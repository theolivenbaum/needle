using System.Collections.Immutable;
using Needle.Math;
using Needle.Model;

namespace Needle.Weights;

/// <summary>
/// Turns a <c>.cact</c> blob back into a named parameter set.
///
/// The format stores neither tensor names nor the architecture geometry — the
/// runtime is expected to know both — so this walks the fixed tensor order from
/// <c>export.py</c> and recovers the dimensions from the shapes as it goes:
/// the embedding gives vocabulary and width, <c>q_norm</c> gives the head width
/// (and hence the head counts), <c>mhc_b_res</c> gives the lane count, and the
/// engram tensors give the table geometry.
///
/// What genuinely is not recoverable — the n-gram orders, which layers the
/// engram sites fire at, the maximum sequence length and the RoPE base — comes
/// from the defaults the released model uses, or from a config passed in.
/// </summary>
public static class CactLayout
{
    /// <summary>A loaded blob: the geometry plus the parameters, ready for the model.</summary>
    /// <param name="Config">Recovered configuration.</param>
    /// <param name="Parameters">Flax-path → tensor map, as <see cref="Needle2Weights.FromFlat"/> wants.</param>
    /// <param name="TokenizerBlob">The embedded SentencePiece dump, when present.</param>
    public sealed record Loaded(
        TransformerConfig Config,
        IReadOnlyDictionary<string, NdArray> Parameters,
        byte[]? TokenizerBlob);

    /// <summary>Load a <c>.cact</c> file into a model-ready weight set.</summary>
    /// <param name="path">Path to the blob.</param>
    /// <param name="template">
    /// Optional configuration supplying what the blob cannot carry (engram orders
    /// and sites, max sequence length, RoPE base).  Defaults to the released
    /// Needle 2 settings.
    /// </param>
    public static Loaded Load(string path, TransformerConfig? template = null) =>
        Load(CactFile.Open(path), template);

    /// <summary>Load an already-parsed blob into a model-ready weight set.</summary>
    public static Loaded Load(CactFile file, TransformerConfig? template = null)
    {
        template ??= new TransformerConfig();
        var parameters = new Dictionary<string, NdArray>();
        var cursor = new Cursor(file);

        // 1. Tied embedding, [vocab, d_model].
        var embedding = cursor.Take(CactDtype.Quantized, 2);
        int vocab = embedding.Shape[0], d = embedding.Shape[1];
        parameters["embedding/embedding"] = embedding;

        // 2. Layer groups of fourteen tensors, until the mHC block (whose second
        //    tensor is fp16 rather than quantised) begins.
        var layers = new List<NdArray[]>();
        while (cursor.Peek(1) is { Dtype: CactDtype.Quantized })
            layers.Add(cursor.TakeMany(14));

        int layerCount = layers.Count;
        int headDim = layers[0][4].Shape[0];             // q_norm
        int attnWidth = layers[0][1].Shape[0];           // q_proj is [attn, d]
        int kvDim = layers[0][2].Shape[0];               // k_proj is [kv, d]
        int hadamardWidth = layers[0][11].Shape[0];      // d1

        StackLayers(parameters, layers, layerCount, d, attnWidth, kvDim, headDim, hadamardWidth);

        // 3. mHC: six fp16 tensors then three quantised projections.
        var aPre = cursor.Take(CactDtype.Float16, 1);
        var aPost = cursor.Take(CactDtype.Float16, 1);
        var aRes = cursor.Take(CactDtype.Float16, 1);
        var bPre = cursor.Take(CactDtype.Float16, 2);
        var bPost = cursor.Take(CactDtype.Float16, 2);
        var bRes = cursor.Take(CactDtype.Float16, 3);
        int lanes = bRes.Shape[1];

        parameters["stack/mhc_a_pre"] = aPre;
        parameters["stack/mhc_a_post"] = aPost;
        parameters["stack/mhc_a_res"] = aRes;
        parameters["stack/mhc_b_pre"] = bPre;
        parameters["stack/mhc_b_post"] = bPost;
        parameters["stack/mhc_b_res"] = bRes;

        foreach (string name in (string[])["mhc_phi_pre", "mhc_phi_post", "mhc_phi_res"])
        {
            // Stored as [layers * fan, lanes*d]; the model wants [layers, lanes*d, fan].
            var phi = cursor.Take(CactDtype.Quantized, 2);
            int fan = phi.Shape[0] / layerCount;
            parameters["stack/" + name] = UnfoldPhi(phi, layerCount, fan, phi.Shape[1]);
        }

        // 4. Engram sites: three quantised tensors and the fp16 taps, repeating
        //    until the fp16 final norm.
        int sites = 0;
        while (cursor.Peek(0) is { Dtype: CactDtype.Quantized })
        {
            var tables = cursor.Take(CactDtype.Quantized, 2);   // [tables*slots, subDim]
            var keyProj = cursor.Take(CactDtype.Quantized, 2);  // [d, tables*subDim]
            var valueProj = cursor.Take(CactDtype.Quantized, 2);
            var taps = cursor.Take(CactDtype.Float16, 2);

            int subDim = tables.Shape[1];
            int tableCount = keyProj.Shape[1] / subDim;
            int slots = tables.Shape[0] / tableCount;

            parameters[$"engrams_{sites}/embedding"] = tables.Reshape(tableCount, slots, subDim);
            parameters[$"engrams_{sites}/key_proj/kernel"] = Transpose(keyProj);
            parameters[$"engrams_{sites}/value_proj/kernel"] = Transpose(valueProj);
            parameters[$"engrams_{sites}/taps"] = taps;
            sites++;
        }

        parameters["stack/final_norm/scale"] = cursor.Take(CactDtype.Float16, 1);

        // 5. Optional probe heads, introduced by a manifest of head codes.
        int contrastiveDim = template.ContrastiveDim;
        if (cursor.Peek(0) is { Dtype: CactDtype.Float16 })
        {
            var manifest = cursor.Take(CactDtype.Float16, 1);
            for (int h = 0; h < manifest.Length; h++)
            {
                var probes = cursor.Take(CactDtype.Float16, 2);
                var proj = cursor.Take(CactDtype.Float16, 2);
                var bias = cursor.Take(CactDtype.Float16, 1);

                string head = (int)manifest[h] switch
                {
                    1 => "contrastive_head",
                    2 => "confidence_head",
                    var code => throw new InvalidDataException($"Unknown head code {code} in .cact manifest."),
                };

                parameters[$"{head}/probes"] = probes;
                parameters[$"{head}/proj/kernel"] = Transpose(proj);
                if (head == "contrastive_head")
                {
                    contrastiveDim = proj.Shape[0];
                    // The exporter drops the temperature; retrieval only needs the
                    // direction of the embedding, so a neutral value is fine.
                    parameters["contrastive_head/log_temp"] = Scalar(MathF.Log(0.07f));
                }
                else
                {
                    parameters["confidence_head/proj/bias"] = bias;
                }
            }
        }

        var config = template with
        {
            VocabSize = vocab,
            DModel = d,
            AttnDim = attnWidth == d ? 0 : attnWidth,
            NumHeads = attnWidth / headDim,
            NumKvHeads = kvDim / headDim,
            NumLayers = layerCount,
            MhcLanes = lanes,
            ContrastiveDim = contrastiveDim,
            KvWindow = file.KvWindow > 0 ? file.KvWindow : template.KvWindow,
            KvBits = file.KvBits > 0 ? file.KvBits : template.KvBits,
            EngramSlots = SlotsOf(parameters, sites, template.EngramSlots),
            EngramLayers = sites == template.EngramLayers.Length
                ? template.EngramLayers
                : [.. Enumerable.Range(0, sites).Select(i => (i + 1) * layerCount / (sites + 1))],
            Dtype = "float32",
        };

        int tokenizer = file.TokenizerIndex;
        return new Loaded(config, parameters,
                          tokenizer >= 0 ? file.Payload(tokenizer).ToArray() : null);
    }

    private static int SlotsOf(IReadOnlyDictionary<string, NdArray> parameters, int sites, int fallback) =>
        sites > 0 && parameters.TryGetValue("engrams_0/embedding", out var tables)
            ? tables.Shape[1]
            : fallback;

    /// <summary>Restack the per-layer tensors into the leading-layer-axis layout.</summary>
    private static void StackLayers(
        Dictionary<string, NdArray> parameters, List<NdArray[]> layers,
        int layerCount, int d, int attnWidth, int kvDim, int headDim, int hadamardWidth)
    {
        const string P = "stack/layers/block/";

        parameters[P + "ZCRMSNorm_0/scale"] = Stack(layers, 0, layerCount, d);
        parameters[P + "self_attn/q_proj/kernel"] = StackTransposed(layers, 1, layerCount, d, attnWidth);
        parameters[P + "self_attn/k_proj/kernel"] = StackTransposed(layers, 2, layerCount, d, kvDim);
        parameters[P + "self_attn/v_proj/kernel"] = StackTransposed(layers, 3, layerCount, d, kvDim);
        parameters[P + "self_attn/q_norm/scale"] = Stack(layers, 4, layerCount, headDim);
        parameters[P + "self_attn/k_norm/scale"] = Stack(layers, 5, layerCount, headDim);
        parameters[P + "self_attn/gate_proj/kernel"] = StackTransposed(layers, 6, layerCount, d, attnWidth);
        parameters[P + "self_attn/out_proj/kernel"] = StackTransposed(layers, 7, layerCount, attnWidth, d);
        parameters[P + "post_attn_norm/scale"] = Stack(layers, 8, layerCount, d);
        parameters[P + "attn_gate"] = Stack(layers, 9, layerCount, 1).Reshape(layerCount);
        parameters[P + "pre_hada_norm/scale"] = Stack(layers, 10, layerCount, d);
        parameters[P + "hadamard_mlp/d1"] = Stack(layers, 11, layerCount, hadamardWidth);
        parameters[P + "hadamard_mlp/d2"] = Stack(layers, 12, layerCount, hadamardWidth);
        parameters[P + "hadamard_mlp/d3"] = Stack(layers, 13, layerCount, hadamardWidth);
    }

    private static NdArray Stack(List<NdArray[]> layers, int slot, int layerCount, int width)
    {
        var result = new NdArray(layerCount, width);
        for (int i = 0; i < layerCount; i++) layers[i][slot].ReadSpan.CopyTo(result.Row(i));
        return result;
    }

    /// <summary>
    /// Stack per-layer kernels, transposing each one back from the exporter's
    /// <c>[out, in]</c> layout to the <c>[in, out]</c> the model multiplies by.
    /// </summary>
    private static NdArray StackTransposed(List<NdArray[]> layers, int slot, int layerCount, int inDim, int outDim)
    {
        var result = new NdArray(layerCount, inDim, outDim);
        for (int i = 0; i < layerCount; i++)
        {
            var source = layers[i][slot];
            var target = result.Slice(i);
            for (int o = 0; o < outDim; o++)
                for (int k = 0; k < inDim; k++)
                    target[k * outDim + o] = source[o * inDim + k];
        }
        return result;
    }

    private static NdArray Transpose(NdArray value)
    {
        int rows = value.Shape[0], cols = value.Shape[1];
        var result = new NdArray(cols, rows);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[c * rows + r] = value[r * cols + c];
        return result;
    }

    /// <summary>Undo <c>phi.transpose(0, 2, 1).reshape(L * fan, width)</c>.</summary>
    private static NdArray UnfoldPhi(NdArray folded, int layerCount, int fan, int width)
    {
        var result = new NdArray(layerCount, width, fan);
        for (int l = 0; l < layerCount; l++)
            for (int f = 0; f < fan; f++)
                for (int w = 0; w < width; w++)
                    result[(l * width + w) * fan + f] = folded[(l * fan + f) * width + w];
        return result;
    }

    private static NdArray Scalar(float value)
    {
        var result = new NdArray(1);
        result[0] = value;
        return result;
    }

    /// <summary>Walks the directory in order, checking each record is the shape it should be.</summary>
    private sealed class Cursor(CactFile file)
    {
        private int _index;

        public CactTensor? Peek(int ahead) =>
            _index + ahead < file.Tensors.Count ? file.Tensors[_index + ahead] : null;

        public NdArray Take(CactDtype dtype, int rank)
        {
            if (_index >= file.Tensors.Count)
                throw new InvalidDataException("Ran off the end of the .cact directory.");
            var tensor = file.Tensors[_index];
            if (tensor.Dtype != dtype || tensor.Shape.Length != rank)
                throw new InvalidDataException(
                    $"Tensor {_index} is {tensor.Dtype} rank {tensor.Shape.Length}, "
                    + $"expected {dtype} rank {rank}. The blob's tensor order does not match this reader.");
            return file.Read(_index++);
        }

        public NdArray[] TakeMany(int count)
        {
            var result = new NdArray[count];
            for (int i = 0; i < count; i++) result[i] = file.Read(_index++);
            return result;
        }
    }
}
