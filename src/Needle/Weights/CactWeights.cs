using System.Collections.Immutable;
using Needle.Math;
using Needle.Model;

namespace Needle.Weights;

/// <summary>
/// Builds a model straight from a <c>.cact</c> blob with the weights left
/// packed.
///
/// <see cref="CactLayout"/> reconstructs float32 tensors, which is what the
/// training flow and the parity harness want. This path instead hands the
/// quantised matrices to the model as-is: decode is bound by streaming weights,
/// so keeping them at two bits each is worth far more than the arithmetic it
/// costs to fold dequantisation into the matmul.
///
/// The tensor walk mirrors <see cref="CactLayout"/> exactly — same fixed order,
/// same geometry recovery — so the two stay in step.
/// </summary>
public static class CactWeights
{
    /// <summary>A blob loaded with its weights still packed.</summary>
    /// <param name="Weights">Model parameters.</param>
    /// <param name="TokenizerBlob">The embedded SentencePiece dump, when present.</param>
    public sealed record Loaded(Needle2Weights Weights, byte[]? TokenizerBlob);

    /// <summary>Load a <c>.cact</c> file without expanding its weights.</summary>
    /// <param name="path">Path to the blob.</param>
    /// <param name="template">
    /// Optional configuration supplying what the blob cannot carry (engram orders
    /// and sites, max sequence length, RoPE base).
    /// </param>
    public static Loaded Load(string path, TransformerConfig? template = null) =>
        Load(CactFile.Open(path), template);

    /// <summary>Load an already-parsed blob without expanding its weights.</summary>
    public static Loaded Load(CactFile file, TransformerConfig? template = null)
    {
        template ??= new TransformerConfig();
        var cursor = new Cursor(file);

        // 1. Tied embedding, [vocab, d_model].
        var embedding = cursor.TakeQuantized();
        int vocab = embedding.Rows, d = embedding.Width;

        // 2. Layer groups of fourteen tensors, until the mHC block (whose second
        //    tensor is fp16 rather than quantised) begins.
        var layers = new List<BlockWeights>();
        var d1 = new List<NdArray>();
        while (cursor.Peek(1) is { Dtype: CactDtype.Quantized })
        {
            var normIn = cursor.TakeDense();
            var qProj = cursor.TakeQuantized();
            var kProj = cursor.TakeQuantized();
            var vProj = cursor.TakeQuantized();
            var qNorm = cursor.TakeDense();
            var kNorm = cursor.TakeDense();
            var gateProj = cursor.TakeQuantized();
            var outProj = cursor.TakeQuantized();
            var postNorm = cursor.TakeDense();
            var attnGate = cursor.TakeDense();
            var preHada = cursor.TakeDense();
            var hd1 = cursor.TakeDense();
            var hd2 = cursor.TakeDense();
            var hd3 = cursor.TakeDense();

            d1.Add(hd1);
            layers.Add(new BlockWeights(
                normIn,
                new QuantizedLinear(qProj), new QuantizedLinear(kProj), new QuantizedLinear(vProj),
                new QuantizedLinear(gateProj), new QuantizedLinear(outProj),
                qNorm, kNorm, postNorm, attnGate[0], preHada, hd1, hd2, hd3));
        }

        int layerCount = layers.Count;
        int headDim = layers[0].QNorm.Length;
        int attnWidth = layers[0].QProj.OutputWidth;
        int kvDim = layers[0].KProj.OutputWidth;

        // 3. mHC: six fp16 tensors then three quantised projections.
        var aPre = cursor.TakeDense();
        var aPost = cursor.TakeDense();
        var aRes = cursor.TakeDense();
        var bPre = cursor.TakeDense();
        var bPost = cursor.TakeDense();
        var bRes = cursor.TakeDense();
        int lanes = bRes.Shape[1];

        var phiPre = cursor.TakeQuantized();
        var phiPost = cursor.TakeQuantized();
        var phiRes = cursor.TakeQuantized();

        var mhc = ImmutableArray.CreateBuilder<MhcWeights>(layerCount);
        for (int i = 0; i < layerCount; i++)
        {
            var preOff = new float[lanes];
            var postOff = new float[lanes];
            int lane = i % lanes;
            for (int n = 0; n < lanes; n++)
            {
                preOff[n] = n == lane ? 4f : -4f;
                postOff[n] = n == lane ? 0f : -4f;
            }

            mhc.Add(new MhcWeights(
                LayerSlice(phiPre, i, lanes),
                LayerSlice(phiPost, i, lanes),
                LayerSlice(phiRes, i, lanes * lanes),
                bPre.Slice(i), bPost.Slice(i), bRes.Slice(i),
                aPre[i], aPost[i], aRes[i], preOff, postOff));
        }

        // 4. Engram sites: three quantised tensors and the fp16 taps, repeating
        //    until the fp16 final norm.
        var engrams = ImmutableArray.CreateBuilder<EngramWeights>();
        int slots = template.EngramSlots;
        while (cursor.Peek(0) is { Dtype: CactDtype.Quantized })
        {
            var tables = cursor.TakeQuantized();     // [tables*slots, subDim]
            var keyProj = cursor.TakeQuantized();    // [d, tables*subDim]
            var valueProj = cursor.TakeQuantized();
            var taps = cursor.TakeDense();

            int subDim = tables.Width;
            int tableCount = keyProj.Width / subDim;
            slots = tables.Rows / tableCount;

            engrams.Add(new EngramWeights(
                new QuantizedEngramTable(tables, tableCount),
                new QuantizedLinear(keyProj), new QuantizedLinear(valueProj), taps));
        }

        var finalNorm = cursor.TakeDense();

        // 5. Optional probe heads, introduced by a manifest of head codes.
        ContrastiveHeadWeights? contrastive = null;
        ConfidenceHeadWeights? confidence = null;
        int contrastiveDim = template.ContrastiveDim;

        if (cursor.Peek(0) is { Dtype: CactDtype.Float16 })
        {
            var manifest = cursor.TakeDense();
            for (int h = 0; h < manifest.Length; h++)
            {
                var probes = cursor.TakeDense();
                var proj = cursor.TakeDense();     // [out, probes*d]
                var bias = cursor.TakeDense();

                var kernel = LinearWeight.Dense(Transpose(proj));
                switch ((int)manifest[h])
                {
                    case 1:
                        contrastiveDim = proj.Shape[0];
                        // The exporter drops the temperature; retrieval only uses
                        // the direction of the embedding, so a neutral value is fine.
                        contrastive = new ContrastiveHeadWeights(probes, kernel, MathF.Log(0.07f));
                        break;
                    case 2:
                        confidence = new ConfidenceHeadWeights(probes, kernel, bias[0]);
                        break;
                    default:
                        throw new InvalidDataException(
                            $"Unknown head code {(int)manifest[h]} in .cact manifest.");
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
            EngramSlots = slots,
            KvWindow = file.KvWindow > 0 ? file.KvWindow : template.KvWindow,
            KvBits = file.KvBits > 0 ? file.KvBits : template.KvBits,
            Dtype = "float32",
        };
        config.Validate();

        var weights = Needle2Weights.FromParts(
            config, new QuantizedEmbedding(embedding), [.. layers], mhc.MoveToImmutable(),
            finalNorm, engrams.ToImmutable(), mtp: null, contrastive, confidence);

        int tokenizer = file.TokenizerIndex;
        return new Loaded(weights, tokenizer >= 0 ? file.Payload(tokenizer).ToArray() : null);
    }

    /// <summary>
    /// A view of one layer's rows inside a projection the exporter folded across
    /// layers: <c>phi.transpose(0, 2, 1).reshape(L * fan, width)</c> stacks each
    /// layer's <c>fan</c> output rows contiguously.
    /// </summary>
    private static LinearWeight LayerSlice(QuantizedMatrix folded, int layer, int fan)
    {
        int rowBytes = folded.RowBytes;
        int groups = folded.Groups;

        var packed = new byte[fan * rowBytes];
        Array.Copy(folded.PackedBytes, layer * fan * rowBytes, packed, 0, packed.Length);

        var norms = new float[fan * groups];
        Array.Copy(folded.GroupNorms, layer * fan * groups, norms, 0, norms.Length);

        return new QuantizedLinear(new QuantizedMatrix(
            packed, norms, fan, folded.Width, folded.Bits, folded.GroupSize, folded.Codebook));
    }

    private static NdArray Transpose(NdArray value)
    {
        int rows = value.Shape[0], columns = value.Shape[1];
        var result = new NdArray(columns, rows);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < columns; c++)
                result[c * rows + r] = value[r * columns + c];
        return result;
    }

    /// <summary>Walks the directory in order, keeping quantised tensors packed.</summary>
    private sealed class Cursor(CactFile file)
    {
        private int _index;

        public CactTensor? Peek(int ahead) =>
            _index + ahead < file.Tensors.Count ? file.Tensors[_index + ahead] : null;

        /// <summary>Take the next tensor as float32 (norms, gates, diagonals, probes).</summary>
        public NdArray TakeDense()
        {
            Expect(CactDtype.Float16);
            return file.Read(_index++);
        }

        /// <summary>Take the next tensor with its Cactus-Quant codes intact.</summary>
        public QuantizedMatrix TakeQuantized()
        {
            Expect(CactDtype.Quantized);
            var tensor = file.Tensors[_index];
            if (tensor.Shape.Length != 2)
                throw new InvalidDataException("Quantised tensors must be two-dimensional.");

            return QuantizedMatrix.FromPayload(
                file.Payload(_index++), tensor.Shape[0], tensor.Shape[1],
                tensor.Bits, tensor.GroupSize, file.Codebook(tensor.Bits));
        }

        private void Expect(CactDtype dtype)
        {
            if (_index >= file.Tensors.Count)
                throw new InvalidDataException("Ran off the end of the .cact directory.");
            if (file.Tensors[_index].Dtype != dtype)
                throw new InvalidDataException(
                    $"Tensor {_index} is {file.Tensors[_index].Dtype}, expected {dtype}. "
                    + "The blob's tensor order does not match this reader.");
        }
    }
}
