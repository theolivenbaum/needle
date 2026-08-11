using Needle.Math;
using Needle.Model;
using Needle.Training.Autodiff;
using Grad = Needle.Training.Autodiff.Ops;

namespace Needle.Training;

/// <summary>
/// The Needle 2 forward pass, expressed on an autodiff tape.
///
/// This is a second implementation of the same arithmetic as
/// <see cref="Needle2Model"/>, and deliberately so.  Inference wants pooled
/// buffers, no retained intermediates and no bookkeeping; training needs every
/// intermediate kept and a backward closure recorded for each. Trying to serve
/// both from one code path would make the inference path pay for training it
/// never does. The two are held together by
/// <see cref="Diagnostics.TrainingCheck"/>, which asserts they produce the same
/// logits.
///
/// The base weights are frozen: only the LoRA factors carry gradients, so
/// activations off the adapted path record nothing at all.
/// </summary>
public sealed class TrainableModel
{
    private readonly Needle2Weights _w;
    private readonly TransformerConfig _cfg;
    private readonly RoPE _rope;
    private readonly float _embedScale;

    // The frozen base, expanded once to float32 kernels the tape can multiply by.
    private readonly NdArray _embedding;
    private readonly NdArray[][] _projections;   // [layer][q, k, v, gate, out]
    private readonly NdArray[] _phiPre;
    private readonly NdArray[] _phiPost;
    private readonly NdArray[] _phiRes;

    /// <summary>Geometry of the model being trained.</summary>
    public TransformerConfig Config => _cfg;

    /// <summary>The frozen base weights.</summary>
    public Needle2Weights Weights => _w;

    /// <param name="weights">
    /// Base weights.  Quantised weights are expanded, since training multiplies
    /// by them from both sides.
    /// </param>
    public TrainableModel(Needle2Weights weights)
    {
        _w = weights;
        _cfg = weights.Config;
        _rope = new RoPE(_cfg.HeadDim, _cfg.MaxSeqLen, _cfg.RopeTheta);
        _embedScale = MathF.Sqrt(_cfg.DModel);

        _embedding = weights.Embedding.ToDense();
        _projections = new NdArray[_cfg.NumLayers][];
        _phiPre = new NdArray[_cfg.NumLayers];
        _phiPost = new NdArray[_cfg.NumLayers];
        _phiRes = new NdArray[_cfg.NumLayers];

        for (int layer = 0; layer < _cfg.NumLayers; layer++)
        {
            var block = weights.Layers[layer];
            _projections[layer] =
            [
                block.QProj.ToDenseKernel(), block.KProj.ToDenseKernel(), block.VProj.ToDenseKernel(),
                block.GateProj.ToDenseKernel(), block.OutProj.ToDenseKernel(),
            ];
            _phiPre[layer] = weights.Mhc[layer].PhiPre.ToDenseKernel();
            _phiPost[layer] = weights.Mhc[layer].PhiPost.ToDenseKernel();
            _phiRes[layer] = weights.Mhc[layer].PhiRes.ToDenseKernel();
        }
    }

    /// <summary>
    /// Record a forward pass and return the logits.
    /// </summary>
    /// <param name="tape">Tape to record on.</param>
    /// <param name="tokens">Token IDs [T].</param>
    /// <param name="adapters">
    /// LoRA factors registered on <paramref name="tape"/>, keyed by parameter
    /// path.  Pass an empty map to run the frozen base.
    /// </param>
    /// <param name="window">Sliding-window width; 0 for plain causal.</param>
    public Value Forward(
        Tape tape, ReadOnlySpan<int> tokens,
        IReadOnlyDictionary<string, (Value A, Value B)> adapters, int window = 0)
    {
        int seqLen = tokens.Length, d = _cfg.DModel, lanes = _cfg.MhcLanes;

        var mask = new SequenceMask(seqLen, window);
        var plan = AttentionPlan.Build(mask);

        // Embedding lookup and the engram sites are frozen and token-driven, so
        // they are computed once, off the tape.
        var reference = new Needle2Model(_w);
        var engram = reference.EngramKeyValues(tokens, mask);

        var embedded = new NdArray(seqLen, d);
        for (int t = 0; t < seqLen; t++)
        {
            _w.Embedding.Row(tokens[t], embedded.Row(t));
            System.Numerics.Tensors.TensorPrimitives.Multiply(embedded.ReadRow(t), _embedScale, embedded.Row(t));
        }

        var x = LaneOps.Broadcast(tape, tape.Constant(embedded, "embed"), lanes, d);

        for (int layer = 0; layer < _cfg.NumLayers; layer++)
        {
            int site = _cfg.EngramLayers.IndexOf(layer);
            x = RunLayer(tape, layer, x, plan, adapters,
                         site >= 0 && engram is not null ? engram.Keys[site] : null,
                         site >= 0 && engram is not null ? engram.Values[site] : null);
        }

        var pooled = LaneOps.Mean(tape, x, lanes, d);
        var hidden = Grad.ZcRmsNorm(tape, pooled, tape.Constant(_w.FinalNorm, "final_norm"));

        return Grad.MatMulTransposed(tape, hidden, _embedding);
    }

    /// <summary>One layer: read the lanes, run the block, write back through the routing.</summary>
    private Value RunLayer(
        Tape tape, int layer, Value lanes, AttentionPlan plan,
        IReadOnlyDictionary<string, (Value A, Value B)> adapters, NdArray? engramKeys, NdArray? engramValues)
    {
        int laneCount = _cfg.MhcLanes, d = _cfg.DModel;
        int seqLen = lanes.Length / (laneCount * d);
        var m = _w.Mhc[layer];

        var flat = Grad.Reshape(tape, lanes, seqLen, laneCount * d);
        var nx = Grad.RmsUnit(tape, flat);

        var readGate = LaneOps.Gate(tape, Grad.MatMul(tape, nx, _phiPre[layer]),
                                    m.APre, m.BPre.ReadSpan, m.PreOffset, doubled: false);
        var u = LaneOps.Read(tape, lanes, readGate, laneCount, d);

        // The engram injection shifts the block input, but the residual is still
        // measured against the un-shifted read — the reference subtracts `u`.
        var blockInput = u;
        if (engramKeys is not null && engramValues is not null)
            blockInput = InjectEngram(tape, u, engramKeys, engramValues);

        var blockOutput = RunBlock(tape, layer, blockInput, plan, adapters);
        var y = Grad.AddScaled(tape, blockOutput, u, -1f);

        var writeGate = LaneOps.Gate(tape, Grad.MatMul(tape, nx, _phiPost[layer]),
                                     m.APost, m.BPost.ReadSpan, m.PostOffset, doubled: true);

        var routingLogits = Grad.MatMul(tape, nx, _phiRes[layer]);
        var biased = AffineRouting(tape, routingLogits, m.ARes, m.BRes);
        var routing = Grad.Sinkhorn(tape, biased, laneCount);

        return LaneOps.Write(tape, lanes, routing, writeGate, y, laneCount, d);
    }

    /// <summary><c>slope * logits + bias</c> over each routing block.</summary>
    private static Value AffineRouting(Tape tape, Value logits, float slope, NdArray bias)
    {
        int blockSize = bias.Length;
        var result = new NdArray((int[])logits.Shape.Clone());
        for (int i = 0; i < result.Length; i++)
            result[i] = slope * logits.Data[i] + bias[i % blockSize];

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!;
            for (int i = 0; i < result.Length; i++) logits.Grad![i] += upstream[i] * slope;
        }, logits);
        return node;
    }

    /// <summary>
    /// Fold one engram site into the block input.  The keys and values are frozen
    /// constants, but the similarity gate depends on <paramref name="u"/>, so the
    /// gradient does flow through it.
    /// </summary>
    private Value InjectEngram(Tape tape, Value u, NdArray keys, NdArray values)
    {
        int d = _cfg.DModel;
        int seqLen = u.Length / d;
        float scale = 1f / MathF.Sqrt(d);

        var unitKeys = keys.Clone();
        Math.Ops.RmsUnit(unitKeys);

        var unitU = Grad.RmsUnit(tape, u);

        // alpha[t] = sigmoid(<unitU[t], unitKeys[t]> / sqrt(d))
        var dot = new NdArray(seqLen, 1);
        for (int t = 0; t < seqLen; t++)
            dot[t] = System.Numerics.Tensors.TensorPrimitives.Dot(
                unitU.Data.ReadRow(t), unitKeys.ReadRow(t)) * scale;

        Value? dotNode = null;
        dotNode = tape.Record(dot, () =>
        {
            var upstream = dotNode!.Grad!;
            for (int t = 0; t < seqLen; t++)
                Math.Ops.AddScaled(unitU.Grad!.Span.Slice(t * d, d), unitKeys.ReadRow(t), upstream[t] * scale);
        }, unitU);

        var alpha = Grad.Sigmoid(tape, dotNode);

        // u + alpha[t] * values[t]
        var result = new NdArray(seqLen, d);
        for (int t = 0; t < seqLen; t++)
        {
            u.Data.ReadRow(t).CopyTo(result.Row(t));
            Math.Ops.AddScaled(result.Row(t), values.ReadRow(t), alpha.Data[t]);
        }

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!;
            for (int t = 0; t < seqLen; t++)
            {
                if (u.RequiresGrad)
                    Math.Ops.Add(u.Grad!.Span.Slice(t * d, d), upstream.ReadRow(t));
                if (alpha.RequiresGrad)
                    alpha.Grad![t] += System.Numerics.Tensors.TensorPrimitives.Dot(
                        upstream.ReadRow(t), values.ReadRow(t));
            }
        }, u, alpha);
        return node;
    }

    /// <summary>One transformer block: gated attention, then the Hadamard MLP.</summary>
    private Value RunBlock(
        Tape tape, int layer, Value x, AttentionPlan plan,
        IReadOnlyDictionary<string, (Value A, Value B)> adapters)
    {
        var block = _w.Layers[layer];
        int d = _cfg.DModel, heads = _cfg.NumHeads, kvHeads = _cfg.NumKvHeads, headDim = _cfg.HeadDim;

        var normed = Grad.ZcRmsNorm(tape, x, tape.Constant(block.NormIn));

        var q = Project(tape, normed, layer, 0, adapters);
        var k = Project(tape, normed, layer, 1, adapters);
        var v = Project(tape, normed, layer, 2, adapters);

        q = AttentionOp.NormAndRotate(tape, q, tape.Constant(block.QNorm), _rope, heads, headDim, 0);
        k = AttentionOp.NormAndRotate(tape, k, tape.Constant(block.KNorm), _rope, kvHeads, headDim, 0);

        var context = AttentionOp.Apply(tape, q, k, v, plan, heads, kvHeads, headDim);
        var gate = Grad.Sigmoid(tape, Project(tape, normed, layer, 3, adapters));
        var attention = Project(tape, Grad.Multiply(tape, context, gate), layer, 4, adapters);

        var post = Grad.ZcRmsNorm(tape, attention, tape.Constant(block.PostAttnNorm));
        var h = Grad.AddScaled(tape, x, post, Needle2Model.Sigmoid(block.AttnGate));

        var mlpIn = Grad.ZcRmsNorm(tape, h, tape.Constant(block.PreHadaNorm));
        return Grad.Add(tape, h, HadamardMlp(tape, mlpIn, block, d));
    }

    /// <summary>A frozen projection plus its LoRA correction, when one is attached.</summary>
    private Value Project(
        Tape tape, Value x, int layer, int slot,
        IReadOnlyDictionary<string, (Value A, Value B)> adapters)
    {
        var baseResult = Grad.MatMul(tape, x, _projections[layer][slot]);

        string name = $"layer{layer:D2}.{LoraSet.Targets[slot]}";
        if (!adapters.TryGetValue(name, out var adapter)) return baseResult;

        var correction = Grad.MatMul(tape, Grad.MatMul(tape, x, adapter.A), adapter.B);
        return Grad.AddScaled(tape, baseResult, correction, ScaleOf(name));
    }

    private float _scale = 1f;

    /// <summary>Scaling applied to every adapter's correction.</summary>
    public float AdapterScale
    {
        get => _scale;
        set => _scale = value;
    }

    private float ScaleOf(string name) { _ = name; return _scale; }

    /// <summary>Two Walsh transforms around a SiLU, with three learned diagonals.</summary>
    private Value HadamardMlp(Tape tape, Value x, BlockWeights block, int d)
    {
        int n = _cfg.HadamardWidth;
        var z = x;

        if (n != d)
        {
            var padded = new NdArray(x.Length / d, n);
            for (int t = 0; t < padded.Shape[0]; t++) x.Data.ReadRow(t).CopyTo(padded.Row(t)[..d]);

            Value? padNode = null;
            padNode = tape.Record(padded, () =>
            {
                var upstream = padNode!.Grad!;
                for (int t = 0; t < padded.Shape[0]; t++)
                    Math.Ops.Add(x.Grad!.Span.Slice(t * d, d), upstream.ReadRow(t)[..d]);
            }, x);
            z = padNode;
        }

        z = Grad.Walsh(tape, Grad.MultiplyRows(tape, z, block.D1));
        z = Grad.Walsh(tape, Grad.Silu(tape, Grad.MultiplyRows(tape, z, block.D2)));
        z = Grad.MultiplyRows(tape, z, block.D3);

        if (n == d) return z;

        var trimmed = new NdArray(z.Length / n, d);
        for (int t = 0; t < trimmed.Shape[0]; t++) z.Data.ReadRow(t)[..d].CopyTo(trimmed.Row(t));

        Value? trimNode = null;
        var source = z;
        trimNode = tape.Record(trimmed, () =>
        {
            var upstream = trimNode!.Grad!;
            for (int t = 0; t < trimmed.Shape[0]; t++)
                Math.Ops.Add(source.Grad!.Span.Slice(t * n, d), upstream.ReadRow(t));
        }, z);
        return trimNode;
    }
}
