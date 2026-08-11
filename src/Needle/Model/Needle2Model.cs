using System.Numerics.Tensors;
using Needle.Diagnostics;
using Needle.Math;

namespace Needle.Model;

/// <summary>
/// Everything the model produced for one sequence, kept so callers (and the
/// parity harness) can inspect intermediate stages rather than only the logits.
/// </summary>
/// <param name="Hidden">Final-normed hidden states [T, D].</param>
/// <param name="Cells">
/// Per-layer residual snapshots [T, L+1, D] — the embedding followed by each
/// layer's lane-average.  Only populated when requested; the pooled heads need it.
/// </param>
public sealed record ForwardResult(NdArray Hidden, NdArray? Cells);

/// <summary>
/// Needle 2 — a decoder-only Simple Attention Network, implemented directly on
/// <see cref="NdArray"/> with the SIMD kernels in <see cref="Ops"/>.
///
/// Port of <c>SimpleAttentionNetwork</c> in
/// <c>.reference/needle/model/architecture.py</c>.  What makes it a SAN rather
/// than a plain transformer:
/// <list type="bullet">
///   <item>a Hadamard MLP — two fixed Walsh transforms around a SiLU, with three
///         learned diagonals and no weight matrix — in place of the FFN,</item>
///   <item>GQA attention with QK-norm and a sigmoid output gate,</item>
///   <item>engram key-value memory: hashed n-gram tables read at a few layers,</item>
///   <item>multi-lane hyper-connections — the residual stream is
///         <see cref="TransformerConfig.MhcLanes"/> lanes wide and mixed each
///         layer by an input-dependent, Sinkhorn-normalised routing matrix,</item>
///   <item>a tied output projection plus auxiliary multi-token-prediction,
///         contrastive (tool retrieval) and confidence heads.</item>
/// </list>
///
/// Everything runs in float32.  So does the reference decoder — <c>decode.py</c>
/// casts the checkpoint with <c>_f32</c> before generating.
/// </summary>
public sealed class Needle2Model
{
    /// <summary>Probe count of the contrastive head (<c>ContrastiveHead.PROBES</c>).</summary>
    public const int ContrastiveProbes = 4;

    /// <summary>Probe count of the confidence head (<c>ConfidenceHead.PROBES</c>).</summary>
    public const int ConfidenceProbes = 8;

    private readonly TransformerConfig _cfg;
    private readonly Needle2Weights _w;
    private readonly RoPE _rope;
    private readonly float _embedScale;

    // Pooled per-pass state.  A model instance is single-threaded, so one arena
    // and one lane buffer serve every layer of every step.
    private readonly ScratchArena _scratch = new();
    private readonly StackStatePool _stacks = new();

    // Packed weights want their input rotated before the dot; the rotation
    // depends on the activation, not the matrix, so consecutive projections of
    // one activation share it.  Ignored entirely by float32 weights.
    private readonly Weights.PreparedActivation _rotation = new();

    /// <summary>Geometry of this model.</summary>
    public TransformerConfig Config => _cfg;

    /// <summary>The parameters this model reads.</summary>
    public Needle2Weights Weights => _w;

    /// <summary>Shared RoPE tables, sized to <see cref="TransformerConfig.MaxSeqLen"/>.</summary>
    public RoPE Rope => _rope;

    /// <summary>Peak scratch demand in floats, for sizing diagnostics.</summary>
    public long ScratchHighWater => _scratch.HighWater;

    /// <summary>
    /// Attach to break a forward pass down by stage.  Null by default, and every
    /// probe is a null check when it is.
    /// </summary>
    public StageProfiler? Profiler { get; set; }

    public Needle2Model(Needle2Weights weights)
    {
        _w = weights;
        _cfg = weights.Config;
        _embedScale = MathF.Sqrt(_cfg.DModel);
        _rope = new RoPE(_cfg.HeadDim, _cfg.MaxSeqLen, _cfg.RopeTheta);
    }

    // ── Embedding and output projection ──────────────────────────────────────

    /// <summary>Scaled token embeddings [T, D].  Port of <c>embedding * embed_scale</c>.</summary>
    public NdArray Embed(ReadOnlySpan<int> tokens)
    {
        int d = _cfg.DModel;
        var x = new NdArray(tokens.Length, d);
        for (int t = 0; t < tokens.Length; t++)
        {
            int id = tokens[t];
            if ((uint)id >= (uint)_cfg.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokens),
                    $"Token {id} at position {t} is outside the vocabulary ({_cfg.VocabSize}).");
            _w.Embedding.Row(id, x.Row(t));
            TensorPrimitives.Multiply(x.ReadRow(t), _embedScale, x.Row(t));
        }
        return x;
    }

    /// <summary>
    /// Project hidden states through the tied embedding: [T, D] → [T, V].
    /// </summary>
    public NdArray Logits(NdArray hidden)
    {
        int d = _cfg.DModel, v = _cfg.VocabSize;
        int t = hidden.Length / d;
        var logits = new NdArray(t, v);
        _w.Embedding.Project(hidden.Reshape(t, d), logits);
        return logits;
    }

    /// <summary>Logits for the last position only — the one generation needs.</summary>
    public NdArray LastLogits(NdArray hidden)
    {
        var logits = new NdArray(1, _cfg.VocabSize);
        LastLogitsInto(hidden, logits);
        return logits.Reshape(_cfg.VocabSize);
    }

    /// <summary>
    /// Logits for the last position, written into <paramref name="destination"/>
    /// — a whole vocabulary row is 32 KB, so a decoding session reuses one buffer
    /// rather than allocating per step.
    /// </summary>
    public void LastLogitsInto(NdArray hidden, NdArray destination)
    {
        int d = _cfg.DModel;
        int t = hidden.Length / d;
        _w.Embedding.Project(hidden.Reshape(t, d).SliceRange(t - 1, 1), destination);
    }

    // ── Full forward pass ────────────────────────────────────────────────────

    /// <summary>
    /// Run the stack over a whole sequence.  Port of
    /// <c>SimpleAttentionNetwork.__call__</c> / <c>hidden_cells</c>.
    /// </summary>
    /// <param name="tokens">Token IDs [T].</param>
    /// <param name="mask">Attention rules; defaults to plain causal.</param>
    /// <param name="collectCells">Also return the per-layer residual snapshots.</param>
    /// <param name="trace">Optional sink for intermediate tensors (parity harness).</param>
    public ForwardResult Forward(
        ReadOnlySpan<int> tokens,
        SequenceMask? mask = null,
        bool collectCells = false,
        ITrace? trace = null)
    {
        int seqLen = tokens.Length;
        mask ??= SequenceMask.Causal(seqLen);
        if (mask.Length != seqLen)
            throw new ArgumentException($"Mask covers {mask.Length} positions, got {seqLen} tokens.", nameof(mask));

        var engram = EngramKeyValues(tokens, mask, trace);
        var plan = AttentionPlan.Build(mask);
        return RunStack(tokens, plan, engram, startPosition: 0, caches: null, collectCells, trace);
    }

    /// <summary>
    /// Push a window of tokens through the stack.  Shared by the full forward
    /// pass and the incremental decoder; the only difference is whether keys and
    /// values come from a cache.
    /// </summary>
    /// <param name="tokens">Token IDs for this window.</param>
    /// <param name="plan">Resolved key positions for the window's queries.</param>
    /// <param name="engram">Engram activations aligned with the window.</param>
    /// <param name="startPosition">Absolute position of the window's first token.</param>
    /// <param name="caches">Per-layer KV caches, or null for a self-contained pass.</param>
    /// <param name="collectCells">Also return the per-layer residual snapshots.</param>
    /// <param name="trace">Optional sink for intermediates.</param>
    public ForwardResult RunStack(
        ReadOnlySpan<int> tokens,
        AttentionPlan plan,
        EngramActivations? engram,
        int startPosition,
        LayerKvCache[]? caches,
        bool collectCells = false,
        ITrace? trace = null)
    {
        int seqLen = tokens.Length;
        var mark = StageProfiler.Mark(Profiler);
        var x0 = Embed(tokens);
        StageProfiler.Add(Profiler, Stage.Embed, mark);
        trace?.Record("embed", x0);

        var stack = _stacks.Rent(_cfg, seqLen);
        stack.Initialise(x0);

        var cells = collectCells ? new NdArray(seqLen, _cfg.NumLayers + 1, _cfg.DModel) : null;
        if (cells is not null) CopyCell(x0, cells, 0);

        // Every layer wants the same temporaries, so they come out of one pooled
        // arena that rewinds between layers rather than off the heap.
        var scratch = _scratch;
        for (int layer = 0; layer < _cfg.NumLayers; layer++)
        {
            scratch.Reset();
            RunLayer(layer, stack, plan, engram, startPosition, caches?[layer], trace, scratch);
            if (cells is not null)
            {
                var cellMark = scratch.Mark;
                CopyCell(stack.LaneMean(scratch), cells, layer + 1);
                scratch.RewindTo(cellMark);
            }
        }
        scratch.Reset();

        mark = StageProfiler.Mark(Profiler);
        var hidden = new NdArray(seqLen, _cfg.DModel);
        stack.LaneMeanInto(hidden);
        Ops.ZcRmsNorm(hidden, _w.FinalNorm.ReadSpan);
        StageProfiler.Add(Profiler, Stage.Logits, mark);
        trace?.Record("hidden", hidden);

        _stacks.Return(stack);
        return new ForwardResult(hidden, cells);
    }

    /// <summary>Copy a [T, D] slab into cell index <paramref name="cell"/> of [T, L+1, D].</summary>
    private void CopyCell(NdArray source, NdArray cells, int cell)
    {
        int d = _cfg.DModel, l1 = _cfg.NumLayers + 1;
        int t = source.Length / d;
        for (int i = 0; i < t; i++)
            source.ReadRow(i).CopyTo(cells.Span.Slice((i * l1 + cell) * d, d));
    }

    // ── One layer of the hyper-connection stack ──────────────────────────────

    /// <summary>
    /// Port of <c>_ScanBody.__call__</c>: read the lanes into a single block
    /// input, run the block, then write the result back through the routing
    /// matrix and the write gate.
    /// </summary>
    private void RunLayer(int layer, StackState stack, AttentionPlan plan, EngramActivations? engram,
                          int startPosition, LayerKvCache? cache, ITrace? trace, ScratchArena scratch)
    {
        int lanes = _cfg.MhcLanes, d = _cfg.DModel, seqLen = stack.SeqLen;
        var m = _w.Mhc[layer];

        // nx = rms_unit(x.reshape(T, lanes*D)) — the shared read of every lane.
        var mark = StageProfiler.Mark(Profiler);
        var nx = scratch.Take(seqLen, lanes * d);
        stack.Lanes.ReadSpan.CopyTo(nx.Span);
        Ops.RmsUnit(nx);
        StageProfiler.Add(Profiler, Stage.LaneRead, mark);

        // hpre = sigmoid(a_pre * (nx @ phi_pre) + b_pre + pre_off)
        mark = StageProfiler.Mark(Profiler);
        // The pre, post and routing gates are three projections of one 2048-wide
        // lane vector, and rotating it is far more work than the projections are.
        _rotation.Bind(nx);
        var hpre = m.PhiPre.Apply(nx, scratch, _rotation);
        ApplyGate(hpre, m.APre, m.BPre.ReadSpan, m.PreOffset, doubled: false);
        StageProfiler.Add(Profiler, Stage.LaneGates, mark);

        // u = einsum("btn,btnc->btc", hpre, x)
        mark = StageProfiler.Mark(Profiler);
        var u = scratch.Take(seqLen, d, clear: true);
        for (int t = 0; t < seqLen; t++)
        {
            var dst = u.Row(t);
            var gates = hpre.ReadRow(t);
            for (int n = 0; n < lanes; n++)
                Ops.AddScaled(dst, stack.Lane(t, n), gates[n]);
        }
        StageProfiler.Add(Profiler, Stage.LaneReduce, mark);

        trace?.Record($"layer{layer:D2}.u", u);

        // The engram sites feed the block a shifted input, but the residual the
        // stack writes back is still measured against the un-shifted read — the
        // reference subtracts `u`, not the injected `bx`.
        var blockInput = u;
        int site = _cfg.EngramLayers.IndexOf(layer);
        if (site >= 0 && engram is not null)
        {
            mark = StageProfiler.Mark(Profiler);
            blockInput = scratch.Take(seqLen, d);
            u.ReadSpan.CopyTo(blockInput.Span);
            InjectEngram(blockInput, engram.Keys[site], engram.Values[site], trace, layer, scratch);
            StageProfiler.Add(Profiler, Stage.Engram, mark);
            trace?.Record($"layer{layer:D2}.bx", blockInput);
        }

        // y = block(bx) - u
        var y = RunBlock(blockInput, _w.Layers[layer], plan, trace, layer, startPosition, cache, scratch);
        TensorPrimitives.Subtract(y.Span, u.ReadSpan, y.Span);
        trace?.Record($"layer{layer:D2}.y", y);

        // hpost = 2 * sigmoid(a_post * (nx @ phi_post) + b_post + post_off)
        mark = StageProfiler.Mark(Profiler);
        var hpost = m.PhiPost.Apply(nx, scratch, _rotation);
        ApplyGate(hpost, m.APost, m.BPost.ReadSpan, m.PostOffset, doubled: true);

        // hres = sinkhorn(a_res * (nx @ phi_res) + b_res)
        var res = m.PhiRes.Apply(nx, scratch, _rotation);
        StageProfiler.Add(Profiler, Stage.LaneGates, mark);
        _rotation.Release();

        mark = StageProfiler.Mark(Profiler);
        var resSpan = res.Span;
        var bRes = m.BRes.ReadSpan;
        for (int t = 0; t < seqLen; t++)
        {
            var block = resSpan.Slice(t * lanes * lanes, lanes * lanes);
            for (int i = 0; i < block.Length; i++) block[i] = m.ARes * block[i] + bRes[i];
            Sinkhorn.Normalize(block, lanes);
        }
        StageProfiler.Add(Profiler, Stage.Sinkhorn, mark);

        mark = StageProfiler.Mark(Profiler);
        stack.Mix(res, hpost, y);
        StageProfiler.Add(Profiler, Stage.LaneMix, mark);
        trace?.Record($"layer{layer:D2}.x", stack.Lanes);
    }

    /// <summary>
    /// <c>gate = (doubled ? 2 : 1) * sigmoid(slope * logits + bias + offset)</c>,
    /// applied row-wise over a [T, lanes] tensor.
    /// </summary>
    private static void ApplyGate(NdArray logits, float slope, ReadOnlySpan<float> bias,
                                  ReadOnlySpan<float> offset, bool doubled)
    {
        int lanes = logits.LastDim;
        int rows = logits.Length / lanes;
        var data = logits.Span;
        for (int r = 0; r < rows; r++)
        {
            var row = data.Slice(r * lanes, lanes);
            for (int n = 0; n < lanes; n++) row[n] = slope * row[n] + bias[n] + offset[n];
            Ops.Sigmoid(row);
            if (doubled) Ops.Scale(row, 2f);
        }
    }

    // ── One transformer block ────────────────────────────────────────────────

    /// <summary>
    /// Port of <c>Block.__call__</c> minus the engram injection, which the caller
    /// has already folded into <paramref name="x"/>.
    /// </summary>
    /// <param name="x">Block input [T, D]; not modified.</param>
    /// <param name="w">Block weights.</param>
    /// <param name="plan">Resolved key positions for each query.</param>
    /// <param name="trace">Optional sink for intermediates.</param>
    /// <param name="layer">Layer index, used only to label traced tensors.</param>
    /// <param name="startPosition">Absolute position of <c>x[0]</c>.</param>
    /// <param name="cache">Optional KV cache for incremental decoding.</param>
    public NdArray RunBlock(NdArray x, BlockWeights w, AttentionPlan plan, ITrace? trace = null,
                            int layer = -1, int startPosition = 0, LayerKvCache? cache = null,
                            ScratchArena? scratch = null)
    {
        int d = _cfg.DModel;
        int seqLen = x.Length / d;
        scratch ??= _scratch;

        var mark = StageProfiler.Mark(Profiler);
        var normed = scratch.Take(seqLen, d);
        x.ReadSpan.CopyTo(normed.Span);
        Ops.ZcRmsNorm(normed, w.NormIn.ReadSpan);
        StageProfiler.Add(Profiler, Stage.BlockNorm, mark);

        var attn = Attention(normed, w, plan, startPosition, cache, scratch);

        mark = StageProfiler.Mark(Profiler);
        Ops.ZcRmsNorm(attn, w.PostAttnNorm.ReadSpan);
        StageProfiler.Add(Profiler, Stage.BlockNorm, mark);
        if (layer >= 0) trace?.Record($"layer{layer:D2}.attn", attn);

        // h = x + sigmoid(attn_gate) * attn
        float gate = Sigmoid(w.AttnGate);
        var h = scratch.Take(seqLen, d);
        x.ReadSpan.CopyTo(h.Span);
        Ops.AddScaled(h.Span, attn.ReadSpan, gate);

        mark = StageProfiler.Mark(Profiler);
        var mlpIn = scratch.Take(seqLen, d);
        h.ReadSpan.CopyTo(mlpIn.Span);
        Ops.ZcRmsNorm(mlpIn, w.PreHadaNorm.ReadSpan);
        StageProfiler.Add(Profiler, Stage.BlockNorm, mark);

        mark = StageProfiler.Mark(Profiler);
        var mlp = HadamardMlp(mlpIn, w, seqLen, scratch);
        StageProfiler.Add(Profiler, Stage.HadamardMlp, mark);
        if (layer >= 0) trace?.Record($"layer{layer:D2}.mlp", mlp);

        Ops.Add(h.Span, mlp.ReadSpan);
        return h;
    }

    /// <summary>
    /// Two orthonormal Walsh transforms sandwiching a SiLU, with a learned
    /// diagonal in front of each.  Port of <c>HadamardMLP</c>.
    /// </summary>
    private NdArray HadamardMlp(NdArray x, BlockWeights w, int seqLen, ScratchArena scratch)
    {
        int d = _cfg.DModel, n = _cfg.HadamardWidth;

        var z = scratch.Take(seqLen, n, clear: n != d);
        if (n == d)
        {
            x.ReadSpan.CopyTo(z.Span);
        }
        else
        {
            for (int t = 0; t < seqLen; t++) x.ReadRow(t).CopyTo(z.Row(t)[..d]);
        }

        Ops.MultiplyRows(z, w.D1.ReadSpan);
        WalshHadamard.TransformRows(z);
        Ops.MultiplyRows(z, w.D2.ReadSpan);
        Ops.Silu(z.Span);
        WalshHadamard.TransformRows(z);
        Ops.MultiplyRows(z, w.D3.ReadSpan);

        if (n == d) return z;

        var trimmed = scratch.Take(seqLen, d);
        for (int t = 0; t < seqLen; t++) z.ReadRow(t)[..d].CopyTo(trimmed.Row(t));
        return trimmed;
    }

    // ── Attention ────────────────────────────────────────────────────────────

    /// <summary>
    /// GQA self-attention with QK-norm, RoPE and a sigmoid output gate.
    /// Port of <c>MultiHeadAttention.__call__</c> (and, with a
    /// <paramref name="cache"/>, of <c>_attn_cached</c> in decode.py).
    /// </summary>
    /// <param name="x">Normalised block input [T, D].</param>
    /// <param name="w">Block weights.</param>
    /// <param name="plan">Resolved key positions for each query in this window.</param>
    /// <param name="startPosition">Absolute position of <c>x[0]</c>.</param>
    /// <param name="cache">
    /// Optional KV cache.  Keys and values are written at their absolute
    /// positions and the whole cache is attended over, which is what turns the
    /// same routine into the incremental decoder.
    /// </param>
    public NdArray Attention(NdArray x, BlockWeights w, AttentionPlan plan, int startPosition,
                             LayerKvCache? cache, ScratchArena? scratch = null)
    {
        scratch ??= _scratch;
        int d = _cfg.DModel, a = _cfg.AttnWidth, kvDim = _cfg.KvDim;
        int heads = _cfg.NumHeads, kvHeads = _cfg.NumKvHeads, headDim = _cfg.HeadDim;
        int repeats = heads / kvHeads;
        int seqLen = x.Length / d;

        var mark = StageProfiler.Mark(Profiler);
        // Query, key, value and the output gate all project the same normalised
        // block input; bind it once so the rotation is not redone four times.
        _rotation.Bind(x);
        var q = w.QProj.Apply(x, scratch, _rotation);
        var k = w.KProj.Apply(x, scratch, _rotation);
        var v = w.VProj.Apply(x, scratch, _rotation);
        StageProfiler.Add(Profiler, Stage.QkvProject, mark);

        mark = StageProfiler.Mark(Profiler);
        NormHeads(q, heads, headDim, w.QNorm.ReadSpan);
        NormHeads(k, kvHeads, headDim, w.KNorm.ReadSpan);
        _rope.ApplyRows(q, heads, startPosition);
        _rope.ApplyRows(k, kvHeads, startPosition);

        NdArray keys = k, values = v;
        if (cache is not null)
        {
            cache.Append(k, v, startPosition);
            keys = cache.Keys;
            values = cache.Values;
        }

        var context = scratch.Take(seqLen, a);
        float scale = 1f / MathF.Sqrt(headDim);

        var scoreBuffer = scratch.TakeSpan(System.Math.Max(1, plan.MaxKeys));

        for (int t = 0; t < seqLen; t++)
        {
            var allowed = plan.Keys(t);
            for (int h = 0; h < heads; h++)
            {
                int kvHead = h / repeats;
                var qHead = q.Row(t).Slice(h * headDim, headDim);
                var outHead = context.Row(t).Slice(h * headDim, headDim);
                if (allowed.Length == 0) { outHead.Clear(); continue; }

                var scores = scoreBuffer[..allowed.Length];
                AttentionKernels.Scores(qHead, keys.ReadSpan, allowed, kvDim, kvHead * headDim, scale, scores);
                Ops.Softmax(scores);
                AttentionKernels.Combine(outHead, values.ReadSpan, allowed, kvDim, kvHead * headDim, scores);
            }
        }

        StageProfiler.Add(Profiler, Stage.AttentionCore, mark);

        // out *= sigmoid(x @ gate_proj)
        mark = StageProfiler.Mark(Profiler);
        var gate = w.GateProj.Apply(x, scratch, _rotation);
        Ops.Sigmoid(gate.Span);
        Ops.Multiply(context.Span, gate.ReadSpan);
        StageProfiler.Add(Profiler, Stage.QkvProject, mark);
        _rotation.Release();

        mark = StageProfiler.Mark(Profiler);
        var projected = w.OutProj.Apply(context, scratch);
        StageProfiler.Add(Profiler, Stage.OutProject, mark);
        return projected;
    }

    /// <summary>Zero-centred RMSNorm applied independently to each head.</summary>
    private static void NormHeads(NdArray x, int heads, int headDim, ReadOnlySpan<float> scale)
    {
        int rows = x.Length / (heads * headDim);
        var data = x.Span;
        int stride = heads * headDim;
        for (int t = 0; t < rows; t++)
        {
            for (int h = 0; h < heads; h++)
            {
                var head = data.Slice(t * stride + h * headDim, headDim);
                float invRms = 1f / MathF.Sqrt(
                    TensorPrimitives.Dot(head, head) / headDim + Ops.Epsilon);
                for (int i = 0; i < headDim; i++) head[i] = (1f + scale[i]) * head[i] * invRms;
            }
        }
    }

    // ── Engram memory ────────────────────────────────────────────────────────

    /// <summary>Keys and values produced by each engram site, one [T, D] pair per site.</summary>
    /// <param name="Keys">Per-site keys.</param>
    /// <param name="Values">Per-site values.</param>
    public sealed record EngramActivations(NdArray[] Keys, NdArray[] Values);

    /// <summary>
    /// Read every engram site across a sequence.
    /// Port of <c>SimpleAttentionNetwork._engram_kv</c>.
    /// </summary>
    public EngramActivations? EngramKeyValues(ReadOnlySpan<int> tokens, SequenceMask mask, ITrace? trace = null)
    {
        if (_w.Engrams.Length == 0) return null;

        var (orders, heads, _) = _cfg.EngramGeometry();
        int tables = orders.Length * heads;
        int maxOrder = orders.Max();
        int seqLen = tokens.Length;

        // ngram_ok[t, table]: may position t still see the token the n-gram reaches back to?
        var ngramOk = new float[seqLen * tables];
        for (int oi = 0; oi < orders.Length; oi++)
        {
            var diag = mask.Diagonal(orders[oi] - 1);
            for (int h = 0; h < heads; h++)
            {
                int table = oi * heads + h;
                for (int t = 0; t < seqLen; t++) ngramOk[t * tables + table] = diag[t];
            }
        }

        var tapOk = new float[EngramConstants.ConvTaps][];
        for (int j = 0; j < EngramConstants.ConvTaps; j++) tapOk[j] = mask.Diagonal(j * maxOrder);

        return EngramCore(tokens, ngramOk, tapOk, trim: 0, trace);
    }

    /// <summary>
    /// Engram read over a short trailing window, for incremental decoding.
    ///
    /// The sites only reach <c>ConvTaps * maxOrder</c> positions back, so a step
    /// re-reads that many tokens of history plus the new ones and discards the
    /// warm-up rows — the same trick as <c>_engram_kv</c> in decode.py, and the
    /// reason a step costs a fixed amount of work rather than growing with the
    /// conversation.
    /// </summary>
    /// <param name="windowTokens">History window, oldest first.</param>
    /// <param name="windowValid">Per-position validity of that window.</param>
    /// <param name="trim">Number of leading warm-up rows to drop.</param>
    public EngramActivations? EngramWindow(ReadOnlySpan<int> windowTokens, ReadOnlySpan<bool> windowValid, int trim)
    {
        if (_w.Engrams.Length == 0) return null;

        var (orders, heads, _) = _cfg.EngramGeometry();
        int tables = orders.Length * heads;
        int maxOrder = orders.Max();
        int windowLen = windowTokens.Length;

        // The cached path checks only that the token an n-gram reaches back to was
        // itself real; the window is far shorter than the sliding window, so the
        // extra mask terms in the batched path can never fire here.
        var ngramOk = new float[windowLen * tables];
        for (int oi = 0; oi < orders.Length; oi++)
        {
            int back = orders[oi] - 1;
            for (int h = 0; h < heads; h++)
            {
                int table = oi * heads + h;
                for (int t = back; t < windowLen; t++)
                    ngramOk[t * tables + table] = windowValid[t - back] ? 1f : 0f;
            }
        }

        var tapOk = new float[EngramConstants.ConvTaps][];
        for (int j = 0; j < EngramConstants.ConvTaps; j++)
        {
            tapOk[j] = new float[windowLen];
            int back = j * maxOrder;
            for (int t = back; t < windowLen; t++)
                tapOk[j][t] = windowValid[t - back] ? 1f : 0f;
        }

        return EngramCore(windowTokens, ngramOk, tapOk, trim, trace: null);
    }

    /// <summary>
    /// Project one site's hashed rows for positions
    /// <c>[startPosition, startPosition + count)</c>, before the causal
    /// convolution.
    ///
    /// This is the piece a session can compute incrementally: the hash reaches
    /// back only <c>maxOrder - 1</c> tokens, so a step needs no window at all.
    /// The taps that do need history are applied by <see cref="Inference.EngramStream"/>
    /// from its own ring buffer.
    /// </summary>
    /// <param name="site">Engram site index.</param>
    /// <param name="history">Full token history; positions before 0 hash as zero.</param>
    /// <param name="startPosition">First absolute position to produce.</param>
    /// <param name="count">How many positions to produce.</param>
    /// <param name="keys">Filled with the site's keys, [count, D].</param>
    /// <param name="values">Filled with the pre-convolution value projections, [count, D].</param>
    public void EngramRows(
        int site, ReadOnlySpan<int> history, int startPosition, int count, NdArray keys, NdArray values)
    {
        var (orders, heads, subDim) = _cfg.EngramGeometry();
        int tables = orders.Length * heads;
        var weights = _w.Engrams[site];
        var mark = _scratch.Mark;

        var e = _scratch.Take(count, tables * subDim);
        for (int i = 0; i < count; i++)
        {
            int position = startPosition + i;
            var row = e.Row(i);

            for (int oi = 0; oi < orders.Length; oi++)
            {
                // The n-gram is legal only once the token it reaches back to
                // exists — the streaming form of _mask_diag.
                bool legal = position - (orders[oi] - 1) >= 0;
                for (int h = 0; h < heads; h++)
                {
                    int table = oi * heads + h;
                    var slot = row.Slice(table * subDim, subDim);
                    if (!legal)
                    {
                        slot.Clear();
                        continue;
                    }
                    weights.Tables.Row(table,
                        EngramHash.Index(history, position, orders[oi], table, _cfg.EngramSlots), slot);
                }
            }
        }

        weights.KeyProj.Apply(e, keys);
        weights.ValueProj.Apply(e, values);
        _scratch.RewindTo(mark);
    }

    /// <summary>
    /// Gather, project and causally convolve the engram rows, shared by the
    /// batched and cached paths.
    /// </summary>
    private EngramActivations EngramCore(
        ReadOnlySpan<int> tokens, float[] ngramOk, float[][] tapOk, int trim, ITrace? trace)
    {
        var (orders, heads, subDim) = _cfg.EngramGeometry();
        int tables = orders.Length * heads;
        int maxOrder = orders.Max();
        int seqLen = tokens.Length, d = _cfg.DModel;

        var indices = new int[seqLen * tables];
        EngramHash.Fill(tokens, orders.AsSpan(), heads, _cfg.EngramSlots, indices);

        var keys = new NdArray[_w.Engrams.Length];
        var values = new NdArray[_w.Engrams.Length];

        for (int s = 0; s < _w.Engrams.Length; s++)
        {
            var site = _w.Engrams[s];

            // e[t] = concat over tables of table[j][idx] * ngram_ok
            var e = new NdArray(seqLen, tables * subDim);
            for (int t = 0; t < seqLen; t++)
            {
                var row = e.Row(t);
                for (int j = 0; j < tables; j++)
                {
                    float ok = ngramOk[t * tables + j];
                    var slot = row.Slice(j * subDim, subDim);
                    site.Tables.Row(j, indices[t * tables + j], slot);
                    TensorPrimitives.Multiply(slot, ok, slot);
                }
            }

            keys[s] = site.KeyProj.Apply(e);

            // v[t] = sum_j taps[j] * v0[t - j*dilation] * tap_ok[j][t]
            var v0 = site.ValueProj.Apply(e);
            var v = new NdArray(seqLen, d);
            for (int j = 0; j < EngramConstants.ConvTaps; j++)
            {
                var taps = site.Taps.ReadRow(j);
                int shift = j * maxOrder;
                for (int t = shift; t < seqLen; t++)
                {
                    float ok = tapOk[j][t];
                    if (ok == 0f) continue;
                    var dst = v.Row(t);
                    var src = v0.ReadRow(t - shift);
                    for (int c = 0; c < d; c++) dst[c] += taps[c] * src[c] * ok;
                }
            }
            values[s] = v;

            if (trim > 0)
            {
                keys[s] = keys[s].SliceRange(trim, seqLen - trim);
                values[s] = values[s].SliceRange(trim, seqLen - trim);
            }

            trace?.Record($"engram{s}.k", keys[s]);
            trace?.Record($"engram{s}.v", values[s]);
        }

        return new EngramActivations(keys, values);
    }

    /// <summary>
    /// Fold one site's memory into the block input:
    /// <c>u += sigmoid(&lt;rms_unit(u), rms_unit(k)&gt; / sqrt(D)) * v</c>.
    /// Port of the engram branch at the top of <c>Block.__call__</c>.
    /// </summary>
    private void InjectEngram(NdArray u, NdArray keys, NdArray values, ITrace? trace, int layer,
                              ScratchArena scratch)
    {
        int d = _cfg.DModel;
        int seqLen = u.Length / d;
        float scale = 1f / MathF.Sqrt(d);
        var mark = scratch.Mark;

        var uUnit = scratch.Take(seqLen, d);
        u.ReadSpan.CopyTo(uUnit.Span);
        Ops.RmsUnit(uUnit);
        var kUnit = scratch.Take(seqLen, d);
        keys.ReadSpan.CopyTo(kUnit.Span);
        Ops.RmsUnit(kUnit);

        var alpha = scratch.TakeSpan(seqLen);
        for (int t = 0; t < seqLen; t++)
        {
            float dot = TensorPrimitives.Dot(uUnit.ReadRow(t), kUnit.ReadRow(t)) * scale;
            alpha[t] = 1f / (1f + MathF.Exp(-dot));
            Ops.AddScaled(u.Row(t), values.ReadRow(t), alpha[t]);
        }

        if (trace is not null)
        {
            var record = new NdArray(seqLen);
            alpha.CopyTo(record.Span);
            trace.Record($"layer{layer:D2}.engram_alpha", record);
        }
        scratch.RewindTo(mark);
    }

    // ── Pooled heads ─────────────────────────────────────────────────────────

    /// <summary>
    /// Attention-pool a cell grid with learned probe queries.
    /// Port of <c>probe_pool</c>.  Returns [probes * D].
    /// </summary>
    /// <param name="cells">Cell grid [T, L+1, D].</param>
    /// <param name="probes">Probe queries [P, D].</param>
    /// <param name="keep">Per-position keep flags [T]; positions ≤ 0 are excluded.</param>
    public static NdArray ProbePool(NdArray cells, NdArray probes, ReadOnlySpan<float> keep)
    {
        int seqLen = cells.Shape[0], perToken = cells.Shape[1], d = cells.Shape[2];
        int probeCount = probes.Shape[0];
        int cellCount = seqLen * perToken;
        float scale = 1f / MathF.Sqrt(d);

        var flat = cells.Reshape(cellCount, d);
        var pooled = new NdArray(probeCount * d);
        var scores = new float[cellCount];

        for (int p = 0; p < probeCount; p++)
        {
            var probe = probes.ReadRow(p);
            for (int c = 0; c < cellCount; c++)
            {
                // Cells are laid out token-major, so cell c belongs to token c / perToken.
                bool live = keep.IsEmpty || keep[c / perToken] > 0f;
                scores[c] = live ? TensorPrimitives.Dot(flat.ReadRow(c), probe) * scale
                                 : float.NegativeInfinity;
            }
            Ops.Softmax(scores);

            var slot = pooled.Span.Slice(p * d, d);
            for (int c = 0; c < cellCount; c++)
            {
                float weight = scores[c];
                if (weight != 0f) Ops.AddScaled(slot, flat.ReadRow(c), weight);
            }
        }
        return pooled;
    }

    /// <summary>
    /// L2-normalised contrastive embedding used for tool retrieval.
    /// Port of <c>encode_contrastive</c>.
    /// </summary>
    /// <param name="tokens">Token IDs [T].</param>
    /// <param name="window">Sliding-window width; 0 disables windowing.</param>
    public NdArray EncodeContrastive(ReadOnlySpan<int> tokens, int window = 0)
    {
        var head = _w.Contrastive
            ?? throw new InvalidOperationException("This checkpoint has no contrastive head.");

        var mask = BuildHeadMask(tokens, window);
        var result = Forward(tokens, mask, collectCells: true);
        var pooled = ProbePool(result.Cells!, head.Probes, KeepFlags(tokens));

        var embedding = head.Proj.Apply(pooled.Reshape(1, pooled.Length)).Reshape(_cfg.ContrastiveDim);
        var span = embedding.Span;
        float norm = MathF.Sqrt(TensorPrimitives.Dot(span, span) + 1e-12f);
        Ops.Scale(span, 1f / norm);
        return embedding;
    }

    /// <summary>
    /// Calibrated confidence logit for a prompt-plus-call sequence.  Port of
    /// <c>forward_confidence</c>; apply a sigmoid for a probability.
    /// </summary>
    public float ConfidenceLogit(ReadOnlySpan<int> tokens, int window = 0)
    {
        var head = _w.Confidence
            ?? throw new InvalidOperationException("This checkpoint has no confidence head.");

        var mask = BuildHeadMask(tokens, window);
        var result = Forward(tokens, mask, collectCells: true);
        var pooled = ProbePool(result.Cells!, head.Probes, KeepFlags(tokens));

        return head.Proj.Apply(pooled.Reshape(1, pooled.Length))[0] + head.Bias;
    }

    /// <summary>Sigmoid of a scalar.</summary>
    public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>The heads mask out padding, so build the padding rule from the tokens.</summary>
    private SequenceMask BuildHeadMask(ReadOnlySpan<int> tokens, int window)
    {
        var valid = new bool[tokens.Length];
        for (int t = 0; t < tokens.Length; t++) valid[t] = tokens[t] != _cfg.PadTokenId;
        return new SequenceMask(tokens.Length, window, valid);
    }

    /// <summary>Per-position keep flags used by the pooled heads.</summary>
    private float[] KeepFlags(ReadOnlySpan<int> tokens)
    {
        var keep = new float[tokens.Length];
        for (int t = 0; t < tokens.Length; t++) keep[t] = tokens[t] != _cfg.PadTokenId ? 1f : 0f;
        return keep;
    }
}

/// <summary>
/// Sink for intermediate tensors, so the parity harness can compare stage by
/// stage instead of only at the logits.
/// </summary>
public interface ITrace
{
    /// <summary>Record a named intermediate.  Implementations copy what they keep.</summary>
    void Record(string name, NdArray value);
}
