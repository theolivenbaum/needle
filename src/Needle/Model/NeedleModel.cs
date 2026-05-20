using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;

namespace Needle.Model;

// ---------------------------------------------------------------------------
// Initialisation helpers
// ---------------------------------------------------------------------------

internal static class Init
{
    /// <summary>Default weight initialisation: N(0, 0.02).</summary>
    public static void Default(Tensor t) =>
        nn.init.normal_(t, mean: 0.0, std: 0.02);

    /// <summary>
    /// Residual weight initialisation: N(0, 0.02 / sqrt(2 * numLayers)).
    /// Scales residual branch weights down to keep activations stable at initialisation.
    /// </summary>
    public static void Residual(Tensor t, int numLayers) =>
        nn.init.normal_(t, mean: 0.0, std: 0.02 / System.Math.Sqrt(2.0 * numLayers));
}

// ---------------------------------------------------------------------------
// ZCRMSNorm — Zero-centred RMSNorm
// ---------------------------------------------------------------------------

/// <summary>
/// Zero-centred RMSNorm: <c>out = (1 + γ) * x / rms(x)</c>.
/// γ (scale) is initialised to 0, so the layer starts as a plain RMS normalisation.
/// </summary>
public sealed class ZCRMSNorm : Module<Tensor, Tensor>
{
    private readonly float _epsilon;

    /// <summary>Learnable per-channel scale, initialised to 0.</summary>
    public readonly Parameter scale;

    public ZCRMSNorm(int dim, float epsilon = 1e-6f, string name = "ZCRMSNorm")
        : base(name)
    {
        _epsilon = epsilon;
        scale = Parameter(zeros(dim));
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        // Cast to float32 for the RMS computation for numerical stability.
        using var xf = x.to(ScalarType.Float32);

        // rms = sqrt(mean(x^2) + eps), keepdim over the last axis.
        using var sq  = xf.pow(2f);
        using var msq = sq.mean(new long[] { -1L }, keepdim: true);
        using var rms = (msq + _epsilon).sqrt();

        // (1 + scale) * x / rms  — scale broadcasts over all leading dimensions.
        using var scaleBcast = (1f + scale).to(x.dtype);
        using var norm       = xf / rms;
        return (scaleBcast * norm).to(x.dtype);
    }
}

// ---------------------------------------------------------------------------
// MultiHeadAttention (GQA + RoPE)
// ---------------------------------------------------------------------------

/// <summary>
/// Multi-head attention with grouped query attention (GQA) and optional RoPE.
/// Includes per-head ZCRMSNorm on queries and keys (QK-norm).
/// </summary>
public sealed class MultiHeadAttention : Module<Tensor, Tensor>
{
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _repeats;
    private readonly bool _ropeKeysOnly;

    public readonly Linear    q_proj;
    public readonly Linear    k_proj;
    public readonly Linear    v_proj;
    public readonly Linear    out_proj;
    public readonly ZCRMSNorm q_norm;
    public readonly ZCRMSNorm k_norm;

    public MultiHeadAttention(
        int  numHeads,
        int  numKvHeads,
        int  dModel,
        int  numLayers,
        bool ropeKeysOnly = false,
        string name       = "MultiHeadAttention")
        : base(name)
    {
        _numHeads    = numHeads;
        _numKvHeads  = numKvHeads;
        _headDim     = dModel / numHeads;
        _repeats     = numHeads / numKvHeads;
        _ropeKeysOnly = ropeKeysOnly;

        int kvDim = numKvHeads * _headDim;

        q_proj   = Linear(dModel, dModel, hasBias: false);
        k_proj   = Linear(dModel, kvDim,  hasBias: false);
        v_proj   = Linear(dModel, kvDim,  hasBias: false);
        out_proj = Linear(dModel, dModel, hasBias: false);

        q_norm = new ZCRMSNorm(_headDim, name: "q_norm");
        k_norm = new ZCRMSNorm(_headDim, name: "k_norm");

        Init.Default(q_proj.weight!);
        Init.Default(k_proj.weight!);
        Init.Default(v_proj.weight!);
        Init.Residual(out_proj.weight!, numLayers);

        RegisterComponents();
    }

    // The typed Module base only supports a single Tensor input/output; we
    // expose the real entry-point as a plain method and forward the stub.
    public override Tensor forward(Tensor input) =>
        throw new InvalidOperationException("Use the explicit Call() overload.");

    /// <summary>
    /// Attention forward.
    /// </summary>
    /// <param name="qInput">Query input [B, T_q, D].</param>
    /// <param name="kvInput">Key/value input [B, T_kv, D].</param>
    /// <param name="mask">Optional bool mask [B, 1, T_q, T_kv]; true = attend.</param>
    /// <param name="rope">Optional (cos, sin) RoPE tables.</param>
    public Tensor Call(
        Tensor  qInput,
        Tensor  kvInput,
        Tensor? mask,
        (Tensor cos, Tensor sin)? rope)
    {
        long B = qInput.shape[0];

        using var q = q_proj.forward(qInput);
        using var k = k_proj.forward(kvInput);
        using var v = v_proj.forward(kvInput);

        // Reshape → [B, heads, T, headDim]
        using var qr = q.reshape(B, -1, _numHeads,   _headDim).transpose(1, 2);
        using var kr = k.reshape(B, -1, _numKvHeads, _headDim).transpose(1, 2);
        using var vr = v.reshape(B, -1, _numKvHeads, _headDim).transpose(1, 2);

        // QK-norm
        using var qn = q_norm.forward(qr);
        using var kn = k_norm.forward(kr);

        // GQA: repeat KV heads to match Q heads
        Tensor kExp, vExp;
        if (_repeats > 1)
        {
            kExp = kn.repeat_interleave(_repeats, dim: 1);
            vExp = vr.repeat_interleave(_repeats, dim: 1);
        }
        else
        {
            kExp = kn.alias();
            vExp = vr.alias();
        }

        // RoPE application
        Tensor qFinal, kFinal;
        if (rope.HasValue)
        {
            var (cos, sin) = rope.Value;
            kFinal = RoPE.ApplyRope(kExp, cos, sin);
            qFinal = _ropeKeysOnly ? qn.alias() : RoPE.ApplyRope(qn, cos, sin);
        }
        else
        {
            qFinal = qn.alias();
            kFinal = kExp.alias();
        }

        // Scaled dot-product: [B, H, T_q, T_kv]
        float scale = MathF.Sqrt(_headDim);
        using var kt     = kFinal.transpose(2, 3);
        using var scores = torch.matmul(qFinal, kt) / scale;

        Tensor attnWeights;
        if (mask is not null)
        {
            // Use the dtype's most negative *finite* value rather than -inf, so
            // a fully-masked row (every position blocked, e.g. a padded query
            // slot in a packed batch) does not produce a NaN softmax row that
            // would then poison the encoder output via matmul.  Matches Python:
            //   attn_weights = jnp.where(mask, scores, jnp.finfo(dtype).min)
            float negFill = scores.dtype == ScalarType.Float64
                ? (float)double.MinValue
                : float.MinValue;
            using var negFinite = torch.full(scores.shape, negFill,
                                             dtype: scores.dtype, device: scores.device);
            using var masked = torch.where(mask, scores, negFinite);
            attnWeights = torch.softmax(masked, dim: -1);
        }
        else
        {
            attnWeights = torch.softmax(scores, dim: -1);
        }

        using var attnOut = torch.matmul(attnWeights, vExp);
        attnWeights.Dispose();
        kExp.Dispose();
        vExp.Dispose();
        qFinal.Dispose();
        kFinal.Dispose();

        // Merge heads → [B, T_q, D]
        using var merged = attnOut.transpose(1, 2).reshape(B, -1, _numHeads * _headDim);
        return out_proj.forward(merged);
    }
}

// ---------------------------------------------------------------------------
// FeedForward
// ---------------------------------------------------------------------------

/// <summary>
/// Gated feed-forward network supporting drelu, swiglu, and geglu activations.
/// </summary>
public sealed class FeedForward : Module<Tensor, Tensor>
{
    private readonly string _activation;

    public readonly Linear gate_proj;
    public readonly Linear up_proj;
    public readonly Linear down_proj;

    public FeedForward(
        int    dModel,
        int    dFf,
        int    numLayers,
        string activation = "drelu",
        string name       = "FeedForward")
        : base(name)
    {
        _activation = activation;

        gate_proj = Linear(dModel, dFf,   hasBias: false);
        up_proj   = Linear(dModel, dFf,   hasBias: false);
        down_proj = Linear(dFf,   dModel, hasBias: false);

        Init.Default(gate_proj.weight!);
        Init.Default(up_proj.weight!);
        Init.Residual(down_proj.weight!, numLayers);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x) => Call(x, ffnMask: null);

    /// <summary>
    /// Feed-forward with an optional per-batch FFN mask.
    /// </summary>
    /// <param name="x">Input [B, T, D].</param>
    /// <param name="ffnMask">
    /// Optional [B, dFf] float mask multiplied into the gated hidden activations
    /// before the down projection.  Used by matryoshka FFN training.
    /// </param>
    public Tensor Call(Tensor x, Tensor? ffnMask)
    {
        using var gate = gate_proj.forward(x);
        using var up   = up_proj.forward(x);

        Tensor h;
        switch (_activation)
        {
            case "swiglu":
            {
                using var silu = F.silu(gate);
                h = silu * up;
                break;
            }
            case "geglu":
            {
                using var gelu = F.gelu(gate);
                h = gelu * up;
                break;
            }
            default: // "drelu" — double ReLU gate
            {
                using var rg = F.relu(gate);
                using var ru = F.relu(up);
                h = rg * ru;
                break;
            }
        }

        if (ffnMask is not null)
        {
            // ffnMask: [B, dFf] → [B, 1, dFf] broadcasts across the T axis.
            using var maskCast = ffnMask.to(h.dtype);
            using var maskB    = maskCast.unsqueeze(1);
            var product = h * maskB;
            h.Dispose();
            h = product;
        }

        using var hD = h;
        return down_proj.forward(h);
    }
}

// ---------------------------------------------------------------------------
// EncoderBlock
// ---------------------------------------------------------------------------

/// <summary>
/// A single encoder block: pre-norm self-attention with a gated residual,
/// optionally followed by a pre-norm feed-forward with a gated residual.
/// </summary>
public sealed class EncoderBlock : Module<Tensor, Tensor>
{
    private readonly bool _noFeedforward;

    public readonly ZCRMSNorm          attn_norm;
    public readonly MultiHeadAttention self_attn;
    public readonly Parameter          attn_gate;

    // Optional FFN sublayer fields
    public readonly ZCRMSNorm?   ffn_norm;
    public readonly FeedForward? ffn;
    public readonly Parameter?   ffn_gate;

    public EncoderBlock(
        int    numHeads,
        int    numKvHeads,
        int    dModel,
        int    dFf,
        int    numLayers,
        string activation  = "drelu",
        bool   noFeedforward = true,
        string name        = "EncoderBlock")
        : base(name)
    {
        _noFeedforward = noFeedforward;

        attn_norm = new ZCRMSNorm(dModel, name: "attn_norm");
        self_attn = new MultiHeadAttention(numHeads, numKvHeads, dModel, numLayers, name: "self_attn");
        attn_gate = Parameter(zeros(1));

        if (!noFeedforward)
        {
            ffn_norm = new ZCRMSNorm(dModel, name: "ffn_norm");
            ffn      = new FeedForward(dModel, dFf, numLayers, activation, name: "ffn");
            ffn_gate = Parameter(zeros(1));
        }

        RegisterComponents();
    }

    // Stub — real logic in Call()
    public override Tensor forward(Tensor input) =>
        throw new InvalidOperationException("Use the explicit Call() overload.");

    public Tensor Call(
        Tensor x,
        Tensor? mask,
        (Tensor cos, Tensor sin)? rope,
        Tensor? ffnMask = null)
    {
        // Self-attention sublayer with scalar gated residual
        using var g1      = torch.sigmoid(attn_gate).squeeze();
        using var normed1 = attn_norm.forward(x);
        using var attnOut = self_attn.Call(normed1, normed1, mask, rope);
        var h = x + g1 * attnOut;

        if (_noFeedforward || ffn is null)
            return h;

        using var g2      = torch.sigmoid(ffn_gate!).squeeze();
        using var normed2 = ffn_norm!.forward(h);
        using var ffnOut  = ffn.Call(normed2, ffnMask);
        using var hD      = h;
        return hD + g2 * ffnOut;
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// <summary>
/// Stack of <see cref="EncoderBlock"/>s followed by a final ZCRMSNorm.
/// </summary>
public sealed class Encoder : Module<Tensor, Tensor>
{
    private readonly ModuleList<EncoderBlock> _layers;
    public readonly ZCRMSNorm final_norm;

    public Encoder(TransformerConfig cfg, string name = "Encoder") : base(name)
    {
        _layers = new ModuleList<EncoderBlock>();
        for (int i = 0; i < cfg.NumEncoderLayers; i++)
        {
            _layers.Add(new EncoderBlock(
                cfg.NumHeads, cfg.NumKvHeads, cfg.DModel, cfg.DFf,
                cfg.TotalLayers, cfg.Activation, cfg.NoFeedforward,
                name: $"layer_{i}"));
        }
        final_norm = new ZCRMSNorm(cfg.DModel, name: "final_norm");
        RegisterComponents();
    }

    public override Tensor forward(Tensor input) =>
        throw new InvalidOperationException("Use the explicit Call() overload.");

    public (Tensor output, Tensor? mask) Call(
        Tensor x,
        Tensor? mask,
        (Tensor cos, Tensor sin)? rope,
        Tensor? ffnMask = null)
    {
        foreach (var layer in _layers)
        {
            var next = layer.Call(x, mask, rope, ffnMask);
            x.Dispose();
            x = next;
        }
        return (final_norm.forward(x), mask);
    }
}

// ---------------------------------------------------------------------------
// DecoderBlock
// ---------------------------------------------------------------------------

/// <summary>
/// A single decoder block: gated self-attention + gated cross-attention,
/// optionally followed by a gated feed-forward sublayer.
/// </summary>
public sealed class DecoderBlock : Module<Tensor, Tensor>
{
    private readonly bool _noFeedforward;

    public readonly ZCRMSNorm          self_attn_norm;
    public readonly MultiHeadAttention self_attn;
    public readonly Parameter          self_attn_gate;

    public readonly ZCRMSNorm          cross_attn_norm;
    public readonly MultiHeadAttention cross_attn;
    public readonly Parameter          cross_attn_gate;

    public readonly ZCRMSNorm?   ffn_norm;
    public readonly FeedForward? ffn;
    public readonly Parameter?   ffn_gate;

    public DecoderBlock(
        int    numHeads,
        int    numKvHeads,
        int    dModel,
        int    dFf,
        int    numLayers,
        string activation  = "drelu",
        bool   noFeedforward = true,
        string name        = "DecoderBlock")
        : base(name)
    {
        _noFeedforward = noFeedforward;

        self_attn_norm  = new ZCRMSNorm(dModel, name: "self_attn_norm");
        self_attn       = new MultiHeadAttention(numHeads, numKvHeads, dModel, numLayers, name: "self_attn");
        self_attn_gate  = Parameter(zeros(1));

        // Cross-attention uses rope only on keys, not queries (rope_keys_only=true)
        cross_attn_norm = new ZCRMSNorm(dModel, name: "cross_attn_norm");
        cross_attn      = new MultiHeadAttention(numHeads, numKvHeads, dModel, numLayers,
                                                  ropeKeysOnly: true, name: "cross_attn");
        cross_attn_gate = Parameter(zeros(1));

        if (!noFeedforward)
        {
            ffn_norm = new ZCRMSNorm(dModel, name: "ffn_norm");
            ffn      = new FeedForward(dModel, dFf, numLayers, activation, name: "ffn");
            ffn_gate = Parameter(zeros(1));
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input) =>
        throw new InvalidOperationException("Use the explicit Call() overload.");

    /// <param name="x">Decoder hidden states [B, T_dec, D].</param>
    /// <param name="encoderOut">Encoder output [B, T_enc, D].</param>
    /// <param name="selfMask">Causal/padding mask [B, 1, T_dec, T_dec].</param>
    /// <param name="crossMask">Cross-attention mask [B, 1, T_dec, T_enc].</param>
    /// <param name="rope">RoPE tables for the self-attention sublayer.</param>
    public Tensor Call(
        Tensor  x,
        Tensor  encoderOut,
        Tensor? selfMask,
        Tensor? crossMask,
        (Tensor cos, Tensor sin)? rope,
        Tensor? ffnMask = null)
    {
        // Self-attention
        using var sg  = torch.sigmoid(self_attn_gate).squeeze();
        using var sn  = self_attn_norm.forward(x);
        using var sa  = self_attn.Call(sn, sn, selfMask, rope);
        var h = x + sg * sa;

        // Cross-attention (no RoPE on queries)
        using var cg  = torch.sigmoid(cross_attn_gate).squeeze();
        using var cn  = cross_attn_norm.forward(h);
        using var ca  = cross_attn.Call(cn, encoderOut, crossMask, null);
        using var hD1 = h;
        h = hD1 + cg * ca;

        if (_noFeedforward || ffn is null)
            return h;

        using var fg  = torch.sigmoid(ffn_gate!).squeeze();
        using var fn2 = ffn_norm!.forward(h);
        using var fo  = ffn.Call(fn2, ffnMask);
        using var hD2 = h;
        return hD2 + fg * fo;
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// <summary>
/// Stack of <see cref="DecoderBlock"/>s followed by a final ZCRMSNorm.
/// </summary>
public sealed class Decoder : Module<Tensor, Tensor>
{
    private readonly ModuleList<DecoderBlock> _layers;
    public readonly ZCRMSNorm final_norm;

    public Decoder(TransformerConfig cfg, string name = "Decoder") : base(name)
    {
        _layers = new ModuleList<DecoderBlock>();
        for (int i = 0; i < cfg.NumDecoderLayers; i++)
        {
            _layers.Add(new DecoderBlock(
                cfg.NumHeads, cfg.NumKvHeads, cfg.DModel, cfg.DFf,
                cfg.TotalLayers, cfg.Activation, cfg.NoFeedforward,
                name: $"layer_{i}"));
        }
        final_norm = new ZCRMSNorm(cfg.DModel, name: "final_norm");
        RegisterComponents();
    }

    public override Tensor forward(Tensor input) =>
        throw new InvalidOperationException("Use the explicit Call() overload.");

    public Tensor Call(
        Tensor  x,
        Tensor  encoderOut,
        Tensor? selfMask,
        Tensor? crossMask,
        (Tensor cos, Tensor sin)? rope,
        Tensor? ffnMask = null)
    {
        foreach (var layer in _layers)
        {
            var next = layer.Call(x, encoderOut, selfMask, crossMask, rope, ffnMask);
            x.Dispose();
            x = next;
        }
        return final_norm.forward(x);
    }
}

// ---------------------------------------------------------------------------
// SimpleAttentionNetwork
// ---------------------------------------------------------------------------

/// <summary>
/// Encoder-decoder transformer with:
/// <list type="bullet">
///   <item>Shared input/output embeddings (tied weights)</item>
///   <item>Rotary position embeddings (RoPE)</item>
///   <item>Grouped query attention (GQA)</item>
///   <item>Zero-centred RMSNorm throughout</item>
///   <item>Contrastive encoding head</item>
/// </list>
/// Port of <c>SimpleAttentionNetwork</c> from architecture.py.
/// </summary>
public sealed class SimpleAttentionNetwork : Module<Tensor, Tensor>
{
    private readonly TransformerConfig _cfg;
    private readonly float             _embedScale;

    // Per-device RoPE cache. Keyed by device descriptor; entry holds a (cos, sin)
    // table sized to at least the longest sequence seen so far on that device.
    // ApplyRope slices to the current T, so a larger table is always usable.
    private readonly Dictionary<string, (Tensor cos, Tensor sin, long maxLen)> _ropeCache
        = new();

    public readonly Embedding embedding;
    public readonly Encoder   encoder;
    public readonly Decoder   decoder;
    public readonly Linear    contrastive_hidden;
    public readonly Linear    contrastive_proj;

    /// <summary>Learnable log-temperature for contrastive loss.</summary>
    public readonly Parameter log_temp;

    public SimpleAttentionNetwork(TransformerConfig cfg, string name = "SimpleAttentionNetwork")
        : base(name)
    {
        _cfg        = cfg;
        _embedScale = MathF.Sqrt(cfg.DModel);

        embedding = nn.Embedding(cfg.VocabSize, cfg.DModel);
        nn.init.normal_(embedding.weight!, std: 0.02);

        encoder = new Encoder(cfg, name: "encoder");
        decoder = new Decoder(cfg, name: "decoder");

        contrastive_hidden = Linear(cfg.DModel, cfg.DModel / 4, hasBias: true);
        contrastive_proj   = Linear(cfg.DModel / 4, cfg.ContrastiveDim, hasBias: false);

        Init.Default(contrastive_hidden.weight!);
        Init.Default(contrastive_proj.weight!);

        log_temp = Parameter(zeros(1));

        RegisterComponents();
    }

    // Stub — use the typed API methods below.
    public override Tensor forward(Tensor input) =>
        throw new InvalidOperationException("Use Forward(), EncodeText(), or Decode() instead.");

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            foreach (var (_, value) in _ropeCache)
            {
                value.cos.Dispose();
                value.sin.Dispose();
            }
            _ropeCache.Clear();
        }
        base.Dispose(disposing);
    }

    // ── RoPE helper ───────────────────────────────────────────────────────────

    /// <summary>
    /// Return cached (cos, sin) RoPE tables for the given device.  The cache
    /// stores tables sized to the largest <paramref name="seqLen"/> seen so far
    /// on that device; <see cref="RoPE.ApplyRope"/> slices to the actual
    /// sequence length so a larger table is always usable.
    ///
    /// Ownership: the returned tensors are owned by the cache.  Callers MUST
    /// NOT dispose them.
    /// </summary>
    private (Tensor cos, Tensor sin) GetRope(long seqLen, Device? device = null)
    {
        device ??= embedding.weight!.device;
        string key = $"{device.type}:{device.index}";

        if (_ropeCache.TryGetValue(key, out var entry) && entry.maxLen >= seqLen)
            return (entry.cos, entry.sin);

        // Grow the table to at least MaxSeqLen so most subsequent calls hit the
        // cache.  If seqLen exceeds that (long-context inference), use seqLen.
        long newMaxLen = System.Math.Max(seqLen, _cfg.MaxSeqLen);

        var (cos, sin) = RoPE.PrecomputeFreqs(_cfg.HeadDim, (int)newMaxLen, _cfg.RopeTheta);
        var cosD = cos.to(device); cos.Dispose();
        var sinD = sin.to(device); sin.Dispose();

        if (_ropeCache.TryGetValue(key, out var old))
        {
            old.cos.Dispose();
            old.sin.Dispose();
        }
        _ropeCache[key] = (cosD, sinD, newMaxLen);
        return (cosD, sinD);
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// <summary>
    /// Encode a source token sequence through the encoder stack.
    /// </summary>
    /// <param name="src">Token IDs [B, T_enc] (int64).</param>
    /// <param name="srcMask">Optional attention mask [B, 1, 1, T_enc] (bool).</param>
    /// <param name="ffnMask">
    /// Optional per-batch-item FFN activation mask [B, dFf] (float).  When given
    /// it is multiplied into the FFN hidden activations of every encoder block —
    /// the matryoshka-FFN path used by <see cref="ForwardMasked"/>.
    /// </param>
    /// <returns>(encoderOutput [B, T_enc, D], mask)</returns>
    public (Tensor encoderOut, Tensor? encMask) EncodeText(
        Tensor src,
        Tensor? srcMask = null,
        Tensor? ffnMask = null)
    {
        using var emb  = embedding.forward(src);
        using var embS = emb * _embedScale;
        var rope   = GetRope(src.shape[1], src.device);
        return encoder.Call(embS, srcMask, rope, ffnMask);
    }

    /// <summary>
    /// Run the decoder and return logits over the vocabulary.
    /// </summary>
    /// <param name="tgt">Target token IDs [B, T_dec] (int64).</param>
    /// <param name="encoderOut">Encoder hidden states [B, T_enc, D].</param>
    /// <param name="selfMask">Optional causal mask [B, 1, T_dec, T_dec] (bool).</param>
    /// <param name="crossMask">Optional cross-attention mask [B, 1, T_dec, T_enc] (bool).</param>
    /// <param name="ffnMask">
    /// Optional per-batch-item FFN activation mask [B, dFf] (float).
    /// </param>
    /// <returns>Logits [B, T_dec, VocabSize] (float32).</returns>
    public Tensor Decode(
        Tensor  tgt,
        Tensor  encoderOut,
        Tensor? selfMask  = null,
        Tensor? crossMask = null,
        Tensor? ffnMask   = null)
    {
        using var emb    = embedding.forward(tgt);
        using var embS   = emb * _embedScale;
        var rope         = GetRope(tgt.shape[1], tgt.device);
        using var hidden = decoder.Call(embS, encoderOut, selfMask, crossMask, rope, ffnMask);

        // Tied output projection: logits = hidden @ embedding.weight^T
        using var hidF = hidden.to(ScalarType.Float32);
        using var wT   = embedding.weight!.to(ScalarType.Float32).t();
        return torch.matmul(hidF, wT);
    }

    /// <summary>
    /// Encode tokens to a normalised contrastive embedding.
    /// Pipeline: encode → mean-pool → ReLU hidden → proj → L2-norm.
    /// </summary>
    /// <param name="tokens">Token IDs [B, T] (int64).</param>
    /// <returns>L2-normalised embeddings [B, ContrastiveDim] (float32).</returns>
    public Tensor EncodeContrastive(Tensor tokens)
    {
        using var srcMask        = MaskUtils.MakePaddingMask(tokens, _cfg.PadTokenId);
        var (encOut, encMask)    = EncodeText(tokens, srcMask);
        using var encOutD        = encOut;

        using var pooled   = MeanPool(encOut, encMask);
        using var hRelu    = F.relu(contrastive_hidden.forward(pooled));
        using var proj     = contrastive_proj.forward(hRelu);
        using var projF    = proj.to(ScalarType.Float32);
        using var sq       = projF.pow(2f);
        using var sumSq    = sq.sum(new long[] { -1L }, keepdim: true);
        using var denom    = (sumSq + 1e-12f).sqrt();
        return (projF / denom).to(proj.dtype);
    }

    /// <summary>
    /// Full encoder-decoder forward pass returning logits [B, T_dec, VocabSize].
    /// </summary>
    public Tensor Forward(
        Tensor  src,
        Tensor  tgt,
        Tensor? srcMask   = null,
        Tensor? tgtMask   = null,
        Tensor? crossMask = null)
    {
        var (encoderOut, encMask) = EncodeText(src, srcMask);
        using var encD = encoderOut;
        var cm = crossMask ?? encMask;
        return Decode(tgt, encoderOut, tgtMask, cm);
    }

    /// <summary>
    /// Encoder-decoder forward pass with a per-batch FFN mask applied to both
    /// stacks.  Used by matryoshka-FFN training.  Port of
    /// <c>SimpleAttentionNetwork.forward_masked</c> in architecture.py.
    /// </summary>
    /// <param name="src">Source token IDs [B, T_enc] (int64).</param>
    /// <param name="tgt">Target token IDs [B, T_dec] (int64).</param>
    /// <param name="srcMask">Optional encoder attention mask.</param>
    /// <param name="tgtMask">Optional decoder self-attention mask.</param>
    /// <param name="crossMask">Optional decoder cross-attention mask.</param>
    /// <param name="ffnMask">
    /// Per-batch FFN activation mask [B, dFf].  Required.
    /// </param>
    /// <returns>Logits [B, T_dec, VocabSize] (float32).</returns>
    public Tensor ForwardMasked(
        Tensor  src,
        Tensor  tgt,
        Tensor  ffnMask,
        Tensor? srcMask   = null,
        Tensor? tgtMask   = null,
        Tensor? crossMask = null)
    {
        var (encoderOut, encMask) = EncodeText(src, srcMask, ffnMask);
        using var encD = encoderOut;
        var cm = crossMask ?? encMask;
        return Decode(tgt, encoderOut, tgtMask, cm, ffnMask);
    }

    /// <summary>
    /// Contrastive twin-encoder forward.  Encodes <paramref name="queryTokens"/>
    /// and <paramref name="toolTokens"/> independently and returns their
    /// L2-normalised projections together with the learnable log-temperature.
    /// Port of <c>SimpleAttentionNetwork.forward_contrastive</c>.
    /// </summary>
    public (Tensor qEmb, Tensor tEmb, Tensor logTemp) ForwardContrastive(
        Tensor queryTokens,
        Tensor toolTokens)
    {
        var qEmb = EncodeContrastive(queryTokens);
        var tEmb = EncodeContrastive(toolTokens);
        return (qEmb, tEmb, log_temp);
    }

    /// <summary>
    /// Build the default prefix FFN mask for matryoshka eval at a single
    /// FFN width.  Sets the first <paramref name="ffWidth"/> neurons to 1.0
    /// and the rest to 0.0, broadcast across the batch.
    /// Port of <c>_make_eval_ffn_mask</c> in architecture.py.
    /// </summary>
    /// <param name="ffWidth">Number of FFN neurons to keep active.</param>
    /// <param name="batchSize">Batch dimension to broadcast to.</param>
    /// <param name="device">Optional device for the returned tensor.</param>
    public Tensor MakeEvalFfnMask(int ffWidth, long batchSize, Device? device = null)
    {
        if (ffWidth < 0 || ffWidth > _cfg.DFf)
            throw new ArgumentOutOfRangeException(nameof(ffWidth),
                $"ffWidth must be in [0, {_cfg.DFf}].");

        using var arange = torch.arange(_cfg.DFf, device: device ?? embedding.weight!.device);
        using var lt     = arange.lt(ffWidth);
        using var asF    = lt.to(ScalarType.Float32);
        return asF.unsqueeze(0).expand(batchSize, _cfg.DFf).contiguous();
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// <summary>
    /// Mean-pool encoder output over non-padded positions → [B, D].
    /// </summary>
    private static Tensor MeanPool(Tensor encoderOut, Tensor? encMask)
    {
        if (encMask is not null)
        {
            // encMask: [B, 1, 1, T] → [B, T]
            using var mask2d  = encMask.squeeze(1).squeeze(1);
            using var maskF   = mask2d.to(encoderOut.dtype);
            using var mask3d  = maskF.unsqueeze(2);
            using var masked  = encoderOut * mask3d;
            using var summed  = masked.sum(new long[] { 1L });
            using var counts  = maskF.sum(new long[] { 1L }, keepdim: true);
            using var safe    = counts.clamp_min(1.0f);
            return summed / safe;
        }
        else
        {
            return encoderOut.mean(new long[] { 1L });
        }
    }
}
