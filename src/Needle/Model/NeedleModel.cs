using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

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
    /// Scales down residual branch weights to keep activations stable at init.
    /// </summary>
    public static void Residual(Tensor t, int numLayers) =>
        nn.init.normal_(t, mean: 0.0, std: 0.02 / Math.Sqrt(2.0 * numLayers));
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
        // Cast to float32 for the RMS computation regardless of input dtype.
        using var xf = x.to(ScalarType.Float32);

        // rms = sqrt(mean(x^2) + eps) over last dimension, keep dims
        using var sq  = xf.pow(2);
        using var msq = sq.mean(dim: [-1], keepdim: true);
        using var rms = (msq + _epsilon).sqrt();

        // (1 + scale) * x / rms — scale is broadcast over all leading dims
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
public sealed class MultiHeadAttention : Module<Tensor, Tensor, Tensor?, (Tensor cos, Tensor sin)?, bool, Tensor>
{
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _repeats;
    private readonly bool _ropeKeysOnly;

    public readonly Linear q_proj;
    public readonly Linear k_proj;
    public readonly Linear v_proj;
    public readonly Linear out_proj;
    public readonly ZCRMSNorm q_norm;
    public readonly ZCRMSNorm k_norm;

    public MultiHeadAttention(
        int numHeads,
        int numKvHeads,
        int dModel,
        int numLayers,
        bool ropeKeysOnly = false,
        string name = "MultiHeadAttention")
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

        // Weight initialisation
        Init.Default(q_proj.weight!);
        Init.Default(k_proj.weight!);
        Init.Default(v_proj.weight!);
        Init.Residual(out_proj.weight!, numLayers);

        RegisterComponents();
    }

    /// <summary>
    /// Forward pass.
    /// </summary>
    /// <param name="qInput">Query input [B, T_q, D].</param>
    /// <param name="kvInput">Key/value input [B, T_kv, D].</param>
    /// <param name="mask">Optional attention mask [B, 1, T_q, T_kv], true = attend.</param>
    /// <param name="rope">Optional (cos, sin) RoPE tables.</param>
    /// <param name="training">Whether in training mode (unused here — kept for API symmetry).</param>
    public override Tensor forward(
        Tensor qInput,
        Tensor kvInput,
        Tensor? mask,
        (Tensor cos, Tensor sin)? rope,
        bool training)
    {
        long B = qInput.shape[0];

        // Linear projections
        using var q = q_proj.forward(qInput);   // [B, T_q, D]
        using var k = k_proj.forward(kvInput);  // [B, T_kv, kvDim]
        using var v = v_proj.forward(kvInput);  // [B, T_kv, kvDim]

        // Reshape to [B, heads, T, headDim]
        using var qr = q.reshape(B, -1, _numHeads,   _headDim).transpose(1, 2);
        using var kr = k.reshape(B, -1, _numKvHeads, _headDim).transpose(1, 2);
        using var vr = v.reshape(B, -1, _numKvHeads, _headDim).transpose(1, 2);

        // QK-norm
        using var qn = q_norm.forward(qr);
        using var kn = k_norm.forward(kr);

        // GQA: expand KV heads to match Q heads
        Tensor kExpanded, vExpanded;
        if (_repeats > 1)
        {
            kExpanded = kn.repeat_interleave(_repeats, dim: 1);
            vExpanded = vr.repeat_interleave(_repeats, dim: 1);
        }
        else
        {
            kExpanded = kn.alias();
            vExpanded = vr.alias();
        }

        // RoPE
        Tensor qFinal, kFinal;
        if (rope.HasValue)
        {
            var (cos, sin) = rope.Value;
            kFinal = RoPE.ApplyRope(kExpanded, cos, sin);
            qFinal = _ropeKeysOnly ? qn.alias() : RoPE.ApplyRope(qn, cos, sin);
        }
        else
        {
            qFinal = qn.alias();
            kFinal = kExpanded.alias();
        }

        // Scaled dot-product attention
        float scale = MathF.Sqrt(_headDim);
        using var kt = kFinal.transpose(2, 3);  // [B, H, headDim, T_kv]
        using var scores = torch.matmul(qFinal, kt) / scale; // [B, H, T_q, T_kv]

        Tensor attnWeights;
        if (mask is not null)
        {
            // Where mask is false, fill with -inf so softmax zeroes those positions.
            using var maskFloat = mask.to(scores.dtype);
            using var negInf    = torch.full(scores.shape, float.NegativeInfinity, dtype: scores.dtype, device: scores.device);
            using var masked    = torch.where(mask, scores, negInf);
            attnWeights = torch.softmax(masked, dim: -1);
        }
        else
        {
            attnWeights = torch.softmax(scores, dim: -1);
        }

        // Weighted sum over values
        using var attnOut = torch.matmul(attnWeights, vExpanded); // [B, H, T_q, headDim]
        attnWeights.Dispose();

        // Merge heads: [B, T_q, D]
        using var merged = attnOut.transpose(1, 2).reshape(B, -1, _numHeads * _headDim);

        // Cleanup GQA temporaries
        if (_repeats > 1) { kExpanded.Dispose(); vExpanded.Dispose(); }
        else              { kExpanded.Dispose(); vExpanded.Dispose(); }
        qFinal.Dispose();
        kFinal.Dispose();

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

    public FeedForward(int dModel, int dFf, int numLayers, string activation = "drelu", string name = "FeedForward")
        : base(name)
    {
        _activation = activation;

        gate_proj = Linear(dModel, dFf, hasBias: false);
        up_proj   = Linear(dModel, dFf, hasBias: false);
        down_proj = Linear(dFf, dModel, hasBias: false);

        Init.Default(gate_proj.weight!);
        Init.Default(up_proj.weight!);
        Init.Residual(down_proj.weight!, numLayers);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var gate = gate_proj.forward(x);
        using var up   = up_proj.forward(x);

        Tensor h;
        switch (_activation)
        {
            case "swiglu":
                using var silu = torch.nn.functional.silu(gate);
                h = silu * up;
                break;
            case "geglu":
                using var gelu = torch.nn.functional.gelu(gate);
                h = gelu * up;
                break;
            default: // "drelu"
                using var rg = torch.relu(gate);
                using var ru = torch.relu(up);
                h = rg * ru;
                break;
        }

        using var hDispose = h;
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
public sealed class EncoderBlock : Module<Tensor, Tensor?, (Tensor cos, Tensor sin)?, bool, Tensor>
{
    private readonly bool _noFeedforward;

    public readonly ZCRMSNorm        attn_norm;
    public readonly MultiHeadAttention self_attn;
    public readonly Parameter        attn_gate;

    // Optional FFN sublayer
    public readonly ZCRMSNorm?  ffn_norm;
    public readonly FeedForward? ffn;
    public readonly Parameter?   ffn_gate;

    public EncoderBlock(
        int numHeads,
        int numKvHeads,
        int dModel,
        int dFf,
        int numLayers,
        string activation  = "drelu",
        bool noFeedforward = true,
        string name        = "EncoderBlock")
        : base(name)
    {
        _noFeedforward = noFeedforward;

        attn_norm = new ZCRMSNorm(dModel, name: "attn_norm");
        self_attn = new MultiHeadAttention(numHeads, numKvHeads, dModel, numLayers, name: "self_attn");
        attn_gate = Parameter(zeros(1));  // scalar gate, sigmoid-activated

        if (!noFeedforward)
        {
            ffn_norm = new ZCRMSNorm(dModel, name: "ffn_norm");
            ffn      = new FeedForward(dModel, dFf, numLayers, activation, name: "ffn");
            ffn_gate = Parameter(zeros(1));
        }

        RegisterComponents();
    }

    public override Tensor forward(
        Tensor x,
        Tensor? mask,
        (Tensor cos, Tensor sin)? rope,
        bool training)
    {
        // Self-attention sublayer with gated residual
        var gate = torch.sigmoid(attn_gate).squeeze();
        using var normed   = attn_norm.forward(x);
        using var attnOut  = self_attn.forward(normed, normed, mask, rope, training);
        var residual = x + gate * attnOut;

        if (_noFeedforward || ffn is null)
            return residual;

        // FFN sublayer with gated residual
        var ffnG = torch.sigmoid(ffn_gate!).squeeze();
        using var normed2  = ffn_norm!.forward(residual);
        using var ffnOut   = ffn.forward(normed2);
        using var rDispose = residual;
        return rDispose + ffnG * ffnOut;
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// <summary>
/// Stack of <see cref="EncoderBlock"/>s followed by a final ZCRMSNorm.
/// Returns (output [B, T, D], mask).
/// </summary>
public sealed class Encoder : Module<Tensor, Tensor?, (Tensor cos, Tensor sin)?, bool, (Tensor output, Tensor? mask)>
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

    public override (Tensor output, Tensor? mask) forward(
        Tensor x,
        Tensor? mask,
        (Tensor cos, Tensor sin)? rope,
        bool training)
    {
        foreach (var layer in _layers)
            x = layer.forward(x, mask, rope, training);

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
public sealed class DecoderBlock : Module<Tensor, Tensor, Tensor?, Tensor?, (Tensor cos, Tensor sin)?, bool, Tensor>
{
    private readonly bool _noFeedforward;

    public readonly ZCRMSNorm          self_attn_norm;
    public readonly MultiHeadAttention self_attn;
    public readonly Parameter          self_attn_gate;

    public readonly ZCRMSNorm          cross_attn_norm;
    public readonly MultiHeadAttention cross_attn;
    public readonly Parameter          cross_attn_gate;

    public readonly ZCRMSNorm?  ffn_norm;
    public readonly FeedForward? ffn;
    public readonly Parameter?   ffn_gate;

    public DecoderBlock(
        int numHeads,
        int numKvHeads,
        int dModel,
        int dFf,
        int numLayers,
        string activation  = "drelu",
        bool noFeedforward = true,
        string name        = "DecoderBlock")
        : base(name)
    {
        _noFeedforward = noFeedforward;

        self_attn_norm  = new ZCRMSNorm(dModel, name: "self_attn_norm");
        self_attn       = new MultiHeadAttention(numHeads, numKvHeads, dModel, numLayers, name: "self_attn");
        self_attn_gate  = Parameter(zeros(1));

        cross_attn_norm = new ZCRMSNorm(dModel, name: "cross_attn_norm");
        cross_attn      = new MultiHeadAttention(numHeads, numKvHeads, dModel, numLayers, ropeKeysOnly: true, name: "cross_attn");
        cross_attn_gate = Parameter(zeros(1));

        if (!noFeedforward)
        {
            ffn_norm = new ZCRMSNorm(dModel, name: "ffn_norm");
            ffn      = new FeedForward(dModel, dFf, numLayers, activation, name: "ffn");
            ffn_gate = Parameter(zeros(1));
        }

        RegisterComponents();
    }

    /// <summary>
    /// Decoder block forward.
    /// </summary>
    /// <param name="x">Decoder hidden states [B, T_dec, D].</param>
    /// <param name="encoderOut">Encoder output [B, T_enc, D].</param>
    /// <param name="selfMask">Causal/padding mask [B, 1, T_dec, T_dec].</param>
    /// <param name="crossMask">Cross-attention mask [B, 1, T_dec, T_enc].</param>
    /// <param name="rope">RoPE tables for self-attention (not used in cross-attention).</param>
    /// <param name="training">Training mode flag.</param>
    public override Tensor forward(
        Tensor x,
        Tensor encoderOut,
        Tensor? selfMask,
        Tensor? crossMask,
        (Tensor cos, Tensor sin)? rope,
        bool training)
    {
        // Self-attention
        var selfG = torch.sigmoid(self_attn_gate).squeeze();
        using var sn = self_attn_norm.forward(x);
        using var sa = self_attn.forward(sn, sn, selfMask, rope, training);
        var h = x + selfG * sa;

        // Cross-attention (no RoPE on queries for cross-attn)
        var crossG = torch.sigmoid(cross_attn_gate).squeeze();
        using var cn = cross_attn_norm.forward(h);
        using var ca = cross_attn.forward(cn, encoderOut, crossMask, null, training);
        using var hDispose = h;
        h = hDispose + crossG * ca;

        if (_noFeedforward || ffn is null)
            return h;

        // FFN
        var ffnG = torch.sigmoid(ffn_gate!).squeeze();
        using var fn2 = ffn_norm!.forward(h);
        using var fo  = ffn.forward(fn2);
        using var hd2 = h;
        return hd2 + ffnG * fo;
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// <summary>
/// Stack of <see cref="DecoderBlock"/>s followed by a final ZCRMSNorm.
/// </summary>
public sealed class Decoder : Module<Tensor, Tensor, Tensor?, Tensor?, (Tensor cos, Tensor sin)?, bool, Tensor>
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

    public override Tensor forward(
        Tensor x,
        Tensor encoderOut,
        Tensor? selfMask,
        Tensor? crossMask,
        (Tensor cos, Tensor sin)? rope,
        bool training)
    {
        foreach (var layer in _layers)
            x = layer.forward(x, encoderOut, selfMask, crossMask, rope, training);

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
/// Port of the Python <c>SimpleAttentionNetwork</c> in architecture.py.
/// </summary>
public sealed class SimpleAttentionNetwork : Module<Tensor, Tensor, Tensor?, Tensor?, Tensor?, Tensor>
{
    private readonly TransformerConfig _cfg;
    private readonly float _embedScale;

    // Sub-modules
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

        embedding = Embedding(cfg.VocabSize, cfg.DModel);
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

    // ── RoPE helpers ─────────────────────────────────────────────────────────

    private (Tensor cos, Tensor sin) GetRope(long seqLen, Device? device = null)
    {
        var (cos, sin) = RoPE.PrecomputeFreqs(_cfg.HeadDim, (int)seqLen, _cfg.RopeTheta);
        if (device is not null)
        {
            cos = cos.to(device);
            sin = sin.to(device);
        }
        return (cos, sin);
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// <summary>
    /// Encode a source token sequence.
    /// </summary>
    /// <param name="src">Token IDs [B, T_enc].</param>
    /// <param name="srcMask">Optional padding/packing mask [B, 1, 1, T_enc].</param>
    /// <returns>(encoderOutput [B, T_enc, D], mask)</returns>
    public (Tensor encoderOut, Tensor? encMask) EncodeText(Tensor src, Tensor? srcMask = null)
    {
        using var emb  = embedding.forward(src);
        using var embS = emb * _embedScale;
        var rope = GetRope(src.shape[1], src.device);
        var result = encoder.forward(embS, srcMask, rope, training: false);
        rope.cos.Dispose();
        rope.sin.Dispose();
        return result;
    }

    /// <summary>
    /// Run the decoder and return logits over the vocabulary.
    /// </summary>
    /// <param name="tgt">Target token IDs [B, T_dec].</param>
    /// <param name="encoderOut">Encoder hidden states [B, T_enc, D].</param>
    /// <param name="selfMask">Optional causal mask [B, 1, T_dec, T_dec].</param>
    /// <param name="crossMask">Optional cross-attention mask [B, 1, T_dec, T_enc].</param>
    /// <returns>Logits [B, T_dec, VocabSize].</returns>
    public Tensor Decode(
        Tensor tgt,
        Tensor encoderOut,
        Tensor? selfMask  = null,
        Tensor? crossMask = null)
    {
        using var emb  = embedding.forward(tgt);
        using var embS = emb * _embedScale;
        var rope = GetRope(tgt.shape[1], tgt.device);
        using var hidden = decoder.forward(embS, encoderOut, selfMask, crossMask, rope, training: false);
        rope.cos.Dispose();
        rope.sin.Dispose();

        // Tied output projection: logits = hidden @ embedding.weight^T
        using var hidF = hidden.to(ScalarType.Float32);
        using var wT   = embedding.weight!.to(ScalarType.Float32).t();
        return torch.matmul(hidF, wT);
    }

    /// <summary>
    /// Encode tokens to a normalised contrastive embedding.
    /// Pipeline: mean-pool encoder output → ReLU hidden → linear proj → L2-norm.
    /// </summary>
    /// <param name="tokens">Token IDs [B, T].</param>
    /// <returns>L2-normalised embeddings [B, ContrastiveDim].</returns>
    public Tensor EncodeContrastive(Tensor tokens)
    {
        var srcMask = MaskUtils.MakePaddingMask(tokens, _cfg.PadTokenId);
        var (encOut, encMask) = EncodeText(tokens, srcMask);

        using var pooled   = MeanPool(encOut, encMask);
        using var h        = torch.relu(contrastive_hidden.forward(pooled));
        using var proj     = contrastive_proj.forward(h);
        using var projF    = proj.to(ScalarType.Float32);
        using var sq       = projF.pow(2);
        using var sumsq    = sq.sum(dim: [-1], keepdim: true);
        using var denom    = (sumsq + 1e-12f).sqrt();
        return (projF / denom).to(proj.dtype);
    }

    /// <summary>
    /// Full encoder-decoder forward pass; returns logits [B, T_dec, VocabSize].
    /// </summary>
    public override Tensor forward(
        Tensor src,
        Tensor tgt,
        Tensor? srcMask   = null,
        Tensor? tgtMask   = null,
        Tensor? crossMask = null)
    {
        var (encoderOut, encMask) = EncodeText(src, srcMask);
        var cm = crossMask ?? encMask;
        return Decode(tgt, encoderOut, tgtMask, cm);
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// <summary>
    /// Mean-pool encoder output over non-padded positions.
    /// </summary>
    private static Tensor MeanPool(Tensor encoderOut, Tensor? encMask)
    {
        if (encMask is not null)
        {
            // encMask: [B, 1, 1, T] → squeeze to [B, T]
            using var mask2d   = encMask.squeeze(1).squeeze(1);            // [B, T]
            using var maskF    = mask2d.to(encoderOut.dtype);              // [B, T]
            using var mask3d   = maskF.unsqueeze(2);                       // [B, T, 1]
            using var masked   = encoderOut * mask3d;                      // [B, T, D]
            using var summed   = masked.sum(dim: 1);                       // [B, D]
            using var counts   = maskF.sum(dim: 1, keepdim: true);        // [B, 1]
            using var safe     = torch.clamp(counts, min: 1.0f);
            return summed / safe;
        }
        else
        {
            return encoderOut.mean(dim: [1]);
        }
    }
}
