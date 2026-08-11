using System.Collections.Immutable;
using Needle.Math;

namespace Needle.Model;

/// <summary>Weights of one transformer block.</summary>
/// <param name="NormIn">Pre-attention norm scale [D].</param>
/// <param name="QProj">Query kernel [D, A].</param>
/// <param name="KProj">Key kernel [D, KV].</param>
/// <param name="VProj">Value kernel [D, KV].</param>
/// <param name="GateProj">Output-gate kernel [D, A].</param>
/// <param name="OutProj">Output kernel [A, D].</param>
/// <param name="QNorm">Query QK-norm scale [headDim].</param>
/// <param name="KNorm">Key QK-norm scale [headDim].</param>
/// <param name="PostAttnNorm">Post-attention norm scale [D].</param>
/// <param name="AttnGate">Scalar residual gate (pre-sigmoid).</param>
/// <param name="PreHadaNorm">Pre-MLP norm scale [D].</param>
/// <param name="D1">Hadamard input diagonal [n].</param>
/// <param name="D2">Hadamard hidden diagonal [n].</param>
/// <param name="D3">Hadamard output diagonal [n].</param>
public sealed record BlockWeights(
    NdArray NormIn,
    LinearWeight QProj,
    LinearWeight KProj,
    LinearWeight VProj,
    LinearWeight GateProj,
    LinearWeight OutProj,
    NdArray QNorm,
    NdArray KNorm,
    NdArray PostAttnNorm,
    float AttnGate,
    NdArray PreHadaNorm,
    NdArray D1,
    NdArray D2,
    NdArray D3);

/// <summary>
/// Multi-lane hyper-connection weights for one layer.  These produce the
/// input-dependent gates and the doubly-stochastic routing matrix that mix the
/// <see cref="TransformerConfig.MhcLanes"/> residual lanes.
/// </summary>
/// <param name="PhiPre">Read gate projection [lanes*D, lanes].</param>
/// <param name="PhiPost">Write gate projection [lanes*D, lanes].</param>
/// <param name="PhiRes">Routing projection [lanes*D, lanes*lanes].</param>
/// <param name="BPre">Read gate bias [lanes].</param>
/// <param name="BPost">Write gate bias [lanes].</param>
/// <param name="BRes">Routing bias [lanes, lanes].</param>
/// <param name="APre">Read gate slope.</param>
/// <param name="APost">Write gate slope.</param>
/// <param name="ARes">Routing slope.</param>
/// <param name="PreOffset">Per-lane read offset, <c>8·onehot − 4</c> [lanes].</param>
/// <param name="PostOffset">Per-lane write offset, <c>−4·(1 − onehot)</c> [lanes].</param>
public sealed record MhcWeights(
    LinearWeight PhiPre,
    LinearWeight PhiPost,
    LinearWeight PhiRes,
    NdArray BPre,
    NdArray BPost,
    NdArray BRes,
    float APre,
    float APost,
    float ARes,
    float[] PreOffset,
    float[] PostOffset);

/// <summary>
/// One engram site: hashed n-gram tables plus the projections and causal
/// convolution taps that turn a row lookup into a key/value pair.
/// </summary>
/// <param name="Tables">Hash tables [tables, slots, subDim].</param>
/// <param name="KeyProj">Key kernel [tables*subDim, D].</param>
/// <param name="ValueProj">Value kernel [tables*subDim, D].</param>
/// <param name="Taps">Per-tap channel gains [ConvTaps, D].</param>
public sealed record EngramWeights(
    EngramTable Tables,
    LinearWeight KeyProj,
    LinearWeight ValueProj,
    NdArray Taps);

/// <summary>Probe-pooled contrastive head used for tool retrieval.</summary>
/// <param name="Probes">Probe queries [4, D].</param>
/// <param name="Proj">Projection [4*D, contrastiveDim].</param>
/// <param name="LogTemp">Learned log-temperature.</param>
public sealed record ContrastiveHeadWeights(NdArray Probes, LinearWeight Proj, float LogTemp);

/// <summary>Probe-pooled confidence head.</summary>
/// <param name="Probes">Probe queries [8, D].</param>
/// <param name="Proj">Projection [8*D, 1].</param>
/// <param name="Bias">Output bias [1].</param>
public sealed record ConfidenceHeadWeights(NdArray Probes, LinearWeight Proj, float Bias);

/// <summary>Auxiliary multi-token-prediction head.</summary>
/// <param name="Combine">Kernel over [hidden ‖ next-token embedding] [2D, D].</param>
/// <param name="EmbNorm">Norm applied to the next-token embedding [D].</param>
/// <param name="FinalNorm">Norm applied to the block output [D].</param>
/// <param name="Block">The extra transformer block.</param>
public sealed record MtpWeights(
    LinearWeight Combine,
    NdArray EmbNorm,
    NdArray FinalNorm,
    BlockWeights Block);

/// <summary>
/// Every parameter of a Needle 2 model, in the layout the forward pass reads
/// them.  Built from a flat name → tensor map by <see cref="FromFlat"/>, whose
/// keys are the reference's Flax paths joined with <c>/</c> — the same names a
/// checkpoint or <c>.cact</c> blob unpacks to.
/// </summary>
public sealed class Needle2Weights
{
    /// <summary>Geometry these weights were built for.</summary>
    public TransformerConfig Config { get; }

    /// <summary>Token embedding, tied with the output projection [V, D].</summary>
    public EmbeddingTable Embedding { get; }

    /// <summary>Per-layer block weights.</summary>
    public ImmutableArray<BlockWeights> Layers { get; }

    /// <summary>Per-layer hyper-connection weights.</summary>
    public ImmutableArray<MhcWeights> Mhc { get; }

    /// <summary>Final norm scale [D].</summary>
    public NdArray FinalNorm { get; }

    /// <summary>Engram sites, one per entry of <see cref="TransformerConfig.EngramLayers"/>.</summary>
    public ImmutableArray<EngramWeights> Engrams { get; }

    /// <summary>Multi-token-prediction head, when the checkpoint carries one.</summary>
    public MtpWeights? Mtp { get; }

    /// <summary>Contrastive retrieval head, when the checkpoint carries one.</summary>
    public ContrastiveHeadWeights? Contrastive { get; }

    /// <summary>Confidence head, when the checkpoint carries one.</summary>
    public ConfidenceHeadWeights? Confidence { get; }

    private Needle2Weights(
        TransformerConfig config,
        EmbeddingTable embedding,
        ImmutableArray<BlockWeights> layers,
        ImmutableArray<MhcWeights> mhc,
        NdArray finalNorm,
        ImmutableArray<EngramWeights> engrams,
        MtpWeights? mtp,
        ContrastiveHeadWeights? contrastive,
        ConfidenceHeadWeights? confidence)
    {
        Config = config;
        Embedding = embedding;
        Layers = layers;
        Mhc = mhc;
        FinalNorm = finalNorm;
        Engrams = engrams;
        Mtp = mtp;
        Contrastive = contrastive;
        Confidence = confidence;
    }

    /// <summary>
    /// Assemble from already-built parts.  Used by the packed <c>.cact</c> path,
    /// which produces quantised views rather than float32 tensors.
    /// </summary>
    public static Needle2Weights FromParts(
        TransformerConfig config,
        EmbeddingTable embedding,
        ImmutableArray<BlockWeights> layers,
        ImmutableArray<MhcWeights> mhc,
        NdArray finalNorm,
        ImmutableArray<EngramWeights> engrams,
        MtpWeights? mtp,
        ContrastiveHeadWeights? contrastive,
        ConfidenceHeadWeights? confidence) =>
        new(config, embedding, layers, mhc, finalNorm, engrams, mtp, contrastive, confidence);

    /// <summary>
    /// Build the weight set from a flat map keyed by Flax parameter path, e.g.
    /// <c>stack/layers/block/self_attn/q_proj/kernel</c>.  Per-layer tensors carry
    /// a leading layer axis (the stack is scanned in the reference) and are sliced
    /// here without copying.
    /// </summary>
    /// <param name="config">Geometry to validate the tensors against.</param>
    /// <param name="flat">Parameter path → tensor.</param>
    public static Needle2Weights FromFlat(TransformerConfig config, IReadOnlyDictionary<string, NdArray> flat)
    {
        config.Validate();

        int d = config.DModel, a = config.AttnWidth, kv = config.KvDim, hd = config.HeadDim;
        int l = config.NumLayers, lanes = config.MhcLanes, nc = lanes * d;
        int hw = config.HadamardWidth;
        var (orders, heads, subDim) = config.EngramGeometry();
        int tables = orders.Length * heads;

        NdArray Get(string name, params int[] shape)
        {
            if (!flat.TryGetValue(name, out var tensor))
                throw new KeyNotFoundException($"Checkpoint is missing '{name}' {NdArray.Describe(shape)}.");
            if (!tensor.Shape.AsSpan().SequenceEqual(shape))
                throw new InvalidOperationException(
                    $"Checkpoint tensor '{name}' has shape {NdArray.Describe(tensor.Shape)}, "
                    + $"expected {NdArray.Describe(shape)}.");
            return tensor;
        }

        // Flax stores gates and temperatures as rank-0 arrays; some writers
        // round-trip them as [1].  Accept either.
        float Scalar(string name)
        {
            if (!flat.TryGetValue(name, out var tensor))
                throw new KeyNotFoundException($"Checkpoint is missing scalar '{name}'.");
            if (tensor.Length != 1)
                throw new InvalidOperationException(
                    $"Checkpoint tensor '{name}' has {tensor.Length} elements, expected a scalar.");
            return tensor[0];
        }

        var embedding = new DenseEmbedding(Get("embedding/embedding", config.VocabSize, d));

        const string BlockPrefix = "stack/layers/block/";
        var normIn = Get(BlockPrefix + "ZCRMSNorm_0/scale", l, d);
        var qProj = Get(BlockPrefix + "self_attn/q_proj/kernel", l, d, a);
        var kProj = Get(BlockPrefix + "self_attn/k_proj/kernel", l, d, kv);
        var vProj = Get(BlockPrefix + "self_attn/v_proj/kernel", l, d, kv);
        var gateProj = Get(BlockPrefix + "self_attn/gate_proj/kernel", l, d, a);
        var outProj = Get(BlockPrefix + "self_attn/out_proj/kernel", l, a, d);
        var qNorm = Get(BlockPrefix + "self_attn/q_norm/scale", l, hd);
        var kNorm = Get(BlockPrefix + "self_attn/k_norm/scale", l, hd);
        var postNorm = Get(BlockPrefix + "post_attn_norm/scale", l, d);
        var attnGate = Get(BlockPrefix + "attn_gate", l);
        var preHada = Get(BlockPrefix + "pre_hada_norm/scale", l, d);
        var hd1 = Get(BlockPrefix + "hadamard_mlp/d1", l, hw);
        var hd2 = Get(BlockPrefix + "hadamard_mlp/d2", l, hw);
        var hd3 = Get(BlockPrefix + "hadamard_mlp/d3", l, hw);

        var layers = ImmutableArray.CreateBuilder<BlockWeights>(l);
        for (int i = 0; i < l; i++)
        {
            layers.Add(new BlockWeights(
                normIn.Slice(i),
                LinearWeight.Dense(qProj.Slice(i)), LinearWeight.Dense(kProj.Slice(i)),
                LinearWeight.Dense(vProj.Slice(i)), LinearWeight.Dense(gateProj.Slice(i)),
                LinearWeight.Dense(outProj.Slice(i)), qNorm.Slice(i), kNorm.Slice(i),
                postNorm.Slice(i), attnGate[i], preHada.Slice(i),
                hd1.Slice(i), hd2.Slice(i), hd3.Slice(i)));
        }

        var phiPre = Get("stack/mhc_phi_pre", l, nc, lanes);
        var phiPost = Get("stack/mhc_phi_post", l, nc, lanes);
        var phiRes = Get("stack/mhc_phi_res", l, nc, lanes * lanes);
        var bPre = Get("stack/mhc_b_pre", l, lanes);
        var bPost = Get("stack/mhc_b_post", l, lanes);
        var bRes = Get("stack/mhc_b_res", l, lanes, lanes);
        var aPre = Get("stack/mhc_a_pre", l);
        var aPost = Get("stack/mhc_a_post", l);
        var aRes = Get("stack/mhc_a_res", l);

        var mhc = ImmutableArray.CreateBuilder<MhcWeights>(l);
        for (int i = 0; i < l; i++)
        {
            // pre_off = 8*onehot - 4, post_off = -4*(1 - onehot), where each layer
            // owns lane (i % lanes).
            var preOff = new float[lanes];
            var postOff = new float[lanes];
            int lane = i % lanes;
            for (int n = 0; n < lanes; n++)
            {
                preOff[n] = n == lane ? 4f : -4f;
                postOff[n] = n == lane ? 0f : -4f;
            }
            mhc.Add(new MhcWeights(
                LinearWeight.Dense(phiPre.Slice(i)), LinearWeight.Dense(phiPost.Slice(i)),
                LinearWeight.Dense(phiRes.Slice(i)),
                bPre.Slice(i), bPost.Slice(i), bRes.Slice(i),
                aPre[i], aPost[i], aRes[i], preOff, postOff));
        }

        var finalNorm = Get("stack/final_norm/scale", d);

        var engrams = ImmutableArray.CreateBuilder<EngramWeights>(config.EngramSites);
        for (int s = 0; s < config.EngramSites; s++)
        {
            engrams.Add(new EngramWeights(
                new DenseEngramTable(Get($"engrams_{s}/embedding", tables, config.EngramSlots, subDim)),
                LinearWeight.Dense(Get($"engrams_{s}/key_proj/kernel", tables * subDim, d)),
                LinearWeight.Dense(Get($"engrams_{s}/value_proj/kernel", tables * subDim, d)),
                Get($"engrams_{s}/taps", EngramConstants.ConvTaps, d)));
        }

        MtpWeights? mtp = null;
        if (flat.ContainsKey("mtp_combine/kernel"))
        {
            const string P = "mtp_block/";
            mtp = new MtpWeights(
                LinearWeight.Dense(Get("mtp_combine/kernel", 2 * d, d)),
                Get("mtp_emb_norm/scale", d),
                Get("mtp_final_norm/scale", d),
                new BlockWeights(
                    Get(P + "ZCRMSNorm_0/scale", d),
                    LinearWeight.Dense(Get(P + "self_attn/q_proj/kernel", d, a)),
                    LinearWeight.Dense(Get(P + "self_attn/k_proj/kernel", d, kv)),
                    LinearWeight.Dense(Get(P + "self_attn/v_proj/kernel", d, kv)),
                    LinearWeight.Dense(Get(P + "self_attn/gate_proj/kernel", d, a)),
                    LinearWeight.Dense(Get(P + "self_attn/out_proj/kernel", a, d)),
                    Get(P + "self_attn/q_norm/scale", hd),
                    Get(P + "self_attn/k_norm/scale", hd),
                    Get(P + "post_attn_norm/scale", d),
                    Scalar(P + "attn_gate"),
                    Get(P + "pre_hada_norm/scale", d),
                    Get(P + "hadamard_mlp/d1", hw),
                    Get(P + "hadamard_mlp/d2", hw),
                    Get(P + "hadamard_mlp/d3", hw)));
        }

        ContrastiveHeadWeights? contrastive = null;
        if (flat.ContainsKey("contrastive_head/probes"))
        {
            contrastive = new ContrastiveHeadWeights(
                Get("contrastive_head/probes", Needle2Model.ContrastiveProbes, d),
                LinearWeight.Dense(
                    Get("contrastive_head/proj/kernel", Needle2Model.ContrastiveProbes * d, config.ContrastiveDim)),
                Scalar("contrastive_head/log_temp"));
        }

        ConfidenceHeadWeights? confidence = null;
        if (flat.ContainsKey("confidence_head/probes"))
        {
            confidence = new ConfidenceHeadWeights(
                Get("confidence_head/probes", Needle2Model.ConfidenceProbes, d),
                LinearWeight.Dense(Get("confidence_head/proj/kernel", Needle2Model.ConfidenceProbes * d, 1)),
                Scalar("confidence_head/proj/bias"));
        }

        return new Needle2Weights(config, embedding, layers.MoveToImmutable(), mhc.MoveToImmutable(),
                                  finalNorm, engrams.MoveToImmutable(), mtp, contrastive, confidence);
    }

    /// <summary>Total number of stored parameters.</summary>
    public long ParameterCount
    {
        get
        {
            long n = (long)Embedding.VocabSize * Embedding.Width + FinalNorm.Length;
            foreach (var b in Layers)
                n += b.NormIn.Length + Elements(b.QProj) + Elements(b.KProj) + Elements(b.VProj)
                     + Elements(b.GateProj) + Elements(b.OutProj) + b.QNorm.Length + b.KNorm.Length
                     + b.PostAttnNorm.Length + 1 + b.PreHadaNorm.Length
                     + b.D1.Length + b.D2.Length + b.D3.Length;
            foreach (var m in Mhc)
                n += Elements(m.PhiPre) + Elements(m.PhiPost) + Elements(m.PhiRes)
                     + m.BPre.Length + m.BPost.Length + m.BRes.Length + 3;
            foreach (var e in Engrams)
                n += (long)e.Tables.Tables * e.Tables.Slots * e.Tables.SubDim
                     + Elements(e.KeyProj) + Elements(e.ValueProj) + e.Taps.Length;
            return n;
        }
    }

    /// <summary>
    /// Bytes the weights actually occupy — float32 when dense, packed codes plus
    /// per-group norms when quantised.  This, not the parameter count, is what
    /// decode throughput tracks.
    /// </summary>
    public long ByteSize
    {
        get
        {
            long bytes = Embedding.ByteSize;
            foreach (var b in Layers)
                bytes += b.QProj.ByteSize + b.KProj.ByteSize + b.VProj.ByteSize
                         + b.GateProj.ByteSize + b.OutProj.ByteSize;
            foreach (var m in Mhc)
                bytes += m.PhiPre.ByteSize + m.PhiPost.ByteSize + m.PhiRes.ByteSize;
            foreach (var e in Engrams)
                bytes += e.Tables.ByteSize + e.KeyProj.ByteSize + e.ValueProj.ByteSize;
            return bytes;
        }
    }

    /// <summary>True when the weights are held in their packed Cactus-Quant form.</summary>
    public bool IsQuantized => Embedding is QuantizedEmbedding;

    private static long Elements(LinearWeight weight) =>
        (long)weight.InputWidth * weight.OutputWidth;
}
