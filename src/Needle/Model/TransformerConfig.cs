using System.Collections.Immutable;

namespace Needle.Model;

/// <summary>
/// Configuration record for the Needle 2 Simple Attention Network.
/// Port of the Python <c>TransformerConfig</c> dataclass in
/// <c>needle/model/architecture.py</c>.
/// </summary>
public record TransformerConfig
{
    /// <summary>Vocabulary size (also the tied output projection width).</summary>
    public int VocabSize { get; init; } = 8192;

    /// <summary>Residual stream width.</summary>
    public int DModel { get; init; } = 512;

    /// <summary>
    /// Width of the attention sub-space.  0 means "same as <see cref="DModel"/>";
    /// use <see cref="AttnWidth"/> for the resolved value.
    /// </summary>
    public int AttnDim { get; init; }

    /// <summary>Number of query heads.</summary>
    public int NumHeads { get; init; } = 8;

    /// <summary>Number of key/value heads (GQA). Must divide <see cref="NumHeads"/>.</summary>
    public int NumKvHeads { get; init; } = 4;

    /// <summary>Number of transformer blocks in the (scanned) stack.</summary>
    public int NumLayers { get; init; } = 12;

    /// <summary>Longest sequence the model is defined for.</summary>
    public int MaxSeqLen { get; init; } = 2048;

    /// <summary>Token ID used for padding.</summary>
    public int PadTokenId { get; init; }

    /// <summary>Output width of the contrastive (tool-retrieval) head.</summary>
    public int ContrastiveDim { get; init; } = 128;

    /// <summary>RoPE base frequency.</summary>
    public float RopeTheta { get; init; } = 100000.0f;

    /// <summary>Compute dtype: "float32", "bfloat16" or "float16".</summary>
    public string Dtype { get; init; } = "float32";

    /// <summary>n-gram orders hashed into the engram tables.</summary>
    public ImmutableArray<int> EngramOrders { get; init; } = [2, 3];

    /// <summary>
    /// Engram heads per order.  0 means "derive from <see cref="DModel"/>";
    /// use <see cref="EngramGeometry"/> for the resolved value.
    /// </summary>
    public int EngramHeads { get; init; }

    /// <summary>Rows per engram hash table.</summary>
    public int EngramSlots { get; init; } = 8192;

    /// <summary>Indices of the layers at which an engram site fires.</summary>
    public ImmutableArray<int> EngramLayers { get; init; } = [2, 15];

    /// <summary>Number of parallel residual lanes in the hyper-connection stack.</summary>
    public int MhcLanes { get; init; } = 4;

    /// <summary>Sliding-window width; 0 means "derive from the KV byte budget".</summary>
    public int KvWindow { get; init; }

    /// <summary>KV-cache quantisation width the model was post-trained for (8 = int8).</summary>
    public int KvBits { get; init; } = 8;

    /// <summary>Activation quantisation width used by the deployment numerics.</summary>
    public int ActBits { get; init; } = 8;

    /// <summary>
    /// Optional mixed-precision weight spec, e.g. <c>"default=4,attn.q_proj=3"</c>.
    /// Empty means "uniform, decided by the exporter".
    /// </summary>
    public string WeightBits { get; init; } = "";

    // ── Derived geometry ─────────────────────────────────────────────────────

    /// <summary>Resolved attention width (<see cref="AttnDim"/> or <see cref="DModel"/>).</summary>
    public int AttnWidth => AttnDim != 0 ? AttnDim : DModel;

    /// <summary>Per-head width = <see cref="AttnWidth"/> / <see cref="NumHeads"/>.</summary>
    public int HeadDim => AttnWidth / NumHeads;

    /// <summary>Combined width of the key (or value) projection.</summary>
    public int KvDim => NumKvHeads * HeadDim;

    /// <summary>Number of engram sites (one per entry of <see cref="EngramLayers"/>).</summary>
    public int EngramSites => EngramLayers.Length;

    /// <summary>Hadamard MLP works in the next power of two at or above <see cref="DModel"/>.</summary>
    public int HadamardWidth => Needle.Math.WalshHadamard.NextPow2(DModel);

    /// <summary>
    /// Resolve the engram table geometry.  Port of <c>engram_geometry</c>.
    /// </summary>
    /// <returns>
    /// <c>heads</c> tables per order, each row <c>subDim</c> wide, for a total of
    /// <c>orders.Length * heads</c> tables.
    /// </returns>
    public (ImmutableArray<int> Orders, int Heads, int SubDim) EngramGeometry()
    {
        var orders = EngramOrders;
        int heads = EngramHeads != 0
            ? EngramHeads
            : System.Math.Max(1, DModel / (orders.Length * EngramConstants.SubDim));
        int subDim = DModel / (orders.Length * heads);
        return (orders, heads, subDim);
    }

    /// <summary>Total number of engram hash tables per site.</summary>
    public int EngramTables
    {
        get
        {
            var (orders, heads, _) = EngramGeometry();
            return orders.Length * heads;
        }
    }

    /// <summary>Throws when the configuration is internally inconsistent.</summary>
    public void Validate()
    {
        if (NumHeads <= 0 || NumKvHeads <= 0 || NumHeads % NumKvHeads != 0)
            throw new ArgumentException(
                $"NumHeads ({NumHeads}) must be a positive multiple of NumKvHeads ({NumKvHeads}).");
        if (AttnWidth % NumHeads != 0)
            throw new ArgumentException(
                $"AttnWidth ({AttnWidth}) must be divisible by NumHeads ({NumHeads}).");
        if (HeadDim % 2 != 0)
            throw new ArgumentException($"HeadDim ({HeadDim}) must be even for RoPE.");
        if (MhcLanes <= 0)
            throw new ArgumentException($"MhcLanes ({MhcLanes}) must be positive.");
        foreach (int layer in EngramLayers)
        {
            if (layer < 0 || layer >= NumLayers)
                throw new ArgumentException(
                    $"Engram site at layer {layer} is outside the stack (NumLayers={NumLayers}).");
        }
        var (orders, heads, subDim) = EngramGeometry();
        if (orders.Length * heads * subDim != DModel)
            throw new ArgumentException(
                $"Engram geometry ({orders.Length} orders x {heads} heads x {subDim}) "
                + $"does not tile DModel ({DModel}).");
    }

    // ── Presets ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Named presets from <c>architecture.py</c>: <c>needle</c>, <c>base</c> and
    /// <c>nano</c>.  Lookup is case-insensitive.
    /// </summary>
    public static TransformerConfig Preset(string name) => name.ToLowerInvariant() switch
    {
        "needle" => new TransformerConfig
        {
            DModel = 768, NumHeads = 12, NumKvHeads = 6, NumLayers = 27,
            EngramLayers = [2, 15],
        },
        "base" => new TransformerConfig
        {
            DModel = 512, NumHeads = 8, NumKvHeads = 4, NumLayers = 27,
            EngramLayers = [2, 15],
        },
        "nano" => new TransformerConfig
        {
            DModel = 256, NumHeads = 4, NumKvHeads = 2, NumLayers = 20,
            VocabSize = 4096, EngramLayers = [2, 15], EngramSlots = 4096,
        },
        _ => throw new ArgumentException($"Unknown preset '{name}'. Known: needle, base, nano."),
    };
}

/// <summary>Fixed engram constants shared by the architecture and the exporter.</summary>
public static class EngramConstants
{
    /// <summary>Nominal per-table row width used to derive the head count.</summary>
    public const int SubDim = 128;

    /// <summary>Number of causal convolution taps applied to engram values.</summary>
    public const int ConvTaps = 4;

    /// <summary>FNV-style hash seed (<c>0x9E3779B9</c>).</summary>
    public const uint Seed = 0x9E3779B9;

    /// <summary>FNV-style hash multiplier (<c>0x01000193</c>).</summary>
    public const uint Prime = 0x01000193;
}
