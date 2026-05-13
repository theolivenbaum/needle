namespace Needle.Model;

/// <summary>
/// Configuration record for the SimpleAttentionNetwork transformer.
/// Port of the Python TransformerConfig dataclass from architecture.py.
/// </summary>
public record TransformerConfig
{
    /// <summary>Vocabulary size.</summary>
    public int VocabSize { get; init; } = 8192;

    /// <summary>Model (embedding) dimension.</summary>
    public int DModel { get; init; } = 128;

    /// <summary>Number of query attention heads.</summary>
    public int NumHeads { get; init; } = 4;

    /// <summary>Number of key/value heads (GQA). Must divide NumHeads evenly.</summary>
    public int NumKvHeads { get; init; } = 2;

    /// <summary>Number of encoder transformer blocks.</summary>
    public int NumEncoderLayers { get; init; } = 2;

    /// <summary>Number of decoder transformer blocks.</summary>
    public int NumDecoderLayers { get; init; } = 2;

    /// <summary>Feed-forward hidden dimension.</summary>
    public int DFf { get; init; } = 512;

    /// <summary>Maximum sequence length supported by RoPE precomputation.</summary>
    public int MaxSeqLen { get; init; } = 128;

    /// <summary>Token ID used for padding (ignored in attention).</summary>
    public int PadTokenId { get; init; } = 0;

    /// <summary>RoPE base frequency theta.</summary>
    public float RopeTheta { get; init; } = 10000.0f;

    /// <summary>Floating-point dtype string: "float32", "bfloat16", or "float16".</summary>
    public string Dtype { get; init; } = "float32";

    /// <summary>Feed-forward activation: "drelu", "swiglu", or "geglu".</summary>
    public string Activation { get; init; } = "drelu";

    /// <summary>Number of persistent memory slots (unused in inference).</summary>
    public int NumMemorySlots { get; init; } = 64;

    /// <summary>Dropout probability (applied during training only).</summary>
    public float DropoutRate { get; init; } = 0.1f;

    /// <summary>Dimensionality of the contrastive projection space.</summary>
    public int ContrastiveDim { get; init; } = 128;

    /// <summary>When true, feed-forward sublayers are skipped (attention-only model).</summary>
    public bool NoFeedforward { get; init; } = true;

    // ── Derived properties ───────────────────────────────────────────────────

    /// <summary>Total number of transformer blocks (encoder + decoder).</summary>
    public int TotalLayers => NumEncoderLayers + NumDecoderLayers;

    /// <summary>Dimension of each attention head = DModel / NumHeads.</summary>
    public int HeadDim => DModel / NumHeads;

    /// <summary>Total dimension of the key/value projections = NumKvHeads * HeadDim.</summary>
    public int KvDim => NumKvHeads * HeadDim;
}
