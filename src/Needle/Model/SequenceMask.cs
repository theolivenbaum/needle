namespace Needle.Model;

/// <summary>
/// Which keys each query position may attend to.
///
/// Rather than materialising a <c>T × T</c> boolean matrix, the rules are kept in
/// the form the model actually uses them: causal, plus an optional sliding
/// window, plus per-position padding validity and pinned "sink" keys that stay
/// visible after they fall out of the window (Needle 2 pins the tool block that
/// way, which is what bounds session memory).
/// </summary>
public sealed class SequenceMask
{
    private readonly bool[]? _valid;
    private readonly bool[]? _sink;

    /// <summary>Sequence length this mask describes.</summary>
    public int Length { get; }

    /// <summary>Sliding-window width; 0 means unbounded (plain causal).</summary>
    public int Window { get; }

    /// <summary>True when some positions are padding.</summary>
    public bool HasPadding => _valid is not null;

    /// <summary>True when some keys are pinned as sinks.</summary>
    public bool HasSinks => _sink is not null;

    /// <param name="length">Sequence length.</param>
    /// <param name="window">Sliding-window width; 0 or negative for unbounded.</param>
    /// <param name="valid">Optional per-position validity (false = padding).</param>
    /// <param name="sink">Optional per-position pin (true = always visible).</param>
    public SequenceMask(int length, int window = 0, bool[]? valid = null, bool[]? sink = null)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        if (valid is not null && valid.Length != length)
            throw new ArgumentException($"valid must have {length} entries.", nameof(valid));
        if (sink is not null && sink.Length != length)
            throw new ArgumentException($"sink must have {length} entries.", nameof(sink));

        Length = length;
        Window = System.Math.Max(0, window);
        _valid = valid;
        _sink = sink;
    }

    /// <summary>A plain causal mask over <paramref name="length"/> positions.</summary>
    public static SequenceMask Causal(int length) => new(length);

    /// <summary>Is position <paramref name="index"/> a real (non-padding) token?</summary>
    public bool IsValid(int index) =>
        (uint)index < (uint)Length && (_valid is null || _valid[index]);

    /// <summary>Is position <paramref name="index"/> pinned as an always-visible key?</summary>
    public bool IsSink(int index) =>
        _sink is not null && (uint)index < (uint)Length && _sink[index];

    /// <summary>
    /// May query <paramref name="query"/> attend to key <paramref name="key"/>?
    /// </summary>
    public bool Allows(int query, int key)
    {
        if (key > query || key < 0 || query >= Length) return false;
        if (!IsValid(key)) return false;
        if (Window > 0 && query - key >= Window && !IsSink(key)) return false;
        return true;
    }

    /// <summary>
    /// Lowest key index inside the sliding window for <paramref name="query"/>.
    /// Keys below it are visible only if they are sinks.
    /// </summary>
    public int WindowStart(int query) =>
        Window > 0 ? System.Math.Max(0, query - Window + 1) : 0;

    /// <summary>
    /// The engram legality flag for an n-gram reaching <paramref name="offset"/>
    /// positions back — the reference's <c>_mask_diag(mask, offset)</c>, which is
    /// just this mask read along its <paramref name="offset"/>-th sub-diagonal.
    /// </summary>
    /// <returns>A [Length] array of 0/1 flags.</returns>
    public float[] Diagonal(int offset)
    {
        var flags = new float[Length];
        for (int t = offset; t < Length; t++)
            if (Allows(t, t - offset)) flags[t] = 1f;
        return flags;
    }
}

/// <summary>
/// The deployment KV-cache budget.  Needle 2 sizes its sliding window so a whole
/// session's cache fits in a fixed number of bytes, which is what keeps total
/// memory near 28 MB however long the conversation runs.
/// Port of <c>kv_budget_window</c> / <c>effective_kv_window</c>.
/// </summary>
public static class KvBudget
{
    /// <summary>Bytes the KV cache may occupy (11.5 MiB).</summary>
    public const long BudgetBytes = 11L * 1024 * 1024 + 512 * 1024;

    /// <summary>Quantisation group size the cache is accounted in.</summary>
    public const int Group = 32;

    /// <summary>Smallest window the budget may shrink to.</summary>
    public const int MinWindow = 160;

    /// <summary>Largest window that fits the byte budget for this geometry.</summary>
    public static int BudgetWindow(TransformerConfig config)
    {
        long kv = (long)config.NumKvHeads * config.HeadDim;
        long d = config.DModel;
        long perPosition = config.NumLayers * (2 * kv + 2 * (kv / Group) * 4)
                           + (long)config.EngramSites * (d + (d / Group) * 4);
        long window = BudgetBytes / perPosition / Group * Group;
        return (int)System.Math.Max(MinWindow, System.Math.Min(window, config.MaxSeqLen));
    }

    /// <summary>
    /// The window actually used: the configured one when set, otherwise the
    /// budget-derived one — and never wider than the budget allows.
    /// </summary>
    public static int EffectiveWindow(TransformerConfig config)
    {
        int budget = BudgetWindow(config);
        return config.KvWindow > 0 ? System.Math.Min(budget, config.KvWindow) : budget;
    }
}
