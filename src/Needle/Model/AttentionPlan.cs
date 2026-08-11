namespace Needle.Model;

/// <summary>
/// The key positions each query may attend to, flattened once into CSR-style
/// arrays.
///
/// The mask rules (causal, sliding window, padding, pinned sinks) do not depend
/// on the layer or the head, so resolving them once per sequence and reusing the
/// result across all 27 layers × 8 heads keeps the attention inner loop to pure
/// arithmetic.
/// </summary>
public sealed class AttentionPlan
{
    private readonly int[] _offsets;   // [queries + 1]
    private readonly int[] _keys;      // concatenated allowed key positions
    private readonly int _used;        // keys actually filled

    /// <summary>Absolute position of the first query this plan covers.</summary>
    public int StartPosition { get; }

    /// <summary>Number of query positions.</summary>
    public int QueryCount { get; }

    private AttentionPlan(int startPosition, int[] offsets, int[] keys, int queries, int used)
    {
        StartPosition = startPosition;
        _offsets = offsets;
        _keys = keys;
        QueryCount = queries;
        _used = used;
    }

    /// <summary>Allowed key positions for query <paramref name="index"/> (0-based within the plan).</summary>
    public ReadOnlySpan<int> Keys(int index) =>
        _keys.AsSpan(_offsets[index], _offsets[index + 1] - _offsets[index]);

    /// <summary>Largest number of keys any single query attends to.</summary>
    public int MaxKeys
    {
        get
        {
            int max = 0;
            for (int i = 0; i < QueryCount; i++) max = System.Math.Max(max, _offsets[i + 1] - _offsets[i]);
            return max;
        }
    }

    /// <summary>
    /// Resolve <paramref name="mask"/> for queries
    /// <c>[startPosition, startPosition + count)</c> against keys
    /// <c>[0, keyCount)</c>.
    /// </summary>
    public static AttentionPlan Build(SequenceMask mask, int startPosition, int count, int keyCount)
    {
        var offsets = new int[count + 1];
        var keys = new List<int>(count * System.Math.Min(keyCount, 64));

        for (int i = 0; i < count; i++)
        {
            offsets[i] = keys.Count;
            int query = startPosition + i;
            int limit = System.Math.Min(query, keyCount - 1);

            if (mask.Window > 0 && mask.HasSinks)
            {
                // Sinks below the window, then the window itself.
                int windowStart = mask.WindowStart(query);
                for (int j = 0; j < windowStart; j++)
                    if (mask.IsSink(j) && mask.IsValid(j)) keys.Add(j);
                for (int j = windowStart; j <= limit; j++)
                    if (mask.IsValid(j)) keys.Add(j);
            }
            else
            {
                int start = mask.Window > 0 ? mask.WindowStart(query) : 0;
                for (int j = start; j <= limit; j++)
                    if (mask.IsValid(j)) keys.Add(j);
            }
        }
        offsets[count] = keys.Count;

        var flat = keys.ToArray();
        return new AttentionPlan(startPosition, offsets, flat, count, flat.Length);
    }

    /// <summary>Resolve the whole sequence at once (the prefill case).</summary>
    public static AttentionPlan Build(SequenceMask mask) => Build(mask, 0, mask.Length, mask.Length);

    /// <summary>
    /// Rebuild in place over buffers this plan already owns.  A decode step
    /// resolves the same shape every time, so reusing the arrays keeps the
    /// per-token allocation at zero.
    /// </summary>
    private AttentionPlan Refill(SequenceMask mask, int startPosition, int count, int keyCount)
    {
        int written = 0;
        for (int i = 0; i < count; i++)
        {
            _offsets[i] = written;
            int query = startPosition + i;
            int limit = System.Math.Min(query, keyCount - 1);
            int start = mask.Window > 0 ? mask.WindowStart(query) : 0;

            if (mask.Window > 0 && mask.HasSinks)
                for (int j = 0; j < start; j++)
                    if (mask.IsSink(j) && mask.IsValid(j)) _keys[written++] = j;

            for (int j = start; j <= limit; j++)
                if (mask.IsValid(j)) _keys[written++] = j;
        }
        _offsets[count] = written;
        return new AttentionPlan(startPosition, _offsets, _keys, count, written);
    }

    /// <summary>
    /// Reuses one plan's storage across steps.  Sized once for the worst case,
    /// then refilled.
    /// </summary>
    public sealed class Builder
    {
        private AttentionPlan? _plan;

        /// <summary>Resolve <paramref name="mask"/> for one window of queries.</summary>
        public AttentionPlan Build(SequenceMask mask, int startPosition, int count, int keyCount)
        {
            int capacity = count * System.Math.Min(keyCount, MaxKeysPerQuery(mask, keyCount));
            if (_plan is null || _plan._keys.Length < capacity || _plan._offsets.Length < count + 1)
                _plan = new AttentionPlan(startPosition, new int[count + 1], new int[capacity], 0, 0);

            return _plan.Refill(mask, startPosition, count, keyCount);
        }

        private static int MaxKeysPerQuery(SequenceMask mask, int keyCount) =>
            mask.Window > 0 ? System.Math.Min(keyCount, mask.Window + keyCount) : keyCount;
    }
}
