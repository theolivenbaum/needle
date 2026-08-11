namespace Needle.Model;

/// <summary>
/// The engram row-index hash.  Port of <c>engram_indices</c> in architecture.py:
/// an FNV-style mix over the last <c>order</c> tokens, one independent hash per
/// (order, head) pair, reduced modulo the table size.
/// </summary>
public static class EngramHash
{
    /// <summary>
    /// Row index for the n-gram of length <paramref name="order"/> ending at
    /// <paramref name="position"/>.
    /// </summary>
    /// <param name="tokens">Token IDs.</param>
    /// <param name="position">Position whose n-gram is being hashed.</param>
    /// <param name="order">n-gram length.</param>
    /// <param name="table">Flat table index (<c>orderIndex * heads + head</c>).</param>
    /// <param name="slots">Number of rows per table.</param>
    public static int Index(ReadOnlySpan<int> tokens, int position, int order, int table, int slots)
    {
        uint acc = unchecked(EngramConstants.Seed * (uint)(table + 1));
        for (int j = 0; j < order; j++)
        {
            // _shift_right by j: the token j positions back, or 0 before the start.
            int src = position - j;
            uint token = src >= 0 ? (uint)tokens[src] : 0u;
            acc = unchecked((acc ^ token) * EngramConstants.Prime);
        }
        acc ^= acc >> 15;
        return (int)(acc % (uint)slots);
    }

    /// <summary>
    /// Fill <paramref name="destination"/> with one row index per table for every
    /// position: <c>destination[t * tables + j]</c>.
    /// </summary>
    /// <param name="tokens">Token IDs [T].</param>
    /// <param name="orders">n-gram orders.</param>
    /// <param name="heads">Independent hash heads per order.</param>
    /// <param name="slots">Rows per table.</param>
    /// <param name="destination">Output buffer of length <c>T * orders.Length * heads</c>.</param>
    public static void Fill(
        ReadOnlySpan<int> tokens, ReadOnlySpan<int> orders, int heads, int slots, Span<int> destination)
    {
        int tables = orders.Length * heads;
        if (destination.Length < tokens.Length * tables)
            throw new ArgumentException("Destination is too small for the index grid.", nameof(destination));

        for (int t = 0; t < tokens.Length; t++)
        {
            for (int oi = 0; oi < orders.Length; oi++)
            {
                for (int h = 0; h < heads; h++)
                {
                    int table = oi * heads + h;
                    destination[t * tables + table] = Index(tokens, t, orders[oi], table, slots);
                }
            }
        }
    }
}
