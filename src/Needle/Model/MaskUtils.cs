using TorchSharp;
using static TorchSharp.torch;

namespace Needle.Model;

/// <summary>
/// Masking utilities for the SimpleAttentionNetwork.
/// Port of the mask-creation helpers in architecture.py.
///
/// All masks are bool tensors where <c>true</c> means "attend" and
/// <c>false</c> means "block" (matching the JAX convention: attention
/// weights are filled with -inf where mask == false).
/// </summary>
public static class MaskUtils
{
    // ── Causal mask ──────────────────────────────────────────────────────────

    /// <summary>
    /// Creates a lower-triangular causal mask of shape [1, 1, seqLen, seqLen].
    /// Position i can attend to position j iff j &lt;= i.
    /// </summary>
    public static Tensor MakeCausalMask(int seqLen)
    {
        // tril(ones) — shape [seqLen, seqLen]
        using var square = torch.ones(seqLen, seqLen, dtype: torch.@bool);
        using var tri = torch.tril(square);
        // Reshape to [1, 1, seqLen, seqLen]
        return tri.reshape(1, 1, seqLen, seqLen);
    }

    // ── Padding mask ─────────────────────────────────────────────────────────

    /// <summary>
    /// Creates a padding mask of shape [batch, 1, 1, seqLen].
    /// Position j is <c>true</c> (attend) when tokens[b,j] != padId.
    /// </summary>
    public static Tensor MakePaddingMask(Tensor tokens, int padId = 0)
    {
        // tokens: [batch, seqLen] (long/int64)
        // mask: tokens != padId → [batch, seqLen] bool
        using var mask = tokens.ne(padId);
        // [batch, 1, 1, seqLen]
        return mask.unsqueeze(1).unsqueeze(1);
    }

    // ── Packing mask ─────────────────────────────────────────────────────────

    /// <summary>
    /// Block-diagonal self-attention mask from segment IDs.
    /// segIds: [batch, T] int tensor where 0 = padding, 1+ = segment number.
    /// Returns [batch, 1, T, T] bool — true where both tokens share the same
    /// non-zero segment ID (i.e. are in the same packed sequence).
    /// </summary>
    public static Tensor MakePackingMask(Tensor segIds)
    {
        // segIds: [B, T]
        // Broadcast compare: [B, T, 1] == [B, 1, T] → [B, T, T]
        using var segLeft  = segIds.unsqueeze(2);   // [B, T, 1]
        using var segRight = segIds.unsqueeze(1);   // [B, 1, T]

        using var sameSegment = segLeft.eq(segRight);       // [B, T, T]
        using var nonPadLeft  = segLeft.gt(0L);             // [B, T, 1] broadcasted
        using var mask = torch.logical_and(sameSegment, nonPadLeft); // [B, T, T]

        return mask.unsqueeze(1); // [B, 1, T, T]
    }

    // ── Causal packing mask ──────────────────────────────────────────────────

    /// <summary>
    /// Block-diagonal causal mask from segment IDs.
    /// Same as <see cref="MakePackingMask"/> but also enforces causal ordering
    /// within each segment (token i can only attend to tokens j &lt;= i).
    /// Returns [batch, 1, T, T] bool.
    /// </summary>
    public static Tensor MakeCausalPackingMask(Tensor segIds)
    {
        long batchSize = segIds.shape[0];
        long T = segIds.shape[1];

        // Causal lower-triangular mask [T, T]
        using var ones   = torch.ones(T, T, dtype: torch.@bool, device: segIds.device);
        using var causal = torch.tril(ones); // [T, T]

        // Block-diagonal segment mask [B, T, T]
        using var segLeft  = segIds.unsqueeze(2);  // [B, T, 1]
        using var segRight = segIds.unsqueeze(1);  // [B, 1, T]

        using var sameSegment = segLeft.eq(segRight);
        using var nonPadLeft  = segLeft.gt(0L);
        using var blockMask   = torch.logical_and(sameSegment, nonPadLeft); // [B, T, T]

        // Combine: [B, T, T] & [T, T] (broadcast over batch)
        using var causalExpanded = causal.unsqueeze(0).expand(batchSize, T, T); // [B, T, T]
        using var combined = torch.logical_and(blockMask, causalExpanded);      // [B, T, T]

        return combined.unsqueeze(1); // [B, 1, T, T]
    }

    // ── Cross packing mask ───────────────────────────────────────────────────

    /// <summary>
    /// Cross-attention mask for packed sequences.
    /// encSegIds: [batch, T_enc]; decSegIds: [batch, T_dec].
    /// Returns [batch, 1, T_dec, T_enc] bool — decoder token d attends to
    /// encoder token e iff they share the same non-zero segment ID.
    /// </summary>
    public static Tensor MakeCrossPackingMask(Tensor encSegIds, Tensor decSegIds)
    {
        // decSegIds: [B, T_dec] → [B, T_dec, 1]
        // encSegIds: [B, T_enc] → [B, 1,     T_enc]
        using var decLeft  = decSegIds.unsqueeze(2);  // [B, T_dec, 1]
        using var encRight = encSegIds.unsqueeze(1);  // [B, 1, T_enc]

        using var sameSegment = decLeft.eq(encRight);       // [B, T_dec, T_enc]
        using var nonPadDec   = decLeft.gt(0L);             // [B, T_dec, 1] broadcasted
        using var mask = torch.logical_and(sameSegment, nonPadDec); // [B, T_dec, T_enc]

        return mask.unsqueeze(1); // [B, 1, T_dec, T_enc]
    }
}
