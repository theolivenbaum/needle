using TorchSharp;
using static TorchSharp.torch;

namespace Needle.Model;

/// <summary>
/// Rotary Position Embeddings (RoPE).
/// Port of <c>precompute_rope_freqs</c> and <c>apply_rope</c> from architecture.py.
/// </summary>
public static class RoPE
{
    /// <summary>
    /// Precompute cosine and sine frequency tables for RoPE.
    /// </summary>
    /// <param name="headDim">
    ///   Dimension of each attention head. Must be even.
    /// </param>
    /// <param name="seqLen">
    ///   Maximum sequence length to precompute.
    /// </param>
    /// <param name="theta">
    ///   Base frequency (default 10000.0).
    /// </param>
    /// <returns>
    ///   A tuple (cos, sin) where each tensor has shape [seqLen, headDim/2]
    ///   and dtype float32.
    /// </returns>
    public static (Tensor cos, Tensor sin) PrecomputeFreqs(
        int headDim,
        int seqLen,
        float theta = 10000.0f)
    {
        // freqs = 1 / (theta ^ (arange(0, head_dim, 2) / head_dim))
        // shape: [headDim/2]
        using var halfRange = torch.arange(
            start: 0,
            stop: headDim,
            step: 2,
            dtype: torch.float32); // [headDim/2]

        using var exponents = halfRange / headDim;          // [headDim/2]
        using var thetaTensor = torch.tensor(theta);
        using var freqs = torch.pow(thetaTensor, exponents); // [headDim/2]
        using var invFreqs = 1.0f / freqs;                   // [headDim/2]

        // t = arange(seq_len), shape [seqLen]
        using var t = torch.arange(seqLen, dtype: torch.float32); // [seqLen]

        // angles = outer(t, freqs) → [seqLen, headDim/2]
        using var angles = torch.outer(t, invFreqs); // [seqLen, headDim/2]

        var cos = torch.cos(angles); // [seqLen, headDim/2]
        var sin = torch.sin(angles); // [seqLen, headDim/2]

        return (cos, sin);
    }

    /// <summary>
    /// Apply rotary position embeddings to a query or key tensor.
    /// </summary>
    /// <param name="x">
    ///   Input tensor of shape [batch, heads, seqLen, headDim].
    /// </param>
    /// <param name="cos">
    ///   Cosine table of shape [maxSeqLen, headDim/2] (from <see cref="PrecomputeFreqs"/>).
    /// </param>
    /// <param name="sin">
    ///   Sine table of shape [maxSeqLen, headDim/2] (from <see cref="PrecomputeFreqs"/>).
    /// </param>
    /// <returns>
    ///   Rotated tensor of the same shape as <paramref name="x"/>.
    /// </returns>
    public static Tensor ApplyRope(Tensor x, Tensor cos, Tensor sin)
    {
        long T    = x.shape[2];
        long half = x.shape[3] / 2;

        // Slice cos/sin to the actual sequence length T
        // cos/sin: [maxSeqLen, headDim/2] → [:T] → [T, headDim/2]
        using var cosT = cos[TensorIndex.Slice(stop: T)];          // [T, headDim/2]
        using var sinT = sin[TensorIndex.Slice(stop: T)];          // [T, headDim/2]

        // Unsqueeze to [1, 1, T, headDim/2] for broadcasting with [B, H, T, headDim/2]
        using var cosB = cosT.unsqueeze(0).unsqueeze(0);            // [1, 1, T, headDim/2]
        using var sinB = sinT.unsqueeze(0).unsqueeze(0);            // [1, 1, T, headDim/2]

        // Split x into first and second halves along the head-dim axis
        using var x1 = x[TensorIndex.Ellipsis, TensorIndex.Slice(stop: half)];    // [B, H, T, half]
        using var x2 = x[TensorIndex.Ellipsis, TensorIndex.Slice(start: half)];   // [B, H, T, half]

        // Rotated halves:
        //   new_x1 = x1 * cos - x2 * sin
        //   new_x2 = x2 * cos + x1 * sin
        using var newX1 = x1 * cosB - x2 * sinB;  // [B, H, T, half]
        using var newX2 = x2 * cosB + x1 * sinB;  // [B, H, T, half]

        // Concatenate along last dimension → [B, H, T, headDim]
        return torch.cat([newX1, newX2], dim: -1);
    }
}
