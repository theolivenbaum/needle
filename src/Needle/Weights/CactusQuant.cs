using Needle.Math;

namespace Needle.Weights;

/// <summary>
/// Cactus-Quants: the weight codec the <c>.cact</c> deployment blob uses.
///
/// A row is split into groups of <see cref="GroupSize"/>, each group is rotated
/// by the orthonormal Walsh-Hadamard transform, scaled to unit length, and its
/// direction stored as one codebook index per element with a single fp16 norm
/// per group.  The rotation is what makes a shared, fixed codebook work: it
/// spreads any outlier in the original basis across the whole group, leaving a
/// near-Gaussian vector that Lloyd-Max centroids quantise well.
///
/// Port of the packing described in <c>.reference/needle/model/export.py</c> and
/// implemented in <c>cq_quantize</c> / <c>_cq_pack</c>.
/// </summary>
public static class CactusQuant
{
    /// <summary>Elements per quantisation group.</summary>
    public const int GroupSize = 128;

    /// <summary>Widths whose codebooks travel in the blob header.</summary>
    public static ReadOnlySpan<int> CodebookBits => [2, 3, 4];

    /// <summary>
    /// Directory <c>bits</c> value that marks a ternary tensor.  Ternary is
    /// nominally 1.58 bits and does not fit the field, so the format spells it 5.
    /// </summary>
    public const int TernaryRecordBits = 5;

    /// <summary>
    /// The analytic 3-level Lloyd-Max centroid.  Unlike the 2/3/4-bit codebooks
    /// this one is not carried in the header, because it is exact.
    /// </summary>
    public const float TernaryCentroid = 1.2240064f;

    /// <summary>The ternary codebook <c>{-c, 0, +c} / sqrt(group)</c>.</summary>
    public static float[] TernaryCodebook(int groupSize = GroupSize)
    {
        float scale = 1f / MathF.Sqrt(groupSize);
        return [-TernaryCentroid * scale, 0f, TernaryCentroid * scale];
    }

    /// <summary>
    /// Bytes one packed row occupies, before the per-group norms.
    /// </summary>
    /// <param name="paddedWidth">Row width rounded up to a whole number of groups.</param>
    /// <param name="bits">Directory bit width (2, 3, 4, or 5 for ternary).</param>
    public static int PackedRowBytes(int paddedWidth, int bits) =>
        bits == TernaryRecordBits ? paddedWidth * 2 / 8 : paddedWidth * bits / 8;

    /// <summary>
    /// Unpack an LSB-first index bitstream.
    ///
    /// Indices are emitted eight at a time: each run of eight is OR'd into one
    /// word at offsets <c>i * bits</c> and written as <c>bits</c> little-endian
    /// bytes — equivalently, one continuous LSB-first bitstream per row.
    /// </summary>
    /// <param name="packed">Packed bytes for one row.</param>
    /// <param name="bits">Bits per index.</param>
    /// <param name="count">Number of indices to produce (a multiple of 8).</param>
    /// <param name="destination">Output indices.</param>
    public static void UnpackIndices(ReadOnlySpan<byte> packed, int bits, int count, Span<byte> destination)
    {
        if (count % 8 != 0)
            throw new ArgumentException("Index runs are packed eight at a time.", nameof(count));
        if (destination.Length < count)
            throw new ArgumentException("Destination is too small.", nameof(destination));

        int chunks = count / 8;
        if (packed.Length < chunks * bits)
            throw new ArgumentException("Packed row is shorter than the declared width.", nameof(packed));

        ulong mask = (1UL << bits) - 1;
        for (int c = 0; c < chunks; c++)
        {
            ulong word = 0;
            for (int b = 0; b < bits; b++) word |= (ulong)packed[c * bits + b] << (8 * b);
            for (int i = 0; i < 8; i++) destination[c * 8 + i] = (byte)((word >> (i * bits)) & mask);
        }
    }

    /// <summary>
    /// Dequantise one packed tensor into a row-major <c>[rows, width]</c> array.
    /// </summary>
    /// <param name="packed">Index bitstream, <c>rows * PackedRowBytes</c> bytes.</param>
    /// <param name="norms">Per-group fp16 norms, already decoded to float.</param>
    /// <param name="rows">Logical row count.</param>
    /// <param name="width">Logical row width (before group padding).</param>
    /// <param name="bits">Directory bit width.</param>
    /// <param name="groupSize">Elements per group.</param>
    /// <param name="codebook">
    /// Centroids for this width; pass the ternary codebook when
    /// <paramref name="bits"/> is <see cref="TernaryRecordBits"/>.
    /// </param>
    public static NdArray Dequantize(
        ReadOnlySpan<byte> packed, ReadOnlySpan<float> norms,
        int rows, int width, int bits, int groupSize, ReadOnlySpan<float> codebook)
    {
        int paddedWidth = (width + groupSize - 1) / groupSize * groupSize;
        int groups = paddedWidth / groupSize;
        int rowBytes = PackedRowBytes(paddedWidth, bits);
        int indexBits = bits == TernaryRecordBits ? 2 : bits;

        if (packed.Length < (long)rows * rowBytes)
            throw new ArgumentException("Packed blob is shorter than the declared shape.", nameof(packed));
        if (norms.Length < (long)rows * groups)
            throw new ArgumentException("Norm blob is shorter than the declared shape.", nameof(norms));

        var result = new NdArray(rows, width);
        var indices = new byte[paddedWidth];
        var scratch = new float[paddedWidth];

        for (int r = 0; r < rows; r++)
        {
            UnpackIndices(packed.Slice(r * rowBytes, rowBytes), indexBits, paddedWidth, indices);

            for (int g = 0; g < groups; g++)
            {
                float norm = norms[r * groups + g];
                var group = scratch.AsSpan(g * groupSize, groupSize);
                for (int i = 0; i < groupSize; i++)
                {
                    int code = indices[g * groupSize + i];
                    // Ternary stores signed crumbs: 3, 0, 1 map to trit 0, 1, 2,
                    // so sign-extending a 2-bit field yields -1, 0, +1 directly.
                    if (bits == TernaryRecordBits) code = code == 3 ? 0 : code + 1;
                    group[i] = codebook[code] * norm;
                }
                // Undo the rotation: H is symmetric and orthonormal, so the same
                // transform inverts it.
                WalshHadamard.Transform(group);
            }

            scratch.AsSpan(0, width).CopyTo(result.Row(r));
        }

        return result;
    }
}
