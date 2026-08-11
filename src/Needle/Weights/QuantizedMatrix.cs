using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using Needle.Math;

namespace Needle.Weights;

/// <summary>
/// A Cactus-Quant matrix kept in its packed form, multiplied without ever
/// materialising float32 weights.
///
/// The trick is the rotation.  A quantised group reconstructs as
/// <c>w = (codebook[idx] * norm) @ H</c>, and <c>H</c> is symmetric and
/// orthonormal, so
///
/// <code>x · (q @ H) = (x @ H) · q</code>
///
/// Transforming the *activation* once per group therefore replaces transforming
/// every weight row, and what is left is a dot product against codebook indices.
/// The weights stay 2 bits each instead of 32 — about sixteen times less memory
/// traffic per token, which is what decode speed is actually bound by.
///
/// Rows are the output dimension and groups run along the input dimension,
/// matching the layout <c>.reference/needle/model/export.py</c> writes.
/// </summary>
public sealed class QuantizedMatrix
{
    private readonly byte[] _packed;
    private readonly float[] _norms;
    private readonly float[] _codebook;

    /// <summary>Codebook value for each raw packed code, ternary remapping folded in.</summary>
    private readonly float[] _byCode;

    /// <summary>
    /// For two-bit packing: one <see cref="Vector4"/> per byte value, holding the
    /// four codebook values that byte encodes.  Turns the inner loop into a
    /// single load and a single fused multiply-add per four weights.
    /// </summary>
    private readonly Vector4[]? _byteTable;

    /// <summary>Output dimension.</summary>
    public int Rows { get; }

    /// <summary>Input dimension, before group padding.</summary>
    public int Width { get; }

    /// <summary>Input dimension rounded up to a whole number of groups.</summary>
    public int PaddedWidth { get; }

    /// <summary>Elements per quantisation group.</summary>
    public int GroupSize { get; }

    /// <summary>Groups per row.</summary>
    public int Groups { get; }

    /// <summary>Directory bit width (2, 3, 4, or 5 for ternary).</summary>
    public int Bits { get; }

    /// <summary>Bytes of packed indices per row.</summary>
    public int RowBytes { get; }

    /// <summary>Bytes this matrix occupies, indices plus norms.</summary>
    public long ByteSize => (long)_packed.Length + (long)_norms.Length * sizeof(float);

    /// <summary>The packed index bitstream, rows concatenated.</summary>
    public byte[] PackedBytes => _packed;

    /// <summary>Per-row, per-group L2 norms.</summary>
    public float[] GroupNorms => _norms;

    /// <summary>The centroids this matrix was quantised against.</summary>
    public float[] Codebook => _codebook;

    /// <param name="packed">Index bitstream, <c>rows * RowBytes</c> bytes.</param>
    /// <param name="norms">Per-group L2 norms, <c>rows * groups</c>.</param>
    /// <param name="rows">Output dimension.</param>
    /// <param name="width">Input dimension.</param>
    /// <param name="bits">Directory bit width.</param>
    /// <param name="groupSize">Elements per group.</param>
    /// <param name="codebook">Centroids for this width.</param>
    public QuantizedMatrix(
        byte[] packed, float[] norms, int rows, int width, int bits, int groupSize, float[] codebook)
    {
        Rows = rows;
        Width = width;
        GroupSize = groupSize;
        Bits = bits;
        PaddedWidth = (width + groupSize - 1) / groupSize * groupSize;
        Groups = PaddedWidth / groupSize;
        RowBytes = CactusQuant.PackedRowBytes(PaddedWidth, bits);

        if (packed.Length < (long)rows * RowBytes)
            throw new ArgumentException("Packed blob is shorter than the declared shape.", nameof(packed));
        if (norms.Length < (long)rows * Groups)
            throw new ArgumentException("Norm blob is shorter than the declared shape.", nameof(norms));

        _packed = packed;
        _norms = norms;
        _codebook = codebook;

        int indexBits = bits == CactusQuant.TernaryRecordBits ? 2 : bits;
        _byCode = new float[1 << indexBits];
        for (int code = 0; code < _byCode.Length; code++)
        {
            int index = code;
            if (bits == CactusQuant.TernaryRecordBits)
            {
                // Signed crumbs: 3, 0, 1 stand for trits 0, 1, 2.  Crumb 2 never
                // occurs; map it onto zero so a corrupt byte cannot index out.
                index = code == 3 ? 0 : code == 2 ? 1 : code + 1;
            }
            _byCode[code] = index < codebook.Length ? codebook[index] : 0f;
        }

        if (indexBits == 2)
        {
            _byteTable = new Vector4[256];
            for (int value = 0; value < 256; value++)
            {
                _byteTable[value] = new Vector4(
                    _byCode[value & 3], _byCode[(value >> 2) & 3],
                    _byCode[(value >> 4) & 3], _byCode[(value >> 6) & 3]);
            }
        }
    }

    /// <summary>
    /// Rotate an activation vector into the basis the codes live in, ready for
    /// <see cref="Dot"/>.  Zero-pads to <see cref="PaddedWidth"/> first, which is
    /// exactly equivalent to dropping the padded weight columns.
    /// </summary>
    /// <param name="x">Activation, <see cref="Width"/> long.</param>
    /// <param name="destination">Scratch buffer, <see cref="PaddedWidth"/> long.</param>
    public void PrepareInput(ReadOnlySpan<float> x, Span<float> destination)
    {
        if (x.Length != Width)
            throw new ArgumentException($"Expected a {Width}-wide activation, got {x.Length}.", nameof(x));

        x.CopyTo(destination);
        destination[Width..PaddedWidth].Clear();
        for (int g = 0; g < Groups; g++)
            WalshHadamard.Transform(destination.Slice(g * GroupSize, GroupSize));
    }

    /// <summary>
    /// Dot a prepared activation with one output row.
    /// </summary>
    /// <param name="prepared">Output of <see cref="PrepareInput"/>.</param>
    /// <param name="row">Output index.</param>
    public float Dot(ReadOnlySpan<float> prepared, int row)
    {
        var codes = _packed.AsSpan(row * RowBytes, RowBytes);
        int normBase = row * Groups;
        float total = 0f;

        if (_byteTable is not null)
        {
            // Two-bit: four weights per byte, one vector multiply-add each.  Four
            // independent accumulators keep the pipeline fed — the table lookup
            // serialises otherwise, and the loop is latency-bound rather than
            // throughput-bound.
            int bytesPerGroup = GroupSize / 4;
            var table = _byteTable;
            for (int g = 0; g < Groups; g++)
            {
                var groupCodes = codes.Slice(g * bytesPerGroup, bytesPerGroup);
                var values = prepared.Slice(g * GroupSize, GroupSize);

                Vector4 a0 = Vector4.Zero, a1 = Vector4.Zero, a2 = Vector4.Zero, a3 = Vector4.Zero;
                int b = 0;
                for (; b + 4 <= bytesPerGroup; b += 4)
                {
                    a0 += table[groupCodes[b]] * new Vector4(values.Slice(b * 4, 4));
                    a1 += table[groupCodes[b + 1]] * new Vector4(values.Slice(b * 4 + 4, 4));
                    a2 += table[groupCodes[b + 2]] * new Vector4(values.Slice(b * 4 + 8, 4));
                    a3 += table[groupCodes[b + 3]] * new Vector4(values.Slice(b * 4 + 12, 4));
                }
                for (; b < bytesPerGroup; b++)
                    a0 += table[groupCodes[b]] * new Vector4(values.Slice(b * 4, 4));

                total += _norms[normBase + g] * Vector4.Dot(a0 + a1 + a2 + a3, Vector4.One);
            }
            return total;
        }

        if (Bits == 4)
        {
            int bytesPerGroup = GroupSize / 2;
            for (int g = 0; g < Groups; g++)
            {
                var groupCodes = codes.Slice(g * bytesPerGroup, bytesPerGroup);
                var values = prepared.Slice(g * GroupSize, GroupSize);
                float sum = 0f;
                for (int b = 0; b < bytesPerGroup; b++)
                {
                    byte packed = groupCodes[b];
                    sum += _byCode[packed & 15] * values[b * 2]
                           + _byCode[packed >> 4] * values[b * 2 + 1];
                }
                total += _norms[normBase + g] * sum;
            }
            return total;
        }

        // Widths that straddle byte boundaries (three bits) take the general path.
        for (int g = 0; g < Groups; g++)
        {
            var values = prepared.Slice(g * GroupSize, GroupSize);
            float sum = 0f;
            for (int i = 0; i < GroupSize; i++)
                sum += _byCode[CodeAt(codes, g * GroupSize + i)] * values[i];
            total += _norms[normBase + g] * sum;
        }
        return total;
    }

    /// <summary>Read code <paramref name="index"/> out of a row's LSB-first bitstream.</summary>
    private int CodeAt(ReadOnlySpan<byte> codes, int index)
    {
        int bitOffset = index * Bits;
        int byteOffset = bitOffset >> 3;
        int shift = bitOffset & 7;

        int window = codes[byteOffset];
        if (shift + Bits > 8 && byteOffset + 1 < codes.Length) window |= codes[byteOffset + 1] << 8;
        return (window >> shift) & ((1 << Bits) - 1);
    }

    /// <summary>
    /// Reconstruct one row to float32 — what a gather needs, since an engram
    /// lookup uses the row itself rather than multiplying by it.
    /// </summary>
    public void DequantizeRow(int row, Span<float> destination)
    {
        if (destination.Length < Width)
            throw new ArgumentException("Destination is too small.", nameof(destination));

        Span<float> scratch = PaddedWidth <= 512
            ? stackalloc float[PaddedWidth]
            : new float[PaddedWidth];

        var codes = _packed.AsSpan(row * RowBytes, RowBytes);
        for (int g = 0; g < Groups; g++)
        {
            float norm = _norms[row * Groups + g];
            var group = scratch.Slice(g * GroupSize, GroupSize);
            for (int i = 0; i < GroupSize; i++)
                group[i] = _byCode[CodeAt(codes, g * GroupSize + i)] * norm;
            WalshHadamard.Transform(group);
        }
        scratch[..Width].CopyTo(destination);
    }

    /// <summary>
    /// Multiply a batch of activations: <c>[tokens, Width] → [tokens, Rows]</c>.
    /// </summary>
    public void Apply(NdArray x, NdArray destination)
    {
        int tokens = x.Length / Width;
        if (destination.Length < (long)tokens * Rows)
            throw new ArgumentException("Destination is too small.", nameof(destination));

        // Prefill has many tokens to spread across cores; a decode step has one,
        // and dispatching a Parallel.For per matmul then costs more than the
        // matmul itself — 135 of them per token, at tens of microseconds each.
        int cores = Environment.ProcessorCount;
        if (tokens >= 4 && cores > 1)
        {
            float[] source = x.Buffer;
            int sourceOffset = x.Offset;
            float[] target = destination.Buffer;
            int targetOffset = destination.Offset;

            Parallel.For(0, cores, worker =>
            {
                int chunk = (tokens + cores - 1) / cores;
                int start = worker * chunk;
                int end = System.Math.Min(start + chunk, tokens);
                if (start >= end) return;

                var scratch = new float[PaddedWidth];
                for (int t = start; t < end; t++)
                {
                    PrepareInput(source.AsSpan(sourceOffset + t * Width, Width), scratch);
                    for (int o = 0; o < Rows; o++) target[targetOffset + t * Rows + o] = Dot(scratch, o);
                }
            });
            return;
        }

        // Preparing the activation costs one Walsh transform per group and is
        // shared by every output row, so it happens once per token.  The buffer
        // is pooled: a decode step runs 135 of these.
        var prepared = ArrayPool<float>.Shared.Rent(PaddedWidth);
        try
        {
            var scratch = prepared.AsSpan(0, PaddedWidth);
            for (int t = 0; t < tokens; t++)
            {
                PrepareInput(x.ReadSpan.Slice(t * Width, Width), scratch);
                var row = destination.Span.Slice(t * Rows, Rows);
                for (int o = 0; o < Rows; o++) row[o] = Dot(scratch, o);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(prepared);
        }
    }

    /// <summary>Reconstruct the whole matrix, row-major <c>[Rows, Width]</c>.</summary>
    public NdArray ToDense()
    {
        var dense = new NdArray(Rows, Width);
        for (int row = 0; row < Rows; row++) DequantizeRow(row, dense.Row(row));
        return dense;
    }

    /// <summary>
    /// Build from a <c>.cact</c> tensor payload: packed indices followed by
    /// fp16 per-group norms.
    /// </summary>
    public static QuantizedMatrix FromPayload(
        ReadOnlySpan<byte> payload, int rows, int width, int bits, int groupSize, float[] codebook)
    {
        int paddedWidth = (width + groupSize - 1) / groupSize * groupSize;
        int groups = paddedWidth / groupSize;
        int rowBytes = CactusQuant.PackedRowBytes(paddedWidth, bits);
        int packedBytes = rows * rowBytes;

        var packed = payload[..packedBytes].ToArray();
        var norms = new float[rows * groups];
        for (int i = 0; i < norms.Length; i++)
            norms[i] = (float)System.Buffers.Binary.BinaryPrimitives.ReadHalfLittleEndian(
                payload.Slice(packedBytes + i * 2, 2));

        return new QuantizedMatrix(packed, norms, rows, width, bits, groupSize, codebook);
    }

    /// <summary>Total bytes of a set of matrices, for reporting.</summary>
    public static long TotalBytes(IEnumerable<QuantizedMatrix> matrices) =>
        matrices.Sum(m => m.ByteSize);
}
