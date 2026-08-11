using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
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

    /// <summary>
    /// For four-bit packing: the sixteen codebook values in lane order, so a
    /// lane permute decodes sixteen weights at once.
    /// </summary>
    private readonly Vector512<float> _codebook512;

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

    /// <summary>
    /// Bits actually occupied by one packed index.  Ternary is nominally 1.58
    /// bits, does not fit the directory field and is spelled 5 there, but the
    /// bitstream stores two-bit crumbs — so the stride to read it by is 2, not
    /// <see cref="Bits"/>.
    /// </summary>
    public int IndexBits { get; }

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
        IndexBits = indexBits;
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

        if (Vector512.IsHardwareAccelerated && _byCode.Length <= 16)
        {
            // The codebook laid out one entry per lane, so a permute decodes a
            // whole vector of weights in one instruction.  Two-bit fills four
            // lanes and four-bit all sixteen; indices are masked to the width, so
            // the unused lanes are never selected.
            var lanes = new float[16];
            _byCode.CopyTo(lanes, 0);
            _codebook512 = Vector512.Create(lanes);
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

        if (_byteTable is not null) return DotTwoBit(prepared, codes, normBase);

        if (Bits == 4) return DotFourBit(prepared, codes, normBase);

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

    /// <summary>
    /// The four-bit inner loop.  Smaller in the blob than the two-bit path but
    /// not smaller in the model: the tied output projection over the vocabulary
    /// and all three hyper-connection gates are stored at four bits, which is
    /// about a third of a packed decode step.
    ///
    /// Four bits means sixteen codebook entries, which is exactly the width of an
    /// AVX-512 lane permute — so sixteen weights decode in one shuffle of the
    /// codebook rather than sixteen dependent table loads.  The indices come from
    /// eight packed bytes read as a single word and spread across the lanes by a
    /// variable shift.
    /// </summary>
    private float DotFourBit(ReadOnlySpan<float> prepared, ReadOnlySpan<byte> codes, int normBase)
    {
        int bytesPerGroup = GroupSize / 2;
        ref byte codeHead = ref MemoryMarshal.GetReference(codes);
        ref float valueHead = ref Unsafe.AsRef(in MemoryMarshal.GetReference(prepared));
        ref float normHead = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(_norms), normBase);
        ref float byCode = ref MemoryMarshal.GetArrayDataReference(_byCode);

        bool wide = Vector512.IsHardwareAccelerated && Avx512F.IsSupported && Avx2.IsSupported
                    && bytesPerGroup >= 8;
        var table = _codebook512;
        var shifts = Vector256.Create(0u, 4u, 8u, 12u, 16u, 20u, 24u, 28u);
        var nibble = Vector256.Create(15u);

        float total = 0f;
        for (int g = 0; g < Groups; g++)
        {
            ref byte groupCodes = ref Unsafe.Add(ref codeHead, g * bytesPerGroup);
            ref float values = ref Unsafe.Add(ref valueHead, g * GroupSize);

            float sum = 0f;
            int b = 0;
            if (wide)
            {
                var acc = Vector512<float>.Zero;
                for (; b + 8 <= bytesPerGroup; b += 8)
                {
                    ulong word = Unsafe.ReadUnaligned<ulong>(ref Unsafe.Add(ref groupCodes, b));
                    var low = Avx2.ShiftRightLogicalVariable(Vector256.Create((uint)word), shifts) & nibble;
                    var high = Avx2.ShiftRightLogicalVariable(Vector256.Create((uint)(word >> 32)), shifts) & nibble;
                    var weights = Avx512F.PermuteVar16x32(table, Vector512.Create(low.AsInt32(), high.AsInt32()));
                    acc += weights * Vector512.LoadUnsafe(ref values, (nuint)(b * 2));
                }
                sum = Vector512.Sum(acc);
            }

            for (; b < bytesPerGroup; b++)
            {
                byte packed = Unsafe.Add(ref groupCodes, b);
                sum += Unsafe.Add(ref byCode, packed & 15) * Unsafe.Add(ref values, b * 2)
                       + Unsafe.Add(ref byCode, packed >> 4) * Unsafe.Add(ref values, b * 2 + 1);
            }

            total += Unsafe.Add(ref normHead, g) * sum;
        }
        return total;
    }

    /// <summary>
    /// The two-bit inner loop — where a packed decode step spends most of its
    /// time, so it is written against raw references rather than spans.
    ///
    /// Every element of the obvious form costs more than the multiply-add it
    /// performs: a span index on the code byte, an array index on the table, a
    /// length check building a <c>Vector4</c>.  Walking all three by reference
    /// removes those, and four independent accumulators keep the pipeline fed
    /// past the latency of the dependent table load.
    /// </summary>
    private float DotTwoBit(ReadOnlySpan<float> prepared, ReadOnlySpan<byte> codes, int normBase)
    {
        if (Vector512.IsHardwareAccelerated && Avx512F.IsSupported && GroupSize >= 16)
            return DotTwoBitWide(prepared, codes, normBase);

        int bytesPerGroup = GroupSize / 4;
        ref Vector4 table = ref MemoryMarshal.GetArrayDataReference(_byteTable!);
        ref byte codeHead = ref MemoryMarshal.GetReference(codes);
        ref float valueHead = ref Unsafe.AsRef(in MemoryMarshal.GetReference(prepared));
        ref float normHead = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(_norms), normBase);

        float total = 0f;
        for (int g = 0; g < Groups; g++)
        {
            ref byte groupCodes = ref Unsafe.Add(ref codeHead, g * bytesPerGroup);
            ref float values = ref Unsafe.Add(ref valueHead, g * GroupSize);

            Vector4 a0 = Vector4.Zero, a1 = Vector4.Zero, a2 = Vector4.Zero, a3 = Vector4.Zero;
            int b = 0;
            for (; b + 4 <= bytesPerGroup; b += 4)
            {
                a0 += Unsafe.Add(ref table, Unsafe.Add(ref groupCodes, b)) * Quad(ref values, b * 4);
                a1 += Unsafe.Add(ref table, Unsafe.Add(ref groupCodes, b + 1)) * Quad(ref values, b * 4 + 4);
                a2 += Unsafe.Add(ref table, Unsafe.Add(ref groupCodes, b + 2)) * Quad(ref values, b * 4 + 8);
                a3 += Unsafe.Add(ref table, Unsafe.Add(ref groupCodes, b + 3)) * Quad(ref values, b * 4 + 12);
            }
            for (; b < bytesPerGroup; b++)
                a0 += Unsafe.Add(ref table, Unsafe.Add(ref groupCodes, b)) * Quad(ref values, b * 4);

            total += Unsafe.Add(ref normHead, g) * Vector4.Dot(a0 + a1 + a2 + a3, Vector4.One);
        }
        return total;
    }

    /// <summary>
    /// The two-bit inner loop where a lane permute is available.
    ///
    /// Sixteen two-bit codes fit one 32-bit word, so a single variable shift
    /// spreads a four-byte read across all sixteen lanes and one permute of the
    /// four-entry codebook turns them into weights.  That replaces four dependent
    /// lookups into a four-kilobyte table with no memory traffic beyond the codes
    /// themselves.  The indices are masked to the codebook width first, so the
    /// bare permute is used rather than the cross-platform shuffle and its
    /// out-of-range guard.
    /// </summary>
    private float DotTwoBitWide(ReadOnlySpan<float> prepared, ReadOnlySpan<byte> codes, int normBase)
    {
        int bytesPerGroup = GroupSize / 4;
        ref byte codeHead = ref MemoryMarshal.GetReference(codes);
        ref float valueHead = ref Unsafe.AsRef(in MemoryMarshal.GetReference(prepared));
        ref float normHead = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(_norms), normBase);

        var table = _codebook512;
        var shifts = Vector512.Create(0u, 2u, 4u, 6u, 8u, 10u, 12u, 14u,
                                      16u, 18u, 20u, 22u, 24u, 26u, 28u, 30u);
        var crumb = Vector512.Create(3u);

        float total = 0f;
        for (int g = 0; g < Groups; g++)
        {
            ref byte groupCodes = ref Unsafe.Add(ref codeHead, g * bytesPerGroup);
            ref float values = ref Unsafe.Add(ref valueHead, g * GroupSize);

            // Two accumulators: the permute has enough latency that a single
            // dependency chain leaves the multiply-add unit idle.
            Vector512<float> a0 = Vector512<float>.Zero, a1 = Vector512<float>.Zero;
            int b = 0;
            for (; b + 8 <= bytesPerGroup; b += 8)
            {
                uint w0 = Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref groupCodes, b));
                uint w1 = Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref groupCodes, b + 4));
                var i0 = Avx512F.ShiftRightLogicalVariable(Vector512.Create(w0), shifts) & crumb;
                var i1 = Avx512F.ShiftRightLogicalVariable(Vector512.Create(w1), shifts) & crumb;
                a0 += Avx512F.PermuteVar16x32(table, i0.AsInt32()) * Vector512.LoadUnsafe(ref values, (nuint)(b * 4));
                a1 += Avx512F.PermuteVar16x32(table, i1.AsInt32()) * Vector512.LoadUnsafe(ref values, (nuint)(b * 4 + 16));
            }
            for (; b + 4 <= bytesPerGroup; b += 4)
            {
                uint word = Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref groupCodes, b));
                var index = Avx512F.ShiftRightLogicalVariable(Vector512.Create(word), shifts) & crumb;
                a0 += Avx512F.PermuteVar16x32(table, index.AsInt32()) * Vector512.LoadUnsafe(ref values, (nuint)(b * 4));
            }

            float sum = Vector512.Sum(a0 + a1);
            ref Vector4 table4 = ref MemoryMarshal.GetArrayDataReference(_byteTable!);
            for (; b < bytesPerGroup; b++)
            {
                sum += Vector4.Dot(Unsafe.Add(ref table4, Unsafe.Add(ref groupCodes, b)),
                                   Quad(ref values, b * 4));
            }

            total += Unsafe.Add(ref normHead, g) * sum;
        }
        return total;
    }

    /// <summary>Four consecutive floats starting <paramref name="at"/> past a reference.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector4 Quad(ref float head, int at) =>
        Unsafe.ReadUnaligned<Vector4>(ref Unsafe.As<float, byte>(ref Unsafe.Add(ref head, at)));

    /// <summary>Read code <paramref name="index"/> out of a row's LSB-first bitstream.</summary>
    private int CodeAt(ReadOnlySpan<byte> codes, int index)
    {
        int bits = IndexBits;
        int bitOffset = index * bits;
        int byteOffset = bitOffset >> 3;
        int shift = bitOffset & 7;

        int window = codes[byteOffset];
        if (shift + bits > 8 && byteOffset + 1 < codes.Length) window |= codes[byteOffset + 1] << 8;
        return (window >> shift) & ((1 << bits) - 1);
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
    /// <param name="x">Activations, <c>[tokens, Width]</c>.</param>
    /// <param name="destination">Result, <c>[tokens, Rows]</c>.</param>
    /// <param name="shared">
    /// Optional rotation shared with the other matrices reading the same
    /// activation.  When supplied and already bound to <paramref name="x"/>, the
    /// Walsh transforms are done once for the group rather than once per matrix.
    /// </param>
    public void Apply(NdArray x, NdArray destination, PreparedActivation? shared = null)
    {
        int tokens = x.Length / Width;
        if (destination.Length < (long)tokens * Rows)
            throw new ArgumentException("Destination is too small.", nameof(destination));

        if (shared is not null)
        {
            var rotated = shared.Rotate(this, x);
            DotRows(rotated, destination, tokens);
            return;
        }

        // Prefill has many tokens to spread across cores; a decode step has one,
        // and dispatching a Parallel.For per matmul then costs more than the
        // matmul itself — 135 of them per token, at tens of microseconds each.
        //
        // The parallel form lives in its own method so that its closure is never
        // built on the single-token path.  Roslyn hoists a captured local into a
        // display class at the top of the method that declares it, so leaving the
        // lambda here would allocate one object per call whether or not it ran.
        if (tokens >= 4 && Environment.ProcessorCount > 1)
        {
            ApplyParallel(x, destination, tokens);
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

    /// <summary>Rotate and multiply a batch of tokens across the thread pool.</summary>
    private void ApplyParallel(NdArray x, NdArray destination, int tokens)
    {
        int cores = Environment.ProcessorCount;
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
    }

    /// <summary>Dot every output row against an already-rotated activation block.</summary>
    private void DotRows(float[] rotated, NdArray destination, int tokens)
    {
        if (tokens >= 4 && Environment.ProcessorCount > 1)
        {
            DotRowsParallel(rotated, destination, tokens);
            return;
        }

        for (int t = 0; t < tokens; t++)
        {
            var scratch = rotated.AsSpan(t * PaddedWidth, PaddedWidth);
            var row = destination.Span.Slice(t * Rows, Rows);
            for (int o = 0; o < Rows; o++) row[o] = Dot(scratch, o);
        }
    }

    /// <summary>The thread-pool form of <see cref="DotRows"/>.</summary>
    private void DotRowsParallel(float[] rotated, NdArray destination, int tokens)
    {
        int cores = Environment.ProcessorCount;
        float[] target = destination.Buffer;
        int targetOffset = destination.Offset;

        Parallel.For(0, cores, worker =>
        {
            int chunk = (tokens + cores - 1) / cores;
            int start = worker * chunk;
            int end = System.Math.Min(start + chunk, tokens);
            for (int t = start; t < end; t++)
            {
                var scratch = rotated.AsSpan(t * PaddedWidth, PaddedWidth);
                for (int o = 0; o < Rows; o++) target[targetOffset + t * Rows + o] = Dot(scratch, o);
            }
        });
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
