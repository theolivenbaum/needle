using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Needle.Math;

/// <summary>
/// SIMD numeric kernels the model is built from.  Everything works on
/// <see cref="Span{T}"/> of float, using <c>System.Numerics.Tensors</c> where it
/// already has a vectorised primitive and <c>Vector&lt;float&gt;</c> where the
/// operation is fused enough to be worth hand-rolling.
/// </summary>
public static class Ops
{
    /// <summary>Epsilon shared by every normaliser in the model.</summary>
    public const float Epsilon = 1e-6f;

    // ── Matrix multiplication ────────────────────────────────────────────────

    /// <summary>
    /// Row-major GEMM: <c>c[m, n] = sum_k a[m, k] * b[k, n]</c>.
    ///
    /// Both operands are row-major, which is the layout the Flax kernels already
    /// use (<c>[in, out]</c>), so the inner loop streams one row of <c>b</c>
    /// contiguously and accumulates into one row of <c>c</c> — SIMD-friendly with
    /// no transposition anywhere.
    /// </summary>
    /// <param name="a">Left operand, <c>m × k</c>.</param>
    /// <param name="b">Right operand, <c>k × n</c>.</param>
    /// <param name="c">Destination, <c>m × n</c>; overwritten.</param>
    public static void MatMul(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c,
                              int m, int k, int n)
    {
        if (a.Length < (long)m * k) throw new ArgumentException("Left operand is too small.", nameof(a));
        if (b.Length < (long)k * n) throw new ArgumentException("Right operand is too small.", nameof(b));
        if (c.Length < (long)m * n) throw new ArgumentException("Destination is too small.", nameof(c));

        c[..(m * n)].Clear();
        MatMulRows(a, b, c, 0, m, k, n);
    }

    /// <summary>
    /// Parallel row-major GEMM.  Splits the output rows across the thread pool
    /// when the operands are large enough to pay for it; otherwise identical to
    /// <see cref="MatMul(ReadOnlySpan{float}, ReadOnlySpan{float}, Span{float}, int, int, int)"/>.
    /// </summary>
    public static void MatMulParallel(NdArray a, NdArray b, NdArray c, int m, int k, int n)
    {
        c.Span[..(m * n)].Clear();

        // Below a few hundred thousand multiply-adds the scheduling overhead
        // dominates and a straight loop wins.
        const long ParallelThreshold = 1 << 18;
        int cores = Environment.ProcessorCount;
        if (cores <= 1 || (long)m * k * n < ParallelThreshold)
        {
            MatMulRows(a.ReadSpan, b.ReadSpan, c.Span, 0, m, k, n);
            return;
        }

        // Spans cannot cross the lambda boundary; the arrays outlive the call.
        float[] aBuffer = a.Buffer, bBuffer = b.Buffer, cBuffer = c.Buffer;
        int aOffset = a.Offset, bOffset = b.Offset, cOffset = c.Offset;

        // Only the prefill has enough output rows to be worth splitting.  A
        // single-token decode step runs 135 of these, each a few tens of
        // microseconds, and the dispatch costs more than the work: splitting one
        // output row across four cores by columns measured 45 tok/s against 60
        // serial.  What is left is streaming the weights, which one core already
        // does at about 10 GB/s here.
        if (m < 8)
        {
            MatMulRows(a.ReadSpan, b.ReadSpan, c.Span, 0, m, k, n);
            return;
        }

        int workers = System.Math.Min(cores, m / 4);
        int chunk = (m + workers - 1) / workers;
        Parallel.For(0, workers, w =>
        {
            int start = w * chunk;
            int count = System.Math.Min(chunk, m - start);
            if (count <= 0) return;
            MatMulRows(aBuffer.AsSpan(aOffset + start * k, count * k),
                       bBuffer.AsSpan(bOffset, k * n),
                       cBuffer.AsSpan(cOffset + start * n, count * n),
                       0, count, k, n);
        });
    }

    /// <summary>
    /// Accumulate <c>rows</c> output rows starting at <paramref name="first"/>.
    /// Four rows of <c>a</c> are processed together so each row of <c>b</c> is
    /// read once and reused four times, which is what turns this from
    /// memory-bound into compute-bound.
    /// </summary>
    private static void MatMulRows(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c,
                                   int first, int rows, int k, int n)
    {
        // The hyper-connection gates project 2048 inputs onto 4 or 16 outputs.
        // Streaming that as one vector call per input row means thousands of
        // calls over four-element spans, where the call overhead dwarfs the
        // arithmetic — so accumulate narrow outputs by hand instead.
        if (n <= 16)
        {
            MatMulNarrow(a, b, c, first, rows, k, n);
            return;
        }

        // Decode projects one token at a time, and one output row cannot reuse a
        // row of `b` the way the four-row block below does.  Blocking along the
        // reduction axis instead is what makes it fast: the accumulator stays in
        // registers across four rows of `b` rather than being read back and
        // written out once per row.
        if (rows == 1)
        {
            MatMulSingleRow(a[(first * k)..], b, c[(first * n)..], k, n);
            return;
        }

        int i = first;
        for (; i + 4 <= first + rows; i += 4)
        {
            var c0 = c.Slice(i * n, n);
            var c1 = c.Slice((i + 1) * n, n);
            var c2 = c.Slice((i + 2) * n, n);
            var c3 = c.Slice((i + 3) * n, n);
            var a0 = a.Slice(i * k, k);
            var a1 = a.Slice((i + 1) * k, k);
            var a2 = a.Slice((i + 2) * k, k);
            var a3 = a.Slice((i + 3) * k, k);

            for (int p = 0; p < k; p++)
            {
                var bRow = b.Slice(p * n, n);
                float s0 = a0[p], s1 = a1[p], s2 = a2[p], s3 = a3[p];
                if (s0 != 0f) TensorPrimitives.MultiplyAdd(bRow, s0, c0, c0);
                if (s1 != 0f) TensorPrimitives.MultiplyAdd(bRow, s1, c1, c1);
                if (s2 != 0f) TensorPrimitives.MultiplyAdd(bRow, s2, c2, c2);
                if (s3 != 0f) TensorPrimitives.MultiplyAdd(bRow, s3, c3, c3);
            }
        }

        for (; i < first + rows; i++)
        {
            var row = c.Slice(i * n, n);
            var aRow = a.Slice(i * k, k);
            for (int p = 0; p < k; p++)
            {
                float scale = aRow[p];
                if (scale != 0f) TensorPrimitives.MultiplyAdd(b.Slice(p * n, n), scale, row, row);
            }
        }
    }

    /// <summary>
    /// Accumulate a single output row: <c>c += sum_p a[p] * b[p, :]</c>, with
    /// <c>c</c> already holding whatever it should be added to.
    ///
    /// Written as four rows of <c>b</c> at a time.  The obvious form — one
    /// <c>MultiplyAdd</c> call per row — reads and writes the whole of <c>c</c>
    /// once per row of the reduction, so a 512×512 projection moves two megabytes
    /// through the accumulator to do a quarter of a million multiply-adds.
    /// Four-way blocking cuts that traffic to a quarter and fuses four library
    /// calls into one loop body.
    ///
    /// The two widths are spelled out rather than written once over
    /// <c>Vector&lt;float&gt;</c>, because on AVX-512 hardware .NET keeps
    /// <c>Vector&lt;T&gt;</c> at 256 bits while <c>TensorPrimitives</c> uses 512 —
    /// so the portable form would hand back half the width the library call it
    /// replaces was already getting, and lose.
    /// </summary>
    private static void MatMulSingleRow(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c,
                                        int k, int n)
    {
        int p = 0;
        if (Vector512.IsHardwareAccelerated && n >= Vector512<float>.Count) p = Blocked512(a, b, c, k, n);
        else if (Vector256.IsHardwareAccelerated && n >= Vector256<float>.Count) p = Blocked256(a, b, c, k, n);

        for (; p < k; p++)
        {
            float scale = a[p];
            if (scale != 0f) TensorPrimitives.MultiplyAdd(b.Slice(p * n, n), scale, c[..n], c[..n]);
        }
    }

    /// <summary>Four reduction rows at a time, 512 bits wide.  Returns rows consumed.</summary>
    private static int Blocked512(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int k, int n)
    {
        const int width = 16;
        int bound = n - n % width;
        ref float cHead = ref MemoryMarshal.GetReference(c);
        ref float bHead = ref Unsafe.AsRef(in MemoryMarshal.GetReference(b));

        int p = 0;
        for (; p + 4 <= k; p += 4)
        {
            var s0 = Vector512.Create(a[p]);
            var s1 = Vector512.Create(a[p + 1]);
            var s2 = Vector512.Create(a[p + 2]);
            var s3 = Vector512.Create(a[p + 3]);
            ref float row = ref Unsafe.Add(ref bHead, p * n);

            for (int j = 0; j < bound; j += width)
            {
                var acc = Vector512.LoadUnsafe(ref cHead, (nuint)j)
                          + s0 * Vector512.LoadUnsafe(ref row, (nuint)j)
                          + s1 * Vector512.LoadUnsafe(ref row, (nuint)(n + j))
                          + s2 * Vector512.LoadUnsafe(ref row, (nuint)(2 * n + j))
                          + s3 * Vector512.LoadUnsafe(ref row, (nuint)(3 * n + j));
                Vector512.StoreUnsafe(acc, ref cHead, (nuint)j);
            }
            TailRows(a, b, c, p, n, bound);
        }
        return p;
    }

    /// <summary>Four reduction rows at a time, 256 bits wide.  Returns rows consumed.</summary>
    private static int Blocked256(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c, int k, int n)
    {
        const int width = 8;
        int bound = n - n % width;
        ref float cHead = ref MemoryMarshal.GetReference(c);
        ref float bHead = ref Unsafe.AsRef(in MemoryMarshal.GetReference(b));

        int p = 0;
        for (; p + 4 <= k; p += 4)
        {
            var s0 = Vector256.Create(a[p]);
            var s1 = Vector256.Create(a[p + 1]);
            var s2 = Vector256.Create(a[p + 2]);
            var s3 = Vector256.Create(a[p + 3]);
            ref float row = ref Unsafe.Add(ref bHead, p * n);

            for (int j = 0; j < bound; j += width)
            {
                var acc = Vector256.LoadUnsafe(ref cHead, (nuint)j)
                          + s0 * Vector256.LoadUnsafe(ref row, (nuint)j)
                          + s1 * Vector256.LoadUnsafe(ref row, (nuint)(n + j))
                          + s2 * Vector256.LoadUnsafe(ref row, (nuint)(2 * n + j))
                          + s3 * Vector256.LoadUnsafe(ref row, (nuint)(3 * n + j));
                Vector256.StoreUnsafe(acc, ref cHead, (nuint)j);
            }
            TailRows(a, b, c, p, n, bound);
        }
        return p;
    }

    /// <summary>The columns past the last whole vector, for one four-row block.</summary>
    private static void TailRows(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c,
                                 int p, int n, int from)
    {
        for (int j = from; j < n; j++)
        {
            c[j] += a[p] * b[p * n + j] + a[p + 1] * b[(p + 1) * n + j]
                    + a[p + 2] * b[(p + 2) * n + j] + a[p + 3] * b[(p + 3) * n + j];
        }
    }

    /// <summary>
    /// GEMM for a narrow output: accumulates <paramref name="n"/> ≤ 16 columns in
    /// registers while streaming the reduction axis, so the whole product is one
    /// pass with no per-row call overhead.
    ///
    /// The hyper-connection gates land here — 2048 inputs onto 4 or 16 outputs,
    /// three times a layer.  That is a trivial number of multiply-adds, so what
    /// decides the cost is the per-element overhead around them: a scalar inner
    /// loop over four columns spends more time on bounds checks and loop control
    /// than on arithmetic.  Accumulating in <see cref="Vector4"/> registers and
    /// walking the operands by reference removes both.
    /// </summary>
    private static void MatMulNarrow(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c,
                                     int first, int rows, int k, int n)
    {
        // Four Vector4 lanes cover every width the model uses (4 and 16); a width
        // that is not a multiple of four falls back to the scalar form.
        if (n % 4 != 0 || n > 16)
        {
            MatMulNarrowScalar(a, b, c, first, rows, k, n);
            return;
        }

        int quads = n / 4;
        ref readonly float bHead = ref MemoryMarshal.GetReference(b);

        for (int i = first; i < first + rows; i++)
        {
            Vector4 v0 = Vector4.Zero, v1 = Vector4.Zero, v2 = Vector4.Zero, v3 = Vector4.Zero;
            ref readonly float aRow = ref Unsafe.Add(ref MemoryMarshal.GetReference(a), i * k);

            for (int p = 0; p < k; p++)
            {
                var scale = new Vector4(Unsafe.Add(ref Unsafe.AsRef(in aRow), p));
                ref readonly float bRow = ref Unsafe.Add(ref Unsafe.AsRef(in bHead), p * n);

                v0 += scale * Load(in bRow, 0);
                if (quads > 1) v1 += scale * Load(in bRow, 4);
                if (quads > 2) v2 += scale * Load(in bRow, 8);
                if (quads > 3) v3 += scale * Load(in bRow, 12);
            }

            var row = c.Slice(i * n, n);
            v0.CopyTo(row);
            if (quads > 1) v1.CopyTo(row[4..]);
            if (quads > 2) v2.CopyTo(row[8..]);
            if (quads > 3) v3.CopyTo(row[12..]);
        }
    }

    /// <summary>Read four consecutive floats starting <paramref name="at"/> past a reference.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector4 Load(ref readonly float head, int at) =>
        Unsafe.ReadUnaligned<Vector4>(
            ref Unsafe.As<float, byte>(ref Unsafe.Add(ref Unsafe.AsRef(in head), at)));

    /// <summary>Narrow GEMM for widths the vector form does not cover.</summary>
    private static void MatMulNarrowScalar(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c,
                                           int first, int rows, int k, int n)
    {
        Span<float> accumulator = stackalloc float[16];

        for (int i = first; i < first + rows; i++)
        {
            accumulator[..n].Clear();
            var aRow = a.Slice(i * k, k);

            for (int p = 0; p < k; p++)
            {
                float scale = aRow[p];
                if (scale == 0f) continue;
                var bRow = b.Slice(p * n, n);
                for (int j = 0; j < n; j++) accumulator[j] += scale * bRow[j];
            }

            accumulator[..n].CopyTo(c.Slice(i * n, n));
        }
    }

    /// <summary>
    /// <see cref="MatMul(ReadOnlySpan{float}, ReadOnlySpan{float}, Span{float}, int, int, int)"/>
    /// over <see cref="NdArray"/> operands: <c>[m, k] × [k, n] → [m, n]</c>.
    /// </summary>
    public static NdArray MatMul(NdArray a, NdArray b)
    {
        int k = a.LastDim;
        int m = a.Length / k;
        if (b.Rank != 2 || b.Shape[0] != k)
            throw new ArgumentException(
                $"Cannot multiply {NdArray.Describe(a.Shape)} by {NdArray.Describe(b.Shape)}.");
        int n = b.Shape[1];

        var shape = (int[])a.Shape.Clone();
        shape[^1] = n;
        var c = new NdArray(shape);
        MatMulParallel(a, b, c, m, k, n);
        return c;
    }

    /// <summary>
    /// Row-major GEMM against a transposed right operand:
    /// <c>c[m, n] = sum_k a[m, k] * bT[n, k]</c>.  Used for the tied output
    /// projection, where the embedding table is stored <c>[vocab, d]</c>.
    /// </summary>
    public static void MatMulTransposed(ReadOnlySpan<float> a, ReadOnlySpan<float> bT, Span<float> c,
                                        int m, int k, int n)
    {
        for (int i = 0; i < m; i++)
        {
            var aRow = a.Slice(i * k, k);
            var cRow = c.Slice(i * n, n);
            for (int j = 0; j < n; j++)
                cRow[j] = TensorPrimitives.Dot(aRow, bT.Slice(j * k, k));
        }
    }

    // ── Normalisation ────────────────────────────────────────────────────────

    /// <summary>
    /// Zero-centred RMSNorm over the last axis, in place:
    /// <c>x = (1 + scale) * x / sqrt(mean(x^2) + eps)</c>.
    /// Port of <c>ZCRMSNorm</c>.
    /// </summary>
    /// <param name="x">Tensor whose last axis has <paramref name="scale"/>'s length.</param>
    /// <param name="scale">Learned per-channel gain, initialised to zero.</param>
    public static void ZcRmsNorm(NdArray x, ReadOnlySpan<float> scale, float epsilon = Epsilon)
    {
        int width = scale.Length;
        if (x.LastDim != width)
            throw new ArgumentException($"Norm width {width} does not match last axis {x.LastDim}.");

        int rows = x.Length / width;
        var data = x.Span;
        for (int r = 0; r < rows; r++)
        {
            var row = data.Slice(r * width, width);
            float invRms = 1f / MathF.Sqrt(TensorPrimitives.Dot(row, row) / width + epsilon);
            ScaleByOnePlus(row, scale, invRms);
        }
    }

    /// <summary><c>row = invRms * (1 + scale) * row</c>, vectorised.</summary>
    private static void ScaleByOnePlus(Span<float> row, ReadOnlySpan<float> scale, float invRms)
    {
        int i = 0, n = row.Length;
        if (Vector.IsHardwareAccelerated && n >= Vector<float>.Count)
        {
            var vInv = new Vector<float>(invRms);
            int bound = n - n % Vector<float>.Count;
            for (; i < bound; i += Vector<float>.Count)
            {
                var v = new Vector<float>(row[i..]);
                var s = new Vector<float>(scale[i..]);
                ((Vector<float>.One + s) * v * vInv).CopyTo(row[i..]);
            }
        }
        for (; i < n; i++) row[i] = (1f + scale[i]) * row[i] * invRms;
    }

    /// <summary>
    /// Scale each row of the last axis to unit RMS, with no learned gain:
    /// <c>x * rsqrt(mean(x^2) + eps)</c>.  Port of <c>_rms_unit</c>.
    /// </summary>
    public static void RmsUnit(NdArray x, float epsilon = Epsilon)
    {
        int width = x.LastDim;
        int rows = x.Length / width;
        var data = x.Span;
        for (int r = 0; r < rows; r++)
        {
            var row = data.Slice(r * width, width);
            float invRms = 1f / MathF.Sqrt(TensorPrimitives.Dot(row, row) / width + epsilon);
            TensorPrimitives.Multiply(row, invRms, row);
        }
    }

    // ── Activations ──────────────────────────────────────────────────────────

    /// <summary>Numerically stable softmax over a span, in place.</summary>
    public static void Softmax(Span<float> x)
    {
        if (x.IsEmpty) return;
        float max = TensorPrimitives.Max(x);
        if (float.IsNegativeInfinity(max))
        {
            // Every entry masked out.  The reference fills blocked scores with the
            // float minimum rather than -inf precisely so this stays finite; keep
            // the uniform fallback for callers that do use -inf.
            x.Fill(1f / x.Length);
            return;
        }
        TensorPrimitives.Subtract(x, max, x);
        TensorPrimitives.Exp(x, x);
        TensorPrimitives.Multiply(x, 1f / TensorPrimitives.Sum(x), x);
    }

    /// <summary>Softmax over the last axis of a tensor, in place.</summary>
    public static void SoftmaxLastAxis(NdArray x)
    {
        int width = x.LastDim;
        int rows = x.Length / width;
        var data = x.Span;
        for (int r = 0; r < rows; r++) Softmax(data.Slice(r * width, width));
    }

    /// <summary>Element-wise sigmoid, in place.</summary>
    public static void Sigmoid(Span<float> x) => TensorPrimitives.Sigmoid(x, x);

    /// <summary>
    /// Element-wise SiLU (<c>x · sigmoid(x)</c>), in place.  Uses the vectorised
    /// <c>TensorPrimitives.Sigmoid</c> through a scratch buffer rather than a
    /// scalar exp loop.
    /// </summary>
    public static void Silu(Span<float> x)
    {
        var scratch = System.Buffers.ArrayPool<float>.Shared.Rent(x.Length);
        try
        {
            var gate = scratch.AsSpan(0, x.Length);
            TensorPrimitives.Sigmoid(x, gate);
            TensorPrimitives.Multiply(x, gate, x);
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(scratch);
        }
    }

    /// <summary>Logarithm of a sum of exponentials, computed stably.</summary>
    public static float LogSumExp(ReadOnlySpan<float> x)
    {
        float max = TensorPrimitives.Max(x);
        if (float.IsNegativeInfinity(max)) return max;
        float sum = 0f;
        foreach (float v in x) sum += MathF.Exp(v - max);
        return max + MathF.Log(sum);
    }

    // ── Element-wise helpers ─────────────────────────────────────────────────

    /// <summary><c>dst += src</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Add(Span<float> dst, ReadOnlySpan<float> src) =>
        TensorPrimitives.Add(dst, src, dst);

    /// <summary><c>dst += scale * src</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AddScaled(Span<float> dst, ReadOnlySpan<float> src, float scale) =>
        TensorPrimitives.MultiplyAdd(src, scale, dst, dst);

    /// <summary><c>dst *= src</c>, element-wise.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Multiply(Span<float> dst, ReadOnlySpan<float> src) =>
        TensorPrimitives.Multiply(dst, src, dst);

    /// <summary><c>dst *= scale</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Scale(Span<float> dst, float scale) =>
        TensorPrimitives.Multiply(dst, scale, dst);

    /// <summary>
    /// Broadcast a per-channel vector across every row of the last axis:
    /// <c>x[..., c] *= gain[c]</c>.
    /// </summary>
    public static void MultiplyRows(NdArray x, ReadOnlySpan<float> gain)
    {
        int width = gain.Length;
        int rows = x.Length / width;
        var data = x.Span;
        for (int r = 0; r < rows; r++)
            TensorPrimitives.Multiply(data.Slice(r * width, width), gain, data.Slice(r * width, width));
    }

    /// <summary>Index of the largest element (first wins on ties).</summary>
    public static int ArgMax(ReadOnlySpan<float> x)
    {
        int best = 0;
        float bestValue = float.NegativeInfinity;
        for (int i = 0; i < x.Length; i++)
        {
            if (x[i] > bestValue) { bestValue = x[i]; best = i; }
        }
        return best;
    }
}
