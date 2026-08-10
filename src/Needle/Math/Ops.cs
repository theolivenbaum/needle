using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

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
        for (int i = 0; i < m; i++)
        {
            var row = c.Slice(i * n, n);
            var aRow = a.Slice(i * k, k);
            for (int p = 0; p < k; p++)
            {
                float scale = aRow[p];
                if (scale == 0f) continue;
                TensorPrimitives.MultiplyAdd(b.Slice(p * n, n), scale, row, row);
            }
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
        MatMul(a.ReadSpan, b.ReadSpan, c.Span, m, k, n);
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
