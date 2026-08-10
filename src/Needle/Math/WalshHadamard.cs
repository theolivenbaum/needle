using System.Numerics;
using System.Numerics.Tensors;

namespace Needle.Math;

/// <summary>
/// The orthonormal Walsh-Hadamard transform, <c>H = H_n / sqrt(n)</c>.
///
/// The reference materialises the matrix and multiplies by it
/// (<c>_walsh_matrix</c> in architecture.py, <c>_cq_hadamard_np</c> in
/// quantize.py); here it runs as the in-place butterfly, which is the same
/// linear map in <c>n log n</c> time with no weights to read.  H is symmetric
/// and orthonormal, so the same routine serves both <c>x @ H</c> and its inverse.
/// </summary>
public static class WalshHadamard
{
    /// <summary>Smallest power of two greater than or equal to <paramref name="n"/>.</summary>
    public static int NextPow2(int n)
    {
        if (n <= 1) return 1;
        // Matches Python's `1 << (n - 1).bit_length()`.
        return 1 << (32 - BitOperations.LeadingZeroCount((uint)(n - 1)));
    }

    /// <summary>True when <paramref name="n"/> is a positive power of two.</summary>
    public static bool IsPow2(int n) => n > 0 && (n & (n - 1)) == 0;

    /// <summary>
    /// Apply the orthonormal transform to a single vector in place.
    /// Its length must be a power of two.
    /// </summary>
    public static void Transform(Span<float> x)
    {
        int n = x.Length;
        if (!IsPow2(n))
            throw new ArgumentException($"Walsh-Hadamard needs a power-of-two length, got {n}.", nameof(x));

        Butterfly(x);
        TensorPrimitives.Multiply(x, 1f / MathF.Sqrt(n), x);
    }

    /// <summary>
    /// Apply the transform to every row of the last axis of <paramref name="x"/>,
    /// in place.  Equivalent to <c>x @ H</c>.
    /// </summary>
    public static void TransformRows(NdArray x)
    {
        int width = x.LastDim;
        if (!IsPow2(width))
            throw new ArgumentException($"Walsh-Hadamard needs a power-of-two width, got {width}.", nameof(x));

        int rows = x.Length / width;
        var data = x.Span;
        float scale = 1f / MathF.Sqrt(width);
        for (int r = 0; r < rows; r++)
        {
            var row = data.Slice(r * width, width);
            Butterfly(row);
            TensorPrimitives.Multiply(row, scale, row);
        }
    }

    /// <summary>
    /// The unnormalised ±1 butterfly.  Blocks of <c>2*half</c> are combined as
    /// <c>(a + b, a - b)</c>, doubling <c>half</c> each pass.
    /// </summary>
    private static void Butterfly(Span<float> x)
    {
        int n = x.Length;
        for (int half = 1; half < n; half <<= 1)
        {
            for (int start = 0; start < n; start += half << 1)
            {
                var lo = x.Slice(start, half);
                var hi = x.Slice(start + half, half);
                CombinePair(lo, hi);
            }
        }
    }

    /// <summary><c>(lo, hi) = (lo + hi, lo - hi)</c>, vectorised.</summary>
    private static void CombinePair(Span<float> lo, Span<float> hi)
    {
        int i = 0, n = lo.Length;
        if (Vector.IsHardwareAccelerated && n >= Vector<float>.Count)
        {
            int bound = n - n % Vector<float>.Count;
            for (; i < bound; i += Vector<float>.Count)
            {
                var a = new Vector<float>(lo[i..]);
                var b = new Vector<float>(hi[i..]);
                (a + b).CopyTo(lo[i..]);
                (a - b).CopyTo(hi[i..]);
            }
        }
        for (; i < n; i++)
        {
            float a = lo[i], b = hi[i];
            lo[i] = a + b;
            hi[i] = a - b;
        }
    }

    /// <summary>
    /// Materialise the orthonormal matrix, row-major <c>[n, n]</c>.  Only used by
    /// tests that check the butterfly against an explicit matrix product.
    /// </summary>
    public static NdArray Matrix(int n)
    {
        if (!IsPow2(n))
            throw new ArgumentException($"Walsh-Hadamard needs a power-of-two order, got {n}.", nameof(n));

        var m = new NdArray(n, n);
        var span = m.Span;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                // H[i, j] = (-1)^popcount(i & j)
                span[i * n + j] = (BitOperations.PopCount((uint)(i & j)) & 1) == 0 ? 1f : -1f;
            }
        }
        TensorPrimitives.Multiply(span, 1f / MathF.Sqrt(n), span);
        return m;
    }
}

/// <summary>
/// Sinkhorn normalisation: turns raw routing logits into a doubly-stochastic
/// mixing matrix.  Port of <c>_sinkhorn</c> in architecture.py — alternating
/// row/column log-sum-exp subtractions in the log domain, then a single exp.
/// </summary>
public static class Sinkhorn
{
    /// <summary>Iteration count used by the reference.</summary>
    public const int DefaultIterations = 20;

    /// <summary>
    /// Normalise a square <c>n × n</c> block in place: logits in, doubly-stochastic
    /// probabilities out.
    /// </summary>
    public static void Normalize(Span<float> logits, int n, int iterations = DefaultIterations)
    {
        if (logits.Length != n * n)
            throw new ArgumentException($"Expected {n * n} logits for an {n}x{n} block, got {logits.Length}.");

        Span<float> reduction = stackalloc float[n];

        for (int it = 0; it < iterations; it++)
        {
            // Rows: subtract logsumexp over the last axis.
            for (int i = 0; i < n; i++)
            {
                var row = logits.Slice(i * n, n);
                float lse = Ops.LogSumExp(row);
                for (int j = 0; j < n; j++) row[j] -= lse;
            }

            // Columns: subtract logsumexp over the second-to-last axis.
            for (int j = 0; j < n; j++)
            {
                float max = float.NegativeInfinity;
                for (int i = 0; i < n; i++) max = MathF.Max(max, logits[i * n + j]);
                float sum = 0f;
                for (int i = 0; i < n; i++) sum += MathF.Exp(logits[i * n + j] - max);
                reduction[j] = max + MathF.Log(sum);
            }
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    logits[i * n + j] -= reduction[j];
        }

        for (int i = 0; i < logits.Length; i++) logits[i] = MathF.Exp(logits[i]);
    }
}
