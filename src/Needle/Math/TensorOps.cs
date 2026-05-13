using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Numerics.Tensors;

namespace Needle.Math;

/// <summary>
/// SIMD-accelerated tensor operations using System.Numerics.Tensors and
/// System.Runtime.Intrinsics where beneficial.
/// </summary>
public static class TensorOps
{
    /// <summary>
    /// Numerically-stable softmax in-place over a span.
    /// Softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Softmax(Span<float> x)
    {
        if (x.IsEmpty) return;

        // Step 1: find max for numerical stability
        float max = TensorPrimitives.Max(x);

        // Step 2: subtract max and exponentiate in-place
        TensorPrimitives.Subtract(x, max, x);
        TensorPrimitives.Exp(x, x);

        // Step 3: sum
        float sum = TensorPrimitives.Sum(x);

        // Step 4: divide by sum
        float invSum = 1.0f / sum;
        TensorPrimitives.Multiply(x, invSum, x);
    }

    /// <summary>
    /// Zero-centred RMSNorm: out = (1 + scale) * x / rms(x), epsilon = 1e-6.
    /// scale is initialized to 0 so the initial transform is identity.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ZCRMSNorm(
        ReadOnlySpan<float> x,
        ReadOnlySpan<float> scale,
        Span<float> output,
        float epsilon = 1e-6f)
    {
        if (x.Length != scale.Length || x.Length != output.Length)
            throw new ArgumentException("Input, scale, and output must have the same length.");

        int n = x.Length;

        // Compute mean of squares: sum(x_i^2) / n
        // Use TensorPrimitives.Dot(x, x) for sum of squares
        float sumSq = TensorPrimitives.Dot(x, x);
        float rms = MathF.Sqrt(sumSq / n + epsilon);
        float invRms = 1.0f / rms;

        // output[i] = (1 + scale[i]) * x[i] * invRms
        // Decompose as: output = x * invRms + scale * x * invRms
        // i.e. output = invRms * (x + scale * x) = invRms * x * (1 + scale)
        int i = 0;

        if (Vector.IsHardwareAccelerated && n >= Vector<float>.Count)
        {
            var vInvRms = new Vector<float>(invRms);
            int simdBound = n - (n % Vector<float>.Count);

            for (; i < simdBound; i += Vector<float>.Count)
            {
                var vx = new Vector<float>(x[i..]);
                var vs = new Vector<float>(scale[i..]);
                var vOut = vInvRms * (Vector<float>.One + vs) * vx;
                vOut.CopyTo(output[i..]);
            }
        }

        for (; i < n; i++)
        {
            output[i] = invRms * (1f + scale[i]) * x[i];
        }
    }

    /// <summary>
    /// L2 normalize in-place: x = x / sqrt(sum(x^2) + eps).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void L2Normalize(Span<float> x, float epsilon = 1e-12f)
    {
        if (x.IsEmpty) return;

        float sumSq = TensorPrimitives.Dot(x, x);
        float norm = MathF.Sqrt(sumSq + epsilon);
        float invNorm = 1.0f / norm;
        TensorPrimitives.Multiply(x, invNorm, x);
    }

    /// <summary>
    /// Element-wise sigmoid: output[i] = 1 / (1 + exp(-x[i])).
    /// Delegates to TensorPrimitives.Sigmoid which is SIMD-vectorised on
    /// platforms that expose AVX/SSE or equivalent instruction sets.
    /// </summary>
    public static void Sigmoid(ReadOnlySpan<float> x, Span<float> output)
    {
        if (x.Length != output.Length)
            throw new ArgumentException("Sigmoid: input and output must have the same length.");

        TensorPrimitives.Sigmoid(x, output);
    }

    /// <summary>
    /// Element-wise ReLU in-place: x[i] = max(0, x[i]).
    /// Uses System.Numerics.Vector for SIMD acceleration with a scalar tail.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ReLU(Span<float> x)
    {
        if (x.IsEmpty) return;

        int n = x.Length;
        int i = 0;

        if (Vector.IsHardwareAccelerated && n >= Vector<float>.Count)
        {
            var vZero = Vector<float>.Zero;
            int simdBound = n - (n % Vector<float>.Count);
            for (; i < simdBound; i += Vector<float>.Count)
            {
                var vx = new Vector<float>(x[i..]);
                Vector.Max(vx, vZero).CopyTo(x[i..]);
            }
        }

        // Scalar tail.
        for (; i < n; i++)
        {
            if (x[i] < 0f) x[i] = 0f;
        }
    }
}
