using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Needle.Math;

/// <summary>
/// The two inner loops of attention, over key and value rows gathered from a KV
/// cache by absolute position.
///
/// Written out here rather than expressed as library calls because of the shape
/// of the work: a head is 64 floats, and a decode step runs about 110,000 of
/// these row operations — eight heads against a 256-position window, twenty-seven
/// layers over. At that size a call into a general vector primitive costs several
/// times the four fused multiply-adds it performs, so the loop is fused instead:
/// the accumulator stays in registers across four gathered rows, and there is one
/// call for the whole window rather than one per row.
///
/// Rows are gathered rather than contiguous because the sliding window keeps
/// pinned sink positions visible after they fall out of it, so the set a query
/// may read is not an interval.
/// </summary>
public static class AttentionKernels
{
    /// <summary>
    /// Scores for one query head: <c>destination[i] = scale * dot(query,
    /// rows[positions[i]] slice)</c>.
    /// </summary>
    /// <param name="query">The query head, <c>headDim</c> long.</param>
    /// <param name="rows">The cache, <c>[capacity, rowStride]</c>.</param>
    /// <param name="positions">Absolute positions to read.</param>
    /// <param name="rowStride">Floats per cache row.</param>
    /// <param name="columnOffset">Where this head starts within a row.</param>
    /// <param name="scale">Applied to every score.</param>
    /// <param name="destination">One score per position.</param>
    public static void Scores(
        ReadOnlySpan<float> query, ReadOnlySpan<float> rows, ReadOnlySpan<int> positions,
        int rowStride, int columnOffset, float scale, Span<float> destination)
    {
        int headDim = query.Length;
        int count = positions.Length;
        ref float rowHead = ref Unsafe.Add(ref Unsafe.AsRef(in MemoryMarshal.GetReference(rows)), columnOffset);
        ref float queryHead = ref Unsafe.AsRef(in MemoryMarshal.GetReference(query));

        if (Vector512.IsHardwareAccelerated && headDim % 16 == 0)
        {
            for (int i = 0; i < count; i++)
            {
                ref float key = ref Unsafe.Add(ref rowHead, (nint)positions[i] * rowStride);
                var acc = Vector512<float>.Zero;
                for (int j = 0; j < headDim; j += 16)
                {
                    acc += Vector512.LoadUnsafe(ref queryHead, (nuint)j)
                           * Vector512.LoadUnsafe(ref key, (nuint)j);
                }
                destination[i] = Vector512.Sum(acc) * scale;
            }
            return;
        }

        if (Vector256.IsHardwareAccelerated && headDim % 8 == 0)
        {
            for (int i = 0; i < count; i++)
            {
                ref float key = ref Unsafe.Add(ref rowHead, (nint)positions[i] * rowStride);
                var acc = Vector256<float>.Zero;
                for (int j = 0; j < headDim; j += 8)
                {
                    acc += Vector256.LoadUnsafe(ref queryHead, (nuint)j)
                           * Vector256.LoadUnsafe(ref key, (nuint)j);
                }
                destination[i] = Vector256.Sum(acc) * scale;
            }
            return;
        }

        for (int i = 0; i < count; i++)
        {
            ref float key = ref Unsafe.Add(ref rowHead, (nint)positions[i] * rowStride);
            float sum = 0f;
            for (int j = 0; j < headDim; j++) sum += Unsafe.Add(ref queryHead, j) * Unsafe.Add(ref key, j);
            destination[i] = sum * scale;
        }
    }

    /// <summary>
    /// The attention output for one head:
    /// <c>destination += sum_i weights[i] * rows[positions[i]] slice</c>.
    /// <paramref name="destination"/> is overwritten, not accumulated into.
    /// </summary>
    /// <param name="destination">The head's output, <c>headDim</c> long.</param>
    /// <param name="rows">The cache, <c>[capacity, rowStride]</c>.</param>
    /// <param name="positions">Absolute positions to read.</param>
    /// <param name="rowStride">Floats per cache row.</param>
    /// <param name="columnOffset">Where this head starts within a row.</param>
    /// <param name="weights">One weight per position — the softmaxed scores.</param>
    public static void Combine(
        Span<float> destination, ReadOnlySpan<float> rows, ReadOnlySpan<int> positions,
        int rowStride, int columnOffset, ReadOnlySpan<float> weights)
    {
        destination.Clear();
        int headDim = destination.Length;
        int count = positions.Length;
        if (count == 0) return;

        ref float rowHead = ref Unsafe.Add(ref Unsafe.AsRef(in MemoryMarshal.GetReference(rows)), columnOffset);
        ref float outHead = ref MemoryMarshal.GetReference(destination);

        // Four gathered rows per pass, so the output vector is read back and
        // written out a quarter as often as it would be one row at a time.
        int i = 0;
        if (Vector512.IsHardwareAccelerated && headDim % 16 == 0)
        {
            for (; i + 4 <= count; i += 4)
            {
                ref float v0 = ref Unsafe.Add(ref rowHead, (nint)positions[i] * rowStride);
                ref float v1 = ref Unsafe.Add(ref rowHead, (nint)positions[i + 1] * rowStride);
                ref float v2 = ref Unsafe.Add(ref rowHead, (nint)positions[i + 2] * rowStride);
                ref float v3 = ref Unsafe.Add(ref rowHead, (nint)positions[i + 3] * rowStride);
                var w0 = Vector512.Create(weights[i]);
                var w1 = Vector512.Create(weights[i + 1]);
                var w2 = Vector512.Create(weights[i + 2]);
                var w3 = Vector512.Create(weights[i + 3]);

                for (int j = 0; j < headDim; j += 16)
                {
                    var acc = Vector512.LoadUnsafe(ref outHead, (nuint)j)
                              + w0 * Vector512.LoadUnsafe(ref v0, (nuint)j)
                              + w1 * Vector512.LoadUnsafe(ref v1, (nuint)j)
                              + w2 * Vector512.LoadUnsafe(ref v2, (nuint)j)
                              + w3 * Vector512.LoadUnsafe(ref v3, (nuint)j);
                    Vector512.StoreUnsafe(acc, ref outHead, (nuint)j);
                }
            }
        }
        else if (Vector256.IsHardwareAccelerated && headDim % 8 == 0)
        {
            for (; i + 4 <= count; i += 4)
            {
                ref float v0 = ref Unsafe.Add(ref rowHead, (nint)positions[i] * rowStride);
                ref float v1 = ref Unsafe.Add(ref rowHead, (nint)positions[i + 1] * rowStride);
                ref float v2 = ref Unsafe.Add(ref rowHead, (nint)positions[i + 2] * rowStride);
                ref float v3 = ref Unsafe.Add(ref rowHead, (nint)positions[i + 3] * rowStride);
                var w0 = Vector256.Create(weights[i]);
                var w1 = Vector256.Create(weights[i + 1]);
                var w2 = Vector256.Create(weights[i + 2]);
                var w3 = Vector256.Create(weights[i + 3]);

                for (int j = 0; j < headDim; j += 8)
                {
                    var acc = Vector256.LoadUnsafe(ref outHead, (nuint)j)
                              + w0 * Vector256.LoadUnsafe(ref v0, (nuint)j)
                              + w1 * Vector256.LoadUnsafe(ref v1, (nuint)j)
                              + w2 * Vector256.LoadUnsafe(ref v2, (nuint)j)
                              + w3 * Vector256.LoadUnsafe(ref v3, (nuint)j);
                    Vector256.StoreUnsafe(acc, ref outHead, (nuint)j);
                }
            }
        }

        for (; i < count; i++)
        {
            ref float value = ref Unsafe.Add(ref rowHead, (nint)positions[i] * rowStride);
            float weight = weights[i];
            for (int j = 0; j < headDim; j++)
                Unsafe.Add(ref outHead, j) += weight * Unsafe.Add(ref value, j);
        }
    }
}
