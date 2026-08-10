using System.Runtime.CompilerServices;

namespace Needle.Math;

/// <summary>
/// A dense, row-major float32 tensor over a plain <c>float[]</c>.
///
/// Deliberately minimal: the model only ever needs contiguous storage, a shape,
/// and cheap views onto sub-blocks.  Everything numeric happens through
/// <see cref="Span{T}"/>, so the SIMD helpers in <see cref="Ops"/> and
/// <c>System.Numerics.Tensors</c> can work on it without copying.
/// </summary>
public sealed class NdArray
{
    /// <summary>Backing storage.  May be longer than this view — see <see cref="Offset"/>.</summary>
    public float[] Buffer { get; }

    /// <summary>Index in <see cref="Buffer"/> where this view starts.</summary>
    public int Offset { get; }

    /// <summary>Row-major dimensions.</summary>
    public int[] Shape { get; }

    /// <summary>Total element count.</summary>
    public int Length { get; }

    public NdArray(params int[] shape) : this(new float[Count(shape)], 0, shape) { }

    public NdArray(float[] buffer, int offset, params int[] shape)
    {
        Shape = shape;
        Length = Count(shape);
        if (offset < 0 || offset + Length > buffer.Length)
            throw new ArgumentOutOfRangeException(nameof(offset),
                $"View of {Length} elements at offset {offset} does not fit a buffer of {buffer.Length}.");
        Buffer = buffer;
        Offset = offset;
    }

    /// <summary>Number of dimensions.</summary>
    public int Rank => Shape.Length;

    /// <summary>Length of the last axis (0 for a scalar).</summary>
    public int LastDim => Shape.Length == 0 ? 1 : Shape[^1];

    /// <summary>Mutable view of the whole tensor.</summary>
    public Span<float> Span => Buffer.AsSpan(Offset, Length);

    /// <summary>Read-only view of the whole tensor.</summary>
    public ReadOnlySpan<float> ReadSpan => Buffer.AsSpan(Offset, Length);

    /// <summary>Product of a shape's dimensions.</summary>
    public static int Count(ReadOnlySpan<int> shape)
    {
        int n = 1;
        foreach (int d in shape)
        {
            if (d < 0) throw new ArgumentException("Shape dimensions must be non-negative.");
            n = checked(n * d);
        }
        return n;
    }

    /// <summary>Elements per index along axis 0 (i.e. the size of one leading slice).</summary>
    public int OuterStride
    {
        get
        {
            int n = 1;
            for (int i = 1; i < Shape.Length; i++) n *= Shape[i];
            return n;
        }
    }

    /// <summary>
    /// A view of <c>this[index]</c> — one slice along axis 0, sharing storage.
    /// </summary>
    public NdArray Slice(int index)
    {
        if (Shape.Length == 0) throw new InvalidOperationException("Cannot slice a scalar.");
        if ((uint)index >= (uint)Shape[0])
            throw new ArgumentOutOfRangeException(nameof(index), $"Axis 0 has length {Shape[0]}.");
        int stride = OuterStride;
        return new NdArray(Buffer, Offset + index * stride, Shape[1..]);
    }

    /// <summary>
    /// A view of <c>this[start .. start+count]</c> along axis 0, sharing storage.
    /// </summary>
    public NdArray SliceRange(int start, int count)
    {
        if (Shape.Length == 0) throw new InvalidOperationException("Cannot slice a scalar.");
        if (start < 0 || count < 0 || start + count > Shape[0])
            throw new ArgumentOutOfRangeException(nameof(start), $"Axis 0 has length {Shape[0]}.");
        int stride = OuterStride;
        var shape = (int[])Shape.Clone();
        shape[0] = count;
        return new NdArray(Buffer, Offset + start * stride, shape);
    }

    /// <summary>Row <paramref name="index"/> of a rank-2 tensor as a span.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<float> Row(int index)
    {
        int width = LastDim;
        return Buffer.AsSpan(Offset + index * width, width);
    }

    /// <summary>Row <paramref name="index"/> of a rank-2 tensor as a read-only span.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ReadOnlySpan<float> ReadRow(int index)
    {
        int width = LastDim;
        return Buffer.AsSpan(Offset + index * width, width);
    }

    /// <summary>
    /// Reinterpret this tensor with a different shape.  The element count must
    /// match; storage is shared, never copied.
    /// </summary>
    public NdArray Reshape(params int[] shape)
    {
        int n = Count(shape);
        if (n != Length)
            throw new ArgumentException(
                $"Cannot reshape {Describe(Shape)} ({Length} elements) to {Describe(shape)} ({n}).");
        return new NdArray(Buffer, Offset, shape);
    }

    /// <summary>A fresh tensor with the same shape and contents.</summary>
    public NdArray Clone()
    {
        var copy = new NdArray((int[])Shape.Clone());
        ReadSpan.CopyTo(copy.Span);
        return copy;
    }

    /// <summary>A fresh zero tensor with the same shape.</summary>
    public NdArray ZerosLike() => new((int[])Shape.Clone());

    /// <summary>Element access by flat index.</summary>
    public float this[int flatIndex]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Buffer[Offset + flatIndex];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set => Buffer[Offset + flatIndex] = value;
    }

    /// <summary>Element access for a rank-2 tensor.</summary>
    public float this[int i, int j]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => Buffer[Offset + i * Shape[1] + j];
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set => Buffer[Offset + i * Shape[1] + j] = value;
    }

    /// <summary>Throws unless the shape matches <paramref name="expected"/> exactly.</summary>
    public NdArray Expect(params int[] expected)
    {
        if (!Shape.AsSpan().SequenceEqual(expected))
            throw new InvalidOperationException(
                $"Expected shape {Describe(expected)}, got {Describe(Shape)}.");
        return this;
    }

    /// <summary>Human-readable shape, e.g. <c>[27, 512, 512]</c>.</summary>
    public static string Describe(ReadOnlySpan<int> shape) => "[" + string.Join(", ", shape.ToArray()) + "]";

    public override string ToString() => $"NdArray{Describe(Shape)}";
}
