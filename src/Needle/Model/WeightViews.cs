using System.Numerics.Tensors;
using Needle.Math;
using Needle.Weights;

namespace Needle.Model;

/// <summary>
/// A projection the model multiplies by, whatever form its weights are stored in.
///
/// The two implementations differ only in memory: <see cref="DenseLinear"/> holds
/// float32 in the checkpoint's <c>[in, out]</c> layout, <see cref="QuantizedLinear"/>
/// holds Cactus-Quant codes in the blob's <c>[out, in]</c> layout and never
/// expands them.  The model does not care which it has.
/// </summary>
public abstract class LinearWeight
{
    /// <summary>Width of the activation this consumes.</summary>
    public abstract int InputWidth { get; }

    /// <summary>Width of the activation this produces.</summary>
    public abstract int OutputWidth { get; }

    /// <summary>Bytes of weight storage — what decode throughput is bound by.</summary>
    public abstract long ByteSize { get; }

    /// <summary>Multiply <c>[tokens, InputWidth]</c> into <c>[tokens, OutputWidth]</c>.</summary>
    /// <param name="x">Activations.</param>
    /// <param name="destination">Result.</param>
    /// <param name="shared">
    /// Rotation shared with the other projections reading the same activation.
    /// Packed weights need one; float32 weights ignore it.
    /// </param>
    public abstract void Apply(NdArray x, NdArray destination, PreparedActivation? shared = null);

    /// <summary>Multiply, allocating the result.</summary>
    public NdArray Apply(NdArray x)
    {
        int tokens = x.Length / InputWidth;
        var destination = new NdArray(tokens, OutputWidth);
        Apply(x, destination);
        return destination;
    }

    /// <summary>Multiply into a tensor taken from <paramref name="scratch"/>.</summary>
    public NdArray Apply(NdArray x, ScratchArena scratch, PreparedActivation? shared = null)
    {
        int tokens = x.Length / InputWidth;
        var destination = scratch.Take(tokens, OutputWidth, clear: false);
        Apply(x, destination, shared);
        return destination;
    }

    /// <summary>
    /// Reconstruct as a float32 <c>[in, out]</c> kernel — the checkpoint layout,
    /// used by the training flow and by tests that compare the two paths.
    /// </summary>
    public abstract NdArray ToDenseKernel();

    /// <summary>Wrap a float32 <c>[in, out]</c> kernel.</summary>
    public static LinearWeight Dense(NdArray kernel) => new DenseLinear(kernel);
}

/// <summary>A float32 projection in the checkpoint's <c>[in, out]</c> layout.</summary>
public sealed class DenseLinear(NdArray kernel) : LinearWeight
{
    /// <summary>The underlying kernel, <c>[in, out]</c>.</summary>
    public NdArray Kernel { get; } = kernel.Rank == 2
        ? kernel
        : throw new ArgumentException($"Expected a rank-2 kernel, got {NdArray.Describe(kernel.Shape)}.");

    public override int InputWidth => Kernel.Shape[0];

    public override int OutputWidth => Kernel.Shape[1];

    public override long ByteSize => (long)Kernel.Length * sizeof(float);

    public override void Apply(NdArray x, NdArray destination, PreparedActivation? shared = null) =>
        Ops.MatMulParallel(x, Kernel, destination, x.Length / InputWidth, InputWidth, OutputWidth);

    public override NdArray ToDenseKernel() => Kernel;
}

/// <summary>
/// A projection held as Cactus-Quant codes, multiplied without expanding them.
/// </summary>
public sealed class QuantizedLinear(QuantizedMatrix matrix) : LinearWeight
{
    /// <summary>The packed matrix, <c>[out, in]</c>.</summary>
    public QuantizedMatrix Matrix { get; } = matrix;

    public override int InputWidth => Matrix.Width;

    public override int OutputWidth => Matrix.Rows;

    public override long ByteSize => Matrix.ByteSize;

    public override void Apply(NdArray x, NdArray destination, PreparedActivation? shared = null) =>
        Matrix.Apply(x, destination, shared);

    public override NdArray ToDenseKernel()
    {
        // Stored [out, in]; the checkpoint layout is [in, out].
        var packed = Matrix.ToDense();
        var kernel = new NdArray(InputWidth, OutputWidth);
        for (int o = 0; o < OutputWidth; o++)
            for (int i = 0; i < InputWidth; i++)
                kernel[i * OutputWidth + o] = packed[o * InputWidth + i];
        return kernel;
    }
}

/// <summary>
/// The token embedding, which the model uses twice: to look up input rows and,
/// tied, to project hidden states back onto the vocabulary.
/// </summary>
public abstract class EmbeddingTable
{
    /// <summary>Number of tokens.</summary>
    public abstract int VocabSize { get; }

    /// <summary>Embedding width.</summary>
    public abstract int Width { get; }

    /// <summary>Bytes of storage.</summary>
    public abstract long ByteSize { get; }

    /// <summary>Copy one token's embedding.</summary>
    public abstract void Row(int id, Span<float> destination);

    /// <summary>Tied projection: <c>[tokens, Width] → [tokens, VocabSize]</c>.</summary>
    public abstract void Project(NdArray hidden, NdArray destination);

    /// <summary>Reconstruct as float32 <c>[vocab, width]</c>.</summary>
    public abstract NdArray ToDense();
}

/// <summary>A float32 embedding table.</summary>
public sealed class DenseEmbedding(NdArray table) : EmbeddingTable
{
    /// <summary>The underlying table, <c>[vocab, width]</c>.</summary>
    public NdArray Table { get; } = table;

    public override int VocabSize => Table.Shape[0];

    public override int Width => Table.Shape[1];

    public override long ByteSize => (long)Table.Length * sizeof(float);

    public override void Row(int id, Span<float> destination) => Table.ReadRow(id).CopyTo(destination);

    public override void Project(NdArray hidden, NdArray destination) =>
        Ops.MatMulTransposed(hidden.ReadSpan, Table.ReadSpan, destination.Span,
                             hidden.Length / Width, Width, VocabSize);

    public override NdArray ToDense() => Table;
}

/// <summary>A Cactus-Quant embedding table.</summary>
public sealed class QuantizedEmbedding(QuantizedMatrix matrix) : EmbeddingTable
{
    /// <summary>The packed table, <c>[vocab, width]</c>.</summary>
    public QuantizedMatrix Matrix { get; } = matrix;

    public override int VocabSize => Matrix.Rows;

    public override int Width => Matrix.Width;

    public override long ByteSize => Matrix.ByteSize;

    public override void Row(int id, Span<float> destination) => Matrix.DequantizeRow(id, destination);

    public override void Project(NdArray hidden, NdArray destination) => Matrix.Apply(hidden, destination);

    public override NdArray ToDense() => Matrix.ToDense();
}

/// <summary>
/// One engram site's hash tables.  These are read by gather rather than
/// multiplied, so the quantised form reconstructs the handful of rows a token
/// touches instead of transforming the activation.
/// </summary>
public abstract class EngramTable
{
    /// <summary>Number of hash tables at this site.</summary>
    public abstract int Tables { get; }

    /// <summary>Rows per table.</summary>
    public abstract int Slots { get; }

    /// <summary>Width of a row.</summary>
    public abstract int SubDim { get; }

    /// <summary>Bytes of storage.</summary>
    public abstract long ByteSize { get; }

    /// <summary>Copy row <paramref name="slot"/> of table <paramref name="table"/>.</summary>
    public abstract void Row(int table, int slot, Span<float> destination);

    /// <summary>Reconstruct as float32 <c>[tables, slots, subDim]</c>.</summary>
    public abstract NdArray ToDense();
}

/// <summary>Float32 engram tables.</summary>
public sealed class DenseEngramTable(NdArray tables) : EngramTable
{
    /// <summary>The underlying tables, <c>[tables, slots, subDim]</c>.</summary>
    public NdArray Tables_ { get; } = tables;

    public override int Tables => Tables_.Shape[0];

    public override int Slots => Tables_.Shape[1];

    public override int SubDim => Tables_.Shape[2];

    public override long ByteSize => (long)Tables_.Length * sizeof(float);

    public override void Row(int table, int slot, Span<float> destination) =>
        Tables_.ReadSpan.Slice((table * Slots + slot) * SubDim, SubDim).CopyTo(destination);

    public override NdArray ToDense() => Tables_;
}

/// <summary>Cactus-Quant engram tables, stored flat as <c>[tables * slots, subDim]</c>.</summary>
public sealed class QuantizedEngramTable(QuantizedMatrix matrix, int tables) : EngramTable
{
    /// <summary>The packed tables.</summary>
    public QuantizedMatrix Matrix { get; } = matrix;

    public override int Tables { get; } = tables;

    public override int Slots => Matrix.Rows / Tables;

    public override int SubDim => Matrix.Width;

    public override long ByteSize => Matrix.ByteSize;

    public override void Row(int table, int slot, Span<float> destination) =>
        Matrix.DequantizeRow(table * Slots + slot, destination);

    public override NdArray ToDense()
    {
        var flat = Matrix.ToDense();
        return flat.Reshape(Tables, Slots, SubDim);
    }
}
