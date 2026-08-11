using Needle.Math;
using Needle.Weights;

namespace Needle.Tests.Weights;

/// <summary>
/// The packed matmul against the same matrix reconstructed to float32.
///
/// <see cref="QuantizedMatrix.Dot"/> never materialises a weight: it rotates the
/// activation instead and dots it against codebook indices, which is only valid
/// because the group rotation is symmetric and orthonormal.  That identity is
/// what the hot loop is allowed to assume, so it is asserted directly here rather
/// than left to the end-to-end parity run — a hand-vectorised kernel needs a test
/// that fails on the kernel, not one that fails on the whole model.
///
/// The packed bytes are arbitrary: the identity holds for any bitstream, so the
/// test does not need an encoder to produce a realistic one.
/// </summary>
public class QuantizedMatrixTests
{
    private const int GroupSize = 64;

    /// <summary>A matrix of pseudo-random codes with pseudo-random group norms.</summary>
    private static QuantizedMatrix Build(int rows, int width, int bits, int seed = 7)
    {
        int padded = (width + GroupSize - 1) / GroupSize * GroupSize;
        int groups = padded / GroupSize;
        int rowBytes = CactusQuant.PackedRowBytes(padded, bits);

        var random = new Random(seed);
        var packed = new byte[rows * rowBytes];
        random.NextBytes(packed);

        var norms = new float[rows * groups];
        for (int i = 0; i < norms.Length; i++) norms[i] = (float)(0.2 + random.NextDouble());

        int entries = 1 << (bits == CactusQuant.TernaryRecordBits ? 2 : bits);
        var codebook = new float[entries];
        for (int i = 0; i < entries; i++) codebook[i] = (float)(random.NextDouble() - 0.5) * 2f;

        return new QuantizedMatrix(packed, norms, rows, width, bits, GroupSize, codebook);
    }

    private static float[] Activation(int width, int seed = 11)
    {
        var random = new Random(seed);
        var x = new float[width];
        for (int i = 0; i < width; i++) x[i] = (float)(random.NextDouble() - 0.5) * 3f;
        return x;
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(CactusQuant.TernaryRecordBits)]
    public void DotMatchesTheDenseReconstruction(int bits)
    {
        var matrix = Build(rows: 37, width: 192, bits);
        var dense = matrix.ToDense();
        var x = Activation(matrix.Width);

        var prepared = new float[matrix.PaddedWidth];
        matrix.PrepareInput(x, prepared);

        for (int row = 0; row < matrix.Rows; row++)
        {
            float expected = 0f;
            var denseRow = dense.ReadRow(row);
            for (int i = 0; i < matrix.Width; i++) expected += x[i] * denseRow[i];

            // Both sides accumulate hundreds of products in float32 in different
            // orders, so the comparison is relative to the magnitude involved.
            float actual = matrix.Dot(prepared, row);
            Assert.True(System.Math.Abs(actual - expected) <= 2e-3f * (1f + System.Math.Abs(expected)),
                $"{bits}-bit row {row}: expected {expected:G6}, got {actual:G6}");
        }
    }

    [Fact]
    public void ApplyMatchesRowWiseDot()
    {
        var matrix = Build(rows: 20, width: 128, bits: 2);
        var x = new NdArray(3, matrix.Width);
        var source = Activation(matrix.Width * 3, seed: 13);
        source.CopyTo(x.Span);

        var actual = new NdArray(3, matrix.Rows);
        matrix.Apply(x, actual);

        var prepared = new float[matrix.PaddedWidth];
        for (int t = 0; t < 3; t++)
        {
            matrix.PrepareInput(x.ReadRow(t), prepared);
            for (int row = 0; row < matrix.Rows; row++)
                Assert.Equal(matrix.Dot(prepared, row), actual[t * matrix.Rows + row], 4);
        }
    }

    /// <summary>
    /// A width that does not fill its last group: the padding must behave exactly
    /// as if the weight columns under it had been dropped.
    /// </summary>
    [Fact]
    public void PaddingDoesNotLeakIntoTheProduct()
    {
        var matrix = Build(rows: 8, width: 100, bits: 2);
        Assert.Equal(128, matrix.PaddedWidth);

        var x = Activation(matrix.Width);
        var prepared = new float[matrix.PaddedWidth];
        // Dirty the scratch so a kernel that forgets to zero the padding fails.
        prepared.AsSpan().Fill(float.NaN);
        matrix.PrepareInput(x, prepared);

        for (int row = 0; row < matrix.Rows; row++)
            Assert.True(float.IsFinite(matrix.Dot(prepared, row)), $"row {row} picked up the padding");
    }
}
