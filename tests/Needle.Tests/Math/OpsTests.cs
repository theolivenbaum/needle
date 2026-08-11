using Needle.Math;

namespace Needle.Tests.Math;

public class NdArrayTests
{
    [Fact]
    public void SliceSharesStorageWithTheParent()
    {
        var parent = new NdArray(3, 2, 4);
        parent.Span.Fill(1f);

        var slice = parent.Slice(1);
        Assert.Equal([2, 4], slice.Shape);
        slice.Span[0] = 7f;

        Assert.Equal(7f, parent[1 * 8 + 0]);
    }

    [Fact]
    public void ReshapeRejectsAnElementCountChange()
    {
        var value = new NdArray(4, 3);
        Assert.Equal([2, 6], value.Reshape(2, 6).Shape);
        Assert.Throws<ArgumentException>(() => value.Reshape(5, 3));
    }

    [Fact]
    public void SliceRangeNarrowsTheLeadingAxis()
    {
        var value = new NdArray(5, 2);
        for (int i = 0; i < value.Length; i++) value[i] = i;

        var window = value.SliceRange(2, 2);
        Assert.Equal([2, 2], window.Shape);
        Assert.Equal([4f, 5f, 6f, 7f], window.ReadSpan.ToArray());
    }
}

public class MatMulTests
{
    [Fact]
    public void MatchesTheNaiveTripleLoop()
    {
        var random = new Random(7);
        const int M = 37, K = 61, N = 53;

        var a = new NdArray(M, K);
        var b = new NdArray(K, N);
        for (int i = 0; i < a.Length; i++) a[i] = (float)random.NextDouble() - 0.5f;
        for (int i = 0; i < b.Length; i++) b[i] = (float)random.NextDouble() - 0.5f;

        var actual = Ops.MatMul(a, b);

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                // Accumulate in float too: a double reference would disagree
                // with any float32 kernel in the last few bits.
                float expected = 0;
                for (int p = 0; p < K; p++) expected += a[i, p] * b[p, j];
                Assert.Equal(expected, actual[i, j], 3);
            }
        }
    }

    [Fact]
    public void ParallelAndSerialPathsAgree()
    {
        // 200 rows crosses the threshold where output rows are split across cores.
        var random = new Random(11);
        const int M = 200, K = 64, N = 64;

        var a = new NdArray(M, K);
        var b = new NdArray(K, N);
        for (int i = 0; i < a.Length; i++) a[i] = (float)random.NextDouble();
        for (int i = 0; i < b.Length; i++) b[i] = (float)random.NextDouble();

        var parallel = Ops.MatMul(a, b);
        var serial = new NdArray(M, N);
        Ops.MatMul(a.ReadSpan, b.ReadSpan, serial.Span, M, K, N);

        Assert.Equal(serial.ReadSpan.ToArray(), parallel.ReadSpan.ToArray());
    }

    [Fact]
    public void TransposedFormMatchesTheDirectForm()
    {
        var random = new Random(3);
        const int M = 5, K = 8, N = 6;

        var a = new NdArray(M, K);
        var bT = new NdArray(N, K);
        for (int i = 0; i < a.Length; i++) a[i] = (float)random.NextDouble();
        for (int i = 0; i < bT.Length; i++) bT[i] = (float)random.NextDouble();

        var actual = new NdArray(M, N);
        Ops.MatMulTransposed(a.ReadSpan, bT.ReadSpan, actual.Span, M, K, N);

        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float expected = 0;
                for (int p = 0; p < K; p++) expected += a[i, p] * bT[j, p];
                Assert.Equal(expected, actual[i, j], 3);
            }
    }
}

public class NormalisationTests
{
    [Fact]
    public void ZeroScaleLeavesAPlainRmsNormalisation()
    {
        var x = new NdArray(1, 4);
        x.Span[0] = 1f; x.Span[1] = 2f; x.Span[2] = 3f; x.Span[3] = 4f;

        Ops.ZcRmsNorm(x, stackalloc float[4]);

        float rms = MathF.Sqrt((1 + 4 + 9 + 16) / 4f + Ops.Epsilon);
        Assert.Equal(1f / rms, x[0], 5);
        Assert.Equal(4f / rms, x[3], 5);
    }

    [Fact]
    public void ScaleActsAsOnePlusGamma()
    {
        var x = new NdArray(1, 2);
        x.Span[0] = 3f; x.Span[1] = 4f;
        float[] scale = [1f, 0f];

        Ops.ZcRmsNorm(x, scale);

        float rms = MathF.Sqrt((9 + 16) / 2f + Ops.Epsilon);
        Assert.Equal(2f * 3f / rms, x[0], 5);
        Assert.Equal(1f * 4f / rms, x[1], 5);
    }

    [Fact]
    public void RmsUnitProducesUnitRootMeanSquare()
    {
        var random = new Random(5);
        var x = new NdArray(3, 16);
        for (int i = 0; i < x.Length; i++) x[i] = (float)(random.NextDouble() * 10 - 5);

        Ops.RmsUnit(x);

        for (int r = 0; r < 3; r++)
        {
            double sum = 0;
            for (int c = 0; c < 16; c++) sum += x[r, c] * x[r, c];
            Assert.Equal(1.0, System.Math.Sqrt(sum / 16), 4);
        }
    }

    [Fact]
    public void SoftmaxSumsToOneAndIsShiftInvariant()
    {
        float[] a = [1f, 2f, 3f];
        float[] b = [101f, 102f, 103f];
        Ops.Softmax(a);
        Ops.Softmax(b);

        Assert.Equal(1.0, a.Sum(), 5);
        for (int i = 0; i < 3; i++) Assert.Equal(a[i], b[i], 5);
    }

    [Fact]
    public void SoftmaxTreatsFloatMinAsFullyMasked()
    {
        // The reference fills blocked scores with the float minimum rather than
        // -inf, so those entries must come out as exactly zero probability.
        float[] scores = [1f, float.MinValue, 1f];
        Ops.Softmax(scores);

        Assert.Equal(0.5, scores[0], 6);
        Assert.Equal(0.0, scores[1], 6);
        Assert.Equal(0.5, scores[2], 6);
    }

    [Fact]
    public void SiluMatchesItsDefinition()
    {
        float[] x = [-3f, -0.5f, 0f, 0.5f, 3f];
        var expected = x.Select(v => v / (1f + MathF.Exp(-v))).ToArray();

        Ops.Silu(x);

        for (int i = 0; i < x.Length; i++) Assert.Equal(expected[i], x[i], 5);
    }
}

public class WalshHadamardTests
{
    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(3, 4)]
    [InlineData(512, 512)]
    [InlineData(513, 1024)]
    [InlineData(768, 1024)]
    public void NextPow2MatchesThePythonBitLengthRule(int input, int expected) =>
        Assert.Equal(expected, WalshHadamard.NextPow2(input));

    [Fact]
    public void ButterflyMatchesAnExplicitMatrixProduct()
    {
        const int N = 16;
        var random = new Random(13);
        var x = new float[N];
        for (int i = 0; i < N; i++) x[i] = (float)random.NextDouble() - 0.5f;

        var matrix = WalshHadamard.Matrix(N);
        var expected = new float[N];
        for (int j = 0; j < N; j++)
        {
            double sum = 0;
            for (int i = 0; i < N; i++) sum += x[i] * matrix[i, j];
            expected[j] = (float)sum;
        }

        var actual = (float[])x.Clone();
        WalshHadamard.Transform(actual);

        for (int i = 0; i < N; i++) Assert.Equal(expected[i], actual[i], 5);
    }

    [Fact]
    public void TransformIsItsOwnInverse()
    {
        const int N = 128;
        var random = new Random(17);
        var original = new float[N];
        for (int i = 0; i < N; i++) original[i] = (float)random.NextDouble();

        var roundTrip = (float[])original.Clone();
        WalshHadamard.Transform(roundTrip);
        WalshHadamard.Transform(roundTrip);

        for (int i = 0; i < N; i++) Assert.Equal(original[i], roundTrip[i], 4);
    }

    [Fact]
    public void TransformRejectsNonPowerOfTwoWidths() =>
        Assert.Throws<ArgumentException>(() => WalshHadamard.Transform(new float[12]));
}

public class SinkhornTests
{
    [Fact]
    public void ProducesADoublyStochasticMatrix()
    {
        var random = new Random(19);
        const int N = 4;
        var logits = new float[N * N];
        for (int i = 0; i < logits.Length; i++) logits[i] = (float)(random.NextDouble() * 4 - 2);

        Sinkhorn.Normalize(logits, N);

        for (int i = 0; i < N; i++)
        {
            double row = 0, column = 0;
            for (int j = 0; j < N; j++)
            {
                row += logits[i * N + j];
                column += logits[j * N + i];
            }
            Assert.Equal(1.0, row, 4);
            Assert.Equal(1.0, column, 4);
        }
    }

    [Fact]
    public void StronglyDiagonalLogitsStayNearTheIdentity()
    {
        // The reference initialises the routing bias at 4*I, so an untrained
        // layer should barely mix the lanes.
        const int N = 4;
        var logits = new float[N * N];
        for (int i = 0; i < N; i++) logits[i * N + i] = 4f;

        Sinkhorn.Normalize(logits, N);

        for (int i = 0; i < N; i++) Assert.True(logits[i * N + i] > 0.9f);
    }
}
