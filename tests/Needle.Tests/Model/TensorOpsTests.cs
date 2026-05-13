using Needle.Math;

namespace Needle.Tests.Model;

public sealed class TensorOpsTests
{
    // ── Softmax ───────────────────────────────────────────────────────────────

    [Fact]
    public void Softmax_AllZero_ProducesUniform()
    {
        float[] x = [0f, 0f, 0f, 0f];
        TensorOps.Softmax(x);
        Assert.All(x, v => Assert.Equal(0.25f, v, precision: 5));
    }

    [Fact]
    public void Softmax_SumsToOne()
    {
        float[] x = [1f, 2f, 3f, 4f];
        TensorOps.Softmax(x);
        Assert.Equal(1f, x.Sum(), precision: 5);
    }

    [Fact]
    public void Softmax_LargeValues_Stable()
    {
        // Would overflow without the max subtraction trick
        float[] x = [1000f, 1001f, 1002f];
        TensorOps.Softmax(x);
        Assert.Equal(1f, x.Sum(), precision: 5);
        Assert.True(x[2] > x[1] && x[1] > x[0]);
    }

    [Fact]
    public void Softmax_SingleElement_IsOne()
    {
        float[] x = [42f];
        TensorOps.Softmax(x);
        Assert.Equal(1f, x[0], precision: 6);
    }

    // ── ZCRMSNorm ─────────────────────────────────────────────────────────────

    [Fact]
    public void ZCRMSNorm_ZeroScale_IsPlainRMSNorm()
    {
        // scale = 0 → output = x / rms(x)
        float[] x     = [3f, 4f, 0f];
        float[] scale = [0f, 0f, 0f];
        float[] output = new float[3];
        TensorOps.ZCRMSNorm(x, scale, output);

        // rms = sqrt((9+16+0)/3) = sqrt(25/3) ≈ 2.887
        double rms = System.Math.Sqrt((9.0 + 16.0) / 3.0 + 1e-6);
        Assert.Equal((float)(3.0 / rms), output[0], precision: 4);
        Assert.Equal((float)(4.0 / rms), output[1], precision: 4);
    }

    [Fact]
    public void ZCRMSNorm_LengthMismatch_Throws()
    {
        float[] x = [1f, 2f];
        float[] scale = [0f];
        float[] output = new float[2];
        Assert.Throws<ArgumentException>(() => TensorOps.ZCRMSNorm(x, scale, output));
    }

    // ── L2Normalize ───────────────────────────────────────────────────────────

    [Fact]
    public void L2Normalize_UnitVector_IsIdempotent()
    {
        float[] x = [1f, 0f, 0f];
        TensorOps.L2Normalize(x);
        Assert.Equal(1f, x[0], precision: 6);
        Assert.Equal(0f, x[1], precision: 6);
    }

    [Fact]
    public void L2Normalize_ProducesUnitNorm()
    {
        float[] x = [3f, 4f];
        TensorOps.L2Normalize(x);
        double norm = System.Math.Sqrt(x[0] * x[0] + x[1] * x[1]);
        Assert.Equal(1.0, norm, precision: 5);
    }

    // ── Sigmoid ───────────────────────────────────────────────────────────────

    [Fact]
    public void Sigmoid_ZeroInput_IsHalf()
    {
        float[] x = [0f];
        float[] output = new float[1];
        TensorOps.Sigmoid(x, output);
        Assert.Equal(0.5f, output[0], precision: 6);
    }

    [Fact]
    public void Sigmoid_OutputInRange()
    {
        float[] x = [-10f, 0f, 10f];
        float[] output = new float[3];
        TensorOps.Sigmoid(x, output);
        Assert.All(output, v => { Assert.True(v > 0f); Assert.True(v < 1f); });
    }

    // ── ReLU ─────────────────────────────────────────────────────────────────

    [Fact]
    public void ReLU_ClampsNegativeToZero()
    {
        float[] x = [-5f, -0.001f, 0f, 0.001f, 5f];
        TensorOps.ReLU(x);
        Assert.Equal(0f, x[0]);
        Assert.Equal(0f, x[1]);
        Assert.Equal(0f, x[2]);
        Assert.Equal(0.001f, x[3], precision: 6);
        Assert.Equal(5f, x[4], precision: 6);
    }
}
