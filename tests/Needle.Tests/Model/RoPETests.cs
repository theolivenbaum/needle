using TorchSharp;
using static TorchSharp.torch;
using Needle.Model;

namespace Needle.Tests.Model;

public sealed class RoPETests
{
    [Fact]
    public void PrecomputeFreqs_OutputShape()
    {
        var (cos, sin) = RoPE.PrecomputeFreqs(headDim: 8, seqLen: 16);
        Assert.Equal([16, 4], cos.shape);
        Assert.Equal([16, 4], sin.shape);
        cos.Dispose(); sin.Dispose();
    }

    [Fact]
    public void PrecomputeFreqs_CosInMinusOneToOne()
    {
        var (cos, sin) = RoPE.PrecomputeFreqs(headDim: 4, seqLen: 8);
        float[] cosData = cos.data<float>().ToArray();
        Assert.All(cosData, v => { Assert.True(v >= -1.01f); Assert.True(v <= 1.01f); });
        cos.Dispose(); sin.Dispose();
    }

    [Fact]
    public void ApplyRope_PreservesShape()
    {
        int B = 2, H = 4, T = 8, D = 16;
        using var x = torch.randn(B, H, T, D);
        var (cos, sin) = RoPE.PrecomputeFreqs(headDim: D, seqLen: T);
        using var result = RoPE.ApplyRope(x, cos, sin);
        Assert.Equal(x.shape, result.shape);
        cos.Dispose(); sin.Dispose();
    }

    [Fact]
    public void ApplyRope_ZeroInput_StaysZero()
    {
        int D = 8;
        using var x = torch.zeros(1, 1, 4, D);
        var (cos, sin) = RoPE.PrecomputeFreqs(headDim: D, seqLen: 4);
        using var result = RoPE.ApplyRope(x, cos, sin);
        float maxAbs = result.abs().max().item<float>();
        Assert.Equal(0f, maxAbs, precision: 6);
        cos.Dispose(); sin.Dispose();
    }
}
