using Needle.Model;
using TorchSharp;
using static TorchSharp.torch;

namespace Needle.Tests.Model;

public sealed class QuantizeTests
{
    [Fact]
    public void FakeQuantizeInt4_PreservesShape()
    {
        using var w = torch.randn(64, 32);
        using var q = Quantize.FakeQuantizeInt4(w);
        Assert.Equal(w.shape, q.shape);
    }

    [Fact]
    public void FakeQuantizeInt8_PreservesShape()
    {
        using var w = torch.randn(64, 32);
        using var q = Quantize.FakeQuantizeInt8(w);
        Assert.Equal(w.shape, q.shape);
    }

    [Fact]
    public void FakeQuantizeInt4_ZeroWeights_StayZero()
    {
        using var w = torch.zeros(new long[] { 32, 8 });
        using var q = Quantize.FakeQuantizeInt4(w);
        Assert.Equal(0f, q.abs().max().item<float>(), precision: 5);
    }

    [Fact]
    public void FakeQuantizeInt4_DoesNotChangeOutputMuchForInt4Friendly()
    {
        // A weight matrix already aligned to int4 grid should be ~unchanged
        using var w = torch.ones(new long[] { 32, 8 });
        using var q = Quantize.FakeQuantizeInt4(w);
        // After fake-quant the value should still be exactly 1.0 (it's the per-group max).
        using var diff = (q - w).abs();
        Assert.True(diff.max().item<float>() < 1e-4f);
    }

    [Fact]
    public void FakeQuantizeInt4_HandlesNonAlignedInputDim()
    {
        // inFeat = 70, groupSize = 32 → needs padding to 96.
        using var w = torch.randn(70, 8);
        using var q = Quantize.FakeQuantizeInt4(w, groupSize: 32);
        Assert.Equal(new long[] { 70, 8 }, q.shape);
    }

    [Fact]
    public void FakeQuantizeInt4_GroupSizeLargerThanInDim_StillWorks()
    {
        // inFeat = 5, groupSize = 32 → effective gs clamped to 5
        using var w = torch.randn(5, 4);
        using var q = Quantize.FakeQuantizeInt4(w, groupSize: 32);
        Assert.Equal(new long[] { 5, 4 }, q.shape);
    }

    [Fact]
    public void FakeQuantizeInt8_NarrowerErrorThanInt4()
    {
        // INT8 should have lower reconstruction error than INT4 on the same data
        using var w  = torch.randn(64, 32);
        using var q4 = Quantize.FakeQuantizeInt4(w);
        using var q8 = Quantize.FakeQuantizeInt8(w);
        float err4 = (q4 - w).pow(2).mean().item<float>();
        float err8 = (q8 - w).pow(2).mean().item<float>();
        Assert.True(err8 <= err4, $"INT8 error ({err8}) should be ≤ INT4 error ({err4})");
    }

    [Fact]
    public void QuantizeParams_QuantizesOnlyWeights()
    {
        var dict = new Dictionary<string, Tensor>
        {
            ["m.weight"]      = torch.randn(64, 32),
            ["m.bias"]        = torch.randn(32),
            ["m.scale"]       = torch.randn(32),  // 1D, untouched
        };

        using var origWeightView = dict["m.weight"].clone();
        Quantize.QuantizeParams(dict);

        // weight tensor pointer changed (replaced with quantised version)
        Assert.Equal(new long[] { 64, 32 }, dict["m.weight"].shape);
        // 1D params untouched
        Assert.Equal(new long[] { 32 }, dict["m.bias"].shape);
        Assert.Equal(new long[] { 32 }, dict["m.scale"].shape);

        // Released to avoid leak (in case the test runner doesn't GC promptly)
        foreach (var v in dict.Values) v.Dispose();
    }

    [Fact]
    public void QuantizeParams_Quantizes3DWeightSliceWise()
    {
        var dict = new Dictionary<string, Tensor>
        {
            ["stack.weight"] = torch.randn(3, 64, 32),
        };
        Quantize.QuantizeParams(dict);
        Assert.Equal(new long[] { 3, 64, 32 }, dict["stack.weight"].shape);
        foreach (var v in dict.Values) v.Dispose();
    }
}
