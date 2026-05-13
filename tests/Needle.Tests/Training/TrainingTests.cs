using Needle.Training;

namespace Needle.Tests.Training;

public sealed class WSDScheduleTests
{
    [Fact]
    public void WSDSchedule_WarmupPhase_LinearIncrease()
    {
        var sched = new WSDSchedule(peakValue: 1.0f, totalSteps: 100, warmupSteps: 10);
        float lr0 = sched.GetLr(0);
        float lr5 = sched.GetLr(4);
        float lr9 = sched.GetLr(9);
        Assert.True(lr0 < lr5, "LR should increase during warmup");
        Assert.True(lr5 < lr9, "LR should keep increasing during warmup");
        Assert.Equal(1.0f, lr9, precision: 3);
    }

    [Fact]
    public void WSDSchedule_StablePhase_ConstantPeak()
    {
        var sched = new WSDSchedule(peakValue: 0.5f, totalSteps: 100, warmupSteps: 10, decayRatio: 0.15f);
        float lr20 = sched.GetLr(20);
        float lr50 = sched.GetLr(50);
        Assert.Equal(0.5f, lr20, precision: 5);
        Assert.Equal(0.5f, lr50, precision: 5);
    }

    [Fact]
    public void WSDSchedule_DecayPhase_Decreases()
    {
        var sched = new WSDSchedule(peakValue: 1.0f, totalSteps: 100, warmupSteps: 10, decayRatio: 0.20f);
        // Stable phase ends at step 90 (100 - 10 warmup - 20 decay = 70 stable → ends at 80)
        // Actually stable = 100 - 10 - 20 = 70; stable phase is steps 10..79, decay is 80..99
        float lrEarlyDecay = sched.GetLr(85);
        float lrLateDecay  = sched.GetLr(99);
        Assert.True(lrEarlyDecay > lrLateDecay, "LR should decrease during decay");
        Assert.True(lrLateDecay > 0f, "LR floor should be > 0 (alphaMin=0.05)");
    }

    [Fact]
    public void WSDSchedule_NeverNegative()
    {
        var sched = new WSDSchedule(1.0f, 50, 5);
        for (int i = 0; i <= 60; i++)
            Assert.True(sched.GetLr(i) >= 0f, $"LR negative at step {i}");
    }
}

public sealed class LossFunctionTests
{
    [Fact]
    public void ZLoss_Scalar_NonNegative()
    {
        using var logits = TorchSharp.torch.randn(2, 4, 16);
        using var z = LossFunctions.ZLoss(logits);
        Assert.True(z.item<float>() >= 0f);
    }

    [Fact]
    public void BuildTokenWeights_MapsCorrectly()
    {
        // class labels: 0=base(1.0), 1=name(3.0), 2=value(2.0), 3=key(1.5)
        using var labels = TorchSharp.torch.tensor(new int[,] { {0, 1, 2, 3} },
                                                   dtype: TorchSharp.torch.ScalarType.Int32);
        using var weights = LossFunctions.BuildTokenWeights(labels);
        float[] w = weights.data<float>().ToArray();
        Assert.Equal(1.0f, w[0], precision: 5);
        Assert.Equal(3.0f, w[1], precision: 5);
        Assert.Equal(2.0f, w[2], precision: 5);
        Assert.Equal(1.5f, w[3], precision: 5);
    }

    [Fact]
    public void TextLoss_AllPaddingMasked_IsZero()
    {
        // If all decSegIds == 0 (all padding), loss should be 0
        using var logits   = TorchSharp.torch.randn(1, 4, 8);
        using var targets  = TorchSharp.torch.zeros(new long[] { 1, 4 }, dtype: TorchSharp.torch.ScalarType.Int64);
        using var weights  = TorchSharp.torch.ones(new long[] { 1, 4 }, dtype: TorchSharp.torch.ScalarType.Float32);
        using var segIds   = TorchSharp.torch.zeros(new long[] { 1, 4 }, dtype: TorchSharp.torch.ScalarType.Int32);
        using var loss     = LossFunctions.TextLoss(logits, targets, weights, segIds);
        Assert.Equal(0f, loss.item<float>(), precision: 5);
    }
}

public sealed class NewtonSchulzTests
{
    [Fact]
    public void Compute_2D_PreservesShape()
    {
        using var G = TorchSharp.torch.randn(4, 6);
        using var result = NewtonSchulz.Compute(G, steps: 3);
        Assert.Equal(G.shape, result.shape);
    }

    [Fact]
    public void Compute_3D_PreservesShape()
    {
        using var G = TorchSharp.torch.randn(3, 4, 6);
        using var result = NewtonSchulz.Compute(G, steps: 3);
        Assert.Equal(G.shape, result.shape);
    }

    [Fact]
    public void Compute_Tall_Matrix_PreservesShape()
    {
        using var G = TorchSharp.torch.randn(8, 3);  // tall (m > n)
        using var result = NewtonSchulz.Compute(G, steps: 3);
        Assert.Equal(G.shape, result.shape);
    }
}
