using TorchSharp;
using static TorchSharp.torch;
using Needle.Model;

namespace Needle.Tests.Model;

/// <summary>
/// Smoke tests for the SimpleAttentionNetwork forward pass.
/// Uses small configs so they run quickly on CPU without a GPU.
/// </summary>
public sealed class NeedleModelTests
{
    private static TransformerConfig SmallConfig() => new TransformerConfig
    {
        VocabSize        = 64,
        DModel           = 16,
        NumHeads         = 2,
        NumKvHeads       = 1,
        NumEncoderLayers = 1,
        NumDecoderLayers = 1,
        DFf              = 32,
        MaxSeqLen        = 8,
        ContrastiveDim   = 8,
        NoFeedforward    = true,
    };

    [Fact]
    public void ZCRMSNorm_Forward_Shape()
    {
        var norm = new ZCRMSNorm(dim: 16);
        using var x = torch.randn(2, 4, 16);
        using var y = norm.forward(x);
        Assert.Equal(x.shape, y.shape);
    }

    [Fact]
    public void ZCRMSNorm_ScaleZero_NormIsApproxOne()
    {
        // With scale = 0 the output should have unit-ish RMS
        var norm = new ZCRMSNorm(dim: 8);
        using var x = torch.randn(1, 8) * 5f;   // large variance
        using var y = norm.forward(x);
        float[] yf = y.data<float>().ToArray();
        double rms = System.Math.Sqrt(yf.Select(v => v * v).Average());
        Assert.Equal(1.0, rms, precision: 1);
    }

    [Fact]
    public void EncodeText_OutputShape()
    {
        var cfg   = SmallConfig();
        var model = new SimpleAttentionNetwork(cfg);
        model.eval();

        int B = 2, T = 5;
        using var src = torch.randint(1, cfg.VocabSize, new long[] { B, T });
        var (enc, _) = model.EncodeText(src);
        Assert.Equal(new long[] { B, T, cfg.DModel }, enc.shape);
        enc.Dispose();
    }

    [Fact]
    public void Decode_OutputShape()
    {
        var cfg   = SmallConfig();
        var model = new SimpleAttentionNetwork(cfg);
        model.eval();

        int B = 1, Te = 4, Td = 3;
        using var src     = torch.randint(1, cfg.VocabSize, new long[] { B, Te });
        using var tgt     = torch.randint(1, cfg.VocabSize, new long[] { B, Td });
        var (enc, encMsk) = model.EncodeText(src);
        using var logits  = model.Decode(tgt, enc, crossMask: encMsk);

        Assert.Equal(new long[] { B, Td, cfg.VocabSize }, logits.shape);
        enc.Dispose();
    }

    [Fact]
    public void EncodeContrastive_OutputNormalized()
    {
        var cfg   = SmallConfig();
        var model = new SimpleAttentionNetwork(cfg);
        model.eval();

        using var tokens = torch.randint(1, cfg.VocabSize, new long[] { 3, 4 });
        using var embs   = model.EncodeContrastive(tokens);

        // Check shape
        Assert.Equal(new long[] { 3, cfg.ContrastiveDim }, embs.shape);

        // Check (approx) unit norm along dim=1
        using var norms = embs.norm(dim: 1);
        float[] nf = norms.data<float>().ToArray();
        Assert.All(nf, n => Assert.Equal(1f, n, precision: 3));
    }

    [Fact]
    public void Forward_FullPass_ProducesLogits()
    {
        var cfg   = SmallConfig();
        var model = new SimpleAttentionNetwork(cfg);
        model.eval();

        int B = 1, Te = 3, Td = 2;
        using var src    = torch.randint(1, cfg.VocabSize, new long[] { B, Te });
        using var tgt    = torch.randint(1, cfg.VocabSize, new long[] { B, Td });
        using var logits = model.Forward(src, tgt);

        Assert.Equal(new long[] { B, Td, cfg.VocabSize }, logits.shape);
        // Logits should be finite
        Assert.False(logits.isnan().any().item<bool>(), "Logits contain NaN");
        Assert.False(logits.isinf().any().item<bool>(), "Logits contain Inf");
    }
}
