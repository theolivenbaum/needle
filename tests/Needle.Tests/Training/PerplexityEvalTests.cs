using Needle.Model;
using Needle.Training;
using TorchSharp;
using static TorchSharp.torch;

namespace Needle.Tests.Training;

public sealed class PerplexityEvalTests
{
    private static TransformerConfig SmallConfig() => new TransformerConfig
    {
        VocabSize        = 16,
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
    public void Perplexity_AggregatesSamples()
    {
        // Two batches: (loss=4, tokens=2) and (loss=6, tokens=4)
        var samples = new List<PerplexityEval.LossSample>
        {
            new(4f, 2L),
            new(6f, 4L),
        };
        // avg nll = (4+6)/(2+4) = 10/6 ≈ 1.6667
        // ppl = e^1.6667 ≈ 5.294
        double ppl = PerplexityEval.Perplexity(samples);
        Assert.InRange(ppl, 5.2, 5.4);
    }

    [Fact]
    public void Perplexity_EmptySamples_ReturnsNaN()
    {
        Assert.True(double.IsNaN(PerplexityEval.Perplexity(Array.Empty<PerplexityEval.LossSample>())));
    }

    [Fact]
    public void Perplexity_ClampsLargeNll()
    {
        var samples = new[] { new PerplexityEval.LossSample(1e10f, 1L) };
        double ppl = PerplexityEval.Perplexity(samples, clampNll: 20f);
        Assert.True(double.IsFinite(ppl));
        Assert.InRange(ppl, System.Math.Exp(20) - 1, System.Math.Exp(20) + 1);
    }

    [Fact]
    public void EvalBatch_RunsOnSmallModel()
    {
        var cfg   = SmallConfig();
        var model = new SimpleAttentionNetwork(cfg);
        model.eval();

        // Build a tiny "packed" batch with single-example packing.
        int B = 2, encLen = 4, decLen = 4;
        var src    = new long[B, encLen];
        var tgtIn  = new long[B, decLen];
        var tgtOut = new long[B, decLen];
        var lm     = new int [B, decLen];
        var encSeg = new int [B, encLen];
        var decSeg = new int [B, decLen];

        for (int i = 0; i < B; i++)
        {
            for (int j = 0; j < encLen; j++)
            {
                src[i, j]    = j + 1;
                encSeg[i, j] = 1;
            }
            for (int j = 0; j < decLen; j++)
            {
                tgtIn[i, j]  = j + 1;
                tgtOut[i, j] = (j + 2) % cfg.VocabSize;
                decSeg[i, j] = 1;
            }
        }

        var batch  = new TrainingBatch(src, tgtIn, tgtOut, lm, encSeg, decSeg);
        var sample = PerplexityEval.EvalBatch(model, batch);

        Assert.True(sample.SumLoss > 0f);
        Assert.Equal(B * decLen, sample.NumTokens);
    }
}
