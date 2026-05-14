using Needle.Model;
using Needle.Training;
using TorchSharp;
using static TorchSharp.torch;

namespace Needle.Tests.Training;

/// <summary>
/// Integration tests for the <see cref="Trainer"/>: ensure that
/// forward+backward+optimizer step runs and that loss actually decreases
/// when the model fits a tiny synthetic batch repeatedly.
/// </summary>
public sealed class TrainerIntegrationTests
{
    private static TransformerConfig TinyConfig() => new TransformerConfig
    {
        VocabSize        = 32,
        DModel           = 16,
        NumHeads         = 2,
        NumKvHeads       = 1,
        NumEncoderLayers = 1,
        NumDecoderLayers = 1,
        DFf              = 32,
        MaxSeqLen        = 16,
        ContrastiveDim   = 8,
        NoFeedforward    = true,
    };

    private static TrainingBatch MakeTinyBatch(int batchSize = 2, int encLen = 4, int decLen = 4)
    {
        var rng = new Random(123);

        var src = new long[batchSize, encLen];
        var tgtIn = new long[batchSize, decLen];
        var tgtOut = new long[batchSize, decLen];
        var lossMask = new int[batchSize, decLen];
        var encSeg = new int[batchSize, encLen];
        var decSeg = new int[batchSize, decLen];

        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < encLen; t++)
            {
                src[b, t]    = rng.Next(1, 30);
                encSeg[b, t] = 1; // single segment, non-padding
            }
            for (int t = 0; t < decLen; t++)
            {
                tgtIn[b, t]    = rng.Next(1, 30);
                tgtOut[b, t]   = rng.Next(1, 30);
                lossMask[b, t] = 0;
                decSeg[b, t]   = 1;
            }
        }

        return new TrainingBatch(src, tgtIn, tgtOut, lossMask, encSeg, decSeg);
    }

    [Fact]
    public void TrainStep_Runs_And_ReturnsFiniteLoss()
    {
        var cfg = TinyConfig();
        var model = new SimpleAttentionNetwork(cfg);

        var trainingCfg = new TrainingConfig
        {
            AdamLr      = 1e-3f,
            MuonLr      = 0.02f,
            WeightDecay = 0.0f,  // Disable weight decay for stability in tiny test
            WarmupRatio = 0.0f,
            DecayRatio  = 0.0f,
            GradClipNorm = 1.0f,
        };

        using var trainer = new Trainer(model, cfg, trainingCfg, totalSteps: 10);

        var batch = MakeTinyBatch();
        var (totalLoss, textLoss, gradNorm) = trainer.TrainStep(batch);

        Assert.False(float.IsNaN(totalLoss), $"totalLoss NaN");
        Assert.False(float.IsInfinity(totalLoss), $"totalLoss Inf");
        Assert.False(float.IsNaN(textLoss), $"textLoss NaN");
        Assert.True(textLoss >= 0f, $"textLoss must be ≥ 0, got {textLoss}");
        Assert.True(gradNorm >= 0f, $"gradNorm must be ≥ 0, got {gradNorm}");
        Assert.Equal(1, trainer.GlobalStep);
    }

    [Fact]
    public void TrainStep_MultipleSteps_LossGenerallyDecreases()
    {
        var cfg = TinyConfig();
        torch.manual_seed(42);
        var model = new SimpleAttentionNetwork(cfg);

        var trainingCfg = new TrainingConfig
        {
            AdamLr      = 5e-3f,
            MuonLr      = 0.05f,
            WeightDecay = 0.0f,
            WarmupRatio = 0.0f,
            DecayRatio  = 0.0f,
            GradClipNorm = 5.0f,
        };

        using var trainer = new Trainer(model, cfg, trainingCfg, totalSteps: 30);

        // Fix the batch — overfit on a single example so loss must drop
        var batch = MakeTinyBatch(batchSize: 2, encLen: 4, decLen: 4);

        float firstLoss = 0f;
        float finalLoss = 0f;
        for (int step = 0; step < 30; step++)
        {
            var (_, textLoss, _) = trainer.TrainStep(batch);
            if (step == 0)  firstLoss = textLoss;
            if (step == 29) finalLoss = textLoss;
            Assert.False(float.IsNaN(textLoss), $"NaN at step {step}");
        }

        Assert.True(finalLoss < firstLoss,
            $"Expected loss to decrease — first={firstLoss}, final={finalLoss}");
    }

    [Fact]
    public void ValidateStep_NoExceptions_ProducesNonNegativeLoss()
    {
        var cfg = TinyConfig();
        var model = new SimpleAttentionNetwork(cfg);

        var trainingCfg = new TrainingConfig();
        using var trainer = new Trainer(model, cfg, trainingCfg, totalSteps: 10);

        var batch = MakeTinyBatch();
        var (sumLoss, numTokens) = trainer.ValidateStep(batch);

        Assert.True(sumLoss >= 0f, $"sumLoss must be ≥ 0, got {sumLoss}");
        Assert.True(numTokens > 0,  $"numTokens must be > 0, got {numTokens}");
    }
}
