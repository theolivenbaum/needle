using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using Needle.Model;

namespace Needle.Training;

// ---------------------------------------------------------------------------
// Configuration records
// ---------------------------------------------------------------------------

/// <summary>Configuration for a training run.</summary>
public record TrainingConfig
{
    public int   Epochs          { get; init; } = 3;
    public int   BatchSize       { get; init; } = 32;
    public float AdamLr          { get; init; } = 1e-3f;
    public float MuonLr          { get; init; } = 0.02f;
    public float WarmupRatio     { get; init; } = 0.05f;
    public float DecayRatio      { get; init; } = 0.15f;
    public float WeightDecay     { get; init; } = 0.01f;
    public float GradClipNorm    { get; init; } = 1.0f;
    public float ContrastiveWeight { get; init; } = 0.1f;
    public int   MaxEncLen       { get; init; } = 1024;
    public int   MaxDecLen       { get; init; } = 512;
    public int   Seed            { get; init; } = 42;
    public string CheckpointDir  { get; init; } = "checkpoints";
    public int   EvalEvery       { get; init; } = 100;

    /// <summary>
    /// Per-class loss weights: [base=1.0, name=3.0, value=2.0, key=1.5].
    /// </summary>
    public float[] TokenWeightMap { get; init; } = [1.0f, 3.0f, 2.0f, 1.5f];
}

/// <summary>One packed batch of training examples.</summary>
public sealed record TrainingBatch(
    long[,] SrcTokens,    // [batch, encLen]
    long[,] TgtInTokens,  // [batch, decLen]
    long[,] TgtOutTokens, // [batch, decLen]
    int[,]  LossMask,     // [batch, decLen] — class labels 0-3
    int[,]  EncSegIds,    // [batch, encLen]
    int[,]  DecSegIds     // [batch, decLen]
);

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// <summary>
/// Manages one training loop iteration over the SimpleAttentionNetwork.
///
/// Applies:
/// <list type="bullet">
///   <item>Adam for embeddings, biases, and 1-D parameters.</item>
///   <item>Muon (orthogonalised gradient + Nesterov) for 2-D/3-D weight matrices.</item>
///   <item>Global gradient-norm clipping before parameter updates.</item>
///   <item>Warmup-Stable-Decay learning rate schedule for both optimisers.</item>
/// </list>
///
/// Port of the training loop in needle/training/train.py.
/// </summary>
public sealed class Trainer : IDisposable
{
    private readonly SimpleAttentionNetwork _model;
    private readonly TransformerConfig      _config;
    private readonly TrainingConfig         _trainingConfig;

    private readonly TorchSharp.Modules.Adam _adam;
    private readonly MuonOptimizer _muon;

    private readonly WSDSchedule _adamSchedule;
    private readonly WSDSchedule _muonSchedule;

    private int _globalStep;
    private bool _disposed;

    public int   GlobalStep    => _globalStep;
    public float CurrentAdamLr => _adamSchedule.GetLr(_globalStep);
    public float CurrentMuonLr => _muonSchedule.GetLr(_globalStep);

    public Trainer(
        SimpleAttentionNetwork model,
        TransformerConfig config,
        TrainingConfig trainingConfig,
        int totalSteps)
    {
        _model          = model;
        _config         = config;
        _trainingConfig = trainingConfig;

        int warmupSteps = System.Math.Max(1, (int)(totalSteps * trainingConfig.WarmupRatio));

        _adamSchedule = new WSDSchedule(trainingConfig.AdamLr, totalSteps, warmupSteps, trainingConfig.DecayRatio);
        _muonSchedule = new WSDSchedule(trainingConfig.MuonLr, totalSteps, warmupSteps, trainingConfig.DecayRatio);

        // Partition model parameters into "muon" (2D/3D kernels) and "adam" (everything else)
        var muonParams = new List<(string, Tensor)>();
        var adamParams = new List<TorchSharp.Modules.Parameter>();

        foreach (var (name, param) in model.named_parameters())
        {
            if (param.dim() >= 2 && name.EndsWith(".weight", StringComparison.Ordinal))
                muonParams.Add((name, param));
            else
                adamParams.Add(param);
        }

        _adam = optim.Adam(adamParams, lr: trainingConfig.AdamLr, weight_decay: trainingConfig.WeightDecay);
        _muon = new MuonOptimizer(muonParams, lr: trainingConfig.MuonLr,
                                  momentum: 0.95f, nsSteps: 5,
                                  weightDecay: trainingConfig.WeightDecay);
    }

    // ── Training step ─────────────────────────────────────────────────────────

    /// <summary>
    /// Perform a single training step (forward + backward + update).
    /// </summary>
    /// <returns>(totalLoss, textLoss, gradNorm)</returns>
    public (float TotalLoss, float TextLoss, float GradNorm) TrainStep(TrainingBatch batch)
    {
        _model.train();
        _adam.zero_grad();

        // Move batch to tensors
        using var src     = torch.tensor(batch.SrcTokens.Cast<long>().ToArray(),
                                         new long[] { batch.SrcTokens.GetLength(0), batch.SrcTokens.GetLength(1) });
        using var tgtIn   = torch.tensor(batch.TgtInTokens.Cast<long>().ToArray(),
                                         new long[] { batch.TgtInTokens.GetLength(0), batch.TgtInTokens.GetLength(1) });
        using var tgtOut  = torch.tensor(batch.TgtOutTokens.Cast<long>().ToArray(),
                                         new long[] { batch.TgtOutTokens.GetLength(0), batch.TgtOutTokens.GetLength(1) });

        var lmFlat   = batch.LossMask.Cast<int>().ToArray();
        var esFlat   = batch.EncSegIds.Cast<int>().ToArray();
        var dsFlat   = batch.DecSegIds.Cast<int>().ToArray();

        long B = batch.SrcTokens.GetLength(0);
        long encLen = batch.SrcTokens.GetLength(1);
        long decLen = batch.TgtInTokens.GetLength(1);

        using var lossMaskTensor = torch.tensor(lmFlat, new long[] { B, decLen }, dtype: ScalarType.Int32);
        using var encSegTensor   = torch.tensor(esFlat, new long[] { B, encLen }, dtype: ScalarType.Int32);
        using var decSegTensor   = torch.tensor(dsFlat, new long[] { B, decLen }, dtype: ScalarType.Int32);

        // Build masks
        using var srcMask   = MaskUtils.MakePackingMask(encSegTensor);
        using var tgtMask   = MaskUtils.MakeCausalPackingMask(decSegTensor);
        using var crossMask = MaskUtils.MakeCrossPackingMask(encSegTensor, decSegTensor);

        // Forward
        using var logits = _model.Forward(src, tgtIn, srcMask, tgtMask, crossMask);

        // Build token weights from class labels
        using var tokenWeights = LossFunctions.BuildTokenWeights(lossMaskTensor, _trainingConfig.TokenWeightMap);

        // Text loss + Z-loss
        using var textLoss = LossFunctions.TextLoss(logits, tgtOut, tokenWeights, decSegTensor);
        using var zLoss    = LossFunctions.ZLoss(logits);
        using var totalLoss = textLoss + zLoss;

        float textLossVal  = textLoss.item<float>();
        float totalLossVal = totalLoss.item<float>();

        // Backward
        totalLoss.backward();

        // Collect gradients for Muon parameters (2D/3D kernels)
        var muonGrads = new Dictionary<string, Tensor>();
        foreach (var (name, param) in _model.named_parameters())
        {
            if (param.dim() >= 2 && name.EndsWith(".weight", StringComparison.Ordinal)
                && param.grad is not null)
            {
                muonGrads[name] = param.grad;
            }
        }

        // Global gradient-norm clipping (Adam params)
        float gradNorm = (float)nn.utils.clip_grad_norm_(_model.parameters(), _trainingConfig.GradClipNorm);

        // Update learning rates via param groups
        foreach (var pg in _adam.ParamGroups)
            pg.LearningRate = CurrentAdamLr;
        _muon.Lr = CurrentMuonLr;

        // Adam step (biases, norms, embeddings, 1-D params)
        _adam.step();

        // Muon step (2D/3D kernels)
        _muon.Step(muonGrads);

        _globalStep++;
        return (totalLossVal, textLossVal, gradNorm);
    }

    // ── Validation step ───────────────────────────────────────────────────────

    /// <summary>
    /// Compute validation loss without updating parameters.
    /// </summary>
    /// <returns>(sumLoss, numTokens) for perplexity computation.</returns>
    public (float SumLoss, int NumTokens) ValidateStep(TrainingBatch batch)
    {
        _model.eval();

        using var noGrad = torch.no_grad();

        long B      = batch.SrcTokens.GetLength(0);
        long encLen = batch.SrcTokens.GetLength(1);
        long decLen = batch.TgtInTokens.GetLength(1);

        using var src    = torch.tensor(batch.SrcTokens.Cast<long>().ToArray(),   new long[] { B, encLen });
        using var tgtIn  = torch.tensor(batch.TgtInTokens.Cast<long>().ToArray(), new long[] { B, decLen });
        using var tgtOut = torch.tensor(batch.TgtOutTokens.Cast<long>().ToArray(), new long[] { B, decLen });

        var esFlat = batch.EncSegIds.Cast<int>().ToArray();
        var dsFlat = batch.DecSegIds.Cast<int>().ToArray();

        using var encSegTensor = torch.tensor(esFlat, new long[] { B, encLen }, dtype: ScalarType.Int32);
        using var decSegTensor = torch.tensor(dsFlat, new long[] { B, decLen }, dtype: ScalarType.Int32);

        using var srcMask   = MaskUtils.MakePackingMask(encSegTensor);
        using var tgtMask   = MaskUtils.MakeCausalPackingMask(decSegTensor);
        using var crossMask = MaskUtils.MakeCrossPackingMask(encSegTensor, decSegTensor);

        using var logits = _model.Forward(src, tgtIn, srcMask, tgtMask, crossMask);

        // Uniform-weight CE for consistent perplexity measurement
        using var paddingMask = decSegTensor.gt(0).to(ScalarType.Float32);

        long seqLen = logits.shape[1];
        using var logitsFlat  = logits.reshape(B * seqLen, -1);
        using var targetsFlat = tgtOut.reshape(B * seqLen);
        using var ceLoss = torch.nn.functional.cross_entropy(logitsFlat, targetsFlat, reduction: nn.Reduction.None);
        using var ceShaped = ceLoss.reshape(B, seqLen);

        float sumLoss  = (ceShaped * paddingMask).sum().item<float>();
        int numTokens  = (int)paddingMask.sum().item<float>();

        return (sumLoss, numTokens);
    }

    // ── IDisposable ───────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _adam.Dispose();
    }
}
