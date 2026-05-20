using TorchSharp;
using static TorchSharp.torch;
using Needle.Model;

namespace Needle.Training;

/// <summary>
/// Validation-loss / perplexity computation over packed batches.
/// Port of <c>compute_perplexity_packed</c> in <c>needle/training/eval.py</c>
/// and <c>_make_val_loss_fn</c> in <c>train.py</c>.
/// </summary>
public static class PerplexityEval
{
    /// <summary>
    /// Per-batch CE-sum + token-count, used to aggregate perplexity across a
    /// pre-packed validation set.
    /// </summary>
    public readonly record struct LossSample(float SumLoss, long NumTokens)
    {
        public LossSample Plus(LossSample other) =>
            new(SumLoss + other.SumLoss, NumTokens + other.NumTokens);
    }

    /// <summary>
    /// Compute the (sum-loss, num-tokens) contribution of a single packed batch
    /// to the dataset-wide perplexity.
    ///
    /// Uses uniform per-token weights (matches the Python validation function,
    /// which deliberately ignores the class-weight map for stable PPL reporting).
    /// </summary>
    public static LossSample EvalBatch(
        SimpleAttentionNetwork model,
        TrainingBatch batch)
    {
        model.eval();
        using var noGrad = torch.no_grad();

        long B      = batch.SrcTokens.GetLength(0);
        long encLen = batch.SrcTokens.GetLength(1);
        long decLen = batch.TgtInTokens.GetLength(1);

        using var src    = torch.tensor(batch.SrcTokens.Cast<long>().ToArray(),    new long[] { B, encLen });
        using var tgtIn  = torch.tensor(batch.TgtInTokens.Cast<long>().ToArray(),  new long[] { B, decLen });
        using var tgtOut = torch.tensor(batch.TgtOutTokens.Cast<long>().ToArray(), new long[] { B, decLen });

        using var encSeg = torch.tensor(batch.EncSegIds.Cast<int>().ToArray(), new long[] { B, encLen },
                                        dtype: ScalarType.Int32);
        using var decSeg = torch.tensor(batch.DecSegIds.Cast<int>().ToArray(), new long[] { B, decLen },
                                        dtype: ScalarType.Int32);

        using var srcMask   = MaskUtils.MakePackingMask(encSeg);
        using var tgtMask   = MaskUtils.MakeCausalPackingMask(decSeg);
        using var crossMask = MaskUtils.MakeCrossPackingMask(encSeg, decSeg);

        using var logits = model.Forward(src, tgtIn, srcMask, tgtMask, crossMask);

        long seqLen = logits.shape[1];
        using var logitsFlat  = logits.reshape(B * seqLen, -1);
        using var targetsFlat = tgtOut.reshape(B * seqLen);

        using var ceFlat = torch.nn.functional.cross_entropy(logitsFlat, targetsFlat, reduction: nn.Reduction.None);
        using var ce     = ceFlat.reshape(B, seqLen);

        using var paddingMask = decSeg.gt(0).to(ScalarType.Float32);
        using var masked      = ce * paddingMask;

        float sumLoss  = masked.sum().item<float>();
        long  numToks  = (long)paddingMask.sum().item<float>();
        return new LossSample(sumLoss, numToks);
    }

    /// <summary>
    /// Aggregate per-batch loss samples into a final perplexity value.
    /// </summary>
    /// <param name="samples">Per-batch (sumLoss, numTokens) pairs.</param>
    /// <param name="clampNll">
    /// Maximum NLL value used when exponentiating to keep the result finite
    /// (Python clamps at 20 — i.e. e^20 ≈ 4.85e8).
    /// </param>
    public static double Perplexity(IEnumerable<LossSample> samples, float clampNll = 20f)
    {
        var total = samples.Aggregate(new LossSample(0f, 0L), (acc, s) => acc.Plus(s));
        if (total.NumTokens <= 0) return double.NaN;
        double avgNll = total.SumLoss / total.NumTokens;
        return System.Math.Exp(System.Math.Min(avgNll, clampNll));
    }

    /// <summary>
    /// One-shot helper: evaluate <paramref name="batches"/> against the model
    /// and return the aggregate perplexity.
    /// </summary>
    public static double ComputePerplexity(SimpleAttentionNetwork model, IEnumerable<TrainingBatch> batches)
    {
        var samples = new List<LossSample>();
        foreach (var batch in batches)
            samples.Add(EvalBatch(model, batch));
        return Perplexity(samples);
    }
}
