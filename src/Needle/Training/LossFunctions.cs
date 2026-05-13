using TorchSharp;
using static TorchSharp.torch;

namespace Needle.Training;

/// <summary>
/// Loss functions for training the SimpleAttentionNetwork.
/// Port of needle/training/train.py loss helpers.
/// </summary>
public static class LossFunctions
{
    /// <summary>
    /// Token weight map for the four token classes.
    /// Index 0 = base (1.0), 1 = name (3.0), 2 = value (2.0), 3 = key (1.5).
    /// </summary>
    public static readonly float[] DefaultTokenWeightMap = [1.0f, 3.0f, 2.0f, 1.5f];

    /// <summary>
    /// Weighted cross-entropy loss over a packed decoder sequence.
    /// Port of the inline loss computation inside <c>_text_loss_fn</c> in train.py.
    /// </summary>
    /// <param name="logits">
    /// Unnormalised model outputs [batch, seqLen, vocabSize], float32.
    /// </param>
    /// <param name="targets">
    /// Ground-truth token IDs [batch, seqLen], int64.
    /// </param>
    /// <param name="tokenWeights">
    /// Per-token loss weight derived from class labels [batch, seqLen], float32.
    /// Class 0=base (weight 1.0), 1=name (3.0), 2=value (2.0), 3=key (1.5).
    /// </param>
    /// <param name="decSegIds">
    /// Decoder segment IDs [batch, seqLen], int32.
    /// Positions with segId &gt; 0 are real tokens; 0 = padding (masked out).
    /// </param>
    /// <returns>Scalar cross-entropy loss (weighted sum over real tokens / number of real tokens).</returns>
    public static Tensor TextLoss(
        Tensor logits,
        Tensor targets,
        Tensor tokenWeights,
        Tensor decSegIds)
    {
        long batch  = logits.shape[0];
        long seqLen = logits.shape[1];

        // Flatten for cross_entropy: [batch*seqLen, vocabSize] and [batch*seqLen]
        using var logitsFlat  = logits.reshape(batch * seqLen, -1);
        using var targetsFlat = targets.reshape(batch * seqLen);

        // Per-token CE loss (reduction = "none"): [batch*seqLen]
        using var ceFlat = torch.nn.functional.cross_entropy(
            logitsFlat,
            targetsFlat,
            reduction: nn.Reduction.None);

        // Reshape back to [batch, seqLen]
        using var ce = ceFlat.reshape(batch, seqLen);

        // Padding mask: 1.0 for real tokens, 0.0 for padding
        using var paddingMask = decSegIds.gt(0).to(ScalarType.Float32);

        // Combined per-token weight: tokenWeight * paddingMask
        using var mask = tokenWeights * paddingMask;

        // Normalise by the number of real tokens (not total positions)
        using var numTokens = torch.clamp(paddingMask.sum(), min: 1.0f);

        // Weighted sum / numTokens
        using var weightedCe = (ce * mask).sum();
        return weightedCe / numTokens;
    }

    /// <summary>
    /// CLIP-style symmetric contrastive loss with a learnable temperature.
    /// Port of <c>_clip_contrastive_loss</c> in train.py.
    /// </summary>
    /// <param name="qEmb">Query embeddings [batch, contrastiveDim], float32, L2-normalised.</param>
    /// <param name="tEmb">Tool embeddings  [batch, contrastiveDim], float32, L2-normalised.</param>
    /// <param name="logTemp">
    /// Scalar learnable log-temperature.
    /// Clamped to the range [-log(100), log(100)] before exponentiation, matching the
    /// reference implementation.
    /// </param>
    /// <returns>Scalar symmetric contrastive loss.</returns>
    public static Tensor ClipContrastiveLoss(Tensor qEmb, Tensor tEmb, Tensor logTemp)
    {
        // Clamp log_temp and exponentiate: temperature ∈ [1/100, 100]
        float logMax = MathF.Log(100.0f);
        using var logTempClamped = logTemp.clamp(-logMax, logMax);
        using var temp = logTempClamped.exp();

        long batchSize = qEmb.shape[0];

        // Similarity matrix [B, B]: logits[i,j] = qEmb[i] · tEmb[j] / temp
        using var tEmbT  = tEmb.t();                           // [contrastiveDim, B]
        using var sim    = torch.matmul(qEmb, tEmbT);          // [B, B]
        using var scaled = sim / temp;                          // [B, B]

        // Ground-truth labels: diagonal pairing (i pairs with i)
        using var labels = torch.arange(batchSize, device: qEmb.device, dtype: ScalarType.Int64);

        // Loss from query perspective: each query should be closest to its paired tool
        using var lossQ = torch.nn.functional.cross_entropy(scaled, labels, reduction: nn.Reduction.Mean);

        // Loss from tool perspective: transpose and repeat
        using var scaledT = scaled.t().contiguous();
        using var lossT   = torch.nn.functional.cross_entropy(scaledT, labels, reduction: nn.Reduction.Mean);

        return (lossQ + lossT) / 2.0f;
    }

    /// <summary>
    /// Z-loss regularisation: penalises large logit magnitudes to prevent softmax saturation.
    /// <c>z_loss = 1e-4 * mean(logsumexp(logits, dim=-1)^2)</c>
    /// Port of the z_loss term in <c>_text_loss_fn</c> in train.py.
    /// </summary>
    /// <param name="logits">Logit tensor of any shape [*, vocabSize], float32.</param>
    /// <returns>Scalar Z-loss.</returns>
    public static Tensor ZLoss(Tensor logits)
    {
        // logsumexp over the last dimension: [*]
        using var lse = torch.logsumexp(logits, dim: -1);

        // Square and average
        using var lse2 = lse.pow(2);
        return 1e-4f * lse2.mean();
    }

    /// <summary>
    /// Build a per-token weight tensor from class-label indices and a weight map.
    /// </summary>
    /// <param name="classLabels">Integer class labels [batch, seqLen], values 0–3.</param>
    /// <param name="weightMap">
    /// Float array of length 4 mapping class → weight.
    /// Defaults to <see cref="DefaultTokenWeightMap"/>.
    /// </param>
    /// <returns>Float32 tensor [batch, seqLen].</returns>
    public static Tensor BuildTokenWeights(Tensor classLabels, float[]? weightMap = null)
    {
        weightMap ??= DefaultTokenWeightMap;

        // Build a weight map tensor on the same device as the labels
        using var mapTensor = torch.tensor(weightMap, device: classLabels.device);

        // Index into the map using the class labels (clamp to valid range for safety)
        using var safeLabels = classLabels.to(ScalarType.Int64).clamp(0, weightMap.Length - 1);
        return mapTensor[safeLabels];
    }
}
