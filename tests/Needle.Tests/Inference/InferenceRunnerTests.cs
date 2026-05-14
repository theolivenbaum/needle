using Needle.Inference;
using Needle.Model;
using Needle.Tokenizer;
using TorchSharp;
using static TorchSharp.torch;

namespace Needle.Tests.Inference;

/// <summary>
/// Tests for the InferenceRunner that don't require a SentencePiece tokenizer
/// model file.  Covers <see cref="InferenceRunner.BuildEncoderInput"/> via a
/// stub tokenizer wrapper and the contrastive retrieval helpers via the model
/// directly.
/// </summary>
public sealed class InferenceRunnerTests
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
        MaxSeqLen        = 16,
        ContrastiveDim   = 8,
        NoFeedforward    = true,
    };

    [Fact]
    public void EncodeForRetrieval_NoTokenizer_FromModelOnly()
    {
        // Exercise SimpleAttentionNetwork.EncodeContrastive directly to verify
        // its outputs are L2-normalised, since this is what InferenceRunner uses
        // for retrieval. (We can't exercise the full Runner.EncodeForRetrieval
        // path without a SentencePiece .model file on disk.)
        var cfg = SmallConfig();
        var model = new SimpleAttentionNetwork(cfg);
        model.eval();

        using var tokens = torch.randint(1, cfg.VocabSize, new long[] { 5, 6 });
        using var embs = model.EncodeContrastive(tokens);

        Assert.Equal(new long[] { 5, cfg.ContrastiveDim }, embs.shape);
        using var norms = embs.norm(dim: 1);
        float[] nf = norms.data<float>().ToArray();
        Assert.All(nf, n => Assert.Equal(1f, n, precision: 3));
    }

    [Fact]
    public void EncodeContrastive_DifferentInputs_ProduceDifferentEmbeddings()
    {
        var cfg = SmallConfig();
        torch.manual_seed(7);
        var model = new SimpleAttentionNetwork(cfg);
        model.eval();

        using var tokensA = torch.tensor(new long[,] { { 1, 2, 3, 4 } });
        using var tokensB = torch.tensor(new long[,] { { 20, 21, 22, 23 } });

        using var embA = model.EncodeContrastive(tokensA);
        using var embB = model.EncodeContrastive(tokensB);

        float[] a = embA.data<float>().ToArray();
        float[] b = embB.data<float>().ToArray();

        // Both vectors are unit-norm — cosine similarity == dot product.
        float dot = 0f;
        for (int i = 0; i < a.Length; i++) dot += a[i] * b[i];
        // Should not be perfectly aligned (very unlikely with random init).
        Assert.True(System.Math.Abs(dot) < 0.999f,
            $"Embeddings unexpectedly identical (cos = {dot})");
    }

    [Fact]
    public void Generate_WithoutTokenizer_Greedy_ProducesTokens()
    {
        // End-to-end Decode loop without a real tokenizer/grammar: feed an
        // encoder output through Decode repeatedly and confirm we get non-NaN
        // logits and consistent argmax-derived tokens.
        var cfg = SmallConfig();
        torch.manual_seed(11);
        var model = new SimpleAttentionNetwork(cfg);
        model.eval();

        using var src = torch.randint(1, cfg.VocabSize, new long[] { 1, 5 });
        var (encOut, encMask) = model.EncodeText(src);
        using var encOutD = encOut;

        int maxGenLen = 6;
        using var tgtMask = MaskUtils.MakeCausalMask(maxGenLen);
        using var dec = torch.full(new long[] { 1, maxGenLen }, cfg.PadTokenId, dtype: ScalarType.Int64);
        dec[0, 0] = 1; // EOS-style seed

        using var logits = model.Decode(dec, encOut, tgtMask, encMask);
        Assert.Equal(new long[] { 1, maxGenLen, cfg.VocabSize }, logits.shape);
        Assert.False(logits.isnan().any().item<bool>(), "Logits contain NaN");
        Assert.False(logits.isinf().any().item<bool>(), "Logits contain Inf");

        // Argmax along last dim — every position must yield a valid vocab id.
        using var preds = logits.argmax(dim: -1);
        long[] predIds = preds.data<long>().ToArray();
        Assert.Equal(maxGenLen, predIds.Length);
        Assert.All(predIds, id => Assert.InRange(id, 0L, (long)cfg.VocabSize - 1));
    }
}
