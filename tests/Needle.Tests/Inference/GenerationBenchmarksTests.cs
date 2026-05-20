using Needle.Inference;

namespace Needle.Tests.Inference;

/// <summary>
/// Tests for pure-functional benchmark helpers in
/// <see cref="GenerationBenchmarks"/>.  The model-dependent helpers
/// (throughput, generation quality, retrieval) require a loaded model
/// and are exercised separately in the integration suite.
/// </summary>
public sealed class GenerationBenchmarksTests
{
    // ── ComputeRepetitionRate ────────────────────────────────────────────────

    [Fact]
    public void RepetitionRate_NoRepetition_ReturnsZero()
    {
        var texts = new[] { "the quick brown fox jumps" };
        Assert.Equal(0.0, GenerationBenchmarks.ComputeRepetitionRate(texts));
    }

    [Fact]
    public void RepetitionRate_AllSame_ReturnsHigh()
    {
        // "aa aa aa aa" → bigrams (aa,aa) repeated 3x → 2/3 repetition.
        var rate = GenerationBenchmarks.ComputeRepetitionRate(new[] { "aa aa aa aa" });
        Assert.Equal(2.0 / 3.0, rate, 6);
    }

    [Fact]
    public void RepetitionRate_ShortTexts_TreatedAsZero()
    {
        var texts = new[] { "single", "" };
        Assert.Equal(0.0, GenerationBenchmarks.ComputeRepetitionRate(texts));
    }

    [Fact]
    public void RepetitionRate_AveragesAcrossTexts()
    {
        // First text: no repetition (0.0).
        // Second text: 1 unique bigram of 3 → 2/3.
        // Mean = 1/3.
        var texts = new[] { "a b c d", "x x x x" };
        var rate = GenerationBenchmarks.ComputeRepetitionRate(texts);
        Assert.Equal((0.0 + 2.0 / 3.0) / 2.0, rate, 6);
    }

    [Fact]
    public void RepetitionRate_CaseInsensitive()
    {
        var lower = GenerationBenchmarks.ComputeRepetitionRate(new[] { "The dog the dog" });
        var mixed = GenerationBenchmarks.ComputeRepetitionRate(new[] { "THE dog THE dog" });
        Assert.Equal(lower, mixed, 6);
    }

    // ── ComputeWer ───────────────────────────────────────────────────────────

    [Fact]
    public void Wer_Identical_ReturnsZero()
    {
        var refs  = new[] { "the quick brown fox" };
        var hyps  = new[] { "the quick brown fox" };
        Assert.Equal(0.0, GenerationBenchmarks.ComputeWer(hyps, refs));
    }

    [Fact]
    public void Wer_SingleSubstitution_ReturnsQuarter()
    {
        var refs  = new[] { "the quick brown fox" };
        var hyps  = new[] { "the quick brown cat" };
        Assert.Equal(0.25, GenerationBenchmarks.ComputeWer(hyps, refs), 6);
    }

    [Fact]
    public void Wer_AllWrong_ReturnsOne()
    {
        var refs  = new[] { "a b c" };
        var hyps  = new[] { "x y z" };
        Assert.Equal(1.0, GenerationBenchmarks.ComputeWer(hyps, refs), 6);
    }

    [Fact]
    public void Wer_EmptyHypothesis_ReturnsOne()
    {
        var refs  = new[] { "hello world" };
        var hyps  = new[] { "" };
        Assert.Equal(1.0, GenerationBenchmarks.ComputeWer(hyps, refs), 6);
    }

    [Fact]
    public void Wer_AveragesOverPairs()
    {
        // First: 0/2 edits, Second: 2/2 edits → 2/4 = 0.5.
        var refs = new[] { "hi there", "good morning" };
        var hyps = new[] { "hi there", "bad evening" };
        Assert.Equal(0.5, GenerationBenchmarks.ComputeWer(hyps, refs), 6);
    }

    [Fact]
    public void Wer_LengthMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            GenerationBenchmarks.ComputeWer(new[] { "a" }, new[] { "a", "b" }));
    }

    [Fact]
    public void Wer_Insertion_CountsAsEdit()
    {
        // ref "a b" → hyp "a b c" : 1 insertion / 2 ref words = 0.5
        var refs = new[] { "a b" };
        var hyps = new[] { "a b c" };
        Assert.Equal(0.5, GenerationBenchmarks.ComputeWer(hyps, refs), 6);
    }

    [Fact]
    public void Wer_Deletion_CountsAsEdit()
    {
        // ref "a b c" → hyp "a c" : 1 deletion / 3 ref words = 1/3.
        var refs = new[] { "a b c" };
        var hyps = new[] { "a c" };
        Assert.Equal(1.0 / 3.0, GenerationBenchmarks.ComputeWer(hyps, refs), 6);
    }
}
