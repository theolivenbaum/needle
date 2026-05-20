using Needle.Tokenizer;
using Needle.Training;

namespace Needle.Tests.Training;

/// <summary>
/// Trivial in-memory tokenizer for unit tests.  Tokenises by splitting on
/// whitespace, assigning each unique word an incrementing ID starting at
/// <c>FirstWordId</c>.  Special tokens occupy the standard reserved IDs
/// 0..5 to match <see cref="NeedleTokenizer"/>.
/// </summary>
internal sealed class FakeTokenizer : INeedleTokenizer
{
    private const int FirstWordId = 10;

    public int PadTokenId      => 0;
    public int EosTokenId      => 1;
    public int BosTokenId      => 2;
    public int ToolCallTokenId => 4;
    public int ToolsTokenId    => 5;
    public int VocabSize       => 256;

    private readonly Dictionary<string, int> _vocab = new();
    private readonly List<string>           _pieces = new();

    public FakeTokenizer()
    {
        // Reserve 0..9 for specials and padding
        for (int i = 0; i < FirstWordId; i++)
            _pieces.Add($"<{i}>");
    }

    public IReadOnlyList<int> Encode(string text)
    {
        var tokens = new List<int>();
        foreach (var word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            if (!_vocab.TryGetValue(word, out int id))
            {
                id = FirstWordId + _vocab.Count;
                _vocab[word] = id;
                _pieces.Add(word);
            }
            tokens.Add(id);
        }
        return tokens;
    }

    public string Decode(IEnumerable<int> ids) =>
        string.Join(' ', ids.Where(id => id >= FirstWordId && id < _pieces.Count).Select(id => _pieces[id]));

    public string IdToPiece(int id) =>
        id >= 0 && id < _pieces.Count ? _pieces[id] : string.Empty;
}

public sealed class JsonlDatasetTests
{
    [Fact]
    public void Load_ParsesValidJsonlLines()
    {
        string path = Path.GetTempFileName();
        File.WriteAllText(path,
            "{\"query\": \"Hi\", \"tools\": \"[]\", \"answers\": \"[]\"}\n" +
            "\n" + // blank line, skipped
            "{\"query\": \"Sup\", \"tools\": \"[]\", \"answers\": \"[]\"}\n");
        try
        {
            var examples = JsonlDataset.Load(path);
            Assert.Equal(2, examples.Count);
            Assert.Equal("Hi",  examples[0].Query);
            Assert.Equal("Sup", examples[1].Query);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_SkipsMalformedLines()
    {
        string path = Path.GetTempFileName();
        File.WriteAllText(path,
            "{\"query\": \"Hi\", \"tools\": \"[]\", \"answers\": \"[]\"}\n" +
            "not json\n" +
            "{\"query\": \"Yo\", \"tools\": \"[]\", \"answers\": \"[]\"}\n");
        try
        {
            var examples = JsonlDataset.Load(path);
            Assert.Equal(2, examples.Count);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_MissingFieldsFallToDefaults()
    {
        string path = Path.GetTempFileName();
        File.WriteAllText(path, "{\"query\": \"X\"}\n");
        try
        {
            var examples = JsonlDataset.Load(path);
            Assert.Single(examples);
            Assert.Equal("X",  examples[0].Query);
            Assert.Equal("[]", examples[0].Tools);
            Assert.Equal("[]", examples[0].Answers);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void PerToolSplit_DistributesAcrossTools()
    {
        var examples = new List<FinetuneExample>();
        for (int i = 0; i < 30; i++)
            examples.Add(new("q", "[]", """[{"name":"get_weather","arguments":{}}]"""));
        for (int i = 0; i < 30; i++)
            examples.Add(new("q", "[]", """[{"name":"send_email","arguments":{}}]"""));

        var (train, val, test) = JsonlDataset.PerToolSplit(examples, valPerTool: 5, testPerTool: 5);
        // 30 - 5 - 5 = 20 train per tool
        Assert.Equal(40, train.Count);
        Assert.Equal(10, val.Count);
        Assert.Equal(10, test.Count);
    }

    [Fact]
    public void PerToolSplit_TinyTool_ProportionalSplit()
    {
        // One tool with 3 examples — needs proportional split.
        var examples = new List<FinetuneExample>
        {
            new("q1", "[]", """[{"name":"rare_tool","arguments":{}}]"""),
            new("q2", "[]", """[{"name":"rare_tool","arguments":{}}]"""),
            new("q3", "[]", """[{"name":"rare_tool","arguments":{}}]"""),
        };
        var (train, val, test) = JsonlDataset.PerToolSplit(examples);
        Assert.Equal(3, train.Count + val.Count + test.Count);
        Assert.NotEmpty(test);
    }
}

public sealed class FinetuneExampleBuilderTests
{
    [Fact]
    public void Build_ProducesEncoderDecoderArrays()
    {
        var tk = new FakeTokenizer();
        var ex = new FinetuneExample(
            Query: "what is weather in SF",
            Tools: """[{"name":"get_weather","parameters":{"location":"string"}}]""",
            Answers: """[{"name":"get_weather","arguments":{"location":"SF"}}]""");

        var built = FinetuneExampleBuilder.Build(tk, ex, maxEncLen: 64, maxDecLen: 64);
        Assert.NotNull(built);

        // Encoder ends with <tools>-sep then tool tokens (since query is short enough).
        Assert.Contains(tk.ToolsTokenId, built!.EncTokens);
        // Decoder starts with [EOS, <tool_call>, ...] and ends with [..., EOS]
        Assert.Equal(tk.EosTokenId,      built.DecInTokens[0]);
        Assert.Equal(tk.ToolCallTokenId, built.DecInTokens[1]);
        Assert.Equal(tk.ToolCallTokenId, built.DecOutTokens[0]);
        Assert.Equal(tk.EosTokenId,      built.DecOutTokens[^1]);

        // Class labels are the same length as the decoder target.
        Assert.Equal(built.DecOutTokens.Length, built.ClassLabels.Length);
        // The <tool_call> and EOS positions should be base class (0).
        Assert.Equal(TokenClass.Base, built.ClassLabels[0]);
        Assert.Equal(TokenClass.Base, built.ClassLabels[^1]);
    }

    [Fact]
    public void Build_AnswerTooLong_ReturnsNull()
    {
        var tk = new FakeTokenizer();
        // 50 unique whitespace-separated tokens in the answer.  Decoder budget
        // is 32 so [<tool_call>, 50 tokens, EOS] = 52 > 32 → builder returns null.
        var longAnswer = "[" + string.Join(' ', Enumerable.Range(0, 50).Select(i => $"key{i}")) + "]";
        var ex = new FinetuneExample("query", "[]", longAnswer);
        var built = FinetuneExampleBuilder.Build(tk, ex, maxEncLen: 64, maxDecLen: 32);
        Assert.Null(built);
    }
}

public sealed class BatchBuilderTests
{
    [Fact]
    public void BuildBatch_ProducesCorrectShapes()
    {
        var tk = new FakeTokenizer();
        var ex = new FinetuneExample(
            "hello world",
            "[]",
            """[{"name":"a","arguments":{}}]""");
        var examples = new List<FinetuneExample> { ex, ex, ex };

        var batch = BatchBuilder.BuildBatch(examples, tk, batchSize: 3);
        Assert.NotNull(batch);

        Assert.Equal(3, batch!.SrcTokens.GetLength(0));
        Assert.Equal(3, batch.TgtInTokens.GetLength(0));
        Assert.Equal(3, batch.TgtOutTokens.GetLength(0));
        Assert.Equal(3, batch.LossMask.GetLength(0));
        Assert.Equal(3, batch.EncSegIds.GetLength(0));
        Assert.Equal(3, batch.DecSegIds.GetLength(0));

        // Encoder seg IDs: 1 for real tokens, 0 for padding (single-example packing).
        Assert.Equal(1, batch.EncSegIds[0, 0]);
        // Decoder seg IDs: 1 for the [<tool_call>, answer..., EOS] positions, 0 for pad
        Assert.Equal(1, batch.DecSegIds[0, 0]);
    }

    [Fact]
    public void Iterate_ChunksExamples()
    {
        var tk = new FakeTokenizer();
        var ex = new FinetuneExample("hi", "[]", """[{"name":"a","arguments":{}}]""");
        var examples = Enumerable.Range(0, 7).Select(_ => ex).ToList();
        var batches = BatchBuilder.Iterate(examples, tk, batchSize: 3).ToList();

        // 7 examples / 3 per batch = batches of sizes 3, 3, 1
        Assert.Equal(3, batches.Count);
        Assert.Equal(3, batches[0].SrcTokens.GetLength(0));
        Assert.Equal(3, batches[1].SrcTokens.GetLength(0));
        Assert.Equal(1, batches[2].SrcTokens.GetLength(0));
    }

    [Fact]
    public void PackBatch_PacksMultipleExamplesPerRow()
    {
        var tk = new FakeTokenizer();
        // 4 small examples that fit two-per-row at maxEncLen=32.
        var ex = new FinetuneExample("hi", "[]", """[{"name":"a","arguments":{}}]""");
        var examples = Enumerable.Range(0, 4).Select(_ => ex).ToList();

        var batch = BatchBuilder.PackBatch(examples, tk, maxEncLen: 32, maxDecLen: 32);
        Assert.NotNull(batch);

        // All rows are width-32.
        Assert.Equal(32, batch!.SrcTokens.GetLength(1));
        Assert.Equal(32, batch.TgtInTokens.GetLength(1));

        // Packing should produce fewer bins than examples (≥ 2 examples per bin).
        int nBins = batch.SrcTokens.GetLength(0);
        Assert.True(nBins < 4, $"Expected packing to reduce bins below 4, got {nBins}");

        // At least one row holds two distinct segment IDs (> 1) — proves the
        // pack actually concatenated two examples.
        bool foundMultiSegment = false;
        for (int r = 0; r < nBins; r++)
        {
            int maxSeg = 0;
            for (int j = 0; j < 32; j++) maxSeg = System.Math.Max(maxSeg, batch.EncSegIds[r, j]);
            if (maxSeg >= 2) { foundMultiSegment = true; break; }
        }
        Assert.True(foundMultiSegment, "expected at least one row with two segments");
    }

    [Fact]
    public void PackBatch_SegmentIdsContiguousFromOne()
    {
        var tk = new FakeTokenizer();
        var ex = new FinetuneExample("hi", "[]", """[{"name":"a","arguments":{}}]""");
        var batch = BatchBuilder.PackBatch(
            new List<FinetuneExample> { ex, ex }, tk, maxEncLen: 32, maxDecLen: 32);
        Assert.NotNull(batch);

        // First non-zero encoder seg ID in row 0 must be 1.
        for (int j = 0; j < 32; j++)
        {
            if (batch!.EncSegIds[0, j] != 0)
            {
                Assert.Equal(1, batch.EncSegIds[0, j]);
                break;
            }
        }
    }

    [Fact]
    public void IteratePacked_RespectsBinsPerBatch()
    {
        var tk = new FakeTokenizer();
        var ex = new FinetuneExample("hi", "[]", """[{"name":"a","arguments":{}}]""");
        var examples = Enumerable.Range(0, 6).Select(_ => ex).ToList();
        var batches = BatchBuilder.IteratePacked(examples, tk, binsPerBatch: 2,
                                                  maxEncLen: 32, maxDecLen: 32).ToList();

        foreach (var b in batches)
            Assert.True(b.SrcTokens.GetLength(0) <= 2);
    }
}
