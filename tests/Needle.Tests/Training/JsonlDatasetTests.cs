using Needle.Training;

namespace Needle.Tests.Training;

public sealed class JsonlDatasetTests : IDisposable
{
    private readonly string _path = Path.GetTempFileName();

    public void Dispose()
    {
        if (File.Exists(_path)) File.Delete(_path);
    }

    private void Write(params string[] lines) => File.WriteAllLines(_path, lines);

    [Fact]
    public void ParsesTheUpstreamRowShape()
    {
        Write("""
            {"query": "dim the kitchen to 10", "tools": [{"name": "set_lights"}], "answers": [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 10}}], "reasoning": "'kitchen' -> room"}
            """);

        var example = Assert.Single(JsonlDataset.Load(_path));

        Assert.Equal("dim the kitchen to 10", example.Query);
        Assert.Equal("set_lights", example.PrimaryTool);
        Assert.Equal("'kitchen' -> room", example.Reasoning);
        Assert.False(example.IsRefusal);
        // Arrays are re-serialised compactly, as the prompt template expects.
        Assert.Equal("""[{"name":"set_lights"}]""", example.ToolsJson);
    }

    [Fact]
    public void AcceptsPreSerialisedStrings()
    {
        Write("""{"query": "q", "tools": "[{\"name\":\"t\"}]", "answers": "[]"}""");

        var example = Assert.Single(JsonlDataset.Load(_path));
        Assert.Equal("""[{"name":"t"}]""", example.ToolsJson);
        Assert.True(example.IsRefusal);
    }

    [Fact]
    public void OffTopicRowsAreKeptAsRefusals()
    {
        Write("""{"query": "what is the meaning of life", "tools": [], "answers": []}""");

        var example = Assert.Single(JsonlDataset.Load(_path));
        Assert.True(example.IsRefusal);
        Assert.Equal("", example.PrimaryTool);
    }

    [Fact]
    public void SkipsBlankAndMalformedLinesAndRowsWithoutAQuery()
    {
        Write(
            "",
            "   ",
            "{not json}",
            """{"tools": [], "answers": []}""",
            """{"query": "keep me", "answers": []}""");

        var example = Assert.Single(JsonlDataset.Load(_path));
        Assert.Equal("keep me", example.Query);
    }

    [Fact]
    public void FallsBackToTheLegacyFunctionCallsField()
    {
        Write("""{"query": "q", "function_calls": [{"name": "legacy", "arguments": {}}]}""");

        Assert.Equal("legacy", Assert.Single(JsonlDataset.Load(_path)).PrimaryTool);
    }

    [Fact]
    public void PerToolSplitHoldsOutEveryTool()
    {
        var examples = new List<FinetuneExample>();
        foreach (string tool in (string[])["alpha", "beta"])
            for (int i = 0; i < 100; i++)
                examples.Add(new FinetuneExample($"q{i}", "[]",
                                                 $$$"""[{"name":"{{{tool}}}","arguments":{}}]"""));

        var (train, validation, test) = JsonlDataset.PerToolSplit(examples);

        Assert.Equal(160, train.Count);
        Assert.Equal(20, validation.Count);
        Assert.Equal(20, test.Count);
        Assert.Equal(2, validation.Select(e => e.PrimaryTool).Distinct().Count());
        Assert.Equal(2, test.Select(e => e.PrimaryTool).Distinct().Count());
    }

    [Fact]
    public void PerToolSplitKeepsRareToolsInTraining()
    {
        // Three examples cannot spare ten for validation and ten for test.
        var examples = Enumerable.Range(0, 3)
            .Select(i => new FinetuneExample($"q{i}", "[]", """[{"name":"rare","arguments":{}}]"""))
            .ToList();

        var (train, validation, test) = JsonlDataset.PerToolSplit(examples);

        Assert.Equal(3, train.Count);
        Assert.Empty(validation);
        Assert.Empty(test);
    }

    [Fact]
    public void PerToolSplitIsDeterministicForASeed()
    {
        var examples = Enumerable.Range(0, 50)
            .Select(i => new FinetuneExample($"q{i}", "[]", """[{"name":"t","arguments":{}}]"""))
            .ToList();

        var first = JsonlDataset.PerToolSplit(examples, seed: 7).Validation.Select(e => e.Query);
        var second = JsonlDataset.PerToolSplit(examples, seed: 7).Validation.Select(e => e.Query);

        Assert.Equal(first, second);
    }
}
