using Needle.Inference;

namespace Needle.Tests.Inference;

public sealed class ToolNormalizerTests
{
    private const string SingleToolJson =
        """[{"name":"getWeather","parameters":{"properties":{"location":{"type":"string"}}}}]""";

    [Fact]
    public void NormalizeTools_ConvertsNameToSnakeCase()
    {
        var (json, nameMap) = ToolNormalizer.NormalizeTools(SingleToolJson);
        Assert.Contains("get_weather", json);
        Assert.DoesNotContain("getWeather", json);
        Assert.True(nameMap.ContainsKey("get_weather"));
        Assert.Equal("getWeather", nameMap["get_weather"]);
    }

    [Fact]
    public void NormalizeTools_InvalidJson_ReturnsOriginal()
    {
        var (json, nameMap) = ToolNormalizer.NormalizeTools("{not json}");
        Assert.Equal("{not json}", json);
        Assert.Empty(nameMap);
    }

    [Fact]
    public void NormalizeTools_EmptyArray_ReturnsEmpty()
    {
        var (json, nameMap) = ToolNormalizer.NormalizeTools("[]");
        Assert.Equal("[]", json);
        Assert.Empty(nameMap);
    }

    [Fact]
    public void RestoreToolNames_ReplacesSnakeCaseName()
    {
        var (_, nameMap) = ToolNormalizer.NormalizeTools(SingleToolJson);
        string pred = """[{"name":"get_weather","arguments":{"location":"SF"}}]""";
        string restored = ToolNormalizer.RestoreToolNames(pred, nameMap);
        Assert.Contains("getWeather", restored);
        Assert.DoesNotContain("get_weather", restored);
    }

    [Fact]
    public void RestoreToolNames_EmptyMap_ReturnsUnchanged()
    {
        string pred = """[{"name":"foo"}]""";
        string result = ToolNormalizer.RestoreToolNames(pred, new Dictionary<string, string>());
        Assert.Equal(pred, result);
    }

    [Fact]
    public void RestoreToolNames_InvalidJson_FallsBackToStringReplace()
    {
        var nameMap = new Dictionary<string, string> { ["foo_bar"] = "FooBar" };
        string pred = "name:foo_bar,args:{}";
        string result = ToolNormalizer.RestoreToolNames(pred, nameMap);
        Assert.Contains("FooBar", result);
    }

    [Fact]
    public void NormalizeTools_MultipleTool_AllConverted()
    {
        const string multiTools =
            """[{"name":"sendEmail","parameters":{}},{"name":"GetStockPrice","parameters":{}}]""";
        var (json, nameMap) = ToolNormalizer.NormalizeTools(multiTools);
        Assert.Contains("send_email", json);
        Assert.Contains("get_stock_price", json);
        Assert.Equal(2, nameMap.Count);
    }
}
