using Needle.Tokenizer;

namespace Needle.Tests.Tokenizer;

public sealed class TokenizerTests
{
    // ── ToSnakeCase ───────────────────────────────────────────────────────────

    [Theory]
    [InlineData("getWeather",        "get_weather")]
    [InlineData("GetWeather",        "get_weather")]
    [InlineData("sendEmail",         "send_email")]
    [InlineData("get.weather",       "get_weather")]
    [InlineData("get-weather",       "get_weather")]
    [InlineData("HTTPRequest",       "http_request")]
    [InlineData("XMLParser",         "xml_parser")]
    [InlineData("simple",            "simple")]
    [InlineData("alreadySnake_case", "already_snake_case")]
    [InlineData("camelCaseMethod",   "camel_case_method")]
    public void ToSnakeCase_ConvertsCorrectly(string input, string expected)
    {
        Assert.Equal(expected, NeedleTokenizer.ToSnakeCase(input));
    }

    [Fact]
    public void ToSnakeCase_EmptyString_ReturnsEmpty()
    {
        Assert.Equal(string.Empty, NeedleTokenizer.ToSnakeCase(string.Empty));
    }

    // ── Special token IDs ─────────────────────────────────────────────────────

    [Fact]
    public void SpecialTokenIds_AreCorrect()
    {
        Assert.Equal(0, NeedleTokenizer.PadId);
        Assert.Equal(1, NeedleTokenizer.EosId);
        Assert.Equal(2, NeedleTokenizer.BosId);
        Assert.Equal(3, NeedleTokenizer.UnkId);
        Assert.Equal(4, NeedleTokenizer.ToolCallId);
        Assert.Equal(5, NeedleTokenizer.ToolsId);
    }
}
