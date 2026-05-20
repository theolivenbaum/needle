using Needle.Cli;

namespace Needle.Tests.Cli;

public sealed class ArgParserTests
{
    [Fact]
    public void Parse_ValueArgs_Captured()
    {
        var p = ArgParserTestProxy.Parse(["--checkpoint", "model.ndlw", "--epochs", "3"]);
        Assert.Equal("model.ndlw", p.Get("checkpoint", "?"));
        Assert.Equal("3",          p.Get("epochs",     "?"));
    }

    [Fact]
    public void Parse_FlagArg_Recognised()
    {
        var p = ArgParserTestProxy.Parse(["--verbose", "--checkpoint", "x"]);
        Assert.True(p.GetFlag("verbose"));
        Assert.Equal("x", p.Get("checkpoint", "?"));
    }

    [Fact]
    public void Parse_MissingValue_DefaultReturned()
    {
        var p = ArgParserTestProxy.Parse(Array.Empty<string>());
        Assert.Equal("default", p.Get("anything", "default"));
    }

    [Fact]
    public void GetRequired_MissingKey_Throws()
    {
        var p = ArgParserTestProxy.Parse(Array.Empty<string>());
        Assert.Throws<InvalidOperationException>(() => p.GetRequired("needed"));
    }

    [Fact]
    public void Parse_PositionalArgsIgnored()
    {
        // Bare words (without leading --) are skipped; only --key value pairs
        // and --flag are recognised.
        var p = ArgParserTestProxy.Parse(["positional", "--checkpoint", "x"]);
        Assert.Equal("x", p.Get("checkpoint", "?"));
    }
}

/// <summary>
/// Tiny adapter so the tests can call the internal <c>ArgParser</c> type.
/// Lives in the same logical "package" as ArgParser via InternalsVisibleTo
/// in Needle.Cli.csproj — but to avoid adding that wiring, we re-implement
/// the small needed surface here using reflection.
/// </summary>
internal static class ArgParserTestProxy
{
    private static readonly Type ArgParserType = typeof(CliEntry).Assembly
        .GetType("Needle.Cli.ArgParser", throwOnError: true)!;

    public static Wrapped Parse(string[] args)
    {
        var method = ArgParserType.GetMethod("Parse")!;
        var inst   = method.Invoke(null, new object[] { args })!;
        return new Wrapped(inst);
    }

    internal sealed class Wrapped
    {
        private readonly object _inner;
        public Wrapped(object inner) { _inner = inner; }

        public string Get(string key, string fallback) =>
            (string)_inner.GetType().GetMethod("Get")!.Invoke(_inner, new object[] { key, fallback })!;

        public string GetRequired(string key)
        {
            try
            {
                return (string)_inner.GetType().GetMethod("GetRequired")!.Invoke(_inner, new object[] { key })!;
            }
            catch (System.Reflection.TargetInvocationException tie) when (tie.InnerException is InvalidOperationException ioe)
            {
                throw ioe;
            }
        }

        public bool GetFlag(string key) =>
            (bool)_inner.GetType().GetMethod("GetFlag")!.Invoke(_inner, new object[] { key })!;
    }
}
