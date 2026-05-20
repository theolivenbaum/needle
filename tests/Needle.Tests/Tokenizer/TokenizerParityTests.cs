using Needle.Tokenizer;

namespace Needle.Tests.Tokenizer;

/// <summary>
/// End-to-end tokenizer-parity tests against the SentencePiece model
/// embedded in the Needle assembly (Cactus-Compute/needle, 8 192 pieces).
/// No network or filesystem path required — the model ships with the
/// build.
///
/// Locks in the cross-runtime parity behaviour established by the
/// <c>scripts/compare/</c> harness so the SentencePiece special-token
/// quirks the .NET wrapper had to work around can't silently regress.
/// </summary>
public sealed class TokenizerParityTests
{
    /// <summary>
    /// Reference token sequences produced by the Python
    /// <c>NeedleTokenizer.encode()</c> (i.e. SentencePiece's
    /// <c>sp.Encode(text, out_type=int)</c>) on the published 8192-piece
    /// model.  Locking these in catches drift in:
    ///   * the SP dummy-prefix ▁ around user-defined symbols,
    ///   * leading/trailing whitespace handling,
    ///   * UTF-8 / byte-fallback decomposition.
    /// </summary>
    public static IEnumerable<object[]> ParityCases() => new[]
    {
        new object[] { "Hello world", new[] { 7318, 363, 5338, 745 } },
        new object[] {
            "What's the weather in San Francisco?",
            new[] { 4279, 8066, 8046, 302, 1149, 362, 711, 327, 1295, 1075, 378, 275, 8047, 8105 },
        },
        new object[] {
            "[{\"name\":\"get_weather\",\"parameters\":{\"location\":\"string\"}}]",
            new[] { 356, 294, 264, 358, 8062, 1331, 265, 318, 282, 506, 264, 315, 503 },
        },
        new object[] {
            "<tool_call>[{\"name\":\"get_weather\",\"arguments\":{\"location\":\"SF\"}}]",
            new[] {
                8041, 4, 8071, 271, 294, 264, 358, 8062, 1331, 265, 393,
                282, 506, 264, 8074, 8095, 503,
            },
        },
    };

    [Theory]
    [MemberData(nameof(ParityCases))]
    public void Encode_MatchesPython(string text, int[] expected)
    {
        using var tok = NeedleTokenizer.LoadDefault();
        var actual = tok.Encode(text);
        Assert.Equal(expected, actual.ToArray());
    }

    [Fact]
    public void RoundTrip_ContainsSpecialTokenLiterals()
    {
        using var tok = NeedleTokenizer.LoadDefault();
        var ids = tok.Encode("<tool_call>{\"name\":\"x\"}");
        string decoded = tok.Decode(ids);

        Assert.StartsWith("<tool_call>", decoded);
        Assert.Contains("\"name\":\"x\"", decoded);
    }

    [Fact]
    public void LoadDefault_VocabSize_MatchesPython()
    {
        using var tok = NeedleTokenizer.LoadDefault();
        Assert.Equal(8192, tok.VocabSize);
    }
}
