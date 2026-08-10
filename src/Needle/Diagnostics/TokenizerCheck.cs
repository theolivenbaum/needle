using System.Text.Json;
using Needle.Tokenizer;

namespace Needle.Diagnostics;

/// <summary>
/// Checks <see cref="CactTokenizer"/> against the reference encoder/decoder.
///
/// Tokenization is the one place where "close enough" is worthless: a single
/// different merge shifts every downstream token, so this compares the exact ID
/// sequences the reference produces for a corpus that exercises chat markers,
/// the dummy prefix, byte fallback and long merge chains.
/// </summary>
public static class TokenizerCheck
{
    /// <summary>Outcome for one text.</summary>
    /// <param name="Text">The input.</param>
    /// <param name="Expected">Reference token IDs.</param>
    /// <param name="Actual">This implementation's token IDs.</param>
    /// <param name="ExpectedDecoded">Reference round-trip text.</param>
    /// <param name="ActualDecoded">This implementation's round-trip text.</param>
    public sealed record CaseResult(
        string Text, int[] Expected, int[] Actual, string ExpectedDecoded, string ActualDecoded)
    {
        /// <summary>Do the encodings agree exactly?</summary>
        public bool EncodeMatches => Expected.AsSpan().SequenceEqual(Actual);

        /// <summary>Do the decodings agree exactly?</summary>
        public bool DecodeMatches => string.Equals(ExpectedDecoded, ActualDecoded, StringComparison.Ordinal);

        /// <summary>Index of the first differing token, or -1.</summary>
        public int FirstDivergence
        {
            get
            {
                int n = System.Math.Min(Expected.Length, Actual.Length);
                for (int i = 0; i < n; i++)
                    if (Expected[i] != Actual[i]) return i;
                return Expected.Length == Actual.Length ? -1 : n;
            }
        }
    }

    /// <summary>Full report: piece-table agreement plus every encode/decode case.</summary>
    /// <param name="PieceMismatches">Vocabulary entries whose surface or type differ.</param>
    /// <param name="Cases">Per-text results.</param>
    public sealed record Report(IReadOnlyList<string> PieceMismatches, IReadOnlyList<CaseResult> Cases)
    {
        /// <summary>True when the piece table and every case agree.</summary>
        public bool Passed =>
            PieceMismatches.Count == 0 && Cases.All(c => c.EncodeMatches && c.DecodeMatches);
    }

    /// <summary>
    /// Compare <paramref name="tokenizer"/> against <c>tokenizer.json</c> as
    /// written by <c>scripts/parity/dump_cact.py</c>.
    /// </summary>
    public static Report Run(CactTokenizer tokenizer, string tokenizerJsonPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(tokenizerJsonPath));
        var root = doc.RootElement;

        var mismatches = new List<string>();
        var pieces = root.GetProperty("pieces");
        if (pieces.GetArrayLength() != tokenizer.VocabSize)
        {
            mismatches.Add($"vocabulary size {tokenizer.VocabSize}, reference has {pieces.GetArrayLength()}");
        }
        else
        {
            foreach (var entry in pieces.EnumerateArray())
            {
                int id = entry.GetProperty("id").GetInt32();
                string piece = entry.GetProperty("piece").GetString() ?? "";
                int type = entry.GetProperty("type").GetInt32();

                if (!string.Equals(tokenizer.Piece(id), piece, StringComparison.Ordinal))
                    mismatches.Add($"piece {id}: '{tokenizer.Piece(id)}' vs reference '{piece}'");
                else if ((int)tokenizer.Type(id) != type)
                    mismatches.Add($"piece {id} ('{piece}'): type {(int)tokenizer.Type(id)} vs reference {type}");

                if (mismatches.Count >= 16) break;
            }
        }

        var cases = new List<CaseResult>();
        foreach (var entry in root.GetProperty("cases").EnumerateArray())
        {
            string text = entry.GetProperty("text").GetString() ?? "";
            var expected = entry.GetProperty("ids").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            string expectedDecoded = entry.GetProperty("decoded").GetString() ?? "";

            var actual = tokenizer.Encode(text).ToArray();
            cases.Add(new CaseResult(text, expected, actual, expectedDecoded, tokenizer.Decode(actual)));
        }

        return new Report(mismatches, cases);
    }
}
