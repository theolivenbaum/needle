using Microsoft.ML.Tokenizers;
using System.Text.RegularExpressions;

namespace Needle.Tokenizer;

/// <summary>
/// Wrapper around Microsoft.ML.Tokenizers <see cref="SentencePieceTokenizer"/> (via
/// <see cref="LlamaTokenizer"/>) providing the interface expected by the Needle
/// inference pipeline.
///
/// The underlying SentencePiece model has 8 192 tokens with the following reserved IDs:
///   0 = PAD, 1 = EOS, 2 = BOS, 3 = UNK, 4 = &lt;tool_call&gt;, 5 = &lt;tools&gt;
///
/// Port of needle/dataset/tokenizer.py.
/// </summary>
public sealed class NeedleTokenizer : INeedleTokenizer, IDisposable
{
    // ── Special token IDs ────────────────────────────────────────────────────

    public const int PadId      = 0;
    public const int EosId      = 1;
    public const int BosId      = 2;
    public const int UnkId      = 3;
    public const int ToolCallId = 4;
    public const int ToolsId    = 5;

    // ── Internals ────────────────────────────────────────────────────────────

    private readonly SentencePieceTokenizer _sp;

    /// <summary>Reverse lookup: ID → raw piece string (e.g. "▁hello").</summary>
    private readonly string[] _idToPiece;

    /// <summary>
    /// Special-token literals in priority order (longer first to avoid
    /// partial-prefix matches when special tokens share a prefix).
    /// </summary>
    private readonly (string Literal, int Id)[] _specials =
    {
        ("<tool_call>", ToolCallId),
        ("<tools>",     ToolsId),
    };

    /// <summary>Set of special-token IDs for fast membership checks in Decode.</summary>
    private readonly HashSet<int> _specialIds;

    /// <summary>
    /// ID of the lone SentencePiece space marker "▁".  Cached at construction
    /// for use as the dummy-prefix token when manually segmenting input around
    /// special tokens.
    /// </summary>
    private readonly int _spaceId;

    // ── Properties ───────────────────────────────────────────────────────────

    public int PadTokenId      => PadId;
    public int EosTokenId      => EosId;
    public int BosTokenId      => BosId;
    public int ToolCallTokenId => ToolCallId;
    public int ToolsTokenId    => ToolsId;

    /// <summary>Total vocabulary size (number of tokens).</summary>
    public int VocabSize => _idToPiece.Length;

    // ── Construction ─────────────────────────────────────────────────────────

    /// <summary>
    /// Load the SentencePiece model from <paramref name="modelPath"/>.
    /// </summary>
    /// <param name="modelPath">Path to the .model file.</param>
    public NeedleTokenizer(string modelPath)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"SentencePiece model not found: {modelPath}", modelPath);

        // Note: we deliberately do NOT pass `specialTokens` to LlamaTokenizer.
        // Microsoft.ML.Tokenizers' special-token preprocessing splits the input
        // around each literal and encodes each segment independently — and each
        // independent segment gets a fresh SentencePiece "dummy prefix" ▁,
        // which does not match the Python reference (which adds the dummy
        // prefix once at the start of the input and treats special tokens as
        // user-defined symbols inside the SP encoder).  We do our own
        // segmentation in Encode() / Decode() below.
        using var stream = File.OpenRead(modelPath);
        _sp = LlamaTokenizer.Create(
            stream,
            addBeginOfSentence: false,
            addEndOfSentence:   false);

        // Build the reverse map: ID → piece string.
        // Vocabulary is IReadOnlyDictionary<string, int> (piece → id).
        int vocabSize = _sp.Vocabulary.Count;
        _idToPiece = new string[vocabSize];

        foreach (var (piece, id) in _sp.Vocabulary)
        {
            if ((uint)id < (uint)vocabSize)
                _idToPiece[id] = piece;
        }

        // Fill any gaps with empty string (unknown/control tokens may be absent).
        for (int i = 0; i < vocabSize; i++)
            _idToPiece[i] ??= string.Empty;

        _specialIds = new HashSet<int>(_specials.Select(s => s.Id));

        // Look up the lone "▁" token id (used to represent the dummy-prefix /
        // word-boundary marker when emitted as its own token).
        _spaceId = _sp.Vocabulary.TryGetValue("▁", out int spaceId) ? spaceId : 8041;
    }

    // ── Encoding ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Encode <paramref name="text"/> into a list of token IDs.
    /// No BOS/EOS tokens are added.  Matches Python
    /// <c>sp.Encode(text, out_type=int)</c> bit-for-bit including the
    /// SentencePiece "dummy prefix" ▁ semantics around user-defined symbols.
    /// </summary>
    public IReadOnlyList<int> Encode(string text)
    {
        if (string.IsNullOrEmpty(text))
            return Array.Empty<int>();

        var spans = FindSpecialTokenSpans(text);
        if (spans.Count == 0)
            return EncodeFirstSegment(text);

        // Manual segmentation around special tokens.  SentencePiece's
        // "dummy prefix" ▁ is added once at the very start of the input; text
        // immediately following a user-defined symbol does NOT receive a
        // fresh dummy prefix.  The three helpers below reproduce that
        // behaviour by:
        //   * Wrapping inputs with a fixed sentinel prefix `"ab\n"` (and a
        //     trailing `"\n"` where we need to preserve trailing whitespace).
        //     The prefix always encodes to exactly two tokens [▁ab, <0x0A>],
        //     so stripping the first two tokens reliably yields the
        //     no-dummy-prefix encoding of whatever followed.
        //   * For the first segment, we add a single space after the sentinel
        //     prefix to materialise the dummy prefix as a ▁ on the segment's
        //     first piece.
        // This both fixes a Microsoft.ML.Tokenizers normalisation quirk that
        // returns [] for several short / whitespace-only inputs and matches
        // the Python reference for every edge case (empty, leading/trailing
        // whitespace, consecutive specials, UTF-8, byte fallback).
        var result = new List<int>();
        int cursor = 0;
        for (int i = 0; i < spans.Count; i++)
        {
            var (start, end, sid) = spans[i];
            string seg = text.Substring(cursor, start - cursor);
            if (i == 0)
                result.AddRange(EncodeFirstSegment(seg));
            else if (seg.Length > 0)
                result.AddRange(EncodeMiddleSegment(seg));
            result.Add(sid);
            cursor = end;
        }

        string tail = text.Substring(cursor);
        if (tail.Length > 0)
            result.AddRange(EncodeTailSegment(tail));

        return result;
    }

    /// <summary>
    /// Raw SP-encode without any manual segmentation.  Used only inside the
    /// sentinel-wrapped helpers; do not call directly with short inputs
    /// because the underlying library normalises some of them to empty.
    /// </summary>
    private IReadOnlyList<int> EncodeRaw(string text) =>
        _sp.EncodeToIds(
            text,
            addBeginningOfSentence: false,
            addEndOfSentence:       false,
            considerPreTokenization: true,
            considerNormalization:   true);

    // The sentinel prefix `"ab\n"` always encodes to exactly two tokens
    // [▁ab, <0x0A>]; the trailing newline serves as a "word boundary" that
    // prevents the next character from merging with `c`.
    private const string LeadSentinel       = "ab\n";
    private const string LeadSentinelSpace  = "ab\n ";
    private const string TailSentinel       = "\n";
    private const int    PrefixTokenCount   = 2;

    private List<int> EncodeFirstSegment(string seg)
    {
        // Includes the dummy prefix ▁ (provided by the single space after the
        // sentinel prefix).  Trailing whitespace in `seg` survives because
        // `seg + "\n"` is followed by a non-whitespace boundary.
        var ids = EncodeRaw(LeadSentinelSpace + seg + TailSentinel);
        return SliceCopy(ids, PrefixTokenCount, ids.Count - 1);
    }

    private List<int> EncodeMiddleSegment(string seg)
    {
        // No dummy prefix.  Trailing whitespace preserved.
        var ids = EncodeRaw(LeadSentinel + seg + TailSentinel);
        return SliceCopy(ids, PrefixTokenCount, ids.Count - 1);
    }

    private List<int> EncodeTailSegment(string seg)
    {
        // No dummy prefix.  Trailing whitespace at the very end of the input
        // is normalised away by SP — matches Python.
        var ids = EncodeRaw(LeadSentinel + seg);
        return SliceCopy(ids, PrefixTokenCount, ids.Count);
    }

    private static List<int> SliceCopy(IReadOnlyList<int> src, int start, int end)
    {
        if (end <= start) return new List<int>();
        var dst = new List<int>(end - start);
        for (int i = start; i < end; i++) dst.Add(src[i]);
        return dst;
    }

    /// <summary>
    /// Locate every non-overlapping occurrence of any special-token literal in
    /// <paramref name="text"/>, in left-to-right order.  Longer literals win
    /// at the same start position so prefixes like <c>&lt;tools&gt;</c> never
    /// shadow longer hypothetical specials in the future.
    /// </summary>
    private List<(int Start, int End, int Id)> FindSpecialTokenSpans(string text)
    {
        var result = new List<(int, int, int)>();
        int pos = 0;
        while (pos < text.Length)
        {
            int bestStart = -1, bestLen = 0, bestId = -1;
            foreach (var (lit, id) in _specials)
            {
                int idx = text.IndexOf(lit, pos, StringComparison.Ordinal);
                if (idx < 0) continue;
                bool better =
                    bestStart < 0 ||
                    idx < bestStart ||
                    (idx == bestStart && lit.Length > bestLen);
                if (better)
                {
                    bestStart = idx; bestLen = lit.Length; bestId = id;
                }
            }
            if (bestStart < 0) break;
            result.Add((bestStart, bestStart + bestLen, bestId));
            pos = bestStart + bestLen;
        }
        return result;
    }

    // ── Decoding ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Decode a sequence of token IDs back to a string.  Matches Python
    /// <c>sp.Decode(ids)</c> including:
    /// <list type="bullet">
    ///   <item>Special token IDs render as their literal strings.</item>
    ///   <item>The SentencePiece ▁ marker maps to a space, except at the very
    ///         start of the result where it is stripped as the dummy prefix.</item>
    ///   <item>Byte-fallback tokens (<c>&lt;0xNN&gt;</c>) accumulate into
    ///         UTF-8 byte sequences and flush as decoded characters.</item>
    /// </list>
    /// </summary>
    public string Decode(IEnumerable<int> ids)
    {
        var sb       = new System.Text.StringBuilder();
        var byteBuf  = new List<byte>();

        void FlushBytes()
        {
            if (byteBuf.Count > 0)
            {
                sb.Append(System.Text.Encoding.UTF8.GetString(byteBuf.ToArray()));
                byteBuf.Clear();
            }
        }

        foreach (var id in ids)
        {
            if (IsByteToken(id))
            {
                string piece = _idToPiece[id];
                int hi = HexValue(piece[3]);
                int lo = HexValue(piece[4]);
                byteBuf.Add((byte)((hi << 4) | lo));
                continue;
            }
            FlushBytes();

            if ((uint)id >= (uint)_idToPiece.Length) continue;

            string p = _idToPiece[id];
            // Specials and regular pieces both go here.  ▁ becomes a space;
            // we'll strip a single dummy prefix at the very end.
            if (p.Length == 0) continue;
            sb.Append(p.Replace('▁', ' '));
        }
        FlushBytes();

        // Strip leading whitespace (the SP dummy prefix).  Multiple leading
        // spaces can occur if a sequence starts with several ▁s; SP collapses
        // them all to nothing at the start.
        string result = sb.ToString();
        int firstNonSpace = 0;
        while (firstNonSpace < result.Length && result[firstNonSpace] == ' ')
            firstNonSpace++;
        return firstNonSpace == 0 ? result : result.Substring(firstNonSpace);
    }

    private static int HexValue(char c) => c switch
    {
        >= '0' and <= '9' => c - '0',
        >= 'a' and <= 'f' => c - 'a' + 10,
        >= 'A' and <= 'F' => c - 'A' + 10,
        _ => 0,
    };

    /// <summary>Decode multiple ID sequences, returning one string per sequence.</summary>
    public IReadOnlyList<string> DecodeBatch(IEnumerable<IEnumerable<int>> sequences)
    {
        var results = new List<string>();
        foreach (var seq in sequences)
            results.Add(Decode(seq));
        return results;
    }

    // ── Vocabulary helpers ───────────────────────────────────────────────────

    /// <summary>
    /// Return the raw piece string for a vocabulary ID, with the SentencePiece
    /// word-boundary marker ▁ (U+2581) replaced by a regular space.
    /// </summary>
    public string IdToPiece(int id)
    {
        if ((uint)id >= (uint)_idToPiece.Length)
            return string.Empty;
        return _idToPiece[id].Replace('▁', ' ');
    }

    /// <summary>
    /// Return <c>true</c> if the token at <paramref name="id"/> is a
    /// SentencePiece byte-fallback token (raw piece matches <c>&lt;0xNN&gt;</c>).
    /// </summary>
    public bool IsByteToken(int id)
    {
        if ((uint)id >= (uint)_idToPiece.Length)
            return false;
        var piece = _idToPiece[id];
        // Byte fallback tokens look like <0x41>, <0xFE>, etc.
        return piece.Length == 6
            && piece[0] == '<'
            && piece[1] == '0'
            && piece[2] == 'x'
            && piece[5] == '>'
            && IsHexChar(piece[3])
            && IsHexChar(piece[4]);
    }

    private static bool IsHexChar(char c) =>
        c is (>= '0' and <= '9') or (>= 'a' and <= 'f') or (>= 'A' and <= 'F');

    // ── Snake-case conversion ─────────────────────────────────────────────────

    /// <summary>
    /// Convert a camelCase, PascalCase, or dot/dash/space-separated identifier
    /// to snake_case.
    ///
    /// Port of Python <c>to_snake_case()</c> in tokenizer.py:
    /// <code>
    /// s = re.sub(r'[^a-zA-Z0-9_]+', '_', name)
    /// s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    /// s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    /// s = re.sub(r'_+', '_', s)
    /// return s.lower().strip('_')
    /// </code>
    /// </summary>
    public static string ToSnakeCase(string name)
    {
        if (string.IsNullOrEmpty(name))
            return name;

        // 1. Replace any non-alphanumeric/underscore runs with a single underscore.
        string s = Regex.Replace(name, @"[^a-zA-Z0-9_]+", "_");

        // 2. Insert underscore before uppercase letters that follow a lowercase letter or digit.
        s = Regex.Replace(s, @"([a-z0-9])([A-Z])", "$1_$2");

        // 3. Insert underscore between consecutive uppercase letters and an uppercase+lowercase pair.
        s = Regex.Replace(s, @"([A-Z]+)([A-Z][a-z])", "$1_$2");

        // 4. Collapse multiple underscores and trim leading/trailing underscores.
        s = Regex.Replace(s, @"_+", "_");

        return s.ToLowerInvariant().Trim('_');
    }

    // ── Factory ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Load a <see cref="NeedleTokenizer"/> from <paramref name="modelPath"/>.
    /// Throws <see cref="FileNotFoundException"/> if the model file does not exist.
    /// </summary>
    public static NeedleTokenizer Load(string modelPath)
        => new NeedleTokenizer(modelPath);

    // ── IDisposable ──────────────────────────────────────────────────────────

    public void Dispose()
    {
        // SentencePieceTokenizer does not implement IDisposable in this version
        // of the library; nothing to release here.
    }
}
