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

        // Special tokens defined in the model at IDs 4 and 5.
        var specialTokens = new Dictionary<string, int>
        {
            { "<tool_call>", ToolCallId },
            { "<tools>",     ToolsId    },
        };

        using var stream = File.OpenRead(modelPath);
        _sp = LlamaTokenizer.Create(
            stream,
            addBeginOfSentence: false,
            addEndOfSentence:   false,
            specialTokens:      specialTokens);

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
    }

    // ── Encoding ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Encode <paramref name="text"/> into a list of token IDs.
    /// No BOS/EOS tokens are added (matches Python <c>sp.Encode(text, out_type=int)</c>).
    /// </summary>
    public IReadOnlyList<int> Encode(string text)
    {
        // addBeginningOfSentence = false, addEndOfSentence = false
        return _sp.EncodeToIds(
            text,
            addBeginningOfSentence: false,
            addEndOfSentence:       false,
            considerPreTokenization: true,
            considerNormalization:   true);
    }

    // ── Decoding ─────────────────────────────────────────────────────────────

    /// <summary>Decode a sequence of token IDs back to a string.</summary>
    public string Decode(IEnumerable<int> ids)
        => _sp.Decode(ids) ?? string.Empty;

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
