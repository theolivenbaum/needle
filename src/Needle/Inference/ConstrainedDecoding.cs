using System.Text.Json;
using Needle.Tokenizer;

namespace Needle.Inference;

// ── Trie ────────────────────────────────────────────────────────────────────

/// <summary>Node in a character-level prefix trie.</summary>
public sealed class TrieNode
{
    public Dictionary<char, TrieNode> Children { get; } = new();
    public bool IsTerminal { get; set; }
}

/// <summary>
/// Character-level prefix tree used to enumerate valid tool names and
/// parameter keys during constrained decoding.
///
/// Port of the Python <c>Trie</c> class in constrained.py.
/// </summary>
public sealed class Trie
{
    private readonly TrieNode _root = new();
    private readonly List<string> _words = new();

    /// <summary>All words that have been inserted into the trie.</summary>
    public IReadOnlyList<string> Words => _words;

    /// <summary>Insert <paramref name="word"/> into the trie.</summary>
    public void Insert(string word)
    {
        if (string.IsNullOrEmpty(word))
            return;

        _words.Add(word);
        var node = _root;
        foreach (char ch in word)
        {
            if (!node.Children.TryGetValue(ch, out var child))
            {
                child = new TrieNode();
                node.Children[ch] = child;
            }
            node = child;
        }
        node.IsTerminal = true;
    }

    /// <summary>
    /// Walk the trie following <paramref name="prefix"/> and return the
    /// node reached, or <c>null</c> if the prefix is not present.
    /// </summary>
    public TrieNode? GetNode(string prefix)
    {
        var node = _root;
        foreach (char ch in prefix)
        {
            if (!node.Children.TryGetValue(ch, out var child))
                return null;
            node = child;
        }
        return node;
    }

    /// <summary>Returns the root node (empty prefix).</summary>
    internal TrieNode Root => _root;
}

// ── ToolConstraints ──────────────────────────────────────────────────────────

/// <summary>
/// Holds the name trie and a per-function parameter trie built from a
/// JSON tool-definition list.
///
/// Extracts property names from <c>parameters.properties</c> — not the
/// schema-level keys (<c>type</c>, <c>properties</c>, <c>required</c>).
///
/// Port of Python <c>ToolConstraints</c> in constrained.py.
/// </summary>
public sealed class ToolConstraints
{
    public Trie NameTrie { get; } = new();
    public Dictionary<string, Trie> ParamTries { get; } = new();

    /// <summary>
    /// Parse <paramref name="toolsJson"/> (a JSON array of tool definitions)
    /// and populate the name and parameter tries.
    /// </summary>
    public ToolConstraints(string toolsJson)
    {
        JsonElement root;
        try
        {
            root = JsonSerializer.Deserialize<JsonElement>(toolsJson);
        }
        catch
        {
            return; // Malformed JSON — leave tries empty.
        }

        if (root.ValueKind != JsonValueKind.Array)
            return;

        foreach (var tool in root.EnumerateArray())
        {
            if (tool.ValueKind != JsonValueKind.Object)
                continue;

            if (!tool.TryGetProperty("name", out var nameProp)
                || nameProp.ValueKind != JsonValueKind.String)
                continue;

            string name = nameProp.GetString() ?? string.Empty;
            if (string.IsNullOrEmpty(name))
                continue;

            NameTrie.Insert(name);

            var paramTrie = new Trie();
            if (tool.TryGetProperty("parameters", out var paramsProp)
                && paramsProp.ValueKind == JsonValueKind.Object
                && paramsProp.TryGetProperty("properties", out var propsProp)
                && propsProp.ValueKind == JsonValueKind.Object)
            {
                foreach (var prop in propsProp.EnumerateObject())
                    paramTrie.Insert(prop.Name);
            }
            ParamTries[name] = paramTrie;
        }
    }

    /// <summary>
    /// Return the parameter trie for <paramref name="functionName"/>, or
    /// <c>null</c> if no such function was found in the tool definitions.
    /// </summary>
    public Trie? GetParamTrie(string functionName)
        => ParamTries.TryGetValue(functionName, out var t) ? t : null;
}

// ── JsonState ────────────────────────────────────────────────────────────────

/// <summary>State of the JSON output state machine.</summary>
public enum JsonState
{
    /// <summary>Not currently inside a constrained span.</summary>
    Free,
    /// <summary>Inside a tool-name string (after <c>"name":"</c>).</summary>
    InName,
    /// <summary>Inside an argument-key string (after <c>{"</c> or <c>,"</c> inside arguments object).</summary>
    InArgKey,
}

// ── JsonStateMachine ─────────────────────────────────────────────────────────

/// <summary>
/// Character-level state machine that tracks the decoder's position within
/// Needle's compact JSON output format:
/// <code>[{"name":"TOOL","arguments":{"key":val,...}}]</code>
///
/// Constrained spans:
/// <list type="bullet">
///   <item>After <c>"name":"</c> → <see cref="JsonState.InName"/></item>
///   <item>After <c>"arguments":{"</c> or <c>,"</c> inside the arguments object
///         → <see cref="JsonState.InArgKey"/></item>
///   <item>Closing <c>"</c> in a constrained state → <see cref="JsonState.Free"/></item>
/// </list>
///
/// Port of Python <c>JsonStateMachine</c> in constrained.py.
/// </summary>
public sealed class JsonStateMachine
{
    // ── Public state ─────────────────────────────────────────────────────────

    /// <summary>Current constrained decoding state.</summary>
    public JsonState State { get; private set; } = JsonState.Free;

    /// <summary>
    /// Characters accumulated so far in the current constrained span
    /// (tool name or argument key prefix).
    /// </summary>
    public string ConstrainedBuf { get; private set; } = string.Empty;

    /// <summary>
    /// The tool name decoded in the most recent <see cref="JsonState.InName"/> span.
    /// </summary>
    public string CurrentFunction { get; private set; } = string.Empty;

    // ── Private tracking ─────────────────────────────────────────────────────

    private string _buffer = string.Empty;
    private bool _inArguments;
    private int  _argumentsDepth;
    private int  _nestingDepth;
    private bool _inString;
    private bool _prevCharEscape;

    // ── Public API ────────────────────────────────────────────────────────────

    /// <summary>
    /// Feed <paramref name="text"/> into the state machine, advancing the
    /// state as each character is processed.
    /// </summary>
    public void Feed(string text)
    {
        foreach (char ch in text)
            FeedChar(ch);
    }

    // ── Private logic ─────────────────────────────────────────────────────────

    private void FeedChar(char ch)
    {
        // In a constrained span: collect chars until the closing quote.
        if (State is JsonState.InName or JsonState.InArgKey)
        {
            if (ch == '"')
            {
                if (State == JsonState.InName)
                    CurrentFunction = ConstrainedBuf;
                ConstrainedBuf = string.Empty;
                State = JsonState.Free;
            }
            else
            {
                ConstrainedBuf += ch;
            }
            _buffer += ch;
            return;
        }

        _buffer += ch;

        // Inside a JSON string value: skip content, handle escapes, watch for close.
        if (_inString)
        {
            if (_prevCharEscape)
            {
                _prevCharEscape = false;
                return;
            }
            if (ch == '\\')
            {
                _prevCharEscape = true;
                return;
            }
            if (ch == '"')
                _inString = false;
            return;
        }

        // Track nesting depth.
        if (ch is '{' or '[')
        {
            _nestingDepth++;
        }
        else if (ch is '}' or ']')
        {
            _nestingDepth = System.Math.Max(0, _nestingDepth - 1);
            if (ch == '}' && _inArguments && _nestingDepth < _argumentsDepth)
                _inArguments = false;
            return;
        }

        // Check for constrained-span triggers.
        if (_buffer.EndsWith("\"name\":\"", StringComparison.Ordinal) && !_inArguments)
        {
            State = JsonState.InName;
            ConstrainedBuf = string.Empty;
            return;
        }

        if (_buffer.EndsWith("\"arguments\":{", StringComparison.Ordinal))
        {
            _inArguments = true;
            _argumentsDepth = _nestingDepth;
            return;
        }

        if (_inArguments
            && _nestingDepth == _argumentsDepth
            && AtArgKeyStart())
        {
            State = JsonState.InArgKey;
            ConstrainedBuf = string.Empty;
            return;
        }

        // Track JSON string values so their contents don't trigger false transitions.
        if (ch == '"' && IsValueQuote())
            _inString = true;
    }

    /// <summary>
    /// Returns <c>true</c> when the buffer ends with <c>{"</c> or <c>,"</c>,
    /// indicating that an argument-key string is opening.
    /// </summary>
    private bool AtArgKeyStart()
    {
        int len = _buffer.Length;
        if (len < 2) return false;
        char prev = _buffer[len - 2];
        char last = _buffer[len - 1];
        return last == '"' && (prev == '{' || prev == ',');
    }

    /// <summary>
    /// Returns <c>true</c> when the current <c>"</c> opens a string value
    /// (the preceding non-whitespace character is a colon).
    /// </summary>
    private bool IsValueQuote()
    {
        for (int j = _buffer.Length - 2; j >= 0; j--)
        {
            char c = _buffer[j];
            if (c is ' ' or '\t' or '\n' or '\r')
                continue;
            return c == ':';
        }
        return false;
    }
}

// ── TokenIndex ───────────────────────────────────────────────────────────────

/// <summary>
/// Maps the first character of each non-empty token string to the list of
/// token IDs that start with that character.  Used to restrict the set of
/// candidates that need trie-validation during constrained decoding.
///
/// Port of Python <c>TokenIndex</c> in constrained.py.
/// </summary>
public sealed class TokenIndex
{
    private readonly Dictionary<char, List<int>> _index = new();
    private readonly List<int> _allNonempty = new();

    public TokenIndex(IReadOnlyList<string> tokenStrings)
    {
        for (int id = 0; id < tokenStrings.Count; id++)
        {
            var s = tokenStrings[id];
            if (string.IsNullOrEmpty(s))
                continue;

            char first = s[0];
            if (!_index.TryGetValue(first, out var list))
            {
                list = new List<int>();
                _index[first] = list;
            }
            list.Add(id);
            _allNonempty.Add(id);
        }
    }

    /// <summary>Token IDs whose string representation starts with <paramref name="firstChar"/>.</summary>
    public IReadOnlyList<int> CandidatesFor(char firstChar)
        => _index.TryGetValue(firstChar, out var list)
            ? list
            : Array.Empty<int>();

    /// <summary>All token IDs with a non-empty string representation.</summary>
    public IReadOnlyList<int> AllNonempty => _allNonempty;
}

// ── ConstrainedDecoder ───────────────────────────────────────────────────────

/// <summary>
/// Batch-aware constrained decoder.  Maintains one <see cref="JsonStateMachine"/>
/// per batch element and applies trie-based logit masking during constrained spans.
///
/// Usage pattern per generation step:
/// <list type="number">
///   <item>Call <see cref="ConstrainLogits"/> to mask invalid tokens.</item>
///   <item>Select the best token (argmax or sample).</item>
///   <item>Call <see cref="Update"/> to advance the state machine.</item>
/// </list>
///
/// Port of Python <c>ConstrainedDecoder</c> in constrained.py.
/// </summary>
public sealed class ConstrainedDecoder
{
    private readonly IReadOnlyList<ToolConstraints> _toolConstraints;
    private readonly IReadOnlyList<string>          _tokenStrings;
    private readonly TokenIndex                     _tokenIndex;
    private readonly JsonStateMachine[]             _machines;

    public ConstrainedDecoder(
        IReadOnlyList<ToolConstraints> toolConstraints,
        IReadOnlyList<string>          tokenStrings,
        TokenIndex                     tokenIndex)
    {
        _toolConstraints = toolConstraints;
        _tokenStrings    = tokenStrings;
        _tokenIndex      = tokenIndex;
        _machines        = new JsonStateMachine[toolConstraints.Count];
        for (int i = 0; i < _machines.Length; i++)
            _machines[i] = new JsonStateMachine();
    }

    /// <summary>
    /// Returns <c>true</c> when batch element <paramref name="batchIdx"/> is
    /// currently inside a constrained span (tool name or argument key).
    /// </summary>
    public bool IsActive(int batchIdx)
        => _machines[batchIdx].State != JsonState.Free;

    /// <summary>
    /// Apply grammar constraints to <paramref name="logits"/> for batch element
    /// <paramref name="batchIdx"/>.  Tokens that cannot validly continue the
    /// current constrained span are set to <see cref="float.NegativeInfinity"/>.
    ///
    /// Returns the original array (possibly modified in-place) if not in a
    /// constrained state, or if no valid tokens are found (fallback).
    /// </summary>
    public float[] ConstrainLogits(float[] logits, int batchIdx)
    {
        var machine = _machines[batchIdx];
        var tc      = _toolConstraints[batchIdx];

        if (machine.State == JsonState.Free)
            return logits;

        Trie? trie;
        if (machine.State == JsonState.InName)
        {
            trie = tc.NameTrie;
        }
        else if (machine.State == JsonState.InArgKey)
        {
            trie = tc.GetParamTrie(machine.CurrentFunction);
            if (trie is null)
                return logits;
        }
        else
        {
            return logits;
        }

        var node = trie.GetNode(machine.ConstrainedBuf);
        if (node is null)
            return logits; // Off-trie — fall back to unconstrained.

        return ApplyConstraints(logits, node);
    }

    /// <summary>
    /// Advance the state machine for batch element <paramref name="batchIdx"/>
    /// with the token that was just selected.
    /// </summary>
    public void Update(int batchIdx, int tokenId)
    {
        string text = (uint)tokenId < (uint)_tokenStrings.Count
            ? _tokenStrings[tokenId]
            : string.Empty;
        _machines[batchIdx].Feed(text);
    }

    // ── Constraint application ────────────────────────────────────────────────

    /// <summary>
    /// Mask logits so only tokens that are valid continuations from
    /// <paramref name="trieNode"/> survive.
    ///
    /// Port of Python <c>apply_constraints</c> and <c>_check_token_valid</c>.
    /// </summary>
    private float[] ApplyConstraints(float[] logits, TrieNode trieNode)
    {
        int vocabSize = logits.Length;
        var mask = new bool[vocabSize];

        // Collect the first characters that are valid from this trie node.
        // If the node is terminal, the closing quote is also valid.
        var validFirstChars = new HashSet<char>(trieNode.Children.Keys);
        if (trieNode.IsTerminal)
            validFirstChars.Add('"');

        bool anyValid = false;
        foreach (char firstChar in validFirstChars)
        {
            foreach (int tid in _tokenIndex.CandidatesFor(firstChar))
            {
                if (mask[tid])
                    continue;
                if (CheckTokenValid(_tokenStrings[tid], trieNode))
                {
                    mask[tid] = true;
                    anyValid  = true;
                }
            }
        }

        if (!anyValid)
            return logits; // No valid tokens found — fall back to unconstrained.

        var result = (float[])logits.Clone();
        for (int i = 0; i < vocabSize; i++)
        {
            if (!mask[i])
                result[i] = float.NegativeInfinity;
        }
        return result;
    }

    /// <summary>
    /// Returns <c>true</c> when <paramref name="tokenText"/> is a valid
    /// continuation from <paramref name="trieNode"/>.
    ///
    /// If <paramref name="tokenText"/> contains a closing <c>"</c>, the node
    /// must be terminal at that point; characters after the <c>"</c> are not
    /// checked (they are structural JSON that the state machine handles).
    ///
    /// Port of Python <c>_check_token_valid</c>.
    /// </summary>
    private static bool CheckTokenValid(string tokenText, TrieNode trieNode)
    {
        var node = trieNode;
        foreach (char ch in tokenText)
        {
            if (ch == '"')
                return node.IsTerminal;
            if (!node.Children.TryGetValue(ch, out var next))
                return false;
            node = next;
        }
        return true;
    }
}

// ── BuildTokenStrings / BuildConstrainedDecoder helpers ──────────────────────

public static class ConstrainedDecodingFactory
{
    /// <summary>
    /// Build the token-string table used by <see cref="ConstrainedDecoder"/>
    /// and <see cref="TokenIndex"/>.
    ///
    /// Maps each vocabulary ID to the characters it contributes to the decoded
    /// output:
    /// <list type="bullet">
    ///   <item>Control and unknown tokens → empty string.</item>
    ///   <item>Byte-fallback tokens <c>&lt;0xNN&gt;</c> → the corresponding character.</item>
    ///   <item>All other tokens → raw piece with ▁ (U+2581) replaced by space.</item>
    /// </list>
    ///
    /// Port of Python <c>build_token_strings</c> in constrained.py.
    /// </summary>
    public static string[] BuildTokenStrings(NeedleTokenizer tokenizer)
    {
        int vocabSize = tokenizer.VocabSize;
        var strings = new string[vocabSize];

        for (int i = 0; i < vocabSize; i++)
        {
            if (tokenizer.IsByteToken(i))
            {
                // Piece looks like <0x41>; extract the hex byte.
                var piece = tokenizer.IdToPiece(i); // already has ▁→' ' but byte tokens don't have ▁
                // Re-read raw to avoid the ▁→space replacement that doesn't apply here.
                // IdToPiece replaces ▁ with ' ' but byte pieces don't contain ▁,
                // so either method is equivalent. We need the original <0xNN> form.
                // Use the vocabulary inverse via internal knowledge: IsByteToken checks
                // the raw _idToPiece form. We call a separate method to get raw piece.
                strings[i] = DecodeBytePiece(piece);
            }
            else
            {
                // IdToPiece already replaces ▁ → ' '.
                // For control/special tokens the piece will be something like
                // <pad>, <eos>, <tool_call>, etc. — we want to return empty
                // for control tokens so they don't appear in constrained matching.
                var piece = tokenizer.IdToPiece(i);
                if (IsControlPiece(piece))
                    strings[i] = string.Empty;
                else
                    strings[i] = piece;
            }
        }
        return strings;
    }

    /// <summary>
    /// Decode a byte-fallback piece (already ▁-substituted, so essentially
    /// the raw piece) to the single character it represents.
    /// Accepts either the raw form <c>&lt;0xNN&gt;</c> or, in the unlikely
    /// case the normaliser already resolved it, the character itself.
    /// </summary>
    private static string DecodeBytePiece(string piece)
    {
        // piece may be the raw form "<0xNN>" or already decoded.
        // We detect the raw form: starts with '<', contains '0x'.
        int hexStart = piece.IndexOf("0x", StringComparison.OrdinalIgnoreCase);
        if (hexStart >= 0 && hexStart + 4 < piece.Length)
        {
            var hexStr = piece.Substring(hexStart + 2, 2);
            if (int.TryParse(hexStr, System.Globalization.NumberStyles.HexNumber, null, out int byteVal)
                && byteVal <= 0xFF)
            {
                return ((char)byteVal).ToString();
            }
        }
        return piece.Length == 1 ? piece : string.Empty;
    }

    /// <summary>
    /// Returns <c>true</c> for special/control token pieces that should map
    /// to empty string in the constrained-decoding token table.
    /// These are pieces enclosed in angle brackets like <c>&lt;pad&gt;</c>,
    /// <c>&lt;eos&gt;</c>, etc., but NOT byte-fallback tokens (which are
    /// handled separately).
    /// </summary>
    private static bool IsControlPiece(string piece)
    {
        if (piece.Length < 2)
            return false;
        // Byte tokens are already handled; here we block other <…> tokens.
        if (piece[0] == '<' && piece[piece.Length - 1] == '>')
        {
            // Exclude byte-fallback tokens that slipped through (shouldn't happen).
            if (piece.StartsWith("<0x", StringComparison.OrdinalIgnoreCase))
                return false;
            return true;
        }
        return false;
    }

    /// <summary>
    /// Convenience factory: build a <see cref="ConstrainedDecoder"/> for a
    /// batch of examples.
    ///
    /// Port of Python <c>build_constrained_decoder</c> in constrained.py.
    /// </summary>
    /// <param name="toolsJsonList">One tool-JSON string per batch element.</param>
    /// <param name="tokenizer">Loaded <see cref="NeedleTokenizer"/>.</param>
    public static ConstrainedDecoder BuildConstrainedDecoder(
        IReadOnlyList<string> toolsJsonList,
        NeedleTokenizer tokenizer)
    {
        var tokenStrings = BuildTokenStrings(tokenizer);
        var tokenIndex   = new TokenIndex(tokenStrings);
        var tcList       = new List<ToolConstraints>(toolsJsonList.Count);
        foreach (var tj in toolsJsonList)
            tcList.Add(new ToolConstraints(tj));
        return new ConstrainedDecoder(tcList, tokenStrings, tokenIndex);
    }
}
