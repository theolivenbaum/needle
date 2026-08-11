using System.Buffers.Binary;
using System.Globalization;
using System.Text;

namespace Needle.Tokenizer;

/// <summary>Piece kinds in the embedded SentencePiece dump.</summary>
public enum PieceType
{
    /// <summary>An ordinary sub-word piece.</summary>
    Normal = 0,

    /// <summary>The unknown-token piece.</summary>
    Unknown = 1,

    /// <summary>A control token (BOS, EOS, PAD) — never produced by encoding.</summary>
    Control = 2,

    /// <summary>A user-defined marker such as <c>&lt;|im_start|&gt;</c>, matched literally.</summary>
    UserDefined = 3,

    /// <summary>A single-byte fallback piece, spelled <c>&lt;0xNN&gt;</c>.</summary>
    Byte = 4,
}

/// <summary>
/// The SentencePiece BPE tokenizer carried inside a <c>.cact</c> blob.
///
/// Port of the <c>RefTokenizer</c> reference encoder/decoder in
/// <c>.reference/needle/model/export.py</c>, which is the normative description
/// of what the deployment blob's tokenizer section means.  Implementing it here
/// keeps the port self-contained: the blob is the only file needed to run the
/// model, tokenizer included.
/// </summary>
public sealed class CactTokenizer
{
    /// <summary>SentencePiece's whitespace marker, U+2581.</summary>
    public const string MetaSpace = "▁";

    private readonly string[] _pieces;
    private readonly float[] _scores;
    private readonly PieceType[] _types;
    private readonly Dictionary<string, int> _pieceToId;
    private readonly Dictionary<int, int> _byteToId;
    private readonly string[] _markers;      // user-defined, longest first
    private readonly bool _addDummyPrefix;
    private readonly bool _byteFallback;

    /// <summary>Padding token ID.</summary>
    public int PadId { get; }

    /// <summary>End-of-sequence token ID.</summary>
    public int EosId { get; }

    /// <summary>Beginning-of-sequence token ID.</summary>
    public int BosId { get; }

    /// <summary>Unknown token ID.</summary>
    public int UnkId { get; }

    /// <summary>Number of pieces in the vocabulary.</summary>
    public int VocabSize => _pieces.Length;

    private CactTokenizer(
        string[] pieces, float[] scores, PieceType[] types,
        int padId, int eosId, int bosId, int unkId,
        bool addDummyPrefix, bool byteFallback)
    {
        _pieces = pieces;
        _scores = scores;
        _types = types;
        PadId = padId;
        EosId = eosId;
        BosId = bosId;
        UnkId = unkId;
        _addDummyPrefix = addDummyPrefix;
        _byteFallback = byteFallback;

        _pieceToId = new Dictionary<string, int>(pieces.Length, StringComparer.Ordinal);
        for (int i = 0; i < pieces.Length; i++) _pieceToId.TryAdd(pieces[i], i);

        _byteToId = new Dictionary<int, int>(256);
        for (int i = 0; i < pieces.Length; i++)
        {
            if (types[i] != PieceType.Byte || pieces[i].Length < 5) continue;
            if (int.TryParse(pieces[i].AsSpan(3, 2), NumberStyles.HexNumber,
                             CultureInfo.InvariantCulture, out int value))
                _byteToId.TryAdd(value, i);
        }

        _markers = pieces.Where((_, i) => types[i] == PieceType.UserDefined)
                         .OrderByDescending(p => p.Length)
                         .ToArray();
    }

    /// <summary>Read the tokenizer out of a <c>.cact</c> blob's raw tokenizer section.</summary>
    public static CactTokenizer FromBlob(ReadOnlySpan<byte> blob)
    {
        // Header: u32 n_pieces, pad, eos, bos, unk; u8 add_dummy_prefix,
        //         u8 byte_fallback, u16 padding.
        int count = (int)BinaryPrimitives.ReadUInt32LittleEndian(blob);
        int pad = (int)BinaryPrimitives.ReadUInt32LittleEndian(blob[4..]);
        int eos = (int)BinaryPrimitives.ReadUInt32LittleEndian(blob[8..]);
        int bos = (int)BinaryPrimitives.ReadUInt32LittleEndian(blob[12..]);
        int unk = (int)BinaryPrimitives.ReadUInt32LittleEndian(blob[16..]);
        bool addDummy = blob[20] != 0;
        bool byteFallback = blob[21] != 0;

        int offset = 24;
        var pieces = new string[count];
        var scores = new float[count];
        var types = new PieceType[count];

        for (int i = 0; i < count; i++)
        {
            // Record: f32 score, u8 type, u16 surface length, then the bytes.
            scores[i] = BinaryPrimitives.ReadSingleLittleEndian(blob[offset..]);
            types[i] = (PieceType)blob[offset + 4];
            int length = BinaryPrimitives.ReadUInt16LittleEndian(blob[(offset + 5)..]);
            offset += 7;
            pieces[i] = Encoding.UTF8.GetString(blob.Slice(offset, length));
            offset += length;
        }

        return new CactTokenizer(pieces, scores, types, pad, eos, bos, unk, addDummy, byteFallback);
    }

    /// <summary>Read the tokenizer out of a <c>.cact</c> file.</summary>
    public static CactTokenizer FromCact(string path)
    {
        var file = Weights.CactFile.Open(path);
        int index = file.TokenizerIndex;
        if (index < 0)
            throw new InvalidDataException($"'{path}' carries no embedded tokenizer.");
        return FromBlob(file.Payload(index));
    }

    /// <summary>The surface form of a token.</summary>
    public string Piece(int id) => _pieces[id];

    /// <summary>The kind of a token.</summary>
    public PieceType Type(int id) => _types[id];

    /// <summary>The ID of a piece, or -1 when it is not in the vocabulary.</summary>
    public int Id(string piece) => _pieceToId.GetValueOrDefault(piece, -1);

    // ── Encoding ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Encode text to token IDs.  No BOS or EOS is added.
    ///
    /// User-defined markers are matched literally and split the input; each
    /// remaining run is escaped (spaces become U+2581, plus a leading one when
    /// the model uses a dummy prefix) and merged greedily by piece score.
    /// </summary>
    public List<int> Encode(string text)
    {
        var ids = new List<int>();
        if (string.IsNullOrEmpty(text)) return ids;

        string escaped = text.Replace(" ", MetaSpace, StringComparison.Ordinal);
        if (_addDummyPrefix) escaped = MetaSpace + escaped;

        var buffer = new StringBuilder();
        int i = 0;
        while (i < escaped.Length)
        {
            string? marker = null;
            foreach (string candidate in _markers)
            {
                if (string.CompareOrdinal(escaped, i, candidate, 0, candidate.Length) == 0)
                {
                    marker = candidate;
                    break;
                }
            }

            if (marker is not null)
            {
                EncodeSegment(buffer.ToString(), ids);
                buffer.Clear();
                ids.Add(_pieceToId[marker]);
                i += marker.Length;
            }
            else
            {
                buffer.Append(escaped[i]);
                i++;
            }
        }
        EncodeSegment(buffer.ToString(), ids);
        return ids;
    }

    /// <summary>Encode and prepend BOS — what every prompt needs.</summary>
    public List<int> EncodeWithBos(string text)
    {
        var ids = new List<int>(1) { BosId };
        ids.AddRange(Encode(text));
        return ids;
    }

    /// <summary>
    /// Greedy BPE over one marker-free run: repeatedly merge the adjacent pair
    /// whose combined piece scores highest, then map the survivors to IDs.
    /// </summary>
    private void EncodeSegment(string segment, List<int> ids)
    {
        if (segment.Length == 0) return;

        // Symbols are Unicode scalar values, not UTF-16 units — the reference
        // iterates Python characters, so an astral glyph counts as one symbol.
        var symbols = new List<string>(segment.Length);
        foreach (var rune in segment.EnumerateRunes()) symbols.Add(rune.ToString());

        while (symbols.Count > 1)
        {
            float bestScore = float.NegativeInfinity;
            int bestIndex = -1;
            for (int j = 0; j < symbols.Count - 1; j++)
            {
                if (_pieceToId.TryGetValue(symbols[j] + symbols[j + 1], out int id)
                    && (bestIndex < 0 || _scores[id] > bestScore))
                {
                    bestScore = _scores[id];
                    bestIndex = j;
                }
            }
            if (bestIndex < 0) break;

            symbols[bestIndex] += symbols[bestIndex + 1];
            symbols.RemoveAt(bestIndex + 1);
        }

        foreach (string symbol in symbols)
        {
            if (_pieceToId.TryGetValue(symbol, out int id))
            {
                ids.Add(id);
            }
            else if (_byteFallback)
            {
                foreach (byte b in Encoding.UTF8.GetBytes(symbol))
                    ids.Add(_byteToId.TryGetValue(b, out int byteId) ? byteId : UnkId);
            }
            else
            {
                ids.Add(UnkId);
            }
        }
    }

    // ── Decoding ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Decode token IDs back to text.  Byte-fallback tokens are reassembled into
    /// UTF-8 before decoding, and control tokens are dropped.
    /// </summary>
    public string Decode(IEnumerable<int> ids)
    {
        var bytes = new List<byte>(64);
        foreach (int id in ids)
        {
            if ((uint)id >= (uint)_pieces.Length) continue;
            switch (_types[id])
            {
                case PieceType.Byte:
                    if (int.TryParse(_pieces[id].AsSpan(3, 2), NumberStyles.HexNumber,
                                     CultureInfo.InvariantCulture, out int value))
                        bytes.Add((byte)value);
                    break;
                case PieceType.Control:
                case PieceType.Unknown:
                    break;
                default:
                    bytes.AddRange(Encoding.UTF8.GetBytes(_pieces[id]));
                    break;
            }
        }

        string text = Encoding.UTF8.GetString(bytes.ToArray())
                              .Replace(MetaSpace, " ", StringComparison.Ordinal);
        if (_addDummyPrefix && text.StartsWith(' ')) text = text[1..];
        return text;
    }
}
