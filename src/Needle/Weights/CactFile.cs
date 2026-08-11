using System.Buffers.Binary;
using Needle.Math;

namespace Needle.Weights;

/// <summary>Element type of a tensor record in a <c>.cact</c> blob.</summary>
public enum CactDtype
{
    /// <summary>IEEE half precision.</summary>
    Float16 = 1,

    /// <summary>IEEE single precision.</summary>
    Float32 = 2,

    /// <summary>Cactus-Quant packed indices plus per-group norms.</summary>
    Quantized = 3,

    /// <summary>Opaque bytes — the embedded tokenizer.</summary>
    Raw = 4,
}

/// <summary>One entry of the blob's nameless tensor directory.</summary>
/// <param name="Dtype">Element type.</param>
/// <param name="Shape">Logical shape (up to four dimensions).</param>
/// <param name="Offset">Byte offset of the payload.</param>
/// <param name="ByteLength">Payload length.</param>
/// <param name="GroupSize">Quantisation group size, 0 when not quantised.</param>
/// <param name="Bits">Quantisation width, 0 when not quantised.</param>
public sealed record CactTensor(
    CactDtype Dtype, int[] Shape, long Offset, long ByteLength, int GroupSize, int Bits);

/// <summary>
/// Reader for the <c>.cact</c> deployment blob — the single self-contained file
/// the released model ships as, holding the quantised weights and the tokenizer.
///
/// Port of the format documented and written by
/// <c>.reference/needle/model/export.py</c>.  Two things the format deliberately
/// omits: tensors have no names, and the architecture geometry is not stored.
/// Both are recovered here from the directory itself — the tensor order is
/// fixed, and the shapes pin down every dimension the model needs.
/// </summary>
public sealed class CactFile
{
    /// <summary>Magic tag at the head of the file.</summary>
    public const uint Tag = 0x05E12A82;

    // Header: u32 tag, num_tensors, codebook_len, kv_window, kv_bits.
    private const int HeaderSize = 20;

    // Record: u8 dtype, u8 ndim, u16 pad, u32 shape[4], u64 offset, u64 nbytes,
    // u32 group_size, u32 bits — packed, so 44 bytes.
    private const int RecordSize = 44;
    private const int ShapeOffset = 4;
    private const int DataOffset = 20;
    private const int LengthOffset = 28;
    private const int GroupOffset = 36;
    private const int BitsOffset = 40;

    private readonly byte[] _raw;

    /// <summary>The tensor directory, in file order.</summary>
    public IReadOnlyList<CactTensor> Tensors { get; }

    /// <summary>
    /// Concatenated Lloyd-Max codebooks from the header: cb2 (4 entries), cb3
    /// (8) and cb4 (16), all on the unit sphere and already divided by
    /// <c>sqrt(group)</c>.
    /// </summary>
    public float[] Codebooks { get; }

    /// <summary>Sliding-window width the model was trained with; 0 means "use the KV budget".</summary>
    public int KvWindow { get; }

    /// <summary>KV-cache width the model was post-trained for (8 = int8).</summary>
    public int KvBits { get; }

    private CactFile(byte[] raw, IReadOnlyList<CactTensor> tensors, float[] codebooks, int kvWindow, int kvBits)
    {
        _raw = raw;
        Tensors = tensors;
        Codebooks = codebooks;
        KvWindow = kvWindow;
        KvBits = kvBits;
    }

    /// <summary>Open a <c>.cact</c> file.</summary>
    public static CactFile Open(string path) => Parse(File.ReadAllBytes(path));

    /// <summary>Parse a <c>.cact</c> blob already in memory.</summary>
    public static CactFile Parse(byte[] raw)
    {
        if (raw.Length < HeaderSize)
            throw new InvalidDataException("File is too short to be a .cact blob.");

        var span = raw.AsSpan();
        uint tag = BinaryPrimitives.ReadUInt32LittleEndian(span);
        if (tag != Tag)
            throw new InvalidDataException($"Bad .cact tag 0x{tag:X8} (expected 0x{Tag:X8}).");

        int tensorCount = (int)BinaryPrimitives.ReadUInt32LittleEndian(span[4..]);
        int codebookLength = (int)BinaryPrimitives.ReadUInt32LittleEndian(span[8..]);
        int kvWindow = (int)BinaryPrimitives.ReadUInt32LittleEndian(span[12..]);
        int kvBits = (int)BinaryPrimitives.ReadUInt32LittleEndian(span[16..]);

        var codebooks = new float[codebookLength];
        int offset = HeaderSize;
        for (int i = 0; i < codebookLength; i++)
            codebooks[i] = BinaryPrimitives.ReadSingleLittleEndian(span[(offset + i * 4)..]);
        offset += codebookLength * 4;

        var tensors = new List<CactTensor>(tensorCount);
        for (int i = 0; i < tensorCount; i++)
        {
            var record = span.Slice(offset, RecordSize);
            offset += RecordSize;

            int ndim = record[1];
            var shape = new int[ndim];
            for (int d = 0; d < ndim; d++)
                shape[d] = (int)BinaryPrimitives.ReadUInt32LittleEndian(record[(ShapeOffset + d * 4)..]);

            tensors.Add(new CactTensor(
                (CactDtype)record[0], shape,
                (long)BinaryPrimitives.ReadUInt64LittleEndian(record[DataOffset..]),
                (long)BinaryPrimitives.ReadUInt64LittleEndian(record[LengthOffset..]),
                (int)BinaryPrimitives.ReadUInt32LittleEndian(record[GroupOffset..]),
                (int)BinaryPrimitives.ReadUInt32LittleEndian(record[BitsOffset..])));
        }

        return new CactFile(raw, tensors, codebooks, kvWindow, kvBits);
    }

    /// <summary>The codebook for a directory bit width.</summary>
    public float[] Codebook(int bits)
    {
        if (bits == CactusQuant.TernaryRecordBits) return CactusQuant.TernaryCodebook();

        int start = 0;
        foreach (int width in CactusQuant.CodebookBits)
        {
            int size = 1 << width;
            if (width == bits) return Codebooks[start..(start + size)];
            start += size;
        }
        throw new NotSupportedException($"No codebook for {bits}-bit tensors in this blob.");
    }

    /// <summary>Raw payload bytes of tensor <paramref name="index"/>.</summary>
    public ReadOnlySpan<byte> Payload(int index)
    {
        var tensor = Tensors[index];
        return _raw.AsSpan((int)tensor.Offset, (int)tensor.ByteLength);
    }

    /// <summary>Decode tensor <paramref name="index"/> to float32.</summary>
    /// <exception cref="InvalidOperationException">The tensor is a raw blob.</exception>
    public NdArray Read(int index)
    {
        var tensor = Tensors[index];
        var payload = Payload(index);

        switch (tensor.Dtype)
        {
            case CactDtype.Float16:
            {
                var result = new NdArray(tensor.Shape);
                var target = result.Span;
                for (int i = 0; i < target.Length; i++)
                    target[i] = (float)BinaryPrimitives.ReadHalfLittleEndian(payload.Slice(i * 2, 2));
                return result;
            }

            case CactDtype.Float32:
            {
                var result = new NdArray(tensor.Shape);
                var target = result.Span;
                for (int i = 0; i < target.Length; i++)
                    target[i] = BinaryPrimitives.ReadSingleLittleEndian(payload.Slice(i * 4, 4));
                return result;
            }

            case CactDtype.Quantized:
            {
                if (tensor.Shape.Length != 2)
                    throw new InvalidDataException("Quantised tensors must be two-dimensional.");
                int rows = tensor.Shape[0], width = tensor.Shape[1];
                int group = tensor.GroupSize;
                int paddedWidth = (width + group - 1) / group * group;
                int rowBytes = CactusQuant.PackedRowBytes(paddedWidth, tensor.Bits);
                int packedBytes = rows * rowBytes;

                var norms = new float[rows * (paddedWidth / group)];
                for (int i = 0; i < norms.Length; i++)
                    norms[i] = (float)BinaryPrimitives.ReadHalfLittleEndian(
                        payload.Slice(packedBytes + i * 2, 2));

                return CactusQuant.Dequantize(payload[..packedBytes], norms, rows, width,
                                              tensor.Bits, group, Codebook(tensor.Bits));
            }

            default:
                throw new InvalidOperationException(
                    $"Tensor {index} is a raw blob; use {nameof(Payload)} instead.");
        }
    }

    /// <summary>Index of the embedded tokenizer blob, or -1 when the file has none.</summary>
    public int TokenizerIndex
    {
        get
        {
            for (int i = Tensors.Count - 1; i >= 0; i--)
                if (Tensors[i].Dtype == CactDtype.Raw) return i;
            return -1;
        }
    }
}
