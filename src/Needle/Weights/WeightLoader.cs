using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using TorchSharp;
using static TorchSharp.torch;

namespace Needle.Weights;

// ---------------------------------------------------------------------------
// Format constants
// ---------------------------------------------------------------------------

/// <summary>
/// Supported tensor element types in the .ndlw binary format.
/// </summary>
public enum NdlwDtype : int
{
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Int32 = 3,
    Int64 = 4,
}

// ---------------------------------------------------------------------------
// ModelManifest
// ---------------------------------------------------------------------------

/// <summary>
/// Describes the layout of every tensor in a .ndlw weight file.
/// Built when saving a model and used when loading to seek directly to each
/// tensor's data without scanning the whole file.
/// </summary>
public sealed class ModelManifest
{
    /// <summary>
    /// Maps a parameter path (e.g. <c>"encoder.layer_0.self_attn.q_proj.weight"</c>)
    /// to its descriptor.
    /// </summary>
    public Dictionary<string, TensorDescriptor> Tensors { get; } = new();

    /// <summary>Adds or replaces a descriptor.</summary>
    public void Add(string name, TensorDescriptor descriptor) =>
        Tensors[name] = descriptor;

    /// <summary>Returns true when the manifest contains the named tensor.</summary>
    public bool Contains(string name) => Tensors.ContainsKey(name);
}

/// <summary>Describes a single tensor inside a weight file.</summary>
public sealed class TensorDescriptor
{
    /// <summary>Parameter path / name used as the dictionary key.</summary>
    public string Name { get; init; } = string.Empty;

    /// <summary>Shape of the tensor (row-major).</summary>
    public int[] Shape { get; init; } = [];

    /// <summary>Element type.</summary>
    public NdlwDtype Dtype { get; init; } = NdlwDtype.Float32;

    /// <summary>
    /// Byte offset of the tensor data inside the .ndlw file's data section
    /// (after the full header has been read).  Set during load; ignored on save.
    /// </summary>
    public long DataOffset { get; init; }

    /// <summary>Number of elements (product of shape dimensions).</summary>
    public int NumElements
    {
        get
        {
            if (Shape.Length == 0) return 1;
            int n = 1;
            foreach (int d in Shape) n *= d;
            return n;
        }
    }

    /// <summary>Size in bytes of one element for this dtype.</summary>
    public int ElementBytes => Dtype switch
    {
        NdlwDtype.Float32  => 4,
        NdlwDtype.Float16  => 2,
        NdlwDtype.BFloat16 => 2,
        NdlwDtype.Int32    => 4,
        NdlwDtype.Int64    => 8,
        _ => throw new NotSupportedException($"Unknown dtype {Dtype}")
    };
}

// ---------------------------------------------------------------------------
// WeightLoader
// ---------------------------------------------------------------------------

/// <summary>
/// Loads and saves Needle model weights from/to a binary <c>.ndlw</c> file
/// or a HuggingFace <c>.safetensors</c> file.
///
/// <para><b>.ndlw binary format</b></para>
/// <code>
/// [4 bytes]  magic  "NDLW"
/// [4 bytes]  version  (uint32 LE, currently 1)
/// [4 bytes]  numTensors  (uint32 LE)
/// For each tensor:
///   [N+1 bytes]  null-terminated UTF-8 name
///   [4 bytes]    ndim  (uint32 LE)
///   [ndim*4 bytes]  shape (uint32 LE each)
///   [4 bytes]    dtype  (uint32 LE, NdlwDtype enum)
///   [numElements * elementBytes]  raw tensor data (row-major)
/// </code>
///
/// <para><b>safetensors format</b></para>
/// <code>
/// [8 bytes]  headerSize  (uint64 LE)
/// [headerSize bytes]  UTF-8 JSON
///   { "__metadata__": {...},
///     "tensor_name": {"dtype": "F32", "shape": [...], "data_offsets": [start, end]},
///     ... }
/// [remaining bytes]  raw tensor data (offsets are relative to start of data section)
/// </code>
/// </summary>
public static class WeightLoader
{
    private static readonly byte[] Magic = "NDLW"u8.ToArray();
    private const uint CurrentVersion = 1u;

    // =========================================================================
    // .ndlw — Save
    // =========================================================================

    /// <summary>
    /// Saves all named tensors in <paramref name="tensors"/> to a .ndlw file at
    /// <paramref name="path"/>.
    /// </summary>
    /// <param name="tensors">
    ///   Dictionary mapping parameter paths to tensors.
    ///   All tensors are converted to float32 before writing unless they are
    ///   already in a supported dtype.
    /// </param>
    /// <param name="path">Destination file path (created or overwritten).</param>
    public static void Save(IReadOnlyDictionary<string, Tensor> tensors, string path)
    {
        using var fs  = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize: 1 << 20);
        using var bw  = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: true);

        // Header
        bw.Write(Magic);
        WriteUInt32LE(bw, CurrentVersion);
        WriteUInt32LE(bw, (uint)tensors.Count);

        foreach (var (name, tensor) in tensors)
        {
            // Resolve dtype
            var (ndlwDtype, torchDtype) = ResolveDtype(tensor.dtype);

            // Ensure CPU + correct dtype
            using var cpu = tensor.cpu().to(torchDtype).contiguous();

            // Name (null-terminated UTF-8)
            bw.Write(Encoding.UTF8.GetBytes(name));
            bw.Write((byte)0);

            // ndim + shape
            int ndim = cpu.shape.Length;
            WriteUInt32LE(bw, (uint)ndim);
            foreach (long dim in cpu.shape)
                WriteUInt32LE(bw, (uint)dim);

            // Dtype tag
            WriteUInt32LE(bw, (uint)ndlwDtype);

            // Raw data
            WriteTensorData(bw, cpu);
        }

        bw.Flush();
    }

    // =========================================================================
    // .ndlw — Load
    // =========================================================================

    /// <summary>
    /// Loads all tensors from a .ndlw file.
    /// </summary>
    /// <param name="path">Path to the .ndlw file.</param>
    /// <param name="device">Target device (default: CPU).</param>
    /// <returns>
    ///   A tuple of (manifest, tensors) where tensors maps parameter paths to
    ///   loaded TorchSharp tensors.
    /// </returns>
    public static (ModelManifest manifest, Dictionary<string, Tensor> tensors) Load(
        string path,
        Device? device = null)
    {
        device ??= torch.CPU;

        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1 << 20);
        using var br = new BinaryReader(fs, Encoding.UTF8, leaveOpen: true);

        // Verify magic
        var magic = br.ReadBytes(4);
        if (!magic.AsSpan().SequenceEqual(Magic))
            throw new InvalidDataException($"Not a valid .ndlw file: bad magic in '{path}'.");

        uint version = ReadUInt32LE(br);
        if (version != CurrentVersion)
            throw new NotSupportedException($".ndlw version {version} is not supported (expected {CurrentVersion}).");

        uint count = ReadUInt32LE(br);

        var manifest = new ModelManifest();
        var tensors  = new Dictionary<string, Tensor>((int)count);

        for (uint i = 0; i < count; i++)
        {
            // Name
            string name = ReadNullTerminatedString(br);

            // Shape
            uint ndim = ReadUInt32LE(br);
            var shape = new long[ndim];
            for (uint d = 0; d < ndim; d++)
                shape[d] = ReadUInt32LE(br);

            // Dtype
            var ndlwDtype = (NdlwDtype)ReadUInt32LE(br);
            var scalarType = NdlwToScalar(ndlwDtype);

            // Data
            long numElements = 1;
            foreach (long s in shape) numElements *= s;
            int elemBytes  = ElementBytes(ndlwDtype);
            int totalBytes = checked((int)(numElements * elemBytes));

            long dataOffset = fs.Position;
            var rawBytes = br.ReadBytes(totalBytes);
            if (rawBytes.Length != totalBytes)
                throw new EndOfStreamException($"Unexpected EOF reading tensor '{name}'.");

            // Build TorchSharp tensor from raw bytes
            var t = BytesToTensor(rawBytes, shape, scalarType, device);

            // Record in manifest
            var intShape = new int[ndim];
            for (int d = 0; d < ndim; d++) intShape[d] = (int)shape[d];
            manifest.Add(name, new TensorDescriptor
            {
                Name       = name,
                Shape      = intShape,
                Dtype      = ndlwDtype,
                DataOffset = dataOffset,
            });

            tensors[name] = t;
        }

        return (manifest, tensors);
    }

    // =========================================================================
    // safetensors — Load
    // =========================================================================

    /// <summary>
    /// Loads all tensors from a HuggingFace <c>.safetensors</c> file.
    /// </summary>
    /// <param name="path">Path to the .safetensors file.</param>
    /// <param name="device">Target device (default: CPU).</param>
    /// <returns>
    ///   A tuple of (manifest, tensors).  The manifest's tensor offsets are
    ///   relative to the start of the file's data section.
    /// </returns>
    public static (ModelManifest manifest, Dictionary<string, Tensor> tensors) LoadSafetensors(
        string path,
        Device? device = null)
    {
        device ??= torch.CPU;

        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1 << 20);
        using var br = new BinaryReader(fs, Encoding.UTF8, leaveOpen: true);

        // 8-byte header size (little-endian uint64)
        ulong headerSize = BinaryPrimitives.ReadUInt64LittleEndian(br.ReadBytes(8));

        // JSON header
        var headerBytes = br.ReadBytes(checked((int)headerSize));
        var headerJson  = Encoding.UTF8.GetString(headerBytes);

        // Start of tensor data section
        long dataStart = 8 + (long)headerSize;

        using var doc = JsonDocument.Parse(headerJson);
        var root = doc.RootElement;

        var manifest = new ModelManifest();
        var tensors  = new Dictionary<string, Tensor>();

        foreach (var prop in root.EnumerateObject())
        {
            // Skip the metadata entry
            if (prop.Name == "__metadata__") continue;

            var entry = prop.Value;

            // dtype string → ScalarType
            string dtypeStr   = entry.GetProperty("dtype").GetString()!;
            var (scalarType, ndlwDtype) = SafetensorsDtype(dtypeStr);

            // shape
            var shapeArr = entry.GetProperty("shape");
            int ndim     = shapeArr.GetArrayLength();
            var shape    = new long[ndim];
            var intShape = new int[ndim];
            for (int i = 0; i < ndim; i++)
            {
                shape[i]    = shapeArr[i].GetInt64();
                intShape[i] = (int)shape[i];
            }

            // data offsets [start, end] relative to data section
            var offsets   = entry.GetProperty("data_offsets");
            long relStart = offsets[0].GetInt64();
            long relEnd   = offsets[1].GetInt64();
            int  byteLen  = checked((int)(relEnd - relStart));

            // Seek and read
            fs.Seek(dataStart + relStart, SeekOrigin.Begin);
            var rawBytes = br.ReadBytes(byteLen);
            if (rawBytes.Length != byteLen)
                throw new EndOfStreamException($"Unexpected EOF reading safetensors tensor '{prop.Name}'.");

            var t = BytesToTensor(rawBytes, shape, scalarType, device);

            manifest.Add(prop.Name, new TensorDescriptor
            {
                Name       = prop.Name,
                Shape      = intShape,
                Dtype      = ndlwDtype,
                DataOffset = dataStart + relStart,
            });

            tensors[prop.Name] = t;
        }

        return (manifest, tensors);
    }

    // =========================================================================
    // Helpers — writing
    // =========================================================================

    private static void WriteUInt32LE(BinaryWriter bw, uint value)
    {
        Span<byte> buf = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(buf, value);
        bw.Write(buf);
    }

    private static unsafe void WriteTensorData(BinaryWriter bw, Tensor t)
    {
        // Obtain a raw byte view of the tensor data using TorchSharp's data<T> accessor.
        // We copy via intermediate byte array to avoid unsafe pointer usage.
        long numElements = 1;
        foreach (long s in t.shape) numElements *= s;

        switch (t.dtype)
        {
            case ScalarType.Float32:
            {
                var data = t.data<float>().ToArray();
                var bytes = new byte[data.Length * 4];
                Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
                bw.Write(bytes);
                break;
            }
            case ScalarType.Float16:
            case ScalarType.BFloat16:
            {
                // Write as-is (16-bit); TorchSharp stores them in native layout.
                var data  = t.data<short>().ToArray();
                var bytes = new byte[data.Length * 2];
                Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
                bw.Write(bytes);
                break;
            }
            case ScalarType.Int32:
            {
                var data  = t.data<int>().ToArray();
                var bytes = new byte[data.Length * 4];
                Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
                bw.Write(bytes);
                break;
            }
            case ScalarType.Int64:
            {
                var data  = t.data<long>().ToArray();
                var bytes = new byte[data.Length * 8];
                Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
                bw.Write(bytes);
                break;
            }
            default:
                throw new NotSupportedException($"Cannot serialise dtype {t.dtype}.");
        }
    }

    // =========================================================================
    // Helpers — reading
    // =========================================================================

    private static uint ReadUInt32LE(BinaryReader br)
    {
        var buf = br.ReadBytes(4);
        return BinaryPrimitives.ReadUInt32LittleEndian(buf);
    }

    private static string ReadNullTerminatedString(BinaryReader br)
    {
        var bytes = new List<byte>(64);
        byte b;
        while ((b = br.ReadByte()) != 0)
            bytes.Add(b);
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    private static Tensor BytesToTensor(byte[] raw, long[] shape, ScalarType dtype, Device device)
    {
        switch (dtype)
        {
            case ScalarType.Float32:
            {
                var data = new float[raw.Length / 4];
                Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
                return torch.tensor(data, dimensions: shape, dtype: dtype).to(device);
            }
            case ScalarType.Float16:
            case ScalarType.BFloat16:
            {
                // Load as int16 (same bit width), then view as the half type.
                var data = new short[raw.Length / 2];
                Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
                using var t16 = torch.tensor(data, dimensions: shape, dtype: ScalarType.Int16);
                // Reinterpret the raw bytes as the target half type via to().
                return t16.to(dtype).to(device);
            }
            case ScalarType.Int32:
            {
                var data = new int[raw.Length / 4];
                Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
                return torch.tensor(data, dimensions: shape, dtype: dtype).to(device);
            }
            case ScalarType.Int64:
            {
                var data = new long[raw.Length / 8];
                Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
                return torch.tensor(data, dimensions: shape, dtype: dtype).to(device);
            }
            default:
                throw new NotSupportedException($"Cannot deserialise dtype {dtype}.");
        }
    }

    // =========================================================================
    // Dtype mapping helpers
    // =========================================================================

    private static (NdlwDtype ndlw, ScalarType torch) ResolveDtype(ScalarType t) => t switch
    {
        ScalarType.Float32  => (NdlwDtype.Float32,  ScalarType.Float32),
        ScalarType.Float16  => (NdlwDtype.Float16,  ScalarType.Float16),
        ScalarType.BFloat16 => (NdlwDtype.BFloat16, ScalarType.BFloat16),
        ScalarType.Int32    => (NdlwDtype.Int32,    ScalarType.Int32),
        ScalarType.Int64    => (NdlwDtype.Int64,    ScalarType.Int64),
        // Fall back to float32 for anything else
        _ => (NdlwDtype.Float32, ScalarType.Float32),
    };

    private static ScalarType NdlwToScalar(NdlwDtype d) => d switch
    {
        NdlwDtype.Float32  => ScalarType.Float32,
        NdlwDtype.Float16  => ScalarType.Float16,
        NdlwDtype.BFloat16 => ScalarType.BFloat16,
        NdlwDtype.Int32    => ScalarType.Int32,
        NdlwDtype.Int64    => ScalarType.Int64,
        _ => throw new NotSupportedException($"Unknown NdlwDtype {d}")
    };

    private static (ScalarType scalarType, NdlwDtype ndlwDtype) SafetensorsDtype(string s) => s switch
    {
        "F32"  or "FLOAT"   => (ScalarType.Float32,  NdlwDtype.Float32),
        "F16"  or "HALF"    => (ScalarType.Float16,  NdlwDtype.Float16),
        "BF16" or "BFLOAT16"=> (ScalarType.BFloat16, NdlwDtype.BFloat16),
        "I32"  or "INT"     => (ScalarType.Int32,    NdlwDtype.Int32),
        "I64"  or "INT64"   => (ScalarType.Int64,    NdlwDtype.Int64),
        _ => throw new NotSupportedException($"Unsupported safetensors dtype string '{s}'.")
    };

    private static int ElementBytes(NdlwDtype d) => d switch
    {
        NdlwDtype.Float32  => 4,
        NdlwDtype.Float16  => 2,
        NdlwDtype.BFloat16 => 2,
        NdlwDtype.Int32    => 4,
        NdlwDtype.Int64    => 8,
        _ => throw new NotSupportedException($"Unknown NdlwDtype {d}")
    };
}
