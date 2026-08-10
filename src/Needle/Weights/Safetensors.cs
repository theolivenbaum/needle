using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using Needle.Math;

namespace Needle.Weights;

/// <summary>
/// Reader and writer for the <c>safetensors</c> container.
///
/// Used as the interchange format between the Python reference and this port:
/// the checkpoint (a pickle of Flax arrays) and every traced intermediate are
/// converted to safetensors, which is a JSON header over a flat data block and
/// needs no dependencies to parse.
///
/// <code>
/// [8 bytes]            header length, uint64 little-endian
/// [headerLength bytes] UTF-8 JSON: name -> {dtype, shape, data_offsets}
/// [remainder]          tensor data; offsets are relative to the end of the header
/// </code>
///
/// Tensors are converted to float32 on read, since that is what the model
/// computes in.
/// </summary>
public static class Safetensors
{
    /// <summary>Reads every tensor in <paramref name="path"/> as float32.</summary>
    public static Dictionary<string, NdArray> Load(string path)
    {
        using var stream = File.OpenRead(path);
        return Load(stream);
    }

    /// <summary>Reads every tensor from <paramref name="stream"/> as float32.</summary>
    public static Dictionary<string, NdArray> Load(Stream stream)
    {
        Span<byte> lengthBytes = stackalloc byte[8];
        stream.ReadExactly(lengthBytes);
        long headerLength = (long)BinaryPrimitives.ReadUInt64LittleEndian(lengthBytes);
        if (headerLength <= 0 || headerLength > 256L * 1024 * 1024)
            throw new InvalidDataException($"Implausible safetensors header length {headerLength}.");

        var headerBytes = new byte[headerLength];
        stream.ReadExactly(headerBytes);

        using var doc = JsonDocument.Parse(Encoding.UTF8.GetString(headerBytes));
        var entries = new List<(string Name, string Dtype, int[] Shape, long Start, long End)>();
        foreach (var property in doc.RootElement.EnumerateObject())
        {
            if (property.Name == "__metadata__") continue;
            var value = property.Value;

            var shapeElement = value.GetProperty("shape");
            var shape = new int[shapeElement.GetArrayLength()];
            for (int i = 0; i < shape.Length; i++) shape[i] = shapeElement[i].GetInt32();

            var offsets = value.GetProperty("data_offsets");
            entries.Add((property.Name, value.GetProperty("dtype").GetString()!, shape,
                         offsets[0].GetInt64(), offsets[1].GetInt64()));
        }

        // Reading in file order keeps this a single forward pass over the stream,
        // which matters for the ~200 MB float32 checkpoint dumps.
        entries.Sort((a, b) => a.Start.CompareTo(b.Start));

        long dataStart = 8 + headerLength;
        var result = new Dictionary<string, NdArray>(entries.Count);
        long cursor = 0;

        foreach (var entry in entries)
        {
            if (entry.Start < cursor)
            {
                stream.Seek(dataStart + entry.Start, SeekOrigin.Begin);
                cursor = entry.Start;
            }
            else if (entry.Start > cursor)
            {
                Skip(stream, entry.Start - cursor);
                cursor = entry.Start;
            }

            int byteLength = checked((int)(entry.End - entry.Start));
            var raw = new byte[byteLength];
            stream.ReadExactly(raw);
            cursor += byteLength;

            result[entry.Name] = Decode(entry.Dtype, raw, entry.Shape, entry.Name);
        }

        return result;
    }

    private static void Skip(Stream stream, long count)
    {
        if (stream.CanSeek) { stream.Seek(count, SeekOrigin.Current); return; }
        var scratch = new byte[System.Math.Min(count, 1 << 20)];
        while (count > 0)
        {
            int chunk = (int)System.Math.Min(count, scratch.Length);
            stream.ReadExactly(scratch, 0, chunk);
            count -= chunk;
        }
    }

    private static NdArray Decode(string dtype, ReadOnlySpan<byte> raw, int[] shape, string name)
    {
        int count = NdArray.Count(shape);
        var array = new NdArray(shape);
        var target = array.Span;

        switch (dtype)
        {
            case "F32":
                if (raw.Length != count * 4) throw Mismatch(name, raw.Length, count * 4);
                for (int i = 0; i < count; i++)
                    target[i] = BinaryPrimitives.ReadSingleLittleEndian(raw.Slice(i * 4, 4));
                break;

            case "F16":
                if (raw.Length != count * 2) throw Mismatch(name, raw.Length, count * 2);
                for (int i = 0; i < count; i++)
                    target[i] = (float)BinaryPrimitives.ReadHalfLittleEndian(raw.Slice(i * 2, 2));
                break;

            case "BF16":
                if (raw.Length != count * 2) throw Mismatch(name, raw.Length, count * 2);
                for (int i = 0; i < count; i++)
                {
                    // bfloat16 is the top 16 bits of a float32.
                    uint bits = (uint)BinaryPrimitives.ReadUInt16LittleEndian(raw.Slice(i * 2, 2)) << 16;
                    target[i] = BitConverter.UInt32BitsToSingle(bits);
                }
                break;

            case "F64":
                if (raw.Length != count * 8) throw Mismatch(name, raw.Length, count * 8);
                for (int i = 0; i < count; i++)
                    target[i] = (float)BinaryPrimitives.ReadDoubleLittleEndian(raw.Slice(i * 8, 8));
                break;

            case "I64":
                if (raw.Length != count * 8) throw Mismatch(name, raw.Length, count * 8);
                for (int i = 0; i < count; i++)
                    target[i] = BinaryPrimitives.ReadInt64LittleEndian(raw.Slice(i * 8, 8));
                break;

            case "I32":
                if (raw.Length != count * 4) throw Mismatch(name, raw.Length, count * 4);
                for (int i = 0; i < count; i++)
                    target[i] = BinaryPrimitives.ReadInt32LittleEndian(raw.Slice(i * 4, 4));
                break;

            default:
                throw new NotSupportedException($"safetensors dtype '{dtype}' (tensor '{name}') is not supported.");
        }

        return array;
    }

    private static InvalidDataException Mismatch(string name, int actual, int expected) =>
        new($"Tensor '{name}' has {actual} bytes of data, expected {expected}.");

    /// <summary>Writes <paramref name="tensors"/> as float32 safetensors.</summary>
    public static void Save(IReadOnlyDictionary<string, NdArray> tensors, string path)
    {
        using var stream = File.Create(path);
        Save(tensors, stream);
    }

    /// <summary>Writes <paramref name="tensors"/> as float32 safetensors.</summary>
    public static void Save(IReadOnlyDictionary<string, NdArray> tensors, Stream stream)
    {
        var ordered = tensors.OrderBy(kv => kv.Key, StringComparer.Ordinal).ToList();

        var header = new StringBuilder("{");
        long offset = 0;
        bool first = true;
        foreach (var (name, tensor) in ordered)
        {
            if (!first) header.Append(',');
            first = false;
            long bytes = (long)tensor.Length * 4;
            header.Append(JsonSerializer.Serialize(name))
                  .Append(":{\"dtype\":\"F32\",\"shape\":[")
                  .Append(string.Join(",", tensor.Shape))
                  .Append("],\"data_offsets\":[").Append(offset).Append(',').Append(offset + bytes)
                  .Append("]}");
            offset += bytes;
        }
        header.Append('}');

        var headerBytes = Encoding.UTF8.GetBytes(header.ToString());
        // The spec wants the data block 8-byte aligned; pad the header with spaces.
        int padding = (int)((8 - (headerBytes.Length % 8)) % 8);
        if (padding > 0)
        {
            var padded = new byte[headerBytes.Length + padding];
            headerBytes.CopyTo(padded, 0);
            padded.AsSpan(headerBytes.Length).Fill((byte)' ');
            headerBytes = padded;
        }

        Span<byte> lengthBytes = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(lengthBytes, (ulong)headerBytes.Length);
        stream.Write(lengthBytes);
        stream.Write(headerBytes);

        var buffer = new byte[1 << 16];
        foreach (var (_, tensor) in ordered)
        {
            var data = tensor.ReadSpan;
            int written = 0;
            while (written < data.Length)
            {
                int chunk = System.Math.Min(buffer.Length / 4, data.Length - written);
                for (int i = 0; i < chunk; i++)
                    BinaryPrimitives.WriteSingleLittleEndian(buffer.AsSpan(i * 4, 4), data[written + i]);
                stream.Write(buffer, 0, chunk * 4);
                written += chunk;
            }
        }
    }
}
