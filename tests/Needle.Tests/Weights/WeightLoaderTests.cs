using System.Buffers.Binary;
using System.IO;
using System.Text;
using System.Text.Json;
using Needle.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace Needle.Tests.Weights;

/// <summary>
/// Tests for <see cref="WeightLoader"/> covering .ndlw save/load round-trips
/// and HuggingFace .safetensors loading.
/// </summary>
public sealed class WeightLoaderTests
{
    private static string TempPath(string suffix)
    {
        return Path.Combine(Path.GetTempPath(),
            $"needle_weightloader_{Guid.NewGuid():N}{suffix}");
    }

    [Fact]
    public void Ndlw_RoundTrip_Float32_PreservesValues()
    {
        string path = TempPath(".ndlw");
        try
        {
            using var a = torch.randn(3, 4);
            using var b = torch.randn(2, 3, 5);
            using var c = torch.tensor(new long[] { 1, 2, 3, 4 });

            var saveDict = new Dictionary<string, Tensor>
            {
                { "encoder.weight",     a },
                { "decoder.block.0.W",  b },
                { "indices",            c },
            };
            WeightLoader.Save(saveDict, path);

            Assert.True(File.Exists(path), "File was not written.");

            var (manifest, tensors) = WeightLoader.Load(path);
            try
            {
                Assert.Equal(3, manifest.Tensors.Count);
                Assert.Equal(3, tensors.Count);

                Assert.True(manifest.Contains("encoder.weight"));
                Assert.True(manifest.Contains("decoder.block.0.W"));
                Assert.True(manifest.Contains("indices"));

                // Shapes round-trip
                Assert.Equal(new long[] { 3, 4 },    tensors["encoder.weight"].shape);
                Assert.Equal(new long[] { 2, 3, 5 }, tensors["decoder.block.0.W"].shape);
                Assert.Equal(new long[] { 4 },       tensors["indices"].shape);

                // Values round-trip (float32)
                var origA = a.data<float>().ToArray();
                var roundA = tensors["encoder.weight"].data<float>().ToArray();
                Assert.Equal(origA.Length, roundA.Length);
                for (int i = 0; i < origA.Length; i++)
                    Assert.Equal(origA[i], roundA[i], precision: 6);

                // Int64 values round-trip
                var origC = c.data<long>().ToArray();
                var roundC = tensors["indices"].data<long>().ToArray();
                Assert.Equal(origC, roundC);

                // NdlwDtype is correct
                Assert.Equal(NdlwDtype.Float32, manifest.Tensors["encoder.weight"].Dtype);
                Assert.Equal(NdlwDtype.Int64,   manifest.Tensors["indices"].Dtype);
            }
            finally
            {
                foreach (var t in tensors.Values) t.Dispose();
            }
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Ndlw_BadMagic_Throws()
    {
        string path = TempPath(".bin");
        try
        {
            File.WriteAllBytes(path, new byte[] { 0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 1, 0, 0, 0, 0 });
            Assert.Throws<InvalidDataException>(() => WeightLoader.Load(path));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Safetensors_RoundTrip_Float32_PreservesValues()
    {
        string path = TempPath(".safetensors");
        try
        {
            // Build a small safetensors file by hand.
            float[] aData = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
            float[] bData = { -1.5f, 0.5f };

            int aBytes = aData.Length * 4;
            int bBytes = bData.Length * 4;

            var header = new
            {
                a = new
                {
                    dtype = "F32",
                    shape = new[] { 2, 3 },
                    data_offsets = new[] { 0, aBytes },
                },
                b = new
                {
                    dtype = "F32",
                    shape = new[] { 2 },
                    data_offsets = new[] { aBytes, aBytes + bBytes },
                },
            };
            string headerJson = JsonSerializer.Serialize(header);
            byte[] headerBytes = Encoding.UTF8.GetBytes(headerJson);

            using (var fs = File.Create(path))
            {
                Span<byte> sizeBuf = stackalloc byte[8];
                BinaryPrimitives.WriteUInt64LittleEndian(sizeBuf, (ulong)headerBytes.Length);
                fs.Write(sizeBuf);
                fs.Write(headerBytes);

                var aBytesArr = new byte[aBytes];
                Buffer.BlockCopy(aData, 0, aBytesArr, 0, aBytes);
                fs.Write(aBytesArr);

                var bBytesArr = new byte[bBytes];
                Buffer.BlockCopy(bData, 0, bBytesArr, 0, bBytes);
                fs.Write(bBytesArr);
            }

            var (manifest, tensors) = WeightLoader.LoadSafetensors(path);
            try
            {
                Assert.Equal(2, manifest.Tensors.Count);
                Assert.True(manifest.Contains("a"));
                Assert.True(manifest.Contains("b"));

                Assert.Equal(new long[] { 2, 3 }, tensors["a"].shape);
                Assert.Equal(new long[] { 2 },    tensors["b"].shape);

                Assert.Equal(aData, tensors["a"].data<float>().ToArray());
                Assert.Equal(bData, tensors["b"].data<float>().ToArray());

                Assert.Equal(NdlwDtype.Float32, manifest.Tensors["a"].Dtype);
            }
            finally
            {
                foreach (var t in tensors.Values) t.Dispose();
            }
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void Safetensors_IgnoresMetadataEntry()
    {
        string path = TempPath(".safetensors");
        try
        {
            float[] data = { 7.0f, 8.0f };
            int bytesLen = data.Length * 4;

            var header = new Dictionary<string, object>
            {
                { "__metadata__", new { format = "pt", note = "should be ignored" } },
                { "x", new
                    {
                        dtype = "F32",
                        shape = new[] { 2 },
                        data_offsets = new[] { 0, bytesLen },
                    }
                },
            };
            string headerJson = JsonSerializer.Serialize(header);
            byte[] headerBytes = Encoding.UTF8.GetBytes(headerJson);

            using (var fs = File.Create(path))
            {
                Span<byte> sizeBuf = stackalloc byte[8];
                BinaryPrimitives.WriteUInt64LittleEndian(sizeBuf, (ulong)headerBytes.Length);
                fs.Write(sizeBuf);
                fs.Write(headerBytes);

                var bArr = new byte[bytesLen];
                Buffer.BlockCopy(data, 0, bArr, 0, bytesLen);
                fs.Write(bArr);
            }

            var (manifest, tensors) = WeightLoader.LoadSafetensors(path);
            try
            {
                Assert.Single(manifest.Tensors);
                Assert.True(manifest.Contains("x"));
                Assert.False(manifest.Contains("__metadata__"));
                Assert.Equal(data, tensors["x"].data<float>().ToArray());
            }
            finally
            {
                foreach (var t in tensors.Values) t.Dispose();
            }
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void TensorDescriptor_NumElements_ProductOfShape()
    {
        var d = new TensorDescriptor { Shape = [3, 4, 5] };
        Assert.Equal(60, d.NumElements);

        var scalar = new TensorDescriptor { Shape = [] };
        Assert.Equal(1, scalar.NumElements);
    }

    [Fact]
    public void TensorDescriptor_ElementBytes_MatchesDtype()
    {
        Assert.Equal(4, new TensorDescriptor { Dtype = NdlwDtype.Float32 }.ElementBytes);
        Assert.Equal(2, new TensorDescriptor { Dtype = NdlwDtype.Float16 }.ElementBytes);
        Assert.Equal(2, new TensorDescriptor { Dtype = NdlwDtype.BFloat16 }.ElementBytes);
        Assert.Equal(8, new TensorDescriptor { Dtype = NdlwDtype.Int64 }.ElementBytes);
    }
}
