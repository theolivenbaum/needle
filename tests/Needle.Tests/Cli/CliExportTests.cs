using Needle.Cli;
using Needle.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace Needle.Tests.Cli;

public sealed class CliExportTests
{
    [Fact]
    public void Export_SlicesFfnAndWritesNdlw()
    {
        string srcPath = Path.GetTempFileName();
        string dstPath = Path.GetTempFileName();

        try
        {
            // Write a minimal model: 8 vocab × 16 d_model, 64 d_ff.
            var input = new Dictionary<string, Tensor>
            {
                ["embedding.weight"]                       = torch.ones(new long[] { 8, 16 }),
                ["encoder.layer_0.gate_proj.weight"]       = torch.ones(new long[] { 16, 64 }),
                ["encoder.layer_0.up_proj.weight"]         = torch.ones(new long[] { 16, 64 }),
                ["encoder.layer_0.down_proj.weight"]       = torch.ones(new long[] { 64, 16 }),
            };
            WeightLoader.Save(input, srcPath);
            foreach (var t in input.Values) t.Dispose();

            int code = CliEntry.Run(
            [
                "export",
                "--checkpoint", srcPath,
                "--factor",     "2",
                "--output",     dstPath,
            ]);

            Assert.Equal(0, code);
            Assert.True(File.Exists(dstPath));
            Assert.True(new FileInfo(dstPath).Length > 0);

            // Verify the exported model has d_ff = 32 (64 / 2).
            var (_, loaded) = WeightLoader.Load(dstPath);
            try
            {
                Assert.Equal(new long[] { 16, 32 }, loaded["encoder.layer_0.gate_proj.weight"].shape);
                Assert.Equal(new long[] { 32, 16 }, loaded["encoder.layer_0.down_proj.weight"].shape);
                // Embedding untouched
                Assert.Equal(new long[] { 8, 16 }, loaded["embedding.weight"].shape);
            }
            finally
            {
                foreach (var t in loaded.Values) t.Dispose();
            }
        }
        finally
        {
            if (File.Exists(srcPath)) File.Delete(srcPath);
            if (File.Exists(dstPath)) File.Delete(dstPath);
        }
    }

    [Fact]
    public void Help_ReturnsZero()
    {
        // Capture stdout to keep the test output clean.
        var prev = Console.Out;
        try
        {
            Console.SetOut(TextWriter.Null);
            Assert.Equal(0, CliEntry.Run(["--help"]));
            Assert.Equal(0, CliEntry.Run([]));
        }
        finally
        {
            Console.SetOut(prev);
        }
    }

    [Fact]
    public void UnknownCommand_ReturnsNonZero()
    {
        var prevOut = Console.Out;
        var prevErr = Console.Error;
        try
        {
            Console.SetOut(TextWriter.Null);
            Console.SetError(TextWriter.Null);
            Assert.NotEqual(0, CliEntry.Run(["this-is-not-a-command"]));
        }
        finally
        {
            Console.SetOut(prevOut);
            Console.SetError(prevErr);
        }
    }
}
