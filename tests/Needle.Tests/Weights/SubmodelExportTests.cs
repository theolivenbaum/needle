using Needle.Model;
using Needle.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace Needle.Tests.Weights;

public sealed class SubmodelExportTests
{
    private static TransformerConfig Config(int dFf = 64) => new TransformerConfig
    {
        VocabSize        = 32,
        DModel           = 16,
        NumHeads         = 2,
        NumKvHeads       = 1,
        NumEncoderLayers = 1,
        NumDecoderLayers = 1,
        DFf              = dFf,
        MaxSeqLen        = 8,
        ContrastiveDim   = 8,
        NoFeedforward    = false,
    };

    [Fact]
    public void SliceParams_HalvesFfnIntermediate_GateAndUpProj()
    {
        var cfg = Config(dFf: 64);
        var src = new Dictionary<string, Tensor>
        {
            ["encoder.layer_0.ffn.gate_proj.weight"] = torch.randn(cfg.DModel, cfg.DFf),
            ["encoder.layer_0.ffn.up_proj.weight"]   = torch.randn(cfg.DModel, cfg.DFf),
            ["encoder.layer_0.ffn.down_proj.weight"] = torch.randn(cfg.DFf, cfg.DModel),
            // Non-FFN tensor — should be passed through
            ["embedding.weight"]                     = torch.randn(cfg.VocabSize, cfg.DModel),
        };

        var (sliced, newCfg) = SubmodelExport.SliceParams(src, cfg, factor: 2);
        try
        {
            Assert.Equal(32, newCfg.DFf);

            Assert.Equal(new long[] { cfg.DModel, 32 }, sliced["encoder.layer_0.ffn.gate_proj.weight"].shape);
            Assert.Equal(new long[] { cfg.DModel, 32 }, sliced["encoder.layer_0.ffn.up_proj.weight"].shape);
            Assert.Equal(new long[] { 32, cfg.DModel }, sliced["encoder.layer_0.ffn.down_proj.weight"].shape);
            Assert.Equal(new long[] { cfg.VocabSize, cfg.DModel }, sliced["embedding.weight"].shape);
        }
        finally
        {
            foreach (var t in src.Values)    t.Dispose();
            foreach (var t in sliced.Values) t.Dispose();
        }
    }

    [Fact]
    public void SliceParams_PreservesFirstNValues()
    {
        var cfg = Config(dFf: 64);
        var src = new Dictionary<string, Tensor>
        {
            ["ffn.gate_proj.weight"] = torch.arange(cfg.DModel * cfg.DFf, dtype: ScalarType.Float32)
                                            .reshape(cfg.DModel, cfg.DFf),
        };

        var (sliced, _) = SubmodelExport.SliceParams(src, cfg, factor: 2);
        try
        {
            // First column (dim 1, index 0) should still be 0..15
            // because we sliced [:, :32] from [16, 64].
            var first = sliced["ffn.gate_proj.weight"][0].data<float>().ToArray();
            for (int i = 0; i < 32; i++)
                Assert.Equal((float)i, first[i]);
        }
        finally
        {
            foreach (var t in src.Values)    t.Dispose();
            foreach (var t in sliced.Values) t.Dispose();
        }
    }

    [Fact]
    public void SliceParams_3DStacked_Weights()
    {
        // Stacked FFN weights (some implementations stack scanned layers)
        var cfg = Config(dFf: 64);
        const int K = 3;
        var src = new Dictionary<string, Tensor>
        {
            ["scanned.gate_proj.weight"] = torch.randn(K, cfg.DModel, cfg.DFf),
            ["scanned.down_proj.weight"] = torch.randn(K, cfg.DFf, cfg.DModel),
        };

        var (sliced, _) = SubmodelExport.SliceParams(src, cfg, factor: 4);
        try
        {
            Assert.Equal(new long[] { K, cfg.DModel, 16 }, sliced["scanned.gate_proj.weight"].shape);
            Assert.Equal(new long[] { K, 16, cfg.DModel }, sliced["scanned.down_proj.weight"].shape);
        }
        finally
        {
            foreach (var t in src.Values)    t.Dispose();
            foreach (var t in sliced.Values) t.Dispose();
        }
    }

    [Fact]
    public void SliceParams_RejectsFactorThatGivesZeroFfn()
    {
        var cfg = Config(dFf: 4);
        var src = new Dictionary<string, Tensor>
        {
            ["ffn.gate_proj.weight"] = torch.randn(cfg.DModel, cfg.DFf),
        };

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SubmodelExport.SliceParams(src, cfg, factor: 8));

        foreach (var t in src.Values) t.Dispose();
    }

    [Fact]
    public void ExportSubmodel_RoundTripsViaNdlw()
    {
        string tmpSrc = Path.GetTempFileName();
        string tmpDst = Path.GetTempFileName();
        try
        {
            var cfg = Config(dFf: 64);
            var src = new Dictionary<string, Tensor>
            {
                ["ffn.gate_proj.weight"] = torch.ones(new long[] { cfg.DModel, cfg.DFf }),
                ["ffn.up_proj.weight"]   = torch.ones(new long[] { cfg.DModel, cfg.DFf }),
                ["ffn.down_proj.weight"] = torch.ones(new long[] { cfg.DFf,   cfg.DModel }),
                ["embedding.weight"]     = torch.ones(new long[] { cfg.VocabSize, cfg.DModel }),
            };
            WeightLoader.Save(src, tmpSrc);
            foreach (var t in src.Values) t.Dispose();

            var newCfg = SubmodelExport.ExportSubmodel(tmpSrc, cfg, factor: 2, tmpDst);
            Assert.Equal(32, newCfg.DFf);

            var (_, loaded) = WeightLoader.Load(tmpDst);
            try
            {
                Assert.Equal(new long[] { cfg.DModel, 32 }, loaded["ffn.gate_proj.weight"].shape);
                Assert.Equal(new long[] { 32, cfg.DModel }, loaded["ffn.down_proj.weight"].shape);
                Assert.Equal(new long[] { cfg.VocabSize, cfg.DModel }, loaded["embedding.weight"].shape);
            }
            finally
            {
                foreach (var t in loaded.Values) t.Dispose();
            }
        }
        finally
        {
            if (File.Exists(tmpSrc)) File.Delete(tmpSrc);
            if (File.Exists(tmpDst)) File.Delete(tmpDst);
        }
    }
}
