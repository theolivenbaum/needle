using Needle.Diagnostics;
using Needle.Math;
using Needle.Model;
using Needle.Tokenizer;
using Needle.Training;
using Needle.Training.Autodiff;
using Needle.Weights;
using Ops = Needle.Training.Autodiff.Ops;

namespace Needle.Tests.Training;

/// <summary>
/// A small random model, big enough to exercise every path — two engram sites,
/// two lanes, grouped-query attention — and small enough to differentiate
/// numerically.
/// </summary>
internal static class TinyModel
{
    public static TransformerConfig Config { get; } = new()
    {
        VocabSize = 32,
        DModel = 16,
        NumHeads = 2,
        NumKvHeads = 1,
        NumLayers = 4,
        MaxSeqLen = 32,
        ContrastiveDim = 8,
        EngramOrders = [2, 3],
        EngramHeads = 1,
        EngramSlots = 13,
        EngramLayers = [1, 3],
        MhcLanes = 2,
    };

    public static Needle2Weights Build(int seed = 1)
    {
        var random = new Random(seed);
        var config = Config;
        int d = config.DModel, a = config.AttnWidth, kv = config.KvDim, hd = config.HeadDim;
        int l = config.NumLayers, lanes = config.MhcLanes, nc = lanes * d;
        int hw = config.HadamardWidth;
        var (orders, heads, subDim) = config.EngramGeometry();
        int tables = orders.Length * heads;

        NdArray Random(params int[] shape)
        {
            var value = new NdArray(shape);
            for (int i = 0; i < value.Length; i++) value[i] = (float)(random.NextDouble() - 0.5) * 0.3f;
            return value;
        }

        var flat = new Dictionary<string, NdArray>
        {
            ["embedding/embedding"] = Random(config.VocabSize, d),
            ["stack/final_norm/scale"] = Random(d),
            ["stack/layers/block/ZCRMSNorm_0/scale"] = Random(l, d),
            ["stack/layers/block/self_attn/q_proj/kernel"] = Random(l, d, a),
            ["stack/layers/block/self_attn/k_proj/kernel"] = Random(l, d, kv),
            ["stack/layers/block/self_attn/v_proj/kernel"] = Random(l, d, kv),
            ["stack/layers/block/self_attn/gate_proj/kernel"] = Random(l, d, a),
            ["stack/layers/block/self_attn/out_proj/kernel"] = Random(l, a, d),
            ["stack/layers/block/self_attn/q_norm/scale"] = Random(l, hd),
            ["stack/layers/block/self_attn/k_norm/scale"] = Random(l, hd),
            ["stack/layers/block/post_attn_norm/scale"] = Random(l, d),
            ["stack/layers/block/attn_gate"] = Random(l),
            ["stack/layers/block/pre_hada_norm/scale"] = Random(l, d),
            ["stack/layers/block/hadamard_mlp/d1"] = Random(l, hw),
            ["stack/layers/block/hadamard_mlp/d2"] = Random(l, hw),
            ["stack/layers/block/hadamard_mlp/d3"] = Random(l, hw),
            ["stack/mhc_phi_pre"] = Random(l, nc, lanes),
            ["stack/mhc_phi_post"] = Random(l, nc, lanes),
            ["stack/mhc_phi_res"] = Random(l, nc, lanes * lanes),
            ["stack/mhc_b_pre"] = Random(l, lanes),
            ["stack/mhc_b_post"] = Random(l, lanes),
            ["stack/mhc_b_res"] = Random(l, lanes, lanes),
            ["stack/mhc_a_pre"] = Random(l),
            ["stack/mhc_a_post"] = Random(l),
            ["stack/mhc_a_res"] = Random(l),
        };

        for (int s = 0; s < config.EngramSites; s++)
        {
            flat[$"engrams_{s}/embedding"] = Random(tables, config.EngramSlots, subDim);
            flat[$"engrams_{s}/key_proj/kernel"] = Random(tables * subDim, d);
            flat[$"engrams_{s}/value_proj/kernel"] = Random(tables * subDim, d);
            flat[$"engrams_{s}/taps"] = Random(EngramConstants.ConvTaps, d);
        }

        return Needle2Weights.FromFlat(config, flat);
    }

    public static int[] Tokens(int length, int seed = 3)
    {
        var random = new Random(seed);
        var tokens = new int[length];
        for (int i = 0; i < length; i++) tokens[i] = random.Next(Config.VocabSize);
        return tokens;
    }
}

public class AutodiffOpTests
{
    /// <summary>
    /// Central differences against the analytic gradient of a scalar reduction
    /// of <paramref name="build"/>'s output.
    /// </summary>
    private static void CheckOp(Func<Tape, Value, Value> build, int[] shape, int seed = 5,
                                float epsilon = 1e-3f, float tolerance = 2e-2f)
    {
        var random = new Random(seed);
        var input = new NdArray(shape);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() - 0.5) * 2f;

        // A random linear readout makes every output coordinate contribute, so a
        // sign error anywhere shows up.
        var readout = new NdArray((int[])shape.Clone());

        float Scalar()
        {
            var scratch = new Tape();
            var value = scratch.Parameter(input);
            var output = build(scratch, value);
            float total = 0f;
            for (int i = 0; i < output.Length; i++) total += output.Data[i] * readout[i % readout.Length];
            return total;
        }

        for (int i = 0; i < readout.Length; i++) readout[i] = (float)(random.NextDouble() - 0.5) * 2f;

        var tape = new Tape();
        var parameter = tape.Parameter(input);
        tape.Backward(SumWithWeights(tape, build(tape, parameter), readout));

        for (int i = 0; i < System.Math.Min(6, input.Length); i++)
        {
            float original = input[i];
            input[i] = original + epsilon;
            float up = Scalar();
            input[i] = original - epsilon;
            float down = Scalar();
            input[i] = original;

            float numeric = (up - down) / (2f * epsilon);
            float analytic = parameter.Grad![i];
            float scale = System.Math.Max(1e-4f, System.Math.Max(System.Math.Abs(numeric), System.Math.Abs(analytic)));
            Assert.True(System.Math.Abs(numeric - analytic) / scale < tolerance,
                $"index {i}: analytic {analytic:G6} vs numeric {numeric:G6}");
        }
    }

    /// <summary>Weighted sum reduction, so the check has a scalar to differentiate.</summary>
    private static Value SumWithWeights(Tape tape, Value x, NdArray weights)
    {
        var result = new NdArray(1);
        for (int i = 0; i < x.Length; i++) result[0] += x.Data[i] * weights[i % weights.Length];

        Value? node = null;
        node = tape.Record(result, () =>
        {
            float seed = node!.Grad![0];
            for (int i = 0; i < x.Length; i++) x.Grad![i] += seed * weights[i % weights.Length];
        }, x);
        return node;
    }

    [Fact]
    public void SigmoidGradient() => CheckOp((tape, x) => Ops.Sigmoid(tape, x), [4, 5]);

    [Fact]
    public void SiluGradient() => CheckOp((tape, x) => Ops.Silu(tape, x), [4, 5]);

    [Fact]
    public void RmsUnitGradient() => CheckOp((tape, x) => Ops.RmsUnit(tape, x), [3, 8]);

    [Fact]
    public void WalshGradient() => CheckOp((tape, x) => Ops.Walsh(tape, x), [3, 8]);

    [Fact]
    public void SoftmaxGradient() => CheckOp((tape, x) => Ops.Softmax(tape, x, 5), [4, 5]);

    [Fact]
    public void SinkhornGradient() => CheckOp((tape, x) => Ops.Sinkhorn(tape, x, 2), [4, 4], tolerance: 5e-2f);

    [Fact]
    public void ZcRmsNormGradientFlowsToInputAndScale()
    {
        var random = new Random(9);
        var input = new NdArray(3, 6);
        var scale = new NdArray(6);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() - 0.5) * 2f;
        for (int i = 0; i < scale.Length; i++) scale[i] = (float)(random.NextDouble() - 0.5);

        var readout = new NdArray(3, 6);
        for (int i = 0; i < readout.Length; i++) readout[i] = (float)(random.NextDouble() - 0.5) * 2f;

        float Scalar()
        {
            var scratch = new Tape();
            var output = Ops.ZcRmsNorm(scratch, scratch.Parameter(input), scratch.Parameter(scale));
            float total = 0f;
            for (int i = 0; i < output.Length; i++) total += output.Data[i] * readout[i];
            return total;
        }

        var tape = new Tape();
        var x = tape.Parameter(input, "x");
        var g = tape.Parameter(scale, "scale");
        var result = Ops.ZcRmsNorm(tape, x, g);
        tape.Backward(SumWithWeights(tape, result, readout));

        const float Epsilon = 1e-3f;
        for (int i = 0; i < 4; i++)
        {
            float original = input[i];
            input[i] = original + Epsilon;
            float up = Scalar();
            input[i] = original - Epsilon;
            float down = Scalar();
            input[i] = original;
            Assert.Equal((up - down) / (2 * Epsilon), x.Grad![i], 2);
        }

        for (int i = 0; i < scale.Length; i++)
        {
            float original = scale[i];
            scale[i] = original + Epsilon;
            float up = Scalar();
            scale[i] = original - Epsilon;
            float down = Scalar();
            scale[i] = original;
            Assert.Equal((up - down) / (2 * Epsilon), g.Grad![i], 2);
        }
    }

    [Fact]
    public void MatMulGradientFlowsToBothOperands()
    {
        var random = new Random(11);
        var input = new NdArray(3, 4);
        var kernel = new NdArray(4, 5);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() - 0.5);
        for (int i = 0; i < kernel.Length; i++) kernel[i] = (float)(random.NextDouble() - 0.5);

        var readout = new NdArray(3, 5);
        for (int i = 0; i < readout.Length; i++) readout[i] = (float)(random.NextDouble() - 0.5) * 2f;

        float Scalar()
        {
            var scratch = new Tape();
            var output = Ops.MatMul(scratch, scratch.Parameter(input), scratch.Parameter(kernel));
            float total = 0f;
            for (int i = 0; i < output.Length; i++) total += output.Data[i] * readout[i];
            return total;
        }

        var tape = new Tape();
        var x = tape.Parameter(input, "x");
        var w = tape.Parameter(kernel, "w");
        tape.Backward(SumWithWeights(tape, Ops.MatMul(tape, x, w), readout));

        const float Epsilon = 1e-3f;
        foreach (var (tensor, handle) in new[] { (input, x), (kernel, w) })
        {
            for (int i = 0; i < System.Math.Min(5, tensor.Length); i++)
            {
                float original = tensor[i];
                tensor[i] = original + Epsilon;
                float up = Scalar();
                tensor[i] = original - Epsilon;
                float down = Scalar();
                tensor[i] = original;
                Assert.Equal((up - down) / (2 * Epsilon), handle.Grad![i], 2);
            }
        }
    }

    [Fact]
    public void CrossEntropyIgnoresMaskedPositions()
    {
        var tape = new Tape();
        var logits = tape.Parameter(new NdArray(2, 4), "logits");
        logits.Data[0] = 5f;    // position 0 strongly prefers token 0
        logits.Data[5] = 5f;    // position 1 strongly prefers token 1

        // Only position 1 is supervised, and it is already correct.
        var loss = Ops.CrossEntropy(tape, logits, [3, 1], [0f, 1f]);
        Assert.True(loss.Data[0] < 0.1f);

        tape.Backward(loss);
        // A masked position must receive no gradient at all.
        for (int c = 0; c < 4; c++) Assert.Equal(0f, logits.Grad![c]);
    }
}

public class TrainableModelTests
{
    [Fact]
    public void TrainingForwardMatchesInferenceForward()
    {
        var weights = TinyModel.Build();
        var delta = TrainingCheck.ForwardMatchesInference(weights, TinyModel.Tokens(9));

        // Two independent implementations of the same arithmetic: anything beyond
        // float32 reassociation means they have drifted apart.
        Assert.True(delta.MaxRelative < 1e-4f, delta.ToString());
        Assert.True(delta.Cosine > 0.999999, delta.ToString());
    }

    [Fact]
    public void UntrainedAdaptersLeaveTheModelUnchanged()
    {
        var weights = TinyModel.Build();
        var tokens = TinyModel.Tokens(7);
        var adapters = LoraSet.Create(weights, rank: 4, alpha: 8f);

        var model = new TrainableModel(weights) { AdapterScale = 2f };
        var tape = new Tape();
        var adapted = model.Forward(tape, tokens, adapters.Register(tape));

        var inference = new Needle2Model(weights);
        var expected = inference.Logits(inference.Forward(tokens).Hidden);

        // B is initialised to zero, so the correction starts at exactly zero.
        var delta = ParityHarness.Delta("zero-init", expected.ReadSpan, adapted.Data.ReadSpan);
        Assert.True(delta.MaxRelative < 1e-4f, delta.ToString());
    }

    [Fact]
    public void GradientsMatchFiniteDifferences()
    {
        var weights = TinyModel.Build();
        var tokens = TinyModel.Tokens(6);
        var targets = TinyModel.Tokens(6, seed: 4);
        var adapters = LoraSet.Create(weights, rank: 4, alpha: 8f, seed: 2);

        // Move B off zero: with B = 0 the gradient w.r.t. A is zero too, which
        // would make the check vacuous.
        var random = new Random(17);
        foreach (var adapter in adapters.Adapters)
            for (int i = 0; i < adapter.B.Length; i++)
                adapter.B[i] = (float)(random.NextDouble() - 0.5) * 0.1f;

        var probes = TrainingCheck.CheckGradients(weights, tokens, targets, adapters,
                                                  probes: 10, epsilon: 5e-3f);

        Assert.NotEmpty(probes);
        foreach (var probe in probes) Assert.True(probe.Agrees(), probe.ToString());

        // The check is only meaningful if the probed gradients were actually
        // resolvable; otherwise it would pass on a backward that returned zero.
        Assert.Contains(probes, p => System.Math.Abs(p.Analytic) > 10f * p.NoiseFloor);
    }

    [Fact]
    public void MergingAnAdapterReproducesTheAdaptedForward()
    {
        var weights = TinyModel.Build();
        var tokens = TinyModel.Tokens(5);
        var adapters = LoraSet.Create(weights, rank: 4, alpha: 8f, seed: 6);

        var random = new Random(23);
        foreach (var adapter in adapters.Adapters)
            for (int i = 0; i < adapter.B.Length; i++)
                adapter.B[i] = (float)(random.NextDouble() - 0.5) * 0.2f;

        var model = new TrainableModel(weights) { AdapterScale = adapters.Alpha / adapters.Rank };
        var tape = new Tape();
        var adapted = model.Forward(tape, tokens, adapters.Register(tape));

        // Folding the adapters into the kernels must give the same model.
        var merged = new Needle2Model(adapters.Merge(weights));
        var expected = merged.Logits(merged.Forward(tokens).Hidden);

        var delta = ParityHarness.Delta("merged", expected.ReadSpan, adapted.Data.ReadSpan);
        Assert.True(delta.MaxRelative < 1e-3f, delta.ToString());
    }
}

public class OptimizerTests
{
    [Fact]
    public void AdamWDescendsAQuadratic()
    {
        // Minimise (w - 3)^2, whose gradient is 2(w - 3).
        var tape = new Tape();
        var w = tape.Parameter(new NdArray(1), "w");
        var optimizer = new AdamW(tape.Trainable, 0.1f);

        for (int step = 0; step < 400; step++)
        {
            optimizer.ZeroGrad();
            w.Grad![0] = 2f * (w.Data[0] - 3f);
            optimizer.Apply();
        }

        Assert.Equal(3.0, w.Data[0], 2);
    }

    [Fact]
    public void ClipReportsTheNormAndBoundsIt()
    {
        var tape = new Tape();
        var a = tape.Parameter(new NdArray(2), "a");
        var optimizer = new AdamW(tape.Trainable, 0.1f);

        a.Grad![0] = 3f;
        a.Grad![1] = 4f;

        Assert.Equal(5f, optimizer.ClipGradientNorm(1f), 4);
        Assert.Equal(0.6f, a.Grad![0], 4);
        Assert.Equal(0.8f, a.Grad![1], 4);
    }

    [Fact]
    public void AtZeroLearningRateNothingMoves()
    {
        // What Finetuner.Warmup leans on: a scratch optimiser can run the whole
        // update — moments, bias correction, clipping — and leave the parameters
        // bit-identical, so warming the JIT cannot perturb the run it precedes.
        var tape = new Tape();
        var w = tape.Parameter(new NdArray(4), "w");
        for (int i = 0; i < w.Length; i++) w.Data[i] = 0.5f - i;

        var before = w.Data.ReadSpan.ToArray();
        var optimizer = new AdamW(tape.Trainable, 0f) { WeightDecay = 0.1f };

        for (int step = 0; step < 3; step++)
        {
            optimizer.ZeroGrad();
            for (int i = 0; i < w.Length; i++) w.Grad![i] = i + 1f;
            optimizer.ClipGradientNorm(1f);
            optimizer.Apply();
        }

        Assert.Equal(before, w.Data.ReadSpan.ToArray());
    }

    [Fact]
    public void WarmupCosineRisesThenDecays()
    {
        var schedule = new WarmupCosine(1e-3f, TotalSteps: 100, WarmupRatio: 0.1f, FinalRatio: 0.1f);

        Assert.True(schedule.At(0) < schedule.At(5));
        Assert.Equal(1e-3f, schedule.At(9), 6);
        Assert.True(schedule.At(50) < schedule.At(9));
        Assert.True(schedule.At(99) < schedule.At(50));
        Assert.True(schedule.At(99) >= 1e-4f * 0.99f);
    }
}

public class FinetunerTests
{
    private static List<FinetuneExample> Examples() =>
    [
        new("dim the kitchen", "[{\"name\":\"set_lights\"}]",
            "[{\"name\":\"set_lights\",\"arguments\":{\"room\":\"kitchen\"}}]", "'kitchen' -> room"),
        new("turn on the porch light", "[{\"name\":\"set_lights\"}]",
            "[{\"name\":\"set_lights\",\"arguments\":{\"room\":\"porch\"}}]"),
        new("what is the meaning of life", "[{\"name\":\"set_lights\"}]", "[]"),
    ];

    [Fact]
    public void LoraTargetsMatchTheReference()
    {
        Assert.Equal(["q_proj", "k_proj", "v_proj", "gate_proj", "out_proj"], LoraSet.Targets);

        var adapters = LoraSet.Create(TinyModel.Build(), rank: 2, alpha: 4f);
        Assert.Equal(TinyModel.Config.NumLayers * 5, adapters.Adapters.Count);
    }

    [Fact]
    public void AdapterFactorsStartAtIdentity()
    {
        var adapters = LoraSet.Create(TinyModel.Build(), rank: 4, alpha: 8f);

        // B is zero, so the correction is zero and training starts from the base.
        foreach (var adapter in adapters.Adapters)
            Assert.All(adapter.B.ReadSpan.ToArray(), v => Assert.Equal(0f, v));
        Assert.Contains(adapters.Adapters, a => a.A.ReadSpan.ToArray().Any(v => v != 0f));
    }

    /// <summary>
    /// The whole flow on the released weights: warm, measure, train, measure
    /// again.  Everything above this tests a piece; this tests that the pieces
    /// compose into an optimiser that descends the loss it reports.
    /// </summary>
    [SkippableFact]
    public void FineTuningDescendsTheLossItReports()
    {
        string? blob = Parity.Fixtures.CactBlob;
        Skip.If(blob is null, "No .cact blob in the fixture directory.");

        var loaded = CactLayout.Load(blob!);
        var weights = Needle2Weights.FromFlat(loaded.Config, loaded.Parameters);
        var tokenizer = CactTokenizer.FromCact(blob!);

        var examples = Examples();
        var finetuner = new Finetuner(weights, tokenizer,
            new FinetuneConfig(Epochs: 3, LearningRate: 1e-3f, Rank: 4, Alpha: 8f,
                               MaxLength: 64, BatchSize: 3));

        // Warming must not train: the loss measured after it has to be the
        // untouched model's, or the "before" number below is already stale.
        float cold = finetuner.Evaluate(examples);
        finetuner.Warmup(examples);
        Assert.Equal(cold, finetuner.Evaluate(examples), 5);

        var history = finetuner.Run(examples);
        Assert.Equal(3, history.Count);

        // The step loss and the evaluation loss have to agree — a mis-scaled
        // backward seed shows up here as a constant factor and nowhere else.
        Assert.Equal(cold, history[0].Loss, 3);
        Assert.True(finetuner.Evaluate(examples) < cold,
            $"loss did not fall: {cold:F4} -> {finetuner.Evaluate(examples):F4}");
    }

    [Fact]
    public void AdaptersRoundTripThroughDisk()
    {
        string path = Path.GetTempFileName();
        try
        {
            var adapters = LoraSet.Create(TinyModel.Build(), rank: 4, alpha: 8f, seed: 5);
            adapters.Adapters[0].B[0] = 0.25f;
            adapters.Save(path);

            var loaded = LoraSet.Load(path);
            Assert.Equal(adapters.Adapters.Count, loaded.Adapters.Count);
            Assert.Equal(0.25f, loaded.Find(adapters.Adapters[0].Name)!.B[0], 5);
            Assert.Equal(adapters.Alpha / adapters.Rank, loaded.Adapters[0].Scale, 5);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
