using System.Diagnostics;
using Needle.Model;
using Needle.Tokenizer;
using Needle.Training.Autodiff;

namespace Needle.Training;

/// <summary>Settings for a LoRA fine-tuning run.</summary>
/// <param name="Epochs">Passes over the training split.</param>
/// <param name="LearningRate">Peak learning rate.</param>
/// <param name="Rank">LoRA rank.</param>
/// <param name="Alpha">LoRA alpha; the correction is scaled by alpha/rank.</param>
/// <param name="MaxLength">Sequence length examples are padded or truncated to.</param>
/// <param name="BatchSize">Examples accumulated per optimiser step.</param>
/// <param name="WeightDecay">Decoupled weight decay.</param>
/// <param name="GradientClip">Global gradient-norm clip; 0 disables it.</param>
/// <param name="Window">Sliding-window width during training; 0 for plain causal.</param>
/// <param name="Seed">Shuffle and initialisation seed.</param>
public sealed record FinetuneConfig(
    int Epochs = 3,
    float LearningRate = 1e-4f,
    int Rank = 16,
    float Alpha = 32f,
    int MaxLength = 256,
    int BatchSize = 4,
    float WeightDecay = 0.0f,
    float GradientClip = 1.0f,
    int Window = 0,
    int Seed = 0);

/// <summary>What one optimiser step did.</summary>
/// <param name="Step">1-based step index.</param>
/// <param name="Epoch">1-based epoch.</param>
/// <param name="Loss">Mean masked cross-entropy over the batch.</param>
/// <param name="GradientNorm">Global gradient norm before clipping.</param>
/// <param name="LearningRate">Rate this step used.</param>
/// <param name="Elapsed">Wall time for the step.</param>
public sealed record TrainingStep(
    int Step, int Epoch, float Loss, float GradientNorm, float LearningRate, TimeSpan Elapsed);

/// <summary>
/// LoRA fine-tuning over the JSONL format the reference uses.
///
/// The base model is frozen and only the adapters move, which keeps the step
/// cost proportional to the adapter size rather than the model's, and means the
/// result merges back into an ordinary checkpoint.
/// </summary>
public sealed class Finetuner
{
    private readonly TrainableModel _model;
    private readonly CactTokenizer _tokenizer;
    private readonly FinetuneConfig _config;

    /// <summary>The adapters being trained.</summary>
    public LoraSet Adapters { get; }

    public Finetuner(Needle2Weights weights, CactTokenizer tokenizer, FinetuneConfig config)
    {
        _model = new TrainableModel(weights);
        _tokenizer = tokenizer;
        _config = config;
        Adapters = LoraSet.Create(weights, config.Rank, config.Alpha, config.Seed);
        _model.AdapterScale = config.Alpha / config.Rank;
    }

    /// <summary>
    /// Run the whole training path — forward, backward, clip, optimiser step —
    /// without changing anything, so the first timed step measures the step
    /// rather than the JIT.
    ///
    /// The tiered JIT promotes a method after about thirty calls, and one pass
    /// touches each kernel once per layer, so a couple of passes is enough to
    /// leave the hot code at tier 1.  It also faults in the expanded float32
    /// weights and the optimiser's moment buffers, which are otherwise first
    /// touched inside step one.  The scratch optimiser runs at a learning rate of
    /// zero, so every parameter comes back bit-identical.
    /// </summary>
    /// <param name="examples">Examples to warm on; the first few are used.</param>
    /// <param name="iterations">Forward/backward passes to run.</param>
    public void Warmup(IReadOnlyList<FinetuneExample> examples, int iterations = 3)
    {
        if (examples.Count == 0 || iterations <= 0) return;

        var tape = new Tape();
        var adapters = Adapters.Register(tape);
        var scratch = new AdamW(tape.Trainable, 0f) { WeightDecay = 0f };

        for (int i = 0; i < iterations; i++)
        {
            scratch.ZeroGrad();
            tape.Clear();
            Accumulate(tape, adapters, examples[i % examples.Count], 1);
            scratch.ClipGradientNorm(_config.GradientClip);
            scratch.Apply();
        }

        // Leave the collector nothing to do mid-measurement.
        GC.Collect();
        GC.WaitForPendingFinalizers();
    }

    /// <summary>
    /// Run the fine-tune.  <paramref name="onStep"/> is called after every
    /// optimiser step.
    /// </summary>
    /// <param name="examples">Training examples.</param>
    /// <param name="onStep">Progress callback.</param>
    /// <returns>Every step's record.</returns>
    public List<TrainingStep> Run(IReadOnlyList<FinetuneExample> examples, Action<TrainingStep>? onStep = null)
    {
        if (examples.Count == 0)
            throw new ArgumentException("Nothing to train on.", nameof(examples));

        var tape = new Tape();
        var adapters = Adapters.Register(tape);
        var optimizer = new AdamW(tape.Trainable, _config.LearningRate)
        {
            WeightDecay = _config.WeightDecay,
        };

        int batchesPerEpoch = (examples.Count + _config.BatchSize - 1) / _config.BatchSize;
        var schedule = new WarmupCosine(_config.LearningRate, _config.Epochs * batchesPerEpoch);

        var order = Enumerable.Range(0, examples.Count).ToArray();
        var random = new Random(_config.Seed);
        var history = new List<TrainingStep>();
        int step = 0;

        for (int epoch = 1; epoch <= _config.Epochs; epoch++)
        {
            Shuffle(order, random);

            for (int start = 0; start < order.Length; start += _config.BatchSize)
            {
                var clock = Stopwatch.StartNew();
                optimizer.ZeroGrad();

                int count = System.Math.Min(_config.BatchSize, order.Length - start);
                float total = 0f;

                // Gradients accumulate across the batch, so a batch costs the
                // memory of one example rather than all of them.
                for (int i = 0; i < count; i++)
                {
                    tape.Clear();
                    total += Accumulate(tape, adapters, examples[order[start + i]], count);
                }

                float gradientNorm = optimizer.ClipGradientNorm(_config.GradientClip);
                optimizer.LearningRate = schedule.At(step);
                optimizer.Apply();
                clock.Stop();

                step++;
                var record = new TrainingStep(step, epoch, total / count, gradientNorm,
                                              optimizer.LearningRate, clock.Elapsed);
                history.Add(record);
                onStep?.Invoke(record);
            }
        }

        return history;
    }

    /// <summary>
    /// Record one example's forward and backward, scaling its loss so the batch
    /// averages rather than sums.
    /// </summary>
    /// <returns>The example's unscaled loss.</returns>
    private float Accumulate(
        Tape tape, IReadOnlyDictionary<string, (Value A, Value B)> adapters,
        FinetuneExample example, int batchSize)
    {
        var (ids, mask) = JsonlDataset.Encode(_tokenizer, example, _config.MaxLength);
        int length = EffectiveLength(ids, mask);
        if (length < 2) return 0f;

        // Next-token prediction: position t predicts token t+1, so the inputs are
        // the sequence minus its last token and the targets are it minus its first.
        var inputs = ids.AsSpan(0, length - 1);
        var targets = ids.AsSpan(1, length - 1);
        var weights = mask.AsSpan(1, length - 1);

        var logits = _model.Forward(tape, inputs, adapters, _config.Window);
        var loss = Autodiff.Ops.CrossEntropy(tape, logits, targets, weights);

        // The loss is already a masked mean over this example's supervised
        // positions; seeding the backward pass with 1/batchSize is what makes the
        // accumulated gradient the batch mean.  Scaling the weights instead would
        // cancel against the mean's own normaliser and silently multiply the
        // effective learning rate by the batch size.
        tape.Backward(loss, 1f / batchSize);

        return loss.Data[0];
    }

    /// <summary>Trim trailing padding, keeping one position past the last supervised token.</summary>
    private static int EffectiveLength(int[] ids, float[] mask)
    {
        int last = -1;
        for (int i = mask.Length - 1; i >= 0; i--)
        {
            if (mask[i] != 0f) { last = i; break; }
        }
        return last < 0 ? 0 : System.Math.Min(ids.Length, last + 1);
    }

    private static void Shuffle(int[] order, Random random)
    {
        for (int i = order.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
    }

    /// <summary>
    /// Mean masked cross-entropy over <paramref name="examples"/> without
    /// touching the parameters — the held-out measurement.
    /// </summary>
    public float Evaluate(IReadOnlyList<FinetuneExample> examples)
    {
        if (examples.Count == 0) return float.NaN;

        var tape = new Tape();
        var adapters = Adapters.Register(tape);
        float total = 0f;
        int counted = 0;

        foreach (var example in examples)
        {
            tape.Clear();
            var (ids, mask) = JsonlDataset.Encode(_tokenizer, example, _config.MaxLength);
            int length = EffectiveLength(ids, mask);
            if (length < 2) continue;

            var logits = _model.Forward(tape, ids.AsSpan(0, length - 1), adapters, _config.Window);
            total += Autodiff.Ops.CrossEntropy(tape, logits, ids.AsSpan(1, length - 1),
                                               mask.AsSpan(1, length - 1)).Data[0];
            counted++;
        }

        return counted == 0 ? float.NaN : total / counted;
    }
}
