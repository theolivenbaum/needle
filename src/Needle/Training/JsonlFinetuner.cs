using Needle.Inference;
using Needle.Model;
using Needle.Tokenizer;
using Needle.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace Needle.Training;

/// <summary>
/// Configuration for <see cref="JsonlFinetuner"/>.
/// </summary>
public sealed record JsonlFinetuneConfig
{
    public int   Epochs        { get; init; } = 1;
    public int   BatchSize     { get; init; } = 8;
    public float AdamLr        { get; init; } = 3e-5f;
    public float MuonLr        { get; init; } = 0.02f;
    public float WarmupRatio   { get; init; } = 0.05f;
    public float DecayRatio    { get; init; } = 0.05f;
    public float GradClipNorm  { get; init; } = 1.0f;
    public int   MaxEncLen     { get; init; } = 1024;
    public int   MaxDecLen     { get; init; } = 512;
    public int   MaxGenLen     { get; init; } = 512;
    public int   Seed          { get; init; } = 42;
    public string? CheckpointDir { get; init; }
    public string  ExperimentName { get; init; } = "needle_finetuned";
    public bool   EvalAfterEachEpoch { get; init; } = true;
    public bool   Verbose       { get; init; } = true;
}

/// <summary>
/// Result of a finetune run.
/// </summary>
public sealed record JsonlFinetuneResult(
    ToolCallMetrics? BaseMetrics,
    ToolCallMetrics? FinetunedMetrics,
    string? BestCheckpointPath);

/// <summary>
/// Drive a local finetune over a JSONL dataset.  Port of
/// <c>needle/training/finetune.py::finetune_local</c>, restricted to the
/// pieces that exist on the .NET side (no HF download, no pickle checkpoints).
/// </summary>
public sealed class JsonlFinetuner
{
    private readonly SimpleAttentionNetwork _model;
    private readonly TransformerConfig _modelConfig;
    private readonly NeedleTokenizer _tokenizer;
    private readonly JsonlFinetuneConfig _config;

    public JsonlFinetuner(
        SimpleAttentionNetwork model,
        TransformerConfig modelConfig,
        NeedleTokenizer tokenizer,
        JsonlFinetuneConfig config)
    {
        _model       = model;
        _modelConfig = modelConfig;
        _tokenizer   = tokenizer;
        _config      = config;
    }

    /// <summary>
    /// Run the full finetune loop over the JSONL file at
    /// <paramref name="jsonlPath"/>.
    /// </summary>
    public JsonlFinetuneResult Run(string jsonlPath)
    {
        var examples = JsonlDataset.Load(jsonlPath);
        if (examples.Count < 3)
            throw new InvalidOperationException($"Need at least 3 examples in {jsonlPath} (got {examples.Count}).");

        Log($"Loaded {examples.Count} examples from {jsonlPath}");

        var (train, val, test) = JsonlDataset.PerToolSplit(examples);
        if (train.Count == 0)
            throw new InvalidOperationException("Per-tool split produced 0 training examples.");

        Log($"Split: {train.Count} train / {val.Count} val / {test.Count} test (per-tool)");

        // ── Base-model evaluation on the held-out test set ──
        ToolCallMetrics? baseMetrics = null;
        if (test.Count > 0)
        {
            Log($"Evaluating base model on {test.Count} test examples...");
            baseMetrics = RunToolCallEval(test);
            if (baseMetrics is not null)
                Log($"  Base: call_f1={baseMetrics.CallF1:P1}, exact={baseMetrics.ExactMatch:P1}");
        }

        // ── Trainer ──
        int batchesPerEpoch = (train.Count + _config.BatchSize - 1) / _config.BatchSize;
        int totalSteps      = System.Math.Max(1, batchesPerEpoch * _config.Epochs);

        var trainerConfig = new TrainingConfig
        {
            Epochs       = _config.Epochs,
            BatchSize    = _config.BatchSize,
            AdamLr       = _config.AdamLr,
            MuonLr       = _config.MuonLr,
            WarmupRatio  = _config.WarmupRatio,
            DecayRatio   = _config.DecayRatio,
            GradClipNorm = _config.GradClipNorm,
            MaxEncLen    = _config.MaxEncLen,
            MaxDecLen    = _config.MaxDecLen,
            Seed         = _config.Seed,
            CheckpointDir = _config.CheckpointDir ?? "checkpoints",
        };

        using var trainer = new Trainer(_model, _modelConfig, trainerConfig, totalSteps);

        var rng = new Random(_config.Seed);

        ToolCallMetrics? bestVal = null;
        string? bestCkptPath = null;
        ToolCallMetrics? finetunedMetrics = null;

        for (int epoch = 0; epoch < _config.Epochs; epoch++)
        {
            // Shuffle each epoch
            Shuffle(train, rng);

            float epochLoss = 0;
            int   epochSteps = 0;
            foreach (var batch in BatchBuilder.Iterate(train, _tokenizer,
                                                       _config.BatchSize,
                                                       _config.MaxEncLen,
                                                       _config.MaxDecLen))
            {
                var (_, textLoss, _) = trainer.TrainStep(batch);
                epochLoss  += textLoss;
                epochSteps += 1;
            }

            float avgLoss = epochSteps > 0 ? epochLoss / epochSteps : float.NaN;
            Log($"Epoch {epoch + 1}/{_config.Epochs}: avg text_loss={avgLoss:F4}");

            // Validation: tool-call F1 on val split
            if (_config.EvalAfterEachEpoch && val.Count > 0)
            {
                var valMetrics = RunToolCallEval(val);
                if (valMetrics is not null)
                {
                    Log($"  Val: call_f1={valMetrics.CallF1:P1}, exact={valMetrics.ExactMatch:P1}");
                    if (bestVal is null || valMetrics.CallF1 > bestVal.CallF1)
                    {
                        bestVal = valMetrics;
                        bestCkptPath = SaveCheckpoint(epoch + 1);
                        Log($"  ** New best call_f1 → {bestCkptPath}");
                    }
                }
            }
        }

        if (bestCkptPath is null)
            bestCkptPath = SaveCheckpoint(_config.Epochs);

        // ── Finetuned-model evaluation on the test set ──
        if (test.Count > 0)
        {
            Log($"Evaluating finetuned model on {test.Count} test examples...");
            finetunedMetrics = RunToolCallEval(test);
            if (finetunedMetrics is not null)
                Log($"  Finetuned: call_f1={finetunedMetrics.CallF1:P1}, exact={finetunedMetrics.ExactMatch:P1}");
        }

        return new JsonlFinetuneResult(baseMetrics, finetunedMetrics, bestCkptPath);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private ToolCallMetrics? RunToolCallEval(IList<FinetuneExample> examples)
    {
        // Drop empty-answer examples (Python finetune does the same in _quick_tool_eval).
        var samples = examples
            .Where(e => e.Answers.Trim() is not "" and not "[]")
            .ToList();
        if (samples.Count == 0) return null;

        using var runner = new InferenceRunner(_model, _tokenizer, _modelConfig);
        const int BATCH = 32;

        var preds = new List<string>();
        for (int i = 0; i < samples.Count; i += BATCH)
        {
            int end = System.Math.Min(i + BATCH, samples.Count);
            var chunk = samples.GetRange(i, end - i);
            var output = runner.GenerateBatch(
                chunk.Select(s => s.Query).ToList(),
                chunk.Select(s => s.Tools).ToList(),
                maxGenLen: System.Math.Min(_config.MaxDecLen, _config.MaxGenLen),
                maxEncLen: _config.MaxEncLen,
                constrained: true);
            preds.AddRange(output);
        }

        return ToolCallEvaluator.Evaluate(
            samples.Select(s => s.Tools).ToList(),
            samples.Select(s => s.Answers).ToList(),
            preds);
    }

    private string SaveCheckpoint(int epoch)
    {
        string dir = _config.CheckpointDir ?? "checkpoints";
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, $"{_config.ExperimentName}_epoch{epoch}.ndlw");

        // Snapshot current parameters to CPU.
        var named = new Dictionary<string, Tensor>();
        foreach (var (name, param) in _model.named_parameters())
            named[name] = param.detach().cpu();

        try
        {
            WeightLoader.Save(named, path);
        }
        finally
        {
            foreach (var t in named.Values) t.Dispose();
        }
        return path;
    }

    private static void Shuffle<T>(IList<T> list, Random rng)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    private void Log(string message)
    {
        if (_config.Verbose)
            Console.WriteLine(message);
    }
}
