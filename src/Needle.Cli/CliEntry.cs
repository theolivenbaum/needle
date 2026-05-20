using Needle.Inference;
using Needle.Model;
using Needle.Tokenizer;
using Needle.Training;
using Needle.Weights;
using TorchSharp;
using static TorchSharp.torch;

namespace Needle.Cli;

/// <summary>
/// Console entry point for the .NET port of <c>needle</c>.
///
/// Subset of the Python CLI from <c>needle/cli.py</c>: only the inference,
/// evaluation, and local finetune commands are wired up — the dataset
/// preparation, distributed-training, and TPU-management commands depend on
/// Python-only infrastructure (HuggingFace datasets, Gemini, gcloud) and are
/// out of scope on .NET.
/// </summary>
public static class CliEntry
{
    private const string Help = """
        needle — Simple Attention Network for on-device function calling

        Usage:
          needle run       --checkpoint <path> --tokenizer <path> [--query <text>] [--tools <json>]
          needle eval      --checkpoint <path> --tokenizer <path> --jsonl <path>
          needle finetune  --checkpoint <path> --tokenizer <path> --jsonl <path> [--epochs N]
          needle export    --checkpoint <path> --factor N --output <path>

        Notes:
          * --checkpoint should be a .ndlw or .safetensors file (pickle .pkl is
            not supported on the .NET port).
          * --tokenizer should point to a SentencePiece .model file matching the
            checkpoint.

        For full documentation see README.md.
        """;

    public static int Run(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            Console.WriteLine(Help);
            return 0;
        }

        var rest = args.Skip(1).ToArray();
        try
        {
            return args[0] switch
            {
                "run"      => RunInference(rest),
                "eval"     => RunEval(rest),
                "finetune" => RunFinetune(rest),
                "export"   => RunExport(rest),
                _ => UnknownCommand(args[0]),
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"error: {ex.Message}");
            return 1;
        }
    }

    private static int UnknownCommand(string cmd)
    {
        Console.Error.WriteLine($"Unknown command: {cmd}");
        Console.Error.WriteLine(Help);
        return 2;
    }

    // ── needle run ───────────────────────────────────────────────────────────

    private static int RunInference(string[] args)
    {
        var opts = ArgParser.Parse(args);
        string checkpoint = opts.GetRequired("checkpoint");
        string tokPath    = opts.GetRequired("tokenizer");
        string query      = opts.Get("query", "What is the weather in San Francisco?");
        string tools      = opts.Get("tools", """[{"name":"get_weather","parameters":{"location":"string"}}]""");
        int maxLen        = int.Parse(opts.Get("max-len", "512"));
        bool noConstrain  = opts.GetFlag("no-constrained");

        var tokenizer = LoadTokenizer(tokPath);
        var (model, config) = LoadModel(checkpoint);

        try
        {
            using var runner = new InferenceRunner(model, tokenizer, config);
            string result = runner.Generate(
                query: query,
                tools: tools,
                maxGenLen: maxLen,
                constrained: !noConstrain);
            Console.WriteLine(result);
        }
        finally
        {
            (model as IDisposable)?.Dispose();
            tokenizer.Dispose();
        }
        return 0;
    }

    // ── needle eval ──────────────────────────────────────────────────────────

    private static int RunEval(string[] args)
    {
        var opts = ArgParser.Parse(args);
        string checkpoint = opts.GetRequired("checkpoint");
        string tokPath    = opts.GetRequired("tokenizer");
        string jsonlPath  = opts.GetRequired("jsonl");
        int maxGenLen     = int.Parse(opts.Get("max-gen-len", "512"));
        int maxEncLen     = int.Parse(opts.Get("max-enc-len", "1024"));
        bool noConstrain  = opts.GetFlag("no-constrained");

        var examples = JsonlDataset.Load(jsonlPath);
        if (examples.Count == 0)
            throw new InvalidOperationException($"No examples in {jsonlPath}");

        var tokenizer = LoadTokenizer(tokPath);
        var (model, config) = LoadModel(checkpoint);

        try
        {
            using var runner = new InferenceRunner(model, tokenizer, config);

            const int BATCH = 32;
            var preds = new List<string>(examples.Count);
            for (int i = 0; i < examples.Count; i += BATCH)
            {
                int end = System.Math.Min(i + BATCH, examples.Count);
                var chunk = examples.GetRange(i, end - i);
                preds.AddRange(runner.GenerateBatch(
                    chunk.Select(e => e.Query).ToList(),
                    chunk.Select(e => e.Tools).ToList(),
                    maxGenLen: maxGenLen,
                    maxEncLen: maxEncLen,
                    constrained: !noConstrain));
            }

            var m = ToolCallEvaluator.Evaluate(
                examples.Select(e => e.Tools).ToList(),
                examples.Select(e => e.Answers).ToList(),
                preds);

            Console.WriteLine($"  N            {m.N,12}");
            Console.WriteLine($"  parse_rate   {m.ParseRate,12:P1}");
            Console.WriteLine($"  exact_match  {m.ExactMatch,12:P1}");
            Console.WriteLine($"  name_f1      {m.NameF1,12:P1}");
            Console.WriteLine($"  call_f1      {m.CallF1,12:P1}");
            Console.WriteLine($"  args_acc     {m.ArgsAcc,12:P1}");
            Console.WriteLine($"  param_haluc  {m.ParamHaluc,12:P1}");
            Console.WriteLine($"  param_miss   {m.ParamMiss,12:P1}");
            Console.WriteLine($"  value_acc    {m.ValueAcc,12:P1}");
        }
        finally
        {
            (model as IDisposable)?.Dispose();
            tokenizer.Dispose();
        }
        return 0;
    }

    // ── needle finetune ──────────────────────────────────────────────────────

    private static int RunFinetune(string[] args)
    {
        var opts = ArgParser.Parse(args);
        string checkpoint = opts.GetRequired("checkpoint");
        string tokPath    = opts.GetRequired("tokenizer");
        string jsonlPath  = opts.GetRequired("jsonl");
        int epochs        = int.Parse(opts.Get("epochs",     "1"));
        int batchSize     = int.Parse(opts.Get("batch-size", "8"));
        string ckptDir    = opts.Get("checkpoint-dir", "checkpoints");
        string expName    = opts.Get("name", "needle_finetuned");

        var tokenizer = LoadTokenizer(tokPath);
        var (model, config) = LoadModel(checkpoint);

        try
        {
            var ftConfig = new JsonlFinetuneConfig
            {
                Epochs         = epochs,
                BatchSize      = batchSize,
                CheckpointDir  = ckptDir,
                ExperimentName = expName,
            };

            var finetuner = new JsonlFinetuner(model, config, tokenizer, ftConfig);
            var result    = finetuner.Run(jsonlPath);

            Console.WriteLine();
            Console.WriteLine("─── Result ──────────────");
            if (result.BaseMetrics is not null)
                Console.WriteLine($"  base call_f1       {result.BaseMetrics.CallF1,12:P1}");
            if (result.FinetunedMetrics is not null)
                Console.WriteLine($"  finetuned call_f1  {result.FinetunedMetrics.CallF1,12:P1}");
            Console.WriteLine($"  best checkpoint:   {result.BestCheckpointPath}");
        }
        finally
        {
            (model as IDisposable)?.Dispose();
            tokenizer.Dispose();
        }
        return 0;
    }

    // ── needle export ────────────────────────────────────────────────────────

    private static int RunExport(string[] args)
    {
        var opts = ArgParser.Parse(args);
        string checkpoint = opts.GetRequired("checkpoint");
        int factor        = int.Parse(opts.GetRequired("factor"));
        string output     = opts.GetRequired("output");

        var (_, tensors) = WeightLoader.Load(checkpoint);

        try
        {
            // Infer config dimensions from the embedding tensor.
            if (!tensors.TryGetValue("embedding.weight", out var embed))
                throw new InvalidOperationException(
                    "Checkpoint missing 'embedding.weight' — cannot infer config.");

            int vocabSize = (int)embed.shape[0];
            int dModel    = (int)embed.shape[1];

            int dFf = InferDFf(tensors);
            var config = new TransformerConfig { VocabSize = vocabSize, DModel = dModel, DFf = dFf };

            var (sliced, newConfig) = SubmodelExport.SliceParams(tensors, config, factor);
            try
            {
                WeightLoader.Save(sliced, output);
                Console.WriteLine($"Exported to {output} (dFf {config.DFf} → {newConfig.DFf})");
            }
            finally
            {
                foreach (var t in sliced.Values) t.Dispose();
            }
        }
        finally
        {
            foreach (var t in tensors.Values) t.Dispose();
        }
        return 0;
    }

    private static int InferDFf(IReadOnlyDictionary<string, Tensor> tensors)
    {
        foreach (var (name, t) in tensors)
        {
            if (name.Contains("gate_proj.weight", StringComparison.Ordinal) && t.dim() >= 2)
                return (int)t.shape[^1];
        }
        throw new InvalidOperationException(
            "Could not infer DFf — no gate_proj.weight tensor found in checkpoint.");
    }

    // ── Loading helpers ──────────────────────────────────────────────────────

    private static NeedleTokenizer LoadTokenizer(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Tokenizer model not found: {path}", path);
        return new NeedleTokenizer(path);
    }

    private static (SimpleAttentionNetwork model, TransformerConfig config) LoadModel(string path)
    {
        // Decide format by extension or by checking the magic header.
        Dictionary<string, Tensor> tensors;
        if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
            (_, tensors) = WeightLoader.LoadSafetensors(path);
        else
            (_, tensors) = WeightLoader.Load(path);

        try
        {
            if (!tensors.TryGetValue("embedding.weight", out var embed))
                throw new InvalidOperationException(
                    "Checkpoint missing 'embedding.weight' — cannot infer config.");

            int vocabSize = (int)embed.shape[0];
            int dModel    = (int)embed.shape[1];

            var config = new TransformerConfig
            {
                VocabSize        = vocabSize,
                DModel           = dModel,
                NumHeads         = InferNumHeads(dModel),
                NumKvHeads       = InferNumHeads(dModel) / 2,
                NumEncoderLayers = CountLayers(tensors, "encoder"),
                NumDecoderLayers = CountLayers(tensors, "decoder"),
                DFf              = TryInferDFf(tensors) ?? dModel * 4,
                MaxSeqLen        = 1024,
                ContrastiveDim   = 128,
                NoFeedforward    = !tensors.Keys.Any(k => k.Contains("gate_proj")),
            };

            var model = new SimpleAttentionNetwork(config);
            // Best-effort weight transfer.  Unknown keys are silently skipped;
            // mismatched shapes log a warning.
            LoadIntoModel(model, tensors);
            return (model, config);
        }
        finally
        {
            foreach (var t in tensors.Values) t.Dispose();
        }
    }

    private static int InferNumHeads(int dModel)
    {
        // The published config uses 8 heads for d=512.  Fall back to a sensible
        // divisor of dModel.
        if (dModel % 8 == 0) return 8;
        if (dModel % 4 == 0) return 4;
        if (dModel % 2 == 0) return 2;
        return 1;
    }

    private static int CountLayers(IReadOnlyDictionary<string, Tensor> tensors, string stack)
    {
        int max = -1;
        foreach (var name in tensors.Keys)
        {
            int idx = name.IndexOf($"{stack}.layer_", StringComparison.Ordinal);
            if (idx < 0) continue;
            int start = idx + $"{stack}.layer_".Length;
            int end   = start;
            while (end < name.Length && char.IsDigit(name[end])) end++;
            if (end > start && int.TryParse(name.AsSpan(start, end - start), out int n))
                if (n > max) max = n;
        }
        return System.Math.Max(1, max + 1);
    }

    private static int? TryInferDFf(IReadOnlyDictionary<string, Tensor> tensors)
    {
        foreach (var (name, t) in tensors)
        {
            if (name.Contains("gate_proj.weight", StringComparison.Ordinal) && t.dim() >= 2)
                return (int)t.shape[^1];
        }
        return null;
    }

    private static void LoadIntoModel(SimpleAttentionNetwork model, IReadOnlyDictionary<string, Tensor> tensors)
    {
        var loaded = 0;
        var skipped = 0;
        var modelParams = model.named_parameters().ToDictionary(p => p.name, p => p.parameter);

        using var noGrad = torch.no_grad();
        foreach (var (name, src) in tensors)
        {
            if (!modelParams.TryGetValue(name, out var dst))
            {
                skipped++;
                continue;
            }
            if (!src.shape.SequenceEqual(dst.shape))
            {
                Console.Error.WriteLine($"  shape mismatch for {name}: ckpt={string.Join('x', src.shape)} model={string.Join('x', dst.shape)}");
                skipped++;
                continue;
            }
            dst.copy_(src.to(dst.dtype));
            loaded++;
        }

        Console.WriteLine($"  loaded {loaded} tensor(s), skipped {skipped}");
    }
}
