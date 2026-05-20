using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
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
          needle run           --checkpoint <path> [--tokenizer <path>] [--query <text>] [--tools <json>]
          needle eval          --checkpoint <path> [--tokenizer <path>] --jsonl <path>
          needle finetune      --checkpoint <path> [--tokenizer <path>] --jsonl <path> [--epochs N]
          needle export        --checkpoint <path> --factor N --output <path>
          needle dump-compare  --checkpoint <path> [--tokenizer <path>] --spec <path> --out <path>

        Notes:
          * --checkpoint should be a .ndlw or .safetensors file (pickle .pkl is
            not supported on the .NET port).
          * --tokenizer is optional: if omitted, the SentencePiece model
            embedded in the Needle assembly (Cactus-Compute/needle) is used.
            Pass an explicit .model file only when overriding it.
          * dump-compare reads a JSON spec and writes deterministic outputs
            (tokenization, generation, retrieval embeddings) for diffing
            against the Python reference.  See scripts/compare/README.md.

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
                "run"          => RunInference(rest),
                "eval"         => RunEval(rest),
                "finetune"     => RunFinetune(rest),
                "export"       => RunExport(rest),
                "dump-compare" => RunDumpCompare(rest),
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
        string tokPath    = opts.Get("tokenizer", "");
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
        string tokPath    = opts.Get("tokenizer", "");
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
        string tokPath    = opts.Get("tokenizer", "");
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

    // ── needle dump-compare ──────────────────────────────────────────────────

    /// <summary>
    /// Read a JSON spec describing test inputs (tokenization, generation,
    /// retrieval) and write a deterministic JSON dump of the C# outputs, for
    /// numerical diffing against the Python reference.  See
    /// scripts/compare/README.md.
    /// </summary>
    private static int RunDumpCompare(string[] args)
    {
        var opts = ArgParser.Parse(args);
        string checkpoint = opts.GetRequired("checkpoint");
        string tokPath    = opts.Get("tokenizer", "");
        string specPath   = opts.GetRequired("spec");
        string outPath    = opts.GetRequired("out");
        int floatPrec     = int.Parse(opts.Get("float-precision", "8"));

        if (!File.Exists(specPath))
            throw new FileNotFoundException($"Spec file not found: {specPath}", specPath);

        var specRoot = JsonNode.Parse(File.ReadAllText(specPath))
            ?? throw new InvalidOperationException("Empty or invalid spec JSON.");

        var tokenizer = LoadTokenizer(tokPath);
        var output    = new JsonObject();

        // ── Tokenization tests ──────────────────────────────────────────────
        if (specRoot["tokenize"] is JsonArray tokSpec)
        {
            var outArr = new JsonArray();
            foreach (var item in tokSpec)
            {
                string text = item?["text"]?.GetValue<string>()
                              ?? throw new InvalidOperationException("tokenize.text missing");
                var ids   = tokenizer.Encode(text);
                var entry = new JsonObject
                {
                    ["text"]    = text,
                    ["ids"]     = new JsonArray([.. ids.Select(i => JsonValue.Create(i))]),
                    ["decoded"] = tokenizer.Decode(ids),
                };
                outArr.Add(entry);
            }
            output["tokenize"] = outArr;
        }

        // ── Tool-name normalization (pure-function check, no model) ─────────
        if (specRoot["tool_normalize"] is JsonArray normSpec)
        {
            var outArr = new JsonArray();
            foreach (var item in normSpec)
            {
                string input = item?.GetValue<string>() ?? "";
                var (normalized, nameMap) = ToolNormalizer.NormalizeTools(input);
                var mapObj = new JsonObject();
                foreach (var (snake, orig) in nameMap)
                    mapObj[snake] = orig;
                outArr.Add(new JsonObject
                {
                    ["input"]    = input,
                    ["tools"]    = normalized,
                    ["name_map"] = mapObj,
                });
            }
            output["tool_normalize"] = outArr;
        }

        // Generation, retrieval, forward-loss, and quantize all need the
        // model; skip loading if none are requested (tokenizer-only diffs
        // don't need weights).
        bool needsModel =
            specRoot["generate"]         is JsonArray  ||
            specRoot["generate_batch"]   is JsonArray  ||
            specRoot["encode_retrieval"] is JsonObject ||
            specRoot["forward_loss"]     is JsonObject ||
            specRoot["quantize"]         is JsonObject;

        if (needsModel)
        {
            var (model, config) = LoadModel(checkpoint);
            try
            {
                using var runner = new InferenceRunner(model, tokenizer, config);

                // ── Generation tests ─────────────────────────────────────────
                if (specRoot["generate"] is JsonArray genSpec)
                {
                    var outArr = new JsonArray();
                    foreach (var item in genSpec)
                    {
                        if (item is not JsonObject obj)
                            throw new InvalidOperationException("generate items must be objects.");

                        string id          = obj["id"]?.GetValue<string>() ?? "";
                        string query       = obj["query"]?.GetValue<string>()
                                              ?? throw new InvalidOperationException("generate.query missing");
                        string tools       = obj["tools"]?.GetValue<string>() ?? "[]";
                        int    maxGen      = obj["max_gen_len"]?.GetValue<int>() ?? 128;
                        int    maxEnc      = obj["max_enc_len"]?.GetValue<int>() ?? 1024;
                        bool   constrained = obj["constrained"]?.GetValue<bool>() ?? false;
                        bool   normalize   = obj["normalize"]?.GetValue<bool>() ?? false;

                        // Capture the actual generated token IDs (in addition
                        // to the decoded text), so the comparator can diff
                        // them deterministically.
                        var pieces = new List<string>();
                        IProgress<string> sink = new Progress<string>(p => pieces.Add(p));

                        string text = runner.Generate(
                            query:      query,
                            tools:      tools,
                            maxGenLen:  maxGen,
                            maxEncLen:  maxEnc,
                            normalize:  normalize,
                            constrained: constrained,
                            progress:   sink);

                        // Re-encode the produced text to get the canonical ID
                        // list (mirrors what the Python comparator can produce).
                        var producedIds = tokenizer.Encode(text);

                        var entry = new JsonObject
                        {
                            ["id"]     = id,
                            ["query"]  = query,
                            ["text"]   = text,
                            ["ids"]    = new JsonArray([.. producedIds.Select(i => JsonValue.Create(i))]),
                            ["pieces"] = new JsonArray([.. pieces.Select(p => JsonValue.Create(p))]),
                        };
                        outArr.Add(entry);
                    }
                    output["generate"] = outArr;
                }

                // ── Batched generation ───────────────────────────────────────
                if (specRoot["generate_batch"] is JsonArray gbSpec)
                {
                    var outArr = new JsonArray();
                    foreach (var entry in gbSpec)
                    {
                        if (entry is not JsonObject obj)
                            throw new InvalidOperationException("generate_batch items must be objects.");
                        string id      = obj["id"]?.GetValue<string>() ?? "";
                        var items      = (obj["items"] as JsonArray)
                                          ?? throw new InvalidOperationException("generate_batch.items missing");
                        int maxGen     = obj["max_gen_len"]?.GetValue<int>() ?? 128;
                        int maxEnc     = obj["max_enc_len"]?.GetValue<int>() ?? 1024;
                        bool constrained = obj["constrained"]?.GetValue<bool>() ?? false;
                        bool normalize   = obj["normalize"]?.GetValue<bool>() ?? false;

                        var queries = items.Select(it => it?["query"]?.GetValue<string>() ?? "").ToList();
                        var tools   = items.Select(it => it?["tools"]?.GetValue<string>() ?? "[]").ToList();

                        var preds = runner.GenerateBatch(
                            queries, tools,
                            maxGenLen: maxGen,
                            maxEncLen: maxEnc,
                            normalize: normalize,
                            constrained: constrained);

                        var results = new JsonArray();
                        for (int k = 0; k < queries.Count; k++)
                        {
                            var ids = tokenizer.Encode(preds[k]);
                            results.Add(new JsonObject
                            {
                                ["query"] = queries[k],
                                ["tools"] = tools[k],
                                ["text"]  = preds[k],
                                ["ids"]   = new JsonArray([.. ids.Select(i => JsonValue.Create(i))]),
                            });
                        }
                        outArr.Add(new JsonObject { ["id"] = id, ["results"] = results });
                    }
                    output["generate_batch"] = outArr;
                }

                // ── Retrieval-embedding tests ────────────────────────────────
                if (specRoot["encode_retrieval"] is JsonObject retSpec)
                {
                    int maxLen = retSpec["max_len"]?.GetValue<int>() ?? 256;
                    var texts  = (retSpec["texts"] as JsonArray)
                                 ?? throw new InvalidOperationException("encode_retrieval.texts missing");
                    var textList = texts.Select(t => t?.GetValue<string>() ?? "").ToList();

                    float[,] embs = runner.EncodeForRetrieval(textList, maxLen: maxLen);
                    int N = embs.GetLength(0);
                    int D = embs.GetLength(1);

                    var embArr = new JsonArray();
                    string fmt = $"F{floatPrec}";
                    for (int i = 0; i < N; i++)
                    {
                        var row = new JsonArray();
                        for (int j = 0; j < D; j++)
                            row.Add(JsonValue.Create(SafeRound(embs[i, j], floatPrec)));
                        embArr.Add(row);
                    }

                    output["encode_retrieval"] = new JsonObject
                    {
                        ["shape"]      = new JsonArray(N, D),
                        ["texts"]      = new JsonArray([.. textList.Select(t => JsonValue.Create(t))]),
                        ["embeddings"] = embArr,
                    };
                }

                // ── Forward + loss (training-math parity) ────────────────────
                if (specRoot["forward_loss"] is JsonObject flSpec)
                {
                    var (lossOut, _) = DumpForwardLoss(model, flSpec, floatPrec);
                    output["forward_loss"] = lossOut;
                }

                // ── Quantization parity ──────────────────────────────────────
                if (specRoot["quantize"] is JsonObject qSpec)
                {
                    output["quantize"] = DumpQuantize(model, qSpec, floatPrec);
                }
            }
            finally
            {
                (model as IDisposable)?.Dispose();
            }
        }

        // Header so the comparator can sanity-check the side it's reading.
        output["meta"] = new JsonObject
        {
            ["side"]            = "csharp",
            ["checkpoint"]      = checkpoint,
            ["tokenizer"]       = tokPath,
            ["float_precision"] = floatPrec,
        };

        tokenizer.Dispose();

        var serializerOpts = new JsonSerializerOptions
        {
            WriteIndented = true,
            Encoder       = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        };
        File.WriteAllText(outPath, output.ToJsonString(serializerOpts));
        Console.WriteLine($"wrote {outPath}");
        return 0;
    }

    /// <summary>
    /// Round a float for JSON output, replacing non-finite values with 0 so
    /// JsonValue.Create doesn't throw on Inf/NaN.  Drift from the Python side
    /// will surface as a tolerance failure in the comparator.
    /// </summary>
    private static double SafeRound(double v, int digits)
    {
        if (double.IsNaN(v) || double.IsInfinity(v)) return 0.0;
        return System.Math.Round(v, digits);
    }

    // ── Forward-loss dump (training-math parity) ──────────────────────────────

    private static (JsonObject Result, Tensor Logits) DumpForwardLoss(
        SimpleAttentionNetwork model,
        JsonObject spec,
        int floatPrec)
    {
        model.eval();
        using var noGrad = torch.no_grad();

        long[,] LoadInts(string key)
        {
            var rows = spec[key] as JsonArray
                ?? throw new InvalidOperationException($"forward_loss.{key} missing");
            int B = rows.Count;
            int T = (rows[0] as JsonArray)?.Count ?? 0;
            var arr = new long[B, T];
            for (int b = 0; b < B; b++)
            {
                var r = rows[b] as JsonArray
                    ?? throw new InvalidOperationException($"forward_loss.{key}[{b}] not array");
                for (int t = 0; t < T; t++) arr[b, t] = r[t]?.GetValue<long>() ?? 0L;
            }
            return arr;
        }

        long[,] src    = LoadInts("src_tokens");
        long[,] tgtIn  = LoadInts("tgt_in_tokens");
        long[,] tgtOut = LoadInts("tgt_out_tokens");
        long[,] lossM  = LoadInts("loss_mask");
        long[,] encSeg = LoadInts("enc_seg_ids");
        long[,] decSeg = LoadInts("dec_seg_ids");

        long B = src.GetLength(0);
        long Te = src.GetLength(1);
        long Td = tgtIn.GetLength(1);

        using var srcT    = torch.tensor(src.Cast<long>().ToArray(),    new long[] { B, Te });
        using var tgtInT  = torch.tensor(tgtIn.Cast<long>().ToArray(),  new long[] { B, Td });
        using var tgtOutT = torch.tensor(tgtOut.Cast<long>().ToArray(), new long[] { B, Td });

        // The training mask helpers expect int32 segment IDs.
        using var encSegT = torch.tensor(encSeg.Cast<long>().Select(x => (int)x).ToArray(),
                                         new long[] { B, Te }, dtype: ScalarType.Int32);
        using var decSegT = torch.tensor(decSeg.Cast<long>().Select(x => (int)x).ToArray(),
                                         new long[] { B, Td }, dtype: ScalarType.Int32);
        using var lossMT  = torch.tensor(lossM.Cast<long>().Select(x => (int)x).ToArray(),
                                         new long[] { B, Td }, dtype: ScalarType.Int32);

        using var srcMask   = MaskUtils.MakePackingMask(encSegT);
        using var tgtMask   = MaskUtils.MakeCausalPackingMask(decSegT);
        using var crossMask = MaskUtils.MakeCrossPackingMask(encSegT, decSegT);

        var logits = model.Forward(srcT, tgtInT, srcMask, tgtMask, crossMask);

        // Token-class weights match the trainer defaults (base, name, value, key).
        var weightMap = LossFunctions.DefaultTokenWeightMap;
        using var tokenWeights = LossFunctions.BuildTokenWeights(lossMT, weightMap);
        using var textLoss     = LossFunctions.TextLoss(logits, tgtOutT, tokenWeights, decSegT);
        using var zLoss        = LossFunctions.ZLoss(logits);
        using var totalLoss    = textLoss + zLoss;

        float ce = textLoss.item<float>();
        float zl = zLoss.item<float>();

        // Probe: first 8 vocab logits at every (batch, position).  Lets the
        // comparator diff numerically rather than just a single scalar.
        long V = logits.shape[2];
        long probeWidth = System.Math.Min(8L, V);
        using var probeT = logits.index(TensorIndex.Ellipsis,
                                       TensorIndex.Slice(stop: probeWidth));
        var flat = probeT.to(ScalarType.Float32).contiguous().data<float>().ToArray();
        var probe = new JsonArray();
        for (long b = 0; b < B; b++)
        {
            for (long t = 0; t < Td; t++)
            {
                var row = new JsonArray();
                long baseIdx = (b * Td + t) * probeWidth;
                for (long k = 0; k < probeWidth; k++)
                    row.Add(JsonValue.Create(SafeRound(flat[baseIdx + k], floatPrec)));
                probe.Add(row);
            }
        }

        var result = new JsonObject
        {
            ["shape"]        = new JsonArray(B, Td, V),
            ["ce_loss"]      = JsonValue.Create(SafeRound((double)ce,      floatPrec)),
            ["z_loss"]       = JsonValue.Create(SafeRound((double)zl,      floatPrec)),
            ["total_loss"]   = JsonValue.Create(SafeRound((double)(ce + zl), floatPrec)),
            ["logits_probe"] = probe,
        };
        return (result, logits);
    }

    // ── Quantization dump ─────────────────────────────────────────────────────

    private static JsonObject DumpQuantize(
        SimpleAttentionNetwork model,
        JsonObject spec,
        int floatPrec)
    {
        string key   = spec["tensor_key"]?.GetValue<string>()
                       ?? throw new InvalidOperationException("quantize.tensor_key missing");
        var rows = spec["rows"] as JsonArray;
        var cols = spec["cols"] as JsonArray;
        int r0 = rows?[0]?.GetValue<int>() ?? 0;
        int r1 = rows?[1]?.GetValue<int>() ?? 32;
        int c0 = cols?[0]?.GetValue<int>() ?? 0;
        int c1 = cols?[1]?.GetValue<int>() ?? 32;
        int gs = spec["group_size"]?.GetValue<int>() ?? 32;
        string prec = spec["precision"]?.GetValue<string>() ?? "int4";

        using var noGrad = torch.no_grad();

        // Find the parameter by name.
        Tensor? tensor = null;
        foreach (var (name, p) in model.named_parameters())
        {
            if (name == key) { tensor = p; break; }
        }
        if (tensor is null)
            throw new InvalidOperationException($"quantize: parameter '{key}' not in model.");

        using var sub = tensor[TensorIndex.Slice(r0, r1), TensorIndex.Slice(c0, c1)]
                        .to(ScalarType.Float32)
                        .contiguous();

        using var quant = prec == "int8"
            ? Quantize.FakeQuantizeInt8(sub, gs)
            : Quantize.FakeQuantizeInt4(sub, gs);

        int H = (int)quant.shape[0], W = (int)quant.shape[1];
        var flat = quant.to(ScalarType.Float32).contiguous().data<float>().ToArray();
        var values = new JsonArray();
        for (int i = 0; i < H; i++)
        {
            var row = new JsonArray();
            for (int j = 0; j < W; j++)
                row.Add(JsonValue.Create(SafeRound(flat[i * W + j], floatPrec)));
            values.Add(row);
        }

        return new JsonObject
        {
            ["tensor_key"] = key,
            ["rows"]       = new JsonArray(r0, r1),
            ["cols"]       = new JsonArray(c0, c1),
            ["group_size"] = gs,
            ["precision"]  = prec,
            ["shape"]      = new JsonArray(H, W),
            ["values"]     = values,
        };
    }

    // ── Loading helpers ──────────────────────────────────────────────────────

    /// <summary>
    /// Load a SentencePiece tokenizer.  An empty/null <paramref name="path"/>
    /// uses the model embedded in the Needle assembly (no file required).
    /// </summary>
    private static NeedleTokenizer LoadTokenizer(string? path)
    {
        if (string.IsNullOrEmpty(path))
            return NeedleTokenizer.LoadDefault();
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
        // Accept both the historical `<stack>.layer_<n>` naming and the
        // TorchSharp ModuleList naming `<stack>._layers.<n>` that the actual
        // model emits via named_parameters().
        string[] prefixes = [$"{stack}._layers.", $"{stack}.layer_"];
        int max = -1;
        foreach (var name in tensors.Keys)
        {
            foreach (var prefix in prefixes)
            {
                int idx = name.IndexOf(prefix, StringComparison.Ordinal);
                if (idx < 0) continue;
                int start = idx + prefix.Length;
                int end   = start;
                while (end < name.Length && char.IsDigit(name[end])) end++;
                if (end > start && int.TryParse(name.AsSpan(start, end - start), out int n))
                    if (n > max) max = n;
                break;
            }
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
