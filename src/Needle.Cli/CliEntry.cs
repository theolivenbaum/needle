using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using Needle.Diagnostics;
using Needle.Inference;
using Needle.Math;
using Needle.Model;
using Needle.Tokenizer;
using Needle.Weights;

namespace Needle.Cli;

/// <summary>
/// Console entry point for the .NET port of Needle 2.
/// </summary>
public static class CliEntry
{
    private const string Help = """
        needle — Needle 2 Simple Attention Network, pure .NET

        Usage:
          needle parity     --fixtures <dir> [--tolerance 0.01] [--verbose]
          needle cact-check --cact <file> --expected <dir> [--tolerance 1e-4] [--verbose]
          needle info       --weights <dir|file|.cact> [--config <path>]
          needle run        --weights <dir|file|.cact> [--config <path>] --tokens 2,100 [--max-new 32]
          needle call       --weights <file.cact> --query <text> [--tools <json|@file>] [--system <text>]

        Commands:
          parity      Compare this implementation against the JAX reference, stage
                      by stage, using fixtures from
                      scripts/parity/dump_reference.py.
          cact-check  Compare this reader's dequantisation of a .cact deployment
                      blob against the reference's own read_export, using the dump
                      from scripts/parity/dump_cact.py.
          info        Print the geometry, parameter count and KV budget.
          run         Greedy-decode from raw token IDs with the KV-cached session.

        Weights may be a .cact blob, a directory holding weights.safetensors +
        config.json, or a .safetensors file (pass --config alongside it).
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
                "parity" => RunParity(rest),
                "cact-check" => RunCactCheck(rest),
                "info" => RunInfo(rest),
                "run" => RunGenerate(rest),
                "call" => RunCall(rest),
                _ => Unknown(args[0]),
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"error: {ex.Message}");
            return 1;
        }
    }

    private static int Unknown(string command)
    {
        Console.Error.WriteLine($"Unknown command: {command}");
        Console.Error.WriteLine(Help);
        return 2;
    }

    // ── parity ───────────────────────────────────────────────────────────────

    private static int RunParity(string[] args)
    {
        var options = ArgParser.Parse(args);
        string fixtures = options.Get("fixtures", "fixtures/parity");
        float tolerance = float.Parse(options.Get("tolerance", "0.01"));
        bool verbose = options.GetFlag("verbose");

        if (!ParityHarness.IsAvailable(fixtures))
        {
            Console.Error.WriteLine($"No parity fixtures in '{fixtures}'.");
            Console.Error.WriteLine("Generate them with:");
            Console.Error.WriteLine($"  python3 scripts/parity/dump_reference.py --out {fixtures}");
            return 1;
        }

        var stopwatch = Stopwatch.StartNew();
        var model = ParityHarness.LoadModel(fixtures);
        Console.WriteLine($"loaded {model.Weights.ParameterCount / 1e6:F1}M parameters "
                          + $"in {stopwatch.ElapsedMilliseconds} ms");
        Describe(model.Config);
        Console.WriteLine();

        bool allPassed = true;
        foreach (string casePath in ParityHarness.CaseFiles(fixtures))
        {
            var report = ParityHarness.RunCase(model, casePath);
            bool passed = report.Passed(tolerance);
            allPassed &= passed;

            Console.WriteLine($"case '{report.Case}': {(passed ? "PASS" : "FAIL")}  "
                              + $"worst relative error {report.WorstRelative:G4}, "
                              + $"argmax {report.ArgmaxMatches}/{report.ArgmaxTotal}, "
                              + $"decode tokens {(report.DecodeTokensMatch ? "match" : "differ")}");

            foreach (var delta in report.Deltas)
                if (verbose || !delta.Within(tolerance))
                    Console.WriteLine("  " + delta);
        }

        Console.WriteLine();
        Console.WriteLine(allPassed
            ? $"all cases within {tolerance:G3} relative error"
            : $"one or more cases exceeded {tolerance:G3} relative error");
        return allPassed ? 0 : 1;
    }

    // ── cact-check ───────────────────────────────────────────────────────────

    private static int RunCactCheck(string[] args)
    {
        var options = ArgParser.Parse(args);
        string cact = options.GetRequired("cact");
        string expected = options.GetRequired("expected");
        float tolerance = float.Parse(options.Get("tolerance", "1e-4"));
        bool verbose = options.GetFlag("verbose");

        var checks = CactCheck.Run(cact, expected);
        var byWidth = new SortedDictionary<int, (int Count, float Worst)>();

        foreach (var check in checks)
        {
            if (verbose)
                Console.WriteLine($"  t{check.Index:D4} {check.Dtype,-9} bits={check.Bits,-2} "
                                  + $"{string.Join("x", check.Shape),-16} {check.Delta}");

            int width = check.Dtype == CactDtype.Quantized ? check.Bits : 0;
            var entry = byWidth.GetValueOrDefault(width);
            byWidth[width] = (entry.Count + 1, System.Math.Max(entry.Worst, check.Delta.MaxRelative));
        }

        Console.WriteLine($"compared {checks.Count} tensors from {Path.GetFileName(cact)}");
        foreach (var (width, entry) in byWidth)
        {
            string label = width == 0 ? "fp16" : width == 5 ? "ternary" : $"CQ{width}";
            Console.WriteLine($"  {label,-8} {entry.Count,4} tensors, worst relative error {entry.Worst:G4}");
        }

        float worst = checks.Count == 0 ? 0f : checks.Max(c => c.Delta.MaxRelative);
        bool passed = worst <= tolerance;
        Console.WriteLine(passed
            ? $"all tensors within {tolerance:G3} relative error"
            : $"worst relative error {worst:G4} exceeds {tolerance:G3}");

        string tokenizerJson = Path.Combine(expected, "tokenizer.json");
        if (File.Exists(tokenizerJson))
        {
            var tokenizer = CactTokenizer.FromCact(cact);
            ChatMarkers.Validate(tokenizer);
            var report = TokenizerCheck.Run(tokenizer, tokenizerJson);

            Console.WriteLine($"tokenizer: {tokenizer.VocabSize} pieces, "
                              + $"{report.Cases.Count(c => c.EncodeMatches && c.DecodeMatches)}"
                              + $"/{report.Cases.Count} cases round-trip identically");
            foreach (string mismatch in report.PieceMismatches)
                Console.WriteLine("  piece mismatch: " + mismatch);
            foreach (var result in report.Cases.Where(c => !c.EncodeMatches || !c.DecodeMatches))
            {
                Console.WriteLine($"  mismatch at token {result.FirstDivergence} for "
                                  + $"{JsonSerializer.Serialize(result.Text)}");
                Console.WriteLine($"    reference: [{string.Join(",", result.Expected)}] -> "
                                  + JsonSerializer.Serialize(result.ExpectedDecoded));
                Console.WriteLine($"    port:      [{string.Join(",", result.Actual)}] -> "
                                  + JsonSerializer.Serialize(result.ActualDecoded));
            }
            passed &= report.Passed;
        }

        return passed ? 0 : 1;
    }

    // ── info ─────────────────────────────────────────────────────────────────

    private static int RunInfo(string[] args)
    {
        var options = ArgParser.Parse(args);
        var model = LoadModel(options);
        Describe(model.Config);
        Console.WriteLine($"  parameters       {model.Weights.ParameterCount / 1e6:F2}M");
        Console.WriteLine($"  heads present    "
                          + $"mtp={(model.Weights.Mtp is not null ? "yes" : "no")} "
                          + $"contrastive={(model.Weights.Contrastive is not null ? "yes" : "no")} "
                          + $"confidence={(model.Weights.Confidence is not null ? "yes" : "no")}");
        return 0;
    }

    private static void Describe(TransformerConfig config)
    {
        Console.WriteLine($"  vocab            {config.VocabSize}");
        Console.WriteLine($"  d_model          {config.DModel} (attn {config.AttnWidth})");
        Console.WriteLine($"  layers           {config.NumLayers}");
        Console.WriteLine($"  heads            {config.NumHeads} q / {config.NumKvHeads} kv "
                          + $"(head dim {config.HeadDim})");
        Console.WriteLine($"  mhc lanes        {config.MhcLanes}");
        Console.WriteLine($"  engram           sites [{string.Join(", ", config.EngramLayers)}], "
                          + $"orders [{string.Join(", ", config.EngramOrders)}], "
                          + $"{config.EngramSlots} slots x {config.EngramTables} tables");
        Console.WriteLine($"  kv window        {KvBudget.EffectiveWindow(config)} "
                          + $"(budget {KvBudget.BudgetWindow(config)}, configured {config.KvWindow})");
    }

    // ── run ──────────────────────────────────────────────────────────────────

    private static int RunGenerate(string[] args)
    {
        var options = ArgParser.Parse(args);
        var model = LoadModel(options);

        var prompt = options.GetRequired("tokens")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(int.Parse)
            .ToArray();
        int maxNew = int.Parse(options.Get("max-new", "32"));

        var session = new NeedleSession(model);
        var stopwatch = Stopwatch.StartNew();
        var generated = session.Generate(prompt, maxNew);
        stopwatch.Stop();

        Console.WriteLine(string.Join(",", generated));
        Console.Error.WriteLine($"{generated.Count} tokens in {stopwatch.ElapsedMilliseconds} ms "
                                + $"({generated.Count * 1000.0 / System.Math.Max(1, stopwatch.ElapsedMilliseconds):F1} tok/s)");
        return 0;
    }

    // ── call ─────────────────────────────────────────────────────────────────

    private static int RunCall(string[] args)
    {
        var options = ArgParser.Parse(args);
        string weightsPath = options.GetRequired("weights");
        var model = LoadModel(options);
        var tokenizer = CactTokenizer.FromCact(weightsPath);

        string tools = ReadInline(options.Get("tools", "[]"));
        string query = options.GetRequired("query");
        string? system = options.GetOrNull("system");
        int maxNew = int.Parse(options.Get("max-new", "256"));

        var agent = new NeedleAgent(model, tokenizer, tools, system);
        var stopwatch = Stopwatch.StartNew();
        var response = agent.Complete(query, maxNew);
        stopwatch.Stop();

        var payload = new JsonObject
        {
            ["type"] = response.Type,
            ["success"] = response.Success,
            ["error"] = response.Error,
            ["function_calls"] = new JsonArray(response.FunctionCalls
                .Select(c => (JsonNode)new JsonObject
                {
                    ["name"] = c.Name,
                    ["arguments"] = c.Arguments.DeepClone(),
                }).ToArray()),
            ["reasoning"] = response.Reasoning,
            ["confidence"] = System.Math.Round(response.Confidence, 4),
            ["prompt_tokens"] = response.PromptTokens,
            ["generated_tokens"] = response.GeneratedTokens,
        };

        Console.WriteLine(payload.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        Console.Error.WriteLine(
            $"{response.PromptTokens} prompt + {response.GeneratedTokens} generated tokens "
            + $"in {stopwatch.ElapsedMilliseconds} ms");
        if (options.GetFlag("raw")) Console.Error.WriteLine("raw: " + response.Text);
        return 0;
    }

    /// <summary>Reads a value that may be given inline or as <c>@path</c>.</summary>
    private static string ReadInline(string value) =>
        value.StartsWith('@') ? File.ReadAllText(value[1..]) : value;

    // ── shared ───────────────────────────────────────────────────────────────

    private static Needle2Model LoadModel(ArgParser options)
    {
        string weights = options.GetRequired("weights");

        if (Directory.Exists(weights)) return ParityHarness.LoadModel(weights);

        if (weights.EndsWith(".cact", StringComparison.OrdinalIgnoreCase))
        {
            var template = options.GetOrNull("config") is { } configFile
                ? CheckpointConfig.Load(configFile)
                : new TransformerConfig();
            var loaded = CactLayout.Load(weights, template);
            return new Needle2Model(Needle2Weights.FromFlat(loaded.Config, loaded.Parameters));
        }

        string configPath = options.GetOrNull("config")
            ?? Path.Combine(Path.GetDirectoryName(Path.GetFullPath(weights)) ?? ".", "config.json");
        if (!File.Exists(configPath))
            throw new FileNotFoundException(
                $"No config found for '{weights}'. Pass --config <path>.", configPath);

        var config = CheckpointConfig.Load(configPath);
        var flat = Safetensors.Load(weights);
        return new Needle2Model(Needle2Weights.FromFlat(config, flat));
    }
}
