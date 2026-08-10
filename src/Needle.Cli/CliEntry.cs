using System.Diagnostics;
using Needle.Diagnostics;
using Needle.Inference;
using Needle.Math;
using Needle.Model;
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
          needle parity  --fixtures <dir> [--tolerance 0.01] [--verbose]
          needle info    --weights <dir|file> [--config <path>]
          needle run     --weights <dir|file> [--config <path>] --tokens 2,100,200 [--max-new 32]

        Commands:
          parity  Compare this implementation against the JAX reference, stage by
                  stage, using fixtures written by
                  scripts/parity/dump_reference.py.
          info    Print the geometry, parameter count and KV budget of a checkpoint.
          run     Greedy-decode from raw token IDs with the KV-cached session.

        Weights may be a directory holding weights.safetensors + config.json, or a
        .safetensors file (pass --config alongside it).
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
                "info" => RunInfo(rest),
                "run" => RunGenerate(rest),
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

    // ── shared ───────────────────────────────────────────────────────────────

    private static Needle2Model LoadModel(ArgParser options)
    {
        string weights = options.GetRequired("weights");

        if (Directory.Exists(weights)) return ParityHarness.LoadModel(weights);

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
