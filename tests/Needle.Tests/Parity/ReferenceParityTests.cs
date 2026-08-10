using Needle.Diagnostics;
using Needle.Model;
using Needle.Tokenizer;
using Needle.Weights;

namespace Needle.Tests.Parity;

/// <summary>
/// Locates the parity fixtures, which are generated on demand rather than
/// committed — the checkpoint alone is 180 MB as float32.
///
/// Set <c>NEEDLE_FIXTURES</c>, or generate them into <c>fixtures/</c> at the repo
/// root with:
/// <code>
/// python3 scripts/parity/dump_reference.py --out fixtures/parity
/// python3 scripts/parity/dump_cact.py      --out fixtures/cact
/// </code>
/// Without them these tests skip; the pure-unit tests still cover the kernels.
/// </summary>
internal static class Fixtures
{
    public static string Root { get; } = Resolve();

    public static string Parity => Path.Combine(Root, "parity");

    public static string Cact => Path.Combine(Root, "cact");

    /// <summary>The released blob, when the fixture directory carries one.</summary>
    public static string? CactBlob
    {
        get
        {
            if (!Directory.Exists(Cact)) return null;
            string? fromEnvironment = Environment.GetEnvironmentVariable("NEEDLE_CACT");
            if (!string.IsNullOrEmpty(fromEnvironment) && File.Exists(fromEnvironment))
                return fromEnvironment;
            return Directory.GetFiles(Cact, "*.cact").FirstOrDefault()
                   ?? Directory.GetFiles(Root, "*.cact").FirstOrDefault();
        }
    }

    private static string Resolve()
    {
        string? fromEnvironment = Environment.GetEnvironmentVariable("NEEDLE_FIXTURES");
        if (!string.IsNullOrEmpty(fromEnvironment)) return fromEnvironment;

        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (Directory.Exists(Path.Combine(directory.FullName, ".git")))
                return Path.Combine(directory.FullName, "fixtures");
            directory = directory.Parent;
        }
        return "fixtures";
    }
}

/// <summary>
/// Every stage of the model against the JAX reference on the released weights.
///
/// These are the tests that would catch a real porting mistake: the unit tests
/// pin the kernels, but only a comparison against the reference catches a
/// misread of the architecture — a residual measured from the wrong point, an
/// off-by-one in the engram lookback, a transposed kernel.
/// </summary>
public class ReferenceParityTests
{
    private static Needle2Model? _model;
    private static readonly Lock Gate = new();

    /// <summary>Loading 43M parameters takes a moment; share one model across the class.</summary>
    private static Needle2Model Model
    {
        get
        {
            lock (Gate) return _model ??= ParityHarness.LoadModel(Fixtures.Parity);
        }
    }

    public static bool Available => ParityHarness.IsAvailable(Fixtures.Parity);

    [SkippableFact]
    public void EveryStageMatchesTheReference()
    {
        Skip.IfNot(Available, $"No parity fixtures in '{Fixtures.Parity}'.");

        foreach (string casePath in ParityHarness.CaseFiles(Fixtures.Parity))
        {
            var report = ParityHarness.RunCase(Model, casePath);

            foreach (var delta in report.Deltas)
            {
                Assert.True(delta.MaxRelative < 1e-3f,
                    $"case '{report.Case}' stage '{delta.Name}': relative error {delta.MaxRelative:G4}");
                Assert.True(delta.Cosine > 0.9999,
                    $"case '{report.Case}' stage '{delta.Name}': cosine {delta.Cosine:F9}");
            }

            Assert.Equal(report.ArgmaxTotal, report.ArgmaxMatches);
            Assert.True(report.DecodeTokensMatch,
                $"case '{report.Case}': the KV-cached continuation diverged from the reference");
        }
    }

    [SkippableFact]
    public void CachedDecodeAgreesWithTheFullForwardPass()
    {
        Skip.IfNot(Available, $"No parity fixtures in '{Fixtures.Parity}'.");

        int[] prompt = [2, 100, 200, 300, 400, 500, 600, 700];

        // The full pass and the cached session share no code below the stack, so
        // agreeing here means the cache, the plan and the engram window all line
        // up with the batched path.
        var full = Model.Forward(prompt, SequenceMask.Causal(prompt.Length));
        var fullLogits = Model.LastLogits(full.Hidden);

        var session = new Needle.Inference.NeedleSession(Model, window: -1);
        var cachedLogits = session.Advance(prompt);

        var delta = ParityHarness.Delta("cached-vs-full", fullLogits.ReadSpan, cachedLogits.ReadSpan);
        Assert.True(delta.MaxRelative < 1e-4f, delta.ToString());
    }
}

/// <summary>
/// The <c>.cact</c> deployment blob: Cactus-Quant reconstruction against the
/// reference's own reader, and the embedded tokenizer against its encoder.
/// </summary>
public class CactParityTests
{
    public static bool Available =>
        Fixtures.CactBlob is not null
        && File.Exists(Path.Combine(Fixtures.Cact, "tensors.safetensors"));

    [SkippableFact]
    public void DequantisationMatchesTheReferenceReader()
    {
        Skip.IfNot(Available, $"No .cact fixtures in '{Fixtures.Cact}'.");

        var checks = CactCheck.Run(Fixtures.CactBlob!, Fixtures.Cact);
        Assert.NotEmpty(checks);

        foreach (var check in checks)
            Assert.True(check.Delta.MaxRelative < 1e-4f,
                $"tensor {check.Index} ({check.Dtype}, {check.Bits} bits): {check.Delta}");
    }

    [SkippableFact]
    public void TokenizerRoundTripsIdenticallyToTheReference()
    {
        Skip.IfNot(Available, $"No .cact fixtures in '{Fixtures.Cact}'.");
        string tokenizerJson = Path.Combine(Fixtures.Cact, "tokenizer.json");
        Skip.IfNot(File.Exists(tokenizerJson), "No tokenizer fixture.");

        var tokenizer = CactTokenizer.FromCact(Fixtures.CactBlob!);
        ChatMarkers.Validate(tokenizer);

        var report = TokenizerCheck.Run(tokenizer, tokenizerJson);
        Assert.Empty(report.PieceMismatches);

        foreach (var result in report.Cases)
        {
            Assert.True(result.EncodeMatches,
                $"encode diverged at token {result.FirstDivergence} for '{result.Text}': "
                + $"[{string.Join(",", result.Expected)}] vs [{string.Join(",", result.Actual)}]");
            Assert.Equal(result.ExpectedDecoded, result.ActualDecoded);
        }
    }

    [SkippableFact]
    public void BlobLoadsIntoAWorkingModel()
    {
        Skip.IfNot(Fixtures.CactBlob is not null, $"No .cact blob in '{Fixtures.Cact}'.");

        var loaded = CactLayout.Load(Fixtures.CactBlob!);
        var model = new Needle2Model(Needle2Weights.FromFlat(loaded.Config, loaded.Parameters));

        // The geometry has to be recovered from tensor shapes alone — the format
        // stores neither names nor dimensions.
        Assert.Equal(8192, model.Config.VocabSize);
        Assert.Equal(512, model.Config.DModel);
        Assert.Equal(27, model.Config.NumLayers);
        Assert.Equal(8, model.Config.NumHeads);
        Assert.Equal(4, model.Config.NumKvHeads);
        Assert.Equal(4, model.Config.MhcLanes);
        Assert.Equal(2, model.Config.EngramSites);
        Assert.NotNull(model.Weights.Contrastive);
        Assert.NotNull(model.Weights.Confidence);
        Assert.NotNull(loaded.TokenizerBlob);

        var logits = model.LastLogits(model.Forward([2, 100, 200]).Hidden);
        Assert.Equal(8192, logits.Length);
        Assert.All(logits.ReadSpan.ToArray(), v => Assert.True(float.IsFinite(v)));
    }
}
