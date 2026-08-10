using Needle.Inference;
using Needle.Math;
using Needle.Model;
using Needle.Weights;

namespace Needle.Diagnostics;

/// <summary>How far one C# tensor drifted from its reference counterpart.</summary>
/// <param name="Name">Stage name, e.g. <c>layer07</c> or <c>logits</c>.</param>
/// <param name="Elements">Number of elements compared.</param>
/// <param name="MaxAbsolute">Largest absolute difference.</param>
/// <param name="MaxRelative">
/// Largest difference relative to the reference's own scale (RMS of the
/// reference tensor), which is the meaningful yardstick for activations that
/// grow through the stack.
/// </param>
/// <param name="Cosine">Cosine similarity, 1 when the tensors point the same way.</param>
public sealed record ParityDelta(string Name, int Elements, float MaxAbsolute, float MaxRelative, double Cosine)
{
    /// <summary>Does this stage agree within <paramref name="tolerance"/> relative error?</summary>
    public bool Within(float tolerance) => MaxRelative <= tolerance;

    public override string ToString() =>
        $"{Name,-28} n={Elements,-9} max|Δ|={MaxAbsolute,-12:G4} rel={MaxRelative,-12:G4} cos={Cosine:F9}";
}

/// <summary>Every stage compared for one prompt, plus the token-level check.</summary>
/// <param name="Case">Case name.</param>
/// <param name="Deltas">Per-stage deltas, in evaluation order.</param>
/// <param name="ArgmaxMatches">Positions whose predicted token matches the reference.</param>
/// <param name="ArgmaxTotal">Positions compared.</param>
/// <param name="DecodeTokensMatch">Whether the KV-cached greedy continuation matched exactly.</param>
public sealed record ParityReport(
    string Case,
    IReadOnlyList<ParityDelta> Deltas,
    int ArgmaxMatches,
    int ArgmaxTotal,
    bool DecodeTokensMatch)
{
    /// <summary>Worst relative error across every stage.</summary>
    public float WorstRelative => Deltas.Count == 0 ? 0f : Deltas.Max(d => d.MaxRelative);

    /// <summary>True when every stage is within tolerance and the tokens agree.</summary>
    public bool Passed(float tolerance) =>
        Deltas.All(d => d.Within(tolerance)) && ArgmaxMatches == ArgmaxTotal && DecodeTokensMatch;
}

/// <summary>
/// Compares this implementation against the JAX reference, stage by stage.
///
/// Consumes the fixtures written by <c>scripts/parity/dump_reference.py</c>: the
/// checkpoint plus, for each prompt, the embeddings, both engram sites, every
/// layer's residual snapshot, the final hidden state, the logits, the MTP
/// logits, the pooled heads, and the KV-cached decode trace.  Checking each
/// stage rather than only the logits means a disagreement localises to the layer
/// that introduced it.
/// </summary>
public static class ParityHarness
{
    /// <summary>File name of the flattened checkpoint inside a fixture directory.</summary>
    public const string WeightsFile = "weights.safetensors";

    /// <summary>File name of the checkpoint config inside a fixture directory.</summary>
    public const string ConfigFile = "config.json";

    /// <summary>Does <paramref name="directory"/> hold a usable fixture set?</summary>
    public static bool IsAvailable(string directory) =>
        Directory.Exists(directory)
        && File.Exists(Path.Combine(directory, WeightsFile))
        && File.Exists(Path.Combine(directory, ConfigFile))
        && CaseFiles(directory).Count > 0;

    /// <summary>Case fixture paths in <paramref name="directory"/>, sorted by name.</summary>
    public static IReadOnlyList<string> CaseFiles(string directory) =>
        Directory.Exists(directory)
            ? Directory.GetFiles(directory, "case-*.safetensors").OrderBy(p => p, StringComparer.Ordinal).ToList()
            : [];

    /// <summary>Load the model described by a fixture directory.</summary>
    public static Needle2Model LoadModel(string directory)
    {
        var config = CheckpointConfig.Load(Path.Combine(directory, ConfigFile));
        var flat = Safetensors.Load(Path.Combine(directory, WeightsFile));
        return new Needle2Model(Needle2Weights.FromFlat(config, flat));
    }

    /// <summary>Run every case in <paramref name="directory"/> against <paramref name="model"/>.</summary>
    public static IReadOnlyList<ParityReport> RunAll(Needle2Model model, string directory) =>
        CaseFiles(directory).Select(path => RunCase(model, path)).ToList();

    /// <summary>Compare one case fixture.</summary>
    public static ParityReport RunCase(Needle2Model model, string casePath)
    {
        var expected = Safetensors.Load(casePath);
        string name = Path.GetFileNameWithoutExtension(casePath);
        if (name.StartsWith("case-", StringComparison.Ordinal)) name = name[5..];

        var tokens = ToTokens(expected["tokens"]);
        var deltas = new List<ParityDelta>();

        // The reference dump uses a plain causal mask over a padding-free prompt,
        // so the port runs the same way here.
        var trace = new CollectingTrace();
        var result = model.Forward(tokens, SequenceMask.Causal(tokens.Length), collectCells: true, trace);

        Compare(deltas, "embed", expected, trace.Get("embed"));

        for (int site = 0; site < model.Config.EngramSites; site++)
        {
            Compare(deltas, $"engram{site}.k", expected, trace.Get($"engram{site}.k"));
            Compare(deltas, $"engram{site}.v", expected, trace.Get($"engram{site}.v"));
        }

        CompareCells(deltas, expected["cells"], result.Cells!, model.Config.NumLayers);
        Compare(deltas, "hidden", expected, result.Hidden);

        var logits = model.Logits(result.Hidden);
        Compare(deltas, "logits", expected, logits);

        var (argmaxMatches, argmaxTotal) = CompareArgmax(expected["logits"], logits);

        if (expected.TryGetValue("contrastive", out var contrastive) && model.Weights.Contrastive is not null)
            Compare(deltas, "contrastive", contrastive, model.EncodeContrastive(tokens));

        if (expected.TryGetValue("confidence", out var confidence) && model.Weights.Confidence is not null)
        {
            float actual = model.ConfidenceLogit(tokens);
            deltas.Add(Delta("confidence", confidence.ReadSpan[..1], [actual]));
        }

        bool decodeMatches = true;
        if (expected.TryGetValue("decode_logits", out var decodeLogits))
        {
            var expectedTokens = ToTokens(expected["decode_tokens"]);
            decodeMatches = CompareDecode(deltas, model, tokens, decodeLogits, expectedTokens);
        }

        return new ParityReport(name, deltas, argmaxMatches, argmaxTotal, decodeMatches);
    }

    /// <summary>
    /// Replay the reference's greedy KV-cached continuation and compare each
    /// step's logits and chosen token.
    /// </summary>
    private static bool CompareDecode(
        List<ParityDelta> deltas, Needle2Model model, int[] prompt,
        NdArray expectedLogits, int[] expectedTokens)
    {
        int steps = expectedLogits.Shape[0];
        int vocab = expectedLogits.Shape[1];

        // The reference dump decodes with unbounded attention (its own
        // generate_cached passes no sink), so match that here.
        var session = new NeedleSession(model, capacity: prompt.Length + steps + 1, window: -1);
        var actual = new NdArray(steps, vocab);
        var produced = new List<int>();

        var logits = session.Advance(prompt);
        logits.ReadSpan.CopyTo(actual.Row(0));

        for (int step = 1; step < steps; step++)
        {
            int next = Ops.ArgMax(logits.ReadSpan);
            produced.Add(next);
            logits = session.Advance(next);
            logits.ReadSpan.CopyTo(actual.Row(step));
        }

        deltas.Add(Delta("decode_logits", expectedLogits.ReadSpan, actual.ReadSpan));

        int compare = System.Math.Min(produced.Count, expectedTokens.Length);
        for (int i = 0; i < compare; i++)
            if (produced[i] != expectedTokens[i]) return false;
        return true;
    }

    private static void CompareCells(List<ParityDelta> deltas, NdArray expected, NdArray actual, int layers)
    {
        int seqLen = expected.Shape[0], perToken = expected.Shape[1], d = expected.Shape[2];
        var expectedSpan = expected.ReadSpan;
        var actualSpan = actual.ReadSpan;

        var left = new float[seqLen * d];
        var right = new float[seqLen * d];
        for (int cell = 0; cell < perToken; cell++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                expectedSpan.Slice((t * perToken + cell) * d, d).CopyTo(left.AsSpan(t * d, d));
                actualSpan.Slice((t * perToken + cell) * d, d).CopyTo(right.AsSpan(t * d, d));
            }
            string label = cell == 0 ? "cells.embed" : $"cells.layer{cell - 1:D2}";
            deltas.Add(Delta(label, left, right));
        }
        _ = layers;
    }

    private static (int Matches, int Total) CompareArgmax(NdArray expected, NdArray actual)
    {
        int rows = expected.Shape[0], vocab = expected.Shape[1];
        int matches = 0;
        for (int t = 0; t < rows; t++)
        {
            if (Ops.ArgMax(expected.ReadSpan.Slice(t * vocab, vocab))
                == Ops.ArgMax(actual.ReadSpan.Slice(t * vocab, vocab)))
                matches++;
        }
        return (matches, rows);
    }

    private static void Compare(List<ParityDelta> deltas, string name,
                                IReadOnlyDictionary<string, NdArray> expected, NdArray? actual)
    {
        if (actual is null || !expected.TryGetValue(name, out var reference)) return;
        deltas.Add(Delta(name, reference.ReadSpan, actual.ReadSpan));
    }

    private static void Compare(List<ParityDelta> deltas, string name, NdArray expected, NdArray actual) =>
        deltas.Add(Delta(name, expected.ReadSpan, actual.ReadSpan));

    /// <summary>Absolute, scale-relative and directional agreement between two tensors.</summary>
    public static ParityDelta Delta(string name, ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        if (expected.Length != actual.Length)
            throw new InvalidOperationException(
                $"{name}: reference has {expected.Length} elements, port produced {actual.Length}.");

        float maxAbs = 0f;
        double sumSquares = 0, dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = System.Math.Abs(expected[i] - actual[i]);
            if (diff > maxAbs) maxAbs = diff;
            sumSquares += (double)expected[i] * expected[i];
            dot += (double)expected[i] * actual[i];
            normA += (double)expected[i] * expected[i];
            normB += (double)actual[i] * actual[i];
        }

        double rms = System.Math.Sqrt(sumSquares / System.Math.Max(1, expected.Length));
        float relative = rms > 0 ? (float)(maxAbs / rms) : maxAbs;
        double cosine = normA > 0 && normB > 0 ? dot / System.Math.Sqrt(normA * normB) : 1.0;

        return new ParityDelta(name, expected.Length, maxAbs, relative, cosine);
    }

    private static int[] ToTokens(NdArray value)
    {
        var tokens = new int[value.Length];
        for (int i = 0; i < tokens.Length; i++) tokens[i] = (int)MathF.Round(value[i]);
        return tokens;
    }

    /// <summary>Keeps a copy of every traced tensor so stages can be compared after the fact.</summary>
    private sealed class CollectingTrace : ITrace
    {
        private readonly Dictionary<string, NdArray> _values = new();

        public void Record(string name, NdArray value) => _values[name] = value.Clone();

        public NdArray? Get(string name) => _values.GetValueOrDefault(name);
    }
}
