using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Needle.Inference;

/// <summary>
/// Throughput measurement result.
/// Port of the dictionary returned by Python <c>measure_throughput</c>.
/// </summary>
public sealed record ThroughputResult(
    double AvgTokensGenerated,
    double AvgLatencySeconds,
    double TokensPerSecond);

/// <summary>
/// Generation-quality benchmark result.
/// Port of the dictionary returned by Python <c>benchmark_generation_quality</c>.
/// </summary>
public sealed record GenerationQualityResult(
    double AvgGenerationLength,
    int    MinGenerationLength,
    int    MaxGenerationLength,
    double BigramRepetitionRate,
    IReadOnlyList<(string Prompt, string Generation)> Generations);

/// <summary>
/// Retrieval benchmark result.
/// Port of the dictionary returned by Python <c>benchmark_retrieval</c>.
/// </summary>
public sealed record RetrievalBenchmarkResult(
    IReadOnlyDictionary<int, double> RecallAtK,
    double Mrr,
    int    NumQueries);

/// <summary>
/// Generation-side eval and benchmarking helpers, ported from
/// <c>needle/training/eval.py</c>.  Complements <see cref="PerplexityEval"/>
/// (which covers the loss-side) and <see cref="ToolCallEvaluator"/>
/// (which covers structured tool-call metrics).
/// </summary>
public static class GenerationBenchmarks
{
    // ── Throughput ───────────────────────────────────────────────────────────

    /// <summary>
    /// Measure single-query generation throughput.  Encodes <paramref name="prompt"/>,
    /// runs <paramref name="numRuns"/> generation passes (plus one warmup pass),
    /// and reports tokens-per-second.
    ///
    /// Port of Python <c>measure_throughput</c>.
    /// </summary>
    public static ThroughputResult MeasureThroughput(
        InferenceRunner runner,
        string prompt    = "What is the weather?",
        int numRuns      = 10,
        int maxGenLen    = 64)
    {
        // Warmup pass — discard timing and tokens.
        runner.Generate(
            query: prompt,
            tools: "[]",
            maxGenLen: maxGenLen,
            normalize: false,
            constrained: false);

        var tokensGenerated = new int[numRuns];
        var latencies       = new double[numRuns];

        for (int run = 0; run < numRuns; run++)
        {
            var sw = Stopwatch.StartNew();
            string text = runner.Generate(
                query: prompt,
                tools: "[]",
                maxGenLen: maxGenLen,
                normalize: false,
                constrained: false);
            sw.Stop();

            tokensGenerated[run] = runner.Tokenizer.Encode(text).Count;
            latencies[run]       = sw.Elapsed.TotalSeconds;
        }

        double sumTokens    = tokensGenerated.Sum();
        double sumLatencies = latencies.Sum();
        double avgTokens    = sumTokens / numRuns;
        double avgLatency   = sumLatencies / numRuns;
        double tps          = sumLatencies > 0 ? sumTokens / sumLatencies : 0.0;

        return new ThroughputResult(avgTokens, avgLatency, tps);
    }

    // ── Repetition rate ──────────────────────────────────────────────────────

    /// <summary>
    /// Compute the mean bigram repetition rate across <paramref name="texts"/>.
    /// For each text, the rate is <c>1 - unique_bigrams / total_bigrams</c>
    /// over whitespace-tokenised, lowercased words.  Texts with fewer than
    /// two words contribute 0.
    ///
    /// Port of Python <c>compute_repetition_rate</c>.
    /// </summary>
    public static double ComputeRepetitionRate(IReadOnlyList<string> texts)
    {
        if (texts.Count == 0) return 0.0;

        double sum = 0.0;
        foreach (var text in texts)
        {
            var words = text.ToLowerInvariant()
                            .Split(new[] { ' ', '\t', '\n', '\r' },
                                   StringSplitOptions.RemoveEmptyEntries);
            if (words.Length < 2)
            {
                continue; // contributes 0
            }

            var seen = new HashSet<(string, string)>(words.Length - 1);
            int total = words.Length - 1;
            for (int i = 0; i < total; i++)
                seen.Add((words[i], words[i + 1]));

            sum += 1.0 - (double)seen.Count / total;
        }

        return sum / texts.Count;
    }

    // ── Generation quality ───────────────────────────────────────────────────

    /// <summary>
    /// Generate text for each prompt and report length statistics and
    /// bigram repetition rate.
    ///
    /// Port of Python <c>benchmark_generation_quality</c>.
    /// </summary>
    public static GenerationQualityResult BenchmarkGenerationQuality(
        InferenceRunner runner,
        IReadOnlyList<string> prompts,
        int maxGenLen = 128)
    {
        if (prompts.Count == 0)
            return new GenerationQualityResult(0.0, 0, 0, 0.0, Array.Empty<(string, string)>());

        var generations = new string[prompts.Count];
        var lengths     = new int[prompts.Count];

        for (int i = 0; i < prompts.Count; i++)
        {
            generations[i] = runner.Generate(
                query: prompts[i],
                tools: "[]",
                maxGenLen: maxGenLen,
                normalize: false,
                constrained: false);
            lengths[i] = runner.Tokenizer.Encode(generations[i]).Count;
        }

        var pairs = new (string, string)[prompts.Count];
        for (int i = 0; i < prompts.Count; i++) pairs[i] = (prompts[i], generations[i]);

        return new GenerationQualityResult(
            AvgGenerationLength:  lengths.Average(),
            MinGenerationLength:  lengths.Min(),
            MaxGenerationLength:  lengths.Max(),
            BigramRepetitionRate: ComputeRepetitionRate(generations),
            Generations:          pairs);
    }

    // ── Word error rate ──────────────────────────────────────────────────────

    /// <summary>
    /// Word error rate (WER) computed via Levenshtein edit distance over
    /// whitespace-tokenised, lowercased words.  Pairs hypotheses with
    /// references by index; lengths must match.
    ///
    /// Port of Python <c>compute_wer</c>.
    /// </summary>
    public static double ComputeWer(
        IReadOnlyList<string> hypotheses,
        IReadOnlyList<string> references)
    {
        if (hypotheses.Count != references.Count)
            throw new ArgumentException(
                "hypotheses and references must have the same length.");

        long totalEdits    = 0;
        long totalRefWords = 0;

        for (int p = 0; p < hypotheses.Count; p++)
        {
            var hyp = hypotheses[p].ToLowerInvariant()
                                   .Split(new[] { ' ', '\t', '\n', '\r' },
                                          StringSplitOptions.RemoveEmptyEntries);
            var refr = references[p].ToLowerInvariant()
                                    .Split(new[] { ' ', '\t', '\n', '\r' },
                                           StringSplitOptions.RemoveEmptyEntries);

            int n = refr.Length;
            int m = hyp.Length;
            totalRefWords += n;

            // Two-row Levenshtein DP for memory efficiency.
            var prev = new int[m + 1];
            var curr = new int[m + 1];
            for (int j = 0; j <= m; j++) prev[j] = j;

            for (int i = 1; i <= n; i++)
            {
                curr[0] = i;
                for (int j = 1; j <= m; j++)
                {
                    if (refr[i - 1] == hyp[j - 1])
                        curr[j] = prev[j - 1];
                    else
                        curr[j] = 1 + System.Math.Min(
                            System.Math.Min(prev[j], curr[j - 1]),
                            prev[j - 1]);
                }
                (prev, curr) = (curr, prev);
            }

            totalEdits += prev[m];
        }

        long denom = System.Math.Max(totalRefWords, 1L);
        return (double)totalEdits / denom;
    }

    // ── Retrieval benchmark ──────────────────────────────────────────────────

    /// <summary>
    /// One example for the retrieval benchmark.  Pairs a query string with a
    /// tools JSON (list of tool objects) and an answers JSON (list of called
    /// tool objects with a "name" field).
    /// </summary>
    public sealed record RetrievalExample(string Query, string Tools, string Answers);

    /// <summary>
    /// Benchmark contrastive-retrieval Recall@k and MRR.
    /// For each example, encodes the query and the tools in that example,
    /// ranks tools by cosine similarity to the query, and checks where the
    /// "positive" tools (those named in <c>answers</c>) land.
    ///
    /// Port of Python <c>benchmark_retrieval</c>.
    /// </summary>
    public static RetrievalBenchmarkResult BenchmarkRetrieval(
        InferenceRunner runner,
        IReadOnlyList<RetrievalExample> dataset,
        int maxLen     = 256,
        int[]? ks      = null,
        int? numSamples = null)
    {
        ks ??= new[] { 1, 2, 3, 4, 5 };

        IReadOnlyList<RetrievalExample> ds = dataset;
        if (numSamples.HasValue && ds.Count > numSamples.Value)
            ds = ds.Take(numSamples.Value).ToList();

        var queries     = new List<string>();
        var allTools    = new List<string>();
        // (start, count, posIndices) per surviving query
        var groups      = new List<(int Start, int Count, HashSet<int> Pos)>();

        foreach (var ex in ds)
        {
            JsonNode? toolsNode;
            try { toolsNode = JsonNode.Parse(ex.Tools); }
            catch (JsonException) { continue; }
            if (toolsNode is not JsonArray toolsArr || toolsArr.Count == 0) continue;

            JsonNode? answersNode;
            try { answersNode = JsonNode.Parse(ex.Answers); }
            catch (JsonException) { answersNode = null; }

            var posNames = new HashSet<string>(StringComparer.Ordinal);
            if (answersNode is JsonArray callsArr)
            {
                foreach (var call in callsArr)
                {
                    if (call is JsonObject obj
                        && obj["name"] is JsonValue nv
                        && nv.TryGetValue<string>(out var name))
                    {
                        posNames.Add(name);
                    }
                }
            }
            if (posNames.Count == 0) continue;

            int start = allTools.Count;
            var posIdx = new HashSet<int>();
            for (int j = 0; j < toolsArr.Count; j++)
            {
                if (toolsArr[j] is not JsonObject tool) continue;

                var toolJson = tool.ToJsonString(new JsonSerializerOptions { WriteIndented = false });
                allTools.Add(toolJson);

                if (tool["name"] is JsonValue tnv
                    && tnv.TryGetValue<string>(out var toolName)
                    && posNames.Contains(toolName))
                {
                    posIdx.Add(allTools.Count - 1 - start);
                }
            }

            if (posIdx.Count == 0) continue;

            queries.Add(ex.Query);
            groups.Add((start, allTools.Count - start, posIdx));
        }

        if (queries.Count == 0)
        {
            return new RetrievalBenchmarkResult(
                RecallAtK:   ks.ToDictionary(k => k, _ => 0.0),
                Mrr:         0.0,
                NumQueries:  0);
        }

        float[,] qEmbs = runner.EncodeForRetrieval(queries, maxLen: maxLen);
        float[,] tEmbs = runner.EncodeForRetrieval(allTools, maxLen: maxLen);
        int dim        = qEmbs.GetLength(1);

        var recall  = ks.ToDictionary(k => k, _ => 0);
        double mrrSum = 0.0;

        for (int i = 0; i < queries.Count; i++)
        {
            var (start, count, posSet) = groups[i];
            var scores = new float[count];
            for (int j = 0; j < count; j++)
            {
                float dot = 0f;
                for (int d = 0; d < dim; d++)
                    dot += qEmbs[i, d] * tEmbs[start + j, d];
                scores[j] = dot;
            }

            // Rank descending by score, tie-break by index ascending (stable).
            var ranked = Enumerable.Range(0, count)
                .OrderByDescending(j => scores[j])
                .ThenBy(j => j)
                .ToArray();

            for (int rank = 0; rank < ranked.Length; rank++)
            {
                if (posSet.Contains(ranked[rank]))
                {
                    mrrSum += 1.0 / (rank + 1);
                    break;
                }
            }

            foreach (int k in ks)
            {
                int top = System.Math.Min(k, ranked.Length);
                for (int r = 0; r < top; r++)
                {
                    if (posSet.Contains(ranked[r])) { recall[k]++; break; }
                }
            }
        }

        int n = queries.Count;
        return new RetrievalBenchmarkResult(
            RecallAtK:  recall.ToDictionary(kv => kv.Key, kv => (double)kv.Value / n),
            Mrr:        mrrSum / n,
            NumQueries: n);
    }
}
