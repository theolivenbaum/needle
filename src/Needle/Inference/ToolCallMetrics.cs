using System.Text.Json;
using System.Text.Json.Nodes;

namespace Needle.Inference;

/// <summary>
/// Aggregated tool-call quality metrics across a batch of predictions.
/// Mirrors the Python <c>benchmark_tool_calls</c> output keys in
/// <c>needle/training/eval.py</c>.
/// </summary>
public sealed record ToolCallMetrics
{
    /// <summary>Number of (reference, prediction) pairs evaluated.</summary>
    public int N { get; init; }

    /// <summary>Fraction of predictions that successfully parsed as JSON.</summary>
    public double ParseRate { get; init; }

    /// <summary>Fraction of predictions that exactly matched the reference set of calls.</summary>
    public double ExactMatch { get; init; }

    /// <summary>F1 over the set of called tool names (set-level TP/FP/FN).</summary>
    public double NameF1 { get; init; }

    /// <summary>F1 over the set of full canonical call keys (name + sorted arguments).</summary>
    public double CallF1 { get; init; }

    /// <summary>Fraction of matched (by-name) calls whose argument dicts equal the reference.</summary>
    public double ArgsAcc { get; init; }

    /// <summary>Hallucinated parameter rate — predicted keys not in the tool schema.</summary>
    public double ParamHaluc { get; init; }

    /// <summary>Missing parameter rate — reference keys not produced.</summary>
    public double ParamMiss { get; init; }

    /// <summary>Fraction of matched (by-name + by-key) values that are exactly correct.</summary>
    public double ValueAcc { get; init; }
}

/// <summary>
/// Compute tool-call evaluation metrics over a batch of
/// <c>(reference, prediction)</c> JSON strings.
///
/// Port of <c>_eval_pool</c> in <c>train.py</c> and the metrics computation in
/// <c>benchmark_tool_calls</c> in <c>training/eval.py</c>.
/// </summary>
public static class ToolCallEvaluator
{
    private record Example(string Tools, string Reference, string Prediction);

    /// <summary>
    /// Evaluate a list of examples.
    /// </summary>
    /// <param name="tools">JSON-array tool definitions used for each example.</param>
    /// <param name="references">Reference tool-call JSON strings.</param>
    /// <param name="predictions">Predicted tool-call JSON strings.</param>
    public static ToolCallMetrics Evaluate(
        IReadOnlyList<string> tools,
        IReadOnlyList<string> references,
        IReadOnlyList<string> predictions)
    {
        int n = references.Count;
        if (predictions.Count != n)
            throw new ArgumentException("references and predictions length mismatch");
        if (tools.Count != n)
            throw new ArgumentException("tools and references length mismatch");

        int parseOk = 0;
        int exact = 0;
        int nameTp = 0, nameFp = 0, nameFn = 0;
        int callTp = 0, callFp = 0, callFn = 0;
        int argsCorrect = 0, argsTotal = 0;
        int hallucParams = 0, totalPredParams = 0;
        int missingParams = 0, totalRefParams = 0;
        int correctValues = 0, matchedParams = 0;

        for (int i = 0; i < n; i++)
        {
            string refText  = (references[i]  ?? string.Empty).Trim();
            string predText = (predictions[i] ?? string.Empty).Trim();

            bool refIsEmpty  = refText is "" or "[]";
            bool predIsEmpty = predText is "" or "[]";

            var refCalls  = TryParseCalls(refText);
            var predCalls = TryParseCallsWithSuccess(predText, out bool predParsed);
            if (predParsed) parseOk++;

            // Exact match
            if (refIsEmpty && predIsEmpty)
            {
                exact++;
            }
            else if (!refIsEmpty && !predIsEmpty)
            {
                var refKeys  = SortedCallKeys(refCalls);
                var predKeys = SortedCallKeys(predCalls);
                if (refKeys.SequenceEqual(predKeys)
                    && refKeys.Count == refCalls.Count
                    && predKeys.Count == predCalls.Count)
                {
                    exact++;
                }
            }

            // Name-set TP/FP/FN
            var refNames  = refCalls.Select(c  => c.Name).Where(n => n is not null).Select(n => n!).ToHashSet();
            var predNames = predCalls.Select(c => c.Name).Where(n => n is not null).Select(n => n!).ToHashSet();
            nameTp += refNames.Intersect(predNames).Count();
            nameFp += predNames.Except(refNames).Count();
            nameFn += refNames.Except(predNames).Count();

            // Call-set TP/FP/FN
            var refKeySet  = refCalls.Select(CallKey).Where(k => k is not null).Select(k => k!).ToHashSet();
            var predKeySet = predCalls.Select(CallKey).Where(k => k is not null).Select(k => k!).ToHashSet();
            callTp += refKeySet.Intersect(predKeySet).Count();
            callFp += predKeySet.Except(refKeySet).Count();
            callFn += refKeySet.Except(predKeySet).Count();

            // Args accuracy & param hallucination / miss / value
            var refByName = new Dictionary<string, List<Dictionary<string, JsonNode?>>>();
            foreach (var c in refCalls)
            {
                if (c.Name is null) continue;
                if (!refByName.TryGetValue(c.Name, out var list))
                {
                    list = new List<Dictionary<string, JsonNode?>>();
                    refByName[c.Name] = list;
                }
                list.Add(c.Arguments ?? new());
            }

            foreach (var c in predCalls)
            {
                if (c.Name is null || !refByName.TryGetValue(c.Name, out var refArgsList))
                    continue;
                argsTotal++;
                string pa = JsonSerializer.Serialize(SortKeys(c.Arguments ?? new()));
                if (refArgsList.Any(ra => JsonSerializer.Serialize(SortKeys(ra)) == pa))
                    argsCorrect++;
            }

            var toolParamMap = ParseToolSchema(tools[i]);
            foreach (var c in predCalls)
            {
                if (c.Name is null || !toolParamMap.TryGetValue(c.Name, out var schemaKeys))
                    continue;

                var pKeys = c.Arguments?.Keys.ToHashSet() ?? new HashSet<string>();
                totalPredParams += pKeys.Count;
                hallucParams    += pKeys.Except(schemaKeys).Count();

                if (refByName.TryGetValue(c.Name, out var refArgsList) && refArgsList.Count > 0)
                {
                    var refArgs = refArgsList[0];
                    var rKeys = refArgs.Keys.ToHashSet();
                    totalRefParams += rKeys.Count;
                    missingParams  += rKeys.Except(pKeys).Count();

                    var common = pKeys.Intersect(rKeys);
                    foreach (var k in common)
                    {
                        matchedParams++;
                        string pv = JsonSerializer.Serialize(c.Arguments![k]);
                        string rv = JsonSerializer.Serialize(refArgs[k]);
                        if (pv == rv) correctValues++;
                    }
                }
            }
        }

        double safeDiv(int num, int den) => den <= 0 ? 0.0 : (double)num / den;
        double f1(int tp, int fp, int fn) =>
            (tp + fp + tp + fn) <= 0 ? 0.0 : 2.0 * tp / ((tp + fp) + (tp + fn));

        return new ToolCallMetrics
        {
            N           = n,
            ParseRate   = n > 0 ? 1.0 - (n - parseOk) / (double)n : 0.0,
            ExactMatch  = safeDiv(exact, n),
            NameF1      = f1(nameTp, nameFp, nameFn),
            CallF1      = f1(callTp, callFp, callFn),
            ArgsAcc     = safeDiv(argsCorrect, argsTotal),
            ParamHaluc  = safeDiv(hallucParams, totalPredParams),
            ParamMiss   = safeDiv(missingParams, totalRefParams),
            ValueAcc    = safeDiv(correctValues, matchedParams),
        };
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private sealed record ParsedCall(string? Name, Dictionary<string, JsonNode?>? Arguments);

    private static List<ParsedCall> TryParseCalls(string json)
    {
        return TryParseCallsWithSuccess(json, out _);
    }

    private static List<ParsedCall> TryParseCallsWithSuccess(string json, out bool ok)
    {
        ok = false;
        var list = new List<ParsedCall>();
        if (string.IsNullOrWhiteSpace(json))
        {
            ok = true;
            return list;
        }

        try
        {
            var node = JsonNode.Parse(json);
            if (node is JsonArray arr)
            {
                foreach (var elem in arr)
                {
                    if (elem is JsonObject obj)
                        list.Add(ParseCallObject(obj));
                }
                ok = true;
                return list;
            }
            if (node is JsonObject single)
            {
                list.Add(ParseCallObject(single));
                ok = true;
                return list;
            }
            ok = true;
            return list;
        }
        catch (JsonException)
        {
            return list;
        }
    }

    private static ParsedCall ParseCallObject(JsonObject obj)
    {
        string? name = obj.TryGetPropertyValue("name", out var nameNode)
            ? nameNode?.GetValue<string>()
            : null;

        Dictionary<string, JsonNode?>? args = null;
        if (obj.TryGetPropertyValue("arguments", out var argsNode) && argsNode is JsonObject argsObj)
        {
            args = new Dictionary<string, JsonNode?>();
            foreach (var kv in argsObj)
                args[kv.Key] = kv.Value?.DeepClone();
        }
        return new ParsedCall(name, args);
    }

    private static string? CallKey(ParsedCall c)
    {
        if (c.Name is null) return null;
        var obj = new JsonObject
        {
            ["name"]      = c.Name,
            ["arguments"] = c.Arguments is null
                ? null
                : (JsonNode)SortKeys(c.Arguments),
        };
        return obj.ToJsonString();
    }

    private static List<string> SortedCallKeys(List<ParsedCall> calls) =>
        calls.Select(CallKey).Where(k => k is not null).Select(k => k!).OrderBy(k => k, StringComparer.Ordinal).ToList();

    private static JsonObject SortKeys(Dictionary<string, JsonNode?> dict)
    {
        var result = new JsonObject();
        foreach (var k in dict.Keys.OrderBy(s => s, StringComparer.Ordinal))
            result[k] = dict[k]?.DeepClone();
        return result;
    }

    private static Dictionary<string, HashSet<string>> ParseToolSchema(string toolsJson)
    {
        var map = new Dictionary<string, HashSet<string>>();
        if (string.IsNullOrWhiteSpace(toolsJson)) return map;

        try
        {
            var node = JsonNode.Parse(toolsJson);
            if (node is not JsonArray arr) return map;

            foreach (var t in arr)
            {
                if (t is not JsonObject tool) continue;
                if (!tool.TryGetPropertyValue("name", out var nameNode)) continue;
                string? name = nameNode?.GetValue<string>();
                if (name is null) continue;

                var keys = new HashSet<string>();
                if (tool.TryGetPropertyValue("parameters", out var p) && p is JsonObject paramObj)
                {
                    foreach (var kv in paramObj) keys.Add(kv.Key);
                }
                map[name] = keys;
            }
        }
        catch (JsonException)
        {
            // Bad tool schema — empty map.
        }
        return map;
    }
}
