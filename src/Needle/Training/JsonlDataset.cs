using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Needle.Tokenizer;

namespace Needle.Training;

/// <summary>
/// A single training example: a natural-language query, the available tools
/// (JSON-array string), and the expected tool-call answer (JSON-array string).
/// </summary>
public sealed record FinetuneExample(string Query, string Tools, string Answers);

/// <summary>
/// Load and prepare JSONL finetune data.
/// Mirrors the input contract of <c>needle/training/finetune.py</c>:
/// each line is a JSON object with <c>query</c>, <c>tools</c>, and
/// <c>answers</c> string fields.
/// </summary>
public static class JsonlDataset
{
    /// <summary>
    /// Read examples from a JSONL file.  Skips blank lines and lines that
    /// fail to parse.
    /// </summary>
    public static List<FinetuneExample> Load(string path)
    {
        var examples = new List<FinetuneExample>();
        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            try
            {
                var node = JsonNode.Parse(line);
                if (node is not JsonObject obj) continue;

                string query   = obj["query"]?.GetValue<string>()   ?? "";
                string tools   = obj["tools"]?.GetValue<string>()   ?? "[]";
                string answers = obj["answers"]?.GetValue<string>() ?? "[]";
                examples.Add(new FinetuneExample(query, tools, answers));
            }
            catch (JsonException)
            {
                // Skip malformed lines
            }
        }
        return examples;
    }

    /// <summary>
    /// Per-tool 80/10/10-style split.  Each unique primary tool name receives
    /// <paramref name="valPerTool"/> validation examples and
    /// <paramref name="testPerTool"/> test examples (proportional fallback for
    /// rare tools).  Mirrors <c>_per_tool_split</c> in finetune.py.
    /// </summary>
    public static (List<FinetuneExample> train, List<FinetuneExample> val, List<FinetuneExample> test)
        PerToolSplit(IList<FinetuneExample> examples, int valPerTool = 10, int testPerTool = 10, int seed = 42)
    {
        var rng = new Random(seed);
        var buckets = new Dictionary<string, List<int>>();

        for (int i = 0; i < examples.Count; i++)
        {
            string primary = "__no_tool__";
            try
            {
                if (JsonNode.Parse(examples[i].Answers) is JsonArray arr && arr.Count > 0
                    && arr[0] is JsonObject c0 && c0["name"]?.GetValue<string>() is string n)
                {
                    primary = n;
                }
            }
            catch (JsonException) { }
            if (!buckets.TryGetValue(primary, out var list))
            {
                list = new List<int>();
                buckets[primary] = list;
            }
            list.Add(i);
        }

        var train = new List<FinetuneExample>();
        var val   = new List<FinetuneExample>();
        var test  = new List<FinetuneExample>();

        foreach (var (_, indices) in buckets)
        {
            // Fisher-Yates shuffle
            for (int i = indices.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            int n = indices.Count;
            int needed = valPerTool + testPerTool;
            int nTest, nVal;
            if (n < needed)
            {
                if (n == 1)      { nTest = 1; nVal = 0; }
                else if (n == 2) { nTest = 1; nVal = 1; }
                else
                {
                    nTest = System.Math.Max(1, n / 3);
                    nVal  = System.Math.Max(1, (n - nTest) / 3);
                }
            }
            else
            {
                nTest = testPerTool;
                nVal  = valPerTool;
            }

            int idx = 0;
            for (int k = 0; k < nTest && idx < n; k++, idx++) test.Add(examples[indices[idx]]);
            for (int k = 0; k < nVal  && idx < n; k++, idx++) val.Add(examples[indices[idx]]);
            for (; idx < n; idx++) train.Add(examples[indices[idx]]);
        }

        return (train, val, test);
    }
}
