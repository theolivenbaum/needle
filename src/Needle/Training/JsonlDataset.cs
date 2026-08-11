using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using Needle.Tokenizer;

namespace Needle.Training;

/// <summary>
/// One fine-tuning example.
///
/// Matches the JSONL contract in the upstream README: a query, the schemas that
/// were available, the calls that satisfy it, and optionally the model's short
/// derivation.  An off-topic example has <c>answers: []</c> — that is how the
/// refusal behaviour is taught, so those rows are load-bearing rather than noise.
/// </summary>
/// <param name="Query">The user request, or a passage to extract from.</param>
/// <param name="ToolsJson">Declared schemas as a compact JSON array.</param>
/// <param name="AnswersJson">Expected calls as a compact JSON array.</param>
/// <param name="Reasoning">Short derivation of each argument; may be empty.</param>
/// <param name="System">Optional system facts turn.</param>
public sealed record FinetuneExample(
    string Query,
    string ToolsJson,
    string AnswersJson,
    string Reasoning = "",
    string System = "")
{
    /// <summary>Name of the first declared call, used to stratify splits.</summary>
    public string PrimaryTool
    {
        get
        {
            try
            {
                if (JsonNode.Parse(AnswersJson) is JsonArray { Count: > 0 } answers
                    && answers[0] is JsonObject first)
                    return first["name"]?.GetValue<string>() ?? "";
            }
            catch (JsonException)
            {
                // Fall through to the empty name.
            }
            return "";
        }
    }

    /// <summary>True when no declared tool can serve the query.</summary>
    public bool IsRefusal => AnswersJson.Trim() is "" or "[]";
}

/// <summary>
/// Reads and prepares Needle 2 fine-tuning data.
/// Port of the JSONL handling in <c>.reference/needle/model/finetune.py</c>.
/// </summary>
public static class JsonlDataset
{
    private static readonly JsonSerializerOptions Compact = new()
    {
        WriteIndented = false,
        // Python's json.dumps escapes non-ASCII by default; keeping the text raw
        // here is the closer match for tool schemas, which are almost always
        // ASCII, and avoids over-escaping punctuation in descriptions.
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    /// <summary>
    /// Read examples from a JSONL file.  <c>tools</c> and <c>answers</c> may be
    /// given either as JSON arrays or as pre-serialised strings; blank and
    /// malformed lines are skipped, as are rows with no <c>query</c>.
    /// </summary>
    public static List<FinetuneExample> Load(string path)
    {
        var examples = new List<FinetuneExample>();
        foreach (string line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            try
            {
                if (JsonNode.Parse(line) is not JsonObject row) continue;
                if (row["query"]?.GetValue<string>() is not { } query) continue;

                examples.Add(new FinetuneExample(
                    query,
                    AsJsonArray(row["tools"]),
                    AsJsonArray(row["answers"] ?? row["function_calls"]),
                    row["reasoning"]?.GetValue<string>() ?? "",
                    row["system"]?.GetValue<string>() ?? ""));
            }
            catch (JsonException)
            {
                // Skip malformed lines rather than failing the whole file.
            }
        }
        return examples;
    }

    /// <summary>Serialise a node to the compact JSON array the template expects.</summary>
    private static string AsJsonArray(JsonNode? node) => node switch
    {
        null => "[]",
        JsonValue value when value.TryGetValue(out string? text) => text ?? "[]",
        _ => node.ToJsonString(Compact),
    };

    /// <summary>
    /// Render and tokenize one example into a training sequence.
    /// Port of <c>_encode</c> in finetune.py.
    /// </summary>
    /// <param name="tokenizer">Tokenizer matching the checkpoint.</param>
    /// <param name="example">The example to encode.</param>
    /// <param name="maxLength">Sequence length; shorter rows are padded.</param>
    /// <returns>
    /// Token IDs and a loss mask that is 1 only over the target continuation, so
    /// the prompt is conditioned on but never trained against.
    /// </returns>
    public static (int[] Ids, float[] LossMask) Encode(
        CactTokenizer tokenizer, FinetuneExample example, int maxLength)
    {
        string prompt = ChatTemplate.Prompt(example.Query, example.ToolsJson,
                                            string.IsNullOrEmpty(example.System) ? null : example.System);
        string target = ChatTemplate.Target(example.AnswersJson, example.Reasoning);

        var promptIds = tokenizer.Encode(prompt);
        var targetIds = tokenizer.Encode(target);

        var ids = new int[maxLength];
        var mask = new float[maxLength];
        ids.AsSpan().Fill(ChatMarkers.PadId);

        int cursor = 0;
        if (cursor < maxLength) ids[cursor++] = tokenizer.BosId;
        foreach (int id in promptIds)
        {
            if (cursor >= maxLength) break;
            ids[cursor++] = id;
        }
        foreach (int id in targetIds)
        {
            if (cursor >= maxLength) break;
            ids[cursor] = id;
            mask[cursor++] = 1f;
        }
        if (cursor < maxLength)
        {
            ids[cursor] = tokenizer.EosId;
            mask[cursor] = 1f;
        }

        return (ids, mask);
    }

    /// <summary>
    /// Stratified split holding out a bounded number of examples per primary
    /// tool, so every tool is represented in validation and test rather than only
    /// the common ones.  Mirrors the per-tool split in the reference finetuner.
    /// </summary>
    public static (List<FinetuneExample> Train, List<FinetuneExample> Validation, List<FinetuneExample> Test)
        PerToolSplit(IReadOnlyList<FinetuneExample> examples,
                     int validationPerTool = 10, int testPerTool = 10, int seed = 42)
    {
        var random = new Random(seed);
        var train = new List<FinetuneExample>();
        var validation = new List<FinetuneExample>();
        var test = new List<FinetuneExample>();

        foreach (var group in examples.GroupBy(e => e.PrimaryTool))
        {
            var shuffled = group.OrderBy(_ => random.Next()).ToList();

            // A small group would otherwise donate everything to the held-out
            // sets, so cap each share at a fifth of the group.
            int cap = System.Math.Max(0, shuffled.Count / 5);
            int validationCount = System.Math.Min(validationPerTool, cap);
            int testCount = System.Math.Min(testPerTool, cap);

            validation.AddRange(shuffled.Take(validationCount));
            test.AddRange(shuffled.Skip(validationCount).Take(testCount));
            train.AddRange(shuffled.Skip(validationCount + testCount));
        }

        return (train, validation, test);
    }
}
