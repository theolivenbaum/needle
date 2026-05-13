using System.Text.Json;
using Needle.Tokenizer;

namespace Needle.Inference;

/// <summary>
/// Utilities for normalising tool names to snake_case before feeding them
/// to the model, and for restoring the original names in the model's output.
///
/// Port of <c>normalize_tools</c> and <c>restore_tool_names</c> from
/// needle/model/run.py.
/// </summary>
public static class ToolNormalizer
{
    // ── Serialiser options ────────────────────────────────────────────────────

    // Compact separators — matches Python json.dumps(separators=(",", ":")).
    private static readonly JsonSerializerOptions _compactOptions = new()
    {
        WriteIndented = false,
    };

    // ── Public API ────────────────────────────────────────────────────────────

    /// <summary>
    /// Normalise tool names in <paramref name="toolsJson"/> to snake_case.
    ///
    /// Returns a tuple of:
    /// <list type="bullet">
    ///   <item><c>Json</c>: the serialised JSON with snake_case names.</item>
    ///   <item><c>NameMap</c>: a map from snake_case name → original name,
    ///         used to restore names in the model output.</item>
    /// </list>
    ///
    /// If <paramref name="toolsJson"/> is not valid JSON the original string
    /// is returned unchanged with an empty map.
    /// </summary>
    public static (string Json, Dictionary<string, string> NameMap) NormalizeTools(string toolsJson)
    {
        JsonElement root;
        try
        {
            root = JsonSerializer.Deserialize<JsonElement>(toolsJson);
        }
        catch
        {
            return (toolsJson, new Dictionary<string, string>());
        }

        if (root.ValueKind != JsonValueKind.Array)
            return (toolsJson, new Dictionary<string, string>());

        var nameMap = new Dictionary<string, string>();
        var tools   = new List<JsonElement>();

        foreach (var tool in root.EnumerateArray())
        {
            if (tool.ValueKind != JsonValueKind.Object
                || !tool.TryGetProperty("name", out var nameProp)
                || nameProp.ValueKind != JsonValueKind.String)
            {
                tools.Add(tool);
                continue;
            }

            string original = nameProp.GetString() ?? string.Empty;
            string snake    = NeedleTokenizer.ToSnakeCase(original);
            nameMap[snake]  = original;

            // Rebuild the tool object with the snake_case name.
            tools.Add(RebuildWithName(tool, snake));
        }

        // Serialise back to compact JSON.
        using var ms     = new System.IO.MemoryStream();
        using var writer = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = false });
        writer.WriteStartArray();
        foreach (var t in tools)
            t.WriteTo(writer);
        writer.WriteEndArray();
        writer.Flush();

        string json = System.Text.Encoding.UTF8.GetString(ms.ToArray());
        return (json, nameMap);
    }

    /// <summary>
    /// Replace snake_case tool names in <paramref name="predText"/> (the raw
    /// model output) with the original names recorded in <paramref name="nameMap"/>.
    ///
    /// Attempts to parse <paramref name="predText"/> as JSON first; falls back
    /// to string-level replacement (longest names first) if parsing fails.
    ///
    /// Port of Python <c>restore_tool_names</c> in run.py.
    /// </summary>
    public static string RestoreToolNames(string predText, Dictionary<string, string> nameMap)
    {
        if (nameMap is null || nameMap.Count == 0)
            return predText;

        JsonElement root;
        try
        {
            root = JsonSerializer.Deserialize<JsonElement>(predText);
        }
        catch
        {
            // Fallback: string replacement, longest snake names first.
            return StringFallbackRestore(predText, nameMap);
        }

        if (root.ValueKind == JsonValueKind.Array)
        {
            var items = new List<JsonElement>();
            foreach (var item in root.EnumerateArray())
                items.Add(RestoreNameInCall(item, nameMap));

            using var ms     = new System.IO.MemoryStream();
            using var writer = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = false });
            writer.WriteStartArray();
            foreach (var item in items)
                item.WriteTo(writer);
            writer.WriteEndArray();
            writer.Flush();
            return System.Text.Encoding.UTF8.GetString(ms.ToArray());
        }

        if (root.ValueKind == JsonValueKind.Object)
        {
            var restored = RestoreNameInCall(root, nameMap);
            using var ms     = new System.IO.MemoryStream();
            using var writer = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = false });
            restored.WriteTo(writer);
            writer.Flush();
            return System.Text.Encoding.UTF8.GetString(ms.ToArray());
        }

        return predText;
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// <summary>
    /// Return a copy of <paramref name="call"/> with its "name" field replaced
    /// by the original name from <paramref name="nameMap"/> (if present).
    /// </summary>
    private static JsonElement RestoreNameInCall(
        JsonElement call, Dictionary<string, string> nameMap)
    {
        if (call.ValueKind != JsonValueKind.Object
            || !call.TryGetProperty("name", out var nameProp)
            || nameProp.ValueKind != JsonValueKind.String)
        {
            return call;
        }

        string snake = nameProp.GetString() ?? string.Empty;
        if (!nameMap.TryGetValue(snake, out string? original))
            return call;

        return RebuildWithName(call, original);
    }

    /// <summary>
    /// Rebuild a JSON object, replacing its "name" property value with
    /// <paramref name="newName"/>.
    /// </summary>
    private static JsonElement RebuildWithName(JsonElement obj, string newName)
    {
        using var ms     = new System.IO.MemoryStream();
        using var writer = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = false });
        writer.WriteStartObject();
        // Write "name" first, then all other properties.
        writer.WriteString("name", newName);
        foreach (var prop in obj.EnumerateObject())
        {
            if (prop.Name == "name")
                continue;
            prop.WriteTo(writer);
        }
        writer.WriteEndObject();
        writer.Flush();

        var bytes = ms.ToArray();
        var doc   = JsonDocument.Parse(bytes);
        return doc.RootElement.Clone();
    }

    /// <summary>
    /// String-level fallback: replace snake_case names longest-first.
    /// </summary>
    private static string StringFallbackRestore(
        string text, Dictionary<string, string> nameMap)
    {
        // Process longest snake names first to avoid partial matches.
        foreach (var (snake, original) in
            nameMap.OrderByDescending(kvp => kvp.Key.Length))
        {
            text = text.Replace(snake, original, StringComparison.Ordinal);
        }
        return text;
    }
}
