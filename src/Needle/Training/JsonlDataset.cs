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

/// <summary>
/// Token-class labels used to bias the cross-entropy loss in
/// <c>training/train.py</c>:
/// 0 = base, 1 = tool name, 2 = argument value, 3 = argument key.
/// </summary>
public static class TokenClass
{
    public const int Base  = 0;
    public const int Name  = 1;
    public const int Value = 2;
    public const int Key   = 3;
}

/// <summary>
/// Build encoder and decoder token sequences (plus class labels) for one
/// <see cref="FinetuneExample"/>, ready to be assembled into a
/// <see cref="TrainingBatch"/>.
///
/// Mirrors the per-example logic inside <c>prepare_tool_call_pairs</c> in
/// <c>needle/dataset/dataset.py</c>.
/// </summary>
public static class FinetuneExampleBuilder
{
    private const string ToolCallText = "<tool_call>";
    private const string ToolsSepText = "<tools>";

    /// <summary>
    /// Result of tokenising one example.
    /// </summary>
    /// <param name="EncTokens">Encoder tokens: <c>[query…, &lt;tools&gt;, tools…]</c></param>
    /// <param name="DecInTokens">Decoder input tokens: <c>[EOS, &lt;tool_call&gt;, answer…]</c></param>
    /// <param name="DecOutTokens">Decoder target tokens: <c>[&lt;tool_call&gt;, answer…, EOS]</c></param>
    /// <param name="ClassLabels">Per-decoder-target-position class label (0..3).</param>
    public sealed record Built(
        int[] EncTokens,
        int[] DecInTokens,
        int[] DecOutTokens,
        int[] ClassLabels);

    /// <summary>
    /// Tokenise one example.  Returns <c>null</c> if the answer is too long for
    /// the decoder budget (the Python pipeline drops these examples).
    /// </summary>
    public static Built? Build(
        INeedleTokenizer tokenizer,
        FinetuneExample example,
        int maxEncLen = 1024,
        int maxDecLen = 512)
    {
        int padId       = tokenizer.PadTokenId;
        int eosId       = tokenizer.EosTokenId;
        int toolsSepId  = tokenizer.ToolsTokenId;
        int toolCallId  = tokenizer.ToolCallTokenId;

        var qToks = tokenizer.Encode(example.Query).ToList();
        var tToks = tokenizer.Encode(example.Tools).ToList();
        var aToks = tokenizer.Encode(example.Answers).ToList();

        int maxQuery = maxEncLen - 2;
        if (qToks.Count > maxQuery) qToks.RemoveRange(maxQuery, qToks.Count - maxQuery);
        int remaining = maxEncLen - qToks.Count - 1;
        if (tToks.Count > remaining) tToks.RemoveRange(remaining, tToks.Count - remaining);

        var enc = new List<int>(qToks.Count + 1 + tToks.Count);
        enc.AddRange(qToks);
        enc.Add(toolsSepId);
        enc.AddRange(tToks);

        // Decoder input needs 2 special tokens + the answer; target needs
        // 1 special + answer + EOS.  Both must fit in maxDecLen.
        if (2 + aToks.Count + 1 > maxDecLen)
            return null;

        var decIn  = new int[2 + aToks.Count];
        var decOut = new int[1 + aToks.Count + 1];

        decIn[0]  = eosId;
        decIn[1]  = toolCallId;
        for (int i = 0; i < aToks.Count; i++) decIn[i + 2] = aToks[i];

        decOut[0] = toolCallId;
        for (int i = 0; i < aToks.Count; i++) decOut[i + 1] = aToks[i];
        decOut[decOut.Length - 1] = eosId;

        var classes = ComputeAnswerClassLabels(tokenizer, example.Answers, aToks, decOut.Length);
        return new Built(enc.ToArray(), decIn, decOut, classes);
    }

    /// <summary>
    /// Compute per-token class labels (0=base, 1=name, 2=value, 3=key) for the
    /// decoder target sequence.  Falls back to all-zeros if the answer cannot
    /// be parsed.
    ///
    /// Simplified port of <c>_token_classes_for_answer</c> in dataset.py.
    /// </summary>
    public static int[] ComputeAnswerClassLabels(
        INeedleTokenizer tokenizer,
        string answersJson,
        IReadOnlyList<int> answerTokens,
        int totalLen)
    {
        // The decoder target has shape [<tool_call>, answer..., EOS] so the
        // class array is also of length totalLen.  Class labels live at
        // positions [1 .. 1 + answerTokens.Count - 1].  The leading
        // <tool_call> and trailing EOS get class 0 (base).
        var classes = new int[totalLen];

        if (answerTokens.Count == 0) return classes;

        // Character-level class array over the original answer string.
        int len = answersJson.Length;
        var charCls = new byte[len];

        try
        {
            var node = JsonNode.Parse(answersJson);
            if (node is not JsonArray calls) return classes;

            foreach (var c in calls)
            {
                if (c is not JsonObject call) continue;
                if (call["name"]?.GetValue<string>() is string name && name.Length > 0)
                    MarkJsonValue(answersJson, charCls, "name", name, TokenClass.Name);
                if (call["arguments"] is JsonObject argsObj)
                {
                    foreach (var (k, v) in argsObj)
                    {
                        MarkJsonKeyInArgs(answersJson, charCls, k, TokenClass.Key);
                        string vStr = v is JsonValue jv && jv.TryGetValue(out string? s) ? s! : v?.ToJsonString() ?? "";
                        MarkJsonValue(answersJson, charCls, k, vStr, TokenClass.Value);
                    }
                }
            }
        }
        catch (JsonException)
        {
            return classes;
        }

        // Walk through tokens and aggregate the max char class touched by each
        // piece.  We need to know each piece's text length — re-encode using a
        // SentencePiece-style walk.  Since IdToPiece returns the piece with
        // ▁ → space, we can lay them down sequentially.
        int pos = 0;
        for (int i = 0; i < answerTokens.Count; i++)
        {
            string piece = tokenizer.IdToPiece(answerTokens[i]);
            int pieceLen = piece.Length;
            byte best = 0;
            if (pieceLen > 0 && pos + pieceLen <= len)
            {
                for (int j = 0; j < pieceLen; j++)
                    if (charCls[pos + j] > best) best = charCls[pos + j];
                pos += pieceLen;
            }
            else
            {
                pos += System.Math.Max(pieceLen, 1);
            }
            classes[1 + i] = best;
        }

        return classes;
    }

    private static void MarkJsonValue(string s, byte[] charCls, string key, string value, int label)
    {
        // Try quoted form first: "key": "value"
        string keyEsc  = Regex.Escape(key);
        string valEsc  = Regex.Escape(value);

        var rxQuoted = new Regex($"\"{keyEsc}\"\\s*:\\s*\"{valEsc}\"");
        var match = rxQuoted.Match(s);
        if (match.Success)
        {
            // Find the value start (after the opening quote of the value).
            int searchStart = match.Index + ($"\"{key}\"").Length;
            int valStart    = s.IndexOf('"', searchStart) + 1;
            int valEnd      = valStart + value.Length;
            for (int i = valStart; i < valEnd && i < charCls.Length; i++)
                if (charCls[i] < label) charCls[i] = (byte)label;
            return;
        }

        // Unquoted form: "key": value
        var rxBare = new Regex($"\"{keyEsc}\"\\s*:\\s*{valEsc}");
        match = rxBare.Match(s);
        if (match.Success)
        {
            int colonIdx = s.IndexOf(':', match.Index);
            int valStart = colonIdx + 1;
            while (valStart < match.Index + match.Length && s[valStart] == ' ') valStart++;
            int valEnd = match.Index + match.Length;
            for (int i = valStart; i < valEnd && i < charCls.Length; i++)
                if (charCls[i] < label) charCls[i] = (byte)label;
        }
    }

    private static void MarkJsonKeyInArgs(string s, byte[] charCls, string key, int label)
    {
        string keyEsc = Regex.Escape(key);
        var rx = new Regex($"\"{keyEsc}\"\\s*:");
        foreach (Match m in rx.Matches(s))
        {
            int start = m.Index + 1;
            int end   = start + key.Length;
            for (int i = start; i < end && i < charCls.Length; i++)
                if (charCls[i] < label) charCls[i] = (byte)label;
        }
    }
}

/// <summary>
/// Assemble <see cref="FinetuneExample"/>s into padded
/// <see cref="TrainingBatch"/> records that the <see cref="Trainer"/>
/// can consume directly.
///
/// Single-example packing: every batch row holds one example with
/// segment ID 1 for the real tokens and 0 for padding.  This produces correct
/// padding masks; full multi-example packing (the Python `pack_sequences`
/// path) is out of scope for the local finetune flow.
/// </summary>
public static class BatchBuilder
{
    /// <summary>
    /// Build a single padded batch from up to <c>batchSize</c> examples.
    /// </summary>
    /// <param name="examples">Source examples (only the first <c>batchSize</c> are used).</param>
    /// <param name="tokenizer">Tokeniser to use.</param>
    /// <param name="batchSize">Target batch size.  Smaller batches are returned if fewer examples remain.</param>
    /// <param name="maxEncLen">Encoder budget per row.</param>
    /// <param name="maxDecLen">Decoder budget per row.</param>
    public static TrainingBatch? BuildBatch(
        IList<FinetuneExample> examples,
        INeedleTokenizer tokenizer,
        int batchSize,
        int maxEncLen = 1024,
        int maxDecLen = 512)
    {
        int padId = tokenizer.PadTokenId;
        int take  = System.Math.Min(batchSize, examples.Count);
        if (take == 0) return null;

        var built = new List<FinetuneExampleBuilder.Built>();
        for (int i = 0; i < take; i++)
        {
            var b = FinetuneExampleBuilder.Build(tokenizer, examples[i], maxEncLen, maxDecLen);
            if (b is not null) built.Add(b);
        }
        if (built.Count == 0) return null;

        int B = built.Count;
        int encLen = built.Max(b => b.EncTokens.Length);
        int decLen = built.Max(b => System.Math.Max(b.DecInTokens.Length, b.DecOutTokens.Length));

        var src    = new long[B, encLen];
        var tgtIn  = new long[B, decLen];
        var tgtOut = new long[B, decLen];
        var lm     = new int [B, decLen];
        var encSeg = new int [B, encLen];
        var decSeg = new int [B, decLen];

        for (int i = 0; i < B; i++) for (int j = 0; j < encLen; j++) src[i, j]    = padId;
        for (int i = 0; i < B; i++) for (int j = 0; j < decLen; j++) tgtIn[i, j]  = padId;
        for (int i = 0; i < B; i++) for (int j = 0; j < decLen; j++) tgtOut[i, j] = padId;

        for (int i = 0; i < B; i++)
        {
            var b = built[i];
            for (int j = 0; j < b.EncTokens.Length; j++)
            {
                src[i, j]    = b.EncTokens[j];
                encSeg[i, j] = 1;
            }
            for (int j = 0; j < b.DecInTokens.Length; j++)
                tgtIn[i, j] = b.DecInTokens[j];
            for (int j = 0; j < b.DecOutTokens.Length; j++)
            {
                tgtOut[i, j] = b.DecOutTokens[j];
                decSeg[i, j] = 1;
                lm[i, j]     = b.ClassLabels[j];
            }
        }

        return new TrainingBatch(src, tgtIn, tgtOut, lm, encSeg, decSeg);
    }

    /// <summary>
    /// Enumerate <paramref name="examples"/> in non-overlapping batches.
    /// </summary>
    public static IEnumerable<TrainingBatch> Iterate(
        IList<FinetuneExample> examples,
        INeedleTokenizer tokenizer,
        int batchSize,
        int maxEncLen = 1024,
        int maxDecLen = 512)
    {
        for (int start = 0; start < examples.Count; start += batchSize)
        {
            int end   = System.Math.Min(start + batchSize, examples.Count);
            var slice = examples.Skip(start).Take(end - start).ToList();
            var batch = BuildBatch(slice, tokenizer, batchSize, maxEncLen, maxDecLen);
            if (batch is not null) yield return batch;
        }
    }
}
