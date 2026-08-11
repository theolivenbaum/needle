using System.Text.Json;
using System.Text.Json.Nodes;
using Needle.Math;
using Needle.Model;
using Needle.Tokenizer;

namespace Needle.Inference;

/// <summary>One call the model asked for.</summary>
/// <param name="Name">Tool name.</param>
/// <param name="Arguments">Arguments, as parsed from the emitted JSON.</param>
public sealed record FunctionCall(string Name, JsonObject Arguments);

/// <summary>
/// One turn's answer.  Mirrors the JSON object the reference package returns
/// from <c>Needle.complete()</c>.
/// </summary>
/// <param name="Type"><c>call</c> when the model asked for tools, else <c>respond</c>.</param>
/// <param name="Success">False when the emitted call could not be parsed.</param>
/// <param name="Error">Why parsing failed, when it did.</param>
/// <param name="FunctionCalls">The calls; empty for a refusal or a plain answer.</param>
/// <param name="Reasoning">The model's short derivation of each argument.</param>
/// <param name="Text">Raw generated text, before parsing.</param>
/// <param name="Confidence">
/// Calibrated confidence in [0, 1]: the smaller of the confidence head's score
/// and the decoding probability of the emitted call.
/// </param>
/// <param name="PromptTokens">Prompt length in tokens.</param>
/// <param name="GeneratedTokens">Number of tokens emitted.</param>
public sealed record NeedleResponse(
    string Type,
    bool Success,
    string? Error,
    IReadOnlyList<FunctionCall> FunctionCalls,
    string Reasoning,
    string Text,
    float Confidence,
    int PromptTokens,
    int GeneratedTokens);

/// <summary>
/// Session-level API over <see cref="NeedleSession"/>: renders the prompt the
/// model was trained on, generates, and parses the emitted call.
///
/// Follows the contract described in the upstream README — one toolset per
/// session, later turns are bare queries against the same tools, a request no
/// declared tool can serve comes back as the empty call <c>[]</c>, and the tool
/// block is pinned as a KV sink so it stays visible however long the
/// conversation runs.
/// </summary>
public sealed class NeedleAgent
{
    private readonly Needle2Model _model;
    private readonly CactTokenizer _tokenizer;
    private readonly string _toolsJson;
    private readonly string? _system;
    private readonly string[]? _tokenStrings;
    private readonly TokenIndex? _tokenIndex;
    private NeedleSession _session;
    private bool _started;

    /// <summary>
    /// Whether the emitted call is constrained to the declared schemas.
    ///
    /// With this on, tool names and argument keys can only ever be ones the
    /// schemas declare — an unsupported argument becomes impossible rather than
    /// merely unlikely.  The derivation stays unconstrained, so it remains
    /// legible free text.
    /// </summary>
    public bool Constrained { get; }

    /// <summary>The tokenizer this agent renders with.</summary>
    public CactTokenizer Tokenizer => _tokenizer;

    /// <summary>The underlying model.</summary>
    public Needle2Model Model => _model;

    /// <param name="model">Loaded model.</param>
    /// <param name="tokenizer">Matching tokenizer, normally from the same blob.</param>
    /// <param name="toolsJson">Declared schemas as a JSON array string.</param>
    /// <param name="system">Optional system facts (never instructions).</param>
    /// <param name="constrained">
    /// Compile the schemas into a decode-time grammar. On by default; turn it off
    /// to see what the model would emit unconstrained.
    /// </param>
    public NeedleAgent(Needle2Model model, CactTokenizer tokenizer, string toolsJson,
                       string? system = null, bool constrained = true)
    {
        _model = model;
        _tokenizer = tokenizer;
        _toolsJson = toolsJson;
        _system = system;
        _session = new NeedleSession(model);
        Constrained = constrained;

        if (constrained)
        {
            // The token→surface table and its first-character index depend only
            // on the vocabulary, so build them once per agent rather than once
            // per turn.
            _tokenStrings = ConstrainedDecodingFactory.BuildTokenStrings(tokenizer);
            _tokenIndex = new TokenIndex(_tokenStrings);
        }
    }

    /// <summary>Rewind the conversation, keeping the tools loaded.</summary>
    public void Reset()
    {
        _session = new NeedleSession(_model);
        _started = false;
    }

    /// <summary>
    /// Run one turn.  The first call renders the full prompt including the tool
    /// block; later calls append a bare turn against the same tools.
    /// </summary>
    /// <param name="text">The query, or a tool result to continue from.</param>
    /// <param name="maxNewTokens">Cap on emitted tokens.</param>
    public NeedleResponse Complete(string text, int maxNewTokens = 256)
    {
        string prompt = _started
            ? ChatTemplate.ResultTurn(text)
            : ChatTemplate.Prompt(text, _toolsJson, _system);

        var promptIds = _started ? _tokenizer.Encode(prompt) : _tokenizer.EncodeWithBos(prompt);
        if (!_started)
        {
            // Pin the tool block: everything up to the end of <tools> stays
            // visible after it slides out of the window.
            int toolsEnd = promptIds.IndexOf(ChatMarkers.ToolsEndId);
            _session.PinPrefix(toolsEnd >= 0 ? toolsEnd + 1 : promptIds.Count);
            _started = true;
        }

        var grammar = _tokenStrings is not null && _tokenIndex is not null
            ? new ConstrainedDecoder([new ToolConstraints(_toolsJson)], _tokenStrings, _tokenIndex)
            : null;

        var generated = new List<int>(maxNewTokens);
        double logProbSum = 0;
        int scored = 0;

        var logits = _session.Advance(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(promptIds));
        for (int i = 0; i < maxNewTokens; i++)
        {
            var scores = logits.ReadSpan;
            int next;
            if (grammar is not null && grammar.IsActive(0))
            {
                var masked = grammar.ConstrainLogits(scores.ToArray(), 0);
                next = Ops.ArgMax(masked);
            }
            else
            {
                next = Ops.ArgMax(scores);
            }

            // Score against the unconstrained distribution: confidence should
            // reflect what the model believed, not what the grammar allowed.
            logProbSum += TokenLogProbability(scores, next);
            scored++;
            grammar?.Update(0, next);

            if (next == _tokenizer.EosId || next == ChatMarkers.ImEndId) break;
            generated.Add(next);
            if (i == maxNewTokens - 1) break;
            logits = _session.Advance(next);
        }

        string output = _tokenizer.Decode(generated);
        float decodeConfidence = scored > 0 ? (float)System.Math.Exp(logProbSum / scored) : 0f;
        return Parse(output, decodeConfidence, promptIds.Count, generated.Count);
    }

    /// <summary>
    /// Score a full prompt-plus-call sequence with the confidence head.  Returns
    /// a probability in [0, 1]; the model's contract is to act above your
    /// threshold and escalate below it.
    /// </summary>
    public float ScoreConfidence(string query, string completion)
    {
        if (_model.Weights.Confidence is null) return float.NaN;

        var ids = _tokenizer.EncodeWithBos(ChatTemplate.Prompt(query, _toolsJson, _system) + completion);
        int window = KvBudget.EffectiveWindow(_model.Config);
        return Needle2Model.Sigmoid(
            _model.ConfidenceLogit(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(ids), window));
    }

    /// <summary>
    /// Rank tool descriptions against a query with the contrastive head.
    ///
    /// Above five declared tools the reference renders only the top five into the
    /// context, so an unselected tool becomes unreachable rather than merely
    /// unlikely.
    /// </summary>
    /// <param name="query">The user's request.</param>
    /// <param name="toolTexts">One text per tool — its schema or description.</param>
    /// <param name="topK">How many to return.</param>
    /// <returns>Indices and cosine scores, best first.</returns>
    public IReadOnlyList<(int Index, float Score)> RetrieveTools(
        string query, IReadOnlyList<string> toolTexts, int topK = 5)
    {
        if (_model.Weights.Contrastive is null)
            throw new InvalidOperationException("This checkpoint has no contrastive head.");

        var queryEmbedding = Encode(query);
        var scores = new (int Index, float Score)[toolTexts.Count];
        for (int i = 0; i < toolTexts.Count; i++)
        {
            var candidate = Encode(toolTexts[i]);
            float dot = 0f;
            for (int d = 0; d < candidate.Length; d++) dot += queryEmbedding[d] * candidate[d];
            scores[i] = (i, dot);
        }

        Array.Sort(scores, (a, b) => b.Score.CompareTo(a.Score));
        return scores.Take(System.Math.Min(topK, scores.Length)).ToArray();
    }

    private NdArray Encode(string text)
    {
        var ids = _tokenizer.EncodeWithBos(text);
        return _model.EncodeContrastive(
            System.Runtime.InteropServices.CollectionsMarshal.AsSpan(ids));
    }

    private static double TokenLogProbability(ReadOnlySpan<float> logits, int token)
    {
        float max = float.NegativeInfinity;
        foreach (float value in logits) max = System.Math.Max(max, value);
        double sum = 0;
        foreach (float value in logits) sum += System.Math.Exp(value - max);
        return logits[token] - max - System.Math.Log(sum);
    }

    // ── Parsing ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Split the generated text into its derivation and its call array.
    /// The call itself is grammar-constrained during decoding, so the JSON
    /// cannot be malformed while the derivation stays free text.
    /// </summary>
    public static NeedleResponse Parse(string text, float confidence, int promptTokens, int generatedTokens)
    {
        string reasoning = Between(text, ChatMarkers.ThinkStart, ChatMarkers.ThinkEnd).Trim();
        string callJson = Between(text, ChatMarkers.ToolCallStart, ChatMarkers.ToolCallEnd).Trim();

        if (callJson.Length == 0)
        {
            // No call block: the model answered in plain text.
            string body = text;
            int thinkEnd = text.IndexOf(ChatMarkers.ThinkEnd, StringComparison.Ordinal);
            if (thinkEnd >= 0) body = text[(thinkEnd + ChatMarkers.ThinkEnd.Length)..];
            return new NeedleResponse("respond", true, null, [], reasoning, body.Trim(),
                                      confidence, promptTokens, generatedTokens);
        }

        try
        {
            var calls = new List<FunctionCall>();
            if (JsonNode.Parse(callJson) is JsonArray array)
            {
                foreach (var entry in array)
                {
                    if (entry is not JsonObject obj) continue;
                    string name = obj["name"]?.GetValue<string>() ?? "";
                    var arguments = obj["arguments"] as JsonObject ?? new JsonObject();
                    calls.Add(new FunctionCall(name, (JsonObject)arguments.DeepClone()));
                }
            }
            return new NeedleResponse("call", true, null, calls, reasoning, text,
                                      confidence, promptTokens, generatedTokens);
        }
        catch (JsonException ex)
        {
            return new NeedleResponse("call", false, ex.Message, [], reasoning, text,
                                      confidence, promptTokens, generatedTokens);
        }
    }

    private static string Between(string text, string open, string close)
    {
        int start = text.IndexOf(open, StringComparison.Ordinal);
        if (start < 0) return "";
        start += open.Length;
        int end = text.IndexOf(close, start, StringComparison.Ordinal);
        return end < 0 ? text[start..] : text[start..end];
    }
}
