using TorchSharp;
using static TorchSharp.torch;
using Needle.Model;
using Needle.Tokenizer;

namespace Needle.Inference;

/// <summary>
/// High-level inference API for the SimpleAttentionNetwork.
/// Wraps encode → greedy-decode → detokenize with optional grammar-constrained
/// decoding and tool-name normalisation.
///
/// Port of <c>generate</c>, <c>generate_batch</c>, <c>encode_for_retrieval</c>,
/// and <c>retrieve_tools</c> in needle/model/run.py.
/// </summary>
public sealed class InferenceRunner : IDisposable
{
    private readonly SimpleAttentionNetwork _model;
    private readonly NeedleTokenizer _tokenizer;
    private readonly TransformerConfig _config;

    /// <summary>The tokenizer this runner was constructed with.</summary>
    public NeedleTokenizer Tokenizer => _tokenizer;

    public InferenceRunner(
        SimpleAttentionNetwork model,
        NeedleTokenizer tokenizer,
        TransformerConfig config)
    {
        _model     = model;
        _tokenizer = tokenizer;
        _config    = config;
    }

    // ── Encoder input construction ────────────────────────────────────────────

    /// <summary>
    /// Build the encoder input token list:
    /// <c>[query_tokens..., &lt;tools&gt;, tool_tokens...]</c> truncated to
    /// <paramref name="maxEncLen"/>.
    ///
    /// Port of Python <c>_build_encoder_input</c> in run.py.
    /// </summary>
    public static int[] BuildEncoderInput(
        NeedleTokenizer tokenizer,
        string query,
        string tools,
        int maxEncLen = 1024)
    {
        int toolsSepId = tokenizer.ToolsTokenId;

        var qToks = new List<int>(tokenizer.Encode(query));
        var tToks = new List<int>(tokenizer.Encode(tools));

        // Truncate query to leave room for sep + at least one tool token
        int maxQuery = maxEncLen - 2;
        if (qToks.Count > maxQuery) qToks.RemoveRange(maxQuery, qToks.Count - maxQuery);

        // Remaining budget for tool tokens
        int remaining = maxEncLen - qToks.Count - 1;
        if (tToks.Count > remaining) tToks.RemoveRange(remaining, tToks.Count - remaining);

        var result = new List<int>(qToks.Count + 1 + tToks.Count);
        result.AddRange(qToks);
        result.Add(toolsSepId);
        result.AddRange(tToks);
        return result.ToArray();
    }

    // ── Single-example generation ─────────────────────────────────────────────

    /// <summary>
    /// Generate a tool-call JSON string for a single query + tools pair.
    /// Port of Python <c>generate()</c> in run.py.
    /// </summary>
    /// <param name="query">Natural-language query.</param>
    /// <param name="tools">JSON array of tool definitions.</param>
    /// <param name="maxGenLen">Maximum decoder output length.</param>
    /// <param name="maxEncLen">Maximum encoder input length.</param>
    /// <param name="normalize">Normalize tool names to snake_case before encoding.</param>
    /// <param name="constrained">Use grammar-constrained decoding.</param>
    /// <param name="progress">Optional progress reporter (receives each decoded token).</param>
    /// <returns>Generated tool-call string (leading &lt;tool_call&gt; stripped).</returns>
    public string Generate(
        string query,
        string tools       = "[]",
        int maxGenLen      = 512,
        int maxEncLen      = 1024,
        bool normalize     = true,
        bool constrained   = true,
        IProgress<string>? progress = null)
    {
        Dictionary<string, string> nameMap = new();
        if (normalize)
            (tools, nameMap) = ToolNormalizer.NormalizeTools(tools);

        // Build encoder input
        int[] encTokens = BuildEncoderInput(_tokenizer, query, tools, maxEncLen);
        using var encInput = torch.tensor(encTokens, dtype: ScalarType.Int64).unsqueeze(0);  // [1, T_enc]

        int padId = _tokenizer.PadTokenId;
        int eosId = _tokenizer.EosTokenId;

        // Encode
        using var srcMask = MaskUtils.MakePaddingMask(encInput, padId);
        var (encoderOut, encMask) = _model.EncodeText(encInput, srcMask);

        // Build fixed-length decoder buffer filled with PAD, first token = EOS
        using var tgtMask  = MaskUtils.MakeCausalMask(maxGenLen);
        var decBuffer = torch.full(new long[] { 1, maxGenLen }, padId, dtype: ScalarType.Int64);
        decBuffer[0, 0] = eosId;

        // Constrained decoder (optional)
        ConstrainedDecoder? cd = constrained
            ? ConstrainedDecodingFactory.BuildConstrainedDecoder([tools], _tokenizer)
            : null;

        var generated = new List<int>();

        // Greedy auto-regressive decode: re-run the decoder over the full
        // buffer after each token is appended.  This mirrors the Python
        // reference (run.py).  There's no KV cache, so each iteration is
        // O(L) in the buffer length; the cost is acceptable for the
        // 32M-param model and small generation lengths.
        Tensor logits = _model.Decode(decBuffer, encoderOut, tgtMask, encMask);

        try
        {
            for (int i = 0; i < maxGenLen - 1; i++)
            {
                using var stepLogits = logits[0, i];   // [vocabSize]

                int nextToken;
                if (cd is not null && cd.IsActive(0))
                {
                    float[] logitsArr = stepLogits.data<float>().ToArray();
                    float[] masked    = cd.ConstrainLogits(logitsArr, 0);
                    nextToken = ArgmaxFloat(masked);
                }
                else
                {
                    nextToken = (int)stepLogits.argmax().item<long>();
                }

                cd?.Update(0, nextToken);

                if (nextToken == eosId) break;

                generated.Add(nextToken);
                decBuffer[0, i + 1] = nextToken;

                string piece = _tokenizer.IdToPiece(nextToken);
                progress?.Report(piece);

                // Re-run the decoder with the updated buffer for the next
                // position's logits.
                logits.Dispose();
                logits = _model.Decode(decBuffer, encoderOut, tgtMask, encMask);
            }
        }
        finally
        {
            logits.Dispose();
            encoderOut.Dispose();
            decBuffer.Dispose();
        }

        string result = _tokenizer.Decode(generated);
        if (result.StartsWith("<tool_call>", StringComparison.Ordinal))
            result = result["<tool_call>".Length..];

        if (normalize && nameMap.Count > 0)
            result = ToolNormalizer.RestoreToolNames(result, nameMap);

        return result;
    }

    // ── Batch generation ──────────────────────────────────────────────────────

    /// <summary>
    /// Generate tool-call JSON strings for a batch of (query, tools) pairs.
    /// Port of Python <c>generate_batch()</c> in run.py.
    /// </summary>
    public IReadOnlyList<string> GenerateBatch(
        IReadOnlyList<string> queries,
        IReadOnlyList<string> toolsList,
        int maxGenLen    = 512,
        int maxEncLen    = 1024,
        bool normalize   = true,
        bool constrained = true)
    {
        int B = queries.Count;
        if (toolsList.Count != B)
            throw new ArgumentException("queries and toolsList must have the same length.");

        // Tool name normalisation
        var nameMaps  = new Dictionary<string, string>[B];
        var normTools = new string[B];
        for (int i = 0; i < B; i++)
        {
            normTools[i] = toolsList[i];
            nameMaps[i]  = new();
            if (normalize)
                (normTools[i], nameMaps[i]) = ToolNormalizer.NormalizeTools(toolsList[i]);
        }

        int padId = _tokenizer.PadTokenId;
        int eosId = _tokenizer.EosTokenId;

        // Build encoder inputs (pad to same length)
        var encTokenLists = new int[B][];
        for (int i = 0; i < B; i++)
            encTokenLists[i] = BuildEncoderInput(_tokenizer, queries[i], normTools[i], maxEncLen);

        int maxEnc = encTokenLists.Max(t => t.Length);
        var encArr  = torch.full(new long[] { B, maxEnc }, padId, dtype: ScalarType.Int64);
        for (int i = 0; i < B; i++)
        {
            for (int j = 0; j < encTokenLists[i].Length; j++)
                encArr[i, j] = encTokenLists[i][j];
        }

        using var srcMask = MaskUtils.MakePaddingMask(encArr, padId);
        var (encoderOut, encMask) = _model.EncodeText(encArr, srcMask);
        encArr.Dispose();

        // Decoder buffer [B, maxGenLen] initialised with PAD, col 0 = EOS
        var decBuffer = torch.full(new long[] { B, maxGenLen }, padId, dtype: ScalarType.Int64);
        for (int i = 0; i < B; i++) decBuffer[i, 0] = eosId;

        using var tgtMask = MaskUtils.MakeCausalMask(maxGenLen);

        // Constrained decoder
        ConstrainedDecoder? cd = constrained
            ? ConstrainedDecodingFactory.BuildConstrainedDecoder(normTools, _tokenizer)
            : null;

        bool[] finished  = new bool[B];
        var genTokens    = Enumerable.Range(0, B).Select(_ => new List<int>()).ToArray();

        // Initial decode
        var logits = _model.Decode(decBuffer, encoderOut, tgtMask, encMask);

        for (int pos = 0; pos < maxGenLen - 1; pos++)
        {
            for (int i = 0; i < B; i++)
            {
                if (finished[i]) continue;

                using var stepLogits = logits[i, pos];  // [vocabSize]

                int nextToken;
                if (cd is not null && cd.IsActive(i))
                {
                    float[] arr    = stepLogits.data<float>().ToArray();
                    float[] masked = cd.ConstrainLogits(arr, i);
                    nextToken = ArgmaxFloat(masked);
                }
                else
                {
                    nextToken = (int)stepLogits.argmax().item<long>();
                }

                cd?.Update(i, nextToken);

                if (nextToken == eosId) { finished[i] = true; continue; }

                genTokens[i].Add(nextToken);
                decBuffer[i, pos + 1] = nextToken;
            }

            if (finished.All(f => f)) break;

            var newLogits = _model.Decode(decBuffer, encoderOut, tgtMask, encMask);
            logits.Dispose();
            logits = newLogits;
        }

        logits.Dispose();
        encoderOut.Dispose();
        decBuffer.Dispose();

        var results = new string[B];
        for (int i = 0; i < B; i++)
        {
            string text = _tokenizer.Decode(genTokens[i]);
            if (text.StartsWith("<tool_call>", StringComparison.Ordinal))
                text = text["<tool_call>".Length..];
            if (normalize && nameMaps[i].Count > 0)
                text = ToolNormalizer.RestoreToolNames(text, nameMaps[i]);
            results[i] = text;
        }
        return results;
    }

    // ── Contrastive retrieval ─────────────────────────────────────────────────

    /// <summary>
    /// Encode a list of texts into contrastive embeddings.
    /// Returns a [N, ContrastiveDim] float array.
    /// Port of Python <c>encode_for_retrieval()</c>.
    /// </summary>
    public float[,] EncodeForRetrieval(
        IReadOnlyList<string> texts,
        int maxLen    = 256,
        int batchSize = 64)
    {
        int N   = texts.Count;
        int dim = _config.ContrastiveDim;
        var result = new float[N, dim];
        int padId = _tokenizer.PadTokenId;

        for (int start = 0; start < N; start += batchSize)
        {
            int end   = System.Math.Min(start + batchSize, N);
            int bSize = end - start;

            var tokenLists = new int[bSize][];
            for (int i = 0; i < bSize; i++)
            {
                var toks = _tokenizer.Encode(texts[start + i]);
                tokenLists[i] = (toks.Count > maxLen
                    ? toks.Take(maxLen)
                    : toks).ToArray();
            }

            int maxT = tokenLists.Max(t => t.Length);
            var tokArr = torch.full(new long[] { bSize, maxT }, padId, dtype: ScalarType.Int64);
            for (int i = 0; i < bSize; i++)
                for (int j = 0; j < tokenLists[i].Length; j++)
                    tokArr[i, j] = tokenLists[i][j];

            using var embs = _model.EncodeContrastive(tokArr);
            tokArr.Dispose();

            // Copy to result[,]
            float[] flat = embs.data<float>().ToArray();
            for (int i = 0; i < bSize; i++)
                for (int j = 0; j < dim; j++)
                    result[start + i, j] = flat[i * dim + j];
        }

        return result;
    }

    /// <summary>
    /// Retrieve the top-k tool descriptions most similar to <paramref name="query"/>
    /// by cosine similarity.
    /// Port of Python <c>retrieve_tools()</c>.
    /// </summary>
    /// <returns>List of (index, score) pairs sorted by descending score.</returns>
    public IReadOnlyList<(int Index, float Score)> RetrieveTools(
        string query,
        IReadOnlyList<string> toolDescriptions,
        int topK   = 5,
        int maxLen = 256)
    {
        float[,] qEmb = EncodeForRetrieval([query], maxLen: maxLen);
        float[,] tEmb = EncodeForRetrieval(toolDescriptions, maxLen: maxLen);

        int N = toolDescriptions.Count;
        int D = _config.ContrastiveDim;

        var scores = new float[N];
        for (int j = 0; j < N; j++)
        {
            float dot = 0f;
            for (int d = 0; d < D; d++)
                dot += qEmb[0, d] * tEmb[j, d];
            scores[j] = dot;
        }

        return scores
            .Select((s, i) => (i, s))
            .OrderByDescending(x => x.s)
            .Take(topK)
            .Select(x => (x.i, x.s))
            .ToList();
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static int ArgmaxFloat(float[] arr)
    {
        int best = 0;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < arr.Length; i++)
        {
            if (arr[i] > bestVal) { bestVal = arr[i]; best = i; }
        }
        return best;
    }

    public void Dispose()
    {
        // model and tokenizer are owned by the caller
    }
}
