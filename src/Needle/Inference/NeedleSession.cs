using Needle.Math;
using Needle.Model;

namespace Needle.Inference;

/// <summary>
/// A single decoding session: token history, per-layer KV caches, and the
/// sliding window with its pinned sinks.
///
/// Port of the KV-cached path in <c>.reference/needle/model/decode.py</c>.
/// A step costs a fixed amount of work regardless of how long the conversation
/// has run: attention reads at most <see cref="Window"/> cached positions plus
/// the pinned sinks, and the engram sites re-read only the short trailing window
/// their convolution taps can reach.
/// </summary>
public sealed class NeedleSession
{
    private readonly Needle2Model _model;
    private readonly TransformerConfig _cfg;
    private readonly LayerKvCache[] _caches;
    private readonly int[] _history;
    private readonly bool[] _valid;
    private readonly bool[] _sink;
    private readonly int _engramWindow;

    /// <summary>Positions consumed so far.</summary>
    public int Length { get; private set; }

    /// <summary>Maximum number of positions this session can hold.</summary>
    public int Capacity { get; }

    /// <summary>Sliding-window width in effect; 0 means unbounded.</summary>
    public int Window { get; }

    /// <summary>The model being decoded.</summary>
    public Needle2Model Model => _model;

    /// <param name="model">Model to decode with.</param>
    /// <param name="capacity">
    /// Maximum session length.  Defaults to the model's <c>MaxSeqLen</c>.
    /// </param>
    /// <param name="window">
    /// Sliding-window width.  Defaults to the checkpoint's KV budget window; pass
    /// -1 for unbounded attention (which is what the batched forward pass does).
    /// </param>
    public NeedleSession(Needle2Model model, int capacity = 0, int window = 0)
    {
        _model = model;
        _cfg = model.Config;
        Capacity = capacity > 0 ? capacity : _cfg.MaxSeqLen;
        Window = window < 0 ? 0 : (window > 0 ? window : KvBudget.EffectiveWindow(_cfg));

        _caches = new LayerKvCache[_cfg.NumLayers];
        for (int i = 0; i < _caches.Length; i++) _caches[i] = new LayerKvCache(Capacity, _cfg.KvDim);

        _history = new int[Capacity];
        _valid = new bool[Capacity];
        _sink = new bool[Capacity];

        var orders = _cfg.EngramOrders;
        _engramWindow = orders.Length == 0 ? 0 : EngramConstants.ConvTaps * orders.Max();
    }

    /// <summary>Forget the conversation, keeping the allocated buffers.</summary>
    public void Reset()
    {
        Length = 0;
        foreach (var cache in _caches) cache.Reset();
        Array.Clear(_history);
        Array.Clear(_valid);
        Array.Clear(_sink);
    }

    /// <summary>
    /// Pin positions <c>[0, count)</c> as KV sinks: they stay visible after they
    /// fall out of the sliding window.  Needle 2 pins the tool block this way.
    /// </summary>
    public void PinPrefix(int count)
    {
        for (int i = 0; i < System.Math.Min(count, Capacity); i++) _sink[i] = true;
    }

    /// <summary>
    /// Consume <paramref name="tokens"/> and return the next-token logits from
    /// the final position.  Works for a whole prompt or for a single token.
    /// </summary>
    public NdArray Advance(ReadOnlySpan<int> tokens)
    {
        if (tokens.Length == 0)
            throw new ArgumentException("Nothing to advance.", nameof(tokens));
        if (Length + tokens.Length > Capacity)
            throw new InvalidOperationException(
                $"Session holds {Capacity} positions; cannot add {tokens.Length} to {Length}.");

        int start = Length;
        tokens.CopyTo(_history.AsSpan(start));
        for (int i = 0; i < tokens.Length; i++) _valid[start + i] = true;
        Length += tokens.Length;

        var mask = new SequenceMask(Length, Window, _valid[..Length], _sink[..Length]);
        var plan = AttentionPlan.Build(mask, start, tokens.Length, Length);
        var engram = BuildEngram(start, tokens.Length);

        var result = _model.RunStack(tokens, plan, engram, start, _caches);
        return _model.LastLogits(result.Hidden);
    }

    /// <summary>Consume a single token and return the next-token logits.</summary>
    public NdArray Advance(int token)
    {
        Span<int> one = [token];
        return Advance(one);
    }

    /// <summary>
    /// Engram keys and values for positions <c>[start, start + count)</c>,
    /// computed over the short trailing window the sites can actually reach.
    /// </summary>
    private Needle2Model.EngramActivations? BuildEngram(int start, int count)
    {
        if (_engramWindow == 0) return null;

        int warmup = System.Math.Min(_engramWindow, start);
        int lo = start - warmup;
        int pad = _engramWindow - warmup;
        int length = _engramWindow + count;

        var windowTokens = new int[length];
        var windowValid = new bool[length];
        for (int i = 0; i < warmup + count; i++)
        {
            windowTokens[pad + i] = _history[lo + i];
            windowValid[pad + i] = _valid[lo + i];
        }

        return _model.EngramWindow(windowTokens, windowValid, trim: _engramWindow);
    }

    /// <summary>
    /// Greedily continue the session until <paramref name="stopToken"/> or
    /// <paramref name="maxNewTokens"/>.
    /// </summary>
    /// <param name="prompt">Prompt tokens; pass an empty span to continue.</param>
    /// <param name="maxNewTokens">Maximum tokens to emit.</param>
    /// <param name="stopToken">Token that ends generation, or -1 for none.</param>
    /// <param name="onToken">Optional per-token callback.</param>
    /// <returns>The generated token IDs, excluding the stop token.</returns>
    public List<int> Generate(
        ReadOnlySpan<int> prompt,
        int maxNewTokens = 96,
        int stopToken = -1,
        Action<int>? onToken = null)
    {
        var logits = prompt.Length > 0 ? Advance(prompt) : throw new ArgumentException(
            "Generate needs at least one prompt token.", nameof(prompt));

        var generated = new List<int>(maxNewTokens);
        for (int i = 0; i < maxNewTokens; i++)
        {
            int next = Ops.ArgMax(logits.ReadSpan);
            if (next == stopToken || Length >= Capacity) break;

            generated.Add(next);
            onToken?.Invoke(next);
            if (i == maxNewTokens - 1) break;

            logits = Advance(next);
        }
        return generated;
    }
}
