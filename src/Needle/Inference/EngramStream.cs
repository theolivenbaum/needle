using System.Numerics.Tensors;
using Needle.Math;
using Needle.Model;

namespace Needle.Inference;

/// <summary>
/// Streams the engram sites across a session, keeping only what the causal
/// convolution can still reach.
///
/// A site's value is a four-tap convolution over past projections, so a step
/// needs the last <c>ConvTaps * maxOrder</c> — twelve — pre-convolution rows.
/// The reference recomputes that whole window on every step, which is free under
/// a JIT that vectorises it; here it means thirteen row projections per token
/// instead of one. Holding those twelve rows in a ring buffer makes a step cost
/// exactly one row per site, and removes the largest remaining per-token
/// allocation.
/// </summary>
public sealed class EngramStream
{
    private readonly Needle2Model _model;
    private readonly int _sites;
    private readonly int _width;
    private readonly int _taps;
    private readonly int _dilation;

    /// <summary>Pre-convolution projections, per site, as a ring of the last <c>_span</c> rows.</summary>
    private readonly NdArray[] _history;
    private readonly int _span;

    /// <summary>Absolute position of the next row to be written.</summary>
    private int _position;

    /// <summary>Reused destination for the projections a call produces.</summary>
    private readonly NdArray _fresh;

    /// <param name="model">Model whose engram weights this streams.</param>
    /// <param name="maxBatch">Largest number of positions consumed in one call.</param>
    public EngramStream(Needle2Model model, int maxBatch)
    {
        _model = model;
        _sites = model.Config.EngramSites;
        _width = model.Config.DModel;
        _taps = EngramConstants.ConvTaps;
        _dilation = model.Config.EngramOrders.Length == 0 ? 1 : model.Config.EngramOrders.Max();

        // Enough room for the reach of the taps plus whatever a prefill adds.
        _span = _taps * _dilation + System.Math.Max(1, maxBatch);
        _history = new NdArray[_sites];
        for (int s = 0; s < _sites; s++) _history[s] = new NdArray(_span, _width);
        _fresh = new NdArray(System.Math.Max(1, maxBatch), _width);
    }

    /// <summary>Number of engram sites.</summary>
    public int Sites => _sites;

    /// <summary>Forget the conversation.</summary>
    public void Reset()
    {
        _position = 0;
        foreach (var site in _history) site.Span.Clear();
    }

    /// <summary>
    /// Produce the engram keys and values for positions
    /// <c>[startPosition, startPosition + count)</c>.
    /// </summary>
    /// <param name="history">The session's full token history.</param>
    /// <param name="startPosition">Absolute position of the first new token.</param>
    /// <param name="count">How many new positions to produce.</param>
    /// <param name="keys">Per-site keys, each [count, D]; filled in.</param>
    /// <param name="values">Per-site values, each [count, D]; filled in.</param>
    public void Advance(
        ReadOnlySpan<int> history, int startPosition, int count, NdArray[] keys, NdArray[] values)
    {
        if (startPosition != _position)
            throw new InvalidOperationException(
                $"Engram stream is at position {_position}, asked to advance from {startPosition}.");
        if (count > _span - _taps * _dilation)
            throw new ArgumentOutOfRangeException(nameof(count),
                $"This stream was sized for at most {_span - _taps * _dilation} positions per call.");

        for (int s = 0; s < _sites; s++)
        {
            // Project the new rows straight into their slots in the ring.
            var fresh = _fresh.SliceRange(0, count);
            _model.EngramRows(s, history, startPosition, count, keys[s], fresh);
            for (int i = 0; i < count; i++)
                fresh.ReadRow(i).CopyTo(Row(s, startPosition + i));

            var taps = _model.Weights.Engrams[s].Taps;
            for (int i = 0; i < count; i++)
            {
                int position = startPosition + i;
                var destination = values[s].Row(i);
                destination.Clear();

                for (int j = 0; j < _taps; j++)
                {
                    int source = position - j * _dilation;
                    // A tap that reaches before the start of the sequence is
                    // masked off, matching _mask_diag on the batched path.
                    if (source < 0) continue;
                    Accumulate(destination, taps.ReadRow(j), Row(s, source));
                }
            }
        }

        _position = startPosition + count;
    }

    /// <summary><c>destination += gain * source</c>, element-wise.</summary>
    private static void Accumulate(Span<float> destination, ReadOnlySpan<float> gain, ReadOnlySpan<float> source)
    {
        for (int c = 0; c < destination.Length; c++) destination[c] += gain[c] * source[c];
    }

    private Span<float> Row(int site, int position) =>
        _history[site].Row(((position % _span) + _span) % _span);
}
