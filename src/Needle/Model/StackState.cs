using System.Numerics.Tensors;
using Needle.Math;

namespace Needle.Model;

/// <summary>
/// The multi-lane residual stream carried through the stack.
///
/// Needle 2 does not have a single residual vector per token: it has
/// <see cref="TransformerConfig.MhcLanes"/> of them.  Each layer reads a
/// gated mixture of the lanes, and writes its output back through a
/// doubly-stochastic routing matrix plus a per-lane write gate.  This class owns
/// that <c>[T, lanes, D]</c> buffer and the two operations the stack performs on
/// it.
/// </summary>
public sealed class StackState
{
    private readonly int _lanes;
    private readonly int _d;
    private readonly NdArray _scratch;

    /// <summary>Sequence length.</summary>
    public int SeqLen { get; }

    /// <summary>The lane buffer, shaped [T, lanes, D].</summary>
    public NdArray Lanes { get; }

    public StackState(TransformerConfig config, int seqLen)
    {
        _lanes = config.MhcLanes;
        _d = config.DModel;
        SeqLen = seqLen;
        Lanes = new NdArray(seqLen, _lanes, _d);
        _scratch = new NdArray(seqLen, _lanes, _d);
    }

    /// <summary>Broadcast the embeddings into every lane — the stack's entry state.</summary>
    public void Initialise(NdArray embeddings)
    {
        for (int t = 0; t < SeqLen; t++)
        {
            var row = embeddings.ReadRow(t);
            for (int n = 0; n < _lanes; n++) row.CopyTo(Lane(t, n));
        }
    }

    /// <summary>Mutable view of lane <paramref name="lane"/> at position <paramref name="t"/>.</summary>
    public Span<float> Lane(int t, int lane) =>
        Lanes.Buffer.AsSpan(Lanes.Offset + (t * _lanes + lane) * _d, _d);

    /// <summary>Read-only view of lane <paramref name="lane"/> at position <paramref name="t"/>.</summary>
    public ReadOnlySpan<float> ReadLane(int t, int lane) =>
        Lanes.Buffer.AsSpan(Lanes.Offset + (t * _lanes + lane) * _d, _d);

    /// <summary>
    /// Apply one layer's update:
    /// <c>x[t, i] = sum_j routing[t, i, j] * x[t, j] + writeGate[t, i] * y[t]</c>.
    /// </summary>
    /// <param name="routing">Doubly-stochastic mixing [T, lanes*lanes].</param>
    /// <param name="writeGate">Per-lane write gates [T, lanes].</param>
    /// <param name="y">This layer's block output minus its input [T, D].</param>
    public void Mix(NdArray routing, NdArray writeGate, NdArray y)
    {
        var scratchSpan = _scratch.Span;
        scratchSpan.Clear();

        for (int t = 0; t < SeqLen; t++)
        {
            var mix = routing.ReadSpan.Slice(t * _lanes * _lanes, _lanes * _lanes);
            var gates = writeGate.ReadRow(t);
            var delta = y.ReadRow(t);

            for (int i = 0; i < _lanes; i++)
            {
                var dst = scratchSpan.Slice((t * _lanes + i) * _d, _d);
                for (int j = 0; j < _lanes; j++)
                {
                    float weight = mix[i * _lanes + j];
                    if (weight != 0f) Ops.AddScaled(dst, ReadLane(t, j), weight);
                }
                Ops.AddScaled(dst, delta, gates[i]);
            }
        }

        scratchSpan.CopyTo(Lanes.Span);
    }

    /// <summary>The lane average, [T, D] — what the stack hands to the next stage.</summary>
    public NdArray LaneMean()
    {
        var mean = new NdArray(SeqLen, _d);
        float inv = 1f / _lanes;
        for (int t = 0; t < SeqLen; t++)
        {
            var dst = mean.Row(t);
            for (int n = 0; n < _lanes; n++) Ops.Add(dst, ReadLane(t, n));
            TensorPrimitives.Multiply(dst, inv, dst);
        }
        return mean;
    }
}

/// <summary>
/// Rolling key/value cache for one layer.  Rows are indexed by absolute
/// position, which is what <see cref="AttentionPlan"/> hands to the attention
/// loop.
/// </summary>
public sealed class LayerKvCache
{
    /// <summary>Cached keys [capacity, kvDim].</summary>
    public NdArray Keys { get; }

    /// <summary>Cached values [capacity, kvDim].</summary>
    public NdArray Values { get; }

    /// <summary>Number of positions written so far.</summary>
    public int Length { get; private set; }

    /// <summary>Maximum number of positions.</summary>
    public int Capacity { get; }

    public LayerKvCache(int capacity, int kvDim)
    {
        Capacity = capacity;
        Keys = new NdArray(capacity, kvDim);
        Values = new NdArray(capacity, kvDim);
    }

    /// <summary>
    /// Write <paramref name="keys"/>/<paramref name="values"/> at absolute
    /// position <paramref name="startPosition"/>.
    /// </summary>
    public void Append(NdArray keys, NdArray values, int startPosition)
    {
        int width = Keys.LastDim;
        int count = keys.Length / width;
        if (startPosition + count > Capacity)
            throw new InvalidOperationException(
                $"KV cache holds {Capacity} positions; cannot write {count} at {startPosition}.");

        keys.ReadSpan.CopyTo(Keys.Span[(startPosition * width)..]);
        values.ReadSpan.CopyTo(Values.Span[(startPosition * width)..]);
        Length = System.Math.Max(Length, startPosition + count);
    }

    /// <summary>Forget everything written so far.</summary>
    public void Reset() => Length = 0;
}
