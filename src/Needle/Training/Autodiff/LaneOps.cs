using System.Numerics.Tensors;
using Needle.Math;

namespace Needle.Training.Autodiff;

/// <summary>
/// The multi-lane hyper-connection reads and writes, as differentiable nodes.
///
/// These are the two places where the lane axis is contracted away and
/// reintroduced, and they are the only ops in the model with no counterpart in
/// an ordinary transformer.
/// </summary>
public static class LaneOps
{
    /// <summary>
    /// Read the lanes into one block input:
    /// <c>u[t, c] = sum_n gate[t, n] * lanes[t, n, c]</c>.
    /// </summary>
    /// <param name="tape">Recording tape.</param>
    /// <param name="lanes">Residual lanes [T, lanes, D].</param>
    /// <param name="gate">Per-lane read gates [T, lanes].</param>
    /// <param name="laneCount">Number of lanes.</param>
    /// <param name="width">Residual width.</param>
    public static Value Read(Tape tape, Value lanes, Value gate, int laneCount, int width)
    {
        int seqLen = lanes.Length / (laneCount * width);
        var result = new NdArray(seqLen, width);

        for (int t = 0; t < seqLen; t++)
        {
            var target = result.Row(t);
            for (int n = 0; n < laneCount; n++)
                Math.Ops.AddScaled(target,
                    lanes.Data.ReadSpan.Slice((t * laneCount + n) * width, width),
                    gate.Data[t * laneCount + n]);
        }

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!;
            for (int t = 0; t < seqLen; t++)
            {
                var up = upstream.ReadSpan.Slice(t * width, width);
                for (int n = 0; n < laneCount; n++)
                {
                    int offset = (t * laneCount + n) * width;
                    if (lanes.RequiresGrad)
                        Math.Ops.AddScaled(lanes.Grad!.Span.Slice(offset, width), up,
                                           gate.Data[t * laneCount + n]);
                    if (gate.RequiresGrad)
                        gate.Grad![t * laneCount + n] +=
                            TensorPrimitives.Dot(up, lanes.Data.ReadSpan.Slice(offset, width));
                }
            }
        }, lanes, gate);
        return node;
    }

    /// <summary>
    /// Write the layer's output back through the routing matrix and the write
    /// gate:
    /// <c>next[t, i, c] = sum_j routing[t, i, j] * lanes[t, j, c] + gate[t, i] * y[t, c]</c>.
    /// </summary>
    /// <param name="tape">Recording tape.</param>
    /// <param name="lanes">Residual lanes [T, lanes, D].</param>
    /// <param name="routing">Doubly-stochastic mixing [T, lanes*lanes].</param>
    /// <param name="gate">Per-lane write gates [T, lanes].</param>
    /// <param name="y">Block output minus its input [T, D].</param>
    /// <param name="laneCount">Number of lanes.</param>
    /// <param name="width">Residual width.</param>
    public static Value Write(
        Tape tape, Value lanes, Value routing, Value gate, Value y, int laneCount, int width)
    {
        int seqLen = y.Length / width;
        var result = new NdArray(seqLen, laneCount, width);

        for (int t = 0; t < seqLen; t++)
        {
            var mix = routing.Data.ReadSpan.Slice(t * laneCount * laneCount, laneCount * laneCount);
            var delta = y.Data.ReadSpan.Slice(t * width, width);

            for (int i = 0; i < laneCount; i++)
            {
                var target = result.Span.Slice((t * laneCount + i) * width, width);
                target.Clear();
                for (int j = 0; j < laneCount; j++)
                    Math.Ops.AddScaled(target,
                        lanes.Data.ReadSpan.Slice((t * laneCount + j) * width, width), mix[i * laneCount + j]);
                Math.Ops.AddScaled(target, delta, gate.Data[t * laneCount + i]);
            }
        }

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!;
            for (int t = 0; t < seqLen; t++)
            {
                var mix = routing.Data.ReadSpan.Slice(t * laneCount * laneCount, laneCount * laneCount);
                var delta = y.Data.ReadSpan.Slice(t * width, width);

                for (int i = 0; i < laneCount; i++)
                {
                    var up = upstream.ReadSpan.Slice((t * laneCount + i) * width, width);

                    for (int j = 0; j < laneCount; j++)
                    {
                        int source = (t * laneCount + j) * width;
                        if (lanes.RequiresGrad)
                            Math.Ops.AddScaled(lanes.Grad!.Span.Slice(source, width), up,
                                               mix[i * laneCount + j]);
                        if (routing.RequiresGrad)
                            routing.Grad![t * laneCount * laneCount + i * laneCount + j] +=
                                TensorPrimitives.Dot(up, lanes.Data.ReadSpan.Slice(source, width));
                    }

                    if (gate.RequiresGrad)
                        gate.Grad![t * laneCount + i] += TensorPrimitives.Dot(up, delta);
                    if (y.RequiresGrad)
                        Math.Ops.AddScaled(y.Grad!.Span.Slice(t * width, width), up,
                                           gate.Data[t * laneCount + i]);
                }
            }
        }, lanes, routing, gate, y);
        return node;
    }

    /// <summary>The lane average, <c>[T, lanes, D] → [T, D]</c>.</summary>
    public static Value Mean(Tape tape, Value lanes, int laneCount, int width)
    {
        int seqLen = lanes.Length / (laneCount * width);
        var result = new NdArray(seqLen, width);
        float inverse = 1f / laneCount;

        for (int t = 0; t < seqLen; t++)
        {
            var target = result.Row(t);
            target.Clear();
            for (int n = 0; n < laneCount; n++)
                Math.Ops.Add(target, lanes.Data.ReadSpan.Slice((t * laneCount + n) * width, width));
            TensorPrimitives.Multiply(target, inverse, target);
        }

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!;
            for (int t = 0; t < seqLen; t++)
            {
                var up = upstream.ReadSpan.Slice(t * width, width);
                for (int n = 0; n < laneCount; n++)
                    Math.Ops.AddScaled(lanes.Grad!.Span.Slice((t * laneCount + n) * width, width),
                                       up, inverse);
            }
        }, lanes);
        return node;
    }

    /// <summary>Broadcast one activation into every lane, <c>[T, D] → [T, lanes, D]</c>.</summary>
    public static Value Broadcast(Tape tape, Value x, int laneCount, int width)
    {
        int seqLen = x.Length / width;
        var result = new NdArray(seqLen, laneCount, width);

        for (int t = 0; t < seqLen; t++)
        {
            var row = x.Data.ReadSpan.Slice(t * width, width);
            for (int n = 0; n < laneCount; n++)
                row.CopyTo(result.Span.Slice((t * laneCount + n) * width, width));
        }

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!;
            for (int t = 0; t < seqLen; t++)
            {
                var target = x.Grad!.Span.Slice(t * width, width);
                for (int n = 0; n < laneCount; n++)
                    Math.Ops.Add(target, upstream.ReadSpan.Slice((t * laneCount + n) * width, width));
            }
        }, x);
        return node;
    }

    /// <summary>
    /// <c>result = (doubled ? 2 : 1) * sigmoid(slope * logits + bias + offset)</c>
    /// over a [T, lanes] tensor — the hyper-connection gate shape.
    /// </summary>
    public static Value Gate(
        Tape tape, Value logits, float slope, ReadOnlySpan<float> bias, ReadOnlySpan<float> offset, bool doubled)
    {
        int lanes = bias.Length;
        int rows = logits.Length / lanes;
        float amplitude = doubled ? 2f : 1f;

        var result = new NdArray(rows, lanes);
        var biasCopy = bias.ToArray();
        var offsetCopy = offset.ToArray();

        for (int r = 0; r < rows; r++)
            for (int n = 0; n < lanes; n++)
            {
                float z = slope * logits.Data[r * lanes + n] + biasCopy[n] + offsetCopy[n];
                result[r * lanes + n] = amplitude / (1f + MathF.Exp(-z));
            }

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!;
            for (int i = 0; i < result.Length; i++)
            {
                float s = result[i] / amplitude;
                logits.Grad![i] += upstream[i] * amplitude * s * (1f - s) * slope;
            }
        }, logits);
        return node;
    }
}
