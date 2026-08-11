using System.Numerics.Tensors;
using Needle.Math;
using Needle.Model;

namespace Needle.Training.Autodiff;

/// <summary>
/// Grouped-query attention with QK-norm and RoPE, as one fused differentiable
/// node.
///
/// Fusing it rather than composing it out of primitives keeps the attention
/// probabilities from being materialised twice and makes the masking explicit:
/// only the key positions the plan admits contribute, so no gradient can leak
/// backwards in time.
/// </summary>
public static class AttentionOp
{
    /// <summary>
    /// Run attention over a whole sequence.
    /// </summary>
    /// <param name="tape">Recording tape.</param>
    /// <param name="q">Queries [T, heads*headDim], already normed and rotated.</param>
    /// <param name="k">Keys [T, kvHeads*headDim], already normed and rotated.</param>
    /// <param name="v">Values [T, kvHeads*headDim].</param>
    /// <param name="plan">Which keys each query may see.</param>
    /// <param name="heads">Query head count.</param>
    /// <param name="kvHeads">Key/value head count.</param>
    /// <param name="headDim">Per-head width.</param>
    /// <returns>Context [T, heads*headDim].</returns>
    public static Value Apply(
        Tape tape, Value q, Value k, Value v, AttentionPlan plan, int heads, int kvHeads, int headDim)
    {
        int seqLen = plan.QueryCount;
        int width = heads * headDim;
        int kvWidth = kvHeads * headDim;
        int repeats = heads / kvHeads;
        float scale = 1f / MathF.Sqrt(headDim);

        var context = new NdArray(seqLen, width);

        // Probabilities are kept: the backward pass needs them, and recomputing
        // would double the cost of the most expensive part of the layer.
        var probabilities = new float[seqLen][];

        for (int t = 0; t < seqLen; t++)
        {
            var allowed = plan.Keys(t);
            probabilities[t] = new float[allowed.Length * heads];

            for (int h = 0; h < heads; h++)
            {
                int kvHead = h / repeats;
                var query = q.Data.ReadSpan.Slice(t * width + h * headDim, headDim);
                var scores = probabilities[t].AsSpan(h * allowed.Length, allowed.Length);

                for (int i = 0; i < allowed.Length; i++)
                    scores[i] = TensorPrimitives.Dot(
                        query, k.Data.ReadSpan.Slice(allowed[i] * kvWidth + kvHead * headDim, headDim)) * scale;
                Math.Ops.Softmax(scores);

                var output = context.Row(t).Slice(h * headDim, headDim);
                for (int i = 0; i < allowed.Length; i++)
                    Math.Ops.AddScaled(output,
                        v.Data.ReadSpan.Slice(allowed[i] * kvWidth + kvHead * headDim, headDim), scores[i]);
            }
        }

        Value? node = null;
        node = tape.Record(context, () =>
        {
            var upstream = node!.Grad!;
            Span<float> dScores = stackalloc float[0];

            for (int t = 0; t < seqLen; t++)
            {
                var allowed = plan.Keys(t);
                var scratch = new float[allowed.Length];

                for (int h = 0; h < heads; h++)
                {
                    int kvHead = h / repeats;
                    var probability = probabilities[t].AsSpan(h * allowed.Length, allowed.Length);
                    var dOutput = upstream.ReadSpan.Slice(t * width + h * headDim, headDim);

                    // dprobability[i] = <dOutput, v_i>, and dv_i += probability[i] * dOutput
                    for (int i = 0; i < allowed.Length; i++)
                    {
                        int offset = allowed[i] * kvWidth + kvHead * headDim;
                        scratch[i] = TensorPrimitives.Dot(dOutput, v.Data.ReadSpan.Slice(offset, headDim));
                        if (v.RequiresGrad)
                            Math.Ops.AddScaled(v.Grad!.Span.Slice(offset, headDim), dOutput, probability[i]);
                    }

                    // Through the softmax: dscore = p ⊙ (dp − <p, dp>).
                    float dot = TensorPrimitives.Dot(probability, scratch);
                    for (int i = 0; i < allowed.Length; i++)
                        scratch[i] = probability[i] * (scratch[i] - dot) * scale;

                    var query = q.Data.ReadSpan.Slice(t * width + h * headDim, headDim);
                    for (int i = 0; i < allowed.Length; i++)
                    {
                        int offset = allowed[i] * kvWidth + kvHead * headDim;
                        if (q.RequiresGrad)
                            Math.Ops.AddScaled(q.Grad!.Span.Slice(t * width + h * headDim, headDim),
                                               k.Data.ReadSpan.Slice(offset, headDim), scratch[i]);
                        if (k.RequiresGrad)
                            Math.Ops.AddScaled(k.Grad!.Span.Slice(offset, headDim), query, scratch[i]);
                    }
                }
            }
            _ = dScores;
        }, q, k, v);

        return node;
    }

    /// <summary>
    /// Zero-centred RMSNorm applied per head, then RoPE — the two things that
    /// happen to queries and keys between projection and attention.
    /// </summary>
    /// <param name="tape">Recording tape.</param>
    /// <param name="x">Projected activations [T, heads*headDim].</param>
    /// <param name="scale">Per-head-channel gain [headDim]; may be trainable.</param>
    /// <param name="rope">Rotation tables.</param>
    /// <param name="heads">Head count.</param>
    /// <param name="headDim">Per-head width.</param>
    /// <param name="startPosition">Absolute position of the first token.</param>
    public static Value NormAndRotate(
        Tape tape, Value x, Value scale, RoPE rope, int heads, int headDim, int startPosition)
    {
        int width = heads * headDim;
        int seqLen = x.Length / width;
        int half = headDim / 2;

        var result = new NdArray(seqLen, width);
        var inverseRms = new float[seqLen * heads];

        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < heads; h++)
            {
                var head = x.Data.ReadSpan.Slice(t * width + h * headDim, headDim);
                float inv = 1f / MathF.Sqrt(TensorPrimitives.Dot(head, head) / headDim + Math.Ops.Epsilon);
                inverseRms[t * heads + h] = inv;

                var target = result.Row(t).Slice(h * headDim, headDim);
                for (int c = 0; c < headDim; c++) target[c] = (1f + scale.Data[c]) * head[c] * inv;
                rope.Apply(target, startPosition + t);
            }
        }

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!;
            Span<float> rotated = stackalloc float[512];

            for (int t = 0; t < seqLen; t++)
            {
                var cos = rope.Cos(startPosition + t);
                var sin = rope.Sin(startPosition + t);

                for (int h = 0; h < heads; h++)
                {
                    var up = upstream.ReadSpan.Slice(t * width + h * headDim, headDim);
                    var normed = rotated[..headDim];

                    // RoPE is a rotation, so its adjoint is the rotation by -angle.
                    for (int c = 0; c < half; c++)
                    {
                        float a = up[c], b = up[c + half];
                        normed[c] = a * cos[c] + b * sin[c];
                        normed[c + half] = b * cos[c] - a * sin[c];
                    }

                    var head = x.Data.ReadSpan.Slice(t * width + h * headDim, headDim);
                    float inv = inverseRms[t * heads + h];

                    if (scale.RequiresGrad)
                        for (int c = 0; c < headDim; c++) scale.Grad![c] += normed[c] * head[c] * inv;

                    if (!x.RequiresGrad) continue;

                    float projection = 0f;
                    for (int c = 0; c < headDim; c++) projection += normed[c] * (1f + scale.Data[c]) * head[c];
                    float shrink = projection * inv * inv * inv / headDim;

                    var dx = x.Grad!.Span.Slice(t * width + h * headDim, headDim);
                    for (int c = 0; c < headDim; c++)
                        dx[c] += normed[c] * (1f + scale.Data[c]) * inv - head[c] * shrink;
                }
            }
        }, x, scale);

        return node;
    }
}
