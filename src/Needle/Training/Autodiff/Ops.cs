using System.Numerics.Tensors;
using Needle.Math;
using Needle.Model;

namespace Needle.Training.Autodiff;

/// <summary>
/// Differentiable versions of the kernels the model is built from.
///
/// Each one computes its forward with the same routine inference uses where
/// possible, and records a closure that pushes the incoming gradient back into
/// its inputs.  The forward arithmetic is therefore identical to
/// <see cref="Needle.Math.Ops"/>; only the bookkeeping differs.
/// </summary>
public static class Ops
{
    // ── Linear algebra ───────────────────────────────────────────────────────

    /// <summary>
    /// <c>[m, k] × [k, n] → [m, n]</c> with a constant right operand — the frozen
    /// base weights, where only the input needs a gradient.
    /// </summary>
    public static Value MatMul(Tape tape, Value x, NdArray kernel)
    {
        int k = kernel.Shape[0], n = kernel.Shape[1];
        int m = x.Length / k;

        var result = new NdArray(m, n);
        Math.Ops.MatMul(x.Data.ReadSpan, kernel.ReadSpan, result.Span, m, k, n);

        Value? node = null;
        node = tape.Record(result, () =>
        {
            // dx[i, p] = sum_j dresult[i, j] * kernel[p, j]
            var upstream = node!.Grad!.ReadSpan;
            var dx = x.Grad!.Span;
            for (int i = 0; i < m; i++)
            {
                var row = upstream.Slice(i * n, n);
                var target = dx.Slice(i * k, k);
                for (int p = 0; p < k; p++)
                    target[p] += TensorPrimitives.Dot(row, kernel.ReadSpan.Slice(p * n, n));
            }
        }, x);
        return node;
    }

    /// <summary>
    /// <c>[m, k] × [k, n] → [m, n]</c> with both operands on the tape — used for
    /// the LoRA factors, where the kernel itself is what is being learned.
    /// </summary>
    public static Value MatMul(Tape tape, Value x, Value kernel)
    {
        int k = kernel.Shape[0], n = kernel.Shape[1];
        int m = x.Length / k;

        var result = new NdArray(m, n);
        Math.Ops.MatMul(x.Data.ReadSpan, kernel.Data.ReadSpan, result.Span, m, k, n);

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!.ReadSpan;

            if (x.RequiresGrad)
            {
                var dx = x.Grad!.Span;
                for (int i = 0; i < m; i++)
                {
                    var row = upstream.Slice(i * n, n);
                    var target = dx.Slice(i * k, k);
                    for (int p = 0; p < k; p++)
                        target[p] += TensorPrimitives.Dot(row, kernel.Data.ReadSpan.Slice(p * n, n));
                }
            }

            if (kernel.RequiresGrad)
            {
                // dkernel[p, j] = sum_i x[i, p] * dresult[i, j]
                var dk = kernel.Grad!.Span;
                for (int i = 0; i < m; i++)
                {
                    var row = upstream.Slice(i * n, n);
                    var input = x.Data.ReadSpan.Slice(i * k, k);
                    for (int p = 0; p < k; p++)
                    {
                        float scale = input[p];
                        if (scale != 0f)
                            TensorPrimitives.MultiplyAdd(row, scale, dk.Slice(p * n, n), dk.Slice(p * n, n));
                    }
                }
            }
        }, x, kernel);
        return node;
    }

    /// <summary>Project through a transposed constant: <c>[m, k] × [n, k]ᵀ → [m, n]</c>.</summary>
    public static Value MatMulTransposed(Tape tape, Value x, NdArray kernelTransposed)
    {
        int n = kernelTransposed.Shape[0], k = kernelTransposed.Shape[1];
        int m = x.Length / k;

        var result = new NdArray(m, n);
        Math.Ops.MatMulTransposed(x.Data.ReadSpan, kernelTransposed.ReadSpan, result.Span, m, k, n);

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!.ReadSpan;
            var dx = x.Grad!.Span;
            for (int i = 0; i < m; i++)
            {
                var row = upstream.Slice(i * n, n);
                var target = dx.Slice(i * k, k);
                for (int j = 0; j < n; j++)
                {
                    float scale = row[j];
                    if (scale != 0f)
                        TensorPrimitives.MultiplyAdd(kernelTransposed.ReadSpan.Slice(j * k, k), scale,
                                                     target, target);
                }
            }
        }, x);
        return node;
    }

    // ── Element-wise ─────────────────────────────────────────────────────────

    /// <summary>Element-wise sum of two identically shaped values.</summary>
    public static Value Add(Tape tape, Value a, Value b)
    {
        var result = new NdArray((int[])a.Shape.Clone());
        TensorPrimitives.Add(a.Data.ReadSpan, b.Data.ReadSpan, result.Span);

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!.ReadSpan;
            if (a.RequiresGrad) TensorPrimitives.Add(a.Grad!.ReadSpan, upstream, a.Grad!.Span);
            if (b.RequiresGrad) TensorPrimitives.Add(b.Grad!.ReadSpan, upstream, b.Grad!.Span);
        }, a, b);
        return node;
    }

    /// <summary><c>a + scale * b</c>, the gated residual form.</summary>
    public static Value AddScaled(Tape tape, Value a, Value b, float scale)
    {
        var result = new NdArray((int[])a.Shape.Clone());
        a.Data.ReadSpan.CopyTo(result.Span);
        TensorPrimitives.MultiplyAdd(b.Data.ReadSpan, scale, result.ReadSpan, result.Span);

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!.ReadSpan;
            if (a.RequiresGrad) TensorPrimitives.Add(a.Grad!.ReadSpan, upstream, a.Grad!.Span);
            if (b.RequiresGrad)
                TensorPrimitives.MultiplyAdd(upstream, scale, b.Grad!.ReadSpan, b.Grad!.Span);
        }, a, b);
        return node;
    }

    /// <summary>Element-wise product of two identically shaped values.</summary>
    public static Value Multiply(Tape tape, Value a, Value b)
    {
        var result = new NdArray((int[])a.Shape.Clone());
        TensorPrimitives.Multiply(a.Data.ReadSpan, b.Data.ReadSpan, result.Span);

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!.ReadSpan;
            if (a.RequiresGrad)
                for (int i = 0; i < upstream.Length; i++) a.Grad![i] += upstream[i] * b.Data[i];
            if (b.RequiresGrad)
                for (int i = 0; i < upstream.Length; i++) b.Grad![i] += upstream[i] * a.Data[i];
        }, a, b);
        return node;
    }

    /// <summary>Multiply every row of the last axis by a per-channel constant.</summary>
    public static Value MultiplyRows(Tape tape, Value x, NdArray gain)
    {
        int width = gain.Length;
        int rows = x.Length / width;

        var result = new NdArray((int[])x.Shape.Clone());
        for (int r = 0; r < rows; r++)
            TensorPrimitives.Multiply(x.Data.ReadSpan.Slice(r * width, width), gain.ReadSpan,
                                      result.Span.Slice(r * width, width));

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!.ReadSpan;
            var dx = x.Grad!.Span;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < width; c++)
                    dx[r * width + c] += upstream[r * width + c] * gain[c];
        }, x);
        return node;
    }

    /// <summary>Element-wise sigmoid.</summary>
    public static Value Sigmoid(Tape tape, Value x)
    {
        var result = new NdArray((int[])x.Shape.Clone());
        TensorPrimitives.Sigmoid(x.Data.ReadSpan, result.Span);

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!.ReadSpan;
            var dx = x.Grad!.Span;
            // d/dx sigmoid = s * (1 - s)
            for (int i = 0; i < upstream.Length; i++)
            {
                float s = result[i];
                dx[i] += upstream[i] * s * (1f - s);
            }
        }, x);
        return node;
    }

    /// <summary>Element-wise SiLU.</summary>
    public static Value Silu(Tape tape, Value x)
    {
        var result = new NdArray((int[])x.Shape.Clone());
        x.Data.ReadSpan.CopyTo(result.Span);
        Math.Ops.Silu(result.Span);

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!.ReadSpan;
            var dx = x.Grad!.Span;
            // d/dx (x·s) = s + x·s·(1 - s), with s = sigmoid(x)
            for (int i = 0; i < upstream.Length; i++)
            {
                float s = 1f / (1f + MathF.Exp(-x.Data[i]));
                dx[i] += upstream[i] * (s + x.Data[i] * s * (1f - s));
            }
        }, x);
        return node;
    }

    // ── Normalisation ────────────────────────────────────────────────────────

    /// <summary>
    /// Zero-centred RMSNorm with a learnable per-channel gain.
    /// </summary>
    /// <param name="tape">Recording tape.</param>
    /// <param name="x">Input, last axis matching <paramref name="scale"/>.</param>
    /// <param name="scale">Gain; may be trainable.</param>
    public static Value ZcRmsNorm(Tape tape, Value x, Value scale, float epsilon = Math.Ops.Epsilon)
    {
        int width = scale.Length;
        int rows = x.Length / width;

        var result = new NdArray((int[])x.Shape.Clone());
        var inverseRms = new float[rows];

        for (int r = 0; r < rows; r++)
        {
            var row = x.Data.ReadSpan.Slice(r * width, width);
            inverseRms[r] = 1f / MathF.Sqrt(TensorPrimitives.Dot(row, row) / width + epsilon);
            var target = result.Span.Slice(r * width, width);
            for (int c = 0; c < width; c++) target[c] = (1f + scale.Data[c]) * row[c] * inverseRms[r];
        }

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!.ReadSpan;

            for (int r = 0; r < rows; r++)
            {
                var row = x.Data.ReadSpan.Slice(r * width, width);
                var up = upstream.Slice(r * width, width);
                float invRms = inverseRms[r];

                if (scale.RequiresGrad)
                    for (int c = 0; c < width; c++) scale.Grad![c] += up[c] * row[c] * invRms;

                if (!x.RequiresGrad) continue;

                // y = g ⊙ x / rms, so dx = invRms·(g ⊙ up) − x·(invRms³/width)·⟨g ⊙ up, x⟩
                float projection = 0f;
                for (int c = 0; c < width; c++) projection += up[c] * (1f + scale.Data[c]) * row[c];
                float shrink = projection * invRms * invRms * invRms / width;

                var dx = x.Grad!.Span.Slice(r * width, width);
                for (int c = 0; c < width; c++)
                    dx[c] += up[c] * (1f + scale.Data[c]) * invRms - row[c] * shrink;
            }
        }, x, scale);
        return node;
    }

    /// <summary>RMS normalisation with no learned gain.</summary>
    public static Value RmsUnit(Tape tape, Value x, float epsilon = Math.Ops.Epsilon)
    {
        int width = x.LastDim();
        int rows = x.Length / width;

        var result = new NdArray((int[])x.Shape.Clone());
        var inverseRms = new float[rows];

        for (int r = 0; r < rows; r++)
        {
            var row = x.Data.ReadSpan.Slice(r * width, width);
            inverseRms[r] = 1f / MathF.Sqrt(TensorPrimitives.Dot(row, row) / width + epsilon);
            TensorPrimitives.Multiply(row, inverseRms[r], result.Span.Slice(r * width, width));
        }

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!.ReadSpan;
            for (int r = 0; r < rows; r++)
            {
                var row = x.Data.ReadSpan.Slice(r * width, width);
                var up = upstream.Slice(r * width, width);
                float invRms = inverseRms[r];
                float shrink = TensorPrimitives.Dot(up, row) * invRms * invRms * invRms / width;

                var dx = x.Grad!.Span.Slice(r * width, width);
                for (int c = 0; c < width; c++) dx[c] += up[c] * invRms - row[c] * shrink;
            }
        }, x);
        return node;
    }

    // ── Structured transforms ────────────────────────────────────────────────

    /// <summary>
    /// The orthonormal Walsh-Hadamard transform over the last axis.  It is
    /// symmetric, so the backward pass is the very same transform.
    /// </summary>
    public static Value Walsh(Tape tape, Value x)
    {
        var result = new NdArray((int[])x.Shape.Clone());
        x.Data.ReadSpan.CopyTo(result.Span);
        WalshHadamard.TransformRows(result);

        Value? node = null;
        node = tape.Record(result, () =>
        {
            int width = x.LastDim();
            int rows = x.Length / width;
            var scratch = new NdArray(rows, width);
            node!.Grad!.ReadSpan.CopyTo(scratch.Span);
            WalshHadamard.TransformRows(scratch);
            TensorPrimitives.Add(x.Grad!.ReadSpan, scratch.ReadSpan, x.Grad!.Span);
        }, x);
        return node;
    }

    /// <summary>
    /// Softmax over the last axis, restricted to <paramref name="width"/> live
    /// entries per row.
    /// </summary>
    public static Value Softmax(Tape tape, Value x, int width)
    {
        int rows = x.Length / width;
        var result = new NdArray((int[])x.Shape.Clone());
        x.Data.ReadSpan.CopyTo(result.Span);
        for (int r = 0; r < rows; r++) Math.Ops.Softmax(result.Span.Slice(r * width, width));

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!.ReadSpan;
            var dx = x.Grad!.Span;
            for (int r = 0; r < rows; r++)
            {
                var p = result.ReadSpan.Slice(r * width, width);
                var up = upstream.Slice(r * width, width);
                float dot = TensorPrimitives.Dot(p, up);
                for (int c = 0; c < width; c++) dx[r * width + c] += p[c] * (up[c] - dot);
            }
        }, x);
        return node;
    }

    /// <summary>
    /// Sinkhorn normalisation of square blocks, differentiated by unrolling the
    /// iteration.
    ///
    /// Each half-step subtracts a log-sum-exp along one axis, whose Jacobian is
    /// <c>I − 1·softmaxᵀ</c> along that axis; running those backwards in reverse
    /// order is the whole derivative.
    /// </summary>
    /// <param name="tape">Recording tape.</param>
    /// <param name="x">Logits, <c>[rows, n*n]</c>.</param>
    /// <param name="n">Block width.</param>
    /// <param name="iterations">Iterations; must match the forward model.</param>
    public static Value Sinkhorn(Tape tape, Value x, int n, int iterations = Needle.Math.Sinkhorn.DefaultIterations)
    {
        int blockSize = n * n;
        int rows = x.Length / blockSize;

        // Keep every half-step's output so the backward pass can rebuild the
        // softmaxes it needs without a second forward.
        var trace = new float[2 * iterations + 1][];
        trace[0] = x.Data.ReadSpan.ToArray();

        var current = (float[])trace[0].Clone();
        for (int it = 0; it < iterations; it++)
        {
            for (int r = 0; r < rows; r++) NormaliseRows(current.AsSpan(r * blockSize, blockSize), n);
            trace[2 * it + 1] = (float[])current.Clone();

            for (int r = 0; r < rows; r++) NormaliseColumns(current.AsSpan(r * blockSize, blockSize), n);
            trace[2 * it + 2] = (float[])current.Clone();
        }

        var result = new NdArray((int[])x.Shape.Clone());
        for (int i = 0; i < current.Length; i++) result[i] = MathF.Exp(current[i]);

        Value? node = null;
        node = tape.Record(result, () =>
        {
            var upstream = node!.Grad!.ReadSpan;

            // Through the final exp.
            var g = new float[upstream.Length];
            for (int i = 0; i < g.Length; i++) g[i] = upstream[i] * result[i];

            for (int it = iterations - 1; it >= 0; it--)
            {
                for (int r = 0; r < rows; r++)
                    BackNormaliseColumns(g.AsSpan(r * blockSize, blockSize),
                                         trace[2 * it + 1].AsSpan(r * blockSize, blockSize), n);
                for (int r = 0; r < rows; r++)
                    BackNormaliseRows(g.AsSpan(r * blockSize, blockSize),
                                      trace[2 * it].AsSpan(r * blockSize, blockSize), n);
            }

            TensorPrimitives.Add(x.Grad!.ReadSpan, g, x.Grad!.Span);
        }, x);
        return node;
    }

    private static void NormaliseRows(Span<float> block, int n)
    {
        for (int i = 0; i < n; i++)
        {
            var row = block.Slice(i * n, n);
            float lse = Math.Ops.LogSumExp(row);
            for (int j = 0; j < n; j++) row[j] -= lse;
        }
    }

    private static void NormaliseColumns(Span<float> block, int n)
    {
        Span<float> column = stackalloc float[16];
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++) column[i] = block[i * n + j];
            float lse = Math.Ops.LogSumExp(column[..n]);
            for (int i = 0; i < n; i++) block[i * n + j] -= lse;
        }
    }

    /// <summary>Adjoint of a row-wise log-sum-exp subtraction, given that step's input.</summary>
    private static void BackNormaliseRows(Span<float> g, ReadOnlySpan<float> input, int n)
    {
        Span<float> softmax = stackalloc float[16];
        for (int i = 0; i < n; i++)
        {
            var row = input.Slice(i * n, n);
            float lse = Math.Ops.LogSumExp(row);
            float sum = 0f;
            for (int j = 0; j < n; j++)
            {
                softmax[j] = MathF.Exp(row[j] - lse);
                sum += g[i * n + j];
            }
            for (int j = 0; j < n; j++) g[i * n + j] -= sum * softmax[j];
        }
    }

    /// <summary>Adjoint of a column-wise log-sum-exp subtraction.</summary>
    private static void BackNormaliseColumns(Span<float> g, ReadOnlySpan<float> input, int n)
    {
        Span<float> column = stackalloc float[16];
        Span<float> softmax = stackalloc float[16];
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++) column[i] = input[i * n + j];
            float lse = Math.Ops.LogSumExp(column[..n]);
            float sum = 0f;
            for (int i = 0; i < n; i++)
            {
                softmax[i] = MathF.Exp(column[i] - lse);
                sum += g[i * n + j];
            }
            for (int i = 0; i < n; i++) g[i * n + j] -= sum * softmax[i];
        }
    }

    // ── Shape ────────────────────────────────────────────────────────────────

    /// <summary>Reinterpret without copying; gradients pass straight through.</summary>
    public static Value Reshape(Tape tape, Value x, params int[] shape)
    {
        var view = x.Data.Reshape(shape);
        Value? node = null;
        node = tape.Record(view, () =>
        {
            TensorPrimitives.Add(x.Grad!.ReadSpan, node!.Grad!.ReadSpan, x.Grad!.Span);
        }, x);

        // Reshape shares storage, so the gradient buffers must be distinct.
        return node;
    }

    /// <summary>Concatenate two values along the last axis.</summary>
    public static Value ConcatColumns(Tape tape, Value a, Value b)
    {
        int leftWidth = a.LastDim(), rightWidth = b.LastDim();
        int rows = a.Length / leftWidth;

        var result = new NdArray(rows, leftWidth + rightWidth);
        for (int r = 0; r < rows; r++)
        {
            a.Data.ReadSpan.Slice(r * leftWidth, leftWidth).CopyTo(result.Row(r));
            b.Data.ReadSpan.Slice(r * rightWidth, rightWidth).CopyTo(result.Row(r)[leftWidth..]);
        }

        Value? node = null;
        node = tape.Record(result, () =>
        {
            int width = leftWidth + rightWidth;
            var upstream = node!.Grad!.ReadSpan;
            for (int r = 0; r < rows; r++)
            {
                if (a.RequiresGrad)
                    TensorPrimitives.Add(a.Grad!.ReadSpan.Slice(r * leftWidth, leftWidth),
                                         upstream.Slice(r * width, leftWidth),
                                         a.Grad!.Span.Slice(r * leftWidth, leftWidth));
                if (b.RequiresGrad)
                    TensorPrimitives.Add(b.Grad!.ReadSpan.Slice(r * rightWidth, rightWidth),
                                         upstream.Slice(r * width + leftWidth, rightWidth),
                                         b.Grad!.Span.Slice(r * rightWidth, rightWidth));
            }
        }, a, b);
        return node;
    }

    // ── Loss ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Next-token cross-entropy over the positions <paramref name="weights"/>
    /// marks, averaged by total weight.
    ///
    /// The mask is what makes this a fine-tuning loss rather than a language
    /// modelling one: it is 1 only over the target continuation, so the prompt is
    /// conditioned on but never trained against.
    /// </summary>
    /// <param name="tape">Recording tape.</param>
    /// <param name="logits">Unnormalised scores <c>[T, vocab]</c>.</param>
    /// <param name="targets">Target token per position <c>[T]</c>.</param>
    /// <param name="weights">Per-position weight <c>[T]</c>; zero to ignore.</param>
    public static Value CrossEntropy(Tape tape, Value logits, ReadOnlySpan<int> targets, ReadOnlySpan<float> weights)
    {
        int vocab = logits.LastDim();
        int rows = logits.Length / vocab;

        float total = 0f, mass = 0f;
        var logSumExp = new float[rows];
        var target = targets.ToArray();
        var weight = weights.ToArray();

        for (int r = 0; r < rows; r++)
        {
            if (weight[r] == 0f) continue;
            var row = logits.Data.ReadSpan.Slice(r * vocab, vocab);
            logSumExp[r] = Math.Ops.LogSumExp(row);
            total += weight[r] * (logSumExp[r] - row[target[r]]);
            mass += weight[r];
        }

        float normaliser = MathF.Max(mass, 1f);
        var result = new NdArray(1);
        result[0] = total / normaliser;

        Value? node = null;
        node = tape.Record(result, () =>
        {
            float seed = node!.Grad![0] / normaliser;
            var dx = logits.Grad!.Span;
            for (int r = 0; r < rows; r++)
            {
                if (weight[r] == 0f) continue;
                var row = logits.Data.ReadSpan.Slice(r * vocab, vocab);
                var target_ = dx.Slice(r * vocab, vocab);
                float factor = seed * weight[r];
                for (int c = 0; c < vocab; c++)
                    target_[c] += factor * MathF.Exp(row[c] - logSumExp[r]);
                target_[target[r]] -= factor;
            }
        }, logits);
        return node;
    }

    /// <summary>Length of a value's last axis.</summary>
    public static int LastDim(this Value value) => value.Data.LastDim;
}
