using System.Numerics.Tensors;
using Needle.Math;
using Needle.Training.Autodiff;

namespace Needle.Training;

/// <summary>
/// AdamW with decoupled weight decay — the optimiser the reference fine-tuner
/// uses (<c>optax.adamw</c>).
///
/// Decoupled means the decay is applied to the parameter directly rather than
/// folded into the gradient, so it does not interact with the adaptive scaling.
/// </summary>
public sealed class AdamW
{
    private readonly List<Value> _parameters;
    private readonly NdArray[] _firstMoment;
    private readonly NdArray[] _secondMoment;
    private int _step;

    /// <summary>Learning rate.</summary>
    public float LearningRate { get; set; }

    /// <summary>Exponential decay for the first moment.</summary>
    public float Beta1 { get; init; } = 0.9f;

    /// <summary>Exponential decay for the second moment.</summary>
    public float Beta2 { get; init; } = 0.999f;

    /// <summary>Denominator floor.</summary>
    public float Epsilon { get; init; } = 1e-8f;

    /// <summary>Decoupled weight decay; 0 disables it.</summary>
    public float WeightDecay { get; init; }

    /// <summary>Steps taken so far.</summary>
    public int Step => _step;

    public AdamW(IEnumerable<Value> parameters, float learningRate)
    {
        _parameters = parameters.ToList();
        LearningRate = learningRate;
        _firstMoment = _parameters.Select(p => new NdArray((int[])p.Shape.Clone())).ToArray();
        _secondMoment = _parameters.Select(p => new NdArray((int[])p.Shape.Clone())).ToArray();
    }

    /// <summary>
    /// Apply one update from the gradients currently accumulated on the
    /// parameters.
    /// </summary>
    public void Apply()
    {
        _step++;
        float correction1 = 1f - MathF.Pow(Beta1, _step);
        float correction2 = 1f - MathF.Pow(Beta2, _step);
        float scale = LearningRate * MathF.Sqrt(correction2) / correction1;

        for (int i = 0; i < _parameters.Count; i++)
        {
            var parameter = _parameters[i];
            var gradient = parameter.Grad;
            if (gradient is null) continue;

            var m = _firstMoment[i].Span;
            var v = _secondMoment[i].Span;
            var w = parameter.Data.Span;
            var g = gradient.ReadSpan;

            for (int j = 0; j < w.Length; j++)
            {
                m[j] = Beta1 * m[j] + (1f - Beta1) * g[j];
                v[j] = Beta2 * v[j] + (1f - Beta2) * g[j] * g[j];
                float update = m[j] / (MathF.Sqrt(v[j]) + Epsilon);
                if (WeightDecay != 0f) update += WeightDecay * w[j];
                w[j] -= scale * update;
            }
        }
    }

    /// <summary>Zero every parameter's gradient, ready for the next step.</summary>
    public void ZeroGrad()
    {
        foreach (var parameter in _parameters) parameter.ClearGrad();
    }

    /// <summary>
    /// Scale gradients so their global L2 norm is at most
    /// <paramref name="maxNorm"/>, and report the norm before clipping.
    /// </summary>
    public float ClipGradientNorm(float maxNorm)
    {
        double sumSquares = 0;
        foreach (var parameter in _parameters)
        {
            if (parameter.Grad is null) continue;
            var g = parameter.Grad.ReadSpan;
            sumSquares += TensorPrimitives.Dot(g, g);
        }

        float norm = (float)System.Math.Sqrt(sumSquares);
        if (maxNorm <= 0f || norm <= maxNorm || norm == 0f) return norm;

        float factor = maxNorm / norm;
        foreach (var parameter in _parameters)
        {
            if (parameter.Grad is null) continue;
            TensorPrimitives.Multiply(parameter.Grad.ReadSpan, factor, parameter.Grad.Span);
        }
        return norm;
    }
}

/// <summary>
/// Warmup then cosine decay, the schedule the reference fine-tuner runs under.
/// </summary>
/// <param name="PeakLearningRate">Rate reached at the end of warmup.</param>
/// <param name="TotalSteps">Steps in the run.</param>
/// <param name="WarmupRatio">Fraction of the run spent warming up.</param>
/// <param name="FinalRatio">Floor, as a fraction of the peak.</param>
public readonly record struct WarmupCosine(
    float PeakLearningRate, int TotalSteps, float WarmupRatio = 0.05f, float FinalRatio = 0.1f)
{
    /// <summary>Learning rate at <paramref name="step"/> (0-based).</summary>
    public float At(int step)
    {
        int warmup = System.Math.Max(1, (int)(TotalSteps * WarmupRatio));
        if (step < warmup) return PeakLearningRate * (step + 1) / warmup;

        float progress = System.Math.Clamp(
            (step - warmup) / (float)System.Math.Max(1, TotalSteps - warmup), 0f, 1f);
        float cosine = 0.5f * (1f + MathF.Cos(MathF.PI * progress));
        return PeakLearningRate * (FinalRatio + (1f - FinalRatio) * cosine);
    }
}
