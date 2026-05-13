using SystemMath = System.Math;

namespace Needle.Training;

/// <summary>
/// Warmup-Stable-Decay learning rate schedule.
/// Linear warmup → constant peak → cosine decay.
/// Port of needle/training/optim.py <c>_wsd_schedule</c>.
/// </summary>
public class WSDSchedule
{
    private readonly float _peakValue;
    private readonly int _warmupSteps;
    private readonly int _stableSteps;
    private readonly int _decaySteps;
    private readonly float _alphaMin;

    /// <summary>
    /// Initialise a WSD schedule.
    /// </summary>
    /// <param name="peakValue">Peak (maximum) learning rate.</param>
    /// <param name="totalSteps">Total number of training steps.</param>
    /// <param name="warmupSteps">Number of linear warm-up steps.</param>
    /// <param name="decayRatio">Fraction of total steps used for cosine decay (default 0.15).</param>
    /// <param name="alphaMin">Cosine decay floor as a fraction of peakValue (default 0.05).</param>
    public WSDSchedule(
        float peakValue,
        int totalSteps,
        int warmupSteps,
        float decayRatio = 0.15f,
        float alphaMin   = 0.05f)
    {
        _peakValue   = peakValue;
        _warmupSteps = warmupSteps;
        _alphaMin    = alphaMin;
        _decaySteps  = SystemMath.Max(1, (int)(totalSteps * decayRatio));
        _stableSteps = SystemMath.Max(0, totalSteps - warmupSteps - _decaySteps);
    }

    /// <summary>
    /// Compute the learning rate at the given training step.
    /// </summary>
    /// <param name="step">Zero-based global training step.</param>
    /// <returns>
    /// <list type="bullet">
    ///   <item>Linear warmup from 0 to <c>peakValue</c> for <c>step &lt; warmupSteps</c>.</item>
    ///   <item>Constant <c>peakValue</c> during the stable phase.</item>
    ///   <item>Cosine decay from <c>peakValue</c> down to <c>peakValue * alphaMin</c> during the decay phase.</item>
    /// </list>
    /// </returns>
    public float GetLr(int step)
    {
        if (step < _warmupSteps)
        {
            // Linear warmup: 0 → peakValue
            // Guard against warmupSteps == 0 (step can never reach here in that case, but be safe)
            float warmupFrac = _warmupSteps > 0
                ? (float)(step + 1) / _warmupSteps
                : 1.0f;
            return _peakValue * warmupFrac;
        }

        int stableEnd = _warmupSteps + _stableSteps;
        if (step < stableEnd)
        {
            // Constant plateau
            return _peakValue;
        }

        // Cosine decay phase
        // Matches optax.cosine_decay_schedule(peakValue, decaySteps, alpha=alphaMin):
        //   lr = peakValue * (alphaMin + 0.5*(1-alphaMin)*(1 + cos(pi * t / decaySteps)))
        int decayStep = step - stableEnd;

        // Clamp so we never go below the floor after decay is complete
        float t = SystemMath.Min(decayStep, _decaySteps);
        float cosineDecay = 0.5f * (1.0f + MathF.Cos(MathF.PI * t / _decaySteps));
        return _peakValue * (_alphaMin + (1.0f - _alphaMin) * cosineDecay);
    }
}
