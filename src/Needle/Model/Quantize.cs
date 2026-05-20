using TorchSharp;
using static TorchSharp.torch;

namespace Needle.Model;

/// <summary>
/// Symmetric group-wise fake quantisation with straight-through estimator.
/// Port of <c>needle/model/quantize.py</c>.
///
/// "Fake" quantisation rounds weights to a discrete grid and immediately
/// dequantises back to float, simulating low-bit deployment numerics while
/// keeping gradients flowing through the original parameter values.
/// </summary>
public static class Quantize
{
    /// <summary>Default per-channel group size for the input dimension.</summary>
    public const int DefaultGroupSize = 32;

    /// <summary>
    /// Symmetric group-wise INT4 fake-quant of a 2D weight matrix [inFeat, outFeat].
    /// Divides axis 0 into groups of <paramref name="groupSize"/> rows; each group
    /// gets its own scale.  Returns a tensor with the same shape and dtype
    /// as the input.
    /// </summary>
    public static Tensor FakeQuantizeInt4(Tensor w, int groupSize = DefaultGroupSize) =>
        FakeQuantize2D(w, groupSize, levels: 7.0f, qMin: -8.0f, qMax: 7.0f);

    /// <summary>
    /// Symmetric group-wise INT8 fake-quant of a 2D weight matrix [inFeat, outFeat].
    /// </summary>
    public static Tensor FakeQuantizeInt8(Tensor w, int groupSize = DefaultGroupSize) =>
        FakeQuantize2D(w, groupSize, levels: 127.0f, qMin: -128.0f, qMax: 127.0f);

    /// <summary>
    /// Quantise a single 2D kernel.  Implements
    /// <c>w + stop_gradient(quant(w) - w)</c> which detaches the rounding step
    /// from the gradient (straight-through estimator).
    /// </summary>
    private static Tensor FakeQuantize2D(Tensor w, int groupSize, float levels, float qMin, float qMax)
    {
        if (w.dim() != 2)
            throw new ArgumentException($"FakeQuantize2D expects a 2D tensor, got {w.dim()}D.", nameof(w));

        long inFeat  = w.shape[0];
        long outFeat = w.shape[1];
        int gs = (int)System.Math.Min(groupSize, inFeat);
        if (gs <= 0) gs = 1;

        long pad = (gs - inFeat % gs) % gs;

        // Pad along axis 0 if needed so that inFeat is a multiple of gs.
        Tensor wPadded;
        if (pad > 0)
        {
            using var zerosPad = torch.zeros(new long[] { pad, outFeat }, dtype: w.dtype, device: w.device);
            wPadded = torch.cat([w, zerosPad], dim: 0);
        }
        else
        {
            wPadded = w.alias();
        }

        try
        {
            long numGroups = wPadded.shape[0] / gs;
            using var grouped = wPadded.reshape(numGroups, gs, outFeat);          // [G, gs, out]

            // Per-group max-abs along the in-group axis (axis 1): [G, 1, out]
            using var absG = grouped.abs();
            using var maxAbs = absG.amax(new long[] { 1L }, keepdim: true);       // [G, 1, out]
            using var scale  = (maxAbs / levels).clamp_min(1e-8f);                // [G, 1, out]

            // Quantise: round(w/scale) clipped to [qMin, qMax], then scale back.
            using var divided = grouped / scale;
            using var rounded = divided.round();
            using var clipped = rounded.clamp(qMin, qMax);
            using var quantG  = clipped * scale;                                  // [G, gs, out]

            // Reshape back to padded shape and trim padding.
            using var quantPadded = quantG.reshape(wPadded.shape[0], outFeat);
            using var quantTrim   = pad > 0
                ? quantPadded[TensorIndex.Slice(stop: inFeat)]
                : quantPadded.alias();

            // Straight-through estimator: forward value == quantTrim,
            // backward gradient flows through w unchanged.
            //
            // result = w + (quantTrim - w).detach()
            using var diff = quantTrim - w;
            using var ste  = diff.detach();
            return w + ste;
        }
        finally
        {
            wPadded.Dispose();
        }
    }

    /// <summary>
    /// Fake-quantise every weight matrix (2-D or 3-D) in <paramref name="parameters"/>.
    /// 1-D parameters (biases, layer-norm scales) are left untouched.
    /// 3-D kernels (e.g. scanned weights) are quantised slice-by-slice along axis 0.
    /// </summary>
    /// <param name="parameters">
    /// Dictionary mapping parameter names to tensors.  The dictionary is updated
    /// in-place — old tensors are disposed and replaced with the quantised
    /// versions.  Pass <c>precision="int4"</c> or <c>"int8"</c>.
    /// </param>
    public static void QuantizeParams(
        IDictionary<string, Tensor> parameters,
        int groupSize    = DefaultGroupSize,
        string precision = "int4")
    {
        Func<Tensor, int, Tensor> qfn = precision == "int8"
            ? FakeQuantizeInt8
            : FakeQuantizeInt4;

        foreach (var key in parameters.Keys.ToList())
        {
            var t = parameters[key];

            // Only quantise weight matrices.  In the Python tree these are flax
            // "kernel" leaves of dim 2 or 3 — here we match by tensor rank +
            // name suffix to keep parity.
            if (t.dim() == 2 && key.EndsWith(".weight", StringComparison.Ordinal))
            {
                var q = qfn(t, groupSize);
                t.Dispose();
                parameters[key] = q;
            }
            else if (t.dim() == 3 && key.EndsWith(".weight", StringComparison.Ordinal))
            {
                // Quantise each [k] slice independently
                long k = t.shape[0];
                var slices = new Tensor[k];
                for (long i = 0; i < k; i++)
                {
                    using var slice = t[i];
                    slices[i] = qfn(slice, groupSize);
                }
                var stacked = torch.stack(slices, dim: 0);
                foreach (var s in slices) s.Dispose();
                t.Dispose();
                parameters[key] = stacked;
            }
        }
    }
}
