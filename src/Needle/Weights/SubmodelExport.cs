using TorchSharp;
using static TorchSharp.torch;
using Needle.Model;

namespace Needle.Weights;

/// <summary>
/// Matryoshka FFN slicing.  Reduces the feed-forward intermediate dimension
/// of a trained checkpoint by a fixed factor (e.g. 2× = half FFN width)
/// to produce a smaller deployment-only model.  Attention, embeddings and
/// norms are unchanged.
///
/// Port of <c>needle/model/export.py</c>.
/// </summary>
public static class SubmodelExport
{
    /// <summary>Suffix used by FFN weight tensors.</summary>
    private static readonly string[] FfnKernels = ["gate_proj", "up_proj", "down_proj"];

    /// <summary>
    /// Slice the FFN-related weights of <paramref name="parameters"/> down to
    /// <c>config.DFf / factor</c> intermediate units and return both the
    /// sliced tensor dictionary and the updated config.
    /// </summary>
    /// <param name="parameters">
    /// Dictionary mapping parameter names to tensors (e.g. as returned by
    /// <see cref="WeightLoader.Load"/>).  The original tensors are <i>not</i>
    /// modified — sliced views are returned in a new dictionary.
    /// </param>
    /// <param name="config">Source transformer config.</param>
    /// <param name="factor">
    /// Integer shrink factor &gt; 0.  The new FFN width is <c>DFf / factor</c>.
    /// </param>
    /// <returns>(slicedTensors, newConfig)</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="factor"/> &lt;= 0 or would produce a zero-width FFN.
    /// </exception>
    public static (Dictionary<string, Tensor> sliced, TransformerConfig newConfig) SliceParams(
        IReadOnlyDictionary<string, Tensor> parameters,
        TransformerConfig config,
        int factor)
    {
        if (factor <= 0)
            throw new ArgumentOutOfRangeException(nameof(factor), "factor must be > 0.");

        int dFfNew = config.DFf / factor;
        if (dFfNew == 0)
            throw new ArgumentOutOfRangeException(nameof(factor),
                $"factor={factor} too large: would produce dFf=0.");

        int dFf = config.DFf;
        var sliced = new Dictionary<string, Tensor>(parameters.Count);

        foreach (var (name, tensor) in parameters)
        {
            sliced[name] = SliceTensor(name, tensor, dFf, dFfNew);
        }

        var newConfig = config with { DFf = dFfNew };
        return (sliced, newConfig);
    }

    /// <summary>
    /// Slice a single tensor based on its name and the FFN dimension change.
    /// Tensors that are not FFN weights are returned unchanged (as a shallow
    /// alias, not a copy).
    /// </summary>
    private static Tensor SliceTensor(string name, Tensor tensor, int dFf, int dFfNew)
    {
        // Only 2-D and 3-D weight kernels are candidates for slicing.
        if (tensor.dim() != 2 && tensor.dim() != 3)
            return tensor.alias();

        string? ffnRole = MatchFfnRole(name);
        if (ffnRole is null)
            return tensor.alias();

        if (tensor.dim() == 2)
        {
            long rows = tensor.shape[0];
            long cols = tensor.shape[1];

            if (ffnRole is "gate_proj" or "up_proj" && cols == dFf)
                return tensor[TensorIndex.Colon, TensorIndex.Slice(stop: dFfNew)].contiguous();
            if (ffnRole == "down_proj" && rows == dFf)
                return tensor[TensorIndex.Slice(stop: dFfNew), TensorIndex.Colon].contiguous();
        }
        else // dim() == 3
        {
            long rows = tensor.shape[1];
            long cols = tensor.shape[2];

            if (ffnRole is "gate_proj" or "up_proj" && cols == dFf)
                return tensor[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(stop: dFfNew)].contiguous();
            if (ffnRole == "down_proj" && rows == dFf)
                return tensor[TensorIndex.Colon, TensorIndex.Slice(stop: dFfNew), TensorIndex.Colon].contiguous();
        }

        return tensor.alias();
    }

    /// <summary>
    /// Return the FFN role (<c>gate_proj</c>, <c>up_proj</c>, or
    /// <c>down_proj</c>) embedded anywhere in the parameter name, or
    /// <c>null</c> if this is not an FFN weight.
    /// </summary>
    private static string? MatchFfnRole(string name)
    {
        foreach (var role in FfnKernels)
        {
            if (name.Contains(role, StringComparison.Ordinal))
                return role;
        }
        return null;
    }

    /// <summary>
    /// Convenience method: slice a checkpoint file and write the result to
    /// a new .ndlw file.  Loads the source, slices, then saves.
    /// </summary>
    /// <param name="sourcePath">Source .ndlw file path.</param>
    /// <param name="config">Source transformer config (passed in because .ndlw doesn't embed it).</param>
    /// <param name="factor">Shrink factor.</param>
    /// <param name="destinationPath">Destination .ndlw file path.</param>
    /// <returns>The new config with the smaller DFf.</returns>
    public static TransformerConfig ExportSubmodel(
        string sourcePath,
        TransformerConfig config,
        int factor,
        string destinationPath)
    {
        var (_, tensors) = WeightLoader.Load(sourcePath);
        try
        {
            var (sliced, newConfig) = SliceParams(tensors, config, factor);
            try
            {
                WeightLoader.Save(sliced, destinationPath);
                return newConfig;
            }
            finally
            {
                foreach (var t in sliced.Values) t.Dispose();
            }
        }
        finally
        {
            foreach (var t in tensors.Values) t.Dispose();
        }
    }
}
