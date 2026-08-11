using Needle.Math;
using Needle.Weights;

namespace Needle.Diagnostics;

/// <summary>
/// Checks this reader's dequantisation of a <c>.cact</c> blob against the
/// reference's own <c>read_export</c>.
///
/// Cactus-Quant reconstruction has three parts that can each be wrong in a way
/// that still produces plausible numbers — the LSB-first index bitstream, the
/// shared unit-sphere codebooks, and the Walsh rotation that undoes the
/// per-group whitening — so the reader is compared tensor by tensor rather than
/// judged on whether the model it produces happens to generate text.
/// </summary>
public static class CactCheck
{
    /// <summary>Per-tensor comparison of a blob against the reference dump.</summary>
    /// <param name="Index">Directory position.</param>
    /// <param name="Dtype">Element type.</param>
    /// <param name="Bits">Quantisation width, 0 when not quantised.</param>
    /// <param name="Shape">Logical shape.</param>
    /// <param name="Delta">Agreement with the reference.</param>
    public sealed record TensorCheck(int Index, CactDtype Dtype, int Bits, int[] Shape, ParityDelta Delta);

    /// <summary>
    /// Compare every tensor of <paramref name="cactPath"/> against
    /// <c>tensors.safetensors</c> in <paramref name="expectedDir"/>, as written
    /// by <c>scripts/parity/dump_cact.py</c>.
    /// </summary>
    public static IReadOnlyList<TensorCheck> Run(string cactPath, string expectedDir)
    {
        var file = CactFile.Open(cactPath);
        var expected = Safetensors.Load(Path.Combine(expectedDir, "tensors.safetensors"));

        var results = new List<TensorCheck>(file.Tensors.Count);
        for (int i = 0; i < file.Tensors.Count; i++)
        {
            var tensor = file.Tensors[i];
            if (tensor.Dtype == CactDtype.Raw) continue;

            string key = $"t{i:D4}";
            if (!expected.TryGetValue(key, out var reference))
                throw new InvalidOperationException($"Reference dump has no tensor '{key}'.");

            var actual = file.Read(i);
            if (!actual.Shape.AsSpan().SequenceEqual(reference.Shape))
                throw new InvalidOperationException(
                    $"Tensor {i}: reference shape {NdArray.Describe(reference.Shape)}, "
                    + $"reader produced {NdArray.Describe(actual.Shape)}.");

            results.Add(new TensorCheck(i, tensor.Dtype, tensor.Bits, tensor.Shape,
                                        ParityHarness.Delta(key, reference.ReadSpan, actual.ReadSpan)));
        }
        return results;
    }

    /// <summary>The embedded tokenizer blob, or null when the file carries none.</summary>
    public static byte[]? TokenizerBlob(string cactPath)
    {
        var file = CactFile.Open(cactPath);
        int index = file.TokenizerIndex;
        return index >= 0 ? file.Payload(index).ToArray() : null;
    }
}
