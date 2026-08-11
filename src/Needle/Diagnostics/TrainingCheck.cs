using Needle.Math;
using Needle.Model;
using Needle.Training;
using Needle.Training.Autodiff;

namespace Needle.Diagnostics;

/// <summary>
/// Checks the training flow against two independent references: the inference
/// forward pass, and finite differences.
///
/// Training runs a second implementation of the model so inference does not have
/// to carry a tape. That buys speed at the cost of a place for the two to drift
/// apart, so the agreement is asserted rather than assumed — and hand-written
/// backward passes are checked numerically, because a wrong one still trains,
/// just to the wrong place.
/// </summary>
public static class TrainingCheck
{
    /// <summary>
    /// Compare the training forward against the inference forward on the same
    /// tokens, with no adapters attached.
    /// </summary>
    public static ParityDelta ForwardMatchesInference(Needle2Weights weights, ReadOnlySpan<int> tokens)
    {
        var inference = new Needle2Model(weights);
        var expected = inference.Logits(inference.Forward(tokens).Hidden);

        var trainable = new TrainableModel(weights);
        var tape = new Tape();
        // Nothing is trainable here, so the tape records only the forward.
        var actual = trainable.Forward(tape, tokens, new Dictionary<string, (Value, Value)>());

        return ParityHarness.Delta("training-vs-inference", expected.ReadSpan, actual.Data.ReadSpan);
    }

    /// <summary>How closely one parameter's analytic gradient tracks a finite difference.</summary>
    /// <param name="Name">Parameter path.</param>
    /// <param name="Index">Flat index within the parameter.</param>
    /// <param name="Analytic">Gradient from the backward pass.</param>
    /// <param name="Numeric">Central finite difference of the loss.</param>
    /// <param name="NoiseFloor">
    /// Smallest gradient the difference could have resolved.  A central
    /// difference divides by <c>2·epsilon</c>, so a loss carrying float32
    /// rounding of about <c>|loss|·2⁻²³</c> cannot see anything below
    /// <c>|loss|·2⁻²³/epsilon</c> — and comparing against it without saying so
    /// turns rounding noise into a spurious failure.
    /// </param>
    public sealed record GradientProbe(string Name, int Index, float Analytic, float Numeric, float NoiseFloor)
    {
        /// <summary>Difference relative to the larger of the two magnitudes.</summary>
        public float RelativeError
        {
            get
            {
                float scale = System.Math.Max(System.Math.Abs(Analytic), System.Math.Abs(Numeric));
                return scale < 1e-9f ? 0f : System.Math.Abs(Analytic - Numeric) / scale;
            }
        }

        /// <summary>
        /// Does the analytic gradient agree, allowing for what the finite
        /// difference could actually resolve?
        /// </summary>
        public bool Agrees(float tolerance = 0.05f) =>
            System.Math.Abs(Analytic - Numeric) <= tolerance * System.Math.Abs(Analytic) + 4f * NoiseFloor;

        public override string ToString() =>
            $"{Name}[{Index}] analytic={Analytic:G6} numeric={Numeric:G6} "
            + $"rel={RelativeError:G4} floor={NoiseFloor:G3}";
    }

    /// <summary>
    /// Spot-check gradients against central differences.
    ///
    /// Only a sample of coordinates is probed: each one costs two extra forward
    /// passes, and a wrong backward is essentially never wrong at just one index.
    /// </summary>
    /// <param name="weights">Base weights.</param>
    /// <param name="tokens">Input tokens.</param>
    /// <param name="targets">Target tokens, same length.</param>
    /// <param name="adapters">Adapters to differentiate.</param>
    /// <param name="probes">Coordinates to check.</param>
    /// <param name="epsilon">Finite-difference step.</param>
    /// <param name="seed">Which coordinates get picked.</param>
    public static List<GradientProbe> CheckGradients(
        Needle2Weights weights, int[] tokens, int[] targets, LoraSet adapters,
        int probes = 8, float epsilon = 1e-3f, int seed = 0)
    {
        var model = new TrainableModel(weights) { AdapterScale = adapters.Alpha / adapters.Rank };
        var weightsMask = Enumerable.Repeat(1f, targets.Length).ToArray();

        var tape = new Tape();
        var registered = adapters.Register(tape);

        float Loss()
        {
            var scratch = new Tape();
            var handles = adapters.Register(scratch);
            var logits = model.Forward(scratch, tokens, handles);
            return Training.Autodiff.Ops.CrossEntropy(scratch, logits, targets, weightsMask).Data[0];
        }

        tape.ZeroGrad();
        var trainingLogits = model.Forward(tape, tokens, registered);
        var loss = Training.Autodiff.Ops.CrossEntropy(tape, trainingLogits, targets, weightsMask);
        tape.Backward(loss);

        // A central difference can only resolve gradients above the loss's own
        // rounding, so probe the coordinates with the largest analytic gradient
        // rather than uniformly random ones — a wrong backward is never wrong at
        // only the small coordinates.
        float noiseFloor = System.Math.Abs(loss.Data[0]) * 1.2e-7f / epsilon;

        var random = new Random(seed);
        var candidates = new List<(NdArray Tensor, Value Handle, int Index, float Gradient)>();
        for (int p = 0; p < probes * 8; p++)
        {
            var adapter = adapters.Adapters[random.Next(adapters.Adapters.Count)];
            bool useB = p % 2 == 1;
            var tensor = useB ? adapter.B : adapter.A;
            var handle = useB ? registered[adapter.Name].B : registered[adapter.Name].A;
            int index = random.Next(tensor.Length);
            candidates.Add((tensor, handle, index, System.Math.Abs(handle.Grad![index])));
        }

        var results = new List<GradientProbe>(probes);
        foreach (var candidate in candidates.OrderByDescending(c => c.Gradient).Take(probes))
        {
            float original = candidate.Tensor[candidate.Index];
            candidate.Tensor[candidate.Index] = original + epsilon;
            float up = Loss();
            candidate.Tensor[candidate.Index] = original - epsilon;
            float down = Loss();
            candidate.Tensor[candidate.Index] = original;

            results.Add(new GradientProbe(
                candidate.Handle.Name, candidate.Index, candidate.Handle.Grad![candidate.Index],
                (up - down) / (2f * epsilon), noiseFloor));
        }

        return results;
    }
}
