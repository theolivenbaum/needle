using Needle.Math;

namespace Needle.Training.Autodiff;

/// <summary>
/// A tensor on the tape: its value, and — when something upstream of it is
/// trainable — the gradient accumulated into it during the backward pass.
/// </summary>
public sealed class Value
{
    /// <summary>The forward value.</summary>
    public NdArray Data { get; }

    /// <summary>
    /// Accumulated gradient, allocated only when this value is on a path from a
    /// trainable parameter.  Frozen activations carry none, which is what keeps
    /// LoRA fine-tuning affordable.
    /// </summary>
    public NdArray? Grad { get; private set; }

    /// <summary>True when a gradient flows to this value.</summary>
    public bool RequiresGrad => Grad is not null;

    /// <summary>Human-readable label, used in gradient-check failures.</summary>
    public string Name { get; init; } = "";

    internal Value(NdArray data, bool requiresGrad)
    {
        Data = data;
        if (requiresGrad) Grad = new NdArray((int[])data.Shape.Clone());
    }

    /// <summary>Shape of the value.</summary>
    public int[] Shape => Data.Shape;

    /// <summary>Element count.</summary>
    public int Length => Data.Length;

    /// <summary>Ensure a gradient buffer exists, e.g. when a branch becomes trainable.</summary>
    internal void EnsureGrad()
    {
        Grad ??= new NdArray((int[])Data.Shape.Clone());
    }

    /// <summary>Zero the gradient buffer.</summary>
    public void ClearGrad() => Grad?.Span.Clear();

    public override string ToString() =>
        $"{(Name.Length > 0 ? Name : "value")}{NdArray.Describe(Shape)}{(RequiresGrad ? " *" : "")}";
}

/// <summary>
/// A reverse-mode autodiff tape.
///
/// This is the training flow's foundation and is deliberately kept away from
/// inference: <see cref="Model.Needle2Model"/> runs allocation-free over pooled
/// buffers and never records anything, while training re-expresses the same
/// arithmetic through <see cref="Ops"/> so every intermediate is retained for
/// the backward pass.  Two implementations of one model, each optimised for what
/// it has to do.
///
/// Values that do not descend from a trainable parameter carry no gradient
/// buffer at all, so freezing the base model costs nothing to represent.
/// </summary>
public sealed class Tape
{
    private readonly List<Action> _backward = [];
    private readonly List<Value> _trainable = [];

    /// <summary>Number of recorded operations.</summary>
    public int Length => _backward.Count;

    /// <summary>Parameters registered on this tape.</summary>
    public IReadOnlyList<Value> Trainable => _trainable;

    /// <summary>Wrap a constant — no gradient flows to it.</summary>
    public Value Constant(NdArray data, string name = "") => new(data, false) { Name = name };

    /// <summary>Wrap a trainable parameter.</summary>
    public Value Parameter(NdArray data, string name = "")
    {
        var value = new Value(data, true) { Name = name };
        _trainable.Add(value);
        return value;
    }

    /// <summary>
    /// Create an intermediate. It requires a gradient exactly when at least one
    /// of its <paramref name="inputs"/> does.
    /// </summary>
    /// <param name="data">Forward value; the op has already filled it.</param>
    /// <param name="backward">
    /// Propagates this value's gradient into its inputs.  Only invoked when the
    /// result actually carries a gradient.
    /// </param>
    /// <param name="inputs">The values this one was computed from.</param>
    public Value Record(NdArray data, Action backward, params Value[] inputs)
    {
        bool needsGrad = false;
        foreach (var input in inputs)
        {
            if (input.RequiresGrad) { needsGrad = true; break; }
        }

        var result = new Value(data, needsGrad);
        if (needsGrad) _backward.Add(backward);
        return result;
    }

    /// <summary>Create an intermediate that never carries a gradient.</summary>
    public Value Detached(NdArray data, string name = "") => new(data, false) { Name = name };

    /// <summary>
    /// Seed <paramref name="loss"/> and run every recorded backward in reverse.
    /// Gradients accumulate, so zero them between steps.
    /// </summary>
    /// <param name="loss">Scalar to differentiate.</param>
    /// <param name="seed">
    /// Weight for this contribution.  Accumulating a batch means seeding each
    /// example with <c>1/batchSize</c>: scaling the loss itself would not work,
    /// because a masked mean renormalises by the same factor and the two cancel.
    /// </param>
    public void Backward(Value loss, float seed = 1f)
    {
        if (loss.Length != 1)
            throw new ArgumentException($"Backward needs a scalar, got {NdArray.Describe(loss.Shape)}.", nameof(loss));
        if (!loss.RequiresGrad)
            throw new InvalidOperationException(
                "The loss does not depend on any trainable parameter — nothing to differentiate.");

        loss.Grad![0] = seed;
        for (int i = _backward.Count - 1; i >= 0; i--) _backward[i]();
    }

    /// <summary>Forget the recorded graph, keeping the registered parameters.</summary>
    public void Clear() => _backward.Clear();

    /// <summary>Zero every registered parameter's gradient.</summary>
    public void ZeroGrad()
    {
        foreach (var parameter in _trainable) parameter.ClearGrad();
    }
}
