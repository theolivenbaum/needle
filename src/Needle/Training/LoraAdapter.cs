using Needle.Math;
using Needle.Model;
using Needle.Training.Autodiff;

namespace Needle.Training;

/// <summary>
/// One low-rank adapter: <c>W + (alpha / rank) · A·B</c> over a frozen kernel.
///
/// <c>B</c> starts at zero so the adapted model begins exactly as the base model
/// did, and only <c>A</c> and <c>B</c> carry gradients — the base weights are
/// never touched, which is what makes fine-tuning affordable here and lets the
/// result merge back into a single <c>.cact</c>.
/// </summary>
public sealed class LoraAdapter
{
    /// <summary>Down-projection [in, rank].</summary>
    public NdArray A { get; }

    /// <summary>Up-projection [rank, out].</summary>
    public NdArray B { get; }

    /// <summary>Scaling applied to the product, <c>alpha / rank</c>.</summary>
    public float Scale { get; }

    /// <summary>Parameter path this adapts, e.g. <c>layer03.q_proj</c>.</summary>
    public string Name { get; }

    /// <summary>Rank of the factorisation.</summary>
    public int Rank => B.Shape[0];

    public LoraAdapter(string name, int inputWidth, int outputWidth, int rank, float alpha, Random random)
    {
        Name = name;
        Scale = alpha / rank;
        A = new NdArray(inputWidth, rank);
        B = new NdArray(rank, outputWidth);

        // The reference draws A from a unit normal divided by the rank and leaves
        // B at zero (needle/model/finetune.py::init_lora).
        for (int i = 0; i < A.Length; i++) A[i] = (float)Gaussian(random) / rank;
    }

    private LoraAdapter(string name, NdArray a, NdArray b, float scale)
    {
        Name = name;
        A = a;
        B = b;
        Scale = scale;
    }

    /// <summary>Rebuild from saved factors.</summary>
    public static LoraAdapter FromFactors(string name, NdArray a, NdArray b, float scale) =>
        new(name, a, b, scale);

    private static double Gaussian(Random random)
    {
        // Box-Muller; the exact stream does not matter, only the distribution.
        double u1 = 1.0 - random.NextDouble();
        double u2 = random.NextDouble();
        return System.Math.Sqrt(-2.0 * System.Math.Log(u1)) * System.Math.Cos(2.0 * System.Math.PI * u2);
    }

    /// <summary>
    /// Fold the adapter into a dense kernel: <c>W += scale · A·B</c>.
    /// This is what turns a trained adapter back into an ordinary checkpoint.
    /// </summary>
    public void MergeInto(NdArray kernel)
    {
        int inputWidth = A.Shape[0], rank = A.Shape[1], outputWidth = B.Shape[1];
        if (kernel.Shape[0] != inputWidth || kernel.Shape[1] != outputWidth)
            throw new ArgumentException(
                $"Adapter '{Name}' is {inputWidth}x{outputWidth}, kernel is "
                + $"{NdArray.Describe(kernel.Shape)}.", nameof(kernel));

        for (int i = 0; i < inputWidth; i++)
        {
            var row = kernel.Row(i);
            for (int r = 0; r < rank; r++)
            {
                float factor = Scale * A[i * rank + r];
                if (factor != 0f) Math.Ops.AddScaled(row, B.ReadSpan.Slice(r * outputWidth, outputWidth), factor);
            }
        }
    }
}

/// <summary>
/// The adapters attached to a model, and the projections they attach to.
///
/// Mirrors the reference's <c>LORA_TARGETS</c>: the five attention projections
/// in every layer.  Everything else — norms, gates, the Hadamard diagonals, the
/// engram tables, the hyper-connection routing — stays frozen.
/// </summary>
public sealed class LoraSet
{
    /// <summary>The projections the reference adapts.</summary>
    public static readonly string[] Targets = ["q_proj", "k_proj", "v_proj", "gate_proj", "out_proj"];

    private readonly Dictionary<string, LoraAdapter> _byName = [];

    /// <summary>Every adapter, in creation order.</summary>
    public List<LoraAdapter> Adapters { get; } = [];

    /// <summary>Rank each adapter was built with.</summary>
    public int Rank { get; }

    /// <summary>Alpha each adapter was built with.</summary>
    public float Alpha { get; }

    private LoraSet(int rank, float alpha)
    {
        Rank = rank;
        Alpha = alpha;
    }

    /// <summary>Total trainable parameters.</summary>
    public long ParameterCount => Adapters.Sum(a => (long)a.A.Length + a.B.Length);

    /// <summary>Look up an adapter by parameter path.</summary>
    public LoraAdapter? Find(string name) => _byName.GetValueOrDefault(name);

    /// <summary>
    /// Build adapters for every targeted projection of <paramref name="weights"/>.
    /// </summary>
    /// <param name="weights">Base weights; not modified.</param>
    /// <param name="rank">Factorisation rank (the reference defaults to 16).</param>
    /// <param name="alpha">Scaling numerator (the reference defaults to 32).</param>
    /// <param name="seed">Initialisation seed.</param>
    public static LoraSet Create(Needle2Weights weights, int rank = 16, float alpha = 32f, int seed = 0)
    {
        var random = new Random(seed);
        var set = new LoraSet(rank, alpha);

        for (int layer = 0; layer < weights.Layers.Length; layer++)
        {
            var block = weights.Layers[layer];
            set.Add($"layer{layer:D2}.q_proj", block.QProj, rank, alpha, random);
            set.Add($"layer{layer:D2}.k_proj", block.KProj, rank, alpha, random);
            set.Add($"layer{layer:D2}.v_proj", block.VProj, rank, alpha, random);
            set.Add($"layer{layer:D2}.gate_proj", block.GateProj, rank, alpha, random);
            set.Add($"layer{layer:D2}.out_proj", block.OutProj, rank, alpha, random);
        }
        return set;
    }

    private void Add(string name, LinearWeight target, int rank, float alpha, Random random)
    {
        var adapter = new LoraAdapter(name, target.InputWidth, target.OutputWidth, rank, alpha, random);
        _byName[name] = adapter;
        Adapters.Add(adapter);
    }

    /// <summary>Register every factor on <paramref name="tape"/> as trainable.</summary>
    public Dictionary<string, (Value A, Value B)> Register(Tape tape)
    {
        var registered = new Dictionary<string, (Value, Value)>(Adapters.Count);
        foreach (var adapter in Adapters)
        {
            registered[adapter.Name] = (
                tape.Parameter(adapter.A, adapter.Name + ".A"),
                tape.Parameter(adapter.B, adapter.Name + ".B"));
        }
        return registered;
    }

    /// <summary>
    /// Save the factors as safetensors, keyed <c>&lt;name&gt;.A</c> /
    /// <c>&lt;name&gt;.B</c>, alongside the scale.
    /// </summary>
    public void Save(string path)
    {
        var tensors = new Dictionary<string, NdArray>();
        foreach (var adapter in Adapters)
        {
            tensors[adapter.Name + ".A"] = adapter.A;
            tensors[adapter.Name + ".B"] = adapter.B;
        }

        var scale = new NdArray(1);
        scale[0] = Alpha / Rank;
        tensors["lora.scale"] = scale;

        Weights.Safetensors.Save(tensors, path);
    }

    /// <summary>Load factors previously written by <see cref="Save"/>.</summary>
    public static LoraSet Load(string path)
    {
        var tensors = Weights.Safetensors.Load(path);
        float scale = tensors.TryGetValue("lora.scale", out var s) ? s[0] : 1f;

        var names = tensors.Keys.Where(k => k.EndsWith(".A", StringComparison.Ordinal))
                                .Select(k => k[..^2])
                                .OrderBy(k => k, StringComparer.Ordinal)
                                .ToList();
        if (names.Count == 0)
            throw new InvalidDataException($"'{path}' holds no LoRA factors.");

        int rank = tensors[names[0] + ".A"].Shape[1];
        var set = new LoraSet(rank, scale * rank);
        foreach (string name in names)
        {
            var adapter = LoraAdapter.FromFactors(name, tensors[name + ".A"], tensors[name + ".B"], scale);
            set._byName[name] = adapter;
            set.Adapters.Add(adapter);
        }
        return set;
    }

    /// <summary>
    /// Merge every adapter into a copy of <paramref name="weights"/>, producing
    /// an ordinary float32 weight set with no adapters left.
    /// </summary>
    public Needle2Weights Merge(Needle2Weights weights)
    {
        var merged = weights.ToDenseParameters();
        foreach (var adapter in Adapters)
        {
            int layer = int.Parse(adapter.Name.AsSpan(5, 2));
            string projection = adapter.Name[(adapter.Name.IndexOf('.') + 1)..];
            string key = $"stack/layers/block/self_attn/{projection}/kernel";

            // Per-layer kernels are stacked; slice out this layer's plane.
            adapter.MergeInto(merged[key].Slice(layer));
        }
        return Needle2Weights.FromFlat(weights.Config, merged);
    }
}
