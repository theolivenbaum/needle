using TorchSharp;
using static TorchSharp.torch;
using TorchSharp.Modules;

namespace Needle.Training;

/// <summary>
/// Newton-Schulz approximate polar decomposition for gradient orthogonalisation.
/// Port of needle/training/optim.py <c>_newton_schulz</c>.
/// </summary>
public static class NewtonSchulz
{
    // Newton-Schulz polynomial coefficients (a, b, c from the reference implementation)
    private const float CoeffA = 3.4445f;
    private const float CoeffB = -4.7750f;
    private const float CoeffC = 2.0315f;

    /// <summary>
    /// Approximate polar decomposition via Newton-Schulz iteration.
    /// </summary>
    /// <param name="G">
    /// Gradient tensor.
    /// <list type="bullet">
    ///   <item>2D [m, n]: applies a single Newton-Schulz iteration sequence.</item>
    ///   <item>3D [k, m, n]: applies the iteration to each slice along the first dimension (vmap equivalent).</item>
    /// </list>
    /// If the matrix is "tall" (m &gt; n), it is transposed before the iteration and transposed back afterwards.
    /// </param>
    /// <param name="steps">Number of Newton-Schulz iterations (default 5).</param>
    /// <returns>Orthogonalised tensor with the same shape and dtype as <paramref name="G"/>.</returns>
    public static Tensor Compute(Tensor G, int steps = 5)
    {
        if (G.dim() == 3)
            return Compute3D(G, steps);

        return Compute2D(G, steps);
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// <summary>Newton-Schulz on a single 2-D matrix.</summary>
    private static Tensor Compute2D(Tensor G, int steps)
    {
        var origDtype = G.dtype;

        // Promote to float32 for numerically stable iteration
        using var Gf = G.to(ScalarType.Float32);

        long m = Gf.shape[0];
        long n = Gf.shape[1];
        bool transposed = m > n;

        // Work on the "wide" form so X ∈ R^{min × max}
        using var X0 = transposed ? Gf.t().contiguous() : Gf.contiguous();

        // Normalise: X ← G / (||G|| + ε)
        using var norm = X0.norm() + 1e-7f;
        Tensor X = X0 / norm;

        // Newton-Schulz iteration: X ← a·X + (b·S + c·S²)·X  where S = X·Xᵀ
        for (int i = 0; i < steps; i++)
        {
            using var Xt    = X.t();
            using var S     = torch.matmul(X, Xt);                  // S = X @ Xᵀ  [p, p]
            using var S2    = torch.matmul(S, S);                   // S² = S @ S  [p, p]
            using var bS    = CoeffB * S;                           // b·S
            using var cS2   = CoeffC * S2;                          // c·S²
            using var poly  = bS + cS2;                             // b·S + c·S²
            using var polyX = torch.matmul(poly, X);                // (b·S + c·S²)·X
            var newX = CoeffA * X + polyX;                          // a·X + (b·S + c·S²)·X
            X.Dispose();
            X = newX;
        }

        Tensor result;
        if (transposed)
        {
            using var Xt2 = X.t();
            result = Xt2.to(origDtype);
        }
        else
        {
            result = X.to(origDtype);
        }

        X.Dispose();
        return result;
    }

    /// <summary>
    /// Newton-Schulz over the first dimension of a 3-D tensor (vmap equivalent).
    /// Iterates over slices [0], [1], … and stacks the results.
    /// </summary>
    private static Tensor Compute3D(Tensor G, int steps)
    {
        long k = G.shape[0];

        var slices = new Tensor[k];
        for (long i = 0; i < k; i++)
        {
            using var slice = G[i];           // [m, n]
            slices[i] = Compute2D(slice, steps);
        }

        var result = torch.stack(slices, dim: 0);

        foreach (var s in slices)
            s.Dispose();

        return result;
    }
}

// ---------------------------------------------------------------------------
// MuonOptimizer
// ---------------------------------------------------------------------------

/// <summary>
/// Muon optimizer: orthogonalise 2-D and 3-D gradient matrices via Newton-Schulz
/// polar decomposition, then apply Nesterov momentum.
/// For 1-D parameters (biases, layer-norm scales) the update is used as-is without
/// orthogonalisation.
/// Port of needle/training/optim.py <c>scale_by_muon</c> + the combined muon_opt chain.
/// </summary>
public class MuonOptimizer
{
    // Parameter tensors and their names
    private readonly List<(string name, Tensor param)> _parameters;

    // Momentum buffers: param name → momentum tensor
    private readonly Dictionary<string, Tensor> _momentums = new();

    private readonly float _momentum;
    private readonly int _nsSteps;
    private readonly float _weightDecay;

    /// <summary>Current learning rate (can be updated between steps).</summary>
    public float Lr { get; set; }

    /// <summary>
    /// Create a MuonOptimizer.
    /// </summary>
    /// <param name="parameters">Named parameters to optimise.</param>
    /// <param name="lr">Initial learning rate.</param>
    /// <param name="momentum">Nesterov momentum coefficient (default 0.95).</param>
    /// <param name="nsSteps">Newton-Schulz iteration count (default 5).</param>
    /// <param name="weightDecay">L2 weight-decay coefficient (default 0.01).</param>
    public MuonOptimizer(
        IEnumerable<(string name, Tensor param)> parameters,
        float lr,
        float momentum    = 0.95f,
        int   nsSteps     = 5,
        float weightDecay = 0.01f)
    {
        Lr           = lr;
        _momentum    = momentum;
        _nsSteps     = nsSteps;
        _weightDecay = weightDecay;

        _parameters = parameters.ToList();

        // Initialise momentum buffers to zero tensors with the same shape/device as each param
        foreach (var (name, param) in _parameters)
        {
            _momentums[name] = torch.zeros_like(param).detach();
        }
    }

    /// <summary>
    /// Perform a single Muon optimiser step.
    /// </summary>
    /// <param name="gradients">
    /// Dictionary mapping parameter names to their current gradients.
    /// Parameters not present in this dictionary are skipped.
    /// </param>
    public void Step(Dictionary<string, Tensor> gradients)
    {
        using var noGrad = torch.no_grad();

        foreach (var (name, param) in _parameters)
        {
            if (!gradients.TryGetValue(name, out var grad))
                continue;

            if (grad is null)
                continue;

            // Orthogonalise 2D/3D gradients; leave 1D (biases, norms) unchanged
            Tensor orthoG;
            if (grad.dim() >= 2)
            {
                orthoG = NewtonSchulz.Compute(grad, _nsSteps);
            }
            else
            {
                orthoG = grad.alias();
            }

            // Nesterov momentum:
            //   mu_new = momentum * mu + orthoG
            //   update = orthoG + momentum * mu_new
            var mu    = _momentums[name];
            var muNew = _momentum * mu + orthoG;
            var update = orthoG + _momentum * muNew;

            _momentums[name].Dispose();
            _momentums[name] = muNew.detach();

            // Weight decay (L2 regularisation applied directly to the parameter)
            if (_weightDecay > 0f)
            {
                param.add_(param * _weightDecay, alpha: -Lr);
            }

            // Parameter update: param ← param - lr * update
            param.add_(update, alpha: -Lr);

            if (grad.dim() >= 2)
                orthoG.Dispose();
            else
                orthoG.Dispose(); // alias — still need to dispose the alias handle
        }
    }

    /// <summary>
    /// Zero all momentum buffers (analogous to optimizer.zero_grad in PyTorch for
    /// parameter gradients — here we reset the momentum state instead).
    /// </summary>
    public void ZeroGrad()
    {
        foreach (var (name, _) in _parameters)
        {
            if (_momentums.TryGetValue(name, out var mu))
            {
                mu.zero_();
            }
        }
    }
}
