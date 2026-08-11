using System.Diagnostics;

namespace Needle.Diagnostics;

/// <summary>
/// The stages a forward pass is broken into for measurement.
///
/// Coarse enough that the probes cost nothing measurable, fine enough that each
/// one maps to a decision: a kernel to rewrite, a redundancy to remove, or a
/// buffer to stop copying.
/// </summary>
public enum Stage
{
    /// <summary>Token lookup and embedding scale.</summary>
    Embed,

    /// <summary>Reading the lanes and normalising them — the shared per-layer read.</summary>
    LaneRead,

    /// <summary>The three hyper-connection gate projections and their Sinkhorn.</summary>
    LaneGates,

    /// <summary>Collapsing the lanes to the block input.</summary>
    LaneReduce,

    /// <summary>Engram projection, lookup and injection.</summary>
    Engram,

    /// <summary>Resolving which cached positions each query may attend to.</summary>
    AttentionPlan,

    /// <summary>The two ZC-RMS norms inside a block.</summary>
    BlockNorm,

    /// <summary>Q, K, V and the output gate projections.</summary>
    QkvProject,

    /// <summary>Scores, softmax and the weighted sum over values.</summary>
    AttentionCore,

    /// <summary>The attention output projection.</summary>
    OutProject,

    /// <summary>The Hadamard MLP.</summary>
    HadamardMlp,

    /// <summary>Sinkhorn-normalising the routing logits into a doubly-stochastic matrix.</summary>
    Sinkhorn,

    /// <summary>Writing the block output back through the routing matrix.</summary>
    LaneMix,

    /// <summary>Final norm and the tied output projection over the vocabulary.</summary>
    Logits,
}

/// <summary>
/// A stage-level stopwatch for the forward pass.
///
/// A sampling profiler says which *function* is hot; it does not say which part
/// of the model that function is hot for, and the same GEMM kernel serves the
/// projections, the gates and the output head.  This closes that gap by timing
/// named stages directly, which is also immune to the attribution skew a sampler
/// shows around short unmanaged helpers like <c>memset</c>.
///
/// Probes are a timestamp read and an array add — about 25 ns a pair, against
/// stages measured in tens of microseconds.  Attach one only when measuring:
/// <see cref="Model.Needle2Model.Profiler"/> is null by default and every probe
/// compiles to a null check.
/// </summary>
public sealed class StageProfiler
{
    private static readonly Stage[] AllStages = Enum.GetValues<Stage>();
    private static readonly double TicksToMs = 1000.0 / Stopwatch.Frequency;

    private readonly long[] _ticks = new long[AllStages.Length];
    private readonly long[] _bytes = new long[AllStages.Length];
    private readonly long[] _calls = new long[AllStages.Length];
    private long _wallStart;
    private long _wallTicks;

    /// <summary>
    /// Also record heap bytes per stage.  Off by default: reading the allocation
    /// counter is a transition into the runtime, and at two reads per probe over
    /// five thousand probes a token it costs more than the stages being measured
    /// — enough to halve the throughput it is supposed to be reporting on.  Turn
    /// it on to attribute allocation, and read the timings from a separate run.
    /// </summary>
    public bool TrackAllocations { get; init; }

    /// <summary>Start of the region the stage totals are a breakdown of.</summary>
    public void StartWall() => _wallStart = Stopwatch.GetTimestamp();

    /// <summary>End that region.</summary>
    public void StopWall() => _wallTicks += Stopwatch.GetTimestamp() - _wallStart;

    /// <summary>Forget everything measured so far.</summary>
    public void Reset()
    {
        Array.Clear(_ticks);
        Array.Clear(_bytes);
        Array.Clear(_calls);
        _wallTicks = 0;
    }

    /// <summary>
    /// The clock and allocation counter at the start of a stage.  A struct, so
    /// taking one does not itself allocate and skew what it is measuring.
    /// </summary>
    public readonly record struct Probe(long Ticks, long Bytes);

    /// <summary>
    /// Read the clock and the allocation counter, or return nothing when
    /// <paramref name="profiler"/> is null — the static form so a call site costs
    /// one null check when nothing is attached.
    /// </summary>
    public static Probe Mark(StageProfiler? profiler) =>
        profiler is null
            ? default
            : new Probe(Stopwatch.GetTimestamp(),
                        profiler.TrackAllocations ? GC.GetAllocatedBytesForCurrentThread() : 0L);

    /// <summary>
    /// Charge the time and heap allocated since <paramref name="mark"/> to a
    /// stage.  Allocation is per-thread, so the figures are the calling thread's:
    /// exact for a decode step, which is single-threaded, and an undercount for a
    /// prefill that fans out.
    /// </summary>
    public static void Add(StageProfiler? profiler, Stage stage, Probe mark)
    {
        if (profiler is null) return;
        profiler._ticks[(int)stage] += Stopwatch.GetTimestamp() - mark.Ticks;
        if (profiler.TrackAllocations)
            profiler._bytes[(int)stage] += GC.GetAllocatedBytesForCurrentThread() - mark.Bytes;
        profiler._calls[(int)stage]++;
    }

    /// <summary>Milliseconds charged to <paramref name="stage"/>.</summary>
    public double Milliseconds(Stage stage) => _ticks[(int)stage] * TicksToMs;

    /// <summary>Bytes charged to <paramref name="stage"/>.</summary>
    public long Bytes(Stage stage) => _bytes[(int)stage];

    /// <summary>Total milliseconds across every stage.</summary>
    public double TotalMilliseconds => _ticks.Sum() * TicksToMs;

    /// <summary>Total bytes across every stage.</summary>
    public long TotalBytes => _bytes.Sum();

    /// <summary>Milliseconds of wall time in the measured region.</summary>
    public double WallMilliseconds => _wallTicks * TicksToMs;

    /// <summary>
    /// A table of stages by cost, with the share of measured wall time each one
    /// accounts for and what is left unattributed.
    /// </summary>
    /// <param name="tokens">Tokens the region processed, for a per-token column.</param>
    public string Report(int tokens = 0)
    {
        double wall = WallMilliseconds > 0 ? WallMilliseconds : TotalMilliseconds;
        var lines = new List<string>
        {
            $"{"stage",-16}{"ms",9}{"% wall",9}{"calls",10}"
            + (tokens > 0 ? $"{"us/token",11}" : "")
            + (tokens > 0 && TrackAllocations ? $"{"B/token",11}" : ""),
        };

        foreach (var stage in AllStages.OrderByDescending(s => _ticks[(int)s]))
        {
            if (_calls[(int)stage] == 0) continue;
            double ms = Milliseconds(stage);
            lines.Add($"{stage,-16}{ms,9:F1}{ms * 100 / wall,8:F1}%{_calls[(int)stage],10}"
                      + (tokens > 0 ? $"{ms * 1000 / tokens,11:F1}" : "")
                      + (tokens > 0 && TrackAllocations
                          ? $"{_bytes[(int)stage] / (double)tokens,11:F0}"
                          : ""));
        }

        double accounted = TotalMilliseconds;
        lines.Add($"{"(unattributed)",-16}{wall - accounted,9:F1}{(wall - accounted) * 100 / wall,8:F1}%");
        lines.Add($"{"total",-16}{wall,9:F1}{100.0,8:F1}%"
                  + (tokens > 0 && TrackAllocations
                      ? $"{"",10}{"",11}{TotalBytes / (double)tokens,11:F0}"
                      : ""));
        return string.Join(Environment.NewLine, lines);
    }
}
