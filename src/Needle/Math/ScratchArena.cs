using System.Buffers;

namespace Needle.Math;

/// <summary>A saved arena position, to be handed back to <see cref="ScratchArena.RewindTo"/>.</summary>
/// <param name="Chunk">Index of the chunk in use.</param>
/// <param name="Offset">Floats consumed within that chunk.</param>
public readonly record struct ArenaMark(int Chunk, int Offset);

/// <summary>
/// A bump allocator for the temporaries a forward pass churns through.
///
/// Every layer allocates the same shapes as the last one — normalised inputs,
/// projections, gates, the routing block — so handing them out of pooled memory
/// and rewinding at the end of each layer removes essentially all of the
/// per-token garbage: a decode step went from roughly a thousand short-lived
/// arrays to none in steady state.
///
/// Memory is chunked and chunks never move, so a view stays valid until the
/// arena is rewound past it.  Only values that die before the next
/// <see cref="Reset"/> may be taken from here.
/// </summary>
public sealed class ScratchArena : IDisposable
{
    private const int DefaultChunkFloats = 1 << 18;   // 1 MB

    private readonly List<float[]> _chunks = [];

    // Shapes repeat every layer, and `params int[]` would allocate one array per
    // call — enough to undo much of the point of the arena.
    private readonly Dictionary<(int, int), int[]> _shapes = [];
    private int _chunk;
    private int _offset;
    private long _highWater;

    /// <param name="initialCapacity">Floats to reserve up front.</param>
    public ScratchArena(int initialCapacity = 1 << 16) =>
        _chunks.Add(ArrayPool<float>.Shared.Rent(System.Math.Max(1024, initialCapacity)));

    /// <summary>Peak simultaneous demand in floats, for sizing diagnostics.</summary>
    public long HighWater => _highWater;

    /// <summary>Total pooled memory currently held, in floats.</summary>
    public long Capacity => _chunks.Sum(c => (long)c.Length);

    /// <summary>Current position.</summary>
    public ArenaMark Mark => new(_chunk, _offset);

    /// <summary>
    /// Take a tensor of <paramref name="shape"/> from the arena.
    /// </summary>
    /// <param name="clear">
    /// Zero the memory.  Needed whenever the caller accumulates into the result
    /// rather than overwriting it — arena memory holds whatever the last layer
    /// left behind.
    /// </param>
    /// <param name="shape">Tensor shape.</param>
    public NdArray Take(bool clear, params int[] shape)
    {
        int count = NdArray.Count(shape);
        var (buffer, offset) = Reserve(count);
        var view = new NdArray(buffer, offset, shape);
        if (clear) view.Span.Clear();
        return view;
    }

    /// <summary>Take an uninitialised tensor.</summary>
    public NdArray Take(params int[] shape) => Take(clear: false, shape);

    /// <summary>
    /// Take a rank-2 tensor.  The shape array is cached and shared, so this
    /// allocates nothing at all once the first pass has run.
    /// </summary>
    public NdArray Take(int rows, int columns, bool clear = false)
    {
        if (!_shapes.TryGetValue((rows, columns), out var shape))
        {
            shape = [rows, columns];
            _shapes[(rows, columns)] = shape;
        }

        var (buffer, offset) = Reserve(rows * columns);
        var view = new NdArray(buffer, offset, shape);
        if (clear) view.Span.Clear();
        return view;
    }

    /// <summary>Take a raw span of <paramref name="count"/> floats.</summary>
    public Span<float> TakeSpan(int count, bool clear = false)
    {
        var (buffer, offset) = Reserve(count);
        var span = buffer.AsSpan(offset, count);
        if (clear) span.Clear();
        return span;
    }

    private (float[] Buffer, int Offset) Reserve(int count)
    {
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count));

        if (_offset + count > _chunks[_chunk].Length)
        {
            _chunk++;
            _offset = 0;

            // Chunks never move, so a request that does not fit the next one gets
            // a fresh chunk spliced in at this position; after the first pass the
            // layout is stable and nothing more is rented.
            if (_chunk >= _chunks.Count || _chunks[_chunk].Length < count)
            {
                var fresh = ArrayPool<float>.Shared.Rent(System.Math.Max(count, DefaultChunkFloats));
                _chunks.Insert(System.Math.Min(_chunk, _chunks.Count), fresh);
            }
        }

        var buffer = _chunks[_chunk];
        int offset = _offset;
        _offset += count;

        long used = _offset;
        for (int i = 0; i < _chunk; i++) used += _chunks[i].Length;
        if (used > _highWater) _highWater = used;

        return (buffer, offset);
    }

    /// <summary>Rewind to the start; everything previously taken becomes invalid.</summary>
    public void Reset()
    {
        _chunk = 0;
        _offset = 0;
    }

    /// <summary>Rewind to a saved position — the nested form, for a scope inside a layer.</summary>
    public void RewindTo(ArenaMark mark)
    {
        _chunk = mark.Chunk;
        _offset = mark.Offset;
    }

    public void Dispose()
    {
        foreach (var chunk in _chunks) ArrayPool<float>.Shared.Return(chunk);
        _chunks.Clear();
        _chunk = 0;
        _offset = 0;
    }
}
