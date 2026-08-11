using Needle.Math;

namespace Needle.Weights;

/// <summary>
/// One activation rotated into the basis the packed codes live in, shared by
/// every matrix that consumes it.
///
/// <see cref="QuantizedMatrix.Dot"/> needs its input rotated by the same Walsh
/// transform the codec applied to the weights, and that rotation depends only on
/// the activation and the group size — not on which matrix is about to read it.
/// The model multiplies the same activation by several matrices in a row: query,
/// key, value and the output gate all read the block input, and the three
/// hyper-connection gates all read the lane vector.  Rotating once per group of
/// callers instead of once per matrix removes that repetition.
///
/// Binding is explicit rather than inferred.  A cache keyed only on the buffer
/// would be wrong, because the arena hands the same memory to a different tensor
/// on the next layer; <see cref="Bind"/> is what says "this is a new activation",
/// so a caller that forgets it recomputes rather than reading something stale.
/// </summary>
public sealed class PreparedActivation
{
    private float[] _rotated = [];
    private float[]? _bound;
    private int _boundOffset;
    private int _tokens;
    private int _width;
    private int _groupSize;
    private int _paddedWidth;
    private bool _ready;

    /// <summary>Rotations avoided since construction, for diagnostics.</summary>
    public long Hits { get; private set; }

    /// <summary>Rotations performed since construction.</summary>
    public long Misses { get; private set; }

    /// <summary>
    /// Declare the activation the next few multiplies will read.  Anything
    /// previously rotated is discarded.
    /// </summary>
    public void Bind(NdArray x)
    {
        _bound = x.Buffer;
        _boundOffset = x.Offset;
        _ready = false;
    }

    /// <summary>Forget the binding, so the next multiply rotates for itself.</summary>
    public void Release()
    {
        _bound = null;
        _ready = false;
    }

    /// <summary>
    /// The rotation of <paramref name="x"/> for <paramref name="matrix"/>, as
    /// <c>[tokens, matrix.PaddedWidth]</c>.  Computed on the first call after a
    /// <see cref="Bind"/> and reused by every matrix with the same geometry.
    /// </summary>
    public float[] Rotate(QuantizedMatrix matrix, NdArray x)
    {
        int tokens = x.Length / matrix.Width;
        if (_ready
            && ReferenceEquals(_bound, x.Buffer) && _boundOffset == x.Offset
            && _tokens == tokens && _width == matrix.Width && _groupSize == matrix.GroupSize)
        {
            Hits++;
            return _rotated;
        }

        int needed = tokens * matrix.PaddedWidth;
        if (_rotated.Length < needed) _rotated = new float[needed];

        for (int t = 0; t < tokens; t++)
        {
            matrix.PrepareInput(x.ReadSpan.Slice(t * matrix.Width, matrix.Width),
                                _rotated.AsSpan(t * matrix.PaddedWidth, matrix.PaddedWidth));
        }

        // Only claim the result for reuse when it belongs to the bound activation;
        // an unbound call still gets a correct rotation, just no caching.
        _tokens = tokens;
        _width = matrix.Width;
        _groupSize = matrix.GroupSize;
        _paddedWidth = matrix.PaddedWidth;
        _ready = ReferenceEquals(_bound, x.Buffer) && _boundOffset == x.Offset;
        Misses++;
        return _rotated;
    }

    /// <summary>Stride between tokens in the buffer <see cref="Rotate"/> returns.</summary>
    public int Stride => _paddedWidth;
}
