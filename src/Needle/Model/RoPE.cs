using Needle.Math;

namespace Needle.Model;

/// <summary>
/// Rotary position embeddings.
/// Port of <c>precompute_rope_freqs</c> and <c>apply_rope</c> in architecture.py.
///
/// The rotation pairs element <c>i</c> with element <c>i + headDim/2</c> (the
/// "split-half" convention), not adjacent elements.
/// </summary>
public sealed class RoPE
{
    private readonly float[] _cos;    // [maxLen, headDim/2]
    private readonly float[] _sin;

    /// <summary>Per-head width the tables were built for.</summary>
    public int HeadDim { get; }

    /// <summary>Number of tabulated positions.</summary>
    public int MaxLen { get; }

    /// <param name="headDim">Per-head width; must be even.</param>
    /// <param name="maxLen">Number of positions to tabulate.</param>
    /// <param name="theta">Base frequency (Needle 2 uses 100 000).</param>
    public RoPE(int headDim, int maxLen, float theta = 100000f)
    {
        if (headDim <= 0 || headDim % 2 != 0)
            throw new ArgumentException($"headDim must be positive and even, got {headDim}.", nameof(headDim));

        HeadDim = headDim;
        MaxLen = maxLen;
        int half = headDim / 2;
        _cos = new float[(long)maxLen * half <= int.MaxValue ? maxLen * half : throw new ArgumentOutOfRangeException(nameof(maxLen))];
        _sin = new float[maxLen * half];

        for (int i = 0; i < half; i++)
        {
            // freqs = 1 / theta^(2i / headDim)
            double inv = 1.0 / System.Math.Pow(theta, (2.0 * i) / headDim);
            for (int t = 0; t < maxLen; t++)
            {
                double angle = t * inv;
                _cos[t * half + i] = (float)System.Math.Cos(angle);
                _sin[t * half + i] = (float)System.Math.Sin(angle);
            }
        }
    }

    /// <summary>Cosine row for absolute position <paramref name="position"/>.</summary>
    public ReadOnlySpan<float> Cos(int position) => _cos.AsSpan(position * (HeadDim / 2), HeadDim / 2);

    /// <summary>Sine row for absolute position <paramref name="position"/>.</summary>
    public ReadOnlySpan<float> Sin(int position) => _sin.AsSpan(position * (HeadDim / 2), HeadDim / 2);

    /// <summary>
    /// Rotate one head vector in place at absolute position
    /// <paramref name="position"/>.
    /// </summary>
    /// <param name="head">A single head's slice, length <see cref="HeadDim"/>.</param>
    public void Apply(Span<float> head, int position)
    {
        if (head.Length != HeadDim)
            throw new ArgumentException($"Expected a {HeadDim}-wide head, got {head.Length}.", nameof(head));
        if ((uint)position >= (uint)MaxLen)
            throw new ArgumentOutOfRangeException(nameof(position),
                $"RoPE table covers {MaxLen} positions.");

        int half = HeadDim / 2;
        var cos = Cos(position);
        var sin = Sin(position);
        for (int i = 0; i < half; i++)
        {
            float a = head[i], b = head[i + half];
            float c = cos[i], s = sin[i];
            head[i] = a * c - b * s;
            head[i + half] = b * c + a * s;
        }
    }

    /// <summary>
    /// Rotate every head of a <c>[tokens, heads * headDim]</c> block, where token
    /// <c>t</c> sits at absolute position <c>startPosition + t</c>.
    /// </summary>
    public void ApplyRows(NdArray x, int heads, int startPosition)
    {
        int width = x.LastDim;
        if (width != heads * HeadDim)
            throw new ArgumentException($"Expected {heads * HeadDim} columns, got {width}.", nameof(x));

        int rows = x.Length / width;
        var data = x.Span;
        for (int t = 0; t < rows; t++)
        {
            var row = data.Slice(t * width, width);
            for (int h = 0; h < heads; h++)
                Apply(row.Slice(h * HeadDim, HeadDim), startPosition + t);
        }
    }
}
