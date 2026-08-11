using Needle.Math;
using Needle.Model;

namespace Needle.Tests.Model;

public class SequenceMaskTests
{
    [Fact]
    public void CausalMaskLetsAPositionSeeItselfAndItsPast()
    {
        var mask = SequenceMask.Causal(4);

        Assert.True(mask.Allows(2, 0));
        Assert.True(mask.Allows(2, 2));
        Assert.False(mask.Allows(2, 3));
    }

    [Fact]
    public void SlidingWindowDropsKeysThatFallOutOfIt()
    {
        var mask = new SequenceMask(10, window: 3);

        Assert.True(mask.Allows(5, 5));
        Assert.True(mask.Allows(5, 3));
        Assert.False(mask.Allows(5, 2));
    }

    [Fact]
    public void PinnedSinksSurviveTheWindow()
    {
        var sink = new bool[10];
        sink[0] = true;
        var mask = new SequenceMask(10, window: 3, sink: sink);

        Assert.True(mask.Allows(9, 0));
        Assert.False(mask.Allows(9, 1));
    }

    [Fact]
    public void PaddingIsNeverVisible()
    {
        var valid = Enumerable.Repeat(true, 5).ToArray();
        valid[2] = false;
        var mask = new SequenceMask(5, valid: valid);

        Assert.False(mask.Allows(4, 2));
        Assert.True(mask.Allows(4, 1));
    }

    [Fact]
    public void DiagonalReadsTheMaskAtAFixedLookback()
    {
        var mask = SequenceMask.Causal(4);

        // Offset 1: every position except the first can see one token back.
        Assert.Equal([0f, 1f, 1f, 1f], mask.Diagonal(1));
        Assert.Equal([0f, 0f, 1f, 1f], mask.Diagonal(2));
    }
}

public class KvBudgetTests
{
    [Fact]
    public void ReleasedGeometryFitsTheDocumentedWindow()
    {
        var config = new TransformerConfig
        {
            DModel = 512, NumHeads = 8, NumKvHeads = 4, NumLayers = 27,
            EngramLayers = [2, 15], MaxSeqLen = 2048, KvWindow = 256,
        };

        // The checkpoint asks for 256 and the budget allows more, so 256 wins.
        Assert.True(KvBudget.BudgetWindow(config) >= 256);
        Assert.Equal(256, KvBudget.EffectiveWindow(config));
    }

    [Fact]
    public void WindowNeverExceedsTheByteBudget()
    {
        var config = new TransformerConfig
        {
            DModel = 512, NumHeads = 8, NumKvHeads = 4, NumLayers = 27,
            EngramLayers = [2, 15], MaxSeqLen = 2048, KvWindow = 4096,
        };

        Assert.Equal(KvBudget.BudgetWindow(config), KvBudget.EffectiveWindow(config));
    }

    [Fact]
    public void UnsetWindowFallsBackToTheBudget()
    {
        var config = new TransformerConfig { EngramLayers = [2, 5], NumLayers = 12 };
        Assert.Equal(KvBudget.BudgetWindow(config), KvBudget.EffectiveWindow(config));
    }
}

public class EngramHashTests
{
    [Fact]
    public void HashIsDeterministicAndTableDependent()
    {
        int[] tokens = [2, 100, 200, 300];

        int a = EngramHash.Index(tokens, 3, order: 2, table: 0, slots: 8192);
        int b = EngramHash.Index(tokens, 3, order: 2, table: 0, slots: 8192);
        int c = EngramHash.Index(tokens, 3, order: 2, table: 1, slots: 8192);

        Assert.Equal(a, b);
        Assert.NotEqual(a, c);
    }

    [Fact]
    public void HashDependsOnTheWholeNgram()
    {
        int[] left = [1, 2, 3];
        int[] right = [1, 9, 3];

        // Order 1 sees only the current token, so both agree.
        Assert.Equal(EngramHash.Index(left, 2, 1, 0, 8192), EngramHash.Index(right, 2, 1, 0, 8192));
        // Order 2 reaches back one token, where they differ.
        Assert.NotEqual(EngramHash.Index(left, 2, 2, 0, 8192), EngramHash.Index(right, 2, 2, 0, 8192));
    }

    [Fact]
    public void PositionsBeforeTheStartHashAsZero()
    {
        int[] tokens = [42];
        // At position 0 an order-3 n-gram reads two out-of-range slots, which the
        // reference pads with zeros rather than wrapping.
        int[] padded = [0, 0, 42];

        Assert.Equal(EngramHash.Index(tokens, 0, 3, 0, 8192),
                     EngramHash.Index(padded, 2, 3, 0, 8192));
    }

    [Fact]
    public void FillMatchesTheScalarIndex()
    {
        int[] tokens = [2, 7, 11, 13, 17];
        int[] orders = [2, 3];
        const int Heads = 2, Slots = 64;

        var grid = new int[tokens.Length * orders.Length * Heads];
        EngramHash.Fill(tokens, orders, Heads, Slots, grid);

        for (int t = 0; t < tokens.Length; t++)
            for (int oi = 0; oi < orders.Length; oi++)
                for (int h = 0; h < Heads; h++)
                {
                    int table = oi * Heads + h;
                    Assert.Equal(EngramHash.Index(tokens, t, orders[oi], table, Slots),
                                 grid[t * orders.Length * Heads + table]);
                }
    }

    [Fact]
    public void IndicesStayInsideTheTable()
    {
        var random = new Random(23);
        var tokens = Enumerable.Range(0, 64).Select(_ => random.Next(8192)).ToArray();

        for (int t = 0; t < tokens.Length; t++)
            for (int table = 0; table < 4; table++)
            {
                int index = EngramHash.Index(tokens, t, 3, table, 8192);
                Assert.InRange(index, 0, 8191);
            }
    }
}

public class AttentionPlanTests
{
    [Fact]
    public void CausalPlanGrowsByOnePerPosition()
    {
        var plan = AttentionPlan.Build(SequenceMask.Causal(5));

        for (int t = 0; t < 5; t++) Assert.Equal(t + 1, plan.Keys(t).Length);
        Assert.Equal([0, 1, 2], plan.Keys(2).ToArray());
    }

    [Fact]
    public void WindowedPlanIsBoundedAndKeepsSinks()
    {
        var sink = new bool[12];
        sink[0] = sink[1] = true;
        var mask = new SequenceMask(12, window: 4, sink: sink);
        var plan = AttentionPlan.Build(mask);

        var keys = plan.Keys(11).ToArray();
        Assert.Equal([0, 1, 8, 9, 10, 11], keys);
        Assert.Equal(6, plan.MaxKeys);
    }

    [Fact]
    public void IncrementalPlanCoversTheWholeCache()
    {
        var mask = new SequenceMask(9);
        var plan = AttentionPlan.Build(mask, startPosition: 8, count: 1, keyCount: 9);

        Assert.Equal(1, plan.QueryCount);
        Assert.Equal(9, plan.Keys(0).Length);
    }
}

public class RoPETests
{
    [Fact]
    public void PositionZeroIsTheIdentity()
    {
        var rope = new RoPE(headDim: 8, maxLen: 16);
        float[] head = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
        var original = (float[])head.Clone();

        rope.Apply(head, 0);

        for (int i = 0; i < head.Length; i++) Assert.Equal(original[i], head[i], 6);
    }

    [Fact]
    public void RotationPreservesPairNorms()
    {
        var rope = new RoPE(headDim: 8, maxLen: 64, theta: 100000f);
        float[] head = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
        var original = (float[])head.Clone();

        rope.Apply(head, 37);

        // Element i pairs with i + headDim/2; the rotation is norm-preserving.
        for (int i = 0; i < 4; i++)
        {
            double before = original[i] * original[i] + original[i + 4] * original[i + 4];
            double after = head[i] * head[i] + head[i + 4] * head[i + 4];
            Assert.Equal(before, after, 4);
        }
    }

    [Fact]
    public void RelativeAngleDependsOnlyOnThePositionGap()
    {
        var rope = new RoPE(headDim: 4, maxLen: 64);
        float[] a = [1f, 0f, 0f, 0f];
        float[] b = [1f, 0f, 0f, 0f];
        float[] c = [1f, 0f, 0f, 0f];
        float[] d = [1f, 0f, 0f, 0f];

        rope.Apply(a, 3);
        rope.Apply(b, 8);
        rope.Apply(c, 20);
        rope.Apply(d, 25);

        double first = a[0] * b[0] + a[2] * b[2];
        double second = c[0] * d[0] + c[2] * d[2];
        Assert.Equal(first, second, 5);
    }

    [Fact]
    public void ApplyRowsRotatesEachHeadAtItsOwnPosition()
    {
        var rope = new RoPE(headDim: 4, maxLen: 32);
        var block = new NdArray(2, 8);   // two tokens, two heads of four
        for (int i = 0; i < block.Length; i++) block[i] = 1f;

        rope.ApplyRows(block, heads: 2, startPosition: 5);

        var head = new float[] { 1f, 1f, 1f, 1f };
        rope.Apply(head, 6);
        for (int i = 0; i < 4; i++) Assert.Equal(head[i], block[1, i], 5);
    }
}

public class TransformerConfigTests
{
    [Fact]
    public void ReleasedGeometryResolvesTheEngramTables()
    {
        var config = new TransformerConfig { DModel = 512, EngramOrders = [2, 3] };
        var (orders, heads, subDim) = config.EngramGeometry();

        Assert.Equal(2, orders.Length);
        Assert.Equal(2, heads);
        Assert.Equal(128, subDim);
        Assert.Equal(4, config.EngramTables);
    }

    [Fact]
    public void ValidateRejectsAnEngramSiteOutsideTheStack()
    {
        var config = new TransformerConfig { NumLayers = 4, EngramLayers = [2, 15] };
        Assert.Throws<ArgumentException>(config.Validate);
    }

    [Fact]
    public void ValidateRejectsMismatchedHeadCounts()
    {
        var config = new TransformerConfig { NumHeads = 8, NumKvHeads = 3 };
        Assert.Throws<ArgumentException>(config.Validate);
    }

    [Theory]
    [InlineData("needle", 768, 27)]
    [InlineData("base", 512, 27)]
    [InlineData("nano", 256, 20)]
    public void PresetsMatchTheReference(string name, int dModel, int layers)
    {
        var config = TransformerConfig.Preset(name);
        Assert.Equal(dModel, config.DModel);
        Assert.Equal(layers, config.NumLayers);
        config.Validate();
    }

    [Fact]
    public void HadamardWidthRoundsUpToAPowerOfTwo()
    {
        Assert.Equal(512, new TransformerConfig { DModel = 512 }.HadamardWidth);
        Assert.Equal(1024, new TransformerConfig { DModel = 768 }.HadamardWidth);
    }
}

public class StackStateTests
{
    private static readonly TransformerConfig Config = new()
    {
        DModel = 4, MhcLanes = 2, NumHeads = 1, NumKvHeads = 1, NumLayers = 2,
        EngramLayers = [], EngramOrders = [2, 3], EngramHeads = 1,
    };

    [Fact]
    public void InitialiseBroadcastsIntoEveryLane()
    {
        var state = new StackState(Config, seqLen: 2);
        var embeddings = new NdArray(2, 4);
        for (int i = 0; i < embeddings.Length; i++) embeddings[i] = i;

        state.Initialise(embeddings);

        Assert.Equal([0f, 1f, 2f, 3f], state.ReadLane(0, 0).ToArray());
        Assert.Equal([0f, 1f, 2f, 3f], state.ReadLane(0, 1).ToArray());
        Assert.Equal([4f, 5f, 6f, 7f], state.ReadLane(1, 1).ToArray());
    }

    [Fact]
    public void MixAppliesTheRoutingMatrixAndTheWriteGate()
    {
        var state = new StackState(Config, seqLen: 1);
        var embeddings = new NdArray(1, 4);
        state.Initialise(embeddings);
        state.Lane(0, 0)[0] = 2f;
        state.Lane(0, 1)[0] = 6f;

        // Swap the lanes, then add half of y into lane 0 and none into lane 1.
        var routing = new NdArray(1, 4);
        routing[0] = 0f; routing[1] = 1f; routing[2] = 1f; routing[3] = 0f;
        var gate = new NdArray(1, 2);
        gate[0] = 0.5f; gate[1] = 0f;
        var y = new NdArray(1, 4);
        y[0] = 10f;

        state.Mix(routing, gate, y);

        Assert.Equal(6f + 5f, state.ReadLane(0, 0)[0], 5);
        Assert.Equal(2f, state.ReadLane(0, 1)[0], 5);
    }

    [Fact]
    public void LaneMeanAveragesAcrossLanes()
    {
        var state = new StackState(Config, seqLen: 1);
        state.Initialise(new NdArray(1, 4));
        state.Lane(0, 0)[0] = 1f;
        state.Lane(0, 1)[0] = 3f;

        Assert.Equal(2f, state.LaneMean()[0, 0], 5);
    }
}

public class LayerKvCacheTests
{
    [Fact]
    public void AppendWritesAtTheAbsolutePosition()
    {
        var cache = new LayerKvCache(capacity: 4, kvDim: 2);
        var keys = new NdArray(1, 2);
        keys[0] = 5f; keys[1] = 6f;

        cache.Append(keys, keys, startPosition: 2);

        Assert.Equal(3, cache.Length);
        Assert.Equal([5f, 6f], cache.Keys.ReadRow(2).ToArray());
    }

    [Fact]
    public void AppendPastCapacityThrows()
    {
        var cache = new LayerKvCache(capacity: 2, kvDim: 2);
        Assert.Throws<InvalidOperationException>(
            () => cache.Append(new NdArray(2, 2), new NdArray(2, 2), startPosition: 1));
    }

    [Fact]
    public void ResetForgetsTheConversation()
    {
        var cache = new LayerKvCache(capacity: 4, kvDim: 2);
        cache.Append(new NdArray(1, 2), new NdArray(1, 2), 0);
        cache.Reset();
        Assert.Equal(0, cache.Length);
    }
}
