using TorchSharp;
using static TorchSharp.torch;
using Needle.Model;

namespace Needle.Tests.Model;

public sealed class MaskUtilsTests
{
    [Fact]
    public void MakeCausalMask_Shape()
    {
        using var m = MaskUtils.MakeCausalMask(4);
        Assert.Equal([1, 1, 4, 4], m.shape);
    }

    [Fact]
    public void MakeCausalMask_LowerTriangular()
    {
        using var m = MaskUtils.MakeCausalMask(3);
        using var s = m.squeeze();   // [3,3]
        // Position [0,0] true, [0,1] false (cannot attend forward)
        Assert.True(s[0, 0].item<bool>());
        Assert.False(s[0, 1].item<bool>());
        Assert.False(s[0, 2].item<bool>());
        Assert.True(s[1, 0].item<bool>());
        Assert.True(s[1, 1].item<bool>());
        Assert.False(s[1, 2].item<bool>());
    }

    [Fact]
    public void MakePaddingMask_Shape()
    {
        using var tokens = torch.tensor(new long[,] { {1, 2, 0}, {3, 0, 0} });
        using var m = MaskUtils.MakePaddingMask(tokens, padId: 0);
        Assert.Equal([2, 1, 1, 3], m.shape);
    }

    [Fact]
    public void MakePaddingMask_PadPositionsFalse()
    {
        using var tokens = torch.tensor(new long[,] { {5, 0} });
        using var m = MaskUtils.MakePaddingMask(tokens, padId: 0);
        using var s = m.squeeze();   // [2]
        Assert.True(s[0].item<bool>());
        Assert.False(s[1].item<bool>());
    }

    [Fact]
    public void MakePackingMask_SameSegmentAttend()
    {
        // Two segments: positions 0,1 in seg=1; position 2 in seg=2
        using var segIds = torch.tensor(new int[,] { {1, 1, 2} }, dtype: ScalarType.Int32);
        using var m = MaskUtils.MakePackingMask(segIds);
        using var s = m.squeeze(); // [3, 3]
        // pos 0 attends to pos 1 (same seg)
        Assert.True(s[0, 1].item<bool>());
        // pos 0 does NOT attend to pos 2 (different seg)
        Assert.False(s[0, 2].item<bool>());
        // pos 2 attends to itself
        Assert.True(s[2, 2].item<bool>());
    }

    [Fact]
    public void MakeCausalPackingMask_NoCausalLeakage()
    {
        using var segIds = torch.tensor(new int[,] { {1, 1, 1} }, dtype: ScalarType.Int32);
        using var m = MaskUtils.MakeCausalPackingMask(segIds);
        using var s = m.squeeze(); // [3, 3]
        // pos 0 cannot attend to pos 1 (causal)
        Assert.False(s[0, 1].item<bool>());
        // pos 1 can attend to pos 0
        Assert.True(s[1, 0].item<bool>());
    }

    [Fact]
    public void MakeCrossPackingMask_Shape()
    {
        using var enc = torch.tensor(new int[,] { {1, 1, 0} }, dtype: ScalarType.Int32);
        using var dec = torch.tensor(new int[,] { {1, 2} }, dtype: ScalarType.Int32);
        using var m = MaskUtils.MakeCrossPackingMask(enc, dec);
        Assert.Equal([1, 1, 2, 3], m.shape);
    }
}
