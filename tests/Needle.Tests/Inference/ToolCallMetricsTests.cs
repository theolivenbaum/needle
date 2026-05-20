using Needle.Inference;

namespace Needle.Tests.Inference;

public sealed class ToolCallMetricsTests
{
    private const string WeatherTool =
        """[{"name":"get_weather","parameters":{"location":"string","unit":"string"}}]""";

    [Fact]
    public void Evaluate_ExactMatch_ScoresOne()
    {
        var tools = new[] { WeatherTool };
        var refs  = new[] { """[{"name":"get_weather","arguments":{"location":"SF"}}]""" };
        var preds = new[] { """[{"name":"get_weather","arguments":{"location":"SF"}}]""" };

        var m = ToolCallEvaluator.Evaluate(tools, refs, preds);

        Assert.Equal(1, m.N);
        Assert.Equal(1.0, m.ExactMatch);
        Assert.Equal(1.0, m.NameF1);
        Assert.Equal(1.0, m.CallF1);
        Assert.Equal(1.0, m.ArgsAcc);
        Assert.Equal(0.0, m.ParamHaluc);
        Assert.Equal(0.0, m.ParamMiss);
        Assert.Equal(1.0, m.ValueAcc);
        Assert.Equal(1.0, m.ParseRate);
    }

    [Fact]
    public void Evaluate_BothEmpty_CountsAsExactMatch()
    {
        var m = ToolCallEvaluator.Evaluate(
            tools: new[] { "[]" },
            references: new[] { "[]" },
            predictions: new[] { "[]" });

        Assert.Equal(1.0, m.ExactMatch);
        Assert.Equal(1.0, m.ParseRate);
    }

    [Fact]
    public void Evaluate_WrongToolName_NameF1Zero()
    {
        var tools = new[] { WeatherTool };
        var refs  = new[] { """[{"name":"get_weather","arguments":{"location":"SF"}}]""" };
        var preds = new[] { """[{"name":"set_weather","arguments":{"location":"SF"}}]""" };

        var m = ToolCallEvaluator.Evaluate(tools, refs, preds);
        Assert.Equal(0.0, m.NameF1);
        Assert.Equal(0.0, m.CallF1);
        Assert.Equal(0.0, m.ExactMatch);
    }

    [Fact]
    public void Evaluate_HallucinatedParam_Detected()
    {
        var tools = new[] { WeatherTool };
        var refs  = new[] { """[{"name":"get_weather","arguments":{"location":"SF"}}]""" };
        var preds = new[] { """[{"name":"get_weather","arguments":{"location":"SF","planet":"Mars"}}]""" };

        var m = ToolCallEvaluator.Evaluate(tools, refs, preds);
        // pred has 2 keys, 1 is hallucinated (planet not in schema)
        Assert.Equal(0.5, m.ParamHaluc, precision: 5);
    }

    [Fact]
    public void Evaluate_MissingParam_Detected()
    {
        var tools = new[] { WeatherTool };
        var refs  = new[] { """[{"name":"get_weather","arguments":{"location":"SF","unit":"C"}}]""" };
        var preds = new[] { """[{"name":"get_weather","arguments":{"location":"SF"}}]""" };

        var m = ToolCallEvaluator.Evaluate(tools, refs, preds);
        // ref has 2 keys, 1 missing (unit)
        Assert.Equal(0.5, m.ParamMiss, precision: 5);
    }

    [Fact]
    public void Evaluate_BadPredJson_CountsParseError()
    {
        var tools = new[] { WeatherTool };
        var refs  = new[] { """[{"name":"get_weather","arguments":{"location":"SF"}}]""" };
        var preds = new[] { """not valid json""" };

        var m = ToolCallEvaluator.Evaluate(tools, refs, preds);
        Assert.Equal(0.0, m.ParseRate);
        Assert.Equal(0.0, m.ExactMatch);
        Assert.Equal(0.0, m.CallF1);
    }

    [Fact]
    public void Evaluate_WrongValue_StillMatchesName_ButLowerValueAcc()
    {
        var tools = new[] { WeatherTool };
        var refs  = new[] { """[{"name":"get_weather","arguments":{"location":"SF","unit":"C"}}]""" };
        var preds = new[] { """[{"name":"get_weather","arguments":{"location":"NY","unit":"C"}}]""" };

        var m = ToolCallEvaluator.Evaluate(tools, refs, preds);
        // location wrong, unit right => 1 of 2 matched values correct
        Assert.Equal(1.0, m.NameF1);
        Assert.Equal(0.0, m.CallF1, precision: 5);
        Assert.Equal(0.5, m.ValueAcc, precision: 5);
    }

    [Fact]
    public void Evaluate_ArgumentOrder_Independent()
    {
        // The reference uses unit before location, prediction uses opposite order.
        var tools = new[] { WeatherTool };
        var refs  = new[] { """[{"name":"get_weather","arguments":{"unit":"C","location":"SF"}}]""" };
        var preds = new[] { """[{"name":"get_weather","arguments":{"location":"SF","unit":"C"}}]""" };

        var m = ToolCallEvaluator.Evaluate(tools, refs, preds);
        Assert.Equal(1.0, m.ExactMatch);
        Assert.Equal(1.0, m.ArgsAcc);
    }

    [Fact]
    public void Evaluate_LengthMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            ToolCallEvaluator.Evaluate(
                tools: new[] { "[]" },
                references: new[] { "[]" },
                predictions: new[] { "[]", "[]" }));
    }
}
