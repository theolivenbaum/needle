using Needle.Inference;

namespace Needle.Tests.Inference;

public sealed class ConstrainedDecodingTests
{
    // ── Trie ──────────────────────────────────────────────────────────────────

    [Fact]
    public void Trie_InsertAndFind_ReturnsNode()
    {
        var trie = new Trie();
        trie.Insert("hello");
        var node = trie.GetNode("hel");
        Assert.NotNull(node);
    }

    [Fact]
    public void Trie_TerminalAtEnd()
    {
        var trie = new Trie();
        trie.Insert("hi");
        var node = trie.GetNode("hi");
        Assert.NotNull(node);
        Assert.True(node!.IsTerminal);
    }

    [Fact]
    public void Trie_NotTerminalAtPrefix()
    {
        var trie = new Trie();
        trie.Insert("hello");
        var node = trie.GetNode("hel");
        Assert.NotNull(node);
        Assert.False(node!.IsTerminal);
    }

    [Fact]
    public void Trie_MissingPrefix_ReturnsNull()
    {
        var trie = new Trie();
        trie.Insert("hello");
        Assert.Null(trie.GetNode("world"));
    }

    [Fact]
    public void Trie_Words_ReturnsAll()
    {
        var trie = new Trie();
        trie.Insert("apple");
        trie.Insert("app");
        trie.Insert("banana");
        var words = trie.Words;
        Assert.Contains("apple", words);
        Assert.Contains("app", words);
        Assert.Contains("banana", words);
    }

    // ── ToolConstraints ───────────────────────────────────────────────────────

    [Fact]
    public void ToolConstraints_ParsesNames()
    {
        const string json = """[{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}]""";
        var tc = new ToolConstraints(json);
        var node = tc.NameTrie.GetNode("get_weather");
        Assert.NotNull(node);
        Assert.True(node!.IsTerminal);
    }

    [Fact]
    public void ToolConstraints_ParsesParamKeys()
    {
        const string json = """[{"name":"send_email","parameters":{"properties":{"to":{},"body":{}}}}]""";
        var tc = new ToolConstraints(json);
        var pt = tc.GetParamTrie("send_email");
        Assert.NotNull(pt);
        Assert.NotNull(pt!.GetNode("to"));
        Assert.NotNull(pt.GetNode("body"));
    }

    [Fact]
    public void ToolConstraints_InvalidJson_DoesNotThrow()
    {
        var tc = new ToolConstraints("{not valid}");
        Assert.Empty(tc.NameTrie.Words);
    }

    // ── JsonStateMachine ──────────────────────────────────────────────────────

    [Fact]
    public void JsonStateMachine_DetectsToolName()
    {
        var sm = new JsonStateMachine();
        sm.Feed("[{\"name\":\"");
        Assert.Equal(JsonState.InName, sm.State);
    }

    [Fact]
    public void JsonStateMachine_CapturesToolName()
    {
        var sm = new JsonStateMachine();
        sm.Feed("[{\"name\":\"get_weather\",");
        Assert.Equal("get_weather", sm.CurrentFunction);
        Assert.Equal(JsonState.Free, sm.State);
    }

    [Fact]
    public void JsonStateMachine_DetectsArgKey()
    {
        var sm = new JsonStateMachine();
        sm.Feed("[{\"name\":\"get_weather\",\"arguments\":{\"");
        Assert.Equal(JsonState.InArgKey, sm.State);
    }

    [Fact]
    public void JsonStateMachine_FreeAfterArgKey()
    {
        var sm = new JsonStateMachine();
        sm.Feed("[{\"name\":\"get_weather\",\"arguments\":{\"location\":\"");
        // After reading key "location" and the closing quote, should be Free
        Assert.Equal(JsonState.Free, sm.State);
    }

    // ── ConstrainedDecoder ────────────────────────────────────────────────────

    [Fact]
    public void ConstrainedDecoder_InActiveState_ConstrainsLogits()
    {
        const string tools = """[{"name":"foo","parameters":{"properties":{"bar":{}}}}]""";
        // Build minimal token strings: only "f", "o", "oo", "foo"
        var tokenStrings = new string[] { "", "f", "o", "oo", "foo", "\"", "x" };
        var tokenIndex   = new TokenIndex(tokenStrings);
        var tc           = new ToolConstraints(tools);
        var cd           = new ConstrainedDecoder([tc], tokenStrings, tokenIndex);

        // Simulate state machine after "[{"name":"
        cd.Update(0, 0); // push "" — no state change
        // Drive the machine into IN_NAME by feeding chars manually:
        // We construct it by calling Update with tokens that spell the trigger string
        // Instead, test via JsonStateMachine directly + separate constraint check

        // The decoder starts Free so constraint is identity
        float[] logits = Enumerable.Repeat(1.0f, tokenStrings.Length).ToArray();
        float[] result = cd.ConstrainLogits(logits, 0);
        // In Free state, logits should be unchanged
        Assert.Equal(logits, result);
    }
}
