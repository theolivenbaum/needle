using System.Collections.Immutable;
using System.Text.Json;
using Needle.Model;

namespace Needle.Weights;

/// <summary>
/// Reads a <see cref="TransformerConfig"/> out of the JSON a checkpoint carries.
///
/// Two spellings are accepted: the reference's own <c>TransformerConfig</c>
/// fields (<c>d_model</c>, <c>num_layers</c>, …), which is what the parity dump
/// writes, and the HuggingFace-style <c>config.json</c> shipped alongside the
/// released weights (<c>hidden_size</c>, <c>num_hidden_layers</c>, …).
/// </summary>
public static class CheckpointConfig
{
    /// <summary>Parse a config file.</summary>
    public static TransformerConfig Load(string path) => Parse(File.ReadAllText(path));

    /// <summary>Parse config JSON.</summary>
    public static TransformerConfig Parse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        var extras = root.TryGetProperty("extras", out var ex) ? ex : default;
        var engram = extras.ValueKind == JsonValueKind.Object
                     && extras.TryGetProperty("engram", out var eg) ? eg : default;
        var quant = root.TryGetProperty("quantization", out var q) ? q : default;

        var config = new TransformerConfig
        {
            VocabSize = Int(root, 8192, "vocab_size"),
            DModel = Int(root, 512, "d_model", "hidden_size"),
            AttnDim = Int(root, 0, "attn_dim"),
            NumHeads = Int(root, 8, "num_heads", "num_attention_heads"),
            NumKvHeads = Int(root, 4, "num_kv_heads", "num_key_value_heads"),
            NumLayers = Int(root, 12, "num_layers", "num_hidden_layers"),
            MaxSeqLen = Int(root, 2048, "max_seq_len", "max_position_embeddings"),
            PadTokenId = Int(root, 0, "pad_token_id"),
            ContrastiveDim = Int(root, 128, "contrastive_dim"),
            RopeTheta = Single(root, 100000f, "rope_theta"),
            Dtype = String(root, "float32", "dtype", "torch_dtype"),
            MhcLanes = Int(root, 4, "mhc_lanes") is var lanes && lanes != 4
                ? lanes
                : Int(extras, 4, "mhc_lanes"),
            EngramSlots = Int(root, 0, "engram_slots") is var slots && slots > 0
                ? slots
                : Int(engram, 8192, "slots"),
            EngramHeads = Int(root, 0, "engram_heads"),
            KvWindow = Int(root, 0, "kv_window", "sliding_window"),
            KvBits = Int(root, 8, "kv_bits") is var kvb && kvb != 8 ? kvb : Int(quant, 8, "kv_cache_bits"),
            ActBits = Int(root, 8, "act_bits") is var ab && ab != 8 ? ab : Int(quant, 8, "activation_bits"),
            WeightBits = String(root, "", "weight_bits") is var wb && wb.Length > 0
                ? wb
                : String(quant, "", "scheme"),
        };

        var orders = Ints(root, "engram_orders") ?? Ints(engram, "orders");
        if (orders is not null) config = config with { EngramOrders = [.. orders] };

        var sites = Ints(root, "engram_layers") ?? Ints(engram, "sites");
        if (sites is not null) config = config with { EngramLayers = [.. sites] };

        // The reference's float32/bfloat16 flag has no meaning here — the port
        // always computes in float32, as does the reference decoder.
        return config with { Dtype = "float32" };
    }

    private static int Int(JsonElement root, int fallback, params string[] names)
    {
        if (root.ValueKind != JsonValueKind.Object) return fallback;
        foreach (var name in names)
            if (root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number)
                return value.GetInt32();
        return fallback;
    }

    private static float Single(JsonElement root, float fallback, params string[] names)
    {
        if (root.ValueKind != JsonValueKind.Object) return fallback;
        foreach (var name in names)
            if (root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number)
                return value.GetSingle();
        return fallback;
    }

    private static string String(JsonElement root, string fallback, params string[] names)
    {
        if (root.ValueKind != JsonValueKind.Object) return fallback;
        foreach (var name in names)
            if (root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String)
                return value.GetString() ?? fallback;
        return fallback;
    }

    private static int[]? Ints(JsonElement root, string name)
    {
        if (root.ValueKind != JsonValueKind.Object) return null;
        if (!root.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.Array) return null;
        var result = new int[value.GetArrayLength()];
        for (int i = 0; i < result.Length; i++) result[i] = value[i].GetInt32();
        return result;
    }
}
