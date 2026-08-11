namespace Needle.Tokenizer;

/// <summary>
/// The reserved token IDs and chat markers of the Needle 2 vocabulary.
/// Port of the constants in <c>.reference/needle/model/tokenizer.py</c>.
///
/// IDs 0–13 are fixed by the tokenizer build, so they can be relied on without
/// looking them up.
/// </summary>
public static class ChatMarkers
{
    /// <summary>Padding.</summary>
    public const int PadId = 0;

    /// <summary>End of sequence.</summary>
    public const int EosId = 1;

    /// <summary>Beginning of sequence.</summary>
    public const int BosId = 2;

    /// <summary>Unknown piece.</summary>
    public const int UnkId = 3;

    /// <summary>Opens a chat turn: <c>&lt;|im_start|&gt;role\n</c>.</summary>
    public const string ImStart = "<|im_start|>";

    /// <summary>Closes a chat turn.</summary>
    public const string ImEnd = "<|im_end|>";

    /// <summary>Opens the model's short derivation of each argument.</summary>
    public const string ThinkStart = "<think>";

    /// <summary>Closes the derivation.</summary>
    public const string ThinkEnd = "</think>";

    /// <summary>Opens the declared tool schemas.</summary>
    public const string ToolsStart = "<tools>";

    /// <summary>Closes the declared tool schemas.</summary>
    public const string ToolsEnd = "</tools>";

    /// <summary>Opens the emitted call array.</summary>
    public const string ToolCallStart = "<tool_call>";

    /// <summary>Closes the emitted call array.</summary>
    public const string ToolCallEnd = "</tool_call>";

    /// <summary>Opens a tool result fed back into the conversation.</summary>
    public const string ToolResultStart = "<tool_result>";

    /// <summary>Closes a tool result.</summary>
    public const string ToolResultEnd = "</tool_result>";

    /// <summary><c>&lt;|im_start|&gt;</c>.</summary>
    public const int ImStartId = 4;

    /// <summary><c>&lt;|im_end|&gt;</c>.</summary>
    public const int ImEndId = 5;

    /// <summary><c>&lt;think&gt;</c>.</summary>
    public const int ThinkStartId = 6;

    /// <summary><c>&lt;/think&gt;</c>.</summary>
    public const int ThinkEndId = 7;

    /// <summary><c>&lt;tools&gt;</c>.</summary>
    public const int ToolsStartId = 8;

    /// <summary><c>&lt;/tools&gt;</c>.</summary>
    public const int ToolsEndId = 9;

    /// <summary><c>&lt;tool_call&gt;</c>.</summary>
    public const int ToolCallStartId = 10;

    /// <summary><c>&lt;/tool_call&gt;</c>.</summary>
    public const int ToolCallEndId = 11;

    /// <summary><c>&lt;tool_result&gt;</c>.</summary>
    public const int ToolResultStartId = 12;

    /// <summary><c>&lt;/tool_result&gt;</c>.</summary>
    public const int ToolResultEndId = 13;

    /// <summary>Every chat marker, in ID order.</summary>
    public static readonly string[] All =
    [
        ImStart, ImEnd, ThinkStart, ThinkEnd,
        ToolsStart, ToolsEnd, ToolCallStart, ToolCallEnd,
        ToolResultStart, ToolResultEnd,
    ];

    /// <summary>Throws unless <paramref name="tokenizer"/> assigns the expected IDs.</summary>
    public static void Validate(CactTokenizer tokenizer)
    {
        for (int i = 0; i < All.Length; i++)
        {
            int expected = ImStartId + i;
            int actual = tokenizer.Id(All[i]);
            if (actual != expected)
                throw new InvalidDataException(
                    $"Tokenizer maps '{All[i]}' to {actual}, expected {expected}. "
                    + "This vocabulary is not the Needle 2 one.");
        }
    }
}

/// <summary>
/// Renders a turn into the prompt format the model was trained on.
/// Port of <c>render_example</c> in <c>.reference/needle/model/finetune.py</c>,
/// which is both the fine-tuning encoder and the inference prompt builder.
/// </summary>
public static class ChatTemplate
{
    /// <summary>
    /// Build the prompt for one turn.  The model continues from
    /// <c>&lt;|im_start|&gt;assistant\n</c>.
    /// </summary>
    /// <param name="query">The user's request, or a passage to extract from.</param>
    /// <param name="toolsJson">The declared schemas as a JSON array string.</param>
    /// <param name="system">
    /// Optional system facts — environment state such as
    /// <c>date: 2026-07-21 Tue 14:30; locale: en-US</c>.  Never instructions.
    /// </param>
    public static string Prompt(string query, string toolsJson, string? system = null)
    {
        string prefix = string.IsNullOrEmpty(system)
            ? ""
            : ChatMarkers.ImStart + "system\n" + system + ChatMarkers.ImEnd + "\n";

        return prefix
               + ChatMarkers.ImStart + "user\n"
               + ChatMarkers.ToolsStart + toolsJson + ChatMarkers.ToolsEnd + "\n"
               + query + ChatMarkers.ImEnd + "\n"
               + ChatMarkers.ImStart + "assistant\n";
    }

    /// <summary>
    /// Build the target continuation for a training example: the derivation
    /// followed by the call array.
    /// </summary>
    /// <param name="answersJson">The expected calls as a JSON array string.</param>
    /// <param name="reasoning">The short derivation, or empty.</param>
    public static string Target(string answersJson, string reasoning = "") =>
        ChatMarkers.ThinkStart + reasoning + ChatMarkers.ThinkEnd + "\n"
        + ChatMarkers.ToolCallStart + answersJson + ChatMarkers.ToolCallEnd + ChatMarkers.ImEnd;

    /// <summary>
    /// Wrap an executed tool's result as the next user turn, so the model can
    /// continue from it.
    /// </summary>
    public static string ResultTurn(string resultJson) =>
        ChatMarkers.ImStart + "user\n"
        + ChatMarkers.ToolResultStart + resultJson + ChatMarkers.ToolResultEnd
        + ChatMarkers.ImEnd + "\n" + ChatMarkers.ImStart + "assistant\n";
}
