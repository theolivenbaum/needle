namespace Needle.Tokenizer;

/// <summary>
/// Subset of <see cref="NeedleTokenizer"/> required by training utilities
/// (<see cref="Training.FinetuneExampleBuilder"/>,
/// <see cref="Training.BatchBuilder"/>).
///
/// Pulled out as an interface so tests can supply lightweight fakes that
/// don't require the SentencePiece model file.  The full
/// <see cref="NeedleTokenizer"/> implements this transparently.
/// </summary>
public interface INeedleTokenizer
{
    int PadTokenId      { get; }
    int EosTokenId      { get; }
    int BosTokenId      { get; }
    int ToolCallTokenId { get; }
    int ToolsTokenId    { get; }
    int VocabSize       { get; }

    IReadOnlyList<int> Encode(string text);
    string Decode(IEnumerable<int> ids);
    string IdToPiece(int id);
}
