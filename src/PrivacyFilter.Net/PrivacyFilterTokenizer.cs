using Microsoft.ML.Tokenizers;

namespace PrivacyFilterNet;

internal sealed class PrivacyFilterTokenizer : IDisposable
{
    // TiktokenTokenizer keeps internal encode caches that are not guaranteed safe
    // for concurrent mutation, so give each thread its own instance rather than
    // serializing all tokenization behind a global lock. Combined with ONNX
    // Runtime's thread-safe Run, this lets a single shared PrivacyFilter serve many
    // concurrent Redact calls at full throughput instead of bottlenecking here.
    private readonly ThreadLocal<TiktokenTokenizer> _tokenizer =
        new(() => TiktokenTokenizer.CreateForEncoding("o200k_base"));

    public TokenizedText Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        TiktokenTokenizer tokenizer = _tokenizer.Value!;
        IReadOnlyList<EncodedToken> encodedTokens =
            tokenizer.EncodeToTokens(text, out _, considerPreTokenization: true, considerNormalization: true);
        int[] ids = encodedTokens.Select(token => token.Id).ToArray();
        if (ids.Length == 0)
        {
            return new TokenizedText(ids, text, [], [], DecodedMismatch: false);
        }

        string decodedText = tokenizer.Decode(ids);
        bool decodedMismatch = !string.Equals(decodedText, text, StringComparison.Ordinal);
        if (decodedMismatch)
        {
            encodedTokens =
                tokenizer.EncodeToTokens(decodedText, out _, considerPreTokenization: true, considerNormalization: true);
            if (encodedTokens.Count != ids.Length ||
                !encodedTokens.Select(token => token.Id).SequenceEqual(ids))
            {
                throw new InvalidDataException(
                    "Tokenizer decode did not produce a stable token sequence for span offsets.");
            }
        }

        var charStarts = new int[encodedTokens.Count];
        var charEnds = new int[encodedTokens.Count];
        for (int index = 0; index < encodedTokens.Count; index++)
        {
            Range offset = encodedTokens[index].Offset;
            charStarts[index] = offset.Start.GetOffset(decodedText.Length);
            charEnds[index] = offset.End.GetOffset(decodedText.Length);
        }

        return new TokenizedText(
            ids,
            decodedText,
            charStarts,
            charEnds,
            DecodedMismatch: decodedMismatch);
    }

    public void Dispose() => _tokenizer.Dispose();
}

internal sealed record TokenizedText(
    int[] TokenIds,
    string DecodedText,
    int[] CharacterStarts,
    int[] CharacterEnds,
    bool DecodedMismatch);
