using Microsoft.ML.Tokenizers;

namespace PrivacyFilterNet;

internal sealed class PrivacyFilterTokenizer
{
    private readonly TiktokenTokenizer _tokenizer;
    private readonly object _gate = new();

    public PrivacyFilterTokenizer()
    {
        _tokenizer = TiktokenTokenizer.CreateForEncoding("o200k_base");
    }

    public TokenizedText Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        lock (_gate)
        {
            IReadOnlyList<EncodedToken> encodedTokens =
                _tokenizer.EncodeToTokens(text, out _, considerPreTokenization: true, considerNormalization: true);
            int[] ids = encodedTokens.Select(token => token.Id).ToArray();
            if (ids.Length == 0)
            {
                return new TokenizedText(ids, text, [], [], DecodedMismatch: false);
            }

            string decodedText = _tokenizer.Decode(ids);
            bool decodedMismatch = !string.Equals(decodedText, text, StringComparison.Ordinal);
            if (decodedMismatch)
            {
                encodedTokens =
                    _tokenizer.EncodeToTokens(decodedText, out _, considerPreTokenization: true, considerNormalization: true);
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
    }
}

internal sealed record TokenizedText(
    int[] TokenIds,
    string DecodedText,
    int[] CharacterStarts,
    int[] CharacterEnds,
    bool DecodedMismatch);
