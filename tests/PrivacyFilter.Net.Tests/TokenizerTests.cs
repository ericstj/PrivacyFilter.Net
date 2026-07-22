using System.Text.Json;

namespace PrivacyFilterNet.Tests;

public sealed class TokenizerTests
{
    [Fact]
    public void MatchesTiktokenOracle()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "tokenizer_oracle.json");
        TokenizerOracleCase[] cases =
            JsonSerializer.Deserialize<TokenizerOracleCase[]>(
                File.ReadAllBytes(path),
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;
        var tokenizer = new PrivacyFilterTokenizer();

        foreach (TokenizerOracleCase testCase in cases)
        {
            TokenizedText actual = tokenizer.Encode(testCase.Text);
            Assert.Equal(testCase.Ids, actual.TokenIds);
            Assert.Equal(testCase.Decoded, actual.DecodedText);
            Assert.Equal(testCase.Starts, actual.CharacterStarts);
            Assert.Equal(testCase.Ends, actual.CharacterEnds);
            Assert.Equal(testCase.Decoded != testCase.Text, actual.DecodedMismatch);
        }
    }

    private sealed record TokenizerOracleCase(
        string Text,
        int[] Ids,
        string Decoded,
        int[] Starts,
        int[] Ends);
}
