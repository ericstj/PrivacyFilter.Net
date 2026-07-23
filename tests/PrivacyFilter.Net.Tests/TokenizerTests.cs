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

    [Fact]
    public void EncodeIsThreadSafeUnderConcurrency()
    {
        // The tokenizer is now per-thread (replacing a global lock), so a single
        // shared instance must serve concurrent Encode calls without corruption.
        // Hammer it from many threads against the oracle: mismatched ids/offsets or
        // an exception would reveal unsafe shared mutable state.
        string path = Path.Combine(AppContext.BaseDirectory, "tokenizer_oracle.json");
        TokenizerOracleCase[] cases =
            JsonSerializer.Deserialize<TokenizerOracleCase[]>(
                File.ReadAllBytes(path),
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;
        var tokenizer = new PrivacyFilterTokenizer();

        Parallel.For(
            0,
            256,
            new ParallelOptions
            {
                MaxDegreeOfParallelism = Math.Max(4, Environment.ProcessorCount),
            },
            _ =>
            {
                foreach (TokenizerOracleCase testCase in cases)
                {
                    TokenizedText actual = tokenizer.Encode(testCase.Text);
                    Assert.Equal(testCase.Ids, actual.TokenIds);
                    Assert.Equal(testCase.Decoded, actual.DecodedText);
                    Assert.Equal(testCase.Starts, actual.CharacterStarts);
                    Assert.Equal(testCase.Ends, actual.CharacterEnds);
                }
            });
    }

    private sealed record TokenizerOracleCase(
        string Text,
        int[] Ids,
        string Decoded,
        int[] Starts,
        int[] Ends);
}
