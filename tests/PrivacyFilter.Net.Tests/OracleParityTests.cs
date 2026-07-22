using System.Text.Json;

namespace PrivacyFilterNet.Tests;

public sealed class OracleParityTests
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    [Fact]
    public void ViterbiMatchesPythonOracle()
    {
        DecodingOracle oracle = LoadOracle();
        LabelSpace labels = LabelSpace.Create(oracle.ClassNames);
        string calibrationPath = Path.Combine(
            Path.GetTempPath(),
            $"privacy-filter-calibration-{Guid.NewGuid():N}.json");
        try
        {
            File.WriteAllText(
                calibrationPath,
                JsonSerializer.Serialize(new
                {
                    operating_points = new
                    {
                        @default = new
                        {
                            biases = oracle.Biases,
                        },
                    },
                }));
            var decoder = new ViterbiDecoder(labels, calibrationPath);

            foreach (ViterbiOracleCase testCase in oracle.ViterbiCases)
            {
                int tokenCount = testCase.Scores.Length;
                float[] scores = testCase.Scores.SelectMany(row => row).ToArray();

                Assert.Equal(testCase.Expected, decoder.Decode(scores, tokenCount));
            }
        }
        finally
        {
            File.Delete(calibrationPath);
        }
    }

    [Fact]
    public void SpanReconstructionMatchesPythonOracle()
    {
        DecodingOracle oracle = LoadOracle();
        LabelSpace labels = LabelSpace.Create(oracle.ClassNames);

        foreach (SpanOracleCase testCase in oracle.SpanCases)
        {
            int[][] actual = SpanDecoder.LabelsToSpans(testCase.Labels, labels)
                .Select(span => new[] { span.Label, span.Start, span.End })
                .ToArray();

            Assert.Equal(testCase.Expected, actual);
        }
    }

    [Fact]
    public void PostprocessingMatchesPythonOracle()
    {
        DecodingOracle oracle = LoadOracle();
        LabelSpace labels = LabelSpace.Create(oracle.ClassNames);
        var tokenizer = new PrivacyFilterTokenizer();

        foreach (PostprocessOracleCase testCase in oracle.PostprocessCases)
        {
            TokenizedText tokenized = tokenizer.Encode(testCase.Text);
            PrivacyFilterOutputMode outputMode =
                string.Equals(testCase.OutputMode, "redacted", StringComparison.Ordinal)
                    ? PrivacyFilterOutputMode.Redacted
                    : PrivacyFilterOutputMode.Typed;
            List<PrivacyFilterSpan> actual = SpanDecoder.Decode(
                testCase.Labels,
                labels,
                tokenized,
                testCase.TrimWhitespace,
                testCase.DiscardOverlaps,
                outputMode);

            Assert.Equal(testCase.ExpectedSpans, actual);
            Assert.Equal(
                testCase.RedactedText,
                PrivacyFilter.ApplyRedactions(tokenized.DecodedText, actual));
        }
    }

    private static DecodingOracle LoadOracle()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "decoding_oracle.json");
        return JsonSerializer.Deserialize<DecodingOracle>(
            File.ReadAllBytes(path),
            JsonOptions)!;
    }

    private sealed record DecodingOracle(
        string[] ClassNames,
        Dictionary<string, float> Biases,
        ViterbiOracleCase[] ViterbiCases,
        SpanOracleCase[] SpanCases,
        PostprocessOracleCase[] PostprocessCases);

    private sealed record ViterbiOracleCase(float[][] Scores, int[] Expected);

    private sealed record SpanOracleCase(int[] Labels, int[][] Expected);

    private sealed record PostprocessOracleCase(
        string Text,
        int[] Labels,
        bool TrimWhitespace,
        bool DiscardOverlaps,
        string OutputMode,
        PrivacyFilterSpan[] ExpectedSpans,
        string RedactedText);
}
