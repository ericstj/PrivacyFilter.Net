using System.Text.Json;

namespace PrivacyFilterNet.Tests;

public sealed class PrivacyFilterTests
{
    private static string ModelDirectory =>
        Path.Combine(AppContext.BaseDirectory, "models", "tiny");

    [Fact]
    public void RunsCompleteOnnxPipeline()
    {
        using var filter = PrivacyFilter.Load(ModelDirectory);

        PrivacyFilterResult result = filter.Redact("Alice");

        PrivacyFilterSpan span = Assert.Single(result.DetectedSpans);
        Assert.Equal("private_person", span.Label);
        Assert.Equal(0, span.Start);
        Assert.Equal(5, span.End);
        Assert.Equal("Alice", span.Text);
        Assert.Equal("<PRIVATE_PERSON>", span.Placeholder);
        Assert.Equal("<PRIVATE_PERSON>", result.RedactedText);
        Assert.Equal(1, result.Summary.SpanCount);
        Assert.False(result.Summary.DecodedMismatch);
    }

    [Fact]
    public void ProcessesMultipleWindows()
    {
        using var filter = PrivacyFilter.Load(
            ModelDirectory,
            new PrivacyFilterOptions
            {
                ContextWindowLength = 1,
            });

        PrivacyFilterResult result = filter.Redact("Alice Bob");

        Assert.Equal(2, result.DetectedSpans.Count);
        Assert.Equal("Alice", result.DetectedSpans[0].Text);
        Assert.Equal("Bob", result.DetectedSpans[1].Text);
        Assert.Equal("<PRIVATE_PERSON> <PRIVATE_PERSON>", result.RedactedText);
    }

    [Fact]
    public void SerializesUpstreamSchemaNames()
    {
        using var filter = PrivacyFilter.Load(
            ModelDirectory,
            new PrivacyFilterOptions
            {
                ModelFileName = Path.Combine("onnx", "model.onnx"),
                OutputMode = PrivacyFilterOutputMode.Redacted,
            });

        string json = filter.Redact("Alice").ToJson(indented: false);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        Assert.Equal(1, root.GetProperty("schema_version").GetInt32());
        Assert.Equal("redacted", root.GetProperty("summary").GetProperty("output_mode").GetString());
        Assert.Equal("<REDACTED>", root.GetProperty("redacted_text").GetString());
        Assert.False(root.TryGetProperty("warning", out _));
    }
}
