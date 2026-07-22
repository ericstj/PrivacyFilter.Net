using System.Text.Json;
using System.Text.Json.Serialization;

namespace PrivacyFilterNet;

/// <summary>One detected character span.</summary>
public sealed record PrivacyFilterSpan(
    [property: JsonPropertyName("label")] string Label,
    [property: JsonPropertyName("start")] int Start,
    [property: JsonPropertyName("end")] int End,
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("placeholder")] string Placeholder);

/// <summary>Summary information for one privacy-filter result.</summary>
public sealed record PrivacyFilterSummary(
    [property: JsonPropertyName("output_mode")] string OutputMode,
    [property: JsonPropertyName("span_count")] int SpanCount,
    [property: JsonPropertyName("by_label")] IReadOnlyDictionary<string, int> ByLabel,
    [property: JsonPropertyName("decoded_mismatch")] bool DecodedMismatch);

/// <summary>The structured result of privacy detection and redaction.</summary>
public sealed record PrivacyFilterResult(
    [property: JsonPropertyName("schema_version")] int SchemaVersion,
    [property: JsonPropertyName("summary")] PrivacyFilterSummary Summary,
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("detected_spans")] IReadOnlyList<PrivacyFilterSpan> DetectedSpans,
    [property: JsonPropertyName("redacted_text")] string RedactedText,
    [property: JsonPropertyName("warning")]
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    string? Warning = null)
{
    /// <summary>Serializes this result using the upstream JSON field names.</summary>
    public string ToJson(bool indented = true)
    {
        return JsonSerializer.Serialize(this, new JsonSerializerOptions
        {
            WriteIndented = indented,
        });
    }
}
