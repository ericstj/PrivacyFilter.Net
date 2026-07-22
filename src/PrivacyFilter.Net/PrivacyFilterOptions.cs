using Microsoft.ML.OnnxRuntime;

namespace PrivacyFilterNet;

/// <summary>Controls model loading, decoding, and output projection.</summary>
public sealed class PrivacyFilterOptions
{
    /// <summary>
    /// Gets or sets the ONNX file relative to the checkpoint directory.
    /// When omitted, the loader prefers q4f16, quantized, fp16, then fp32.
    /// </summary>
    public string? ModelFileName { get; set; }

    /// <summary>Gets or sets the maximum number of tokens processed per ONNX invocation.</summary>
    public int ContextWindowLength { get; set; } = 4096;

    /// <summary>Gets or sets whether leading and trailing whitespace is removed from spans.</summary>
    public bool TrimWhitespace { get; set; } = true;

    /// <summary>Gets or sets whether overlapping spans are discarded independently per label.</summary>
    public bool DiscardOverlappingSpans { get; set; }

    /// <summary>Gets or sets whether entity labels are retained or collapsed to redacted.</summary>
    public PrivacyFilterOutputMode OutputMode { get; set; } = PrivacyFilterOutputMode.Typed;

    /// <summary>Gets or sets the token-label decoding algorithm.</summary>
    public PrivacyFilterDecodeMode DecodeMode { get; set; } = PrivacyFilterDecodeMode.Viterbi;

    /// <summary>
    /// Gets or sets ONNX Runtime session options. The caller retains ownership of this object.
    /// </summary>
    public SessionOptions? SessionOptions { get; set; }
}

/// <summary>Controls how detected entity labels are represented.</summary>
public enum PrivacyFilterOutputMode
{
    /// <summary>Retain the model's entity labels and typed placeholders.</summary>
    Typed,

    /// <summary>Collapse all labels and placeholders to redacted.</summary>
    Redacted,
}

/// <summary>Selects the token-label decoding algorithm.</summary>
public enum PrivacyFilterDecodeMode
{
    /// <summary>Use constrained BIOES Viterbi decoding.</summary>
    Viterbi,

    /// <summary>Select the highest-scoring label independently for each token.</summary>
    ArgMax,
}
