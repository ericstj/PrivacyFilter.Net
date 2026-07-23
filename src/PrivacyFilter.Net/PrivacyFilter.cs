using System.Buffers;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace PrivacyFilterNet;

/// <summary>
/// Runs OpenAI Privacy Filter ONNX inference and managed BIOES span decoding.
/// </summary>
public sealed class PrivacyFilter : IDisposable
{
    private static readonly string[] DefaultModelCandidates =
    [
        Path.Combine("onnx", "model_q4f16.onnx"),
        Path.Combine("onnx", "model_quantized.onnx"),
        Path.Combine("onnx", "model_q4.onnx"),
        Path.Combine("onnx", "model_fp16.onnx"),
        Path.Combine("onnx", "model.onnx"),
    ];

    private readonly InferenceSession _session;
    private readonly PrivacyFilterTokenizer _tokenizer;
    private readonly LabelSpace _labels;
    private readonly ViterbiDecoder _viterbi;
    private readonly PrivacyFilterOptions _options;
    private bool _disposed;

    private PrivacyFilter(
        InferenceSession session,
        PrivacyFilterTokenizer tokenizer,
        LabelSpace labels,
        ViterbiDecoder viterbi,
        PrivacyFilterOptions options)
    {
        _session = session;
        _tokenizer = tokenizer;
        _labels = labels;
        _viterbi = viterbi;
        _options = options;
    }

    /// <summary>Loads a reusable privacy-filter instance from a Hugging Face checkpoint folder.</summary>
    public static PrivacyFilter Load(string checkpointDirectory, PrivacyFilterOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointDirectory);
        string fullCheckpointDirectory = Path.GetFullPath(checkpointDirectory);
        if (!Directory.Exists(fullCheckpointDirectory))
        {
            throw new DirectoryNotFoundException(
                $"Privacy Filter checkpoint directory was not found: {fullCheckpointDirectory}");
        }

        options ??= new PrivacyFilterOptions();
        if (options.ContextWindowLength <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                options.ContextWindowLength,
                "ContextWindowLength must be positive.");
        }

        ModelConfig config = ModelConfig.Load(fullCheckpointDirectory);
        LabelSpace labels = LabelSpace.Create(config.ClassNames);
        string modelPath = ResolveModelPath(fullCheckpointDirectory, options.ModelFileName);
        string calibrationPath = Path.Combine(fullCheckpointDirectory, "viterbi_calibration.json");
        var session = options.SessionOptions is null
            ? new InferenceSession(modelPath)
            : new InferenceSession(modelPath, options.SessionOptions);
        try
        {
            ValidateModelContract(session, labels.TokenClassNames.Length);
            return new PrivacyFilter(
                session,
                new PrivacyFilterTokenizer(),
                labels,
                new ViterbiDecoder(labels, File.Exists(calibrationPath) ? calibrationPath : null),
                CloneOptions(options));
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    /// <summary>Detects private spans and returns structured and redacted output.</summary>
    public PrivacyFilterResult Redact(string text)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(text);

        TokenizedText tokenized = _tokenizer.Encode(text);
        if (tokenized.TokenIds.Length == 0)
        {
            return BuildResult(tokenized, []);
        }

        int classCount = _labels.TokenClassNames.Length;
        int scoreCount = checked(tokenized.TokenIds.Length * classCount);
        float[] scoreBuffer = ArrayPool<float>.Shared.Rent(scoreCount);
        try
        {
            Span<float> scores = scoreBuffer.AsSpan(0, scoreCount);
            RunModel(tokenized.TokenIds, scores);
            int[] decodedLabels = _options.DecodeMode == PrivacyFilterDecodeMode.Viterbi
                ? _viterbi.Decode(scores, tokenized.TokenIds.Length)
                : ArgMax(scores, tokenized.TokenIds.Length, classCount);
            List<PrivacyFilterSpan> spans = SpanDecoder.Decode(
                decodedLabels,
                _labels,
                tokenized,
                _options.TrimWhitespace,
                _options.DiscardOverlappingSpans,
                _options.OutputMode);
            return BuildResult(tokenized, spans);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(scoreBuffer, clearArray: true);
        }
    }

    /// <summary>Detects private spans and returns only the redacted text.</summary>
    public string RedactText(string text) => Redact(text).RedactedText;

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _session.Dispose();
        _disposed = true;
    }

    private void RunModel(int[] tokenIds, Span<float> scores)
    {
        int classCount = _labels.TokenClassNames.Length;
        if (scores.Length != tokenIds.Length * classCount)
        {
            throw new ArgumentException("Score dimensions do not match the token input.", nameof(scores));
        }

        for (int windowStart = 0;
             windowStart < tokenIds.Length;
             windowStart += _options.ContextWindowLength)
        {
            int windowLength = Math.Min(_options.ContextWindowLength, tokenIds.Length - windowStart);
            long[] ids = ArrayPool<long>.Shared.Rent(windowLength);
            long[] mask = ArrayPool<long>.Shared.Rent(windowLength);
            try
            {
                for (int index = 0; index < windowLength; index++)
                {
                    ids[index] = tokenIds[windowStart + index];
                    mask[index] = 1;
                }

                var idTensor = new DenseTensor<long>(
                    ids.AsMemory(0, windowLength),
                    [1, windowLength]);
                var maskTensor = new DenseTensor<long>(
                    mask.AsMemory(0, windowLength),
                    [1, windowLength]);
                using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(
                [
                    NamedOnnxValue.CreateFromTensor("input_ids", idTensor),
                    NamedOnnxValue.CreateFromTensor("attention_mask", maskTensor),
                ]);
                DisposableNamedOnnxValue? logitsValue =
                    results.FirstOrDefault(result => result.Name == "logits");
                if (logitsValue is null)
                {
                    throw new InvalidDataException("The ONNX model did not return logits.");
                }

                Tensor<float> logits = logitsValue.AsTensor<float>();
                if (logits.Dimensions.Length != 3 ||
                    logits.Dimensions[0] != 1 ||
                    logits.Dimensions[1] != windowLength ||
                    logits.Dimensions[2] != classCount)
                {
                    throw new InvalidDataException(
                        $"The ONNX logits shape must be [1,{windowLength},{classCount}].");
                }

                int scoreOffset = windowStart * classCount;
                int windowScoreCount = windowLength * classCount;
                Span<float> windowScores = scores.Slice(scoreOffset, windowScoreCount);
                if (logits is DenseTensor<float> denseLogits)
                {
                    denseLogits.Buffer.Span.CopyTo(windowScores);
                }
                else
                {
                    for (int index = 0; index < windowScoreCount; index++)
                    {
                        windowScores[index] = logits.GetValue(index);
                    }
                }
            }
            finally
            {
                ArrayPool<long>.Shared.Return(ids, clearArray: true);
                ArrayPool<long>.Shared.Return(mask, clearArray: true);
            }
        }
    }

    internal static void LogSoftmax(
        float[] source,
        float[] destination,
        int sourceTokenCount,
        int classCount,
        int destinationTokenOffset)
    {
        for (int token = 0; token < sourceTokenCount; token++)
        {
            int sourceOffset = token * classCount;
            float maximum = float.NegativeInfinity;
            for (int label = 0; label < classCount; label++)
            {
                maximum = Math.Max(maximum, source[sourceOffset + label]);
            }

            double exponentialSum = 0;
            for (int label = 0; label < classCount; label++)
            {
                exponentialSum += Math.Exp(source[sourceOffset + label] - maximum);
            }

            float logDenominator = maximum + (float)Math.Log(exponentialSum);
            int destinationOffset = (destinationTokenOffset + token) * classCount;
            for (int label = 0; label < classCount; label++)
            {
                destination[destinationOffset + label] =
                    source[sourceOffset + label] - logDenominator;
            }
        }
    }

    internal static int[] ArgMax(ReadOnlySpan<float> scores, int tokenCount, int classCount)
    {
        var labels = new int[tokenCount];
        for (int token = 0; token < tokenCount; token++)
        {
            int offset = token * classCount;
            int bestLabel = 0;
            float bestScore = scores[offset];
            for (int label = 1; label < classCount; label++)
            {
                float score = scores[offset + label];
                if (score > bestScore)
                {
                    bestScore = score;
                    bestLabel = label;
                }
            }

            labels[token] = bestLabel;
        }

        return labels;
    }

    private PrivacyFilterResult BuildResult(
        TokenizedText tokenized,
        IReadOnlyList<PrivacyFilterSpan> spans)
    {
        var byLabel = spans
            .GroupBy(span => span.Label, StringComparer.Ordinal)
            .OrderBy(group => group.Key, StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => group.Count(), StringComparer.Ordinal);
        string outputMode = _options.OutputMode == PrivacyFilterOutputMode.Typed
            ? "typed"
            : "redacted";
        string redactedText = ApplyRedactions(tokenized.DecodedText, spans);
        string? warning = tokenized.DecodedMismatch
            ? "Input text did not exactly match tokenizer round-trip decode; spans are based on decoded token text."
            : null;
        return new PrivacyFilterResult(
            SchemaVersion: 1,
            new PrivacyFilterSummary(outputMode, spans.Count, byLabel, tokenized.DecodedMismatch),
            tokenized.DecodedText,
            spans,
            redactedText,
            warning);
    }

    internal static string ApplyRedactions(string text, IReadOnlyList<PrivacyFilterSpan> spans)
    {
        if (spans.Count == 0)
        {
            return text;
        }

        var pieces = new System.Text.StringBuilder(text.Length);
        int cursor = 0;
        foreach (PrivacyFilterSpan span in spans)
        {
            pieces.Append(text, cursor, span.Start - cursor);
            pieces.Append(span.Placeholder);
            cursor = span.End;
        }

        pieces.Append(text, cursor, text.Length - cursor);
        return pieces.ToString();
    }

    private static string ResolveModelPath(string checkpointDirectory, string? configuredPath)
    {
        if (!string.IsNullOrWhiteSpace(configuredPath))
        {
            string path = Path.IsPathRooted(configuredPath)
                ? configuredPath
                : Path.Combine(checkpointDirectory, configuredPath);
            path = Path.GetFullPath(path);
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("Configured ONNX model was not found.", path);
            }

            return path;
        }

        foreach (string candidate in DefaultModelCandidates)
        {
            string path = Path.Combine(checkpointDirectory, candidate);
            if (File.Exists(path))
            {
                return path;
            }
        }

        throw new FileNotFoundException(
            $"No supported ONNX model was found under {Path.Combine(checkpointDirectory, "onnx")}.");
    }

    private static void ValidateModelContract(InferenceSession session, int classCount)
    {
        if (!session.InputMetadata.TryGetValue("input_ids", out NodeMetadata? inputIds) ||
            inputIds.ElementType != typeof(long))
        {
            throw new InvalidDataException("The ONNX model must define an INT64 input_ids input.");
        }

        if (!session.InputMetadata.TryGetValue("attention_mask", out NodeMetadata? attentionMask) ||
            attentionMask.ElementType != typeof(long))
        {
            throw new InvalidDataException("The ONNX model must define an INT64 attention_mask input.");
        }

        if (!session.OutputMetadata.TryGetValue("logits", out NodeMetadata? logits) ||
            logits.ElementType != typeof(float) ||
            logits.Dimensions.Length != 3 ||
            (logits.Dimensions[2] > 0 && logits.Dimensions[2] != classCount))
        {
            throw new InvalidDataException(
                $"The ONNX model must define FLOAT logits shaped [batch,sequence,{classCount}].");
        }
    }

    private static PrivacyFilterOptions CloneOptions(PrivacyFilterOptions options)
    {
        return new PrivacyFilterOptions
        {
            ModelFileName = options.ModelFileName,
            ContextWindowLength = options.ContextWindowLength,
            TrimWhitespace = options.TrimWhitespace,
            DiscardOverlappingSpans = options.DiscardOverlappingSpans,
            OutputMode = options.OutputMode,
            DecodeMode = options.DecodeMode,
            SessionOptions = null,
        };
    }
}
