using System.Buffers;
using System.Text.Json;

namespace PrivacyFilterNet;

internal sealed class ViterbiDecoder
{
    private const float NegativeInfinity = -1e9f;

    private static readonly string[] BiasKeys =
    [
        "transition_bias_background_stay",
        "transition_bias_background_to_start",
        "transition_bias_inside_to_continue",
        "transition_bias_inside_to_end",
        "transition_bias_end_to_background",
        "transition_bias_end_to_start",
    ];

    private readonly LabelSpace _labels;
    private readonly float[] _startScores;
    private readonly float[] _endScores;
    private readonly int[] _predecessorOffsets;
    private readonly byte[] _predecessors;
    private readonly float[] _transitionScores;

    public ViterbiDecoder(LabelSpace labels, string? calibrationPath)
    {
        _labels = labels;
        ViterbiBiases biases = ViterbiBiases.Load(calibrationPath);
        int classCount = labels.TokenClassNames.Length;
        if (classCount > byte.MaxValue + 1)
        {
            throw new InvalidDataException("Viterbi decoding supports at most 256 token classes.");
        }

        _startScores = Enumerable.Repeat(NegativeInfinity, classCount).ToArray();
        _endScores = Enumerable.Repeat(NegativeInfinity, classCount).ToArray();
        var validPredecessors = new List<byte>[classCount];
        var validTransitionScores = new List<float>[classCount];
        for (int next = 0; next < classCount; next++)
        {
            validPredecessors[next] = [];
            validTransitionScores[next] = [];
        }

        for (int previous = 0; previous < classCount; previous++)
        {
            char? previousTag = labels.TokenBoundaryTags[previous];
            int previousSpan = labels.TokenToSpanLabel[previous];
            if (previous == labels.BackgroundTokenLabel || previousTag is 'B' or 'S')
            {
                _startScores[previous] = 0;
            }

            if (previous == labels.BackgroundTokenLabel || previousTag is 'E' or 'S')
            {
                _endScores[previous] = 0;
            }

            for (int next = 0; next < classCount; next++)
            {
                char? nextTag = labels.TokenBoundaryTags[next];
                int nextSpan = labels.TokenToSpanLabel[next];
                if (IsValidTransition(previous, previousTag, previousSpan, next, nextTag, nextSpan))
                {
                    validPredecessors[next].Add((byte)previous);
                    validTransitionScores[next].Add(
                        TransitionBias(previous, previousTag, previousSpan, next, nextTag, nextSpan, biases));
                }
            }
        }

        _predecessorOffsets = new int[classCount + 1];
        for (int next = 0; next < classCount; next++)
        {
            _predecessorOffsets[next + 1] =
                _predecessorOffsets[next] + validPredecessors[next].Count;
        }

        _predecessors = new byte[_predecessorOffsets[^1]];
        _transitionScores = new float[_predecessors.Length];
        for (int next = 0; next < classCount; next++)
        {
            validPredecessors[next].CopyTo(_predecessors, _predecessorOffsets[next]);
            validTransitionScores[next].CopyTo(_transitionScores, _predecessorOffsets[next]);
        }
    }

    public int[] Decode(ReadOnlySpan<float> emissions, int tokenCount)
    {
        int classCount = _labels.TokenClassNames.Length;
        if (emissions.Length != tokenCount * classCount)
        {
            throw new ArgumentException("Emission dimensions do not match the label space.", nameof(emissions));
        }

        if (tokenCount == 0)
        {
            return [];
        }

        Span<float> previousScores = stackalloc float[classCount];
        Span<float> nextScores = stackalloc float[classCount];
        int backpointerCount = Math.Max(0, tokenCount - 1) * classCount;
        byte[]? backpointerBuffer =
            backpointerCount == 0 ? null : ArrayPool<byte>.Shared.Rent(backpointerCount);
        Span<byte> backpointers = backpointerBuffer.AsSpan(0, backpointerCount);
        try
        {
            for (int label = 0; label < classCount; label++)
            {
                previousScores[label] = emissions[label] + _startScores[label];
            }

            for (int token = 1; token < tokenCount; token++)
            {
                int emissionOffset = token * classCount;
                int backpointerOffset = (token - 1) * classCount;
                for (int next = 0; next < classCount; next++)
                {
                    float bestScore = float.NegativeInfinity;
                    byte bestPrevious = 0;
                    int predecessorEnd = _predecessorOffsets[next + 1];
                    for (int predecessorIndex = _predecessorOffsets[next];
                         predecessorIndex < predecessorEnd;
                         predecessorIndex++)
                    {
                        byte previous = _predecessors[predecessorIndex];
                        float score = previousScores[previous] +
                            _transitionScores[predecessorIndex];
                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestPrevious = previous;
                        }
                    }

                    nextScores[next] = bestScore + emissions[emissionOffset + next];
                    backpointers[backpointerOffset + next] = bestPrevious;
                }

                Span<float> temporaryScores = previousScores;
                previousScores = nextScores;
                nextScores = temporaryScores;
            }

            int lastLabel = 0;
            float bestFinalScore = float.NegativeInfinity;
            for (int label = 0; label < classCount; label++)
            {
                float score = previousScores[label] + _endScores[label];
                if (score > bestFinalScore)
                {
                    bestFinalScore = score;
                    lastLabel = label;
                }
            }

            var path = new int[tokenCount];
            path[^1] = lastLabel;
            for (int token = tokenCount - 2; token >= 0; token--)
            {
                lastLabel = backpointers[(token * classCount) + lastLabel];
                path[token] = lastLabel;
            }

            return path;
        }
        finally
        {
            if (backpointerBuffer is not null)
            {
                ArrayPool<byte>.Shared.Return(backpointerBuffer, clearArray: true);
            }
        }
    }

    private bool IsValidTransition(
        int previous,
        char? previousTag,
        int previousSpan,
        int next,
        char? nextTag,
        int nextSpan)
    {
        bool previousIsBackground =
            previous == _labels.BackgroundTokenLabel || previousSpan == _labels.BackgroundSpanLabel;
        bool nextIsBackground =
            next == _labels.BackgroundTokenLabel || nextSpan == _labels.BackgroundSpanLabel;

        if (previousIsBackground || previousTag is 'E' or 'S')
        {
            return nextIsBackground || nextTag is 'B' or 'S';
        }

        if (previousTag is 'B' or 'I')
        {
            return previousSpan == nextSpan && nextTag is 'I' or 'E';
        }

        return false;
    }

    private float TransitionBias(
        int previous,
        char? previousTag,
        int previousSpan,
        int next,
        char? nextTag,
        int nextSpan,
        ViterbiBiases biases)
    {
        bool previousIsBackground =
            previous == _labels.BackgroundTokenLabel || previousSpan == _labels.BackgroundSpanLabel;
        bool nextIsBackground =
            next == _labels.BackgroundTokenLabel || nextSpan == _labels.BackgroundSpanLabel;

        if (previousIsBackground)
        {
            return nextIsBackground ? biases.BackgroundStay : biases.BackgroundToStart;
        }

        if (previousTag is 'B' or 'I')
        {
            return nextTag == 'I' ? biases.InsideToContinue : biases.InsideToEnd;
        }

        if (previousTag is 'E' or 'S')
        {
            return nextIsBackground ? biases.EndToBackground : biases.EndToStart;
        }

        return 0;
    }

    private sealed record ViterbiBiases(
        float BackgroundStay,
        float BackgroundToStart,
        float InsideToContinue,
        float InsideToEnd,
        float EndToBackground,
        float EndToStart)
    {
        public static ViterbiBiases Load(string? path)
        {
            if (path is null)
            {
                return new(0, 0, 0, 0, 0, 0);
            }

            if (!File.Exists(path))
            {
                throw new FileNotFoundException("Viterbi calibration file was not found.", path);
            }

            using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(path));
            JsonElement root = document.RootElement;
            JsonElement biases = root.GetProperty("operating_points").GetProperty("default").GetProperty("biases");
            var keys = biases.EnumerateObject().Select(property => property.Name).ToHashSet(StringComparer.Ordinal);
            if (!keys.SetEquals(BiasKeys))
            {
                throw new InvalidDataException("Viterbi calibration biases do not match the expected schema.");
            }

            return new(
                ReadBias(biases, BiasKeys[0]),
                ReadBias(biases, BiasKeys[1]),
                ReadBias(biases, BiasKeys[2]),
                ReadBias(biases, BiasKeys[3]),
                ReadBias(biases, BiasKeys[4]),
                ReadBias(biases, BiasKeys[5]));
        }

        private static float ReadBias(JsonElement biases, string name)
        {
            JsonElement value = biases.GetProperty(name);
            if (value.ValueKind != JsonValueKind.Number || !value.TryGetSingle(out float result))
            {
                throw new InvalidDataException($"Viterbi calibration bias '{name}' must be numeric.");
            }

            return result;
        }
    }
}
