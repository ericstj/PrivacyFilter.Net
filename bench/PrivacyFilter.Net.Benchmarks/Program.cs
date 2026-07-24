using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using PrivacyFilterNet;

BenchmarkSwitcher.FromAssembly(typeof(PrivacyFilterBenchmarks).Assembly).Run(args);

[MemoryDiagnoser]
public class PrivacyFilterBenchmarks
{
    private const string ShortInput =
        "Alice Smith can be reached at alice@example.com or +1 (425) 555-0100.";

    private readonly string _longInput = string.Join(
        Environment.NewLine,
        Enumerable.Repeat(ShortInput, 50));
    private PrivacyFilter? _filter;

    [GlobalSetup]
    public void Setup()
    {
        string modelDirectory =
            Environment.GetEnvironmentVariable("PRIVACY_FILTER_MODEL_DIR")
            ?? throw new InvalidOperationException(
                "Set PRIVACY_FILTER_MODEL_DIR to an openai/privacy-filter checkpoint.");
        _filter = PrivacyFilter.Load(modelDirectory);
    }

    [GlobalCleanup]
    public void Cleanup() => _filter?.Dispose();

    [Benchmark(Baseline = true)]
    public PrivacyFilterResult ShortText() => _filter!.Redact(ShortInput);

    [Benchmark]
    public PrivacyFilterResult LongText() => _filter!.Redact(_longInput);
}

[MemoryDiagnoser]
public class ManagedPostprocessingBenchmarks
{
    private const int TokenCount = 4096;
    private float[] _scores = null!;
    private float[] _logProbabilities = null!;
    private float[] _denseStartScores = null!;
    private float[] _denseEndScores = null!;
    private float[] _denseTransitionScores = null!;
    private ViterbiDecoder _decoder = null!;
    private int _classCount;

    [GlobalSetup]
    public void Setup()
    {
        LabelSpace labels = LabelSpace.Create(CreateClassNames());
        _decoder = new ViterbiDecoder(labels, calibrationPath: null);
        _classCount = labels.TokenClassNames.Length;
        _scores = new float[TokenCount * _classCount];
        _logProbabilities = new float[_scores.Length];
        var random = new Random(42);
        for (int index = 0; index < _scores.Length; index++)
        {
            _scores[index] = (random.NextSingle() * 20) - 10;
        }

        PrivacyFilter.LogSoftmax(
            _scores,
            _logProbabilities,
            TokenCount,
            _classCount,
            destinationTokenOffset: 0);
        CreateDenseScores(
            _classCount,
            out _denseStartScores,
            out _denseEndScores,
            out _denseTransitionScores);
    }

    [Benchmark]
    public void LogSoftmax() =>
        PrivacyFilter.LogSoftmax(
            _scores,
            _logProbabilities,
            TokenCount,
            _classCount,
            destinationTokenOffset: 0);

    [Benchmark(Baseline = true)]
    public int[] ViterbiDense() =>
        DecodeDense(
            _logProbabilities,
            TokenCount,
            _classCount,
            _denseStartScores,
            _denseEndScores,
            _denseTransitionScores);

    [Benchmark]
    public int[] ViterbiSparse() => _decoder.Decode(_logProbabilities, TokenCount);

    [Benchmark]
    public int[] ArgMaxScalar() =>
        ArgMaxScalar(_logProbabilities, TokenCount, _classCount);

    [Benchmark]
    public int[] ArgMaxTensorPrimitives() =>
        PrivacyFilter.ArgMax(_logProbabilities, TokenCount, _classCount);

    private static int[] ArgMaxScalar(
        ReadOnlySpan<float> scores,
        int tokenCount,
        int classCount)
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

    private static int[] DecodeDense(
        float[] emissions,
        int tokenCount,
        int classCount,
        float[] startScores,
        float[] endScores,
        float[] transitionScores)
    {
        var previousScores = new float[classCount];
        var nextScores = new float[classCount];
        var backpointers = new int[(tokenCount - 1) * classCount];
        for (int label = 0; label < classCount; label++)
        {
            previousScores[label] = emissions[label] + startScores[label];
        }

        for (int token = 1; token < tokenCount; token++)
        {
            int emissionOffset = token * classCount;
            int backpointerOffset = (token - 1) * classCount;
            for (int next = 0; next < classCount; next++)
            {
                float bestScore = float.NegativeInfinity;
                int bestPrevious = 0;
                for (int previous = 0; previous < classCount; previous++)
                {
                    float score = previousScores[previous] +
                        transitionScores[(previous * classCount) + next];
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestPrevious = previous;
                    }
                }

                nextScores[next] = bestScore + emissions[emissionOffset + next];
                backpointers[backpointerOffset + next] = bestPrevious;
            }

            (previousScores, nextScores) = (nextScores, previousScores);
        }

        int lastLabel = 0;
        float bestFinalScore = float.NegativeInfinity;
        for (int label = 0; label < classCount; label++)
        {
            float score = previousScores[label] + endScores[label];
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

    private static void CreateDenseScores(
        int classCount,
        out float[] startScores,
        out float[] endScores,
        out float[] transitionScores)
    {
        const float invalid = -1e9f;
        startScores = Enumerable.Repeat(invalid, classCount).ToArray();
        endScores = Enumerable.Repeat(invalid, classCount).ToArray();
        transitionScores = Enumerable.Repeat(invalid, classCount * classCount).ToArray();
        for (int previous = 0; previous < classCount; previous++)
        {
            char? previousTag = Boundary(previous);
            if (previous == 0 || previousTag is 'B' or 'S')
            {
                startScores[previous] = 0;
            }

            if (previous == 0 || previousTag is 'E' or 'S')
            {
                endScores[previous] = 0;
            }

            for (int next = 0; next < classCount; next++)
            {
                if (IsValidTransition(previous, next))
                {
                    transitionScores[(previous * classCount) + next] = 0;
                }
            }
        }
    }

    private static bool IsValidTransition(int previous, int next)
    {
        char? previousTag = Boundary(previous);
        char? nextTag = Boundary(next);
        if (previous == 0 || previousTag is 'E' or 'S')
        {
            return next == 0 || nextTag is 'B' or 'S';
        }

        return SpanLabel(previous) == SpanLabel(next) && nextTag is 'I' or 'E';
    }

    private static char? Boundary(int label) => label == 0 ? null : "BIES"[(label - 1) % 4];

    private static int SpanLabel(int label) => label == 0 ? 0 : ((label - 1) / 4) + 1;

    private static string[] CreateClassNames()
    {
        var names = new List<string> { "O" };
        foreach (string label in new[]
        {
            "account_number",
            "private_address",
            "private_date",
            "private_email",
            "private_person",
            "private_phone",
            "private_url",
            "secret",
        })
        {
            foreach (char boundary in "BIES")
            {
                names.Add($"{boundary}-{label}");
            }
        }

        return names.ToArray();
    }
}
