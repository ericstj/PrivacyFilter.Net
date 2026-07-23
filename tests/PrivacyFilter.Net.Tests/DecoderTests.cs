namespace PrivacyFilterNet.Tests;

public sealed class DecoderTests
{
    private static readonly string[] ClassNames = CreateClassNames();

    [Fact]
    public void ViterbiRejectsIllegalInsideStart()
    {
        LabelSpace labels = LabelSpace.Create(ClassNames);
        var decoder = new ViterbiDecoder(labels, calibrationPath: null);
        int classCount = ClassNames.Length;
        var emissions = Enumerable.Repeat(-10f, 2 * classCount).ToArray();
        emissions[ClassIndex("I-private_person")] = 10;
        emissions[ClassIndex("B-private_person")] = 9;
        emissions[classCount + ClassIndex("E-private_person")] = 10;

        int[] path = decoder.Decode(emissions, tokenCount: 2);

        Assert.Equal(
            [ClassIndex("B-private_person"), ClassIndex("E-private_person")],
            path);
    }

    [Fact]
    public void DecodersAreInvariantToLogSoftmax()
    {
        LabelSpace labels = LabelSpace.Create(ClassNames);
        var decoder = new ViterbiDecoder(labels, calibrationPath: null);
        int classCount = ClassNames.Length;
        var random = new Random(42);

        for (int testCase = 0; testCase < 128; testCase++)
        {
            int tokenCount = random.Next(1, 65);
            var logits = new float[tokenCount * classCount];
            for (int index = 0; index < logits.Length; index++)
            {
                logits[index] = (random.NextSingle() * 40) - 20;
            }

            var logProbabilities = new float[logits.Length];
            PrivacyFilter.LogSoftmax(
                logits,
                logProbabilities,
                tokenCount,
                classCount,
                destinationTokenOffset: 0);

            Assert.Equal(
                decoder.Decode(logProbabilities, tokenCount),
                decoder.Decode(logits, tokenCount));
            Assert.Equal(
                PrivacyFilter.ArgMax(logProbabilities, tokenCount, classCount),
                PrivacyFilter.ArgMax(logits, tokenCount, classCount));
        }
    }

    [Fact]
    public void SparseViterbiMatchesDenseReference()
    {
        LabelSpace labels = LabelSpace.Create(ClassNames);
        var decoder = new ViterbiDecoder(labels, calibrationPath: null);
        int classCount = ClassNames.Length;
        var random = new Random(42);

        for (int testCase = 0; testCase < 64; testCase++)
        {
            int tokenCount = random.Next(1, 33);
            var emissions = new float[tokenCount * classCount];
            for (int index = 0; index < emissions.Length; index++)
            {
                emissions[index] = (random.NextSingle() * 20) - 10;
            }

            Assert.Equal(
                DecodeDenseReference(emissions, tokenCount, classCount),
                decoder.Decode(emissions, tokenCount));
        }

        var tiedEmissions = new float[8 * classCount];
        Assert.Equal(
            DecodeDenseReference(tiedEmissions, 8, classCount),
            decoder.Decode(tiedEmissions, 8));
    }

    [Fact]
    public void SpanDecoderTrimsAndRedacts()
    {
        LabelSpace labels = LabelSpace.Create(ClassNames);
        var tokenized = new TokenizedText(
            [1],
            " Alice ",
            [0],
            [7],
            DecodedMismatch: false);

        List<PrivacyFilterSpan> spans = SpanDecoder.Decode(
            [ClassIndex("S-private_person")],
            labels,
            tokenized,
            trimWhitespace: true,
            discardOverlappingSpans: false,
            PrivacyFilterOutputMode.Redacted);

        PrivacyFilterSpan span = Assert.Single(spans);
        Assert.Equal("redacted", span.Label);
        Assert.Equal(1, span.Start);
        Assert.Equal(6, span.End);
        Assert.Equal("Alice", span.Text);
        Assert.Equal("<REDACTED>", span.Placeholder);
    }

    private static int ClassIndex(string name) => Array.IndexOf(ClassNames, name);

    private static int[] DecodeDenseReference(float[] emissions, int tokenCount, int classCount)
    {
        const float invalid = -1e9f;
        var previousScores = new float[classCount];
        var nextScores = new float[classCount];
        var backpointers = new int[(tokenCount - 1) * classCount];
        for (int label = 0; label < classCount; label++)
        {
            previousScores[label] = emissions[label] +
                (label == 0 || Boundary(label) is 'B' or 'S' ? 0 : invalid);
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
                        (IsValidTransition(previous, next) ? 0 : invalid);
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
            float score = previousScores[label] +
                (label == 0 || Boundary(label) is 'E' or 'S' ? 0 : invalid);
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
