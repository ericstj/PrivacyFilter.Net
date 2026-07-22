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
