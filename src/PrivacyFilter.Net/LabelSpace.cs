namespace PrivacyFilterNet;

internal sealed class LabelSpace
{
    private LabelSpace(
        string[] tokenClassNames,
        string[] spanClassNames,
        int[] tokenToSpanLabel,
        char?[] tokenBoundaryTags,
        int backgroundTokenLabel,
        int backgroundSpanLabel)
    {
        TokenClassNames = tokenClassNames;
        SpanClassNames = spanClassNames;
        TokenToSpanLabel = tokenToSpanLabel;
        TokenBoundaryTags = tokenBoundaryTags;
        BackgroundTokenLabel = backgroundTokenLabel;
        BackgroundSpanLabel = backgroundSpanLabel;
    }

    public string[] TokenClassNames { get; }

    public string[] SpanClassNames { get; }

    public int[] TokenToSpanLabel { get; }

    public char?[] TokenBoundaryTags { get; }

    public int BackgroundTokenLabel { get; }

    public int BackgroundSpanLabel { get; }

    public static LabelSpace Create(string[] classNames)
    {
        ArgumentNullException.ThrowIfNull(classNames);

        var spanClassNames = new List<string> { "O" };
        var spanLabelLookup = new Dictionary<string, int>(StringComparer.Ordinal)
        {
            ["O"] = 0,
        };
        var boundaryLabels = new Dictionary<string, HashSet<char>>(StringComparer.Ordinal);
        var tokenToSpanLabel = new int[classNames.Length];
        var tokenBoundaryTags = new char?[classNames.Length];
        int backgroundTokenLabel = -1;

        for (int index = 0; index < classNames.Length; index++)
        {
            string name = classNames[index];
            if (name == "O")
            {
                if (backgroundTokenLabel >= 0)
                {
                    throw new InvalidDataException("The label space contains more than one O label.");
                }

                backgroundTokenLabel = index;
                tokenToSpanLabel[index] = 0;
                continue;
            }

            int separator = name.IndexOf('-');
            if (separator != 1 || name.Length <= 2)
            {
                throw new InvalidDataException($"Token label '{name}' is not a BIOES label.");
            }

            char boundary = name[0];
            if (boundary is not ('B' or 'I' or 'E' or 'S'))
            {
                throw new InvalidDataException($"Token label '{name}' has an unsupported boundary.");
            }

            string baseLabel = name[2..];
            if (!spanLabelLookup.TryGetValue(baseLabel, out int spanLabel))
            {
                spanLabel = spanClassNames.Count;
                spanClassNames.Add(baseLabel);
                spanLabelLookup.Add(baseLabel, spanLabel);
            }

            tokenToSpanLabel[index] = spanLabel;
            tokenBoundaryTags[index] = boundary;
            if (!boundaryLabels.TryGetValue(baseLabel, out HashSet<char>? boundaries))
            {
                boundaries = [];
                boundaryLabels.Add(baseLabel, boundaries);
            }

            if (!boundaries.Add(boundary))
            {
                throw new InvalidDataException($"Token label '{name}' is duplicated.");
            }
        }

        if (backgroundTokenLabel < 0)
        {
            throw new InvalidDataException("The label space must contain O.");
        }

        foreach ((string label, HashSet<char> boundaries) in boundaryLabels)
        {
            if (!boundaries.SetEquals(['B', 'I', 'E', 'S']))
            {
                throw new InvalidDataException($"Span label '{label}' must define B, I, E, and S tokens.");
            }
        }

        return new LabelSpace(
            (string[])classNames.Clone(),
            spanClassNames.ToArray(),
            tokenToSpanLabel,
            tokenBoundaryTags,
            backgroundTokenLabel,
            backgroundSpanLabel: 0);
    }
}
