using System.Text;

namespace PrivacyFilterNet;

internal static class SpanDecoder
{
    public static List<PrivacyFilterSpan> Decode(
        int[] labelsByToken,
        LabelSpace labels,
        TokenizedText tokenizedText,
        bool trimWhitespace,
        bool discardOverlappingSpans,
        PrivacyFilterOutputMode outputMode)
    {
        List<TokenSpan> tokenSpans = LabelsToSpans(labelsByToken, labels);
        List<CharacterSpan> characterSpans = TokenSpansToCharacterSpans(
            tokenSpans,
            tokenizedText.CharacterStarts,
            tokenizedText.CharacterEnds);
        if (trimWhitespace)
        {
            characterSpans = TrimWhitespace(characterSpans, tokenizedText.DecodedText);
        }

        if (discardOverlappingSpans)
        {
            characterSpans = DiscardOverlappingSpansByLabel(characterSpans);
        }

        var detected = new List<PrivacyFilterSpan>();
        foreach (CharacterSpan span in characterSpans)
        {
            if (span.Start < 0 ||
                span.End <= span.Start ||
                span.End > tokenizedText.DecodedText.Length)
            {
                continue;
            }

            string label = span.Label >= 0 && span.Label < labels.SpanClassNames.Length
                ? labels.SpanClassNames[span.Label]
                : $"label_{span.Label}";
            detected.Add(new PrivacyFilterSpan(
                label,
                span.Start,
                span.End,
                tokenizedText.DecodedText[span.Start..span.End],
                CreatePlaceholder(label)));
        }

        List<PrivacyFilterSpan> nonOverlapping = SelectNonOverlappingSpans(detected);
        if (outputMode == PrivacyFilterOutputMode.Redacted)
        {
            return nonOverlapping
                .Select(span => span with { Label = "redacted", Placeholder = "<REDACTED>" })
                .ToList();
        }

        return nonOverlapping;
    }

    internal static List<TokenSpan> LabelsToSpans(int[] labelsByToken, LabelSpace labels)
    {
        var spans = new List<TokenSpan>();
        int? currentLabel = null;
        int? startIndex = null;
        int? previousIndex = null;

        for (int tokenIndex = 0; tokenIndex < labelsByToken.Length; tokenIndex++)
        {
            int labelId = labelsByToken[tokenIndex];
            if ((uint)labelId >= (uint)labels.TokenToSpanLabel.Length)
            {
                previousIndex = tokenIndex;
                continue;
            }

            int spanLabel = labels.TokenToSpanLabel[labelId];
            char? boundary = labels.TokenBoundaryTags[labelId];
            if (spanLabel == labels.BackgroundSpanLabel)
            {
                CloseCurrent(spans, currentLabel, startIndex, tokenIndex);
                currentLabel = null;
                startIndex = null;
                previousIndex = tokenIndex;
                continue;
            }

            switch (boundary)
            {
                case 'S':
                    CloseCurrent(spans, currentLabel, startIndex, previousIndex + 1);
                    spans.Add(new TokenSpan(spanLabel, tokenIndex, tokenIndex + 1));
                    currentLabel = null;
                    startIndex = null;
                    break;
                case 'B':
                    CloseCurrent(spans, currentLabel, startIndex, previousIndex + 1);
                    currentLabel = spanLabel;
                    startIndex = tokenIndex;
                    break;
                case 'I':
                    if (currentLabel != spanLabel)
                    {
                        CloseCurrent(spans, currentLabel, startIndex, previousIndex + 1);
                        currentLabel = spanLabel;
                        startIndex = tokenIndex;
                    }
                    break;
                case 'E':
                    if (currentLabel == spanLabel && startIndex is not null)
                    {
                        spans.Add(new TokenSpan(spanLabel, startIndex.Value, tokenIndex + 1));
                    }
                    else
                    {
                        CloseCurrent(spans, currentLabel, startIndex, previousIndex + 1);
                        spans.Add(new TokenSpan(spanLabel, tokenIndex, tokenIndex + 1));
                    }

                    currentLabel = null;
                    startIndex = null;
                    break;
                default:
                    CloseCurrent(spans, currentLabel, startIndex, previousIndex + 1);
                    currentLabel = null;
                    startIndex = null;
                    break;
            }

            previousIndex = tokenIndex;
        }

        CloseCurrent(spans, currentLabel, startIndex, previousIndex + 1);
        return spans;
    }

    private static void CloseCurrent(
        List<TokenSpan> spans,
        int? label,
        int? start,
        int? end)
    {
        if (label is not null && start is not null && end is not null && end > start)
        {
            spans.Add(new TokenSpan(label.Value, start.Value, end.Value));
        }
    }

    private static List<CharacterSpan> TokenSpansToCharacterSpans(
        List<TokenSpan> spans,
        int[] characterStarts,
        int[] characterEnds)
    {
        var result = new List<CharacterSpan>();
        foreach (TokenSpan span in spans)
        {
            if (span.Start < 0 ||
                span.End <= span.Start ||
                span.End > characterStarts.Length)
            {
                continue;
            }

            int start = characterStarts[span.Start];
            int end = characterEnds[span.End - 1];
            if (end > start)
            {
                result.Add(new CharacterSpan(span.Label, start, end));
            }
        }

        return result;
    }

    private static List<CharacterSpan> TrimWhitespace(List<CharacterSpan> spans, string text)
    {
        var result = new List<CharacterSpan>(spans.Count);
        foreach (CharacterSpan span in spans)
        {
            int start = span.Start;
            int end = span.End;
            while (start < end && char.IsWhiteSpace(text[start]))
            {
                start++;
            }

            while (end > start && char.IsWhiteSpace(text[end - 1]))
            {
                end--;
            }

            if (end > start)
            {
                result.Add(span with { Start = start, End = end });
            }
        }

        return result;
    }

    private static List<CharacterSpan> DiscardOverlappingSpansByLabel(List<CharacterSpan> spans)
    {
        var result = new List<CharacterSpan>();
        foreach (IGrouping<int, CharacterSpan> group in spans.GroupBy(span => span.Label))
        {
            var kept = new List<CharacterSpan>();
            foreach (CharacterSpan span in group
                .OrderBy(span => span.Start)
                .ThenByDescending(span => span.End - span.Start))
            {
                if (kept.Any(existing => span.End > existing.Start && span.Start < existing.End))
                {
                    continue;
                }

                kept.Add(span);
            }

            result.AddRange(kept);
        }

        return result
            .OrderBy(span => span.Start)
            .ThenBy(span => span.End)
            .ThenBy(span => span.Label)
            .ToList();
    }

    private static List<PrivacyFilterSpan> SelectNonOverlappingSpans(List<PrivacyFilterSpan> spans)
    {
        var result = new List<PrivacyFilterSpan>();
        int cursor = 0;
        foreach (PrivacyFilterSpan span in spans
            .OrderBy(span => span.Start)
            .ThenByDescending(span => span.End - span.Start)
            .ThenBy(span => span.Label, StringComparer.Ordinal))
        {
            if (span.Start < cursor || span.End <= span.Start)
            {
                continue;
            }

            result.Add(span);
            cursor = span.End;
        }

        return result;
    }

    private static string CreatePlaceholder(string label)
    {
        var builder = new StringBuilder(label.Length + 2);
        builder.Append('<');
        bool previousWasSeparator = false;
        foreach (char character in label)
        {
            if (char.IsAsciiLetterOrDigit(character))
            {
                builder.Append(char.ToUpperInvariant(character));
                previousWasSeparator = false;
            }
            else if (!previousWasSeparator && builder.Length > 1)
            {
                builder.Append('_');
                previousWasSeparator = true;
            }
        }

        if (builder[^1] == '_')
        {
            builder.Length--;
        }

        if (builder.Length == 1)
        {
            builder.Append("REDACTED");
        }

        builder.Append('>');
        return builder.ToString();
    }

    internal sealed record TokenSpan(int Label, int Start, int End);

    private sealed record CharacterSpan(int Label, int Start, int End);
}
