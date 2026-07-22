using System.Globalization;
using System.Text.Json;

namespace PrivacyFilterNet;

internal sealed record ModelConfig(string[] ClassNames)
{
    public static ModelConfig Load(string checkpointDirectory)
    {
        string path = Path.Combine(checkpointDirectory, "config.json");
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("Privacy Filter config.json was not found.", path);
        }

        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(path));
        JsonElement root = document.RootElement;
        if (root.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"{path} must contain a JSON object.");
        }

        if (root.TryGetProperty("encoding", out JsonElement encodingElement))
        {
            string? encoding = encodingElement.GetString();
            if (!string.Equals(encoding, "o200k_base", StringComparison.Ordinal))
            {
                throw new NotSupportedException($"Tokenizer encoding '{encoding}' is not supported.");
            }
        }

        if (!root.TryGetProperty("id2label", out JsonElement labelsElement) ||
            labelsElement.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"{path} must contain an id2label object.");
        }

        var labels = new SortedDictionary<int, string>();
        foreach (JsonProperty property in labelsElement.EnumerateObject())
        {
            if (!int.TryParse(property.Name, NumberStyles.None, CultureInfo.InvariantCulture, out int id) ||
                id < 0 ||
                property.Value.ValueKind != JsonValueKind.String)
            {
                throw new InvalidDataException($"{path} contains an invalid id2label entry.");
            }

            string? label = property.Value.GetString();
            if (string.IsNullOrWhiteSpace(label) || !labels.TryAdd(id, label))
            {
                throw new InvalidDataException($"{path} contains an invalid or duplicate label id.");
            }
        }

        if (labels.Count == 0 || labels.Keys.First() != 0 || labels.Keys.Last() != labels.Count - 1)
        {
            throw new InvalidDataException($"{path} id2label keys must be contiguous and start at zero.");
        }

        return new ModelConfig(labels.Values.ToArray());
    }
}
