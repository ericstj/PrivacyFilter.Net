using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using PrivacyFilterNet;

BenchmarkRunner.Run<PrivacyFilterBenchmarks>();

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
