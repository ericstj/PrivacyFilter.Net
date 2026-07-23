<p align="center">
  <img src="https://raw.githubusercontent.com/ericstj/PrivacyFilter.Net/main/eng/icon.png" alt="PrivacyFilter.Net logo" width="128" height="128">
</p>

# PrivacyFilter.Net

A managed C# port of the inference API and decoding pipeline from
[OpenAI Privacy Filter](https://github.com/openai/privacy-filter). It uses the
official ONNX model, `Microsoft.ML.Tokenizers` for byte-compatible `o200k_base`
tokenization, and managed constrained BIOES Viterbi decoding, span reconstruction,
and redaction.

> **Unofficial.** This project is not affiliated with or endorsed by OpenAI.

## Features

- Loads the official fp32, fp16, q4, q4f16, or quantized ONNX checkpoints.
- Uses `Microsoft.ML.Tokenizers` for `o200k_base` token ids and per-token
  character offsets.
- Reproduces the upstream 33-label BIOES space and constrained Viterbi decoder.
- Supports typed placeholders such as `<PRIVATE_PERSON>` or a single
  `<REDACTED>` output mode.
- Processes long inputs in non-overlapping token windows, matching the upstream
  CPU runtime default of 4,096 tokens.
- Returns the upstream schema fields and JSON names.

## Usage

Download an ONNX variant and its metadata from
[`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter). Keep the
ONNX external-data file beside its `.onnx` graph:

```text
privacy-filter/
  config.json
  viterbi_calibration.json
  onnx/
    model_q4f16.onnx
    model_q4f16.onnx_data
```

For a smaller CPU checkpoint:

```pwsh
hf download openai/privacy-filter `
  --revision 7ffa9a043d54d1be65afb281eddf0ffbe629385b `
  --include config.json viterbi_calibration.json onnx/model_q4f16.onnx onnx/model_q4f16.onnx_data `
  --local-dir C:\models\privacy-filter
```

```csharp
using PrivacyFilterNet;

using var filter = PrivacyFilter.Load(@"C:\models\privacy-filter");
PrivacyFilterResult result =
    filter.Redact("Alice's email is alice@example.com.");

Console.WriteLine(result.RedactedText);
foreach (PrivacyFilterSpan span in result.DetectedSpans)
{
    Console.WriteLine($"{span.Label}: {span.Text} [{span.Start}, {span.End})");
}
```

To collapse all detected types:

```csharp
using var filter = PrivacyFilter.Load(
    @"C:\models\privacy-filter",
    new PrivacyFilterOptions
    {
        OutputMode = PrivacyFilterOutputMode.Redacted,
    });
```

The loader prefers `model_q4f16.onnx`, then the quantized, q4, fp16, and fp32
variants. Set `ModelFileName` to choose one explicitly. ONNX Runtime is a native
runtime dependency; tokenization and all post-model decoding are managed C#.

## Scope

PrivacyFilter.Net ports inference and redaction. Training, fine-tuning, and the
PyTorch/Triton sparse-MoE implementation are out of scope because OpenAI publishes
official ONNX graphs that preserve the model architecture directly.

## Building and testing

```pwsh
dotnet build PrivacyFilter.Net.sln -c Release
dotnet test PrivacyFilter.Net.sln -c Release
```

Tests validate tokenizer ids, decoded text, and offsets against Python `tiktoken`
across deterministic Unicode and special-token corpora; compare Viterbi paths and
BIOES span reconstruction against committed Python oracles; and run the complete
pipeline through a tiny ONNX fixture with the same input/output contract as the
official model.

## Benchmarks

Set `PRIVACY_FILTER_MODEL_DIR` to a downloaded checkpoint, then run:

```pwsh
dotnet run -c Release --project bench\PrivacyFilter.Net.Benchmarks
```

The BenchmarkDotNet suite measures short single-record inference and a longer
multi-record input while reporting managed allocations. Current measurements and
the upstream Python comparison are in [bench/results.md](bench/results.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
