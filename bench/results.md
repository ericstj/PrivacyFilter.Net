# Benchmark results

Measured 2026-07-21 on Windows 11 with an Intel Core i7-13800H (14 physical,
20 logical cores).

## .NET

- .NET 10.0.10
- Microsoft.ML.OnnxRuntime 1.27.1
- Official `model_q4f16.onnx` checkpoint (~809 MB external data)
- BenchmarkDotNet 0.15.8 default job
- Model loading excluded

| Workload | Mean | StdDev | Managed allocation |
| --- | ---: | ---: | ---: |
| Short text | 356.6 ms | 10.82 ms | 16.6 KB |
| Long text (50 copies) | 2,048.4 ms | 22.09 ms | 644.85 KB |

## Same-model Python comparison

For an apples-to-apples model comparison, a Python harness ran the same
`model_q4f16.onnx` graph with ONNX Runtime 1.27.0, Python `tiktoken`, and the
upstream Python Viterbi/span postprocessing.

| Workload | .NET | Python, same q4f16 ONNX | .NET speedup |
| --- | ---: | ---: | ---: |
| Short text | 356.6 ms | 1,913.6 ms | 5.37x |
| Long text (50 copies) | 2,048.4 ms | 3,001.8 ms | 1.47x |

The Python harness used one warmup followed by 10 short-text and 5 long-text
iterations. The ONNX Runtime package versions differ slightly: .NET used 1.27.1
and Python used 1.27.0.

## Default upstream Python comparison

- Python upstream `opf` at commit `f7f00ca7fb869683eb732c010299d901457f19c3`
- PyTorch 2.13.0 CPU, 14 threads
- Official original bf16 checkpoint (~2.8 GB)
- One warmup followed by 10 short-text and 5 long-text iterations
- Model loading excluded

| Workload | Mean | Median | Range | .NET speedup |
| --- | ---: | ---: | ---: | ---: |
| Short text | 663.5 ms | 655.8 ms | 610.2-789.8 ms | 1.86x |
| Long text (50 copies) | 28,497.6 ms | 30,806.4 ms | 23,059.8-31,633.8 ms | 13.91x |

This second comparison represents practical default deployment choices, not an
isolated language-runtime comparison: .NET uses OpenAI's quantized q4f16 ONNX
export while the upstream Python implementation uses the original bf16 PyTorch
checkpoint. Both produced the same person, email, and phone spans for the short
benchmark input, but quantization can change near-tie labels on other inputs.
