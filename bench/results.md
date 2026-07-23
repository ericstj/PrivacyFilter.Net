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

## Managed postprocessing optimization

Measured 2026-07-22 with 4,096 tokens and 33 classes. The dense and sparse
Viterbi implementations used the same nontrivial score input.

| Workload | Mean | Managed allocation | Relative time |
| --- | ---: | ---: | ---: |
| Log-softmax removed from production | 1.093 ms | 0 B | - |
| Dense Viterbi baseline | 4.527 ms | 544.28 KB | 1.00x |
| Sparse Viterbi with pooled backpointers | 2.065 ms | 16.02 KB | 0.46x |

Sparse valid-predecessor tables reduced Viterbi time by 54% and managed
allocation by 97%. Decoding raw logits also removes the separate log-softmax
pass without changing argmax or Viterbi results.

The end-to-end comparison used 5 warmups and 10 measurement iterations for
each implementation:

| Workload | Before | After | Allocation before | Allocation after |
| --- | ---: | ---: | ---: | ---: |
| Short text | 234.4 ms | 232.6 ms | 16.69 KB | 7.93 KB |
| Long text (50 copies) | 1,048.7 ms | 1,078.4 ms | 644.85 KB | 222.09 KB |

End-to-end latency remained effectively model-bound: the short workload
improved by 0.8%, while the noisier long workload measured 2.8% slower with
overlapping confidence intervals. Managed allocation fell by 52% and 66%,
respectively.

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
