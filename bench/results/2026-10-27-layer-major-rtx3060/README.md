# Qwen streamed-expert layer-major measurements

Structured run-level results are in [results.json](results.json). They were captured with the repository harness at revision `7b1b6e638c237f622cae5c76f03d2fc7f4d4dd00` on Windows x64, an Intel Core i5-14600K, NVIDIA GeForce RTX 3060 12 GB (driver 610.74), CUDA 13.3.73, and the Swift Qwen3.8-Flash-Next IQ3_XXS pack. Runs used `--prefill auto`, 1,000 expert-cache slots unless shown otherwise, `--spec 2`, greedy sampling, eight generated tokens, the shipped expert profile, and mmap-backed experts.

| Prompt | Layer-major status | Automatic prefill | Layer-major prefill | Automatic / layer-major H2D | Greedy output IDs |
| ---: | --- | ---: | ---: | ---: | --- |
| 4K | Guard fallback | 273.53 tok/s | 272.12 tok/s | 83,264.9 / 83,264.9 MB | Match |
| 8K | Guard fallback | 286.70 tok/s | 317.57 tok/s | 125,804.6 / 125,804.6 MB | Match |
| 16K | Active, one round | 383.43 tok/s | 517.96 tok/s | 200,688.6 / 102,274.4 MB | Match |
| 32K | Active, two rounds | 544.28 tok/s mean | 625.54 tok/s mean | 382,392.6 / 177,566.7 MB | Match in both rounds |

The 32K mean is based on two interleaved rounds with reversed order in round 2: automatic measured 563.53 and 525.02 tok/s; layer-major measured 622.13 and 628.95 tok/s. The 14.9% prompt-throughput improvement and 53.6% lower reported expert H2D volume apply only to this model, cache size, and machine. Automatic cache sizing did not leave enough scratch VRAM at 32K, so the active comparison fixed both arms at 1,000 slots.

Three pre-guard 8K active attempts are also retained in the JSON: cache/target variants each produced a greedy continuation mismatch after the first generated token. The cause is unresolved. These runs are diagnostic evidence, not valid performance comparisons. The current implementation gates prompts below 16,384 tokens to the ordinary path.

Prompt token files were generated deterministically by `bench/layer_major_ab.py`; the JSON includes each prompt SHA-256, exact generated IDs, per-run timings, H2D bytes, and sampled peak memory. The OS file cache was not cleared, so these measurements are not cold-cache or physical NVMe-read results. Decode measurements varied, and no decode performance claim is made.
