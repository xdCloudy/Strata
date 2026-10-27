# Qwen streamed-expert layer-major measurements

Structured run-level results are in [results.json](results.json). They were captured with the repository harness on Windows x64, an Intel Core i5-14600K, NVIDIA GeForce RTX 3060 12 GB (driver 610.74), CUDA 13.3.73, and the Swift Qwen3.8-Flash-Next IQ3_XXS pack. Runs used `--prefill auto`, 1,000 expert-cache slots unless shown otherwise, `--spec 2`, greedy sampling, eight generated tokens, the shipped expert profile, and mmap-backed experts. Because the investigation spans the pre-guard and post-guard revisions, each case records its tested source revision in the JSON.

| Prompt | Layer-major status | Automatic prefill | Layer-major prefill | Automatic / layer-major H2D | Greedy output IDs |
| ---: | --- | ---: | ---: | ---: | --- |
| 4K | Guard fallback | 273.53 tok/s | 272.12 tok/s | 83,264.9 / 83,264.9 MB | Match |
| 8K | Guard fallback | 286.70 tok/s | 317.57 tok/s | 125,804.6 / 125,804.6 MB | Match |
| 16K | Active, two rounds (fresh current-revision retest) | 429.08 tok/s mean | 553.00 tok/s mean | 200,688.6 / 102,274.4 MB | Match in both rounds |
| 32K | Active, two rounds | 544.28 tok/s mean | 625.54 tok/s mean | 382,392.6 / 177,566.7 MB | Match in both rounds |

The 32K mean is based on two interleaved rounds with reversed order in round 2: automatic measured 563.53 and 525.02 tok/s; layer-major measured 622.13 and 628.95 tok/s. The 14.9% prompt-throughput improvement and 53.6% lower reported expert H2D volume apply only to this model, cache size, and machine. Automatic cache sizing did not leave enough scratch VRAM at 32K, so the active comparison fixed both arms at 1,000 slots.

The exact 16,384-token minimum was retested after fixing the off-by-one in the eligibility guard. Both reversed-order rounds entered layer-major and matched all eight generated IDs. Automatic measured 409.51 and 411.00 tok/s; layer-major measured 560.21 and 529.79 tok/s. These two samples document threshold behavior and should not be treated as a general performance estimate.

A fresh two-round retest after rebuilding revision `9b17dc454fd53e4df12259272ab4872df95b69a5` also entered layer-major in both rounds and matched all eight generated token IDs. Automatic measured 395.46 and 462.69 tok/s; layer-major measured 553.62 and 552.37 tok/s. The layer-major arm reported 102,274.4 MB H2D in each round versus 200,688.6 MB for automatic. The machine-readable rows are retained separately under `16384_current_revision_9b17dc45_retest`; this is a second two-round sample, not a replacement for the earlier boundary run.

A separate 8K current-revision check with `--adapt-every 0` did not establish output parity: both arms completed, and the layer-major arm reported the expected `a short prompt` fallback, but the two output sequences differed. Two later automatic-only controls and two later feature-flag fallback controls on the same prompt all produced the same alternate eight-token sequence. Treat the A/B sample as inconclusive; it neither establishes a guard regression nor supports a parity claim. The pair and control summary are retained under `8192_current_revision_fallback_inconclusive`.

A second 16K active-path retest used `docs/DETAILS.md` as a distinct prompt body (SHA-256 `fd904fa30298992cd2c50628310c23d8e48e50a6ff629f3bcb0bcd181d1e14d7`). Both reversed-order rounds matched all eight generated IDs. Automatic measured 384.69 and 438.88 tok/s; layer-major measured 413.90 and 423.66 tok/s. This verifies parity on a second prompt body; it is a small sample and does not establish a general speedup. Rows are under `16384_details_corpus_current_revision_retest`.

Three pre-guard 8K active attempts are also retained in the JSON: cache/target variants each produced a greedy continuation mismatch after the first generated token. The cause is unresolved. These runs are diagnostic evidence, not valid performance comparisons. The current implementation gates prompts below 16,384 tokens to the ordinary path.

An isolated pre-guard diagnostic captured every verify window's FP32 logits while running the ordinary greedy sequence. Both arms chose token `248068` first. At position 8,192, automatic prefill ranked `198` over `271` by only 0.151 logit; layer-major ranked `271` over `198` by 0.961, matching the observed second-token divergence. The final residual hashes already differed at the first verify window (position 8,191), although its argmax remained `248068`. This locates the greedy flip after a prompt-state difference; it does not prove whether the underlying cause is accumulation order, recurrent-state progression, or another numerical effect. The temporary per-window instrumentation is described in the JSON and is not part of the submitted branch.

Prompt token files were generated deterministically by `bench/layer_major_ab.py`; the JSON includes each prompt SHA-256, exact generated IDs, per-run timings, H2D bytes, and sampled peak memory. The OS file cache was not cleared, so these measurements are not cold-cache or physical NVMe-read results. Decode measurements varied, and no decode performance claim is made.

A follow-up one-round check at 16,383 tokens, immediately below the minimum, also reported the explicit short-prompt fallback and exact eight-token parity. Its run-level data is included in `results.json`.

## 500-slot cache limitation

The 16,384-token eligibility threshold does **not** guarantee output parity at smaller expert-cache capacities. Four current-binary checks at 16K with 500 cache slots, using the `docs/DETAILS.md` prompt body and the same exact token IDs, all entered layer-major and returned a different greedy continuation after the shared first token. Automatic began `[248068, 198, ...]`; layer-major began `[248068, 271, ...]`.

| Layer-major setting | Automatic prefill | Layer-major prefill | Automatic / layer-major H2D | Greedy IDs match |
| --- | ---: | ---: | ---: | --- |
| Host residuals, adaptive tiering | 307.30 tok/s | 373.54 tok/s | 543,615.2 / 109,090.9 MB | No |
| VRAM residuals, adaptive tiering | 314.15 tok/s | 388.14 tok/s | 543,615.2 / 109,090.9 MB | No |
| Host residuals, `--adapt-every 0` | 313.00 tok/s | 375.29 tok/s | 543,615.2 / 109,090.9 MB | No |
| Host residuals, 4,096-token super-chunks | 310.53 tok/s | 352.16 tok/s | 543,615.2 / 177,330.9 MB | No |

Each check generated eight tokens. The result persisted across residual storage tier, adaptive-tier freeze, and smaller super-chunks. These timings and H2D counts are diagnostic only and do not constitute successful performance comparisons. The reason for the numerical divergence remains unresolved. Until it is diagnosed and fixed, do not treat 500-slot cache configurations as parity-validated, even when the active path is available. Full compact per-run records are in `results.json` under `low_cache_16k_parity_checks`.

One-round checks at 625 and 750 slots matched automatic and layer-major exactly; both produced `[248068, 271, 248069, 271, 2495, 450, 369, 264]`, consistent with the existing two-round 1,000-slot result. Automatic at 500 slots instead produced `[248068, 198, 760, 1156, 6587, 728, 310, 59987]`. This prompt therefore changes continuation across cache-capacity execution paths. The 625/750 checks are single samples and do not establish a supported minimum; their full rows are in `results.json` under `cache_capacity_controls`.
