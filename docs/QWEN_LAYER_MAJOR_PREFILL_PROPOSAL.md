# Streamed-expert layer-major prefill

Status: **opt-in implementation, draft pending correctness and hardware validation**. The feature is disabled by default.

## Behavior

With `STRATA_PREFILL_LAYER_MAJOR=1`, eligible single-GPU prompts process several ordinary prefill chunks through one model layer before moving to the next layer. Routed expert IDs are grouped across those chunks, so each streamed expert blob is staged at most once per layer and super-chunk. Strata's existing chunk path remains the fallback.

The default super-chunk target is 8192 tokens and is rounded to the initialized chunk size. `STRATA_PREFILL_LM_TOKENS` changes the target. `STRATA_PREFILL_LM_R=vram|host` chooses where the super-chunk residual rows live; unset selects VRAM when the target fits the current free-memory budget and pinned host memory otherwise. `STRATA_PREFILL_LM_MARGIN_MIB` reserves 96 MiB by default. `STRATA_LM_PREFETCH=0` disables prefetching the layer's expert blobs while its attention chunks run. `STRATA_PREFILL_LM_RETAIN=1` keeps scratch between successful layer-major runs; fallback runs release retained scratch. `STRATA_PREFILL_LM_SPLIT_ALLOC=1` uses separate allocations for the scratch blocks. Scratch is bounded by the free-VRAM check and the host buffers required by the selected mode.

The scheduler currently requires at least 16,384 prompt tokens. A lower-cache 8K run entered the optimized path but generated a different greedy continuation after its first token, so prompts below 16K stay on the proven chunk-major path until that boundary divergence is understood. This minimum applies even when `STRATA_PREFILL_LM_TOKENS` is set.

The scheduler retains the existing per-chunk attention, GDN, QSA and PLE work in token order. It reports final residual chunks in order after the last layer. Mid-super-chunk callbacks are marked so prompt-cache checkpoints are not saved from the later state at an earlier token boundary. Expert accumulation is deterministic by expert/group order, but may differ by FP32 rounding from the chunk path's router-order accumulation.

The path currently requires the Qwen streamed-expert geometry used by this implementation (2560-wide embeddings, 512 experts, top-10 routing, 640-wide routed and shared expert FFNs, and the supported hyper-connection layout). Other model geometries fall back to the normal path. It also falls back for layer splits, peer/helper GPUs, image rows, DFlash2 taps, concurrent batch yields, forced planned chunks, route observers/debug dumps, non-MMQ expert layers, fused expert layout, or scratch that does not fit. Decode and the default prompt path are unchanged. The mode currently has no support for splitting expert work across GPUs.

## Implementation basis and related work

- Strata issue [#1670](https://github.com/Niko1221/Strata/issues/1670) contains the original work by Tomas Vizdal: layer-major kernels, staging skip/synchronization and the scheduler. The author reported 62–99 to 378–443 prompt tok/s on an RTX 3060 Laptop 6 GB on an older Strata revision; those results have not been reproduced here.
- This port starts from the current-Strata adaptation at [durikim1-cpu/Strata commit `ab57cca4`](https://github.com/durikim1-cpu/Strata/commit/ab57cca4814a55b3ca2ddfc8a4fbb877441a3517), co-authored by Tomas Vizdal and other contributors. The later experimental cache-arena commit `4cbc1533` was not included: it is HIP-only, compact-MMQ-specific and not needed by the bounded independent scratch allocator used here.
- NInfer-all's [Qwen notes](https://github.com/iamwavecut/ninfer-all/blob/master/docs/qwen3-8-flash-next.md) and [executor](https://github.com/iamwavecut/ninfer-all/tree/master/src/models/qwen4_exp) use bounded spans, grouped expert work and overlapped staging. Those concepts agree with the Strata port. **No NInfer source code is copied**; its implementation is Apache-2.0, while Strata remains MIT-licensed.
- The October 9 [P4 roofline report](../bench/results/2026-10-09-community-p4-prefill-roofline/README.md) measured 262.79 GB transferred for an 8K prompt with fixed 512-token chunks versus 30.87 GB of distinct streamed expert bytes. It did not implement this scheduler or establish a speedup; it is not the baseline for a performance claim. Current Strata's auto-prefill policy must be used for A/B measurements.
- Current upstream PR [#4283](https://github.com/Niko1221/Strata/pull/4283) separately skips streamed experts unused by a chunk. That optimization is complementary, but changes the same prefill issuer area and should be reconciled during upstream review.

## Validation

Passed:

- CUDA 13.3 builds from the PR checkout passed for `strata`, `strata_prefill`, and `layer_major_kernel_test` at both `sm_75` and the RTX 3060's native `sm_86`.
- Nine CPU-side CTest checks passed: model family, prefill planning/share rules, memory guard, file expert source, expert profile save/version, expert layout, and batch rows.
- The current `layer_major_kernel_test` passed on the RTX 3060. It checks shared-expert initialization, routed scatter accumulation, and `copy_i32`.
- After adding the minimum-prompt guard, the native `sm_86` build and CUDA kernel test passed again. The updated A/B harness ran exact 4K and 8K prompts with 1,000 expert slots and eight greedy output tokens per arm; both sizes reported the explicit `a short prompt` fallback and matched all output token IDs against automatic prefill.
- A 16,383-token one-round boundary check immediately below the 16,384-token minimum also fell back with exact eight-token parity.
- On the local IQ3_XXS Qwen3.8 pack, the 16K pair passed exact 8-token greedy parity (one round). The 32K pair passed exact parity in both rounds with the same flags: `--prefill auto --expert-cache 1000 --expert-profile data/expert-profile.bin --spec 2 --max-new 8 --greedy --mmap-experts`.
- For that 32K, two-round run, mean prompt throughput was 544.3 tok/s automatic and 625.5 tok/s layer-major (+14.9%). Reported expert H2D bytes averaged 382,393 MB and 177,567 MB respectively (−53.6%). Hardware: GeForce RTX 3060 12 GB, driver 610.74, CUDA 13.3, `sm_86`; IQ3_XXS Qwen3.8-Flash-Next; 1,000 expert slots; two interleaved rounds with reversed arm order. The OS cache was not cleared, so these are not controlled cold-cache or physical-NVMe-read results.
- The reproducible harness is `bench/layer_major_ab.py`. It creates exact 4K/8K/16K/32K prompts, records command lines and raw logs, verifies generated token IDs, captures prefill/decode rates, transfer bytes, peak RSS/system RAM and VRAM, and rejects an inactive layer-major path unless `--allow-fallback` is explicitly selected.
- Structured per-run measurements, exact generated IDs, prompt hashes, and the post-gate 4K/8K plus active 16K/32K summaries are checked in under `bench/results/2026-10-27-layer-major-rtx3060/`. Full logs remain local because they include the full tokenized prompts.

Observed limitation:

- Before the 16K minimum gate was added, the low-cache 8K IQ3_XXS run entered layer-major and produced a different greedy continuation from automatic prefill after the first generated token. This was reproduced with different super-chunk targets and cache sizes. The affected prompt range now falls back to ordinary prefill, confirmed by the post-gate 4K/8K parity run. The underlying divergence is not yet explained, and additional tail/state coverage is still required before expanding the eligible range.
- With automatic cache sizing, the 32K layer-major run did not fit its scratch budget and safely fell back; the measured 32K comparison used 1,000 expert slots on both arms to leave scratch VRAM available.
- Decode samples varied substantially across the four short runs, so no decode speedup or non-regression claim is made. The model logits dump was empty in native-pack mode, and residual/mixed dumps were zero, so neither provided a usable state comparison.
- A 4K/8K performance claim is not made; these sizes use the fallback under the current 16K gate.

Not run:

- HIP build and execution: no `hipcc`/ROCm toolchain is available in this Windows environment.
- Cancellation/recovery, tool-calling, multi-turn, and broader GDN/QSA/PLE state tests; cold-cache physical disk reads; NInfer side-by-side benchmarks; and complete memory-tier sweeps.
- The historical 4–6× report has not been reproduced. The PR remains draft until the 8K divergence is understood and broader end-to-end correctness/backend validation passes.
