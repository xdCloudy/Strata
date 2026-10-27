# Streamed-expert layer-major prefill

Status: **opt-in implementation, draft pending correctness and hardware validation**. The feature is disabled by default.

## Behavior

With `STRATA_PREFILL_LAYER_MAJOR=1`, eligible single-GPU prompts process several ordinary prefill chunks through one model layer before moving to the next layer. Routed expert IDs are grouped across those chunks, so each streamed expert blob is staged at most once per layer and super-chunk. Strata's existing chunk path remains the fallback.

The default super-chunk target is 8192 tokens and is rounded to the initialized chunk size. `STRATA_PREFILL_LM_TOKENS` changes the target. `STRATA_PREFILL_LM_R=vram|host` chooses where the super-chunk residual rows live; unset selects VRAM when the target fits the current free-memory budget and pinned host memory otherwise. `STRATA_PREFILL_LM_MARGIN_MIB` reserves 96 MiB by default. `STRATA_LM_PREFETCH=0` disables prefetching the layer's expert blobs while its attention chunks run. `STRATA_PREFILL_LM_RETAIN=1` keeps scratch between successful layer-major runs; fallback runs release retained scratch. `STRATA_PREFILL_LM_SPLIT_ALLOC=1` uses separate allocations for the scratch blocks. Scratch is bounded by the free-VRAM check and the host buffers required by the selected mode.

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

- `cmake -S . -B ..\build-layermajor-cuda -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_COMPILER=nvcc` — configured with Visual Studio 2022, CUDA 13.3, `sm_75`.
- `cmake --build ..\build-layermajor-cuda --target strata_prefill layer_major_kernel_test --config Release -j 8` — CUDA prefill library and the new test target built.
- `ctest --test-dir ..\build-layermajor-cuda -C Release -R layer_major_kernel_test --output-on-failure` — the shared-expert initialization and routed-group accumulation kernel comparison passed on the NVIDIA GeForce RTX 3060 12 GB.
- `cmake --build ..\build-layermajor-cuda --target strata --config Release -j 8` — the engine and generation integration compiled.

Not run:

- Full prompt parity against the chunk-major path, including multi-token generation, GDN/QSA/PLE state, incomplete tails, cancellation/recovery, decode continuation and tool/multi-turn prompts.
- HIP build and execution. No `hipcc`/ROCm toolchain is available in this Windows environment; HIP validation is required before review under Strata's contribution rules.
- Model throughput, transfer-byte, disk-read, peak-memory or decode-regression measurements. No matched model A/B has been run; the available GPU was already in use by another workload, and the historical 4–6× report is not reproduced evidence.

The available GPU kernel test validates the two new accumulation kernels only. It does not establish end-to-end prompt correctness or performance. Keep this PR in draft until the full model and backend checks above are supplied.
