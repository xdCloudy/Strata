# Qwen streamed-expert prefill: layer-major design review

Status: **scope proposal only**. No engine implementation or performance claim is included in this change. This is intended for maintainer review before a substantial engine modification, as required by CONTRIBUTING.md.

## Problem and existing work

Qwen3.8-Flash-Next prefill with streamed experts can reload the same layer/expert blobs for successive prompt chunks. The existing chunk-major walk trades activation storage for repeated transfers. On small-VRAM cards this can make long prompts transfer-bound.

This is **not a new discovery**:
- [#1670](https://github.com/Niko1221/Strata/issues/1670) includes a working, opt-in layer-major implementation and its [source branch](https://github.com/vizdatom/Strata/tree/perf-ryzen), including two kernels, stager synchronization and prefetch handling. Its author reported 62–99 to 378–443 prompt tok/s on a 6 GB RTX 3060 Laptop, on an older Strata revision. Those figures are the author's measurements, **not reproduced here**. The issue was closed during tracker cleanup; follow-up comments reported the read pattern on later versions.
- [October 9 P4 roofline](../bench/results/2026-10-09-community-p4-prefill-roofline/README.md) measured 262.79 GB transferred for an 8K prompt with fixed 512-token chunks, versus 30.87 GB distinct streamed expert bytes, on a Tesla P4. It explicitly did **not** implement layer-major scheduling or establish a speedup.
- [DeepSeek V4.1](DEEPSEEK41.md) already uses a layer-major prompt path. The Qwen integration should evaluate reuse of existing Strata scheduling abstractions, not introduce a second generic runtime.
- [NInfer-all](https://github.com/iamwavecut/ninfer-all) provides a separate CUDA implementation and [Qwen technical notes](https://github.com/iamwavecut/ninfer-all/blob/master/docs/qwen3-8-flash-next.md) describing grouped prompt-chunk processing and expert reuse. Its reported numbers are external benchmarks, not Strata results.

## Proposed single-feature scope

Add an **opt-in Qwen Flash Next streamed-expert layer-major prefill** mode, off by default. Reuse existing expert cache, stager and CUDA kernels wherever possible. Before any code port, compare the #1670 branch with the current prefill implementation and NInfer-all's grouped execution. Preserve exact causal state progression for GDN/QSA/PLE and token order, while grouping routed expert work for the current layer across multiple sub-chunks.

Proposed implementation constraints:
1. Gate by supported model, backend, single-GPU topology, streamed experts, minimum prompt size, and a bounded activation scratch budget; fall back to existing prefill otherwise.
2. Preserve existing decode cache ownership and restore it after prefill, including on cancellation/errors. No default-path changes.
3. Enforce staging-buffer ownership and CUDA stream/event ordering before any overlapping prefetch. Never reuse pinned staging memory while a transfer reads it.
4. Track actual expert bytes transferred, distinct layer/expert blobs, cache hit/miss counts, prompt throughput and peak scratch memory.
5. Keep Linux-specific O_DIRECT and warm-up experiments from #1670 **out of scope**. No unconditional NInfer runtime import.

## Acceptance checks for a future engine PR

- CPU-only schedule/unit tests for layer boundaries, chunk tails, capacity limits and fallback.
- CUDA build and execution, plus other touched backends built as required by CONTRIBUTING.md.
- Numerical comparison against the existing prefill path across multiple tokens, with attention/recurrent state, long-context continuation, tool-calling and cancellation.
- Same-day interleaved A/B against **current** Strata main and its auto-prefill policy, including 4K/8K/16K+ prompts, cold and warm file cache, peak VRAM, physical NVMe reads, H2D bytes and decode regression.
- Target 6–12 GB NVIDIA cards; report actual device, OS, model quant, context, flags, individual runs and failures. No claim of speedup without those results.

## Licensing

Strata is MIT-licensed; NInfer-all is Apache-2.0. If source is reused, identify exact upstream files and commits, retain applicable copyright and NOTICE information, and satisfy Apache-2.0 attribution/change requirements. This proposal copies **no** NInfer code.

## Decision requested

Given the closed #1670 thread and existing DeepSeek implementation, is a **Qwen-only, opt-in layer-major prefill** implementation still wanted? If yes, should it be based on the published #1670 branch, an adaptation of Strata's DeepSeek scheduler, or a smaller expert-transfer reuse primitive? No performance or correctness validation has been performed for this proposal.
