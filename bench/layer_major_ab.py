#!/usr/bin/env python3
"""Matched automatic-prefill vs layer-major runs for a native Qwen expert pack.

The script uses one model, prompt token sequence, sampling policy, and cache mode for both arms. It
alternates run order by round, preserves every engine log, compares generated token IDs, and writes a
machine-readable summary. Run it only when the GPU is otherwise idle. File-cache state is not forcibly
cleared; the interleaved runs therefore measure the host's naturally warmed storage tier.

Example (PowerShell):
  python bench/layer_major_ab.py --exe ..\b\Release\strata.exe `
    --pack D:/Strata-data/packs/iq2_xs `
    --native D:/Strata-data/models/sc117/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS-00001-of-00002.gguf `
    --tokenizer D:/Strata-data/packs/iq2_xs/tokenizer --out ..\layer-major-bench
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # memory sampling is optional; model correctness and engine timing still work
    psutil = None


ROOT = Path(__file__).resolve().parents[1]
NOTES = (
    "A reliable cache keeps keys stable, records ownership, and never evicts a value while an asynchronous reader still uses it.",
    "For each request, measure prompt processing separately from decoding, and record the selected chunk size with the result.",
    "A streamed expert can be staged once for a group of prompt chunks, then reused until its last matrix product finishes.",
    "Attention and recurrent updates remain ordered by token position even when expert computation is grouped across chunks.",
    "A bounded allocation failure must return to the ordinary prompt path without changing the session state.",
    "Completion events protect staging buffers until each transfer and its dependent computation have finished.",
    "The automatic-prefill and layer-major paths receive the same deterministic token IDs and greedy decoding settings.",
)
OUTPUT_RE = re.compile(r"^output\s*:\s*(.*)$", re.MULTILINE)
PREFILL_RE = re.compile(r"^\s*prefill\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", re.MULTILINE)
DECODE_RE = re.compile(r"^\s*decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", re.MULTILINE)
H2D_RE = re.compile(r"experts streamed\s+(\d+)\s+\((\d+) by DMA, ([\d.]+) MB H2D, host ([\d.]+) ms\)")
LM_RE = re.compile(r"layer-major:.*?(\d+) expert blobs streamed \(([\d.]+) MB\).*?; ([\d.]+) ms", re.DOTALL)
LM_OFF_RE = re.compile(r"layer-major off for this prompt: ([^\r\n]+)")


def load_tokenizer(directory: Path):
    sys.path.insert(0, str(ROOT / "tools"))
    import strata_tokenizer as ST

    vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for token, token_id in vocab.items():
        tokens[token_id] = token
    return ST.Tokenizer(
        tokens,
        (directory / "merges.txt").read_text(encoding="utf-8").split("\n"),
        json.loads((directory / "token_type.json").read_text(encoding="utf-8")),
    )


def prompt_ids(tokenizer, count: int) -> list[int]:
    prefix = tokenizer.encode("<|im_start|>user\n", parse_special=True)
    suffix = tokenizer.encode(
        "\nSummarize the engineering notes in one sentence.\n<|im_end|>\n<|im_start|>assistant\n",
        parse_special=True,
    )
    body_text = "\n".join(f"Note {i}: {line}" for i in range(128) for line in NOTES)
    body = tokenizer.encode(body_text)
    need = count - len(prefix) - len(suffix)
    if need <= 0:
        raise ValueError(f"prompt length {count} is too short for the chat wrapper")
    while len(body) < need:
        body.extend(body)
    return prefix + body[:need] + suffix


def run_one(args, prompt_file: Path, prompt_hash: str, n_tokens: int, mode: str, round_no: int) -> dict:
    env = os.environ.copy()
    env["STRATA_PREFILL_LAYER_MAJOR"] = "1" if mode == "layer-major" else "0"
    if mode == "layer-major" and args.layer_major_tokens:
        env["STRATA_PREFILL_LM_TOKENS"] = str(args.layer_major_tokens)
    else:
        env.pop("STRATA_PREFILL_LM_TOKENS", None)
    max_context = max(args.max_context, n_tokens + args.max_new)
    command = [
        str(args.exe), "--pack", str(args.pack), "--native", str(args.native),
        "--tokens-file", str(prompt_file), "--max-context", str(max_context),
        "--prefill", args.prefill, "--expert-cache", args.expert_cache,
        "--spec", "2", "--max-new", str(args.max_new), "--greedy",
    ]
    if args.adapt_every is not None:
        command += ["--adapt-every", str(args.adapt_every)]
    if args.expert_profile:
        command += ["--expert-profile", str(args.expert_profile)]
    if args.mmap_experts:
        command.append("--mmap-experts")
    if args.kv:
        command += ["--kv", args.kv]
    started = time.perf_counter()
    baseline_vram = gpu_memory_mib()
    baseline_ram = psutil.virtual_memory().used if psutil else None
    peak = {"vram_mib": baseline_vram, "rss_bytes": 0, "system_ram_bytes": baseline_ram or 0}
    stop = threading.Event()
    process = subprocess.Popen(command, env=env, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, errors="replace")

    def sample() -> None:
        proc = psutil.Process(process.pid) if psutil else None
        while not stop.is_set():
            if proc:
                try:
                    peak["rss_bytes"] = max(peak["rss_bytes"], proc.memory_info().rss)
                    peak["system_ram_bytes"] = max(peak["system_ram_bytes"], psutil.virtual_memory().used)
                except psutil.Error:
                    pass
            vram = gpu_memory_mib()
            if vram is not None:
                peak["vram_mib"] = max(peak["vram_mib"] or 0, vram)
            stop.wait(1.0)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    output, _ = process.communicate()
    stop.set()
    sampler.join()
    elapsed = time.perf_counter() - started
    log_path = args.out / f"{n_tokens}-{mode}-round{round_no}.log"
    log_path.write_text(output, encoding="utf-8", errors="replace")
    out_match = OUTPUT_RE.search(output)
    output_ids = [int(x) for x in out_match.group(1).split()] if out_match else None
    prefill = PREFILL_RE.search(output)
    decode = DECODE_RE.search(output)
    h2d = H2D_RE.search(output)
    lm = LM_RE.search(output)
    lm_off = LM_OFF_RE.search(output)
    return {
        "mode": mode,
        "round": round_no,
        "prompt_tokens": n_tokens,
        "prompt_sha256": prompt_hash,
        "max_context": max_context,
        "command": command,
        "returncode": process.returncode,
        "wall_seconds": elapsed,
        "output_ids": output_ids,
        "prefill_ms": float(prefill.group(2)) if prefill else None,
        "prefill_tok_s": float(prefill.group(3)) if prefill else None,
        "decode_tokens": int(decode.group(1)) if decode else None,
        "decode_ms": float(decode.group(2)) if decode else None,
        "decode_tok_s": float(decode.group(3)) if decode else None,
        "expert_streamed_blobs": int(h2d.group(1)) if h2d else None,
        "expert_dma_blobs": int(h2d.group(2)) if h2d else None,
        "expert_h2d_mb": float(h2d.group(3)) if h2d else None,
        "expert_host_ms": float(h2d.group(4)) if h2d else None,
        "layer_major_active": bool(lm) if mode == "layer-major" else None,
        "layer_major_fallback_reason": lm_off.group(1) if lm_off else None,
        "layer_major_streamed_experts": int(lm.group(1)) if lm else None,
        "layer_major_streamed_mb": float(lm.group(2)) if lm else None,
        "layer_major_reported_ms": float(lm.group(3)) if lm else None,
        "peak_vram_mib": peak["vram_mib"],
        "peak_vram_delta_mib": (peak["vram_mib"] - baseline_vram)
                               if baseline_vram is not None and peak["vram_mib"] is not None else None,
        "peak_process_rss_bytes": peak["rss_bytes"] or None,
        "peak_system_ram_delta_bytes": (peak["system_ram_bytes"] - baseline_ram)
                                       if baseline_ram is not None else None,
        "log": str(log_path),
    }


def gpu_memory_mib() -> int | None:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10)
        return int(result.stdout.splitlines()[0].strip()) if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", required=True, type=Path)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--native", required=True, type=Path, help="first shard of the model GGUF")
    ap.add_argument("--tokenizer", required=True, type=Path, help="pack tokenizer directory")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--engine-revision", default=None,
                    help="optional source revision/label for the executable (default: current checkout HEAD)")
    ap.add_argument("--sizes", default="4096,8192,16384,32768")
    ap.add_argument("--rounds", type=int, default=2,
                    help="interleaved matched rounds per size; even rounds reverse arm order (default: 2)")
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--max-context", type=int, default=0,
                    help="minimum context allocation; each prompt expands it as needed (default: prompt + generation)")
    ap.add_argument("--prefill", default="auto", help="same Strata --prefill value for both arms (default: auto)")
    ap.add_argument("--expert-cache", default="auto")
    ap.add_argument("--adapt-every", type=int, default=None,
                    help="optional adaptive-tier interval passed identically to both arms; 0 freezes tier changes")
    default_profile = ROOT / "data" / "expert-profile.bin"
    ap.add_argument("--expert-profile", type=Path, default=default_profile if default_profile.is_file() else None,
                    help="residency table required by native-pack speculative verification (default: shipped profile if present)")
    ap.add_argument("--layer-major-tokens", type=int, default=0,
                    help="optional STRATA_PREFILL_LM_TOKENS override; 0 keeps the engine default")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="continue and record a successful ordinary-path fallback when layer-major is ineligible or lacks scratch VRAM")
    ap.add_argument("--kv", choices=("fp16", "int8", "q4_0", "k8v4"))
    ap.add_argument("--mmap-experts", action=argparse.BooleanOptionalAction, default=True,
                    help="read native expert spans from their GGUF shards (default: enabled)")
    args = ap.parse_args()
    if args.rounds < 1 or args.max_new < 1:
        ap.error("--rounds and --max-new must be positive")
    sizes = [int(x) for x in args.sizes.split(",")]
    if not sizes or any(x < 1 for x in sizes):
        ap.error("--sizes must be comma-separated positive token counts")
    args.exe = args.exe.resolve()
    args.pack = args.pack.resolve()
    args.native = args.native.resolve()
    args.tokenizer = args.tokenizer.resolve()
    if args.expert_profile:
        args.expert_profile = args.expert_profile.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.tokenizer)
    results = []
    for n_tokens in sizes:
        ids = prompt_ids(tokenizer, n_tokens)
        prompt_file = args.out / f"prompt-{n_tokens}.tokens"
        prompt_file.write_text("\n".join(map(str, ids)) + "\n", encoding="ascii")
        prompt_hash = hashlib.sha256(prompt_file.read_bytes()).hexdigest()
        for round_no in range(1, args.rounds + 1):
            order = ("automatic", "layer-major") if round_no % 2 else ("layer-major", "automatic")
            cell = []
            for mode in order:
                result = run_one(args, prompt_file, prompt_hash, n_tokens, mode, round_no)
                results.append(result)
                cell.append(result)
                print(f"{n_tokens:>6} {mode:>12}: rc={result['returncode']} "
                      f"prefill={result['prefill_tok_s']} tok/s log={result['log']}", flush=True)
            left = next(x for x in cell if x["mode"] == "automatic")
            right = next(x for x in cell if x["mode"] == "layer-major")
            same = left["output_ids"] is not None and left["output_ids"] == right["output_ids"]
            measurements_present = all(
                x["prefill_tok_s"] is not None and x["decode_tok_s"] is not None and x["expert_h2d_mb"] is not None
                for x in cell
            )
            layer_major_active = right["layer_major_active"]
            for item in cell:
                item["output_matches_other_arm"] = same
                item["measurements_present"] = measurements_present
            fallback_allowed = args.allow_fallback and not layer_major_active and right["layer_major_fallback_reason"] is not None
            if left["returncode"] != 0 or right["returncode"] != 0 or not same or not measurements_present or (not layer_major_active and not fallback_allowed):
                (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
                print(f"FAIL: nonzero exit, missing metrics, inactive layer-major path, or token mismatch at {n_tokens} tokens; see logs and results.json",
                      file=sys.stderr)
                return 1
    report = {
        "commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip(),
        "engine_revision": args.engine_revision or subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip(),
        "hardware": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                     "--format=csv,noheader"], text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL).stdout.strip(),
        "model": str(args.native),
        "pack": str(args.pack),
        "prefill": args.prefill,
        "expert_cache": args.expert_cache,
        "adapt_every": args.adapt_every if args.adapt_every is not None else "engine default",
        "expert_profile": str(args.expert_profile) if args.expert_profile else None,
        "kv": args.kv,
        "mmap_experts": args.mmap_experts,
        "max_new": args.max_new,
        "layer_major_tokens": args.layer_major_tokens or "engine default",
        "allow_fallback": args.allow_fallback,
        "cache_methodology": "interleaved rounds with reversed arm order; OS file cache is not forcibly cleared, so this is not a controlled cold-cache test",
        "results": results,
    }
    result_file = args.out / "results.json"
    result_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    active = sum(1 for x in results if x["mode"] == "layer-major" and x["layer_major_active"])
    fallbacks = [x for x in results if x["mode"] == "layer-major" and not x["layer_major_active"]]
    print(f"PASS: all generated token IDs matched; layer-major active in {active} run(s), "
          f"{len(fallbacks)} fallback(s); raw logs and summary: {result_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
