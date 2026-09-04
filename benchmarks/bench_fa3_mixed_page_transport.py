"""Mixed-page paged KV attention through the FA3 batch prefill kernel (sm90).

Same shape and transport construction as bench_xqa_mixed_page_transport.py so
the two hosts are directly comparable; reports the FA3 stock A16 kernel, the
mixed kernel with every page A16 (transport_a16), and static E4M3 / E2M1 /
dynamic mixed pages, for one or more query lengths per request.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch

from flashinfer.mixed_page_prefill import (
    mixed_page_prefill_jit_args,
    mixed_page_prefill_run_args,
)
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
from flashinfer.quantization.kv_cache_fp8 import MixedKVPagedCache


def make_transport(shape, mode: str, device: torch.device) -> MixedKVPagedCache:
    pages, page_size, heads, head_dim = shape
    scale_shape = (pages, page_size, heads, head_dim // 16)
    fp4_shape = (pages, page_size, heads, head_dim // 2)
    if mode == "mixed":
        page_format = (torch.arange(pages, device=device) % 3).to(torch.uint8)
    else:
        page_format = torch.full((pages,), {"a16": 0, "fp8": 1, "fp4": 2}[mode],
                                 dtype=torch.uint8, device=device)
    scalar = torch.ones((), dtype=torch.float32, device=device)
    # Zero payloads: finite values, no NaN hazards, same bytes moved as real data.
    return MixedKVPagedCache(
        torch.zeros(shape, dtype=torch.uint8, device=device),
        torch.zeros(shape, dtype=torch.uint8, device=device),
        torch.full(scale_shape, 0x38, dtype=torch.uint8, device=device),
        torch.full(scale_shape, 0x38, dtype=torch.uint8, device=device),
        torch.zeros(fp4_shape, dtype=torch.uint8, device=device),
        torch.zeros(fp4_shape, dtype=torch.uint8, device=device),
        torch.full(scale_shape, 0x38, dtype=torch.uint8, device=device),
        torch.full(scale_shape, 0x38, dtype=torch.uint8, device=device),
        page_format,
        torch.empty((pages, 2), dtype=torch.float32, device=device),
        torch.empty((0, page_size, heads, 4), dtype=torch.float32, device=device),
        torch.empty(4, dtype=torch.float32, device=device),
        scalar, scalar, scalar, scalar,
    )


def time_call(call, repeats: int, trials: int) -> dict[str, float]:
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repeats)
    return {"median_us": statistics.median(samples), "min_us": min(samples), "max_us": max(samples)}


def main() -> None:
    import flashinfer
    from flashinfer.jit import env as _jit_env

    print(f"flashinfer: {flashinfer.__file__}", file=sys.stderr)
    print(f"csrc: {_jit_env.FLASHINFER_CSRC_DIR}", file=sys.stderr)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=17)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--q-lens", type=int, nargs="+", default=[1, 4, 64])
    parser.add_argument("--modes", nargs="+",
                        choices=("stock_a16", "transport_a16", "fp8", "fp4", "mixed"),
                        default=("stock_a16", "transport_a16", "fp8", "fp4", "mixed"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trials", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    page_size = 16
    pages_per_request = args.sequence_length // page_size
    pages = args.batch_size * pages_per_request
    heads_q = args.kv_heads * args.group_size
    shape = (pages, page_size, args.kv_heads, args.head_dim)
    k_cache = torch.zeros(shape, dtype=dtype, device=device)
    v_cache = torch.zeros(shape, dtype=dtype, device=device)
    transports = {m: make_transport(shape, m, device) for m in ("a16", "fp8", "fp4", "mixed")}
    kv_indptr = torch.arange(0, (args.batch_size + 1) * pages_per_request, pages_per_request,
                             dtype=torch.int32, device=device)
    kv_indices = torch.arange(pages, dtype=torch.int32, device=device)
    last_page_len = torch.full((args.batch_size,), page_size, dtype=torch.int32, device=device)
    workspace = torch.empty(256 << 20, dtype=torch.uint8, device=device)
    sm_scale = args.head_dim ** -0.5
    # One mixed module per static format (transport_a16 -> 0, fp8 -> 1, fp4 -> 2,
    # mixed -> dynamic); the URI names the format.
    static_of = {"transport_a16": 0, "fp8": 1, "fp4": 2, "mixed": None}
    jit_args_of = {
        m: mixed_page_prefill_jit_args(dtype, dtype, dtype, args.head_dim, static_format=sf)
        for m, sf in static_of.items() if m in args.modes
    }
    for m, ja in jit_args_of.items():
        print(f"module {m}: {ja[0]} variant {ja[11]}", file=sys.stderr)

    results = []
    for q_len in args.q_lens:
        qo_indptr = torch.arange(0, (args.batch_size + 1) * q_len, q_len, dtype=torch.int32,
                                 device=device)
        q = torch.randn(args.batch_size * q_len, heads_q, args.head_dim, dtype=dtype, device=device)
        causal = q_len > 1

        def plan(w):
            w.plan(qo_indptr, kv_indptr, kv_indices, last_page_len, heads_q, args.kv_heads,
                   args.head_dim, page_size, causal=causal, q_data_type=dtype, kv_data_type=dtype)

        stock = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa3")
        plan(stock)
        mixed = {}
        for m, ja in jit_args_of.items():
            mixed[m] = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa3",
                                                           jit_args=ja)
            plan(mixed[m])
        for mode in args.modes:
            if mode == "stock_a16":
                call = lambda: stock.run(q, (k_cache, v_cache))
            else:
                t = transports["a16" if mode == "transport_a16" else mode]
                run_args = mixed_page_prefill_run_args(t, sm_scale, static_of[mode])
                w = mixed[mode]
                call = lambda: w.run(q, (k_cache, v_cache), *run_args)
            timing = time_call(call, args.repeats, args.trials)
            results.append({"q_len": q_len, "mode": mode, **timing})
            print(f"q_len {q_len:4d} {mode:14s} median {timing['median_us']:9.2f} us",
                  file=sys.stderr)
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
