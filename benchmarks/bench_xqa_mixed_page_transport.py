"""Compare mixed-page transport against the unmodified A16 XQA path."""

from __future__ import annotations

import argparse
import sys
import json
import statistics

import torch

from flashinfer.decode import xqa_batch_decode_with_kv_cache
from flashinfer.quantization.kv_cache_fp8 import MixedKVPagedCache


def pack_causal_mask(batch_size: int, q_len: int, device: torch.device) -> torch.Tensor:
    words = (q_len + 31) // 32
    columns = torch.arange(words * 32, device=device).view(1, -1)
    rows = torch.arange(q_len, device=device).view(-1, 1)
    bits = 1 << torch.arange(32, device=device, dtype=torch.int64)
    packed = (((columns <= rows).view(q_len, words, 32)) * bits).sum(-1)
    return packed.to(torch.uint32).expand(batch_size, -1, -1).contiguous().view(torch.uint16)


def make_transport(
    shape: tuple[int, int, int, int], mode: str, device: torch.device
) -> MixedKVPagedCache:
    pages, page_size, heads, head_dim = shape
    scale_shape = (pages, page_size, heads, head_dim // 16)
    fp4_shape = (pages, page_size, heads, head_dim // 2)
    if mode == "mixed":
        page_format = (torch.arange(pages, device=device) % 3).to(torch.uint8)
    else:
        page_format = torch.full(
            (pages,), {"a16": 0, "fp8": 1, "fp4": 2}[mode],
            dtype=torch.uint8,
            device=device,
        )
    scalar = torch.ones((), dtype=torch.float32, device=device)
    return MixedKVPagedCache(
        torch.empty(shape, dtype=torch.float8_e4m3fn, device=device),
        torch.empty(shape, dtype=torch.float8_e4m3fn, device=device),
        torch.full(scale_shape, 0x38, dtype=torch.uint8, device=device),
        torch.full(scale_shape, 0x38, dtype=torch.uint8, device=device),
        torch.empty(fp4_shape, dtype=torch.uint8, device=device),
        torch.empty(fp4_shape, dtype=torch.uint8, device=device),
        torch.full(scale_shape, 0x38, dtype=torch.uint8, device=device),
        torch.full(scale_shape, 0x38, dtype=torch.uint8, device=device),
        page_format,
        torch.empty((pages, 2), dtype=torch.float32, device=device),
        torch.empty((0, page_size, heads, 4), dtype=torch.float32, device=device),
        torch.empty(4, dtype=torch.float32, device=device),
        scalar,
        scalar,
        scalar,
        scalar,
    )


def time_call(call, repeats: int, trials: int, use_graph: bool) -> dict[str, float]:
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    replay = call
    graph = None
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        replay = graph.replay
        for _ in range(3):
            replay()
        torch.cuda.synchronize()

    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repeats)
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
    }


def kv_transport_bytes(args: argparse.Namespace, mode: str) -> int:
    values_per_page = args.page_size * args.kv_heads * args.head_dim * 2
    pages = args.batch_size * (args.sequence_length // args.page_size)
    bytes_per_page = {
        "a16": values_per_page * 2,
        "fp8": values_per_page + values_per_page // 16,
        "fp4": values_per_page // 2 + values_per_page // 16,
    }
    if mode == "baseline_a16":
        return pages * bytes_per_page["a16"]
    if mode == "transport_a16":
        return pages * bytes_per_page["a16"] + pages
    if mode in ("fp8", "fp4", "native_fp8", "native_block_fp8", "native_fp4"):
        if mode == "native_fp8":
            return pages * values_per_page
        if mode == "native_block_fp8":
            return pages * bytes_per_page["fp8"]
        storage_mode = "fp4" if mode == "native_fp4" else mode
        return pages * bytes_per_page[storage_mode] + (0 if mode == "native_fp4" else pages)
    counts = {name: (pages + 2 - index) // 3 for index, name in enumerate(("a16", "fp8", "fp4"))}
    return sum(counts[name] * bytes_per_page[name] for name in counts) + pages


def main() -> None:
    # A measurement is only attributable if the source tree it exercised is
    # known: print the imported package and the csrc directory the JIT compiles.
    import flashinfer
    from flashinfer.jit import env as _jit_env

    print(f"flashinfer: {flashinfer.__file__}", file=sys.stderr)
    print(f"csrc: {_jit_env.FLASHINFER_CSRC_DIR}", file=sys.stderr)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=17)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--q-lens", type=int, nargs="+", default=[1, 64])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("baseline_a16", "transport_a16", "fp8", "fp4", "native_fp8", "native_block_fp8", "native_fp4", "mixed"),
        default=("baseline_a16", "transport_a16", "fp8", "fp4", "mixed"),
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()

    if args.sequence_length % args.page_size:
        raise ValueError("sequence length must be page aligned")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    pages_per_request = args.sequence_length // args.page_size
    pages = args.batch_size * pages_per_request
    shape = (pages, args.page_size, args.kv_heads, args.head_dim)
    canonical_k = torch.empty(shape, dtype=dtype, device=device)
    canonical_v = torch.empty_like(canonical_k)
    page_table = torch.arange(pages, dtype=torch.int32, device=device).reshape(
        args.batch_size, pages_per_request
    )
    seq_lens = torch.full(
        (args.batch_size,), args.sequence_length, dtype=torch.int32, device=device
    )
    transports = {
        mode: make_transport(shape, mode, device)
        for mode in ("a16", "fp8", "fp4", "mixed")
    }

    report = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "shape": vars(args),
        "measurements": [],
    }
    for q_len in args.q_lens:
        query = torch.empty(
            (args.batch_size * q_len, args.kv_heads * args.group_size, args.head_dim),
            dtype=dtype,
            device=device,
        )
        mask = None if q_len == 1 else pack_causal_mask(args.batch_size, q_len, device)
        common = dict(
            block_tables=page_table,
            seq_lens=seq_lens,
            max_seq_len=args.sequence_length,
            bmm1_scale=args.head_dim**-0.5,
            q_len_per_req=q_len,
            mask=mask,
        )
        for mode in args.modes:
            workspace = torch.zeros(256 << 20, dtype=torch.uint8, device=device)
            output = torch.empty_like(query)
            native_fp8 = mode == "native_fp8"
            native_block_fp8 = mode == "native_block_fp8"
            native_fp4 = mode == "native_fp4"
            transport = (
                None
                if mode in ("baseline_a16", "native_fp8", "native_block_fp8", "native_fp4")
                else transports[mode.removeprefix("transport_")]
            )
            static_format = {
                "transport_a16": 0,
                "fp8": 1,
                "fp4": 2,
            }.get(mode)
            kv_cache = (
                (transports["fp4"].fp4_k_payload, transports["fp4"].fp4_v_payload)
                if native_fp4
                else (transports["fp8"].fp8_k_payload, transports["fp8"].fp8_v_payload)
                if native_fp8 or native_block_fp8
                else (canonical_k, canonical_v)
            )
            kv_cache_sf = (
                (transports["fp4"].fp4_k_scales, transports["fp4"].fp4_v_scales)
                if native_fp4
                else (
                    transports["fp8"].fp8_k_scales.view(torch.float8_e4m3fn),
                    transports["fp8"].fp8_v_scales.view(torch.float8_e4m3fn),
                )
                if native_block_fp8
                else None
            )

            def call(
                transport=transport,
                workspace=workspace,
                output=output,
                kv_cache=kv_cache,
                kv_cache_sf=kv_cache_sf,
                static_format=static_format,
            ):
                return xqa_batch_decode_with_kv_cache(
                    query,
                    kv_cache,
                    workspace,
                    out=output,
                    page_transport=transport,
                    page_transport_static_format=static_format,
                    kv_cache_sf=kv_cache_sf,
                    **common,
                )

            timing = time_call(
                call, args.repeats, args.trials, not args.no_cuda_graph
            )
            transport_bytes = kv_transport_bytes(args, mode)
            report["measurements"].append(
                {
                    "q_len": q_len,
                    "mode": mode,
                    "kv_transport_bytes": transport_bytes,
                    "effective_kv_gbps": transport_bytes / timing["median_us"] / 1e3,
                    **timing,
                }
            )
            print(json.dumps(report["measurements"][-1]), flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
