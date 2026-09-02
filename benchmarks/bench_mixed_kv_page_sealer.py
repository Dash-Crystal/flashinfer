"""Measure graph-replayed mixed FP4/FP8/A16 page sealing."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from flashinfer.quantization.kv_cache_fp8 import (
    MixedKVPagedCache,
    seal_mixed_kv_pages_cuda,
)


def make_cache(
    pages: int,
    page_size: int,
    heads: int,
    head_dim: int,
    thresholds: list[float],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, MixedKVPagedCache]:
    shape = (pages, page_size, heads, head_dim)
    scale_shape = (pages, page_size, heads, head_dim // 16)
    fp4_shape = (pages, page_size, heads, head_dim // 2)
    k = torch.randn(shape, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    global_scale = torch.ones((), dtype=torch.float32, device=device)
    cache = MixedKVPagedCache(
        torch.empty(shape, dtype=torch.float8_e4m3fn, device=device),
        torch.empty(shape, dtype=torch.float8_e4m3fn, device=device),
        torch.empty(scale_shape, dtype=torch.uint8, device=device),
        torch.empty(scale_shape, dtype=torch.uint8, device=device),
        torch.empty(fp4_shape, dtype=torch.uint8, device=device),
        torch.empty(fp4_shape, dtype=torch.uint8, device=device),
        torch.empty(scale_shape, dtype=torch.uint8, device=device),
        torch.empty(scale_shape, dtype=torch.uint8, device=device),
        torch.zeros(pages, dtype=torch.uint8, device=device),
        torch.empty((pages, 2), dtype=torch.float32, device=device),
        torch.empty((pages, page_size, heads, 4), dtype=torch.float32, device=device),
        torch.tensor(thresholds, dtype=torch.float32, device=device),
        global_scale,
        global_scale,
        global_scale,
        global_scale,
    )
    return k, v, cache


def time_replay(call, repeats: int, trials: int) -> dict[str, float]:
    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repeats)
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-capacity", type=int, default=17)
    parser.add_argument("--completed-counts", type=int, nargs="+", default=[1, 17])
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--trials", type=int, default=7)
    args = parser.parse_args()

    device = torch.device("cuda")
    policies = {
        "a16": [float("-inf"), float("inf")] * 2,
        "fp8": [float("-inf"), float("inf"), float("inf"), float("inf")],
        "fp4": [float("inf")] * 4,
    }
    report = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "shape": vars(args),
        "measurements": [],
    }
    for completed_count in args.completed_counts:
        if completed_count > args.event_capacity:
            raise ValueError("completed count exceeds fixed event capacity")
        for policy, thresholds in policies.items():
            k, v, cache = make_cache(
                args.event_capacity,
                args.page_size,
                args.heads,
                args.head_dim,
                thresholds,
                device,
            )
            reused_pages = torch.arange(
                args.event_capacity, dtype=torch.int32, device=device
            )
            completed_pages = reused_pages.clone()
            reused_count = torch.tensor(
                args.event_capacity, dtype=torch.int32, device=device
            )
            completed_count_tensor = torch.tensor(
                completed_count, dtype=torch.int32, device=device
            )

            def call() -> None:
                seal_mixed_kv_pages_cuda(
                    k,
                    v,
                    reused_pages,
                    reused_count,
                    completed_pages,
                    completed_count_tensor,
                    cache,
                )

            timing = time_replay(call, args.repeats, args.trials)
            item = {
                "completed_pages": completed_count,
                "policy": policy,
                **timing,
            }
            report["measurements"].append(item)
            print(json.dumps(item), flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
