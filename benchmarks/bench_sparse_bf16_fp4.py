# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Correctness and CUDA graph latency for dense/sparse NVFP4 W4A16."""

import argparse
import json
import math
import statistics

import torch
import torch.nn.functional as F

from flashinfer import SfLayout, mm_bf16_fp4, nvfp4_quantize, prepare_bf16_fp4_weights
from flashinfer.autotuner import autotune
from flashinfer.gemm.gemm_bf16_fp4 import _unswizzle_sf_128x4
from flashinfer.gemm.gemm_bf16_fp4_sparse import (
    mm_bf16_fp4_sparse,
    prepare_bf16_fp4_sparse_weights,
)


def measure(fn):
    functions = fn if isinstance(fn, list) else [fn]
    for _ in range(5):
        for operation in functions:
            operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(5):
            for operation in functions:
                operation()
    samples = []
    for _ in range(5):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(20):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 10 / len(functions))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 7, 16, 56])
    parser.add_argument("--m-tiles", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--split-k", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--prepared-a", action="store_true")
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--autotune", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(20260916)
    n, k = args.n, args.k
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
    for chunk in weight.split(256):
        grouped = chunk.reshape(chunk.shape[0], -1, 4, 2 if args.paired else 1)
        energy = grouped.float().square().sum(-1)
        positions = energy.topk(2, dim=-1).indices
        mask = torch.zeros_like(energy, dtype=torch.bool).scatter_(-1, positions, True)
        grouped.mul_(mask.unsqueeze(-1))
    global_scale = 2688 / weight.abs().max().float()
    packed, swizzled = nvfp4_quantize(
        weight,
        global_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cute-dsl",
    )
    alpha = global_scale.reciprocal().reshape(1)
    sparse = prepare_bf16_fp4_sparse_weights(packed, swizzled, paired=args.paired)
    dense = prepare_bf16_fp4_weights(packed, swizzled, alpha, backend="cute-dsl")
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    unscaled = torch.empty(n, k, device="cuda", dtype=torch.float32)
    sf = _unswizzle_sf_128x4(swizzled, n, k // 16).view(torch.float8_e4m3fn)
    for start in range(0, n, 256):
        block = packed[start : start + 256]
        codes = torch.stack((block & 15, block >> 4), -1).reshape(block.shape[0], k)
        unscaled[start : start + block.shape[0]] = lut[codes.long()] * sf[
            start : start + block.shape[0]
        ].float().repeat_interleave(16, -1)
    del codes, positions, mask, packed, sf, swizzled, chunk, grouped, energy, block
    torch.cuda.empty_cache()
    inputs = {
        m: torch.randn(m, k, device="cuda", dtype=torch.bfloat16) for m in args.rows
    }
    references = {
        m: (x.float() @ unscaled.T * alpha).to(torch.bfloat16)
        for m, x in inputs.items()
    }
    del unscaled
    dense_layers = [dense] + [
        tuple(t.clone() if t is not None else None for t in dense)
        for _ in range(args.layers - 1)
    ]
    sparse_layers = [sparse] + [
        tuple(t.clone() for t in sparse) for _ in range(args.layers - 1)
    ]
    weight_layers = [weight] + [weight.clone() for _ in range(args.layers - 1)]
    print(
        json.dumps(
            {
                "kind": "configuration",
                **vars(args),
                "device": torch.cuda.get_device_name(),
                "traffic": "rotating independent weight allocations",
            }
        ),
        flush=True,
    )
    for m, x in inputs.items():
        reference = references[m]
        dense_functions = [
            lambda operands=operands: mm_bf16_fp4(x, *operands, backend="cute-dsl")
            for operands in dense_layers
        ]
        dense_fn = dense_functions[0]
        with autotune():
            dense_fn()
        print(
            json.dumps(
                {
                    "m": m,
                    "n": n,
                    "k": k,
                    "backend": "dense",
                    "us": measure(dense_functions),
                }
            ),
            flush=True,
        )
        print(
            json.dumps(
                {
                    "m": m,
                    "n": n,
                    "k": k,
                    "backend": "bf16",
                    "us": measure(
                        [
                            lambda weight=weight: F.linear(x, weight)
                            for weight in weight_layers
                        ]
                    ),
                }
            ),
            flush=True,
        )
        for mt in [None] if args.autotune else args.m_tiles:
            for sk in [None] if args.autotune else args.split_k:
                if sk is not None and sk > 1 and sk * m * n * 4 > 128 * 1024**2:
                    continue
                functions = [
                    lambda operands=operands: mm_bf16_fp4_sparse(
                        x,
                        *operands,
                        alpha,
                        m_tiles=mt,
                        split_k=sk,
                        prepared_a=None if args.autotune else args.prepared_a,
                        paired=args.paired,
                    )
                    for operands in sparse_layers
                ]
                fn = functions[0]
                with autotune(args.autotune):
                    fn()
                out = fn()
                error = torch.zeros((), device="cuda")
                magnitude = torch.zeros_like(error)
                for actual, expected in zip(
                    out.split(32), reference.split(32), strict=True
                ):
                    error += (actual.float() - expected.float()).square().sum()
                    magnitude += expected.float().square().sum()
                relative_l2 = (error / magnitude).sqrt()
                if not torch.isfinite(out).all() or relative_l2.item() > 0.005:
                    raise AssertionError(
                        f"Sparse result error {relative_l2.item()}: "
                        f"actual={out[0, :8]}, expected={reference[0, :8]}"
                    )
                print(
                    json.dumps(
                        {
                            "m": m,
                            "n": n,
                            "k": k,
                            "backend": "sparse",
                            "m_tiles": mt,
                            "split_k": sk,
                            "us": measure(functions),
                            "relative_l2": relative_l2.item(),
                            "weight_bytes": sum(
                                t.numel() * t.element_size() for t in sparse
                            ),
                            "dense_weight_bytes": sum(
                                t.numel() * t.element_size()
                                for t in dense
                                if t is not None
                            ),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
