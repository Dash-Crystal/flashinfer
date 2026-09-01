"""Compare paged A16, per-tensor FP8, and block-scaled FP8 KV transport."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import flashinfer
import torch
from flashinfer.quantization.kv_cache_fp8 import quantize_block_scaled_fp8


def _time_ms(fn: Callable[[], torch.Tensor], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeat):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    return statistics.median(samples)


def _tensor_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = x.abs().amax().float().clamp_min_(1e-12) / 448.0
    return (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), scale


def _causal_mask(batch_size: int, q_len: int) -> torch.Tensor | None:
    if q_len == 1:
        return None
    words = (q_len + 31) // 32
    q_idx = torch.arange(q_len, device="cuda", dtype=torch.int64)[:, None]
    k_idx = torch.arange(words * 32, device="cuda", dtype=torch.int64)[None, :]
    bits = (k_idx <= q_idx).reshape(q_len, words, 32)
    shifts = torch.arange(32, device="cuda", dtype=torch.int64)
    packed = (bits.to(torch.int64) << shifts).sum(-1).to(torch.uint32)
    return packed[None].expand(batch_size, -1, -1).contiguous().view(torch.uint16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument(
        "--a16-baseline",
        choices=("paged", "xqa"),
        default="paged",
        help="Use the generic paged wrapper or the compact A16 XQA kernel.",
    )
    args = parser.parse_args()

    torch.manual_seed(17)
    dtype = getattr(torch, args.dtype)
    pages_per_request = args.seq_len // args.page_size
    total_pages = args.batch_size * pages_per_request
    query_flat = torch.randn(
        args.batch_size * args.q_len,
        args.q_heads,
        args.head_dim,
        dtype=dtype,
        device="cuda",
    )
    query = (
        query_flat.reshape(
            args.batch_size, args.q_len, args.q_heads, args.head_dim
        ).unsqueeze(1)
        if args.q_len > 1
        else query_flat.unsqueeze(1)
    )
    mask = _causal_mask(args.batch_size, args.q_len)
    key = torch.randn(
        total_pages,
        args.page_size,
        args.kv_heads,
        args.head_dim,
        dtype=dtype,
        device="cuda",
    )
    value = torch.randn_like(key)
    key_bs = quantize_block_scaled_fp8(key, optimize_scales=False)
    value_bs = quantize_block_scaled_fp8(value, optimize_scales=False)
    key_fp8, key_fp8_scale = _tensor_fp8(key)
    value_fp8, value_fp8_scale = _tensor_fp8(value)

    logical = torch.arange(pages_per_request, dtype=torch.int32, device="cuda")
    physical = (logical * 40503) % pages_per_request
    bases = (
        torch.arange(args.batch_size, dtype=torch.int32, device="cuda")
        * pages_per_request
    )
    page_table = (bases[:, None] + physical[None]).contiguous()
    page_indices = page_table.flatten()
    page_indptr = (
        torch.arange(args.batch_size + 1, dtype=torch.int32, device="cuda")
        * pages_per_request
    )
    last_page_len = torch.full(
        (args.batch_size,), args.page_size, dtype=torch.int32, device="cuda"
    )
    seq_lens = torch.full(
        (args.batch_size, 1), args.seq_len, dtype=torch.int32, device="cuda"
    )

    baseline = None
    if args.a16_baseline == "paged":
        baseline_workspace = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )
        if args.q_len == 1:
            baseline = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                baseline_workspace, kv_layout="NHD", use_tensor_cores=False
            )
            baseline.plan(
                page_indptr,
                page_indices,
                last_page_len,
                args.q_heads,
                args.kv_heads,
                args.head_dim,
                args.page_size,
                q_data_type=dtype,
                kv_data_type=dtype,
                sm_scale=args.head_dim**-0.5,
            )
        else:
            query_indptr = (
                torch.arange(args.batch_size + 1, dtype=torch.int32, device="cuda")
                * args.q_len
            )
            baseline = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                baseline_workspace, kv_layout="NHD"
            )
            baseline.plan(
                query_indptr,
                page_indptr,
                page_indices,
                last_page_len,
                args.q_heads,
                args.kv_heads,
                args.head_dim,
                args.page_size,
                causal=True,
                q_data_type=dtype,
                kv_data_type=dtype,
                sm_scale=args.head_dim**-0.5,
            )

    baseline_out = torch.empty_like(query_flat)
    fp8_out = torch.empty_like(query)
    bs_out = torch.empty_like(query)

    def workspace() -> tuple[torch.Tensor, torch.Tensor]:
        storage = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        return storage[: 8 * 1024 * 1024], storage[8 * 1024 * 1024 :]

    fp8_sem, fp8_scratch = workspace()
    bs_sem, bs_scratch = workspace()
    a16_sem, a16_scratch = workspace()
    bridge_sem, bridge_scratch = workspace()
    bridge_query = torch.empty_like(query, dtype=torch.float16)
    bridge_out_fp16 = torch.empty_like(query, dtype=torch.float16)
    bridge_out = torch.empty_like(query)

    def run_a16() -> torch.Tensor:
        if baseline is not None:
            return baseline.run(query_flat, (key, value), out=baseline_out)
        flashinfer.xqa(
            query,
            key,
            value,
            page_table,
            seq_lens,
            baseline_out.reshape_as(query),
            a16_scratch,
            a16_sem,
            args.kv_heads,
            args.page_size,
            enable_pdl=False,
            q_seq_len=args.q_len,
            mask=mask,
        )
        return baseline_out

    def run_fp8() -> torch.Tensor:
        flashinfer.xqa(
            query,
            key_fp8,
            value_fp8,
            page_table,
            seq_lens,
            fp8_out,
            fp8_scratch,
            fp8_sem,
            args.kv_heads,
            args.page_size,
            q_scale=key_fp8_scale / value_fp8_scale,
            kv_scale=value_fp8_scale,
            enable_pdl=False,
            q_seq_len=args.q_len,
            mask=mask,
        )
        return fp8_out

    def run_bsfp8() -> torch.Tensor:
        flashinfer.xqa(
            query,
            key_bs.payload,
            value_bs.payload,
            page_table,
            seq_lens,
            bs_out,
            bs_scratch,
            bs_sem,
            args.kv_heads,
            args.page_size,
            q_scale=key_bs.global_scale / value_bs.global_scale,
            kv_scale=value_bs.global_scale,
            enable_pdl=False,
            q_seq_len=args.q_len,
            mask=mask,
            k_sf_cache=key_bs.scales,
            v_sf_cache=value_bs.scales,
        )
        return bs_out

    def run_bsfp8_fp16_bridge() -> torch.Tensor:
        bridge_query.copy_(query)
        flashinfer.xqa(
            bridge_query,
            key_bs.payload,
            value_bs.payload,
            page_table,
            seq_lens,
            bridge_out_fp16,
            bridge_scratch,
            bridge_sem,
            args.kv_heads,
            args.page_size,
            q_scale=key_bs.global_scale / value_bs.global_scale,
            kv_scale=value_bs.global_scale,
            enable_pdl=False,
            q_seq_len=args.q_len,
            mask=mask,
            k_sf_cache=key_bs.scales,
            v_sf_cache=value_bs.scales,
        )
        bridge_out.copy_(bridge_out_fp16)
        return bridge_out

    functions = {
        "flashinfer_a16": run_a16,
        "xqa_tensor_fp8": run_fp8,
        "xqa_bsfp8_16": run_bsfp8,
    }
    if dtype == torch.bfloat16:
        functions["xqa_bsfp8_16_fp16_bridge"] = run_bsfp8_fp16_bridge
    outputs = {name: fn().clone() for name, fn in functions.items()}
    timings = {
        name: _time_ms(fn, args.warmup, args.repeat) for name, fn in functions.items()
    }
    reference = (
        outputs["flashinfer_a16"]
        .reshape(args.batch_size, args.q_len, args.q_heads, args.head_dim)
        .unsqueeze(1)
        .float()
        if args.q_len > 1
        else outputs["flashinfer_a16"].unsqueeze(1).float()
    )
    errors = {}
    for name, output in outputs.items():
        if name == "flashinfer_a16":
            continue
        delta = output.float() - reference
        errors[name] = float(
            delta.square().mean().sqrt() / reference.square().mean().sqrt()
        )
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "q_len": args.q_len,
                "dtype": args.dtype,
                "page_size": args.page_size,
                "effective_bits_per_value": {
                    "xqa_tensor_fp8": 8.0,
                    "xqa_bsfp8_16": 8.5,
                },
                "latency_ms": timings,
                "speedup_vs_flashinfer_a16": {
                    name: timings["flashinfer_a16"] / elapsed
                    for name, elapsed in timings.items()
                },
                "relative_rms_vs_flashinfer_a16": errors,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
