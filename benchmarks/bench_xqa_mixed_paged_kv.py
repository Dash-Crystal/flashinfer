"""Benchmark XQA's shared reduction over A16 and native NVFP4 page runs."""

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
    values = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values)


def _quantize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global_scale = x.abs().amax().float().clamp_min_(1e-12) / (448.0 * 6.0)
    packed, scales = flashinfer.nvfp4_quantize(
        x,
        global_scale.reciprocal(),
        sfLayout=flashinfer.SfLayout.layout_linear,
    )
    return (
        packed,
        scales.reshape(*packed.shape[:-1], scales.shape[-1]),
        global_scale,
    )


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


def _quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = x.abs().amax().float().clamp_min_(1e-12) / 448.0
    packed = (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return packed, scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--q4-pages-per-period", type=int, default=4)
    parser.add_argument("--fp8-pages-per-period", type=int, default=0)
    parser.add_argument("--period-pages", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(17)
    dtype = torch.bfloat16
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
    spec_mask = _causal_mask(args.batch_size, args.q_len)
    key = torch.randn(
        total_pages,
        args.page_size,
        args.kv_heads,
        args.head_dim,
        dtype=dtype,
        device="cuda",
    )
    value = torch.randn_like(key)
    key_q4, key_sf, key_global = _quantize(key)
    value_q4, value_sf, value_global = _quantize(value)
    key_fp8, key_fp8_scale = _quantize_fp8(key)
    value_fp8, value_fp8_scale = _quantize_fp8(value)
    # Scale search affects only page sealing, not the read kernel. Keep this
    # full-cache latency benchmark memory bounded by using direct scales.
    key_bsfp8 = quantize_block_scaled_fp8(key, optimize_scales=False)
    value_bsfp8 = quantize_block_scaled_fp8(value, optimize_scales=False)

    logical = torch.arange(pages_per_request, dtype=torch.int32, device="cuda")
    physical_in_request = (logical * 40503) % pages_per_request
    request_base = (
        torch.arange(args.batch_size, dtype=torch.int32, device="cuda")
        * pages_per_request
    )
    page_table = (request_base[:, None] + physical_in_request[None]).contiguous()
    period_index = logical % args.period_pages
    q4_columns = period_index < args.q4_pages_per_period
    fp8_columns = (period_index >= args.q4_pages_per_period) & (
        period_index < args.q4_pages_per_period + args.fp8_pages_per_period
    )
    if args.q_len > 1:
        # The live tail carries the speculative causal mask and remains A16.
        q4_columns[-1] = False
        fp8_columns[-1] = False
    q4_table = page_table[:, q4_columns].contiguous()
    fp8_table = page_table[:, fp8_columns].contiguous()
    a16_table = page_table[:, ~(q4_columns | fp8_columns)].contiguous()
    seq_lens = torch.full(
        (args.batch_size, 1), args.seq_len, dtype=torch.int32, device="cuda"
    )
    q4_lens = torch.full_like(seq_lens, q4_table.shape[1] * args.page_size)
    fp8_lens = torch.full_like(seq_lens, fp8_table.shape[1] * args.page_size)
    a16_lens = torch.full_like(seq_lens, a16_table.shape[1] * args.page_size)

    page_indices = page_table.flatten()
    page_indptr = (
        torch.arange(args.batch_size + 1, dtype=torch.int32, device="cuda")
        * pages_per_request
    )
    last_page_len = torch.full(
        (args.batch_size,), args.page_size, dtype=torch.int32, device="cuda"
    )
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
    q4_out = torch.empty_like(query)
    fp8_out = torch.empty_like(query)
    bsfp8_out = torch.empty_like(query)
    mixed_out = torch.empty_like(query)

    def xqa_workspace() -> tuple[torch.Tensor, torch.Tensor]:
        storage = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        return storage[: 8 * 1024 * 1024], storage[8 * 1024 * 1024 :]

    q4_semaphore, q4_scratch = xqa_workspace()
    fp8_semaphore, fp8_scratch = xqa_workspace()
    bsfp8_semaphore, bsfp8_scratch = xqa_workspace()
    mixed_semaphore, mixed_scratch = xqa_workspace()

    def run_baseline() -> torch.Tensor:
        return baseline.run(query_flat, (key, value), out=baseline_out)

    def run_q4() -> torch.Tensor:
        flashinfer.xqa(
            query,
            key_q4,
            value_q4,
            page_table,
            seq_lens,
            q4_out,
            q4_scratch,
            q4_semaphore,
            args.kv_heads,
            args.page_size,
            q_scale=key_global / value_global,
            kv_scale=value_global,
            enable_pdl=False,
            q_seq_len=args.q_len,
            mask=spec_mask,
            k_sf_cache=key_sf,
            v_sf_cache=value_sf,
        )
        return q4_out

    def run_fp8() -> torch.Tensor:
        flashinfer.xqa(
            query,
            key_fp8,
            value_fp8,
            page_table,
            seq_lens,
            fp8_out,
            fp8_scratch,
            fp8_semaphore,
            args.kv_heads,
            args.page_size,
            q_scale=key_fp8_scale / value_fp8_scale,
            kv_scale=value_fp8_scale,
            enable_pdl=False,
            q_seq_len=args.q_len,
            mask=spec_mask,
        )
        return fp8_out

    def run_bsfp8() -> torch.Tensor:
        flashinfer.xqa(
            query,
            key_bsfp8.payload,
            value_bsfp8.payload,
            page_table,
            seq_lens,
            bsfp8_out,
            bsfp8_scratch,
            bsfp8_semaphore,
            args.kv_heads,
            args.page_size,
            q_scale=key_bsfp8.global_scale / value_bsfp8.global_scale,
            kv_scale=value_bsfp8.global_scale,
            enable_pdl=False,
            q_seq_len=args.q_len,
            mask=spec_mask,
            k_sf_cache=key_bsfp8.scales,
            v_sf_cache=value_bsfp8.scales,
        )
        return bsfp8_out

    def run_mixed() -> torch.Tensor:
        reduction_splits = 3 if fp8_table.shape[1] else 2
        flashinfer.xqa(
            query,
            key_q4,
            value_q4,
            q4_table,
            q4_lens,
            mixed_out,
            mixed_scratch,
            mixed_semaphore,
            args.kv_heads,
            args.page_size,
            q_scale=key_global / value_global,
            kv_scale=value_global,
            enable_pdl=False,
            q_seq_len=args.q_len,
            mask=spec_mask,
            local_subseq_override=1,
            reduction_subseq_total=reduction_splits,
            reduction_subseq_base=0,
            apply_spec_mask=False,
            k_sf_cache=key_sf,
            v_sf_cache=value_sf,
        )
        if fp8_table.shape[1]:
            flashinfer.xqa(
                query,
                key_fp8,
                value_fp8,
                fp8_table,
                fp8_lens,
                mixed_out,
                mixed_scratch,
                mixed_semaphore,
                args.kv_heads,
                args.page_size,
                q_scale=key_fp8_scale / value_fp8_scale,
                kv_scale=value_fp8_scale,
                enable_pdl=False,
                q_seq_len=args.q_len,
                mask=spec_mask,
                local_subseq_override=1,
                reduction_subseq_total=reduction_splits,
                reduction_subseq_base=1,
                apply_spec_mask=False,
            )
        flashinfer.xqa(
            query,
            key,
            value,
            a16_table,
            a16_lens,
            mixed_out,
            mixed_scratch,
            mixed_semaphore,
            args.kv_heads,
            args.page_size,
            q_scale=1.0,
            kv_scale=1.0,
            enable_pdl=False,
            q_seq_len=args.q_len,
            mask=spec_mask,
            local_subseq_override=1,
            reduction_subseq_total=reduction_splits,
            reduction_subseq_base=reduction_splits - 1,
        )
        return mixed_out

    functions = {
        "flashinfer_a16": run_baseline,
        "xqa_fp8": run_fp8,
        "xqa_bsfp8_16": run_bsfp8,
        "xqa_nvfp4": run_q4,
        "xqa_mixed_page_runs": run_mixed,
    }
    outputs = {name: fn().clone() for name, fn in functions.items()}
    torch.cuda.synchronize()
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
    q4_fraction = float(q4_columns.float().mean())
    fp8_fraction = float(fp8_columns.float().mean())
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "q_len": args.q_len,
                "page_size": args.page_size,
                "q4_fraction": q4_fraction,
                "fp8_fraction": fp8_fraction,
                "effective_bits_per_value": q4_fraction * 4.5
                + fp8_fraction * 8.0
                + (1.0 - q4_fraction - fp8_fraction) * 16.0,
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
