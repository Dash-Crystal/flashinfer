"""Benchmark the fused direct-scale BSFP8 K/V page-seal epilogue."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from flashinfer.quantization.kv_cache_fp8 import quantize_block_scaled_fp8_cuda


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(71)
    shape = (args.pages, args.page_size, args.kv_heads, args.head_dim)
    key = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    key_global = key.abs().amax().float() / (448.0 * 448.0)
    value_global = value.abs().amax().float() / (448.0 * 448.0)
    key_payload = torch.empty_like(key, dtype=torch.float8_e4m3fn)
    value_payload = torch.empty_like(value, dtype=torch.float8_e4m3fn)
    scale_shape = (*shape[:-1], shape[-1] // 16)
    key_scales = torch.empty(scale_shape, device="cuda", dtype=torch.uint8)
    value_scales = torch.empty_like(key_scales)

    def seal() -> None:
        quantize_block_scaled_fp8_cuda(
            key,
            key_global,
            payload_out=key_payload,
            scales_out=key_scales,
        )
        quantize_block_scaled_fp8_cuda(
            value,
            value_global,
            payload_out=value_payload,
            scales_out=value_scales,
        )

    for _ in range(args.warmup):
        seal()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.repeat):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        seal()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    elapsed_ms = statistics.median(samples)
    input_bytes = 2 * key.numel() * key.element_size()
    output_bytes = 2 * (key_payload.numel() + key_scales.numel())
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "pages": args.pages,
                "page_size": args.page_size,
                "kv_heads": args.kv_heads,
                "head_dim": args.head_dim,
                "latency_ms": elapsed_ms,
                "us_per_page": elapsed_ms * 1000.0 / args.pages,
                "input_gb_per_s": input_bytes / elapsed_ms / 1e6,
                "input_plus_output_gb_per_s": (input_bytes + output_bytes)
                / elapsed_ms
                / 1e6,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
