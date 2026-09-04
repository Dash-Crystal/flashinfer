"""Streaming HBM bandwidth probe: the achievable ceiling that byte rooflines divide by.

One kernel per CUDA-event pair (no back-to-back bursts) so a co-tenant's
time-slicing, which kicks in for bursts above ~0.5 ms, shows up as a spread
between min and median instead of silently inflating every sample.

Ops: read-only (``x.sum()``), read+write (``y.copy_(x)``), write-only
(``y.fill_``).  Bytes are counted at the DRAM: read = n*itemsize, copy = 2x.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch


def timed_us(fn, trials: int, burst: int = 1) -> list[float]:
    """Per-kernel time in us; ``burst`` back-to-back launches per event pair
    (longer bursts expose a co-tenant's time-slicing)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(burst):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / burst)
    return samples


try:  # optional: a hand-written streaming read kernel (no reduction tree, no second pass)
    import triton
    import triton.language as tl

    @triton.jit
    def _triton_read_kernel(x_ptr, out_ptr, n, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        base = pid.to(tl.int64) * CHUNK
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for i in range(0, CHUNK, BLOCK):
            offs = base + i + tl.arange(0, BLOCK)
            acc += tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
        tl.store(out_ptr + pid, tl.sum(acc, axis=0))

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


def triton_read(x: torch.Tensor, out: torch.Tensor, chunk: int = 1 << 16, num_warps: int = 8) -> None:
    """Read-only stream: each program sums a contiguous ``chunk`` of elements
    (256 KB for fp32 at the default) with 16 elements per lane per step and
    writes one float."""
    grid = (triton.cdiv(x.numel(), chunk),)
    _triton_read_kernel[grid](
        x, out, x.numel(), CHUNK=chunk, BLOCK=512 * num_warps, num_warps=num_warps
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes-mb", type=float, nargs="+", default=[285.212672, 1024, 2048],
        help="tensor sizes in MB (decimal); default includes the bench's A16 KV footprint",
    )
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--burst", type=int, default=1, help="back-to-back launches per event pair")
    parser.add_argument("--ops", nargs="*", default=None, help="subset of ops to run")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--triton", action="store_true", help="add a Triton streaming-read kernel")
    parser.add_argument("--triton-chunk", type=int, default=1 << 16, help="elements per Triton program")
    parser.add_argument("--triton-warps", type=int, default=8)
    args = parser.parse_args()
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(
        json.dumps({"device": props.name, "sm_count": props.multi_processor_count,
                    "total_mem_gb": props.total_memory / 1e9, "torch": torch.__version__}),
        file=sys.stderr,
    )
    rows = []
    for size_mb in args.sizes_mb:
        itemsize = torch.finfo(dtype).bits // 8
        n = int(size_mb * 1e6) // itemsize
        x = torch.ones(n, dtype=dtype, device=device)
        y = torch.empty_like(x)
        nbytes = n * x.element_size()
        # Two read-only probes: the full reduction (two-pass, ~45 us fixed
        # cost at 285 MB on H200) and a row reduction over 8192-element rows
        # (one pass, output n/8192 elements: read-only at the DRAM).
        ops = {
            "read_sum": (lambda: x.sum(), nbytes),
            "copy": (lambda: y.copy_(x), 2 * nbytes),
            "write_fill": (lambda: y.fill_(2.0), nbytes),
        }
        if n % 8192 == 0:
            rows_view = x.view(-1, 8192)
            row_out = torch.empty(rows_view.shape[0], dtype=x.dtype, device=device)
            col_out = torch.empty(8192, dtype=x.dtype, device=device)
            ops["read_rowsum"] = (lambda: torch.sum(rows_view, dim=1, out=row_out), nbytes)
            ops["read_colsum"] = (lambda: torch.sum(rows_view, dim=0, out=col_out), nbytes)
        if args.triton and HAVE_TRITON:
            chunk, warps = args.triton_chunk, args.triton_warps
            tri_out = torch.empty((n + chunk - 1) // chunk, dtype=torch.float32, device=device)
            ops["read_triton"] = (lambda: triton_read(x, tri_out, chunk, warps), nbytes)
        for name, (fn, traffic) in ops.items():
            if args.ops and name not in args.ops:
                continue
            us = timed_us(fn, args.trials, args.burst)
            us_sorted = sorted(us)
            row = {
                "size_mb": nbytes / 1e6,
                "op": name,
                "traffic_bytes": traffic,
                "trials": args.trials,
                "burst": args.burst,
                **({"triton_chunk": args.triton_chunk, "triton_warps": args.triton_warps} if name == "read_triton" else {}),
                "median_us": statistics.median(us),
                "min_us": us_sorted[0],
                "p90_us": us_sorted[int(0.9 * (len(us) - 1))],
                "max_us": us_sorted[-1],
            }
            row["tb_s_median"] = traffic / row["median_us"] / 1e6
            row["tb_s_min_time"] = traffic / row["min_us"] / 1e6
            rows.append(row)
            print(json.dumps(row), flush=True)
        del x, y
        torch.cuda.empty_cache()
    print(json.dumps({"rows": rows}, indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()
