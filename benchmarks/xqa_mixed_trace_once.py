"""Launch the XQA mixed-page decode once per mode and let the kernel's
MIXED_KV_TRACE printf reach stdout (one warm-up launch first for JIT).

Same shape as bench_xqa_mixed_page_transport.py; no timing.  Usage:
  python benchmarks/xqa_mixed_trace_once.py --modes fp4 transport_a16 --q-len 1
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench_xqa_mixed_page_transport import make_transport  # noqa: E402

from flashinfer.decode import xqa_batch_decode_with_kv_cache  # noqa: E402


def main() -> None:
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
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--modes", nargs="+", default=["fp4", "transport_a16"])
    parser.add_argument("--launches", type=int, default=1,
                        help="traced launches after the warm-up launch")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    pages_per_request = args.sequence_length // args.page_size
    pages = args.batch_size * pages_per_request
    shape = (pages, args.page_size, args.kv_heads, args.head_dim)
    canonical_k = torch.zeros(shape, dtype=dtype, device=device)
    canonical_v = torch.zeros_like(canonical_k)
    page_table = torch.arange(pages, dtype=torch.int32, device=device).reshape(
        args.batch_size, pages_per_request
    )
    seq_lens = torch.full((args.batch_size,), args.sequence_length, dtype=torch.int32,
                          device=device)
    query = torch.zeros((args.batch_size * args.q_len, args.kv_heads * args.group_size,
                         args.head_dim), dtype=dtype, device=device)
    for mode in args.modes:
        transport = make_transport(shape, mode.removeprefix("transport_"), device)
        static_format = {"transport_a16": 0, "fp8": 1, "fp4": 2}.get(mode)
        workspace = torch.zeros(256 << 20, dtype=torch.uint8, device=device)
        output = torch.empty_like(query)

        def call():
            return xqa_batch_decode_with_kv_cache(
                query, (canonical_k, canonical_v), workspace, out=output,
                page_transport=transport, page_transport_static_format=static_format,
                block_tables=page_table, seq_lens=seq_lens,
                max_seq_len=args.sequence_length, bmm1_scale=args.head_dim ** -0.5,
                q_len_per_req=args.q_len, mask=None,
            )

        call()
        torch.cuda.synchronize()
        sys.stdout.flush()
        for i in range(args.launches):
            print(f"=== MODE {mode} q_len {args.q_len} launch {i} ===", flush=True)
            call()
            torch.cuda.synchronize()
            sys.stdout.flush()
        print(f"=== END {mode} ===", flush=True)


if __name__ == "__main__":
    main()
