"""Plain runner for the FA3 mixed-page conformance matrix (no test framework).

Usage: python tests/attention/run_fa3_mixed_page_transport.py [--first]
  --first  run a single case (fp4, NHD, q_len 4) so a build error surfaces once.
Exit code is the number of failing cases.
"""

import importlib.util
import pathlib
import sys
import traceback

_spec = importlib.util.spec_from_file_location(
    "_fa3_mixed_test", pathlib.Path(__file__).with_name("test_fa3_mixed_page_transport.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

Q_LENS = [1, 4, 64, 130]
LAYOUTS = ["NHD", "HND"]
PAGE_MODES = ["a16", "fp8", "fp4", "a16_fp8_runs", "a16_fp8", "a16_fp4", "fp8_fp4", "mixed"]


def main() -> int:
    cases = [(4, "NHD", "fp4")] if "--first" in sys.argv else [
        (q, l, m) for m in PAGE_MODES for l in LAYOUTS for q in Q_LENS
    ]
    failures = 0
    for q_len, layout, mode in cases:
        name = f"[{mode}-{layout}-{q_len}]"
        try:
            _mod.test_fa3_mixed_page_transport_matches_a16_expansion(q_len, layout, mode)
            print(f"PASS {name}", flush=True)
        except BaseException as exc:  # noqa: BLE001 - report every case
            if type(exc).__name__.endswith("Skipped"):
                print(f"SKIP {name} {exc}", flush=True)
                continue
            failures += 1
            print(f"FAIL {name}", flush=True)
            traceback.print_exc()
            if "--first" in sys.argv:
                break
    # Cross-work-item protocol (C7): more work items than CTAs, compressed first tile.
    if "--first" not in sys.argv:
        for mode in ("fp8", "mixed"):
            cases.append((1, "NHD", mode))
            failures += _run_many_items(mode)
        # [23] decode corner cases: E4M3 subnormal payload values (and -0), the
        # maximal block scale 448 and subnormal block scales, FP8 and FP4 pages.
        for mode in ("fp8", "mixed"):
            for q_len in (1, 64):
                cases.append((q_len, "NHD", mode))
                failures += _run_extremes(mode, q_len)
    print(f"{len(cases) - failures} passed, {failures} failed")
    return failures


def _extreme_transport(shape, dtype, dev, mode):
    """_make_transport with FP8 payload bytes drawn from all of E4M3 (subnormals,
    +-0, +-448, normals; no NaN) and block scales from {448, 2^-9, 2^-7, 2^-6, 1, 256}
    for both formats; the A16 reference is recomputed for the compressed pages."""
    import torch
    ck, cv, rk, rv, t = _mod._make_transport(shape, dtype, dev, mode)
    num_pages, page_size, num_heads, head_dim = shape
    g = torch.Generator(device=dev)
    g.manual_seed(23)
    vals = torch.tensor(
        [*range(0x00, 0x08)] * 4 + [*range(0x80, 0x88)] * 4 + [0x7E, 0xFE] * 8
        + [*range(0x08, 0x7E)] + [*range(0x88, 0xFE)], dtype=torch.uint8, device=dev)
    scale_vals = torch.tensor([0x7E, 0x7E, 0x01, 0x04, 0x08, 0x38, 0x7C], dtype=torch.uint8,
                              device=dev)

    def pick(table, shp):
        return table[torch.randint(0, table.numel(), shp, device=dev, generator=g)]

    for payload, scales in ((t.fp8_k_payload, t.fp8_k_scales), (t.fp8_v_payload, t.fp8_v_scales)):
        payload.view(torch.uint8).copy_(pick(vals, payload.shape))
        scales.copy_(pick(scale_vals, scales.shape))
    for scales in (t.fp4_k_scales, t.fp4_v_scales):
        scales.copy_(pick(scale_vals, scales.shape))
    scale_shape = (num_pages, page_size, num_heads, head_dim // 16)

    def dec8(payload, scales):
        return (payload.float().reshape(*scale_shape, 16)
                * scales.view(torch.float8_e4m3fn).float().unsqueeze(-1)).reshape(shape)

    def dec4(payload, scales):
        return (_mod._xqa_mod._decode_fp4(payload).reshape(*scale_shape, 16)
                * scales.view(torch.float8_e4m3fn).float().unsqueeze(-1)).reshape(shape)

    fp8_pages = t.page_format == 1
    fp4_pages = t.page_format == 2
    rk[fp8_pages] = dec8(t.fp8_k_payload, t.fp8_k_scales)[fp8_pages].to(dtype)
    rv[fp8_pages] = dec8(t.fp8_v_payload, t.fp8_v_scales)[fp8_pages].to(dtype)
    rk[fp4_pages] = dec4(t.fp4_k_payload, t.fp4_k_scales)[fp4_pages].to(dtype)
    rv[fp4_pages] = dec4(t.fp4_v_payload, t.fp4_v_scales)[fp4_pages].to(dtype)
    return ck, cv, rk, rv, t


def _run_extremes(mode: str, q_len: int) -> int:
    import torch
    from flashinfer.mixed_page_prefill import mixed_page_prefill_jit_args, mixed_page_prefill_run_args
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
    name = f"[extremes-{mode}-{q_len}]"
    try:
        dev, dtype = torch.device("cuda"), torch.bfloat16
        B, H, D, P, pages_per_req = 2, 2, 128, 16, 18
        shape = (B * pages_per_req, P, H, D)
        ck, cv, rk, rv, t = _extreme_transport(shape, dtype, dev, mode)
        kv_len = pages_per_req * P - 3
        qo_indptr = torch.arange(0, (B + 1) * q_len, q_len, dtype=torch.int32, device=dev)
        kv_indptr = torch.arange(0, (B + 1) * pages_per_req, pages_per_req, dtype=torch.int32, device=dev)
        kv_indices = torch.arange(B * pages_per_req, dtype=torch.int32, device=dev)
        last = torch.full((B,), kv_len - (pages_per_req - 1) * P, dtype=torch.int32, device=dev)
        q = torch.randn(B * q_len, H * 4, D, dtype=dtype, device=dev)
        ws = torch.empty(128 << 20, dtype=torch.uint8, device=dev)
        static = {"fp8": 1}.get(mode)
        w_ref = BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend="fa3",
            jit_args=mixed_page_prefill_jit_args(dtype, dtype, dtype, D, static_format=0))
        w = BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend="fa3",
            jit_args=mixed_page_prefill_jit_args(dtype, dtype, dtype, D, static_format=static))
        for x in (w_ref, w):
            x.plan(qo_indptr, kv_indptr, kv_indices, last, H * 4, H, D, P, causal=q_len > 1,
                   q_data_type=dtype, kv_data_type=dtype)
        a16 = t._replace(page_format=torch.zeros_like(t.page_format))
        ref = w_ref.run(q, (rk, rv), *mixed_page_prefill_run_args(a16, D ** -0.5, 0, "NHD"))
        out = w.run(q, (ck, cv), *mixed_page_prefill_run_args(t, D ** -0.5, static, "NHD"))
        torch.cuda.synchronize()
        assert not torch.isnan(ref).any(), "reference has NaN"
        assert torch.equal(out, ref), "not bit-exact"
        print(f"PASS {name}", flush=True)
        return 0
    except BaseException:  # noqa: BLE001
        print(f"FAIL {name}", flush=True)
        traceback.print_exc()
        return 1


def _run_many_items(mode: str) -> int:
    """136 (batch x kv-head) work items on a persistent grid: every CTA runs >1 item."""
    import torch
    from flashinfer.mixed_page_prefill import mixed_page_prefill_jit_args, mixed_page_prefill_run_args
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
    name = f"[many-items-{mode}]"
    try:
        dev, dtype = torch.device("cuda"), torch.bfloat16
        B, H, D, P, pages_per_req = 17, 8, 128, 16, 18
        shape = (B * pages_per_req, P, H, D)
        ck, cv, rk, rv, t = _mod._make_transport(shape, dtype, dev, mode)
        kv_len = pages_per_req * P - 3
        qo_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=dev)
        kv_indptr = torch.arange(0, (B + 1) * pages_per_req, pages_per_req, dtype=torch.int32, device=dev)
        kv_indices = torch.arange(B * pages_per_req, dtype=torch.int32, device=dev)
        last = torch.full((B,), kv_len - (pages_per_req - 1) * P, dtype=torch.int32, device=dev)
        q = torch.randn(B, H * 4, D, dtype=dtype, device=dev)
        ws = torch.empty(128 << 20, dtype=torch.uint8, device=dev)
        static = {"fp8": 1}.get(mode)
        w_ref = BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend="fa3",
            jit_args=mixed_page_prefill_jit_args(dtype, dtype, dtype, D, static_format=0))
        w = BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend="fa3",
            jit_args=mixed_page_prefill_jit_args(dtype, dtype, dtype, D, static_format=static))
        for x in (w_ref, w):
            x.plan(qo_indptr, kv_indptr, kv_indices, last, H * 4, H, D, P, causal=False,
                   q_data_type=dtype, kv_data_type=dtype)
        a16 = t._replace(page_format=torch.zeros_like(t.page_format))
        ref = w_ref.run(q, (rk, rv), *mixed_page_prefill_run_args(a16, D ** -0.5, 0, "NHD"))
        out = w.run(q, (ck, cv), *mixed_page_prefill_run_args(t, D ** -0.5, static, "NHD"))
        torch.cuda.synchronize()
        assert torch.equal(out, ref), "not bit-exact"
        print(f"PASS {name}", flush=True)
        return 0
    except BaseException:  # noqa: BLE001
        print(f"FAIL {name}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
