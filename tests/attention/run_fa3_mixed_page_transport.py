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
        # [24b] page-parity ownership meets the src-size-0 tail on an odd page:
        # kv_len = 96 + 3 x 16 + 5 = 149 -> the last tile has three full pages
        # (0, 1, 2) and a 5-token page 3, owned by the odd-parity warp group.
        for mode in ("fp8", "fp4", "mixed"):
            for q_len in (1, 64):
                cases.append((q_len, "NHD", mode))
                failures += _run_parity_tail(mode, q_len)
        # [24] the FP8 cold path (fold vote fails) on a *partial* page: the
        # extremes transport (block scales incl. 448, g = 1) under kv_len 149, so
        # the odd-parity page 3's five valid rows can carry a 448 scale in the
        # same warp as the src-size-0 zero-filled rows 5..15 (the whole warp then
        # takes the exact two-multiply path with zero payload and zero scale).
        for mode in ("fp8", "mixed"):
            for q_len in (1, 64):
                cases.append((q_len, "NHD", mode))
                failures += _run_parity_tail(mode, q_len, extremes=True)
        # [23] decode corner cases: E4M3 subnormal payload values (and -0), the
        # maximal block scale 448 and subnormal block scales, FP8 and FP4 pages.
        # [24a] the same payloads under three FP8 global scales (C9): g = 1 (the
        # 448 / 256 blocks fail the 2^120 fold vote, the others pass, within one
        # page), g = 0.5 (every block folds, products up to 224 x 2^120) and
        # g = 1.1 x 2^-118 (below the 2^-117 lower bound: the sentinel sends every
        # block down the exact path; the reference scale is a bf16 subnormal).
        for mode in ("fp8", "mixed"):
            for q_len in (1, 64):
                for gs in (1.0, 0.5, 1.1 * 2.0 ** -118):
                    cases.append((q_len, "NHD", mode))
                    failures += _run_extremes(mode, q_len, gs)
    print(f"{len(cases) - failures} passed, {failures} failed")
    return failures


def _extreme_transport(shape, dtype, dev, mode, gs=1.0):
    """_make_transport with FP8 payload bytes drawn from all of E4M3 (subnormals,
    +-0, +-448, normals; no NaN) and block scales from {448, 2^-9, 2^-7, 2^-6, 1, 256}
    for both formats; the FP8 global scale is `gs` (FP4's stays 1); the A16
    reference is recomputed for the compressed pages as
    A16(x * A16(float(s) * g)) - the kernel's contract (dataflow C9)."""
    import torch
    ck, cv, rk, rv, t = _mod._make_transport(shape, dtype, dev, mode)
    g8 = torch.full((), gs, dtype=torch.float32, device=dev)
    t = t._replace(fp8_k_global_scale=g8, fp8_v_global_scale=g8)
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
        # sf = A16(float(s) * g): one rounding; x * sf is exact in f32 (<= 4 + 8
        # significant bits), so .to(dtype) below is the second, single rounding.
        sf = (scales.view(torch.float8_e4m3fn).float() * g8).to(dtype).float()
        return (payload.float().reshape(*scale_shape, 16) * sf.unsqueeze(-1)).reshape(shape)

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


def _run_extremes(mode: str, q_len: int, gs: float = 1.0) -> int:
    import torch
    from flashinfer.mixed_page_prefill import mixed_page_prefill_jit_args, mixed_page_prefill_run_args
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
    name = f"[extremes-{mode}-{q_len}-g{gs:.3g}]"
    try:
        dev, dtype = torch.device("cuda"), torch.bfloat16
        B, H, D, P, pages_per_req = 2, 2, 128, 16, 18
        shape = (B * pages_per_req, P, H, D)
        ck, cv, rk, rv, t = _extreme_transport(shape, dtype, dev, mode, gs)
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


def _run_parity_tail(mode: str, q_len: int, extremes: bool = False) -> int:
    """kv_len 149 = one full tile + 3 full pages + 5 tokens of page 3 (odd parity).
    ``extremes``: the payload / block-scale set of _extreme_transport (448 scales
    among them, g = 1), so the FP8 fold vote fails inside the partial page."""
    import torch
    from flashinfer.mixed_page_prefill import mixed_page_prefill_jit_args, mixed_page_prefill_run_args
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
    name = f"[parity-tail-{'extremes-' if extremes else ''}{mode}-{q_len}]"
    try:
        dev, dtype = torch.device("cuda"), torch.bfloat16
        B, H, D, P, pages_per_req = 3, 2, 128, 16, 10
        shape = (B * pages_per_req, P, H, D)
        if extremes:
            ck, cv, rk, rv, t = _extreme_transport(shape, dtype, dev, mode, 1.0)
        else:
            ck, cv, rk, rv, t = _mod._make_transport(shape, dtype, dev, mode)
        kv_len = 96 + 3 * P + 5
        assert kv_len <= pages_per_req * P
        qo_indptr = torch.arange(0, (B + 1) * q_len, q_len, dtype=torch.int32, device=dev)
        kv_indptr = torch.arange(0, (B + 1) * pages_per_req, pages_per_req, dtype=torch.int32, device=dev)
        kv_indices = torch.arange(B * pages_per_req, dtype=torch.int32, device=dev)
        last = torch.full((B,), kv_len - (pages_per_req - 1) * P, dtype=torch.int32, device=dev)
        q = torch.randn(B * q_len, H * 4, D, dtype=dtype, device=dev)
        ws = torch.empty(128 << 20, dtype=torch.uint8, device=dev)
        static = {"fp8": 1, "fp4": 2}.get(mode)
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
