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
        # [25c] NaN-pattern tail: the same kv_len 149 with page 0 unreferenced by
        # every request and filled with E4M3 NaN codes (payload and scales), and
        # rows 5..15 of each request's partial page filled the same way.  Pages
        # past kv_len are mapped to page 0 by the chunk table and rows past
        # `valid` must copy with src-size 0 (D4): a FULL copy arm misused on
        # V(last) (design 10.1) would land NaN scale bytes -> NaN sf2 -> NaN in O,
        # which the zero-filled buffers of the other cases could not show.
        for mode in ("fp8", "fp4", "mixed"):
            for q_len in (1, 64):
                cases.append((q_len, "NHD", mode))
                failures += _run_parity_tail(mode, q_len, nan_tail=True)
        # [25d] the dynamic module on uniform streams: every tile has 6 FP8 pages
        # (or 6 FP4 pages), i.e. the other format's mask is 0 and one format's
        # loop runs three full two-page steps; with kv_len 285 the last tile has
        # a partial page.  (The `a16_fp8_runs` mode covers all-A16 tiles - no
        # pending operand - next to all-FP8 ones; `a16_fp4` covers 0 FP8 pages
        # with FP4 pages present.)
        for mode in ("fp8", "fp4"):
            for q_len in (1, 64):
                cases.append((q_len, "NHD", mode))
                failures += _run_dynamic_uniform(mode, q_len)
        # [26] the dynamic module's per-slot predicated exact form on whole
        # operands: six pages of one format per tile with the extremes payload /
        # scale set, at a global scale that fails the fold for some blocks and
        # passes it for others (fp8, g = 1: the 448 / 256 blocks exceed 255.5,
        # the {1, 2^-6, 2^-7, 2^-9} blocks fold; fp4, g = 0.5: 448 x 0.5 = 224
        # exceeds 3.99, 1 x 0.5 folds), so the per-operand vote fails in some
        # warps and passes in others while every slot of the tile runs the same
        # format's body (C9 / C16 term for term through the predicated form).
        for mode, gs in (("fp8", 1.0), ("fp4", 0.5)):
            cases.append((1, "NHD", mode))
            failures += _run_dynamic_uniform(mode, 1, extremes_gs=gs)
        # [26] multi-chunk items, stated in tiles (CTA_KV = 96 tokens, a chunk is
        # 16 tiles, the chunk table a 32-row ring).  Every case above has
        # pages_per_req <= 18 = at most 3 tiles per item, i.e. rows 0..2 of
        # buffer 0: buffer 1, the ring wrap, the j == 0 gather, the j == 8 store +
        # group barrier and the next-chunk countdown were never bit-checked.
        #  * 193 pages, kv_len 3085 -> 33 tiles (entries 0..32): chunk 0 = rows
        #    0..15, chunk 1 = rows 16..31 (buffer 1), chunk 2 = entry 32 -> row 0
        #    (the wrap; the pair with V at entry 31 reads its K row at row 0);
        #    two gathers (entries 0, 16), two stores (8, 24), countdown 0 at 32.
        #  * 130 pages, kv_len 2075 -> 22 tiles (entries 0..21): buffer 1 without
        #    a wrap, one gather / store, countdown 0 at entry 16.
        # Run on the F25 kernel first (the tests' own baseline), then per step.
        for pages_per_req, kv_len, tiles in ((193, 193 * 16 - 3, 33), (130, 130 * 16 - 5, 22)):
            for mode in ("fp8", "fp4", "mixed"):
                for q_len in (1, 64):
                    cases.append((q_len, "NHD", mode))
                    failures += _run_multi_chunk(mode, q_len, pages_per_req, kv_len, tiles)
        # [23] decode corner cases: E4M3 subnormal payload values (and -0), the
        # maximal block scale 448 and subnormal block scales, FP8 and FP4 pages.
        # [24a] the same payloads under three FP8 global scales (C9): g = 1 (the
        # 448 / 256 blocks fail the 2^120 fold vote, the others pass, within one
        # page), g = 0.5 (every block folds, products up to 224 x 2^120) and
        # g = 1.1 x 2^-118 (below the 2^-117 lower bound: the sentinel sends every
        # block down the exact path; the reference scale is a bf16 subnormal).
        # [25b] the FP4 global scale takes the same three values (C16): the 2^126
        # fold is exact iff |s g| < 3.9921875, so at g = 1 the 448 / 256 blocks
        # fail the per-operand vote (cold arm, 8 HMUL2 by 2^126 first) while the
        # {1, 2^-6, 2^-7, 2^-9} blocks of other warps take the hot arm; g = 0.5
        # halves the products (448 x 0.5 still fails); g = 1.1 x 2^-118 trips the
        # sentinel for the whole operand.
        for mode in ("fp8", "fp4", "mixed"):
            for q_len in (1, 64):
                for gs in (1.0, 0.5, 1.1 * 2.0 ** -118):
                    cases.append((q_len, "NHD", mode))
                    failures += _run_extremes(mode, q_len, gs)
    print(f"{len(cases) - failures} passed, {failures} failed")
    return failures


def _extreme_transport(shape, dtype, dev, mode, gs=1.0):
    """_make_transport with FP8 payload bytes drawn from all of E4M3 (subnormals,
    +-0, +-448, normals; no NaN) and block scales from {448, 2^-9, 2^-7, 2^-6, 1, 256}
    for both formats; the FP8 and the FP4 global scales are `gs` ([25b]); the A16
    reference is recomputed for the compressed pages as
    A16(x * A16(float(s) * g)) - the kernel's contract (dataflow C9 / C16)."""
    import torch
    ck, cv, rk, rv, t = _mod._make_transport(shape, dtype, dev, mode)
    g8 = torch.full((), gs, dtype=torch.float32, device=dev)
    t = t._replace(fp8_k_global_scale=g8, fp8_v_global_scale=g8,
                   fp4_k_global_scale=g8, fp4_v_global_scale=g8)
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
        # Same contract as dec8: sf = A16(float(s) * g), one rounding of x * sf
        # (E2M1 x 2^ee has <= 2 significant bits: the f32 product is exact).
        sf = (scales.view(torch.float8_e4m3fn).float() * g8).to(dtype).float()
        return (_mod._xqa_mod._decode_fp4(payload).reshape(*scale_shape, 16)
                * sf.unsqueeze(-1)).reshape(shape)

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


def _run_parity_tail(mode: str, q_len: int, extremes: bool = False, nan_tail: bool = False) -> int:
    """kv_len 149 = one full tile + 3 full pages + 5 tokens of page 3: the partial
    tile kv_tile_idx, whose K is issued by the K(last)-alone call and whose V by
    the peeled first pair ([25c]: the two partial-arm call sites).
    ``extremes``: the payload / block-scale set of _extreme_transport (448 scales
    among them, g = 1), so the FP8 fold vote fails inside the partial page.
    ``nan_tail``: one extra physical page 0 that no request references, filled
    with E4M3 NaN codes, and rows 5..15 of every request's last page filled the
    same way (payload, scales and the A16 reference rows), so that any byte copied
    from past kv_len or from page 0 poisons the output."""
    import torch
    from flashinfer.mixed_page_prefill import mixed_page_prefill_jit_args, mixed_page_prefill_run_args
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
    tag = "extremes-" if extremes else ("nan-" if nan_tail else "")
    name = f"[parity-tail-{tag}{mode}-{q_len}]"
    try:
        dev, dtype = torch.device("cuda"), torch.bfloat16
        B, H, D, P, pages_per_req = 3, 2, 128, 16, 10
        extra = 1 if nan_tail else 0
        shape = (B * pages_per_req + extra, P, H, D)
        if extremes:
            ck, cv, rk, rv, t = _extreme_transport(shape, dtype, dev, mode, 1.0)
        else:
            ck, cv, rk, rv, t = _mod._make_transport(shape, dtype, dev, mode)
        kv_len = 96 + 3 * P + 5
        assert kv_len <= pages_per_req * P
        if nan_tail:
            tail_rows = slice(kv_len - (pages_per_req - 1) * P, P)  # rows 5..15 of the last page
            poisoned = [(0, slice(0, P))] + [
                (extra + r * pages_per_req + pages_per_req - 1, tail_rows) for r in range(B)]
            for pg, rows in poisoned:
                for payload, scales in ((t.fp8_k_payload, t.fp8_k_scales),
                                        (t.fp8_v_payload, t.fp8_v_scales)):
                    payload[pg, rows].view(torch.uint8).fill_(0xFF)  # E4M3 NaN code
                    scales[pg, rows].fill_(0x7F)                     # E4M3 NaN scale
                for payload, scales in ((t.fp4_k_payload, t.fp4_k_scales),
                                        (t.fp4_v_payload, t.fp4_v_scales)):
                    payload[pg, rows].fill_(0xFF)
                    scales[pg, rows].fill_(0x7F)
                for ref in (rk, rv, ck, cv):
                    ref[pg, rows].view(torch.int16).fill_(0x7FC0)   # bf16 NaN
        qo_indptr = torch.arange(0, (B + 1) * q_len, q_len, dtype=torch.int32, device=dev)
        kv_indptr = torch.arange(0, (B + 1) * pages_per_req, pages_per_req, dtype=torch.int32, device=dev)
        kv_indices = torch.arange(extra, B * pages_per_req + extra, dtype=torch.int32, device=dev)
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


def _run_dynamic_uniform(mode: str, q_len: int, extremes_gs=None) -> int:
    """A pure fp8 / fp4 transport run through the DYNAMIC module (static_format
    None): 6 pages of one format per tile, the other format's mask 0.
    ``extremes_gs``: use the extremes payload / block-scale set with that global
    scale ([26]: the per-slot predicated exact form on whole operands)."""
    import torch
    from flashinfer.mixed_page_prefill import mixed_page_prefill_jit_args, mixed_page_prefill_run_args
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
    tag = "" if extremes_gs is None else f"-extremes-g{extremes_gs:.3g}"
    name = f"[dynamic-uniform-{mode}-{q_len}{tag}]"
    try:
        dev, dtype = torch.device("cuda"), torch.bfloat16
        B, H, D, P, pages_per_req = 2, 2, 128, 16, 18
        shape = (B * pages_per_req, P, H, D)
        if extremes_gs is None:
            ck, cv, rk, rv, t = _mod._make_transport(shape, dtype, dev, mode)
        else:
            ck, cv, rk, rv, t = _extreme_transport(shape, dtype, dev, mode, extremes_gs)
        kv_len = pages_per_req * P - 3
        qo_indptr = torch.arange(0, (B + 1) * q_len, q_len, dtype=torch.int32, device=dev)
        kv_indptr = torch.arange(0, (B + 1) * pages_per_req, pages_per_req, dtype=torch.int32, device=dev)
        kv_indices = torch.arange(B * pages_per_req, dtype=torch.int32, device=dev)
        last = torch.full((B,), kv_len - (pages_per_req - 1) * P, dtype=torch.int32, device=dev)
        q = torch.randn(B * q_len, H * 4, D, dtype=dtype, device=dev)
        ws = torch.empty(128 << 20, dtype=torch.uint8, device=dev)
        w_ref = BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend="fa3",
            jit_args=mixed_page_prefill_jit_args(dtype, dtype, dtype, D, static_format=0))
        w = BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend="fa3",
            jit_args=mixed_page_prefill_jit_args(dtype, dtype, dtype, D, static_format=None))
        for x in (w_ref, w):
            x.plan(qo_indptr, kv_indptr, kv_indices, last, H * 4, H, D, P, causal=q_len > 1,
                   q_data_type=dtype, kv_data_type=dtype)
        a16 = t._replace(page_format=torch.zeros_like(t.page_format))
        ref = w_ref.run(q, (rk, rv), *mixed_page_prefill_run_args(a16, D ** -0.5, 0, "NHD"))
        out = w.run(q, (ck, cv), *mixed_page_prefill_run_args(t, D ** -0.5, None, "NHD"))
        torch.cuda.synchronize()
        assert not torch.isnan(ref).any(), "reference has NaN"
        assert torch.equal(out, ref), "not bit-exact"
        print(f"PASS {name}", flush=True)
        return 0
    except BaseException:  # noqa: BLE001
        print(f"FAIL {name}", flush=True)
        traceback.print_exc()
        return 1


def _run_multi_chunk(mode: str, q_len: int, pages_per_req: int, kv_len: int, tiles: int) -> int:
    """One item of ``tiles`` KV tiles (CTA_KV = 96): pages_per_req pages, kv_len
    tokens with a partial tail page.  33 tiles cross both chunk-table buffers and
    wrap the 32-row ring (entry 32 -> row 0); 22 tiles use buffer 1 without a
    wrap.  Compressed (static fp8 / fp4 or dynamic mixed) against the a16
    module's expansion of the same stream, bit-exact."""
    import torch
    from flashinfer.mixed_page_prefill import mixed_page_prefill_jit_args, mixed_page_prefill_run_args
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
    name = f"[multi-chunk-{mode}-{q_len}-t{tiles}]"
    try:
        dev, dtype = torch.device("cuda"), torch.bfloat16
        B, H, D, P = 2, 2, 128, 16
        assert (kv_len + 95) // 96 == tiles, "case is stated in tiles"
        assert (pages_per_req - 1) * P < kv_len <= pages_per_req * P, "partial tail page"
        shape = (B * pages_per_req, P, H, D)
        ck, cv, rk, rv, t = _mod._make_transport(shape, dtype, dev, mode)
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
