import pytest
import torch

from flashinfer.decode import xqa_batch_decode_with_kv_cache
from flashinfer.quantization.kv_cache_fp8 import MixedKVPagedCache
from flashinfer.utils import get_compute_capability


def _pack_query_mask(batch_size: int, q_len: int, device: torch.device) -> torch.Tensor:
    words = (q_len + 31) // 32
    padded = words * 32
    row = torch.arange(q_len, device=device).view(-1, 1)
    column = torch.arange(padded, device=device).view(1, -1)
    causal = (column <= row).view(q_len, words, 32)
    bits = 1 << torch.arange(32, device=device, dtype=torch.int64)
    packed = (causal.to(torch.int64) * bits).sum(-1).to(torch.uint32)
    return packed.expand(batch_size, -1, -1).contiguous().view(torch.uint16)


def _decode_fp4(payload: torch.Tensor) -> torch.Tensor:
    low = payload & 0xF
    high = payload >> 4
    nibbles = torch.stack((low, high), dim=-1).flatten(-2)
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=payload.device,
    )
    values = magnitudes[(nibbles & 7).long()]
    return torch.where((nibbles & 8).bool(), -values, values)


def _make_transport(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    device: torch.device,
    page_mode: str = "mixed",
    regime: str = "normal",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MixedKVPagedCache]:
    """``regime`` selects the FP8 payload / block-scale value ranges:

    * ``normal``: payload ~ N(0, 2), block scales in {0.5, 1, 2}.
    * ``subnormal``: every finite E4M3 payload code appears (subnormals 0x01..0x07,
      zeros, the maximum 448) with block scales {0.5, 1, 2, 128 (the sealer's FP8
      maximum)}.
    * ``maxscale``: the same payloads with block scales {448 (E4M3 maximum), 2^-9
      (E4M3 minimum subnormal), 1}; 448 x global 1 lies outside the sm90 converter's
      folded-scale range, so its exact fallback multiply is exercised.
    * ``tinyglobal``: the same payloads with block scales {2, 4, 128, 448} and a
      global scale of 2^-118 (< 2^-117, the sm90 fold's ``foldOk`` bound): every
      block scale x global is fp32-normal (>= 2^-117) but the fold is refused for
      the whole stream, so the two-multiply fallback runs on every span.

    The host reference models the kernels' two roundings: ``bf16(x * bf16(s * g))``
    (the block scale times the global scale rounded to the activation type first,
    then one rounding of the product; with ``g == 1`` this is ``bf16(x * s)``, exact
    in the product since E4M3 / E2M1 magnitudes have <= 4 significant bits).
    """
    num_pages, page_size, num_heads, head_dim = shape
    canonical_k = torch.randn(shape, dtype=dtype, device=device)
    canonical_v = torch.randn(shape, dtype=dtype, device=device)
    reference_k = canonical_k.clone()
    reference_v = canonical_v.clone()

    scale_shape = (num_pages, page_size, num_heads, head_dim // 16)
    global_scale = 1.0
    if regime == "normal":
        fp8_k = (torch.randn(shape, device=device) * 2).to(torch.float8_e4m3fn)
        fp8_v = (torch.randn(shape, device=device) * 2).to(torch.float8_e4m3fn)
        scale_values = torch.tensor([0.5, 1.0, 2.0], device=device)
    else:
        def all_codes() -> torch.Tensor:
            # Uniform over the 254 finite E4M3 codes (NaN 0x7F / 0xFF remapped).
            codes = torch.randint(0, 256, shape, dtype=torch.int32, device=device)
            codes = torch.where((codes & 0x7F) == 0x7F, codes ^ 0x01, codes)
            return codes.to(torch.uint8).view(torch.float8_e4m3fn)

        fp8_k = all_codes()
        fp8_v = all_codes()
        if regime == "subnormal":
            scale_values = torch.tensor([0.5, 1.0, 2.0, 128.0], device=device)
        elif regime == "maxscale":
            scale_values = torch.tensor([448.0, 2.0**-9, 1.0], device=device)
        elif regime == "tinyglobal":
            scale_values = torch.tensor([2.0, 4.0, 128.0, 448.0], device=device)
            global_scale = 2.0**-118
        else:
            raise ValueError(regime)
    fp8_k_scale_f = scale_values[
        torch.randint(0, len(scale_values), scale_shape, device=device)
    ].to(torch.float8_e4m3fn)
    fp8_v_scale_f = scale_values[
        torch.randint(0, len(scale_values), scale_shape, device=device)
    ].to(torch.float8_e4m3fn)
    fp8_k_scales = fp8_k_scale_f.view(torch.uint8)
    fp8_v_scales = fp8_v_scale_f.view(torch.uint8)

    fp4_shape = (num_pages, page_size, num_heads, head_dim // 2)
    fp4_k = torch.randint(0, 256, fp4_shape, dtype=torch.uint8, device=device)
    fp4_v = torch.randint(0, 256, fp4_shape, dtype=torch.uint8, device=device)
    fp4_k_scale_f = scale_values[
        torch.randint(0, len(scale_values), scale_shape, device=device)
    ].to(torch.float8_e4m3fn)
    fp4_v_scale_f = scale_values[
        torch.randint(0, len(scale_values), scale_shape, device=device)
    ].to(torch.float8_e4m3fn)
    fp4_k_scales = fp4_k_scale_f.view(torch.uint8)
    fp4_v_scales = fp4_v_scale_f.view(torch.uint8)

    format_cycles = {
        "a16_fp8": (0, 1),
        "a16_fp4": (0, 2),
        "fp8_fp4": (1, 2),
        "mixed": (0, 1, 2),
    }
    if page_mode in format_cycles:
        cycle = torch.tensor(format_cycles[page_mode], device=device, dtype=torch.uint8)
        page_format = cycle[torch.arange(num_pages, device=device) % len(cycle)]
    elif page_mode == "a16_fp8_runs":
        page_format = ((torch.arange(num_pages, device=device) // 16) % 2).to(torch.uint8)
    else:
        page_format = torch.full(
            (num_pages,), {"a16": 0, "fp8": 1, "fp4": 2}[page_mode],
            dtype=torch.uint8, device=device,
        )
    fp8_pages = page_format == 1
    fp4_pages = page_format == 2

    def effective_scale(block_scale_f: torch.Tensor) -> torch.Tensor:
        # The kernels apply bf16(s * g) (convertE4M3ScaleToA16Bits / the folded form).
        return (block_scale_f.float() * global_scale).to(dtype).float().unsqueeze(-1)

    fp8_k_decoded = (
        fp8_k.float().reshape(*scale_shape, 16) * effective_scale(fp8_k_scale_f)
    ).reshape(shape)
    fp8_v_decoded = (
        fp8_v.float().reshape(*scale_shape, 16) * effective_scale(fp8_v_scale_f)
    ).reshape(shape)
    fp4_k_decoded = (
        _decode_fp4(fp4_k).reshape(*scale_shape, 16) * effective_scale(fp4_k_scale_f)
    ).reshape(shape)
    fp4_v_decoded = (
        _decode_fp4(fp4_v).reshape(*scale_shape, 16) * effective_scale(fp4_v_scale_f)
    ).reshape(shape)
    reference_k[fp8_pages] = fp8_k_decoded[fp8_pages].to(dtype)
    reference_v[fp8_pages] = fp8_v_decoded[fp8_pages].to(dtype)
    reference_k[fp4_pages] = fp4_k_decoded[fp4_pages].to(dtype)
    reference_v[fp4_pages] = fp4_v_decoded[fp4_pages].to(dtype)

    scalar = torch.full((), global_scale, dtype=torch.float32, device=device)
    transport = MixedKVPagedCache(
        fp8_k,
        fp8_v,
        fp8_k_scales,
        fp8_v_scales,
        fp4_k,
        fp4_v,
        fp4_k_scales,
        fp4_v_scales,
        page_format,
        torch.empty((num_pages, 2), dtype=torch.float32, device=device),
        torch.empty((0, page_size, num_heads, 4), dtype=torch.float32, device=device),
        torch.empty(4, dtype=torch.float32, device=device),
        scalar,
        scalar,
        scalar,
        scalar,
    )
    return canonical_k, canonical_v, reference_k, reference_v, transport


@pytest.mark.skipif(
    get_compute_capability(torch.device("cuda"))[0] not in (9, 12),
    reason="mixed-page XQA targets SM90 and SM12x",
)
@pytest.mark.parametrize(
    "q_len,kv_layout",
    [(1, "NHD"), (4, "NHD"), (64, "NHD"), (4, "HND")],
)
@pytest.mark.parametrize(
    "page_mode",
    ["a16", "fp8", "fp4", "a16_fp8_runs", "a16_fp8", "a16_fp4", "fp8_fp4", "mixed"],
)
def test_xqa_mixed_page_transport_matches_register_expansion(
    q_len, kv_layout, page_mode
):
    """Every query-span mode consumes one interleaved FP4/FP8/A16 page stream."""

    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 2
    page_size = 16
    pages_per_request = 18
    num_pages = batch_size * pages_per_request
    num_kv_heads = 2
    group_size = 4
    head_dim = 128
    shape = (num_pages, page_size, num_kv_heads, head_dim)
    canonical_k, canonical_v, reference_k, reference_v, transport = _make_transport(
        shape, dtype, device, page_mode
    )
    page_table = torch.arange(
        num_pages, dtype=torch.int32, device=device
    ).reshape(batch_size, pages_per_request)
    seq_lens = torch.full(
        (batch_size,), pages_per_request * page_size - 3, dtype=torch.int32, device=device
    )
    query = torch.randn(
        batch_size * q_len,
        num_kv_heads * group_size,
        head_dim,
        dtype=dtype,
        device=device,
    )
    mask = None if q_len == 1 else _pack_query_mask(batch_size, q_len, device)
    workspace_ref = torch.zeros(64 << 20, dtype=torch.uint8, device=device)
    workspace_mixed = torch.zeros_like(workspace_ref)

    if kv_layout == "HND":
        canonical_k = canonical_k.transpose(1, 2)
        canonical_v = canonical_v.transpose(1, 2)
        reference_k = reference_k.transpose(1, 2)
        reference_v = reference_v.transpose(1, 2)
        transport = transport._replace(
            fp8_k_payload=transport.fp8_k_payload.transpose(1, 2),
            fp8_v_payload=transport.fp8_v_payload.transpose(1, 2),
            fp8_k_scales=transport.fp8_k_scales.transpose(1, 2),
            fp8_v_scales=transport.fp8_v_scales.transpose(1, 2),
            fp4_k_payload=transport.fp4_k_payload.transpose(1, 2),
            fp4_v_payload=transport.fp4_v_payload.transpose(1, 2),
            fp4_k_scales=transport.fp4_k_scales.transpose(1, 2),
            fp4_v_scales=transport.fp4_v_scales.transpose(1, 2),
        )

    kwargs = dict(
        block_tables=page_table,
        seq_lens=seq_lens,
        max_seq_len=int(seq_lens.max()),
        bmm1_scale=head_dim**-0.5,
        kv_layout=kv_layout,
        q_len_per_req=q_len,
        mask=mask,
        enable_pdl=True,
    )
    a16_reference_transport = transport._replace(
        page_format=torch.zeros_like(transport.page_format)
    )
    expected = xqa_batch_decode_with_kv_cache(
        query,
        (reference_k, reference_v),
        workspace_ref,
        page_transport=a16_reference_transport,
        page_transport_static_format=0,
        **kwargs,
    )
    actual = xqa_batch_decode_with_kv_cache(
        query,
        (canonical_k, canonical_v),
        workspace_mixed,
        page_transport=transport,
        page_transport_static_format={"fp8": 1, "fp4": 2}.get(page_mode),
        **kwargs,
    )
    assert not torch.isnan(expected).any()
    assert not torch.isnan(actual).any()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    get_compute_capability(torch.device("cuda"))[0] not in (9, 12),
    reason="the native block-scaled E4M3 XQA arm targets SM90 and SM12x",
)
@pytest.mark.parametrize("q_len", [1, 64])
def test_xqa_native_block_fp8_matches_a16_expansion(q_len):
    """The format-specialized FP8 arm is the mixed-router FP8 consumer."""

    torch.manual_seed(11)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 2
    page_size = 16
    pages_per_request = 18
    num_pages = batch_size * pages_per_request
    num_kv_heads = 2
    group_size = 4
    head_dim = 128
    shape = (num_pages, page_size, num_kv_heads, head_dim)
    _, _, reference_k, reference_v, transport = _make_transport(
        shape, dtype, device, "fp8"
    )
    page_table = torch.arange(
        num_pages, dtype=torch.int32, device=device
    ).reshape(batch_size, pages_per_request)
    seq_lens = torch.full(
        (batch_size,), pages_per_request * page_size - 3,
        dtype=torch.int32, device=device,
    )
    query = torch.randn(
        batch_size * q_len, num_kv_heads * group_size, head_dim,
        dtype=dtype, device=device,
    )
    mask = None if q_len == 1 else _pack_query_mask(batch_size, q_len, device)
    kwargs = dict(
        block_tables=page_table,
        seq_lens=seq_lens,
        max_seq_len=int(seq_lens.max()),
        bmm1_scale=head_dim**-0.5,
        q_len_per_req=q_len,
        mask=mask,
        enable_pdl=True,
    )
    expected = xqa_batch_decode_with_kv_cache(
        query, (reference_k, reference_v),
        torch.zeros(64 << 20, dtype=torch.uint8, device=device), **kwargs,
    )
    actual = xqa_batch_decode_with_kv_cache(
        query, (transport.fp8_k_payload, transport.fp8_v_payload),
        torch.zeros(64 << 20, dtype=torch.uint8, device=device),
        kv_cache_sf=(
            transport.fp8_k_scales.view(torch.float8_e4m3fn),
            transport.fp8_v_scales.view(torch.float8_e4m3fn),
        ),
        **kwargs,
    )
    assert not torch.isnan(actual).any()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# Tail / long-CTA / value-range cases, run by run_xqa_mixed_page_transport.py:
# (seq_len, XQA_NB_SUB_SEQ override, regime, q_len_per_req).  q=1 is the sm90 GMMA host
# (mha_sm90.cu; mha.cu on sm120); q=4 is the SPEC_DEC mha.cu build (on sm90 the M_TILESIZE 16
# compact build with the [44] placement expansion, which has its own fold / fallback bodies).
TAIL_CASES = [
    # nbIters < kAhead + 1 tails (C7 class): one and two 64-token tiles per CTA, odd tails.
    (50, None, "normal", 1),
    (100, None, "normal", 1),
    (130, None, "normal", 1),
    # 35 tiles in one CTA (single sub-sequence): more than two metadata chunks, so the
    # chunk refill and its ready barrier are exercised (tiles 32.. read a refilled chunk).
    (2200, 1, "normal", 1),
    # converter decode value ranges (folded-scale fast path and its fallbacks), both hosts:
    # every finite E4M3 payload code incl. subnormals and 448 through the decode; FP4 spans
    # with |s g| >= 4 (448 x 1: the fold's 255.5 * 2^120 bound fails, two-multiply body);
    # |g| < 2^-117 (foldOk false: fallback on every span, exact s * g).
    (285, None, "subnormal", 1),
    (285, None, "maxscale", 1),
    (285, None, "tinyglobal", 1),
    (285, None, "subnormal", 4),
    (285, None, "maxscale", 4),
    (285, None, "tinyglobal", 4),
]


def test_xqa_mixed_page_transport_tails_and_value_ranges(
    page_mode, seq_len, nb_sub_seq, regime, q_len=1
):
    """Short and odd sequence tails, 35-tile CTAs, and extreme E4M3 / scale values (q=1, q=4)."""

    import os

    if get_compute_capability(torch.device("cuda"))[0] not in (9, 12):
        raise RuntimeError("Skipped: mixed-page XQA targets SM90 and SM12x")
    torch.manual_seed(13)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 2
    page_size = 16
    pages_per_request = (seq_len + page_size - 1) // page_size
    num_pages = batch_size * pages_per_request
    num_kv_heads = 2
    group_size = 4
    head_dim = 128
    shape = (num_pages, page_size, num_kv_heads, head_dim)
    canonical_k, canonical_v, reference_k, reference_v, transport = _make_transport(
        shape, dtype, device, page_mode, regime
    )
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).reshape(
        batch_size, pages_per_request
    )
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    query = torch.randn(
        batch_size * q_len, num_kv_heads * group_size, head_dim, dtype=dtype, device=device
    )
    mask = None if q_len == 1 else _pack_query_mask(batch_size, q_len, device)
    kwargs = dict(
        block_tables=page_table,
        seq_lens=seq_lens,
        max_seq_len=seq_len,
        bmm1_scale=head_dim**-0.5,
        q_len_per_req=q_len,
        mask=mask,
        enable_pdl=True,
    )
    saved = os.environ.get("XQA_NB_SUB_SEQ")
    if nb_sub_seq is not None:
        os.environ["XQA_NB_SUB_SEQ"] = str(nb_sub_seq)
    try:
        expected = xqa_batch_decode_with_kv_cache(
            query,
            (reference_k, reference_v),
            torch.zeros(64 << 20, dtype=torch.uint8, device=device),
            page_transport=transport._replace(
                page_format=torch.zeros_like(transport.page_format)
            ),
            page_transport_static_format=0,
            **kwargs,
        )
        actual = xqa_batch_decode_with_kv_cache(
            query,
            (canonical_k, canonical_v),
            torch.zeros(64 << 20, dtype=torch.uint8, device=device),
            page_transport=transport,
            page_transport_static_format={"fp8": 1, "fp4": 2}.get(page_mode),
            **kwargs,
        )
        torch.cuda.synchronize()
    finally:
        if saved is None:
            os.environ.pop("XQA_NB_SUB_SEQ", None)
        else:
            os.environ["XQA_NB_SUB_SEQ"] = saved
    assert not torch.isnan(expected).any()
    assert not torch.isnan(actual).any()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    get_compute_capability(torch.device("cuda"))[0] not in (9, 12),
    reason="mixed-page XQA targets SM90 and SM12x",
)
@pytest.mark.parametrize("pages_per_request", [18, 256])
def test_xqa_mixed_a16_stream_matches_stock_decode(pages_per_request):
    """Independent numeric check of the mixed-page consumer.

    The bit-exact matrix compares mixed streams against the all-A16 mixed
    stream, i.e. the same kernel with different page converters; a consumer
    bug that changes both sides identically would pass it.  This case runs the
    all-A16 mixed stream (mha_sm90.cu on SM90 at q=1) against the stock bf16
    XQA decode (mha.cu), which shares none of the mixed kernel's consumer code.
    The two kernels differ in P precision and reduction order, so the
    comparison has a tolerance calibrated on the unmodified kernel (max |diff|
    1.95e-3 at |ref| <= 0.38 for 18 pages, 4.9e-4 at |ref| <= 0.10 for 256:
    one bf16 ulp; the tolerance below allows ~3).  256 pages per request cover
    64 tiles, the running max/sum update on every tile and the multi-block
    merge.
    """

    torch.manual_seed(13)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 2
    page_size = 16
    num_pages = batch_size * pages_per_request
    num_kv_heads = 2
    group_size = 4
    head_dim = 128
    shape = (num_pages, page_size, num_kv_heads, head_dim)
    canonical_k, canonical_v, _, _, transport = _make_transport(
        shape, dtype, device, "a16"
    )
    page_table = torch.arange(
        num_pages, dtype=torch.int32, device=device
    ).reshape(batch_size, pages_per_request)
    seq_lens = torch.full(
        (batch_size,), pages_per_request * page_size - 3, dtype=torch.int32, device=device
    )
    query = torch.randn(
        batch_size, num_kv_heads * group_size, head_dim, dtype=dtype, device=device
    )
    kwargs = dict(
        block_tables=page_table,
        seq_lens=seq_lens,
        max_seq_len=int(seq_lens.max()),
        bmm1_scale=head_dim**-0.5,
        q_len_per_req=1,
        mask=None,
        enable_pdl=True,
    )
    expected = xqa_batch_decode_with_kv_cache(
        query, (canonical_k, canonical_v),
        torch.zeros(64 << 20, dtype=torch.uint8, device=device), **kwargs,
    )
    actual = xqa_batch_decode_with_kv_cache(
        query, (canonical_k, canonical_v),
        torch.zeros(64 << 20, dtype=torch.uint8, device=device),
        page_transport=transport, page_transport_static_format=0, **kwargs,
    )
    assert not torch.isnan(expected).any()
    assert not torch.isnan(actual).any()
    err = (actual.float() - expected.float()).abs().max().item()
    ref = expected.float().abs().max().item()
    print(f"[a16_vs_stock-{pages_per_request}] max|diff|={err:.3e} max|ref|={ref:.3e}",
          flush=True)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=2e-3)
