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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MixedKVPagedCache]:
    num_pages, page_size, num_heads, head_dim = shape
    canonical_k = torch.randn(shape, dtype=dtype, device=device)
    canonical_v = torch.randn(shape, dtype=dtype, device=device)
    reference_k = canonical_k.clone()
    reference_v = canonical_v.clone()

    fp8_k = (torch.randn(shape, device=device) * 2).to(torch.float8_e4m3fn)
    fp8_v = (torch.randn(shape, device=device) * 2).to(torch.float8_e4m3fn)
    scale_shape = (num_pages, page_size, num_heads, head_dim // 16)
    scale_values = torch.tensor([0.5, 1.0, 2.0], device=device)
    fp8_k_scale_f = scale_values[
        torch.randint(0, 3, scale_shape, device=device)
    ].to(torch.float8_e4m3fn)
    fp8_v_scale_f = scale_values[
        torch.randint(0, 3, scale_shape, device=device)
    ].to(torch.float8_e4m3fn)
    fp8_k_scales = fp8_k_scale_f.view(torch.uint8)
    fp8_v_scales = fp8_v_scale_f.view(torch.uint8)

    fp4_shape = (num_pages, page_size, num_heads, head_dim // 2)
    fp4_k = torch.randint(0, 256, fp4_shape, dtype=torch.uint8, device=device)
    fp4_v = torch.randint(0, 256, fp4_shape, dtype=torch.uint8, device=device)
    fp4_k_scale_f = scale_values[
        torch.randint(0, 3, scale_shape, device=device)
    ].to(torch.float8_e4m3fn)
    fp4_v_scale_f = scale_values[
        torch.randint(0, 3, scale_shape, device=device)
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
    fp8_k_decoded = (
        fp8_k.float().reshape(*scale_shape, 16)
        * fp8_k_scale_f.float().unsqueeze(-1)
    ).reshape(shape)
    fp8_v_decoded = (
        fp8_v.float().reshape(*scale_shape, 16)
        * fp8_v_scale_f.float().unsqueeze(-1)
    ).reshape(shape)
    fp4_k_decoded = (
        _decode_fp4(fp4_k).reshape(*scale_shape, 16)
        * fp4_k_scale_f.float().unsqueeze(-1)
    ).reshape(shape)
    fp4_v_decoded = (
        _decode_fp4(fp4_v).reshape(*scale_shape, 16)
        * fp4_v_scale_f.float().unsqueeze(-1)
    ).reshape(shape)
    reference_k[fp8_pages] = fp8_k_decoded[fp8_pages].to(dtype)
    reference_v[fp8_pages] = fp8_v_decoded[fp8_pages].to(dtype)
    reference_k[fp4_pages] = fp4_k_decoded[fp4_pages].to(dtype)
    reference_v[fp4_pages] = fp4_v_decoded[fp4_pages].to(dtype)

    scalar = torch.ones((), dtype=torch.float32, device=device)
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
