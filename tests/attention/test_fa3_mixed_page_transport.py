"""Mixed-page (ragged format) paged KV through the FA3 batch prefill kernel on sm90.

Every query shape shares this path (prefill, continued prefill, draft
verification, decode); the reference is the same kernel with every page tagged
A16 over the explicitly expanded cache, so the comparison is bit-exact.
"""

import importlib.util
import pathlib

import pytest
import torch

from flashinfer.mixed_page_prefill import (
    mixed_page_prefill_jit_args,
    mixed_page_prefill_run_args,
)
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
from flashinfer.utils import get_compute_capability

_xqa_test = importlib.util.spec_from_file_location(
    "_xqa_mixed_test", pathlib.Path(__file__).with_name("test_xqa_mixed_page_transport.py")
)
_xqa_mod = importlib.util.module_from_spec(_xqa_test)
_xqa_test.loader.exec_module(_xqa_mod)
_make_transport = _xqa_mod._make_transport


def _skip_unless_sm90():
    if get_compute_capability(torch.device("cuda"))[0] != 9:
        pytest.skip("the mixed-page FA3 producer targets sm90")


def _plan(wrapper, qo_indptr, kv_indptr, kv_indices, last_page_len, num_qo_heads, num_kv_heads,
          head_dim, causal):
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        16,
        causal=causal,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )


@pytest.mark.parametrize("q_len", [1, 4, 64, 130])
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize(
    "page_mode", ["a16", "fp8", "fp4", "a16_fp8_runs", "a16_fp8", "a16_fp4", "fp8_fp4", "mixed"]
)
def test_fa3_mixed_page_transport_matches_a16_expansion(q_len, kv_layout, page_mode):
    _skip_unless_sm90()
    torch.manual_seed(11)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 2
    page_size = 16
    pages_per_request = 18
    num_pages = batch_size * pages_per_request
    num_kv_heads = 2
    group_size = 4
    num_qo_heads = num_kv_heads * group_size
    head_dim = 128
    shape = (num_pages, page_size, num_kv_heads, head_dim)
    canonical_k, canonical_v, reference_k, reference_v, transport = _make_transport(
        shape, dtype, device, page_mode
    )
    kv_len = pages_per_request * page_size - 3
    if q_len > kv_len:
        pytest.skip("query longer than the cache")

    qo_indptr = torch.arange(0, (batch_size + 1) * q_len, q_len, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(
        0, (batch_size + 1) * pages_per_request, pages_per_request, dtype=torch.int32, device=device
    )
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    last_page_len = torch.full((batch_size,), kv_len - (pages_per_request - 1) * page_size,
                               dtype=torch.int32, device=device)
    q = torch.randn(batch_size * q_len, num_qo_heads, head_dim, dtype=dtype, device=device)
    sm_scale = head_dim**-0.5

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

    jit_args = mixed_page_prefill_jit_args(dtype, dtype, dtype, head_dim)
    workspace = torch.empty(128 << 20, dtype=torch.uint8, device=device)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(
        workspace, kv_layout, backend="fa3", jit_args=jit_args
    )
    _plan(wrapper, qo_indptr, kv_indptr, kv_indices, last_page_len, num_qo_heads, num_kv_heads,
          head_dim, causal=q_len > 1)

    a16_reference = transport._replace(page_format=torch.zeros_like(transport.page_format))
    expected = wrapper.run(
        q, (reference_k, reference_v),
        *mixed_page_prefill_run_args(a16_reference, sm_scale, 0, kv_layout),
    )
    actual = wrapper.run(
        q, (canonical_k, canonical_v),
        *mixed_page_prefill_run_args(transport, sm_scale, {"fp8": 1, "fp4": 2}.get(page_mode),
                                     kv_layout),
    )
    assert not torch.isnan(expected).any()
    assert not torch.isnan(actual).any()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # Sanity against the stock FA3 paged kernel (different accumulation order: tolerance).
    stock = BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout, backend="fa3")
    _plan(stock, qo_indptr, kv_indptr, kv_indices, last_page_len, num_qo_heads, num_kv_heads,
          head_dim, causal=q_len > 1)
    baseline = stock.run(q, (reference_k, reference_v))
    torch.testing.assert_close(actual.float(), baseline.float(), rtol=2e-2, atol=2e-2)
