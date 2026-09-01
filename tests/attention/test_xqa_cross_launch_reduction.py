import pytest
import torch

import flashinfer
from flashinfer.fp4_quantization import e2m1_and_ufp8sf_scale_to_float
from flashinfer.quantization.kv_cache_fp8 import (
    dequantize_block_scaled_fp8,
    quantize_block_scaled_fp8,
)
from flashinfer.utils import get_compute_capability


def _quantize_xqa_nvfp4(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    amax = x.abs().amax().float().clamp_min_(1e-12)
    global_scale = amax / (448.0 * 6.0)
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


def _dequantize_xqa_nvfp4(
    packed: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    flat = e2m1_and_ufp8sf_scale_to_float(
        packed.reshape(-1, packed.shape[-1]),
        scales.reshape(-1, scales.shape[-1]),
        global_scale,
        sf_vec_size=16,
        is_sf_swizzled_layout=False,
    )
    return flat.reshape(*packed.shape[:-1], packed.shape[-1] * 2).to(torch.bfloat16)


@pytest.mark.skipif(
    get_compute_capability(torch.device("cuda"))[0] not in (9, 10, 12),
    reason="XQA requires SM90 or newer",
)
def test_xqa_block_scaled_fp8_matches_explicit_dequantization() -> None:
    """The fused scale stream preserves both K and V token/lane layouts."""
    torch.manual_seed(23)
    batch_size = 4
    page_size = 16
    pages_per_request = 8
    num_kv_heads = 2
    num_q_heads = 4
    head_dim = 256
    total_pages = batch_size * pages_per_request

    query = torch.randn(
        batch_size, 1, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    key = torch.randn(
        total_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    value = torch.randn_like(key)
    key_q8 = quantize_block_scaled_fp8(key)
    value_q8 = quantize_block_scaled_fp8(value)
    key_dq = dequantize_block_scaled_fp8(*key_q8).to(torch.bfloat16)
    value_dq = dequantize_block_scaled_fp8(*value_q8).to(torch.bfloat16)
    page_table = torch.arange(total_pages, dtype=torch.int32, device="cuda").reshape(
        batch_size, pages_per_request
    )
    seq_lens = torch.full(
        (batch_size, 1),
        pages_per_request * page_size,
        dtype=torch.int32,
        device="cuda",
    )

    def run(
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        k_scales: torch.Tensor | None = None,
        v_scales: torch.Tensor | None = None,
        q_scale: torch.Tensor | float = 1.0,
        kv_scale: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        storage = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        output = torch.empty_like(query)
        flashinfer.xqa(
            query,
            k_cache,
            v_cache,
            page_table,
            seq_lens,
            output,
            storage[8 * 1024 * 1024 :],
            storage[: 8 * 1024 * 1024],
            num_kv_heads,
            page_size,
            q_scale=q_scale,
            kv_scale=kv_scale,
            enable_pdl=False,
            k_sf_cache=k_scales,
            v_sf_cache=v_scales,
        )
        return output

    reference = run(key_dq, value_dq)
    actual = run(
        key_q8.payload,
        value_q8.payload,
        k_scales=key_q8.scales,
        v_scales=value_q8.scales,
        q_scale=key_q8.global_scale / value_q8.global_scale,
        kv_scale=value_q8.global_scale,
    )
    torch.testing.assert_close(actual, reference, rtol=1e-2, atol=4e-3)


@pytest.mark.skipif(
    get_compute_capability(torch.device("cuda"))[0] not in (9, 10, 12),
    reason="XQA requires SM90 or newer",
)
def test_xqa_cross_launch_page_scatter_reduction() -> None:
    """Two format-style page runs reproduce one ordinary XQA invocation."""
    torch.manual_seed(17)
    batch_size = 4
    page_size = 16
    pages_per_request = 8
    num_kv_heads = 2
    num_q_heads = 4
    head_dim = 256
    total_pages = batch_size * pages_per_request

    query = torch.randn(
        batch_size, 1, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    key = torch.randn(
        total_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    value = torch.randn_like(key)
    page_table = torch.arange(total_pages, dtype=torch.int32, device="cuda").reshape(
        batch_size, pages_per_request
    )
    seq_lens = torch.full(
        (batch_size, 1),
        pages_per_request * page_size,
        dtype=torch.int32,
        device="cuda",
    )

    def workspace() -> tuple[torch.Tensor, torch.Tensor]:
        storage = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        return storage[: 8 * 1024 * 1024], storage[8 * 1024 * 1024 :]

    reference = torch.empty_like(query)
    ref_semaphore, ref_scratch = workspace()
    flashinfer.xqa(
        query,
        key,
        value,
        page_table,
        seq_lens,
        reference,
        ref_scratch,
        ref_semaphore,
        num_kv_heads,
        page_size,
        q_scale=1.0,
        kv_scale=1.0,
        enable_pdl=False,
    )

    actual = torch.empty_like(query)
    semaphore, scratch = workspace()
    for split_base, columns in enumerate((slice(0, None, 2), slice(1, None, 2))):
        split_table = page_table[:, columns].contiguous()
        split_lens = torch.full_like(seq_lens, split_table.shape[1] * page_size)
        flashinfer.xqa(
            query,
            key,
            value,
            split_table,
            split_lens,
            actual,
            scratch,
            semaphore,
            num_kv_heads,
            page_size,
            q_scale=1.0,
            kv_scale=1.0,
            enable_pdl=False,
            local_subseq_override=1,
            reduction_subseq_total=2,
            reduction_subseq_base=split_base,
        )

    torch.testing.assert_close(actual, reference, rtol=1e-2, atol=4e-3)


@pytest.mark.skipif(
    get_compute_capability(torch.device("cuda"))[0] not in (9, 10, 12),
    reason="XQA requires SM90 or newer",
)
def test_xqa_ragged_query_indptr_matches_per_request_invocations() -> None:
    """Ragged draft verification uses the same XQA query/mask machinery."""
    torch.manual_seed(31)
    q_lens = (4, 2, 3)
    batch_size = len(q_lens)
    max_q_len = max(q_lens)
    page_size = 16
    pages_per_request = 4
    num_kv_heads = 2
    num_q_heads = 4
    head_dim = 256
    total_pages = batch_size * pages_per_request

    query = torch.randn(
        sum(q_lens), num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    key = torch.randn(
        total_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    value = torch.randn_like(key)
    page_table = torch.arange(total_pages, dtype=torch.int32, device="cuda").reshape(
        batch_size, pages_per_request
    )
    seq_lens = torch.full(
        (batch_size, 1),
        pages_per_request * page_size,
        dtype=torch.int32,
        device="cuda",
    )
    q_cu_seq_lens = torch.tensor((0, 4, 6, 9), dtype=torch.int32, device="cuda")
    ragged_mask = torch.full(
        (sum(q_lens), 2), 0xFFFF, dtype=torch.uint16, device="cuda"
    )

    def workspace() -> tuple[torch.Tensor, torch.Tensor]:
        storage = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        return storage[: 8 * 1024 * 1024], storage[8 * 1024 * 1024 :]

    actual = torch.empty_like(query)
    semaphore, scratch = workspace()
    flashinfer.xqa(
        query,
        key,
        value,
        page_table,
        seq_lens,
        actual,
        scratch,
        semaphore,
        num_kv_heads,
        page_size,
        enable_pdl=False,
        q_seq_len=max_q_len,
        q_cu_seq_lens=q_cu_seq_lens,
        mask=ragged_mask,
    )

    expected = torch.empty_like(query)
    offset = 0
    for request, q_len in enumerate(q_lens):
        request_query = query[offset : offset + q_len].reshape(
            1, 1, q_len, num_q_heads, head_dim
        )
        request_output = expected[offset : offset + q_len].reshape_as(request_query)
        request_mask = torch.full(
            (1, q_len, 2), 0xFFFF, dtype=torch.uint16, device="cuda"
        )
        request_semaphore, request_scratch = workspace()
        flashinfer.xqa(
            request_query,
            key,
            value,
            page_table[request : request + 1],
            seq_lens[request : request + 1],
            request_output,
            request_scratch,
            request_semaphore,
            num_kv_heads,
            page_size,
            enable_pdl=False,
            q_seq_len=q_len,
            mask=request_mask,
        )
        offset += q_len

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=4e-3)


@pytest.mark.skipif(
    get_compute_capability(torch.device("cuda"))[0] != 12,
    reason="XQA NVFP4 KV requires SM120",
)
def test_xqa_cross_launch_a16_nvfp4_reduction() -> None:
    """A16 and native packed-Q4 producers share XQA's stable reduction."""
    torch.manual_seed(29)
    batch_size = 4
    page_size = 16
    pages_per_request = 8
    num_kv_heads = 2
    num_q_heads = 4
    head_dim = 256
    total_pages = batch_size * pages_per_request

    query = torch.randn(
        batch_size, 1, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    key = torch.randn(
        total_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    value = torch.randn_like(key)
    key_q4, key_sf, key_global = _quantize_xqa_nvfp4(key)
    value_q4, value_sf, value_global = _quantize_xqa_nvfp4(value)
    key_dq = _dequantize_xqa_nvfp4(key_q4, key_sf, key_global)
    value_dq = _dequantize_xqa_nvfp4(value_q4, value_sf, value_global)

    page_table = torch.arange(total_pages, dtype=torch.int32, device="cuda").reshape(
        batch_size, pages_per_request
    )
    seq_lens = torch.full(
        (batch_size, 1),
        pages_per_request * page_size,
        dtype=torch.int32,
        device="cuda",
    )

    def workspace() -> tuple[torch.Tensor, torch.Tensor]:
        storage = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        return storage[: 8 * 1024 * 1024], storage[8 * 1024 * 1024 :]

    reference = torch.empty_like(query)
    ref_semaphore, ref_scratch = workspace()
    flashinfer.xqa(
        query,
        key_q4,
        value_q4,
        page_table,
        seq_lens,
        reference,
        ref_scratch,
        ref_semaphore,
        num_kv_heads,
        page_size,
        q_scale=key_global / value_global,
        kv_scale=value_global,
        enable_pdl=False,
        k_sf_cache=key_sf,
        v_sf_cache=value_sf,
    )

    actual = torch.empty_like(query)
    semaphore, scratch = workspace()
    a16_table = page_table[:, 0::2].contiguous()
    q4_table = page_table[:, 1::2].contiguous()
    split_lens = torch.full_like(seq_lens, a16_table.shape[1] * page_size)
    flashinfer.xqa(
        query,
        key_dq,
        value_dq,
        a16_table,
        split_lens,
        actual,
        scratch,
        semaphore,
        num_kv_heads,
        page_size,
        q_scale=1.0,
        kv_scale=1.0,
        enable_pdl=False,
        local_subseq_override=1,
        reduction_subseq_total=2,
        reduction_subseq_base=0,
    )
    flashinfer.xqa(
        query,
        key_q4,
        value_q4,
        q4_table,
        split_lens,
        actual,
        scratch,
        semaphore,
        num_kv_heads,
        page_size,
        q_scale=key_global / value_global,
        kv_scale=value_global,
        enable_pdl=False,
        local_subseq_override=1,
        reduction_subseq_total=2,
        reduction_subseq_base=1,
        k_sf_cache=key_sf,
        v_sf_cache=value_sf,
    )

    torch.testing.assert_close(actual, reference, rtol=1e-2, atol=4e-3)
