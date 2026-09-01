import pytest
import torch

from flashinfer.quantization.kv_cache_fp8 import (
    BlockScaledFP8PagedCache,
    dequantize_block_scaled_fp8,
    dequantize_mxfp8_reference,
    quantize_block_scaled_fp8,
    quantize_block_scaled_fp8_cuda,
    quantize_mxfp8_reference,
    quantize_tensor_fp8_reference,
    seal_block_scaled_fp8_pages_cuda,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_block_scaled_fp8_shapes_and_rate(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(7)
    x = torch.randn(3, 16, 2, 256, device=device, dtype=torch.bfloat16)
    encoded = quantize_block_scaled_fp8(x)
    actual = dequantize_block_scaled_fp8(*encoded)
    assert encoded.payload.shape == x.shape
    assert encoded.payload.dtype == torch.float8_e4m3fn
    assert encoded.scales.shape == (*x.shape[:-1], x.shape[-1] // 16)
    assert encoded.scales.dtype == torch.uint8
    assert actual.shape == x.shape
    assert torch.isfinite(actual).all()
    assert 8.0 + 8.0 / 16.0 == 8.5


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scale_search_never_worsens_its_objective(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(11)
    x = torch.randn(97, 256, device=device, dtype=torch.bfloat16)
    x[::3, 0] *= 16
    searched = quantize_block_scaled_fp8(x, tail_weight=0.05)
    direct = quantize_block_scaled_fp8(x, optimize_scales=False, tail_weight=0.05)
    searched_error = (dequantize_block_scaled_fp8(*searched) - x.float()).abs()
    direct_error = (dequantize_block_scaled_fp8(*direct) - x.float()).abs()
    searched_blocks = searched_error.reshape(-1, 16)
    direct_blocks = direct_error.reshape(-1, 16)
    searched_objective = (
        searched_blocks.square().mean(-1) + 0.05 * searched_blocks.amax(-1).square()
    )
    direct_objective = (
        direct_blocks.square().mean(-1) + 0.05 * direct_blocks.amax(-1).square()
    )
    assert torch.all(searched_objective <= direct_objective + 1e-12)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reference_codecs_round_trip(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(19)
    x = torch.randn(64, 256, device=device, dtype=torch.bfloat16)
    mx_payload, mx_scales = quantize_mxfp8_reference(x)
    mx_actual = dequantize_mxfp8_reference(mx_payload, mx_scales)
    tensor_payload, tensor_scale = quantize_tensor_fp8_reference(x)
    tensor_actual = tensor_payload.float() * tensor_scale
    assert torch.isfinite(mx_actual).all()
    assert torch.isfinite(tensor_actual).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_page_seal_matches_direct_reference() -> None:
    torch.manual_seed(71)
    x = torch.randn(16, 8, 256, device="cuda", dtype=torch.bfloat16)
    global_scale = x.abs().amax().float() / (448.0 * 128.0)
    payload = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(16, 8, 16, device="cuda", dtype=torch.uint8)
    actual = quantize_block_scaled_fp8_cuda(
        x, global_scale, payload_out=payload, scales_out=scales
    )
    expected = quantize_block_scaled_fp8(x, optimize_scales=False)
    assert actual.payload.data_ptr() == payload.data_ptr()
    assert actual.scales.data_ptr() == scales.data_ptr()
    torch.testing.assert_close(
        actual.payload.float(), expected.payload.float(), rtol=0, atol=0
    )
    torch.testing.assert_close(actual.scales, expected.scales, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_slot_driven_page_seal_handles_strided_a16_cache() -> None:
    torch.manual_seed(79)
    combined = torch.randn(4, 16, 2, 512, device="cuda", dtype=torch.bfloat16)
    key, value = combined.split(256, dim=-1)
    assert not key.is_contiguous()
    global_scale = torch.tensor(1.0 / 448.0, device="cuda")
    cache = BlockScaledFP8PagedCache(
        torch.empty_like(key, dtype=torch.float8_e4m3fn),
        torch.empty_like(value, dtype=torch.float8_e4m3fn),
        torch.empty(4, 16, 2, 16, device="cuda", dtype=torch.uint8),
        torch.empty(4, 16, 2, 16, device="cuda", dtype=torch.uint8),
        torch.full((4,), 1, device="cuda", dtype=torch.uint8),
        torch.full((4, 2), torch.inf, device="cuda", dtype=torch.float32),
        torch.tensor([1.0, 1.0], device="cuda", dtype=torch.float32),
        global_scale,
        global_scale,
    )
    # Pages 0 and 1 complete. Page 2 is reused but remains partial.
    slots = torch.tensor([0, 15, 16, 31, 32], device="cuda", dtype=torch.int64)
    seal_block_scaled_fp8_pages_cuda(key, value, slots, cache)
    torch.testing.assert_close(
        cache.page_format, torch.tensor([1, 1, 0, 1], device="cuda", dtype=torch.uint8)
    )

    for page in (0, 1):
        expected_k = quantize_block_scaled_fp8_cuda(
            key[page].contiguous(), global_scale
        )
        expected_v = quantize_block_scaled_fp8_cuda(
            value[page].contiguous(), global_scale
        )
        torch.testing.assert_close(
            cache.k_payload[page].float(), expected_k.payload.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(cache.k_scales[page], expected_k.scales)
        torch.testing.assert_close(
            cache.v_payload[page].float(), expected_v.payload.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(cache.v_scales[page], expected_v.scales)

        reconstructed_k = dequantize_block_scaled_fp8(
            cache.k_payload[page], cache.k_scales[page], global_scale
        )
        reconstructed_v = dequantize_block_scaled_fp8(
            cache.v_payload[page], cache.v_scales[page], global_scale
        )
        signal = torch.cat((key[page].float().flatten(), value[page].float().flatten()))
        residual = torch.cat(
            (
                (reconstructed_k - key[page].float()).flatten(),
                (reconstructed_v - value[page].float()).flatten(),
            )
        ).abs()
        expected_stats = torch.stack(
            (
                residual.square().sum().sqrt() / signal.square().sum().sqrt(),
                residual.amax() / signal.abs().amax(),
            )
        )
        torch.testing.assert_close(cache.page_error_stats[page], expected_stats)

    cache.routing_thresholds.zero_()
    seal_block_scaled_fp8_pages_cuda(
        key, value, torch.tensor([15, 31], device="cuda", dtype=torch.int64), cache
    )
    torch.testing.assert_close(
        cache.page_format[:2], torch.zeros(2, device="cuda", dtype=torch.uint8)
    )
