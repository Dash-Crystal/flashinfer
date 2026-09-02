import pytest
import torch

from flashinfer.quantization.kv_cache_fp8 import (
    MixedKVPagedCache,
    dequantize_block_scaled_fp8,
    dequantize_mxfp8_reference,
    quantize_block_scaled_fp8,
    quantize_block_scaled_fp8_cuda,
    quantize_mxfp8_reference,
    quantize_tensor_fp8_reference,
    seal_mixed_kv_pages_cuda,
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
@pytest.mark.parametrize("selected_format", [0, 1, 2])
def test_page_event_driven_mixed_seal_handles_strided_a16_cache(
    selected_format: int,
) -> None:
    torch.manual_seed(79)
    combined = torch.randn(4, 16, 2, 512, device="cuda", dtype=torch.bfloat16)
    key, value = combined.split(256, dim=-1)
    assert not key.is_contiguous()
    fp8_global_scale = torch.tensor(1.0 / 448.0, device="cuda")
    fp4_global_scale = torch.tensor(1.0 / 448.0, device="cuda")
    fp8_k_payload = torch.full_like(key, 0x5A, dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    fp8_v_payload = torch.full_like(value, 0x5A, dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    fp8_k_scales = torch.full(
        (4, 16, 2, 16), 0x5A, device="cuda", dtype=torch.uint8
    )
    fp8_v_scales = torch.full_like(fp8_k_scales, 0x5A)
    fp4_k_payload = torch.full(
        (4, 16, 2, 128), 0x5A, device="cuda", dtype=torch.uint8
    )
    fp4_v_payload = torch.full_like(fp4_k_payload, 0x5A)
    fp4_k_scales = torch.full_like(fp8_k_scales, 0x5A)
    fp4_v_scales = torch.full_like(fp8_k_scales, 0x5A)
    if selected_format == 2:
        thresholds = [float("inf")] * 4
    elif selected_format == 1:
        thresholds = [float("-inf"), float("inf"), float("inf"), float("inf")]
    else:
        thresholds = [float("-inf"), float("inf")] * 2
    cache = MixedKVPagedCache(
        fp8_k_payload,
        fp8_v_payload,
        fp8_k_scales,
        fp8_v_scales,
        fp4_k_payload,
        fp4_v_payload,
        fp4_k_scales,
        fp4_v_scales,
        torch.full((4,), 1, device="cuda", dtype=torch.uint8),
        torch.full((4, 2), torch.inf, device="cuda", dtype=torch.float32),
        torch.tensor(thresholds, device="cuda", dtype=torch.float32),
        fp8_global_scale,
        fp8_global_scale,
        fp4_global_scale,
        fp4_global_scale,
    )
    # Pages 0 and 1 complete. Page 2 is reused but remains partial.
    reused = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    completed = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    seal_mixed_kv_pages_cuda(
        key,
        value,
        reused,
        torch.tensor(3, device="cuda", dtype=torch.int32),
        completed,
        torch.tensor(2, device="cuda", dtype=torch.int32),
        cache,
    )
    torch.testing.assert_close(
        cache.page_format,
        torch.tensor(
            [selected_format, selected_format, 0, 1],
            device="cuda",
            dtype=torch.uint8,
        ),
    )

    for page in (0, 1):
        page_values = torch.cat((key[page].float(), value[page].float()))
        left = torch.cat(
            (key[page, :-1].float().reshape(-1), value[page, :-1].float().reshape(-1))
        )
        right = torch.cat(
            (key[page, 1:].float().reshape(-1), value[page, 1:].float().reshape(-1))
        )
        neighbor_cos = torch.dot(left, right) / (
            torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        )
        blocks = page_values.reshape(-1, 32)
        block_peak_rms = blocks.abs().amax(-1) / blocks.square().mean(-1).sqrt()
        torch.testing.assert_close(
            cache.page_router_stats[page],
            torch.stack((neighbor_cos, block_peak_rms.amax())),
            rtol=2e-5,
            atol=2e-5,
        )

        if selected_format == 1:
            for source, payload, scales in (
                (key[page], cache.fp8_k_payload[page], cache.fp8_k_scales[page]),
                (value[page], cache.fp8_v_payload[page], cache.fp8_v_scales[page]),
            ):
                scale_values = scales.contiguous().view(torch.float8_e4m3fn).float()
                denominator = fp8_global_scale * scale_values.repeat_interleave(16, -1)
                expected = (source.float() / denominator).clamp(-448, 448).to(
                    torch.float8_e4m3fn
                )
                torch.testing.assert_close(
                    payload.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
                )
                reconstructed = payload.float() * denominator
                tensor_payload, tensor_scale = quantize_tensor_fp8_reference(source)
                tensor_reconstructed = tensor_payload.float() * tensor_scale
                block_residual = (reconstructed - source.float()).reshape(-1, 16).abs()
                tensor_residual = (
                    tensor_reconstructed - source.float()
                ).reshape(-1, 16).abs()
                block_objective = (
                    block_residual.square().mean(-1)
                    + 0.05 * block_residual.amax(-1).square()
                ).mean()
                tensor_objective = (
                    tensor_residual.square().mean(-1)
                    + 0.05 * tensor_residual.amax(-1).square()
                ).mean()
                assert block_objective <= tensor_objective
        elif selected_format == 2:
            e2m1_magnitude = torch.tensor(
                [0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda"
            )
            for source, payload, scales in (
                (key[page], cache.fp4_k_payload[page], cache.fp4_k_scales[page]),
                (value[page], cache.fp4_v_payload[page], cache.fp4_v_scales[page]),
            ):
                scale_values = scales.contiguous().view(torch.float8_e4m3fn).float()
                denominator = fp4_global_scale * scale_values.repeat_interleave(16, -1)
                normalized = source.float() / denominator
                magnitude_codes = (
                    (normalized.abs().unsqueeze(-1) - e2m1_magnitude)
                    .abs()
                    .argmin(-1)
                    .to(torch.uint8)
                )
                codes = magnitude_codes | ((normalized < 0).to(torch.uint8) << 3)
                expected = codes[..., 0::2] | (codes[..., 1::2] << 4)
                torch.testing.assert_close(payload, expected, rtol=0, atol=0)

    fp8_changed = (cache.fp8_k_payload.view(torch.uint8)[:2] != 0x5A).any()
    fp4_changed = (cache.fp4_k_payload[:2] != 0x5A).any()
    assert bool(fp8_changed) == (selected_format == 1)
    assert bool(fp4_changed) == (selected_format == 2)
