"""Reference codecs for block-scaled FP8 KV-cache transport.

The encode path is intentionally accuracy first.  It is used while sealing a
paged-attention page; the latency-critical decode path lives in XQA and only
loads an E4M3 payload byte and one amortized E4M3 scale byte per 16 values.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN_SUBNORMAL = 2.0**-9
# The register-dequantized A16 path multiplies payload by block scale before
# applying the global scale.  Capping the encoded scale at 128 guarantees
# 448 * 128 = 57,344 remains finite in IEEE FP16.
FP8_E4M3_A16_SCALE_MAX = 128.0


class BlockScaledFP8(NamedTuple):
    payload: torch.Tensor
    scales: torch.Tensor
    global_scale: torch.Tensor


class MixedKVPagedCache(NamedTuple):
    """Selected-only FP4/FP8/A16 page-seal operands.

    ``routing_thresholds`` contains the two FP4 signature limits followed by
    the two FP8 limits.  The producer computes the signature from the A16 page,
    chooses one format, writes only that compressed pool (if any), then
    publishes ``page_format`` last.
    """

    fp8_k_payload: torch.Tensor
    fp8_v_payload: torch.Tensor
    fp8_k_scales: torch.Tensor
    fp8_v_scales: torch.Tensor
    fp4_k_payload: torch.Tensor
    fp4_v_payload: torch.Tensor
    fp4_k_scales: torch.Tensor
    fp4_v_scales: torch.Tensor
    page_format: torch.Tensor
    page_router_stats: torch.Tensor
    routing_thresholds: torch.Tensor
    fp8_k_global_scale: torch.Tensor
    fp8_v_global_scale: torch.Tensor
    fp4_k_global_scale: torch.Tensor
    fp4_v_global_scale: torch.Tensor


def _check_input(x: torch.Tensor, block_size: int) -> None:
    if not x.is_floating_point():
        raise TypeError("KV-cache input must be floating point")
    if x.shape[-1] % block_size:
        raise ValueError(f"head dimension must be divisible by {block_size}")


@torch.no_grad()
def quantize_block_scaled_fp8(
    x: torch.Tensor,
    *,
    block_size: int = 16,
    optimize_scales: bool = True,
    tail_weight: float = 0.05,
    rows_per_chunk: int = 65536,
) -> BlockScaledFP8:
    """Quantize KV data to the XQA accuracy-first block-scaled FP8 format.

    ``tail_weight`` adds the squared maximum element error in each block to
    the ordinary mean-squared objective.  Scale search changes only page-seal
    cost; payload rate and the attention read kernel are unchanged.
    """

    if block_size != 16:
        raise ValueError("the current XQA transport kernel requires block_size=16")
    if tail_weight < 0:
        raise ValueError("tail_weight must be non-negative")
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    _check_input(x, block_size)

    head_dim = x.shape[-1]
    blocks_per_row = head_dim // block_size
    flat = x.reshape(-1, head_dim)
    tensor_amax = torch.zeros((), dtype=torch.float32, device=x.device)
    for row_begin in range(0, flat.shape[0], rows_per_chunk):
        chunk = flat[row_begin : row_begin + rows_per_chunk]
        tensor_amax = torch.maximum(tensor_amax, chunk.abs().amax().float())
    tensor_amax = tensor_amax.clamp_min(torch.finfo(torch.float32).tiny)
    global_scale = tensor_amax / (FP8_E4M3_MAX * FP8_E4M3_A16_SCALE_MAX)
    payload = torch.empty(flat.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty(
        (flat.shape[0], blocks_per_row), dtype=torch.uint8, device=x.device
    )
    factors = torch.tensor(
        (0.75, 0.8125, 0.875, 0.9375, 1.0, 1.0625, 1.125, 1.25, 1.5),
        dtype=torch.float32,
        device=x.device,
    )
    # Candidate search has an extra candidate axis; cap its chunk to keep page
    # sealing bounded even when this reference helper is run on a full pool.
    encode_chunk_rows = min(rows_per_chunk, 4096) if optimize_scales else rows_per_chunk
    for row_begin in range(0, flat.shape[0], encode_chunk_rows):
        row_end = min(row_begin + encode_chunk_rows, flat.shape[0])
        blocks = flat[row_begin:row_end].float().reshape(-1, blocks_per_row, block_size)
        block_amax = blocks.abs().amax(dim=-1)
        required_sf = block_amax / (global_scale * FP8_E4M3_MAX)

        if optimize_scales:
            # Search both sides of max-normalization. Slightly smaller scales
            # may clip one outlier but improve the other 15 values; larger
            # scales avoid clipping after E4M3 scale rounding.
            sf_candidates = (required_sf.unsqueeze(-1) * factors).clamp(
                FP8_E4M3_MIN_SUBNORMAL, FP8_E4M3_A16_SCALE_MAX
            )
            sf_candidates = sf_candidates.to(torch.float8_e4m3fn).float()
            denominators = global_scale * sf_candidates
            payload_candidates = (
                blocks.unsqueeze(-2) / denominators.unsqueeze(-1)
            ).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
            payload_candidates = payload_candidates.to(torch.float8_e4m3fn)
            residual = (
                payload_candidates.float() * denominators.unsqueeze(-1)
                - blocks.unsqueeze(-2)
            ).abs()
            objective = residual.square().mean(dim=-1)
            if tail_weight:
                objective = objective + tail_weight * residual.amax(dim=-1).square()
            selected = objective.argmin(dim=-1, keepdim=True)
            scales_f32 = sf_candidates.gather(-1, selected).squeeze(-1)
            # CPU PyTorch does not implement gather for float8, but raw-byte
            # selection is exactly equivalent and is also cheaper.
            payload_chunk = (
                payload_candidates.contiguous()
                .view(torch.uint8)
                .gather(
                    -2,
                    selected.unsqueeze(-1).expand(*selected.shape, block_size),
                )
                .squeeze(-2)
                .view(torch.float8_e4m3fn)
            )
        else:
            scales_f32 = (
                required_sf.clamp(FP8_E4M3_MIN_SUBNORMAL, FP8_E4M3_A16_SCALE_MAX)
                .to(torch.float8_e4m3fn)
                .float()
            )
            payload_chunk = (
                (blocks / (global_scale * scales_f32).unsqueeze(-1))
                .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
                .to(torch.float8_e4m3fn)
            )

        payload[row_begin:row_end].copy_(payload_chunk.reshape(-1, head_dim))
        scales[row_begin:row_end].copy_(
            scales_f32.to(torch.float8_e4m3fn).contiguous().view(torch.uint8)
        )

    return BlockScaledFP8(
        payload.reshape_as(x),
        scales.reshape(*x.shape[:-1], blocks_per_row),
        global_scale,
    )


@torch.no_grad()
def quantize_block_scaled_fp8_cuda(
    x: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    payload_out: torch.Tensor | None = None,
    scales_out: torch.Tensor | None = None,
) -> BlockScaledFP8:
    """Fast direct-scale CUDA page-seal kernel for XQA block-scaled FP8.

    ``x`` may have arbitrary leading dimensions and a head dimension divisible
    by 16. ``global_scale`` is the dequantization scale and must be a CUDA
    float32 scalar tensor. Accuracy-optimized candidate search remains in
    :func:`quantize_block_scaled_fp8`.
    """

    if not x.is_cuda:
        raise ValueError("the fused page-seal kernel requires a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError("the fused page-seal kernel requires contiguous input")
    _check_input(x, 16)
    if global_scale.dtype != torch.float32 or global_scale.numel() != 1:
        raise TypeError("global_scale must be a float32 scalar tensor")
    if global_scale.device != x.device:
        raise ValueError("global_scale must be on the same CUDA device as x")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("the fused page-seal kernel accepts float16 or bfloat16")

    from .fp4_quantization import get_fp4_kv_quantization_module

    head_dim = x.shape[-1]
    flat = x.reshape(-1, head_dim)
    if payload_out is None:
        payload_out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    if payload_out.shape != x.shape or payload_out.dtype != torch.float8_e4m3fn:
        raise ValueError("payload_out must match x.shape and use float8_e4m3fn")
    expected_scale_shape = (*x.shape[:-1], head_dim // 16)
    if scales_out is None:
        scales_out = torch.empty(
            expected_scale_shape, dtype=torch.uint8, device=x.device
        )
    if scales_out.shape != expected_scale_shape or scales_out.dtype != torch.uint8:
        raise ValueError("scales_out must match x[..., ::16] and use uint8")
    if payload_out.device != x.device or scales_out.device != x.device:
        raise ValueError("output tensors must be on the same CUDA device as x")
    if not payload_out.is_contiguous() or not scales_out.is_contiguous():
        raise ValueError("output tensors must be contiguous")
    payload_storage = payload_out.reshape(-1, head_dim).view(torch.uint8)
    scales = scales_out.reshape(flat.shape[0], head_dim // 16)
    get_fp4_kv_quantization_module().bsfp8_kv_quant(
        flat, global_scale.reshape(1), payload_storage, scales
    )
    return BlockScaledFP8(
        payload_out,
        scales_out,
        global_scale.reshape(()),
    )


@torch.no_grad()
def seal_mixed_kv_pages_cuda(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    reused_pages: torch.Tensor,
    reused_count: torch.Tensor,
    completed_pages: torch.Tensor,
    completed_count: torch.Tensor,
    cache: MixedKVPagedCache,
) -> None:
    """Route and seal completed PagedAttention pages as FP4, FP8, or A16.

    ``k_cache`` and ``v_cache`` use NHD page interiors. ``reused_pages`` and
    ``completed_pages`` are fixed-capacity explicit page-index operands; their
    device scalar counts delimit valid entries. Reused pages are demoted to A16
    before completed pages are routed from their page-local spatial/coherence
    signature. The chosen compressed tier alone is encoded; A16 pages retain
    the canonical cache and write no sidecar payload. Both transitions are
    stream ordered and require no host synchronization.
    """

    if not k_cache.is_cuda or not v_cache.is_cuda:
        raise ValueError("page sealing requires CUDA caches")
    if k_cache.shape != v_cache.shape or k_cache.ndim != 4:
        raise ValueError("K and V caches must have matching [page, token, head, dim]")
    if k_cache.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("A16 caches must use float16 or bfloat16")
    for name, pages in (
        ("reused_pages", reused_pages),
        ("completed_pages", completed_pages),
    ):
        if pages.dtype != torch.int32 or pages.ndim != 1:
            raise TypeError(f"{name} must be a 1D int32 tensor")
    for name, count in (
        ("reused_count", reused_count),
        ("completed_count", completed_count),
    ):
        if count.dtype != torch.int32 or count.numel() != 1:
            raise TypeError(f"{name} must be a scalar int32 tensor")
    expected_scales = (*k_cache.shape[:-1], k_cache.shape[-1] // 16)
    if (
        cache.fp8_k_payload.shape != k_cache.shape
        or cache.fp8_v_payload.shape != v_cache.shape
    ):
        raise ValueError("BSFP8 payload pools must match the A16 cache shapes")
    if (
        cache.fp8_k_scales.shape != expected_scales
        or cache.fp8_v_scales.shape != expected_scales
        or cache.fp4_k_scales.shape != expected_scales
        or cache.fp4_v_scales.shape != expected_scales
    ):
        raise ValueError("block-scale pools must have head_dim / 16 scale bytes")
    expected_fp4 = (*k_cache.shape[:-1], k_cache.shape[-1] // 2)
    if (
        cache.fp4_k_payload.shape != expected_fp4
        or cache.fp4_v_payload.shape != expected_fp4
    ):
        raise ValueError("BSFP4 payload pools must pack two coefficients per byte")
    if cache.page_format.shape != (k_cache.shape[0],):
        raise ValueError("page_format must have one byte per physical page")
    if cache.page_router_stats.shape != (k_cache.shape[0], 2):
        raise ValueError("page_router_stats must have shape [num_pages, 2]")
    if cache.page_router_stats.dtype != torch.float32:
        raise TypeError("page_router_stats must use float32")
    if cache.routing_thresholds.shape != (4,):
        raise ValueError("routing_thresholds must contain FP4 and FP8 signature limits")
    if cache.routing_thresholds.dtype != torch.float32:
        raise TypeError("routing_thresholds must use float32")
    if (
        cache.page_router_stats.device != k_cache.device
        or cache.routing_thresholds.device != k_cache.device
    ):
        raise ValueError("routing state must be on the same CUDA device as the cache")

    from .fp4_quantization import get_fp4_kv_quantization_module

    get_fp4_kv_quantization_module().mixed_kv_quant_pages(
        k_cache,
        v_cache,
        reused_pages,
        reused_count,
        completed_pages,
        completed_count,
        cache.fp8_k_global_scale.reshape(1),
        cache.fp8_v_global_scale.reshape(1),
        cache.fp4_k_global_scale.reshape(1),
        cache.fp4_v_global_scale.reshape(1),
        cache.fp8_k_payload,
        cache.fp8_v_payload,
        cache.fp8_k_scales,
        cache.fp8_v_scales,
        cache.fp4_k_payload,
        cache.fp4_v_payload,
        cache.fp4_k_scales,
        cache.fp4_v_scales,
        cache.page_format,
        cache.page_router_stats,
        cache.routing_thresholds,
    )


@torch.no_grad()
def dequantize_block_scaled_fp8(
    payload: torch.Tensor,
    scales: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    block_size: int = 16,
) -> torch.Tensor:
    """Reference dequantization matching XQA's register dequantization."""

    if payload.dtype != torch.float8_e4m3fn:
        raise TypeError("payload must use torch.float8_e4m3fn")
    if scales.dtype != torch.uint8:
        raise TypeError("scales must contain raw E4M3 bytes in torch.uint8 storage")
    _check_input(payload, block_size)
    expected_shape = (*payload.shape[:-1], payload.shape[-1] // block_size)
    if scales.shape != expected_shape:
        raise ValueError(f"scale shape must be {expected_shape}, got {scales.shape}")
    blocks = payload.float().reshape(*expected_shape, block_size)
    scales_f32 = scales.contiguous().view(torch.float8_e4m3fn).float()
    return (blocks * (global_scale * scales_f32).unsqueeze(-1)).reshape(payload.shape)


@torch.no_grad()
def quantize_mxfp8_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear-layout OCP MXFP8 reference: E4M3 plus UE8M0 per 32 values."""

    block_size = 32
    _check_input(x, block_size)
    blocks = x.float().reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    amax = blocks.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    scales = torch.pow(2.0, torch.ceil(torch.log2(amax / FP8_E4M3_MAX)))
    payload = (
        (blocks / scales.unsqueeze(-1))
        .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
    )
    ue8m0 = (scales.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    return payload.reshape_as(x), ue8m0


@torch.no_grad()
def dequantize_mxfp8_reference(
    payload: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Dequantize linear-layout OCP MXFP8 to float32."""

    block_size = 32
    _check_input(payload, block_size)
    expected_shape = (*payload.shape[:-1], payload.shape[-1] // block_size)
    if scales.shape != expected_shape:
        raise ValueError(f"scale shape must be {expected_shape}, got {scales.shape}")
    decoded = torch.pow(
        torch.tensor(2.0, device=scales.device), scales.to(torch.int64).sub(127).float()
    )
    return (
        payload.float().reshape(*expected_shape, block_size) * decoded.unsqueeze(-1)
    ).reshape(payload.shape)


@torch.no_grad()
def quantize_tensor_fp8_reference(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor E4M3 baseline used by the existing XQA FP8 cache path."""

    scale = x.float().abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
    scale = scale / FP8_E4M3_MAX
    payload = (
        (x.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    )
    return payload, scale
