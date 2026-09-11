"""Reference codecs for block-scaled FP8 KV-cache transport.

The encode path is intentionally accuracy first.  It is used while sealing a
paged-attention page; the latency-critical decode path lives in XQA and only
loads an E4M3 payload byte and one amortized E4M3 scale byte per 16 values.
"""

from __future__ import annotations

from enum import IntEnum
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
    publishes ``page_format`` last. Payload views may start at their A16 row's
    address; row strides are retained and scales must not overlap source rows.
    The two compressed formats may share payload and scale storage.

    ``page_router_stats`` stores neighbor cosine and block peak/RMS. The scale
    search evaluates reconstruction error in registers without materializing
    alternate page encodings.
    """

    fp8_k_payload: torch.Tensor | None
    fp8_v_payload: torch.Tensor | None
    fp8_k_scales: torch.Tensor | None
    fp8_v_scales: torch.Tensor | None
    fp4_k_payload: torch.Tensor | None
    fp4_v_payload: torch.Tensor | None
    fp4_k_scales: torch.Tensor | None
    fp4_v_scales: torch.Tensor | None
    page_format: torch.Tensor | None
    page_router_stats: torch.Tensor
    routing_thresholds: torch.Tensor
    fp8_k_global_scale: torch.Tensor
    fp8_v_global_scale: torch.Tensor
    fp4_k_global_scale: torch.Tensor
    fp4_v_global_scale: torch.Tensor

    page_storage: torch.Tensor | None = None
    page_addresses: torch.Tensor | None = None
    page_geometry: tuple[int, int, int] | None = None
    native_mma: bool = False


class MixedKVPageUpdatePhase(IntEnum):
    """Host-realized scheduling phases of the arena update ABI."""

    PLACE = 1
    SEAL = 2
    PLACE_AND_SEAL = 3


class MixedKVPageArena:
    """One byte allocation with format-tagged addresses and reusable size classes.

    ``update`` accepts a ``MixedKVPageUpdatePhase`` as its final operand.
    PLACE prepares writable A16 pages and scatters the new values. SEAL owns
    each completed page in one CTA, using the original routing metric and
    codec, and publishes the selected encoding after its writes finish.
    Consumers must finish reading before SEAL can retire the old encoding;
    later readers and arena reuse must depend on its stream completion.
    """

    def __init__(
        self,
        capacity_bytes: int,
        num_blocks: int,
        pages_per_block: int,
        page_values: list[int],
        device: torch.device,
    ) -> None:
        from .fp4_quantization import get_fp4_kv_quantization_module

        if not page_values or any(n <= 0 or n % 16 for n in page_values):
            raise ValueError("Page coefficient counts must be positive multiples of 16")
        self.classes = sorted(
            {(extent, n) for n in page_values for extent in self.page_extents(n)}
        )
        self.extents = [extent for extent, _ in self.classes]
        self.page_values = sorted(set(page_values))
        self.a16_bytes = torch.tensor(
            [self.page_extents(n)[0] for n in self.page_values],
            dtype=torch.int32,
            device=device,
        )
        self.a16_classes = torch.tensor(
            [self.page_values.index(n) for _, n in self.classes],
            dtype=torch.int32,
            device=device,
        )
        self.slab_bytes = max(1 << 20, 128 * ((max(self.extents) + 127) // 128))
        num_slabs = capacity_bytes // self.slab_bytes
        if num_slabs == 0 or num_blocks <= 0 or pages_per_block <= 0:
            raise ValueError(
                "The arena must contain at least one slab and logical page"
            )
        self.data = torch.empty(capacity_bytes, dtype=torch.uint8, device=device)
        self.pages = torch.full(
            (num_blocks, pages_per_block), -1, dtype=torch.int64, device=device
        )
        self.slabs = torch.zeros((num_slabs, 4), dtype=torch.int32, device=device)
        bitmap_words = (self.slab_bytes // min(self.extents) + 63) // 64
        self.occupied = torch.zeros(
            (num_slabs, bitmap_words), dtype=torch.int64, device=device
        )
        self.available = torch.zeros(
            (len(self.extents) + 1, (num_slabs + 63) // 64),
            dtype=torch.int64,
            device=device,
        )
        self.available[-1].fill_(-1)
        if num_slabs % 64:
            self.available[-1, -1] = (1 << (num_slabs % 64)) - 1
        self.hints = torch.zeros(
            len(self.extents) + 1, dtype=torch.int32, device=device
        )
        self.counters = torch.zeros(
            4 + len(self.page_values), dtype=torch.int64, device=device
        )
        self.reservations = torch.zeros(num_blocks, dtype=torch.int32, device=device)
        self.capacity_snapshot = torch.empty(
            6 + 2 * len(self.page_values), dtype=torch.int64, device=device
        )
        module = get_fp4_kv_quantization_module()
        self.update = module.mixed_kv_arena_update
        self._blocks = module.mixed_kv_arena_blocks
        self._capacity = module.mixed_kv_arena_capacity
        self._block_operands = (
            self.data,
            self.pages,
            self.slabs,
            self.occupied,
            self.available,
            self.hints,
            self.counters,
            self.reservations,
            self.a16_classes,
            self.slab_bytes,
        )

    @staticmethod
    def page_extents(values: int) -> tuple[int, int, int]:
        """A16, block-scaled FP8 and FP4 extents, including alignment."""
        a16, fp8, fp4 = (
            ((size + 127) // 128) * 128
            for size in (values * 2, values + values // 16, values // 2 + values // 16)
        )
        return a16, fp8, fp4

    def size_classes(self, values: int) -> torch.Tensor:
        return torch.tensor(
            [self.classes.index((size, values)) for size in self.page_extents(values)],
            dtype=torch.int32,
            device=self.data.device,
        )

    def reset_blocks(self, blocks: torch.Tensor) -> None:
        """Release every old layer page before a logical block is reassigned."""
        self._blocks(*self._block_operands, blocks, None, True)

    def release_blocks(self, blocks: torch.Tensor) -> None:
        """Release unreferenced storage and its unwritten-page reservation."""
        self._blocks(*self._block_operands, blocks, None, False)

    def capacity(self) -> torch.Tensor:
        """Snapshot allocator capacity after the current stream's publications."""
        self._capacity(*self._block_operands, self.a16_bytes, self.capacity_snapshot)
        return self.capacity_snapshot

    def copy_blocks(
        self, sources: torch.Tensor, destinations: torch.Tensor, num_blocks: int
    ) -> None:
        """Copy committed encodings; subsequent writes promote private pages."""
        self._blocks(*self._block_operands, destinations, sources, True)


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
    *,
    page_router_partials: torch.Tensor | None = None,
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

    partial_shape = (completed_pages.numel(), *k_cache.shape[1:3], 4)
    if page_router_partials is None:
        page_router_partials = torch.empty(
            partial_shape, dtype=torch.float32, device=k_cache.device
        )
    if (
        page_router_partials.shape != partial_shape
        or page_router_partials.dtype != torch.float32
        or page_router_partials.device != k_cache.device
    ):
        raise ValueError(
            "page_router_partials must match the completed-page row geometry"
        )

    get_fp4_kv_quantization_module().mixed_kv_quant_pages(
        k_cache,
        v_cache,
        reused_pages,
        reused_count,
        completed_pages,
        completed_count,
        cache.fp8_k_global_scale,
        cache.fp8_v_global_scale,
        cache.fp4_k_global_scale,
        cache.fp4_v_global_scale,
        cache.fp8_k_payload,
        cache.fp8_v_payload,
        cache.fp8_k_scales,
        cache.fp8_v_scales,
        cache.fp4_k_payload,
        cache.fp4_v_payload,
        cache.fp4_k_scales,
        cache.fp4_v_scales,
        page_router_partials,
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
