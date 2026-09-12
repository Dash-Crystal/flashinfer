# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""CUTLASS matrix multiplication with caller-owned row visibility."""

import logging
from functools import cache
from typing import Any, NamedTuple

import torch

from ..jit.gemm.core import gen_masked_gemm_module
from ..utils import get_compute_capability


@cache
def get_masked_gemm_module(*, ready_tma: bool = False, math_registers: int = 0):
    return gen_masked_gemm_module(
        ready_tma=ready_tma, math_registers=math_registers
    ).build_and_load()


class MaskedGemmInfo(NamedTuple):
    module: Any
    resources: tuple[int, int, int]
    row_tile: int


@cache
def masked_gemm_info(
    device: int, dtype: torch.dtype, *, publish: bool, tiles: bool
) -> MaskedGemmInfo:
    """Return compiled resources and geometry, independent of publication groups."""
    module = get_masked_gemm_module()
    registers, shared, threads, rows = module.prepare(
        device, dtype == torch.bfloat16, publish, tiles
    )
    return MaskedGemmInfo(module, (registers, shared, threads), rows)


def _validate_mm_operands(x, weight, out):
    if (
        x.dtype not in (torch.float16, torch.bfloat16)
        or weight.dtype != x.dtype
        or out.dtype != x.dtype
        or any(t.device != x.device for t in (weight, out))
        or x.device.type != "cuda"
        or any(t.ndim != 2 for t in (x, weight, out))
        or x.shape[1] != weight.shape[0]
        or out.shape != (x.shape[0], weight.shape[1])
        or x.stride(1) != 1
        or weight.stride(0) != 1
        or out.stride(1) != 1
        or x.stride(0) < x.shape[1]
        or weight.stride(1) < weight.shape[0]
        or out.stride(0) < out.shape[1]
        or any(
            s % 8
            for s in (
                x.shape[1],
                weight.shape[1],
                x.stride(0),
                weight.stride(1),
                out.stride(0),
            )
        )
        or any(t.data_ptr() % 16 for t in (x, weight, out))
        or any(d >= 2**31 for t in (x, weight, out) for d in (*t.shape, *t.stride()))
    ):
        raise ValueError("Expected aligned CUDA TN GEMM operands")


def mm_masked_tiles(
    x,
    weight,
    out,
    is_padding,
    *,
    row_offset=0,
    peer_output=None,
    publication=None,
    peer_publication=None,
):
    """Write GEMM tiles containing visible rows; wholly padded tiles stay untouched.

    The caller must consume outputs with the same row mask. Arbitrary holes in
    the mask are supported; partially visible tiles use ordinary GEMM arithmetic.
    Operands are A row-major, B column-major, and output row-major, aligned to
    eight FP16/BF16 values. An optional peer mapping receives the same epilogue
    stores. Its owner must publish kernel completion to peer readers with system
    synchronization, and order reuse after the readers finish. Optional local
    and peer publication arrays have four rows: local-ready, peer-ready,
    fragment completion, and consumed rows, with one column per 32-row group.
    Compute tiles retain the ordinary GEMM geometry; only the first subgroup
    counts N-fragment completion, then all its row groups are published.
    The consumer acquires both flags and the last reader resets its local flags
    and consumed-row counter. Both ranks join consumption before slot reuse.
    No preparation tensor or launch is introduced.
    """
    _validate_mm_operands(x, weight, out)
    if (
        is_padding.device != x.device
        or is_padding.dtype != torch.bool
        or is_padding.ndim != 1
        or not is_padding.is_contiguous()
        or row_offset < 0
        or row_offset + x.shape[0] > is_padding.numel()
    ):
        raise ValueError("Padding must cover the logical GEMM rows")
    if peer_output is not None and (
        peer_output.device != x.device
        or peer_output.dtype != out.dtype
        or peer_output.shape != out.shape
        or peer_output.stride() != out.stride()
        or peer_output.data_ptr() % 16
        or torch._C._overlaps(peer_output, out)
    ):
        raise ValueError(
            "Peer output must be a distinct mapping with the output layout"
        )
    if publication is not None or peer_publication is not None:
        if (
            peer_output is None
            or any(
                p is None
                or p.device != x.device
                or p.dtype != torch.int32
                or p.ndim != 2
                or p.shape[0] != 4
                or p.shape[1] < (x.shape[0] + 31) // 32
                or not p.is_contiguous()
                for p in (publication, peer_publication)
            )
            or publication.shape != peer_publication.shape
        ):
            raise ValueError("Tile publication requires matching local/peer counters")
    if x.shape[0] and weight.shape[1]:
        masked_gemm_info(
            x.device.index,
            x.dtype,
            publish=peer_output is not None,
            tiles=publication is not None,
        ).module.run(
            x,
            weight,
            out,
            is_padding,
            row_offset,
            peer_output,
            publication,
            peer_publication,
        )
    return out


class ReadyGemmInfo(NamedTuple):
    module: Any
    sms: int
    occupancy: int
    row_tile: int
    workspace_bytes: int
    concurrent_producer: bool


@cache
def ready_gemm_info(
    device: int,
    dtype: torch.dtype,
    producers: tuple[tuple[int, int, int], ...] = (),
) -> ReadyGemmInfo:
    ready_tma = get_compute_capability(torch.device("cuda", device))[0] == 12
    module = get_masked_gemm_module(ready_tma=ready_tma)
    args = (device, dtype == torch.bfloat16, producers)
    sms, occupancy, row_tile, workspace_bytes, budget, concurrent = (
        module.prepare_ready(*args)
    )
    if ready_tma and budget:
        module = get_masked_gemm_module(ready_tma=True, math_registers=budget)
        sms, occupancy, row_tile, workspace_bytes, _, concurrent = module.prepare_ready(
            *args
        )
        logging.getLogger(__name__).info(
            "Ready TMA resource plan: producers=%s, row_tile=%d, "
            "math_registers=%d, shared_sm=%s",
            producers,
            row_tile,
            budget,
            bool(concurrent),
        )
    return ReadyGemmInfo(
        module, sms, occupancy, row_tile, workspace_bytes, bool(concurrent)
    )


def create_ready_workspace(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Allocate the selected kernel's workspace before model graph capture."""
    info = ready_gemm_info(device.index, dtype)
    return torch.zeros(info.workspace_bytes, dtype=torch.uint8, device=device)


def mm_ready_rows(
    x,
    weight,
    out,
    readiness,
    *,
    group_rows,
    reserved_blocks,
    workspace,
    producers=(),
):
    """Multiply once, consuming row groups published by a concurrent producer.

    Operands follow ``mm_masked_tiles``'s layout contract. The caller initializes
    the int32 workspace to zero before capture. Each producer row contributes
    one device-release increment to its group, after storing every column.
    Padded rows must also be stored and counted. The final workspace element is
    reserved for consumer completion; the last GEMM CTA resets the workspace.
    The caller joins both kernels before reusing any operand or workspace.

    ``producers`` lists compiled registers/thread, shared bytes and threads.
    SM120 budgets registers for all producers, then checks their joint allocation
    with CUDA's occupancy calculator. Fitting kernels share SMs; otherwise
    ``reserved_blocks`` leaves producer SMs available.
    The resource plan reserves execution capacity; the caller also owns stream
    dependencies and producer progress. SM120 uses the native TMA pipeline with
    separate loading and compute warps. Other architectures use CUTLASS Stream-K.
    The resource plan is cached per compiled producer before graph replay.
    Communication groups do not select the GEMM's compute tile. SM120 traverses
    N tiles within a row tile before advancing to later published rows.
    ``workspace`` is allocated and zeroed once with ``create_ready_workspace``;
    Stream-K partials and barriers occupy fixed, disjoint regions across shapes.
    The TMA path needs zero scratch bytes. Execution adds no allocation or
    reset kernel.
    """
    _validate_mm_operands(x, weight, out)
    if (
        readiness.device != x.device
        or readiness.dtype != torch.int32
        or readiness.ndim != 1
        or not readiness.is_contiguous()
        or group_rows <= 0
        or group_rows >= 2**31
        or readiness.numel() < (x.shape[0] + group_rows - 1) // group_rows + 1
        or any(torch._C._overlaps(readiness, t) for t in (x, weight, out))
    ):
        raise ValueError("Readiness requires one counter per row group and completion")
    info = ready_gemm_info(x.device.index, x.dtype, producers)
    if (
        workspace.device != x.device
        or workspace.dtype != torch.uint8
        or workspace.ndim != 1
        or not workspace.is_contiguous()
        or workspace.data_ptr() % 128
        or workspace.numel() < info.workspace_bytes
        or any(torch._C._overlaps(workspace, t) for t in (x, weight, out, readiness))
        or not 0 < reserved_blocks < info.sms
    ):
        raise ValueError("Ready GEMM requires distinct workspace and producer capacity")
    if x.shape[0] and weight.shape[1]:
        info.module.run_ready(
            x,
            weight,
            out,
            readiness,
            workspace,
            group_rows,
            info.sms,
            info.occupancy,
            0 if info.concurrent_producer else reserved_blocks,
        )
    return out
