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
def get_masked_gemm_module(
    *,
    ready_tma: bool = False,
    math_registers: int = 0,
    row_tile: int = 128,
    startup_ramp: bool = False,
):
    return gen_masked_gemm_module(
        ready_tma=ready_tma,
        math_registers=math_registers,
        row_tile=row_tile,
        startup_ramp=startup_ramp,
    ).build_and_load()


class MaskedGemmInfo(NamedTuple):
    module: Any
    resources: tuple[int, int, int]
    row_tile: int
    sms: int
    occupancy: int
    workspace_bytes: int


@cache
def masked_gemm_info(
    device: int,
    dtype: torch.dtype,
    *,
    publish: bool,
    tiles: bool,
    stream_k: bool = False,
    persistent_dp: bool = False,
) -> MaskedGemmInfo:
    """Return compiled resources and geometry, independent of publication groups."""
    if persistent_dp and (stream_k or not tiles):
        raise ValueError("Persistent DP requires tile publication without Stream-K")
    module = get_masked_gemm_module()
    registers, shared, threads, rows, sms, occupancy, workspace_bytes = module.prepare(
        device, dtype == torch.bfloat16, publish, tiles, stream_k, persistent_dp
    )
    return MaskedGemmInfo(
        module, (registers, shared, threads), rows, sms, occupancy, workspace_bytes
    )


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
    workspace=None,
    stream_k=False,
    persistent_dp=False,
    native_region: int = 0,
    task_state: torch.Tensor | None = None,
    retire_readiness: torch.Tensor | None = None,
    retire_group_rows: int = 128,
    retire_rows: int = 0,
    retire_workers: int = 0,
    publication_epoch: int = 0,
):
    """Compute visible GEMM tiles and publish complete reduced output rows.

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
    The prepared schedule selects direct output tiles or Stream-K. Stream-K
    uses caller-owned FP32 partial workspace. Both share MMA and publication;
    wholly padded tiles skip MMA but still retire contributors and publish zeros.
    Workspace is initialized once and reused only after this launch completes.
    Supplying only ``publication`` selects local-only publication: the same
    full GEMM tiles store solely to ``out`` and publish row-zero flags with
    system release ordering after every N tile has finished. Rows one and three
    are untouched; fragment counters in row two reset as usual. The caller
    resets ready flags only after copies and readers finish, before slot reuse.
    ``persistent_dp=True`` uses a bounded resident producer grid with full
    output tiles and no K splitting. It requires publication counters and is
    mutually exclusive with ``stream_k``. Use the matching prepared workspace.
    Optional three-word zeroed int32 ``task_state`` gives persistent DP workers
    a shared full-tile claim pool. ``retire_workers`` workers finish their tile
    then release their slots once the first ``retire_rows`` normalized rows are
    published in ``retire_readiness``. At least one worker must remain. A latch
    survives consumer readiness resets; the last producer CTA clears all state.
    The caller must prove the finite readiness publisher can co-reside with A.
    ``native_region`` optionally points to a device tp_region::DeviceRegion whose
    registered source is exactly ``out``. Its groups cover contiguous 128-row
    BF16 stripes. The final column tile enqueues one copy and signal, including
    the actual partial-tail byte count. Keep its registration and proxy alive
    until consumption finishes; only the consumer advances signal epochs.
    """
    _validate_mm_operands(x, weight, out)
    if publication_epoch and (
        publication_epoch < 0 or peer_output is None or not persistent_dp
    ):
        raise ValueError("Epoch publication requires a persistent peer producer")
    if native_region and (
        native_region < 0
        or x.dtype != torch.bfloat16
        or publication is None
        or peer_output is not None
        or peer_publication is not None
        or not persistent_dp
        or not out.is_contiguous()
    ):
        raise ValueError(
            "Native region requires contiguous BF16 local persistent publication"
        )
    if task_state is not None:
        if (
            not persistent_dp
            or publication is None
            or task_state.device != x.device
            or task_state.dtype != torch.int32
            or task_state.shape != (3,)
            or not task_state.is_contiguous()
            or retire_workers < 0
            or any(
                torch._C._overlaps(task_state, t)
                for t in (x, weight, out, publication, workspace)
                if t is not None
            )
        ):
            raise ValueError(
                "Claimed producer requires private three-word state and persistent DP"
            )
    elif retire_workers or retire_readiness is not None:
        raise ValueError("Producer retirement requires claimed task state")
    if retire_workers and (
        retire_readiness is None
        or retire_readiness.device != x.device
        or retire_readiness.dtype != torch.int32
        or retire_readiness.ndim != 1
        or not retire_readiness.is_contiguous()
        or retire_group_rows <= 0
        or not 0 < retire_rows <= x.shape[0]
        or retire_readiness.numel()
        < (retire_rows + retire_group_rows - 1) // retire_group_rows
        or torch._C._overlaps(task_state, retire_readiness)
    ):
        raise ValueError("Producer retirement requires a valid counter prefix")
    if persistent_dp and (stream_k or publication is None):
        raise ValueError("Persistent DP requires tile publication without Stream-K")
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
        local_only = peer_output is None and peer_publication is None
        counters = (publication,) if local_only else (publication, peer_publication)
        if (
            (not local_only and peer_output is None)
            or any(
                p is None
                or p.device != x.device
                or p.dtype != torch.int32
                or p.ndim != 2
                or p.shape[0] != 4
                or p.shape[1] < (x.shape[0] + 31) // 32
                or not p.is_contiguous()
                for p in counters
            )
            or (not local_only and publication.shape != peer_publication.shape)
        ):
            raise ValueError(
                "Tile publication requires valid local or matching peer counters"
            )
    if x.shape[0] and weight.shape[1]:
        info = masked_gemm_info(
            x.device.index,
            x.dtype,
            publish=peer_output is not None,
            tiles=publication is not None,
            stream_k=stream_k,
            persistent_dp=persistent_dp,
        )
        if info.workspace_bytes and (
            workspace is None
            or workspace.device != x.device
            or workspace.dtype != torch.uint8
            or workspace.ndim != 1
            or not workspace.is_contiguous()
            or workspace.data_ptr() % 128
            or workspace.numel() < info.workspace_bytes
        ):
            raise ValueError("Published GEMM requires its prepared partial workspace")
        info.module.run(
            x,
            weight,
            out,
            is_padding,
            row_offset,
            peer_output,
            publication,
            peer_publication,
            workspace,
            info.sms,
            info.occupancy,
            stream_k,
            persistent_dp,
            native_region,
            task_state,
            retire_readiness,
            retire_group_rows,
            retire_rows,
            retire_workers,
            publication_epoch,
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
    *,
    row_tile: int = 128,
    startup_ramp: bool = False,
) -> ReadyGemmInfo:
    ready_tma = get_compute_capability(torch.device("cuda", device))[0] == 12
    if startup_ramp and (not ready_tma or row_tile != 128):
        raise ValueError("Geometric startup requires SM120 and a 128-row steady tile")
    module = get_masked_gemm_module(
        ready_tma=ready_tma, row_tile=row_tile, startup_ramp=startup_ramp
    )
    args = (device, dtype == torch.bfloat16, producers)
    sms, occupancy, row_tile, workspace_bytes, budget, concurrent = (
        module.prepare_ready(*args)
    )
    if ready_tma and budget:
        module = get_masked_gemm_module(
            ready_tma=True,
            math_registers=budget,
            row_tile=row_tile,
            startup_ramp=startup_ramp,
        )
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


def create_ready_task_state(rows: int, columns: int, device, *, row_tile: int = 128):
    """Create replayable eligible-task state before capture; one owner per launch."""
    if rows < 0 or columns <= 0 or row_tile not in (32, 64, 96, 128):
        raise ValueError("Expected nonnegative rows and a supported full tile shape")
    return torch.zeros(
        ((rows + row_tile - 1) // row_tile) * (((columns + 63) // 64) + 1),
        dtype=torch.int32,
        device=device,
    )


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
    row_tile=128,
    startup_ramp=False,
    tile_swizzle=1,
    task_state: torch.Tensor | None = None,
):
    """Multiply once, consuming row groups published by a concurrent producer.

    Optional ``task_state`` comes from ``create_ready_task_state`` with the same
    shape and row tile. It maps scheduler tickets to currently eligible full
    tiles, preserving a single shared coordinate choice across loader and math
    warpgroups. The final CTA resets it for replay. Concurrent producers must
    still progress when no input tile is ready. Startup ramp and swizzle are
    excluded from this path.

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
    ``startup_ramp`` consumes the first 32 rows, the next 64 rows, then the
    remaining rows with 128-row tiles in one persistent SM120 launch. Readiness
    groups retain global row coordinates, including partial final groups.
    ``tile_swizzle`` groups neighboring SM120 output tiles for weight reuse;
    it does not alter compute geometry or readiness publication.
    """
    _validate_mm_operands(x, weight, out)
    if tile_swizzle not in (1, 2, 4, 8):
        raise ValueError("Tile swizzle must be 1, 2, 4 or 8")
    if tile_swizzle != 1 and get_compute_capability(x.device)[0] != 12:
        raise ValueError("Ready GEMM tile swizzle requires SM120")
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
    info = ready_gemm_info(
        x.device.index,
        x.dtype,
        producers,
        row_tile=row_tile,
        startup_ramp=startup_ramp,
    )
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
    if task_state is not None and (
        startup_ramp
        or tile_swizzle != 1
        or task_state.device != x.device
        or task_state.dtype != torch.int32
        or task_state.ndim != 1
        or not task_state.is_contiguous()
        or task_state.numel()
        != ((x.shape[0] + row_tile - 1) // row_tile)
        * (((weight.shape[1] + 63) // 64) + 1)
        or any(
            torch._C._overlaps(task_state, t)
            for t in (x, weight, out, readiness, workspace)
        )
    ):
        raise ValueError(
            "Eligible tasks require distinct zeroed exact-size state "
            "and unswizzled tiles"
        )
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
            tile_swizzle,
            task_state,
        )
    return out
