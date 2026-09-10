# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""CUTLASS matrix multiplication with caller-owned row visibility."""

from functools import cache
from typing import Any, NamedTuple

import torch

from ..jit.gemm.core import gen_masked_gemm_module


@cache
def get_masked_gemm_module():
    return gen_masked_gemm_module().build_and_load()


@cache
def _prepared_module(device: int, dtype: torch.dtype, publish: bool):
    module = get_masked_gemm_module()
    module.prepare(device, dtype == torch.bfloat16, publish)
    return module


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


def mm_masked_tiles(x, weight, out, is_padding, *, row_offset=0, peer_output=None):
    """Write GEMM tiles containing visible rows; wholly padded tiles stay untouched.

    The caller must consume outputs with the same row mask. Arbitrary holes in
    the mask are supported; partially visible tiles use ordinary GEMM arithmetic.
    Operands are A row-major, B column-major, and output row-major, aligned to
    eight FP16/BF16 values. An optional peer mapping receives the same epilogue
    stores. Its owner must publish kernel completion to peer readers with system
    synchronization, and order reuse after the readers finish.
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
    if x.shape[0] and weight.shape[1]:
        _prepared_module(x.device.index, x.dtype, peer_output is not None).run(
            x, weight, out, is_padding, row_offset, peer_output
        )
    return out


class ReadyGemmInfo(NamedTuple):
    module: Any
    sms: int
    row_tile: int
    workspace_bytes: int


@cache
def ready_gemm_info(device: int, dtype: torch.dtype) -> ReadyGemmInfo:
    module = get_masked_gemm_module()
    sms, row_tile, workspace_bytes = module.prepare_ready(
        device, dtype == torch.bfloat16
    )
    return ReadyGemmInfo(module, sms, row_tile, workspace_bytes)


def create_ready_workspace(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Allocate one reusable Stream-K workspace before model graph capture."""
    info = ready_gemm_info(device.index, dtype)
    return torch.zeros(info.workspace_bytes, dtype=torch.uint8, device=device)


def mm_ready_rows(x, weight, out, readiness, *, group_rows, reserved_blocks, workspace):
    """Multiply once, consuming row groups published by a concurrent producer.

    Operands follow ``mm_masked_tiles``'s layout contract. The caller initializes
    the int32 workspace to zero before capture. Each producer row contributes
    one device-release increment to its group, after storing every column.
    Padded rows must also be stored and counted. The final workspace element is
    reserved for consumer completion; the last GEMM CTA resets the workspace.
    The caller joins both kernels before reusing any operand or workspace.

    ``reserved_blocks`` is the producer's maximum resident CTA count. The GEMM
    grid leaves that many SMs available, so readiness waits cannot occupy the
    producer's execution capacity. CUTLASS Stream-K distributes K iterations as
    well as complete output tiles. ``workspace`` is allocated and zeroed once
    with ``create_ready_workspace``; FP32 partials and barriers occupy fixed,
    disjoint regions across shapes. Execution adds no allocation or reset kernel.
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
    info = ready_gemm_info(x.device.index, x.dtype)
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
        raise ValueError("Stream-K requires distinct workspace and producer capacity")
    if x.shape[0] and weight.shape[1]:
        info.module.run_ready(
            x, weight, out, readiness, workspace, group_rows, info.sms, reserved_blocks
        )
    return out
