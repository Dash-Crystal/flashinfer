# Copyright (c) 2026 by FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0
"""BF16 paired gate/up GEMM with a compact tanh-GeGLU epilogue."""

import os
from functools import cache
from hashlib import sha256

import torch

from ..jit import env as jit_env
from ..jit.core import current_compilation_context, gen_jit_spec


@cache
def _module(ready: bool = False, math_registers: int = 0):
    source = jit_env.FLASHINFER_CSRC_DIR / "paired_geglu.cu"
    header = jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer/gemm/paired_geglu.cuh"
    base = header.with_name("masked_gemm.cuh")
    contents = source.read_bytes() + header.read_bytes() + base.read_bytes()
    contents += header.with_name("region_channel.cuh").read_bytes()
    if ready:
        contents += header.with_name("paired_ready_geglu.cuh").read_bytes()
        contents += header.with_name("ready_tma_gemm.cuh").read_bytes()
        contents += header.with_name("eligible_ready_scheduler.cuh").read_bytes()
    identity = sha256(contents).hexdigest()[:16]
    return gen_jit_spec(
        f"paired_geglu_{identity}_ready{int(ready)}_r{math_registers}",
        [source],
        extra_include_paths=[os.environ["TP_PORTCHANNEL_INCLUDE_DIR"]],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[12] if ready else [8, 9, 10, 11, 12]
        )
        + [
            f"-DFLASHINFER_PAIRED_READY={int(ready)}",
            "-DFLASHINFER_READY_ROW_TILE=128",
            f"-DFLASHINFER_READY_MATH_REGISTERS={math_registers}",
            "-DFLASHINFER_READY_STARTUP_RAMP=0",
        ],
    ).build_and_load()


@cache
def paired_geglu_info(device: int):
    """Prepare the kernel before capture and return registers/shared bytes/threads."""
    return tuple(_module().prepare(device))


def pack_geglu_weight(weight: torch.Tensor) -> torch.Tensor:
    """Pack local [gate; up] linear weights (2I,K) once into column-major (K,2I)."""
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or weight.shape[0] % 8:
        raise ValueError(
            "Expected BF16 local gate/up weight (2I,K), with 2I divisible by 8"
        )
    gate, up = weight.chunk(2, dim=0)
    return torch.stack((gate, up), dim=1).reshape_as(weight).contiguous().t()


def mm_paired_geglu(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    out: torch.Tensor,
    is_padding: torch.Tensor | None = None,
    *,
    _validate_only: bool = False,
) -> torch.Tensor:
    """Compute BF16-round(GELU_tanh(BF16(xWg))) * BF16(xWu), then round to BF16.

    Fully padded 128-row tiles leave output untouched. Prepare with
    paired_geglu_info before CUDA graph capture; packing is a load-time operation.
    """
    tensors = (x, packed_weight, out)
    if any(
        t.ndim != 2 or t.dtype != torch.bfloat16 or t.device != x.device
        for t in tensors
    ):
        raise ValueError("Expected same-device BF16 matrices")
    if (
        x.device.type != "cuda"
        or x.shape[1] != packed_weight.shape[0]
        or packed_weight.shape[1] != 2 * out.shape[1]
        or out.shape[0] != x.shape[0]
        or x.shape[1] == 0
        or out.shape[1] == 0
        or x.shape[1] % 8
        or packed_weight.shape[1] % 8
        or x.stride(1) != 1
        or packed_weight.stride(0) != 1
        or out.stride(1) != 1
        or x.stride(0) < x.shape[1]
        or packed_weight.stride(1) < packed_weight.shape[0]
        or out.stride(0) < out.shape[1]
        or x.stride(0) % 8
        or packed_weight.stride(1) % 8
        or out.stride(0) % 4
        or any(t.data_ptr() % 16 for t in (x, packed_weight))
        or out.data_ptr() % 8
    ):
        raise ValueError("Unsupported paired GeGLU shape, layout, or alignment")
    if is_padding is not None and (
        is_padding.shape != (x.shape[0],)
        or is_padding.dtype != torch.bool
        or is_padding.device != x.device
        or not is_padding.is_contiguous()
    ):
        raise ValueError("Expected contiguous per-row CUDA boolean padding")
    if x.shape[0] and not _validate_only:
        paired_geglu_info(x.device.index)
        _module().run(x, packed_weight, out, is_padding)
    return out


@cache
def _ready_plan(device: int, producers: tuple[tuple[int, int, int], ...]):
    module = _module(True)
    info = tuple(module.prepare_ready(device, producers))
    budget = info[6]
    if budget:
        module = _module(True, budget)
        info = tuple(module.prepare_ready(device, producers))
    return module, info[:6]


@cache
def ready_paired_geglu_info(
    device: int, producers: tuple[tuple[int, int, int], ...] = ()
):
    """Return registers, shared bytes, threads, occupancy, SMs, concurrent-producer flag."""
    return _ready_plan(device, producers)[1]


def mm_ready_paired_geglu(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    out: torch.Tensor,
    readiness: torch.Tensor,
    *,
    group_rows: int,
    reserved_blocks: int = 0,
    producers: tuple[tuple[int, int, int], ...] = (),
) -> torch.Tensor:
    """Paired GeGLU on the existing SM120 READY TMA mainloop, with compact output.

    Readiness uses the same row counters and terminal reset as mm_ready_rows.
    Prepare ready_paired_geglu_info before capture. No weight packing occurs here.
    """
    mm_paired_geglu(x, packed_weight, out, _validate_only=True)
    if group_rows <= 0:
        raise ValueError("group_rows must be positive")
    groups = (x.shape[0] + group_rows - 1) // group_rows
    if (
        readiness.dtype != torch.int32
        or readiness.device != x.device
        or readiness.ndim != 1
        or not readiness.is_contiguous()
        or readiness.numel() < groups + 1
    ):
        raise ValueError("Expected row readiness counters and terminal completion slot")
    if not x.shape[0]:
        return out
    _, _, _, _, sms, concurrent = ready_paired_geglu_info(x.device.index, producers)
    if reserved_blocks < 0 or reserved_blocks >= sms:
        raise ValueError("reserved_blocks must be in [0, SM count)")
    if producers and not concurrent and not reserved_blocks:
        raise ValueError(
            "Producer cannot co-reside; provide a progress-safe reservation"
        )
    available_sms = sms if concurrent else sms - reserved_blocks
    module, _ = _ready_plan(x.device.index, producers)
    module.run_ready(
        x,
        packed_weight,
        out,
        readiness,
        group_rows,
        readiness.numel() - 1,
        available_sms,
    )
    return out
