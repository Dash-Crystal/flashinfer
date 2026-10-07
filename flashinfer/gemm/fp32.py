# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0
"""FP32 storage and accumulation using CUTLASS's three-product TF32 program."""

from functools import cache

import torch

from ..jit.gemm.fp32 import gen_fp32_tf32x3_module


@cache
def get_fp32_tf32x3_module():
    return gen_fp32_tf32x3_module().build_and_load()


@torch.library.custom_op("flashinfer::bmm_fp32_tf32x3", mutates_args=())
def bmm_fp32_tf32x3(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor | None = None,
    beta: float = 0.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Compute alpha*A@B + beta*C with explicit TF32x3 arithmetic.

    Operands are CUDA FP32 matrices or batches with aligned row/column strides.
    FP32 storage and accumulation are retained. This is an approximate arithmetic
    choice and must be qualified for the caller's complete numerical program.
    """
    if a.ndim not in (2, 3) or b.ndim != a.ndim:
        raise ValueError("Expected two matrices or two batched matrices")
    if (
        a.dtype != torch.float32
        or b.dtype != a.dtype
        or a.device.type != "cuda"
        or b.device != a.device
        or a.shape[-1] != b.shape[-2]
        or a.shape[:-2] != b.shape[:-2]
    ):
        raise ValueError("Expected compatible CUDA FP32 GEMM operands")
    shape = (*a.shape[:-2], a.shape[-2], b.shape[-1])
    out = a.new_empty(shape)
    if c is None:
        if beta != 0:
            raise ValueError("A nonzero beta requires an input matrix")
        c = out
    elif c.shape != shape or c.dtype != a.dtype or c.device != a.device:
        raise ValueError("Expected a matching FP32 epilogue input")
    matrices = (
        (a, b, c, out)
        if a.ndim == 3
        else tuple(value.unsqueeze(0) for value in (a, b, c, out))
    )
    get_fp32_tf32x3_module().bmm(*matrices, alpha, beta)
    return out


@bmm_fp32_tf32x3.register_fake
def _fake(a, b, c=None, beta=0.0, alpha=1.0):
    return a.new_empty((*a.shape[:-2], a.shape[-2], b.shape[-1]))


def baddbmm_fp32_tf32x3(input, batch1, batch2, *, beta=1.0, alpha=1.0):
    return bmm_fp32_tf32x3(batch1, batch2, input, beta, alpha)
