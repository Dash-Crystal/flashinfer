# Copyright (c) 2026 by FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common TMA execution substrate for three ready projections."""

import os
from functools import cache
from hashlib import sha256

from ..jit import env as jit_env
from ..jit.core import current_compilation_context, gen_jit_spec


@cache
def _module():
    source = jit_env.FLASHINFER_CSRC_DIR / "shared_tma_task_pipeline.cu"
    headers = jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer/gemm"
    dependencies = [source] + [
        headers / name
        for name in (
            "shared_tma_task_pipeline.cuh",
            "eligible_ready_scheduler.cuh",
            "ready_tma_gemm.cuh",
            "masked_gemm.cuh",
        )
    ]
    identity = sha256(b"".join(p.read_bytes() for p in dependencies)).hexdigest()[:16]
    return gen_jit_spec(
        f"shared_tma_tasks_{identity}",
        [source],
        extra_include_paths=[os.environ["TP_PORTCHANNEL_INCLUDE_DIR"]],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[12]
        )
        + [
            "-DFLASHINFER_READY_TMA_SM120=1",
            "-DFLASHINFER_READY_MATH_REGISTERS=0",
            "-DFLASHINFER_READY_ROW_TILE=128",
            "-DFLASHINFER_READY_STARTUP_RAMP=1",
            "--fmad=false",
        ],
    ).build_and_load()
