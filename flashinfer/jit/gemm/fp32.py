# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0
"""Build the stock CUTLASS three-product TF32 batched GEMM adapter."""

from hashlib import sha256

from .. import env as jit_env
from ..core import current_compilation_context, gen_jit_spec


def gen_fp32_tf32x3_module():
    source = jit_env.FLASHINFER_CSRC_DIR / "fp32_tf32x3_gemm.cu"
    identity = sha256(source.read_bytes()).hexdigest()[:16]
    return gen_jit_spec(
        f"fp32_tf32x3_{identity}",
        [source],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[8, 9, 10, 11, 12]
        ),
    )
