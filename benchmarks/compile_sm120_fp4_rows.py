"""Compile scalar and row-scaled SM120 GEMM variants without launching kernels."""

import os

os.environ["CUTE_DSL_ARCH"] = "sm_120a"

import cutlass

from flashinfer.gemm.gemm_mm_fp4_cute_dsl import _make_blockscaled_gemm_compile_fn
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120_b12x import (
    Sm120B12xBlockScaledDenseGemmKernel,
)


def main():
    for rowwise in (False, True):
        for swap in (False, True):
            for dtype in (cutlass.BFloat16, cutlass.Float16):
                kernel = Sm120B12xBlockScaledDenseGemmKernel(
                    16,
                    (64, 32) if swap else (128, 128),
                    (1, 1),
                    swap_ab=swap,
                    rowwise_alpha=rowwise,
                )
                compile_kernel = _make_blockscaled_gemm_compile_fn(
                    kernel,
                    ab_cutlass_dtype=cutlass.Uint8,
                    sf_dtype=cutlass.Float8E4M3FN,
                    c_cutlass_dtype=dtype,
                    ab_assumed_align=32,
                    swap_ab=False,
                    sf_m=9,
                    sf_n=30,
                    sf_k=60,
                    batch_size=1,
                    max_active_clusters=1,
                    rowwise_alpha=rowwise,
                )
                compile_kernel()
                print(
                    f"compiled rowwise={rowwise} swap={swap} dtype={dtype}", flush=True
                )


if __name__ == "__main__":
    main()
