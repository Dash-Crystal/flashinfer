# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Prepared scalar 2:4 NVFP4 operands for sparse W4A16 on SM120."""

import functools

import torch

from ..api_logging import flashinfer_api
from ..autotuner import AutoTuner, TunableRunner, TuningConfig
from ..trace.templates.gemm import mm_bf16_fp4_sparse_trace
from .gemm_bf16_fp4 import _unswizzle_sf_128x4
from .gemm_bf16_fp4_cute_dsl import _e4m3_to_s0e5m3


def prepare_bf16_fp4_sparse_weights(b, b_descale, *, paired=False):
    """Pack already-sparse NVFP4; never prune or requantize supplied values.

    Args:
        b: Contiguous uint8 [N, K/2] E2M1 codes, at most two nonzeros per four.
        b_descale: NVFP4's 128x4-swizzled E4M3 scales.
        paired: Permute adjacent pairs within each group of eight into scalar
            2:4 order. Pass the same flag to execution; no values change.

    Returns:
        Three int32 tensors: compressed values, block scales, and metadata.
        N must be divisible by 64 and K by 32. Prepare outside graph capture.
    """
    if b.dtype != torch.uint8 or b.ndim != 2 or not b.is_contiguous():
        raise ValueError("b must be contiguous uint8 [N,K/2]")
    n, k = b.shape[0], b.shape[1] * 2
    if not b.is_cuda or n <= 0 or k <= 0 or n % 64 or k % 32:
        raise ValueError("Sparse W4A16 requires N divisible by 64 and K by 32")
    expected_sf_bytes = ((n + 127) // 128) * ((k // 16 + 3) // 4) * 512
    if (
        b_descale.dtype != torch.uint8
        or b_descale.device != b.device
        or not b_descale.is_contiguous()
        or b_descale.numel() != expected_sf_bytes
    ):
        raise ValueError("b_descale must contain contiguous 128x4-swizzled E4M3 bytes")
    # Bound preparation scratch even for vocabulary projections.
    packed, scales, metadata = [], [], []
    linear_sf = _unswizzle_sf_128x4(b_descale, n, k // 16)
    if (linear_sf > 126).any().item():
        raise ValueError("NVFP4 block scales must be nonnegative finite E4M3 values")
    for start in range(0, n, 256):
        block = b[start : start + 256]
        rows = block.shape[0]
        codes = torch.stack((block & 15, block >> 4), -1)
        if paired:
            codes = codes.reshape(rows, k // 8, 4, 2).transpose(-1, -2).contiguous()
        codes = codes.reshape(rows, k // 4, 4)
        nonzero = (codes & 7) != 0
        if paired:
            nonzero = nonzero.reshape(rows, k // 8, 2, 4).any(dim=2)
        if (nonzero.sum(-1) > 2).any().item():
            raise ValueError("Weights violate scalar 2:4 sparsity")
        indices = nonzero.to(torch.int32).argsort(dim=-1, descending=True, stable=True)
        indices = indices[..., :2].sort(-1).values
        if paired:
            indices = indices.repeat_interleave(2, dim=1)
        retained = codes.gather(-1, indices).to(torch.int32)
        values = (retained[..., 0] | retained[..., 1] << 4).to(torch.uint8)
        values = values.reshape(rows // 16, 16, k // 32, 8)
        values = values.permute(0, 2, 1, 3).reshape(rows // 16, k // 32, 2, 8, 2, 4)
        values = values.permute(0, 1, 3, 5, 4, 2).contiguous()
        packed.append(values.reshape(rows // 16, k // 32, 128).view(torch.int32))
        nibble = indices[..., 0] | indices[..., 1] << 2
        shifts = torch.arange(4, device=b.device) * 4
        if paired:
            nibble = nibble[:, ::2].reshape(rows // 16, 16, k // 32, 4)
            word = (nibble << shifts).sum(-1).permute(0, 2, 1)
            word = word.reshape(rows // 16, k // 32, 2, 8)
            metadata.append((word[:, :, 0] | word[:, :, 1] << 16).to(torch.int32))
        else:
            nibble = nibble.reshape(rows // 16, 16, k // 32, 2, 4)
            word = (nibble << shifts).sum(-1).permute(0, 2, 1, 3)
            word = word.reshape(rows // 16, k // 32, 2, 8, 2)
            metadata.append(
                (word[:, :, 0] | word[:, :, 1] << 16)
                .to(torch.int32)
                .reshape(rows // 16, k // 32, 16)
            )
        sf = _e4m3_to_s0e5m3(linear_sf[start : start + rows])
        sf = sf.reshape(rows // 16, 16, k // 32, 2).permute(0, 2, 1, 3)
        sf = sf.reshape(rows // 16, k // 32, 2, 8, 2)
        sf = sf.permute(0, 1, 3, 4, 2).contiguous()
        scales.append(sf.reshape(rows // 16, k // 32, 32).view(torch.int32))
    return torch.cat(packed), torch.cat(scales), torch.cat(metadata)


@functools.cache
def _compiled(m, n, k, m_tiles, split_k, paired, prepared_a, stages):
    import cutlass
    import cutlass.cute as cute
    from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from .kernels.cute_dsl import sparse_gemm_bf16_fp4_sm120 as kernel

    def tensor(dtype, shape):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=16,
        )

    def compile_kernel():
        implementation = (
            kernel.SparseGemmBf16Fp4Tma
            if prepared_a and split_k == 1 and m >= 128 and k % 128 == 0
            else kernel.SparseGemmBf16Fp4
        )
        return cute.compile(
            implementation(m, n, k, m_tiles, split_k, paired, prepared_a, stages),
            tensor(
                cutlass.Int32,
                (((m + m_tiles * 8 - 1) // (m_tiles * 8)) * m_tiles * k // 8, 32)
                if prepared_a
                else (m, k // 2),
            ),
            tensor(cutlass.Int32, (n // 16, k // 32, 32)),
            tensor(cutlass.Int32, (n // 16, k // 32, 8)),
            tensor(cutlass.Int32, (n // 16, k // 32, 8 if paired else 16)),
            tensor(cutlass.Float32, (1,)),
            tensor(
                cutlass.BFloat16 if split_k == 1 else cutlass.Float32, (split_k, m, n)
            ),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi --opt-level 2",
        )

    return build_and_load_cute_dsl_kernel(
        "sparse_bf16_fp4",
        f"m{m}_n{n}_k{k}_mt{m_tiles}_sk{split_k}_paired{int(paired)}_pa{int(prepared_a)}_st{stages}",
        compile_kernel,
        extra_key_files=(__file__, kernel.__file__),
    )


@functools.cache
def _activation_packer(m, k, padded_m, paired):
    import cutlass
    import cutlass.cute as cute
    from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from .kernels.cute_dsl import sparse_gemm_bf16_fp4_sm120 as kernel

    return build_and_load_cute_dsl_kernel(
        "sparse_bf16_fp4_activation",
        f"m{m}_k{k}_pad{padded_m}_p{int(paired)}",
        lambda: cute.compile(
            kernel.PackActivations(m, k, padded_m, paired),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (m, k // 2), stride_order=(1, 0), assumed_align=16
            ),
            cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (padded_m * k // 64, 32),
                stride_order=(1, 0),
                assumed_align=16,
            ),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi --opt-level 2",
        ),
        extra_key_files=(__file__, kernel.__file__),
    )


def _run_sparse(
    a,
    b,
    b_descale,
    metadata,
    alpha,
    *,
    m_tiles=2,
    split_k=1,
    paired=False,
    prepared_a=False,
    stages=3,
):
    """BF16 activations against compressed scalar 2:4 NVFP4 weights."""
    m, k = a.shape
    n = b.shape[0] * 16
    if a.dtype != torch.bfloat16 or not a.is_cuda or not a.is_contiguous():
        raise ValueError("a must be a contiguous CUDA BF16 matrix")
    if torch.cuda.get_device_capability(a.device) != (12, 0):
        raise ValueError("Sparse NVFP4 W4A16 currently supports SM120")
    out = torch.empty(
        (split_k, m, n),
        device=a.device,
        dtype=torch.bfloat16 if split_k == 1 else torch.float32,
    )
    x = a.view(torch.int32)
    if prepared_a:
        padded_m = (m + m_tiles * 8 - 1) // (m_tiles * 8) * m_tiles * 8
        x = torch.empty((padded_m * k // 64, 32), device=a.device, dtype=torch.int32)
        _activation_packer(m, k, padded_m, paired)(a.view(torch.int32), x)
    _compiled(m, n, k, m_tiles, split_k, paired, prepared_a, stages)(
        x, b, b_descale, metadata, alpha, out
    )
    return out[0] if split_k == 1 else out.sum(0).to(torch.bfloat16)


def _tactics(m, n, k):
    return [
        (mt, sk, pa, stages)
        for mt in (1, 2, 4, 8)
        for sk in (1, 4, 16)
        for pa in ((False, True) if m <= 8 else (True,))
        if (mt <= max(1, (m + 7) // 8))
        and sk <= k // 32
        and (sk == 1 or sk * m * n * 4 <= 128 * 1024**2)
        for stages in ((2, 3) if pa and sk == 1 and m >= 128 and k % 128 == 0 else (3,))
    ]


class _SparseRunner(TunableRunner):
    def __init__(self, paired):
        self.paired = paired

    def get_cache_key_extras(self, inputs):
        return (self.paired,)

    def get_valid_tactics(self, inputs, profile):
        a, b, *_ = inputs
        return _tactics(a.shape[0], b.shape[0] * 16, a.shape[1])

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        a, b, *_ = inputs
        if tactic == -1:
            m, k = a.shape
            n = b.shape[0] * 16
            mt = 1 if m <= 8 else 2 if m <= 64 else 8
            ctas = n // 64 * ((m + mt * 8 - 1) // (mt * 8))
            sk = 16 if ctas < 128 else 4 if ctas < 512 else 1
            while sk > k // 32 or (sk > 1 and sk * m * n * 4 > 128 * 1024**2):
                sk = 4 if sk == 16 else 1
            tactic = (mt, sk, m > 8, 3)
        return _run_sparse(
            *inputs,
            m_tiles=tactic[0],
            split_k=tactic[1],
            prepared_a=tactic[2],
            stages=tactic[3],
            paired=self.paired,
        )


@flashinfer_api(trace=mm_bf16_fp4_sparse_trace)
def mm_bf16_fp4_sparse(
    a,
    b,
    b_descale,
    metadata,
    alpha,
    *,
    paired=False,
    m_tiles=None,
    split_k=None,
    prepared_a=None,
):
    """Sparse NVFP4 W4A16 on SM120, with FP32 accumulation and BF16 output.

    Prepare weights once with :func:`prepare_bf16_fp4_sparse_weights`. Use the
    same ``paired`` flag for preparation and execution. Autotune before graph
    capture using ``flashinfer.autotune``; explicit tile knobs bypass tuning.
    Activations retain their full BF16 range. Split-K uses FP32 scratch;
    layout staging and reduction are included in tuning measurements.
    """
    if (
        a.ndim != 2
        or a.dtype != torch.bfloat16
        or not a.is_cuda
        or not a.is_contiguous()
    ):
        raise ValueError("a must be a contiguous CUDA BF16 matrix")
    m, k = a.shape
    if b.ndim != 3:
        raise ValueError("b must be the prepared three-dimensional weight")
    n = b.shape[0] * 16
    if k <= 0 or n <= 0 or k % 32 or n % 64:
        raise ValueError("Sparse W4A16 requires N divisible by 64 and K by 32")
    for name, tensor, last in (
        ("b", b, 32),
        ("b_descale", b_descale, 8),
        ("metadata", metadata, 8 if paired else 16),
    ):
        if (
            tensor.dtype != torch.int32
            or tensor.device != a.device
            or not tensor.is_contiguous()
            or tensor.shape != (n // 16, k // 32, last)
        ):
            raise ValueError(
                f"{name} has invalid prepared shape, dtype, device, or stride"
            )
    if (
        alpha.dtype != torch.float32
        or alpha.device != a.device
        or alpha.shape != (1,)
        or not alpha.is_contiguous()
    ):
        raise ValueError("alpha must be contiguous CUDA float32 [1] on a's device")
    if m == 0:
        return a.new_empty((0, n))
    inputs = [a, b, b_descale, metadata, alpha]
    runner = _SparseRunner(paired)
    if any(v is not None for v in (m_tiles, split_k, prepared_a)):
        mt = 2 if m_tiles is None else m_tiles
        sk = 1 if split_k is None else split_k
        pa = bool(prepared_a)
        if mt not in (1, 2, 4, 8) or sk not in (1, 4, 16):
            raise ValueError("Unsupported sparse MMA tile or split-K count")
        return runner(inputs, tactic=(mt, sk, pa, 3))
    chosen, tactic = AutoTuner.get().choose_one(
        "sparse_bf16_fp4", [runner], TuningConfig(), inputs
    )
    return chosen(inputs, tactic=tactic)
