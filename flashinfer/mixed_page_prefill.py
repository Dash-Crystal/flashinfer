"""
Copyright (c) 2026 by FlashInfer team.
SPDX-License-Identifier: Apache-2.0

Mixed-page (ragged-format) paged KV attention on Hopper through the FA3 batch
prefill kernel.  Pages carry a one-byte format tag (A16 / block-scaled E4M3 /
block-scaled E2M1); the producer warp group expands compressed pages to A16 in
shared memory and the unchanged FA3 consumer does the math.  This serves every
query shape (prefill, continued prefill, draft verification, decode); the XQA
mixed-page kernel remains the low-FLOP choice for single-token decode with a
small group size.

Usage::

    wrapper = BatchPrefillWithPagedKVCacheWrapper(
        workspace, kv_layout, backend="fa3",
        jit_args=mixed_page_prefill_jit_args(q.dtype, kv.dtype, q.dtype, head_dim))
    wrapper.plan(..., page_size=16, causal=True)
    out = wrapper.run(q, (k_cache, v_cache),
                      *mixed_page_prefill_run_args(transport, sm_scale, static_format))
"""

from typing import List, Optional, Tuple

import torch

from .quantization.kv_cache_fp8 import MixedKVPagedCache
from .jit.utils import filename_safe_dtype_map

# Order matters: the generated Run() signature is [tensors..., scalars...] in
# these lists' order, and BatchPrefillWithPagedKVCacheWrapper.run(*args)
# forwards user args positionally in that order.
_FA3_BASE_TENSOR_NAMES = [
    "maybe_prefix_len_ptr",
    "maybe_token_pos_in_items_ptr",
    "maybe_max_item_len_ptr",
    "maybe_scale_v",
]
_FA3_BASE_TENSOR_DTYPES = ["uint32_t", "uint16_t", "uint16_t", "float"]
_FA3_BASE_SCALAR_NAMES = ["logits_soft_cap", "sm_scale", "scale_v_scalar", "token_pos_in_items_len"]
_FA3_BASE_SCALAR_DTYPES = ["double", "double", "double", "int64_t"]

MIXED_TENSOR_NAMES = [
    "mixed_page_format",
    "mixed_fp8_k_payload",
    "mixed_fp8_v_payload",
    "mixed_fp8_k_scales",
    "mixed_fp8_v_scales",
    "mixed_fp4_k_payload",
    "mixed_fp4_v_payload",
    "mixed_fp4_k_scales",
    "mixed_fp4_v_scales",
    "mixed_fp8_k_global_scale",
    "mixed_fp8_v_global_scale",
    "mixed_fp4_k_global_scale",
    "mixed_fp4_v_global_scale",
]
MIXED_TENSOR_DTYPES = ["uint8_t"] * 9 + ["float"] * 4
MIXED_SCALAR_NAMES = [
    "mixed_fp8_payload_stride_page",
    "mixed_fp8_payload_stride_token",
    "mixed_fp8_payload_stride_head",
    "mixed_fp8_scale_stride_page",
    "mixed_fp8_scale_stride_token",
    "mixed_fp8_scale_stride_head",
    "mixed_fp4_payload_stride_page",
    "mixed_fp4_payload_stride_token",
    "mixed_fp4_payload_stride_head",
    "mixed_fp4_scale_stride_page",
    "mixed_fp4_scale_stride_token",
    "mixed_fp4_scale_stride_head",
    "mixed_static_format",
]
MIXED_SCALAR_DTYPES = ["int64_t"] * len(MIXED_SCALAR_NAMES)

MIXED_PAGE_SIZE = 16


def _static_format_code(static_format: Optional[int]) -> int:
    if static_format not in (None, 0, 1, 2):
        raise ValueError("static_format must be None, 0, 1 or 2")
    return -1 if static_format is None else int(static_format)


def mixed_page_prefill_jit_args(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    use_sliding_window: bool = False,
    static_format: Optional[int] = None,
) -> Tuple:
    """``jit_args`` for ``BatchPrefillWithPagedKVCacheWrapper(backend="fa3", jit_args=...)``.

    The kernel selects the mixed-page producer because the generated
    ``AdditionalParams`` carries ``mixed_page_format`` (see
    ``has_mixed_page_format_v`` in ``hopper/sparse_mixed_mainloop.cuh``).

    ``static_format`` (None / 0 / 1 / 2) is a compile-time constant of the module
    (variant ``MixedPageAttention<N>``, URI suffix ``_static_format_N``): a module
    built for one format compiles only that format's copy path and no tag loads;
    ``None`` builds the dynamic module with the per-page format switch.  The
    same value must be passed to :func:`mixed_page_prefill_run_args`.
    """
    if head_dim != 128:
        raise ValueError("mixed-page FA3 attention is implemented for head_dim == 128")
    if dtype_kv not in (torch.bfloat16, torch.float16):
        raise ValueError("mixed-page FA3 attention expands to a bf16/f16 KV cache dtype")
    code = _static_format_code(static_format)
    uri = (
        f"batch_prefill_mixed_page_sm90_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"head_dim_{head_dim}_swa_{use_sliding_window}_static_format_{code}"
    )
    return (
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        torch.int32,
        head_dim,
        head_dim,
        _FA3_BASE_TENSOR_NAMES + MIXED_TENSOR_NAMES,
        _FA3_BASE_TENSOR_DTYPES + MIXED_TENSOR_DTYPES,
        _FA3_BASE_SCALAR_NAMES + MIXED_SCALAR_NAMES,
        _FA3_BASE_SCALAR_DTYPES + MIXED_SCALAR_DTYPES,
        f"MixedPageAttention<{code}>",
        "#include<flashinfer/attention/hopper/variants.cuh>",
    )


def mixed_page_prefill_jit_kwargs(use_sliding_window: bool = False) -> dict:
    return {"use_sliding_window": use_sliding_window}


def _byte_strides(t: torch.Tensor, kv_layout: str) -> Tuple[int, int, int]:
    """(page, token, head) strides in bytes of a 1-byte-element transport tensor laid out
    [pages, tokens, heads, bytes] (NHD) or [pages, heads, tokens, bytes] (HND)."""
    if t.element_size() != 1:
        raise TypeError("mixed-page payload and scale tensors use 1-byte elements")
    if kv_layout == "NHD":
        return t.stride(0), t.stride(1), t.stride(2)
    if kv_layout == "HND":
        return t.stride(0), t.stride(2), t.stride(1)
    raise ValueError(f"unknown kv_layout {kv_layout!r}")


def mixed_page_prefill_run_args(
    transport: MixedKVPagedCache,
    sm_scale: float,
    static_format: Optional[int] = None,
    kv_layout: str = "NHD",
) -> List:
    """Positional ``*args`` for ``wrapper.run(q, (k_cache, v_cache), *args)``.

    ``static_format`` (None / 0 / 1 / 2) must equal the ``static_format`` the
    wrapper's module was built with (:func:`mixed_page_prefill_jit_args`); the
    kernel launcher rejects a mismatch.  A static module promises every page
    visible to the call has that format and never reads ``page_format``.
    """
    code = _static_format_code(static_format)
    if transport.page_format.dtype != torch.uint8 or transport.page_format.dim() != 1:
        raise ValueError("page_format must be one uint8 tag per physical page")
    fp8_p = _byte_strides(transport.fp8_k_payload, kv_layout)
    fp8_s = _byte_strides(transport.fp8_k_scales, kv_layout)
    fp4_p = _byte_strides(transport.fp4_k_payload, kv_layout)
    fp4_s = _byte_strides(transport.fp4_k_scales, kv_layout)
    for a, b in ((transport.fp8_k_payload, transport.fp8_v_payload),
                 (transport.fp8_k_scales, transport.fp8_v_scales),
                 (transport.fp4_k_payload, transport.fp4_v_payload),
                 (transport.fp4_k_scales, transport.fp4_v_scales)):
        if a.stride() != b.stride():
            raise ValueError("K and V transport tensors must share strides")
    # The FP8 expansion folds 2^112 into (block_scale * global_scale) in bf16
    # ([23]); the fold is exact while that product is below 2^16 for every block
    # scale an E4M3 byte can hold (448).  One host sync per call site of this
    # helper (the wrappers build the argument list once per shape).
    if code in (-1, 1):
        for gs in (transport.fp8_k_global_scale, transport.fp8_v_global_scale):
            if gs is not None and gs.numel() == 1 and float(gs.item()) * 448.0 >= 2.0 ** 16:
                raise ValueError("mixed KV pages: fp8 global scale must be below 2^16 / 448")
    tensors = [
        None,  # maybe_prefix_len_ptr
        None,  # maybe_token_pos_in_items_ptr
        None,  # maybe_max_item_len_ptr
        None,  # maybe_scale_v
        transport.page_format,
        transport.fp8_k_payload,
        transport.fp8_v_payload,
        transport.fp8_k_scales,
        transport.fp8_v_scales,
        transport.fp4_k_payload,
        transport.fp4_v_payload,
        transport.fp4_k_scales,
        transport.fp4_v_scales,
        transport.fp8_k_global_scale,
        transport.fp8_v_global_scale,
        transport.fp4_k_global_scale,
        transport.fp4_v_global_scale,
    ]
    scalars = [
        0.0,  # logits_soft_cap
        float(sm_scale),
        1.0,  # scale_v_scalar
        0,  # token_pos_in_items_len
        *fp8_p,
        *fp8_s,
        *fp4_p,
        *fp4_s,
        code,
    ]
    return tensors + scalars
