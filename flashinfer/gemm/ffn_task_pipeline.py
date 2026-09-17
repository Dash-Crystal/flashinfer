# Copyright (c) 2026 by FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0
"""Experimental full-tile, single-grid local FFN task pipeline."""

import os
from functools import cache
from hashlib import sha256

import torch

from ..jit import env as jit_env
from ..jit.core import current_compilation_context, gen_jit_spec


@cache
def _module():
    source = jit_env.FLASHINFER_CSRC_DIR / "ffn_task_pipeline.cu"
    includes = jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer/gemm"
    dependencies = [
        source,
        includes / "ffn_task_pipeline.cuh",
        includes / "paired_geglu.cuh",
        includes / "masked_gemm.cuh",
        includes / "region_channel.cuh",
    ]
    identity = sha256(b"".join(p.read_bytes() for p in dependencies)).hexdigest()[:16]
    return gen_jit_spec(
        f"ffn_task_pipeline_{identity}",
        [source],
        extra_include_paths=[os.environ["TP_PORTCHANNEL_INCLUDE_DIR"]],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[8, 9, 10, 11, 12]
        )
        + ["--fmad=false"],
    ).build_and_load()


@cache
def ffn_task_pipeline_info(device: int, *, push: bool = False):
    """Return registers, shared bytes, threads, occupancy and physical SM count."""
    prepare = _module().prepare_push if push else _module().prepare
    return tuple(prepare(device))


def create_ffn_task_state(rows: int, device):
    """Allocate zeroed claim/completion state; only ordered launches may reuse it."""
    return torch.zeros(3 * ((rows + 127) // 128) + 2, dtype=torch.int32, device=device)


def run_ffn_task_pipeline(
    x: torch.Tensor,
    paired_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    activation: torch.Tensor,
    out: torch.Tensor,
    state: torch.Tensor,
    *,
    readiness: torch.Tensor | None = None,
    group_rows: int = 128,
    publication: torch.Tensor | None = None,
    reserved_sms: int = 0,
    producer_progress_proven: bool = False,
    worker_blocks_per_sm: int | None = None,
    retire_readiness: torch.Tensor | None = None,
    retire_group_rows: int = 128,
    retire_rows: int = 0,
    retire_workers: int = 0,
    native_region: int = 0,
    admission_state: torch.Tensor | None = None,
    peer_output: torch.Tensor | None = None,
    peer_publication: torch.Tensor | None = None,
    publication_epoch: int = 0,
):
    """Compute a complete local GeGLU FFN using one resident worker grid.

    Weights have contiguous PyTorch linear layouts (2I,H) and (H,I); gate/up
    channels are interleaved. Activation scratch is compact (M,I), output (M,H).
    Optional readiness counts completed normalized input rows per group. The
    final worker resets these input counts and internal state after all issued
    tasks drain. Input readiness must belong exclusively to this invocation.

    Optional int32 publication[4,G] emits local-ready flags in row zero after all
    output columns of each 128-row stripe finish. Row-two completion counts reset;
    rows one and three are untouched. The caller clears ready flags only after
    copies and readers finish, before reuse. All scratch/outputs must be disjoint.

    Concurrent input production requires a caller-established progress proof.
    ``worker_blocks_per_sm`` can limit per-SM resources for a co-resident producer;
    ``reserved_sms`` can instead bound the total resident worker count. Neither
    option guarantees placement or launch ordering. Set ``producer_progress_proven``
    only after checking the producer's resources and graph scheduling contract.
    This API never synchronizes CUDA or reads a device scalar on the host.

    Optional int32 ``admission_state[2]`` holds next-ticket and started flag.
    The caller zeros both before each invocation after prior region joins.
    Actual admission ticket zero cannot readiness-retire and system-publishes
    started=1. The final CTA resets the ticket counter, never the started flag.
    This proves a permanent FFN worker has entered; upstream row producers must
    still have an independent progress guarantee.

    Optional retirement returns a fixed subset of worker CTAs after the first
    ``retire_rows`` downstream input rows become ready. The workers finish their
    current full-K task before retiring. A device latch survives downstream
    readiness reset; final-CTA cleanup resets it for the next ordered invocation.
    At least one worker remains until every FFN task finishes. The caller must
    prove the downstream readiness publisher progresses with the initial grid.
    ``native_region`` optionally points to a device tp_region::DeviceRegion whose
    registered source is exactly ``out``. Its groups cover contiguous 128-row
    BF16 stripes. The final column tile enqueues one copy and signal, including
    the actual partial-tail byte count. Keep its registration and proxy alive
    until consumption finishes; only the consumer advances signal epochs.

    With ``peer_output`` and ``peer_publication``, the down-projection epilogue
    stores the same rounded BF16 fragments locally and to the mapped peer receive
    buffer. Row one of the peer publication array signals arrival after every
    column tile completes. The caller orders both ranks' resets before production
    and drains receive readers before reusing these buffers. Gate/up and its
    expanded activation remain local; GEMM K loops and task geometry are unchanged.
    """
    push = peer_output is not None
    if publication_epoch and (publication_epoch < 0 or not push):
        raise ValueError("Epoch publication requires a peer producer")
    if push != (peer_publication is not None):
        raise ValueError("Peer push requires both output and publication mappings")
    if admission_state is not None and (
        admission_state.device != x.device
        or admission_state.dtype != torch.int32
        or admission_state.shape != (2,)
        or not admission_state.is_contiguous()
        or any(
            torch._C._overlaps(admission_state, t)
            for t in (
                x,
                paired_up_weight,
                down_weight,
                activation,
                out,
                state,
                readiness,
                publication,
                retire_readiness,
            )
            if t is not None
        )
    ):
        raise ValueError(
            "Admission state must be a distinct contiguous CUDA int32 pair"
        )
    operands = (x, paired_up_weight, down_weight, activation, out)
    if (
        any(
            t.device != x.device
            or t.dtype != torch.bfloat16
            or t.ndim != 2
            or not t.is_contiguous()
            or t.data_ptr() % 16
            for t in operands
        )
        or x.device.type != "cuda"
    ):
        raise ValueError("Expected aligned contiguous same-device BF16 matrices")
    rows, hidden = x.shape
    intermediate = activation.shape[1]
    if (
        min(hidden, intermediate) <= 0
        or hidden % 8
        or intermediate % 8
        or paired_up_weight.shape != (2 * intermediate, hidden)
        or down_weight.shape != (hidden, intermediate)
        or activation.shape[0] != rows
        or out.shape != x.shape
        or group_rows <= 0
    ):
        raise ValueError("Invalid local FFN shapes or readiness group size")
    if (
        state.device != x.device
        or state.dtype != torch.int32
        or state.ndim != 1
        or not state.is_contiguous()
        or state.numel() < 3 * ((rows + 127) // 128) + 2
    ):
        raise ValueError("Insufficient int32 FFN task state")
    if readiness is not None and (
        readiness.device != x.device
        or readiness.dtype != torch.int32
        or readiness.ndim != 1
        or not readiness.is_contiguous()
        or readiness.numel() < (rows + group_rows - 1) // group_rows
    ):
        raise ValueError("Invalid input readiness counters")
    if publication is not None and (
        publication.device != x.device
        or publication.dtype != torch.int32
        or publication.ndim != 2
        or publication.shape[0] != 4
        or publication.shape[1] < (rows + 31) // 32
        or not publication.is_contiguous()
    ):
        raise ValueError("Invalid output publication counters")
    if push and (
        native_region
        or publication is None
        or peer_output.device != x.device
        or peer_output.dtype != out.dtype
        or peer_output.shape != out.shape
        or not peer_output.is_contiguous()
        or peer_output.data_ptr() % 16
        or peer_publication.device != x.device
        or peer_publication.dtype != torch.int32
        or peer_publication.shape != publication.shape
        or not peer_publication.is_contiguous()
        or any(
            torch._C._overlaps(peer_output, t) for t in (*operands, state, publication)
        )
        or any(
            torch._C._overlaps(peer_publication, t)
            for t in (*operands, state, publication, peer_output)
        )
    ):
        raise ValueError(
            "Peer push requires distinct matching payload and flag mappings"
        )
    if not rows:
        return out
    _, _, _, occupancy, sms = ffn_task_pipeline_info(x.device.index, push=push)
    if not 0 <= reserved_sms < sms or occupancy < 1:
        raise ValueError("FFN workers require available resident SM capacity")
    if readiness is not None and not producer_progress_proven:
        raise ValueError(
            "Concurrent input readiness requires a producer progress proof"
        )
    blocks_per_sm = occupancy if worker_blocks_per_sm is None else worker_blocks_per_sm
    if not 1 <= blocks_per_sm <= occupancy:
        raise ValueError("Worker block limit exceeds kernel occupancy")
    work_tiles = ((rows + 127) // 128) * ((2 * intermediate + 63) // 64)
    workers = min((sms - reserved_sms) * blocks_per_sm, work_tiles)
    if not 0 <= retire_workers < workers:
        raise ValueError("Retirement must leave at least one FFN worker")
    if retire_workers and (
        not producer_progress_proven
        or retire_readiness is None
        or retire_readiness.device != x.device
        or retire_readiness.dtype != torch.int32
        or retire_readiness.ndim != 1
        or not retire_readiness.is_contiguous()
        or not 0 < retire_rows <= rows
        or retire_group_rows <= 0
        or retire_readiness.numel()
        < (retire_rows + retire_group_rows - 1) // retire_group_rows
    ):
        raise ValueError("Retirement requires proved-progress downstream row readiness")
    if native_region and (native_region < 0 or publication is None):
        raise ValueError("Native region requires local publication counters")
    run = _module().run_push if push else _module().run
    run(
        x,
        paired_up_weight,
        down_weight,
        activation,
        out,
        state,
        readiness,
        group_rows,
        publication,
        workers,
        retire_readiness,
        retire_group_rows,
        retire_rows,
        retire_workers,
        native_region,
        admission_state,
        *((peer_output, peer_publication, publication_epoch) if push else ()),
    )
    return out
