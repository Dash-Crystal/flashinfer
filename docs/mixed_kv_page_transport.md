# Mixed KV page transport

This branch transports post-RoPE K and projected V as vLLM 16-token pages in one of three physical formats: A16 (`0`), block-scaled E4M3 (`1`), or block-scaled E2M1 (`2`). Both compressed formats use one E4M3 scale per 16 coefficients; attention consumes A16 values without writing a reconstructed A16 cache to global memory.

## Runtime API

Pass `MixedKVPagedCache` as `page_transport` to `xqa_batch_decode_with_kv_cache`. Its `page_format` tensor has one device-resident byte per physical page, and its FP8/FP4 payload and scale tensors use the same page, token, and KV-head indexing as the canonical A16 paged cache.

`page_transport_static_format=0|1|2` selects a format-specialized module when every page visible to the call has that format. Omitting it preserves device-side per-page routing. A caller must not provide a static format that disagrees with the routed pages; the scheduler already owns this fact and can pass it without inspecting device data or synchronizing a CUDA graph.

The page sealer is graph-safe: it consumes fixed-capacity page-completion operands, computes page-local residual statistics, and publishes payload, scales, and the format tag. Sealing is not on the attention dependency chain except when a newly completed page is first made visible to a later attention call.

## Kernel structure

The SM90 decode specialization retains FlashInfer XQA's two-stage K, V, and softmax buffers. Its existing four-warp I/O producer group is divided into two cooperative K loaders and two cooperative V loaders; the first K loader also performs the one-time Q load. Independent four-warp K and V conversion groups expand the next compressed stages while the existing QK and PV GMMA groups consume the current A16 stages. Block-scale conversion is hoisted once per 16-value block. The mbarriers are CTA-local stage ownership; there is no grid-wide producer handoff or host/device synchronization.

The conversion and pipeline organization follows the in-tree NVIDIA references:

- `3rdparty/cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp`
- `3rdparty/cutlass/examples/python/CuTeDSL/blackwell/mixed_input_fmha/mixed_input_fmha_decode.py`
- `include/flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh`
- `csrc/xqa/mha_sm90.cu`

Architecture separation is normative: see `mixed_kv_page_transport_backends.md` (one backend per compute capability; dispatch asserted, not inferred) and `mixed_kv_page_transport_cutlass_references.md` (per-stage in-tree references, cost model, and acceptance targets).

The SM120 and architecture-neutral arms keep conversion in register fragments. Static specialization removes the other two format paths from the hot module; arbitrary mixed pages still require format scatter plus partial-attention/LSE merge to obtain the same performance without the dynamic code union.

## Current validation

Every transport arm is bit-exact against attention over an explicitly expanded A16 paged cache. Fresh builds of the 34-case suite pass on both H200/SM90 and RTX 5090/SM120. The cases cover A16, FP8, FP4, mixed pages, NHD/HND, partial final pages, decode, and query spans up to 64.

At batch 17, context 4096, 8 KV heads, group size 4, head dimension 128, BF16, and CUDA graph replay, fresh RTX 5090 medians are:

| Query span | A16 baseline | static block FP8 | static block FP4 | dynamic mixed |
| --- | ---: | ---: | ---: | ---: |
| 1 | 177.3 us | 137.5 us | 98.3 us | 233.5 us |
| 64 | 421.4 us | 682.7 us | 785.6 us | 1386.5 us |

The SM120 decode prologues therefore provide same-run speedups of 1.29x for FP8 and 1.80x for FP4. The dynamic code union and the query-span path are not wins.

Fresh H200 medians for the same shape are:

| Query span | A16 baseline | A16 transport | static block FP8 | static block FP4 | dynamic mixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 109.4 us | 92.3 us | 195.5 us | 760.9 us | 780.3 us |
| 64 | 1459.0 us | 1507.8 us | 2868.3 us | 9621.3 us | 10256.4 us |

Splitting the I/O producer group corrected static FP8 decode from 725.8 to 195.5 microseconds, but it remains 1.79x slower than the unmodified A16 call. SM90 FP4 uses software E2M1 conversion and is not competitive. Neither is a speedup claim.

## Distortion contract

Kernel equality to explicit A16 expansion detects transport, unpacking, scaling, masking, and service errors. It does not establish a model-quality rate-distortion curve. End-to-end acceptance must measure on-policy whole-sequence logit KL and hidden-state or embedding trajectory divergence, then verify that changing the page-routing thresholds changes those sequence metrics monotonically. Absence of that control relationship rejects a claimed precision-related result rather than reclassifying kernel or service divergence as quantization loss.

The next integration step is a CUDA-graph-compatible scatter of page runs into A16, FP8, and FP4 specialized attention work, followed by FlashInfer-style partial-state/LSE merge. Decode, draft verification, and continued prefill must share that attention implementation and differ only in their mask modifier.
