# Mixed-precision paged-KV transport (experimental)

This branch extends XQA's native multi-block reduction so independently
specialized page runs can contribute to one attention result.  The intended
unit of classification is one vLLM/PagedAttention page (16 tokens), not one
value and not an attention-derived estimate.

## Data path

The scheduler scatters each request's block table into format-homogeneous page
lists and gathers their physical page IDs.  Each nonempty list launches the
existing format-specialized XQA producer.  A global split count and per-launch
split base give every producer a disjoint slot in XQA's existing scratch
layout.  The last producer CTA performs XQA's stable merge of FP32 row maxima,
row sums, and output numerators.

For native NVFP4, XQA double-buffers packed K/V page data and separate FP8
scale tiles in shared memory.  `ldmatrix` unpacks E2M1 values, converts the
scale for each 16-value block, applies it in registers, and supplies BF16/FP16
MMA operands.  No reconstructed A16 cache is written to global memory.

The branch also adds an accuracy-first 8-bit format: E4M3 payload, one E4M3
scale per 16 values, and separate launch-level K/V FP32 scales.  Its rate is
8.5 bits/value plus negligible launch metadata.  The scale tile follows the
same compact companion-page layout as NVFP4.  XQA double-buffers payload and
scale loads, applies the scale while producing A16 operands, and keeps QK,
softmax, and XV arithmetic on an A16 path.

This is deliberately not OCP MXFP8.  MXFP8 uses a UE8M0 power-of-two scale per
32 values (8.25 bits/value).  The finer 16-value block and E4M3 scale spend
0.25 extra bit/value to reduce scale rounding and within-block range loss.
The page-seal encoder can additionally search nine neighboring E4M3 scales
under `MSE + tail_weight * max_error^2`; that changes seal cost but not the
read kernel or stored rate.

The direct page-seal CUDA epilogue maps one 16-lane subgroup to each block and
writes into preallocated payload and scale pools.  It never materializes a
full FP32 cache.  The searched encoder is row-chunked and remains a reference
path pending a fused multi-candidate seal kernel.

For a speculative/continued query, historical compressed page runs disable
the tail mask. Every page that intersects the active query span stays A16 and
is read by the mask-carrying A16 producer. Only pages strictly before
`(sequence_length - query_length)` may enter the mask-disabled compressed
producer. This remains true when a query spans several pages and prevents
compacted historical pages from being mistaken for the live attention span.

The page sealer also computes two page-local statistics from the representation
it just encoded: relative RMS reconstruction error and peak reconstruction
error normalized by the page amax. The page is marked BSFP8 only when both
statistics satisfy caller-provided thresholds; otherwise it remains A16. This
router is part of the sealing operation, not an optional analysis pass, and
does not materialize a dequantized page.

## SM120 measurements

RTX 5090, BF16 query/output, Hq=16, Hkv=8, D=256, page size 16, batch 32,
context 4096.  CUDA-event medians; quantization and page-list construction are
outside the timed attention call.  The reference is FlashInfer's ordinary
paged A16 decode/prefill wrapper over the same physical cache and page table.

| Page transport | Query tokens | Effective bits/value | Latency (ms) | A16 latency (ms) | Speedup |
|---|---:|---:|---:|---:|---:|
| uniform native NVFP4 | 1 | 4.50 | 0.327 | 0.646 | 1.98x |
| 80% Q4, 20% A16 | 1 | 6.79 | 0.404 | 0.646 | 1.60x |
| 81% Q4, 14% FP8, 5% A16 | 1 | 5.53 | 0.372 | 0.652 | 1.75x |
| 80% Q4, 20% A16 | 16 | 6.84 | 0.460 | 0.658 | 1.43x |
| 81% Q4, 14% FP8, 5% A16 | 16 | 5.58 | 0.450 | 0.659 | 1.46x |

Uniform Q8 comparison on the same SM120/B32/4K geometry:

| Page transport | Query tokens | Effective bits/value | Latency (ms) | A16 latency (ms) | Speedup | Relative RMS vs A16 |
|---|---:|---:|---:|---:|---:|---:|
| per-tensor E4M3 | 1 | 8.00 | 0.343 | 0.648 | 1.89x | 0.03836 |
| block-scaled E4M3-16, direct scale | 1 | 8.50 | 0.387 | 0.648 | 1.67x | 0.03733 |
| per-tensor E4M3 | 16 | 8.00 | 0.360 | 0.658 | 1.83x | 0.03846 |
| block-scaled E4M3-16, direct scale | 16 | 8.50 | 0.410 | 0.658 | 1.61x | 0.03748 |

The block-scale read premium is 43.8 us for q_len=1 and 50.1 us for q_len=16
over the one-scale FP8 XQA path.  It retains 84-86% of that path's latency
reduction versus A16.

Direct seal timings for K and V together, with preallocated outputs, are
18.75 us for one page, 18.59 us for 32 pages, and 26.43 us for 128 pages.
At 32 pages this is 0.581 us/page; at 128 pages the measured aggregate input
plus output traffic is 972 GB/s.  Kernel-launch overhead dominates the
single-page case, so serving should batch simultaneously sealed pages.

The integrated Dash Crystal/vLLM prototype was also measured with its real
page scatter and two format-specialized XQA launches.  On RTX 5090 at batch
17, context 4096, Hq=16, Hkv=8, D=256, and 5.01% A16 pages, compact A16 XQA
took 0.3580 ms.  Mixed attention took 0.2504 ms, route scatter took 0.01235 ms,
and route plus attention took 0.2636 ms: 1.36x versus compact A16.  Batch 17
is the largest over-16 batch that fit beside the resident service; no service
process was stopped for the measurement.

For a uniform 64-token continued-prefill span at batch 17 and context 4096,
compact A16 XQA took 0.6040 ms and mixed XQA including route scatter took
0.7767 ms (0.778x). The corrected active-span protection keeps the 64 new
tokens, 1.5625% of values read, in the masked A16 lane. The output relative RMS
against A16 was 0.03754. This result establishes that the same operation runs
the longer masked span; it is not yet a performance success on SM120.

## SM90 measurements

H200, B32, 4K, page-16, Hq=16, Hkv=8, D=256 configuration. Tensor FP8 uses
FlashInfer's Hopper-specialized path. BSFP8-16 uses the register-dequantized
warp-MMA path: compressed payload and scales are double buffered in shared
memory, expanded into FP16 MMA operand registers, and consumed without an A16
shared-memory reconstruction tile.

| Page transport / model I/O | Query tokens | Latency (ms) | A16 latency (ms) | Speedup | Relative RMS vs A16 |
|---|---:|---:|---:|---:|---:|
| per-tensor E4M3 / BF16 | 1 | 0.1423 | 0.3252 | 2.29x | 0.05298 |
| block-scaled E4M3-16 / native BF16 I/O, FP16 register MMA | 1 | 0.2648 | 0.3246 | 1.23x | 0.03707 |
| block-scaled E4M3-16 / FP16 math | 1 | 0.2595 | 0.3257 | 1.26x | 0.03695 |
| block-scaled E4M3-16 / BF16 I/O, FP16 math bridge | 1 | 0.2702 | 0.3252 | 1.20x | 0.03703 |

The native specialization keeps BF16 Q, A16 pages, output, and reduction
scratch as the public ABI. It converts Q and softmax words plus block-scaled
K/V directly into FP16 register fragments and executes FP16 `mma.sync`; no
extra kernel or shared A16 reconstruction is used. The bridge measurement is
retained as a cross-check: Nsight Systems reports a median 0.2569 ms for its
FP16 attention kernel and about 7.2 us of GPU execution for both conversion
kernels together. The native form is faster and remains compatible with
mixed BF16-A16/BSFP8 cross-launch reduction.

The format now caps the encoded E4M3 block scale at 128 and chooses the
reference global scale accordingly. This guarantees the intermediate
`abs(payload * block_scale) <= 448 * 128 = 57,344`, below FP16's finite maximum;
the global scale is applied after MMA. The direct producer applies the same
cap, so an undersized caller-provided global scale clips quantization rather
than creating an FP16 infinity. The direct K+V seal kernel remains 14.58 us
for 32 pages, 0.456 us/page, and 441 GB/s counting input plus output traffic.

The Dash Crystal/vLLM integration at batch 32, context 4096, and 5.00% A16
pages now measures 0.3277 ms for compact A16 XQA, 0.3225 ms for mixed XQA,
0.00589 ms for route scatter, and 0.3260 ms for route plus attention: 1.005x.
At context 8192 the same 5.00% A16 mix measures 0.6246 ms A16 versus 0.5879 ms
mixed attention and 0.5895 ms including scatter: 1.060x. The fixed second
launch/reduction cost hides most of the 4K producer gain; it amortizes at the
longer streaming context. Moving page-run maintenance to seal time removes
the remaining per-call scatter cost.

With the mandatory page-local router and its default thresholds, B32/4K decode
measured 0.3269 ms for A16 and 0.3078 ms for routing plus mixed attention
(1.062x). A uniform 64-token continued-prefill span measured 1.1292 ms for A16,
1.1213 ms for mixed attention, and 1.1283 ms including routing (1.001x). The
active 64-token span is A16 while the strictly historical prefix is eligible
for BSFP8. Output relative RMS against A16 was 0.03759.

The alternative SM90 TMA/WGMMA experiment was also corrected. Its producer
no longer waits for its own transaction: two compressed and two expanded
buffers form independent mbarrier rings, and each K/V leader issues the next
TMA before joining a 64-lane expansion of the current tile. That reduced the
B32/4K kernel from 1.72 ms to 0.824 ms, but it remains inferior because it
writes a complete reconstructed A16 tile to shared memory. It is not selected
by the XQA dispatcher. The register-consumer organization is the production
direction on SM90.

The three-run result includes three XQA launches and the native shared
reduction.  The 8-bit lane in these measurements is E4M3 with one scale for
the format pool.  It is a transport/performance proxy and is deliberately not
labeled block-scaled Q8.

## Correctness scope

Two A16 page runs reproduce one ordinary XQA call with relative RMS 0.00336;
the difference is BF16 accumulation order.  A mixed native-NVFP4/A16 producer
test, in which the A16 pages contain the dequantized values from the same Q4
cache, matches the uniform native-NVFP4 result within the same tolerance.

Random-normal synthetic KV gives relative output RMS around 0.12 for the
80%-Q4 mixes.  This is a stress input, not a model accuracy result and not a
logit KL estimate.  Output-distribution KL requires captured per-layer KV from
the target model and an on-policy full-model replay.

The fused block-scaled FP8 read agrees with attention over its explicitly
dequantized BF16 cache at relative RMS 0.00377 and maximum absolute difference
0.003906, within the existing BF16 reduction-order tolerance.  A diagnostic
with a distinct scale on every one of the 16 page tokens verifies an identity
token-to-scale map; this catches the different `ldmatrix.trans` lane ordering
between packed FP4 and unpacked FP8.

On a sparse-outlier synthetic attention case, per-tensor FP8 had relative RMS
0.05976 and maximum error 0.9375.  Direct BSFP8 scaling was worse (0.07473,
1.625), while the fixed-rate searched scale was substantially better (0.03585,
0.6875).  This is why the direct max-derived scale is only a fast epilogue,
not an accuracy claim. The direct encoded residual now drives the mandatory
BSFP8/A16 routing decision. Incorporating the searched-scale residual into the
fused sealer remains future work.

For an outlier-heavy pointwise codec input, per-tensor FP8, MXFP8-32, direct
BSFP8-16, and searched BSFP8-16 had RMS errors 0.03449, 0.03598, 0.03245, and
0.02401 respectively.  Corresponding maximum errors were 1.714, 2.000, 1.714,
and 1.204.  These are codec diagnostics, not model-level KL.

Ragged query indptr plumbing is exercised with query lengths `[4, 2, 3]` and
uniform multi-page active spans with query lengths 48 and 64. The ragged call
matches separate per-request invocations, and the multi-page cases catch the
active-span/page-boundary mask error that a short speculative test cannot see.

## Remaining work

- Move page-list maintenance into the serving scheduler so the attention-time
  scatter becomes an incrementally maintained format index rather than a
  per-call operation.
- Replace the prototype's padded embedded sidecar (currently two A16 cache
  footprints) with allocator-owned A16 and BSFP8 pools so compression releases
  physical KV capacity instead of only reducing read traffic.
- Fuse the nine-candidate accuracy search into the page-seal epilogue.  The
  direct-scale epilogue already writes preallocated payload and scale pools.
- Validate contexts above 4K when the SM120 worker has sufficient free memory.
- Exercise the vLLM metadata-builder path in a normal model server for decode,
  continued-prefill, and draft verification. The operation and wrapper tests
  pass, but that is not an end-to-end scheduler result.
- Run Gemma-4-12B text/vision captures and report elementwise attention error,
  layerwise logit drift, teacher-forced sequence KL, and on-policy sequence KL.
