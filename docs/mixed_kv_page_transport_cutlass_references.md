# Mixed KV page transport: what the in-tree references say to do

Every stage of a streaming (de)quantization prologue or epilogue for paged attention has
a shipped NVIDIA or FlashInfer implementation in this repository. This document maps each
stage to that implementation, states the cost model that follows from it, and fixes the
acceptance targets. Paths are relative to the repository root; CUTLASS is pinned at
`3rdparty/cutlass` (`b46b16d`).

## 0. Cost model

Per 16-token × `D`-channel page, one KV head, `D = 128`:

| Format | payload | scales | bytes/page | decode ALU |
| --- | ---: | ---: | ---: | --- |
| A16 | 4096 B | 0 | 4096 | none |
| block-scaled E4M3 | 2048 B | 128 B | 2176 | 1 `cvt` + 1 FMA per pair (sm90); 1 packed `cvt` + 1 FMA per pair (sm120) |
| block-scaled E2M1 | 1024 B | 128 B | 1152 | `prmt`-LUT x8: ~6 instr per 8 elems (sm90); 1 packed `cvt` + 1 FMA per pair (sm120) |

Two consequences that must govern the warp topology:

1. **Load cost is O(1) instructions per page and independent of format** when the copy is
   a bulk/TMA copy issued by one elected lane. There is therefore no format-dependent
   reason to vary *load* warps. The load side is not where FP4 and FP8 differ.
2. **Decode cost is O(elements) and is the only format-dependent cost.** The element count
   per tile is the same for all three formats; the per-element instruction count is
   0 / small / larger for A16 / E4M3 / E2M1. What must scale with the format is the
   **conversion** throughput: the number of converter warps (sm90) or the acceptable
   per-fragment decode budget (sm120). FP4 needs roughly twice the conversion issue
   slots of FP8 to keep the same A16 consumption rate — at one quarter the DRAM bytes.

A tile is 64 tokens (4 pages). At sm90 SS-GMMA consumption rates the FP8 K+V tile needs
~200 issue-cycles of decode against ~960 cycles of DRAM time per SM; the FP4 tile needs
~400 against ~480. Both fit only if decode runs **concurrently** with the copy engine and
the MMA, in warps that are not the MMA warps. Hopper predates this prologue shape and
will always spend more of the compressed-format headroom on decode than Blackwell does;
it is still faster than A16 in any credible implementation at ≥16k context, because the
kernel remains DRAM-bound once the decode overlaps.

## 1. Loading compressed pages: TMA / bulk copy, one elected lane

Reference:

- `3rdparty/cutlass/include/cute/arch/copy_sm90_tma.hpp:327` (`SM90_TMA_LOAD`) and
  `include/cute/atom/copy_traits_sm90_tma.hpp:1038` (`to_CUtensorMapDataType`) — TMA
  tensor maps are typed by element width; `CU_TENSOR_MAP_DATA_TYPE_UINT8` describes a
  compressed page as a byte tensor. `csrc/xqa/tensorMap.cpp:12` already accepts it.
- `sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp:428-446, 711` — the **scale
  tensor gets its own TMA descriptor** (`tma_load_scale`) and its bytes are added to the
  stage's transaction count (`tma_transaction_bytes + TmaTransactionBytesExtra`); the copy
  is issued under `if (cute::elect_one_sync())`.
- `include/flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh:241-290` — the
  in-tree sm120 NVFP4 decode: *"IO gather: scalar scales → fence → expect_tx + bulks."*
  One IO warp; scales by `__ldg` into smem, then `cp.async.bulk` for the payload.
- `csrc/xqa/mha_sm90.cu:2255-2260` — upstream XQA's own `KVTilePartLoader::loadData`
  issues one `tma::loadAsync` per page from a `CUtensorMap`. This is the baseline being
  measured against.

Rule: the mixed path must issue **one bulk copy per (page, operand) for payload and one
for scales**, from one elected lane, with `arrive_tx` transaction bytes equal to the
compressed byte count. Per-lane `cp.async` of 8–16 B with per-block 64-bit address
arithmetic (`copyMixedPartialHeadsAsync`) is not an implementation of this stage; its
instruction count scales with element count, i.e. it is 4× more expensive for FP4 than
for A16 per byte moved, and it is the dominant cost in the measured sm90 kernels.

Per-page format dispatch is compatible with TMA: choose the tensor map (A16 / E4M3 /
E2M1) per page; a 4-page tile is at most 4 + 4 copy issues instead of 1. Under
`MIXED_PAGE_STATIC_FORMAT >= 0` it is exactly 1 + 1.

## 2. Pipeline and barriers: `PipelineTmaAsync`

Reference:

- `3rdparty/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp:271` (`PipelineTmaAsync`)
  and `media/docs/cpp/pipeline.md` — producer `acquire → issue TMA → (transaction bytes
  complete the barrier)`; consumer `wait → consume → release`. Stage ownership is
  CTA-local mbarriers. No named-barrier round trip, no `__syncthreads`, no grid or host
  synchronization inside the mainloop.
- `3rdparty/cutlass/examples/48_hopper_warp_specialized_gemm/` and
  `54_hopper_fp8_warp_specialized_gemm/` — the producer warpgroup is one warp doing
  useful work; the rest of its warpgroup is register-deallocated (`setmaxnreg`).

Rule: upstream XQA sm90 already has this shape (`kBar`/`vBar` produced/consumed with
`arrive_tx`). Keep it. Add stages only by adding smem buffers, not by adding
synchronization.

## 3. Where dequantization happens

### sm90

Reference:

- `sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp:869-899` — CUTLASS's mixed-input
  Hopper collective converts the narrow operand into registers **inside the consumer
  warpgroup** (`copy_tensors_MK` → transform → RS-GMMA). This is correct for GEMM,
  where each `m64nNk16` WGMMA with `N ≥ 128` takes ≥64 cycles and hides ~14 instructions
  of decode per lane.
- For decode attention with `SWAP_AB`, `N = ctaNbQHeads` is 8–32 and the same WGMMA takes
  ~8 cycles. In-consumer decode is then 2–3× longer than the MMA it feeds and serializes
  with it. The CUTLASS RS pattern is therefore the wrong template for this kernel's
  consumer; the **SS** collective is the right one:
  `3rdparty/cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp`.

Rule: dequantize in **dedicated converter warps** that read the compressed stage and
write the A16 stage in the swizzled layout the unchanged SS-GMMA consumer already reads.
`convertWarpsPerOperand` is a function of `MIXED_PAGE_STATIC_FORMAT`: 0 for A16, `n` for
E4M3, `2n` for E2M1, with `n` chosen so decode issue-cycles per tile stay below the
tile's DRAM time (Section 0). The consumer code path is byte-for-byte upstream XQA.

### sm120

Reference:

- `include/flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh` and its
  `model/scale_convert.cuh` — packed K/V plus scale factors are decoded in the consumer's
  fragment load with native `cvt` and fed to `mma.sync` from registers.

Rule: the register-fragment consumer in `csrc/xqa/mha.cu` (`loadMixedKPageFragment`,
`loadMixedVPageFragment`) is the correct shape and is kept. Improvement on sm120 comes
from Section 1 (bulk loads) and Section 5 (occupancy), not from moving the decode.

## 4. Conversion primitives

Reference:

- `3rdparty/cutlass/include/cutlass/numeric_conversion.h:3833-3835` (`_e2m1_to_bf16_x8`)
  and `:3956-3958` (`_e2m1_to_half_x8`): eight E2M1 values in one 32-bit word to four
  packed A16 pairs via `prmt` lookup, ~6 instructions. This is the sm90 E2M1 decoder.
  Nothing on sm90 may decode E2M1 one byte or one pair at a time.
- `3rdparty/cutlass/include/cutlass/detail/collective/mixed_input_utils.hpp:555, 863, 1156`
  (`UseScaleLookupTable`): block scales are applied once per 16-value block, hoisted
  out of the per-pair path.
- sm120: `cvt.rn.bf16x2.e2m1x2` / `cvt.rn.bf16x2.e4m3x2` (`csrc/xqa/mhaUtils.cuh:372-393`,
  `__CUDA_ARCH__`-guarded). One instruction per pair; nothing to import.

`csrc/xqa/mhaUtils.cuh:421` (`convertE2M1x8ToA16`) already wraps the CUTLASS x8 helpers.
Every sm90 E2M1 decode site must call it on a 32-bit word obtained by `ldmatrix`/`lds.32`,
never on individual bytes.

## 5. Occupancy and compile shape

Reference:

- `media/docs/cpp/efficient_gemm.md` (warp specialization, persistent kernels):
  latency is hidden by the pipeline depth and by other resident CTAs; a one-CTA-per-SM
  kernel exposes every pipeline bubble.
- Examples 48/54/55: the producer warpgroup uses `setmaxnreg` to release registers so the
  CTA fits alongside a second CTA or a deeper pipeline.

Rule: the mixed sm90 CTA must not force `__launch_bounds__(..., 1)`. Converter warps are
register-light (they hold one 16-value block at a time); size the smem stages and register
budget so at least two CTAs are resident, or use the baseline's occupancy as the floor.
Fully-unrolled containers of fragments indexed by a runtime page format
(`Vec<Vec<RegMatAFrag, M>, 2>` under `#pragma unroll` over `nbGmmaInstK` with a
per-iteration format switch) are not a compile-time shape any reference uses; they
produce the multi-hour `ptxas` runs observed on sm90 and are prohibited.

## 6. Quantization / page sealing (producer side) — complete

The routing filters — single-pass, linear-time, non-recurrent page statistics that
separate high-outlier-loss pages (image and poorly-normalized multimodal embeddings)
from low-outlier-loss pages (normalized text-vocabulary embeddings) — are implemented
and validated:

- `csrc/fp4_kv_quantization.cu:461` (`mixed_kv_route_rows_kernel`), `:535`
  (`mixed_kv_finalize_route_kernel`), `:394` (`mixed_kv_reset_reused_pages_kernel`)
- `flashinfer/quantization/kv_cache_fp8.py:232` (`seal_mixed_kv_pages_cuda`),
  `routing_thresholds`, `page_router_stats`, `page_router_partials`
- Exactness and graph-replay tests pass on sm90 and sm120.

This stage is finished and is not to be reopened by transport-side work. The only
interface it exposes to the consumer is `page_format` (one byte per page) and the
payload/scale tensors described in `include/flashinfer/attention/page_transport.cuh`.

## 7. Measurement discipline

Two conditions on the shared test hosts silently invalidate numbers:

1. **Import path.** The venvs carry an editable install of another checkout.
   `python -m pytest` from a checkout imports that checkout (cwd is on
   `sys.path`), but `python benchmarks/foo.py` puts the *script directory* on
   `sys.path` and imports the editable one — the JIT then compiles the other
   tree's `csrc`. Every measurement process must pin `PYTHONPATH` to the
   checkout under test and print `flashinfer.__file__` and
   `flashinfer.jit.env.FLASHINFER_CSRC_DIR`; the benchmark does this itself.
   Confirm with the `-c` source paths in the workspace's `build.ninja`.
2. **Co-tenant time slicing.** When another process holds the GPU (e.g. a
   `VLLM::EngineCore` at 99% SM), kernels are time-sliced with it once a burst
   exceeds roughly half a millisecond, and every sustained number is inflated
   ~2x — memory-bound and compute-carrying kernels alike, with the threshold
   depending on the burst length. Check `nvidia-smi pmon -s u`. On such a host
   report short bursts (`--no-cuda-graph --repeats 3 --trials 15`) and treat
   graph-replay medians as lower-quality; clocks and throttle reasons are not
   the explanation (verify with `nvidia-smi --query-gpu=clocks.sm,...`).

## 8. Acceptance targets and current state

B = 17, S = 4096, H_kv = 8, D = 128, K+V, BF16 math. Bytes: A16 285 MB, E4M3 152 MB
(payload + scales), E2M1 80 MB.

| Device | A16 XQA (measured) | achieved BW | E4M3 target | E2M1 target |
| --- | ---: | ---: | ---: | ---: |
| H200 / sm90 (~4.8 TB/s) | 88-91 µs | 3.2 TB/s | ≤ 60 µs | ≤ 35 µs |
| RTX 5090 / sm120 (~1.8 TB/s) | 177 µs | 1.6 TB/s | ≤ 95 µs | ≤ 55 µs |

Targets are the A16 kernel's *achieved* bandwidth applied to the compressed byte count,
with a 5–10% allowance for decode on sm120 and ~15% on sm90. A compressed-format result
is accepted only when (a) it is bit-exact against explicit A16 expansion, (b) the
asserted kernel family matches the architecture (see `mixed_kv_page_transport_backends.md`),
and (c) the same-run A16 baseline is within 10% of the figure above, so that the
compressed result can be expressed as achieved bandwidth rather than as a ratio to a
drifting baseline.

### sm90 state (TMA loader + converter warps, this branch)

H200 burst measurements (Section 7 discipline; co-tenant present), q = 1, all 34
transport cases bit-exact:

| mode | before (cp.async gather) | now | tiles/s note |
| --- | ---: | ---: | --- |
| static A16 through the mixed kernel | 92 µs | 88-91 µs | reference cadence |
| static E4M3 | 318 µs | ~104 µs | 3.1x faster; parity with A16 |
| static E2M1 | 455 µs | ~111 µs | 4.1x faster; parity with A16 |
| dynamic mixed pages | 477 µs | ~148 µs | 3.2x faster |

What was fixed, in order of measured effect: (1) per-lane `cp.async` gather replaced by
one elected-lane TMA per page (byte-typed maps for compressed pages, whole-head rows
once per tile); (2) the load warp no longer waits for its scale copies before issuing
the TMA (that serialized two DRAM round trips per part and alone accounted for the
E4M3 deficit); (3) the grid is split by an occupancy-aware chooser so the ~83 KB CTA
(two per SM) fills every SM without a straggler wave — 136 CTAs at two per SM had been
running on 68 SMs.

What remains: the compressed kernels now run at the same per-tile cadence as the A16
kernel (~1.4 µs per 64-token tile per CTA), which is latency- rather than
bandwidth-bound; K stage depth 3 and 4 measured identical to 2 (4 also drops to one CTA
per SM). Turning the byte reduction into wall-clock therefore needs the cadence itself
shortened — more tiles in flight per SM through the softmax/X hand-off, not deeper K/V
staging — and that change applies to the A16 kernel equally. Compile time for
`mha_sm90.cu` is ~4 s.
