# Mixed KV page transport: one backend per architecture

This document is normative for `csrc/xqa/` and `include/flashinfer/attention/` on the
mixed-KV-page branch. It exists because of the following review finding, quoted exactly:

> **sm90 ≠ sm120, and it keeps trying to unify them.** On sm120 `mma.sync` takes
> register operands and `cvt.rn.bf16x2.e2m1x2` is native → in-fragment decode is
> correct and *already works* (79.6 vs 120 µs). On sm90 `wgmma` reads smem for free
> and FP4 decode is software → expand-to-A16-smem in dedicated converter warps and
> leave the baseline SS-GMMA consumer byte-for-byte untouched. Its current RS-GMMA
> sm90 path puts ~14 instr/lane of shuffle+cvt+FMA in front of each ~8-cycle
> `m64n16k16` wgmma — the decode is 2–3× longer than the MMA it feeds, and the FP4
> arm of `loadMixedKTileFragment` **still reads smem byte-by-byte with the scalar x2
> cvt** despite the "official x8 substitution" claim (the x8 helper exists in
> `mhaUtils.cuh`; this path doesn't call it).

## Design specification

The producer/consumer data flow, its stall-freedom conditions, and the
invariants an implementation must satisfy are fixed in
`mixed_kv_page_transport_dataflow.md`. Kernels are written once from that
specification and reviewed by reading against it; measurement confirms
conformance, it does not tune.

## Rule

Every CUDA compute capability that differs in any instruction used by the attention
prologue, mainloop, or epilogue is a **separate backend**. sm80, sm86, sm89, sm90,
sm100, sm103, sm120, and any later revision are distinct where their ISA, tensor-core
operand sourcing, copy engines, or conversion instructions differ. A kernel is shared
between two architectures only when the *identical* source compiles to the intended
instruction sequence on both; that is a property to be checked, not assumed.

Concretely, for this branch:

| Concern | sm90 (Hopper) | sm120 (Blackwell consumer) |
| --- | --- | --- |
| MMA operand source | `wgmma` A from **smem** (SS) or registers (RS); B from smem | `mma.sync` A and B from **registers** |
| Bulk copy | TMA tensor maps (`cp.async.bulk.tensor`) — one elected lane per copy | `cp.async.bulk` (non-tensor) — one elected lane per copy |
| E4M3 → A16 | `cvt` via f16 pair | native packed `cvt.rn.bf16x2.e4m3x2` |
| E2M1 → A16 | **software** (`prmt` LUT, x8 per 32-bit word) | native `cvt.rn.bf16x2.e2m1x2` |
| Correct place to dequantize | dedicated converter warps → A16 smem stage → unchanged SS-GMMA consumer | inside the consumer's register-fragment load (the path that already wins) |
| Source file | `include/flashinfer/attention/hopper/sparse_mixed_mainloop.cuh` (FA3 host, all query shapes); `csrc/xqa/mha_sm90.cu` (reference) | `csrc/xqa/mha.cu` |
| Bandwidth-side reference in tree | `3rdparty/cutlass/.../sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp` | `include/flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh` |

The two right-hand columns are not converging. Do not port the sm120 register-fragment
decode into the sm90 GMMA consumer; do not port sm90 smem-expansion into sm120.
Shared code between the two is limited to: page-format metadata gather
(`gatherMixedPageFormats`), scale arithmetic (`scaleA16x2`), the conversion helpers in
`mhaUtils.cuh` that are themselves `__CUDA_ARCH__`-guarded, and the host API.

## Dispatch must be explicit and asserted

`flashinfer/jit/xqa.py` chooses which source enters the module. Which kernel executes
for a given (architecture, page format) pair is a fact the test suite must assert, not
infer from timings. Requirements:

1. Every JIT module exports the architecture it was compiled for and the kernel family
   (`mha_sm90` GMMA vs. architecture-neutral `mha`) it will launch; tests read this and
   fail if a measurement is attributed to the wrong kernel.
2. Excluding `mha_sm90.cu` for a format (as `jit/xqa.py` currently does for
   `block_scaled_fp8`) is a **dispatch decision** that must be visible in the module
   URI and in benchmark output. Silent fallback to `mha.cu` on an sm90 device is a
   failure, not a fallback.
3. A performance number for "sm90 FP8/FP4" is only admissible when the asserted kernel
   family is `mha_sm90`.

## What this rules out

- Any change to `mha_sm90.cu` motivated by an sm120 measurement, or vice versa.
- Any single "unified" mixed-input consumer that must be correct on both architectures.
- Any topology argument ("N load warps per operand", "converter warps in the consumer")
  that is not stated per architecture with the instruction it is balancing against.

## What this does not rule out

- Identical **host** APIs, page layouts, scale layouts, and `page_format` semantics
  across architectures. The storage contract in `include/flashinfer/attention/page_transport.cuh`
  is architecture-independent by design.
- Identical **acceptance tests**: bit-exactness against explicit A16 expansion, and the
  roofline targets in `mixed_kv_page_transport_cutlass_references.md`, apply to both.

## sm120 (RTX 5090) state: mixed streams through the XQA host

Measured on an RTX 5090 (B=17, S=4096, 8 KV heads, GQA 4; bursts; 34/34 cases
bit-exact against explicit A16 expansion):

| q per sequence | A16 stock | A16 transport | FP8 | FP4 | mixed |
|---|---|---|---|---|---|
| 1 (batched AR decode) | 179.9 µs | 179.8 | 139.8 | 83.7 | **145.3** |
| 4 (draft verification) | 183.5 | 184.6 | 153.7 | 100.6 | **156.9** |
| 64 (prefill; XQA is not the host for this) | 437.8 | 460.4 | 785.0 | 689.3 | 905.4 |

Two structural changes got the dynamically mixed stream from 1.3× *slower* than
A16 (237 µs) to 1.24× faster:

1. **Page tags are gathered one tile ahead** with a two-deep page-index prefetch
   (`loadPages` rotates `pageIdx ← pageIdxNext`, issues the tag load for the
   landed indices, and requests the indices two tiles ahead; the tag is
   broadcast with shuffles a tile later).  Before, the loading warp — which on
   sm120 is the compute warp — waited on a dependent `page_format[pageIdx]` load
   at every tile part.  Alone this did not change the mixed time (237 → 233 µs);
   it removed a latency chain, not the bottleneck.
2. **Tiles with compressed pages are expanded to A16 in shared memory and fed to
   the stock A16 GEMM** (`MIXED_COMPACT_PAGES=0`, now the default).  The
   register-side "compact" form dispatched on the page tag per (block, page)
   inside the unrolled MMA loop, instantiating all three fragment converters:
   `cuobjdump` showed 17 688 instructions, 647 branches and 338 local loads for
   the dynamic kernel against 6.5–8.3 K instructions, 130–190 branches and
   118–168 local loads for the static-format kernels.  With expansion the dynamic
   kernel is 12 064 instructions / 262 branches, and the mixed stream moves
   172 MB at 1.19 TB/s.  Pure FP8 at q=4 pays 13 % for this (136 → 154 µs);
   mixed streams are the product.

Open on sm120: no prefill-shaped host exists for mixed pages (the FA3 host is
sm90-only); q ≥ 64 through XQA is 1.6–2× slower than A16 and should route to the
sm12x prefill backend once one carries the page-transport contract.

## sm90 (H200) XQA decode host: where the compressed-format time goes

Measured on H200 (B=17, S=4096, 8 KV heads, GQA 4, q=1; bursts; 34/34 bit-exact).
Byte roofline at 4.8 TB/s: A16 59 µs, FP8 32, FP4 17, mixed 36.

### Attribution with the kernel's built-in experiment bits

| mode | production | converters skipped | no compressed TMA | no TMA, no scales |
|---|---|---|---|---|
| A16 transport | 83.7 | 83.6 | 83.8 | 83.7 |
| FP8 | 90.6 | 65.2 | 80.9 | 75.0 |
| FP4 | 94.9 | 61.8 | 86.0 | 80.6 |
| mixed | 108.5 | 73.8 | (n/a) | (n/a) |

Read as a stack for FP4: ≈47 µs is the consumer chain itself (gemm0 → softmax →
P hand-off → gemm1 at 2 CTAs/SM; 45 % of PC samples are barrier spins), ≈15 µs
exposed compressed loads, ≈33 µs exposed conversion.  Mixed-dtype-ness adds
nothing beyond its share of conversion.  E2M1/E4M3 arithmetic is ~10 % of samples.

### Per-tile timeline (MIXED_KV_TRACE, CTA 0, steady state)

| | FP4 | A16 |
|---|---|---|
| consumer cadence per 64-token tile | 1.95 µs | 1.6 µs (bandwidth-bound: 5–7 µs landing) |
| loader TMA issue per tile (8 boxes) | 1.0–1.7 µs | 1.0 µs |
| TMA issue → bytes landed | ~1.7 µs | 5–7 µs |
| converter ready → done | 1.0–1.6 µs | 0.1 µs |

With two K stages the compressed chain per stage is issue + landing + conversion
≈ 4.3 µs → one tile every ~2 µs: pipeline-depth-bound, not ALU- or bandwidth-bound.

### Structural changes made, and what each measured

1. **Converter warps copy compressed pages themselves** (cp.async, one warp per
   page, warp-contiguous chunk ownership, scales included); the loader's TMA is
   A16-only.  Loader issue per tile fell 1.0–1.7 → 0.2 µs.  cp.async landing
   under the saturated memory system measured **~1.9 µs**, so with two stages the
   cadence became landing + expansion = 2.8 µs — worse.  Landing latency at
   3.4 TB/s of aggregate traffic is a fact to cover with depth, not to remove.
2. **Copies two tiles ahead → three K and V stages.**  +32 KB of shared memory
   (84 → 114 KB) cost the second CTA per SM (limit 112.5 KB), which cost more
   than it gained; with the P ring 4 → 2 (110.6 KB) two CTAs fit again — but
   only after the register budget was fixed (below).  Converter period is then
   ≈ (landing + expansion) / 2 ≈ 1.5 µs, at gemm0's floor.
3. **Register budget.** The kernel has no `setmaxnreg`; every warp lives with the
   launch allocation, so `__launch_bounds__` minBlocksPerSM *is* the register
   budget.  The mixed build declared 1 (and `OPTIMIZE_FOR_LATENCY=1` drops the
   floor entirely), so the allocation floated to 58 registers/thread and two
   640-thread CTAs (≤ 51) no longer fit; the driver then also chose a one-block
   shared-memory carveout.  Declaring `(640, 2)` unconditionally gives 48
   registers and both occupancy limits = 2.  The SPEC_DEC (q > 1) variant wants
   226 registers statically and spills heavily (`LDL` hot in PC sampling; mixed
   q=4 935 µs vs FP4 276) — a separate item.
4. **Converter hazard barrier**: the read-before-write hazard of in-place
   expansion is intra-warp (a warp covers one page's (token, head-part) lanes),
   so the 128-thread named barrier per tile per operand is a `__syncwarp`.

Shared-memory budget (2 CTAs/SM ⇒ ≤ 112.5 KB): K/V stage 16 KB each, P entry
4 KB, Q 8 KB, scales ring 1 KB/entry.  (K3, V3, X2) and (K3, V2, X4) both fit;
(K3, V3, X4) does not.  The P ring was sized 4 because at 2 the two consumer
groups run in lock step; which trade wins is measured, not assumed.

### Converter budget, resolved by the full profile

Level-3 trace (converter sub-stamps, fp4, steady state): landed → ready 0.17 µs,
expansion 1.69, proxy fence + `produced` arrive 0.45, copy issue 0.64 (was 1.25
before the lean issue path), commit → next tile landed 0.09.  Sum ≈ 3.0 µs =
the cadence; landing is fully hidden by the two-ahead copies.

Kernel-level scheduler statistics (`ncu --set full`): Issue Slots Busy 58 %,
IPC 2.78, 9.2 active warps per scheduler, 1.7 eligible, **13.3 cycles per issued
instruction per warp**.  The converter's ~230 instructions per lane per tile at
13 cycles each *is* its fair share of a busy scheduler.  What keeps the
scheduler busy: the barrier retry loops — `SYNCS.PHASECHK … TRYWAIT`,
`SYNCS.CCTL.IV`, `BRA` — are *issued* instructions (stall reasons `selected` /
`not_selected` / `long_sb` on the try-wait result, not `sleeping`), ~45 % of all
samples.  XQA's sm90 `MBarrier::poll` spins on `mbarrier.try_wait` with a
suspend hint the implementation caps; waiting warps steal issue from the
converter warps they are waiting for — or so the sample mix suggested.  A bounded
exponential `__nanosleep` backoff (32 → 512 ns) in the poll loop was measured and
**rejected**: fp4 96.2 → 101.1 µs, mixed 114.4 → 124.3, issue-active 58 → 66 %,
`sleeping` still 0.00, converter expansion unchanged at 1.6 µs.  The retry loop
is not what paces the converter.

**Where this leaves the sm90 decode host.**  FP8 (≈100 converter instructions per
lane per tile) and FP4 (≈200) expand in the same ~1.6 µs, so the converter's
wall time is not its ALU count either; it is the cost of the warp group's
shared-memory writes, proxy fence and barrier traffic under a scheduler shared
with 36 other warps.  With the converter warp group pacing the pipeline the
cadence is ≈ 2.9 µs/tile → FP8 91 / FP4 96 / mixed 114 µs at q=1 against A16 83
(consumer floor ≈ 62).  The remaining structural levers are more converter lanes
per operand (the IO group's two Q warps and two loader warps are idle most of
the tile) or a four-warp-group layout (fewer threads → 64 registers each,
loader merged into the converters).  Both are real work with a bounded payoff
(~20–30 %); neither is a tuning knob.

### P0.3 — consumer trace, tile-time multiplier, RT constants (H200, 2026-09-03)

Method: `MIXED_KV_TRACE=1` build (`FLASHINFER_EXTRA_CUDAFLAGS`), bench shape
(B=17, S=4096, 8 KV heads, GQA 4, q=1), one launch per measurement under the GPU
lock, 10 launches per mode, cleanest launch reported.  The trace build now
records per-CTA `%globaltimer`/`clock64` entry and main-loop-end stamps in a
device buffer printed once by the last merging CTA (per-CTA `printf` at exit
was measured to hold each slot for ~0.5 ms and destroyed the wave structure);
`MIXED_KV_TRACE_TAIL=1` traces the last 8 tiles of CTA 0 instead of the first.
Driver: `benchmarks/xqa_mixed_trace_once.py`; parser
`benchmarks/microbench/parse_xqa_trace.py`.  The SM clock measured from the
stamps was 1.79–1.80 GHz (not the 1.98 GHz nominal; VLLM co-tenant present at
100 % SM on device 0 throughout).  Cycles below are at that clock (1000 cyc ≈
0.56 µs).

**(a) Consumer chain, CTA 0 steady state (tiles 2–7 medians, cycles).**

| segment | FP4 production | FP4, converters skipped (EXPERIMENT=1) | A16 transport |
|---|---|---|---|
| s1−s0 gemm0 8× HGMMA (2 commit groups) | 738 | 589 | 542 |
| s2−s1 colMax sync + softmax (2 mbarrier RT) | 800 | 609 | 458 |
| s3−s2 xBar.consumed wait + X store + fence + arrive | 540 | 544 | 400 |
| T_g0 = s3−s0 | 2048 | 1762 | 1407 |
| s5−s4 gemm1 wait for X | 1140 | 566 | 130 |
| s6−s5 rescale (2 mbarrier RT) | 514 | 473 | 400 |
| s7−s6 PV 4×(2 HGMMA + commit + wait) | 740 | 636 | 544 |
| gemm1 work = s7−s5 | 1261 | 1123 | 944 |
| gemm0 K-wait = s0(t)−s3(t−1) | 622 | 192 | 2930 |
| X hand-off lag = s5(t)−s3(t) | 67 | 69 | 1204 |
| cadence s0(t)−s0(t−1) | 2672 (1.49 µs) | 1960 (1.09 µs) | 4330 (2.40 µs) |

Cadence hypothesis: **max(), gemm0-bound**, not lock-step sum.  s3−s2 is
identical (540 vs 544) whether gemm1 is busy or not and gemm1 has released
X(t−2) 3–4k cycles before gemm0 needs the buffer, so gemm0 never waits on
gemm1; s5−s4 is gemm1 waiting for gemm0's X (it observes X 67–123 cycles
after gemm0's arrive).  cadence = T_g0 + K-wait; gemm1 has 840–1400 cycles of
slack per tile.  Consequences: [11] (X depth 3) is not built; gemm1 levers
[0]/[1] have no wall effect until T_g0 + K-wait < ~1150 cyc; gemm0 levers
[2]/[4]/[6] and the converter period are the only wall-visible consumer
levers.  Consumer floor (converters skipped) is 1.09 µs/tile, not 1.5.
Converter pacing in production: gemm0 gets K 174–311 cycles after the K
converter's last expansion stamp (fence + arrive + RT); K-wait 622–824.

**(b) Tile-time multiplier and tail (FP4 production, cleanest launch, main-loop
end 96.5 µs; bench 96.0–96.5).**  680 CTAs on 264 slots.  Wave 1 (264 CTAs)
lifetime median 29.5 µs but bimodal: the two co-resident CTAs of an SM differ
by 10.5 µs median (fast ≈ 24, slow ≈ 34; CTA 0 is a fast one) — the converter
groups of the two CTAs share the SM unfairly.  A16 pairs differ by 0.4 µs,
converters-skipped by 1.2 µs.  Wave 2 (263 CTAs, start 26–44 µs) lifetime
27.6, wave 3 (153 CTAs, start 54–70 µs) 25.4 (fewer co-runners → faster).
Per-SM last end: p10 80.7, median 82.9, p90 95.0, max 96.5 µs → the 39-tile
model (3 rounds × 27.6 = 83) describes the median SM; the wall is 13.6 µs
(14 %) later because the last 153 CTAs are handed slots as they free (50–75 µs)
and the last ones start at ~70 µs.  Perfect balance (Σ lifetimes / 264) would
be 71.7 µs.  Fixed cost per CTA: prologue to first K ready 4.36 µs FP4
(7.75 µs A16: TMA landing under full-wave DRAM contention), epilogue after the
last PV 0.11 µs; per-CTA lifetime = 13 × cadence + ~4.5 µs, so 3 rounds pay
~13 µs of fills.  Converters skipped: wall 63.3 µs = wave-1 lifetime 19.8 ×
~3 + tail; Σ/264 = 48.2 µs (31 % quantization loss).  A16: waves 36.3 / 25.5
/ 18.4 µs (DRAM-bound: lifetime scales with co-runners), 112 two-CTA slots
idle from ~65 µs.

**(c) Zero-code calibration `XQA_NB_SUB_SEQ=3` vs default 5 (production build,
5 repeats × 5 trials, two rounds).**  A16 82.6/82.7 → 82.8/82.7 (×1.00), FP8
91.4/91.5 → 96.9/96.7 (×1.059), FP4 96.5/96.0 → 103.3/103.0 (×1.072), mixed
114.8/114.7 → 127.2/126.8 (×1.107).  Predicted 44/39 = 1.128 for a pure
tile-time model; with the measured ~4.5 µs per-CTA fill the model gives
(44 T + 2 F)/(39 T + 3 F) ≈ 1.05 plus tail differences, matching 1.06–1.11.
The wave model holds once the per-CTA fill is included; [8]'s payoff is 2 of
3 fills (~9 µs) plus the 13.6 µs slot-assignment tail.

**(d) RT constants (`benchmarks/microbench/sm90_rt_constants.cu`, nvcc
sm_90a, 128-thread CTA, 4000 iterations, cycles per iteration).**

| primitive | 1 CTA/SM | +4 co-resident spinning warp groups | +9 (40 warps/SM, XQA-like) |
|---|---|---|---|
| mbarrier arrive(release) + try_wait(acquire) poll, 128 threads | 87 | 88 | 140–154 |
| bar.sync id, 128 | 20.7 | 20.7 | 21 |
| __syncwarp | 2 | 2 | 2 |
| mbarrier arrive only (128 arrivals/phase) | 14 | 17–19 | 17–20 |
| wgmma.fence + 1× m64n8k16 SS bf16 + commit + wait_group 0 | 57.7 | 57.7 | 57.7 |
| 4× m64n8k16 + commit + wait | 113.5 | 113.6 | 113.6 |
| 8× m64n8k16 + commit + wait (+18.3 cyc per extra MMA) | 185.5 | 185.6 | 185.6 |
| 2× atomicMax smem (4 lanes/warp) | 26 | 26 | 26 |

In the kernel the two-RT segments measure 452–556 cycles (rescale) → loaded
RT ≈ 200–230 cycles; replacing an mbarrier RT with bar.sync saves ~120–200
cycles per site under load (~65 isolated).  gemm0's 8 HGMMAs take 589–738
cycles against a 185–227-cycle microbenchmark floor: ~400 cycles of
descriptor arithmetic/issue on the critical path per tile (candidate lever).
