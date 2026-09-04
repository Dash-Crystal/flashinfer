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

### P0.4 [12] — converter SASS class counts and PC sampling (nkcut2 H200, static fp4 / fp8 builds)

Build: bench modes `fp4` (`MIXED_PAGE_STATIC_FORMAT=2`) and `fp8` (`=1`), q=1, B=17, S=4096, 8 KV
heads; `xqa_mha_sm90.cuda.o`: REG 48, STACK 0, 3336 (fp4) / 3248 (fp8) SASS instructions.  Source
attribution from a `FLASHINFER_JIT_LINEINFO=1` build whose SASS is byte-identical (address + text)
to the production build (`nvdisasm --print-line-info-inline`); regions = any inline frame inside
`expandPackedStage` (:2734-2819) / `issueCompressedPageCopies` (:2538-2607), K/V by the outermost
converter-loop line.  ncu 2025.3.1 `--set full`, a 4x finer `--warp-sampling-interval 4` rerun, and a
`--clock-control none` run (1.80 GHz effective, 95.8 us = production duration) agree within 2-3
points; the fine run is quoted.  ncu "Instructions Executed" = 34816 (= 4 warps x 8704 tiles) for
every expansion instruction: static count = dynamic count, no branch inside the expansion is taken.
Tool: `benchmarks/mixed_kv_converter_pcsample.py` (sass / sample subcommands).

**SASS per lane per tile (K converter; V identical +-1).**

| region | fp4 | fp8 |
|---|---|---|
| `expandPackedStage` | **405**: PRMT 96, LOP3 78, SHF 28, HMUL2 32, HADD2 4, F2FP 6, LDS 4 (U8 tag, scale word, 2x LDS.128), STS.128 8, IMAD 104, CS2R 14, BRA 5 (+5 BSSY, +5 BSYNC), UMOV 8, FMUL 4, misc 6 | **287**: F2FP 70 (32 F16.E4M3.UNPACK_B, 32 BF16.F32.PACK_AB, 6 scale), HADD2.F32 68, HMUL2 32, LOP3 15, SHF 4, LDS 6 (tag, scale, 4x LDS.128), STS.128 8, IMAD 31, CS2R 14, BRA 6 (+5, +5), LEA 4, misc 22 |
| `issueCompressedPageCopies` per call (executed / static) | 71 / 73: LDGSTS 3, LDS 8, IMAD 24, LOP3 10, IADD3 7, SHF 4, VIADD 4, ULDC 3, LDC 2, ISETP 2, BRA 2 | 90 / 92: LDGSTS 5, LDS 8, IMAD 20, IADD3 10, LOP3 9, VIADD 7, ULOP3 5, ULEA 5, USHF 3, ULDC 3, LEA 2, LDC 2, misc 11 |
| `consumed.wait_parity` before the copies | 13 (K) / 23 (V, incl. retries) | 14 / 25 |
| loop overhead (DEPBAR wait_group, syncwarp, `kLoadReady.arrive_and_wait` incl. 6-9 retry iterations, fence, `produced.arrive`, commit) | 61 (K) / 53 (V) | 53 / 47 |
| K converter total per tile | 565 | 460 |

fp4 expansion attribution (405): cutlass `_e2m1_to_bf16_x8` LUT decode, 8 calls = **259** (96 PRMT,
64 LOP3, 32 IMAD.SHL, 31 `IMAD.U32 R,RZ,RZ,UR` constant re-materialisation — one set per call because
48 registers cannot keep the LUT constants live —, 24 SHF, 8 UMOV, 4 IMAD.MOV); HMUL2 32; STS.128 +
swizzled store addressing 34 (8 STS, 18 IMAD, 7 LOP3, 1); packed-row/scale LDS + swizzle addressing
25; per-block `isFP4` predication + `LdGrain{}` zeroing 25 (14 CS2R, 4 BRA, 4 BSSY, 3 IMAD.MOV — the
predicate survives a static format because the bad-page tag is still runtime); scale prep 13; lane
index prep 5; misc 8.  The plan's "~140 essential / 48 PRMT" assumed a 6-SASS LUT; the cutlass LUT
is 32 SASS per 8 values, so essential (decode + HMUL2 + LDS + STS) is ~303 and non-essential ~100.
fp8 (287): per-pair chain 5 SASS x 32 pairs = 160 (the [16] target), store 34, LDS/addr 30,
predication/zeroing 26, scale 14, misc 23.

**Stall shares, all samples, K converter (V within 3 points).**

| PC range | not_selected | selected | wait | math | short_sb | no_inst | dispatch | branch | mio | long_sb | lg / barrier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| expansion fp4 | 25.5 | 22.3 | 18.3 | 15.4 | 7.3 | 7.0 | 2.5 | 1.7 | 0.1 | 0 | 0 |
| expansion fp8 | 19.3 | 23.8 | 16.2 | 6.8 | 20.7 | 5.2 | 1.8 | 3.7 | 2.6 | 0 | 0 |
| copy-issue body fp4 | 26.4 | 13.7 | 19.9 | 11.2 | 14.5 | 3.4 | 6.1 | 1.8 | 0.9 | 2.1 | 0 |
| copy-issue body fp8 | 18.1 | 12.5 | 13.8 | 6.6 | 30.7 | 0.4 | 3.3 | 0.9 | 4.9 | 8.9 | 0 |
| consumed.wait_parity K / V | 21-25 / 12-14 | 12-16 / 6-10 | 22-27 / 12-14 | | | | | | | 20-30 / 50-66 | |
| loop overhead fp4 / fp8 | 7.6 / 5.4 | 3.4 / 4.4 | 7.6 / 7.3 | | | | | | 1.0 / 3.5 | 71.9 / 75.7 | |

Hot spots: fp4 — ISETP after the format `LDS.U8` (138 samples, 64 % short_sb: the tag load gates
every block's predicate), CS2R zeroing 14.6 samples/instr (40 % short_sb: WAR on registers still
being read by the previous STS.128), STS.128 15/instr (35 % wait), PRMT/LOP3/SHF 7.7-8.9/instr
(not_selected 27-28 %, math 16-23 %: ALU-pipe throttle), HMUL2 6.2/instr.  fp8 — `F2FP.F16.E4M3`
first use after LDS.128 (short_sb 34 %), CS2R 19.4/instr (51 % short_sb), STS.128 13/instr (mio
34 %), HMUL2 10.5/instr (not_selected 43 %).  Copy issue: the ISETP after the metadata LDS carries
83-96 % short_sb (106 / 215 samples).

**Converter time split (share of K-converter samples, fp4 / fp8):** expansion 54.8 / 42.4 %, copy-issue
body 13.6 / 22.4, consumed.wait 3.0 / 3.2, loop overhead 23.8 / 24.2 (kLoadReady wait 10.4 / 8.3,
`produced.arrive` 4.7 / 5.4, FENCE.VIEW.ASYNC 2.3 / 4.1, wait_group + counter 6.5 / 6.5), prologue +
setup 4.8 / 7.7.  At the 3.0 us fp4 period: expansion 1.64 us (level-3 trace 1.69), copy issue 0.50
(trace 0.64), loop overhead 0.71 (trace 0.17 + 0.45 + 0.09 = 0.71) — sampling and trace agree.
Expansion rate: 1.69 us x 1.98 GHz / 405 = 8.3 cyc per warp-instruction (fp4); fp8 ~8.5.
Non-converter roles issue 1250-1400 warp-instructions per converter tile-equivalent, of which
SYNCS 380-470 + BRA 210-256 + NANOSLEEP 180-220 are barrier retry loops (the arbitration the
converters lose).  `produced.arrive` compiles to one `SYNCS.ARRIVE.TRANS64.RED.A1T0` per warp
(warp-reduced), i.e. 4 transactions per operand per tile, not 128.

**Classification (plan P0.4 rule).** Not MIO/drain-bound: mio + lg <= 2.7 % in the expansion (short_sb
7 % fp4 / 21 % fp8 is LDS-first-use and STS-source WAR exposure, not throttle).  Not
no_instruction-bound (5-8 %).  The expansion is issue-arbitration-bound (selected + not_selected
43-48 %; the converter is eligible about half the time and wins 45-55 % of those cycles) and
dependency/ILP-bound (wait + math 34 % fp4 / 23 % fp8, plus fp8's 21 % short_sb).  Selected paths:
count levers first (every fp4 SASS = 8.3 cyc; audit target is ~100 non-essential of 405: 31 constant
re-materialisations, 25 predication/zeroing, 18 lane-constant store IMADs, 8 UMOV; fp8 [16] -32 plus
~60), then [15] 64-register layout (what stops ptxas from keeping LUT constants live and hoisting
the next block's LDS above the current STS).  [14] store-drain overlap is bounded by the measured
fence + arrive share, 7-9.5 % of converter time (0.2-0.3 us/tile) — J1's -0.1..-0.2, not J2's
-0.45.  Code-size cuts: not selected.  Spin/arbitration reduction (P0.5, [5]) is the remaining lever
for the 20-27 % not_selected.

### P0.5 [34] — fair-share vs latency-bound discriminator (nkcut2 H200, fp4 build, converters skipped)

Build: `-DMIXED_KV_EXPERIMENT=1 -DMIXED_KV_TRACE=1` (expansion skipped, copies still issued and
landed; `kc:done - kc:ready` = 26-31 cyc), `xqa_mha_sm90.cuda.o` REG 48.  q=1, S=4096, 8 KV heads,
bf16, 5x5 bursts under the GPU lock, SM clock 1980 MHz throughout (co-tenant VLLM at 100 % SM).
Cadence = `slot0(t+1) - slot0(t)` (gemm0 warp 0, after `kBar.produced` wait), tiles 2-7 of CTA 0,
median over 25 timed launches; `slot7` (gemm1) agrees within 5 cyc.  Parser:
`benchmarks/parse_xqa_trace.py --mode <bench mode> --skip-launches 6`.  Occupancy of CTA 0's SM
was *measured*, not assumed: a trace-only probe (`mixedKvTraceSmResident[%smid]`, +1 at CTA start,
-1 at end, sampled at tile 4) prints `residentCtasAtTile4`.

| config | CTAs | CTAs on CTA 0's SM (probe) | warps / scheduler | fp4 cadence | A16 cadence |
|---|---|---|---|---|---|
| B=17 `XQA_NB_SUB_SEQ=1` (the plan's "136 CTAs = 1/SM") | 136 | **2** (136 > 132 SMs: the 4 excess CTAs land on SMs 124/128 where CTA 0 runs) | 10 | 1981 cyc = **1.001 us** | 1851 = 0.935 us |
| B=17 default (n=5) | 680, 3 waves | 2 | 10 | 1986 cyc = **1.003 us** | 3443 = 1.739 us (DRAM-bound) |
| B=16 `XQA_NB_SUB_SEQ=2` | 256, 1 wave | 2 | 10 | 1981 cyc = **1.001 us** | 3505 = 1.770 us (DRAM-bound) |
| B=16 `XQA_NB_SUB_SEQ=1` | 128, 1 wave | **1** | 5 | 1662 cyc = **0.839 us** | 1598 = 0.807 us |

The fp4 cadence with data always ready (the converter releases tile t 2267 cyc at 1 CTA/SM /
2716 cyc at 2 CTAs/SM *before* gemm0 passes its wait; gemm0's end(t-1) -> kwait-done(t) is 158 /
190 cyc, gemm1's 214 / 252, xBar produced -> xwait-done 42 / 65: no consumer wait is on data) is
**1.00 us at 10 warps/scheduler and 0.84 us at 5**: halving the warps per scheduler removes 16 %
(320 cyc) of the tile, not 50 %.  Every intra-tile segment shrinks by the same 4-19 % (gemm0 mma
596 -> 529, colMax 660 -> 550, xarr 446 -> 420; gemm1 xwait 624 -> 502, rescale 446 -> 376, mma
592 -> 570), none halves.  **Verdict: latency-bound** (round trips + dependent chains; issue-share
tax <= 0.16 us/tile).  Per the plan's P0.5 mapping: only round-trip-removal levers pay ([4] [0] [2]
[1] [33c] [11]); spin/warp-removal as an issue-relief argument ([9], [38]'s 1-CTA/SM arithmetic,
[15]/[34] justified by "converters starved by spinners") is rejected — at most 0.16 us/tile of the
consumer chain, and 0 of the converter's fair share is recoverable that way.

Two corrections to the plan's numbers: (i) the plan's thresholds ("~1.0 -> fair-share, ~1.7 ->
latency-bound") assumed a 10-warp baseline of 1.5-1.7 us/tile (62 us / 39); the measured
converters-skipped consumer cadence is 1.00 us, so the absolute values must not be read against those
thresholds — the ratio is the discriminator.  (ii) Wall vs cadence (no-trace `EXPERIMENT=1` build,
same bursts): fp4 default 57.1 us = 39 tile-times x 1.0 + 18 us; B=16 n=4 (512 CTAs, 2 waves x 16)
47.2 = 32 + 15; B=16 n=2 (256, 1 wave x 32) 42.85 = 32 + 11; B=16 n=1 (128, 1 x 64 at 0.84)
51.2 (< 64 x 0.84 = 53.7: the trace window, tiles 2-7, runs ~10 % slower than the run average at
1 CTA/SM); B=17 n=1 (136, the 4 doubled SMs pace the wall at 1.0 us) 71.9.  A16 transport (same
build): default 82.3, n=1 84.5, B=16 n=1/2/4: 71.0 / 73.1 / 77.2 (3.4-4.0 TB/s).  The per-wave fixed
cost (fill ~1.1-1.3 us to slot0(t=0), drain, merge) is 6-11 us and is P0.3's multiplier question.
The `MIXED_KV_TRACE=1` build's own wall numbers are unusable (device printf: 950-990 us for every
mode and split).
## H200 host calibration (plan P0.1 / lever [32], 2026-09-03, nkcut2)

Recorded so that later H200 numbers divide by a measured ceiling and name the
kernel they measured.  Artifacts: `benchmarks/bench_xqa_mixed_page_transport.py`
now prints `kernel_family` (mha.cu vs mha_sm90.cu, read back from the dispatch
in `flashinfer/xqa.py`) and `module_uri` on every line;
`benchmarks/probe_hbm_bandwidth.py` is the streaming probe (one kernel or a
short burst per CUDA-event pair).  All timing under `flock /tmp/mixedkv-gpu0.lock`.

### Kernel family per bench mode (B=17, S=4096, 8 KV heads, GQA 4, D=128, bf16)

| mode | q=1 | q=4 |
|---|---|---|
| baseline_a16 (stock bf16 KV) | mha.cu, 108-110 us (2.6 TB/s) | mha.cu SPEC_DEC, 128 us |
| transport_a16 (mixed build, static A16) | **mha_sm90.cu**, 81-83 us (3.45-3.52 TB/s) | mha.cu SPEC_DEC, 135 us |
| fp8 / fp4 / mixed | mha_sm90.cu: 91.4 / 96.2 / 114.8 us | mha.cu SPEC_DEC: 226 / 277-285 / 441-452 us |

The "A16 83 us" reference in the targets table is transport_a16 through
mha_sm90.cu, not the stock mha.cu kernel (which is 1.3x slower at q=1).  At q=4
every mode, A16 included, runs mha.cu with `SPEC_DEC=1 SPEC_Q_SEQ_LEN=4`
(`run_sm90_fp8_mha` is cleared for swap-AB-eligible spans).  Module URIs:
`xqa_input_bf16_kv_cache_bf16_block_scaled_fp8_False_mixed_page_{False|True}_static_format_{-1|0|1|2}_..._use_spec_dec_{False|True}_spec_q_seq_len_{1|4}`.

### Streaming ceiling (probe, kernel-only in 5-kernel bursts, medians of 15)

| footprint | read (Triton stream, 8 KiB/CTA) | copy (r+w, `copy_`) | write (`fill_`) |
|---|---|---|---|
| 80.2 MB (= FP4 bytes) | 23.0 us, 3.49 TB/s | 3.73 TB/s | |
| 151.5 MB (= FP8) | 38.6 us, 3.93 TB/s | 3.99 TB/s | |
| 172.3 MB (= mixed) | 43.0 us, 4.01 TB/s | 4.01 TB/s | |
| 285.2 MB (= A16) | 67.5 us, 4.23 TB/s | 4.19-4.21 TB/s (135.4 us) | 63.3 us, 4.51 TB/s |
| 1-2 GB sustained | 4.31-4.50 TB/s | 4.27 TB/s | 4.63-4.67 TB/s |

Read fit over the four footprints: t = 5.6 us + bytes / 4.61 TB/s.  Theoretical
at the observed 3201 MHz HBM3e clock x 6144 bit = 4.92 TB/s; spec sheet 4.8.
`torch.sum` is a poor probe below 1 GB (two-pass, ~45 us fixed).  Single-kernel
(non-burst) event timing adds the CPU launch gap (5-10 us C++ launches, 10-15 us
Triton/Python), which is why `--repeats 1` bench numbers run 8-20 us high.

**Corrected byte rooflines (measured achievable, replacing the 4.8 TB/s paper
values 59/32/17/36 us): A16 67.5, FP8 38.6, FP4 23.0, mixed 43.0 us.**

**Verdict on the A16 plateau:** the host reads 285 MB in 67.5 us (4.23 TB/s) and
sustains 4.5 TB/s, so transport_a16's 81-83 us (3.45-3.52 TB/s) is kernel-side:
~14 us (17-19 %) above the achievable read at the same footprint - the plan's
"probe shows 4.5+" branch (per-tile TMA issue limiter, A16-side loader item,
fold into [15]/[34]), with the headroom being ~14 us rather than the 24 us the
4.8 TB/s figure implied.  Gate targets (relative to measured A16) are unchanged.

### Co-tenant and clocks

`VLLM::EngineCore` (PID 1373615, 27.8 GB, 99 % SM in `nvidia-smi pmon`, ~120 W
draw, i.e. an idle spin) is resident for days.  Clocks at max and locked
(SM 1980 MHz, HBM 3201 MHz, throttle reasons 0x0) before and after every run.
Burst-length experiment (graph replay, per-kernel medians):

| kernel | burst | per-kernel time |
|---|---|---|
| copy 2 GB | 1 / 2 / 3 kernels = 0.96 / 1.92 / 2.88 ms | 960 / 959 / **1787 us** |
| mixed q=4 | 1 / 2 / 3 / 5 / 10 = 0.45 / 0.9 / 1.3 / 2.3 / 4.7 ms | 452 / 448 / 441 / **938 / 937** |
| fp4 q=4 | 1 / 5 / 10 = 0.28 / 1.4 / 2.8 ms | 285 / 277 / **523** |
| baseline_a16 q=1 | 1 / 5 / 20 = 0.13 / 0.54 / 2.2 ms | 130 / 109 / **229** |
| transport_a16 q=1 | 1 / 5 / 20 / 40 = 0.09 / 0.41 / 1.6 / 3.2 ms | 90 / 83 / 81 / **143** |

Bursts below ~1.9 ms are clean (min/median/max within 1 %); bursts of >= 2.2 ms
are time-sliced 1:1 with the co-tenant (1.8-2.1x).  Rule for every H200 number
while the co-tenant is present: `repeats x kernel_time < 1.5 ms` (repeats 5 is
fine up to ~300 us kernels; use repeats 2-3 for the q=4 mixed kernel).  The
previously recorded **q=4 mixed 935 us is this artifact**; the kernel itself
takes 441-452 us (fp4 q=4 277-285, fp8 q=4 226, transport_a16 q=4 135,
baseline_a16 q=4 128).
**P0.6 / lever [5] step 1 — elected converter `produced` arrive: negative.**
The K/V converter warps' `kBar/vBar.produced.arrive()` (32 same-address
arrivals per warp, 128 per barrier) was replaced by `__syncwarp(); if (lane==0)
arrive(32)` (counts unchanged).  SASS (fp4 module, `xqa_mha_sm90.cuda.o`): the
two sites went from `SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ,[bar],RZ` (unpredicated)
to `@!P SYNCS.ARRIVE.TRANS64.RED.ART0 RZ,[bar],R3` with R3 = 0x20 on lane 0;
the release fence that `mbarrier.arrive.release.cta` lowers to —
`MEMBAR.ALL.CTA` followed by `FENCE.VIEW.ASYNC.S` — stays unpredicated (every
lane), 48 registers either way, +8 instructions.  Level-3 trace, fp4 q=1, CTA 0
tiles 2–7, 24 launches per variant interleaved A/B under the GPU lock, H200 at
1980 MHz with a co-tenant: segment slot 13 → slot 8 (expansion done → fenced +
arrived) baseline median 186 cyc (0.094 µs; per-launch medians 108–354), elected
181 cyc (0.091 µs; 125–269); expansion 1656 vs 1654 cyc, copy issue 686 vs 693,
converter period 2968 vs 2971 cyc, gemm0 period 2940 vs 2945.  No segment moved
beyond run-to-run spread, so the 128 same-address arrivals do not serialize
measurably; what the segment contains is the per-lane `MEMBAR.ALL.CTA` drain of
the expansion's `STS.128`s, which an elected arrive cannot remove.  [5] step 2
(elected arrivals at all 17 sites) is closed; the drain is attacked only by
[14].  Note on units: today's baseline segment is 186 cyc, not the 0.45 µs
(≈890 cyc at 1.98 GHz) quoted above.  The other level-3 segments quoted above
(0.17 / 1.69 / 0.64 / 0.09) coincide with today's cycle counts (176 / 1656 /
686 / 113) read as cycles/1000, which suggests that row was in kilo-cycles and
that the arrive segment differed (450 vs 186 cyc) either through a code change
since or through conditions; today's interleaved A/B is self-contained.
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

### P0.2 [39] SPEC_DEC attribution, Track S step 1 (no-op), Track W [29] C2 fix (2026-09-04, nkcut2 H200 + ws-1 RTX 5090)

**Attribution (P0.2).** The bench's `kernel_family` line reports `mha.cu
spec_dec=True` for every q=4 mode (module URIs `..._use_spec_dec_True_spec_q_seq_len_4`);
q=1 compressed modes run `mha_sm90.cu`.  The q=4 `build.ninja` carries no
`-DM_TILESIZE`, so `M_TILESIZE` is the `defines.h` default 32 and
`mha.cu:3521` (`__CUDA_ARCH__ == 900 && M_TILESIZE == 16`) is false: the sm90
SPEC_DEC build already has `nbCtaPerSM = 1`, i.e. `__launch_bounds__(256, 1)`
and the 255-register cap.  `cuobjdump -res-usage` of the baseline q=4 `xqa_mha.cuda.o`:

| module (sm90, q=4) | REG | STACK | LDL | STL | LDGSTS |
|---|---|---|---|---|---|
| fp4 static (format 2) | 251 | 48 | 114 | 14 | 263 |
| fp8 static (format 1) | 255 | 48 | 114 | 14 | 263 |
| mixed dynamic (format -1) | 226 | 112 | 378 | 38 | 455 |

sm120 shows the same STACK 48 / 112 and LDL 118 / 338 at REG 166-242 (far from
any cap), so the stack and the LDLs are the runtime-indexed register arrays
(C2), not register-pressure spills, and the record's "226 registers, spills
heavily" reading was wrong: the kernel was never register-capped at 128.
**Track S step 1 (launch bounds) is therefore a no-op and was not applied.**

**Where the LDLs were** (`-lineinfo` build, `nvdisasm --print-line-info`, sm120
fp4 q=1, 118 LDL): `copyMixedPartialHeadsAsync` block loop `pages[localPage]` /
`formats.values[localPage]` (54), its scale loop (59), `mixedPageTagLane`
`pages[lane]` (8), the rest `Vec::operator[]`/shuffle spill traffic of the same
sites.  After fixing those, the dynamic module still had 160 LDL / 40 STL, all
in the A16 path it takes for all-A16 tiles: `HeadPtr::operator+`
`pageIndices[absoluteTokenIdx / tokensPerPage]` (`mhaUtils.cuh:84`, 38 LDL +
`Vec::operator[]` 10 + the `ldgsts` sites 8), with the STLs at the
`HeadPtr src{..., pageIdx, ...}` constructions (`mha.cu:2320/2853`) that spill
the page vector so it can be indexed.

**The fix ([29], `csrc/xqa/mhaUtils.cuh`, `csrc/xqa/mha.cu`).**
(i) `tokenOffset` is removed from `copyMixedPartialHeadsAsync` /
`expandMixedPartialHeadsInPlace`; the callers `static_assert` that every tile
origin (`ctaTile.x`, `warpTile.x`, `warpTile.x * nbXTilesPerXIter`,
`cacheVTileSeqStride`, `cacheVTileSeqLen`) is a multiple of `tokensPerPage`
(16, already required by `smemQKPartGemmMixed`), so a block's page is an
unrolled-iteration constant.  (ii) `selectByIndex(Vec, idx)` - an unrolled
compare/select chain - replaces every register-vector index that stays
lane-dependent (scale loop `lane / 16`, `mixedPageTagLane` `pages[lane]`,
`HeadPtr::operator+`); it folds to the element when the index is constant.
(iii) The expansion form issues only payload copies: `second` and (FP4) the
upper 8 B of `first` are rewritten by the expansion before any read, so their
zero-fill `cp.async` was dead - 3 -> 2 (FP8) / 1 (FP4) LDGSTS per block,
13 -> 9 / 5 per K part per lane on sm120.  The block format is the page's own
format also for `!valid` blocks (zero-filled payload -> expansion yields zeros,
the same tile bytes as the former "treat as A16, zero-fill 32 B").

**Artifact (final tree, `cuobjdump -res-usage` / `-sass` of `xqa_mha.cuda.o`).**

| module | REG | STACK | LDL | STL | LDGSTS | SASS instr |
|---|---|---|---|---|---|---|
| sm120 fp4 q=1 | 178 -> 166 | 48 -> 0 | 118 -> 0 | 18 -> 0 | 214 -> 94 | 6800 -> 6184 |
| sm120 fp8 q=1 | 175 -> 169 | 48 -> 0 | 118 -> 0 | 18 -> 0 | 214 -> 154 | 6424 -> 5864 |
| sm120 mixed q=1 | 186 -> 192 | 112 -> 0 | 338 -> 0 | 59 -> 0 | 374 -> 374 | 12064 -> 11040 |
| sm120 a16 q=1 (format 0) | - -> 176 | - -> 0 | - -> 0 | - -> 0 | - -> 169 | - -> 5392 |
| sm120 fp4 q=4 | 215 -> 206 | 48 -> 0 | 118 -> 0 | 18 -> 0 | 234 -> 114 | 8952 -> 8032 |
| sm120 fp8 q=4 | 214 -> 212 | 48 -> 0 | 118 -> 0 | 18 -> 0 | 234 -> 174 | 8552 -> 7712 |
| sm120 mixed q=4 | 242 -> 237 | 112 -> 0 | 338 -> 0 | 59 -> 0 | 394 -> 394 | 14512 -> 13792 |
| sm90 fp4 q=4 | 251 -> 254 | 48 -> 0 | 114 -> 0 | 14 -> 0 | 263 -> 119 | 11112 -> 9968 |
| sm90 fp8 q=4 | 255 -> 254 | 48 -> 0 | 114 -> 0 | 14 -> 0 | 263 -> 191 | 10088 -> 9016 |
| sm90 mixed q=4 | 226 -> 239 | 112 -> 0 | 378 -> 0 | 38 -> 0 | 455 -> 455 | 18312 -> 17912 |

LDGSTS by variant, sm120 fp4 q=1: `E.64` 60 + `E.64.ZFILL` 60 + `BYPASS.128`
68 + `E` 25 -> `E.64.ZFILL` 60 + `BYPASS.128` 8 + `E.ZFILL` 25: exactly the 60
upper-8-B zero-fills and the 60 `second`-grain zero-fills are gone; the 8
remaining 16-B copies are the A16 path, the 25 4-B copies are the scales.  The
dynamic module keeps the same static LDGSTS count (the FP8/FP4/A16 copies are
now uniform branches on the page format); its executed count is measured with
ncu below.  Correctness: `run_xqa_mixed_page_transport.py` 34/34 bit-exact on
both hosts (the matrix covers the 285 = 18 x 16 - 3 token tail: partial page,
partial tile and `kBAD_PAGE_INDEX` pages in the last tile).

**Timing (flock'd; each row is the median of per-round medians, rounds
interleaved baseline / final).  The baseline is a second pristine checkout
(`dash-flashinfer-claude-wtEbase`, `git archive` of the unpatched tree) with its
own workspace - a cached workspace is *not* a frozen baseline, because
`JitSpec.try_load()` returns None for JIT specs and ninja's dependency scan
rebuilds the cached module from whatever source the checkout now holds (first
attempt at a "baseline" re-time produced the patched kernel's numbers).**

ws-1 RTX 5090, B=17 S=4096 8 KV heads GQA 4 D=128 bf16, repeats 5 x trials 5, 3 rounds:

| mode | q=1 baseline | q=1 final | ratio | q=4 baseline | q=4 final | ratio |
|---|---|---|---|---|---|---|
| transport_a16 | 180.3 | 174.7 | 0.969 | 184.3 | 179.1 | 0.972 |
| fp8 | 139.5 | 118.8 | 0.851 | 153.7 | 125.0 | 0.814 |
| fp4 | 83.4 | 65.5 | 0.785 | 100.8 | 81.9 | 0.813 |
| mixed | 145.8 | 126.4 | 0.867 | 157.3 | 132.3 | 0.841 |

Per-round medians agree within 0.5 %.  The sm120 targets (FP8 <= 125, FP4 <= 79,
mixed <= 135) all pass from [29] alone, well beyond the lever's own model
(FP4 84 -> 78-80, FP8/mixed +-2 %): the sm120 kernel was issue-bound on the
copy path's local-memory traffic, not only latency/request-bound.  Branch (a)
of Track W ([25] `nbSubSeqPerSeq`) and [26]/[27] are now optional headroom, not
gate items.

nkcut2 H200 (co-tenant present; q=4 repeats 3 x trials 7 so that repeats x t < 1.5 ms,
q=1 repeats 5 x trials 5; 3 rounds for q=4):

| mode | q=4 baseline (r1/r2/r3) | q=4 final (r1/r2/r3) | ratio | q=1 baseline -> final (mha_sm90.cu control) |
|---|---|---|---|---|
| transport_a16 | 137.6 / 137.7 / 138.4 | 131.4 / 131.3 / 131.5 | 0.955 | 83.0 -> 83.0 |
| fp8 | 230.4 / 231.2 / 234.1 | 198.8 / 198.2 / 197.6 | 0.857 | 91.6 -> 91.4 |
| fp4 | 276.7 / 276.5 / 276.4 | 239.5 / 238.9 / 239.8 | 0.866 | 96.5 -> 96.1 |
| mixed | 436.9 / 438.9 / 435.7 | 424.8 / 422.8 / 422.2 | 0.968 | 115.1 -> 115.9 |

Two further final-tree rounds from the first (contaminated-baseline) chain agree
(fp4 240.1/239.6, fp8 199.5/198.8, mixed 421.3/423.1, a16 130.6/131.5).  The
q=1 control is unchanged because sm90 q=1 runs `mha_sm90.cu`, which has its own
copy/expand code.  **Track S verdict: steps 1-2 leave mixed q=4 at 422.8 /
239.5 = 1.77x fp4 q=4 (acceptance <= 1.5x not met); the static modules gain
13-14 %, the dynamic module 3 %.**

**ncu, sm90 q=4, one launch (136 CTAs), `--launch-skip 1 --launch-count 1`
(counts are contention-independent; durations are ncu-serialized with the
co-tenant present):**

| metric | mixed base | mixed final | fp4 base | fp4 final |
|---|---|---|---|---|
| registers / occupancy limit (regs) | 226 / 1 | 239 / 1 | 251 / 1 | 254 / 1 |
| smsp__inst_executed_op_local_ld.sum | 741,472 | **0** | 458,592 | **0** |
| smsp__inst_executed_op_local_st.sum | 208,352 | **0** | 64,736 | **0** |
| smsp__inst_executed_op_ldgsts.sum (issued) | 924,800 | 924,800 | 935,680 | 361,216 |
| ..._pred_on_any.sum (any lane active) | 924,800 | 655,677 | 935,680 | 361,216 |
| sm__inst_executed_pipe_lsu.sum | 5,395,776 | 4,757,664 (-12 %) | 6,219,008 | 5,096,736 (-18 %) |
| smsp__inst_executed.sum | 49.09 M | 48.11 M | 53.40 M | 48.41 M |
| gpu__time_duration | 548.8 us | 540.7 us | 360.9 us | 318.8 us |

The dynamic module's FP8/FP4 copies compile to predicated LDGSTS (same issued
count, 29 % now predicated off = the removed zero-fills for the 1/3 fp8 + 1/3
fp4 pages); the fp4 static module issues 3 -> 1 per block.  ncu sections on the
final tree (sm90 q=4): mixed runs at **10.0 warp-cycles per issued instruction
vs 5.1 (fp4) / 5.2 (fp8)** with the same executed-instruction count as fp4
(48.1 M vs 48.4 M); the section report attributes 6.3 of those cycles to "waiting
to be selected to fetch an instruction" (no_instruction: the 17.9 K-instruction
dynamic kernel jumps between its A16, FP8 and FP4 copy/expand paths every
iteration).  All three run at 12.5 % occupancy (1 CTA/SM: registers *and* shared
memory both limit to 1), issue 0.20 (mixed) / 0.39 (fp4) instructions per
scheduler-cycle with ~2 active warps per scheduler, DRAM at 7-14 %.  Grid 136 =
`nbSubSeqPerSeq` 1 (`132 / (17 x 8)` rounds to 0) on 132 SMs: one full wave plus
a 4-CTA second wave, i.e. the q=4 kernels take two CTA-durations.

**Zero-code calibration `XQA_NB_SUB_SEQ` on the final tree (q=4, repeats 3 x
trials 5, flock'd):**

| mode | n=1 (136 CTAs) | n=2 (272) | n=4 (544) | n=8 (1088) | wave model n=2 / 4 / 8 |
|---|---|---|---|---|---|
| transport_a16 | 131.3 | 117.1 (0.892) | **111.7 (0.850)** | 125.9 (0.959) | 0.75 / 0.625 / 0.56 |
| fp8 | 197.6 | 176.5 (0.893) | **167.9 (0.850)** | 188.6 (0.954) | |
| fp4 | 240.1 | 230.0 (0.958) | **204.7 (0.852)** | 241.4 (1.005) | |
| mixed | 420.4 | 382.0 (0.909) | **359.0 (0.854)** | 368.8 (0.877) | |

n=4 gives 0.85x on every q=4 mode (the tail wave is real) but the pure wave
model over-predicts because each extra CTA carries the P0.8 fixed cost
(prologue + multi-block merge, 4.8-7 us on sm120); n=8 loses it again.  A host
default for the sm90 SPEC_DEC shape (n=4 here) is a candidate lever worth ~15 %
on all q=4 modes; it was not applied because the right n needs the per-CTA
fixed-cost model, not the wave count alone.  It does not change the mixed/fp4
ratio (1.75x at n=4): **the Track S residual is the dynamic module's
instruction-fetch stall, which is a code-size / dispatch problem of the sm90
SPEC_DEC dynamic build (287 KB of SASS), and the plan's step 3 (route q=4 mixed
to `mha_sm90.cu`, whose converters are format-uniform per stage) or a
code-size lever in `mha.cu`'s dynamic path is the next item.**
