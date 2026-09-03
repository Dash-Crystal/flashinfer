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
