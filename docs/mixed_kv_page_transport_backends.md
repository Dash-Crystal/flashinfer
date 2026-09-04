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

### Phase 2 [16] + converter SASS audit — bit-placement decode, source-ordered conflict-free layout, hoisted lane constants (nkcut2 H200, 2026-09-04, wt/B)

Baseline = post-[13] tree (`f4369f61`) rebuilt in its own checkout and workspace; every number below
is an interleaved A/B (base, new, base, new, ...) under the GPU lock with the VLLM co-tenant present.
Tool: `benchmarks/microbench/xqa_sm90_converter_sass.py --paths` (executed path of the steady state
= prologue + scale prep/vote + the folded-scale loop), lineinfo build byte-identical to production.

**What changed (sm90, BF16 math; sm120 keeps its native cvt path, `__CUDA_ARCH__ < 1000` guard).**
1. E4M3 -> BF16 by bit placement: `PRMT` (byte spread + sign replicate) + `SHF` + `LOP3` per two
   values gives bf16(x) * 2^-120 exactly (subnormals included; `mul.rn.bf16x2` handles subnormal
   inputs); the 2^120 is folded into the per-block scale when every scale of the warp's tile stays
   finite after the fold (one `__all_sync` vote per tile on fp32 `|s * global * 2^120| < 255.5 * 2^120`),
   else one extra packed multiply per pair.  4 SASS per pair (3 + HMUL2) instead of 5
   (F2FP.E4M3 -> 2 HADD2.F32 -> F2FP.BF16 -> HMUL2).  E2M1 likewise (placement = mag * 2^-126, fold
   iff |s * global| < 4): 5 per pair with the fallback multiply, 4 folded, instead of the cutlass
   LUT's 32 SASS per 8 values (= 8 per pair) + HMUL2.  Exactness argument: both operands keep their
   mantissas, only exponents shift, so the product rounds identically as long as `scale * global` is
   fp32-normal; `|global| >= 2^-117` is checked once per warp (below it: two-multiply form).
2. Lane cut: lanes 0-15 = one page's 16 tokens at head part 0, 16-31 at part 1, so every 8-lane
   LDS.128/STS.128 phase is 8 consecutive tokens of one part.  Stores are source-ordered (block b,
   half g at chunk `(2b+g) ^ (token % 8)` = one lane base XOR an immediate, 7 LOP3 per tile) and
   conflict-free; the previous (token = tid/2, p = tid%2) cut made every STS.128 2-way conflicted
   (parts 0 and 1 of a token share bank groups) and the destination-ordered variant tried first
   would have been 4-way.  E4M3 packed rows keep the 128 B swizzle (reads = lane base XOR b*16,
   conflict-free); E2M1 packed rows now use an 80 B stride (5 chunks; bank group `5*token + c`
   walks all 8 groups over 8 tokens), so the lane's 4 blocks are 32 contiguous bytes at immediate
   offsets (2 LDS.128 [R+UR+imm]).  Compressed pages are cp.async-copied by the converters, so the
   layout is free (no TMA box uses it).
3. All lane constants (A16 row base, packed row bases, scale offset) are computed once per warp
   (`ExpandLane`) and kept in registers; per tile the address work is one `IADD3/VIADD` of the stage
   base per stream (the compiler emits `[R+UR+imm]` for the rest).  The old body recomputed ~45
   address instructions per tile.  The zero-initialised `LdGrain first{}/second{}` (14 CS2R) and
   the BSSY/BSYNC pairs are gone.

**SASS, `xqa_mha_sm90.cuda.o` (K converter; V identical +-1).**

| build | total SASS | REG / STACK | expandPackedStage static | executed per lane per tile (steady state) |
|---|---|---|---|---|
| fp4 post-[13] | 3360 | 45 / 0 | 394: PRMT 96, LOP3 78, SHF 27, IMAD 98, HMUL2 32, HADD2 4, F2FP 6, FMUL 4, LDS 3, STS 8, CS2R 14, UMOV 8, BRA 5 | ~385 |
| fp4 [16] | 3344 | 48 / 0 | 399 (fold loop + two-multiply loop + zero fill) | **188**: PRMT 36, LOP3 39, SHF/IMAD.SHL 42, HMUL2 32, STS 8, LDS 3, F2FP 4, HADD2 4, FMUL 4, FMNMX 3, VOTE 1, FSETP 1, ISETP 1, IADD3/VIADD/U* 5, BRA 3, NOP 2 (fallback path 223) |
| fp8 post-[13] | 3200 | 48 / 0 | 272: F2FP 70, HADD2 68, HMUL2 32, LOP3 15, IMAD 28, LEA 6, LDS 5, STS 8, CS2R 14, BRA 6 | ~262 |
| fp8 [16] | 3472 | 48 / 0 | 390 | **187**: PRMT 36, LOP3 42, SHF/IMAD.SHL 34, HMUL2 32, STS 8, LDS 5, F2FP 4, HADD2 4, FMUL 4, FMNMX 3, VOTE 1, misc 14 (fallback 222) |
| mixed post-[13] | 4096 | 48 / 16 | 600 | - |
| mixed [16] | 4096 | 48 / 0 | 768 (both formats x fold/fallback) | - |
| a16 | 2504 -> 2488 | 40 / 0 | - | - |

Plan [16] verification items: HADD2.F32 in the converter body 68 -> 4 (the four block-scale
conversions); F2FP 70 -> 4 (+2 static in the fallback); LOP3 for stores 16 -> 7; STACK 0 in every
build (the mixed build's STACK 16 is gone; an intermediate version of this change with two more
hoisted registers spilled 8 B there, fixed by recomputing the fold multiplier per tile in the mixed
build only); 52/52 bit-exact on H200 (34-case matrix + tails + E4M3-subnormal + max-scale regimes;
the max-scale regime drives the two-multiply fallback) and 52/52 on the RTX 5090 (sm120 path).

**Audit of the remaining fp4 lane-tile (188).**  Essential 139: 32 PRMT + 32 SHF + 32 LOP3
(placement) + 8 SHF (`w << 4`, one per packed word so both nibbles of a byte take the same shift)
+ 32 HMUL2 + 8 STS.128 + 2 LDS.128 + 1 LDS scale word.  Non-essential 49: scale conversion 13
(SHF, 2 F2FP.E4M3, 4 HADD2.F32, 4 FMUL, 2 F2FP.BF16 pack — fp32 route required for bit-exactness
with an arbitrary fp32 global scale), fold vote 6 (3 FMNMX, FSETP, VOTE, NOP), store XOR 7 (the
TMA swizzle vs source order; unavoidable without SEL-based register permutation), addressing 6
(ULEA/USHF/ULOP3/IMAD.U32/IADD3/VIADD: stage base and `scales[t % 4]`), 4 PRMT scale broadcasts
(ptxas fused them into HMUL2 `.H0_H0/.H1_H1` in an earlier variant but not here), format test +
branches 4, `__syncwarp` NOP 1.  The plan's <= 180 target assumed a 6-SASS LUT (48 PRMT essential);
with the placement decode the floor is 139 + the 13 scale + 6 vote = 158 and the residual 30 is
addressing/branch glue worth <= 0.15 us at 10.8 cyc/instr.  A further -32 (fold 2^126 into the E2M1
scale) is already taken when the fold applies (|s * global| < 4, true for the bench's unit scales);
the fallback costs +32 HMUL2 only for tiles with large block scales.

**Trace (MIXED_KV_TRACE=3, CTA 0, tiles 2-7, 15 launches each, cycles at 1.98 GHz).**

| segment | fp4 base -> new | fp8 base -> new | mixed base -> new |
|---|---|---|---|
| K expansion (s13-s12) | 1440 -> 750 | 1490 -> 816 | 2202 -> 1545 |
| V expansion (s15-s14) | 1662 -> 1068 | 2118 -> 966 | 3736 -> 2465 |
| K converter period | 2512 -> 2228 | 3160 -> 2432 | 4066 -> 2584 |
| copy issue (s9-s8, includes the wait for the stage release) | 700 -> 1242 | 1088 -> 1100 | 1078 -> 1013 |
| gemm0 cadence (s0(t)-s0(t-1)) | 2620 -> 2544 | 3268 -> 2702 | 3881 -> 3254 |
| gemm0 K-wait (s0(t)-s3(t-1)) | 262 -> 312 | 870 -> 311 | 1012 -> 431 |

The fp4 converter now waits for gemm0's stage release (copy-issue segment 700 -> 1242 while its own
work shrank), i.e. the converter no longer paces fp4; fp8 and mixed lose 0.3-0.4 us of gemm0 cadence
because the K-wait (converter late) collapses to the fp4 level.

**Production bench (q=1, B=17, S=4096, 8 KV heads, GQA 4, D=128, bf16; `--repeats 5 --trials 5`,
two interleaved rounds, medians in us; all four modes dispatch to `mha_sm90.cu`).**

| mode | post-[13] base (r1/r2) | [16] (r1/r2) | delta |
|---|---|---|---|
| transport_a16 | 82.18 / 81.95 | 81.79 / 81.84 | -0.3 (noise) |
| fp8 | 86.20 / 86.84 | 77.17 / 77.25 | **-9.3 us (-10.8 %)** |
| fp4 | 90.78 / 91.01 | 73.12 / 73.34 | **-17.7 us (-19.4 %)** |
| mixed | 106.91 / 108.17 | 83.38 / 83.53 | **-24.1 us (-22.4 %)** |

Plan [16] predicted fp8 "245 -> ~213 SASS, -5 us or more"; measured 262 -> 187 executed and -9.3 us
(accept).  The plan's fp4 gain from hoisting alone was -16 SASS / -0.09 us; the placement decode
plus the layout/hoisting cut gives -197 SASS and -17.7 us.  The expansion segment scaled with the
count (1440 -> 750 cyc for 385 -> 188 SASS), so the converter is count-bound here, not
MIO-rate-bound; fp4 is now paced by gemm0 (consumer floor), so the next fp4 lever is on the consumer
side ([4]/[7]/[8]) or in the copy-issue chain ([14]/[19]), not in the decode.

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

### Track S step 3 — [40] per-page format dispatch in the mha.cu dynamic path (2026-09-04, nkcut2 H200, worktree E)

**Route decision (arithmetic before any build).**  Plan step 3 proposed routing
q=4 mixed pages to `mha_sm90.cu` SPEC_DEC after deepening its `nbKBuf = nbVBuf = 2`.
`sizeof(SharedMem)` of `mha_sm90.cu` with the mixed build's flags
(`CACHE_ELEM_ENUM=5 SPEC_DEC=1 SPEC_Q_SEQ_LEN=4 HEAD_GRP_SIZE=4 HEAD_ELEMS=128`,
`ctaNbQHeads` = 16, `nbXBuf` = 2), measured by compiling a copy of the file with
`nbKBuf/nbVBuf` forced to `MIXED_KV_KDEPTH/VDEPTH` and an incomplete-type print
of the constants (cap for two CTAs per SM: 112.5 KB = 115,200 B):

| stages K / V | K bufs | V bufs | X/out ring (2 x max(2 KB X, 4 KB out)) | Q | rest (scales 4 KB, meta 1.25 KB, colmax/sum, barriers) | sizeof(SharedMem) | 2 CTAs/SM |
|---|---|---|---|---|---|---|---|
| 2 / 2 (today) | 32,768 | 32,768 | 8,192 | 4,096 | ~7,168 | **84,992 B (83.0 KB)** | yes |
| 3 / 2 | 49,152 | 32,768 | 8,192 | 4,096 | ~7,168 | **101,376 B (99.0 KB)** | yes |
| 3 / 3 | 49,152 | 49,152 | 8,192 | 4,096 | ~7,168 | **117,760 B (115.0 KB)** | **no (+2,560 B over the cap)** |

(q=1 control from the same method: K3/V3 = the shipping 108.5 KB layout, `ctaNbQHeads` 8, X entry 2 KB, Q 2 KB.)
So 3/3 only fits at 2 CTAs/SM with a 4 KB `OutSwizzleBuf` aliasing change to the
K ring (X entries back to 2 KB -> 113,664 B); 3/2 fits as is.  But the smem is
not the blocker: in the mixed q=4 module the compiled `xqa_mha_sm90.cuda.o`
kernel is a **16-instruction, 4-register stub** — `mha_sm90.cu:1089` guards the
body with `(IS_SUPPORTED_F16_CASE || CACHE_ELEM_ENUM == 2)`, and
`IS_SUPPORTED_F16_CASE` (`:53`) requires `!SPEC_DEC` for enum 5.  Routing q=4
mixed pages there means writing the SPEC_DEC (SWAP_AB) variant of the mixed
loader/converter kernel in a Track A/B-owned file, on top of the known
full-draft-mask defect of that path (upstream #4199 / #4198, the reason
`xqa.py:566-575` forces `mha.cu` for `swap_ab_eligible` shapes).  Prediction if
it were built (q=1 cadences fp8 91 / fp4 96 / mixed 114 are converter-bound at
~1.0-1.3 us/tile; the SWAP_AB consumer at N=16 instead of 8 adds elementwise
softmax work, not GMMA issue): fp8 ~95-110, fp4 ~100-115, mixed ~120-135 us at
2 CTAs/SM with K3/V2 — attractive, but a multi-day kernel item, not a step-3
routing change.  **Route taken: (b), the code-size lever in `mha.cu`.**

**Attribution of the 17.9 K instructions (`-lineinfo` cubins, `nvdisasm
--print-line-info-inline`, instructions bucketed by the `kernel_mha_impl`-level
call site; sm90 q=4, per module).**  On sm90 `compactMixedPages` is false, so
the MMA loop was already the stock A16 loop; the excess is entirely in the copy
and expansion helpers, which were instantiated with a per-block runtime format
branch inside their fully unrolled block loops (8 iterations x 3 predicated
LDGSTS variants per K part, 8 x 2 expansion bodies), and the dynamic module in
addition carried the stock A16 `copyPartialHeadsAsync` path at every site:

| call site (mha.cu, a93d090e lines) | dyn (-1) | fp4 (2) | a16 (0) |
|---|---|---|---|
| `runGemm0` :2576 total | 6,031 | 2,957 | 1,620 |
| .. `loadKTilePart` inside gemm0 :2497 (2 parts) | 3,105 | 1,150 | 1,321 |
| .. K `expandMixedPartialHeadsInPlace` :2519 | 2,543 | 1,481 | 2 |
| .. `smemQKPartGemm` :2559 | 373 | 316 | 289 |
| V `expandMixedPartialHeadsInPlace` :3179 | 2,686 | 1,449 | 0 |
| `loadVTilePart` in loop :3076 | 2,338 | 698 | 1,409 |
| `loadKTilePart` prologue :2443 | 1,604 | 531 | 752 |
| `loadVTilePart` prologue :3014 | 1,250 | 357 | 817 |
| everything else (softmax, rescale, merge, output) | ~3,660 | ~3,395 | ~3,405 |
| **total** | **17,912** | **9,968** | **8,552** |

dyn - fp4 = 7,944: copy paths +5.5 K (A16 fast path 2.1 K at K + 2.3 K at V, the
FP8 LDGSTS variant and its address math), expansion +2.3 K (the FP8 bodies).
The hot footprint of one tile iteration (mixed copy + both expansion bodies for
2 K parts + V + consumer) was ~145 KB of the 287 KB dyn kernel vs ~108 KB (fp4)
and ~75 KB (a16), which matches the "instruction fetch" stall appearing only
for the dynamic module.

**Lever [40] (`csrc/xqa/mhaUtils.cuh copyMixedPartialHeadsAsync`,
`expandMixedPartialHeadsInPlace`; `csrc/xqa/mha.cu` K/V copy call sites).**
A warp instruction of either helper covers blocks of a single page (blocksPerSpan
= 16 tokens x blocksPerPart >= 32 for every supported part width), so the format
is warp-uniform per page.  Both loops are now page-outer: per page span, read
`pages[]`/`formats[]` once (`selectByIndex` chain), branch once, and run a
format-specialised body (`MixedFormatTag<f>` generic lambda; `if constexpr` on
A16/FP8/FP4) over that page's unrolled block iterations.  The page loop is
`#pragma unroll 1` in the dynamic module (`mixedPageLoopUnroll` = 1 when
`MIXED_PAGE_STATIC_FORMAT < 0`) and fully unrolled in the static modules, where
the tag is the build constant, the branch vanishes and the code shape is the
previous one.  The dynamic module routes every tile through the per-page copy
(`kA16CopyFastPath = MIXED_PAGE_STATIC_FORMAT >= 0` gates the stock
`copyPartialHeadsAsync` call), its A16 body being the stock two-grain (2 x 16 B)
copy; FP8 2 x 8 B and FP4 1 x 8 B per block are unchanged from [29].  A16-only
tiles still skip the expansion via `kNeedsExpansion`.  Prediction: dyn SASS
17.9 K -> 6-8 K (a16 base ~4.3 K non-copy code + one copy of each body per
site), hot footprint ~70 KB, no_instruction stall -> the static modules' level,
warp-cycles/instruction 10 -> ~5, mixed q=4 420 -> 220-250 us (<= 1.05x fp4);
static modules within +-2 %.

**Artifact (sm90 q=4, `cuobjdump -sass` / `-res-usage` of `xqa_mha.cuda.o`).**

| module | SASS instr before -> after | LDGSTS static | BRA | REG | STACK / LDL / STL |
|---|---|---|---|---|---|
| dyn (format -1) | **17,912 -> 8,824** | 455 -> 137 | 306 -> 206 | 239 -> 249 | 0 / 0 / 0 |
| fp4 (2) | 9,968 -> 9,792 | 119 -> 119 | 130 -> 132 | 254 -> 240 | 0 / 0 / 0 |
| fp8 (1) | 9,016 -> 8,912 | 191 -> 191 | 140 | 254 -> 238 | 0 / 0 / 0 |
| a16 (0) | 8,552 -> 8,552 (identical) | 221 | 98 | 236 | 0 / 0 / 0 |

dyn after, by call site: `runGemm0` 2,308 (copy 1,251 for 2 parts, K expansion
718, MMA 327); the dynamic module is now smaller than the fp4 static module.

**ncu (sm90 q=4, one launch of 136 CTAs, `--launch-skip 1 --launch-count 1`,
co-tenant present so durations are ncu-serialized; the same metric list on the
pristine a93d090e checkout `dash-flashinfer-claude-wtEs3base` and on the patched
tree):**

| metric | mixed before (a93d090e) | mixed after | fp4 after |
|---|---|---|---|
| smsp__average_warp_latency_per_inst_issued (warp-cycles / issued instr) | 10.01 (record 10.04) | **4.52** | 5.06 |
| smsp__average_warps_issue_stalled_no_instruction_per_issue_active | 6.21 (record: "6.3 cycles waiting to be selected to fetch an instruction") | **0.52** | 0.77 |
| smsp__inst_executed.sum | 48.11 M | 46.36 M | 48.31 M |
| smsp__inst_executed_op_ldgsts.sum (issued) | 924,800 | 555,648 | 361,216 |
| issued warp per scheduler | 0.20 | 0.44 | 0.39 |
| registers / achieved occupancy | 239 / 12.3 % | 249 / 12.2 % | 240 / 12.3 % |
| gpu__time_duration (serialized) | 545.6 (record 540.7) | 283.6 | 311.0 |

The fetch stall is gone (0.52 warps stalled per issue-active cycle, below the
fp4 module's 0.77); the dynamic module now issues 4 % fewer instructions than
fp4 (no predicated-off LDGSTS address math for the other two formats: issued
LDGSTS 924,800 -> 555,648 = the real 2 x 1/3 FP8 + 1 x 1/3 FP4 + 2 x 1/3 A16 copies
plus scales).

**Timing (flock'd, `--repeats 2 --trials 5`, three rounds interleaved with other
agents' GPU use; medians per round; q=1 is the untouched `mha_sm90.cu` control):**

| mode | q=4 a93d090e (prev. record r1/r2/r3) | q=4 [40] r1 / r2 / r3 | q=4 [40] min / max over rounds | q=1 control before -> after |
|---|---|---|---|---|
| transport_a16 (a16 module, byte-identical SASS) | 131.4 / 131.3 / 131.5 | 136.4 / 136.4 / 136.8 | 135.9 / 137.6 | 83.0 -> 82.9 |
| fp8 | 198.8 / 198.2 / 197.6 | 199.2 / 198.7 / 198.0 | 196.4 / 199.5 | 91.4 -> 91.2 |
| fp4 | 239.5 / 238.9 / 239.8 | 237.5 / 236.1 / 236.9 | 234.8 / 238.6 | 96.1 -> 96.0 |
| mixed | 424.8 / 422.8 / 422.2 | **217.1 / 216.4 / 216.0** | 214.8 / 218.3 | 115.9 -> 115.6 |

Same-session pristine-baseline round (`dash-flashinfer-claude-wtEs3base`, SASS 17,912 confirmed, one locked round after the three [40] rounds): a16 136.2, fp8 201.3, fp4 242.7, mixed 426.6 — so within this session mixed is 426.6 -> 216.4 (0.507x), fp4 242.7 -> 236.9 (-2.4 %), fp8 201.3 -> 198.7 (-1.3 %), a16 136.2 vs 136.4 (byte-identical module, +0.1 %).  The a16 module is
byte-identical, so its +3.8 % against the record is the session's co-tenant/clock
offset (the record's rounds were 137.7 before [29]); against the same-session
baseline fp8/fp4 gain 1-2 % from the per-page copy (page-stride products once
per page instead of once per block).
**mixed q=4: 422.8 -> 216.4 us (0.51x), 0.915x fp4 q=4 — Track S acceptance
(<= 1.5x fp4) met; the mixed stream (2/3 compressed pages) is now faster than
the pure fp4 stream, as its expansion work is smaller.**  Correctness:
`run_xqa_mixed_page_transport.py` 34/34 bit-exact (fresh workspace, includes
the q=4 NHD/HND mixed cases).

**Against the q=4 targets (FP8 <= 94, FP4 <= 59, mixed <= 101; A16 135):** FP8
198, FP4 236, mixed 216 — none pass; all three sit at 1 CTA/SM (255-register
SPEC_DEC build, 12 % occupancy, 0.4 issued instructions per scheduler-cycle,
DRAM 7-14 %).  The remaining lever set for sm90 q=4 is occupancy/waves, not the
mixed dispatch: `XQA_NB_SUB_SEQ=4` (0.85x on every mode, needs the host
`nbSubSeqPerSeq` default from the per-CTA fixed-cost model), and a 2-CTA/SM
SPEC_DEC build (register budget 128 with the C2-free code, or the `mha_sm90.cu`
SPEC_DEC mixed path whose smem arithmetic is above).

### Track S step 4 — [41] 2-CTA/SM SPEC_DEC build and [42] host `nbSubSeqPerSeq` default (2026-09-03, nkcut2 H200 + ws-1 RTX 5090, worktree E)

**Occupancy arithmetic (before any build).**  The sm90 q=4 modules after [40]
ran 1 CTA/SM for two independent reasons, both read from the step-3 ncu launch
statistics: `Block Limit Registers 1` (REG 236-249 under the 255-register
`__launch_bounds__(256, 1)` of the SPEC_DEC build, `mha.cu` `nbCtaPerSM`) and
`Block Limit Shared Mem 1` (`Dynamic Shared Memory Per Block 163.58 KB` =
167,504 B against 228 KB per SM).  Neither limit alone would have been enough
to lift:

| SharedMem member (sm90, enum 5, SPEC_DEC) | M_TILESIZE 32, K 128 B / V 64 rows (shipping) | M_TILESIZE 16, K 64 B / V 32 rows ([41]) |
|---|---|---|
| `k[4 warps][2 bufs]` = 4 x 2 x 64 tokens x part bytes | 65,536 | 32,768 |
| `v[2 grps][1][2 bufs]` = 2 x 2 x rows x 256 B | 65,536 | 32,768 |
| `q` = warpTile.y x 256 B | 8,192 | 4,096 |
| `x[4]` = 4 x warpTile.y x 128 B | 16,384 | 8,192 |
| scales, formats, row max/sum, barriers, padding | ~11,856 | ~5,888 |
| **sizeof(SharedMem)** (ncu `launch__shared_mem_per_block_dynamic`) | **167,504 B (163.6 KB)** | **83,712 B (81.8 KB)** |
| 2 x (size + 1 KB driver) vs 228 KB | 329 KB: no | 165.5 KB: **yes** (3 CTAs: 248 KB, no) |

The two intermediate layouts do not fit two CTAs either: K 128 B / V 32 rows at
M 16 is 116,480 B (2 x 117.5 KB = 235 KB > 228 KB, over by 1.5 KB), K 64 B /
V 64 rows at M 16 is the same size.  So the smem lever is the sm120 K/V ring
shape (64 B K parts, 32-row V tiles; `mha.cu` `preferedKHeadPartBytes` /
`cacheVTileSeqLen`) *and* the 16-row M tile.

Registers: `__launch_bounds__(256, 2)` means <= 128 registers per thread
(65,536 / 512).  What drives the 239-249 of the shipping build is the M-tile:
at `M_TILESIZE` 32 a gemm0 warp holds a 32 x 64 fp32 accumulator (64 registers)
and a gemm1 warp a 32 x 64 fp32 output accumulator (64), plus the copy/expand
address state of a 128 B part (16 grains per lane per part); the compiler then
spends the rest of the 255 budget on scheduling freedom (STACK 0 both before and
after [29], so none of it was demand).  With q x GQA = 16 rows the 32-row tile
is half padding (`nbValidRows = rowsPerBlock = M_TILESIZE` in SPEC_DEC): the
MMA and softmax rows 16-31 are computed and discarded.  `M_TILESIZE 16` halves
both accumulators (32 + 32) and the 64 B parts halve the per-part copy state.
The register test already existed in the cache: the q=1 non-SPEC_DEC enum-5
sm90 modules are built at `__launch_bounds__(256, 2)` with the same
warpTile.y 16 mixed copy/expand code (128 B parts) and compile to **REG 128,
STACK 8-24, 2-5 LDL/STL pairs all outside the tile loop** (store at the
prologue, `LDL.LU` at the epilogue); SPEC_DEC adds the mask words, the 64 B
parts remove half the address state.  Prediction: REG 128, STACK 0-32, no LDL
inside the tile loop; the per-tile instruction count drops (half the MMA and
softmax rows) while the K pipeline runs 4 parts per tile instead of 2.

Effect model: the kernel is latency-bound (step 3: 0.4 issued instructions per
scheduler-cycle, DRAM 7-14 %, warp latency 4.5-5.1 cycles per issued
instruction at 8 warps/SM), so a second resident CTA hides the K/V landing and
barrier round trips of the other; and the sub-sequence sweep on the 1-CTA/SM
build (n=4: 0.85x on every mode; 136 CTAs on 132 SMs is 1.03 waves, the 4-CTA
tail wave costs almost a whole CTA time) says the wave geometry is worth
another ~15 %.  Modelled: fp4 236 -> 130-150, fp8 199 -> 115-135, mixed 216 ->
125-155, a16 136 -> 95-105 (a16 sits near the 67.5 us byte floor plus the
per-CTA fixed cost); the q=4 targets (94 / 59 / 101) were not predicted to be
met by occupancy alone.

**Levers.**
- **[41]** `csrc/xqa/mha.cu`: `#elif __CUDA_ARCH__ == 900 && CACHE_ELEM_ENUM == 5 && SPEC_DEC && M_TILESIZE == 16` selects `preferedKHeadPartBytes = 64`, `cacheVTileSeqLen = 32`; `nbCtaPerSM` for the sm90 SPEC_DEC `M_TILESIZE == 16` case is `2 * (smemSize + 1024) <= 228 KB ? 2 : 1` (the smem arithmetic decides, not the flag).  `flashinfer/jit/xqa.py`: `-DM_TILESIZE=16` for the mixed-page SPEC_DEC modules when `q_seq_len x head_group_ratio <= 16` (one 16-row token block; the module URI already encodes both, so no name change).  The sm120 SPEC_DEC modules take the M tile too (their K/V ring is unchanged; their 99 KB cap still gives 1 CTA/SM).
- **[42]** `csrc/xqa/mha.cu launchMHAFlashInfer`: `chooseNbSubSeqPerSeq` replaces `max(1, SMs / (B x H))` (which is 1 for every B x H >= SMs).  slots = SMs x `cudaOccupancyMaxActiveBlocksPerMultiprocessor(kernel_mha, 256, smemSize)`; cost(n) = ceil(nbSeq x n / slots) x (1 + nbTiles / n) in tile units (per-CTA fixed cost ~ one tile: 5.5-7 us measured on both hosts against 4-11 us per tile); n > 1 only for a modelled gain > 5 % (merge scratch traffic is not in the model).  Picks n = 5 at 264 slots, n = 4 at 132 slots (the measured optimum of the 1-CTA/SM build), n = 1 at 170 slots (ws-1; P0.8 measured every n > 1 slower there).  `XQA_NB_SUB_SEQ` still overrides.

**Artifact (sm90, `cuobjdump -res-usage` / `-sass` of `xqa_mha.cuda.o`, q=4 modules, workspace `/tmp/mixedkv-wtE-s4-v1`; `build.ninja` carries `-DM_TILESIZE=16`).**

| module (sm90 q=4) | REG before -> after | STACK / LDL / STL | SASS instr before -> after | LDGSTS static | BRA |
|---|---|---|---|---|---|
| a16 (0) | 236 -> **128** | 0 / 0 / 0 | 8,552 -> 7,008 | 221 -> 177 | 98 -> 125 |
| fp8 (1) | 238 -> **128** | 0 / 0 / 0 | 8,912 -> 8,160 | 191 -> 102 | 140 -> 174 |
| fp4 (2) | 240 -> **128** | 0 / 0 / 0 | 9,792 -> 9,104 | 119 -> 102 | 132 -> 174 |
| mixed dyn (-1) | 249 -> **128** | 0 / 0 / 0 | 8,824 -> 9,288 | 137 -> 122 | 206 -> 289 |

sm120 q=4 (ws-1, triton `cuobjdump`, same flags): a16 213 -> 200, fp8 235 -> 196,
fp4 214 -> 178, mixed 240 -> 202 registers, STACK/LDL/STL 0 everywhere, SASS
7,160 -> 5,968 / 7,240 -> 6,000 / 7,392 -> 6,184 / 9,256 -> 8,056.

**ncu (sm90 q=4, one launch, `--launch-skip 1 --launch-count 1`, co-tenant
present so durations are serialized; "before" = the step-3 record of the same
metrics on the [40] tree).**

| metric | mixed before | mixed after | fp4 before | fp4 after | fp8 after | a16 after |
|---|---|---|---|---|---|---|
| launch__grid_size | 136 | 408 (v1, n=3) | 136 | 408 | 408 | 408 |
| launch__registers_per_thread | 249 | **128** | 240 | **128** | 128 | 128 |
| launch__shared_mem_per_block_dynamic | 167,504 B | **83,712 B** | 167,504 | 83,712 | 83,712 | 83,712 |
| launch__occupancy_limit_registers / _shared_mem (blocks) | 1 / 1 | **2 / 2** | 1 / 1 | 2 / 2 | 2 / 2 | 2 / 2 |
| launch__waves_per_multiprocessor | 1.03 | 1.55 | 1.03 | 1.55 | 1.55 | 1.55 |
| sm__warps_active (% of peak, active cycles); theoretical | 12.2 %; 12.5 | **21.1 %; 25** | 12.3 % | 21.5 % | 21.7 % | 22.1 % |
| smsp__issue_active (% of active cycles) | 44 (0.44 issued/scheduler-cycle) | 43.9 | 39 | 43.7 | 41.5 | 33.9 |
| smsp__inst_executed.sum | 46.36 M | 50.21 M | 48.31 M | 48.79 M | 40.50 M | 24.54 M |
| warp-cycles per issued instruction | 4.52 | 7.61 | 5.06 | 7.87 | 8.25 | 10.31 |
| .. of which no_instruction (warps per issue-active cycle) | 0.52 | 2.16 | 0.77 | 2.42 | 1.64 | 1.86 |
| .. long_scoreboard | - | 1.19 | - | 1.35 | 1.63 | - |
| dram__throughput (% of peak, elapsed) | - | 21.1 | - | 11.9 | 21.7 | - |
| gpu__time_duration (serialized) | 283.6 us | **198.3** | 311.0 | **189.2** | 173.2 | 123.0 |

The register and smem limits are both 2 blocks and the achieved warps per SM
went 7.8 -> 13.5-14.1 of 16 (the 1.55-wave tail keeps it under the theoretical
25 %).  Issue utilisation per active cycle did not double: the second CTA
raises the warp latency per issued instruction from 4.5-5.1 to 7.6-8.3 cycles,
and the largest single component is now instruction fetch (`no_instruction`
2.2-2.4 warps per issue cycle, up from 0.5-0.8): the four warps per SMSP (a
gemm0 and a gemm1 warp of each CTA) sit in four code regions instead of two.
The gain is in wall time, not in SM issue rate: instructions per kernel are
within 5 % (the 64 B parts add copy/expand iterations, the 16-row tile removes
MMA/softmax rows, and 3x more CTAs pay the prologue) while the serialized
kernel time fell 30-39 %.

**XQA_NB_SUB_SEQ sweep on the 2-CTA/SM build (nkcut2, flock'd, `--repeats 2
--trials 5`, medians; grid = 136 x n CTAs on 264 slots).**

| n (waves) | transport_a16 | fp8 | fp4 | mixed |
|---|---|---|---|---|
| 1 (0.52; 1 CTA/SM in practice) | 105.3 | 165.4 | 194.3 | 201.1 |
| 2 (1.03) | 116.2 | 166.7 | 183.2 | 181.8 |
| 3 (1.55) | 100.5 | 128.4 | 148.1 | 151.8 |
| 4 (2.06) | 101.3 | 136.7 | 154.2 | 153.9 |
| **5 (2.58)** | **98.6** | **123.7** | **144.3** | **138.3** |
| 6 (3.09) | 99.1 | 131.1 | 153.5 | 154.5 |
| 8 (4.12) | 98.7 | 127.5 | 144.4 | 159.7 |

Every "just over an integer wave" count (2, 4, 6) pays a nearly full CTA time
for its tail, exactly the discrete-wave reading of the 1-CTA/SM sweep (n=1 =
1.03 waves there); the n=1 row isolates [41] at 1 CTA/SM: fp4 237 -> 194
(0.82x), fp8 199 -> 165, a16 136 -> 105, mixed 216 -> 201 (the M-tile work
halving minus the doubled K round trips).  The host model's first version
(ceil tiles per CTA, no hysteresis) chose n=3; the sweep calibrated it to the
mean tiles per CTA (16/5 = 3.2: 4- and 3-tile CTAs backfill each other) with
the 5 % hysteresis that keeps ws-1 at n=1 - both hosts' sweeps are reproduced
by the same constants.

**Timing (flock'd, `--repeats 2 --trials 5`, three rounds of the shipped
default; the `kernel_family` line reports `mha.cu spec_dec=True` and the
`..._spec_q_seq_len_4` URIs for every q=4 mode; ncu `launch__grid_size` 680 =
136 x 5, `launch__waves_per_multiprocessor` 2.58, `launch__occupancy_limit_registers`
2, `launch__occupancy_limit_shared_mem` 2 on the shipped build).  q=1 is the
untouched `mha_sm90.cu` control (session offset +1-3 % against the step-3
record: 83.0 / 91.2 / 96.0 / 115.6).**

| mode | q=4 step-3 record ([40]) | q=4 [41]+[42] r1 / r2 / r3 | ratio | q=4 target | q=1 control |
|---|---|---|---|---|---|
| transport_a16 | 136.4 | 100.0 / 99.7 / 99.6 | **0.73x** | 135 (pass) | 85.6 |
| fp8 | 198.7 | 124.2 / 124.3 / 124.8 | **0.63x** | <= 94 (open, 1.32x) | 92.2 |
| fp4 | 236.1 | 144.1 / 144.3 / 145.7 | **0.61x** | <= 59 (open, 2.45x) | 97.2 |
| mixed | 216.4 | 137.8 / 137.8 / 136.0 | **0.64x** | <= 101 (open, 1.36x) | 117.0 |

Against the prediction (fp4 130-150, fp8 115-135, mixed 125-155, a16 95-105):
every mode landed inside its band.  mixed stays below fp4 (0.95x) and the
Track S acceptance (mixed <= 1.5x fp4) holds.  Correctness: 34/34 bit-exact on
nkcut2 and ws-1 with the shipped default and again with `XQA_NB_SUB_SEQ=2`
(the multi-block merge path of the 2-CTA/SM SPEC_DEC build; the test shape's
2 tiles never trigger it from the model).

**sm120 regression table (ws-1 RTX 5090, flock'd, `--repeats 5 --trials 5`,
three rounds; q=1 modules are unchanged by [41] and the host rule keeps n = 1
there - torch.profiler grid `[1, 8, 17]` at q=1 and q=4; q=4 modules take the
16-row M tile).**

| mode | q=1 record | q=1 after (r1 / r2 / r3) | q=4 record ([29]) | q=4 after (r1 / r2 / r3) | q=4 target |
|---|---|---|---|---|---|
| transport_a16 | 174.7 | 172.8 / 173.4 / 173.2 | 179-184 | 176.2 / 176.0 / 176.1 | - |
| fp8 | 100.5 | 100.6 / 100.7 / 100.3 | 125.0 | **115.1 / 115.1 / 115.4** | <= 128 (pass) |
| fp4 | 59.5 | 59.6 / 59.8 / 59.8 | 81.9 | **65.8 / 65.5 / 65.7** | <= 81 (pass, was within 1 %) |
| mixed | 113.5 | 113.5 / 113.6 / 113.5 | 132.3 | **119.0 / 118.7 / 119.6** | <= 138 (pass) |

**What is left for the sm90 q=4 targets (94 / 59 / 101).**  The kernel now
runs 13.5-14 warps per SM at 42-44 % issue-active with 7.6-8.5 warp-cycles per
issued instruction, 2.2-2.8 of them instruction-fetch (`no_instruction`) and
1.2-1.6 long-scoreboard; DRAM is at 12-22 % of peak.  Occupancy is exhausted
(3 CTAs/SM would need <= 85 registers and 3 x 82 KB of smem), so the remaining
levers are per-tile issue count and code footprint: the dynamic instruction
count per warp-tile (2,900 for mixed, 2,800 fp4, 2,300 fp8 at 8 warps) and the
four-region i-fetch pattern of two co-resident CTAs, i.e. a smaller, less
unrolled per-tile body (the 64 B parts run 4 copy/expand/MMA rounds per tile)
or the `mha_sm90.cu` SPEC_DEC route whose smem arithmetic is in the step-3
section (K3/V2 fits 2 CTAs/SM at 99 KB).  The a16 mode is 1.48x its 67.5 us
byte floor with 680 CTAs x ~6 us of fixed cost = 15 us of the gap.
### Phase 1 — sm90 consumer cheap set (track A, H200 nkcut2, 2026-09-04)

Levers [4] [0] [2] [1] [33c] [35] of `docs/mixed_kv_speed_plan.md`, one commit
each on branch `wt/A` (`csrc/xqa/mha_sm90.cu`, gemm0 / gemm1 warp-group
bodies only; the barrier-init block and the IO / converter roles are
untouched, [2] adds one `SharedMem` member).  Every step was verified in
this order: SASS opcode counts of the two consumer regions
(`benchmarks/microbench/sass_consumer_regions.py` on `cuobjdump -sass` of
the fp4 `static_format_2` `mha_sm90` object, `cuobjdump -res-usage` REG:48
STACK:0 at every step), the 34-case bit-exact matrix plus two new
independent-reference cases (`tests/attention/run_xqa_mixed_page_transport.py`:
all-A16 mixed stream on `mha_sm90.cu` vs the stock `mha.cu` decode, tolerance
one bf16 ulp), the `MIXED_KV_TRACE=1` consumer slots (CTA 0, tiles 2-7
medians over 26 launches, two rounds, fp4 and transport_a16 production and
fp4 with `-DMIXED_KV_EXPERIMENT=1` converters skipped), and only then the
stopwatch (`bench_xqa_mixed_page_transport.py --repeats 5 --trials 5` under
the GPU lock, base and steps interleaved in one session; VLLM co-tenant at
100 % SM throughout, nvidia-smi SM clock 1980 MHz, cycles below are clock64
deltas).

**SASS, consumer regions (fp4 static build; kernel totals in parentheses).**

| step | gemm0: BAR.SYNC / PHASECHK / ARRIVE / ATOMS / DEPBAR | gemm1: BAR.SYNC / PHASECHK / ARRIVE / DEPBAR | kernel instr |
|---|---|---|---|
| base | 0 / 6 / 15 / 4 / 1 | 0 / 12 / 16 / 4 | 3336 (BAR.SYNC 5, PHASECHK 91, ARRIVE 48, ATOMS 4, DEPBAR 5) |
| [4] | 2 / 4 / 13 / 4 / 1 | 2 / 9 / 14 / 4 | 3344 (9, 80, 44, 4, 5) |
| [0] | 2 / 4 / 13 / 4 / 1 | 2 / 9 / 14 / **1** | 3336 (9, 80, 44, 4, **2**) |
| [2] | **1** / 3 / 12 / **0** / 1 | 2 / 9 / 13 / 1 | 3352 (8, 77, 43, **0**, 2) |
| [1] | 1 / 3 / 12 / 0 / 1 | **1** / 7 / 11 / 1 | 3312 (7, 71, 41, 0, 2) |
| [33c] | 1 / 3 / 12 / 0 / 1 (arrive moved ahead of the K wait, PHASECHK before the STSM) | 1 / 7 / 11 / 1 | 3320 |
| [35] | identical to [33c] (byte-identical SASS; source −232 lines) | | 3320 |

HGMMA stays 8 + 8 at every step; no register-A (`R`-form) HGMMA anywhere.
PHASECHK counts include the out-of-line retry loops (two sites per wait).

**Trace, fp4 with converters skipped (`-DMIXED_KV_EXPERIMENT=1`, the consumer
floor; cycles, two rounds).**  Segments: s1-s0 K ready -> 8 QK HGMMA done;
s2-s1 colMax exchange + softmax; s3-s2 X store + colMax/colSum STS + release
fence + xBar.produced arrive; s5-s4 gemm1 wait for X; s6-s5 rescale; s7-s6
8 PV HGMMA; T_g0 = s3-s0; gemm1 work = s7-s5; cadence = s0(t+1)-s0(t).

| step | s1-s0 | s2-s1 | s3-s2 | s5-s4 | s6-s5 | s7-s6 | T_g0 | gemm1 work | cadence | K-wait |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 669/668 | 686/678 | 484/480 | 696/690 | 456/448 | 638/646 | 1860/1853 | 1094/1092 | 2048/2045 | 190/192 |
| [4] | 682/682 | 481/479 | 526/527 | 706/710 | 262/254 | 691/698 | 1715/1705 | 956/956 | 1904/1890 | 186/188 |
| [4][0] | 678/678 | 482/480 | 530/525 | 710/724 | 260/260 | 694/694 | 1708/1722 | 955/958 | 1906/1912 | 186/187 |
| +[2] | 682/675 | 448/450 | 535/538 | 688/690 | 271/273 | 712/716 | 1691/1688 | 982/993 | 1900/1895 | 196/198 |
| +[1] | 699/698 | 452/452 | 503/508 | 780/766 | 206/213 | 664/662 | 1672/1670 | 874/872 | 1879/1864 | 189/192 |
| +[33c] (reverted) | 694/690 | 458/468 | 500/500 | 808/784 | 209/208 | 660/656 | 1680/1672 | 881/876 | 1892/1898 | 202/198 |
| base, final session | 670/674 | 682/686 | 468/466 | 706/720 | 446/451 | 638/636 | 1848/1836 | 1082/1082 | 2039/2030 | 190/192 |
| **final tree**, final session | 700/699 | 454/459 | 501/509 | 780/787 | 208/210 | 658/661 | 1676/1683 | 869/868 | **1884/1888** | 190/192 |

13-tile window (`-DMIXED_KV_TRACE_TILES=13`, CTA 0 tiles 1-12): the cadence
is uniform across the sub-sequence (base 1960-2085 per tile, final 1814-1926;
tile 4->5 carries the residency probe's global atomic), mean 2055-2058 ->
1897-1898 cyc (-7.7 %); CTA 0's main loop (s0(t0) -> s7(t12)) 28.4k -> 26.4k
cyc.

**Trace, fp4 production (cycles).**  Cadence is converter-paced (P0.3): the
consumer's savings turn into K-wait.

| step | s2-s1 | s3-s2 | s6-s5 | s7-s6 | T_g0 | gemm1 work | cadence | K-wait |
|---|---|---|---|---|---|---|---|---|
| base | 872/859 | 600/590 | 572/574 | 748/749 | 2359/2359 | 1326/1331 | 2700/2708 | 269/336 |
| [4] | 601/604 | 589/609 | 326/326 | 872/874 | 2106/2116 | 1201/1208 | 2636/2663 | 571/522 |
| [4][0] | 620/598 | 594/598 | 314/324 | 884/880 | 2114/2091 | 1198/1214 | 2632/2644 | 550/549 |
| +[2] | 462/470 | 486/478 | 324/335 | 840/828 | 1826/1809 | 1168/1156 | 2648/2646 | 812/860 |
| +[1] | 446/444 | 484/475 | 291/280 | 724/714 | 1804/1757 | 1012/996 | 2659/2683 | 872/873 |
| +[33c] (reverted) | 443/456 | 548/552 | 304/290 | 724/726 | 1898/1916 | 1031/1020 | 2679/2668 | 775/784 |
| base, final session | 870/884 | 590/596 | 568/570 | 749/742 | 2341/2372 | 1314/1318 | 2678/2704 | 269/285 |
| **final tree**, final session | 444/449 | 478/480 | 286/276 | 714/719 | 1792/1794 | 993/1005 | 2655/2714 | 895/928 |

transport_a16 (DRAM-bound, K-wait 400-1000): s2-s1 502/510 -> 291/296, s6-s5
408/411 -> 189/192, T_g0 1666/1672 -> 1454/1470 at [1]; cadence unchanged
(2500-2900, set by TMA landing).  Final session (K-wait 2200-2700): s2-s1
464/468 -> 274/277, s6-s5 411/407 -> 182/180, T_g0 1540/1568 -> 1356/1362,
cadence 4140-4600 either way.

**Walls (median_us, `--repeats 5 --trials 5`, CUDA graph, GPU lock, two
interleaved rounds; q=1 unless noted).**

| step | fp4 floor (EXPERIMENT=1) | transport_a16 | fp8 | fp4 | mixed |
|---|---|---|---|---|---|
| base | 57.00/57.02 | 82.59/82.87 | 91.47/91.23 | 96.35/96.04 | 115.32/114.34 |
| [4] | 56.75/56.81 | 82.79/82.50 | 90.80/90.84 | 95.21/95.10 | 114.93/113.38 |
| [4][0] | 57.01/56.99 | 82.46/82.25 | 91.16/91.05 | 95.06/95.12 | 114.00/114.35 |
| +[2] | 54.90/54.93 | 82.06/82.33 | 89.15/89.25 | 93.96/94.44 | 111.19/111.01 |
| +[1] | 55.07/54.94 | 81.70/82.00 | 88.75/88.88 | 92.78/92.72 | 109.56/109.96 |
| +[33c] (reverted) | 56.45/56.27 | 82.07/82.11 | 89.06/89.16 | 93.84/93.86 | 110.10/109.95 |
| +[35] (= [33c] code, SASS-identical) | 56.58/56.58 | 82.33/82.30 | 88.74/88.70 | 93.25/93.38 | 110.20/109.16 |
| base, re-timed in the final session | 57.22/57.50 | 83.03/82.69 | 91.76/91.03 | 96.40/95.87 | 115.73/114.05 |
| **final tree** ([4][0][2][1][35]; [33c] reverted) | **55.04/55.11** | **82.25/82.02** | **88.78/88.72** | **92.72/92.65** | **109.74/109.41** |

q=4 (SPEC_DEC, `mha.cu`, not touched by this track; a16 / fp8 / fp4 / mixed):
base 137.1 / 230.6-228.0 / 277.1-274.6 / 934.3-934.6 vs final 136.6-136.5 /
228.5-226.0 / 276.0-275.0 / 933.8-935.1 (step session); final session base
136.9/136.0 / 227.3/226.4 / 277.0/277.4 / 930.4/936.1 vs final tree 135.5/134.7 /
225.8/227.6 / 277.5/275.1 / 935.4/937.5.  The mixed q=4 figure at `--repeats 5`
is the repeats x time > 1.5 ms co-tenant artifact (kernel ~440 us, P0.1).

**Final tree, verified as a whole (separate session, base re-timed alongside).**
`cuobjdump -sass` of the final fp4 static `xqa_mha_sm90.cuda.o`: 3312
instructions, REG:48 STACK:0, instruction stream byte-identical to the [1]-step
object (`cmp` of the opcode/operand text: 0 differing lines; the [35] deletion
and the [33c] revert leave the code of step [1]); consumer regions gemm0
BAR.SYNC 1 / PHASECHK 3 / ARRIVE 12 / ATOMS 0 / DEPBAR 1, gemm1 1 / 7 / 11 / 1,
HGMMA 8 + 8, no register-A HGMMA.  Correctness 36/36 (34 bit-exact + the two
independent-reference cases at max |diff| 1.953e-3 / 4.883e-4, the calibration
values of the unmodified kernel).  One semantic note on [2]: the replaced float
`atomicMax` (`utils.cuh:305`, signed-int max / unsigned min on the bit pattern)
and `fmax` agree on every finite value and on -0/+0 (the downstream
`exp2(x - max)` is identical either way); they differ only when a QK score is
NaN (the bit-pattern max propagates it as the column max, `fmax` drops it),
which no valid input produces and the matrix does not exercise.

**Per-lever verdicts.**

- [4] named barriers: accepted.  Both two-RT segments lose ~200 cyc (s2-s1
  686 -> 480, s6-s5 452 -> 258), i.e. ~100 cyc per mbarrier round trip
  replaced (between the isolated 65 and the loaded 120-200 of P0.3 (d));
  consumer floor cadence -7 %.  The plan's "s2-s1 <= 300 + shuffles" was
  optimistic: the segment also holds the qkScale multiply, the mask test, the
  three shuffle rounds x 2 values and the 4 exp2 of softmax.
- [0] single commit / wait for PV: negative on its criterion (s7-s6 <= 250):
  691/698 -> 694/694 with DEPBAR 4 -> 1 in SASS, i.e. the PV segment is not
  drain-bound; eight back-to-back m64n8k16 SS HGMMAs cost the same ~690 cyc
  as four commit/wait pairs (microbenchmark floor 185).  Kept: bit-exact,
  fewer instructions, no protocol change; the ~85 cyc per HGMMA (also seen in
  gemm0's 8 QK HGMMAs, 670-700 cyc) is the issue-side item P0.3 flagged
  (descriptor arithmetic / tensor-pipe sharing between the four resident
  warp groups), not the wait.
- [2] one barrier per tile: accepted on SASS (ATOMS 4 -> 0, gemm0 sync sites
  2 -> 1 per tile), s2-s1 480 -> 449 (-30: the bar.sync and the two
  ATOMS.MAX/MIN pairs it replaced, minus three extra LDS); the criterion
  "s2-s1 <= 250" is not met for the reason above.  The wall moved here:
  floor 56.8-57.0 -> 54.9 us, production fp4 95.1 -> 94.0-94.4, mixed
  114.0-114.9 -> 111.0-111.2.
- [1] register-resident colMax/colSum: accepted on SASS (gemm1 sync sites
  per tile 2 -> 0, one bar.sync left before finalize; REG 48 STACK 0),
  s6-s5 271 -> 206-213, gemm1 work 982-993 -> 872-874; the criterion
  "s6-s5 <= 100" is not met - the remaining ~210 cyc is the dependent LDS ->
  compare -> ballot -> exp -> shuffle -> FMUL chain plus 4 LDS + adds and the
  stamp itself.  Production: fp4 92.7-92.8, fp8 88.8-88.9, mixed 109.6-110.0,
  a16 81.7-82.0.
- [33c] split xBar.consumed arrive/wait: rejected and reverted (see the
  revert commit): s3-s2 unchanged in the floor (503-508 -> 500), +70 in
  production, floor wall +1.4 us, production fp4 +1.1 us.  The consumed
  phase is complete long before the store (P0.3), so nothing was hidden.
- [35] RS loaders deleted: SASS byte-identical (diff 0 lines), source -232
  lines; closes [18] (below).

**Where Phase 1 leaves the sm90 consumer.**  Floor cadence 2046 -> 1870 cyc
(-8.5 %, tiles 2-7) with T_g0 1856 -> 1671 and gemm1 work 1093 -> 873; the
floor wall moved 57.0 -> 55.0 us (-3.5 %), less than 39 x 0.09 us because the
wall carries ~18 us of per-wave fill/drain/tail (P0.3 (b), P0.5) that the
per-tile chain does not touch - that is [8]'s territory.  Production moved
more than the plan's "0 +-2 us": fp4 96.0-96.4 -> 92.7-92.8 (-3.4 us), fp8
91.2-91.5 -> 88.8-88.9 (-2.6), mixed 114.3-115.3 -> 109.6-110.0 (-4.9), a16
82.6-82.9 -> 81.7-82.0 (-0.8), at an unchanged converter-paced cadence
(2700 -> 2660-2680): the consumer now idles ~870 cyc per tile in K-wait
instead of spinning on mbarriers and shared-memory atomics, which frees issue
slots for the co-resident converter groups (P0.3's fast/slow pair asymmetry
is the likely beneficiary).  The consumer chain that remains (T_g0 ~1670:
QK HGMMA 690, colMax+softmax 450, X store + release + arrive 505; K-wait
190) is issue/latency-bound with one bar.sync and one mbarrier arrive/wait
per tile per group; the next consumer items are the HGMMA issue cost (P0.3's
candidate lever) and the X-store release path, both outside this cheap set.

**RS-decode in the GEMM groups: negative result (plan [35], closes [18]).**
The register-A wgmma route (`loadMixedKTileFragment` / `loadMixedVTileFragment`,
per-format A16 / E4M3 / E2M1 fragment decoders that were never instantiated)
is deleted.  Arithmetic: decoding one m64n8k16 A fragment per k-step costs
35-50 SASS per lane (nibble/byte extraction, `cvt`, block-scale multiply,
shuffles for the FP8 pair layout) against a 10-20 cycle wgmma; at 4 cyc/instr
that is +1.2-1.4 us per group per tile, 2-3x more at the 10.8-13.3 cyc/instr
the converters actually achieve (P0.4), on the gemm0 chain that P0.3 shows is
the binding consumer role.  The converter warp groups keep the expansion
(`expandPackedStage`); the SASS of the sm90 TU is byte-identical before and
after the deletion, confirming the loaders were dead code.

### Round 2 baseline — wt/A + wt/B merged, and the setmaxnreg that never existed (2026-09-04, nkcut2 H200, main @ de2351fc / fe2e9a33)

Matrix 54/54 (32 register-expansion + 2 native FP8 + 18 tail/value-range + 2
independent stock-decode reference).  Locked bench, 5 x 5, B=17 S=4096 8 KV
heads GQA 4 D=128, min/median/max:

| q | mode | main @ de2351fc (A+B, split dropped) | main @ fe2e9a33 (split honored) | target |
|---|---|---|---|---|
| 1 | transport_a16 | 81.2 / **81.4** / 81.8 | 81.2 / **81.8** / 81.9 | parity (stock mha.cu 108-110) |
| 1 | fp8 | 75.4 / **75.8** / 76.3 | 76.6 / **76.9** / 77.1 | <= 58 |
| 1 | fp4 | 73.9 / **74.2** / 74.4 | 70.2 / **70.7** / 70.8 | <= 36 |
| 1 | mixed | 82.9 / **83.3** / 83.8 | 79.3 / **79.5** / 80.5 | <= 62 |
| 4 | transport_a16 | 93.5 / 93.7 / 94.0 | 93.4 / 94.1 / 94.7 | parity |
| 4 | fp8 | 118.8 / 119.9 / 120.8 | 118.8 / 120.1 / 120.9 | <= 94 |
| 4 | fp4 | 136.0 / 140.0 / 141.3 | 138.3 / 139.0 / 141.1 | <= 59 |
| 4 | mixed | 132.1 / 135.3 / 135.9 | 135.3 / 136.6 / 136.9 | <= 101 |

A+B combined equals wt/B alone (77.2 / 73.1 / 83.4): Track A's consumer
levers add nothing on top of B — reading (ii) of the round-1 synthesis
("the consumer does not pace"), settled below by the same-launch trace.

**setmaxnreg was silently dropped.**  `ptxas -v` on the merged TU (real
ninja flags, `-arch=sm_90a`) prints
`(C7507) Potential Performance Loss: 'setmaxnreg' ignored to maintain minimum
register requirements` and the SASS has zero `USETMAXREG`.  The IO group's
`.dec 24` was below that role's own register need, and ptxas then drops
*every* setmaxnreg in the kernel, so the converters ran at the launch cap of
48 all along (the mixed module carried a 16-byte stack frame, 24 bytes of
spill; the fp4 module 8 bytes).  Re-assembling the PTX with candidate splits:

| IO .dec | converter .inc | C7507 | stack / spill | USETMAXREG in SASS |
|---|---|---|---|---|
| 24 | 56 or 64 | ignored | 16 B / 24 B | 0 |
| 32 | 56 or 64 | honored | 32 B / 120 B (IO spills) | 2 |
| 40 | 56 or 64 | honored | 0 / 0 | 2 |

Pool balance at `__launch_bounds__(640, 2)`: 640 x 48 = 30720; the fix
(fe2e9a33) is IO **and both GEMM groups** `.dec 40`, converters `.inc 56`:
3 x 128 x 40 + 2 x 128 x 56 = 29696.  Verified: no C7507, 0 spill,
`USETMAXREG.DEALLOC.CTAPOOL 0x28` + `USETMAXREG.TRY_ALLOC.CTAPOOL UP1, 0x38`,
REG 48 STACK 0 on the mixed module, 2 CTAs/SM (ncu: registers and smem both
limit 2).  Effect: fp4 74.2 -> 70.7, mixed 83.3 -> 79.5, fp8 unchanged —
i.e. the register-starved converter was the FP4 pace, and the FP8 pace is
elsewhere (below).  Rule added to the dataflow doc (A4): every sm90 build
check is `ptxas -v` free of C7507 and two `USETMAXREG` in the SASS; the
`cuobjdump -res-usage` REG line cannot show this (it reports the launch cap).

**Same-launch role periods** (MIXED_KV_TRACE=1 on de2351fc, i.e. before the
split fix; CTA 0, tiles 3-7, medians over 3 launches, 1.98 GHz; nbIters 13,
nbSubSeq 5, grid 1 x 5 x 136 = 680 CTAs = 2.58 waves at 264 slots):

| mode | gemm0 period | gemm1 | K load issue | V load issue | K convert done | V convert done | 13-tile CTA body |
|---|---|---|---|---|---|---|---|
| fp8 | 1.38 us | 1.46 | 1.43 | 1.39 | **1.62** | 1.02 | ~18 us |
| fp4 | 1.14 | 1.15 | 1.13 | 1.17 | 1.18 | 1.12 | ~15 us |
| a16 | 2.52 | 2.42 | 2.38 | 2.68 | 2.88 | 2.51 | ~33 us |

- fp8 is K-converter paced (1.62 us against a 1.4 us consumer cadence; the V
  converter runs the same code in 1.02) — the [43] parity question is now
  "why is K 60 % slower than V on fp8", not the fp4 V-slower reading from
  round 1.  Both converters have 56 registers only from fe2e9a33 on.
- fp4 has no single pacing role: every period sits at 1.13-1.18 us, the
  latency floor (P0.3: 1.00 us/tile consumer floor at 2 CTAs/SM).
- a16 is transport paced (2.5 us/tile = 32 KB x 264 / 2.5 us = 3.4 TB/s).
- The CTA body is 15-18 us of a 71-77 us wall: with 680 CTAs over 264 slots
  the wall is ~3 lifetimes x (body + ~4-5 us fill/first-K latency) + tail.
  Wave quantisation and per-CTA fill are 55-60 % of the fp8/fp4 wall — [8]
  (persistent 264-CTA pull with cross-item prefetch) is the decisive lever:
  8840 tile-iterations / 264 CTAs = 33.5 tiles per CTA -> fp8 ~47 + fill ~
  53 us, fp4 ~39 + fill ~ 45 us at unchanged periods.

### Track S step 5 — [43] 128 B K parts, rolled tile loops, one copy body per format (2026-09-04, nkcut2 H200 + ws-1 RTX 5090, worktree E)

**Smem arithmetic (before any build; sm90, enum 5, SPEC_DEC, M_TILESIZE 16,
`!grpLoadV`, tokensPerPage 16, D 128, bf16).**  Cap for two CTAs per SM:
2 x (sizeof(SharedMem) + 1,024 B reserve) <= 228 KB = 233,472 B, i.e.
sizeof(SharedMem) <= 115,712 B (allocation granularity 128 B).

| SharedMem member | step 4: K 64 B / V 32 rows | K 128 B / V 32 rows | K 64 B / V 64 rows | [43]: K 128 B / V 32 rows, 4 B vScales |
|---|---|---|---|---|
| `q[1][1]` 16 rows x 256 B | 4,096 | 4,096 | 4,096 | 4,096 |
| `k[4][2]` 64 rows x part bytes | 32,768 | 65,536 | 32,768 | 65,536 |
| `x[1][4]` 16 rows x 128 B | 8,192 | 8,192 | 8,192 | 8,192 |
| `v[2][2][2]` rows x 128 B | 32,768 | 32,768 | 65,536 | 32,768 |
| `kFormats[4][2]` (4 tags) / `vFormats[2][2][2]` (2 tags) | 32 + 16 | 32 + 16 | 32 + 32 | 32 + 16 |
| `kNeedsExpansion` / `vNeedsExpansion` | 8 + 8 | 8 + 8 | 8 + 8 | 8 + 8 |
| `kScales[4][2][65][4]` (4 B = max(4, part/32)) | 2,080 | 2,080 | 2,080 | 2,080 |
| `vScales[2][2][2][rows+1][stride]` | 33 x 8: 2,112 | 33 x 8: 2,112 | 65 x 8: 4,160 | 33 x 4: **1,056** |
| `warpRowMax` / `warpRowSum` / `ctaRowMax` [1][4] x 128 B | 3 x 512 | 3 x 512 | 3 x 512 | 3 x 512 |
| `qBarrier[1]` 8 B + `xBarriers[1][4]` 4 x 16 B | 72 | 72 | 72 | 72 |
| sum / `alignas(128)` | 83,688 / **83,712** | 116,456 / **116,480** | 118,568 / 118,656 | 115,400 / **115,456** |
| 2 x (size + 1,024) vs 233,472 | 169,472 yes | 235,008 **no (+768 B)** | 239,360 no | 232,960 **yes (512 B spare)** |

The step-4 column reproduces ncu's 83,712 B, so the accounting is exact.  The V
scale rows were the slack: `copyMixedPartialHeadsAsync` writes them at
`scaleLoadBytes = max(4, blocksPerPart)` = 4 B for a warp's 128 B half row
(`nbPartsPerHead = gemm1WarpsPerGrp = 2`) and `expandMixedPartialHeadsInPlace`
reads them at the same stride, while `mixedVScaleBytes` allocated 8 B per row
(the whole-head value that only `grpLoadV` uses).  No other member is
compressible without touching the consumers (`SMemWarpRowMax` holds 32 rows
for a 16-row tile: 768 B, the exact deficit, but it is indexed by the
gemm0/gemm1 softmax code; the X ring is already depth 1; kScales' dump row is
32 B).  64-row V tiles do not fit in any combination.

**Per-tile rounds and code footprint (predicted from the step-4 SASS).**  The
gemm0 K-part loop is fully unrolled for 16-bit tiles (`nbUnroll =
nbPartsPerKHead`), so the dynamic module's hot gemm0 loop (3,239 SASS,
back-edge 0x02ee0-0x0f940) held four part bodies of ~790 instructions (each:
copy with 3 formats x {isFull, partial} = 6 bodies, expansion with 2 format
bodies, 16 HMMA, and the per-round fixed cost: wait_group, tag broadcast, flag
stores, page advance); the gemm1 X-tile loop (`#pragma unroll` over
`nbXItersPerCtaTile` = 4) held four V bodies of ~875 (3,500 SASS,
0x16080-0x23b30).  Two co-resident CTAs put a gemm0 and a gemm1 warp of each
on every SMSP: 6.7 K instructions = 108 KB of hot code per SMSP, hence the
`no_instruction` 2.2-2.8 warps per issue cycle of step 4.  [43]: 128 B parts
(2 rounds per K tile, the round's fixed cost paid twice not four times), the
part loop rolled (one 128 B body: 32 HMMA, 10 LDGSTS, 2 expansion iterations
per format), the X-tile loop rolled (one V body), and every K/V copy through
the bounds-checked variant (`isFullTile` forced false in the compact build;
one ISETP per block instead of a second copy body).  Predicted hot footprint:
dyn ~1,300 + ~920, i.e. 6,739 -> ~2,200-2,500 (below the fp4 static module's
6,647 and the a16 module's 4,426); executed instructions -5 to -8 % (two fewer
rounds of fixed cost, minus ~30-50 loop-control instructions per rolled
iteration); no_instruction -> <= 0.8; q=4 mixed 95-115, fp8 88-105, fp4
100-125, a16 90-100 us.

**Levers.**  `csrc/xqa/mha.cu`: the `__CUDA_ARCH__ == 900 && CACHE_ELEM_ENUM ==
5 && SPEC_DEC && M_TILESIZE == 16` branch selects `preferedKHeadPartBytes =
128`, `cacheVTileSeqLen = 32` and defines `MIXED_COMPACT_TILE_LOOPS 1` ->
`kCompactTileLoops`; `SharedMem::mixedVScaleBytes` = 4 B under
`kCompactTileLoops && !grpLoadV` (8 B elsewhere, so every other build keeps its
SharedMem layout byte-for-byte); gemm0 `nbUnroll = 1` and gemm1 `#pragma unroll
1` (a `#if MIXED_COMPACT_TILE_LOOPS` branch around the stock `#pragma unroll`)
under the compact build; K and V `isFullTile = !kCompactTileLoops && ...` with
the bounds-checked copy's `nbHeadsAvail` clamped to the tile only there.  No
change to `mhaUtils.cuh` (the page loop of the mixed helpers is already rolled
in the dynamic module: 9-10 LDGSTS per body confirm it in the SPEC_DEC SASS).

**Artifact (sm90 q=4 modules, `cuobjdump -sass` / `-res-usage`, workspace
`/tmp/mixedkv-wtE-s5-v1`; loops delimited by their back-edges).**

| module | SASS total | gemm0 loop | gemm1 loop | hot total (code bytes) | LDGSTS / LDS / STS static | REG / STACK / LDL / STL |
|---|---|---|---|---|---|---|
| dyn (-1) before | 9,288 | 3,239 (4 x 790 part bodies) | 3,500 (4 x 875) | 6,739 (105 KB) | 121 / 340 / 82 | 128 / 0 / 0 / 0 |
| dyn (-1) after | **4,968** | **1,588** (part body 921: 32 HMMA, 10 LDGSTS) | **888** (V body: 16 HMMA, 9 LDGSTS) | **2,476 (39 KB)** | 54 / 152 / 35 | 124 / 0 / 0 / 0 |
| fp4 (2) before / after | 9,104 / **4,688** | 2,816 / 1,818 | 3,831 / 771 | 6,647 / **2,589** | 101 / 342 / 98 -> 46 / 131 / 43 | 128 / 0 -> 128 / 0 |
| fp8 (1) before / after | 8,160 / **4,352** | 2,342 / 1,591 | 3,352 / 652 | 5,694 / **2,243** | 101 / 342 / 98 -> 46 / 131 / 43 | 128 / 0 -> 127 / 0 |
| a16 (0) before / after | 7,008 / **3,928** | 2,230 / 1,205 | 2,196 / 443 | 4,426 / **1,648** | 176 / 119 / 34 -> 64 / 59 / 19 | 128 / 0 -> 127 / 0 |

Every module's hot footprint is below the step-4 fp4 module's (the target) and
below the step-4 a16 module's.  The gemm0 part body carries 32 HMMA / 20 LDSM
/ 10 LDGSTS / 2 x (F2FP 20, PRMT 24) - the 128 B round predicted above.

**ncu (q=4, one launch, `--launch-skip 1 --launch-count 1`, co-tenant present so
durations are serialized; "before" = step-4 record).**

| metric | mixed before -> after | fp4 before -> after | fp8 before -> after | a16 before -> after |
|---|---|---|---|---|
| launch__shared_mem_per_block_dynamic | 83,712 -> **115,456** | same | same | same |
| launch__occupancy_limit_registers / _shared_mem | 2 / 2 -> 2 / 2 | 2 / 2 | 2 / 2 | 2 / 2 |
| launch__registers_per_thread | 128 -> 124 | 128 -> 128 | 128 -> 127 | 128 -> 127 |
| launch__grid_size / waves | 680 / 2.58 | 680 / 2.58 | 680 / 2.58 | 680 / 2.58 |
| sm__warps_active (% peak, active) | 21.1 -> 21.1 | 21.5 -> 20.9 | 21.7 -> 21.5 | 22.1 -> 21.0 |
| smsp__issue_active (% active cycles) | 43.9 -> **58.3** | 43.7 -> **58.1** | 41.5 -> 52.0 | 33.9 -> 50.4 |
| smsp__inst_executed.sum | 50.21 M -> 51.47 M (+2.5 %) | 48.79 -> 50.85 M (+4.2 %) | 40.50 -> 42.76 M (+5.6 %) | 24.54 -> 29.85 M (+21.6 %) |
| warp-cycles per issued instruction | 7.61 -> **5.78** | 7.87 -> **5.76** | 8.25 -> 6.61 | 10.31 -> 6.61 |
| .. no_instruction (warps per issue-active cycle) | 2.16 -> **0.51** | 2.42 -> **0.30** | 1.64 -> **0.15** | 1.86 -> **0.14** |
| .. long_scoreboard | 1.19 -> 0.93 | 1.35 -> 0.99 | 1.63 -> 1.39 | - -> 2.81 |
| .. short_scoreboard / wait / barrier | 0.51 / 1.24 / 0.12 | 0.69 / 0.97 / 0.13 | 1.32 / 1.02 / 0.18 | 0.28 / 0.92 / 0.23 |
| dram__throughput (% peak, elapsed) | 21.1 -> 28.3 | 11.9 -> 15.4 | 21.7 -> 27.1 | - -> 66.9 |
| gpu__time_duration (serialized) | 198.3 -> **148.7** | 189.2 -> **147.7** | 173.2 -> 139.3 | 123.0 -> 98.0 |

The i-fetch stall is gone (0.14-0.51) and issue utilisation rose by a third;
the executed instruction count did **not** fall: the rolled loops' control and
dynamic indexing (~60-100 instructions per iteration, e.g. `idxXTile`,
`smemQOffset`, the CircIdx and page-advance selects, 8 R2UR / 35 UMOV in the
part body) cancel the two saved rounds of fixed cost, and the a16 module's
single bounds-checked copy adds a compare per block (+22 %; a16 is DRAM-side,
67 % of peak, so it still gained 20 %).  The remaining latency per issued
instruction is `wait` (fixed-latency dependencies, 1.0-1.2) and long scoreboard
(0.9-1.4): the next lever for the sm90 q=4 targets is the executed instruction
count per tile (51 M for mixed against 30 M for a16 on the same tiles), not
occupancy or fetch.

**Timing (nkcut2, flock'd, `--repeats 2 --trials 5`, three rounds; the
co-tenant rule holds: 2 x 116 us = 0.23 ms per event pair).  q=1 is the
untouched `mha_sm90.cu` control.**

| mode | q=4 step-4 record | q=4 [43] r1 / r2 / r3 | ratio | predicted band | q=4 target | q=1 control |
|---|---|---|---|---|---|---|
| transport_a16 | 99.7 | 86.1 / 86.0 / 86.7 | **0.86x** | 90-100 (better) | 135 (pass) | 82.3 |
| fp8 | 124.3 | 113.8 / 114.0 / 114.9 | **0.92x** | 88-105 (9 us above) | <= 94 (open, 1.21x) | 91.3 |
| fp4 | 144.3 | 114.8 / 116.4 / 116.5 | **0.80x** | 100-125 (in band) | <= 59 (open, 1.96x) | 96.2 |
| mixed | 137.8 | 116.2 / 116.1 / 115.9 | **0.84x** | 95-115 (1 us above) | <= 101 (open, 1.15x) | 114.7 |

Track S acceptance (mixed <= 1.5x fp4) holds at 1.00x.  fp8 gained least
because it had the smallest i-fetch component (1.64) and now sits on short
scoreboard (1.32: the LDS -> F2FP chains of the FP8 expansion, P0.4's finding
on sm90-io).

**Correctness.**  34/34 bit-exact on nkcut2 (default and `XQA_NB_SUB_SEQ=2`)
and on ws-1 (default and `XQA_NB_SUB_SEQ=2`) on the final tree, and on each
intermediate cut (four builds per host).

**sm120 regression table (ws-1 RTX 5090, flock'd, `--repeats 5 --trials 5`,
three rounds, final code: the sm120 modules are unchanged by [43] - the guard
is sm90-only and their SharedMem layout is byte-for-byte the step-4 one).**

Regression proof by construction first: `cuobjdump -sass` (addresses stripped) of the
eight sm120 mixed-page modules (static formats -1/0/1/2 x q=1/q=4) built from
this tree is **byte-identical** to a pristine `e777da96` checkout built in the
same session for 7 of 8 (`fmt2/fmt1/fmt0 q=1`, all four q=4; REG 146 / 151 /
163 / 178 / 196 / 202 / 200, STACK 0); the dynamic q=1 module differs between
the two pristine builds themselves (step-4 workspace `1604033e8955` == this
tree, fresh pristine build `cb834699b703`): ptxas build-to-build variation, not
a source effect.  Two earlier cuts of this step did change the sm120 SASS and
were reverted: the vScales stride shrink for every `!grpLoadV` build, and a
`#pragma unroll(nbXItersPerCtaTile)` spelling of the stock `#pragma unroll`
(nvcc lays the 4-trip loop out differently).  Timing, interleaved pristine
base / this tree per round (a VLLM tensor-parallel pair was resident on the
GPU during this session, so absolute numbers sit 0-2 % above the idle-GPU
records; the interleaving is the control):

| mode | q=1 record | q=1 base r1/r2/r3 | q=1 [43] r1/r2/r3 | q=4 record | q=4 base r1/r2/r3 | q=4 [43] r1/r2/r3 | q=4 target |
|---|---|---|---|---|---|---|---|
| transport_a16 | 174.7 | 173.2 / 173.5 / 172.9 | 173.7 / 173.3 / 174.6 | 176.1 | 176.7 / 176.1 / 176.3 | 176.6 / 176.0 / 176.1 | - |
| fp8 | 100.5 | 100.6 / 100.5 / 100.6 | 100.3 / 100.7 / 100.6 | 115.1 | 116.4 / 115.6 / 116.8 | 116.9 / 116.7 / 116.4 | <= 128 (pass) |
| fp4 | 59.5 | 59.8 / 59.7 / 60.5 | 60.7 / 59.9 / 59.8 | 65.7 | 65.9 / 65.7 / 65.9 | 65.9 / 65.8 / 65.6 | <= 81 (pass) |
| mixed | 113.5 | 115.9 / 115.5 / 116.1 | 115.5 / 115.4 / 115.6 | 119.0 | 123.5 / 123.5 / 123.3 | 123.2 / 123.3 / 123.1 | <= 138 (pass) |

Identical binaries, identical numbers within the round-to-round spread
(<= 0.9 us); the mixed q=4 +4 us against the record is the co-tenant session
offset, present in the base column too.

#### Correction to the round-2 baseline pacing reading (2026-09-04)

The "fp8 is K-converter paced (1.62 vs 1.02 us)" table above was traced on
de2351fc, where ptxas had dropped setmaxnreg and the converters ran at 48
registers with spills.  The lever-[14] design (worktree wt/r2p14,
docs/mixed_kv_speed_round2_lever14.md) re-traced fe2e9a33 (56-register
converters, MIXED_KV_TRACE=3, 2 x 6 launches per mode under the lock): the
K-converter period equals the gemm0 cadence (fp8 2468 vs 2467 cyc, fp4 2247
vs 2222; mixed 3574 with the converter throttled by gemm0's stage release) and
produced(t) precedes gemm0's wait by 1124 / 787 / 2140 cyc.  So after the
register fix no compressed mode is converter paced; [43] (K/V parity) closes
as a pacing question, [14] and [19] are 0 today (their 90-141 cyc segments
lie inside the converter slack), and the round-2 wall is wave quantisation +
fill ([8]) on top of the gemm0 cadence (1.25 / 1.13 / 1.8 us per tile at
1.98 GHz).

### Track F [24] — FA3 compressed prefill: E4M3 decode floor (F24a), second producer warp group (F24b), dynamic page masks (F24c) (2026-09-04, nkcut2 H200, worktree F24, design docs/mixed_kv_speed_round2_fa3_consumer_decode.md)

Commits d6539806 (F24a), e202ef2a (F24b), 5d493a71 (F24c), 6b95ddb0 (review
fixes), 35706f8a (verification fixes) on wt/F24 over 5cc416fd ([23]).  Shared
Hopper files touched by F24b (`kernel_traits.cuh`, `named_barrier.cuh`,
`prefill_sm90.cuh`, `epilogue.cuh`, `sparse_mainloop.cuh`; `variants.cuh` and
`mainloop_mma.cuh` untouched): the brief listed them as untouched, the design
rev 2 requires them (`NUM_COPY_THREADS` hard-coded 128 would give a 128-count
producer barrier, a consumer thread index off by 128 and a 384-count
`kValueEmpty` -> hang); **user sign-off is still owed**; the acceptance used
here is the byte-identity of the stock paged kernel and of the a16 module
against `5cc416fd` (below).

**SASS (`cuobjdump -sass`, persistent kernel of the `*_paged_sm90_kernel_mask_1`
object, producer region = `USETMAXREG.DEALLOC .. EXIT`; artifacts
`nkcut2:/tmp/mixedkv-wtF24-art/sf{-1,0,1,2}_mask1.sass`, `counts4.log`,
`dynmap4.log`, `ptxas_*.log`, `sass_all4.log`).**

| module | region instr | STACK (ptxas) | USETMAXREG | LDGSTS | STS.128 | LDS | VOTE.ALL | F2FP.E4M3 | FMUL / .FTZ | HMUL2 | BAR.SYNC | FENCE | LDL/STL in pair loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp8 (`static_format_1`) | **2093** (2687 at [23]) | 0 B, 0 spills | DEALLOC 0x48 + TRY_ALLOC 0xB8 | 18 x .128 + 18 x .64 (= 6 + 6 per thread per pair over 3 copy sites; [23] 36 + 36) | 30 (2 per block site, 15 sites; [23] 60) | .64 30, .32 63, .128 6 | 15 (one per block site) | 15 (one per site: the cold recompute is CSE'd with the hot scale, so ~15 not the ~24 written) | 30 / 2 (the two `.FTZ` are `g x 2^120` in make_bases) | 240 (8 hot + 8 cold x 2^120 per site) | 4 = 3 x (id 0xe, **256**) + 1 x (id 0x8, **512**) | 3 | **0** |
| fp4 (`static_format_2`) | 2058 | 0 B | 0x48 / 0xB8 | 36 x .64 (payload 8 B + scale) | 30 | .32 93, .128 6 | 0 (no fold) | 15 | 15 / 0 | 120 | 4 | 3 | **0** |
| dynamic (`static_format_-1`) | **3235** (7271 at [23]) | 32 B (32 B stores / 52 B loads) | 0x48 / 0xB8 | 24 x .128 + 18 x .64 | 20 (5 FP8 + 5 FP4 sites) | .64 20, .32 172 | 5 | 10 | 15 / 2 | 120 | 4 | 3 | **4 LDL per pair** (2 per operand) + 5 STL in the item prologue |
| a16 (`static_format_0`) | 12-warp, 0x48 / 0xD8 | 0 B | | | | | | | | | | | byte-identical to `5cc416fd` (2 kernels x 4 mask objects) |
| stock paged (`batch_prefill_with_kv_cache_*_sm90`) | | | | | | | | | | | | | byte-identical to `5cc416fd` (2 kernels x 4 mask objects) |

`ptxas -v` (replayed nvcc commands, `ptxas_{-1,0,1,2}.log`): **no C7507** in any
module; 128 registers at launch for the 16-warp modules (512 threads), 168 for
the 12-warp a16; `USETMAXREG` pairs as designed.  `LD.E/ST.E` 0 in every
producer region.  fp8 protocol counts: `LDG.E` 33 = 18 ([23]) + 15 cold-path
global-scale reloads (one per block site); `UTMALDG` 2 sites; `WARPSYNC.ALL` 4
plus **`BRA.DIV` 22**: ptxas guards every `__syncwarp()` it cannot prove
converged with `UMOV UR, -1; BRA.DIV UR, <out-of-line WARPSYNC>` - +2
instructions per block site over the one `WARPSYNC` budgeted;
**`BSSY`/`BSYNC` 24** (2 at [23]): the `pending` conditionals of the finish
step and the FULL / partial copy arms, **none around the vote** - the vote is
`FSETP.GEU.FTZ |v|, 255.5 x 2^120; VOTE.ALL P0, !P0; @P0 BRA hot` (3
instructions, as budgeted); the cold body (`LDG`, 8 `HMUL2` by `2^120`, `FMUL`,
`F2FP.PACK`, `BRA`) is the not-taken fall-through and the hot path takes one
branch - the mirror of the design's wording, same cost.  Per FP8 block, hot
path, read from the SASS: `LDS.64 x2, LDS.32, PRMT, F2FP.E4M3, HADD2.F32, FMUL,
FSETP, 8 x (PRMT, IMAD.SHL, LOP3), VOTE.ALL, BRA, F2FP.PACK, UMOV, 8 HMUL2,
BRA.DIV, 2 STS.128` plus two `IMAD` stage-address folds (`STS.128` forms 26
`[R+imm]` / 10 `[R]`, [23]: 56 / 10 over 60).

**Correctness.**  `run_fa3_mixed_page_transport.py`: **88 / 88 bit-exact** (64
matrix + 2 many-items + 6 parity-tail + 4 parity-tail-extremes + 12 extremes)
on objects rebuilt after the run's start (`.o` mtimes checked).  Two findings
were fixed on the way (design doc, "as written" F24a / F24c): the f32 scale
product was `FMUL.FTZ` under `-use_fast_math` and flushed `f32(s) g = 1.1 x
2^-127` to 0 (`[extremes-fp8-64-g3.31e-36]`, V side, 400 outputs one ulp off;
`mul.rn.f32` without `.ftz` now); the dynamic module's pair loop had 10
`LDL/STL` per pair (four hoisted landing half addresses per operand spilled +
a 16-bit `"h"` asm operand through the frame), now 4 (the two landing bases
per operand; `STACK` 40 -> 32).

**Bench** (`bench_fa3_mixed_page_transport.py --q-lens 1 64 --repeats 1
--trials 5`, nkcut2 lock, us, min / median / max; `bench_final.txt`;
F24a-only build = commit d6539806 in checkout `dash-flashinfer-claude-wtF24a`,
`f24a_bench.txt`):

| mode (us) | [23] fa13ad89, q=1 / q=64 (medians) | F24a only (d6539806), q=1 ; q=64 | **F24a+b+c (35706f8a)** q=1 min / med / max | q=64 min / med / max | design row (section 5) |
|---|---|---|---|---|---|
| stock_a16 | 299.8 / 309.4 | 299.1 / 299.7 / 300.3 ; 308.7 / 309.1 / 309.6 | 299.5 / 299.8 / 301.3 | 308.6 / 310.5 / 311.6 | 297-303 / 306-312: control holds, no session offset |
| transport_a16 | 283.4 / 286.9 | - | 282.0 / 282.5 / 285.3 | 286.7 / 288.4 / 289.0 | 281-290 / 284-292: holds (module byte-identical) |
| fp8 | 474.0 / 483.0 | **493.9 / 494.6 / 495.3 ; 510.7 / 511.5 / 513.7** (slower than [23]) | **459.7 / 460.2 / 460.7** | **472.2 / 475.6 / 477.6** | accept <= 330 (band 290-345): **reject**, and outside the 331-360 re-derive band |
| fp4 | 507.2 / 517.5 | - | 491.6 / 495.7 / 504.9 | 509.3 / 512.4 / 518.3 | accept <= 330 (band 295-350): **reject** |
| mixed | 880.1 / 906.9 | - | 715.9 / 718.1 / 718.5 | 726.7 / 728.2 / 733.2 | recorded against gate 2E (-18 %) |

Clocks 1980 MHz (nvidia-smi during the runs); the co-tenant `VLLM::EngineCore`
resident; single-kernel bursts (`--repeats 1`), spreads <= 2.5 % except fp4
q=1 (2.7 %).

**Trace** (`MIXED_FA3_TRACE`, q = 1, us per pair, CTA 0 items; `trace_q1.txt`):

| mode (trace build, q=1, 44 pairs per item, w0 / w1 = thread 0 of producer WG0 / WG1) | us per pair | acq | bar | gat | iss | fin = wait / expK / expV / fence+commits | gap (trace overhead) |
|---|---|---|---|---|---|---|---|
| transport_a16 (12 warps) | 2.02-2.04 | **1.00-1.05** | 0.04-0.05 | 0.05 | 0.73-0.75 | 0 | 0.14-0.20 |
| fp8 | 2.59-2.89 | **0.10** | 0.14-0.19 | 0.12-0.18 | **0.62-0.69** | **1.00-1.16** = 0.03 / 0.43-0.59 / 0.40-0.44 / 0.06 | 0.54-0.67 |
| fp4 | 2.75-2.94 | 0.10 | 0.04 | 0.17-0.22 | 0.69-0.70 | 1.27-1.32 = 0.03 / 0.62-0.66 / 0.51-0.53 / 0.05 | 0.47-0.57 |
| mixed (dynamic) | 4.59-5.17 | 0.35-0.39 | 0.21-0.24 | 0.16-0.22 | **1.71-2.15** | 1.19-1.27 = 0.03 / 0.56-0.63 / 0.49-0.54 / 0.05 | 0.93-1.09 |

[23] for comparison: fp8 3.3 = acq 0.2 + iss 0.85 + fin 1.2 (expK 0.57-0.63,
expV 0.48-0.53) + gap 0.7; fp4 fin 1.27-1.36; mixed 5.6 (iss 2.0-2.2, fin
1.85); transport_a16 2.03 (acq 1.05).  Predicted for F24b: iss <= 0.45, fin <=
0.6, acq >= 0.25.  Measured: each producer thread now issues half the copies
and expands half the blocks, but `iss` fell only 0.85 -> 0.66 and `fin` only
1.2 -> 1.0-1.16 - the per-warp rate roughly halved as the warps doubled - and
`acq` stayed ~0.1: **the producer still paces**.  The 16-warp module's
chunk-table segments (`bar` + `gat` 0.26-0.37 us) are 3x the 12-warp module's
(0.10): the 256-thread named barrier with the gather by threads < 128.

**ncu** (fp8, q = 1, `--repeats 1`, third launch; `ncu_fp8*`):

| metric (fp8, q=1, third launch, `--clock-control none`, 1980 MHz) | [23] (`ncu_fp8e`) | F24 (`ncu_fp8`) | design (section 5) |
|---|---|---|---|
| `gpu__time_duration` | 468.0 us | 456.9 us | |
| producer-region `inst_executed` | 81.8 M = **3417 / pair** (854 / warp) | 106.1 M = **4432 / pair** (554 / warp) | ~3600 / pair, <= 500 / warp |
| `smsp__issue_active` | 44.6 % | 52.9 % | 70-80 % |
| warps active per SMSP | 3.0 | 4.0 | |
| producer stall mix (pc sampling, region) | wait 27.5, selected 22.8, dispatch 14.9, not_selected 9.9, short_sb 8.9, long_sb 8.8, no_inst 3.4, math 3.1 | wait 25.5, **dispatch 18.6**, **not_selected 15.6**, selected 15.2, **math 8.1**, long_sb 6.3, branch_resolving 4.0, short_sb 3.8 | dispatch down, selected >= 18 |
| dispatch stalls per issue-active | 0.876 | 1.252 | |
| LSU shared wavefronts, total / per pair | 61.5 M / 2570 | 47.3 M / **1974** | <= ~2050 (C11): met |
| by class: op_gmma / op_st / op_ld / ldgsts (remainder) | 23.95 M / 19.28 M / 8.70 M / 9.6 M | 23.95 M / 10.00 M / 10.89 M / 2.4 M | ~1000 / <= 450 / ~365 / <= 250 per pair -> 1001 / 418 / 455 / 101 |
| `STS.128` wavefronts per instruction (bank conflicts st) | 8.0 (9.76 M) | **4.41** (0.58 M) | 4.0 |
| `LDGSTS` bank conflicts | 8.84 M | 0.29 M | scale copies <= 4 wf |
| `LDS` instructions | 3.58 M | 7.15 M (two half loads per block) | |
| LSU pipe % of peak | 55.7 | 43.8 | |
| ALU / FMA / XU pipe % of peak (SM) | 30.9 / 20.3 / 20.8 | 33.5 / 23.9 / 21.4 | ALU <= ~70 % of producer share |
| tensor pipe active | 39.9 % | 41.1 % | within 3 % of transport_a16 (not measured this round) |

Top producer PCs by stall samples (region): the copy-issue address chain
(`IADD3.X`, `VIADD R, UR`, `IMAD.WIDE.U32`, `ISETP.NE`) with `dispatch` /
`math` stalls, the chunk-table `valid` extraction (`LOP3 R, R19, 0xff0000`,
`short_sb` behind the LDS.128), and the acquire / commit waits (`@!P0 BRA`
loops, `SYNCS.ARRIVE`, `long_sb`).  The expansion body itself is not among the
top 28.

**Verdict against the design's accept / reject rows (section 5):**

- **fp8 / fp4: reject.**  460 / 476 and 496 / 512 us against <= 330; the
  trace's `acq` is ~0.1 us (< 0.25) - the producer still paces - so by the
  design's own row the finding is "per-warp IPC did not hold; re-attribute
  (pipe classes, ALU) before any further step".  The re-attribution from this
  round's counters: (i) the executed count per pair rose 30 % (4432 vs 3417;
  the model allowed +5 %): the per-warp protocol (chunk-table row reads and
  `valid` decode, pipeline acquire / commit, pending bookkeeping, the 256-thread
  barrier + gather, item loop) is duplicated across 8 warps, and the 16-warp
  module's `bar` + `gat` segments are 3x the 12-warp module's; (ii) the
  producer's per-warp IPC fell from 0.13 to 0.10 (554 warp-instr in ~2.7 us):
  `not_selected` 9.9 -> 15.6 %, `dispatch` 14.9 -> 18.6 %, `math` 3.1 -> 8.1 %
  - issue-slot sharing among four warps per SMSP and ALU / FMA dispatch
  contention, not F2FP (XU 21 %) and not the smem pipe (C11 met, LSU 43.8 %);
  (iii) `iss` barely moved (0.85 -> 0.66 with half the copies): the copy
  phase is a latency chain (table LDS -> valid -> predicates -> IMAD.WIDE ->
  LDGSTS -> arrive), not an issue-count problem.
- **Gate 6.3 fails retroactively.**  F24a alone (decode floor, 12 warps) is
  slower than [23]: fp8 494.6 / 511.5 vs 474.0 / 483.0 in the same session
  (stock control equal).  The [23] SASS had no `WARPSYNC` / `BRA.DIV` (ptxas
  proved the `__syncwarp` converged); the vote's data-dependent branch makes it
  unprovable and every block site carries `UMOV + BRA.DIV` plus the taken
  branch over the cold body.  `branch_resolving` appears in the stall mix
  (4.0 %, was 0.5 %).  The decode floor's -5 instructions per block did not pay
  for the guard; the design's step order (F24a's trace and ncu before F24b)
  would have stopped here.
- **Gate 6.4 (vote structure): passes** - no `BSSY/BSYNC` around the votes, the
  guard is 3 instructions; the cold body is the fall-through and the hot path
  takes the branch (mirror layout, same cost).
- **Gate 6.1 (registers): passes** - no C7507, `STACK 0` for fp8 / fp4 / a16
  at 72 / 184 (and 72 / 216), two `USETMAXREG` per kernel; the dynamic module
  keeps a 32 B frame with 4 `LDL` per pair (2 per operand: the landing bases),
  not the 0 the F24c gate asked for.
- **Gate 6.5 (C11): passes** - 1974 wavefronts per pair, `STS.128` 4.41.
- **Gate 6.2 (16-warp consumer control, `SKIP_EXPAND`): not run** - with `acq`
  ~0.1 the consumer is not the pacing side, so the control decides nothing
  yet; it is the first measurement once the producer is below the cadence.
- **mixed: 718 / 728 (from 880 / 907).**  The dynamic module's `iss` 2.0-2.2 ->
  1.7-2.1 and `fin` 1.85 -> 1.2 per pair; the F24c pc-sampling gate (2E) was
  not run in isolation (the F24b-only dynamic build was not built); recorded,
  not accepted.
- **Correctness: 88 / 88 bit-exact**, a16 module and stock paged kernel
  byte-identical to `5cc416fd`, and C9 now holds down to bf16 subnormal scale
  factors (FTZ fix).
- **What the numbers say to do next (not done here, per the rules):** the
  lever is the per-warp protocol and the copy-issue latency chain, not the
  decode: (a) one chunk-table row read and one `valid` decode per warp per
  pair shared by the 8 blocks (the `LOP3 0xff0000 / short_sb` PC), (b) the
  copy addresses as one `IMAD.WIDE` per page from a per-item 64-bit base
  instead of the `IADD3.X / VIADD` chains, (c) the vote hoisted per operand
  (one `VOTE` per pair per operand, block bodies branch-free so the
  `__syncwarp` is provably converged again and the 22 `BRA.DIV` guards
  disappear), (d) the a16-style 12-warp layout re-measured with (a)-(c) before
  deciding whether the second warp group pays for its protocol duplication.
