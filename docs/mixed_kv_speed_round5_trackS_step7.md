# Track S step 7 — [45] register-only dependency cuts for the sm90 SPEC_DEC q=4 build (design rev 2 + implementation; no build or GPU run in this worktree)

Tree `claude/mixed-kv-sm90-tma` @ `659eacfa`, worktree `wt/S7`.  Design only: no build, no GPU
timing.  Every number below is either arithmetic on the source at this tip or a reading of the
step-6 artefacts on nkcut2 (`/tmp/mixedkv-wtS6-a0/`: `v2-fmt{-1,0,1,2}.nvdis` with lineinfo,
`v2-{fp8,fp4,mixed}.source.csv` and `new-transport_a16.source.csv` = ncu `SourceCounters` PC
sampling of one launch of the [44] tree; `/tmp/mixedkv-wtE-s5-v1-sass` for step 5).  The SPEC_DEC
path of `csrc/xqa/mha.cu` and the [44] helpers in `csrc/xqa/mhaUtils.cuh` are unchanged between
the wtS6v2 tree those artefacts were taken from and this tip (the commits in between are [8] on
`mha_sm90.cu` and F26 on the FA3 headers), so the attribution applies to the tip.  Scope excludes
`csrc/xqa/mha_sm90.cu` and the FA3 headers.

## 0. State and the question

nkcut2 H200, locked medians, q=4, B=17, S=4096, 8 KV heads, GQA 4, D=128, bf16
(backends.md "Merged-tree confirmation @ 67a6b4aa"): transport_a16 83.4, fp8 113.5, fp4 101.5,
mixed 107.8 us.  Targets (targets.md): fp8 <= 94 (0.83x), fp4 <= 59 (0.58x; step 6 section 5.3:
below the A16-materialising mma.sync floor), mixed <= 101 (0.94x).

Build: `mha.cu:147-169` (sm90, enum 5, SPEC_DEC, `M_TILESIZE 16`: 128 B K parts, 32-row V tiles,
`MIXED_COMPACT_TILE_LOOPS`), `:216-226` ([44] guards), `:230-233` (`nbKBuffers 2`, `nbVBuffers 2`,
`nbXBuffers 1`), `:3680-3700` (`nbCtaPerSM` from the smem sum, `__launch_bounds__(256, 2)`),
`SharedMem` `:411-543` = 115,456 B of the 115,712 B cap.  Step-6 ncu ([44] tree): REG 124-127,
STACK 0, 2 CTAs/SM, `smsp__inst_executed.sum` a16 29.9 M / fp8 36.0 M / fp4 36.5 M / mixed 42.4 M,
warp-cycles per issued instruction 6.66 / 8.13 / 7.30 / 6.56, issue-active 51 / 43 / 48 / 52 %.

Step 6's closing reading: "fp8's remaining time is landing latency (long scoreboard on the K/V
ring and the page/tag LDGs); the next lever is pipeline depth (kAhead / a third K buffer), which
the 2-CTA smem budget forbids at 128 B parts".  The question for this step: does any smem layout
buy that depth at 2 CTAs/SM, and if not, what does?

Answer in one line: **no layout buys a third 128 B K buffer at 2 CTAs/SM, and none is needed —
the step-6 PC samples put < 0.1 % of the time on the `cp.async.wait_group` (`DEPBAR`) of either
ring.  The long-scoreboard time is (i) the per-CTA pipeline fill (the gemm1 warps' `xBar.produced`
wait, 9.6 % of all fp8 samples, is quantitatively one gemm0 tile-time per gemm1 warp per CTA
start — rev 2, section 2: the two roles run in lock-step with ~0 steady-state slack, so the wall
follows max(gemm0 tile, gemm1 tile) and every cut is priced as min(gemm0 cut, gemm1 cut)), (ii)
`R2UR` at the page-index load sites (ptxas parks the warp-uniform page index in a uniform
register as soon as it is loaded, exposing the L2 round trip there) and register WAR behind the
`LDGSTS` batch (MIO-queue drain, not recoverable), and (iii) the barrier poll's `NANOSLEEP`
(2.3 %, part of the fill).  The lever is the per-warp dependency depth of both roles' tile bodies
— register-only items under the existing [44] guards, no SharedMem change.**

**Revision 2 (judge blockers, before code).**  Rev 2 is the design of record: the build / accept
gate is the gap-free table of section 5 (fp8 99-105 predicted, the <= 94 target NOT reached by
this step; fp4 88-93; mixed 92-97; a16 80-82), and A0 (the `XQA_NB_SUB_SEQ=1` ncu run on the
pristine tip) is a precondition of A4 timing.  Any brief that still quotes rev 1's fp8 91-99 /
"target <= 94 marginal" is superseded by this section.  Five blockers were raised against rev 1
and are answered in place: (1) [45b] in the dynamic module would have paired spans of different formats
inside one format branch — [45b] is now static-modules-only (section 3.2); (2) the "gemm0 paces,
gemm1 has 20 % slack" reading was wrong — the produced-wait samples are the pipeline fill, two
barrier sites were mislabelled, and the wall is now budgeted as min(gemm0 cut, gemm1 cut) with a
discriminating measurement specified (section 2); (3) the 6.3 % / 7.2 % "page-index / tag LDG
consumer" bucket is `R2UR`-at-load-site plus address-register WAR, which a deeper prefetch does
not touch — [45d] is redesigned as lane-distributed page indices that never enter the uniform
datapath, and the WAR part is priced at zero (sections 2, 3.4); (4) the budget is re-derived per
item with a mechanism-specific realisation factor (section 3.7): the defensible cut is -9..-12 %
on gemm0 and -7.5..-11 % on gemm1, fp8 lands at 99-105 us, the <= 94 target is not reached by
this step; (5) the accept / reject table is gap-free and the produced-wait rule is replaced
(section 5 A2).  Notes taken: the (a') column of the smem table sums to 117,000 (not 117,008;
conclusion unchanged); [45e] is budgeted on the 111 `MATCH.ANY`-group samples (the 33 at
`0x11410` are an unrelated `IMAD.SHL`); [45f] is a CTA-lifetime saving with a ~1-2 % wall share;
the dynamic module's per-span fold predicate is a *select* on the register tag, never a multiply
by a possibly stale smem scale word; the [45c] flag / tag rotation is two named registers, not a
`bool[2]` indexed by the `CircIdx`; the loader lambdas become `#if MIXED_HOISTED_COPY <new> #else
<stock verbatim> #endif` so the sm120 and sm90 q=1 preprocessed source is unchanged.

## 1. Smem arithmetic first: the four depth options

Cap for two CTAs per SM: `2 x (sizeof(SharedMem) + 1,024) <= 228 KB = 233,472 B`, i.e.
`sizeof(SharedMem) <= 115,712 B` (`mha.cu:3684`; `alignas(128)` `:411`).  One CTA per SM:
`kMAX_SMEM_SIZE` 227 KB = 232,448 B (`utils.cuh:55-56`, `mha.cu:543`).  Member sizes from the
declarations (`mha.cu:434-509`; K/V buffer = rows x parts-per-row x 16 B grains; the step-5 table
reproduced ncu's 115,456 with this accounting):

| member (sm90, enum 5, SPEC_DEC, M16, `!grpLoadV`) | today: K2 x 128 B / V2 | (a) K3 x 64 B / V2 | (b) K2 x 128 B / V3 | (c) K4 x 128 B / V4, 1 CTA/SM | (a') K3 x 64 B / V3 |
|---|---|---|---|---|---|
| `q[1][1]` 16 x 256 B | 4,096 | 4,096 | 4,096 | 4,096 | 4,096 |
| `k[4][nbK]` 64 rows x part bytes | 65,536 | **49,152** | 65,536 | **131,072** | 49,152 |
| `x[1][4]` 16 x 128 B | 8,192 | 8,192 | 8,192 | 8,192 | 8,192 |
| `v[2][2][nbV]` 32 rows x 128 B | 32,768 | 32,768 | **49,152** | **65,536** | 49,152 |
| `kFormats[4][nbK]` (4 B) + `vFormats[2][2][nbV]` (2 B) | 32 + 16 | 48 + 16 | 32 + 24 | 64 + 32 | 48 + 24 |
| `kNeedsExpansion[4][nbK]` + `vNeedsExpansion[2][2][nbV]` | 8 + 8 | 12 + 8 | 8 + 12 | 16 + 16 | 12 + 12 |
| `kScales[4][nbK][65][4]` | 2,080 | 3,120 | 2,080 | 4,160 | 3,120 |
| `vScales[2][2][nbV][33][4]` | 1,056 | 1,056 | 1,584 | 2,112 | 1,584 |
| `warpRowMax` / `warpRowSum` / `ctaRowMax` `[1][4]` x 128 B | 1,536 | 1,536 | 1,536 | 1,536 | 1,536 |
| `qBarrier[1]` 8 B + `xBarriers[1][4]` 4 x 16 B | 72 | 72 | 72 | 72 | 72 |
| sum -> `alignas(128)` | 115,400 -> **115,456** | 100,076 -> **100,096** | 132,324 -> **132,352** | 216,904 -> **216,960** | 117,000 -> **117,120** |
| 2 CTAs/SM (<= 115,712) | yes (256 B spare) | **yes (15,616 spare)** | **no (+16,640)** | no (1 CTA: 216,960 <= 232,448 yes) | **no (+1,408)** |

(d) M tile: `warpTile.y = roundUp(nbValidRows, 16)` (`mha.cu:104-111`) is 16 for q x GQA = 16 and
is what puts Q at 4 KB, X at 8 KB and the row-max arrays at 3 x 512 B.  At M 32 the same K/V rings
give 116,480 + 4,096 + 8,192 = 128,768 B (step 4's "K 128 B / V 32 rows at M 32" column, 1 CTA/SM);
the M16 build is the only one that fits two CTAs with 128 B parts, and its compressible members are
`SMemWarpRowMax` (32 rows held for a 16-row tile: 768 B), the four metadata arrays (80 B — which
this step stops reading anyway) and the scale dump rows (48 B): 896 B against the 32,768 B a third
128 B K buffer needs or the 16,924 B a third V buffer needs.  There is no (d) lever.

**Landing cover in tile-times (measured on the [44] fp8 module).**  Per 64-token gemm0 warp-tile:
2,067 executed instructions (part loop 1,451 = 725 per part; `smsp__inst_executed` of the tile
loop / 8,704 warp-tiles, section 2) at 8.13 warp-cycles per issued instruction = **~16,700 cycles
(8.5 us at 1.98 GHz); one K part body ~6,070 cycles (3.1 us)**.  Per gemm1 V iteration
(4 per tile): (4,744 - 953 idle) samples x 4.31 cycles per sample / 4 = **~4,100 cycles (2.1 us)**.
Cross-check: 3.2 CTA-tiles per CTA (2,176 tiles / 680 CTAs) x 8.5 us = 27 us per CTA lifetime; ncu:
138.2 us x 57.7 % active / (5.15 CTAs per SM / 2 resident) = 31 us.  With two buffers the copy of
part p+1 is issued at the top of iteration p (`mha.cu:2611-2613`) and waited at the top of p+1
(`:2619`, `waitGroup<1>`), so the cover is one part body = **0.36 tile-times = 3.1 us** for K and
one V iteration = **0.25 tile-times = 2.1 us** for V (`:3223-3224`, `:3135`).  The LDGSTS landing
latency under load is 1-2 us (the [8] trace: `kl_iss -> kc_ready` 1.2-2.0 us steady state incl.
conversion; fill 3-4 us) — the cover is already 1.5-3x the latency, which is what the samples say:

| `DEPBAR.LE SB0, 0x1` site ([44] fp8 module) | samples on the DEPBAR | samples on the next 8 instructions | of which long_sb |
|---|---|---|---|
| gemm1 V wait (`0x03280`, `mha.cu:3135`) | 5 | 178 (ISETP on the `vNeedsExpansion` LDS.U8: 125 short_sb) | 0 |
| gemm0 K wait (`0x0b950`, `mha.cu:2619`) | 1 | 160 (ISETP on the `kNeedsExpansion` LDS.U8: 114 short_sb) | 0 |
| prologue (`0x079f0`) | 0 | 1 | 1 |

(fp4 and mixed: 3 / 2 and 4 / 2 samples on the DEPBARs, 0 long_sb after them; a16 identical shape.)
The whole `ldgsts.cuh` bucket is 36 / 29 / 106 samples (0.4 / 0.3 / 1.1 %) on fp8 / fp4 / mixed.

**Predicted stall change per option, from the step-6 numbers** (whole-kernel stall shares from
the same CSVs: fp8 short_sb 23.5 %, long_sb 22.4, selected 13.3, wait 12.6, not_selected 8.7,
no_inst 5.2; fp4 21.0 / 19.8 / 14.6 / 14.5 / 9.2 / 7.1; mixed wait 24.0, long_sb 20.0, selected
16.8, short_sb 9.9, not_selected 9.9, no_inst 7.7; a16 long_sb 37.7, selected 16.7, wait 14.7):

- **(a) K3 x 64 B / V2 (100,096 B, fits).**  Cover 2 x (64 B body ~0.55 x 6,070) = 6,700 cycles:
  +10 % cover on a wait that carries < 0.1 % of the samples -> long_sb share 22.4 -> 22.3 %.  Cost:
  the 64 B part runs four rounds per K tile (step 4 -> 5 measured the four-round form as the slower
  one: +150 U of round fixed cost per extra round, +300 U = +7 % instructions), and the [44] cut
  `static_assert`s `partBytes == 128` / 64 blocks per span (`mhaUtils.cuh:1053-1061`) — the
  expansion and the hoisted copy (`:1226`) would both be re-templated for one block per lane.
  **Rejected: +3..+7 % time for a stall that is not there.**
- **(b) K2 / V3 (132,352 B).**  Does not fit; the V-side DEPBAR carries 5 samples.  **Rejected.**
- **(c) 1 CTA/SM, K4 / V4 (216,960 B, fits one CTA).**  Two warps per SMSP instead of ~3.4 active:
  even with all long_sb (1.81) and no_inst (0.43) removed, warp-cycles per issued instruction
  8.13 -> 5.9 gives IPC 2 / 5.9 = 0.34 per SMSP against today's 3.4 / 8.13 = 0.42: **>= +24 %
  time** before any real loss; step 4 measured the 1 -> 2 CTA/SM move at 0.74-0.75x with equal
  code (backends.md "Track S step 4", `n=1` row vs `n=5`).  **Rejected.**
- **(d)** no lever (above).

Conclusion of the arithmetic: pipeline depth is neither available nor the bottleneck; the record's
"landing latency" is two other things, decomposed next.
## 2. What the step-6 samples say (fp8 [44] module, 9,960 samples; fp4 / mixed in parentheses) — rev 2

Regions are the SASS loops delimited by back-edges (`v2-fmt1.nvdis`; the gemm1 code is laid out
first).  Per warp-tile = region `Instructions Executed` / 8,704.  Both roles are compared by
*samples* (rev 1 mixed an instruction-count estimate for gemm0 with samples for gemm1; by
instructions the gemm0 tile would be ~16,700 cycles, by samples ~14,000 — the ratio below is the
one that matters).

| region | SASS | samples | inst per warp-tile | stall mix |
|---|---|---|---|---|
| gemm1 V-iteration loop `[0x2390, 0x50f0]` (`mha.cu:3173-3345`) | 726 | 4,744 (47.6 %), of which **953 on the `xBar.produced` TRYWAIT branch `0x2e40` (`mha.cu:3229`)**; busy 3,791 | **1,959** (4 iterations) | long_sb 26 %, short_sb 25, selected 15, wait 12 |
| gemm0 tile loop `[0xa790, 0x11740]` (`:2570-2790`) | 1,787 | **3,873** (38.9 %) | **2,067** | short_sb 27-29 %, long_sb 11-16, wait 14, selected 13-14, not_selected 10-11, no_inst 6 |
| .. gemm0 part loop `[0xa990, 0xef30]` (`:2594-2696`) | 1,114 | 2,817 (28.3 %) | 1,451 (2 parts) | short_sb 33, selected 13, wait 12, long_sb 11, not_selected 11, no_inst 6, mio 5 |
| .. gemm0 outer (softmax, X hand-off) | ~670 | ~700 | ~600 | FMNMX <- SHFL chains, `MATCH.ANY` group (below), MUFU |
| out-of-line barrier poll (`NANOSLEEP` `0x11770`, `barriers.cuh:352-378`) | | 228 (2.3 %) | | long_sb (the gemm1 `produced` poll's sleep) |

**Pacing (rev 2): lock-step, no steady-state slack.**  gemm0 3,873 samples per launch vs gemm1
busy 3,791: the two tile bodies are equal within 2 %.  Through a single X buffer (`nbXBuffers 1`)
that is what lock-step looks like — either role can be at most a few per cent shorter than the
other before it waits.  The 953 + 228 = 1,181 `xBar.produced` samples are the **per-CTA fill**:
each of the 4 gemm1 warps waits one gemm0 tile-time at CTA start (its first `produced` cannot
arrive before gemm0's first tile), while the gemm0 warps `EXIT` early after their last tile
(`0x11750`).  Expected fill = 4 warps x 680 CTAs x (gemm0 samples per warp-tile 3,873 / (3.2 x 4
x 680)) = 3,873 / 3.2 = **1,210 samples**, i.e. the whole wait (fp4 862 + its sleep, mixed 873
+ sleep: same arithmetic, same conclusion).  Rev 1 read this as 20 % gemm1 slack; it is 0-2 %.
Two barrier sites were also mislabelled in rev 1: `0xf190` (fp8, 73 long_sb) / `0xfb10` (mixed,
178 = 1.8 %) are `SYNCS.PHASECHK.TRANS64.TRYWAIT [+0x1c290]` = `getAndFlip` (`mhaUtils.cuh:1677`)
inside `xBar.consumed.wait_parity` (`mha.cu:2777`) — **gemm0 waiting for gemm1**, i.e. gemm1
already paces part of the time (most on mixed); `0xa600` waits on `[+0x1c280]` before the tile
loop = the `qBarrier`, not `xBar.consumed`.

Consequences for the design: (i) the wall follows max(gemm0 tile, gemm1 tile), so every item is
priced on both roles and the achieved cut is **min(gemm0 cut, gemm1 cut)**; (ii) the gemm1
(V-side) items are load-bearing from the first commit, not a reserve; (iii) the fill (1 gemm0
tile-time per CTA ~ 1,181 samples ~ 8.5 us of a 31 us CTA lifetime, overlapped with the
co-resident CTA) is [42]'s domain (`nbSubSeqPerSeq`, `kFixedCostInTiles`), not this step's.
**Discriminating measurement (A0 of this step, one ncu `SourceCounters` run of the pristine tip at
`XQA_NB_SUB_SEQ=1`, 136 CTAs x 16 tiles instead of 680 x 3.2):** if the `0x2e40` + `NANOSLEEP`
share drops ~5x (one CTA start per 16 tiles), the wait is fill and this section stands; if it
stays ~12 %, it is steady-state slack and rev 1's reading was right — either way the number is
known before the first timing run.  In the same run the `R2UR` sites (next paragraph) should
keep their share: they are per-load, not per-CTA.

**Where the gemm0 tile goes (samples -> cycles at 4.31 cycles per sample), rev 2 attribution:**

| item (gemm0, per warp-tile) | evidence (address, samples, top reason) | cycles per tile |
|---|---|---|
| expansion scale chain: `LDS.U16 -> F2FP.F16.E4M3 -> HADD2 -> FMUL -> FMNMX -> FSETP -> VOTE.ALL -> BRA`, once **per span** (4 per K part) | `F2FP.E4M3` `0xbaf0/0xc640/0xd100/0xdbc0` 72+72+62+31 short_sb; `LDS.U16` `0xc620/0xd0e0` 71+48; fold `BRA` `0xc6e0/0xd1a0/0xdc60/0xbc20` 50+48+41+27 (`mhaUtils.cuh:1107-1144`) | ~2,300 measured; of which ~440 is the serialisation of four ~110-cycle chains (the rest is XU / issue at 4 warps per SMSP) |
| next span's payload `LDS.128` WAR on the previous span's `STS.128` registers | `0xdbb0` 51 short_sb (`mhaUtils.cuh:1111`) | ~220 (MIO-queue: the STS operands are read late) |
| `kNeedsExpansion` flag: `LDS.U8 -> ISETP -> BRA` right after the DEPBAR | `0xb970` 114 short_sb (`mha.cu:2629`) | ~490 (MIO-queue congestion behind the just-issued LDGSTS batch: mio stalls at `0xb920-0xb940`; the queue wait moves to the next MIO op if only this LDS is removed) |
| page tags: `LDG.E.U8` (lane) -> 4 x `SHFL.IDX` -> `PRMT` x3 -> `STS.U8` flag + `STS` formats | `0xaa60` 90 short_sb (`mha.cu:2409`, `:2454-2456`, `mhaUtils.cuh:219-227`) | ~390 (dead work in the static modules) |
| **(a) `R2UR` at the page-index LOAD site** (`mhaUtils.cuh:1519` `getPage`): `@!P6 LDG.E R40,[R40]` `0xb650` -> `@!P6 R2UR UR39, R40` `0xb710` 12 instructions later | `0xb710` 75, `0xb7d0` 32, `0xb870` 40, `0x97a0` 26 long_sb = **173** (mixed `0x3ff0` `@!P1 R2UR UR29, R13` 269 + 43 = 312, 3.1 %) | ~750 (mixed ~1,350).  ptxas keeps the warp-uniform page index in a uniform register (register pressure at 124-127) and moves it there as soon as it is loaded: the full L2 round trip is exposed at the load, whatever the distance to the copy that uses it.  A deeper rotation gives ptxas more values to `R2UR`, not fewer stalls. |
| **(b) WAR on address registers of in-flight `LDGSTS` / `LDG`**: `IADD3 R8` `0x2cc0` rewrites the LDGSTS `0x2be0` address; `LDC.64 R32` `0xb200` the LDGSTS `0xb110` address; `IMAD.WIDE R14` `0xa370`; `0x2e30` rewrites R8 after `LDG.E.U8 [R8.64]` `0x2e00`; `0x2d00` R44 (LDGSTS `0x2c60`); `0xb700` R41 (LDG `0xb650` high half) | 66 + 55 + 48 + 84 + 41 + 41 = **335** long_sb | ~1,440 — **not recoverable by this step**: a store-class instruction holds its source registers until the MIO has read them, so this is the LDGSTS batch draining; keeping the address registers live longer only moves the wait onto the next LDGSTS issue (`lg_throttle`).  Priced at 0. |
| (c) `0xf190` 73 = `xBar.consumed` wait (above), rev 1 booked it as a scale-loop `BRA` | | gemm1-pacing time, not a gemm0 item |
| `computeRowSum` quad broadcast with a lane-dependent mask -> `WARPSYNC / MATCH.ANY / REDUX / VOTEU / BRA.DIV` | `0x11300-0x11360` = 111 samples (`0x11310` 66 branch_resolving, `0x11350` 30, `0x11360` 15) (`mha.cu:802`); `0x11410` `IMAD.SHL.U32 R0, R2, 0x8` 33 is unrelated | ~480 |
| `HMMA <- LDSM` first-use waits | `0xee70` 59 short_sb | ~250 |
| loop control / address chains (`wait`: fixed-latency dependent ALU pairs) | 338 samples spread over the part loop | ~1,460 |
| issue (selected) | 366 | ~1,580 (1,451 instructions) |

gemm1 shows the same items at its scale (per V iteration: `PRMT <- SHFL` `0x2510` 151, flag
`ISETP` `0x32c0` 125, `LDS.128` `0x3f90` 127, fold `BRA` 93+80, `F2FP` 91+70, `R2UR` at the V
page load ~150, WAR `0x2cc0/0x2d00/0x2e30` 191) plus its own rescale ballot (`FSETP.NEU` `0x2eb0`
112).  Mixed (dyn) adds the per-span format `LDS.U8 [UR]` from `smem.kFormats` (`mhaUtils.cuh:35`,
228 `wait` samples = 2.3 %) and has the largest `R2UR` share (312 = 3.1 %) plus format / scale
branches 178+137+40+31.

Two facts the record did not have: (1) in the **static fp8 / fp4 / a16 modules the whole tag
pipeline is dead work** — `needsMixedPageExpansion` is a constant (`mhaUtils.cuh:231-236`), the
hoisted copy ignores `formats` (`:1319`, `:1341-1345`), the expansion ignores them (`:1149-1152`)
— yet the tag `LDG.E.U8`, the four shuffles, the `PRMT`s and the lane-0 stores execute per copy
call because the stores are side effects (`mha.cu:2409`, `:2454-2457`, `:2965`, `:2972-2979`).
(2) The per-span fold vote is a control dependency: the `asm volatile` loads of span s+1 cannot be
hoisted above span s's `if (fold)` (`mhaUtils.cuh:1140-1144`), so the four chains of a K part run
in series (~4 x 110 cycles) — step 6's "loads-first" order helped inside a span, not across spans.

## 3. Design [45]: six register-only items, all under the [44] guards (rev 2)

All code changes sit in `#if MIXED_HOISTED_COPY` / `MIXED_BF16_PLACEMENT_EXPANSION` /
`MIXED_COMPACT_TILE_LOOPS` blocks of `mha.cu` (kernel body: preprocessor guards, as [44] item 1
explains — an `if constexpr` there is still instantiated) and in the two [44] helpers of
`mhaUtils.cuh` (never instantiated by sm120, sm90 q=1 or `mla_sm120`), so every other build's
preprocessed source is unchanged.  The loader lambdas (`loadPages` `mha.cu:2360-2392`, the flag /
format stores `:2450-2457`, V `:2851-2867`, `:2966-2979`) are today under `#if
ENABLE_MIXED_KV_CACHE` only; they become `#if MIXED_HOISTED_COPY <new> #else <stock verbatim>
#endif`.  No `SharedMem` member is added or removed (115,456 B stays; the four metadata arrays
become unread in the guarded build but keep their slots so the barrier addresses do not move).
A new derived guard `MIXED_ALL_HOISTED_COPY = MIXED_HOISTED_COPY && MIXED_PAGE_STATIC_FORMAT != 0`
marks the modules whose every K / V copy is the hoisted copy (fp8 / fp4: the stock A16 path is
dead once the expansion flag is the constant true; dyn: `kA16CopyFastPath` is false); the a16
module keeps the stock A16 copy, whose `HeadPtr` wants the page vector.

### 3.1 [45a] One fold vote per call (expansion helper, `mhaUtils.cuh:1090-1162`)

Static modules: load every span's scale word first (K: 4 x `LDS.U16` at `scaleAddr + 64 span`,
V: 2) and span 0's payload (loads-first, as [44]), run the chain once — 4 x `F2FP.F16.E4M3`
(independent, back-to-back on the XU pipe), 4 x `HADD2.F32`, 8 x `FMUL`, `FABS`/`FMNMX` tree,
`FSETP`, one `VOTE.ALL`, one `BRA` — then two whole-call bodies (fold / fallback) with the spans
unrolled and branch-free inside.  `sf2` per span is recomputed from the kept scale word inside the
body (1 `F2FP` + 2 `HADD2` + 2 `FMUL` + 2 `F2FP.PACK`, ~30 cycles, no vote, no branch), so the
live set across the vote is `s01[nbSpans]` (4) + span 0's payload (8) = 12 registers against
today's `s01 + r0 + r1 + f0 + f1` (5) + payload (8) = 13: **-1 register**, +3 instructions per
span.  Dynamic module: the pre-vote pass is unrolled over the 4 spans with each span's `(g 2^k,
foldOk)` **selected** by its tag byte from the register format word ([45c]) — `f_i = compressed ?
|r_i g 2^k| : 0` is a select, never a multiply of a possibly stale smem word (A16 / BAD-page scale
rows are never written by the copy's scale loop, `mhaUtils.cuh:1350-1372`, so a stale NaN would
otherwise force a spurious whole-call fallback); `foldOk` is ANDed over the formats present; one
vote; then the rolled span loop with the per-span format branch and `fold ? body<true> :
body<false>` selected from the call-level flag (warp-uniform, no vote).  The rolled body re-reads
its span's scale word (`LDS.U16`, register-neutral) instead of indexing `s01[]` at run time.
Bit-exactness is unchanged: both bodies give the reference's single rounding ([44] rev 2), and a
call takes the fallback iff any span would have (per-span bounds are still evaluated exactly).
The fold frequency is unchanged on the bench (FP8 scales capped at 128, unit global scales) and
the `tinyglobal` / `maxscale` matrix rows exercise the fallback.  Pre-existing and out of scope
(follow-up): a BAD page inside a compressed static module's tail part has a stale scale word
(payload zero-filled) — bit-identical to today (both bodies give NaN x 0 for that span, as today),
but under [45a] the stale word can also send the *other* spans of the call to the fallback body
(bit-exact, perf-only); a zero-length `cp.async` for BAD-page scale rows would make it
deterministic.

Dependency depth per K part: today 4 x (11-deep chain, ~110 cycles, serialised by the branch) =
~440 cycles of exposed latency + a branch-resolve `wait` per span; after: 1 x ~130 + 4 x ~30 =
~250: **-190 cycles per part, -380 per tile** (V: 2 x 110 -> 130 + 60: -90 per iteration, -360 per
tile).  The remaining ~1,900 measured cycles of the chain items are XU conversion + issue at 4
warps per SMSP and are not claimed.

### 3.2 [45b] Two register sets, block-pipelined spans (static modules only)

With the branch gone, a static span body is straight-line; issue span s+1's block-0 payload `LDS`
before span s's block-0 decode + `STS.128` x2, and span s+1's block-1 `LDS` before span s's
block-1 decode + stores, into the *other* register set (`packed[2][2]`, indexed by the unrolled
span parity — compile-time indices).  The next load then never WARs on registers a still-draining
`STS` reads (the load is issued before those `out` registers exist), and each `LDS` latency is
covered by ~24 instructions of the previous block's decode.  Peak live set: `packed[set][1]` (4) +
`packed[nxt][0]` (4) + `packed[nxt][1]` (4) + `out` (8) + `sf2` (2) = 22 vs today's 18: **+4
registers (FP8), +2 (FP4)** (`ldsB64` fills two words per block).  **Dynamic module: one set** —
one span = one page and adjacent spans may differ in format (`mhaUtils.cuh:1140-1152` branches per
span), so pairing spans inside one format branch would decode an FP4 page with the FP8 body; the
format-agnostic alternative (issue `LDS.128` x2 for every span and let the FP4 decode consume the
low 8 B) was not taken because the dyn module's short_sb is the smallest (0.73) and the [45b]
gain there is < 0.3 %.  Budget: the `LDS.128` short_sb samples (51 gemm0 / 127 gemm1 on fp8) at
an MIO-queue realisation (section 3.7).

### 3.3 [45c] Metadata in registers: flags, formats, tags (`mha.cu` K and V loaders) — rev 2

- **Static modules (`MIXED_PAGE_STATIC_FORMAT >= 0`): no tags at all.**  The decision
  `needsExpansion` is the build constant `kMixedStaticNeedsExpansion = MIXED_PAGE_STATIC_FORMAT
  > 0` (fp8 / fp4: every part is expanded; a16: never).  `mixedPageTagLane`,
  `broadcastMixedPageTags`, the `STS.U8` flag + `STS` formats, the `__syncwarp` and the `LDS.U8 ->
  ISETP -> BRA` after the DEPBAR disappear (the copy and the expansion ignored the tags already:
  section 2 fact 1).  In fp8 / fp4 the stock A16 copy branch is provably dead and is removed by
  the preprocessor (`MIXED_ALL_HOISTED_COPY`); in a16 the hoisted branch is dead and removed, the
  stock A16 copy stays, so the a16 module changes only by losing its tag pipeline.  Dropping the
  loader `__syncwarp()` is safe: it existed only for the lane-0 `STS -> LDS` flag path (the stock
  loaders issue `cp.async` with no `__syncwarp` before it).
- **Dynamic module: one packed tag word per buffer, in registers.**  `pageTagLane` (lane s holds
  the tag of page s, BAD / past-the-tile lanes 0 = `kA16`) is packed once per copy call by
  `packMixedPageTags`: one `redux.sync.or.b32` over `tag << 8 lane` (sm80+; the word is
  warp-uniform by construction and lands in the uniform datapath) instead of 4 `SHFL.IDX` + 3
  `PRMT`.  The word is (i) passed by value to the hoisted copy, whose per-span format is
  `(word >> 8 span) & 0xFF` (uniform ALU; the scale loop's per-lane page uses the same shift with
  a lane-dependent count), (ii) kept in **two named registers** `kTagWordNext` (written by the
  copy of buffer *next*) / `kTagWordCurr` (read by the expansion of buffer *curr*), copied at the
  two `idxCurrSMemKBuf++` sites (prologue and loop) — a `bool[2]` / `u32[2]` indexed by the runtime
  `CircIdx` would go to local memory (the `mixedVExpandParity` comment records the same trap; the
  LDL/STL 0 gate would catch it), (iii) the expansion's `needsExpansion` is `word != 0` (`ISETP` on
  a register) and its per-span format is the same byte extract (today `LDS.U8 [UR]`, 228 `wait`
  samples on mixed).  V: `vTagWordNext` / `vTagWordCurr` at the two `idxCurrSMemVBuf++` sites.
  **Register cost: +2 (two words; `pageFormats` and the flag disappear: net +1).**
- Not done in rev 2: rev 1's "every lane loads the four tags itself" (4 `LDG.E.U8` to the same
  address) — uniform addresses would give ptxas four more values to `R2UR` at the load site
  (blocker 3); the lane-distributed tag load stays and is consumed by one `REDUX`.
- Probe build (`MIXED_KV_PROBE_C`: placement expansion on, hoisted copy off): the expansion takes
  the word from the staged `MixedPageFormats` (`mixedPageTagsWord`), so that build still compiles.

### 3.4 [45d] Lane-distributed page indices (replaces rev 1's "prefetch one stage deeper")

Rev 1 proposed a third index vector and a second tag word.  Section 2 (a) shows the stall is at
the *load* site: `getPage` loads the same address in every lane, ptxas proves the value uniform
and, at 124-127 registers, parks it in a uniform register with an `R2UR` 12 instructions after the
`LDG` — the L2 round trip is exposed there regardless of how far away the copy is.  Rotating more
uniform vectors changes nothing.  The fix is structural: **the page index must never be provably
uniform**.  `getPageLaneUngated<nbLoadedPages>` has lane s (s < nbLoadedPages) load page
`idxPageBeg + s` (one predicated `LDG` per warp, one 16 B sector, `ld.global.nc` in an `asm
volatile` so the compiler cannot fold the [45f] bounds predicate into the `nbPages` select) and
returns one `KVCachePageIndex` per lane (BAD for lanes past the tile and pages past `nbPages`):
**1 register instead of 4 (K) / 2 (V) per rotation stage, no `R2UR`**.  The tag load becomes
`mixedPageTagOfLane(transport, pageLane)` (each lane loads the tag of its own page; no
`selectByIndex`).  Consumers read the value in the copy with `__shfl_sync(~0, pageLane, span)`
— one `SHFL.IDX` per span (K 4, V 2) plus one per scale-loop iteration with a lane-dependent
source (K 2, V 1) — so the value stays in vector registers until its use, and the `SHFL` result
is not uniform-provable either.  Cost: +6 `SHFL` per K copy call (+12 issue slots per tile,
~50 cycles), +3 per V call; today's 4 `SHFL` + 3 `PRMT` of the tag broadcast are gone ([45c]).
Applies to the `MIXED_ALL_HOISTED_COPY` modules (fp8 / fp4 / dyn); the a16 module keeps the
`Vec` form (its stock copy's `HeadPtr` indexes the vector).  The prefetch distance is unchanged
(indices two tiles ahead, tags one tile ahead): the `R2UR` was never a distance problem.
Register accounting (review correction): step 6's `R2UR` evidence says the `Vec` stages WERE in
uniform registers, which do not count against the 128 vector-register cap, so the honest vector
delta is **+2** (`pageLane`, `pageLaneNext`; `pageTagLane` existed already), plus possibly +2 if
ptxas hoists the per-lane 64-bit `kvCachePageList + 4 (maxNbPagesPerSeq idxReq + lane)` address
out of the tile loop, and the copy's per-span `IMAD.WIDE` chain now runs in the vector datapath
(same count, vector registers instead of URs).  A REG 126 -> 128 reading after the [45d] commit
is this item's, not [45a]'s / [45b]'s.  (b) is
not addressed: the address-register WAR is the LDGSTS batch draining; the design prices it at 0
and gates only that the `LDGSTS` count is unchanged.  Budget: the `R2UR` sites (173 samples fp8
gemm0, ~150 gemm1; 312 mixed) at a latency realisation (section 3.7).  A1 gates: `R2UR` fed by
the page-index `LDG` 0 in every module; `LDG` in `loadPages` 1 (index) + 1 (tag, dyn) per call
instead of 4 + 1.

### 3.5 [45e] `computeRowSum` quad broadcast with a constant mask (`mha.cu:798-805`)

`__shfl_sync(0xF << (laneId() / 4 * 4), rowSum[i], 0, 4)` has a lane-dependent mask, so ptxas
emits `WARPSYNC / MATCH.ANY / REDUX.OR / VOTEU / BRA.DIV` convergence code around every shuffle
(`0x11300-0x11360`, 111 samples = 1.1 % of fp8, on gemm0's critical path).  The same broadcast as
`__shfl_sync(~0U, rowSum[i], laneId() & ~3U)` (full mask, quad-base source lane, width 32) is one
`SHFL.IDX`.  Values are identical (a broadcast of lane 0 of the quad either way);
`computeRowSum` runs unconditionally in the converged gemm0 warp (`mha.cu:2777`).  Guarded under
`#if MIXED_COMPACT_TILE_LOOPS` inside the function body so the sm120 and M32 modules keep their
SASS.  **-480 cycles per gemm0 tile modelled.**

### 3.6 [45f] Prologue: page-list load not gated on `cacheSeqLen` (`mha.cu:2280-2305`, `:2385-2392`)

The per-CTA chain before the first K copy is `getCacheSeqLen` (`LDG`, `:2280`) -> `nbPages`
(`:2304`) -> `getPage` (`idxPage < nbPages ? LDG : BAD`) -> tag `LDG` -> copy -> landing: three
dependent round trips plus the landing.  The page-list read does not need `nbPages`:
`kvCachePageList[maxNbPagesPerSeq x idxReq + idxPage]` is in bounds for every `idxPage <
maxNbPagesPerSeq` (`maxNbPagesPerSeq = page_table.shape[-1]`, `flashinfer/xqa.py:430-431`,
`mha.cu:3851`: the row stride `getPage` already indexes with), so load predicated on
`idxPage < maxNbPagesPerSeq` (a kernel parameter) and select BAD on `idxPage < nbPages` after
both loads land (`cacheSeqLen == 0` gives `nbPages 0` and the select reproduces today's values).
The load is an `asm volatile ld.global.nc` so the compiler cannot merge the two predicates back
into one that depends on `nbPages`.  Removes one round trip per CTA (3 -> 2 dependent round trips
in the dyn module, whose tag `LDG` still depends on the BAD select; 2 -> 1 in the static modules,
which load no tags after [45c]).  Under `SLIDING_WINDOW=1` `idxPageBeg` depends on `cacheSeqLen`
and the item is a no-op (still correct); the bench builds have `SLIDING_WINDOW=0`.  Guarded
(`MIXED_COMPACT_TILE_LOOPS`) at the two `loadPages` call sites: `kCompactTileLoops` is defined in
`mha.cu` after `mhaUtils.cuh` is included, so the helper is a separate `getPageUngated` (rev 1
said "guarded in getPage", which cannot see the macro).  **Wall share (rev 2):** ~0.7-1 us per
CTA lifetime of 31 us, overlapped with the co-resident CTA's issue: ~2.6 waves x ~1 us / 113 us
= **~1-2 %** on every mode incl. a16 (rev 1's 2.5-5 % counted the per-CTA saving 1:1).  The
`XQA_NB_SUB_SEQ` sweep of step 4 is *not* re-run in this step (the [42] constants stay).

### 3.7 Budget (rev 2): per item, per role, with the realisation factor of its mechanism

Realisation factors, by mechanism (step 6's lesson: removed stalls migrate): **deterministic
instruction removal** (convergence code, dead work) 0.8-0.9; **exposed latency of a dependency
chain** (R2UR at load, serialised chains) 0.5-0.7 — the three co-resident warps per SMSP already
cover part of it, and `not_selected` rises when it is removed; **MIO-queue waits** (an LDS behind a
just-issued LDGSTS batch, STS -> LDS register WAR) 0.3-0.5 — the queue wait moves to the next MIO
op in the same warp.  Cycles per tile at 4.31 cycles per sample, fp8 module.

| item | mechanism | gemm0 modelled | gemm0 budget (x r) | gemm1 modelled | gemm1 budget | registers |
|---|---|---|---|---|---|---|
| [45a] one vote per call | serialised chains (latency) | -380 | **-190..-270** | -360 | -180..-250 | -1 |
| [45b] block-pipelined spans (static) | STS -> LDS WAR (MIO queue) | -220 (51) | **-70..-110** | -550 (127) | -160..-270 | +4 FP8 / +2 FP4 |
| [45c] flag in registers | LDS.U8 behind the LDGSTS batch (MIO queue) | -490 (114) | -150..-250 | -540 (125) | -160..-270 | +1 (dyn), -1 (static) |
| [45c] tags: dead work / REDUX | deterministic removal (90 `PRMT <- SHFL` + ~24 instructions per tile) | -390 - 100 | -350..-440 | -650 (151) - 100 | -520..-680 | 0 |
| [45d] lane-distributed pages | R2UR at load (latency); +6 SHFL per K call | -750 (173) + 50 | **-320..-470** | -650 (~150) + 50 | -270..-400 | +2 (..+4) vector: the `Vec` stages were URs (see 3.4) |
| [45e] rowSum mask | deterministic removal | -480 (111) | **-380..-430** | 0 | 0 | 0 |
| **sum** | | -2,760 | **-1,460..-1,970 = -8.7..-11.8 % of 16,700** | -2,800 | **-1,290..-1,870 = -7.9..-11.4 % of 16,340** | net +4..+8 over 124-127 (static: -1 -1 +2..4 +4; dyn: +1 +2..4) |
| [45f] prologue | one round trip per CTA lifetime | | -1..-2 % of the wall, all modes | | | 0 |
| (b) address-register WAR | LDGSTS batch drain | 1,440 | **0** | ~820 | 0 | |

Wall model (lock-step): `t = today x (1 - min(gemm0 cut, gemm1 cut)) x (1 - [45f])` with the
fill unchanged in absolute terms (it is one gemm0 tile-time per CTA start and shrinks with the
tile).  fp8: cut 7.9-11.4 % + 1-2 % -> **99-105 us**.  fp4 (101.5; its chain / F2FP / LDS
samples are larger, R2UR similar): **88-93**.  mixed (107.8; R2UR 3.1 % + format `LDS.U8 [UR]`
2.3 % + branch `wait` on the staged flag, all removed or turned into register ALU; no [45b]):
**92-97**.  transport_a16 (83.4; only the tag pipeline, [45e] and [45f] apply; its long_sb is DRAM
landing at 67 % of peak): **80-82**.

| mode | today | predicted band | target | verdict on the target |
|---|---|---|---|---|
| fp8 | 113.5 | **99-105** | <= 94 | **not reached by this step** (rev 1's 91-99 assumed 20 % gemm1 slack and a 1:1 [45f]); the residual is the fill / wave tail ([42]) and the LDGSTS-batch drain (b), i.e. copy shape (A2/D6), both out of scope here |
| fp4 | 101.5 | **88-93** | <= 59 | not this kernel (step 6 section 5.3) |
| mixed | 107.8 | **92-97** | <= 101 | **pass predicted** (margin 4-9 us) |
| transport_a16 | 83.4 | **80-82** | 135 | pass |

## 4. Order of work, the register gate, live-set arithmetic per item

One commit per item, in the order **[45c] -> [45e] -> [45f] -> [45d] -> [45a] -> [45b]** (cheapest
and most certain first; the register-costing item last).  Gate after every commit (remote, not in
this worktree): `cuobjdump -res-usage` REG <= 128, STACK 0, LDL 0, STL 0 on all four sm90 q=4
modules.  Live-set deltas at the kernel's pressure point (the gemm0 part body: `acc` 32 + Q / K
fragments + loop state + the expansion's 18-22), against the 124 (dyn) / 127 (a16) / 126 / 126
(fp8 / fp4) of step 6:

| commit | static fp8 / fp4 | dyn | why |
|---|---|---|---|
| [45c] | -1 (`pageTagLane`, `pageFormats` word and the flag register go) | +1 (`kTagWordCurr/Next` +2, `pageFormats` -1) | by construction: nothing new is live across the part body except the two words |
| [45e] | 0 | 0 | one SHFL replaces a convergence sequence |
| [45f] | 0 | 0 | same values, one more predicate |
| [45d] | **+2 vector** (`pageLane`, `pageLaneNext`; the replaced `Vec<int,4>` stages were URs per the step-6 `R2UR`, i.e. outside the 128 cap), +2 more if the per-lane list address is hoisted; the per-span `IMAD.WIDE` chain moves UR -> vector (count unchanged) | same (+2..+4) | the lane value must live in a vector register - that is the point of the item |
| [45a] | -1 | 0 (pre-vote temporaries die before the loop; the body re-reads the word) | `s01[4]` replaces `s01, r0, r1, f0, f1` |
| [45b] | +4 (FP8) / +2 (FP4) | 0 (not applied) | second payload set |

Net +4..+8 over 126 (static; dyn +3..+5 over 124): [45d] and [45b] are the two items that can
cross 128 - [45d] by +2..+4 (a 126 -> 128 reading after its commit is expected and is its own),
[45b] by +4 / +2 on top; both are gated, [45b] last.  Fallbacks if a
commit breaks the gate: [45b] -> `MIXED_EXPANSION_PIPELINED_SPANS 0` (single set; drop the item);
[45a] -> keep the per-span vote in the dyn module only (static modules keep the call vote);
[45c] -> keep the format word in smem for the dyn module; [45d] -> none needed (it frees).  Nothing
is traded for registers: 2 CTAs/SM is the step-4 lever and outranks every item here.

## 5. Verification artifacts (mechanism first, stopwatch last) — rev 2

- **A0 (before any timing; pristine tip `659eacfa`, one ncu `SourceCounters` launch at
  `XQA_NB_SUB_SEQ=1` next to the step-6 default run):** the `0x2e40` + `NANOSLEEP` share at
  n = 1 vs n = 5 (fill: ~5x lower; slack: unchanged) — decides whether section 2's lock-step
  reading holds (if not, the rev 1 budget applies and this doc is amended before A4).  Same run:
  the `R2UR` sites at `mhaUtils.cuh:1519` keep their share (per-load).
- **A1 (SASS, `cuobjdump -sass` / `-res-usage`, loops delimited by back-edges as in step 5; all
  four sm90 q=4 modules):** `VOTE.ALL` in the expansion 6 -> 2 (one per call site: K, V; dyn 2);
  `F2FP.F16.E4M3` in the static expansion 4 per K call issued before the vote; `MATCH.ANY` /
  `BRA.DIV` in `computeRowSum` 0; `SHFL.IDX` in the tile loops: 0 in the static modules for tags
  (rev 1's "0 everywhere" is superseded: [45d] adds 6 per K copy call / 3 per V call for the page
  index — `SHFL.IDX` count per copy call = nbSpans + scale iterations, no `PRMT` after them);
  `REDUX` 1 per copy call in the dyn module (`packMixedPageTags`), 0 in static; `R2UR` whose source
  is the page-index `LDG` 0 in every module; `LDG` in `loadPages` 1 (+1 tag in dyn) per call; `LDS.U8`
  at the flag offsets (`+0x1b030`, `+0x1b038` today) 0; `STS.U8` / `STS` flag and format stores
  0; `LDG.E.U8` tag loads 0 (static) / 1 per `loadPages` (dyn); `LDGSTS` static counts unchanged
  (47 / 47 / 55 / 65); `LDS.128` / `LDS.64` 2 and `STS.128` 4 per span-call unchanged; the static
  fold body's `LDS` of span s+1 precede the `STS` of span s ([45b]) **and, for every `LDS.128` /
  `LDS.64` in the two static fold bodies, no destination register is a source of an `STS.128`
  within the preceding ~10 instructions (the STS -> LDS register WAR [45b] exists to remove;
  ptxas is free to recycle the just-stored `out` registers, section 8.6) - if any is,
  `MIXED_EXPANSION_PIPELINED_SPANS 0` is taken**; `DEPBAR.LE SB0, 0x1` 3 per
  module unchanged; REG <= 128, STACK 0, LDL 0, STL 0; hot loops gemm0 part <= 1,200 SASS
  (1,114), gemm1 V <= 800 (726), dyn total hot <= 2,800.
- **A2 (ncu one launch, `--launch-skip 1 --launch-count 1`, the step-6 metric list plus
  `SourceCounters` on the `-lineinfo` build whose stripped SASS must be byte-identical to
  production):** `smsp__inst_executed.sum` fp8 36.0 -> 34-35.5 M, fp4 36.5 -> 34.5-36, mixed
  42.4 -> 39.5-41.5, a16 29.9 -> 28.8-29.9; warp-cycles per issued instruction fp8 8.13 -> <= 7.3,
  fp4 7.30 -> <= 6.6, mixed 6.56 -> <= 6.0; short_scoreboard fp8 2.01 -> <= 1.5, fp4 1.64 ->
  <= 1.25; long_scoreboard fp8 1.81 -> <= 1.45, mixed 1.34 -> <= 1.05 (the WAR part, ~3.4 %,
  stays by design); `launch__shared_mem_per_block_dynamic` 115,456 B and occupancy limits 2 / 2
  unchanged.  **PC sampling by the section-2 regions (the produced-wait rule of rev 1 is
  replaced):** (i) gemm0 tile-loop samples per warp-tile 0.445 (3,873 / 8,704) -> 0.39-0.41 and
  gemm1 busy samples 3,791 -> 3,350-3,500 — both roles shrink, that is the sign the model is right;
  (ii) the `xBar.produced` wait stays ~1,150-1,250 samples in absolute terms (one gemm0 tile-time
  per CTA start; its *share* rises to 11-12 % as the loops shrink — expected, not a failure);
  (iii) the `xBar.consumed` TRYWAIT (`mhaUtils.cuh:1677` / `mha.cu:2777`, the `0xf190`-class site)
  0.7 % fp8 / 1.8 % mixed -> if > 3 %, gemm1 has become the pacing role and the V-side items
  decide the next step; (iv) `R2UR` long_sb at the page loads 1.7 % -> 0; flag `ISETP` 2.4 % -> 0;
  `PRMT <- SHFL` 2.4 % -> 0; `F2FP.E4M3` + fold `BRA` 4.0 % -> <= 2 %; the `MATCH.ANY` group
  1.1 % -> 0; the WAR sites (b) ~3.4 % -> unchanged or moved onto the `LDGSTS` themselves
  (`lg_throttle`); the three DEPBAR neighbourhoods stay at ~0 long_sb.
- **A3 (correctness and byte-identity — shared files):** `tests/attention/run_xqa_mixed_page_transport.py`
  bit-exact (all cases; exit code 0) on nkcut2 (default and `XQA_NB_SUB_SEQ=2`) and on ws-1,
  after every commit; ws-1: all eight sm120 `xqa_mha` modules (formats -1/0/1/2 x q=1/q=4) and
  `mla_sm120` stripped SASS byte-identical to a pristine `659eacfa` build made in the same session
  (the dyn q=1 pair may show the known ptxas pristine-vs-pristine variation: compare against two
  pristine builds before reading it as a leak); nkcut2: the sm90 q=1 `xqa_mha_sm90` and q=1
  `xqa_mha` objects byte-identical; the sm90 q=4 a16 module changes (its tag pipeline is removed;
  the stock A16 copy is kept) and is accepted on its A1 counts.  `ptxas -v`: no C7507 anywhere
  (dataflow.md A4).
- **A4 (bench):** three locked rounds `--repeats 2 --trials 5` (2 x 117 us < 1.5 ms), pristine
  `659eacfa` checkout with its own JIT workspace interleaved (memory: a cached workspace rebuilds
  from the checkout's source and is not a baseline), q=1 control rows included; the table below.

**Accept / reject (gap-free; medians of three rounds; record = merged-tree confirmation @
67a6b4aa; "keep" = the step is merged, "open" = the target row stays open in targets.md):**

| mode | interval | outcome |
|---|---|---|
| fp8 | t <= 94 | accept, target met |
| fp8 | 94 < t <= 105 | **accept** (predicted band), target open; A2 counters reported |
| fp8 | 105 < t <= 110 | keep, target open, **model missed**: A2 (i)-(iv) are read item by item and the item whose counter did not move is bisected by commit (each commit is independently revertible) |
| fp8 | 110 < t <= 113.5 | keep only if fp4 and mixed are in their accept rows and A2 (i) shows both loops shrank; otherwise revert the commit(s) whose A2 counter did not move; target open |
| fp8 | 113.5 < t <= 115.8 (1.02x) | within the co-tenant band: three more rounds interleaved with the base; if still > 113.5 revert item by item until <= 113.5 |
| fp8 | t > 115.8 | **reject the step** (revert all six) |
| fp4 | t <= 90 | accept |
| fp4 | 90 < t <= 96 | keep, open (as the fp8 105-110 row) |
| fp4 | 96 < t <= 101.5 | keep only with fp8 and mixed in their accept rows; otherwise bisect |
| fp4 | 101.5 < t <= 103.5 | co-tenant band: re-run, then bisect |
| fp4 | t > 103.5 | reject the step |
| mixed | t <= 97 | accept, target met |
| mixed | 97 < t <= 101 | accept, target met (low end of the model) |
| mixed | 101 < t <= 105 | keep, target open, bisect by A2 (the R2UR and format-LDS counters are the mixed-specific ones) |
| mixed | 105 < t <= 107.8 | keep only with fp8 and fp4 in their accept rows; otherwise bisect |
| mixed | 107.8 < t <= 110 | co-tenant band: re-run, then bisect |
| mixed | t > 110 | reject the step |
| transport_a16 | t <= 84 | accept |
| transport_a16 | 84 < t <= 87 | keep (a16 changes only by [45c]'s dead-work removal, [45e], [45f]; the step-6 a16 sessions ran 86.2-86.8 on byte-identical SASS: session offset); confirm on the interleaved base |
| transport_a16 | t > 87 | reject the step (the a16 guard leaked or the tag removal changed the A16 copy) |

## 6. Do not build if

1. A0 shows the `0x2e40` + `NANOSLEEP` share unchanged at `XQA_NB_SUB_SEQ=1` (steady-state
   slack, rev 1's reading): the wall follows gemm0 alone and the budget is re-derived (rev 1
   section 3.7 numbers minus blockers 3-4) before A4 — the code is the same either way.
2. Any item needs a `SharedMem` change, a third K/V buffer, 64 B K parts or 1 CTA/SM — rejected by
   section 1; do not revisit without new DEPBAR samples.
3. REG > 128 or STACK > 0 after [45c]+[45e]+[45f]+[45d] (the register-freeing items): stop,
   report; do not build [45a] / [45b] on a spilling base.
4. The sm120 SASS changes (guard leak) at any commit — fix the guard before any timing.
5. Another track has touched `mha.cu` `:2339-2545` / `:2851-3135` / `:3200-3345` or the two [44]
   helpers since `659eacfa` — checked on the merged tip at coding time (659eacfa merged F26 and
   [8] after the step-6 artefacts; `mha.cu` and `mhaUtils.cuh` are byte-identical to the wtS6v2
   tree, md5 `7478...5f` / `24fc...85`).
6. Not in scope, whatever the numbers say: `csrc/xqa/mha_sm90.cu` (SPEC_DEC route), the FA3
   headers, the copy ownership / LDGSTS shape (A2/D6 — the (b) WAR bucket lives there), unrolling
   the dyn module's span or page loops (step 3/4 fetch stalls), the [42] `nbSubSeqPerSeq`
   constants, the BAD-page scale-row zero fill (follow-up).
7. The step is judged against fp4 <= 59 — it is not this kernel's number (step 6 section 5.3).

## 7. Go / no-go

**Go** for [45] as a register-only package under the existing guards, in the section-4 order with
the register gate after every commit; **[45b] static-only, [45d] as lane-distributed page indices,
[45a]'s dynamic-module predicate by select.**  **No-go** for every smem-depth option (section 1):
none fits at 2 CTAs/SM except K3 x 64 B, and the wait it would deepen carries < 0.1 % of the
samples.  Predicted (rev 2): fp8 99-105 (target 94 **not reached**; what remains is the fill / wave
tail and the LDGSTS-batch drain), fp4 88-93, mixed 92-97 (target 101 passes), a16 80-82.  Main
risks: the realisation factors (three of the six items are latency- or queue-class), the
128-register cap (+4 for [45b] over a -2..-8 base), the dyn module's `wait`-dominated dispatch
chains (24 % of mixed, touched only through the format word), and the co-tenant on nkcut2 (every
A4 round interleaved with the pristine base).

Artefacts read for this design (no new remote jobs; scripts left at
`nkcut2:/tmp/mixedkv-wtS7-longsb.py`, `/tmp/mixedkv-wtS7-regions.py` — read-only over the step-6
CSVs and nvdis files): `nkcut2:/tmp/mixedkv-wtS6-a0/{v2-fmt-1,v2-fmt0,v2-fmt1,v2-fmt2}.nvdis`,
`{v2-fp8,v2-fp4,v2-mixed,new-transport_a16}.source.csv`, `buckets.py`, `stalls.py`.

## 8. As written (code state of this worktree; one subsection per commit, appended as each lands)

### 8.0 Stock-view identity of the shared files (checked at review, before any remote build)

`unifdef -k -DMIXED_HOISTED_COPY=0 -DMIXED_ALL_HOISTED_COPY=0 -DMIXED_BF16_PLACEMENT_EXPANSION=0
-DMIXED_COMPACT_TILE_LOOPS=0 -DMIXED_PAGE_STATIC_FORMAT={-1,0,1,2}` of `d9a317e4`'s and the
[45b] tip's `csrc/xqa/mha.cu`: the non-directive code lines are identical for all four formats
(only `#if` spellings differ), so the sm120 and sm90 q=1 preprocessed kernels are unchanged.  The
`mhaUtils.cuh` edits are confined to the two [44] templates (instantiated only under
`MIXED_BF16_PLACEMENT_EXPANSION`) plus new never-called inline / template helpers; the removed
`foldScalePairFinite` has no other user; `mha_sm90.cu` includes `mhaUtils.cuh` but instantiates
none of the changed templates.  The ws-1 SASS byte-compare (A3) stays the binding gate (expect the
known ptxas variation on the dyn q=1 pair only).

### 8.1 [45c] — flags, formats, tags in registers (commit "[45c]")

Files: `csrc/xqa/mhaUtils.cuh` (`packMixedPageTags<nbPages>`: `redux.sync.or` of `tag << 8 lane`
over lanes `< nbPages`, shuffle fallback below sm80 / host pass; `mixedPageTagsWord` for the
probe build; `mixedPageTagOfSpan(word, span)`; `expandMixedPartialHeadsInPlaceBF16Placement` and
`copyMixedPartialHeadsAsyncHoisted` take `uint32_t formatWord` instead of `MixedPageFormats
const&` — the dynamic module's per-span format is `(word >> 8 span) & 0xFF`, the static modules
`unused()` it), `csrc/xqa/mha.cu` (new guards `MIXED_ALL_HOISTED_COPY = MIXED_HOISTED_COPY &&
MIXED_PAGE_STATIC_FORMAT != 0` and `kMixedStaticNeedsExpansion = MIXED_PAGE_STATIC_FORMAT > 0`
(static modules only); K and V loaders, expansion sites and buffer-advance sites).  Every
change in `mha.cu` is `#if MIXED_HOISTED_COPY <new> #else <stock verbatim> #endif`: `unifdef
-DMIXED_HOISTED_COPY=0 -DMIXED_ALL_HOISTED_COPY=0 -DMIXED_BF16_PLACEMENT_EXPANSION=0
-DMIXED_COMPACT_TILE_LOOPS=0` of the old and the new `mha.cu` differ only in the new `#define`
block (the sm120 / sm90 q=1 preprocessed source is unchanged; A3 confirms on the objects).

Data flow, per copy call (K part / V half-tile), by module:

```
static fp8 / fp4 (MIXED_ALL_HOISTED_COPY, kMixedStaticNeedsExpansion = true):
  loadPages: pageIdx <- pageIdxNext; pageIdxNext <- getPage(...)      (no tag load: pageTagLane is gone)
  copy:      copyMixedPartialHeadsAsyncHoisted(..., pageIdx, tagWord = 0 (constexpr), ...)
  expansion: if constexpr (true) expand...(..., 0u, ...)                 (no flag load, no branch)
static a16 (kMixedStaticNeedsExpansion = false):
  loadPages: as above; copy: stock bounds-checked copyPartialHeadsAsync (the only body; isFullTile
             is constant false under kCompactTileLoops); expansion: if constexpr (false) -> discarded
dynamic (MIXED_PAGE_STATIC_FORMAT < 0):
  loadPages: pageIdx <- pageIdxNext; pageTagLane <- mixedPageTagLane(pageIdx) (lane s: tag of page s);
             pageIdxNext <- getPage(...)
  copy:      tagWord = packMixedPageTags(pageTagLane)   REDUX.OR: byte s = tag of span s, 0 = all A16
             kTagWordNext = tagWord  (V: vTagWordNext)
             copyMixedPartialHeadsAsyncHoisted(..., pageIdx, tagWord, ...)  per-span format = byte span;
             scale loop: pageValid ? byte localPage : kA16
  advance:   idxCurrSMemKBuf++ ; kTagWordCurr = kTagWordNext        (prologue site and loop site; V same)
  expansion: if (kTagWordCurr != 0) expand...(..., kTagWordCurr, ...)  per-span format = byte span
```

Control flow: no lane-0 `STS.U8` / `STS`, no `__syncwarp()` in the loaders (it served only the
flag path); the K copy site has one body per module chosen by the preprocessor (`MIXED_ALL_
HOISTED_COPY`: hoisted; a16: stock); the a16 module's `HeadPtr src` is still built, the
all-hoisted modules build none.  The two words are named registers copied at the four
`idxCurrSMemKBuf++` / `idxCurrSMemVBuf++` sites (no `u32[2]` indexed by the `CircIdx`).  Rotation
trace (K): prologue `loadKTilePart(part 0)` writes buffer 0 and `kTagWordNext = T(tile 0)`;
`idxCurr++` -> 0, `Curr = T(0)`; loop p=0: `loadKTilePart(part 1)` -> buffer 1, `Next = T(0)`;
expansion of buffer 0 reads `Curr = T(0)`; `idxCurr++`, `Curr = T(0)`; p=1: `loadKTilePart(tile
1, part 0)` -> buffer 0, `Next = T(1)`; expansion of buffer 1 reads `Curr = T(0)`; `idxCurr++`,
`Curr = T(1)`.  `smem.kFormats / vFormats / kNeedsExpansion / vNeedsExpansion` stay declared
(unread in the guarded build; the compact-register path still names them) so the barrier
addresses do not move.  `SharedMem` is unchanged (115,456 B).

Expected SASS (A1): static modules — `LDG.E.U8` 0, `SHFL.IDX` for tags 0, `PRMT` of the tag
broadcast 0, flag `STS.U8` / `LDS.U8` 0, `WARPSYNC` in the loaders 0; the a16 module loses the
same and its hoisted copy body (dead); dyn — 1 `REDUX` per copy call (2 per tile), `ISETP` on a
register for the expansion decision, `SHF.R + LOP3` per span for the format (uniform datapath
if ptxas keeps the REDUX result in a UR), `LDS.U8 [UR]` of `kFormats` 0.  Registers: static -1
(`pageTagLane`, `pageFormats`, the flag), dyn +1 (`Next/Curr` +2, `pageFormats` -1).  REG <= 128,
STACK 0, LDL / STL 0.

### 8.2 [45e] — `computeRowSum` quad broadcast with a constant mask (commit "[45e]")

File: `csrc/xqa/mha.cu` `computeRowSum` (`:~805`), inside the `#pragma unroll` loop over
`QuadRegRowMax::size` (4): `#if MIXED_COMPACT_TILE_LOOPS __shfl_sync(0xFFFFFFFFU, rowSum[i],
laneId() & ~3U) #else <stock> #endif`.  `MIXED_COMPACT_TILE_LOOPS` is defined at `mha.cu:~186`,
before the function.  Data flow: `rowSum[i]` (per lane) -> `SHFL.IDX` from lane `4 (lane / 4)`
-> `rowSum[i]`; the stock form reads the same source lane (lane 0 of the 4-wide segment) under a
4-lane mask.  Control flow: none added; the function runs once per tile in the converged gemm0
warp (`mha.cu:2777`), so the full mask is the true active set.  Expected SASS: 4 `SHFL.IDX` (was
4 `SHFL.IDX` + `WARPSYNC` / `MATCH.ANY` / `REDUX.OR` / `VOTEU.ANY` / `LOP3` / `BRA.DIV` per
shuffle); `MATCH.ANY` 0 and `BRA.DIV` 0 in the gemm0 tile loop.  Registers: 0.  sm120 / M32:
the `#else` text is byte-for-byte the stock line.

### 8.3 [45f] — page-list load not gated on the sequence length (commit "[45f]")

Files: `csrc/xqa/mhaUtils.cuh` (`ldgNcPageIndex`: `asm volatile ld.global.nc.b32`;
`getPageUngated<nbLoadedPages>(cacheList, idxReq, idxPageBeg, nbPages)`), `csrc/xqa/mha.cu` (K
and V `loadPages`: `#if MIXED_COMPACT_TILE_LOOPS getPageUngated(...) #else getPage(...) #endif`;
the stock `getPage` is untouched and stays the body of every other build).  Data flow per
entry i: `idxPage = idxPageBeg + i`; `idxPage < maxNbPagesPerSeq` (kernel parameter) predicates
the `LDG` of `kvCachePageList[maxNbPagesPerSeq idxReq + idxPage]`; `idxPage < nbPages` selects
`loaded` or `kBAD_PAGE_INDEX`.  The `asm volatile` keeps the load's predicate free of `nbPages`
(a plain load whose only use is the select could legally be predicated on both conditions by
the compiler, which would restore the dependency).  Control flow: predication only.  Values:
identical to `getPage` for every `idxPage` (`nbPages <= maxNbPagesPerSeq` by construction of the
page table; `cacheSeqLen == 0` -> `nbPages 0` -> every entry BAD).  Prologue chain: static
modules `seqLen || list -> copy` (was `seqLen -> list -> copy`; the tag load is gone since
[45c]); dyn `seqLen || list -> tag -> copy` (was three dependent round trips).  Under
`SLIDING_WINDOW=1` `idxPageBeg` itself depends on `cacheSeqLen` (still correct, no gain); the
bench builds have `SLIDING_WINDOW=0`.  Expected SASS: the list `LDG.E.CONSTANT` (`.nc`) no
longer under a predicate derived from the `seqLenList` load; `SEL` after it; count of `LDG` per
`loadPages` unchanged (4 K / 2 V; [45d] takes it to 1).  Registers: 0 (one more predicate).

### 8.4 [45d] — lane-distributed page indices (commit "[45d]")

Files: `csrc/xqa/mhaUtils.cuh` (`getPageLaneUngated<nbLoadedPages>`: lane s < nbLoadedPages
loads page `idxPageBeg + s` with the [45f] predicate split, returns one `KVCachePageIndex` per
lane; `mixedPageTagOfLane(transport, pageLane)`: each lane loads the tag of its own page (0 for
BAD); `copyMixedPartialHeadsAsyncHoisted<..., isK, nbPages>(dst, scales, transport,
KVCachePageIndex pageLane, uint32_t formatWord, ...)` — `nbPages` is now an explicit template
argument (it was deduced from the vector), the span loop reads `page = __shfl_sync(~0,
pageLane, span)` and the scale loop `__shfl_sync(~0, pageLane, localPage)` with the
lane-dependent source), `csrc/xqa/mha.cu` (`MIXED_ALL_HOISTED_COPY` modules: `KVCachePageIndex
pageLane, pageLaneNext` replace `KCachePageIndices pageIdx, pageIdxNext` (V: the
`VCachePageIndices` pair); `loadPages` rotates the lane value, the dyn module's tag load is
`mixedPageTagOfLane(pageLane)`; the two hoisted copy calls pass `pageLane` and
`nbPagesPerWarpTile` / `nbPagesPerVTile`).  The a16 module and every non-hoisted build keep the
vector form (`#else` = stock / [45f] text; the stock-view `unifdef` of `mha.cu` is
token-identical to the tip's).

Data flow (fp8 / fp4 / dyn), per `loadPages(p)` call: `pageLane <- pageLaneNext` (indices of the
tile about to be copied; loaded two calls ago), dyn: `pageTagLane <- page_format[pageLane]` (lane
s, predicated on `pageLane != BAD`), `pageLaneNext <- getPageLaneUngated(p)` (one predicated
`LDG` per warp: lanes 0..3 (K) / 0..1 (V), consecutive words of one 16 B sector; BAD elsewhere).
Per copy call: span s -> `SHFL.IDX pageLane, s` -> `pageValid`, `laneOff` (3 `IMAD.WIDE`) ->
LDGSTS x2 as before; scale loop i -> `SHFL.IDX pageLane, 2 i + lane / 16` -> `pageValid`, scale
`LDGSTS`.  Nothing about the page index is ever warp-uniform-provable (each lane's value is its
own load), so ptxas cannot park it in a uniform register: the `R2UR` 12 instructions after the
`LDG` (section 2 (a)) has no source.  Prefetch distance unchanged (indices two tiles ahead, tags
one tile ahead); the V prime loop and `advanceVPages` are untouched (they call `loadPages` as
before).  Control flow: unchanged; the copy's per-span branch (dyn) is on the tag word, not on the
page.  `SHFL.IDX` per copy call: K 4 + 2 = 6, V 2 + 1 = 3 (the tag broadcast's 4 `SHFL` + 3 `PRMT`
are gone since [45c]).  Expected SASS: `LDG` in `loadPages` 1 (+1 tag, dyn) per call instead of
4 + 1 (K) / 2 + 1 (V); `R2UR` fed by the page-index `LDG` 0; `LDGSTS` count unchanged.
Registers (corrected in review): **+2 vector** (`pageLane`, `pageLaneNext`) - the replaced `Vec`
stages were uniform registers (the step-6 `R2UR` is the proof), which never counted against the
128 cap - plus up to +2 if ptxas hoists the per-lane 64-bit list address; the per-span address
`IMAD.WIDE`s move from the uniform to the vector datapath (same count).  A 126 -> 128 reading
after this commit is attributed here.

### 8.5 [45a] — one fold vote per call (commit "[45a]")

File: `csrc/xqa/mhaUtils.cuh`, `expandMixedPartialHeadsInPlaceBF16Placement` body (the
signature, lane constants and entry / exit `__syncwarp` are [44]'s; the stock helper is
untouched).  `foldScalePairFinite` is replaced by `kMixedFoldBound = 255.5 * 2^120` and
`foldScalesFinite(finite, foldOk) = __all_sync(~0, foldOk && finite)`.  Four building-block
lambdas with compile-time format / fold tags: `block(fmt, fold, p, a, b, sf2)` (decode + two
`STS.128`), `loadBlock(fmt, addr, p)` (`LDS.128` / `LDS.64`), `scalePair(fmt, fold, s01, sf2_0,
sf2_1)` (the span's two bf16x2 scale broadcasts from its scale word), `spanFinite(s01, gFold)`
(`fmaxf(|s_0 g 2^k|, |s_1 g 2^k|) < bound` — the [44] per-span compare, so a NaN pair (zero
scales with `|g| >= 2^8`) still fails and forces the fallback).

```
static fp8 / fp4:
  s01[s] = LDS.U16 [scaleAddr + 64 s]  for s < nbSpans          (4 K / 2 V, independent)
  packed[0] = LDS [addr0], packed[1] = LDS [addr2]              (span 0, before the vote: loads-first)
  finite = AND_s spanFinite(s01[s], g 2^k)                       (F2FP.E4M3, 2 HADD2, 2 FMUL, FMNMX, FSETP per span)
  fold = VOTE.ALL(foldOk && finite)                              (ONE per call; was one per span)
  spans<fold>: for s (unrolled): s > 0 ? LDS [addr0 + 2048 s], LDS [addr2 + 2048 s]
               sf2 pair = scalePair(s01[s])                      (recompute: 1 F2FP + 2 HADD2 + 2 FMUL + 2 PACK)
               block(packed[0] -> addr0, addr1); block(packed[1] -> addr2, addr3)
dynamic:
  pre-vote (unrolled over the 4 spans): fmt = byte s of formatWord; s01 = LDS.U16;
     finite &= (fmt is FP8 | FP4) ? spanFinite(s01, fmt == FP8 ? g8 2^120 : g4 2^126) : true   (SELECT)
     foldOk &= fmt == FP8 ? |g8| >= 2^-117 : fmt == FP4 ? |g4| >= 2^-117 : true
  fold = VOTE.ALL(foldOk && finite)
  for span (rolled, #pragma unroll 1): fmt byte -> if FP8 / FP4: expandSpan<fmt>:
     s01 = LDS.U16 (again; a register array indexed by the rolled span would go to local memory),
     packed = LDS x2, fold ? body<true> : body<false>  (uniform branch on the call-level flag)
```

Exactness: unchanged — both bodies give the reference's single rounding `bf16_rn(x *
bf16_rn(s g))` ([44] rev 2), so moving a span from the fold body to the fallback body changes no
bit; the call takes the fallback iff some span's pair fails its own per-span compare (evaluated
exactly as before) or a present format has `|g| < 2^-117`.  Dyn A16 / BAD spans contribute
`true` by select, never through a multiply of their (unwritten) scale word.  Static BAD spans
(tail parts) keep their pre-existing stale word: bit-identical to [44] for that span, and a
possible perf-only fallback of the other spans of that call (section 3.1 follow-up).
Live set across the vote (static): `s01[nbSpans]` (4) + span 0's payload (8) = 12 vs [44]'s `s01,
r0, r1, f0, f1` (5) + payload (8): **-1 register**; the dyn pre-vote temporaries die before the
loop.  Expected SASS (A1): `VOTE.ALL` in the expansion 2 per module (K call, V call; fp8 module
had 6), `F2FP.F16.E4M3` in the static K body 4 before the vote + 1 per span in each fold body,
one `BRA` on the vote per call; `LDS.U16` per K call 4 (static) / 8 (dyn: pre-vote + body);
`LDS.128` / `LDS.64` 2 and `STS.128` 4 per span unchanged; no LDL / STL.  The static `s01[nbSpans]`
(and [45b]'s `packed[2][2]`) are register arrays only while `#pragma unroll` fully unrolls the
span loops (nbSpans 4 / 2); a failure to unroll would put them in local memory - the LDL / STL 0
+ STACK 0 check after this commit and after [45b] is what catches it, and is not skipped.

### 8.6 [45b] — block-pipelined spans, two register sets, static modules only (commit "[45b]")

File: `csrc/xqa/mhaUtils.cuh`, the static branch of `expandMixedPartialHeadsInPlaceBF16Placement`
(`MIXED_PAGE_STATIC_FORMAT > 0`); the dynamic branch is untouched (one set; section 3.2).  Switch
`MIXED_EXPANSION_PIPELINED_SPANS` (default 1; 0 = the [45a] single-set body, the section-4
register-gate fallback — a build constant, not a tuning knob).  `packed[2][2]` = `[set][block]`,
indexed only by the unrolled span parity (compile-time after `#pragma unroll`; the LDL / STL 0
gate catches any failure to unroll).

```
spans<fold>, span s (unrolled), set = s % 2, nxt = (s + 1) % 2:
  sf2 pair = scalePair(s01[s])
  s + 1 < nbSpans:  LDS [addr0 + 2048 (s+1)] -> packed[nxt][0]      (before any STS of span s)
  decode packed[set][0] -> STS.128 [addr0 + 2048 s], [addr1 + 2048 s]
  s + 1 < nbSpans:  LDS [addr2 + 2048 (s+1)] -> packed[nxt][1]
  decode packed[set][1] -> STS.128 [addr2 + 2048 s], [addr3 + 2048 s]
```

Why the WAR should disappear, and where the argument is incomplete (review): the `LDS` that
fills `packed[nxt][0]` is issued before block(set, 0)'s `out` registers exist, and the `LDS` into
`packed[nxt][1]` before block(set, 1)'s, so neither can land on the registers of the `STS` pair it
interleaves with, and each `LDS` lands under the ~24 instructions of the following decode.  But
the `LDS` into `packed[nxt][1]` is issued *right after* block(set, 0)'s two `STS.128`, and the
`LDS` into `packed[nxt][0]` follows the previous span's block-1 `STS.128` pair after only
`scalePair` (~7 instructions); the `out` registers of those STSs are dead at exactly that point,
so the allocator is free to put the `LDS` destination on them - the very STS -> LDS WAR the item
claims to remove.  The structure removes the reason to *prefer* those registers, not the
possibility.  Hence the A1 gate above: read each fold body's `LDS` destinations against the
sources of the `STS.128`s in the preceding ~10 instructions; if any collides, the item is
`MIXED_EXPANSION_PIPELINED_SPANS 0` (single set, [45a] body) rather than +4 registers for
nothing.  (Issuing both of span s+1's loads before block(set, 0) would not change this: the
first of them still follows the previous span's last `STS` pair; it only adds 4 live registers.)  Shared-memory ordering: spans are 2,048 B apart, so a load of span
s+1 never aliases a store of span s; within a span both blocks are loaded (one in the previous
iteration, one now) before the span's first store, as in [44].  Peak live set: `packed[set][1]`
+ `packed[nxt][0]` + `packed[nxt][1]` (12 FP8 / 6 FP4) + `out` (8) + `sf2` (2) = 22 / 16 vs
[45a]'s 18 / 14: **+4 (FP8) / +2 (FP4)**.  Expected SASS (A1): in each static fold body the
`LDS.128` / `LDS.64` of span s+1 precede the `STS.128` of span s (address immediates `+2048 (s+1)`
vs `+2048 s`); counts per span unchanged (2 `LDS`, 4 `STS.128`); REG <= 128, STACK 0 — if not,
`MIXED_EXPANSION_PIPELINED_SPANS 0`.


## 9. As measured (2026-09-04, nkcut2 H200 + ws-1 RTX 5090; full tables in backends.md "Track S step 7")

Final tree `d453cbcb` = [45c] + [45e] + [45f] + [45a] (static-module call vote); three
design-prescribed fallbacks were taken in this order, each followed by the full re-gate (register
gate, 72/72 matrix default + `XQA_NB_SUB_SEQ=2` on nkcut2, 72/72 on ws-1, sm90 q=1 objects
byte-identical, all eight sm120 modules byte-identical, lineinfo == production):

1. **[45b] -> `MIXED_EXPANSION_PIPELINED_SPANS 0`** (section 8.6 gate): in the fp4 fold body
   ptxas placed span 2's `LDS.64 R36` on the registers of span 0's `STS.128 [R51], R36` seven
   instructions earlier (`0xc090` -> `0xc100`).  The two fp8 "hits" of the linear scan were
   cross-branch (tail of one body / entry of the other) and prologue, not collisions.  The
   two-set code stays under the switch.
2. **[45d] reverted (`0148a3bf`)**: the accept table's fp8 110-113.5 row.  Its A2 counter (`R2UR`
   long-scoreboard at the page load 1.7 % -> 0) moved literally, but the same load's round trip
   re-appeared on the BAD `SEL` of `getPageLaneUngated` (fp8 3.5 %, fp4 3.4 %, mixed 5.0 %) and the
   copy's new `SHFL.IDX -> ISETP` page-valid chain took ~2.7 % short-scoreboard; the per-commit
   bisect priced the commit at fp8 +3.4 us, fp4 +2.1 us (mixed -0.6).
3. **[45a] dyn -> per-span vote (`MIXED_DYN_CALL_VOTE 0`)**, the section-4 fallback: the bisect
   priced the dynamic module's call-level vote at mixed +2.8 us (its pre-vote pass runs all four
   spans' `LDS.U16 -> F2FP` chains whatever their format; dyn `F2FP.E4M3` samples 1.1 -> 2.0 %),
   while the static modules' call vote gained fp8 -2.7 / fp4 -2.5.

Register gate: met at every commit (REG 122-128, STACK 0, LDL 0, STL 0; the static modules sit at
the 128 cap from [45c] on, so the "-1" of section 4 was spent by ptxas, not saved).  Final REG 124
/ 128 / 128 / 128 (dyn / a16 / fp8 / fp4).

**A0 (section 2's discriminating run): intermediate.**  Produced-wait group at n = 5: 1,293
samples = 13.0 % (905 on the `0x2e40` BRA, 265 on the NANOSLEEP); at n = 1: 765 = 7.9 % (687 on
the BRA, 4 on the NANOSLEEP).  A 1.7x drop, not ~5x and not unchanged: the NANOSLEEP part is fill,
~690 samples on the TRYWAIT branch survive at 16 tiles per CTA and are gemm1 waiting on gemm0 in
steady state (part of it the 1-CTA/SM exposure of the 136-CTA run).  "0-2 % slack" in section 2
is amended to "~5 % gemm1-on-gemm0 wait in steady state"; the accept table was not changed (the
code is the same either way), and the measured cuts were compared against it as written.  `R2UR`
at `getPage` 2.0 % -> 3.9 % at n = 1: per-load, as predicted.

**A4 (medians of three locked rounds, base = pristine `659eacfa` interleaved in the same
session; record base in parentheses):** transport_a16 **85.3** vs 86.7 (83.4) — keep row 84-87,
faster than the interleaved base; fp8 **110.3** vs 116.5 (113.5) — row 110-113.5 by 0.3 us
(spread 109.8-110.5), predicted 99-105 **missed**; fp4 **95.2** vs 103.7 (101.5) — row 90-96,
predicted 88-93 missed; mixed **102.5** vs 110.3 (107.8) — row 101-105, predicted 92-97 missed,
target <= 101 not met on the raw median (the same-session base runs 2.2-3.3 us above the record).
Ratios vs the interleaved base 0.984x / 0.947x / 0.918x / 0.929x; q=1 controls equal; no reject
row triggered.  Bisect (two rounds): [45c] fp8 -5.1 / fp4 -7.2 / mixed -7.1 / a16 -1.6; [45e]
-0.5..-1.1; [45f] neutral; [45d] +3.4 / +2.1 / -0.6; [45a] -2.7 / -2.5 static, +2.8 dyn.

**A2:** `smsp__inst_executed.sum` fp8 36.0 -> 32.5 M, fp4 36.5 -> 33.1 M, mixed 42.4 -> 39.6 M, a16
29.9 -> 27.4 M (all beyond the predicted bands); warp-cycles per issued instruction fp8 8.13 ->
8.36, fp4 7.31 -> 7.18, mixed 6.51 -> 6.43 (every <= target missed); short_scoreboard fp8 2.01 ->
1.75, fp4 1.65 -> 1.32; long_scoreboard fp8 1.81 -> 2.48, mixed 1.38 -> 1.44 (rose); occupancy
2 / 2, 115,456 B, `LDGSTS` counts unchanged.  PC sampling: (i) gemm0 tile-loop samples per
warp-tile 0.423 -> 0.368 and gemm1 busy ~3,660 -> ~3,460 — both roles shrank; (ii) the produced
wait stays ~980 samples absolute; (iii) `xBar.consumed` 0 % fp8 / fp4, 1.9 % mixed, **3.6 % a16**
(gemm1 pacing part of the a16 module); (iv) flag `ISETP` -> 0, `PRMT <- SHFL` -> 0, `MATCH.ANY`
-> 0 as predicted; `R2UR` at the page load -> 0.3 % **but the BAD `SEL` of `getPageUngated` takes
4.7 % (fp8) / 4.1 % (fp4) / 5.6 % (a16)**, the dyn module's `getPage` `R2UR` stays 5.5 %;
`F2FP.E4M3` + fold `BRA` 6.7 % -> 5.0 % (<= 2 % missed).

**Corrections to this design, from the measurement.**  (1) Section 2 (a): the exposed round trip
at the page-index load is not a uniform-register artefact — whatever instruction first consumes
the loaded index (`R2UR` in [44], the BAD `SEL` in [45f] / [45d]) is scheduled directly behind the
load, so the two-tile prefetch distance is unused; the lever is to defer the first consumer to the
copy (raw loaded vector, select at the use), not to change the datapath.  Both [45d] and [45f] as
written leave this bucket at 4-6 %.  (2) Section 3.1 for the dynamic module: the call-level vote
costs more than the four serialised chains it removes (+2.8 us), because the pre-vote pass runs
every span's chain regardless of format; the per-span vote is the right form there.  (3) Section
3.2: the two-set body does not stop ptxas from reusing the just-stored `out` registers as the next
`LDS` destination (one hit in four bodies); a guaranteed form would need the loads issued before
the stores of the *previous* span's second block as well (+4 more live registers), which the
128 cap does not allow.  (4) The realisation factors of section 3.7 were still too high: the
instruction cut over-delivered (fp8 -9.6 % vs -6..-9 % implied) and the stall side under-delivered
(warp-cycles per instruction rose); the wall moved -5.3 % (fp8) against the modelled -8..-11 %.
Kept: [45c] (the whole tag pipeline was dead work in the static modules: -5..-7 us) and [45e];
[45f] is neutral and kept for the shorter prologue chain; [45a] static kept (-2.5..-2.7).
