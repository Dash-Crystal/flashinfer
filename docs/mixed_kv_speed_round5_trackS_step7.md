# Track S step 7 — [45] register-only dependency cuts for the sm90 SPEC_DEC q=4 build (design; no kernel edits in this worktree)

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
ring; the long-scoreboard time is (i) the gemm1 warps waiting on `xBar.produced` for gemm0 (9.6 %
of all fp8 samples: gemm0 is the pacing role, gemm1 has 20 % slack in its loop), (ii) the
consumers of the page-index / page-tag `LDG`s inside the copy (6.3 % fp8, 7.2 % mixed) and (iii)
the barrier poll's `NANOSLEEP` (2.3 %).  The lever is the per-warp dependency depth of the gemm0
tile — register-only items under the existing [44] guards, no SharedMem change.**

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
| sum -> `alignas(128)` | 115,400 -> **115,456** | 100,076 -> **100,096** | 132,324 -> **132,352** | 216,904 -> **216,960** | 117,008 -> **117,120** |
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

## 2. What the step-6 samples say (fp8 [44] module, 9,960 samples; fp4 / mixed in parentheses)

Regions are the SASS loops delimited by back-edges (`v2-fmt1.nvdis`; the gemm1 code is laid out
first).  Per warp-tile = region `Instructions Executed` / 8,704.

| region | SASS | samples | inst per warp-tile | stall mix |
|---|---|---|---|---|
| gemm1 V-iteration loop `[0x2390, 0x50f0]` (`mha.cu:3173-3345`) | 726 | 4,744 (47.6 %) | **1,959** (4 iterations) | long_sb 26 % (**953 = the `xBar.produced` TRYWAIT branch at `0x2e40`, `mha.cu:3229` = 20 % of the loop**), short_sb 25, selected 15, wait 12 |
| gemm0 tile loop `[0xa790, 0x11740]` (`:2570-2790`) | 1,787 | 3,628-3,873 (36-39 %) | **1,773-2,067** | short_sb 27-29 %, long_sb 11-16, wait 14, selected 13-14, not_selected 10-11, no_inst 6 |
| .. gemm0 part loop `[0xa990, 0xef30]` (`:2594-2696`) | 1,114 | 2,817 (28.3 %) | 1,451 (2 parts) | short_sb 33, selected 13, wait 12, long_sb 11, not_selected 11, no_inst 6, mio 5 |
| .. gemm0 outer (softmax, X hand-off) | ~670 | ~700 | ~600 | FMNMX <- SHFL chains, `MATCH.ANY` group (below), MUFU |
| out-of-line barrier poll (`NANOSLEEP` `0x11770`, `barriers.cuh:352-378`) | | 228 (2.3 %) | | long_sb (mostly gemm1's `produced` poll) |

**The pacing role is gemm0.**  gemm1 idles 20 % of its loop on `xBar.produced` (fp4 862 samples =
9.4 % of all, mixed 873 = 8.9 %); gemm0's own `xBar.consumed` wait (`0xa600`, `mha.cu:2777`) has
73 samples (0.7 %).  So the wall follows the gemm0 warp-tile (~16,700 cycles), and a gemm1-only
gain is worth nothing until gemm0 has gained ~20 %.

**Where the gemm0 tile goes (samples -> cycles at 4.31 cycles per sample):**

| item (gemm0, per warp-tile) | evidence (address, samples, top reason) | cycles per tile |
|---|---|---|
| expansion scale chain: `LDS.U16 -> F2FP.F16.E4M3 -> HADD2 -> FMUL -> FMNMX -> FSETP -> VOTE.ALL -> BRA`, once **per span** (4 per K part) | `F2FP.E4M3` `0xbaf0/0xc640/0xd100/0xdbc0` 72+72+62+31 short_sb; `LDS.U16` `0xc620/0xd0e0` 71+48; fold `BRA` `0xc6e0/0xd1a0/0xdc60/0xbc20` 50+48+41+27 (`mhaUtils.cuh:1107-1144`) | ~2,300 |
| next span's payload `LDS.128` WAR on the previous span's `STS.128` registers | `0xdbb0` 51 short_sb (`mhaUtils.cuh:1111`) | ~220 (+ the hidden serialisation) |
| `kNeedsExpansion` flag: `LDS.U8 -> ISETP -> BRA` right after the DEPBAR | `0xb970` 114 short_sb (`mha.cu:2629`) | ~490 |
| page tags: `LDG.E.U8` (lane) -> 4 x `SHFL.IDX` -> `PRMT` x3 -> `STS.U8` flag + `STS` formats | `0xaa60` 90 short_sb (`mha.cu:2409`, `:2454-2456`, `mhaUtils.cuh:219-227`) | ~390 |
| page-index / tag consumers in the hoisted copy: `R2UR UR, R(page)`, `LDC.64 c[..]` (per-format strides), `IADD3.X`, `IMAD.WIDE`, `IMAD.U32 <- UR` | `0xb710/0xb870/0xb7d0/0x97a0` 75+40+32+26 long_sb; `0xb200` 55; `0xb360` 35; `0xa370/0xb390` 48+17; `0xb700` 41; scale-loop `BRA` `0xf190` 73 (`mhaUtils.cuh:1294`, `:1319`, `:1368-1370`, `:1519`, `:1677`) | ~1,340 |
| `computeRowSum` quad broadcast with a lane-dependent mask -> `WARPSYNC / MATCH.ANY / REDUX / VOTEU / BRA.DIV` | `0x11310` 66 branch_resolving, `0x11410` 33, `0x11350` 30, `0x11360` 15 (`mha.cu:802`) | ~650 |
| `HMMA <- LDSM` first-use waits | `0xee70` 59 short_sb | ~250 |
| loop control / address chains (`wait`: fixed-latency dependent ALU pairs) | 338 samples spread over the part loop | ~1,460 |
| issue (selected) | 366 | ~1,580 (1,451 instructions) |

gemm1 shows the same items at its scale (per V iteration: `PRMT <- SHFL` `0x2510` 151, flag `ISETP`
`0x32c0` 125, `LDS.128` `0x3f90` 127, fold `BRA` 93+80, `F2FP` 91+70, page/tag long_sb `R2UR`
`0x2d00`-`0x2dc0` + `IADD3` `0x2cc0` ~ 300) plus its own rescale ballot (`FSETP.NEU` `0x2eb0` 112).
Mixed (dyn) adds the per-span format `LDS.U8 [UR]` from `smem.kFormats` (`mhaUtils.cuh:35`, 228
`wait` samples = 2.3 %) and has the largest page/tag long_sb share (`R2UR` `0x3ff0` 269 + 43,
format/scale branches 178+137+40+31: 7.2 % of all samples).

Two facts the record did not have: (1) in the **static fp8 / fp4 / a16 modules the whole tag
pipeline is dead work** — `needsMixedPageExpansion` is a constant (`mhaUtils.cuh:231-236`), the
hoisted copy ignores `formats` (`:1319`, `:1341-1345`), the expansion ignores them (`:1149-1152`)
— yet the tag `LDG.E.U8`, the four shuffles, the `PRMT`s and the lane-0 stores execute per copy
call because the stores are side effects (`mha.cu:2409`, `:2454-2457`, `:2965`, `:2972-2979`); and
the tag address depends on the page-index `LDG` (`mhaUtils.cuh:212`, 66 long_sb samples), which is
the dependent-load pair the "two tiles ahead" prefetch was built to hide (`mha.cu:2360-2392`).
(2) The per-span fold vote is a control dependency: the `asm volatile` loads of span s+1 cannot be
hoisted above span s's `if (fold)` (`mhaUtils.cuh:1140-1144`), so the four chains of a K part run
in series (~4 x 110 cycles) — step 6's "loads-first" order helped inside a span, not across spans.

## 3. Design [45]: six register-only items, all under the [44] guards

All code changes sit in `#if MIXED_BF16_PLACEMENT_EXPANSION` / `MIXED_HOISTED_COPY` blocks of
`mha.cu` (kernel body: preprocessor guards, as [44] item 1 explains — an `if constexpr` there is
still instantiated) and in the two [44] helpers of `mhaUtils.cuh` (never instantiated by sm120,
sm90 q=1 or `mla_sm120`), so every other build's preprocessed source is unchanged.  No
`SharedMem` member is added or removed (115,456 B stays; the four metadata arrays become unread
in the guarded build but keep their slots so the barrier addresses do not move).

### 3.1 [45a] One fold vote per call (expansion helper, `mhaUtils.cuh:1090-1162`)

Load every span's scale word first (K: 4 x `LDS.U16` at `scaleAddr + 64 span`, V: 2), run the
chain once — 4 x `F2FP.F16.E4M3` (independent, back-to-back on the XU pipe), 4 x `HADD2.F32`,
8 x `FMUL`, a 7-deep `FMNMX` tree, `FSETP`, one `VOTE.ALL`, one `BRA` — then the span bodies.
Static modules: two whole-call bodies (fold / fallback), spans unrolled and branch-free inside.
Dynamic module: `f_i` per span with the span's own `(g, 2^k)` selected by its tag (A16 spans
contribute 0), one vote over the call, then the rolled span loop with the per-span format branch
selecting `body<fold>` from the call-level flag (warp-uniform).  `sf2` per span is recomputed from
the kept scale word after the vote (1 `F2FP` + 1 `HADD2` + 2 `FMUL` + 2 `F2FP.PACK`, a ~30-cycle
chain with no vote and no branch; +2 live registers) rather than kept for all spans (+8).
Bit-exactness is unchanged: both bodies give the reference's single rounding ([44] rev 2); a call
takes the fallback iff any span would have.  The fold frequency is unchanged on the bench (FP8
scales capped at 128, unit global scales) and the `tinyglobal` / `maxscale` matrix rows exercise
the fallback.

Dependency depth per K part: today 4 x (11-deep chain, ~110 cycles, serialised by the branch) =
~440 cycles of exposed latency + the branch-resolve `wait` per span; after: 1 x ~130 + 4 x ~30 =
~250: **-190 cycles per part, -380 per tile**, plus -10 instructions per K call and -5 per V call
(-160 U).  Counted conservatively against the measured 2,300 cycles of chain samples per tile:
the design budgets **-660 cycles per gemm0 tile** (the measured chain time is 2,300 because the
XU conversion and the vote are also issue-limited at 4 warps per SMSP; only the serialisation part
is removed).

### 3.2 [45b] Two register sets, software-pipelined spans (same helper)

With the branch gone, a span body is straight-line; issue span s+1's payload `LDS.128` x2 (FP8) /
`LDS.64` x2 (FP4) into the *other* register set before span s's decode and `STS.128` x4, so the
next load never WARs on registers the previous stores still read and each span's LDS latency is
covered by ~40 instructions of the previous span's decode.  +8 registers (FP8) / +4 (FP4).  Static
modules: explicit `packedA` / `packedB` in the unrolled spans; dynamic module: unroll the span
loop by 2 inside the format branch only if the hot loop stays <= 2,800 SASS (step-6 A1 bound),
else keep one set (the dyn module's short_sb is 0.73, the smallest).  Budget: -30..-40 cycles per
span -> **-240..-320 per gemm0 tile**, and the `LDS.128` short_sb samples (51 / 127) -> ~0.

### 3.3 [45c] Metadata in registers: flags, formats, tags (`mha.cu` K and V loaders)

- `kNeedsExpansion` / `vNeedsExpansion`: the value is warp-uniform and produced by the same warp
  one part / one iteration earlier (`mha.cu:2450-2457`, `:2966-2979`); keep it in a two-entry
  register rotation with the `CircIdx` instead of the lane-0 `STS.U8` + `__syncwarp` + `LDS.U8 ->
  ISETP -> BRA` after the DEPBAR (`:2629`, `:3310`).  Static modules: `if constexpr` on
  `MIXED_PAGE_STATIC_FORMAT > 0` (the flag is the constant `true`), so the flag store, its
  `__syncwarp` and the read disappear entirely.  Removes the 114 + 125 (fp8) short_sb samples:
  **-490 cycles per gemm0 tile**, -540 per gemm1 tile.
- `kFormats` / `vFormats`: pass the tag word by value (one `u32` = 4 tags, two rotated registers)
  to the expansion instead of by reference to smem; the dynamic module's `selectByIndex` then
  works on a register (`mhaUtils.cuh:1149-1152`; today `LDS.U8 [UR]`, 228 `wait` samples on mixed).
- Tags without shuffles: static modules **do not load tags at all** (dead work, section 2 fact 1):
  `mixedPageTagLane` / `broadcastMixedPageTags` and the `STS` of formats are skipped under
  `MIXED_PAGE_STATIC_FORMAT >= 0` (the a16 module too: `kA16CopyFastPath`, `mha.cu:94`).  The
  dynamic module has every lane load the four tags itself (4 predicated `LDG.E.U8` to the same
  addresses: one L1 wavefront each) into byte registers packed once, in place of 1 `LDG` + 4
  `SHFL.IDX` + 3 `PRMT` per copy call (`mhaUtils.cuh:205-227`).  Removes the `PRMT <- SHFL`
  samples (90 gemm0 / 151 gemm1 on fp8): **-390 cycles per gemm0 tile**.  Register cost: +2 (flag
  rotation folds into the tag word: A16 tag is 0; `needsExpansion = tagWord != 0`), +3 for the
  direct tags in the dyn module.

### 3.4 [45d] Metadata prefetch one stage deeper (dyn module; static modules need only the pages)

Today (`mha.cu:2360-2392`, `:2531-2536`; V `:2851-2867`, `:2881-2919`): `loadPages(p)` moves
`pageIdxNext -> pageIdx`, issues the tag `LDG` for `pageIdx` and the index `LDG` for `pageIdxNext`;
the K copy for tile t+1 consumes `pageIdx` and the tag one part body (~3 us) after the tag issue,
one tile after the index issue; the V side consumes them one V iteration (~2 us) later.  The
samples say the consumers still stall (section 2: 6.3 % fp8, 7.2 % mixed on `R2UR` / `LDC.64` /
`IADD3.X` / `IMAD.WIDE` / the tag-address `IADD3`).  Two changes, both register-only: (i) in the
static modules the tag load and its dependency on the index disappear with [45c], so the index has
a full tile of cover before its only consumer (the copy); (ii) in the dynamic module rotate three
index vectors (`pageIdx`, `pageIdxNext`, `pageIdxNext2`: +4 registers K, +2 V) and two tag words
(+1), so the tag `LDG` for tile t+2 is issued at tile t from indices loaded at tile t-1, and every
consumer is >= one tile behind its load.  Budget: the copy's page/tag long_sb (~1,340 cycles per
gemm0 tile fp8; ~2,500 mixed) -> **<= 200**.  This is the "pipeline depth" the step-6 followup
asked for, in the only pipeline where the samples show exposure.

### 3.5 [45e] `computeRowSum` quad broadcast with a constant mask (`mha.cu:798-805`)

`__shfl_sync(0xF << (laneId() / 4 * 4), rowSum[i], 0, 4)` has a lane-dependent mask, so ptxas
emits `WARPSYNC / MATCH.ANY / REDUX.OR / VOTEU / BRA.DIV` convergence code around every shuffle
(`0x11300-0x11410`, ~150 samples = 1.5 % of fp8, on gemm0's critical path).  The same broadcast as
`__shfl_sync(~0U, rowSum[i], laneId() & ~3U)` (full mask, quad-base source lane, width 32) is one
`SHFL.IDX`.  Values are identical (a broadcast of lane 0 of the quad either way).  Guarded under
`#if MIXED_COMPACT_TILE_LOOPS` so the sm120 and M32 modules keep their SASS.  **-650 cycles per
gemm0 tile.**

### 3.6 [45f] Prologue: page-list load not gated on `cacheSeqLen` (`mha.cu:2280-2305`, `:2385-2392`)

The per-CTA chain before the first K copy is `getCacheSeqLen` (`LDG`, `:2280`) -> `nbPages`
(`:2304`) -> `getPage` (`idxPage < nbPages ? LDG : BAD`, `mhaUtils.cuh` `getPage`) -> tag `LDG` ->
copy -> landing: three dependent round trips plus the landing, ~3-5 us of a ~31 us CTA lifetime
(the [42] host model already carries it as `kFixedCostInTiles = 1`, `mha.cu:3941`; at n = 5 a CTA
runs 3.2 tiles).  The page-list read does not need `nbPages`: `kvCachePageList[maxNbPagesPerSeq x
idxReq + idxPage]` is in bounds for every `idxPage < maxNbPagesPerSeq`, so load unconditionally
(predicated on `idxPage < maxNbPagesPerSeq`, a kernel parameter) and select `BAD` after both loads
land.  Removes one round trip per CTA (~0.7-1.5 us): **-2.5..-5 % of the wall on every mode**,
including a16.  Guarded (`kCompactTileLoops`); the `XQA_NB_SUB_SEQ` sweep of step 4 is *not*
re-run in this step (the [42] constants stay).

### 3.7 Budget: per gemm0 warp-tile (fp8) and the predicted wall

| item | gemm0 cycles per tile (of 16,700) | gemm1 cycles per tile (of 16,340 busy) | registers |
|---|---|---|---|
| [45a] vote once | -660 | -440 | +2 |
| [45b] two sets | -280 | -240 | +8 / +4 |
| [45c] flags / formats / tags in registers | -880 | -1,190 | +2 (+3 dyn) |
| [45d] prefetch depth | -1,140 | -1,700 | +5 (dyn) |
| [45e] rowSum mask | -650 | 0 | 0 |
| **sum** | **-3,610 (-21.6 %)** | -3,570 | +12..+20 |
| [45f] prologue | -0.7..-1.5 us per CTA lifetime (31 us) | | 0 |

After the cut gemm0 ~13,100 cycles, gemm1 ~12,800: gemm0 still paces, gemm1's slack shrinks from
20 % to ~2-5 % (its `xBar.produced` samples fall from 9.6 % to 3-6 %, which is the sign the
model is right).  Realisation: the removed stalls are one warp's, and the three co-resident warps
per SMSP take part of the freed issue slots (not_selected rises); step 5 -> 6 taught that a
count-only model over-predicts, so the design takes **0.7-0.9 of the modelled cut**.  Wall model:
`t = fill + active x (1 - r x 0.216)` with fill ~5 us on fp8/fp4/mixed (a16: DRAM-side, only [45c]
tags, [45d], [45e], [45f] apply, and its long_sb is DRAM landing at 67 % of peak).

| mode | today | modelled (r = 0.7 / 0.9, + [45f] -2.5..-5 %) | **predicted band** | accept | reject (revert the step) | target |
|---|---|---|---|---|---|---|
| fp8 | 113.5 (116.7 in the step-6 session) | 96.5 / 91.5 | **91-99** | <= 99 | > 116 (1.02x) or a16 > 87 | <= 94: **marginal** — needs r >= 0.85 and [45f] |
| fp4 | 101.5 | 86 / 81.5 | **82-90** | <= 90 | > 104 | <= 59: not this kernel (step 6 section 5.3) |
| mixed | 107.8 | 91 / 85.5 (its page/tag + format-LDS share is the largest) | **88-97** | <= 99 | > 110 | <= 101: **pass** |
| transport_a16 | 83.4 | 80 / 77 | **77-84** | 77-86 | > 87 | 135: pass |

The honest statement for the targets table: **mixed <= 101 is predicted with margin; fp8 <= 94 is
inside the band but at its good end** — it needs the gemm0 cut to land at >= 85 % of the model
*and* [45f]; fp8 in 95-99 with the A2 counters at their predicted values means the model was right
about the mechanism and the remaining 1-5 us is the fill / wave tail ([42]'s n and the per-CTA
fixed cost: a re-sweep of `XQA_NB_SUB_SEQ` on the new build is then the follow-up, not another
dependency cut).

## 4. Order of work and the register gate

One commit per item, measured by A1 after each; the order puts the cheapest, most certain items
first and the register-hungriest last: **[45c] -> [45e] -> [45f] -> [45d] -> [45a] -> [45b]**.
Gate after every commit: `cuobjdump -res-usage` REG <= 128, STACK 0, LDL 0, STL 0 on all four
sm90 q=4 modules.  If a commit breaks the gate, its fallback is: [45b] -> single set (drop the
item); [45a] -> recompute `sf2` per span (already the default) or keep the per-span vote in the
dyn module only; [45d] -> static modules only (no extra registers); [45c] -> keep the format word
in smem for the dyn module.  Nothing is traded for registers: 2 CTAs/SM is the step-4 lever and
outranks every item here.

## 5. Verification artifacts (mechanism first, stopwatch last)

- **A1 (SASS, `cuobjdump -sass` / `-res-usage`, loops delimited by back-edges as in step 5; all
  four sm90 q=4 modules):** `VOTE.ALL` 6 -> 2 (fp8, fp4: one per call site; dyn 2), `F2FP.F16.E4M3`
  in the expansion 4 per K call issued back-to-back before the vote, `MATCH.ANY` / `REDUX` /
  `BRA.DIV` in `computeRowSum` 0, `SHFL.IDX` in the tile loops 0 in every module (tags), `LDS.U8`
  at the flag offsets (`+0x1b030`, `+0x1b038` today) 0, `STS.U8` / `STS` flag and format stores 0
  in the static modules, `LDG.E.U8` tag loads 0 (static) / 4 per copy call (dyn), `LDGSTS` static
  counts unchanged (47 / 47 / 55 / 65), `LDS.128` / `LDS.64` 2 and `STS.128` 4 per span-call
  unchanged, `DEPBAR.LE SB0, 0x1` 3 per module unchanged; REG <= 128, STACK 0, LDL 0, STL 0; hot
  loops gemm0 part <= 1,200 SASS (1,114), gemm1 V <= 800 (726), dyn total hot <= 2,800.
- **A2 (ncu one launch, `--launch-skip 1 --launch-count 1`, the step-6 metric list plus
  `SourceCounters` on the `-lineinfo` build whose stripped SASS must be byte-identical to
  production):** `smsp__inst_executed.sum` fp8 36.0 -> 33.5-35 M, fp4 36.5 -> 34-35.5, mixed
  42.4 -> 39-41, a16 29.9 -> 28.5-29.9; warp-cycles per issued instruction fp8 8.13 -> <= 6.8,
  fp4 7.30 -> <= 6.2, mixed 6.56 -> <= 5.7; short_scoreboard fp8 2.01 -> <= 1.3, fp4 1.64 ->
  <= 1.1; long_scoreboard fp8 1.81 -> <= 1.2, mixed 1.34 -> <= 0.9; issue-active fp8 43 -> >= 50 %;
  no_instruction <= 0.8; `launch__shared_mem_per_block_dynamic` 115,456 B and occupancy limits
  2 / 2 unchanged.  PC sampling by the section-2 regions: `xBar.produced` TRYWAIT-branch samples
  9.6 % -> 3-6 % (gemm1's slack consumed — if it stays >= 8 %, gemm0 did not shorten and the
  item counts are read again); page/tag consumer long_sb in the copy 6.3 % -> <= 1 %; flag `ISETP`
  2.4 % -> 0; `PRMT <- SHFL` 2.4 % -> 0; `F2FP.E4M3` + fold `BRA` 4.0 % -> <= 1.2 %; the
  `MATCH.ANY` group 1.5 % -> 0; the three DEPBAR neighbourhoods stay at ~0 long_sb (the
  no-depth claim, re-checked on the new build).
- **A3 (correctness and byte-identity — shared files):** `tests/attention/run_xqa_mixed_page_transport.py`
  72/72 bit-exact on nkcut2 (default and `XQA_NB_SUB_SEQ=2`) and on ws-1, after every commit;
  ws-1: all eight sm120 `xqa_mha` modules (formats -1/0/1/2 x q=1/q=4) and `mla_sm120` stripped
  SASS byte-identical to a pristine `659eacfa` build made in the same session (the dyn q=1 pair may
  show the known ptxas pristine-vs-pristine variation: compare against two pristine builds before
  reading it as a leak); nkcut2: the sm90 q=1 `xqa_mha_sm90` and q=1 `xqa_mha` objects
  byte-identical; the sm90 q=4 a16 module changes (its tag pipeline is removed) and is accepted on
  its A1 counts.  `ptxas -v`: no C7507 anywhere (dataflow.md A4).
- **A4 (bench):** three locked rounds `--repeats 2 --trials 5` (2 x 117 us < 1.5 ms), pristine
  `659eacfa` checkout with its own JIT workspace interleaved (memory: a cached workspace rebuilds
  from the checkout's source and is not a baseline), q=1 control rows included; accept / reject
  per the section-3.7 table.

## 6. Do not build if

1. The pristine-tip PC sampling (A2 run on `659eacfa` before any edit) does not reproduce the
   section-2 shape: gemm1 `xBar.produced` wait >= 7 % of samples, DEPBAR neighbourhoods <= 0.2 %,
   page/tag consumer long_sb >= 4 % — if gemm1 has no slack the pacing role has changed and the
   budget is re-derived first.
2. Any item needs a `SharedMem` change, a third K/V buffer, 64 B K parts or 1 CTA/SM — rejected by
   section 1; do not revisit without new DEPBAR samples.
3. REG > 128 or STACK > 0 after [45c]+[45e]+[45f]: stop, report; do not build the register-costing
   items on a spilling base.
4. The sm120 SASS changes (guard leak) at any commit — fix the guard before any timing.
5. Another track has touched `mha.cu` `:2339-2545` / `:2851-3135` / `:3200-3345` or the two [44]
   helpers since `659eacfa` — re-derive section 2 on the merged tip.
6. Not in scope, whatever the numbers say: `csrc/xqa/mha_sm90.cu` (SPEC_DEC route), the FA3
   headers, the copy ownership / LDGSTS shape (A2/D6), unrolling the dyn module's span or page
   loops beyond [45b]'s factor 2 (step 3/4 fetch stalls), the [42] `nbSubSeqPerSeq` constants.
7. The step is judged against fp4 <= 59 — it is not this kernel's number (step 6 section 5.3).

## 7. Go / no-go

**Go** for [45] as a register-only package under the existing guards, in the section-4 order with
the register gate after every commit.  **No-go** for every smem-depth option (section 1): none
fits at 2 CTAs/SM except K3 x 64 B, and the wait it would deepen carries < 0.1 % of the samples.
Predicted: fp8 91-99 (target 94 marginal — reachable only with >= 85 % realisation plus [45f]),
fp4 82-90, mixed 88-97 (target 101 passes), a16 77-84.  Main risks: the realisation factor at
four warps per SMSP (not_selected growth), the 128-register cap (+12..+20 registers over
124-127), the dyn module's `wait`-dominated dispatch chains (24 % of mixed, untouched except the
format LDS), gemm1 becoming the pacing role once gemm0 gains more than ~20 % (then the V-side
items decide), and the co-tenant on nkcut2 (every A4 round interleaved with the pristine base).

Artefacts read for this design (no new remote jobs; scripts left at
`nkcut2:/tmp/mixedkv-wtS7-longsb.py`, `/tmp/mixedkv-wtS7-regions.py` — read-only over the step-6
CSVs and nvdis files): `nkcut2:/tmp/mixedkv-wtS6-a0/{v2-fmt-1,v2-fmt0,v2-fmt1,v2-fmt2}.nvdis`,
`{v2-fp8,v2-fp4,v2-mixed,new-transport_a16}.source.csv`, `buckets.py`, `stalls.py`.
