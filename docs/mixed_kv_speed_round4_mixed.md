# Round 4 — the mixed module's period: attribution, and the code-footprint / A16-issue lever (design)

Kernel: `csrc/xqa/mha_sm90.cu` at `354914cc` (production kernel state = lever [8],
`039ba5c7`; the r2p8 checkout on nkcut2 is byte-identical: md5
`1ad06789…` for `mha_sm90.cu`, `24fc3529…` for `mhaUtils.cuh`), the
`MIXED_KV_PERSISTENT` q=1 build, module `static_format_-1` ("mixed").  Bench shape
B=17, S=4096, 8 KV heads, GQA 4, D=128, q=1; the bench's mixed stream is
`page_format[p] = p % 3` (`benchmarks/bench_xqa_mixed_page_transport.py:32-33`):
pages cycle A16, FP8, FP4, so a 64-token tile (4 pages) is `[0 1 2 0]`, `[1 2 0 1]`
or `[2 0 1 2]` — **every tile carries at least one FP8 and one FP4 page and 1-2 A16
pages; no tile is single-format.**  Line references are into the worktree file
unless prefixed `r2p8:` (the nkcut2 checkout, whose line numbers differ by a few
lines in the IO region).

State (nkcut2 H200, locked 5x5 medians): transport_a16 78.8, fp8 67.8, fp4 60.5,
mixed 64.4 us.  Target mixed <= 62 (gap 2.4-2.6 us).

Method: attribution by reading (existing artefacts + trace-only / profiler-only
measurements on the existing r2p8 checkouts, no production edit, no build of
production code), then a data-flow / control-flow design, budgets by arithmetic,
one confirmation run after review.

## 0. Verdict in one paragraph

The premise "mixed period 1.44 vs fp8 1.38 vs fp4 1.15" is an artefact of the
round-2 body derivation (`79.5 = 3 x (13 T + 4.4) + 10` assumed the fp4 fill for
every mode; mixed's fill is ~1 us longer than fp8's).  Under [8] the mixed module
is **faster** than fp8 per tile in production: bench 64.4 vs 67.8, ncu 82.85 vs
87.62 us at ncu's locked clock (0.946x), executing 0.865x the warp-instructions
and moving 1.13x the DRAM bytes (172 vs 152 MB, 3.3 TB/s during the body — not
the 4.1 TB/s the a16 module reaches, so mixed is **not** at a bandwidth floor;
the byte-weighted-mean hypothesis predicts 1.72 us/tile and is refuted).  The
per-page dynamic dispatch costs 3 warp-uniform branches per warp-tile and no
predicated dual body (5 of 773 expansion instructions predicated) — the sm120
[40] pathology is absent.  What *is* mixed-specific in production: (i) the SM
is more latency-bound (eligible warps 1.38 vs 1.59, issue-active 57.8 vs 62.9 %,
gemm1 executes +15 % spin instructions), and (ii) **instruction-fetch stalls
are up 55 %** (`no_instruction` 2.90 vs 1.87 warps per issue-active cycle;
PC sampling: gemm0 — the pacing chain — spends 17.4 % of its samples fetching
vs 8.2 % in fp8, spread over its softmax / HGMMA body, not at a wait site; the
expansion bodies 43.5 vs 27.9 no_inst samples per M instructions).  The mixed
module's SASS is 5720 instructions (91.5 KB) with a hot footprint of ~62 KB
against fp8's 4696 (75 KB) / ~41 KB, because the K and V converter groups are two
textual copies (1248 + 1264 instructions, both formats' copy and expansion
bodies each) and the loader carries the TMA branch (536).  A natural experiment
calibrates the knee: the trace build (+7 KB of code, identical data flow) leaves
fp8 at production speed but slows mixed by +4 us of body (level-1) / +14 us
(level-2, which also spills): **no existing trace build is a valid instrument
for the mixed module's absolute period** (section 1.4).  The smallest sound
change that attacks the mixed-specific mechanism is a **code-footprint cut**:
one converter code path for both operands (operand-selected shared addresses,
as [8] did for the loaders) with the A16 pages' TMA boxes issued by the owning
converter warp (`expect_tx` + 2 boxes at copy-issue time) so the loader is the
static-module loop in every module.  Footprint 91.5 -> ~70 KB total, hot ~62 ->
~36 KB; predicted mixed **61.3 us (60.0-63.4)**, fp8 -0..-1, fp4 -0..-0.5, a16
unchanged (TMA branch stays in the a16 module).  Go, gated on `ptxas -v` 0
spill at 56 registers and on the mechanism check (PC-sampling gemm0 no_inst
share <= 10 %), not on the wall alone.

## 1. Attribution evidence

### 1.1 Production numbers (bench, ncu; r2p8 checkout, `/tmp/mixedkv-r2p8`)

Bench (locked 5x5, `/tmp/mixedkv-r2p8.bench.txt`, sessions r2p8 / r3p15 / r3pair):
mixed 64.4 / 64.2 / 65.0, fp8 67.8 / 67.6 / 68.1, fp4 60.5 / 60.4 / 60.6.
P = 132 control (one CTA per SM, r3p15 section 1.1): mixed 73.4, fp8 72.7, fp4
68.0 — lone-CTA tile times 0.96 / 0.95 / 0.89 us.

ncu, one launch each, `--launch-skip 4`, production objects rebuilt at 14:52
from the identical source (the workspace re-generated `build.ninja` under the
profiler; `md5sum` of the three kernel sources equals the worktree's; SASS
totals 5720 / 4696 as recorded in the [8] confirmation), clock-locked by ncu so
durations are ratios only (`/tmp/r4mixed_ncu_{mixed,fp8}.log`):

| metric | mixed | fp8 | mixed / fp8 |
|---|---:|---:|---:|
| `gpu__time_duration` | 82.85 us | 87.62 us | 0.946 (bench 0.950) |
| `smsp__inst_executed.sum` | 31.73 M | 36.70 M | 0.865 |
| per tile-CTA (8712) | 3642 warp-instr | 4213 | -571 |
| `sm__sass_inst_executed_op_ldgsts` | 185.7 K (21.3 / tile-CTA) | 348.2 K (40.0) | 0.53 — exactly 8/3 x (4+1) + 8/3 x (2+1) vs 8 x (4+1) |
| `sm__sass_inst_executed_op_branch` | 1.78 M (205 / tile-CTA) | 1.52 M (175) | +30 per tile-CTA (~4 per converter warp) |
| `dram__bytes_read.sum` | 175.8 MB | 156.2 MB | 1.13 (bench transport bytes 172.3 / 151.5 MB) |
| `dram__throughput` % peak (locked clock) | 51.0 | 43.5 | |
| `smsp__issue_active` % | 57.8 | 62.9 | |
| eligible warps / scheduler / cycle | 1.38 | 1.59 | |
| active warps / scheduler | 9.71 | 9.46 | |
| warp latency per issued instruction | 16.79 cyc | 15.03 | |
| smem wavefronts / bank conflicts | 7.93 M / 0.72 M | 9.58 M / 0.89 M | |
| stall (warps per issue-active cycle): no_instruction | **2.90** | **1.87** | +55 % |
| long_scoreboard | 4.75 | 4.23 | |
| barrier | 2.45 | 2.13 | |
| wait | 1.55 | 1.37 | |
| not_selected | 1.38 | 1.53 | |
| short_scoreboard | 1.01 | 1.20 | |
| math_pipe / mio / dispatch / imc | 0.41 / 0.12 / 0.19 / 0.08 | 0.48 / 0.20 / 0.23 / 0.05 | |

Reading: mixed does 13.5 % less SM work (one converter warp in three has an
A16 page and expands nothing; FP4 pages copy 6+3 instead of 12+3 `LDGSTS`)
and moves 13 % more bytes, and finishes only 5.4 % sooner.  The scheduler
picture is *less* issue-bound (57.8 % active, 1.38 eligible) and more
latency-bound; the one stall class that grows is instruction fetch.

PC sampling (`--section SourceCounters`, same launch recipe, SASS attributed
by kernel-relative address to roles via the lineinfo listing of the same
source, `/tmp/r2p8_ptx/li-1.nvdis` / `li1.nvdis`; `/tmp/r4mixed_pcs_*.csv`;
6486 / 6924 samples):

| role (mixed / fp8) | samples | no_inst samples | no_inst share | instructions executed (M) | no_inst per M instr |
|---|---:|---:|---:|---:|---:|
| gemm0 | 1432 / 1689 | **249 / 139** | **17.4 % / 8.2 %** | 15.2 / 15.8 | 16.4 / 8.8 |
| gemm1 | 1183 / 1349 | 101 / 84 | 8.5 % / 6.2 % | 18.4 / 16.0 | 5.5 / 5.3 |
| io (4 warps) | 805 / 820 | 59 / 47 | 7.3 % / 5.7 % | 10.8 / 10.7 | |
| K conv expand | 349 / 462 | 152 / 128 | 43.6 % / 27.7 % | 4.57 / 6.55 | 33 / 20 |
| V conv expand | 448 / 543 | 247 / 238 | 55 % / 44 % | 4.61 / 6.58 | 54 / 36 |
| K+V conv issue | 384 / 484 | 49 / 52 | 12.8 % / 10.7 % | 3.98 / 6.20 | |
| total | 6486 / 6924 | 999 / 840 | 15.4 % / 12.1 % | 63.7 / 66.9 | |

- gemm0's no_inst samples in the mixed module sit on `FMNMX` / `ISETP` / `ULEA`
  / `WARPGROUP.ARRIVE` / `BRA` throughout its softmax and HGMMA body (top PCs
  `04900 04850 04300 04880 04d80 04400`, 11-18 samples each); in fp8 they
  concentrate on the wait site (`040c0 S2UR` 36, `04180 SYNCS.ARRIVE` 16) — a
  spin-loop refetch, not a miss pattern.  gemm0 is the pacing dependent chain
  (P0.3, r3p15 section 1.2); +9 points of its time on fetch is ~0.13 us per
  1.5 us tile.
- gemm1 executes +2.4 M instructions in mixed (spin in `try_wait` loops): it
  waits longer for V / X, consistent with a longer gemm0 chain feeding it.
- The converters' expansion bodies are straight-line code; 44-55 % no_inst
  there is fetch, not control flow.  It is off the exposed path (gemm0 waits
  0.2 us per tile on K in every trace, and the K lead is >= 3 us, 1.3) and
  counts only through SM fetch pressure.

### 1.2 SASS footprint (lineinfo listings, flags identical to the production build)

Role totals (`/tmp/r4mixed_sass.py` over `li-1` / `li1` / `li2`; static counts):

| role | mixed | fp8 | fp4 | notes |
|---|---:|---:|---:|---|
| gemm0 | 282 | 282 | 281 | identical |
| gemm1 | 455 | 455 | 455 | identical |
| io (4 warps) | 1820 | 1707 | 1705 | mixed: TMA loop 536 (record LDS x3, 4 tag decodes, `arrive_tx`, up to 8 `UTMALDG` with 64-bit address chains); static loop 127 |
| K converters | **1248** | 801 | 712 | mixed: copy-issue 327 (fp8 133 + fp4 66 + scales 67 + common), expansion 773 (fp8 365, fp4 374, bad 21, a16 3) |
| V converters | **1264** | 799 | 710 | textual copy of the K block |
| other | 651 | 652 | 649 | prologue scan, merge warp, epilogue |
| total (production `cuobjdump`) | **5720 = 91.5 KB** | 4696 = 75.1 KB | ~4600 | |
| hot per SM (loop bodies of all five groups) | ~3850 = 62 KB | ~2540 = 41 KB | ~2400 = 38 KB | gemm0 282 + gemm1 455 + io loop 600 / 200 + K conv + V conv |

Dispatch cost in the mixed module: `BRA` 43 vs 33 (fp8) in the K converter
block; the format decision is three warp-uniform branches per warp-tile
(`issueCompressedPageCopies` :3158 bad/A16 and :3170 fp8/fp4;
`expandPackedStage` :3470 A16, :3485 bad, :3505 fp8/fp4) and the executed
bodies are the static modules' bodies (fp8 365 vs 361, fp4 374 vs 371;
predicated instructions 5 of 773).  Both fold branches are compiled in every
module.  The [40]-class cost (unrolled per-page loops with predicated
variants of every body) does not exist here.  Register pressure: the mixed
converter is at the 56-register cliff already (`foldMultiplier` :3412
recomputes the fold per tile because holding both formats' folds spilled 8 B).

### 1.3 Trace-only runs this round (r2p8-trace and r3pair-trace2 checkouts, `--modes mixed fp8 --launches 3`)

`/tmp/r4mixed_trace_r2p8.log` (level-1 build, CTA-0 window tiles 0-7, per-CTA
`ctarec`), `/tmp/r4mixed_trace_r3pair2.log` (level-2 build with per-role
segment accumulators and `%warpid`), parsed with `benchmarks/parse_xqa_trace.py`
and `/tmp/r4mixed_{gaps,smid,pair_split}.py` (local copies under `/tmp/`).

Level-1 build, same session:

| | mixed | fp8 |
|---|---|---|
| fill (start -> first K ready), median | 9.0 / 9.0 / 9.7 | 8.1 / 8.4 / 8.3 |
| body median (fast / slow member by SM pair) | 56.3 (55.6 / 57.2, **delta 1.5-1.7**) | 52.5 (47.2-47.6 / 57.5-57.7, **delta 10.1-10.7**) |
| body per tile fast / slow | 1.68 / 1.73 | 1.43 / 1.75 |
| end median / max (excl. co-tenant) | 68.1 / 73.4 | 64.4 / ~71 |
| CTA 0 cadence tiles 2-7: gemm0 / gemm1 / K loader / K conv | 1.459 / 1.424 / 1.657 / 1.575 | 1.258 / 1.263 / 1.413 / 1.256 |
| gemm0 K-wait -> mma / -> smax / -> xarr | 0.566 / 0.222 / 0.261 | 0.527 / 0.246 / 0.282 |
| gemm1 vwait -> xwait / -> rs / -> mma | 0.348 / 0.201 / 0.448 | 0.448 / 0.169 / 0.371 |
| K conv expand (done - ready), V | 0.378 (warp 0 alternates A16/FP8/FP4) / 0.543 | 0.455 / 0.519 |
| loader own work (iss - start) K / V | **0.294 / 0.324** (2.67 TMA boxes, elected lane) | 0.063 / 0.086 |
| gemm0 idle on K: kwait(t) - xarr(t-1), cycles | 237-2682, median ~400 (0.2 us) | 214-1127, median ~300 |
| TMA lead: g0 kwait(t) - kl_iss(t) | 5900-7000 cyc (3.0-3.5 us) | n/a |

Level-2 build (accumulators over all 264 CTAs, ns per tile, fast / slow slot):
mixed gemm0 341+763+392+388 = 1883 / 1903, gemm1 1928 / 1969, K loader
start 905 iss **999** / 949 / 994, K conv issued 814 landed 160 ready 93 expand
794 / 836, V conv 1085 + 832; body 70.6 / 71.4 (2.14 / 2.17 per tile).  fp8
same build: gemm0 1515 / 1736, K loader 1165 + 322, K conv 618 / 210 / 132 /
541 (slow 687), body 56.7 / 64.6 (1.72 / 1.96).

What the traces establish regardless of build validity (ratios inside one
launch): (a) in the mixed module the two co-resident CTAs are **symmetric**
(delta 0.7-1.7 us of 56-71) while fp8 shows the round-3 slot split (8-10.6 us):
the slot-priority penalty appears only when the SM is issue-saturated, and mixed
is not; (b) gemm0 is not starved on K (0.2 us idle per tile, same as fp8) and
the A16 TMA bytes are issued >= 3 us before gemm0 needs them: **the TMA landing
is not exposed**; (c) every role period is equal (closed loop); the loader's
serial TMA issue is 0.3 us (CTA 0 window) to 1.0 us (all tiles, incl. chunk
fills) per tile — inside its 0.9 us stage-release slack, so not pacing.

### 1.4 The trace build is not a valid instrument for the mixed module

Production ordering: mixed faster than fp8 by 3.4 us (bench) / 4.8 us (ncu).
Level-1 trace build: mixed *slower* by 3.7 us (end median) / 3.8 (body); level-2:
slower by 10 us.  fp8's trace body (52.5, slow 57.6) reproduces its production
slow-member body (57.1); mixed's does not (trace 56.3-57.2 vs production
64.4 - 9.3 - 2.8 = 52.3 for the slowest CTAs).  Build facts: level-1 modules
STACK 136 / STL 28 / LDL 0 for both fp8 and mixed (printf frames; no reload
spill); level-2 mixed STACK 216 / **LDL 15** / STL 52 (reload spills; fp8
level-2: LDL 0).  Code size: mixed 5720 -> 6160 (level-1, +7.0 KB) -> 6488
(level-2); fp8 4696 -> 5112 (+6.7 KB) -> 5392.  Same data flow, same
instruction counts per tile within a few per cent, same stamps in both modules:
the only variable that separates the two modules' response is where each sits
relative to an instruction-fetch capacity knee.  Consequences: (i) the
"mixed 1.44 us" premise had no valid trace behind it (the [8] design said so:
"body-derived; no same-launch trace exists"); (ii) every mixed-module trace
period in `mixed_kv_speed_round3_fill.md` (firstk 10.05, body 60.7 -> 71.2)
carries this bias; (iii) this lever's confirmation must use a **ctarec-only
trace build** (`MIXED_KV_TRACE` with the 16 per-tile stamps compiled out; 4
`%globaltimer` stores per CTA) and the production PC-sampling recipe of 1.1,
and the ctarec-only build must first reproduce the production ordering
(mixed end-max < fp8 end-max) before its fill / body numbers are accepted.

### 1.5 The candidates of the task, answered

| candidate | finding | consequence |
|---|---|---|
| per-page dynamic dispatch (branch per page, both bodies resident, register pressure) | 3 warp-uniform branches per warp-tile, no predicated dual body, executed bodies = static bodies (+4/+3 instr); register pressure is at the cliff but 0 spill; **both bodies resident in two textual copies = +20 KB of hot code**, fetch stalls +55 % SM-wide, gemm0 fetch share 17 % vs 8 % | the code footprint is the mixed-specific cost, not the branches |
| A16 tiles' TMA by the loader vs cp.async by the converters (two landing paths into one `kBar`) | loader issue 0.3-1.0 us per tile inside its 0.9 us slack; TMA lead >= 3 us; gemm0 idle 0.2 us per tile as in fp8 | not on the period; ~0.5-1 us on the fill (a16-branch preamble + start burst) |
| metadata format tags read per page | one `LDS.U8` per warp per tile at copy issue + 4 tag decodes in the loader; fill reads `page_format` in every module (R4 unmerged) | negligible |
| X ring / colMax | gemm0 / gemm1 SASS identical across modules (282 / 455) | none |
| A16 tile period / byte-weighted mean | mixed 172 MB in a ~52 us body = 3.3 TB/s vs a16's 4.1; weighted mean would be 1/3 x 2.4 + 2/3 x 1.38 = 1.72 us/tile vs measured <= 1.585 | mixed is not at its bandwidth floor (body floor 172 / 4.1 = 42 us) |

Two-rate / slot model for mixed, production: fill 9.3 (trace 9.0-9.7 in both
builds vs fp8 8.1-8.9, i.e. +0.9; production fp8 fill 8.5), tail 2.8; body
= 64.4 - 9.3 - 2.8 = 52.3 -> T = 1.585 us/tile for the slowest CTAs.  The trace
says the pair is symmetric in mixed, so T_f ~ T_s ~ 1.585 (fp8: 1.41 / 1.73).
Target 62 <=> 33 T <= 49.9 <=> T <= 1.512 (-4.6 %), or fill -2.6, or a mix.

## 2. Current data flow and control flow of the touched roles (as written at 354914cc)

- **Loader, mixed / a16 modules** (IO warps 0 / 1, :2159-2244): per tile `g`:
  record `LDS` x3 (pages `uint4`, formats, head) :2170-2172 ->
  `stageBar[stage].consumed.arrive_and_wait()` :2176 -> elected lane: 4 tag
  decodes, `nbA16` :2190-2205 -> `arrive_tx(produced, nbA16 x 2 x 2 KB, 32)`
  :2212 -> up to 8 `tma::loadAsync(dst, tensorMap, {partElems x p, head, 0,
  page}, produced)` :2214-2226 -> `__syncwarp` -> chunk refill at `g + 4`
  :2229-2234.  Static modules (:2245-2276): `arrive_and_wait` + refill only.
  Barrier init (:1352-1372): `produced` count = 128 GEMM + 128 converters +
  (32 loader iff `mixedLoaderTma` :156); `consumed` = 128 + 32.
- **K converters** (z = 3, :2633-2712): per warp `w` = page `w`; lane
  constants `kLane` (:2659, `makeExpandLane` :578-604: a16 / fp8 / fp4 / scale
  offsets) and `kGlobals` (:2660, `ExpandScales` :561-574: 4 floats + 2 bools);
  `issueKCopies(t)` (:2661-2675): chunk parity wait at `t % 16 == 0`,
  `kBar[t%3].consumed.wait_parity(toParity<3>(t))`, `issueCompressedPageCopies`
  (:3137-3186: tag `LDS.U8` from the record; A16 / bad -> return; else page +
  head `LDS`, 12 (FP8) or 6 (FP4) `LDGSTS.128` into the packed landing + 1 x
  8 B scale `LDGSTS` for lanes < 16).  Loop (:2680-2712): `waitGroup<1>`,
  `__syncwarp`, `expandPackedStage` (:3467-3627; A16 -> `__syncwarp` return;
  bad -> zero rows; FP8 / FP4 bodies, fold / two-multiply variants), `fence.
  proxy.async`, `kBar.produced.arrive()`, tag rotate, `issueKCopies(t+2)`,
  `commitGroup`.
- **V converters** (z = 4, :2713-2760): textual mirror with `vBar`, `vScales`,
  `vBuf`, `vMeta = 1`, `fp8VGlobalScale / fp4VGlobalScale`, stamps 14 / 15.
- **gemm0 / gemm1**: unchanged by this design (`kBar.produced.arrive_and_wait`
  :1515; `vBar` :1761).

## 3. New data flow and control flow

Scope: `ENABLE_MIXED_KV_CACHE && !SPEC_DEC` sm90 kernel, all four q=1
modules compile the same source; the a16 module keeps its loader TMA path
(`mixedLoaderTma := (MIXED_PAGE_STATIC_FORMAT == 0)`) and must stay SASS-
byte-identical; the compressed and mixed modules get one converter code path.

### 3.1 One converter code path for both operands (footprint cut)

    z in {3, 4}:  operand = z - 3
      stageBase   = operand ? cvta(&smem.vBufs[0]) : cvta(&smem.k[0])          (shared-window u32)
      stageBar    = operand ? &smem.vBar[0] : &smem.kBar[0]                    (CtaBarrierPair*, stride sizeof)
      metaReady   = operand ? &smem.vMetaReady[0] : &smem.kMetaReady[0]
      scalesBase  = operand ? cvta(&smem.vScales[0]) : cvta(&smem.kScales[0])
      recBase     = tileRecordAddr(smem, operand, 0)                            (:786; + (g % 32) * 32)
      globals     = operand ? {v_global_scale x2} : {k_global_scale x2}        (2 floats; folds derived per tile)
      lane        = makeExpandLane(warpIdx.x)                                   (offsets are stage-relative already)
      tensorMap   = operand ? &tensorMapVLLMV : &tensorMapVLLMK                 (mixed module only)
      loop over g exactly as :2680-2712 with these bases; stamps slot 12 + 2 x operand / 13 + 2 x operand

Stage `s` part `p` of K is `smem.k[2s + p]`, of V `smem.vBufs[s][p]`; both are
64 rows x 128 B per part, 16 KB per stage (the loader already relies on this,
`static_assert` :2164-2166), so one `stageBase + s * 16384 + p * 8192` formula
serves both operands.  `issueCompressedPageCopies` takes `isK` today (:3139
selects `k_payload` / `v_payload`, `k_scales` / `v_scales`); it becomes a
runtime `operand` selecting the two pointers once per warp into registers (2 x
64-bit) — no struct reference is selected (A6 code-shape rule).  The per-tile
loop body is textually one; ptxas emits it once.

### 3.2 A16 pages by the owning converter warp (mixed module)

    issue(g), warp w, after the chunk-parity wait and consumed[s].wait_parity(toParity<3>(g)):
      tag = LDS.U8 rec(g) + 16 + w
      if tag == bad or (static module):           as today
      if tag == A16 (mixed module):
          page = LDS.32 rec(g) + 4w ; head = LDS.32 rec(g) + 28
          if elect: produced[s].expect_tx(2 x 16 x 128 B = 4096)         (new MBarrier::expect_tx, no arrive)
                    for p in {0,1}: tma::loadAsync(stageBase + s*16384 + p*8192 + w*2048,
                                                   tensorMap, {64 p, head, 0, page}, produced[s])
          __syncwarp ; return tag
      else:                                        compressed copies as today (:3160-3185)
    iteration g (unchanged): waitGroup<1> ; __syncwarp ; expand (A16: return) ; fence ; produced[s].arrive()

The loader in the mixed module runs the static loop (:2245-2276): fills and
`consumed` arrives only.  `produced` count = 128 + 128 = 256 in fp8 / fp4 /
mixed; the a16 module keeps 288 (loader `arrive_tx`).

## 4. Barrier, parity and ownership invariants

Existing: D1-D6, C1-C7 (dataflow doc), C8-C13 ([8]).  Added:

- **C14 (tx registration is phase-correct and never premature).**  Warp `w`
  issues `expect_tx(4096)` on `produced[s]` at `issue(g)`, after
  `consumed[s].wait_parity(toParity<3>(g))` — the completion of consumed for
  tile `g - 3`, which requires gemm0's arrive after its `produced` wait for
  `g - 3`; hence `produced[s]`'s phase for `g - 3` is complete and the barrier
  is in tile `g`'s phase.  Other warps' 32-lane arrivals for tile `g` may
  precede this `expect_tx` (a warp may run up to two iterations ahead, bounded
  by its own `consumed` waits), leaving the phase with pending arrivals > 0 and
  tx 0; the phase cannot complete before warp `w`'s own 32 arrivals, which are
  program-ordered after its `expect_tx` (iteration `g` vs `g - 2`), so tx is
  positive or zero-after-landing when the last arrival comes.  `complete_tx`
  cannot precede its `expect_tx`: same lane, program order, same barrier.
  This is the loader's existing `arrive_tx` argument (:2212) moved to the
  converter's issue point, with arrive and expect split.
- **C15 (page ownership, A16).**  Page `w`'s rows in both parts are written
  only by warp `w`'s two TMA boxes; no converter `STS` touches them (every
  warp's `ExpandLane` offsets lie in its own page's rows, :578-604), and no
  `cp.async` is issued for an A16 page, so the in-place packed landing of the
  compressed pages (part 1 rows of *their* pages) never overlaps.  D3 / D6
  unchanged.
- **C16 (WAR on the stage).**  TMA writes for tile `g` into stage `s` are
  ordered after gemm0's reads of tile `g - 3` by `consumed[s]` completion ->
  `wait_parity` (acquire) -> issue, the edge the converters already use for
  `cp.async` (C8: the waiter is never two phases behind).  The loader used the
  same edge through a token wait; no new ordering is introduced.
- **C17 (proxy order).**  TMA (async proxy) completion is tracked by the
  transaction count; the converters' generic `STS` are fenced by
  `fence.proxy.async` before `arrive` (existing).  gemm0's wgmma reads are
  ordered by its `wait_group<0>` before `consumed.arrive` (existing).
- **C4 (counts).**  `produced` = 256 in fp8 / fp4 / mixed (was 288 in mixed),
  288 in a16; `consumed` = 160 everywhere (unchanged); `kMetaReady` /
  `vMetaReady` unchanged (loader arrives, converters wait).  `mixedProducedExtra`
  :1352 uses `mixedLoaderTma` with the new definition.
- **C6 (issue budget).**  Per A16 warp-tile the converter executes ~30
  instructions (3 `LDS`, elect, `expect_tx`, two 64-bit address + coordinate
  chains + `UTMALDG`) in place of ~150-200 loader instructions per tile; the
  compressed paths are unchanged.
- **C9 / C10 (records).**  Readers of record `g` are unchanged in time: the
  converter reads pages / tag / head at `issue(g)` after its `kMetaReady` wait
  (A16 warps now read page + head there too, as compressed warps already do).
- **C12 / Q / merge**: untouched.

## 5. Budgets

**Registers** (`__launch_bounds__(640, 2)`, `setmaxnreg` 40 / 40 / 40 / 56 / 56,
pool 29696 of 30720): converter live set today: `ExpandLane` 4, `ExpandScales`
6, tags 1, loop / stage / addresses ~10, expansion temporaries ~30 (P0.4: 188
SASS per lane-tile at 56 with 0 spill in the static modules; mixed at the
cliff, `foldMultiplier` :3412).  Change: +5 operand-selected bases (stage,
barrier, metaReady, scales, record) and +2 payload / scale pointers (64-bit,
selected once per warp) = +9; -4 by shrinking `ExpandScales` to the two
globals (fold multipliers and `FoldOk` re-derived per tile: 2 `FMUL` + 2 `FSETP`
= 4 instructions per warp-tile, the mixed module does one of them already);
the tensor-map address is a constant-bank operand.  Net +5 against a role at
its cliff: **the design does not assert it fits; `ptxas -v` decides** (do-not-
build item 1).  Fallbacks, in order: keep the scale-slot base as
`stageBase`-relative immediates (the `TileScales` arrays sit at fixed offsets
from the stage arrays for each operand -> one base fewer), and IO at 48 with
GEMMs at 40 does not help the converters (the pool is exactly 30720 either way);
if the converter needs 64 the layout is [15]'s and this lever is folded there.
gemm0 / gemm1 / IO: unchanged live sets (IO loses the TMA branch in the mixed
module).

**Shared memory**: unchanged, `sizeof(SharedMem)` 113 664 B (2 CTAs/SM); the
trace build's ctarec-only variant adds 0 B.

**Code footprint** (from the 1.2 counts): converters 1248 + 1264 -> ~1270 (one
body incl. the A16 issue branch ~40); io 1820 -> ~1430 (TMA loop 536 -> static
loop 127 + walker); total **5720 -> ~4350 instructions = 91.5 -> ~70 KB**; hot
per SM 62 -> ~36 KB (below fp8's 41 KB today).  fp8 / fp4 modules: 4696 ->
~3900 (75 -> 62 KB), hot 41 -> 28 KB.

**Issue** per tile-CTA (mixed): loader -150..-200, converters +30 x 1.33 A16
warps x 2 operands = +80 -> net -70..-120 of 3642 (-2..-3 %); DRAM bytes,
`LDGSTS`, `UTMALDG` count per tile unchanged (2.67 boxes per operand per tile,
now issued by 1-2 warps in parallel instead of one lane serially).

**Time** per A16 warp-tile: 2 boxes x 100-200 ns of issue on the elected lane
(A1) = 0.2-0.4 us, on a warp that otherwise idles for the tile; the compressed
warps' copy issue (~0.3 us) runs in parallel, so the tile's issue phase does
not lengthen.  TMA lead to gemm0 grows by the loader's serial delay (0.3-1.0
us) — irrelevant while the lead is >= 3 us.

## 6. Predicted periods and wall (two-rate / slot model)

Model: `wall = fill + 33 x T_s + tail`, `T_s` the slower co-resident member's
tile time; mixed today: 9.3 + 33 x 1.585 + 2.8 = 64.4, symmetric pair (1.3).

Period.  The pacing dependent chain is gemm0's (K-wait -> HGMMA -> colMax /
softmax -> X store: 1.05 us of work + 0.2 us hand-off in the level-1 trace;
gemm1's chain is 1.0 + 0.35).  gemm0 spends 17.4 % of its samples on
instruction fetch in the mixed module vs 8.2 % in fp8 (1.1) = 0.27 vs 0.13 us
per tile; the change brings the hot footprint below fp8's (36 vs 41 KB), so the
mixed gemm0 fetch share is predicted to fall to fp8's: **-0.13 us per tile on
gemm0's chain**.  gemm1's share 8.5 -> 6.2 % = -0.04.  The period follows the
longer chain: -0.04 (if gemm1 becomes the bound) to -0.13 us -> T_s 1.585 ->
1.455-1.545.  Issue relief (-2..-3 % of a 57.8 %-active scheduler): <= -0.02,
not counted.  The slot split is not expected to reappear (issue load falls).

Fill.  Mixed 9.3 vs fp8 8.5: the loader's a16-branch preamble (r3fill: ~1.5 us
from the `__syncthreads` to the first TMA issue on a16) and the serial TMA of
tiles 0-2 leave the K loader's first `consumed` arrive late; with the static
loop and A16 boxes issued at the converters' prologue `issue(0..1)` the mixed
fill is predicted at fp8's: **-0.5..-0.8 us** (RT4 does not shrink with bytes,
r3fill; the A16 bytes join the LDGSTS burst).

| mode | today | period change | fill change | predicted | target |
|---|---:|---|---|---:|---|
| mixed | 64.4 | 33 x (-0.04..-0.13) = -1.3..-4.3 | -0.5..-0.8 | **61.3** (60.0-63.4; central -0.08/tile, -0.6 fill) | <= 62 |
| fp8 | 67.8 | fetch share 8.2 % already; footprint 41 -> 28 KB: 0..-0.03/tile | 0 | 66.8-67.8 | (not this lever) |
| fp4 | 60.5 | 0..-0.02/tile | 0 | 59.8-60.5 | |
| a16 | 78.8 | unchanged SASS | 0 | 78.8 | parity |

Honest statement if the mechanism does not deliver: mixed's structure-level
floor at [8] with fill 9.3 and tail 2.8 is 33 T + 12.1; T cannot go below the
lone-CTA chain 0.96 us x the pair stretch, and the shared levers ([15]+[7]:
mixed ~62; + fill cut ~58) remain the route to and below 62; mixed is **not** at
a bandwidth floor (42 us body at 4.1 TB/s).

## 7. Verification artifacts, accept / reject

Build (each module a16 / fp8 / fp4 / mixed, `ptxas -v` with the ninja flags via
`/tmp/main_ptx/ninja_flags.py`, `cuobjdump -sass`):

1. No C7507; 0 bytes stack, 0 spill stores / loads; `USETMAXREG` 2 (0x28 /
   0x38); `LDL` = `STL` = 0; REG 48; `sizeof(SharedMem)` 113 664.
2. **a16 module SASS byte-identical** to 354914cc's (`cmp` of `cuobjdump -sass`
   after stripping addresses) — the loader TMA path is untouched there.
3. `UTMALDG`: a16 8, **mixed 4** (K / V x 2 parts, converter block), fp8 / fp4 0;
   `mbarrier.expect_tx` (no arrive) sites: 2 in mixed (`SYNCS.ARRIVE.TRANS64`
   variants counted per role), 0 elsewhere; `LDGSTS` 42 / 30 / 18 unchanged.
4. Footprint: total SASS mixed <= 4500 (from 5720), fp8 <= 4000 (from 4696);
   per-role split (`/tmp/r2p8_role_split.py` with the new line ranges): one
   converter region only (z == 3 and z == 4 branch to the same body), gemm0 /
   gemm1 counts unchanged (282 / 455; PHASECHK 8 / 17, ARRIVE 11 / 13).
5. Barrier init: `kBar` / `vBar` `produced` count 256 in fp8 / fp4 / mixed,
   288 in a16 (read from the SASS immediates or a `static_assert` on
   `mixedProducedExtra`).

Conformance: `python tests/attention/run_xqa_mixed_page_transport.py` (60 cases;
the mixed cases already contain tiles with A16 pages owned by a warp whose
previous tile was compressed and vice versa, tail tiles with bad tags in every
warp position, P = 1 / 3 / 5 and T < P) -> 60 / 60.

Mechanism (production build, before the bench; the accept gate of the lever):
`ncu --section SourceCounters --section WarpStateStats` with the 1.1 recipe
(`/tmp/r4mixed_pcs.sh`, `/tmp/r4mixed_pcs_roles.py` with the new lineinfo
listing): mixed gemm0 no_inst share **<= 10 %** (17.4 today; fp8 8.2), total
no_inst per issue-active cycle <= 2.1 (2.90 / 1.87), `smsp__inst_executed`
30.5-31.5 M (-0.2..-1.2 M for the loader), `dram__bytes_read` 176 +- 1 MB.
Reject and re-attribute if gemm0's share stays > 14 % with the footprint cut
confirmed by item 4 (then fetch is not the mechanism and the wall will not
move).

Timing (locked, `flock /tmp/mixedkv-gpu0.lock bash /home/bigboi/mixedkv_remote_run.sh
<checkout> r4mixed sm90 transport_a16 fp8 fp4 mixed`, 5 x 5, q=1 rows; q=4 rows
unchanged within spread; a same-session [8] control interleaved):

| mode | accept | predicted | reject if |
|---|---|---|---|
| mixed | median <= 62.0 (target) ; lever gate <= 63.4 | 61.3 | > 63.4 |
| fp8 | 66.8-68.1 | 67.3 | > 68.4 |
| fp4 | 59.8-60.8 | 60.2 | > 61.0 |
| a16 | 78.5-79.3 | 78.8 | > 80 |

Trace (ctarec-only build, section 1.4; `xqa_mixed_trace_once.py --modes mixed
fp8 --launches 3`): first the validity check — mixed end-max < fp8 end-max as
in production; then mixed fill median <= 8.7 (9.0-9.7 today), body/tile
(slowest decile) <= 1.55, pair delta unchanged (<= 2 us).  If the ctarec-only
build still inverts the ordering, the per-tile stamps are not the cause and
the trace path is closed for mixed until re-attributed.

## 8. Do not build if

1. `ptxas -v` shows any spill in the converter role at 56 with the merged body
   and the `ExpandScales` shrink, and also with the scale-base fold-in (5): the
   converters then need [15]'s 64-register layout and this lever is folded
   into [15] rather than built alone.
2. The a16 module's SASS is not byte-identical (the loader change leaked into
   `MIXED_PAGE_STATIC_FORMAT == 0`).
3. The merged converter body exceeds the sum of today's bodies' executed path
   by more than 8 instructions per warp-tile (operand selection turned into
   per-tile selects instead of per-warp bases) — check the per-lane-tile SASS
   count with `xqa_sm90_converter_sass.py --paths`: fp8 expand 188 +- 2, fp4
   187 +- 2.
4. `sizeof(SharedMem)` moves or the occupancy calculator returns != 2.
5. Total mixed SASS after the change is > 4800 instructions (the footprint cut
   did not happen: e.g. ptxas duplicated the body by unswitching on
   `operand`) — the mechanism premise is void; stop before timing.
6. The mechanism gate (section 7) fails: gemm0 no_inst share > 14 % with the
   footprint confirmed.  Then the fetch reading was wrong, the wall prediction
   is withdrawn, and the honest mixed number stays 64.4 pending the shared
   levers.
7. The tx accounting cannot be expressed with a per-warp `expect_tx` (e.g. the
   barrier helper only offers `arrive.expect_tx`): an arrive at issue time
   would count the A16 warp twice per phase — never substitute it.
8. The conformance runner's mixed cases do not execute the A16-by-converter
   path on every warp position (add a case with `page_format = (p + k) % 3`
   for k in 0..2 if the per-warp coverage cannot be shown from the existing
   pattern).

## 9. Go / no-go

**Go**, as a code-footprint lever with the A16 issue moved to the owning
converter warp, gated on `ptxas -v` (item 1) and on the PC-sampling mechanism
check (section 7) before the wall is read.  Predicted mixed 61.3 us (60.0-63.4),
target 62 inside the band but not assured; fp8 / fp4 unchanged to -1 us; a16
byte-identical.  Recorded for the plan: (a) the trace build is not a valid
instrument for the mixed module (1.4) and the r3fill / r2p8 mixed trace rows
must be read with that caveat; (b) the "mixed period 1.44" premise is
withdrawn — mixed is 0.93x fp8 per tile in production and not bandwidth-bound;
(c) the slot-priority penalty is absent in the mixed module (symmetric pair),
so a pair pool would have gained nothing there; (d) the remaining gap after
this lever is the shared fill (9.3 vs 3.5 designed) and the consumer chain
([15] + [7]).

Artefacts (nkcut2): `/tmp/r4mixed_trace_r2p8.log`, `/tmp/r4mixed_trace_r3pair2.log`,
`/tmp/r4mixed_ncu_{mixed,fp8}.log`, `/tmp/r4mixed_pcs_{mixed,fp8}.{ncu-rep,csv,details}`,
`/tmp/r2p8_ptx/li{-1,1,2}.nvdis`, scripts `/tmp/r4mixed_{trace,ncu,pcs}.sh`,
`/tmp/r4mixed_{sass,pcs_roles}.py`; local `/tmp/r4mixed_{gaps,smid,pair_split}.py`.
