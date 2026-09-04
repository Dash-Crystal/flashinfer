# Round 4 — the mixed module's period: attribution, and the code-footprint lever (design, rev 2)

Kernel: `csrc/xqa/mha_sm90.cu` at `354914cc` (production kernel state = lever [8],
`039ba5c7`; the r2p8 checkout on nkcut2 compiles the identical kernel source —
`md5` `1ad06789…` for the worktree's `mha_sm90.cu` at 354914cc and 039ba5c7; the
r2p8 copy differs only in the IO-region line numbers, so rev-1 references
prefixed `r2p8:` and the `/tmp/r4mixed_*.py` role ranges are r2p8 lines; every
`:NNNN` below is a worktree line).  The `MIXED_KV_PERSISTENT` q=1 build, module
`static_format_-1` ("mixed").  Bench shape B=17, S=4096, 8 KV heads, GQA 4,
D=128, q=1; the bench's mixed stream is `page_format[p] = p % 3`
(`benchmarks/bench_xqa_mixed_page_transport.py:32-33`): pages cycle A16, FP8,
FP4, so a 64-token tile (4 pages) is `[0 1 2 0]`, `[1 2 0 1]` or `[2 0 1 2]` —
every tile carries at least one FP8 and one FP4 page and 1-2 A16 pages.

State (nkcut2 H200, locked 5x5 medians): transport_a16 78.8, fp8 67.8, fp4 60.5,
mixed 64.4 us.  Target mixed <= 62 (gap 2.4).

Method: attribution by reading (existing artefacts + profiler-only measurements
of the existing production objects in the r2p8 checkout — no production edit,
no build of production code), then a data-flow / control-flow design, budgets
by arithmetic, one confirmation run after review.

**Rev 2 changes.**  Every blocker and note of the rev-1 review is quoted and
answered in section 1.  The four cheap discriminators the review asked for were
run (PC sampling at P = 132 for mixed / fp8 / fp4, fp4 and a16 at P = 264, a16 at
P = 132; a line-position analysis of the existing samples).  The fill term, the
"symmetric pair" claim and the "register cliff" claim are withdrawn.  The design
is now one source shape for all four modules (converters own every page's
transport, the loader is the static loop everywhere), the a16 gate is a wall
band plus role counts instead of byte-identity, and the prediction is a
bounded band whose one unmeasured factor — the exposed fraction of the fetch
stall — is what the confirmation run measures.

## 0. Verdict in one paragraph

The mixed module's gemm0 — the pacing dependent chain — spends 17.4 % of its
time stalled on instruction fetch at 2 CTAs/SM (fp8 8.2 %, fp4 2.5 %, a16
2.6 %), and 9.2 % at 1 CTA/SM (fp8 4.8, fp4 2.5, a16 3.8).  Eight measured
points order strictly by the module's static hot code footprint (a16 ~26 KB,
fp4 37.7, fp8 40.6, mixed 61.6) at both occupancies, with a knee between 37.7
and 40.6 KB; the module with the highest memory load (a16, 3.6 TB/s) has the
lowest stalls, so the alternative "fetch latency follows DRAM / L2 load" is
refuted; 85 % of mixed gemm0's fetch-stall samples sit on the first instruction
of a 128 B line (3 of 249 after a branch): line-crossing fetch, not control
flow.  The stall's *exposure* on the tile time is the one thing the profiler
cannot give: at P = 132 the mixed-vs-fp8 excess is 0.042 us/tile of stall
against a measured tile-time difference of 0.011 us (<= 26 % exposed if fp8 is
the right baseline; the fp4 baseline leaves the fraction unconstrained).  The
design cuts the mixed module's hot footprint to ~36 KB — below fp4's 37.7 KB,
which sits at the floor — by compiling the K and V converter groups as one
code path (operand-selected shared bases, as [8] did for the loaders) and
letting the owning converter warp issue the A16 pages' two TMA boxes
(`expect_tx` + 2 boxes at copy-issue time), so the loader is the static loop in
every module and the a16 module's 536-instruction TMA loop is gone too.  The
same source compiles all four modules (module constants prune dead format
branches, as today); no module is byte-identical to today.  Converters are at
R37 under the 56 cap (measured), so the change (+4..+6 registers) fits without any
fallback.  Predicted mixed **64.4 - 7.9 r** us where r is the exposed fraction
of gemm0's fetch stall: 62.3 at the P = 132 calibration r = 0.26, 60.4 at
r = 0.5; fp8 67.8 - 3.2 r; fp4 / a16 unchanged.  **Go**, gated before any
timing on `ptxas -v` (0 spill, no C7507), on the SASS role counts (mixed hot
<= 37 KB by construction), and on the PC-sampling mechanism check (mixed gemm0
no_inst <= 4 %); the accept line is the target (mixed median <= 62.0), a pass
of the mechanism gate with a wall in (62.0, 63.2] is recorded as "r < 0.30" and
the kernel is not merged as a speed lever.

## 1. The rev-1 review, blocker by blocker

Judge verdict on rev 1 (`4b00e360`): approve = false, four blockers, six notes.
Each is quoted in full, then answered.

### 1.1 Blocker 1 — mechanism vs the lone-CTA data

> Mechanism not reconciled with the doc's own lone-CTA data. At P = 132 (r3p15
> section 1.1, cited in 1.1 of this doc) mixed runs 0.96 us/tile vs fp8 0.95 with
> the SAME 62 KB vs 41 KB hot footprints and the same per-SMSP role mix; a
> capacity/footprint explanation (LRU stack distance is set by the distinct hot
> lines, not by warp count) predicts the +0.13 us/tile gemm0 fetch penalty there
> too, and it is absent. The observed penalty is therefore occupancy-dependent
> (fetch bandwidth / co-tenant demand, or misses hidden under chain latency), and
> the extrapolation 'hot 36 KB < fp8's 41 KB => gemm0 fetch share falls to fp8's
> 8.2 %' rests on ONE calibration point (the trace natural experiment) plus a
> monotonic-footprint assumption. Cheap, allowed, profiler-only discriminators
> were not run: PC-sample the P = 132 mixed and fp8 launches (if mixed gemm0
> no_inst is ~8 % there, the mechanism is not footprint; if ~17 % yet the chain is
> unaffected, the -0.13 us/tile upper bound is not additive) and PC-sample fp4
> (38 KB hot) to test monotonicity. Without these, the -0.04..-0.13 us/tile period
> term is a correlation, not a verified mechanism, and building is guess-and-check.

Conceded and measured.  The discriminators were run on the existing production
objects (r2p8 checkout, `/tmp/mixedkv-r2p8`; `ncu --section SourceCounters
--section WarpStateStats -k regex:kernel_mha --launch-skip 4 --launch-count 1`,
`XQA_PERSISTENT_CTAS=132` for the lone-CTA launches; script
`/tmp/r4mixed2_pcs.sh`; SASS attributed to roles by kernel-relative address
through the lineinfo listings `/tmp/r2p8_ptx/li{0,1,2,-1}.nvdis` of the same
source with `/tmp/r4mixed_pcs_roles.py`; line-position analysis
`/tmp/r4mixed2_linepos.py`).  Results in section 2.  In the review's own
terms: at P = 132 mixed gemm0 no_inst is **9.2 %** against fp8's **4.8 %** —
neither "~8 % (not footprint)" nor "~17 % (not additive)": the excess persists
at one CTA per SM at about half its P = 264 size, and it orders by footprint
there too.  The monotonicity test passes with margin: fp4 (37.7 KB) is at the
floor (2.5 %) at both occupancies, and the added a16 control (~26 KB, 3.6 TB/s
of DRAM traffic — the highest memory load of the four modules) is at the floor
as well (2.6 / 3.8 %).  The review is right that a pure LRU-capacity model does
not predict an occupancy dependence of the *magnitude*: the absolute stall on
gemm0 per tile is 0.28 us at P = 264 vs 0.09 at P = 132 for mixed (0.14 vs
0.05 for fp8), a factor ~3 for the same distinct lines.  The design does not
rest on a cache model; it rests on the eight-point ordering (section 2.2): at
both occupancies the stall is at the floor for footprints <= 37.7 KB and
above it for >= 40.6 KB, and the design moves the mixed module to ~36 KB by
construction (section 6, gate 9.5).

What the P = 132 data do and do not say about exposure (the "additive" half of
the blocker): the mixed-minus-fp8 fetch excess at P = 132 is 4.4 points of
gemm0's time = 0.042 us per 0.96 us tile, while the measured lone-CTA tile
times differ by 0.011 us (73.4 vs 72.7 us walls over 66 tiles, common fill /
tail) — **at most 26 % of the excess is exposed** if fp8 is the right baseline
for "what mixed's chain would run at without the fetch excess".  Against fp4
(0.89 us/tile: 8 x (2+1) LDGSTS per tile-CTA, like mixed's 21) the difference
is 0.082 us with a fetch bound of 0.066, but fp8 itself is 0.071 slower than
fp4 with a fetch bound of only 0.024, so >= 0.047 us of the fp8-vs-fp4 gap is
non-fetch environment (bytes, copy issue), and the fp4 baseline cannot
separate mixed's fetch share from its 172 MB of traffic.  Hence r (exposed
fraction at P = 264) is bounded above by 0.26 under the fp8 baseline and
unconstrained under the fp4 baseline; the prediction (section 7) is written
as a function of r and the confirmation run measures it.  The rev-1 term
"-0.04..-0.13 us/tile, central -0.08" is withdrawn.

### 1.2 Blocker 2 — the fill term and the symmetric pair are taken from an invalid instrument

> The prediction's fill term and the 'symmetric pair' conclusion are taken from
> the level-1 trace build that section 1.4 itself declares invalid for the mixed
> module. Production mixed fill 9.3 exists only as the trace number (9.0-9.7); the
> derived body 52.3 / T = 1.585 and the -0.5..-0.8 us fill gain are circular on it
> (the r3fill fill chain shows the converters' issue(0) already waits
> kMetaReady[0], the same gate the loader's TMA sits behind, so at most the
> loader's tile-0 serial box issue ~0.3-0.5 us is removable). Dropping the fill
> term leaves the central prediction at ~61.9-62.0 with the target at the edge,
> not inside, and claim (c) 'slot-priority penalty absent in mixed / T_f ~ T_s' is
> unverified in production (round3_pair's mixed 1.310 / 1.609 split was
> body-derived, not measured).

Conceded on all three points.  (i) The fill term is dropped: the design counts
**0** for the fill.  The converters' `issue(0)` waits `kMetaReady[0]` (:2666)
exactly as the loader's tile-0 TMA does today (:2155-2156 then :2184), and
both then wait `consumed[0]`; the only fill-side difference is that today's
elected lane issues tile 0-2's boxes serially (~0.3 us per tile, `kl iss` in
the CTA-0 window) while 1-2 converter warps issue 2 boxes each in parallel —
a <= 0.3 us shift of the A16 landing that is on the fill's path only if the
A16 boxes land after the compressed pages' expansion; not counted.  (ii) No
production mixed fill or body number is used anywhere in rev 2: the
prediction is a wall delta (`64.4 - 33 x r x delta`) that needs neither.  The
"T = 1.585" and "fill 9.3" of rev 1 are struck; where a mixed period is needed
for a share-to-time conversion (section 2.3) it is bracketed as 1.59-1.61
(`(64.4 - fill - 2.8) / 33` with fill 8.5-9.3) and only the bound is used.
(iii) Claim (c) is withdrawn: "the pair is symmetric in mixed" is a
trace-build observation and stays unverified in production; the prediction
does not use it (a saving on gemm0's chain applies to the slower member
whatever the split).  The trace-only validity check of rev-1 1.4 is kept
(section 8) for the ctarec-only build.

### 1.3 Blocker 3 — a16 byte-identity vs a source-level converter merge

> Internal inconsistency in the gates: 7.2 and 8.2 require the a16 module's SASS
> to be byte-identical, but 3.1 makes one converter code path for both operands in
> source that 'all four q=1 modules compile'. The a16 module's converter block
> (two textual copies today) would necessarily change unless the merge is
> preprocessor-gated on MIXED_PAGE_STATIC_FORMAT == 0, leaving two source shapes
> of the converter block; the doc states neither, so either the gate or the code
> shape must be revised.

The code shape is revised to **one source shape for all four modules** and the
gate is revised accordingly.  The converter body is one text; the loader loop
is the static loop (:2245-2276) in every module; `mixedLoaderTma` (:156) and
`mixedProducedExtra`'s loader term (:1352-1356) are removed; the A16 TMA issue
lives in the converter's issue path behind the same runtime `format` test that
already selects FP8 / FP4 / bad there (:3149-3158), so the static compressed
modules (`MIXED_PAGE_STATIC_FORMAT` 1 / 2) prune it as dead code exactly as
they prune the other format's body today, and the a16 module (`== 0`) prunes
the compressed copies.  No preprocessor gate is added.  Consequently the a16
module's SASS **changes** (its loader loses the 536-instruction TMA loop; its
converters gain the A16 issue) and byte-identity cannot be the gate.  The a16
gate becomes: gemm0 / gemm1 SASS counts unchanged (278 / 451 in li0), `UTMALDG`
8 -> 8 (moved from the io region to the converter region), locked wall
78.3-79.3 (today 78.8, spread +-0.3), conformance 60 / 60 including the a16
cases.  a16 is DRAM-bound at 4.1 TB/s during its body (r3p15 1.4) with the
loader's TMA issue (0.65 us of a 2.05 us tile) off its critical path, so the
period prediction is "unchanged"; its fill chain is the same
`kMetaReady[0]` -> `consumed[0]` -> issue chain with 8 boxes issued by 4 warps
in parallel instead of one lane serially.

### 1.4 Blocker 4 — the register fallback is wrong

> Register fallback (section 5, item 'keep the scale-slot base as
> stageBase-relative immediates') is factually wrong: kScales / vScales are
> adjacent arrays (mha_sm90.cu:426-427) while k[] and vBufs[] are at :240 / :278,
> so the kScales-from-k offset differs from vScales-from-vBufs by ~sizeof(k[]);
> the scale slot also rotates t % 4 (nbScaleTiles) against the stage's t % 3. A
> second selected base is unavoidable, so the stated 'one base fewer' fallback
> does not exist and the register plan at the 56-register cliff has one fewer
> escape than claimed.

Conceded: `kScales[4]` / `vScales[4]` are adjacent (:426-427, 2048 B each), `k[]`
is at :240 and `vBufs[]` at :278 with the X ring between them, so
`kScales - k != vScales - vBufs`, and the scale slot rotates `t % 4` (:425,
:2675) against the stage's `t % 3`.  The fallback is struck.  The premise it
served — "the converter is at the 56-register cliff" — is also wrong: the
highest register index in the converter regions of the production SASS is
**R37** in every module (li-1 / li1 / li2 kconv and vconv: 37; gemm0 27, gemm1
33, io 35; `/tmp/r4mixed2` register scan over the lineinfo listings), i.e. 18
registers of headroom under the `setmaxnreg` 56, and `ptxas -v` reports 0 bytes
stack / 0 spill for the kernel (`/tmp/r2p8_ptx/f-1.ptxas.log`).  The
`foldMultiplier` comment (:3408-3411, "would spill one (STACK 8)") describes
the pre-[16] converter and is stale.  Register plan (section 6.1): five
operand-selected shared bases (stage, scales, barrier pair, metaReady, record)
+ one operand register for the parameter-bank offsets = **+6, R37 -> <= R43**,
no fallback needed.  If ptxas nevertheless spills, the correct escape is the
**operand-strided shared layout**: `{parts[6] 49152 B, scales[4] 2048 B,
meta[32] 1024 B}` per operand = 52224 B = 51 x 1024, two blocks back to back
(same 104448 bytes the six arrays occupy today, `sizeof(SharedMem)` unchanged
at 113 664), so every converter address is `[opBase + imm]` with one register;
barrier pairs and metaReady stay adjacent arrays at strides 48 / 16 B off the
same `operand` register.  It is a fallback, not the design, because it moves
`smem.k[]` / `vBufs[]` and therefore the GEMM groups' immediates.

### 1.5 Notes

> [0] Section 2 misdescribes the current copy issue: FP8 pages are 4 LDGSTS.128 +
> 1 scale per lane and FP4 2 + 1 (mha_sm90.cu:3167-3187), not '12 (FP8) or 6
> (FP4)'; the ncu table's 8/3 x (4+1) + 8/3 x (2+1) = 21.3 is the correct one.

Fixed (section 3): per lane 4 + 1 (FP8, :3170-3181 and :3195-3201) / 2 + 1
(FP4, :3182-3194); per warp-tile 5 / 3 `LDGSTS` instructions.

> [1] The change bundles two levers: (1) merged converter body (-20 KB hot, no
> protocol change) and (2) A16 TMA moved to the owning converter warp (-6.5 KB
> hot, new C14 tx-split protocol, speculative fill gain). By the doc's own knee
> argument (1) alone brings mixed hot to ~42 KB ~ fp8's 41 KB, which is claimed
> to be below the knee; staging (1) first is the single-variable experiment and
> attributes the fetch mechanism cleanly before adding (2).

The knee is now located: fp8 at 40.6 KB is **above** it (8.2 % / 4.8 %) and
fp4 at 37.7 KB is at the floor.  (1) alone lands the mixed module at ~42 KB
(282 + 455 + ~600 io TMA loop + ~1300 converters = 2637 instructions), i.e. at
fp8's level: the expected outcome of that single-variable experiment is
gemm0 no_inst ~8 % and a gain bound of only (17.4 - 8.2) % x T — it would
attribute the mechanism (already attributed by section 2) at the price of a
second build and run.  (2) is what takes the io region from ~600 to ~200 hot
instructions and the module under the knee (~36 KB).  In the mechanism's own
variable the bundle is one change — "hot footprint 61.6 -> 36 KB" — and (2)
carries no independent performance claim any more (fill counted 0).  Its
protocol (C14) was checked by the review (note [2]) and it removes the
`mixedLoaderTma` special case rather than adding one.  The bundle is kept; the
mechanism gate (gemm0 no_inst <= 4 %) is read before the wall so that a
footprint cut that does not remove the stall is recognised as such.

> [2] C14-C17 hold against the code as written: consumed parity at issue(g)
> implies produced[s] phase g-3 complete (gemm0 :1515 -> :1590/1595, priming
> :1462/1465); expect_tx and the warp's 32 arrivals are program-ordered on the
> same lane/warp; no cross-warp phase skew is possible; A16 warps' empty cp.async
> groups are harmless; ExpandLane offsets are stage-relative (makeExpandLane
> :571-604 ignores its base pointer) so a u32 stage base works for both operands;
> mbarrier.expect_tx without arrive already exists (barriers.cuh:217), so 8.7 is
> satisfied.

Kept as written (section 5), now for all four modules.

> [3] The two-stage accept structure (ptxas gate -> PC-sampling mechanism gate <=
> 10 % gemm0 no_inst -> wall) and the ctarec-only trace validity check are sound
> and should be kept in any revision; the bench mixed pattern page_format = p % 3
> does exercise every warp position with A16 across tiles (formats (4t+w) % 3), so
> 8.8 is likely already met.

Kept; the mechanism threshold is tightened from 10 % to **4 %** because the
design's footprint is below fp4's, whose measured floor is 2.5 % (section 8).
The per-warp A16 coverage is stated as met by the bench pattern.

> [4] no_instruction PC samples are reported when no instruction is available
> regardless of whether the warp would otherwise have been dependency-stalled, so
> 17.4 % vs 8.2 % of gemm0 samples bounds the chain effect from above; the doc
> uses -0.13 as the upper end and -0.08 central without justification for the
> central value.

Adopted as the structure of the prediction: the share difference x period is
the **bound**, r the exposed fraction; no central value is asserted, the
P = 132 calibration (r <= 0.26 under the fp8 baseline) is given as the one
data point, and the accept line is the target rather than a central estimate.

> [5] Line references checked: :156, :1352, :1515, :1761, :2159, :2212,
> :2633-2712, :3137, :3412, :3466 correct; :2176 (consumed wait) is actually
> :2184 in the worktree.

Fixed; all references in rev 2 are worktree lines re-read for this revision.

## 2. Attribution evidence

### 2.1 Production numbers (unchanged from rev 1; bench, ncu at P = 264)

Bench (locked 5x5, sessions r2p8 / r3p15 / r3pair): mixed 64.4 / 64.2 / 65.0,
fp8 67.8 / 67.6 / 68.1, fp4 60.5 / 60.4 / 60.6, a16 78.8-79.0.  P = 132 control
(r3p15 1.1, same session interleaved): mixed 73.4, fp8 72.7, fp4 68.0, a16 79.5
-> lone-CTA tile times over 66 tiles with a common fill / tail 7.2 / 2.8:
0.96 / 0.95 / 0.89 / 1.05 us.

ncu, production objects, `--launch-skip 4` (`/tmp/r4mixed_ncu_{mixed,fp8}.log`,
ratios only): mixed / fp8 duration 82.85 / 87.62 us (0.946), `smsp__inst_executed`
31.73 / 36.70 M (0.865), `LDGSTS` 21.3 / 40.0 per tile-CTA, DRAM bytes 175.8 /
156.2 MB, issue-active 57.8 / 62.9 %, eligible warps 1.38 / 1.59, stall
no_instruction **2.90 / 1.87** warps per issue-active cycle (+55 %), all other
stall classes within +-15 %.  The mixed module does 13.5 % less SM work, moves
13 % more bytes, finishes 5.4 % sooner; the one stall class that grows is
instruction fetch.

### 2.2 The eight-point PC-sampling matrix (new; production objects, r2p8 checkout)

`ncu --section SourceCounters --section WarpStateStats`, one launch each, roles
by kernel-relative address (`/tmp/r4mixed_pcs_roles.py`); gemm0 has 278-282
instructions in every module and identical SASS across the compressed modules.
"hot" = static instruction count of the five groups' loop bodies (gemm0 + gemm1
+ io loop incl. chunk fill + K conv + V conv) x 16 B.

| module | hot SASS | DRAM avg (bench) P264 / P132 | gemm0 no_inst share P264 | P132 | gemm1 P264 / P132 | all roles P264 / P132 | K-conv expand P264 / P132 |
|---|---:|---|---:|---:|---|---|---|
| transport_a16 | ~1630 = **26 KB** | 3.61 / 3.59 TB/s | 34 / 1294 = **2.6 %** | 51 / 1350 = **3.8 %** | 1.9 / 3.2 % | 2.0 / 3.2 % | idle |
| fp4 | 2358 = **37.7 KB** | 1.33 / 1.18 | 32 / 1296 = **2.5 %** | 34 / 1344 = **2.5 %** | 3.3 / 4.5 | 3.4 / 3.9 | 5.2 / 7.7 % |
| fp8 | 2537 = **40.6 KB** | 2.24 / 2.09 | 139 / 1689 = **8.2 %** | 64 / 1341 = **4.8 %** | 6.2 / 7.2 | 12.1 / 8.5 | 27.7 / 17.0 |
| mixed | 3849 = **61.6 KB** | 2.68 / 2.35 | 249 / 1432 = **17.4 %** | 123 / 1338 = **9.2 %** | 8.5 / 10.7 | 15.4 / 13.8 | 43.6 / 42.5 |

Poisson error of the shares: +-0.4 (fp4 / a16), +-0.7 (fp8), +-1.1 (mixed)
points.  Warp cycles per issued instruction (WarpStateStats): P = 264 a16 29.8,
fp4 14.2, fp8 15.0, mixed 16.6; P = 132: 16.1, 8.9, 8.9, 10.4.

Readings:

1. **Ordering.**  At both occupancies gemm0's fetch-stall share orders as the
   static hot footprint: a16 ~ fp4 (floor, 2.5-3.8 %) < fp8 < mixed.  The same
   ordering holds for gemm1 and for the converters' expansion bodies (fp4 5 %,
   fp8 28 %, mixed 44 % at P = 264).  The knee lies between fp4's 37.7 KB and
   fp8's 40.6 KB at both occupancies.
2. **Not memory load.**  a16 has the highest DRAM rate (3.6 TB/s) and the
   smallest footprint and sits at the floor; fp4 vs fp8 differ by 3 KB of code
   and 0.9 TB/s of traffic and by 3.3x in stall share while mixed vs fp8 differ
   by 21 KB and 0.4 TB/s and by 2.1x.  A load-driven fetch latency would order
   a16 first; it is last.
3. **Line-crossing fetch, not control flow** (`/tmp/r4mixed2_linepos.py`):
   of mixed gemm0's 249 no_inst samples at P = 264, **212 (85 %) fall on the
   first instruction of a 128 B line** (slot 0 of 8), 3 on the instruction
   after a `BRA` / `BSYNC`; at P = 132 89 of 123 (72 %).  fp8: 82 / 139 (59 %)
   and 27 / 64.  fp4 and a16: no concentration (8 / 32, 13 / 34).  The same
   slot-0 concentration holds in the converter bodies (mixed 166 / 239 K,
   218 / 303 V).  Mixed gemm0's stalls touch 33 of its 38 lines (fp8 23, fp4
   10, a16 12): the whole body misses, not a few sites.
4. **Occupancy dependence of the magnitude.**  Absolute gemm0 fetch stall per
   tile (share x period; periods P = 264 ~1.6 mixed / 1.7 fp8 / 1.5 fp4 / 2.0
   a16 from `(wall - fill - tail) / 33`, P = 132 the lone-CTA times):
   mixed 0.28 (P264) / 0.088 (P132), fp8 0.14 / 0.046, fp4 0.04 / 0.022, a16
   0.05 / 0.040.  The mixed-minus-fp8 excess is **0.14 us/tile at P = 264 and
   0.042 at P = 132**; the mixed-minus-floor excess 0.24 / 0.066.  A capacity
   model with equal distinct lines does not predict the factor ~3; the design
   does not need the model (1.1).

### 2.3 What the lone-CTA data bound (exposure)

At P = 132 gemm0's chain is the cadence with no arbitration loss (r3p15 1.2:
no consumer wait is on data; issue-active 56 %, 0.97 eligible warps).  The
mixed-vs-fp8 tile-time difference is 0.011 us (2.1) against a fetch excess of
0.042: **exposed fraction r_132 <= 0.26** under the fp8 baseline (the two
modules' gemm0 SASS is identical; their environments differ by 21 vs 40
LDGSTS per tile-CTA and 172 vs 152 MB).  Under the fp4 baseline (24 LDGSTS
per tile-CTA) mixed is 0.082 slower with a fetch bound of 0.066 — but fp8 is
0.071 slower than fp4 with a fetch bound of 0.024, so at least 0.047 of that
gap is non-fetch (bytes / copy issue), and the fp4 baseline does not bound r.
At P = 264 the chain is stretched 1.35x by arbitration between the two CTAs
(r3p15 6.2); a fetch stall that coincides with a lost arbitration slot is
hidden (lower r), a longer stall (2.4: per-miss stall is ~3x longer at
P = 264) exceeds the dependency slack more often (higher r).  r at P = 264 is
therefore **unmeasured, bounded by [0, 1], with 0.26 the one calibration
point**; the confirmation run's wall reads it off directly (section 7).

### 2.4 Trace-only runs (rev 1, kept for the record; not used in the prediction)

Level-1 (`/tmp/r4mixed_trace_r2p8.log`) and level-2 (`/tmp/r4mixed_trace_r3pair2.log`)
builds invert the production ordering for the mixed module (mixed slower than
fp8 by 3.7-10 us where production has it faster by 3.4-4.8) while reproducing
fp8's production body; the level-2 mixed build spills (LDL 15).  The trace
build adds 6.7-7.0 KB of code; by 2.2 that moves fp8 from 40.6 to ~47 KB
(already above the knee) and mixed from 61.6 to ~68 KB — consistent with the
fetch mechanism but not needed by it.  Consequences kept: (i) no existing
trace build is a valid instrument for the mixed module's absolute period or
fill; (ii) the mixed rows of `mixed_kv_speed_round3_fill.md` carry this bias;
(iii) the ctarec-only build (16 per-tile stamps compiled out, 4 `%globaltimer`
stores per CTA) must reproduce the production ordering before its numbers are
read (section 8).  What the traces establish as within-launch ratios: gemm0 is
not starved on K (0.2 us idle per tile in both modules), the A16 TMA bytes
are issued >= 3 us before gemm0 needs them, the loader's serial TMA issue is
0.3-1.0 us per tile inside its stage-release slack.

### 2.5 The candidates of the task, answered (rev 1 table, corrected)

| candidate | finding | consequence |
|---|---|---|
| per-page dynamic dispatch | 3 warp-uniform branches per warp-tile (:3158 bad / A16, :3170 fp8 / fp4; :3489 A16, :3504 bad, :3524 fp8 / fp4); executed bodies = static bodies +3..4 instructions; 5 of 773 expansion instructions predicated; **both formats' bodies resident in two textual copies = +21 KB of hot code**; fetch stalls order by footprint (2.2) | the code footprint is the mixed-specific cost, not the branches |
| A16 tiles by loader TMA vs converter cp.async (two landing paths) | loader issue 0.3-1.0 us per tile inside its slack; TMA lead >= 3 us; gemm0 idle on K 0.2 us as in fp8 | not on the period; the loader's TMA loop is 536 instructions of hot code (2.2) |
| metadata tags per page | one `LDS.U8` per warp per tile + 4 tag decodes in the loader | negligible |
| X ring / colMax | gemm0 / gemm1 SASS identical across modules | none |
| A16 tile period / byte-weighted mean | 172 MB in a ~52 us body = 3.3 TB/s vs a16's 4.1; weighted mean 1.72 us/tile vs measured ~1.6 | mixed is not at its bandwidth floor |

## 3. Current data flow and control flow of the touched roles (as written at 354914cc)

- **Loader, mixed / a16 modules** (IO warps 0 / 1, :2145-2244; `mixedLoaderTma`
  :156): chunk-0 fill + `metaReady[0].arrive` (:2155-2156); per tile `g`:
  record `LDS` x3 (pages `uint4`, formats, head) :2177-2180 ->
  `stageBar[stage].consumed.arrive_and_wait()` :2184 -> elected lane: 4 tag
  decodes, `nbA16` :2195-2210 -> `arrive_tx(produced, nbA16 x 2 x 2 KB, 32)`
  :2212-2213 -> up to 8 `tma::loadAsync(dst, tensorMap, {64 p, head, 0,
  page}, produced)` :2214-2226 -> `__syncwarp` -> chunk refill at
  `g + MIXED_KV_META_LEAD` :2230-2235.  Static modules (:2245-2276):
  `arrive_and_wait` + refill only.  Barrier init (:1352-1372): `produced` count
  = 128 GEMM + 128 converters + (32 loader iff `mixedLoaderTma`); `consumed` =
  128 + 32; `kMetaReady` / `vMetaReady` count 32 (:1393-1394).
- **K converters** (z = 3, :2633-2712): per warp `w` = page `w`; lane
  constants `kLane` (:2659, `makeExpandLane` :571-604: a16 / fp8 / fp4 / scale
  offsets, all stage-relative) and `kGlobals` (:2660, `ExpandScales` :561-568:
  2 globals, 2 folds, 2 bools; the folds are dead in the mixed module,
  `foldMultiplier` :3412-3422); `issueKCopies(t)` (:2661-2676): chunk-parity
  wait at `t % 16 == 0` (:2663-2668), `kBar[t % 3].consumed.wait_parity(
  toParity<3>(t))` (:2671), `issueCompressedPageCopies` (:3137-3195: tag
  `LDS.U8` from the record; A16 / bad -> return; else page + head `LDS`, per
  lane **4 (FP8, :3170-3181) / 2 (FP4, :3182-3194) `LDGSTS.128`** into the packed landing (part 1 rows
  of the warp's page) **+ 1 x 8 B scale `LDGSTS`** (:3195-3201) for lanes < 16 into
  `kScales[t % 4]`).  Loop (:2681-2712): `waitGroup<1>`, `__syncwarp`,
  `expandPackedStage` (:3467-3627; A16 -> `__syncwarp` return; bad -> zero
  rows; FP8 / FP4 bodies), `fence.proxy.async`, `kBar.produced.arrive()`, tag
  rotate, `issueKCopies(t + 2)`, `commitGroup`.
- **V converters** (z = 4, :2713-2760): textual mirror with `vBar`, `vScales`,
  `vBuf`, `vMeta = 1`, `v_global_scale`, stamps 14 / 15.
- **gemm0 / gemm1**: unchanged by this design (`kBar.produced.arrive_and_wait`
  :1515 with priming arrives :1462 / :1465; `vBar` :1761).
- **Shared memory** (:231-470): `k[6]` :240 (49152 B, alignas 1024), X ring,
  `vBufs[3]` :278 (49152), `q[2]`, col vectors, `meta[2][2][16]` :401 (2048),
  `sched`, `kScales[4]` / `vScales[4]` :426-427 (2048 each), barriers
  :436-452; `sizeof(SharedMem)` 113 664 (2 CTAs/SM, limit 115 712).
- **Registers** (measured, production SASS): gemm0 R27, gemm1 R33, io R35,
  K / V converters **R37**, other R26; `setmaxnreg` 40 / 40 / 40 / 56 / 56;
  `ptxas -v` 48 registers, 0 stack, 0 spill, no C7507.

## 4. New data flow and control flow

Scope: `ENABLE_MIXED_KV_CACHE && !SPEC_DEC` sm90 kernel; **one source shape for
all four q=1 modules**; the format branches are pruned per module by the
existing `MIXED_PAGE_STATIC_FORMAT` constant folding (:3149-3153, :3477-3481).
q=4 (`SPEC_DEC`) is untouched.

### 4.1 One converter code path for both operands

    z in {3, 4}:  operand = z - 3                                             (warp-uniform, per warp)
      stageBase   = cvta(operand ? &smem.vBufs[0] : &smem.k[0])              u32, stage s part p at + s*16384 + p*8192
      scalesBase  = cvta(operand ? &smem.vScales[0] : &smem.kScales[0])      u32, tile t at + (t % 4) * 512
      stageBar    = operand ? &smem.vBar[0] : &smem.kBar[0]                  CtaBarrierPair*, stage s at [s]
      metaReady   = operand ? &smem.vMetaReady[0] : &smem.kMetaReady[0]
      recBase     = tileRecordAddr(smem, operand, 0)                          (:786-791; tile g at + (g % 32) * 32)
      paramOff    = operand * 8                                                byte offset of {v_payload, v_scales} from {k_payload, k_scales} in FormatSpan (page_transport.cuh:28-31)
      tensorMap   = &tensorMapVLLMK + operand                                  grid-constant parameter, address selected once
      globals     = {formats[FP8].{k,v}_global_scale, formats[FP4].{k,v}_global_scale}[operand]   2 floats
      lane        = makeExpandLane(warpIdx.x)                                  (offsets already stage-relative)
      loop over g exactly as :2681-2712 with these bases; stamps slot 12 + 2 x operand / 13 + 2 x operand

`issueCompressedPageCopies` takes `operand` (runtime) in place of `isK`; the
payload / scale pointers are `LDC [span + paramOff]` per call (as the `isK`
select is a constant-bank load today), no pointer is held across the loop, no
struct reference is selected (A6 code-shape rule).  `expandPackedStage` is
unchanged except that `ExpandScales` shrinks to the two globals (the mixed
module's `foldMultiplier` already recomputes the fold per tile; `FoldOk` is one
`FSETP` per tile).  The per-tile loop body is textually one; ptxas emits it
once (gate 9.5 checks that it did not unswitch on `operand`).

### 4.2 A16 pages by the owning converter warp (all modules)

    issue(g), warp w, after the chunk-parity wait and consumed[s].wait_parity(toParity<3>(g)):
      tag = LDS.U8 rec(g) + 16 + w
      if tag == bad:                                 return tag                 (as today; expansion zero-fills)
      if tag == A16:                                                             (pruned in fp8 / fp4 modules)
          page = LDS.32 rec(g) + 4w ; head = LDS.32 rec(g) + 28
          if elect: produced[s].expect_tx(2 x 16 x 128 B = 4096)                (MBarrier::expect_tx, barriers.cuh:217 — no arrive)
                    for p in {0,1}: tma::loadAsync(stageBase + s*16384 + p*8192 + w*2048,
                                                   tensorMap, {64 p, head, 0, page}, produced[s])
          __syncwarp ; return tag
      else:                                          compressed copies as today  (pruned in the a16 module)
    iteration g (unchanged): waitGroup<1> ; __syncwarp ; expand (A16: return) ; fence ; produced[s].arrive()

The loader runs the static loop (:2245-2276) in every module: chunk fills and
`consumed` arrives only; the TMA loop (:2159-2244), `mixedLoaderTma` (:156) and
the loader term of `mixedProducedExtra` (:1352-1356) are deleted.  `produced`
count = 128 + 128 = **256 in all four modules**.  The a16 module's TMA moves
from one elected lane issuing 8 boxes per tile to 4 warps issuing 2 boxes
each; the compressed modules never reach the A16 branch (constant-folded
away); the mixed module's A16 warps issue their 2 boxes where they idle today.

## 5. Barrier, parity and ownership invariants

Existing: D1-D6, C1-C7 (dataflow doc), C8-C13 ([8]).  Added / restated:

- **C14 (tx registration is phase-correct and never premature).**  Warp `w`
  issues `expect_tx(4096)` on `produced[s]` at `issue(g)`, after
  `consumed[s].wait_parity(toParity<3>(g))` — the completion of consumed for
  tile `g - 3`, which requires gemm0's arrive after its `produced` wait for
  `g - 3` (:1515 -> :1590 / :1595; priming :1462 / :1465); hence
  `produced[s]`'s phase for `g - 3` is complete and the barrier is in tile
  `g`'s phase.  Other warps' 32-lane arrivals for tile `g` may precede this
  `expect_tx` (a warp runs up to two iterations ahead, bounded by its own
  `consumed` waits); the phase cannot complete before warp `w`'s own 32
  arrivals, which are program-ordered after its `expect_tx` (iteration `g` vs
  `g - 2`).  `complete_tx` cannot precede its `expect_tx`: same lane, program
  order, same barrier.  This is the loader's existing `arrive_tx` argument
  (:2212) moved to the converter's issue point with arrive and expect split.
- **C15 (page ownership, A16).**  Page `w`'s rows in both parts are written
  only by warp `w`'s two TMA boxes; no converter `STS` touches them (every
  warp's `ExpandLane` offsets lie in its own page's rows, :571-604), and no
  `cp.async` is issued for an A16 page, so the in-place packed landing of the
  compressed pages (part-1 rows of *their* pages) never overlaps.  D3 / D6
  unchanged.
- **C16 (WAR on the stage).**  TMA writes for tile `g` into stage `s` are
  ordered after gemm0's reads of tile `g - 3` by `consumed[s]` completion ->
  `wait_parity` (acquire) -> issue, the edge the converters already use for
  `cp.async` (C8).  The loader used the same edge through a token wait; no new
  ordering is introduced.
- **C17 (proxy order).**  TMA (async proxy) completion is tracked by the
  transaction count; the converters' generic `STS` are fenced by
  `fence.proxy.async` before `arrive` (existing).  gemm0's wgmma reads are
  ordered by its `wait_group<0>` before `consumed.arrive` (existing).
- **C4 (counts).**  `produced` = 256 in all modules (was 288 in a16 / mixed);
  `consumed` = 160 everywhere (unchanged); `kMetaReady` / `vMetaReady`
  unchanged (loader arrives, converters wait).
- **C6 (issue budget).**  Per A16 warp-tile the converter executes ~30
  instructions (3 `LDS`, elect, `expect_tx`, two address + coordinate chains,
  2 `UTMALDG`) in place of ~150-200 loader instructions per tile in the a16 /
  mixed modules; the compressed paths are unchanged.
- **C9 / C10 (records).**  Readers of record `g` are unchanged in time: the
  converter reads pages / tag / head at `issue(g)` after its `metaReady` wait
  (A16 warps now read page + head there too, as compressed warps already do).
- **C12 / Q / merge**: untouched.

## 6. Budgets

### 6.1 Registers

`__launch_bounds__(640, 2)`, `setmaxnreg` 40 / 40 / 40 / 56 / 56 (pool
3 x 128 x 40 + 2 x 128 x 56 = 29 696 of 30 720).  Converters today: **R37**
(section 3).  Change: +5 operand-selected shared bases (stage, scales, barrier
pair, metaReady, record) + 1 `paramOff` / `operand` register = +6; -2 from
the `ExpandScales` shrink (bools re-derived per tile) -> **R41 +- 2, <= 56 with
>= 13 to spare**.  The tensor-map address is a parameter-bank operand selected
once (2 registers if held; counted in the +-2).  gemm0 / gemm1: unchanged live
sets (their code is unchanged); IO: loses the TMA branch.  Fallback if `ptxas
-v` shows a converter spill anyway: the operand-strided layout of 1.4 (one
base register).  No 64-register layout is needed.

### 6.2 Shared memory

Unchanged: `sizeof(SharedMem)` 113 664 B (2 CTAs/SM); the ctarec-only trace
build adds 0 B.

### 6.3 Code footprint (from the section 2.2 counts; static)

| region | mixed today | mixed after | fp8 today -> after | fp4 today -> after | a16 today -> after |
|---|---:|---:|---|---|---|
| gemm0 / gemm1 | 282 / 455 | 282 / 455 | 282 / 455 | 281 / 455 | 278 / 451 |
| io | 1820 (TMA loop 536, static loop 127, fill + walker + Q + merge) | ~1410 | 1707 -> ~1410 | 1705 -> ~1410 | 1821 -> ~1410 |
| converters | 1248 + 1264 | **~1300** (one body incl. A16 issue ~40) | 801 + 799 -> ~820 | 712 + 710 -> ~730 | 151 + 152 -> ~190 |
| other | 651 | 651 | 652 | 649 | 651 |
| total | 5720 = 91.5 KB | **~4100 = 66 KB** | 4696 -> ~3620 | 4512 -> ~3530 | 3504 -> ~2980 |
| hot (bodies + io loop ~200) | 3849 = 61.6 KB | **~2240 = 35.8 KB** | 2537 = 40.6 -> ~1760 = 28 KB | 2358 = 37.7 -> ~1670 = 27 KB | ~1630 = 26 -> ~1320 = 21 KB |

The design's hot footprint (35.8 KB) is below fp4's measured floor point
(37.7 KB) by 1.9 KB = 120 instructions; gate 9.5 requires the converter region
<= 1380 instructions so that hot stays <= 37 KB.

### 6.4 Issue and time

Per tile-CTA (mixed): loader -150..-200, converters +30 x 1.33 A16 warps x 2
operands = +80 -> net -70..-120 of 3642 warp-instructions (-2..-3 %); DRAM
bytes, `LDGSTS` and `UTMALDG` counts per tile unchanged (2.67 boxes per operand
per tile, issued by 1-2 warps in parallel instead of one lane serially).  Per
A16 warp-tile: 2 boxes x 100-200 ns of issue on the elected lane (A1) =
0.2-0.4 us on a warp that otherwise idles; the compressed warps' copy issue
(~0.3 us) runs in parallel, so the tile's issue phase does not lengthen.  TMA
lead to gemm0 grows by the loader's serial delay (0.3-1.0 us) — irrelevant at
a >= 3 us lead.  a16 module: 8 boxes per tile by 4 warps instead of one lane,
issue segment 0.65 -> ~0.2 us on a DRAM-bound tile: no period change.

## 7. Predicted wall

Model: `wall_after = wall_today - 33 x r x delta`, `delta` = gemm0's fetch-stall
excess over the floor per tile at P = 264 (2.2 reading 4), `r` = the exposed
fraction of that stall on gemm0's chain (2.3), fill and tail unchanged (1.2).

| mode | today | delta (us/tile) | wall(r) | r = 0.26 (P = 132 calibration) | r = 0.5 | r = 1 | target / accept |
|---|---:|---:|---|---:|---:|---:|---|
| mixed | 64.4 | (17.4 - 2.5) % x 1.6 = **0.24** | 64.4 - 7.9 r | 62.3 | 60.4 | 56.5 | <= 62.0 (needs r >= 0.30) |
| fp8 | 67.8 | (8.2 - 2.5) % x 1.7 = 0.097 | 67.8 - 3.2 r | 67.0 | 66.2 | 64.6 | (not this lever) 66.0-68.1 |
| fp4 | 60.5 | 0 (at the floor) | 60.5 | 60.5 | 60.5 | 60.5 | 60.0-60.8 |
| a16 | 78.8 | 0 (at the floor; DRAM-bound) | 78.8 | 78.8 | 78.8 | 78.8 | 78.3-79.3 |

Honest statement: the target is reached iff r >= 0.30 at P = 264, and the
one calibration point (P = 132, fp8 baseline) is r <= 0.26.  Two effects pull
r at P = 264 in opposite directions (2.3); neither is measurable without the
build.  The run therefore has a defined outcome in both cases: a wall <= 62.0
confirms r >= 0.30 and the lever; a wall in (62.0, 63.2] with the mechanism
gate passed measures r in [0.15, 0.30) and records the fetch stall as mostly
hidden under gemm0's chain at 2 CTAs/SM — the kernel is then not merged as a
speed lever (section 10).  If the mechanism does not deliver at all: mixed's
structure-level floor at [8] remains `33 T + fill + tail` with T bounded below
by the lone-CTA chain 0.96 x the pair stretch, and the shared levers ([15] +
[7], fill) remain the route to and below 62; mixed is not at a bandwidth floor
(42 us body at 4.1 TB/s).

## 8. Verification artifacts, accept / reject

Build (each module a16 / fp8 / fp4 / mixed, `ptxas -v` with the ninja flags via
`/tmp/main_ptx/ninja_flags.py`, `cuobjdump -sass`, lineinfo listing for the
role split):

1. No C7507; 0 bytes stack, 0 spill stores / loads; `USETMAXREG` 2 (0x28 /
   0x38); `LDL` = `STL` = 0; REG 48; `sizeof(SharedMem)` 113 664; occupancy
   calculator 2 / 2.
2. Role SASS counts (`/tmp/r4mixed_sass.py` with the new line ranges): gemm0 /
   gemm1 unchanged in every module (282 / 455 compressed and mixed, 278 / 451
   a16; PHASECHK 8 / 17, ARRIVE 11 / 13); **one converter region** (z == 3 and
   z == 4 branch to the same body; converter region <= 1380 in mixed, <= 850
   fp8, <= 760 fp4); io region <= 1450 in every module (no TMA loop); total
   mixed <= 4300, fp8 <= 3800, fp4 <= 3700, a16 <= 3150.  Hot footprint
   computed from these counts <= 37 KB for mixed (gate 9.5).
3. `UTMALDG`: a16 8 and mixed 4, both in the converter region (io region 0);
   fp8 / fp4 0.  `mbarrier.expect_tx` without arrive: 1 site (converter body)
   in a16 and mixed, 0 in fp8 / fp4.  `LDGSTS` 42 (mixed) / 30 (fp8) / 18 (fp4)
   / 0 (a16) unchanged.
4. Barrier init: `kBar` / `vBar` `produced` count 256 in all four modules
   (`static_assert(mixedProducedExtra == 128)` or read from the SASS
   immediates).
5. Per-lane-tile executed path (`xqa_sm90_converter_sass.py --paths`): fp8
   expand 188 +- 2, fp4 187 +- 2 (operand selection did not become per-tile
   selects); converter max register index <= 45.

Conformance: `python tests/attention/run_xqa_mixed_page_transport.py` (60 cases;
the mixed cases contain tiles with A16 pages in every warp position — formats
`(4t + w) % 3` — A16 pages owned by a warp whose previous tile was compressed
and vice versa, tail tiles with bad tags in every warp position, P = 1 / 3 / 5
and T < P; the a16 cases exercise the 4-warp TMA issue on every tile) ->
60 / 60.

Mechanism (production build, before the bench; the accept gate of the lever):
`ncu --section SourceCounters --section WarpStateStats` with the 1.1 recipe
(`/tmp/r4mixed_pcs.sh` adapted to the new checkout, `/tmp/r4mixed_pcs_roles.py`
with the new lineinfo listing): mixed gemm0 no_inst share **<= 4 %** (17.4
today; floor 2.5), gemm1 <= 5 %, K / V expand <= 10 % (43 / 55 today); fp8
gemm0 <= 4 % (8.2 today); a16 unchanged (<= 4 %); `smsp__inst_executed`
30.5-31.5 M for mixed (-0.2..-1.2 M for the loader), `dram__bytes_read` 176 +- 1
MB.  Reject and re-attribute if mixed gemm0's share stays > 8 % with the
footprint confirmed by build item 2 (then the footprint is not the variable
and the wall will not move).

Timing (locked, `flock /tmp/mixedkv-gpu0.lock bash /home/bigboi/mixedkv_remote_run.sh
<checkout> r4mixed sm90 transport_a16 fp8 fp4 mixed`, 5 x 5, q=1 rows; q=4 rows
unchanged within spread; a same-session [8] control interleaved):

| mode | accept (merge as speed lever) | record only | reject if |
|---|---|---|---|
| mixed | median <= 62.0 | (62.0, 63.2] with the mechanism gate passed: r in [0.15, 0.30) recorded | > 63.2, or mechanism gate failed |
| fp8 | 66.0-68.1 (no regression) | | > 68.4 |
| fp4 | 60.0-60.8 | | > 61.0 |
| a16 | 78.3-79.3 | | > 80.0 |

Trace (ctarec-only build, 2.4; `xqa_mixed_trace_once.py --modes mixed fp8
--launches 3`): first the validity check — mixed end-max < fp8 end-max as in
production; then mixed fill median, body/tile deciles and pair delta are
recorded as the first valid mixed-module trace numbers.  If the ctarec-only
build still inverts the ordering, the per-tile stamps are not the cause and
the trace path is closed for mixed until re-attributed; the timing verdict
above does not depend on it.

## 9. Do not build if

1. `ptxas -v` shows any spill in the converter role at 56 with the merged body,
   and also with the operand-strided layout of 1.4 applied.
2. The a16 module's gemm0 / gemm1 SASS counts change, or its io region still
   contains `UTMALDG` (the loader change did not reach the a16 module — two
   source shapes exist).
3. The merged converter body exceeds today's per-lane-tile executed path by
   more than 8 instructions (build item 5).
4. `sizeof(SharedMem)` moves or the occupancy calculator returns != 2.
5. Hot footprint from the role counts (build item 2) is > 37 KB for the mixed
   module — e.g. ptxas duplicated the converter body by unswitching on
   `operand`, or the io loop did not shrink: the mechanism premise is void;
   stop before timing.
6. The mechanism gate fails: mixed gemm0 no_inst > 8 % with the footprint
   confirmed.  Then the footprint reading was wrong, the wall prediction is
   withdrawn, and the honest mixed number stays 64.4 pending the shared levers.
7. The tx accounting cannot be expressed with a per-warp `expect_tx` (the
   helper exists at barriers.cuh:217; an `arrive.expect_tx` would count the A16
   warp twice per phase — never substitute it).
8. The conformance runner's mixed cases do not execute the A16-by-converter
   path on every warp position, or the a16 cases do not execute the 4-warp TMA
   issue (add `page_format = (p + k) % 3`, k in 0..2, if it cannot be shown
   from the existing pattern).

## 10. Go / no-go

**Go**, as a code-footprint lever (one converter code path for both operands,
A16 transport owned by the converter warp, loader = static loop in every
module), gated before any timing on `ptxas -v`, on the SASS role counts (hot
<= 37 KB by construction) and on the PC-sampling mechanism check (gemm0
no_inst <= 4 %).  The mechanism — instruction-fetch stalls on gemm0's pacing
chain that order strictly by the module's static hot footprint at both
occupancies, with a knee between 37.7 and 40.6 KB and the memory-load
alternative refuted by the a16 control — is established by profiler-only
measurement of the production objects (section 2).  The exposed fraction r of
that stall at 2 CTAs/SM is the one unmeasured quantity; the target needs
r >= 0.30 and the one calibration point says r <= 0.26 at 1 CTA/SM, so the
target is **not assured**: the accept line is the target itself (mixed median
<= 62.0), a wall in (62.0, 63.2] with the mechanism gate passed is recorded as
a measurement of r and the kernel is not merged as a speed lever, and a wall
> 63.2 or a failed mechanism gate rejects.  Recorded for the plan: (a) the
trace builds are not a valid instrument for the mixed module (2.4) and the
r3fill / r2p8 mixed trace rows carry that caveat; (b) the "mixed period 1.44"
premise is withdrawn — mixed is 0.93x fp8 per tile in production and not
bandwidth-bound; (c) the fetch-stall floor of the [8] kernel is ~2.5 % of
gemm0's time and fp8 sits above the knee too (8.2 %): the same change takes
fp8 to the floor (bound 3.2 us x r); (d) any later lever that adds hot code to
the compressed modules (trace stamps, pool protocol, [7]'s fused consumer)
must budget against the 37.7-40.6 KB knee — the r3pair protocol warp and the
level-2 trace build both crossed it; (e) the converter groups have 18
registers of headroom under 56 (R37), not a cliff.

Artefacts (nkcut2): `/tmp/r4mixed2_pcs_{fp4,mixed,fp8,transport_a16}_P{264,132}.{ncu-rep,csv,details,log}`
(fp4 / a16 at P = 264, all four at P = 132), `/tmp/r4mixed_pcs_{mixed,fp8}.*`
(P = 264, rev 1), `/tmp/r4mixed_ncu_{mixed,fp8}.log`, `/tmp/r4mixed_trace_r2p8.log`,
`/tmp/r4mixed_trace_r3pair2.log`, `/tmp/r2p8_ptx/li{0,1,2,-1}.nvdis` (lineinfo
listings of the a16 / fp8 / fp4 / mixed modules), scripts `/tmp/r4mixed2_pcs.sh`,
`/tmp/r4mixed2_linepos.py`, `/tmp/r4mixed_pcs_roles.py`, `/tmp/r4mixed_sass.py`,
`/tmp/r4mixed_{trace,ncu,pcs}.sh`; local `/tmp/r4mixed_{gaps,smid,pair_split}.py`.
