# Round 4, lever [7] — two fused consumer groups on alternate tiles at one CTA per SM (cadence lever), **rev 2**

Kernel: `csrc/xqa/mha_sm90.cu` at `354914cc` / `39ef0531` (branch
`claude/mixed-kv-sm90-tma`; production q=1 kernel state = lever [8]; the kernel
source in this worktree is byte-identical to production, `git diff 354914cc --
csrc` empty).  Bench shape B=17, S=4096, 8 KV heads, GQA 4, D=128, 64-token
tiles, 16-token pages.  Line references are into `mha_sm90.cu` of this worktree.
**No production kernel edit, no build of a modified kernel, no timed run in
this phase.**  New in rev 2 is one *reading* measurement: the per-role
attribution of the dynamic instruction counts that today's production-kernel
ncu `SourceCounters` runs already contain (`/tmp/r4mixed2_pcs_*.ncu-rep` on
nkcut2, production checkout `dash-flashinfer-claude-r2p8`, taken for the
round-4 mixed design), plus a compile-only build of the current source with and
without `-lineinfo` (`/tmp/r4p7_sass/f{1,2,-1}{,_li}.cubin`): the lineinfo
SASS is identical to the production-flag SASS of the same source (`diff` of
the address-stripped `cuobjdump -sass` listings: 0 lines; 4 696 / 4 512 /
5 720 instructions, REG 48, STACK 0 for fp8 / fp4 / mixed).

State (nkcut2 H200, locked 5x5 medians, q=1 us): transport_a16 79.4, fp8 67.9,
fp4 60.6, mixed 64.6.  Targets fp8 <= 58, fp4 <= 36 (analytically out of
reach, plan gate), mixed <= 62.

## 0. Verdict in one paragraph (rev 2)

Rev 1 was judged not approvable on five blockers: a `qBar` deadlock at item
entry (produced count 256 + 32 against an owner-only arrive), 1-tile items
breaking the `qBar` phase, a double V-stage release at item boundaries plus a
missing `qBar.consumed` pre-arrive, a converter instruction share halved by a
"(2 CTAs)" error, and an IPC argument resting on a 1.38 GHz DRAM-shifted ncu
run.  Rev 2 answers each (section 1).  The protocol blockers are resolved by
construction (section 4: strict item-ordinal sequence with catch-up, one V
release site per group plus epilogue, an exchange counter for the partial
slots, full prologue pre-arrive table) and every arrive/wait pair is
enumerated for multi-tile items, 1-tile items at either parity, item
boundaries at either group, and range start/end (section 5).  The budget
blocker is resolved by measurement and it decides the lever: **on the
production kernel at P = 132 the SM executes ~4 150 warp-instructions per
tile, of which the eight converter warps execute 2 632 (63 %), the two consumer
groups 1 462 (35 %, of which 668 are uniform-datapath wgmma-descriptor
arithmetic) and the IO warps ~75** (section 2).  The fused consumer removes
~270 instructions per tile (-18 % of the consumers, **-6.5 % of the SM's
work**).  The cadence is then bounded on one side by the SM's issue rate at
16-20 warps (production steady-state IPC 2.21 lone / 2.46 paired at 1.98 GHz,
not the locked-clock ncu values) and on the other by the eight converter
warps' own busy time (329 instructions per warp per tile at 4.0-5.8 cycles
each = 0.66-0.96 us); the self-consistent point is **0.83-0.85 us per tile-SM
(band 0.80-0.89)** for fp8, 0.77-0.80 for fp4, 0.80-0.85 for mixed (section
7).  Walls: **fp8 64.6 (62.6-68.5), fp4 60.5 (59-63), mixed 63.8 (59.5-68.7),
a16 ~80** — central gains -3.3 / +0 / -0.8 us, no target inside the central
case, fp8 <= 58 outside the band, mixed <= 62 only at the optimistic edge.
**Recommendation: no-go (section 9).**  Rev 1's own do-not-build condition 4
(converter share > 45 %) is met by the production kernel itself (63 %), so
this lever cannot be the cadence lever until the converter instruction count
is cut; the measured budget names that cut (copy-issue 720, expansion 1 504,
consumer descriptor arithmetic 668 per tile-SM) as the levers that act on the
SM's actual bottleneck.

## 1. Judge blockers and notes (rev 1, journal `wf_f88417cb-2ef`), each quoted and answered

### B1 — item-entry deadlock on the `first` bit

> "Deadlock on the bench shape as written (3.3 item entry uses the record
> `first` bit): only the owner of an item's first tile resets O/m/l and does
> `qBar[ord&1].produced.arrive_and_wait`, but 3.5 / C4 set the produced count to
> 256 + 32 (both groups).  The partner group never sees `first` on its own tiles
> (with S=4096, 64-tile items and even CTA starts x0 = 66k, C0 owns every `first`
> tile and C1 never does), so C1 never arrives: the produced phase never
> completes and C0 blocks at item 0.  Even if `first` is read as 'first owned
> tile' (ord != prevOrd), C1 also never resets its accumulator under the written
> rule.  3.3 and 3.5/C21 contradict each other; the mechanism for 'first owned
> tile of the item' is not designed."

Accepted; rev 1 was inconsistent.  Rev 2 (4.2, 5.1): item entry for a group is
**"the record's item ordinal differs from the group's current ordinal"**, never
the `first` bit.  The group keeps `curOrd` (starts at -1); on every owned tile
it reads the 3-bit ordinal field of the record and advances `curOrd` by the
(mod-8) difference, which is 0 (same item), 1 (next item) or 2 (one skipped
1-tile item of the partner — a group can skip at most one consecutive item,
5.3).  On any advance the group resets (O, m, l) and runs the Q protocol for
every ordinal it advanced over, in order.  `qBar[b].produced` is **arrived by
the Q warp only (count 32)**; the groups `wait_parity` on it and never arrive
(4.3), so no rendezvous between the groups exists at item entry.  The `first`
bit keeps its present meaning (used by the finalizer to recognise a solo item:
`first && last` on its own last tile).

### B2 — solo items break the `qBar` protocol

> "Solo items break the qBar protocol regardless of the fix above: for a 1-tile
> item the non-owner never observes the item (no tile, no record), so it never
> arrives on `qBar[ord&1].produced` (count 256 + 32) -> the owner deadlocks, or,
> if the non-owner later arrives for item ord+2 on the same slot, that arrival
> is absorbed into item ord's phase and Q(ord+2) can be read before the Q warp
> stored it.  The doc lists S=64/128 `XQA_PERSISTENT_CTAS`=1/3/5 as the
> conformance cases that exercise this; those runs would hang.  A consistent
> fix (not in the doc): produced count = 32 (Q warp only) with both groups using
> `wait_parity(toParity<2>(ord))` instead of arrive_and_wait; consumed stays 128
> finalizer + 32."

Accepted, with one correction to the suggested fix.  `produced` count 32 with
group-side `wait_parity` is adopted (4.3).  **`consumed` = 128 finalizer + 32
is not adopted**: with a parity wait, a group that skips a solo item and later
waits for item `ord + 2` on the same slot can find the barrier one phase
*behind* (Q(ord) not yet stored because the partner — up to `dK` tiles behind
through the K ring — has not yet finalized item `ord - 2`); `try_wait.parity`
then returns true for the wrong phase and the group reads a Q buffer the Q
warp has not written (5.6 gives the construction).  Rev 2 therefore makes both
groups **process every item ordinal in strict sequence**: a group that
observes an ordinal jump of 2 first *catches up* on the skipped ordinal `k`
(wait `produced[k & 1]` parity `toParity<2>(k)`, then arrive
`consumed[k & 1]`), then handles its own item.  `consumed` count is **256 +
32**: both groups arrive exactly once per ordinal (participants after their
last QK of the item, the non-participant at catch-up after the produced wait),
plus the Q warp's `arrive_and_wait`.  With that rule a group has waited every
earlier phase of the slot in order, so the barrier is at the awaited phase or
one beyond and the parity wait is unambiguous (5.6).  The Q warp's code
(:2278-2298) is textually unchanged; only barrier init counts and the prologue
pre-arrivals change.  Conformance cases with 1- and 2-tile items at both
parities remain required (8.7).

### B3 — double V-stage release; missing `qBar.consumed` pre-arrive

> "Double V-stage release (C19 / 3.3): the [QK] step does
> `vBar[(g-2)%dV].consumed.arrive` unconditionally 'if g >= 2', but an item
> branch at tile g-2 (last or nextLast) already released V(g-2) after its own
> wait_group<0>.  Two 128-arrivals on one consumed phase complete the NEXT phase
> early, so the V converters' `wait_parity(toParity<6>(t))` for the stage's next
> tile passes before its owner has consumed it -> cp.async overwrites a V stage
> the PV wgmma is still reading (silent corruption, first at every item
> boundary).  Needs a per-group 'released' flag; the doc states such a guard
> only for the epilogue.  Related omission: the prologue (3.3) pre-arrives only
> kBar/vBar consumed; today gemm0 also pre-arrives `qBar[*].consumed`
> (:1459-1461) and the Q warp's `consumed.arrive_and_wait` at items 0/1 needs it
> -> Q warp hangs before any tile."

Accepted.  Rev 2 has **one V release site per group**: V(g) is released at the
group's next owned tile `g + 2`, after that tile's QK `wait_group<0>` (which
retires PV(g) because a warpgroup's wgmma groups complete in order), and the
epilogue releases the group's last owned tile.  Item branches execute
`wait_group<0>` (finalize / publish need O) but **do not release V**; no flag
is needed because the release site is unique (4.2 [QK] and epilogue).  Cost:
the V stage of an item's last tile is held until the group's next tile, which
`dV = dK + 1` already accounts for (6.2).  Prologue pre-arrivals are now a
table (5.2): `kBar[*].consumed` and `vBar[*].consumed` 128 by C0 (count 128 +
32 loader in step A, phase 0 = "stage free"); **`qBar[b].consumed` 128 by C0
and 128 by C1 for b = 0, 1** (count 256 + 32; phase 0 completes with the Q
warp's own `arrive_and_wait` for items 0 and 1, as today with 128 + 32);
`partialBar[p].consumed` 128 by C0 for p = 0, 1 (count 128).

### B4 — converter instruction share mis-derived

> "Converter instruction share is mis-derived and trips the doc's own gate 4:
> 5.5 takes 8 warps x (188 + 71-90 + 50) = ~2 600 per tile and halves it to
> ~1 300 by calling it '(2 CTAs)'.  Each tile's 8 pages are expanded by one
> CTA's 8 warps, so the per-tile-SM converter count is ~2 500-2 600 (the [15]
> doc 6.2 itself uses 2 x 4 x 188 = 1 504 for the expansion alone).  Against the
> ncu total of 4 150-4 213 per tile that is ~40-60 %, above the 45 %
> do-not-build threshold, and the arithmetic is inconsistent (2 100 consumer +
> 2 600 converter + 150 IO = 4 850 > 4 213 measured), so either the 'today
> ~2 100' consumer baseline or the 3 600 per-tile-SM gate is wrong.  The gate
> values of 7.2 / 8.2 / 8.4 must be re-derived from a SASS role count before
> they can be used as hard stops."

Accepted; both the halving and the "consumer ~2 100" were wrong.  Rev 2
measures instead of estimating: section 2 attributes the per-PC "Instructions
Executed" of the production kernel (ncu `SourceCounters`, P = 132 fp8 / fp4 /
mixed and P = 264 fp4) to roles through the lineinfo listing of the same
source.  Result (fp8, per tile-CTA = per tile-SM at P = 132): **gemm0 927,
gemm1 535, IO ~75, K converters 1 314, V converters 1 318; total 4 122 body +
151 item-level = 4 273 vs 4 150 in the uninstrumented pass** (the difference
is the spin iterations the instrumented pass adds, 2.3).  Converter share 63 %
(fp8), 62 % (fp4), 57 % (mixed).  The [15] doc's "consumer ~2 100" was 1 462;
rev 1's "converters ~1 300" was 2 632.  Rev 1's gate 4 (> 45 %) is met by the
production kernel, i.e. by rev 1's own rule the lever is not buildable until
the converter count is cut (section 9).

### B5 — the performance claim does not follow from the cited evidence

> "(a) the IPC values 2.24 / 2.51 come from the pair-doc ncu run locked at 1.38
> GHz, which the pair doc (9.1, note 12) explicitly says is DRAM-shifted and
> 'not used for this claim'; at the production 1.98 GHz the launch-average IPC
> is 36.7 M / (132 x 67.9 us x 1.98 GHz) ~ 2.07, so the issue model at IPC
> 2.1-2.3 with 3 600-3 800 instr/tile gives 0.79-0.91 us, central ~0.85 -> fp8
> ~66, not 62; (b) the 'issue floor' of 1.3 is a tautology (cadence =
> instructions / IPC) — 56-63 % issue-active with ~1 eligible warp per scheduler
> is the latency-bound signature the [15] and P0.5 docs already recorded, so
> 'two models that must agree' is one chain model plus a restatement, not
> independent confirmation; (c) the chain model's 1.31 us is built from
> unmeasured deltas (PV issue 0.30 vs 0.39, V wait 0.08 vs 0.19, rescale 0.06
> vs 0.19, P store 0.14 vs 0.22) — with the [15] doc's own fused-chain estimate
> of 1.55 us and the same stretch band the cadence is 0.92-1.05, i.e. no gain
> over today's 0.865.  Even the doc's central case (fp8 62.0, mixed 62.8) misses
> fp8 <= 58 and mixed <= 62, and the accept threshold (fp8 <= 63.5) is set to
> accept a target miss."

Accepted on all three points.  (a) Rev 2 derives IPC from production
cadences at 1.98 GHz: steady-state lone `4 150 / (0.95 us x 1.98 GHz) = 2.21`,
paired `4 213 / (0.865 x 1.98) = 2.46` (fp8); the locked-clock ncu values are
quoted only as the source of the instruction counts, which are clock-invariant
(7.1).  The launch-average 2.07 includes fill and tail and is listed as the
lower edge.  (b) Rev 2 drops the "two models" framing: the issue model is the
model, and its unknown (the IPC the fused SM sustains at 16-20 warps) is
bounded from the *other* side by a constraint rev 1 did not have — the
converter warps' own busy time per tile, which is measured (expansion 4.0
cycles per instruction lone, 5.8-7.4 in the pair) and which the fused design
does not change (7.2).  The two constraints cross at 0.83-0.85 us; that is
not a tautology but a feasibility bound.  (c) The chain model is demoted to a
sanity check with only measured segments (7.3); it does not bind.  The central
prediction is now fp8 64.6 / fp4 60.5 / mixed 63.8, the accept thresholds are
withdrawn with the recommendation, and the lever is a no-go.

### Notes (N1-N7), acknowledged

- N1 (files reviewed) — noted.
- N2 "Register budget is sound: dropping setmaxnreg at
  __launch_bounds__(640,1) gives a uniform 96-register cap ... the fused
  consumer estimate (~52 peak, <= 56) is consistent" — kept unchanged (6.1).
- N3 "Shared memory at dK=5/dV=6 (~206-207 KB) fits ... Side effect not
  mentioned: L1 shrinks to ~28-49 KB per SM; cp.async.cg bypasses L1 so likely
  harmless." — recorded in 6.2 as a fact the confirmation run would have to
  check (`l1tex__t_sector_hit_rate` of the Q / record / page-table loads).
- N4 "The wait_parity two-phase ambiguity on partialBar is safe only because
  the groups are coupled within ~dK tiles through the K ring with dK odd (tile
  g and g-5 have different owners); this is not stated and would silently
  weaken if dK were made even." — made explicit: invariant C20b' in 5.6
  requires `dK` odd and `dK <= 5` with the proof, and a `static_assert`.  The
  partial exchange is also re-indexed by an *exchange counter* (multi-tile
  items only) because a solo item publishes nothing and would otherwise skip a
  phase — a second defect of rev 1's `toParity<2>(ord)` that N4 did not name
  (4.4).
- N5 "The pipeline-to-parallel argument is the real lever: today's lone CTA
  already overlaps two chains (gemm0(g+1) || gemm1(g)) at 0.95 us; the fused
  design's gain over that is exactly the protocol overhead removed ... divided
  by two.  Trace-side proxies do exist for parts of it (the xBar RT and the
  rescale segment are directly stamped), so 1.5's 'no proxy' claim is too
  strong" — agreed and adopted as the framing of section 7: the gain is the
  removed instructions (measured per line in 2.2: xBar wait / arrive, xCol
  exchange, ballot rescale, second loop) and the removed slot asymmetry;
  nothing else.
- N6 (staging shape, `partialDone[isCtaLast]`) — kept (4.5).
- N7 (fp4 <= 36 out of reach) — unchanged.

## 2. Attribution evidence, rev 2: the per-role instruction budget of the production kernel

### 2.1 Method (reading only)

Inputs already on nkcut2: `/tmp/r4mixed2_pcs_{fp8_P132,fp4_P132,fp4_P264,
mixed_P132}.ncu-rep` (production checkout `dash-flashinfer-claude-r2p8`, whose
`mha_sm90.cu` differs from this worktree's only in trace-build diagnostics —
`persistentCtaIndex()` / `MIXED_KV_TRACE_REVERSE_RANGES`, `diff` = 22 lines;
the wrapper returns `blockIdx.x` in production builds, so the algorithm and
every per-tile path are the same; both sources compile to 4 696 SASS
instructions at REG 48 / STACK 0, with scheduling-level differences in the
listing — the attribution below therefore uses the r2p8 lineinfo listing that
matches the profiled binary exactly), `--section SourceCounters --section
WarpStateStats`, `--launch-skip 4 --launch-count 1`, one launch of the locked
bench (B = 17, S = 4096).  The source page (`--page source --csv`) gives
"Instructions Executed" (warp-level `inst_executed`) per SASS address; each
address is attributed to a role by the outermost `mha_sm90.cu` frame of the
lineinfo listing `/tmp/r2p8_ptx/li{1,2,-1}.nvdis` (r2p8 line ranges gemm0
1434-1687, gemm1 1687-2107, IO 2107-2616, K converters 2616-2697, V converters
2697-2745; this worktree's lines are +17 from :688 on).  Scripts:
`/tmp/r4p7_roles.py` (per role / opcode / source line) and `/tmp/r4p7_body.py`
(body / spin split) on nkcut2.  8 712 tiles per launch (132 x 66); at P = 132
one tile-CTA is one tile-SM.

Two caveats the numbers carry.  (i) The `SourceCounters` pass is instrumented
and slows the kernel, so barrier spin loops (`SYNCS.PHASECHK` + `NANOSLEEP` +
`BRA`) iterate more than in production: its total is 6 621 per tile (fp8 P132)
against `smsp__inst_executed.sum / 8 712 = 4 150` in the uninstrumented pass.
The split below therefore caps every PC at one execution per tile per warp
("body") and reports the excess separately ("spin"); PCs executed less than
once per two tiles are "rare" (item-level, prologue, chunk fills).  (ii) ncu
locked the SM clock at ~1.38 GHz (`gpc__cycles_elapsed` 130 274 in 94.6 us);
instruction counts are clock-invariant and are the only thing used from these
runs.

### 2.2 Result: warp-instructions per tile-CTA (= per tile-SM at P = 132)

| role (warps) | fp8 P132 body | uniform-datapath | spin (instr. pass) | rare | fp4 P132 body | fp4 P264 body | mixed P132 body (+rare) |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemm0 (4) | **927** | 376 | 595 | 5 | 923 | 903 | 927 |
| gemm1 (4) | **535** | 292 | 863 | 20 | 527 | 530 | 535 |
| IO (4: K/V loader, Q, merge) | 28 (+46 rare) | 10 | 850 | 46 | 27 (+46) | 27 (+59) | 114 (+47) |
| K converters (4) | **1 314** | 200 | 5 | 25 | 1 189 | 1 193 | 401 (+589) |
| V converters (4) | **1 318** | 200 | 34 | 25 | 1 193 | 1 204 | 400 (+592) |
| prologue / other | 0 | | | 29 | 0 (+28) | 37 (+20) | 0 (+29) |
| **body + rare** | **4 273** | | 2 347 | | 4 003 | 4 065 | 3 659 |
| uninstrumented `smsp__inst_executed.sum` / 8 712 | **4 150** | | | | 3 881 | 3 941 | 3 580 |
| HGMMA per tile | 64 (32 + 32) | | | | 64 | 64 | 64 |

(For mixed, one converter warp in three holds an A16 page and skips the
expansion, so its expansion PCs execute 2/3 of the time and land in "rare";
converter total = 1 012 + 1 032 = 2 044.)  Body + rare exceeds the
uninstrumented total by 3 % because the wait loops' first iteration is counted
in "body"; in production the `mbarrier.try_wait` suspend hint makes the spin
share small (the K-wait is already complete when gemm0 reaches it, P0.3).

**Shares (fp8):** converters 2 632 / 4 150 = **63 %**; consumers 1 462 =
**35 %**, of which uniform-datapath (`ULEA / UIADD3 / ULOP3 / USHF / UMOV /
UIMAD`, i.e. wgmma descriptor arithmetic and stage/record addressing) **668 =
16 % of the SM's instructions**; IO ~75 = 2 %.  fp4: converters 62 %.  mixed:
57 %.

Per source line (fp8 P132, body per tile, this worktree's line numbers; r2p8
numbers in parentheses):

| gemm0 line | body / tile | what | gemm1 line | body / tile | what |
|---|---:|---|---|---:|---|
| :1504 (1487) loop | 16 | loop overhead | :1752 (1735) loop | 16 | loop overhead |
| :1514-1515 (1497-1498) `kBar.produced.arrive_and_wait` | 12 + 68 | stage index, arrive + one try_wait | :1757-1761 (1740-1744) `vBar` wait | 12 + 56 | |
| :1522-1524 (1505-1507) record LDS, bits | 32 | | :1770-1773 (1753-1756) record LDS, bits | 24 | |
| :1572 (1555) Q descriptor `addAddr` | 68 | **uniform ops** | :1792 (1775) `xBar.produced.arrive_and_wait` | 32 | **removed by [7]** |
| :1575 (1558) K descriptor `addAddr` | 172 | **uniform ops** | :1846 (1829) `rescaleGemm1AccForNewColMax` | 124 | LDS xColMax / xColSum, ballot, shfl, expf, 8 FMUL: **replaced by a register rescale (~40)** |
| :1595 (1578) `kBar.consumed.arrive` | 28 | | :1871 (1854) `gmma::fence` | 12 | |
| :1602 (1585) `acc * qkScale` | 16 | | :1923 (1906) X descriptor `addAddr` | 192 | **uniform ops** |
| :1639 (1622) `computeWarpGrpColMax_sync` | 140 | STS slot, `bar.sync`, LDS, fold | :1933 (1916) V descriptor | 12 | |
| :1641 (1624) `warpGrpOnlineSoftmax` | 40 | | :2112 (2095) `xBar.consumed.arrive` | 24 | **removed** |
| :1651 (1634) `computeWarpColSum` | 60 | | :2114 (2097) `vBar.consumed.arrive` | 8 | |
| :1657 (1640) `acc * kE4M3_MAX` | 16 | | | | |
| :1663 (1646) `storeGemm0AccToShm` | 104 | bf16 pack + `stmatrix` (~44) + `xBar.consumed.arrive_and_wait` (~60, **removed**) | | | |
| :1674-1676 (1657-1659) `xColMax / xColSum` STS | 32 | **removed** | | | |
| :1689-1690 (1672-1673) fence + `xBar.produced.arrive` | 24 + 24 | fence stays (P visibility), arrive **removed** | | | |

The HGMMA count is exactly 8 per warp per GEMM (unrolled; no k-loop), so the
descriptor lines are straight-line code executed once per tile: **the 8
`m64n8k16` QK instructions cost 240 descriptor instructions and the 8 PV
instructions 204** — the "~400 cycles of descriptor arithmetic per GEMM" of
P0.3 (d) is this count.  The `SYNCS` / `NANOSLEEP` / `BRA` columns of the
opcode histogram (gemm0 281 / 126 / 155, gemm1 386 / 181 / 211 in the
instrumented pass) are the spin loops and are not part of the body.

### 2.3 What the fused consumer removes and adds (per tile-CTA, from 2.2)

    removed  gemm0: xBar.consumed wait inside storeGemm0AccToShm ~60, xColMax/xColSum STS 32, xBar.produced.arrive 24
             gemm1: xBar.produced wait 32, ballot rescale 124, xBar.consumed.arrive 24, second loop + record LDS + stage index 40
             total ~ -336
    added    register rescale (8 FMUL + 2 MUFU + folds) ~40, bar.sync for P visibility ~8, deferred-release bookkeeping ~16
             total ~ +64
    net      ~ -270 per tile-CTA  ->  fused consumer body ~1 190 (today 1 462), SM total ~3 880 fp8 / 3 610 fp4 / 3 310 mixed

The descriptor arithmetic (668) is untouched by [7]; so are the converters
(2 632).  Everything else in 2.2 is unchanged by construction (3.4).

### 2.4 IPC at production clocks (replaces rev 1 section 1.3)

| quantity | fp8 | fp4 | mixed | source |
|---|---:|---:|---:|---|
| instr per tile-SM, uninstrumented | 4 150 (P132) / 4 213 (P264) | 3 881 / 3 941 | 3 580 / ~3 600 | ncu `smsp__inst_executed.sum` / 8 712 |
| production lone cadence, us | 0.95 | 0.89 | 0.96 | [15] doc 1.1 (`XQA_PERSISTENT_CTAS=132` ctarec) |
| production paired per-tile-SM cadence, us | 0.858 | 0.75 | 0.81 | (wall - fill - tail) / 66 with fill 8.5 and tail 2.8 / 2.6 / 2.8 |
| **steady-state SM IPC, lone (20 warps)** | **2.21** | **2.27** | **1.88** | instr / (cadence x 1.98 GHz) |
| **steady-state SM IPC, paired (40 warps)** | **2.48** | **2.65** | **2.24** | |
| launch-average IPC, paired | 2.07 | 2.16 | 1.95 | 36.7 M / (132 x wall x 1.98 GHz), includes fill / tail |
| ncu issue-active (locked clock) | 0.56 (P132) / 0.63 (P264) | 0.56 / 0.65 | 0.48 / 0.58 | rev 1 1.3, mixed doc 1.1 |

The locked-clock ncu IPC (2.24 / 2.51) and the production-derived
steady-state IPC (2.21 / 2.48) agree within 2 % for fp8 — the DRAM shift the
pair doc warned about matters for durations, not for the fp8 issue picture
(DRAM 43 %) — but rev 2 uses the production-derived values throughout.

### 2.5 Converter warp busy time per tile (the new bound)

One converter warp executes 329 (fp8) / 297 (fp4) body instructions per tile
(2.2), of which expansion 188, copy issue + parity wait ~104, loop glue ~37.
Its issue rate is measured on the expansion segment (no stamps inside it):

| layout | expansion segment (trace, ns) | cycles per instruction (188 instr, 1.98 GHz) | warp busy per tile (fp8, 329 instr) |
|---|---:|---:|---:|
| lone CTA, P = 132 | 383 (K) / 409 (V) | **4.0 / 4.3** | 0.66-0.72 us |
| pair, fast member | 552 / 602 | 5.8 / 6.3 | 0.96-1.05 us |
| pair, slow member | 707 / 742 | 7.4 / 7.8 | 1.23-1.30 us |

(pair doc 1.3; the P0.4 rate of 8.3-8.5 was the pre-[16] kernel at 2 CTAs/SM.)
The converter warp's busy time is a serial per-tile requirement — one warp
expands one page per tile — so **the cadence cannot be below the converter
warp's busy time at whatever issue rate the SM's load leaves it**.  Rev 1
noted "71-73 % of the cadence, co-critical" but derived it from the halved
count; with the measured 329 instructions the converter warp is at 83-91 % of
a 0.79 us cadence even at its lone-CTA issue rate.

## 3. Data flow (rev 1 section 3.2 with the rev 2 changes marked)

    records          fill (loader warp) -> meta[op][chunk][g % 16]; NEW bits: 20 nextLast, 21-23 item ordinal mod 8 (C23)
    K stage g        converters (page w of tile g) -> k[(g % dK) * 2 + part]  -> kBar[g % dK].produced       (unchanged code, dK = 5)
    V stage g        converters -> vBufs[g % dV]                              -> vBar[g % dV].produced       (unchanged code, dV = 6)
    Q(ord)           Q warp -> q[ord & 1] -> qBar[ord & 1].produced (count 32, Q warp only)               (unchanged code)
    group c, tile g  kBar wait -> QK wgmma (K(g), Q(ord)) -> Gemm0Acc -> mask, colMax (group-private slots) -> softmax, colSum
                     -> bf16 -> P[c] -> fence.proxy.async + bar.sync(id_c) -> rescale O(c) in registers -> vBar wait
                     -> PV wgmma (V(g), P[c]) -> O(c) += ; commit, wait deferred to the next owned tile's QK wait
    item end         publisher (owner of b_j - 1) -> partial[nx & 1] (O 4 KB, m, l) -> partialBar[nx & 1].produced
                     finalizer (owner of b_j) combines, finalizes (output or scratch 2c + isCtaLast), publishes to the merger
                     (nx = exchange counter over multi-tile items, 4.4)

Deleted as in rev 1: the X ring, `xBar[]`, `xColMax / xColSum`, `gemm1AccColMax
/ Sum` as inter-group state, `finalizedItems` (-> `partialDone[2]`, C22).  Two
layouts as in rev 1 3.1 (step A five groups `__launch_bounds__(640, 1)` no
`setmaxnreg`; step B four groups) — unchanged and still the staging order had
the lever been a go.

## 4. Control flow of a consumer group, rev 2 (one body, c = warpIdx.z in {0, 1})

### 4.1 Group state

    curOrd  = -1      item ordinal the group is in (full counter, reconstructed from the record's 3 bits)
    nx      = 0       exchanges seen (multi-tile items completed by this group's item branch)
    pendV   = none    V stage whose release is deferred (stage index of the previous owned tile)
    O (8), m (dup form, 2), l (per warp, 2)

### 4.2 Loop

    prologue   (5.2 table): c == 0: kBar[*].consumed.arrive, vBar[*].consumed.arrive, partialBar[*].consumed.arrive
               both c:      qBar[0].consumed.arrive, qBar[1].consumed.arrive
    for g = c; g < G; g += 2:
      [K]      kBar[g % dK].produced.arrive_and_wait
               word = LDS meta[K][g].tile ; last, nextLast, first, partial, ctaLast bits ; ord3 = bits 21-23
               d = (ord3 - (curOrd & 7)) & 7                         -- 0, 1 or 2 (5.3); assert d <= 2
               if d != 0:                                            -- item entry (B1)
                 for k = curOrd + 1 .. curOrd + d - 1:               -- catch-up, at most one iteration (B2)
                   qBar[k & 1].produced.wait_parity(toParity<2>(k))
                   qBar[k & 1].consumed.arrive
                 curOrd += d ; O = 0 ; m = init ; l = 0
                 qBar[curOrd & 1].produced.wait_parity(toParity<2>(curOrd))
      [QK]     gmma::fence ; 8 x m64n8k16 SS (K(g), Q(curOrd)) ; commit ; wait_group<0>
               if pendV != none: vBar[pendV].consumed.arrive ; pendV = none      -- the ONLY in-loop V release (B3)
               kBar[g % dK].consumed.arrive
               if last or nextLast:  (this is the group's last QK of item curOrd)
                 -- the group's Q reads of the item are retired by every thread's wait_group<0>; the arrive
                 -- below is issued after the colMax bar.sync of [S] so that all 128 threads have passed it
      [S]      acc *= qkScale ; mask (:1618-1632 unchanged) ; colMax = computeWarpGrpColMax_sync(.., warpColMax[c][(g/2)%2], m, acc)
               if last or nextLast: qBar[curOrd & 1].consumed.arrive                                    (Q release, 5.4)
               scale = exp2((m_old - m_new) log2e) ; warpGrpOnlineSoftmax ; colSum ; l = l * scale + colSum (per warp)
      [P]      acc *= kE4M3_MAX ; bf16 ; stmatrix.x4 -> P[c] ; fence.proxy.async.shared::cta ; bar.sync(id_c, 128)   (C20)
      [R]      rescaleAcc(O, scale) (8 FMUL) ; gmma::fence
      [V]      vBar[g % dV].produced.arrive_and_wait
      [PV]     8 x SS wgmma (V(g), P[c]) ; commit ; NO wait ; pendV = g % dV
      [item]   if last:                                              -- finalizer
                 wait_group<0>                                       -- O complete (V(g) stays held: released at g + 2 or epilogue)
                 if not (first):                                     -- multi-tile item: combine the partner's partial
                   partialBar[nx & 1].produced.wait_parity(toParity<2>(nx))
                   read partial[nx & 1] -> combine (O, m, l) (rev 1 5.4)
                   partialBar[nx & 1].consumed.arrive ; nx++
                 fold the 4 per-warp l -> group l (slot STS, bar.sync(id_c), 4 LDS) ; publish m, l -> pubColMax / pubColSum[c]
                 finalizeAndWriteOut_sync (group barrier = bar.sync(id_c)) -> output, or scratch 2 x cta + ctaLast
                 thread 0: st.release.cta partialDone[ctaLast] = 1 (C22)
               else if nextLast:                                     -- publisher (the item has >= 2 tiles by definition)
                 wait_group<0>
                 partialBar[nx & 1].consumed.wait_parity(toParity<2>(nx))
                 fold l ; STS O (8 floats / thread), m, l -> partial[nx & 1] ; partialBar[nx & 1].produced.arrive ; nx++
    epilogue   wait_group<0> ; if pendV != none: vBar[pendV].consumed.arrive

Every participating group hits exactly one item branch per item: tiles
alternate, so a group's last owned tile of item j is `b_j` (it is the
finalizer) or `b_j - 1` (it is the publisher); a solo item has only the
finalizer branch with `first && last`.

### 4.3 Q protocol (B1, B2)

    qBar[b].produced   count 32 (Q warp, after store + fence.proxy.async)   phase n <-> Q(b + 2n) stored
                       waiters: both groups, wait_parity(toParity<2>(ord)) at item entry or catch-up; never arrive
    qBar[b].consumed   count 256 + 32                                        phase n <-> item b + 2n - 2 released (phase 0 pre-arrived)
                       arrivals per item ord: C0 once, C1 once (participant: after its last QK, at [S]; non-participant:
                       at catch-up, after the produced wait), Q warp once (arrive_and_wait before storing Q(ord + 2))

Q warp code (:2278-2298) unchanged: `consumed.arrive_and_wait` -> register
load / store -> `fence.proxy.async` -> `produced.arrive`, item counter `idxItem`
= the fill's ordinal (both enumerate the same `ItemCursor` items in order).

### 4.4 Partial exchange (N4, and rev 1's skipped-phase defect)

    partialBar[p].produced   count 128 (publisher)   phase k <-> exchange p + 2k published
    partialBar[p].consumed   count 128 (finalizer)   phase k <-> exchange p + 2k - 2 read (phase 0 pre-arrived by C0)

The exchange index `nx` counts **multi-tile items only** and is kept per group;
both groups own a tile in every multi-tile item, so both see every exchange
and their counters agree.  A solo item has no exchange and does not advance
`nx` (rev 1 indexed by the item ordinal, which a solo item would have made
skip a phase on the publisher / finalizer barrier — the parity would then be
wrong for every later exchange in that slot).  The finalizer waits `produced`
phase `nx >> 1`, the publisher waits `consumed` phase `nx >> 1` (`toParity<2>
(nx)` on both, as today's `toParity<2>(idxItem)` pattern).

### 4.5 Merge publication, converters, loaders

Unchanged from rev 1: `partialDone[isCtaLast]` per-slot flag with
`st.release.cta` / `ld.acquire.cta` (C22, N6); converters and loaders
textually unchanged except `dK = 5`, `dV = 6`, `nbKScaleTiles = 6`,
`nbVScaleTiles = 7`, `kAhead = 4`, `vAhead = 5`, and the barrier counts of
5.2.  The stage protocol they see is: one 128-arrival on `consumed[s]` per
tile from the owner of that tile, exactly as today from gemm0 / gemm1 (C8
with alternating owners).

## 5. Protocol enumeration: every arrive / wait pair, every case

### 5.1 Record bits (C23, rev 2)

`fillTileMeta` (:3310-3360) computes per tile `first`, `last` from the piece
(`x == x0 || t == 0`; `x + 1 == xEnd || t + 1 == tiles`).  Rev 2 adds:
`nextLast = (x + 2 == xEnd) || (t + 2 == tiles)` (the tile after this one is
the item's last — same piece data, no lookahead), and `ord & 7` in bits 21-23
where `ord` is a new `ItemCursor` field (:717-724) incremented when
`tileInSeq == tl` (item completed; *not* per piece, since `next(limit)` clips
pieces at chunk boundaries and one item can span two fills).  The K and V
loaders each run their own cursor and produce identical bits.  Both operands'
records carry the bits; the group reads the K record (as gemm0 does today
:1523).

### 5.2 Barrier table (C4, rev 2)

| barrier | count produced / consumed | producer arrive | consumer arrive | waits | prologue pre-arrive (phase 0) |
|---|---|---|---|---|---|
| `kBar[s]`, s < 5 | 128 + 128 K conv (+ 32 loader A) / 128 + 32 loader (A) | converters after expansion; loader `arrive_tx` (A) | owner group after QK `wait_group<0>` at tile g | owner `arrive_and_wait`; converters `wait_parity`; loader `arrive_and_wait` | `consumed`: 128 by C0 |
| `vBar[s]`, s < 6 | mirror | mirror | owner group at tile g + 2 [QK] or epilogue | mirror | `consumed`: 128 by C0 |
| `qBar[b]`, b < 2 | 32 / 256 + 32 | Q warp | C0 once + C1 once per ordinal + Q warp | groups `wait_parity`; Q warp `arrive_and_wait` on consumed | `consumed`: 128 by C0 **and** 128 by C1 |
| `partialBar[p]`, p < 2 | 128 / 128 | publisher group | finalizer group | finalizer `wait_parity`(produced); publisher `wait_parity`(consumed) | `consumed`: 128 by C0 |
| named barriers 3 (C0), 4 (C1) | 128 | | | colMax exchange, P visibility, l fold, finalize's internal syncs | |
| `kMetaReady / vMetaReady` | 32 | loader | | converters `wait_parity` | unchanged |
| `xBar[*]`, `gemm0WarpGrpBar`, `gemm1WarpGrpBar` | deleted (the latter two become `cBar[c]` for finalize's `warpGrpBar` argument :4549) | | | | |

### 5.3 Ordinal reconstruction (why d <= 2)

A group misses an item only if it owns none of the item's tiles, i.e. the
item is a solo tile of the other parity.  Two consecutive solo items sit on
tiles t and t + 1 of opposite parity, so a group can miss at most one item in
a row: between two of its owned tiles the ordinal advances by 0, 1 or 2.
Three bits (mod 8) are therefore enough; `assert(d <= 2)`.

### 5.4 Q release ordering (C15 / C21, rev 2)

The Q warp overwrites `q[b]` for item `ord + 2` after `consumed[b]` phase
`(ord >> 1) + 1` completes, which needs: C0's arrive for `ord`, C1's arrive for
`ord`, its own.  A participant's arrive is issued at [S] after the colMax
`bar.sync`, which every thread reaches only after its own `wait_group<0>` of
the item's last QK — so all 128 threads' wgmma reads of `q[b]` are retired
before the arrive.  A non-participant's arrive (catch-up) is issued after it
has waited `produced` for `ord`; it never read `q[b]` for `ord`.  No group
reads Q(`ord`) after its arrive for `ord`.  Solo items: the owner is the only
reader; the non-owner arrives at catch-up.

### 5.5 Case walk-throughs

Notation: item j occupies tiles `[a_j, b_j]`, `n_j = b_j - a_j + 1`, `own(t) =
t & 1`, exchange counter `nx` as in 4.4.

**(a) Multi-tile item j, both groups.**  Group c's first owned tile of j is
`a_j` or `a_j + 1`: ordinal advance d = 1 -> reset, `produced[j & 1]` wait.
Its last owned tile is `b_j` (finalizer) or `b_j - 1` (publisher): the
`consumed[j & 1]` arrive at [S] of that tile.  Publisher at `b_j - 1`: waits
`partialBar[nx & 1].consumed` phase `nx >> 1`, writes, arrives `produced`,
`nx++`.  Finalizer at `b_j`: waits `produced` phase `nx >> 1` (published one
tile earlier), reads, arrives `consumed`, `nx++`, finalizes, publishes
`partialDone`.  Both groups' `nx` advance by one.  V: publisher's V(`b_j - 1`)
released at its tile `b_j + 1` [QK]; finalizer's V(`b_j`) at `b_j + 2` [QK]
(or epilogue).  K: released at each tile's own [QK].

**(b) Solo item j owned by C0 (`a_j = b_j`, even).**  C0 at `b_j`: d = 1 ->
reset, wait `produced[j & 1]`, QK, `consumed[j & 1]` arrive at [S] (last),
finalize with `first && last` (no exchange, `nx` unchanged), `partialDone`.
C1 never owns a tile of j: at its next owned tile `b_j + 1` (item j + 1) it
reads `ord3 = (j + 1) & 7`, d = 2: catch-up `k = j`: wait `produced[j & 1]`
parity `toParity<2>(j)` (Q(j) was stored before C0's QK of `b_j`, so this is
already complete unless C1 is ahead of the Q warp — then it blocks until Q(j)
lands, 5.6), arrive `consumed[j & 1]`; then item j + 1: reset, wait
`produced[(j + 1) & 1]`.  `consumed[j & 1]`'s phase for item j gets C0 (at
`b_j`), C1 (at `b_j + 1`), Q warp (before storing Q(j + 2)).  Neither `nx`
advances.

**(c) Solo item j owned by C1 (`b_j` odd).**  Symmetric: C1 finalizes at
`b_j`; C0 catches up at `b_j + 1`.

**(d) Item boundary inside a group's tile pair, either group.**  Tiles `b_j`
(item j) and `b_j + 1` (item j + 1) have different owners, so one group ends
item j (finalizer) while the other starts item j + 1 (publisher-to-be or
finalizer of j + 1).  The group starting j + 1 at `b_j + 1` waits `produced[(j
+ 1) & 1]`, whose Q(j + 1) was stored after `consumed[(j + 1) & 1]` phase for
item j - 1 — both groups' arrives for j - 1 (issued at their last QK of j - 1,
tiles <= `b_{j-1}` < `b_j`) and the Q warp's.  No dependency on item j's
finalize.  The finalizer of j at `b_j` needs the publisher's partial from
`b_j - 1`, one tile earlier by the other group.

**(e) 2-tile item j (`a_j = b_j - 1`).**  Publisher = owner of `a_j`
(its first and last owned tile of j: d = 1 -> reset, wait produced, QK,
`consumed` arrive, then the publisher branch: `nextLast`), finalizer = owner of
`b_j` (`first` not set on `b_j`, so it combines).  `nx` advances on both.

**(f) CTA range start.**  C0 at g = 0: `curOrd` -1 -> ordinal 0, d = 1, wait
`produced[0]` phase 0 (Q(0) stored by the Q warp after its `consumed[0]`
`arrive_and_wait` phase 0 = pre-arrived 128 + 128 + its own 32).  C1 at g = 1:
if tile 1 is in item 0, d = 1 -> wait `produced[0]` phase 0 too; if tile 0 was a
solo item, d = 2 -> catch-up on 0 (wait `produced[0]`, arrive `consumed[0]`)
then item 1 (wait `produced[1]` phase 0).  Stage barriers: `consumed` phase 0
pre-arrived by C0 (and the loaders' own arrive in step A), converters issue
tiles 0..kAhead-1 as today.  `partialBar[*].consumed` phase 0 pre-arrived, so
the first two exchanges' publishers do not wait.

**(g) CTA range end.**  The last item's finalizer is the owner of `G - 1`; the
other group's last owned tile is `G - 2` (publisher of the last item if
multi-tile, else its own finalizer of item `last - 1`).  Epilogue of each
group: `wait_group<0>`, release `pendV`.  The loaders' `consumed`
`arrive_and_wait` at tile g needs the release of tile `g - d`, which happens at
tile `g - d + 2 <= G - 1` for every `g < G` with `d >= 3` — inside the loop; the
epilogue releases are never waited on.  The Q warp exits after storing the
last item's Q; the merge warp polls `partialDone` for the CTA's partial items
(at most two) and exits.  `qBar` phases for items > last are never waited or
arrived.

**(h) Items beginning mid-pipeline, many items per CTA, T < P (C7 class).**
Nothing in 4.2 depends on the item's position in the range or on the number of
items; every wait is on a barrier phase indexed by `ord`, `nx` or `g`, each
advanced identically by both groups (5.3, 4.4) or by the CTA-local tile
counter (C9).

### 5.6 Phase-ambiguity proofs (`try_wait.parity` is safe only if the barrier is at the awaited phase or one beyond)

- **`qBar[b].produced`, waiter c for item ord = b + 2n.**  c has waited (or
  caught up on) every earlier ordinal in order, in particular ord - 2 in the
  same slot: the barrier is at phase >= n.  It cannot be at phase n + 2: Q(ord
  + 2) is stored only after `consumed[b]` phase for ord completes, which needs
  c's own arrive for ord, issued after this wait.  Hence the barrier is at n
  or n + 1: unambiguous.  (Without the catch-up rule — B2's suggested fix — c
  could wait for ord while Q(ord - 2) is not yet stored: if ord - 2 is a solo
  item of the other group and the other group, up to dK tiles behind, has not
  yet finalized ord - 4, then `consumed` for ord - 4 is incomplete, Q(ord - 2)
  is not stored, the barrier is at phase n - 1 and the parity test for n
  passes immediately.)
- **`qBar[b].consumed`, Q warp.**  `arrive_and_wait` uses the arrival token
  (phase-exact), no parity.
- **`partialBar[p].produced`, finalizer of exchange nx.**  Phase `(nx >> 1) -
  1` (exchange nx - 2, same slot) is complete: its publisher was c (program
  order; the 128th arrive precedes c's later `bar.sync`s) or the other group,
  in which case c finalized nx - 2 and waited that phase.  Phase `(nx >> 1) +
  1` cannot be complete: the publisher of exchange nx + 2 first waits
  `consumed` phase `(nx >> 1) + 1`, which needs the finalizer of nx (c, now)
  to arrive.
- **`partialBar[p].consumed`, publisher of exchange nx at tile t — invariant
  C20b' (N4).**  Needs the finalizer of exchange nx - 4 (same slot, two
  exchanges back) to have read.  If that finalizer was c: program order.  If
  it was the other group, at tile b'''': exchanges nx - 3, nx - 2, nx - 1 each
  span >= 2 tiles and the item of exchange nx has its publisher at `b - 1`
  with `b - 1 >= b'''' + 7`, so `b'''' <= t - 7`.  c has passed `kBar.produced
  (t)`, so the converters issued K(t), so `kBar.consumed` for tile `t - dK`
  completed, so the owner of `t - dK` passed its QK `wait_group<0>` of that
  tile; **with dK odd** that owner is the other group, and its finalize at
  b'''' (an item branch of tile b'''') precedes in program order the QK of
  tile `b'''' + 2`; it is complete if `b'''' + 2 <= t - dK`, i.e. **dK <= 5**.
  `static_assert(dK % 2 == 1 && dK <= 5)`.  (A flag-based exchange —
  monotone `partialFree = nx` with `ld.acquire` — would remove the coupling
  dependence; not chosen because the mbarrier form matches today's code and
  the bound holds with margin 0 at dK = 5; the assert makes it explicit.)
- **`kBar / vBar` `wait_parity` by converters and `consumed` by loaders**:
  unchanged from today (C8): exactly one 128-arrival per tile on `consumed[s]`
  (the owner), phases advance once per tile, the converters' parity index is
  `t / d` as today.

### 5.7 Liveness

Every wait is on an action at a strictly earlier tile of the CTA or on the Q
warp / converters / loaders, whose own waits are on consumer releases of
strictly earlier tiles: `produced[k]` (Q warp, needs `consumed` for k - 2:
groups' arrives at tiles < a_k), `partialBar` (partner at b_j - 1 < b_j;
finalizer of nx - 2 at a tile before the publisher of nx), stage barriers
(C9).  The catch-up wait of a non-participant on `produced[k]` is on Q(k),
which the item's owner needed before its QK of k's only tile, one tile
earlier.  No cycle.

## 6. Budgets, rev 2

### 6.1 Registers — unchanged (N2)

Persistent 36 + transient <= 16 -> peak ~52, fits 64 under the uniform 96
cap of `__launch_bounds__(640, 1)` with no `setmaxnreg` (C7507 class cannot
arise).  The rev 2 state adds `curOrd`, `nx`, `pendV` (3 registers, inside the
"loop" line of the rev 1 table).  Today's ptxas for the production source
(compile-only, this worktree): fp8 / fp4 / mixed REG 48, 0 stack, 0 spill,
4 696 / 4 512 / 5 720 SASS.

### 6.2 Shared memory — unchanged (N3)

dK = 5, dV = 6: ~206 400 B production (~207 400 trace) <= 232 448; one CTA per
SM by smem alone.  N3's side effect recorded: L1 shrinks to 232 - 206 = ~26 KB
(+ carve-out granularity); the Q / record / page-table loads are `ld.global`
through L1 (small), the page bytes are `cp.async.cg` (L2 only) and TMA — a
confirmation run would read `l1tex__t_sector_hit_rate` for the loaders' and
merger's loads.

### 6.3 Issue budget per tile-SM (replaces rev 1 5.5; every number from 2.2 / 2.3)

| item | today, fp8 P132 (per tile-SM) | fused design (step A) | fp4 today -> fused | mixed today -> fused |
|---|---:|---:|---:|---:|
| gemm0 + gemm1 body (incl. 668 descriptor / addressing uniform ops) | 1 462 | ~1 190 | 1 450 -> ~1 180 | 1 462 -> ~1 190 |
| converters, 8 warps | 2 632 | 2 632 | 2 382 -> 2 382 | 2 044 -> 2 044 |
| IO warps (fills, Q, merge, TMA in mixed) | ~75 | ~75 | ~75 | ~160 |
| item-level (finalize, partial exchange ~90 per item x 2-3 items / 66 tiles) | ~30 | ~35 | ~30 | ~30 |
| **total** | **~4 200 (ncu 4 150)** | **~3 930 (-6.5 %)** | 3 940 -> 3 670 | 3 700 -> 3 430 |
| converter share | 63 % | **67 %** | 60 % -> 65 % | 55 % -> 60 % |

Per scheduler (5 warps: 2 converter, 2 consumer, 1 IO), fp8 fused: 658 + 298 +
~25 = ~980 issue slots per tile; at a 0.79 us cadence (1 564 cycles) that is
63 % issue-active per scheduler — the pair's level (63 %) with half its warps
(the lone CTA today runs 56 %).

## 7. Predicted cadence and wall, rev 2

### 7.1 Issue model

    cadence = N / (IPC x 1.98 GHz),  N from 6.3,  IPC from 2.4

| mode | N fused | IPC 2.07 (launch avg, pair) | IPC lone 20 warps (2.21 / 2.27 / 1.88) | IPC paired 40 warps (2.48 / 2.65 / 2.24) |
|---|---:|---:|---:|---:|
| fp8 | 3 930 | 0.96 | **0.90** | **0.80** |
| fp4 | 3 670 | 0.86 | **0.82** | **0.70** |
| mixed | 3 430 | 0.89 | **0.92** | **0.77** |

The fused SM has today's lone warp count (20) with two independent consumer
chains instead of today's two pipelined stages; nothing measured says whether
that alone lifts the IPC from the lone value toward the paired value, which is
exactly B5 (a)/(b).  The converter bound decides where in the band the design
can sit.

### 7.2 Converter feasibility (2.5)

The eight converter warps must each issue 329 (fp8) / 297 (fp4) instructions
per tile; their cycles-per-instruction rises with the SM's issue load (4.0
lone at IPC 2.21; 5.8 fast-member at IPC 2.48 with 10 warps per scheduler).
Requiring `converter busy <= cadence` at the same IPC:

| scenario (fp8) | IPC | cadence (issue model) | converter cyc/instr (linear between the two measured points) | converter busy per tile | feasible |
|---|---:|---:|---:|---:|---|
| lone-CTA issue rate | 2.21 | 0.90 | 4.0 | 0.66 | yes (converters 73 % busy) |
| | 2.35 | 0.845 | 5.0 | 0.83 | marginal |
| pair issue rate | 2.48 | 0.80 | 5.8 | 0.96 | **no** (converter warp longer than the tile) |

The self-consistent point is **IPC ~2.35, cadence ~0.84 us (fp8)**; the pair's
own converters at 40 warps needed 0.96-1.30 us per tile per warp and were
hidden only because two CTAs' tiles alternate on the SM (each CTA's tile is
1.7-2.0 us long).  At 5 warps per scheduler the converters' `not_selected`
share should be lower than at 10, so the linear interpolation is a pessimistic
side; the band is taken as **0.80-0.89 (fp8)**, central 0.83-0.85.  fp4: 297
instructions at 4.2-6.3 cyc -> busy 0.63-0.95; crossing with the issue model
(N 3 670, IPC 2.27-2.65) at ~0.77-0.80.  mixed: converters at 2/3 duty are
not binding (busy 0.50-0.73); the issue model with mixed's fetch-bound IPC
(1.88 lone, 2.24 paired; mixed doc 1.1: `no_instruction` stalls +55 %)
gives 0.77-0.92, central ~0.83.

### 7.3 Chain sanity check (measured segments only, N5)

Fused chain per group = K-wait tail 0.17 + QK 0.37 + colMax / softmax 0.19 +
P store 0.14 (X store 0.274 minus the stamped xBar RT 83-233 cyc, x 0.81) +
register rescale 0.03-0.06 (today's stamped rescale 0.238 x 0.81 = 0.19 is
the ballot form) + V wait ~0.05 + PV issue 0.30-0.39 = **1.25-1.37 us**; two
groups without interference 0.63-0.69; with the P0.5 / pair stretch band
1.19-1.35 -> **0.75-0.93**.  Consistent with 7.1-7.2 and not tighter; the
[15] doc's 1.55 us fused-chain figure (B5 (c)) gives 0.92-1.05 and is the
pessimistic edge.  The chain model does not bind; the converter / issue
bound does.

### 7.4 Wall

    wall = fill(1 CTA/SM) + 66 x cadence + tail        fill 7.0 / 6.7 / 7.2 / 10 (P = 132 trace), tail 2.8 / 2.6 / 2.8 / 3

| mode | today | cadence central (band) | **predicted** (band) | target | gain (central) |
|---|---:|---:|---:|---:|---:|
| fp8 | 67.9 | 0.83 (0.80-0.89) | **64.6** (62.6-68.5) | <= 58 | -3.3 |
| fp4 | 60.6 | 0.78 (0.75-0.82) | **60.5** (59.2-63.9) | <= 36 | +0 |
| mixed | 64.6 | 0.815 (0.75-0.89) | **63.8** (59.5-68.7) | <= 62 | -0.8 |
| a16 | 79.4 | DRAM-bound body 68 | **~80** (79-81) | parity | 0 |

Step B (no IO warps, 16 warps) removes ~75 instructions per tile (2 %):
<= -0.02 us per tile, inside the band.  The fill cut is rejected (round 3) and
not assumed.  **No target is inside the central case; fp8 <= 58 is outside
the band; mixed <= 62 needs the optimistic third of the band.**  Rev 1's
central 62.0 / 58.1 / 62.8 came from the halved converter count and the
unmeasured chain deltas.

### 7.5 Why the pair beats the lone CTA, restated with the measured budget

The SM's per-tile work is 4 150-4 213 warp-instructions in both layouts; the
pair issues them at IPC 2.48 because 40 warps give 1.58 eligible warps per
scheduler, the lone CTA at 2.21 with 0.97.  The consumer chains are 35 % of
the instructions and, at one CTA per SM, mostly *wait*: their fusion shortens
the chain but cannot raise the SM's IPC beyond what the converter warps —
63 % of the instructions, latency-bound at 4-6 cycles per instruction on
their own dependency chains (LDS -> F2FP / PRMT -> HMUL2 -> STS, P0.4) — can
supply.  The pair's advantage is two CTAs' converter sets alternating; one
CTA's eight converter warps cannot alternate with themselves.

## 8. Verification artifacts that would gate a build (kept for the record; corrected gates)

Had the lever been a go, the pre-run gates would be, in this order:

1. `ptxas -v` at `(640, 1)`: 0 stack, 0 spill; per-role REG from the lineinfo
   split (`/tmp/r3pair_regs.py` style): consumer <= 64, converters <= 64, IO
   <= 48; no `USETMAXREG`.
2. **SASS body count per tile-SM from the lineinfo listing, by role** (the
   `/tmp/r4p7_body.py` method on a `SourceCounters` run of the *production*
   kernel is the baseline; for the new build the static loop body per role,
   unrolled and straight-line as today): consumer body <= 1 200 per tile
   (today 1 462), converter body = today's 2 632 +- 8 (unchanged code), total
   <= 3 950 (fp8).  If the consumer body is not <= 1 250 the removal did not
   happen; do not run.
3. `cuobjdump -sass` counts: HGMMA 16 in one consumer body; `SYNCS.PHASECHK`
   in the loop = 2 (K, V) + item-level (qBar produced, partialBar);
   `SYNCS.ARRIVE` in the loop = 2 (K, deferred V) + item-level (qBar consumed,
   partialBar); `BAR.SYNC` per tile = 2 (ids 3 / 4 by group); `STSM` 1;
   `FENCE.VIEW.ASYNC.S` 1; no `VOTE` / `SHFL` in the rescale; `LDL = STL = 0`;
   converter bodies byte-identical to today's except ring immediates
   (`toParity<5>` / `<6>`); `UTMALDG` 8 (a16, mixed) / 0.
4. `cuobjdump -res-usage`: STACK 0; `sizeof(SharedMem)` 206 4xx (207 4xx
   trace); occupancy calculator 1; grid 132.
5. Conformance: `tests/attention/run_xqa_mixed_page_transport.py`, all cases,
   **plus** the cases of 8.7 below.
6. Trace (`MIXED_KV_TRACE 1`, per-group stamps): cadence = stamp(g) ->
   stamp(g + 2) halved, medians over tiles 4-60; K / V wait segments; converter
   `ready` (idle) segment — if < 15 % of the cadence the converters pace (7.2
   predicts exactly this); partial publish -> combine-done <= 0.3 us; catch-up
   waits (solo items) visible only in the short-sequence cases.
7. ncu (fp8, `--launch-skip 4`): `smsp__warps_active` ~5 per scheduler,
   `smsp__issue_active`, `sm__inst_executed` <= 34.4 M (-7 %),
   `l1tex__t_sector_hit_rate` for the IO loads (N3).

Do-not-build conditions of rev 1 carried over with corrected values: 1
(registers), 2 (instruction count, now: consumer body > 1 250 or total >
3 950), 3 (smem), **4 (converter share > 45 % of the per-tile-SM total: met
today at 63 %, 67 % after fusion — this is the condition that makes the lever
a no-go)**, 5 (occupancy), 6 (rescale needs SHFL / VOTE), 7 (conformance
matrix lacks 1- and 2-tile items at both parities: S = 64 / 128 with
`XQA_PERSISTENT_CTAS` 1 / 3 / 5 — required cases: solo owned by C0, solo
owned by C1, two consecutive solos, 2-tile item at each parity, solo as the
CTA's first item, solo as the CTA's last item), 8 (bit-identity demand), 9
(step B before step A's trace), 10 (P = 132 control re-read).

## 9. Go / no-go

**No-go.**  Reasons, in the order they were established:

1. **The budget, measured, is not the one the lever assumed.**  The SM's
   per-tile work is 63 % converter instructions (2 632 of 4 150, fp8), 35 %
   consumer (1 462, of which 46 % is wgmma-descriptor / addressing arithmetic
   the fusion does not touch), 2 % IO.  The fused consumer removes ~270
   instructions per tile: -6.5 % of the SM's issue demand (2.2-2.3, 6.3).
2. **The cadence is converter-bounded before it is chain-bounded.**  Each
   converter warp needs 329 instructions per tile at 4.0-5.8 cycles each =
   0.66-0.96 us; the issue model's optimistic end (0.80 at the pair's IPC)
   is infeasible for the converters at that IPC, and the self-consistent
   cadence is 0.83-0.85 us (7.2).  Predicted walls fp8 64.6 (62.6-68.5), fp4
   60.5, mixed 63.8 (59.5-68.7): -3.3 / 0 / -0.8 us central, no target
   inside the central case, fp8 <= 58 outside the band (7.4).
3. **Rev 1's own gate 4 fails on the production kernel** (converter share
   > 45 %), and the ordering it prescribes — cut the converter instruction
   count first — is what the measurement says as well.
4. **The protocol is now sound but not cheap**: strict item-ordinal sequence
   with catch-up, an exchange counter, a dK-parity coupling invariant (C20b')
   and eleven new or restated invariants (sections 4-5) for a lever whose
   central gain is inside the band's zero for two of three modes.  Boring
   kernels with the right structural property are the rule; this one has the
   structure (one loop, two groups meeting only at item ends) but not the
   property (the SM's bottleneck is elsewhere).

What the measured budget points at (recorded, not designed here — each is a
separate lever with its own design, data flow and gates):

- **Converter copy issue**: 90 (fp8) / 71 (fp4) instructions per warp per tile
  for 5 / 3 `LDGSTS` = **720 / 570 per tile-SM = 17 % / 15 %** of the SM's
  instructions, i.e. more than the whole gain of [7]; the q=4 build's
  `copyMixedPartialHeadsAsyncHoisted` ([44] item 3: per-lane / per-page
  address constants hoisted, `[R+imm]` destinations, one 64-bit add per
  iteration) is the existing precedent for a ~-50 % cut on this path.
- **Converter expansion**: 188 x 8 = 1 504 per tile-SM = 36 %; the [16] audit
  (backends doc Phase 2) left ~40 non-essential of 188 per lane-tile; a
  further cut needs a decode with fewer than 32 PRMT per 8 values or the
  64-register layout of [15] (constants kept live).
- **Consumer descriptor arithmetic**: 668 per tile-SM = 16 %: the 8 + 8
  `addAddr` descriptor recomputations per tile are uniform-datapath ops that
  a base-descriptor-plus-immediate form (CUTLASS's `DescriptorIterator`
  pattern: one 64-bit add per k-step, or immediates on a per-stage base)
  would cut to ~1/4.  This is also the only consumer-side lever that does not
  need [7]'s protocol.

These three together are ~2 900 of 4 150 instructions per tile-SM; a -25..-30
% cut of the SM's work at the pair's 2.48 IPC would move the fp8 body from
56.6 toward ~42-45 us — the first estimate on record that reaches the fp8
target's neighbourhood, and it applies to the *existing* two-CTA layout
without a new protocol.  [7] should be re-evaluated only after those cuts,
when the consumer chain may again be the bound.

Predicted for the record (step A, central): **fp8 64.6, fp4 60.5, mixed 63.8,
a16 ~80 us**; cadence 0.83-0.85 us per tile-SM (band 0.80-0.89).  Not built.

## Appendix: artifacts of this revision

- ncu reports read (production kernel, r2p8 checkout, 2026-09-04 15:27-15:28
  UTC): `/tmp/r4mixed2_pcs_{fp8_P132,fp4_P132,fp4_P264,mixed_P132}.{ncu-rep,
  csv,details}`; `smsp__inst_executed.sum` 36 161 737 / 33 808 585 /
  34 330 702 / 31 188 740; `smsp__issue_active` 0.56 / 0.56 / 0.65 / 0.48;
  `gpc__cycles_elapsed.avg` 130 274 in 94.6 us (1.38 GHz lock).
- Lineinfo listings: `/tmp/r2p8_ptx/li{1,2,-1}.nvdis` (r2p8 source); new
  compile-only builds of this worktree's source `/tmp/r4p7_sass/f{1,2,-1}
  {,_li}.{cubin,sass,nvdis,ptxas.log,res}` (fp8: 4 696 SASS, REG 48, STACK 0;
  lineinfo SASS text identical to the production text).
- Scripts (nkcut2): `/tmp/r4p7_roles.py` (per role / opcode / source line),
  `/tmp/r4p7_body.py` (body / uniform / spin / rare split), `/tmp/r4p7_sass/
  build.sh` (recipe: `/tmp/main_ptx/ninja_flags.py` flags with the checkout
  path and `-DMIXED_PAGE_STATIC_FORMAT` substituted, `-lineinfo -cubin`).
- Remote checkout `/home/bigboi/dash-flashinfer-claude-r4p7` (rsync of this
  worktree, symlinks set); no workspace built, no GPU job started.
