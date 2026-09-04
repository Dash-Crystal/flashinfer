# Round 3 — co-resident CTA pair asymmetry: attribution and per-SM cooperative tile pull (design)

Kernel: `csrc/xqa/mha_sm90.cu` at `e113026a` (kernel state [8], merged at
`039ba5c7`), `ENABLE_MIXED_KV_CACHE && !SPEC_DEC` q=1 build.  Bench shape B=17,
S=4096, 8 KV heads, GQA 4, D=128, q=1.  Line references are into `mha_sm90.cu`
of worktree `wt/r3pair` (this commit: production code identical to `e113026a`;
only `MIXED_KV_TRACE` code and the host `XQA_TRACE_CLUSTER` probe were added).

> **Amended after review (sections 9-10).**  Section 9 answers the judge
> blockers and supersedes sections 3.2, 3.3, 3.5-3.8, 6 and 7 wherever it
> differs; section 10 is the "as written" record of the kernel.  The headline
> changes: the wall model is a paired / lone **two-rate** model (ceiling ~2 us,
> base protocol ~1.3 us body, ~0.6-0.7 us net after the merger's tag round
> trip; the mixed target is **not** reachable by this lever); the merger reads
> **no pool word** (per-chunk sequence tags + tile-count arrivals replace the
> weights and the pool-word enumeration, which also removes the SOLO hazard);
> claims are one `atom.add` (no CAS on the per-2-tile path), issued in a loader
> iteration of their own, so every loader iteration carries at most one global
> round trip against its one-period slack; the rendezvous runs in the merge
> warp after the `__syncthreads`, off the prologue path.

State after [8] (nkcut2 H200, locked, medians): a16 78.8, fp8 67.8, fp4 60.5,
mixed 64.4 us.  Targets fp8 <= 58, fp4 <= 36, mixed <= 62.

This document is the design that precedes the change (method rule: attribution
by reading first, data-flow / control-flow design second, build checks third,
one confirmation run last).  No production kernel edit was made in this phase.

## 1. Attribution: the slow CTA of every SM is the one in the upper warp slots

### 1.1 Trace instrumentation added (trace builds only, committed)

- `TRACE ctarec` gains `warpid` (`%warpid` of warp 0 = the CTA's physical warp
  slot base), `cluster`/`rank` (`%clusterid.x`, `%cluster_ctarank`, 0 without a
  cluster launch) and 16 **per-role segment accumulators**: `traceStamp` now
  runs for every tile (not only the 8-tile print window) and adds
  `now - lastStampOfThisRole` into `smem.traceAcc[slot]` (:1326-1358, +176 B of
  shared memory, trace build 114 864 B, still 2 CTAs/SM).  `traceAcc[s]` is
  therefore the total time a role spent in the segment that ends at stamp `s`,
  for all 264 CTAs of a launch, so the fast / slow sets can be compared per
  role and per segment without choosing a CTA to print.
- Host: `XQA_TRACE_CLUSTER=n` (trace + persistent builds only)
  launches with cluster dimension `n` for the placement probe of 1.5 (:5168-5187).
- Parser: `benchmarks/parse_xqa_pair_trace.py` pairs CTAs by `smid`, splits
  each pair into fast / slow by body time, and prints per-set medians of fill /
  body / tail and of every segment (us and ns per tile); a launch with one CTA
  per SM is reported as `lone`; cluster launches print SMs per cluster and the
  slot pattern.

Caveat on absolute numbers: the accumulator stamps (six lanes doing
`CS2R + LDS + IADD + STS x2` per stamp, 2-4 stamps per role per tile) slow the
traced kernel by ~0.3 us/tile (fast body 56.6 us here vs 46.5 us with the [8]
`ctarec`-only trace).  Everything below is read as **ratios within one launch**
(fast vs slow vs lone), not as production absolutes; production absolutes are
the [8] confirmation numbers (fp8 46.5 / 57.1 us, fp4 44.7 / 53.8).

### 1.2 Per-slot evidence (fp8, 3 launches; fp4 identical in shape)

Runs: `/tmp/r3pair_trace.log` (P = 264), `/tmp/r3pair_trace132.log`
(`XQA_PERSISTENT_CTAS=132`, one CTA per SM, 66 tiles each),
`/tmp/r3pair_trace2.log` (`MIXED_KV_TRACE 2`: K-converter copy-issue / landing
split), `/tmp/r3pair_cluster{2,4,8}.log` on nkcut2; copies under
`/tmp/r3pair/` locally.

| launch | pairs | slow member has the higher blockIdx | slow member has the higher `%warpid` | warp slots fast / slow | body(slow) - body(fast) |
|---|---|---|---|---|---|
| fp8 0 / 1 / 2 | 132 each | 132 / 132 / 132 | 132 / 132 / 132 | {0..3} / {20..23} | 8.6 / 8.4 / 9.2 us |
| fp4 0 / 1 / 2 | 132 each | 132 / 132 / 132 | 132 / 132 / 132 | {0..3} / {20..23} | 8.5 / 8.2 / 8.8 us |
| a16 0 / 1 / 2 | 132 each | 56 / 76 / 59 (random) | 56 / 76 / 59 | mixed | 1.8 / 1.6 / 2.6 us |

- The slow CTA is **the one whose warps occupy physical slots 20..39** of the
  SM (its warp 0 reads `%warpid` 20-23; the fast CTA's warp 0 reads 0-3), in
  396 of 396 fp8 pairs and 396 of 396 fp4 pairs.  The `%globaltimer` start
  order does **not** decide it (slow started later in only 47 / 66 / 65 of 132
  pairs; start deltas within +-128 ns).
- The slow set by blockIdx is ids 32-39 and 140-263 (fast: 0-31, 40-139),
  identical in every launch and mode: the dispatcher's SM order is not the id
  order, which is why the [8] doc saw a "140 boundary".  So a pairing by
  blockIdx (`c`, `c + 132`) would form 8 both-fast and 8 both-slow pairs
  (measured: max-of-pair body 67.7 us vs 65.1 median; harmonic-balanced 60.6) —
  the wall would not move.  **Pairing must be by SM.**
- Fill (start -> first K ready) is the same for both members (8.0 vs 8.7 us);
  the asymmetry is in the body only.  Tail: the fast member's tail is longer
  (3.5-3.8 vs 2.5 us) because it is the one that waits for the other's partials.

### 1.3 Which segments stretch (fp8 launch 0, ns per tile; lone = P 132 control)

| segment (accumulator) | lone (1 CTA/SM) | fast (slots 0-19) | slow (slots 20-39) | slow / fast | lone -> fast |
|---|---|---|---|---|---|
| g0 K-wait (xarr(t-1) -> kwait(t)) | 212 | 261 | 262 | 1.00 | +23 % |
| g0 kwait -> mma (8 HGMMA + wait, Q wait on item start) | 455 | 580 | 672 | **1.16** | +27 % |
| g0 mma -> smax (colMax exchange, softmax) | 233 | 325 | 399 | **1.23** | +39 % |
| g0 smax -> xarr (X store, xBar) | 274 | 346 | 424 | **1.23** | +26 % |
| g1 V-wait | 234 | 282 | 340 | 1.21 | |
| g1 vwait -> xwait | 241 | 380 | 432 | 1.14 | |
| g1 xwait -> rs (rescale) | 238 | 294 | 320 | 1.09 | |
| g1 rs -> mma (8 PV HGMMA + wait) | 484 | 600 | 704 | **1.17** | +24 % |
| K conv done -> ready (issue t+2, land t) | 765 | 894 | 1017 | 1.14 | |
| **K conv expand** | 383 | 552 | 707 | **1.28** | **+44 %** (slow: +85 %) |
| V conv done -> ready | 763 | 896 | 1021 | 1.14 | |
| **V conv expand** | 409 | 602 | 742 | **1.23** | +47 % |
| body per tile | 1.331 us | 1.715 us | 1.973 us | 1.15 | |

Level-2 split of the K converter (fp8, `MIXED_KV_TRACE 2`, ns per tile):

| segment | lone | fast | slow | slow / fast |
|---|---|---|---|---|
| done -> copies issued (parity wait + `issueCompressedPageCopies`) | 521 | 600 | 670 | 1.12 |
| **issued -> landed (`cp.async` wait_group)** | 155 | 222 | 234 | **1.05** |
| landed -> ready (`__syncwarp`) | 99 | 130 | 134 | 1.03 |
| expand | 391 | 543 | 688 | 1.27 |

a16 control (launch 1): gemm0 kwait->mma 519 vs 521, expand 136 vs 139 ns per
tile — every segment equal within 2 %; the a16 tile is 40 % K-wait (792
ns/tile: DRAM-bound), so the SM's issue slots are not contended and no slot
priority is visible.

ncu (fp8, production r2p8 build, one launch, clocks locked by ncu at 1.38 GHz
so durations are not comparable to the bench):

| metric | P = 264 (2 CTAs/SM) | P = 132 (1 CTA/SM) |
|---|---|---|
| `smsp__issue_active` (% of cycles a scheduler issued) | 62.9 | 56.3 |
| eligible warps per scheduler per cycle | 1.58 | 0.97 |
| active warps per scheduler | 9.41 | 4.99 |
| warp cycles per issued instruction | 15.0 | 8.9 |
| `sm__inst_executed` per cycle | 2.51 | 2.24 |
| shared wavefronts / bank conflicts | 9.58 M / 0.89 M | 9.66 M / 1.04 M |

### 1.4 Reading

1. **Mechanism: fixed-priority arbitration by physical warp slot** between the
   two co-resident CTAs (candidate (1) of the task, extended to every
   shared SM pipe, not only the converters).  Every *work* segment of every
   role — HGMMA issue + wait, the colMax/softmax ALU + `bar.sync` exchange, the
   X store, the converters' LDS/decode/STS expansion — is 16-28 % longer in the
   slots-20..39 CTA, while the *memory* segments are equal: `cp.async` landing
   222 vs 234 ns (+5 %), gemm0's K-wait 261 vs 262.  Both CTAs run identical
   code on identical work; the SM's four schedulers issue 63 % of cycles with
   1.58 eligible warps on average, i.e. ties between eligible warps are
   frequent, and the tie-break favours the lower slot: the CTA dispatched
   first onto the SM (slots 0-19) wins them.  The expansion, the most
   issue-dense segment (LDS.128, F2FP/PRMT, HMUL2, STS.128 back-to-back), shows
   the largest ratio (1.28); the lone control shows the same segment at 383 ns,
   so sharing the SM costs the fast CTA +44 % and the slow one +85 % on it.
2. **Not the memory system, not the tile range, not the SM, not the
   co-tenant.**  Landing latency is equal in both members (1.3); the slow set
   follows the warp slot, not the range (`MIXED_KV_TRACE_REVERSE_RANGES` in the
   [8] confirmation: slow ids unchanged while ranges moved) and not the SM (132
   of 132 SMs hold exactly one of each, every launch); a16 (same grid, same
   co-tenant, same start burst) has no slot correlation; co-tenant time slices
   appear as 1-8 outlier CTAs per launch with tails of 25-280 us and are
   excluded.  LSU / smem-port arbitration (candidate (2)) is part of the same
   slot-priority mechanism (the expansion's LDS/STS share the MIO queue) and is
   not separable from issue arbitration with these stamps; it does not change
   the fix.  Barrier polling (candidate (3)): the wait segments of the slow
   member are not longer than the pipeline stretch, and total issue is 63 %,
   so polls are not the contention; `mbarrier.try_wait` suspends in hardware.
3. **Consequence for the wall.**  The static partition gives both members 33
   tiles; the wall follows the slow one: fp8 8.5 fill + 57.1 body + 2.8 tail =
   68.4 ~ 67.8 measured.  The SM's *pair throughput* is 66 tiles / 57.1 us with
   the fast member idle for 10.6 us; the lone control (1 CTA/SM) is 1.33
   us/tile in the trace build, i.e. 2 CTAs/SM still delivers 1.45x the SM
   throughput of one, so the occupancy stays at 2 and the fix is to give the
   slots-0..19 CTA more tiles.
4. **Software cannot move the priority.**  Warp slot `w` maps to SM sub-partition
   `w % 4`; both CTAs place 5 warps on every sub-partition whatever the role
   layout inside the CTA, so no warp-id remap separates the two CTAs'
   converters (fix (b) of the task is infeasible; recorded).  Thread-block
   clusters do not help either (1.5).

### 1.5 Cluster placement probe (why the pair cannot be a cluster)

`XQA_TRACE_CLUSTER=2 / 4 / 8`, fp8, two launches each: the CTAs of a cluster
land on **distinct SMs** in every cluster (2 -> 2 SMs, 4 -> 4, 8 -> 8), and the
cluster is slot-homogeneous: with size 2, 66 clusters are both-fast and 66
both-slow; with size 4, 31-33 all-fast, 30 all-slow, 3-5 mixed; with size 8 the
2-CTA/SM uniformity breaks (124 SMs used, some with 3 CTAs).  A cluster-local
pool (DSMEM atomics, no global protocol) would therefore balance nothing.  The
bimodality itself is unchanged under clusters (fast 1.71 / slow 1.97 us per
tile), confirming it is per-SM slot priority.

### 1.6 Rejected fixes (so they are not re-proposed)

- Static asymmetric split by dispatch slot (ids by the observed pattern):
  fragile (8 + 8 CTAs already violate the "second half is slow" pattern), the
  ratio is mode-dependent (fp8 1.228, fp4 1.204) and would be tuned — banned.
- `%warpid`-based role choice: PTX defines `%warpid` as volatile /
  diagnostic; a design whose *correctness* depends on it is out.  It is used
  here only as evidence.
- End-of-range stealing (fast CTA takes the partner's remaining tiles when it
  finishes): the pipeline lookahead (records 4 tiles ahead, copies 2 ahead, 3
  stages: ~9-12 claimed-but-unfinished tiles ~ 15-20 us of the slow CTA) is as
  large as the imbalance (10.6 us) — nothing is left to steal.  Balancing must
  run from the first tile.
- Any scheme whose per-CTA tile set is not contiguous (single shared counter,
  both CTAs forward): an item per claimed unit, a finalize + partial write per
  2 tiles (+14 % on gemm1) and ~30 partials per sequence.
- One 1280-thread CTA with two pipelines sharing a smem queue: over the 1024
  threads/CTA limit.

## 2. Current data flow and control flow of the touched roles (as written)

Per-tile machinery (stages, barriers, converters, gemm0/gemm1 per-tile path)
is untouched by this design; only the *work decomposition* changes.  Touched:

- **Prologue scan** (IO warp 3, `persistentPrologueScan` :3286-3362, called at
  :1437 before the `__syncthreads` :1451): T = H x sum tiles(r); this CTA's
  range `x0 = ceil(cT/P)`, `x1 = ceil((c+1)T/P)`; `req0/head0/tile0/Lseq0/
  seqLen0/seqLen1` of x0; published in `smem.sched` (:405-413).  Every role's
  loop bound is `nbCtaTiles = x1 - x0` (:1461).
- **ItemCursor** (:725-790): register walker `{x, xEnd, x0, req, head,
  tileInSeq, seqLen, nextSeqLen, Lseq}`, `next(limit)` emits the current item
  clipped to `min(xEnd, limit)` with `partial = !(Lseq >= x0 && Lseq + tl <=
  xEnd)`, `ctaLast = (Lseq + tl >= xEnd)`.  Forward only.
- **Record fill** (`fillTileMeta` :3364-3442): 16-tile chunk, lane owns entries
  (tile lane/4, page lane%4) and (+8); phase A walks pieces with the cursor and
  computes the tile word `validBeg | validEnd << 8 | first << 16 | last << 17 |
  partial << 18 | ctaLast << 19` (:397-400); phase B issues the page-table pair
  per lane; phase C gathers formats and writes the 32 B record
  (`tileRecordAddr` :794: slot `g % 32`).
- **K / V loaders** (IO warps 0 / 1, :2186-2316): prologue `fillTileMeta(chunk
  0)` + `kMetaReady[0].arrive()` (:2195-2196); per tile `stageBar[stage].
  consumed.arrive_and_wait()` (:2222 / :2290), (a16/mixed module) elected TMA
  boxes, then `ahead = g + MIXED_KV_META_LEAD; if (ahead % 16 == 0 && ahead <
  nbCtaTiles) fillTileMeta(ahead) ; metaReady[(ahead/16) % 2].arrive()`
  (:2269-2273 / :2300-2304).
- **Converters** (:2701-2760 K, :2768-2800 V): `issueKCopies(t)`: at `t % 16
  == 0` `kMetaReady[(t/16) % 2].wait_parity(toParity<1>(t/32))` (:2703-2707);
  `kBar[t % 3].consumed.wait_parity(toParity<3>(t))`; copies from record t.
  Loop over `g < nbCtaTiles` with `kAhead = 2`.
- **gemm0** (:1544-1580): after `kBar.produced` wait reads the tile word
  (:1562), `tileFirst` -> `runningColMax` reset + `qBar[j&1].produced` wait
  (:1566-1569); `tileLast` -> `qBar.consumed.arrive`, `j++` (:1735).
- **gemm1** (:1792-2100): `tileFirst` -> acc reset (:1814); `tileLast` (:2032):
  publish colMax/colSum, `reqHead = ldsU64(rec + 24)`, `itemIsPartial` ->
  scratch chunk `2 * persistentCtaIndex() + isCtaLast` (:2050-2051,
  `ScratchMem{scratch, 2 * gridDim.x, 1}`), else direct output; then
  `st.release.cta finalizedItems = j + 1` (:2082).
- **Q warp** (IO warp 2, :2317-2337): own cursor, per item `QCvt::load`,
  `qBar[j&1].consumed.arrive_and_wait`, store, fence, `produced.arrive`.
- **Merge warp** (IO warp 3, :2339-2500): own cursor; per partial item:
  `c0 = floor(Lseq P / T)`, `c1 = floor((Lend-1) P / T)` (:2357-2358),
  `nbPartials = min(c1 - c0 + 1, tiles)` (:2364); poll `finalizedItems > j`
  (:2367-2378, `__nanosleep(1000)`); lane 0 `atom.acq_rel.gpu.inc
  semaphores[idxSeq], nbPartials - 1` (:2383-2385); last arriver reads chunks
  `{2c+1 | c0 <= c < c1} + {2c1 + (x_{c1+1} == Lend)}` (:2401-2404), combines,
  writes output.
- **Host** (:4922-4938, :5023-5029, :5141-5152): `P = ctasPerSm x SMs`
  (`XQA_PERSISTENT_CTAS` override), `dimGrid = {P, 1, 1}`, scratch `2P`
  chunks, semaphores `H x B` words in `workspace[:8 MB]` (zero at first use).

Invariant that the design relies on and that the current code already
satisfies (stated here because the new design removes the `metaReady`
barriers): **the loader's per-tile `consumed.arrive` at iteration g is
program-ordered after every fill it issued at iterations <= g - 1**, and the
converters' `consumed.wait_parity` for tile g is an acquire on that arrive.
The fill of tile g happens at loader iteration g - LEAD <= g - 1, so the
converters' record read at `issue(g)` is ordered after the fill by the
`consumed` chain alone; `kMetaReady` is a second, redundant edge (C8 of [8]).

## 3. New data flow and control flow: per-SM two-ended tile pool

### 3.1 Work decomposition

Unchanged static ranges `R_c = [x_c, x_{c+1})`, `x_c = ceil(cT/P)` (C13: 33
or 32 tiles for the bench).  Each range is split into a **prefix**
`[x_c, x_c + F)`, `F = 8` tiles, that only its owner ever processes, and a
**pool part** `[x_c + F, x_{c+1})`.  Pairing is *enabled* for the launch iff
`floor(T/P) >= 2F` (every range has a pool part of >= F tiles; the bench: 32
>= 16); otherwise every CTA runs today's static path (its whole range is its
prefix; every rule below degenerates to [8] with `W = 1`).

Two co-resident CTAs A (the **leader**, the first of the two to register on
the SM) and B (the **partner**) share a **virtual pool** = pool(R_A) ++
pool(R_B) of `n = (|R_A| - F) + (|R_B| - F)` tiles (50 for the bench):

    virtual v in [0, |R_A| - F)              <->  real x_A + F + v          (A's pool part)
    virtual v in [|R_A| - F, n)              <->  real x_B + F + (v - (|R_A| - F))   (B's pool part)

A claims virtual tiles **from the bottom** (`a` claimed), B **from the top**
(`b` claimed); the pool is closed when `a + b == n`.  Consequently:

- A's tiles are `[x_A, x_A + F + a_A) ∪ [x_B + F, x_B + F + a_B)` where
  `a = a_A + a_B` (A enters R_B's pool only after exhausting R_A's);
- B's tiles are `[x_B + F + n_B - b_B, x_{B+1}) ∪ [x_{A+1} - b_A, x_{A+1})`
  where `b = b_B + b_A`, and additionally B's own prefix `[x_B, x_B + F)`.

Every CTA's tile set inside one static range is at most two contiguous runs
(B: prefix run + backward run in its own range), each processed in one
direction, so every range has at most **four partial items per CTA-run** and
the scratch slot rule stays per-CTA and static (3.6).  Balance: A and B take
tiles at their own rates until the pool closes; the wall follows the pair's
mean rate instead of the slow member (section 6).

**Processing order.**  A: prefix forward, continuing without a boundary into
its own pool part, then (seam: forced item boundary) R_B's pool part forward
from `x_B + F`.  B: prefix forward (`[x_B, x_B + F)`), then (forced boundary)
its own pool part **backward** from `x_{B+1} - 1` down, then (seam) R_A's pool
part backward from `x_{A+1} - 1`.  Tile order inside an item does not matter
for the online softmax; the merge combine is order-free.  A CTA with no
partner (solo) walks its own range forward exactly as today.

**Why the role-independent prefix.**  The rendezvous (3.2) costs up to three
dependent global round trips (~4.5 us under the start burst).  The first
`F / u = 4` units of a CTA are its own prefix in the forward direction whatever
its role, so the K loader's first *pool* claim — unit index 4, needed at
`fill(3)` = loader iteration `3u - LEAD = 2` — is ~11 us after start (fill 8.5
+ 2 tiles), after the rendezvous result has landed in `smem.pair`.  Nothing
about the rendezvous is on the prologue path.

### 3.2 Rendezvous (IO warp 2 — the Q warp — before its first item)

Global state in the semaphore region (`workspace[:8 MB]`, zero at first use;
requirement #9 of [8], unchanged in kind), after the `H x B` sequence
semaphores:

    uint32 epoch;                 // launch counter, incremented by the last CTA to finish
    uint32 finished;              // atom.inc, limit P - 1 (self-resetting, as the semaphores)
    uint64 smTable[1024];         // per %smid: {epoch:32, leaderId:32}
    uint64 pool0[P];              // per leader: {a:24, b:24, joined:1, closed:1, pad}
    uint64 pool1[P];              // per leader: {epoch:32, partnerId:32}
    uint32 pairInfo[P];           // per CTA: leaderId (self if leader or solo)

Protocol of CTA c (one lane of IO warp 2; results broadcast to smem.pair with
`st.release.cta`):

    E  = ld epoch ; S = ld smTable[%smid]                     (one round trip, both loads together)
    pairInfo[c] = c (provisional)
    if S.epoch != E:                                          // no leader of this launch on this SM yet
        pool0[c] = 0 ; pool1[c] = 0                           // fresh pool (ordered before the CAS below)
        if CAS(smTable[%smid], S, {E, c}) succeeds: role = LEADER, leader = c
        else: S = the observed value; fall through to the partner path
    if S.epoch == E (a leader L = S.leaderId exists):
        pool1[L] = {E, c}                                      // partner id first (release)
        if CAS(pool0[L], observed with joined = 0 && closed = 0, same with joined = 1) succeeds:
            role = PARTNER, leader = L ; pairInfo[c] = L
        else: role = SOLO                                      // pool already joined or closed: c keeps its whole range
    publish smem.pair = {role, leader, partnerRangeBase/Len (partner: leader's range; leader: read pool1[c] lazily), ready}

Properties: no CTA ever waits for another (a failed CAS is a decision, not a
retry loop on the other CTA's progress; the only retry is the `smTable` CAS
re-read, at most once per CTA); a third CTA on an SM, or one arriving after the
pair closed its pool, becomes SOLO; stale entries from a previous launch are
rejected by `epoch`; a leader's `pool0/pool1` reset is ordered before it
publishes itself in `smTable` (release CAS), and a partner reads `smTable`
before touching `pool0` (acquire), so no partner can observe pre-reset pool
words.  The leader learns its partner (if any) only when a claim CAS observes
`joined = 1`: it then reads `pool1[c]` (written by the partner before its
joining CAS, release/acquire) and extends its virtual pool by `|R_B| - F`.  If
the leader closes its pool before a partner joins, the partner's join CAS fails
(closed) and it runs SOLO — correct, merely unbalanced.

Kernel end: every CTA `atom.inc finished, P - 1`; the CTA that receives `P - 1`
stores `epoch + 1` (plain store; the next launch is ordered by the kernel
boundary).  Nothing else needs resetting: `smTable` entries are validated by
epoch, pool words are reset by their leader at the start of each launch.

### 3.3 Claims (K loader, IO warp 0) and their publication

Unit `u = 2` tiles.  Fill of unit k (tiles `[2k, 2k+2)` of this CTA's tile
counter g) at loader iteration `2k - LEAD`, `LEAD = 4` (as today: WAR-safe for
LEAD <= 15 by [8] 8.1; the fill's page-table pair has ~2 tile periods of
slack).  Units 0..3 are the prefix (no claim).  Unit k >= 4 is a **pool claim**
made at `fill(k - 1)` time (one unit ahead, so the `last` flag of unit k - 1's
top tile is known when its records are written):

    loop:  w = ld pool0[L]            (first time: from the CAS return value)
           n = (|R_L| - F) + (w.joined ? |R_partner| - F : 0)
           free = n - w.a - w.b
           if free == 0 or w.closed: closed -> this CTA takes nothing more
           take = min(u, free) ; closing = (take == free)
           w' = w with (role == LEADER ? a += take : b += take), closed |= closing
           if CAS(pool0[L], w, w') succeeds: claimed [virtual range of `take` tiles] ; break
           (else retry with the returned value; at most a few iterations: two claimers, one claim per ~2.8 us each)

The claiming lane publishes `smem.claim = {unitsClaimed, closed}` with
`st.release.cta` after the CAS (the K loader's own cursor advances in
registers).  `take < u` happens only on the closing claim (remainder), and the
unit whose records are written next is the last unit of the CTA's pool run.

The virtual-to-real mapping of a claimed unit is warp-uniform arithmetic
(3.1); the K loader's cursor is positioned at the unit's real tile before the
fill.  A's cursor moves forward; B's cursor moves backward (3.4).  The
`ItemCursor` gets a direction and a `limit` that is the claimed frontier
(records past the frontier are never written; the fill is called only for
claimed units).

Consumers of `smem.claim` (all `ld.acquire.cta`, `__nanosleep(200)` when the
frontier has not reached the reader's position; expected zero waits in steady
state because the K loader claims one unit before the V loader needs it):

- **V loader** (IO warp 1): before `fill(k)` for k >= 4 waits `unitsClaimed >= k
  - 3` (its records are the same tiles); mirrors the K loader's cursor.
- **Q warp / merge warp**: their cursors are limited by the frontier; when a
  cursor reaches the frontier and `closed == 0` they poll; `closed == 1`
  terminates the item stream.  The merge warp's item extent is therefore
  final when it reads it (its `partial`, `weight` and `slot` are then the same
  values the K loader wrote into the record).

### 3.4 Bidirectional `ItemCursor` and the per-tile record

`ItemCursor` gains `dir in {+1, -1}` and `runEnd` (the current run's limit:
prefix end, seam, or claimed frontier).  `next()` emits the current item piece
clipped to the run: forward as today; backward, the piece is `[max(Lseq, lo),
x + 1)` processed from x downward (`tileInSeq` decreases; on `tileInSeq == 0`
the cursor moves to `head - 1`, then `req - 1` — the scan's prefix sums give
`Lseq` of the previous sequence: `Lseq -= tiles(prev)`, with `seqLen[req-1]`
loaded one sequence ahead as `nextSeqLen` is today).  Per-tile flags for tile x
of sequence s (`t` = in-use tile index, `tl` = tiles in use):

    forward:   first = (t == 0)      || (x == runStart)        last = (t + 1 == tl) || (x + 1 == runEnd)
    backward:  first = (t + 1 == tl) || (x == runStart)        last = (t == 0)      || (x == runEnd)   (runEnd = lowest tile)
    validBeg / validEnd: unchanged (static per tile)

`runEnd` for the pool run is the claimed frontier; `last` for the top tile of
unit k is decided at `fill(k)` because unit k + 1's claim outcome is already
known (3.3).  Item-level fields written into the record of the item's **last**
tile (the only tile whose record gemm1 reads them from, :2032-2082):

    partial  = !(item covers all of s)                  (Lseq < itemStart || Lend > itemEnd; both ends known at the last tile)
    slot     = 3 bits, 3.6                              (per CTA: 0..5)
    weight   = 3 bits, 3.7                              (1..4)

The tile word (:397-400) has bits 20..31 free: `slot` at 20..22, `weight` at
23..25; `tileCtaLastBit` (bit 19) is no longer needed (the slot replaces it).
Records written by the filler for a backward run map `g, g+1 <-> x, x-1`.

The `kMetaReady` / `vMetaReady` barriers are **removed**: with fills every 2
tiles they would need 4 barriers per operand and a parity argument (waiter at
`issue(t)`, `t` even, vs fill f+1 of the same barrier at loader iteration `t +
2NB - 4`, gated on gemm0's consumption of `t + 2NB - 7` — sound for NB >= 4),
while the `consumed` chain already orders every fill before the converters'
record read (section 2, last paragraph; the converters' `consumed.wait_parity`
for tile g is an acquire on the loader's arrive at its iteration g, which is
program-ordered after `fill` of tile g at iteration g - LEAD).  Removing them
deletes two PHASECHK sites per 16 tiles from each converter's path and 32 B of
shared memory; it changes no count of any other barrier.

### 3.5 gemm0 / gemm1

gemm0: unchanged (reads `first`/`last`/`validBeg`/`validEnd` from the tile
word; runs the same per-tile path).  gemm1: on `tileLast` reads `partial`,
`slot`, `weight`, `idxReq`, `idxHeadGrp` from the record; partial -> scratch
chunk `6 * persistentCtaIndex() + slot` (`ScratchMem{scratch, 6 * gridDim.x,
1}`); the `weight` is written into `smem.itemWeight[j & 3]` next to the
existing `st.release.cta finalizedItems = j + 1` for the merge warp (or the
merge warp recomputes it from its own cursor: identical by construction; the
smem copy is the cheaper cross-check).  Everything else (finalize, direct
output, Q hand-off) is as written.

### 3.6 Scratch slots: six per CTA, static

A CTA's processing sequence is at most three runs: `own-forward` (prefix, and
for A the continuation into its pool part), `own-backward` (B's pool part of
its own range), `partner` (the other range's pool part, forward for A,
backward for B).  Inside one run only the first and the last item can be
partial (a contiguous run that contains a sequence boundary on both sides of
an item contains the whole sequence — C11 unchanged, per run).  Slot:

    slot = 2 * run + (item is not the first item of its run)      run: own-forward 0, own-backward 1, partner 2

A run with a single item uses its "first" slot.  The merger derives the same
slot from static data plus the pool word (3.8): a piece of s held by CTA h in
run r is the run's first item iff the piece contains the run's entry tile
(forward: the run's lowest tile; backward: its highest).

### 3.7 Arrival weights: the per-range total is static

Per static range `R_c` and sequence `s`, define the parts `P1 = s ∩ prefix_c`
and `P23 = s ∩ pool_c`.  The expected arrival total for `s` is

    W(s) = sum over non-empty ranges c in [c0, c1] of ( [P1 != ∅] x 1 + [P23 != ∅] x 3 )

(static: `x_c`, `F`, `L_s`, `L_end`; the T < P "sparse" rule of [8] counts only
non-empty ranges, as today).  A finalized partial piece arrives with weight

    (piece intersects the prefix ? 1 : 0)
    + (piece covers all of P23 ? 3 : piece is the forward part of a cut P23 ? 1 : piece is the backward part ? 2 : 0)

Every configuration sums to W: solo or A-own uncut (1 + 3); A-own cut + B
backward (1 + 1 + 2); B prefix (1) + [B backward all (3) | A forward all (3) |
A forward cut (1) + B backward cut (2)]; P1-only or P23-only sequences give the
corresponding terms alone.  "Covers all of P23" is decidable by the holder at
its item's last tile: forward holder — the item reached `min(L_end, x_{c+1})`
(no closure cut inside s); backward holder — the item reached `max(L_s, x_c +
F)`; a closure cut inside s makes both holders' items end at the same virtual
point m, so both see "cut".  `m == L_end` (the remainder ended exactly at a
sequence end) is "uncut" for the forward holder and the backward holder's next
item belongs to the next sequence — consistent.

The semaphore is `old = atom.add.acq_rel.gpu semaphores[idxSeq], weight`; the
last arriver is `old + weight == W(s)`; it performs the merge and then stores
`semaphores[idxSeq] = 0` (it is the only remaining user in this launch; the
next launch is ordered by the kernel boundary).  This replaces the
`atom.inc`-with-limit self-reset (:2383-2385); still exactly one atomic per
partial item, one `ATOMG` site in the SASS.

### 3.8 Merge warp: enumeration of pieces from static data plus pool words

For the last arriver of s: for each non-empty range `c in [c0, c1]` (<= 3 for
the bench: 64-tile sequences over 32/33-tile ranges):

    L = pairInfo[c] ; w = pool0[L] ; partner = (w.joined && pool1[L].epoch == E) ? pool1[L].partnerId : none
    forward part of pool_c  = A's claims that fall in R_c:  [x_c + F, x_c + F + fwd_c)
    backward part of pool_c = B's claims that fall in R_c:  [x_{c+1} - bwd_c, x_{c+1})
      with fwd_c = clamp(w.a - (c == L ? 0 : |R_L| - F), 0, |R_c| - F),
           bwd_c = clamp(w.b - (c == partner ? 0 : |R_partner| - F), 0, |R_c| - F)   (0 if no partner)
    pieces of s in R_c:  P1 (owner c, run own-forward; for c == L the item may extend into fwd_c),
                         s ∩ forward part (holder L, run own-forward if c == L else partner),
                         s ∩ backward part (holder partner, run own-backward if c == partner else partner)
    slot per piece by 3.6 ; chunk = 6 * holder + slot ; read SumMax + tokens ; online-combine (unchanged arithmetic)

Reading the pool word *after* the acquire on the semaphore is sufficient even
if the pool is not closed: a piece arrives only after its holder has claimed
the tiles it covers, `a` and `b` are monotone and `a + b <= n` always, so the
decision "s ∩ pool_c lies entirely in the forward part" (or the backward part,
or is split at the closure point, which is then final because pieces cut by
closure are last items finalized after the closing claim) is stable from the
moment the last arrival happens.  Visibility of `pairInfo[c]`, `pool1[L]`:
written by their CTAs before any pool claim (release chain through the claim
CAS and the holder's arrival — release cumulativity, the same chain the [8]
merge relies on); the merger reads them only for pool pieces, and prefix-only
pieces need neither (owner = c, run 0).

Cost: one more dependent L2 round trip per merge (pool words before the chunk
reads): ~0.5-0.7 us on the tail when the last merge of the launch is the last
arriver (section 6).

### 3.9 Host

`P` as today; pairing enabled by the kernel from `floor(T/P) >= 2F` (the scan
knows T).  Scratch `6P` chunks (bench: 1584 x 4 heads x 264 B = 1.7 MB).
Semaphore region layout as 3.2 (`H x B` + 2 + 1024 + 3P words).  Conformance
knob `XQA_PAIR_FORCE=1` (env, kernel parameter): the rendezvous indexes
`smTable` by `blockIdx.x / 2` instead of `%smid`, so CTAs (2k, 2k+1) form a
pair whatever their placement — the same protocol, deterministic for the
matrix (co-residency is a performance assumption only; the test knob makes the
pool paths reachable with P = 2, 4, 6).

## 4. Invariants (D1-D6, C1-C13 of the transport and [8] docs) and the new C-items

Untouched: D1-D6 (copies/expansion), C1-C7 (FA3 producer, not this host), C3
register budgets (section 5), C4 barrier counts (`kBar/vBar/xBar/qBar` counts
unchanged; two barriers deleted), C5-C6, C9 (item-agnostic tile stream: the
records still carry everything; `g` stays the only per-tile index), C10
(record visibility: the same loader -> consumed -> converter -> produced chain;
the redundant `metaReady` edge is removed), C12 (gemm1 -> merge warp counter),
C13 (balanced *static* partition; the dynamic split refines it).

- **C8'** (record visibility without `metaReady`).  Converter `issue(g)` reads
  record g after `kBar[g%3].consumed.wait_parity(toParity<3>(g))`; the phase it
  waits for completes with the loader's `consumed.arrive` at loader iteration
  g, which is program-ordered after the loader's `fill` of tile g (iteration `g
  - LEAD`, LEAD >= 1) and after the flag patch of tile g - 1 written at
  `fill(g)`.  gemm0 / gemm1 read record g after `produced(g)`, which follows the
  converters' arrive (release) after their read.  Prologue: `consumed[s]`
  phase 0 needs the loader's iteration-s arrive, after the prologue fills.
  WAR: as [8] 8.1 (LEAD <= 15).
- **C11'** (slots per run).  Inside one run (own-forward, own-backward,
  partner) a CTA's tiles are contiguous and processed monotonically, so only
  the run's first and last item can be partial; `slot = 2 run + (not first)`
  is unique per CTA (six slots) and the merger reproduces it from `(L_s, L_end,
  x_c, F, a, b, |R_L|, |R_partner|)`.  Forced item boundaries at the prefix end
  (B only), at the seam between ranges, and at the closure point keep every
  item inside one run and one static range.
- **C14** (pool exclusivity and completeness).  `pool0` is modified only by
  64-bit CAS; a claim moves `a` or `b` by `take = min(u, n - a - b)` and sets
  `closed` iff it takes the remainder.  Hence `a + b <= n` always, the forward
  and backward tile sets are disjoint, every pool tile is claimed exactly once
  before `closed`, and after `closed` every claim returns "nothing".  A CTA
  that never learns of a partner (`joined = 0` until it closes) processes its
  own pool alone; a partner whose join fails processes its own range alone
  (SOLO); in both cases every tile of every range is processed exactly once
  because a range's pool is either in exactly one open pool or processed by
  its owner.
- **C15** (static arrival total).  `W(s)` depends only on `(L_s, L_end, x_c,
  F)`; the weights of 3.7 sum to `W(s)` for every possible split (case list in
  3.7).  Arrivals are `atom.add`; the last arriver resets the word.  A sequence
  with a single non-partial holder never touches the semaphore (as today).
- **C16** (merger decision stability).  A piece's arrival is preceded (program
  order, release) by the claims covering it; `a`, `b` are monotone with `a + b
  <= n`; pieces cut by the closure point are finalized after the closing CAS.
  Therefore after the last arrival for s every range's `(fwd_c, bwd_c)` read
  from `pool0[L]` covers `s ∩ pool_c`, and which part holds which tiles cannot
  change afterwards.
- **C17** (no inter-CTA waits).  New inter-CTA interactions are single CAS /
  load / store operations with a decision on every outcome (leader, partner,
  solo; take, nothing).  No CTA waits for another CTA's progress; co-residency
  remains a performance assumption.  Intra-CTA: V loader / Q warp / merge warp
  poll `smem.claim` (monotone, `st.release.cta` / `ld.acquire.cta`, the C12
  pattern); the K loader waits on nothing new.  Liveness: the K loader's claims
  depend only on its own `consumed` waits (gemm0 progress), which depend on
  records the K loader already wrote; the merge warp's `finalizedItems` wait
  depends on gemm1, which never waits on the merge warp.  Acyclic.
- **C18** (epoch and reset).  `epoch` is read once per CTA at start and
  incremented once per launch by the CTA whose `atom.inc finished, P-1`
  returns `P - 1`, after every other CTA has passed its last use of the pool
  words; the increment is a plain store ordered before the next launch by the
  kernel boundary.  `smTable`, `pool0`, `pool1`, `pairInfo` entries are never
  cleared: `epoch` mismatch invalidates `smTable` and `pool1`; `pool0` is
  reset by its leader before the leader becomes visible in `smTable`;
  `pairInfo[c]` is rewritten by c every launch before any pool claim.
  Requirement (restated after review): the region is zero at first use (as
  [8] #9) **and** the pair-state words are at the same addresses in every
  launch that shares the workspace — the header has fixed, shape-independent
  offsets (do-not-build 4, section 10 `PairGlobals`).
- **D3/D6, C6**: untouched (no change in the copy / expansion code).

## 5. Budgets

**Registers** (`__launch_bounds__(640,2)`, pool 30720; split 3 x 128 x 40 + 2
x 128 x 56 = 29696 as today):

| role | budget | added live state | expectation |
|---|---|---|---|
| gemm0 | 40 | none (same word bits) | unchanged, 0 spill |
| gemm1 | 40 | `slot` / `weight` transient at item end | 0 spill |
| IO warp 0 (K loader) | 40 | cursor 9 + `dir`, `runEnd` (2) + claim state `{L, a/b local, n, partnerBase, partnerLen, role}` (~6) + CAS temporaries | at risk; fallback IO at 48 ([8] §4: 2x128x40 + 128x48 + 2x128x56 = 30720 exactly, three `USETMAXREG`) or claim state in smem (`smem.pair`, warp-uniform LDS at claim time only) |
| IO warp 1 (V loader) | 40 | cursor + `dir`, `runEnd` (2) | 0 spill |
| IO warp 2 (Q warp) | 40 | rendezvous transients (prologue only) | unchanged |
| IO warp 3 (merge) | 40 | per range: `L`, `a`, `b`, `partner` (4, transient inside the merge) | fallback as above |
| converters | 56 | none (two PHASECHK sites fewer) | unchanged |

**Shared memory** (113 664 B today; limit 115 712 for 2 CTAs/SM):
`smem.pair` {role, leader, partner, partnerLen, ready} 16 B; `smem.claim`
{units, closed} 8 B; `smem.itemWeight[4]` 16 B; `kMetaReady/vMetaReady` -32 B;
net +8 B -> **113 664 B** (128-aligned, unchanged); trace build 114 864 + 8 ->
114 944 B.  `static_assert(sizeof(SharedMem) + 1024 <= 233472 / 2)` stays.

**Issue / latency budget on the per-tile paths** (premise: role periods
unchanged): K loader: one fill of 2 tiles every 2 tiles instead of 16 every
16 — one dependent page-table pair (0.75-1.25 us) per 2.8 us; the loader's
only other duty is the `consumed` arrive + wait, and the fill's latency delays
that arrive by <= 1.25 us against the converters' 2-tile (2.8 us) slack ([8]
8.2's argument, now per unit); one CAS (~1 us under load, issued one unit
before use) per 2 tiles.  V loader: one `ld.acquire.cta` per fill.  Converters:
-2 PHASECHK per 16 tiles.  gemm0 / gemm1 / Q warp: unchanged per tile.  Merge
warp: +1 dependent L2 round trip per merge.  Prologue: unchanged (the
rendezvous runs in IO warp 2 in parallel with the scan and the first fills;
its result is first needed at loader iteration 2).

**Global memory**: 3 CAS + ~4 loads per CTA for the rendezvous (264 x ~7 ops,
once), one CAS per 2 tiles per CTA (~4350 total), semaphore `atom.add`
unchanged in count.  Scratch 1.7 MB (was 0.56 MB).

## 6. Predicted periods and wall

Per-tile periods are the [8] confirmation's: fast member `T_f`, slow `T_s`
(production `ctarec` bodies / 33): fp8 1.409 / 1.730, fp4 1.355 / 1.630, mixed
(body-derived: 64.4 - 8.5 - 2.8 = 53.1 -> 1.609; fast by the fp8 ratio 1.228
-> 1.310).  Pair pool of 66 tiles; each CTA holds `L` claimed-but-unfinished
tiles when the pool closes: `L = LEAD + 2u + 3 stages + 1 (claim one unit
ahead) ~ 12` (u = 2).  Closure time `t_c = (66 - 2L) / (1/T_f + 1/T_s)`; body
wall = `t_c + L x T_s` (the slow member drains its queue last):

| mode | T_f / T_s | today (slow body) | t_c | body = t_c + 12 T_s | ideal (both end together) | body gain |
|---|---|---|---|---|---|---|
| fp8 | 1.409 / 1.730 | 57.1 | 32.6 | **53.4** | 51.3 | -3.7 (ideal -5.8) |
| fp4 | 1.355 / 1.630 | 53.8 | 31.1 | **50.7** | 48.8 | -3.1 (ideal -5.0) |
| mixed | 1.310 / 1.609 | 53.1 | 30.3 | **49.6** | 47.6 | -3.5 (ideal -5.5) |
| a16 | no split (DRAM-bound) | 70.3 | — | unchanged | — | 0 |

Wall = fill 8.5 (unchanged: prefix keeps the prologue path) + body + tail 2.8
+ 0.6 (merger's pool-word round trip on the last merge):

| mode | today | predicted (base, u = 2) | with the endgame rule (step 2) | target |
|---|---|---|---|---|
| fp8 | 67.8 | **64.7** | ~62.7 | <= 58 (not reached; the remaining gap is the fill 8.5 -> 3.5 and the periods themselves) |
| fp4 | 60.5 | **58.0** | ~56.1 | <= 36 (no) |
| mixed | 64.4 | **61.5** | ~59.6 | <= 62 (borderline base; pass with step 2) |
| a16 | 78.8 | 78.8-79.5 | — | parity |

**Step 2 — endgame rule** (built only if step 1's per-pair end skew histogram
shows >= 1.5 us median, which the model predicts): the systematic skew is `L x
(T_s - T_f)` = 12 x 0.32 = 3.9 us (fp8) because both members hold the same
queue at closure.  Rule, evaluated by the claiming lane when `free <= 2L`: the
claimant X takes the unit iff `(q_X + u) x T_X <= (q_Y + free - u) x T_Y`,
where `q` = claimed-minus-finished tiles and `T` = EMA tile period (both
published per CTA in global memory by gemm1 thread 0 once per tile: one
`st.relaxed.gpu` of `{finishedTiles, periodEMA}`; read by the partner with one
L2 load per claim in the endgame only).  A CTA that declines stops claiming;
the other takes the rest.  Both finish within one unit of each other:
body -> ideal + ~u x T_s / 2 ~ ideal + 1.7 -> fp8 ~53.0 -> the "with step 2"
column above (the residual is the u-quantization).  The rule uses runtime
measurements, not constants; its verification is the skew histogram (section
7), not the wall.

Sensitivity: if the dispatcher does not place exactly one fast and one slow
CTA per SM (e.g. under a different co-tenant state), pairs of equal speed split
50/50 and the wall equals today's — the design cannot be worse than [8] except
for the +0.6 us merger round trip and the u = 2 quantization (<= 1.7 us) on an
already-balanced SM.

## 7. Verification artifacts, accept / reject

Build checks (each of the four q=1 modules):

1. `ptxas -v` with the ninja flags: no C7507; 0 bytes stack, 0 spill stores /
   loads.  If the IO group spills at 40: rebuild with IO at 48 (three
   `USETMAXREG`, record) before any other change.
2. `cuobjdump -sass`: `USETMAXREG` = 2 (or 3 with the fallback); `LDL` = `STL`
   = 0; exactly one `ATOMG...ADD` (semaphore; the `INC` disappears) plus one
   `ATOMG...INC` (`finished`) and the rendezvous / claim `ATOMG...CAS` sites
   (expected 3: `smTable`, join, claim) — all outside the per-tile GEMM /
   converter paths (verified by the per-role SASS split
   `/tmp/r2p8_role_split.py`); converters: PHASECHK 9 -> 7 (K) / 15 -> 13 (V),
   HGMMA 8 + 8, gemm0 / gemm1 barrier sites unchanged (8/11/1, 17/13/1).
3. `cuobjdump -res-usage`: REG 48, STACK 0; `sizeof(SharedMem)` 113 664;
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor` = 2 (ncu: registers 2,
   shared memory 2).

Conformance (`python tests/attention/run_xqa_mixed_page_transport.py`, exit
code = failures): existing 60 cases byte-identical (pairing disabled when
`floor(T/P) < 16`: P = 1 / 3 / 5 with seq 2200 / 4096 have T/P >= 16 and DO
pair if co-resident — on the test GPU they are not, so add:) `XQA_PAIR_FORCE=1`
cases: (4096, P = 2), (4096, P = 4), (2200, P = 6), (50/100/130, P = 4: 1-3-tile
items inside prefix / pool, seams inside sequences), (285, P = 2, subnormal),
each x fp8 / fp4 / mixed = 15 cases -> 75 total, all PASS.  A debug-build
assert in the merge warp: `sum of weights received == W(s)` and every
enumerated chunk's `SumMax` written this launch (a per-chunk `epoch` word,
debug only).

Trace (`MIXED_KV_TRACE 1`, 3 launches, `parse_xqa_pair_trace.py`):

- role periods (CTA 0 tiles 3-7 / 11-18 / 27-32): fp8 1.2-1.35, fp4 1.2-1.25
  within +-5 % of [8]'s table — the premise; reject if any period moved > 5 %;
- per-pair `end(slow) - end(fast)` histogram: base design median <= 4.5 us
  (model 3.9) and body(fast) + body(slow) per SM unchanged within 3 % (the SM's
  throughput is the invariant; only the split moves); step 2: median <= 1.8
  us, max <= 3.5 us;
- claims per CTA: fast 29-33 pool units (58-66 tiles incl. prefix), slow
  21-25; CAS retries per CTA <= 2 (counted in a trace accumulator);
  rendezvous outcome: 132 leader + 132 partner, 0 solo;
- fill median unchanged (8.5 / 7.4 / 6.6 us): the rendezvous is off the path;
- `kc_ready` at every fill tile at the steady-state period (fills every 2
  tiles are not exposed); reject and go to `LEAD` 6 if not.

Timing (locked, `flock /tmp/mixedkv-gpu0.lock bash /home/bigboi/mixedkv_remote_run.sh
<checkout> r3pair sm90 transport_a16 fp8 fp4 mixed`, 5 x 5, q=1 rows; q=4 rows
must equal today within spread):

| mode | accept (base) | predicted | reject if |
|---|---|---|---|
| fp8 | median <= 65.5 | 64.7 | > 66.5 or any period moved > 5 % |
| fp4 | median <= 58.8 | 58.0 | > 59.8 |
| mixed | median <= 62.3 | 61.5 | > 63.3 |
| a16 | median <= 80 | 78.8-79.5 | > 81 |

ncu (one launch): `sm__cycles_active.avg/.max` >= 0.95 (was 0.948 / 0.962),
`dram__bytes_read.sum` unchanged, `launch__grid_size` 264, occupancy limits 2 / 2.

## 8. Do not build if

1. `ptxas -v` spills in the K loader or merge warp at 40 **and** at IO 48
   **and** with the claim state moved to `smem.pair`: the IO group then needs
   the [15] register layout first.
2. `sizeof(SharedMem)` would exceed 115 712 B (it should not move: +8 B).
3. The confirmation trace on the *unchanged* [8] build (re-run before this
   change) no longer shows the pair asymmetry (e.g. a driver / co-tenant change
   made the two members equal): the lever's premise is the 20-23 % split.
4. The semaphore region is not zero at first use in the deployment (as [8]
   #9), **or** the pair-state words are not at the same addresses in every
   launch that shares the workspace.  `epoch`, `smTable` and `pool` are never
   cleared (validated by `epoch`), so their offsets must not depend on B, H or
   P: a layout placed after the H x B semaphores (the first version of this
   change) re-reads old `epoch` / `smTable` / `pool` words as `smTable` /
   `pool` / `tags` of the next launch when B changes (production reuses
   `workspace_u8[:8MB]` across steps of varying B, `flashinfer/decode.py`),
   pairing a CTA with a dead or live-but-unrelated pool and presenting old pool
   words as chunk tags.  Restated C18: the header (`epoch`, `finished`,
   `smTable[1024]`, `pool[2048]`, `tags[6 x 2048]`, 18 496 words) sits at fixed
   offsets at the head of the region and the sequence semaphores follow it, in
   every mixed launch (persistent q=1 and SPEC_DEC q>1 alike).
5. `floor(T/P) < 16` for the bench shape (it is 32) — pairing would be
   disabled and the change is a no-op for the gate.
6. The V loader / Q warp / merge warp polls on `smem.claim` show up in the trace
   as steady-state waits (they must be zero in steady state: the K loader
   claims one unit ahead); a non-zero count means the claim schedule is late
   and `LEAD` / the claim-ahead distance must be re-derived, not tuned.
7. The per-role SASS split shows a new PHASECHK / ARRIVE / CAS site inside the
   gemm0, gemm1 or converter per-tile paths.
8. The merge warp's enumeration needs more than the 3 x (pairInfo, pool0,
   pool1) reads for the bench (sequences over > 3 ranges only for T/P < 22
   tiles: then the tail estimate of +0.6 us is wrong and must be re-derived).
9. Step 2 (endgame rule) is built before step 1's skew histogram is read.

## 9. Amendments after review (resolved before code; the amendment wins)

Each item names the judge finding it answers.  Numbers re-derived from
`/tmp/r3pair/r3pair_trace.log` (P = 264, 3 launches per mode) and
`r3pair_trace132.log` (P = 132 lone control), pairing CTAs by `smid`, taking
per-launch medians over the 132 pairs, excluding co-tenant outliers (body >
80 us; none in these launches):

| mode / launch | body fast (us) | body slow | slow.last - fast.last | fill | lone body (66 tiles) |
|---|---|---|---|---|---|
| fp8 0 / 1 / 2 | 56.61 / 56.80 / 55.97 | 65.10 / 65.20 / 65.15 | 9.23 / 8.83 / 9.31 | 8.66 / 8.26 / 8.80 | 87.87 / 87.97 (1.331 / 1.333 per tile) |
| fp4 0 / 1 / 2 | 57.55 / 57.34 / 57.25 | 65.78 / 65.55 / 65.42 | 8.77 / 8.94 / 8.10 | 7.71 / 8.27 / 7.74 | 84.58 / 84.98 (1.281 / 1.288) |
| a16 0 / 1 / 2 | 66.98 / 67.94 / 67.76 | 68.80 / 69.66 / 70.66 | 0.26 / 0.43 / -0.22 | — | — |

### 9.1 Blocker 1 — the slow member runs two regimes; section 6 is withdrawn

Section 6 modelled the slow member with one rate `T_s = body / 33`.  The
trace says the slow member spends `G = slow.last - fast.last` = 8.8-9.3 us
(fp8) **alone** on the SM after the fast member exits, and the lone control
runs at 1.331 us/tile — 22 % *faster* than the paired fast member (1.715).
So the slow member's paired rate is much worse than 1.973:

    trace (fp8, launch medians):  alone tiles  = G / T_lone = 9.1 / 1.331 = 6.8
                                  paired tiles = 33 - 6.8 = 26.2 in body(fast) = 56.6 us
                                  T_s(paired)  = 56.6 / 26.2 = 2.16 us/tile      (T_f = 1.715)
                                  pair rate    = 1/1.715 + 1/2.16 = 1.046 tiles/us
                                  ideal body (both end together) = 66 / 1.046 = 63.1 vs 65.1  ->  -2.0 us

    production (fp8, [8] ctarec): T_f = 46.5/33 = 1.409 ; G = 57.1 - 46.5 = 10.6 ; T_lone = 1.03-1.09
      (1.331 - 0.30 stamp overhead, or 1.331 x 1.409/1.715)
                                  alone tiles 9.7-10.3 ; paired 22.7-23.3 in 46.5 -> T_s(paired) = 2.00-2.05
                                  pair rate 1.198-1.210 ; ideal body 54.5-55.1 vs 57.1     ->  -2.0..-2.6 us
    fp4:                          T_f 1.355, G 9.1, T_lone ~0.99 -> T_s(paired) 1.88, rate 1.270,
                                  ideal 52.0 vs 53.8                                       ->  -1.8 us

The SM's paired throughput is therefore 1.20 tiles/us against 0.97 lone
(**1.23x**, not the "1.45x" of 1.4 item 3, which mixed production pair bodies
with trace lone bodies; the ncu P132/P264 comparison at the 1.38 GHz lock is
DRAM-shifted and is not used for this claim either — note 12).  Occupancy 2
still wins; the imbalance costs the ~10 us in which one CTA runs alone.

**Base protocol re-derived** (two rates, drain of `L` claimed-but-unfinished
tiles per CTA at closure).  `L` from the schedule of 9.3: the claim of unit m
is made at loader iteration `2m - LEAD - 3`; iteration g runs when gemm0 has
released tile g-3 and gemm1 finishes tile g-3 about half a tile later, so at
the closing claim gemm1 stands at ~`2m - 10.5` while tiles up to `2m - 1` are
claimed: **L ~ 10** (LEAD 4; the 12 of section 6 counted the unit twice).

    t_c = (66 - 2L) / rate ; fast ends at t_c + L T_f ; slow: paired until the fast member ends,
    then alone at T_lone for the remainder of its L tiles.
    fp8 (production): t_c = 46 / 1.198 = 38.4 ; fast 38.4 + 14.1 = 52.5 ; slow paired 14.1 / 2.05 = 6.9 tiles,
                      3.1 tiles alone x 1.03 = 3.2 -> body 55.7        gain 1.4 us (u = 2 quantisation +-1 tile: 1.0-1.8)
    fp4:              t_c = 46 / 1.270 = 36.2 ; fast 36.2 + 13.6 = 49.8 ; slow 7.2 paired, 2.8 x 0.99 = 2.8 -> 52.6   gain 1.2
    mixed (rates scaled from fp8 by 53.1/57.1):                                                                     gain ~1.3

Wall = fill (unchanged) + body + tail (2.8 + 0.6 for the merger's tag round
trip, 9.2).  Predicted **fp8 67.8 -> ~67.0**, **fp4 60.5 -> ~59.9**, **mixed
64.4 -> ~63.7**; a16 unchanged.  The mixed target (62) is out of reach of this
lever in any form: even the ideal split gives ~62.6 before the +0.6 tail.  The
"ideal -5.8 us" / body 53.4 figures of section 6 are withdrawn.

**Verdict on cost / benefit.**  Net ~0.6-1.0 us on fp8 / fp4 / mixed for a
protocol of ~500 lines whose correctness rests on a per-launch global
rendezvous.  It is built (the task says pursue levers whose targets are out of
reach), but with the cheapest protocol that keeps the invariants (9.2-9.4:
no CAS on the per-tile path, no pool word in the merger, no new per-tile
operation in any GEMM or converter role), behind a host switch
(`XQA_PAIR_DISABLE=1` gives the [8] kernel for A/B), and it is **kept only if**
the confirmation shows the gain in 9.6 with every role period unchanged.  The
dominant remaining lever is the per-tile issue count of the paired kernel
(63 % issue-active with 9.4 warps/scheduler; both members slowed vs lone),
i.e. [15]/[34], not the split.  The endgame rule (step 2) is not built: it
adds a per-tile `st.relaxed.gpu` to gemm1, which is a do-not-build-7 item
(note 11), and its upside above the base protocol is <= 0.6 us.

### 9.2 Blocker 2 and notes 5-7 — the merger reads no pool word

The pool-word enumeration of 3.7-3.8 (weights 1/2/3, `pool0[pairInfo[c]]`,
the `[P23 != empty]` guard, the c == L double count, the SOLO stale word) is
**replaced** by two static mechanisms:

1. **Tile-count arrivals.**  A partial item of sequence s arrives with
   `old = atom.add.acq_rel.gpu semaphores[idxSeq], nb` where `nb` is the
   item's tile count (the merge warp's own walker, the same enumeration gemm1
   ran); the last arriver is `old + nb == tiles(s)` (the static in-use tile
   count of s — every tile of s is processed exactly once, C14, so the sum of
   the pieces is `tiles(s)` whatever the split), merges, and stores 0 (it is
   the last user of the word in this launch; the next launch is ordered by
   the kernel boundary).  This is the [8] `atom.inc` protocol with "1 per
   range" generalised to "tiles per piece"; there is nothing to prove per
   configuration.  Still one `ATOMG` per partial item.
2. **Range-indexed scratch chunks with sequence tags.**  Scratch has **6
   chunks per static range** `c`: slot = `2 x part + isLast` with `part` 0 =
   prefix `[x_c, x_c+F)`, 1 = forward pool run, 2 = backward pool run; a
   partial item writes slot `2 x part(run entry tile)` if it is the first item
   of its run, else `2 x part(run exit tile) + 1` (C11': only a run's first
   and last item can be partial).  Within one range exactly one run enters at
   `x_c` (own-forward: SOLO, leader or partner), one forward run may enter at
   `x_c + F` (the leader's run inside the partner's range), one backward run
   enters at `x_{c+1} - 1`, and each run exits once, so the six slots are
   collision-free.  gemm1 thread 0 writes `tags[chunk] = idxSeq + 1` after the
   finalize's group barrier and before `st.release.cta finalizedItems` (the
   tag rides the same release chain as the chunk data: gemm1 stores -> cta
   barrier -> st.release.cta -> merge warp ld.acquire.cta -> atom.acq_rel.gpu
   -> last arriver's acquire, C12).  The last arriver reads the tags of the
   six slots of every non-empty range in `[c0, c1]` (lanes in parallel: lane
   l reads range `c0 + l/6`, slot `l%6`, groups of five ranges for the sparse
   case; one round trip), ballots `tag == idxSeq + 1`, combines exactly the
   matching chunks (data loads as today, one per matching chunk), then clears
   those tags.  A tag written this launch is cleared by its sequence's merger
   in this launch (every partial item's sequence has >= 2 pieces, hence a
   merge), so the tag region is all-zero at every kernel end and stale tags
   cannot exist; the region is part of the semaphore area (zero at first use,
   [8] #9), not of `scratch`.

Consequences: `pairInfo`, `pool1` and the merger's pool-word round trip are
gone; the SOLO role has **no global footprint** (a CTA whose join failed, or
that lost the `smTable` CAS and then failed to join, walks its whole range
forward and writes slots 0 / 3 of its own range like any other holder);
"leader without partner" and SOLO are indistinguishable and need not be
distinguished.  The tail cost is the one tag round trip (+0.6 us on the last
merge), the same as the pool-word round trip it replaces.

### 9.3 Blocker 3 — one global operation per loader iteration, one-period slack

Confirmed by reading the barrier chain: the converter's `issue(t)` waits on
`kBar[t%3].consumed` phase t, which completes with gemm0's `release(t-3)` **and
the K loader's arrive at iteration t**; the loader's iteration t-1 unblocks at
`release(t-4)`, so whatever the loader does inside one iteration has **one tile
period** (~1.2-1.4 us) before it delays a converter issue.  ([8] 8.2's "two
periods" was off by one; a 0.75-1.25 us fill every 16 tiles fitted anyway.)
Design consequences:

- **Claims are a single `atom.add.relaxed.gpu.u64`**, not a CAS loop.  The
  pool word is `{a:16, b:16, partner:16, joined:1}`; the leader adds `u` to
  `a`, the partner adds `u << 16` to `b`; each claimer computes from the
  *returned* word `n = nOwn + (joined ? nOther : 0)` (partner: `n = nL +
  nOwn` always), `free = n - a - b` (signed), `take = clamp(free, 0, u)`,
  `closed = take < u || a + b + take == n`; the leader's claimed virtual tiles
  are `[a, a + take)`, the partner's `[n - b - take, n - b)`.  Atomics on one
  word are totally ordered and every claimer's `take` is computed from a state
  that includes all earlier adds, so the two claimers' virtual ranges are
  disjoint and cover `[0, n)` (the fields may overshoot `n` by <= 2u after
  closure; nobody reads them for anything but this rule).  A partner joining
  after the leader has exhausted its own pool (`a >= nL`) fails its join CAS
  and runs SOLO; a join before that succeeds, and the leader's next add returns
  `joined = 1` (with the partner id in the same word — no second word, no
  separate release chain).
- **Schedule.**  Fill of unit k (tiles `[2k, 2k+2)`) at loader iteration `2k -
  LEAD` (even iterations); **claim of unit m at iteration `2m - LEAD - 3`**
  (odd iterations): the claim of unit k+1 lands one iteration before the fill
  of unit k, so the top tile of unit k knows whether the CTA continues
  (`last` / `ctaEnd` flags) without any patch.  Each iteration therefore holds
  at most one dependent global chain: the fill's page-table pair (2 dependent
  loads, 0.75-1.25 us) **or** one atomic (~1 us), each within one period.
  Prologue: units `0 .. LEAD/2 - 1` (tiles `0 .. LEAD-1`, prefix, role-free);
  the first pool unit (unit `F/u` = 4) is **pre-claimed by the rendezvous**
  (the leader's fresh pool word is `{a = u}`; the partner's join CAS sets `b =
  u`), so the K loader's first atomic is unit 5 at iteration 3 and the fill of
  unit 3 (iteration 2, the first whose flags depend on the role) needs only
  `smem.pair.ready`.  The K loader's iterations 0-2 run at kernel start (their
  `consumed` waits are the phase-0 free arrivals), so the `ready` poll at
  iteration 2 waits for the rendezvous (~5 us worst case under the start
  burst) while nothing needs the loader's iteration-3 arrive before `issue(3)`
  at ~first-K + 1 period ~ 10 us.
- The `LEAD 6` fallback of section 7 is **dropped** (it does not shorten an
  iteration).  `MIXED_KV_META_LEAD` must be even, `2 <= LEAD <= 6 <= F`
  (static_assert); 4 stays.
- The V loader mirrors the K loader's fills (same walker, same unit schedule)
  and takes the frontier from `smem.claim` (`ld.acquire.cta`, one word
  `gClaimed | closed << 31`) before each fill: it waits until `gClaimed >=
  2k + 3 || closed`, exactly the state in which the K loader decided unit k's
  top-tile flags, so both operands' records carry identical words.  Expected
  zero waits (the V loader lags the K loader by gemm1's lag behind gemm0).

### 9.4 Blocker 4 — rendezvous placement

The rendezvous runs in **IO warp 3 (the merge warp) after the `__syncthreads`**
(the warp is idle until the first partial item finalizes, >= 10 us after
start), not in the Q warp before its first item: the Q warp's first publish
stays on the fill path untouched.  Chain: `ld epoch` + `ld.acquire smTable[key]`
(one round trip), `st pool[c]` + `atom.acq_rel.cas smTable` (one), and for a
partner `ld pool[L]` + `atom.acq_rel.cas pool[L]` (one or two): 2-4 round
trips, ~3-6 us under the start burst, finished before the K loader's iteration
2 needs `smem.pair.ready` (9.3).  The merge warp then computes the partner's
seam positions (`locate(x_{c+1} - 1)`, `locate(x_{L+1} - 1)`: the scan's
pass-2 search, one seqLen round trip each; note 10) and publishes `smem.pair`
(`st.release.cta ready`).  The leader's seam (`x_partner + F`) is computed by
the merge warp when the K loader publishes `otherKnown` (first claim return
with `joined = 1`; the merge warp checks it inside its two poll loops and at
its own run switch); it is needed by the walkers only when the leader's claims
cross into the partner's pool (>= nOwn = 24 tiles after learning the partner).
Waits: K loader on `pair.ready` (iteration 2) and on `pair.seamReady[i]` at a
run switch; V loader / Q warp / merge warp on `smem.claim` (frontier) and
`seamReady`; all intra-CTA, all monotone flags (`st.release.cta` /
`ld.acquire.cta`, C12 pattern), acyclic (the merge warp's seam computation
depends only on global loads; the K loader's claims only on gemm0 progress).

### 9.5 Notes 8-12

- **Registers (note 9).**  IO budget 40 first; claim constants (`nOwn`,
  `nOther`, `x_other`, leader index, epoch) live in `smem.pair` and are read
  by warp-uniform LDS at claim / run-switch time, not held in registers.
  Fallback `-DMIXED_KV_IO_REGS=48`: `setmaxnreg.dec 40` (z <= 1), `.dec 48`
  (z == 2), `.inc 56` (z >= 3): 2x128x40 + 128x48 + 2x128x56 = 30720 exactly;
  three `USETMAXREG` in the SASS, recorded against the `== 2` check.
- **Shared memory (note 8).**  `PairInfo` 26 words + `claim` 1 word - 8 words
  (`kMetaReady` / `vMetaReady`) = +76 B as designed; as written the delta is
  **+92 B**: `PersistentSched` grew by 16 B (9 -> 13 words: `pairEnabled`,
  `prefixTiles`, `nbTilesMax` and `SeamPos pos0` (7 words) replace the six words
  `req0 / head0 / tile0 / Lseq0 / seqLen0 / seqLen1`).  113 664 =
  888 x 128 is exact, so `sizeof(SharedMem)` becomes 113 792 (889 x 128) either
  way; 2 x (113 792 + 1024) = 229 632 <= 233 472: still 2 CTAs/SM.  Gate 115 712.
  Confirmed by the build check (`cudaFuncSetAttribute` size / ncu
  `launch__shared_mem_per_block_dynamic`, section 11).
- **Seams (note 10).**  Three searches at most per CTA (partner 2, leader 1),
  each one lane-parallel prefix-sum pass over the batch (B/32 iterations of
  seqLen loads, one round trip) plus one round trip for the neighbouring
  request's seqLen; all in the merge warp, none in a loader iteration.
  Registers: the walker carries `adjSeqLen` (the next request's length in the
  walk direction) exactly as `nextSeqLen` today.
- **Acceptance figures (note 7).**  Pool 50 tiles = 25 units; expected split
  with the corrected rates: fast ~28-30 pool tiles (36-38 incl. prefix), slow
  ~20-22 (28-30).  The SM invariant is *tiles per SM-second*: `66 / (end of
  the later member - start)` per SM must not fall below today's `66 / 57.1` by
  more than 3 %; `body(fast) + body(slow)` is not an invariant under the two
  rates.
- **Step 2 / endgame rule (note 11).**  Not built; recorded under do-not-build
  7 (a per-tile global store in gemm1).
- **ncu (note 12).**  The 1.38 GHz-locked P132 vs P264 comparison (93.3 vs
  87.7 us) is not used for any throughput claim; the 1.23x figure of 9.1 is
  from the 1.98 GHz `%globaltimer` trace and the production ctarec.

### 9.6 Re-derived accept / reject (replaces the section 7 bands)

Build checks as section 7 items 1-3 with: `ATOMG` sites = one `ADD` (semaphore
arrivals), one `INC` (`finished`), one `ADD.64` (claims), two `CAS.64`
(`smTable`, join) — all in IO warps 0 / 3, none in the gemm0 / gemm1 /
converter per-tile paths (per-role SASS split); converter PHASECHK 9 -> 7 (K),
15 -> 13 (V); `USETMAXREG` 2 (3 with the IO-48 fallback); `sizeof(SharedMem)`
113 664 or 113 792.

Conformance: the 60 cases byte-identical (pairing disabled for `floor(T/P) <
16`, P = 1, or `XQA_PAIR_DISABLE=1`) plus `XQA_PAIR_FORCE=1` cases (pairs by
`blockIdx.x / 2` instead of `%smid`): (4096, P 2), (4096, P 4), (2200, P 6),
(285, P 2, subnormal) x fp8 / fp4 / mixed = 12 cases -> 72 total.

Trace (3 launches, `parse_xqa_pair_trace.py`): role periods within +-5 % of
[8]'s table (reject otherwise); per-pair `end(slow) - end(fast)` median <= 2 x
u x T_s ~ 4 us (the quantisation + L drain; today 9-10 us); tiles per SM-second
within 3 % of today; rendezvous outcome 132 leader + 132 partner + 0 solo;
fill median unchanged (8.5 / 7.4 / 6.6); no steady-state waits on
`smem.claim` / `seamReady` (trace accumulator, must be 0 after tile 8).

| mode | today | predicted | accept if median | reject if |
|---|---|---|---|---|
| fp8 | 67.8 | ~67.0 | <= 67.3 | > 67.8 or any period moved > 5 % |
| fp4 | 60.5 | ~59.9 | <= 60.2 | > 60.5 |
| mixed | 64.4 | ~63.7 | <= 64.0 | > 64.4 |
| a16 | 78.8 | 78.8-79.4 | <= 79.5 | > 80 |

Reject = revert to [8] (`XQA_PAIR_DISABLE` is the A/B, the code is removed,
the two-rate finding stays on record).  Do-not-build additions: 10. the
confirmation trace on the unchanged [8] build no longer shows G >= 8 us (the
lever's premise is the alone phase); 11. the IO group spills at 40 **and** 48.

## 10. As written (kernel state after this change; line references into `csrc/xqa/mha_sm90.cu` of this commit)

**Superseded in part by 12.5**: the K / V loader, Q warp and merge-warp bullets
below describe the amendment-1 kernel (claims and fills in the K loader,
rendezvous in the merge warp) that 11.3 rejected; 12.5 is the as-written state
of the amendment-2 kernel (protocol warp).  The constants, `PairGlobals`,
`RunWalker`, prologue, gemm0 / gemm1, converter, trace and test bullets stand.

Production code changed in the `MIXED_KV_PERSISTENT` (q=1 mixed) build only;
the non-mixed and SPEC_DEC kernels are untouched.  Not built or run here: the
code is reviewed by reading first (method rule); build checks, conformance
(72 cases) and the one confirmation run follow 9.6.

- **Constants / knobs**: `pairPrefixTiles` F = 8, `pairUnitTiles` u = 2,
  `pairScratchSlots` 6, roles, `pairFlagForce` / `pairFlagDisable` (:829-870);
  `MIXED_KV_META_LEAD` (even, 2..6; 4) and `MIXED_KV_IO_REGS` (40; fallback 48)
  (:135-149); kernel argument `pairFlags` (:1544) from `mixedKvPairFlags()`
  (`XQA_PAIR_FORCE`, `XQA_PAIR_DISABLE`; :5662) at both launch sites.
- **Global state** `PairGlobals` (:839): a fixed, shape-independent header at
  the head of the semaphore region — word offsets `epoch` 0, `finished` 1,
  `smTable[1024]` at 4 (u64 {epoch, leader}), `pool[2048]` at 2052 (u64 {a:16,
  b:16, partner:16, joined}), `tags[6 x 2048]` at 6148, header end 18 496 —
  followed by the H x B sequence semaphores (`seqSemaphores()`; the SPEC_DEC
  mixed kernel uses the same base).  `P <= PairGlobals::maxCtas` (2048) is
  checked by `choosePersistentGridSize` (throws).  Rationale: `epoch`,
  `smTable`, `pool` persist across launches and are validated by `epoch`, so
  a layout that moved with B / H / P (review must-fix) would alias them into
  the next launch's `smTable` / `pool` / `tags`.  Zero at first use as before.  Scoped PTX accessors (:760-828): `ld/st.relaxed.gpu`,
  `ld.acquire.gpu`, `atom.acq_rel.gpu.cas.b64`, `atom.relaxed.gpu.add.u64`,
  `atom.acq_rel.gpu.add/inc.u32`, `st.release.cta` / `ld.acquire.cta`.
- **Shared memory** (:249-460): `TileRecord.tile` bits 19 ctaEnd, 20-22 slot,
  23 otherRange; `PersistentSched {x0, x1, T, pairEnabled, prefixTiles,
  nbTilesMax, SeamPos pos0}`; `PairInfo` (12 words + 2 `SeamPos`) ; `claim`
  (`gClaimed | closed << 31`); `kMetaReady` / `vMetaReady` removed.
- **`RunWalker`** (:894-1110): register walker of runs (own-forward,
  own-backward, partner range); `init` from `sched`, `bindRole` from
  `smem.pair`, lazy `settle` / `switchRun` at a run end (never when the run end
  is the closed frontier), `next(gEnd, frontierFinal)` emits a piece with the
  item flags (`itemBegins`, `itemEnds`, `partial`, `ctaEnd`, `otherRange`,
  `slot`); backward stepping mirrors the [8] forward cursor with `adjSeqLen`
  one request ahead in the walk direction.
- **Prologue** (:3921): `persistentPrologueScan` decides `pairEnabled`
  (`floor(T/P) >= 2F`, `P >= 2`, field ranges, host flag), publishes `sched`,
  the initial `claim` (F open / whole range closed) and the pair flag words;
  `locateTile` (:3879) is the shared pass-2 search (used for `pos0` and the
  seams).
- **K / V loaders** (:2537-2801): one loop for both operands and both
  static / TMA modules; per iteration g: frontier check (K: registers; V:
  `smem.claim`), `consumed.arrive_and_wait`, TMA boxes (a16 module), then K:
  role load at `gRoleLoad` = 2, `claimUnit` at odd g >= `gFirstClaim` = 3
  (:2601), `fillTileUnit(k)` at even `g + LEAD` (:4103); V: frontier wait
  `gClaimed >= 2k + 3 || closed`, `bindRole` at k = 3, fill.  K loader end:
  `finished` inc, epoch store by the last CTA (:2796).
- **Q warp** (:2802): Q load at the item start without a frontier wait; the
  advance past the item polls `smem.claim`.
- **Merge warp** (:2860-3000): `pairRendezvous` (:3998) then per item: frontier
  poll (with `settle<true>` and the leader's seam[1] service), `finalizedItems`
  poll, `atom.add.acq_rel.gpu semaphores[idxSeq], nb`, last arriver =
  `old + nb == tiles(s)`: lane-parallel tag reads over the 6 slots of every
  non-empty range in `[c0, c1]` (:2942), ballot, combine matching chunks
  (both halves of a lane's 16 elements per chunk; SumMax loaded once per
  chunk), clear tags, reset the semaphore.
- **gemm0** (:1904): `ctaEnd` from the tile word ends the loop; loop bound
  `nbTilesMax`.  **gemm1** (:2157, :2400-2430): chunk = `6 x range + slot`
  (range = own or `smem.pair.otherIdx`), tag store by thread 0 after the
  finalize barrier and before `st.release.cta finalizedItems`; `ctaEnd` ends
  the loop.
- **Converters** (:3295, :3362): `metaReady` waits removed (C8'); `kEnd` /
  `vEnd` tighten from `nbTilesMax` when a record read at issue time carries
  `ctaEnd`; no copy is issued past it.
- **Trace**: every `MIXED_KV_TRACE` stamp unchanged; `ctarec` prints the
  processed tile count (final frontier) and appends `role`, `other` (the other
  range index), `runsw k v q m` (run switches of each IO warp's `RunWalker`),
  `closetake` (closing claim's `take` + 1: 0 SOLO / never closed by a claim,
  1 take 0, 2 odd closure take 1, 3 take 2) and `switchns` (`%globaltimer` of
  the K loader's switch into the partner's range; compared with the partner's
  `last` stamp it shows whether the partner was still active at the crossing).
  These are the review's should-fix 4 path counters: the confirmation trace
  must show leader crossings with an active partner and odd closures at least
  once each.
- **Merge warp seam service** (review should-fix 1): `serviceSeam` first tests
  `seamReady[1]`; a seam[1] computed by the walker's own `switchRun<true>`
  clears `seam1Pending` instead of being recomputed and re-stored while the
  K/V/Q walkers may read it.
- **Tests**: `tests/attention/test_xqa_mixed_page_transport.py` `PAIR_CASES`
  (+12 cases under `XQA_PAIR_FORCE=1`), run by
  `run_xqa_mixed_page_transport.py` (72 cases).

Predicted build-check state (to verify): `USETMAXREG` 2 (3 with
`MIXED_KV_IO_REGS=48`), `ATOMG` sites ADD.32 (merge), INC (finished), ADD.64
(claim), CAS.64 x 2 (rendezvous), all in IO warps; converter PHASECHK -2 each;
`sizeof(SharedMem)` 113 664 or 113 792.

## 11. Build, verification and confirmation (2026-09-04, worktree `mixedkv-wt-r3pair`, nkcut2 H200)

### 11.1 Review fixes landed before the build (commit "review fixes")

- **must-fix, `PairGlobals` layout**: the header is now shape-independent and
  sits at the head of the semaphore region (word offsets `epoch` 0, `finished`
  1, `smTable[1024]` 4, `pool[2048]` 2052, `tags[6 x 2048]` 6148; header end
  18 496 words); the H x B sequence semaphores follow (`seqSemaphores()`).  The
  SPEC_DEC mixed kernel indexes its semaphores from the same base so a q=1 /
  q>1 interleaving on one workspace never overlays the header.  `P <= 2048` is
  checked on the host (`choosePersistentGridSize` throws).  C18 and
  do-not-build 4 restated (sections 4, 8).  Remaining assumption (unchanged
  from [8]): a *non-mixed* XQA kernel sharing the same 8 MB region would
  `atom.inc` the header words; the wrappers never mix the two on one buffer.
- **should-fix 1**: `serviceSeam` tests `seamReady[1]` first; a seam computed
  by the merge warp's own `switchRun<true>` clears `seam1Pending` instead of
  being recomputed and re-stored.
- **should-fix 2/3**: 9.5 smem delta corrected to +92 B; C18 / do-not-build 4 /
  section 10 restated; the layout is in the `PairGlobals` header comment.
- **should-fix 4**: trace-build path counters in the `ctarec` line (`runsw`
  per IO warp, `closetake`, `switchns`, `other`), analysed in 11.4.
- Pre-existing compile error of the unbuilt kernel: `smemAddr` was defined a
  second time (`mhaUtils.cuh` has it); the duplicate is removed.

### 11.2 Register-budget gate (`ptxas -v`, four q=1 modules F = 0, 1, 2, -1)

Recipe: `nvcc $(ninja flags) -ptx` + `ptxas -arch=sm_90a -v` + `nvdisasm
--print-line-info` and a per-source-line attribution of STL / LDL / CALL
(`/tmp/r3pair_attr.py`, `/tmp/r3pair_regs.py` on nkcut2; artefacts in
`/tmp/r3pair_ptx`).

1. **As committed before this step (IO 40, setmaxnreg chain before the role
   chain)**: `Used 48 registers, 136 bytes stack frame, 428 bytes spill
   stores, 412 bytes spill loads`, LDL 103 / STL 112, no C7507, USETMAXREG 3
   (2 x DEALLOC 0x28 from the two `.dec 40` sites + TRY_ALLOC 0x38).  Every
   spill is in the IO group: the K loader's unit fill (`fillTileUnit` inlined
   at the K fill site: 16 STL / 10 LDL; V fill sites 8 / 5 each) and the merge
   warp (rendezvous, per-item walker, chunk combine).  `-DMIXED_KV_IO_REGS=48`
   changed **nothing** (identical counts).
2. **Attribution by reading the SASS** (`maxReg` per role region): gemm0 R27,
   gemm1 R31, **IO R37, converters R37**.  With the setmaxnreg if-chain placed
   *before* the role chain, every role's code is reachable from every
   setmaxnreg path, and ptxas allocates all of it at the smallest budget in the
   kernel (40).  So in [8] the converters never used their 56 registers and the
   IO group was compiled at 40 whatever `.dec` value the z == 2 branch carried;
   the `IO_REGS=48` fallback of 9.5 could not work as written.
3. **Fix (structural, no tuning)**: each group's setmaxnreg is the first
   statement of its own role branch (`if (z <= 1) { dec 40; gemm0 | gemm1 }
   else if (z == 2) { [dec IO]; IO } else { inc 56; converters }`).  Result
   with IO 40: converters R50, IO R37, spills unchanged (428 B; the IO group
   really needs more than 40).  With IO 48 (no setmaxnreg in the IO branch;
   2x128x40 + 128x48 + 2x128x56 = 30720 = the launch pool): **IO R45,
   converters R50, `88 bytes stack frame, 220 bytes spill stores, 200 bytes
   spill loads`, LDL 50 / STL 59 (F = 1, 2; 58 / 60 for F = 0, -1), no C7507,
   USETMAXREG 2 (one DEALLOC 0x28, one TRY_ALLOC 0x38)**.  Shipped:
   `MIXED_KV_IO_REGS` default 48.
4. **Where the remaining spills are** (IO 48, F = 1, by source line): zero in
   the K loader, the V loader and the Q warp (no STL / LDL at any fill, claim
   or copy-issue site).  All 59 STL / 50 LDL are in the merge warp: the
   rendezvous (`rangeStart` div_u64 CALL ABI saves, once per CTA), the walker
   save / restore around the per-item chunk combine (`walker.next<true>` /
   `settle<true>` sites: 13 + 11 + 11 STL), and the combine's own state (16
   accumulator floats + range enumeration).  Every site is per item (5-6 items
   per CTA, 2-3 of them partial), none per tile.  Do-not-build 1 is read as
   "spills on a per-tile path": not triggered; the merge-warp per-item spills
   are recorded as a known cost (tens of ns per item).
5. **Other counts** (F = 1): CALL.REL.NOINC 14 = 4 prologue scan (div_u64) +
   4 gemm1 finalize (unchanged from [8]) + 2 claim (`rangeStart`, only on the
   first `joined` observation) + 2 rendezvous + 2 merge (c0 / c1); none per
   tile.  ATOMG 5, all in the IO warps: ADD.64 (claim), ADD.32 (semaphore),
   CAS.64 x 2 (rendezvous), INC (finished).  Role split (lineinfo): gemm0
   PHASECHK 8 / ARRIVE 11 / HGMMA 8 (unchanged), gemm1 PHASECHK 18 (+1, the
   partial / final finalize split) / ARRIVE 13 / STG 4 (+1: the chunk tag),
   converters PHASECHK 9 each (the `metaReady` waits are gone), no ATOMG /
   CAS outside the IO warps: do-not-build 7 not triggered.  LDGSTS 30 (F = 1),
   UTMALDG 8 (F = 0 / -1): unchanged.

### 11.3 First confirmation run of the kernel as written (r3pair @ 746cb063): rejected by reading

Matrix: the SPEC_DEC (q>1) mixed module did not compile — the kernel took the
`pairFlags` argument unconditionally while the SPEC_DEC / non-mixed launch
sites pass 19 arguments (`cuda_runtime.h(286): ... cannot be called with the
given argument list`); the q=1 modules built.  Bench (locked 5x5, q=1):
transport_a16 91.9 / **93.2** / 94.1, fp8 91.5 / **92.4** / 92.8, fp4 84.7 /
**85.4** / 85.9, mixed 122.8 / **123.3** / 124.5 us — +18 / +36 / +41 / +91 %
against [8].  Object: REG 48 STACK 88, USETMAXREG 2 (0x28 / 0x38), ncu
`launch__shared_mem_per_block_dynamic` 113.66 KB, occupancy limits 2 / 2.

Trace (r3pair-trace, MIXED_KV_TRACE=1, 3 launches per mode, ctarec with the
new counters; `/tmp/r3pair/pair_analysis.py`):

| mode | fill med | body min/med/max | tail | end med / max | tiles leader / partner | SM pairs both-slow / both-fast / mixed |
|---|---|---|---|---|---|---|
| fp8 | 16.1-17.0 | 77-79 / 85-87 / 96-100 | 2.7 | 106-108 / 120-124 | 32 / 34 | ~30 / ~33 / ~27 (of 89-93) |
| fp4 | 15.6 | 74 / 82 / 90-94 | 2.5 | 101 / 116-119 | 32 / 34 | ~30 / ~30 / ~35 |
| a16 | 15.0-15.6 | 67-68 / 75-77 / 83-86 | 2.3 | 94 / 107-109 | 34 / 32 | ~40 / ~37 / ~30 |

Protocol facts: every launch pairs all 264 CTAs (132 leaders + 132 partners,
pool partner == smid co-resident in every pair), K/V/Q walkers switch runs
once per CTA (261 of 264; the rest are outliers), 71-83 partners reach the
leader's range, all 132 leaders cross into the partner's range — but only
after the partner's last tile (`partner.last > switchns` in 0 cases), i.e. the
leader crosses only into the partner's leftover.  Both members are equally
slow (|body gap| median 3 us; lower / upper warp slot medians equal): the
asymmetry is gone because the whole SM is slower, not because the fast member
took more tiles (32 / 34 split).  The warm-up launch (fresh zero workspace)
ran every CTA SOLO: with `epoch` = 0 a zero `smTable` entry reads as
"leader 0 of the current epoch" (first-launch defect, fixed in 12.1).  The
merge-warp counters (`runsw[3]`, `closetake`, `switchns`) print constant
garbage in every CTA — a trace-build defect investigated with a canary in 12.

**Role periods, tiles 3-7 (fp8; [8] in parentheses)**: gemm0 kwait cadence
1.90 (1.21), gemm1 mma 2.51 (1.20), K loader 2.38 (1.18), K converter 1.96
(1.19) us; the K loader's per-tile cadence alternates 3.1 / 1.6 us (6296 /
3241 / 6026 / 2935 / 5838 cycles) and its `kl_start -> kl_iss` segment is
**1.51 us (0.059 in [8])**; V loader `vl_start -> vl_iss` 0.96 (0.08).  a16:
K loader 2.69 (segment 2.25).  Reading: the K loader is now the pacing role.
Between its stamps it runs, every second iteration, the unit fill
(`fillTileUnit`: one dependent global round trip for the page indices, a
second for `page_format`, on the loader's own path) and, on the other
iterations, the claim (`atom.add.u64` whose result is consumed by the shfl in
the same iteration: a blocking gpu-scope round trip).  In [8] the fill was
one chunk of 16 tiles every 16 iterations (its round trips amortised to
~0.1 us/tile and taken inside a 3-deep K ring), so the per-tile loader work
was 0.06 us.  The round-3 unit granularity (u = 2, "one global operation per
loader iteration", 9.3) put ~1.2 us of dependent global latency per tile on
the pacing role — 9.3's assumption that the operation's latency is hidden
by one period of slack does not hold when the loader itself consumes the
result in the same iteration.  Every other role follows (the converters and
GEMMs merely wait longer per tile: kwait 1.90).  fill 16 us (8.5): the same
mechanism at start (prefix fills 4 units x 2 round trips, the role load wait at
iteration 2, the claim of unit 5 at iteration 3).

Verdict: the protocol works (roles, claims, seams, merges, 60-case matrix
apart from the q>1 build), the register gate passes, but the as-written
loader structure fails accept / reject on every period (> 5 %); the data
flow must change so that no dependent global latency sits on a loader
iteration.  This is a structural defect, not a tuning matter; section 12 is
the amended design, built and confirmed once.

## 12. Amendment 2 (2026-09-04): protocol warp — claims, fills, Q and seams off the loaders

### 12.1 Data flow

IO warp roles (warpIdx.z == 2):

- **warp 0, K loader**: per tile g: frontier check (`smem.claim`, acquire) ->
  `kBar[g%3].consumed.arrive_and_wait` -> publish `smem.kProg = g`
  (st.release.cta) -> TMA boxes for A16 pages (a16 / mixed modules; reads its
  record) -> next tile.  No walker, no claims, no fills, no global loads or
  atomics.  (The [8] loader minus the chunk fill.)
- **warp 1, V loader**: the same on `vBar`, without `kProg`.
- **warp 2, protocol warp** (was the Q warp): owns the `RunWalker`, the pool
  claims, the record fills of BOTH operands, the Q loads, the rendezvous, the
  seams, the item table and the epoch / finished epilogue.
- **warp 3, merge warp**: unchanged consumer of finalized items (walker for
  extents, `finalizedItems` poll, semaphore arrival, tag-enumerated merge);
  no rendezvous, no seams (it waits for `seamReady[i]`, `seamOwner = false`),
  no Q.

Words in shared memory (all C12 monotone flag words, st.release.cta /
ld.acquire.cta):

- `claim` = **filled frontier**: `nbFilled | final << 31`, published by the
  protocol warp after the records of the tiles `[.., nbFilled)` are stored for
  both operands.  Semantics for the readers: tiles `[0, nbFilled)` are this
  CTA's and their records are visible; `final`: no more tiles.  (It was the
  claimed frontier published before the fill; every reader — V loader, Q,
  merge — already used it as "tiles that exist"; the K loader now reads it
  too.)
- `kProg`: the K loader's iteration counter, published after its `consumed`
  wait of tile g.  Meaning (through the barrier chain, 12.3): every reader's
  LDS of the records of tiles `<= g - 5` is complete.
- `pair.*`, `seamReady[i]`, `otherKnown`, `finalizedItems`: as before;
  the writers of `ready`, `seam[i]`, `otherKnown` are now all the protocol
  warp.
- `qItems[16]` (`uint2 {req, head}`): the protocol warp's private ring of
  item starts discovered by the fills but not yet Q-loaded (no flag; one
  writer and reader).

Global: unchanged (`PairGlobals` header, pool words, tags, semaphores).
First-launch fix: `smTable` entries carry `epoch + 1` in the high word, so a
zero region (fresh workspace) has no valid leader entry.

### 12.2 Control flow of the protocol warp

```
fill units 0, 1 (prefix; no role needed)  -> claim := 4
rendezvous (epoch, smTable CAS / join CAS) -> smem.pair, ready := 1
partner: seams 0 and 1 (locateTile x2)     (leader: seam 1 on otherKnown)
fill units 2, 3                            -> claim := 8
k = 4; gClaimed = F + u (paired, unit 4 pre-claimed) | nbTilesStatic, closed (solo)
loop:
  Q service: if a Q is pending and its buffer's consumed phase tests ready
             (mbarrier.test_wait on the token taken when the item was found):
             load Q(req, head) -> store q[j & 1] -> fence.proxy.async -> produced.arrive
  lead bound: wait until kProg >= 2k - 7 (fill at most 8 tiles ahead of the K
             loader; polls run the Q service)
  claim:     if !closed: atom.add.u64 for unit k + 1 (blocking; this warp has
             the slack) -> take, closed, (leader) otherKnown -> seam 1
  if 2k >= gClaimed: break                (nothing left to fill)
  fill unit k: walker pieces -> page indices (one round trip; page_format
             only in the mixed module) -> records of K and V (STS x2) ->
             item starts into qItems, arrive on qBar[j & 1].consumed for the
             next pending Q
  publish claim = min(gClaimed, 2k + 2) | final if it equals the closed pool
  k++
drain: remaining pending Qs with blocking waits
finished.inc, epoch store by the last CTA
```

The claim of unit k + 1 precedes the fill of unit k (the top tile of unit k
needs `ctaEnd` / `last` when the pool closes at unit k + 1 with take 0), as
in 9.3.  Prefix fills need no role: with pairing on, every range has >= 2F
tiles, so tile F - 1 is never a CTA's last tile in either role.

### 12.3 Invariants (additions to section 4)

- **C8''** (record visibility).  A reader of record g is one of: K / V loader
  at iteration g (after its `claim > g` acquire), gemm0 at iteration g (after
  `kBar.produced(g)`, which needs the K loader's arrive of iteration g), gemm1
  at iteration g (`vBar.produced(g)`, the V loader's arrive), K / V converter
  when issuing tile g's copies (after `consumed.wait_parity(g)`, which needs
  the loader's arrive of iteration g).  Every path passes through a loader's
  iteration g, which acquired `claim > g` after the protocol warp's release
  store of the records.  Acyclic: the protocol warp waits on `kProg` only,
  the loaders wait on `claim` only, and `claim > g` for the K loader's tile g
  needs `kProg >= 2 floor(g/2) - 7 <= g - 1`, already published.
- **C19** (record ring WAR).  Filling unit k rewrites the slots of tiles
  2k - 32, 2k - 31.  The lead bound `kProg >= 2k - 7` implies gemm0 finished
  tile 2k - 10 (K ring), gemm1 finished 2k - 12 (X ring), the V converter
  issued 2k - 10 and the K converter 2k - 8: every read of records <= 2k - 12
  is complete, with 19 tiles of margin.  The same bound keeps `qItems[16]`
  safe (items pending Q lie within the 11 tiles between gemm0 and the fill
  position, plus 2).
- **C20** (Q buffers).  `qBar[b].consumed` phase m needs gemm0's arrive for
  item 2m + b - 2 and the protocol warp's arrive for item 2m + b; the warp
  arrives for item j as soon as it has stored Q(j - 2) (its previous wait on
  that barrier completed) and tests the phase without blocking; one pending Q
  at a time, in item order — the same phases as the [8] Q warp.
- **C17** unchanged: no warp waits on another CTA.  **C14 / C18** unchanged.
- Claims hoard at most 10 unprocessed tiles per CTA (unit k + 1 claimed when
  the K loader is at 2k - 7), against 7 in 9.3: the endgame imbalance bound of
  9.1 grows by <= 3 tiles.

### 12.4 Budgets and expectations (to verify by reading)

- Registers: K / V loaders lose the walker (17) and the claim state; the
  protocol warp holds the walker, the claim state, one 64-bit barrier token
  and the fill temporaries (as the old K loader minus the copy state); the
  merge warp is unchanged minus rendezvous.  Gate: `ptxas -v` no spill in
  warps 0-2; IO at 48.
- Shared memory: + `kProg` (4 B) + `qItems` (128 B) + trace canaries (trace
  build only): 113 792 -> <= 113 920 or the same 128-B multiple; gate 115 712.
- Periods: K loader segment `kl_start -> kl_iss` back to ~0.06 us; all role
  periods back to the [8] cadence (1.18-1.22 us) or better; the pool split
  then follows the members' rates (fast member > 33 tiles).  fill back to
  ~8.5 us (prefix fills of 2 units before anything, rendezvous overlapped
  with the loaders' first tiles).
- Accept / reject as 9.6.

### 12.5 As written (amendment 2; line references into `csrc/xqa/mha_sm90.cu` of this commit)

Reviewed by reading against 12.1-12.4 and the method rules (explicit call
sites, no runtime-indexed register arrays, no runtime `cond ? V : K`
references beyond the [8] loader's pre-existing `isK ?` operand selection, u32
smem addresses with immediate offsets, unrolled block loops, data-flow /
control-flow comment blocks at each changed region).  Not built or run for
this commit.

- **Shared memory** (:429-469): `PersistentSched` unchanged from section 10;
  `PairInfo` (:446) written by the protocol warp only; `claim` (:463) = filled
  frontier `nbFilled | final << 31`; `kProg` (:466); `qItems[16]` (:469).  Trace
  build adds `traceCanary0 / 1` around the path counters (:506).
- **IO group** (:2625-3150; comment block :2636).  `MIXED_KV_SETMAXNREG_IO()`
  is the first statement of the branch (:2625; no-op at IO 48, 11.2).
  - K / V loader (:2668-2773): per tile g: `readClaim` (ld.acquire.cta of
    `claim`; poll while `g >= gClaimed && !closed`; break at `g >= gClaimed`),
    `consumed.arrive_and_wait`, K publishes `kProg = g` (:2699-2701,
    st.release.cta), A16 TMA boxes from the tile's own record (warp-uniform
    LDS), trace stamps 8 / 9 (K) and 10 / 11 (V) unchanged.  No walker, no
    claim, no fill, no global load or atomic.
  - Protocol warp (:2774-2948): `fillLeadTiles` 8 (:2785, static_assert 4..24);
    `qService(blocking)` (:2799: one pending Q, `qBar[b].consumed` arrive once
    then `test_wait_parity` under `__all_sync`, or `wait_parity` when draining;
    Q load / store / `fence.proxy.async` / `produced.arrive` as the [8] Q warp);
    `publish(nbFilled)` (:2826: `final` iff closed and `nbFilled == gClaimed`);
    `fillUnit(k)` (:2831: `fillTileUnit` then `publish(min(gClaimed, 2k+2))`);
    `claimUnit` (:2842: lane 0 `atom.add.relaxed.gpu.u64` of `u` (leader) or
    `u << 16` (partner), shfl of the returned word, `n / free / take / closed`
    as 9.3, a leader's first `joined` publishes `otherIdx / xOther0 / xOther1 /
    nOther` then `st.release.cta otherKnown` and computes seam 1 (`x_o + F`)
    before `gClaimed` grows; trace `traceCloseTake`).  Order (:2885-2938):
    fills of units 0 and 1 (prefix; `gClaimed = F` paired, whole range solo) ->
    `qService(false)` -> `pairRendezvous` (:2894; sets `role`, `poolWord =
    pool + leaderIdx`, `otherKnown`, `walker.bindRole`; solo: `gClaimed =
    nbTilesStatic, closed`; paired: `gClaimed = F + u`) -> fills of units 2 ..
    F/u - 1 -> main loop (:2911): end check (`closed && 2k >= gClaimed`:
    publish final, break; 12.6) -> lead bound (`kProg >= 2k + 1 - 8`, polling
    with `qService(false)`) -> `claimUnit` for unit k + 1 if open (take 0 with
    `2k >= gClaimed`: publish final, break) -> `fillUnit(k)` ->
    `qService(false)`.  Epilogue: `qService(true)`, `finished.inc` and the
    epoch store by the last CTA (:2943).
  - Merge warp (:2950-3150): waits `pair.ready` then `bindRole` (:2964);
    per item: frontier poll with `settle<false>` (no seam ownership), item
    from `walker.next<false>`, partial items: `finalizedItems > j` poll,
    `atom.add.acq_rel.gpu semaphores[idxSeq], nb`, last arriver (`old + nb ==
    tiles(s)`) enumerates the six tags of every non-empty range in `[c0, c1]`
    (:3018; lane = (range, slot); groups of five ranges), ballots `tag ==
    idxSeq + 1`, combines the matching chunks, clears the tags after the last
    head pass, resets the semaphore.  Unchanged from amendment 1 apart from the
    removed rendezvous / seam computation.
- **`RunWalker`** (:948): `bindRole` (:985), `next` (:1025), `switchRun`
  (:1121) unchanged in logic; `seamOwner = true` is no longer instantiated
  (both walkers wait on `seamReady[i]`); the trace path counter records the
  partner-range switch time from the protocol warp (IO warp 2), which owns the
  fills (12.6).
- **Prologue** (`persistentPrologueScan` :4008, `locateTile` :3966): `claim =
  0` (nothing filled), `kProg = 0`, `pair.ready / otherKnown / seamReady = 0`.
- **`pairRendezvous`** (:4086; protocol warp): `epoch` load, `smTable[key]`
  acquire load; an entry is valid iff its high word == `epoch + 1` (12.1
  first-launch fix); leader: `pool[c] = {a = u}` then CAS `{epoch + 1, c}`
  into `smTable`; partner: join CAS loop on `pool[L]` (`b = u`, partner id,
  `joined`), SOLO when `joined` is set or `a >= nL`; `smem.pair` fields,
  partner seams 0 (`x1 - 1`) and 1 (`x_{L+1} - 1`) via `pairComputeSeam`
  (:4061), then `st.release.cta ready`.
- **`fillTileUnit`** (:4194): walker pieces of unit k (item starts into
  `qItems[nbItems % 16]`), one page-index load per lane, page_format load only
  in the mixed module (static-format modules store the module's format, which
  every consumer substitutes anyway; 12.6), K and V records stored identically
  (unrolled operand loop, STS.32 per page + STS.128 for `{formats, word, req,
  head}`).
- **gemm0 / gemm1 / converters / trace / host**: as section 10 (:2001, :2254,
  :2518 tag store; :3373 `kEnd`, :3440 `vEnd`; :3494 `ctarec` with `role other
  runsw closetake switchns canary`; :5778 `mixedKvPairFlags`).

### 12.6 Fixes after the first amendment-2 build (this worktree, by reading; no run)

1. **Deadlock of CTAs with fewer than `2k + 1` tiles** (commit b3e97090, found
   by the conformance matrix of the amendment-2 build): the main loop waited
   for `kProg >= 2k + 1 - 8` before checking whether any tile was left; a
   closed pool with `gClaimed <= 2k` (short unpaired ranges of the 60-case
   matrix, SOLO CTAs) waited for K-loader progress that never comes.  The end
   check now precedes the lead bound (12.2 order: end check, lead bound, claim,
   fill).  When tiles remain the bound is reachable: the K loader can reach
   tile `2k - 1 >= 2k - 7` because units `< k` are filled and published.
2. **Trace path counter** `switchns`: the stamp was taken by IO warp 0 (the
   amendment-1 K loader); under amendment 2 warp 0 owns no walker, so the field
   would always print 0.  It is now taken by the protocol warp (IO warp 2);
   `runsw[0]` and `runsw[1]` are expected to print 0, `runsw[2]` (protocol) and
   `runsw[3]` (merge) the run switches.  The canaries `C0FFEE00 / C0FFEE01`
   must print intact; otherwise the 11.3 "constant garbage" is a layout / init
   defect of the trace block, to be read before any counter is trusted.
3. **Static-format modules skip the page_format load** in `fillTileUnit`
   (12.2 said "page_format only in the mixed module"; the code loaded it in
   every module).  Every consumer of the tag (loader TMA decision, K and V
   converters) substitutes `MIXED_PAGE_STATIC_FORMAT` for any tag but
   `kMixedBadPageFormat`, so the record now carries that value directly and
   the fp8 / fp4 / a16 modules issue one dependent round trip per unit fill
   instead of two.  No output change (the substituted value is what the
   consumers computed).
4. Stale comments naming the K loader / merge warp as writers of `PairInfo`,
   `otherIdx`, seams and the frontier are updated to the protocol warp.

Confirmation of the amendment-2 kernel (build checks of 9.6 / 12.4, 72-case
matrix, one locked 5x5 run, one trace of 3 launches) is still to be run; the
accept / reject bands are 9.6, with 12.4's additional expectation that the K
loader's `kl_start -> kl_iss` segment is back at ~0.06 us and every role
period is within 5 % of the [8] table.

### 12.7 Review fixes before the confirmation (2026-09-04, r3pair @ 9560ef7a) and the gate read

Must-fix items of the amendment-2 review (both change what the conformance
matrix proves, not the kernel's data flow):

1. **`mixedKvPairFlags()` is read per launch** (:5804), exactly as
   `choosePersistentGridSize` reads `XQA_PERSISTENT_CTAS` (:5780).  It was a
   process-static (`static uint32_t const flags = []{...}()`); the matrix runs
   all cases in one process and the fp8 / fp4 / mixed q=1 modules are first
   launched by `TAIL_CASES` with the variable unset, so every `PAIR_CASES`
   launch of amendment 1 and 2 ran **unforced** (P = 2 / 4 / 6 on 132 SMs: every
   CTA a leader without partner, SOLO after the join timeout never happened —
   no partner role, join CAS, backward run, seam, pool crossing, `otherRange`
   scratch index or multi-slot tag merge was executed by the matrix; the 11.3
   trace, not the matrix, is what showed the protocol live).  The same defect
   would have affected an `XQA_PAIR_DISABLE` A/B sharing a process.
2. **Paired conformance cases have an independent reference with a
   tolerance** (`tests/attention/test_xqa_mixed_page_transport.py`
   `_pairing_active`, `test_xqa_mixed_page_transport_tails_and_value_ranges`):
   when the grid pairs (`floor(T/P) >= 2F`, mirrored from
   `persistentPrologueScan`), `expected` is the stock bf16 decode (`mha.cu`,
   no page transport) on the dequantized KV, compared at the tolerance of
   `test_xqa_mixed_a16_stream_matches_stock_decode` (rtol 1e-2, atol 2e-3,
   ~3 bf16 ulp); cases that cannot pair ((285, P 2): `floor(20/2) = 10 < 16`)
   stay bit-exact against the a16 module.  Reason (accepted property, also of
   production): the pool's closure point is a race between the two CTAs of a
   pair, so the item split, the per-item `colMax` / `colSum` and the bf16
   partials in scratch differ between launches and between modules; paired
   launches are **run-to-run nondeterministic at the bf16-ulp level**, unlike
   [8].  The a16 module (static format 0, the `transport_a16` bench module,
   TMA loader reading the records) is added to the pair loop: 4 cases x 4
   modules = 16 pair cases, 76 in total.

Should-fix items landed in the same commit: `static_assert(fillLeadTiles + 2
<= 16)` with the derivation of the `qItems` bound (:2794-2801: pending Qs
start in `[2k - lead + 1, 2k + 1]`, <= lead + 1 with 1-tile items; 24 is the
record-ring bound only); `qService` returns whether it stored a Q and the
prologue drains every testable Q before the rendezvous (`while
(qService(false)) {}`, :2913); the dead `MIXED_KV_META_LEAD` knob is removed
(:131-135 now point at `fillLeadTiles`); the loaders' and merge warp's
frontier lambda is `readFilled(nbFilled, final)` (:2674; `final` = filled ==
total, distinct from the protocol warp's `closed` = pool exhausted); the
persistent-scheduling header (:750-760), the `pairRendezvous` forward
declaration (:1228) and the prologue's frontier comment (:4070) name the
protocol warp; the `finished` requirement of C18 is restated at the
`PairGlobals` header (:857-865, with the benign-but-silent failure mode).

**Register / SASS gate, re-read for the protocol-warp layout** (`ptxas -v`
on the TU with the ninja flags, all four q=1 modules F = 1, 2, 0, -1; role
attribution by nvdisasm line info with the as-built ranges gemm0 :1934-2192,
gemm1 :2192-2632, K/V loaders :2679-2785, protocol :2785-2971, merge
:2971-3304, K converters :3363-3448, V converters :3448-3497):

- Every module: `Used 48 registers, 112 bytes stack frame, 282 bytes spill
  stores, 328 bytes spill loads`, no C7507, `USETMAXREG` 2 (DEALLOC 0x28 +
  TRY_ALLOC 0x38), `CALL.REL.NOINC` 14 (4 prologue div_u64, 4 gemm1 finalize
  rcp, 2 rendezvous, 2 claim `rangeStart`, 2 merge c0 / c1: none per tile),
  `ATOMG` 5 = ADD.64 (claim, protocol :2950), ADD.32 (semaphore, merge :3032),
  CAS.64 x 2 (rendezvous :2916), INC (`finished`, :2965) — all in IO warps 2 /
  3; HGMMA 16; UTMALDG 8 (F = 0, -1) / 0; LDGSTS 30 / 18 / 0 / 42.
- Per role (fp8 module): gemm0 maxReg R39, **0 STL / 0 LDL on its path** (the
  one LDL that line info attributes to :2184 sits at SASS 0xbee0 between the
  merge warp's frontier poll (:3000-3003) and the IO group's `readFilled`
  (:2663) code and reloads `[R1+0x14]`, stored at 0x8880 by the IO group's
  `RunWalker::init` (:2669): IO-group code with a stale line marker, not
  gemm0); gemm1 R31, 0 / 0; K / V loaders R5 (R12 with TMA), 0 / 0, 54 SASS;
  converters R51, 0 / 0; **protocol warp R45, 21 STL / 29 LDL**; **merge warp
  R45, 45 STL / 45 LDL**.
- Protocol-warp spill sites by source line: rendezvous / bindRole (:2913-2923,
  once per CTA), prefix fills (:2928), main-loop head (:2937-2939),
  `claimUnit` (:2950: 2 STL / 8 LDL), `fillUnit(k)` (:2957: 3 STL / 5 LDL),
  `qService` (:2944, :2958: 2 STL / 3 LDL).  So the per-unit path (every 2
  tiles) of IO warp 2 carries ~7 STL + ~16 LDL of L1-resident local traffic.
  This **misses 12.4's gate** ("no spill in warps 0-2") for warp 2 while
  warps 0 / 1 (the loaders, on the per-tile path) and every GEMM / converter
  path are spill-free.  The budget cannot be moved: gemm0 needs 40 (R39),
  the converters 56 (R51), and 2x40 + 48 + 2x56 = 240 x 128 = 30 720 is the
  whole launch pool; a smaller protocol-warp state is a design change, not a
  fix.  Read as: the protocol warp is not a pacing role (its per-unit work is
  dominated by the ~1 us claim round trip it exists to absorb), so the spill
  cost is tens of ns per unit; recorded as a gate deviation to be judged by
  the trace's role periods, not waived.
- Barrier split (fp8): gemm0 PHASECHK 8 / ARRIVE 11 / BAR.SYNC 1 / HGMMA 8,
  gemm1 18 / 13 / 1 / 8 (unchanged from 11.2), loaders PHASECHK 3 / ARRIVE 1
  (K: consumed wait + kProg publish), protocol 5 / 8, converters PHASECHK 9 /
  ARRIVE 1 each (as 11.2; the "7 / 13" of the review expectation counted the
  amendment-1 `metaReady` sites that were already gone), kernel totals
  PHASECHK 53 / ARRIVE 35 / BAR.SYNC 4.

### 12.8 Confirmation of amendment 2 (2026-09-04, r3pair @ 441d156f, nkcut2 H200): REJECTED — the protocol warp starves the loaders

One standard confirmation run (matrix + locked 5x5 bench + object / occupancy),
one matrix re-run after the reference fix of 12.7 (kernel unchanged), two
controls in the same session (the [8] kernel from its cached r2p8 workspace;
`XQA_PAIR_DISABLE=1` on this kernel), and traces (3 launches x fp8 / fp4 /
transport_a16, q=1) of the paired kernel at tiles 0-7 and 27-34, of the
disabled kernel, and — after adding two trace-only counters — of both again.
Clocks sampled at 200 ms during the runs: 1980 MHz SM / 3201 MHz memory
throughout, co-tenant at 100 % utilization as always.

**Conformance.**  First matrix (stock-decode reference at atol 2e-3): 68 / 76,
the 8 failures all paired fp8 / fp4 / mixed cases.  Isolation
(`/tmp/r3pair_diag.py`: fp32 torch attention, stock decode, a16 static / paired,
mode static / paired x2 / unforced, per (seq, P, mode)):

| case | max abs ref | kernel variants vs fp32 | stock vs fp32 | a16 static vs mode static | paired vs static | paired vs paired (2 launches) |
|---|---|---|---|---|---|---|
| 4096 P2 fp8 | 8.81 | 2.93e-2 (all 6 variants) | 3.40e-2 | 0 | 3.13e-2 | 0 |
| 4096 P2 a16 | 0.10 | 3.2e-4 | 4.5e-4 | 0 | 4.9e-4 | 4.9e-4 |
| 4096 P4 fp4 | 11.0 | 3.1e-2 .. 4.5e-2 | 6.1e-2 | 0 | 6.25e-2 | 3.13e-2 |
| 2200 P6 mixed | 11.75 | 4.6e-2 .. 5.0e-2 | 6.6e-2 | 0 | 6.25e-2 | 6.25e-2 |
| 2200 P6 fp8 | 8.88 | 2.3e-2 .. 2.7e-2 | 3.5e-2 | 0 | 3.13e-2 | 3.13e-2 |

Every variant is within one bf16 ulp of the output magnitude (ulp 0.0625 at
[8, 16)) of the fp32 reference, closer than the stock kernel; the paired
protocol is numerically correct and run-to-run nondeterministic at that ulp
(the accepted property of 12.7).  The reference of the paired cases is now the
fp32 attention with a tolerance of 3 bf16 ulps of the largest output (+ 2e-3);
**second matrix: 76 passed, 0 failed** (paired margins: max diff 2.7e-2 .. 4.8e-2
against tolerances 0.106 .. 0.140; a16 4.2e-4 against 3.4e-3).  Trace path
counters (canaries intact): 132 leaders + 132 partners + 0 solo per launch,
pool partner == smid co-resident in 100 % of pairs, closing claims take 0 /
1 / 2 = 132 / 8 / 124, protocol-warp and merge-warp run switches equal
(260-263 of 264), 60-96 leaders cross into the partner's pool while the partner
is still active.  With `XQA_PAIR_DISABLE=1`: 264 solo, no switches, no claims.

**Object.**  As 12.7 (REG 48, STACK 112, USETMAXREG 2 x, no C7507); ncu
`launch__shared_mem_per_block_dynamic` 114.69 KB, occupancy limits registers 2 /
shared memory 2, `smsp__issue_active` 55.6 % (fp4 q=1; [8]: 62.9 %).

**Locked bench, q=1, min / median / max us (same session, same co-tenant):**

| mode | [8] control (r2p8, this session) | [8] recorded | this kernel, XQA_PAIR_DISABLE=1 | this kernel, paired (production) | 9.6 accept / reject |
|---|---|---|---|---|---|
| transport_a16 | 78.5 / **78.7** / 78.9 | 78.8 | 83.9 / **84.1** / 84.8 | 91.3 / **92.2** / 92.7 | <= 79.5 / > 80 -> **reject** |
| fp8 | 67.9 / **68.1** / 68.2 | 67.8 | 73.9 / **74.2** / 74.2 | 85.6 / **85.8** / 86.1 | <= 67.3 / > 67.8 -> **reject** |
| fp4 | 60.3 / **60.6** / 61.1 | 60.5 | 68.1 / **68.5** / 68.8 | 80.8 / **80.9** / 81.5 | <= 60.2 / > 60.5 -> **reject** |
| mixed | 64.4 / **65.0** / 65.2 | 64.4 | 87.4 / **88.0** / 88.2 | 104.4 / **105.1** / 106.0 | <= 64.0 / > 64.4 -> **reject** |

q=4 (SPEC_DEC, untouched): 83.6 / 111.4 / 113.8 / 114.9 — as [8].  The [8]
control reproduces the recorded numbers within 0.6 us, so the machine state is
not the cause.  The amendment-2 kernel is slower than [8] **even with the pair
protocol disabled** (+5.4 / +6.1 / +7.9 / +23.0 us), and pairing adds +8.1 /
+11.6 / +12.4 / +17.1 us on top.

**Trace (trace build, medians of 3 launches; co-tenant outliers excluded).**

| mode / variant | fill | body | tail | end median | wall excl. outliers | tiles leader / partner | SM pairs both-slow / both-fast / mixed | lower- / upper-slot body |
|---|---|---|---|---|---|---|---|---|
| fp8 paired | 8.1-8.5 | 80.3-81.0 | 2.4 | 91.6-91.8 | 103-107 | 34 / 32 | 26-28 / 23-24 / 38-46 | 80.3 / 80.7 |
| fp8 disabled | 8.7 | 74.6-75.3 | 4.9 | 87.7-88.4 | 95-96 | 33 / 33 | — | — |
| fp8 [8] (r3pair section 1 trace) | 8.0-8.7 | 56.6 fast / 65 slow | 2.5-3.8 | — | — | 33 / 33 | 0 / 0 / 132 | 56.6 / 65 |
| fp4 paired | 7.8-8.1 | 72.0-72.4 | 2.3 | 82.1-83.2 | 94.6-96 | 34 / 32 | 25-28 / 19-29 / 43-47 | 72.3 / 71.7 |
| fp4 disabled | 8.0-8.1 | 65.9-66.5 | 4.8-5.3 | 77.3-77.8 | 84.5-84.9 | 33 / 33 | — | — |
| a16 paired | 11.4-11.6 | 71.7-72.2 | 2.1 | 85.6-86.2 | 99-101 | 36 / 30 | 47-49 / 46-49 / 24-29 | 71.8 / 72.1 |
| a16 disabled | 10.1-11.7 | 70.8-73.0 | 3.4-3.5 | 86.0-86.5 | 92-95 | 33 / 33 | — | — |
| a16 [8] | 6.6 | 70.3 | — | 79.4 | — | 33 / 33 | random | no split |

The slot asymmetry of section 1 is gone (lower / upper slot bodies equal) —
not because the fast member took more tiles (34 / 32) but because **both
members are equally starved**: per-role segment accumulators (cycles / 1980
MHz, ns per tile, fp8 paired, lower / upper slot; section 1.3 fast / slow in
parentheses): gemm0 K-wait 490 / 533 (261 / 262), kwait->mma 736 / 750 (580 /
672), mma->smax 358 / 410 (325 / 399), smax->xarr 485 / 463 (346 / 424); gemm1
V-wait 602 / 622 (282 / 340), rs->mma 841 / 860 (600 / 704); K converter
done->ready 1093 / 1117 (894 / 1017), **expand 950 / 1017 (552 / 707)**, V expand
1188 / 1176 (602 / 742).  The a16 module's segments are unchanged in cycles
(expand 266 vs 269, kwait->mma 966 vs 1028), which also confirms the clock.

Role cadences (`TRACE tile` window, us; [8] 1.18-1.22 for every role): fp8
paired tiles 1-6: gemm0 1.74-1.94, gemm1 1.43-1.67, K loader 1.39-1.74, K
converter 1.90-2.17; fp8 paired tiles 28-32 (TILE0 = 27 build): gemm0
1.66-2.32, K loader 1.68-1.69, K converter 1.80-2.17; fp8 disabled tiles 1-6:
gemm0 1.56-1.84, K loader 1.65-1.82, K converter 1.73-1.91; a16 paired tiles
1-6: 1.09-1.16 (unchanged), mid-body 1.37-1.99.  Loader own work
`kl_start -> kl_iss` 0.15-0.21 us (0.06 in [8]; 1.51 in amendment 1): the
loader's path is short again, as 12.4 required — but its cadence is not.

**Attribution (trace-only counters added at 441d156f: per CTA, the
%globaltimer ns a loader spent polling `claim` because `nbFilled <= g`, and
the number of such tiles).**

| mode | variant | K-loader frontier wait, median per CTA (max) | tiles waited (of 32-33) | wait / body | V-loader wait |
|---|---|---|---|---|---|
| fp8 | paired | 21.3-21.9 us (45-49) | 8-9 (16-17) | **27 %** | 7.9-11.5 us |
| fp8 | disabled | 9.8-11.9 us (18-21) | 5-6 (8-9) | 13-16 % | 2.2-5.3 us |
| fp4 | paired | 31.9-33.4 us (47-49) | 14-15 (17) | **44-46 %** | 10.4-12.3 us |
| fp4 | disabled | 9.5-11.9 us (23-24) | 5-6 (12) | 15-18 % | 1.8-4.6 us |
| a16 | paired | 36.6-39.8 us (49-53) | 14 (16-17) | 50-55 % (DRAM-bound tiles overlap it) | 22-27 us |
| a16 | disabled | 1.5-1.8 us (18-28) | 1 (8-9) | 2.5 % | 1.5-1.8 us |

The loaders wait on the **filled frontier**, i.e. on the protocol warp, for a
quarter (fp8) to a half (fp4, a16) of the body when paired and for 13-18 % of
it (fp8 / fp4) when not.  The protocol warp's unit loop is a chain of
*dependent* global round trips — `atom.add` claim (the result decides the
fill's flags) -> walker -> page-index `LDG` (mixed: + `page_format`) -> STS ->
release — gated by `kProg >= 2k - 7`; under the fp8 / fp4 memory saturation
(264 CTAs x 30 `cp.async` per tile; round trip ~1.5-2.5 us, section 1.3) that
is 3-5 us per 2-tile unit paired and 1.5-2.5 us unpaired, against a 2.4 us
unit demand at the [8] cadence.  Its *throughput* is therefore at or below
the pipeline's; the 8-tile lead only delays the first starvation (fp8: tiles
1-6 already show it), and the a16 module escapes unpaired only because its
TMA traffic leaves the LDG latency short (2.5 %).  Amendment 1 put ~1.2 us
of dependent latency on the pacing loader (11.3); amendment 2 moved it to a
warp of its own but kept it **serial per unit**, so the same latency became
the pacing rate through the frontier.  The starvation also explains the
slower converter expansions: with K-ready arriving late and irregularly the
two CTAs' converters expand in bursts against each other and against the
GEMMs (issue_active 55.6 % vs 62.9 %: less overlap, not more).  Secondary
costs, read from the same traces: (i) the a16 fill +5 us in both variants
(11.6 vs 6.6): Q(0) is now loaded after the fills of units 0 and 1 (two
dependent round trips) whereas the [8] Q warp loaded it in parallel, and
`firstk` includes gemm0's Q wait; (ii) the mixed module's second round trip
per unit (`page_format`) doubles the chain: +23 us unpaired; (iii) the
protocol warp's spills (12.7) are not visible as a period.

**Verdict: reject (9.6).**  Every accept band is missed, every role period
moved > 5 %, and the `XQA_PAIR_DISABLE` A/B does not recover [8] either, so
this kernel cannot ship even with the protocol off.  The premise of the lever
(section 1: slot-priority asymmetry, the fast member idle 10.6 us) and the
two-rate model (9.1) stand; the do-not-build addition 10 was not triggered
([8]'s alone phase is still there).  What the confirmation adds to the
record: (a) a per-2-tile pool protocol needs a **pipelined** producer — claims
for several units in flight and fills whose loads are issued before the
previous unit's store (or [8]'s 16-tile chunk fill with coarser claims); a
serial claim -> fill chain is slower than the tile cadence under fp8 / fp4
memory saturation whatever warp runs it; (b) Q(0) must not queue behind the
fills; (c) the pair protocol itself is correct (76 / 76 with forced pairing,
within 1 bf16 ulp of fp32) and its production nondeterminism is at that ulp.

**Disposition.**  Per 9.6, reject = revert the production kernel to [8]
(`e113026a`, merged at `039ba5c7`); the round-3 branch keeps this document,
the review-fix commits and the test infrastructure (fp32 reference for
paired cases, `_pairing_active`, the frontier-wait counters and the two-line
`ctarec`).  The revert / removal of the pair-pool code is a separate decision
recorded in `mixed_kv_speed_plan.md`; nothing of round 3 is merged.

Artefacts: nkcut2 `/tmp/r3pair2_confirm.log`, `/tmp/mixedkv-r3pair.bench.txt`,
`/tmp/mixedkv-r3pair-disable.bench.txt`, `/tmp/mixedkv-r2p8-ctrl.bench.txt`,
`/tmp/r3pair2_matrix.log`, `/tmp/r3pair2_diag.log`, `/tmp/r3pair2_trace.log`
(paired, tiles 0-7), `/tmp/r3pair2_trace27.log` (tiles 27-34),
`/tmp/r3pair2_trace_dis.log`, `/tmp/r3pair3_trace{,_dis}.log` (frontier
counters), `/tmp/r3pair2_clocks.csv`, `/tmp/r3pair_ptx/x{1,2,0,-1}.*`; local
copies and scripts under `/tmp/r3pair/` (`pair_analysis2.py`, `cadence.py`,
`ctarec2.py`, `r3pair_diag.py`).
