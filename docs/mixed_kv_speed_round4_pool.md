# Round 4 — per-SM tile pool with a pipelined, pre-filled producer: static [8] steady state, closure negotiated in the endgame (design)

Kernel: `csrc/xqa/mha_sm90.cu` at `354914cc` (production kernel state = lever
[8], `e113026a` / merged `039ba5c7`), `ENABLE_MIXED_KV_CACHE && !SPEC_DEC`
q=1 build.  Bench shape B=17, S=4096, 8 KV heads, GQA 4, D=128, q=1.  Line
references are into `mha_sm90.cu` of this worktree (`wt/r4pool`, identical to
`354914cc`).  Design only: no production kernel edit, no build, no run in this
phase.  All numbers below are read from the recorded artefacts of rounds 2 and
3 (`mixed_kv_speed_round2_lever8.md` §10, `mixed_kv_speed_round3_pair.md`
§1, §9.1, §11-12, `mixed_kv_speed_round3_15.md` §1.1-1.2,
`mixed_kv_speed_round3_fill.md` §1.2-1.3).

State (nkcut2 H200, locked 5x5 medians, q=1 us): transport_a16 79.4, fp8
67.9, fp4 60.6, mixed 64.6.  Targets: fp8 <= 58, fp4 <= 36, mixed <= 62.

## 0. Verdict in one paragraph

The round-3 pool failed on its *producer* (a serial claim -> walk -> LDG ->
STS -> release chain of 3-5 us per 2-tile unit under fp8/fp4 memory
saturation, 12.8), not on its balancing idea.  This design removes the
producer problem structurally: the steady state is exactly [8] (16-tile chunk
fills at lead 4 on the loaders, 0.06 us of loader work per tile, no global
operation on any pacing path), the records of every tile a CTA can possibly
take from its partner are **pre-filled once** (one page-table LDG burst per
operand, 8 tiles, on the otherwise idle merge warp, ~10 us after start and
needed >= 35 us after start), and the **only** thing negotiated at run time is
the closure of each range's last D = 8 tiles, in 2-tile units, by <= 8
`atom.cas` per CTA issued from the idle merge warp five loader iterations
before the result is needed (Lambda = 5: 5.6-8.2 us of slack against a
1.5-2 us round trip).  The producer is therefore cheap by construction and
the round-3 frontier-wait counters are the gate (0 after tile 8).  But the
**ceiling** of any pair split is set by the SM's paired throughput, and the
corrected two-rate (paired / lone) model of round 3 §9.1, confirmed by the
P = 132 control of `round3_15.md` §1.1 (lone CTA 0.95 us/tile), gives fp8
body 57.1 -> 55.5-55.9 (-1.2..-1.6 us; ideal -1.7), fp4 -1.2..-1.5, mixed
-1.2..-1.5: **fp8 ~66.5, fp4 ~59.3, mixed ~63.3, a16 unchanged**.  No target
is reached; mixed <= 62 is unreachable by any split (ideal 63.0).  The naive
model the task names (T_f 1.41 / T_s 1.73 as one rate per member) predicts
-4.1 us but is falsified by the lone-rate measurement (section 6.1).  The
only path toward fp8 <= 58 on record ([15]+[7]+fill, one CTA per SM) removes
the pair — and with it this lever's premise.  **Recommendation: no-go as a
standalone build** (section 9); the design is complete so it can be built if
the layout stays at 2 CTAs/SM and a ~1.3 us gain is worth ~400 lines of
protocol with bf16-ulp run-to-run nondeterminism.

## 1. Attribution evidence the lever rests on

### 1.1 The asymmetry is deterministic, per SM, and worth 10.6 us on the wall (round 3 §1)

| fact | value | source |
|---|---|---|
| slow member = CTA in physical warp slots 20-39 | 396 / 396 fp8 pairs, 396 / 396 fp4 pairs, 3 launches each | r3pair §1.2 (`%warpid` of warp 0) |
| body fast / slow (production `ctarec`, fp8) | 46.5 / 57.1 us for identical 33-tile work | lever8 §10 |
| body fast / slow fp4 | 44.7 / 53.8 | lever8 §10 |
| fill (start -> first K ready) fast / slow | 8.0 / 8.7 (trace) — equal; asymmetry is body only | r3pair §1.2 |
| issue-bound segments slow / fast | 1.16-1.28 (HGMMA, softmax, X store, expansion) | r3pair §1.3 |
| memory segments slow / fast | 1.00-1.05 (`cp.async` landing 222 / 234 ns, K-wait 261 / 262) | r3pair §1.3 |
| a16 (DRAM-bound) | no slot correlation, body gap 1.6-2.6 us random | r3pair §1.2 |
| mechanism | fixed-priority issue arbitration between the two co-resident CTAs' warps; software cannot move it (both CTAs put 5 warps on every sub-partition; clusters land on distinct SMs) | r3pair §1.4-1.5 |

### 1.2 Two regimes of the slow member — the ceiling of any split (round 3 §9.1, round3_15 §1.1)

    lone control (XQA_PERSISTENT_CTAS=132, production, fp8): T_lone = (72.7 - 7.2 - 2.8) / 66 = 0.95 us/tile
    trace build: lone 1.331 vs paired fast 1.715 vs paired slow 1.973 us/tile
    slow member's alone phase G = slow.last - fast.last = 8.8-9.3 us (trace), 57.1 - 46.5 = 10.6 (production)

    fp8 production:  T_f = 46.5 / 33 = 1.409
                     alone tiles = 10.6 / T_lone = 10.3-11.2 (T_lone 1.03-0.95)  ->  paired tiles 22.7-21.8 in 46.5 us
                     T_s(paired) = 2.05-2.13 ;  pair rate R = 1/1.409 + 1/T_s(paired) = 1.198-1.178 tiles/us
                     ideal body (both end together) = 66 / R = 55.1-56.0     vs 57.1 today  ->  -1.1..-2.0 us
    fp4 production:  T_f = 44.7 / 33 = 1.355 ; G = 9.1 ; T_lone 0.89-0.99 ; T_s(paired) 1.88-1.94 ; R 1.270-1.253
                     ideal 52.0-52.7 vs 53.8  ->  -1.1..-1.8 us

This is the number every design in this family is bounded by.  The one-rate
reading (T_s = 57.1 / 33 = 1.73 for the whole slow body) predicts ideal 66 /
(1/1.409 + 1/1.73) = 51.3 (-5.8) and is inconsistent with the lone control:
it requires the slow member to run at 1.73 us/tile *while alone*, whereas a
lone CTA runs at 0.95-1.03.

### 1.3 Why the round-3 pool failed: producer throughput, not the pool (round 3 §12.8)

| item | value |
|---|---|
| loaders waiting on the filled frontier (protocol warp), fp8 paired | 21.3-21.9 us per CTA = 27 % of body; fp4 44-46 %; a16 50-55 % |
| the same with `XQA_PAIR_DISABLE=1` (protocol warp still fills 2-tile units) | 13-18 % (fp8 / fp4) |
| producer chain per 2-tile unit | `atom.add` (result decides the flags) -> walker -> page LDG (mixed: + `page_format` LDG) -> STS -> release: 3-5 us paired, 1.5-2.5 unpaired, against 2.4 us of unit demand |
| amendment 1 (claims + fills on the K loader) | `kl_start -> kl_iss` 1.51 us (0.06 in [8]); every period +60-100 % |
| Q(0) behind two fill round trips | a16 fill +5 us (11.6 vs 6.6) |
| the protocol itself | correct: 76 / 76 forced-pairing cases, 132 leaders + 132 partners, within 1 bf16 ulp of fp32 |

Consequence for this design: (a) no dependent global load may be issued *for*
a unit after it is claimed; the records must exist before the claim; (b) no
global operation whose result the same warp consumes in the same iteration may
sit on a loader; (c) Q(0) stays where [8] has it (Q warp, right after the
`__syncthreads`, in smem at 3.07 us, needed at 8.7-9.0: fill doc §1.2).

### 1.4 What [8]'s producer costs today (the steady state this design keeps)

Chunk fill of 16 tiles at loader iteration 16k - 4 (`ahead = g + 4`,
:2229-2233 / :2260-2264): one dependent page-table pair (L2 hits 0.16 /
0.29-0.45 us in the prologue, fill doc §1.2; <= 1.25 us under load, lever8
§8.2) per 16 tiles, inside the loader's one-period slack (r3pair §9.3);
confirmed `kc_ready(16)`, `kc_ready(32)` at the steady-state period (lever8
§10).  Loader own work `kl_start -> kl_iss` 0.06-0.09 us per tile.

## 2. Current data flow and control flow of the touched roles (as written at `354914cc`)

Per-tile machinery (stages, barriers, converters, gemm0 / gemm1 per-tile
path, copy issue, expansion) is untouched.  Touched: the work decomposition,
the IO group's four warps, gemm1's item end, the loop bounds.

- **Shared memory** (:387-416, :451-452): `TileRecord` 32 B {pages[4],
  formats, tile word (bits 0-7 validBeg, 8-15 validEnd, 16 first, 17 last, 18
  partial, 19 ctaLast), idxReq, idxHeadGrp}; `meta[2][2][16]` ring of 32
  slots per operand (`tileRecordAddr` :786-791: slot `g % 32`);
  `PersistentSched {x0, x1, nbTotalTiles, req0, head0, tile0, Lseq0, seqLen0,
  seqLen1}`; `finalizedItems`; `kMetaReady[2]`, `vMetaReady[2]` (count 32).
  `sizeof(SharedMem)` 113 664 B (lever8 §10); trace build 114 688.
- **Prologue** (:1401-1417): IO warp 3 runs `persistentPrologueScan`
  (:3232-3308: T, `x0 = ceil(cT/P)`, `x1`, pass-2 search for `(req0, head0,
  tile0, Lseq0, seqLen0, seqLen1)`), thread 0 zeroes `finalizedItems`,
  `__syncthreads` :1417; every role's loop bound `nbCtaTiles = x1 - x0`
  (:1421).  `setmaxnreg.dec 40` for z <= 2, `.inc 56` for z >= 3 (:1445-1449).
- **ItemCursor** (:717-784): register walker `{x, xEnd, x0, req, head,
  tileInSeq, seqLen, nextSeqLen, Lseq}`; `next(limit)` emits the current
  item clipped to `min(xEnd, limit)` with `partial = !(Lseq >= x0 && Lseq + tl
  <= xEnd)`, `ctaLast = (Lseq + tl >= xEnd)`.  Forward only.
- **Chunk fill** `fillTileMeta` (:3310-3400): phase A walks pieces with the
  cursor (lane owns entries tile `lane/4` and `+8`, page `lane%4`), computes
  the tile word with `first = (x == x0) || (t == 0)`, `last = (x+1 == xEnd)
  || (t+1 == tl)`; phase B one page-table load + one `page_format` load per
  lane; phase C gathers formats (3 shfl) and stores the record (STS.32 per
  page, STS.128 for `{formats, word, req, head}` by lane j == 0).
- **K / V loaders** (IO warps 0 / 1, :2147-2276): `ItemCursor::init`
  (:2146); `fillTileMeta(chunk 0)` + `metaReady[0].arrive` (:2155-2156); per
  tile g: [TMA module: record LDS :2177-2180] `stageBar[stage].consumed.
  arrive_and_wait` (:2185 / :2250), [elected A16 TMA boxes :2195-2226],
  `ahead = g + 4; if (ahead % 16 == 0 && ahead < nbCtaTiles) fillTileMeta(ahead)
  ; metaReady[(ahead/16)%2].arrive()` (:2229-2233 / :2260-2264).
- **Q warp** (IO warp 2, :2277-2296): own cursor; per item `QCvt::load`,
  `qBar[j&1].consumed.arrive_and_wait`, store, `fence.proxy.async`,
  `produced.arrive`.  Q(0) is issued right after the `__syncthreads`, in
  parallel with the loaders' chunk-0 fill.
- **Merge warp** (IO warp 3, :2297-2450): own cursor; per partial item:
  `c0 = floor(Lseq P / T)`, `c1 = floor((Lend-1) P / T)` (:2311-2312),
  `nbPartials = min(c1 - c0 + 1, tiles)` (:2318), poll `finalizedItems > j`
  (:2328-2338), lane 0 `atom.acq_rel.gpu.inc semaphores[idxSeq],
  nbPartials - 1` (:2343-2346), last arriver combines chunks `{2c+1 | c0 <= c
  < c1} + {2c1 + (x_{c1+1} == Lend)}` (:2356-2420) with `ScratchMem{scratch,
  2 * nbCtas, 1}`.
- **gemm0** (:1504-1697): `kBar.produced` wait (:1516), tile word LDS
  (:1522), `tileFirst` -> `runningColMax` reset + `qBar[j&1].produced` wait
  (:1526-1529); `tileLast` -> `qBar[j&1].consumed.arrive`, `j++`
  (:1694-1697).  Loop `idxIter < nbCtaTiles`.
- **gemm1** (:1752-2054): record :1770; `tileLast` (:1990): publish colMax /
  colSum, `reqHead = ldsU64(rec + 24)`, `itemIsPartial` -> chunk `2 *
  persistentCtaIndex() + isCtaLast` (:2011), finalize to scratch or output,
  thread 0 `st.release.cta finalizedItems = j + 1` (:2041-2046).
- **Converters** (:2661-2711 K, :2728-2760 V): `issueKCopies(t)`: at `t % 16
  == 0` `kMetaReady[(t/16)%2].wait_parity(toParity<1>(t/32))` (:2663-2668);
  `kBar[t%3].consumed.wait_parity(toParity<3>(t))` (:2671); copies from
  record t.  Prologue issues t = 0, 1 (:2677-2680); loop `idxIter <
  nbCtaTiles` with `kAhead = 2`.
- **Host** (:4868-4877, :4973-4975, :5096-5099): `P = ctasPerSm x SMs`
  (`XQA_PERSISTENT_CTAS` override), `dimGrid = {P, 1, 1}`, scratch `2P`
  chunks, semaphores `H x B` words in `workspace[:8 MB]` (zero at first use).

Invariant the design keeps relying on (already true today, lever8 §8.1 /
r3pair §2 last paragraph): **the loader's `consumed.arrive` at iteration g is
program-ordered after every record store it made at iterations <= g**, and
every reader of record g (converters at `issue(g)`, gemm0 after
`produced(g)`, gemm1 after `produced(g)`, the TMA loader at its own iteration
g) sits behind that arrive.  Anything the loader writes into record g *before
its arrive at iteration g* is visible to every reader of g.

## 3. New data flow and control flow

### 3.1 Work decomposition: static prefix, negotiated tail

Static ranges `R_c = [x_c, x_{c+1})`, `x_c = ceil(cT/P)` as [8] (33 or 32
tiles for the bench).  Constants: `D = 8` (pool tiles per range, 4 units),
`u = 2` (claim unit), `Lambda = 5` (claim lead in loader iterations).  Each
range is split into

    prefix_c = [x_c, x_c + S_c),  S_c = |R_c| - D          (25 or 24 tiles; owned by c, never negotiated)
    pool_c   = [x_c + S_c, x_{c+1})                         (D tiles; owner takes from the bottom, partner from the top)

Pairing is enabled for the launch iff `floor(T/P) >= 2D` (every range has a
prefix >= D tiles; the bench: 32 >= 16) and `P >= 2`; otherwise every CTA runs
[8] unchanged (S_c = |R_c|, no pool, no protocol).  Two co-resident CTAs (by
`%smid`, rendezvous 3.5) A (leader) and B (partner) each claim from **two**
pool words, their own range's from the bottom and the partner's from the top:

    stream of CTA c:   [x_c, x_c + S_c)                        static prefix                (forward)
                    ++ [x_c + S_c, m_c)                        own pool bottom, lo_c tiles  (forward; same run, NO item boundary at S_c)
                    ++ (x_{p+1} - 1, x_{p+1} - 2, ..., m_p)    partner's pool top, hi_p tiles (DESCENDING linear order; forced item boundary at the run switch)

`m_c = x_c + S_c + lo_c`, `lo_c + hi_c = D` at closure (C21).  The owner's run
inside R_c is one contiguous forward run `[x_c, m_c)`; the partner's run
inside R_c is one contiguous set `[m_c, x_{c+1})` processed in strictly
descending tile order (so its sequences are visited monotonically: <= 2 items
in 8 tiles, only the first and the last can be partial — C22).  Tile order
inside an item is irrelevant to the online softmax; the merge combine is
order-free.  A CTA processes <= `|R_c| + D` tiles (`nbTilesMax`, the loop
bound of every role) and its stream ends at a tile whose record carries
`ctaEnd`.

**Why D = 8 is a derivation, not a tuning.**  With two rates the ideal fast
share is `66 R_f / (R_f + R_s) = 33 (1 + (r-1)/(r+1))`, r = T_s(paired) /
T_f: fp8 r = 1.455-1.51 -> shift 6.1-6.7 tiles; fp4 r = 1.387-1.43 -> 5.4-5.9.
The pool must contain the shift plus one unit of quantisation: D >= 8 for r <=
41/25 = 1.64.  A larger D costs nothing in the steady state (claims are off
every pacing path) and would only enlarge the pre-filled table (64 B per tile
per operand); D = 8 keeps the table at 512 B (section 5).  If a future
measurement shows r > 1.64 the constant is re-derived, not tuned (do-not-build
6).

**Why the closure is the only negotiation.**  The two members' loaders reach
their pools at different times (fast at ~S T_f = 35 us after first K, slow at
~S T_s = 52 us).  Claims are made as the loaders need the decision (Lambda
iterations ahead), so the fast member has taken its own 8 and 4-6 of the
partner's top before the slow member's first claim; the pool arithmetic gives
the fast member 37-39 tiles and the slow one 27-29 without any rate being
measured or published — the split *is* the arrival order.  Nothing about the
first S_c = 25 tiles differs from [8].

### 3.2 The pre-filled partner table (merge warp, once per CTA)

    smem.poolRec[2 operands][D = 8] : TileRecord      (512 B)   records of the partner's pool tiles, index v <-> linear x_{p+1} - 1 - v

Filled by IO warp 3 as soon as the partner's index p is known (3.5): one
`locateTile(x_{p+1} - D)` (the scan's pass-2 search: B/32 lane-parallel
`seqLen` loads, one round trip; `persistentPrologueScan` :3262-3299 already
contains it) positions a cursor; then `fillTileMeta`'s phases A-C over the 8
tiles for **both operands** in one call (lanes 0-31 = 8 tiles x 4 pages: one
entry per lane; the phase-B load pair is issued once and stored twice —
K and V records of a tile are identical except for nothing: both operands'
records carry the same pages / formats / word / req / head, as today's two
fills compute the same values).  The word written into the table is
**sequence-relative**: bit 16 `seqFirst = (t == 0)`, bit 17 `seqLast = (t + 1
== tl)`, bits 0-15 `validBeg / validEnd`, no item bits.  Then
`st.release.cta smem.pair.tableReady = 1`.  Cost: 2-3 dependent round trips
(~1-3 us), issued ~5-12 us after start on an idle warp; first use by a loader
at stream tile `S_c + lo_c >= 25`, i.e. >= 35 us after first K.  The static
modules skip the `page_format` load (the consumers substitute
`MIXED_PAGE_STATIC_FORMAT`; r3pair 12.6 item 3).

### 3.3 Claims (merge warp), frontier word, and what the loaders do with it

Global pool word per range (`PairGlobals`, 3.5): `pool[c] : u32 = {lo:8,
hi:8, closed:1}`.  A claim is one `atom.cas.acq_rel.gpu.b32` loop (blocking,
<= 2 retries because there are two claimers):

    claim(word, side):  w = ld ; loop { if w.closed: return 0
                                        free = D - w.lo - w.hi ; take = min(u, free)
                                        w' = w ; (side == bottom ? w'.lo : w'.hi) += take ; w'.closed = (take == free)
                                        (w, ok) = cas(word, w, w') ; if ok: return take }

CAS (not `atom.add`) because the merger reads `lo` / `hi` as exact tile
counts (3.6); an add-based word can overshoot after closure and the split
becomes ambiguous ((6,4) = owner 6 / partner 2 or owner 4 / partner 4).  CAS
is acceptable here because claims are <= 8 per CTA and run on the idle merge
warp, never on a loader or GEMM path (the round-3 objection to CAS was to a
per-2-tile CAS *on the loader*).

Schedule, in terms of the K loader's published progress `kProg` (3.4): the
claim for the stream unit whose first tile is `u0` is issued when `kProg >=
u0 - Lambda`.  Own-pool units: `u0 = S_c + 2i`, i = 0..3; partner-pool units
(only after the own pool is closed and the partner is known): `u0 = S_c +
lo_c + 2i`.  Result publication, one `st.release.cta` of

    smem.frontier : u32 = { nbDecided:16, ownClosed:1, streamClosed:1, partnerRun:1 }

where `nbDecided` = number of stream tiles whose existence is decided (initial
`S_c`; += take after each claim), `ownClosed` when the own pool word closed
(the stream's own run ends at `nbDecided` at that moment: `m_c - x_c`),
`streamClosed` when no further tile can be added (own closed and (no partner
| partner pool closed for us)), `partnerRun` when tiles >= the own-run end
belong to the partner's pool.  `smem.pair` (3.5) holds `{role, partnerIdx,
xP0, xP1, S_own, ownEnd (= m_c - x_c once ownClosed), tableReady, ready}`.

**Why one outstanding claim is enough** (the task's "at most 2 outstanding"):
the claim has no dependent successor — the records already exist — so the
producer chain is one round trip.  Demand: one unit per 2 tiles = 2.8 us
(fast member) / 4.1 us (slow); a CAS round trip under saturation is 1.5-2 us
(round-3 12.8's 3-5 us per unit was 2-3 dependent round trips).  The claim
for the unit at `u0` is issued at loader iteration `u0 - 5`; its result is
needed by the loader at iteration `u0 - 1` (the `last` flag of tile `u0 - 1`
depends on whether tile `u0` exists): 4 iterations = 5.6 us (fast) / 8.3 us
(slow) of slack against <= 2 us.  Gate: the loaders' frontier-wait counters
(section 7) must be 0 after tile 8.

**Loader per-iteration flow (K; V identical with operand-selected addresses;
the a16 / mixed TMA branch keeps its record LDS + boxes exactly as :2177-2226).
The record of tile g + 1 is prepared at iteration g, after this iteration's
arrive and before the next one's (C23):**

    for g in [0, nbTilesMax):
      F = ld.acquire.cta smem.frontier                                  (warp-uniform LDS)
      while (g >= F.nbDecided && !F.streamClosed) { nanosleep(200) ; reload F }   -> trace counter frontierWait[op]
      if (g >= F.nbDecided) break                                       (stream ended: tile g does not exist)
      [TMA branch: LDS record g (pages, formats, head) as today]
      consumed[g%3].arrive_and_wait                                     (:2185 / :2250, unchanged)
      K only: st.release.cta smem.kProg = g                             (+1 STS per tile)
      [TMA branch: elected arrive_tx + A16 boxes, unchanged]
      if (g + 4) % 16 == 0 && g + 4 < nbTilesMax: fillTileMeta(g + 4) ; metaReady[(g+4)/16 % 2].arrive   (unchanged rule; bound nbTilesMax)
      if pairEnabled && g + 2 >= S_c:                                    (endgame only; warp-uniform; prepares tile n = g + 1)
        n = g + 1
        while (n + 1 >= F.nbDecided && !F.streamClosed) { nanosleep(200) ; reload F }   (needs to know whether tile n+1 exists; counter)
        nextExists = (n + 1 < F.nbDecided)
        if n < ownEnd:                                                  (own run: record n is the chunk fill's)
            if n == ownEnd - 1 && !(F.partnerRun || ownEnd == |R_c|):   ... i.e. the run's last tile when m_c < x_{c+1}:
                word(n) |= last ; partial |= !staticLast(word(n)) ; ctaEnd = !nextExists      (LDS + STS of the word)
        else:                                                           (partner run: v = n - ownEnd, descending)
            wait tableReady (counter; expected 0)
            rec = LDS.128 x2 poolRec[op][v]
            word = validBeg/validEnd(rec) | first (v == 0 || seqLast(rec)) | last (seqFirst(rec) || !nextExists)
                 | partial (!(seqFirst(rec) && itemTopSeqLast)) | otherRange | ctaEnd (!nextExists)
            STS.128 x2 into ring slot n % 32 ; itemTopSeqLast := (first ? seqLast(rec) : itemTopSeqLast)

`ownEnd` = `m_c - x_c` once `ownClosed` (from `smem.pair`), else `S_c + D`
(no own tile can be the run's last before closure).  The decision "does tile
n + 1 exist" needs `nbDecided >= n + 2 = g + 3` or `streamClosed`: the claim
for the unit at `u0 <= g + 3` was issued at `kProg >= u0 - 5 <= g - 2`, two
iterations earlier.  The chunk fill for tiles `[16k, 16k+16)` at iteration
`16k - 4` always precedes the copies into those slots (`16k - 4 < g` for `g >=
16k - 1`), and the copy for tile g + 1 rewrites slot `(g+1) % 32` whose
previous occupant (tile g - 31) is dead (WAR margin of [8] 8.1: LEAD <= 15;
here the effective lead is 1).

For own-run tiles nothing is copied: their records are [8]'s chunk-fill
records (the stream index equals the static index for `g < ownEnd`).  Only
the word of the run's last tile is patched: `last = 1`; `partial |=
!staticLast` (the static word's `last` bit is "last in sequence or last of the
static range": if the sequence continues past `m_c` the item is now partial;
if `m_c == x_{c+1}` no patch is needed at all — the stream equals [8]'s);
`ctaEnd = (no partner run follows)`.

Record word bits (bits 20-31 of the word are free today): 19 `ctaEnd`
(replaces `ctaLast`; the scratch slot no longer uses it), 20 `otherRange`.

### 3.4 Other roles

- **gemm0** (:1504-1697): loop bound `nbTilesMax`; after the tile-word LDS,
  `if (tileWord & ctaEnd) lastTile = true` and the loop exits after this
  tile.  Nothing else changes (first / last / validBeg / validEnd bits as
  today).
- **gemm1** (:1752-2054): loop bound `nbTilesMax`, `ctaEnd` exit; at
  `tileLast`: `run = (tileWord & otherRange) != 0`; `itemInRun` counter (reset
  when `run` changes); `slot = 2 run + (itemInRun > 0)`; `range = run ?
  smem.pair.partnerIdx : persistentCtaIndex()`; chunk `4 range + slot`,
  `ScratchMem{scratch, 4 * gridDim.x, 1}`.  One LDS of `partnerIdx` per
  partner-run item end.  No new global operation.
- **Converters** (:2661-2760): loop bound `nbTilesMax`; `issue(t)` reads the
  record it already reads and, if it carries `ctaEnd`, sets `kEnd = t + 1`
  (`vEnd`); no copies are issued for `t >= kEnd` and the expansion loop ends
  at `kEnd`.  The `metaReady` parity waits are unchanged (one chunk fill per
  32 tiles still holds for the bounded stream, C24).  PHASECHK counts
  unchanged.
- **Q warp** (:2277-2296): own-run items from the [8] cursor with limit
  `min(xEnd, x_c + F.nbDecided)`; when the cursor reaches the limit and
  `!ownClosed` it polls `frontier` (counter; expected 0 exposure: the Q warp
  runs <= 2 items ahead of gemm0 and the frontier is >= 4 tiles ahead of the
  loader); after `ownClosed` and if `partnerRun`: items from `poolRec[K][v]`
  descending (`(req, head)` runs over v = 0..hi-1 with `hi` from the frontier
  once `streamClosed`, or as far as `nbDecided` allows).  Q loads / stores /
  `qBar` phases exactly as today ([8] 8.5; C20 of round 3).  Q(0) unchanged.
- **Merge warp** (:2297-2450): after the `__syncthreads`: rendezvous (3.5),
  partner table fill (3.2), then one loop that interleaves three duties with
  no blocking wait on any of them except the claim CAS itself:

      loop:
        claims:  if (open && kProg >= nextU0 - Lambda) { take = claim(...) ; publish frontier ; advance nextU0 / switch pool / close }
                 leader: at rendezvous and at each own-pool claim also `ld pair[c]` (same round trip) -> partnerKnown -> table fill
        merges:  if (finalizedItems > j) { item j from the own-run cursor (limit as the Q warp's) or the partner table ; if partial: 3.6 }
        exit:    when streamClosed and every item of the stream is merged
        nanosleep(200) when nothing was done

  Per partial item: `atom.add.acq_rel.gpu semaphores[idxSeq], nb` (tile count
  of the item, C25); last arriver iff `old + nb == tiles(s)`; it merges and
  stores 0 (last user of the word this launch).  Enumeration 3.6.

### 3.5 Rendezvous and global state

Reused from round 3 as verified live (12.5 / 12.8: 132 leaders + 132
partners, 0 solo, partner == `%smid` co-resident in 100 % of pairs; 76 / 76
with `XQA_PAIR_FORCE`), reduced to what this design needs.  Fixed-offset
header at the head of the semaphore region (`workspace[:8 MB]`, zero at first
use, shape-independent — r3pair C18 / do-not-build 4):

    word 0      epoch                 (u32; incremented by the last CTA to finish: atom.inc finished, P-1)
    word 1      finished
    word 4      smTable[1024]         (u64 {epoch+1 : 32, leaderIdx : 32}; a zero region has no valid entry)
    word 2052   pool[2048]            (u32 {lo:8, hi:8, closed:1})
    word 4100   pair[2048]            (u32 partnerIdx + 1, 0 = none)
    word 6148   sequence semaphores H x B  (as today, moved behind the header)

Protocol of CTA c (merge warp, one lane, after the `__syncthreads`;
`XQA_PAIR_FORCE=1` indexes `smTable` by `blockIdx.x / 2`, `XQA_PAIR_DISABLE=1`
makes every CTA SOLO; both read per launch):

    E = ld epoch ; S = ld.acquire smTable[key]
    if S.hi != E + 1:                      # no leader of this launch on this SM yet
        st pool[c] = 0 ; st pair[c] = 0    # fresh words, ordered before the release CAS
        if cas(smTable[key], S, {E+1, c}) ok:  role = LEADER ; partner = (ld pair[c] != 0 ? it - 1 : none)
        else S = observed ; fall through
    if S.hi == E + 1:  L = S.lo
        st pool[c] = 0                      # ordered before the join CAS (release)
        if cas(pair[L], 0, c + 1) ok:  role = PARTNER ; partner = L
        else:                          role = SOLO
    SOLO (or pairing disabled / a third CTA): st pool[c] = {lo = D, hi = 0, closed = 1} ; frontier = {|R_c|, ownClosed, streamClosed}
    publish smem.pair {role, partnerIdx, xP0 = x_p, xP1 = x_{p+1}, S_own, ready}  (st.release.cta)

No CTA waits for another (every CAS outcome is a decision); a leader that
never learns a partner before its own pool closes ends its stream at `m_c`
(= `x_{c+1}` when unopposed: [8] behaviour); a partner whose leader's pool is
already closed gets `take = 0` from it and runs its own range (also [8]
behaviour).  Kernel end: `atom.inc finished, P-1`; the CTA receiving `P-1`
stores `epoch + 1`.  2-4 dependent round trips (~3-6 us under the start burst)
on the merge warp, whose first duty (a partial item's finalize) is >= 14 us
after start; nothing of it is on the fill path (the loaders' prologue is
[8]'s).

### 3.6 Merger enumeration from static data plus the exact pool words

For the last arriver of sequence s (`[Lseq, Lend)`), for each non-empty range
`c in [c0, c1]` (<= 3 for the bench; the T < P stepping of :2356-2366 finds
the non-empty ones as today):

    pieceOwner   = s ∩ [x_c, m_c)              m_c = x_c + S_c + lo_c   (needs pool[c] only if s ∩ pool_c != ∅; else m_c := x_{c+1})
    piecePartner = s ∩ [m_c, x_{c+1})
    slot(pieceOwner)   = pieceOwner contains x_c         ? 0 : 1      (first item of the owner's run, else its last)
    slot(piecePartner) = piecePartner contains x_{c+1}-1 ? 2 : 3      (first item of the partner's descending run, else its last)
    chunk = 4 c + slot ; non-empty pieces are the partials (a piece equal to all of s is never partial: single holder, no arrival)

`pool[c]` for the <= 3 ranges is loaded **together with the semaphore atomic**
(same round trip); a word that reads `closed` is final (C26; `closed` is
monotone); a word that does not is re-read after the atomic's acquire (rare:
it requires the load to have been served before the closing CAS became
visible while the arrival chain already implies closure).  Expected tail cost
0; worst case one L2 round trip (+0.6 us) on the last merge.  Unpaired ranges
(SOLO, pairing disabled) carry `{lo = D, closed}` from the rendezvous, so
one merger rule serves every configuration; the [8] rule `2c + isCtaLast` is
replaced by the run-entry rule in gemm1 and here alike.

### 3.7 Host

`P` as today; scratch `4P` chunks (bench 1056 x 4 heads x 264 B = 1.1 MB, was
0.56 MB); the semaphore region gets the header (6148 words) before the `H x
B` semaphores; `P <= 2048` checked (throws); `mixedKvPairFlags()` read per
launch; `XQA_PAIR_FORCE` / `XQA_PAIR_DISABLE` as round 3 §12.7.  SPEC_DEC
mixed indexes its semaphores from the same base (r3pair 11.1).

### 3.8 Rejected alternatives (so they are not re-proposed)

- 2-tile fills after each claim (round 3, both amendments): the producer's
  dependent chain paces the pipeline (12.8).  Here no fill follows a claim.
- Pipelined 2-tile fills on a protocol warp with 2 claims in flight: still
  2-3 round trips per unit issued from one warp; at 2.4 us demand it needs
  the pipelining to be exact, and the a16 / mixed `page_format` chain makes
  it 3 round trips.  Pre-filling the 8 possible partner tiles once makes the
  question moot.
- Pool over the whole range (round 3's F = 8 prefix + 50-tile pool):
  balancing needs only the D = 8 tiles of imbalance; a larger pool buys
  nothing (3.1) and puts claims on the first 25 tiles.
- Backward own-range run (round 3's partner walking its own pool backward):
  needs a bidirectional walker (round-3 `RunWalker`, the source of the
  merge-warp spills); here the owner always walks forward and only the
  partner's 8 tiles are walked descending, from a static table.
- Endgame rate rule (round 3 step 2): needs per-tile progress publication in
  gemm1 (do-not-build 7 of round 3); its upside above the arrival-order
  split is <= 0.3-0.6 us (section 6).
- Sequence tags on the scratch chunks (round 3 §9.2): +1 STG per partial
  item in gemm1 and a tag clear per merge; the exact CAS pool word gives the
  merger the same information with no gemm1 change.
- gemm1 -> merge warp item ring (removing the merge warp's walker): the ring
  needs a WAR guard, i.e. gemm1 waiting on the merge warp — forbidden ([8]
  2.5).  The merge warp keeps the [8] cursor (0 spills at 40 in [8]).
- Claims on the K loader with the result consumed one iteration later:
  works on paper (no dependent use in the issuing iteration) but puts an
  `ATOMG` on the pacing loader whose *issue* can back up behind the MIO /
  LDGSTS queue (fill doc §10: two pre-arrived loader iterations took 2.5 +
  1.7 us behind the converters' burst).  The idle merge warp absorbs it.
- Learned asymmetric static split (previous launch's body times per smid):
  cross-launch state deciding the partition = tuning by another name (r3pair
  1.6).

## 4. Invariants (D1-D6, C1-C20 of the transport / [8] / round-3 docs) and new C-items

Untouched: D1-D6, C1-C7 (FA3), C3 (register budgets, section 5), C4
(barrier counts unchanged: `kBar/vBar/xBar/qBar/kMetaReady/vMetaReady` all
stay), C5-C6, C9 (item-agnostic tile stream: records carry everything; `g`
stays the only per-tile index), C10 (record visibility chain, extended by
C23), C12 (gemm1 -> merge warp counter), C13 (balanced static partition; the
endgame refines it by <= D tiles), C17 (no inter-CTA waits: rendezvous and
claims are single CAS / load / store operations with a decision on every
outcome), C18 (epoch and reset; fixed header offsets), C20 (Q buffer
phases: the Q warp's per-item protocol is unchanged).

- **C8 (consumer-gated `metaReady` parity) still holds** because the chunk
  fill rule is unchanged and bounded by `nbTilesMax <= |R_c| + D <= 41`: fills
  at iterations 12 and 28 for chunks 1 and 0 (second fill); the converters'
  waits at t = 16 and t = 32 are each preceded by exactly one fill of that
  chunk since the previous wait.  A fill whose chunk start lies beyond the
  stream's end (e.g. the fill at g = 28 for a stream ending at 30) arrives a
  phase nobody waits for — harmless; a stream ending at <= 28 never reaches
  the fill and no converter waits for t = 32 (`kEnd <= 30`).  (C24.)
- **C21 (pool exclusivity and completeness).**  `pool[c]` changes only by
  CAS; a claim moves `lo` or `hi` by `take = min(u, D - lo - hi)` and sets
  `closed` iff `take == free`; `lo + hi <= D` always and `lo + hi == D` iff
  closed; owner tiles `[x_c + S_c, m_c)` and partner tiles `[m_c, x_{c+1})`
  are disjoint and cover `pool_c` exactly once when closed.  A SOLO / unpaired
  owner stores `{D, 0, closed}` at rendezvous.  Every tile of every range is
  processed exactly once: prefix by its owner; pool by the owner up to `m_c`
  and by the partner above `m_c`; a range whose partner never claims has
  `m_c = x_{c+1}`.
- **C22 (runs and slots).**  The owner's run `[x_c, m_c)` is contiguous and
  forward; the partner's run `[m_c, x_{c+1})` is contiguous and strictly
  descending (unit order descending, tiles inside a unit descending).  In a
  contiguous monotone run only the first and the last item can be partial
  (an interior item covers a whole sequence), so `slot = 2 run + (not the
  run's first item)` is unique per range (4 slots) and the merger reproduces
  it from `(Lseq, Lend, x_c, m_c, x_{c+1})`: the run's first item is the one
  containing the run's entry tile (`x_c` / `x_{c+1} - 1`).  A run with a
  single item uses its "first" slot.  The forced boundary at the run switch
  keeps every item inside one run and one static range.
- **C23 (record visibility for copied / patched records).**  Every reader
  of record g sits behind the loader's `consumed.arrive` at iteration g
  (section 2, last paragraph).  The loader writes / patches record `g + 1`
  at iteration g (after its arrive of g, before its arrive of g + 1), so the
  store is program-ordered before the arrive that every reader of g + 1
  acquires.  WAR: slot `(g+1) % 32` held tile `g - 31`, whose readers
  finished before the loader's `consumed` wait at iteration g completed
  (gemm0 released g - 3).  The partner table is written once by the merge
  warp before `st.release.cta tableReady` and read by the loaders / Q warp /
  merge warp after `ld.acquire.cta tableReady` — never rewritten.
- **C25 (tile-count arrivals).**  Each partial piece arrives with its tile
  count `nb`; the pieces of s partition its `tiles(s)` in-use tiles (C21), so
  `old + nb == tiles(s)` identifies the last arriver whatever the split; the
  last arriver resets the word (next launch ordered by the kernel boundary).
  One `ATOMG.ADD` per partial item, replacing the `INC`-with-limit.
- **C26 (merger decision stability).**  A piece of s that intersects
  `pool_c` is finalized by its holder only after `pool[c]` closed (the
  owner's run end `m_c` and the partner's lowest tile are decided by the
  closing claim; the loader writes those tiles' `last` flags after reading the
  frontier that the merge warp published after the CAS).  The holder's
  arrival (merge warp, `atom.add.acq_rel`) is program-ordered after that
  CAS in the same warp (or after the CAS that observed the other side's
  closing CAS), so the last arriver's acquire orders a subsequent read of
  `pool[c]` after the closure; a speculative earlier read that shows
  `closed` is equally final because `closed` never clears within a launch.
- **C27 (frontier monotonicity and liveness).**  `frontier.nbDecided` is
  monotone and published only by the merge warp with `st.release.cta`;
  loaders / Q warp / merge cursor read it with `ld.acquire.cta`.  Claims
  depend only on `kProg` (loader progress), which depends on gemm0's
  releases, which depend on records the loader has already written from a
  frontier value published earlier (the claim for the unit at `u0` is
  triggered at `kProg = u0 - 5`, when the frontier already covers `u0`).
  The merge warp's `finalizedItems` polls are non-blocking within its loop.
  No cycle; every wait is on a monotone intra-CTA word or on this CTA's own
  progress.

## 5. Budgets

**Registers** (`__launch_bounds__(640,2)`: pool 30 720; split 3 x 128 x 40 +
2 x 128 x 56 = 29 696 as today; round-3 11.2 item 3's structural fix — each
group's `setmaxnreg` as the first statement of its own role branch — is
adopted so the budgets are real):

| role | budget | added live state | expectation / fallback |
|---|---|---|---|
| gemm0 | 40 (R27-39 today) | `ctaEnd` test (transient) | 0 spill |
| gemm1 | 40 (R31) | `itemInRun` (1), `run` (1) | 0 spill |
| K / V loader | 40 (IO group fits 40 in [8] with the 9-register cursor) | `S_own` (1), frontier word (1), `ownEnd` (1), `itemTopSeqLast` (1), copy temporaries (transient) | 0 spill; the round-3 amendment-2 loaders without a walker were R5-R12 — the [8] loader with cursor plus ~4 words stays below 40 |
| Q warp | 40 | frontier limit (1), partner-table index (1) | 0 spill |
| merge warp | 40 | [8] cursor 9 + combine ~20 (0 spills in [8]) + claim state `{nextU0, side, open}` (3); rendezvous / table-fill transients once per CTA; `partnerIdx, xP0, xP1, poolAddr` read from `smem.pair` by LDS at claim time, not held | at risk: fallback `-DMIXED_KV_IO_REGS=48` (2x40 + 48 + 2x56 = 240 x 128 = 30 720 exactly, three `USETMAXREG`) with the per-branch `setmaxnreg` placement; per-item spills in the merge warp (as round 3 found, tens of ns per item, none per tile) are recorded, not waived |
| converters | 56 | none (record read at issue as today; `kEnd` 1 register) | unchanged |

**Shared memory** (113 664 B today; limit 115 712 for 2 CTAs/SM):

| item | bytes |
|---|---|
| `poolRec[2][8]` (partner table, both operands) | +512 |
| `pair` {role, partnerIdx, xP0, xP1, S_own, ownEnd, tableReady, ready} | +32 |
| `frontier`, `kProg` | +8 |
| `PersistentSched` +`nbTilesMax`, `pairEnabled` | +8 |
| total | 113 664 + 560 = 114 224 -> **114 304** (128-aligned) |
| trace build (114 688 + 560 + round-3 accumulators 176 + counters 32) | **115 456** <= 115 712 (margin 256 B) |

`static_assert(sizeof(SharedMem) + 1024 <= 233472 / 2)` stays; `sizeof` is
recorded at the build check.

**Issue / latency budget on the per-tile paths** (premise: role periods
unchanged):

| path | added per tile | comparison |
|---|---|---|
| K loader, steady state (g < S - 1) | 1 LDS.acquire + compare (frontier), 1 STS.release (`kProg`) | ~5 instructions vs [8]'s ~60 per tile (`kl_start -> kl_iss` 0.06 us) |
| K / V loader, endgame own tiles | + at one tile: LDS + STS (word patch) | once |
| K / V loader, endgame partner tiles (<= 8 per CTA) | 2 LDS.128 + ~10 ALU + 2 STS.128 + 4 STS.32 (lanes 0-3) | ~20 instructions, MIO 8 wavefronts, vs the converters' ~1 500 MIO wavefronts per tile |
| gemm0 / gemm1 | 1 bit test (`ctaEnd`); gemm1 +1 LDS per partner-run item end | nil |
| converters | 1 bit test per issue | nil |
| merge warp | <= 8 CAS (1.5-2 us each, blocking on an idle warp), <= 4 `pair[c]` loads, rendezvous 2-4 round trips, table fill 2-3 round trips, per merge <= 3 pool-word loads overlapped with the semaphore atomic | 0 on any pacing path; merges themselves as [8] |

**Global memory:** header 6148 words; per CTA <= 8 CAS + <= 4 loads (claims),
3-4 ops (rendezvous), 1 `atom.inc` (finished); semaphore `atom.add` count
unchanged (one per partial item); scratch 1.1 MB.

## 6. Predicted periods and wall

### 6.1 Two models, and which one the record supports

Inputs (production, fp8): `T_f = 1.409`, fast body 46.5, slow body 57.1, fill
8.5, tail 2.8 (lever8 §10); `T_lone = 0.95` (round3_15 §1.1) to `1.03`
(r3pair §9.1, trace-scaled).

*(a) One rate per member, T_f 1.409 / T_s 1.730 (the task's numbers).*  Pool of
66, both claim until closure with `L = 10` claimed-but-unfinished tiles each:

    t_c = (66 - 2L) / (1/1.409 + 1/1.730) = 46 / 1.288 = 35.7 ; fast ends 35.7 + 14.1 = 49.8 ; slow ends 35.7 + 17.3 = 53.0
    body 53.0 (-4.1) ; ideal (no drain) 66 / 1.288 = 51.3 (-5.8)  ->  fp8 wall 63.8 / 62.1

This model says the slow member processes tiles at 1.73 us whether or not the
fast member is present.  The lone control says a CTA alone processes a tile in
0.95-1.03 us; the traced alone phase of the slow member (G = 8.8-9.3 us) is
consistent with 6.8 tiles at the lone rate, not 5.3 at 1.73.  **(a) is
withdrawn**, as round 3 §9.1 already withdrew it.

*(b) Paired / lone two-rate model (used).*  `T_s(paired) = 46.5 / (33 - 10.6 /
T_lone)`: 2.048 (T_lone 1.03) .. 2.133 (0.95); central `T_lone = 1.00 ->
T_s(paired) = 2.076`, `R = 0.7097 + 0.4817 = 1.191 tiles/us`.

    ideal (both end together):        66 / R = 55.4     (55.1 .. 56.0)                   -> -1.7 (-2.0 .. -1.1)
    uniform drain L = 10 each:        t_c = 46 / 1.191 = 38.6 ; fast ends 38.6 + 14.1 = 52.7 ;
                                      slow paired 14.1 / 2.076 = 6.8 tiles, 3.2 alone x 1.00 = 3.2  -> body 55.9   (-1.2)
    arrival-order split (3.1; closure at the slow member's second own claim, ~45.7 us after first K):
                                      fast claimed 39 / processed 32.4 (6.6 left) ; slow claimed 27 / processed 22.0 (5.0 left)
                                      fast ends 45.7 + 9.3 = 55.0 ; slow paired 9.3 / 2.076 = 4.5, 0.5 alone -> 55.5   (-1.6)

Body prediction **55.5-55.9 us (-1.2..-1.6; band -0.8..-1.9 over T_lone
0.95-1.03)**.  fp4: `T_f 1.355`, `T_s(paired) = 44.7 / (33 - 9.1 / 0.95) =
1.91`, `R = 1.262`; ideal 52.3 (-1.5); drain / arrival-order 53.8 -> 52.3-52.6
(-1.2..-1.5).  mixed: body 64.6 - fill 10.0 (fill doc: firstk 10.05) - 2.8 =
51.8; no fast / slow ctarec exists for mixed — scaled by the fp8 ratio: -1.2..-1.5.
a16: no asymmetry (1.1) -> the split lands 33/33 +- 1 unit; claims cost 0 on
the wall; predicted unchanged.

### 6.2 Wall

    wall = fill (unchanged: the loaders' prologue is [8]'s; rendezvous / table fill on the merge warp) + body + tail (2.8, + 0.6 worst case if a pool word must be re-read on the last merge)

| mode | today | predicted | band | target | reached |
|---|---|---|---|---|---|
| fp8 | 67.9 | **66.5** | 66.0-67.1 (+0.6 worst-case tail) | <= 58 | no (ideal split 65.8) |
| fp4 | 60.6 | **59.3** | 59.0-59.8 | <= 36 | no |
| mixed | 64.6 | **63.3** | 63.0-63.8 | <= 62 | **no — ideal split 63.0 before any tail cost** |
| a16 | 79.4 | 79.4 | 79.1-79.8 | parity | yes |

Per-role periods: unchanged by premise (no new operation on any per-tile path
of gemm0, gemm1 or the converters; the loaders gain ~5 instructions per tile
and ~20 on <= 9 tiles).  The claim's slack is 4 loader iterations (5.6 / 8.3
us) against a 1.5-2 us CAS; the table's slack is >= 20 us.

Sensitivity: if the dispatcher pairs two equal-speed CTAs (no asymmetry), the
arrival-order split is 33/33 +- 1 unit and the wall equals today's plus the
u = 2 quantisation on an already-balanced SM (<= T_s = 2 us on one SM, ~0 on
the wall median); the design cannot be worse than [8] by more than the
worst-case +0.6 us tail.

## 7. Verification artifacts, accept / reject

Build checks (each of the four q=1 modules F = 1, 2, 0, -1; recipe of r3pair
11.2: `nvcc -ptx` with the ninja flags + `ptxas -arch=sm_90a -v` + `nvdisasm
--print-line-info`, per-role attribution):

1. `ptxas -v`: no C7507; 0 bytes stack, 0 spill stores / loads in gemm0,
   gemm1, loaders, Q warp, converters.  Merge-warp per-item spills at 40 are
   recorded; if the *loaders* spill, `MIXED_KV_IO_REGS=48` with the
   per-branch `setmaxnreg` placement before any other change.
2. `cuobjdump -sass` per role: `USETMAXREG` 2 (3 with IO 48); `ATOMG` sites =
   `CAS.32` (claims, merge warp), `CAS.64` x 2 (rendezvous), `ADD.32`
   (semaphore), `INC` (`finished`) — all in IO warp 3; **none in gemm0,
   gemm1, loaders or converters**; converter PHASECHK 9 / 15 (K / V,
   unchanged), HGMMA 8 + 8, gemm0 PHASECHK / ARRIVE / BAR.SYNC 8 / 11 / 1,
   gemm1 17 / 13 / 1 (unchanged); `LDGSTS` 30 / 18 / 0 / 42, `UTMALDG` 8 / 0
   (unchanged); loader SASS: +1 `LDS` + `STS` on the per-tile path, the copy
   block under a warp-uniform branch.
3. `cuobjdump -res-usage`: REG 48, STACK 0; `sizeof(SharedMem)` 114 304
   (production) / 115 456 (trace);
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor` = 2; ncu
   `launch__occupancy_limit_registers` = `_shared_mem` = 2.

Conformance (`python tests/attention/run_xqa_mixed_page_transport.py`; exit
code = failures): the 60 [8] cases byte-identical when pairing is disabled
(`floor(T/P) < 16`, P = 1, `XQA_PAIR_DISABLE=1`) plus the round-3
`PAIR_CASES` under `XQA_PAIR_FORCE=1` with the fp32 reference at 3 bf16
output ulps (r3pair 12.7-12.8: paired launches are run-to-run
nondeterministic at that ulp because the closure point is a race): (4096, P
2), (4096, P 4), (2200, P 6), (285, P 2) x fp8 / fp4 / mixed / a16 = 16 -> 76
cases, all PASS; the trace-build path counters must show, over the matrix,
closing claims with take 0 / 1 / 2 each at least once, leaders reaching the
partner's pool while the partner is active, and partner-run items with a
sequence boundary inside the 8 tiles.

Trace (`MIXED_KV_TRACE 1`, 3 launches per mode, `parse_xqa_pair_trace.py` /
`pair_analysis2.py` ported from `wt/r3pair`):

- **Gate (the design's own):** the round-3 frontier-wait counters (per CTA:
  ns spent polling `frontier` because `g >= nbDecided`, and the number of such
  tiles; K loader, V loader, Q warp; plus the loaders' `tableReady` wait) —
  **0 ns and 0 tiles after tile 8** in every CTA of every launch (the only
  admissible waits are at tiles < 8 during the start burst, and the design
  predicts none there either since the frontier starts at S = 25).  Any
  nonzero count means the claim schedule is late: `Lambda` is re-derived
  from the measured CAS latency (stamp the CAS issue / return in the merge
  warp: two `%globaltimer` stamps per claim, printed in `ctarec`), not tuned.
- Role periods (CTA 0 tiles 3-7 / 11-18 / 27-32, and the TILE0 = 27 window
  covering the endgame tiles 25-34): every role within +-5 % of [8]'s table
  (fp8 1.2-1.35, fp4 1.2-1.25); `kl_start -> kl_iss` <= 0.10 us on endgame
  tiles (0.06-0.09 today); `kc_ready(16)`, `kc_ready(32)` at the steady-state
  period.
- Per-pair: tiles 66 per SM; fast member 37-40 tiles, slow 26-29; `end(slow)
  - end(fast)` median <= 2.5 us (model 0.5-3.2; today 9-10.6); tiles per
  SM-second within 3 % of today's `66 / 57.1` or better; rendezvous outcome
  132 leaders + 132 partners + 0 solo; claims per CTA <= 8, CAS retries <= 2
  per CTA (counter).
- Fill median unchanged (8.5 / 7.4 / 10.0 / 6.6 us fp8 / fp4 / mixed / a16);
  Q(0) `produced.arrive` at ~3 us (fill doc stamp 22).
- Merger: pool-word re-reads after the atomic (counter) — expected 0; the last
  merge's duration within 0.3 us of today's.

Timing (locked, `flock /tmp/mixedkv-gpu0.lock bash
/home/bigboi/mixedkv_remote_run.sh <checkout> r4pool sm90 transport_a16 fp8
fp4 mixed`, 5 x 5, q=1 rows; q=4 rows must equal today within spread; an
[8] control (r2p8 cached workspace) in the same session):

| mode | today | predicted | accept if median | reject if |
|---|---|---|---|---|
| fp8 | 67.9 | 66.5 | <= 67.0 | > 67.6, or any period moved > 5 %, or any frontier wait after tile 8 |
| fp4 | 60.6 | 59.3 | <= 59.8 | > 60.3 |
| mixed | 64.6 | 63.3 | <= 63.8 | > 64.3 |
| a16 | 79.4 | 79.4 | <= 79.8 | > 80.4 |

ncu (one launch): `sm__cycles_active.avg/.max` >= 0.95, `dram__bytes_read.sum`
unchanged per mode, `launch__grid_size` 264, occupancy limits 2 / 2,
`smsp__issue_active` within 2 points of [8]'s 62.9 % (fp8).

Reject = revert to [8]; `XQA_PAIR_DISABLE=1` is the A/B and must reproduce
[8] within 0.3 us (it did **not** in round 3: the fill restructuring cost 5-23
us even with the protocol off; here the disabled path *is* [8] plus a
frontier LDS per tile).

## 8. Do not build if

1. The confirmation trace on the unchanged [8] build no longer shows the
   pair asymmetry (G = slow.last - fast.last >= 8 us, 396 / 396 slot
   correlation): the lever's premise.
2. The roadmap keeps [15]+[7] (one CTA per SM) as the path to fp8 <= 58: the
   pool's premise (two co-resident CTAs) disappears and this code is dead on
   arrival.  Build the pool only if the layout stays at 2 CTAs/SM.
3. `ptxas -v` spills in the K / V loaders at 40 **and** at IO 48 (with the
   per-branch `setmaxnreg` placement): the loader's copy block then needs a
   smaller state, which is a design change.
4. `sizeof(SharedMem)` > 115 712 in the production **or** trace build
   (predicted 114 304 / 115 456; the trace margin is 256 B — one more
   accumulator does not fit).
5. The semaphore region is not zero at first use, or the header's offsets
   would depend on B / H / P (r3pair C18 / do-not-build 4).
6. A measurement shows `T_s(paired) / T_f > 1.64` for the bench (D = 8 could
   not hold the ideal shift): re-derive D, do not tune it.
7. The per-role SASS split shows a new `ATOMG` / `CAS` / PHASECHK / ARRIVE
   site inside the gemm0, gemm1, converter or loader per-tile paths beyond
   the loader's one `LDS.acquire` + `STS.release`.
8. `floor(T/P) < 16` for the bench shape (it is 32): pairing disabled, the
   change is a no-op for the gate.
9. The frontier-wait or `tableReady` counters are nonzero after tile 8 in
   the trace: the claim schedule is late and `Lambda` / the table-fill
   trigger must be re-derived from the stamped CAS latency before timing —
   never by moving a constant until the wall improves.
10. Any change to the Q warp's placement or to the loaders' prologue is
    proposed "while at it": Q(0) at ~3 us and the loaders' [8] prologue are
    the fill's known-good state (round 3 fill: -5 us a16 when Q queued behind
    fills; kMetaReady[0] gate behind the LDGSTS burst).
11. The expected gain (<= 1.6 us fp8, <= 1.5 fp4 / mixed) is judged not
    worth ~400 lines of protocol whose paired outputs are nondeterministic at
    the bf16 ulp (r3pair 12.7: an accepted property, but a property [8] does
    not have).

## 9. Go / no-go

**No-go as a standalone build now.**

- What is settled by this design: a per-SM pool *can* have a cheap producer —
  pre-filled records (one 8-tile LDG burst per operand, once), claims as <= 8
  blocking CAS on the idle merge warp five iterations ahead of need, [8]'s
  chunk fills and Q warp untouched; the steady state is [8] to the
  instruction, the endgame adds ~20 instructions on <= 9 loader tiles.  Its
  correctness rests on C21-C27, all of which are single-writer monotone words
  or exact CAS state, and on the round-3 rendezvous that ran 132 / 132 pairs
  live.  The gate is binary (frontier waits 0 after tile 8).
- What is also settled: the payoff.  Both members' rates are fixed by the
  hardware arbitration; the SM's paired throughput is 1.18-1.20 tiles/us and
  the lone rate ~1.0 tiles/us, so the best possible split saves 1.1-2.0 us of
  body on fp8 (55.1-56.0 vs 57.1), 1.1-1.8 on fp4, ~1.5 on mixed.  Predicted
  walls 66.5 / 59.3 / 63.3 against targets 58 / 36 / 62: none reached;
  mixed's ideal is 63.0 before any tail cost.  The one-rate model that
  promised -4 to -6 us is falsified by the lone control.
- The lever that can reach fp8 <= 58 on record ([15]+[7]+fill: ~57.5,
  round3_15 §6.3) runs one CTA per SM and removes the pair; the pool would
  then be dead code.  Building the pool is justified only if that path is
  abandoned and a ~1.3 us gain is accepted as worth the protocol and the
  bf16-ulp nondeterminism (do-not-build 2, 11).
- If it is built: exactly as sections 3-5, in the order (i) build checks
  (section 7 items 1-3), (ii) 76-case matrix, (iii) one trace of 3 launches
  read for the frontier counters and periods, (iv) one locked 5x5 bench with
  the [8] control in the same session; accept / reject per section 7.

Artefacts read for this design (no new remote run): `/tmp/r3pair/` (r3pair
trace, `r3pair_trace132.log`, `pair_analysis2.py`, `cadence.py`),
`/tmp/mixedkv-r2p8-ctrl.bench.txt`, `/tmp/r3pair2_trace*.log`,
`/tmp/r3pair3_trace{,_dis}.log` (frontier counters), `/tmp/r3fill/trace5*`
(fill stamps), `/tmp/r3pair_ptx/x{1,2,0,-1}.*` (register gate recipe);
documents as listed at the top.
