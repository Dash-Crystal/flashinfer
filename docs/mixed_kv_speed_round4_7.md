# Round 4, lever [7] — two fused consumer groups on alternate tiles (L8) on the one-CTA-per-SM substrate: the cadence lever (design only)

Kernel: `csrc/xqa/mha_sm90.cu` at `354914cc` (branch `claude/mixed-kv-sm90-tma`;
production q=1 kernel state = lever [8] at `039ba5c7`), the `MIXED_KV_PERSISTENT`
q=1 build (a16 / fp8 / fp4 / mixed).  Bench shape B=17, S=4096, 8 KV heads,
GQA 4, D=128, 64-token tiles, 16-token pages.  Line references are into
`mha_sm90.cu` at `354914cc`.  **No production kernel edit, no build and no
remote run in this phase**: every number below is taken from the round-2/3
records (`mixed_kv_speed_round2_lever8.md` section 10,
`mixed_kv_speed_round3_pair.md` sections 1 and 9.1,
`mixed_kv_speed_round3_15.md` sections 1, 5 and 6, `mixed_kv_speed_round3_fill.md`
sections 1.3 and 10, `mixed_kv_page_transport_backends.md` P0.3 / P0.5) or is
arithmetic on them.  No trace-only measurement was run: the one unknown that
decides this lever (the SM's sustained issue rate with 16-20 warps at a higher
instruction density) has no proxy on the existing kernel, see 1.5.

State (nkcut2 H200, locked 5x5 medians, q=1 us): transport_a16 79.4, fp8 67.9,
fp4 60.6, mixed 64.6.  Targets fp8 <= 58, fp4 <= 36 (analytically out of
reach, plan gate), mixed <= 62.

## 0. Verdict in one paragraph

The lever fuses gemm0 -> softmax -> gemm1 of one tile into one 128-thread
consumer group and runs two such groups on alternate tiles of the same CTA at
one CTA per SM, so the SM overlaps two dependent GEMM chains from one CTA —
what the 2-CTA layout does today by accident and pays for with the
slots-20..39 penalty (every issue-bound segment 16-28 % longer in the slow
member; wall follows the slow member: fp8 46.5 / 57.1).  The design is
straightforward on the consumer side (accumulators are 4 + 8 registers; a
fused group needs <= 56 registers, 3.1 / 5.1), removes the X ring, the xBar
pair, the xColMax / xColSum exchange and the ballot rescale from the per-tile
path (-300..-500 warp-instructions of ~4 000 per tile-SM), and couples the two
groups only once per item through a partial (O, m, l) exchange (5.4: ~0.15 us
per item, 2-3 items per CTA).  Its payoff, however, is bounded by a fact the
round-3 records make explicit (1.3): the pair already runs the SM at 63 %
issue-active with 40 warps, i.e. **the pair's 0.86 us per tile-SM is within
10 % of the issue-throughput floor of the current instruction count (0.76-0.84
us at 63 %)**; a one-CTA layout has 16-20 warps and sustained 56 % (2.24 IPC)
on this kernel.  With the fused consumer's instruction cut the model gives a
tile-SM cadence of **0.79 us (band 0.70-0.87)** — the task's <= 0.85 is met
centrally — and walls of **fp8 ~62 (57-67), fp4 ~58.5 (55.5-62), mixed ~63.5
(59-68), a16 ~79-80 (DRAM-bound, unchanged)** with the one-CTA fill (7.0 /
6.7 / 7.2 / 10) and today's tail.  fp8 <= 58 and mixed <= 62 sit at the
optimistic end of the band and need the fill cut on top; the central case is
fp8 -6, fp4 -2, mixed -1.  Recommendation (section 9): **go, staged** — step A
builds the fused consumer on the *existing* five-group layout at
`__launch_bounds__(640, 1)` (consumer branches, SharedMem, barrier counts and
one record bit change; IO and converters untouched; ~1/3 of the [15] + [7]
diff), gated by the pre-run SASS instruction-count read of 7.1 (a build whose
steady-state loop body is not >= 300 warp-instructions per tile below today's
is not run); step B ([15]'s loader merge) only if step A's trace shows the IO
warps' issue share on the critical path.  Do-not-build conditions in section 8.

## 1. Attribution evidence (what the lever acts on, and what bounds it)

### 1.1 The consumer chain is the cadence at one CTA per SM ([15] doc, 1.2)

P = 132 trace build (`/tmp/r3pair_trace132.log`), fp8, per-tile role segments
(us, medians over 132 CTAs; fp4 in parentheses):

| role | seg 1 | seg 2 | seg 3 | seg 4 | total |
|---|---|---|---|---|---|
| gemm0 | kwait 0.212 (0.201) | QK HGMMA + wait 0.455 (0.449) | colMax + softmax 0.233 (0.230) | X store + xBar.consumed wait + fence + arrive 0.274 (0.261) | **1.174 (1.141)** |
| gemm1 | vwait 0.234 (0.226) | xwait 0.241 (0.222) | rescale 0.238 (0.224) | PV HGMMA + wait 0.484 (0.490) | **1.197 (1.162)** |
| K converter | idle 0.765 (0.714) | expansion 0.383 (0.402) | | | 1.149 |
| V converter | idle 0.763 (0.726) | expansion 0.409 (0.415) | | | 1.172 |

No consumer wait is on data (K `done(t)` precedes gemm0's wait by 1 800-3 000
cycles; V by 1 000-1 500); the "kwait" segment is the loop tail plus an
already-complete `arrive_and_wait` (~190 cycles, P0.3 (d)).  Production
lone-CTA cadence 0.95 us fp8 / 0.89 fp4 / 0.96 mixed (trace-to-production
factor 0.81).  The pair (P = 264) delivers a tile-SM every 0.865 (fp8) /
0.815 (fp4) us because two CTAs' chains overlap, each stretched 1.35x (gemm0
1.17 -> 1.61 trace).  The chain, not the converters, is the bound: this is
the segment list the fused design re-arranges.

### 1.2 Slot priority: what one CTA per SM removes ([pair] 1.3-1.4, 9.1)

Slow-member / fast-member segment ratios (fp8): QK 1.16, colMax/softmax
1.23, X store 1.23, PV 1.17, expansion 1.28; memory segments 1.00-1.05.  396
of 396 pairs: the slow CTA occupies warp slots 20-39.  Two-rate model (9.1):
paired throughput 1.20 tiles/us (0.83 us per tile-SM while both run), the
slow member alone for 9.7-10.3 tiles at 1.03; the *balanced* pair would end
at body 54.5-55.1 vs 57.1 -> the asymmetry itself is worth **-2.0..-2.6 us**
fp8, -1.8 fp4.  A one-CTA design removes it by construction; anything beyond
-2.6 us must come from a cadence below 0.83.

### 1.3 The issue-throughput floor (ncu, [pair] 1.3 / [15] 1.3)

| metric | P = 264 (2 CTAs/SM, 40 warps) | P = 132 (1 CTA/SM, 20 warps) |
|---|---|---|
| `smsp__inst_executed.sum` per launch | 36.70 M | 36.16 M |
| per tile (8 712 tiles, incl. fill / merge) | 4 213 warp-instr | 4 150 |
| SM IPC (`sm__inst_executed` per active cycle) | 2.51 | 2.24 |
| issue active (% of cycles a scheduler issued) | 62.9 | 56.3 |
| active / eligible warps per scheduler | 9.41 / 1.58 | 4.99 / 0.97 |

Steady-state instruction count per tile-SM ~3 900-4 000 (the fill and merge
are ~5 % of the launch).  At IPC 2.51 that is 1 570 cycles = **0.79 us at 1.98
GHz**; the measured pair cadence 0.865 is 9 % above it (the remainder is the
slot-asymmetry drain of 1.2).  At IPC 2.24 (one CTA, 20 warps): 0.89 us — the
lone CTA's measured 0.95 is 7 % above *that*.  Reading: **both layouts run
within 10 % of what the SM issues at their warp count**; the chain-latency
reading of [15] 1.3 and this issue reading are the same fact seen from two
sides (a latency-bound chain interleaved with issue-dense converter warps
shows as 56-63 % issue-active with ~1 eligible warp).  Consequence for [7]:
its cadence is `instructions per tile-SM / (IPC x f)` with IPC in [2.2, 2.5]
for 16-20 warps, so the lever's real payoff is the instruction cut, and the
chain-overlap argument only matters insofar as it keeps ~1 eligible warp per
scheduler (which two independent chains plus the converters do, 1.1).

### 1.4 Chain doubling without converters: the P0.5 control (backends P0.5)

Converters skipped (`EXPERIMENT=1`), consumers only: cadence 0.839 us at 1
CTA/SM (5 warps/scheduler) vs 1.001 at 2 CTAs/SM (10 warps/scheduler) —
doubling the number of concurrent gemm0/gemm1 chains stretched each by
**1.19x** and gave 1.68x the SM throughput.  With converters present the
stretch was 1.35x (1.1).  These two numbers bracket the stretch a second
fused chain imposes on the first: s in [1.19, 1.35] (6.1).

### 1.5 What has no proxy today

The sustained issue rate of an SM with 16-20 warps at a per-cycle demand ~1.2x
today's (two fused chains + 8 converter warps) is not measurable on the
existing kernel: `XQA_PERSISTENT_CTAS=132` gives 20 warps at today's density
(2.24 IPC), P = 264 gives 40 warps at 2.51.  The model therefore carries IPC
as a band [2.2, 2.5] (6.1) and the design's first artifact (7.1) is the SASS
instruction count, readable before any run.

### 1.6 Fill and tail carried over

One-CTA fill (P = 132 trace): fp8 7.2, fp4 6.7, mixed ~7.2, a16 ~10 (start
burst of 132 x 3 tiles instead of 264; the fill-cut design of round 3 is
rejected and not assumed here).  Tail 2.6-2.8 us (last item's finalize +
cross-CTA merge), unchanged by the lever.  Item boundary +0.4 us on gemm0
(the 128-arrival `qBar` wait, fill doc finding 5) — the design keeps a `qBar`
wait per group per item (3.3) so this cost stays, x2 items per CTA.

## 2. Current data flow and control flow of the touched roles (as written at `354914cc`)

Five warp groups (`ctaWarpGroups = 5` :184, `__launch_bounds__(640, 2)` :1151),
`setmaxnreg.dec 40` (z <= 2) / `.inc 56` (z >= 3) :1444-1449.  `SharedMem`
:233-470: `k[nbKBuf x 2 parts]` (16 KB per stage), `reusedXVOutSwizzleBuf[nbXBuf]`
(union of the 1 KB bf16 X buffer and the 2 KB `OutSwizzleBuf`) :270-277,
`vBufs[nbVBuf]` :278, `q[2]` :323, `xColMax[nbXBuf]`, `xColSum[nbXBuf][4]`
:329-330, `gemm1AccColMax / Sum` :338-339, `gemm0WarpColMax[2][4]` :350,
`TileRecord` (32 B: pages[4], formats, tile word bits 0-7 validBeg / 8-15
validEnd / 16 first / 17 last / 18 partial / 19 ctaLast, idxReq, idxHeadGrp)
:388-396, `meta[2][2][16]` :401, `finalizedItems` :417, `kScales / vScales
[nbScaleTiles = 4]` :425-427, barriers :444-465.  `sizeof(SharedMem)` 113 664
B; `nbKBuf = nbVBuf = 3`, `nbXBuf = 2` (:140-149).

- **Barrier init** :1342-1398: `kBar[s]` produced = 128 gemm0 + 128 K
  converters (+ 32 loader when `mixedLoaderTma`), consumed = 128 + 32;
  `vBar[s]` mirror; `xBar[x]` 256 / 256 :1380; `qBar[b]` 160 / 160 :1386;
  `gemm0WarpGrpBar / gemm1WarpGrpBar` 128 :1388-1389; `kMetaReady / vMetaReady`
  32 :1393-1394.  Scan by IO warp 3 :1402-1405; `finalizedItems = 0` :1407;
  `__syncthreads` :1420; `nbCtaTiles = sched.x1 - sched.x0`.
- **gemm0** (z = 0) :1451-1703.  Pre-arrives `qBar[*].consumed`, `kBar[*].consumed`
  :1459-1464; `runningColMax` in registers (`RegColWiseVec` = 2 floats) :1476;
  per tile g: `kBar[g%3].produced.arrive_and_wait` :1515 -> tile word LDS
  :1523 -> on first: reset colMax, `qBar[j&1].produced.arrive_and_wait`
  :1526-1529 -> 2 parts x 4 k-steps of `m64n8k16` SS wgmma (K from
  `smem.k[stage*2+part]`, Q from `q[j&1][part]`) :1541-1590 -> `wait_group<0>`,
  `kBar.consumed.arrive` :1594-1595 -> scale + mask from the record :1601-1615
  -> `computeWarpGrpColMax_sync` (per-warp slot STS, `bar.sync 3,128`, fold 4
  slots into the running max) :1639 / :3711-3767 -> `warpGrpOnlineSoftmax`
  (exp2) :1641 / :3826 -> `computeWarpColSum` :1651 / :3846 -> `acc *
  kE4M3_MAX` -> `storeGemm0AccToShm` (bf16 convert, `xBar[g%2].consumed
  .arrive_and_wait` inside, 1 `stmatrix.x4`) :1663 / :3876-3925 -> lanes 0-3 STS
  `xColMax` (warp 0) and `xColSum[warp]` :1665-1677 -> `__syncwarp`,
  `fence.proxy.async.shared::cta`, `xBar.produced.arrive` :1687-1690 -> on
  last: `qBar[j&1].consumed.arrive`, j++ :1695-1698.
- **gemm1** (z = 1) :1704-2123.  Pre-arrives `vBar[*].consumed`, `xBar[*].consumed`
  :1706-1714; `accColMax / accColSum` register-resident `RegColWiseVecNoDup`
  (1 float each) :1724-1725; `Gemm1Acc acc{}` (= `GmmaAcc<128, 8>`, 8 floats)
  :1744; per tile: `vBar[g%3].produced.arrive_and_wait` :1761 -> record :1772
  -> on first: acc = 0, running max / sum reset :1777-1782 ->
  `xBar[g%2].produced.arrive_and_wait` :1793 -> `rescaleGemm1AccForNewColMax`
  (LDS xColMax, ballot mask, `expf`, shfl per column, 8 FMUL, LDS 4 xColSum)
  :1846 / :4321-4374 -> `gmma::fence` :1871 -> 4 k-steps x 2 M-instructions of
  SS wgmma (V from `vBufs[g%3][m]`, X as B operand), one commit and one
  `wait_group<0>` per tile :1873-1957 -> on last: publish colMax / colSum to
  smem :1996-1999, `gemm1WarpGrpSync`, route by record :2001-2006, partial ->
  scratch chunk `2c + isCtaLast` + `finalizeAndWriteOut_sync` :2007-2024, else
  output :2025-2036, thread 0 `st.release.cta finalizedItems = j + 1`
  :2039-2044, j++ -> `xBar.consumed.arrive`, `vBar.consumed.arrive` :2112-2114.
- **IO group** (z = 2) :2124-2632: K / V loader warps (chunk fills of `meta`,
  `kMetaReady`, per-tile `stageBar.consumed.arrive_and_wait` + A16 TMA), Q warp
  (`qBar[j&1].consumed.arrive_and_wait`, register load / store, `produced
  .arrive`) :2278-2298, merge warp (polls `finalizedItems` :2309, semaphore,
  last-arriver combine) :2299-2447.  Unchanged by this lever except barrier
  counts (3.6) and the finalize publication (4, C22).
- **Converters** (z = 3 K :2633-2712, z = 4 V :2713-2760): warp w owns page w;
  `issueKCopies(t)`: `kMetaReady` wait at `t % 16 == 0`, `kBar[t%3].consumed
  .wait_parity(toParity<3>(t))`, `issueCompressedPageCopies` :2660-2676;
  prologue issues `kAhead = nbKBuf - 1` tiles :2677-2681; per tile
  `waitGroup<kAhead-1>`, `__syncwarp`, `expandPackedStage`, fence, `produced
  .arrive`, issue `g + kAhead`, `commitGroup` :2682-2711.  Unchanged except
  ring depth (`nbKBuf`, `nbVBuf`, `nbScaleTiles`) and the consumed count (3.6).
- **Host**: `choosePersistentGridSize` (`XQA_PERSISTENT_CTAS` override),
  occupancy calculator -> P = ctasPerSm x SMs, `ScratchMem{scratch, 2P, 1}`.

## 3. New data flow and control flow

### 3.1 Layouts (the substrate choice is a staging decision, not a design one)

The consumer design (3.2-3.5) is identical in both layouts; only the
non-consumer roles and the launch bounds differ.

    step A  five groups, one CTA per SM:  z = 0, 1  consumer groups C0, C1 (fused, alternate tiles)
                                          z = 2     IO group as today (K / V loader, Q warp, merge warp)
                                          z = 3, 4  K / V converters as today
            __launch_bounds__(640, 1): 65 536 / 640 = 102 -> 96 registers per thread for every role,
            NO setmaxnreg (:1444-1449 deleted; A4's USETMAXREG count 2 -> 0).  One CTA per SM by
            registers (640 x 96 = 61 440 > 32 768) and by shared memory (5.2).
    step B  four groups ([15] 3.1): C0, C1, K side (converters + own-page A16 TMA + chunk fill),
            V side; __launch_bounds__(512, 1), 128 registers; merge in the finalizing group (C14),
            Q by TMA from the finalizing group (C21).

Register options enumerated for the fused consumer (5.1 gives the need,
<= 56):

| layout | threads | regs / thread (65 536 / T, rounded down to 8) | setmaxnreg | fits the need | note |
|---|---|---|---|---|---|
| (640, 1) five groups | 640 | 96 | none | yes (consumer <= 56, converters <= 64, IO <= 48) | step A |
| (512, 1) four groups | 512 | 128 | none | yes | step B ([15]) |
| (384, 1) three groups: C0, C1 + one converter group for K and V | 384 | 168 | none | registers yes; **issue budget no**: one converter warp would own two pages per tile (2 x ~330 warp-instr = 1.2-1.3 us of issue per 0.8 us tile) | rejected |
| (640, 2) today | 640 | 48 pool, 40 / 56 split | yes | consumer 56 > 40 (C7507 or spills) | rejected: [7] cannot live at 2 CTAs/SM |
| (768, 1) six groups: 3 consumer groups on tiles mod 3 | 768 | 80 | none | registers yes; smem: three private P buffers and three partials fit; issue: no gain once the SM is at its issue rate (1.3), and a third chain raises the per-item merge to 3 partials | not designed; the model of 6.1 says the second chain already reaches the issue floor |

### 3.2 Data flow (per CTA; g = CTA-local tile counter, group c = g & 1)

    records          fill (loader warp / K-side warp 0) -> meta[op][chunk][g % 16]; NEW bit 20 nextLast
    K stage g        converters (page w of tile g) -> k[(g % dK) * 2 + part]  -> kBar[g % dK].produced
    V stage g        converters -> vBufs[g % dV]                              -> vBar[g % dV].produced
    Q(j)             q[j & 1] (Q warp today; TMA from the finalizer of item j - 2 in step B)
    group c, tile g  kBar wait -> QK wgmma (K(g), Q(j)) -> Gemm0Acc (4 regs) -> mask, colMax (group-private
                     slots gemm0WarpColMax[c][g/2 % 2][4]) -> softmax, colSum -> bf16 -> P[c] (private 1 KB)
                     -> fence.proxy.async + bar.sync(c) -> rescale O(c) by exp(m_old - m_new) (registers only)
                     -> vBar wait -> PV wgmma (V(g), P[c]) -> O(c) += ... (Gemm1Acc, 8 regs) -> commit
                     (wait deferred to the next tile's QK wait_group<0>)
    item end         non-finalizing group publishes (O, m, l) -> smem.partial[j & 1] (4 KB + 64 B)
                     finalizing group (owner of the item's last tile) combines: O = O_L e^(m_L - m) + O_O e^(m_O - m),
                     l likewise -> finalize (output, or scratch 2c + isCtaLast) -> publication to the merger

The X ring (`reusedXVOutSwizzleBuf[nbXBuf]`), `xBar[]`, `xColMax`, `xColSum`,
`gemm1AccColMax / Sum` as inter-group state, and the `RegColWiseVecNoDup`
representation disappear.  Per group: `P[c]` (union with the group's
`OutSwizzleBuf`, 2 KB), `warpColMax[c][2][4]` (256 B), `pubColMax / pubColSum
[c]` (64 B, finalize publish), and the shared `partial[2]` slots with
`partialBar[2]` pairs.

### 3.3 Control flow of a consumer group (one body, parameterised by c = warpIdx.z)

    prologue   c == 0: pre-arrive kBar[*].consumed and vBar[*].consumed (phase-0 free; one arrival of 128 per
               stage, so exactly one group does it)
               O = 0 ; m = safeInitRowMax (dup form, 2 floats) ; l = 0 ; j = 0 (item counter, both groups count
               every item: j advances on any tile with the last bit, including tiles the group does not own -
               see "item tracking" below)
    for g = c, c + 2, c + 4, ... < G:
      [K]      kBar[g % dK].produced.arrive_and_wait                                   (1 mbarrier RT, data long ready)
               word = LDS meta[K][g].tile ; first / last / nextLast / partial / ctaLast bits
               if first:  O = 0 ; m = init ; l = 0 ; qBar[j & 1].produced.arrive_and_wait
      [QK]     gmma::fence ; 8 x m64n8k16 SS (K(g), Q(j)) ; commit ; wait_group<0>
               -- this wait also retires PV(g - 2) (same warpgroup, in-order groups): only now
               vBar[(g - 2) % dV].consumed.arrive  (if g >= 2)                        (deferred V release, C19)
               kBar[g % dK].consumed.arrive
      [S]      acc *= qkScale ; mask from the word (:1601-1615 unchanged)
               colMax = computeWarpGrpColMax_sync(warpRank, warpColMax[c][(g/2) % 2], m_new = m, acc)
                        (:3711 unchanged: per-warp slot STS, bar.sync(id_c, 128), fold; returns the new running max)
               scale = exp2((m_old - m_new) * log2e)    (2 values per lane; == 1 where the max did not move)
               warpGrpOnlineSoftmax(acc, m_new) ; colSum = computeWarpColSum(acc) ; l = l * scale + colSum
               (l is the group-wide sum: computeWarpColSum already reduces across the warp's 16 tokens; the
                4 warps' sums are folded through the same slot mechanism as colMax - one extra STS/LDS of
                4 x 32 B per tile, or the sum is kept per warp and folded once per item at publish/finalize;
                the design keeps it PER WARP (no per-tile exchange) and folds at item end, 5.4)
      [P]      acc *= kE4M3_MAX (kept for bit-compat of the bf16 P quantisation with today's X path; xvoScale
               keeps 1 / kE4M3_MAX) ; bf16 convert ; stmatrix.x4 into P[c]   (:3876 minus the barConsumed wait)
               fence.proxy.async.shared::cta ; bar.sync(id_c, 128)                       (C20: P visible to the
                                                                                          warpgroup's async proxy)
      [R]      rescaleAcc(O, scale)   (:4427: 8 FMUL, dup-form scale; no LDS, no ballot, no shfl)
               gmma::fence            (O written by non-wgmma instructions)
      [V]      vBar[g % dV].produced.arrive_and_wait
      [PV]     4 k-steps x 2 M-inst SS wgmma (V(g) as A, P[c] as B) ; commit ; NO wait
      [item]   if last (this group finalizes):
                 wait_group<0> ; vBar[g % dV].consumed.arrive
                 if not solo (first && last is solo): partialBar[j & 1].produced.wait_parity(toParity<2>(j)) ;
                     read partial -> combine into O, m, l (5.4) ; partialBar[j & 1].consumed.arrive
                 fold the 4 per-warp l into the group sum (slot STS + bar.sync + 4 LDS) ; publish m, l to
                     pubColMax / pubColSum[c] ; finalizeAndWriteOut_sync (:4547, group barrier = bar.sync(id_c))
                     to output or scratch 2 x cta + ctaLast
                 publication: step A  thread 0: st.release.cta partialDone[ctaLast] = 1 (C22) ;
                                      qBar[j & 1].consumed.arrive (count 128 + 32)
                              step B  C14 merge in this group ; elected lane: Q(j + 2) TMA into q[j & 1] (C21)
                 j++
               else if nextLast (the partner finalizes this item; this is the group's last tile of it):
                 wait_group<0> ; vBar[g % dV].consumed.arrive
                 partialBar[j & 1].consumed.wait_parity(toParity<2>(j)) (pre-arrived for j = 0, 1)
                 fold l ; STS O (8 floats per thread, 4 KB), m, l -> partial[j & 1] ; partialBar[j & 1].produced.arrive
                 j++
               (a group whose tile is neither last nor nextLast does nothing at item level)
    epilogue   wait_group<0> ; the last owned tile's V release (if not already done by an item branch)

Item tracking: both groups must count the same j for the same item because
`q[j & 1]`, `partial[j & 1]` and `partialBar[j & 1]` are indexed by it.  A
group increments j on a tile it owns that carries `last` or `nextLast`
(exactly one of the two groups' tiles of every item with >= 2 tiles carries
`nextLast`, the other carries `last`; a solo item carries `last` only, so the
non-owner never sees it).  Hence for the non-owner of a solo item, j does not
advance — **so j is not the item index for that group**.  Fix by
construction: the record already carries the item's identity (`idxReq`,
`idxHeadGrp`); the design adds nothing to the record for j but derives the
Q-buffer and partial-slot parity from a per-CTA *item ordinal* written into
the record by the fill (`tile` word bits 21-23: item ordinal mod 8 — the fill
counts items as its cursor crosses them; C23).  Both groups then index
`q[ord & 1]`, `partial[ord & 1]`, `partialBar[ord & 1]` and the parity
`toParity<2>(ord)` from the record, never from a private counter.  The
per-group `j` is dropped.

### 3.4 What the converters and loaders see

Nothing per tile: stage `s = g % d` is released by the owner group of tile g
with one arrival of 128 on `consumed[s]`; the converters' `wait_parity` on
`consumed[s]` and the loaders' `arrive_and_wait` are unchanged.  Ring depth:
`dK = 5`, `dV = 6` (5.2 / 5.3; V is released one group-tile later because of
the deferred PV wait), `nbScaleTiles = d + 1` per operand (already a per-operand
constant in the scale ring's use: `kScales[t % nbScaleTiles]`, so it becomes
`nbKScaleTiles = 6`, `nbVScaleTiles = 7`).  `kAhead = dK - 1 = 4`, `vAhead = 5`
tiles of copies in flight; the K and V converter code is otherwise textually
unchanged.

### 3.5 Merge publication and Q hand-off in step A

- `finalizedItems` (:417, :1407, :2039-2044, :2309) is replaced by two words
  `partialDone[2]` (index = the item's `ctaLast` bit): the finalizer of a
  partial item writes its word with `st.release.cta` after
  `finalizeAndWriteOut_sync`; the merge warp polls the word of the partial
  item it is about to merge (`ld.acquire.cta`, as today) — C22.  The merge
  warp's item enumeration (which of the CTA's items are partial, and their
  chunk tags) is static from `sched` as today; only the readiness signal
  changes from a counter to a per-slot flag, because two groups finalize
  alternate items and the counter's monotonicity is no longer guaranteed
  (finalize(j + 1) by the other group can complete before finalize(j)).
- Q warp (:2278-2298): unchanged code; `qBar[b].produced` count 256 + 32 (both
  groups `arrive_and_wait` at their first tile of the item), `qBar[b].consumed`
  count 128 + 32 (**the finalizer only** arrives, after its finalize; a group
  with no tile in a solo item never arrives, and the finalizer's arrive is
  ordered after the partner's last QK of the item through the partial
  protocol, C21).  Item j's Q is needed by both groups at their first owned
  tile of the item; the Q warp stores Q(j + 1) into `q[(j+1) & 1]` after
  `consumed` of item j - 1, as today.

## 4. Barrier, parity and ownership invariants

D1-D6, C1-C7 (`mixed_kv_page_transport_dataflow.md`), C8-C13 ([8] design
section 3 / 8.1-8.11) and C14-C17 ([15] design section 4) remain the reference.
Restated or added:

- **C4 (barrier accounting), restated.**

  | barrier | produced count | tx | consumed count | waiters |
  |---|---|---|---|---|
  | `kBar[s]`, s < dK = 5 | 128 (owner group) + 128 K converters (+ 32 loader in step A a16 / mixed) | loader `arrive_tx` (A) / converter `expect_tx` (B) for A16 pages | 128 (owner) + 32 loader (A) / 128 (B) | owner `arrive_and_wait`; converters `wait_parity`; loader `arrive_and_wait` (A) |
  | `vBar[s]`, s < dV = 6 | mirror | mirror | mirror | mirror |
  | `xBar[*]` | **deleted** | | | |
  | `qBar[b]` | 256 + 32 (A) / 256 + 1 (B, elected `arrive.expect_tx`) | 2 KB (B) | 128 + 32 (A) / deleted (B) | both groups `arrive_and_wait`; Q warp (A) |
  | `partialBar[p]`, p < 2 | 128 (publisher group) | - | 128 (finalizer group) | finalizer `wait_parity`(produced); publisher `wait_parity`(consumed), pre-arrived phase 0 by C0's prologue for both slots |
  | named barriers | ids 3 (C0) and 4 (C1), 128 threads each: colMax exchange, P visibility, l fold, finalize's internal syncs | | | `gemm0WarpGrpBar / gemm1WarpGrpBar` mbarriers become `cBar[c]` (finalize's `warpGrpBar` argument :4549) |

- **C8 (consumer-gated parity) with alternating owners.**  Phase k + 1 of
  `consumed[s]` completes with the single 128-arrival of the owner of tile
  `kd + s` (+ the loader's 32 in step A), which follows that owner's
  `produced(kd + s)` wait, which needs every converter warp's arrive for the
  tile, which follows each warp's `issue(kd + s)` wait on phase k.  The
  argument never used *which* group arrives, only that exactly one group
  arrives once per tile — true by construction (tile g is owned by g & 1 and
  by nobody else; the prologue pre-arrive is done by C0 alone).  Holds for
  every d >= 2.
- **C9 / C13 (partition, rings and stage indices independent of items)**
  unchanged: P = 132, 66 tiles per CTA; the group parity is `g & 1`, a
  function of the CTA-local counter only.
- **C10 (record visibility).**  Owner of tile g reads `meta[op][g]` after
  `produced(g)`, the same synchronizes-with chain as today (fill -> `metaReady`
  -> converter `issue(g)` acquire -> converter arrive -> owner's wait).  The
  finalizer of item j additionally reads the item ordinal / next-item identity
  from record g = b_j only (its own tile).  Step B's Q(j + 2) identity: the
  record of tile b_j + 1 (the next item's first tile) is visible to a thread
  that passed `produced(b_j)`: every converter warp's arrive for b_j is
  program-ordered after its `issue(b_j + 1)` (issued at iteration
  `b_j + 1 - kAhead <= b_j - 1` for kAhead >= 2), which acquired the chunk of
  b_j + 1 through `kMetaReady`.  Holds for dK >= 3.  WAR of record slots
  (C10 of [8] 8.1): the newest reader of record g is now the owner group (at
  its `consumed.arrive`, after the LDS, as today :1519-1523) — the bound
  `L <= 15 - x` loses its x term (no X ring): `L <= 15`; today's lead 4 holds.
- **C11 (scratch slot rule)** unchanged: `2c + isCtaLast`, at most two
  partial items per CTA; the finalizer writes the chunk whichever group it is.
- **C12 (merge hand-off)** restated as **C22**: readiness of a partial item is
  a per-slot flag `partialDone[isCtaLast]` written by the finalizer with
  `st.release.cta` after `finalizeAndWriteOut_sync` (whose closing group
  barrier orders every thread's scratch stores before thread 0's release) and
  read by the merge warp with `ld.acquire.cta`; the counter is gone because
  finalizes are no longer totally ordered across the two groups.  In step B,
  C14 (merge in the finalizing group) applies verbatim with "gemm1" read as
  "the finalizing group".
- **C15 / C21 (Q buffer ownership).**  `q[b]` for item j (b = ord(j) & 1) is
  read by both groups' QK wgmma during item j.  Step A: the Q warp overwrites
  `q[b]` for item j + 2 after `qBar[b].consumed` phase completes, which needs
  the finalizer of item j's 128 arrivals, issued after its finalize, which is
  after (a) its own `wait_group<0>` of the item's last QK (program order) and
  (b) the partner's publish (partialBar produced), which the partner issued
  after its own `wait_group<0>` of its last QK of the item (program order) —
  or, for a solo item, there is no partner read.  Step B: the finalizer's
  elected lane issues the Q(j + 2) TMA at the same point; same argument;
  `qBar[b].produced` count 256 + 1, phases alternate strictly because the
  expect_tx for item j + 2 is issued after both groups' `produced` waits for
  item j (the finalizer's own wait precedes in program order; the partner's
  wait for item j precedes its publish, which precedes the finalizer's
  combine).
- **New C18 (alternate-tile ownership).**  Tile g of the CTA is consumed
  (both stage waits, both stage releases, its P, its colMax slots) by group
  `g & 1` only.  Items are the only place the groups meet (C20).  A
  consequence is that the running (m, l, O) of a group covers *its* tiles of
  the item — an online-softmax partial in the split-KV sense, combined at
  item end with the standard formula (the same arithmetic as the cross-CTA
  combine :2333-2432 and FA3's `combine`); the single-chain result is
  recovered up to fp32 rounding (the conformance reference is fp32 attention
  at 3 bf16 ulps since the pair track's fix, so no test change is needed; a
  solo item is bit-identical to today's path).
- **New C19 (deferred PV release).**  A group releases V(g) (`vBar.consumed
  .arrive`) only after `wait_group<0>` at tile g + 2's QK step (or at an item
  branch / the epilogue).  wgmma groups of one warpgroup complete in order, so
  that wait covers PV(g).  Cost: one more V stage live per group -> dV = dK + 1
  (5.2).  Alternative (not chosen): wait after PV(g) and release immediately,
  dV = dK, +~0.1 us per group-tile exposed (6.1 lists both).
- **New C20 (P buffer, within one group).**  Writers: the group's 4 warps
  (`stmatrix`, generic proxy).  Reader: the group's PV wgmma (async proxy).
  RAW: `fence.proxy.async.shared::cta` by every writer thread, then `bar.sync
  id_c, 128`, then the wgmma — the CUTLASS / FA3 register-P-to-smem pattern;
  today's equivalent is the fence + `xBar.produced.arrive` / `arrive_and_wait`
  edge between two groups.  WAR: P(g + 2) is written after this group's
  `wait_group<0>` at tile g + 2's QK, which retired PV(g) (in-order
  completion within the warpgroup): a single P buffer per group suffices;
  no cross-group hazard because P[c] is private.
- **New C20b (partial exchange).**  `partial[p]`, p = ord & 1, is written by
  the publisher (owner of tile b_j - 1) at that tile and read by the
  finalizer (owner of b_j) at b_j.  RAW: `partialBar[p].produced` (128
  arrivals after the STS; the finalizer's `wait_parity(toParity<2>(ord))`
  acquires).  WAR: the next writer of slot p is the publisher of item j + 2
  at tile b_{j+2} - 1 >= b_j + 1; it waits `partialBar[p].consumed` phase
  `toParity<2>(ord + 2)`, completed by the finalizer of item j's 128 arrivals
  after its reads.  Liveness: every wait in the protocol is on a group's
  action at a *strictly earlier tile* of the CTA (publisher at b_j - 1 <
  finalizer at b_j; finalizer's read at b_j < publisher's next write at
  b_{j+2} - 1), and stage production does not depend on items (C9), so no
  cycle exists; the only stalls are the finalizer waiting for a publish that
  is at most one group-tile away, and a publisher two items ahead of a slow
  finalizer (bounded by one finalize, ~1-2 us, only with 1-2-tile items).
- **New C23 (record bits).**  The fill sets bit 20 `nextLast` on tile g iff
  g + 1 is the item's last tile (the cursor knows the item's remaining tile
  count without touching the next chunk) and bits 21-23 = item ordinal mod 8
  (a per-CTA item counter the cursor increments when it enters a new item;
  only bit 21 is consumed, the others are slack for a deeper Q / partial
  ring).  `solo = first && last`.  These bits are the only fill change; the
  record stays 32 B.
- **C16 / C17** ([15]) apply to step B unchanged.  **C7-class** cases (items
  begin mid-pipeline, many items per CTA, T < P) remain in the matrix; the
  2-group protocol adds the cases "1-tile item between two long items" (solo
  with the non-owner idle at item level), "2-tile item" (publisher and
  finalizer on consecutive tiles), and "item whose first tile is odd" (C1
  owns the first tile) — all exercised by the existing `XQA_PERSISTENT_CTAS`
  = 1 / 3 / 5 runs with short sequences (S = 64, 128) which the conformance
  runner has.

## 5. Budgets

### 5.1 Registers (fused consumer; per-thread live sets, no runtime-indexed arrays — C2)

| state | registers | lifetime |
|---|---|---|
| `Gemm1Acc` O (`GmmaAcc<128, 8>` = 2 x 1 core mats x 4) | 8 | whole item |
| running max m (dup form `RegColWiseVec`, 2 floats) ; running sum l (per warp, 2) | 4 | whole item |
| wgmma descriptors: Q base, K base, V base, P base (64-bit) | 8 | loop |
| `qkScale`, `xvoScale`, `kE4M3_MAX` fold | 3 | loop |
| loop: g, G, warpRank, lane, c, tile word, stage indices (2), record address | 9 | loop |
| smem addresses (P[c], warpColMax[c], partial, barriers) | 4 | loop |
| **persistent total** | **36** | |
| transient at [S]: `Gemm0Acc` 4, colMax 2, colSum 2, scale 2, exp temporaries 4 | 14 | QK -> P |
| transient at [P]: bf16 packed acc 2, stmatrix address 1 | 3 | |
| transient at item end: combine — partner O 8 (loaded 2 at a time: 2), partner m / l 4, scales 2 | 8 | last tile |
| **peak** | **~52** | |

Today's ptxas reads (pair doc 11.2): gemm0 R27-R39, gemm1 R31, converters
R50-R51, IO R45.  The fused group's need is <= 56 (the sum of the two
persistent sets plus the larger transient), so it fits **64** (step-A cap 96,
step-B cap 128 with margin).  Converters unchanged (R51 <= 64 at either cap;
at 96 / 128 ptxas may use more without spilling — record REG, require STACK
0).  Any consumer REG > 64 is a code-shape regression (C2 / C3), not a need.
No `setmaxnreg`: with `__launch_bounds__(640, 1)` (A) or `(512, 1)` (B) every
role compiles at the launch cap and the C7507 class of failure cannot occur.

### 5.2 Shared memory (one CTA per SM: 232 448 B usable)

    smem(dK, dV) = 16 384 (dK + dV)                         K / V stages (64 tokens x 2 parts x 128 B)
                 + 2 x 2 048                                P[c] / OutSwizzleBuf union per group
                 + 2 x 2 048                                q[2]
                 + 2 048                                    records meta[2][2][16] x 32 B
                 + 512 (dK + 1) + 512 (dV + 1)              scale rings
                 + 2 x (4 096 + 64)                         partial[2] (O 128 x 8 fp32 + m, l)
                 + 2 x 256 + 2 x 64                         warpColMax[c][2][4], pubColMax / Sum[c]
                 + sched 48 + partialDone 8 + barriers ~250 + trace 1 032 (trace build)

| dK | dV | bytes (production) | fits 232 448 | forces 1 CTA/SM (> 115 712) | note |
|---|---|---|---|---|---|
| 3 | 3 | 121 700 | yes | yes | today's depth; K landing slack 1 tile at cadence 0.8 (6.1) — too shallow |
| 4 | 5 | 172 500 | yes | yes | minimum with the deferred V release (5.3) |
| **5** | **6** | **206 400** | yes | yes | **specification** |
| 5 | 5 | 189 500 | yes | yes | fallback without the deferred V release (C19 alternative) |
| 6 | 7 | 240 300 | **no** | - | |

`static_assert(smemSize <= 232448)` replaces :478; the host already sets
`cudaFuncAttributeMaxDynamicSharedMemorySize` from `hostSmemSize`; the
occupancy calculator returns 1 (do-not-build 5 if it does not).

### 5.3 Ring depth from the lookahead (the "2 x lookahead" question)

Stages of one operand live at any instant: `consuming` (one per group: 2) +
`released late` (V only, C19: 1) + `landed, waiting` (>= 1 so the owner never
waits) + `in flight` (copy latency / cadence).  Copy latency (fill doc 1.3:
RT4 1.54 fp8 / 1.70 fp4 steady 1.2-2.0 us; pair doc 1.3 issued -> landed
155-234 ns when the memory system is not saturated) at cadence 0.79:
1.5-2.5 tiles.  dK = 2 + 1 + 2 = 5 covers 2 tiles in flight with one landed;
dV = 6.  This is "2 x lookahead" in the sense that two consumers each hold a
stage while the same two-tile copy lead of today (kAhead = 2) stays; the
converters' lookahead becomes kAhead = 4 / vAhead = 5 tiles — the same
copy-lead-in-tiles as the two-CTA layout has *per SM* today (2 CTAs x 2).

### 5.4 Per-item partial exchange cost

Publisher: fold l (4 STS + bar.sync + 4 LDS x 2 floats), 8 STS.32 of O per
thread (4 KB per group; 32 wavefronts), 2 STS of m / l, `produced.arrive`:
~40 warp-instructions, ~0.05 us.  Finalizer: `wait_parity` (already complete
in the steady state: the publisher ran one tile earlier), 8 LDS + 2 LDS,
m = max(m_L, m_O), a = exp2((m_L - m) log2e), b = exp2((m_O - m) log2e),
O = a O_L + b O_O (8 FMA), l = a l_L + b l_O: ~50 warp-instructions, ~0.05 us
plus one mbarrier RT (~0.1 us under load, P0.3 (d)).  **<= 0.15-0.2 us per
item, 2-3 items per CTA at S = 4096 (<= 0.5 us per CTA, on the finalizer's
chain only, overlapped by the partner).**  The exchange replaces today's per
item `gemm1AccColMax / Sum` publish + `gemm1WarpGrpSync`, so the net per-item
cost is ~0.1 us.  For short sequences (S = 512: 8-tile items, 8 items per
CTA) the cost is ~1.5 us per CTA (~2 % of a ~70 us wall) — recorded, not a
gate for the bench shape.

### 5.5 Issue budget per tile-SM

| item | today (warp-instr per tile-SM, pair) | fused design | evidence |
|---|---|---|---|
| gemm0 body (wait, 8 HGMMA + descriptors, mask, colMax exchange, softmax, colSum, bf16, stmatrix, xCol STS, fence, 2 mbarrier ops + 1 X wait) | ~1 150 | | P0.3 (d): 8 HGMMA = 185 cyc floor vs 589-738 measured -> ~400 cyc of descriptor / issue |
| gemm1 body (2 waits, LDS xColMax + 4 xColSum, ballot, shfl, expf, 8 FMUL, 8 HGMMA, commit, wait, 2 arrives) | ~950 | | |
| **fused consumer body** (2 waits, 16 HGMMA + descriptors, mask, colMax exchange, softmax, colSum, bf16, stmatrix, fence, bar.sync, 8 FMUL rescale, 2 arrives) | | **~1 650-1 800** | removes: X wait / arrive x 2 groups (4 mbarrier ops ~ 60), xColMax / xColSum STS + LDS (~40), ballot + shfl + expf rescale (~80), `gemm1WarpGrpSync` per item, second record LDS, second loop overhead (~100) |
| converters: 8 warps x (expansion 188 + copy issue 71-90 + glue 50) | ~2 600 (2 CTAs) -> per tile-SM 1 300 x 2 tiles / 2 = **~1 300** | ~1 300 | P0.4 / [16] counts; unchanged code |
| loaders, Q, merge (step A) | ~150 | ~150 (0 in step B) | [15] 5.3 |
| **total per tile-SM** | **~3 900-4 000** (ncu 4 150-4 213 incl. fill / merge) | **~3 400-3 600 (A), ~3 250-3 450 (B)** | |

Converter warp busy time per tile at cadence 0.79: expansion 0.31-0.33
(production, from 0.383-0.41 trace x 0.81) + copy issue 0.2 + glue 0.05 =
0.56-0.58 us = **71-73 % of the cadence**, before contention stretch; at the
pair's fast-member stretch (1.44x on the expansion) 0.70 = 89 %.  The
converters are therefore co-critical at the design cadence and are the first
suspect if the trace shows the owner waiting on data (7, gate 4).

## 6. Predicted periods and wall

### 6.1 Cadence: two models that must agree

**(a) Chain model.**  Fused chain of one group, production us (trace
segments of 1.1 x 0.81, adjusted per 3.3):

    K wait + loop tail          0.21 x 0.81 = 0.17
    QK 8 HGMMA + wait           0.455 x 0.81 = 0.37
    colMax exchange + softmax   0.233 x 0.81 = 0.19
    P store + fence + bar.sync  (0.274 - 0.096 xBar RT) x 0.81 = 0.14
    rescale (registers)         0.06
    V wait (complete)           0.08
    PV 8 HGMMA issue (wait deferred)   0.30      (0.484 x 0.81 = 0.39 with the wait)
    T_chain = 1.31 (deferred wait) / 1.40 (immediate wait)

Two groups, no interference: 0.66 / 0.70.  Interference from the second chain
and the converters: s in [1.19, 1.35] (1.4): **0.78-0.88 (deferred), 0.83-0.95
(immediate)**.

**(b) Issue model.**  instr per tile-SM / (IPC x 1.98 GHz): 3 400-3 600 (A) at
IPC 2.2-2.5 -> **0.69-0.83**; step B 3 250-3 450 -> 0.66-0.79.

The two bands overlap at **0.78-0.83**; the design value is **0.79** (A),
0.76 (B), band 0.70-0.87.  The task's target <= 0.85 is met centrally and
missed at the pessimistic end (s = 1.35 with the immediate wait, or IPC 2.2
with no instruction cut).  fp4 scales by the lone ratio 0.89 / 0.95 = 0.94
(lighter converters): 0.74; mixed by 0.96 / 0.95: 0.80.

### 6.2 Two-rate / slot model, restated for one CTA per SM

Today (fp8): wall = fill 8.5 + [paired phase: 66 - 2L tiles at 1.20 tiles/us]
+ [fast member finishes its L, slow member alone at 1.03 for ~10 tiles] +
tail 2.8 = 67.9 (pair doc 9.1).  One CTA per SM has one rate: wall = fill +
66 x cadence + tail; the ~10 alone-tiles at 1.03 and the 2-2.6 us imbalance
disappear by construction, and the fill's 264-CTA start burst halves (7.0 vs
8.5).  Both rates are in the cadence band above; no slot term remains.

### 6.3 Wall

    wall = fill(1 CTA/SM) + 66 x cadence + tail

| mode | today | fill | cadence (band) | tail | **predicted** (band) | target | gain (central) |
|---|---|---|---|---|---|---|---|
| fp8 | 67.9 | 7.0 | 0.79 (0.70-0.87) | 2.8 | **62.0** (56.0-67.2) | <= 58 | -5.9 |
| fp4 | 60.6 | 6.7 | 0.74 (0.66-0.82) | 2.6 | **58.1** (52.9-63.4) | <= 36 | -2.5 |
| mixed | 64.6 | 7.2 | 0.80 (0.71-0.88) | 2.8 | **62.8** (56.9-68.1) | <= 62 | -1.8 |
| a16 | 79.4 | 10 | DRAM: 279 MB / 4.1 TB/s = 68 us body (1.03 per tile-SM) | 3 | **~80** (79-81) | parity | 0 |

Step B (no IO warps): -0.03 us per tile -> fp8 ~60, mixed ~61 central.  With
the fill item (a working design, not the rejected round-3 one; -2..-4 us on
every compressed mode): fp8 58-60, mixed 59-61 central — **the fp8 target is
reached only at the optimistic end of this lever's band plus the fill; the
mixed target centrally only with step B + fill.**  fp4's target is out of
reach as the plan's gate check states.

### 6.4 Why the plan's original [7] number (0.25-0.3 us per tile per CTA) does not hold

It assumed ~900 cycles per tile per group and two groups at 2 CTAs/SM; the
measured gemm0 chain alone is 1 880 cycles in production (0.95 us) with
data always ready (P0.3, [15] 1.2), the 8-HGMMA QK step costs 590-740 cycles
against a 185-cycle floor, and the SM issues at 2.2-2.5 IPC.  The fused
chain is ~2 600 cycles; two of them give 1 300 cycles = 0.66 us only without
interference, and the issue model caps the SM at ~0.7.  The 62 -> 9-21 us
"consumer floor" of the plan is superseded by the issue floor of 1.3.

## 7. Verification artifacts, accept / reject (read before the confirmation run; one run)

Build (each module a16 / fp8 / fp4 / mixed; ptxas recipe as [15] 7):

1. `ptxas -v`: 0 bytes stack, 0 spill; REG <= 96 (A) / 128 (B) by
   construction — **record REG per role region** (`nvdisasm --print-line-info`
   + the pair track's `/tmp/r3pair_regs.py`): consumer <= 64, converters <=
   64, IO <= 48 (A); no `USETMAXREG` (count 0; A4 updated).
2. `cuobjdump -sass`, steady-state loop bodies (line-info bounded):
   consumer body **<= 1 800 warp-level SASS instructions per tile** and the
   kernel total per tile-SM (consumer + 8 x converter body + IO bodies) **<=
   3 600** (A) — computed from the SASS, not measured; if the consumer body is
   not >= 300 below today's gemm0 + gemm1 bodies (~2 100), the issue model
   predicts >= 0.82 and the build is **not run** (do-not-build 2).  Counts:
   HGMMA 16 in one consumer body (the two groups share it: 16 total in the
   consumer path, 0 elsewhere); `SYNCS.PHASECHK` in the consumer loop = 2 (K,
   V) + item-level (qBar, partialBar); `SYNCS.ARRIVE.TRANS64` in the loop = 2
   (K, V consumed) + item-level; `BAR.SYNC` per tile = 2 (colMax, P) with ids
   3 / 4 by group; `STSM` 1 per tile; `FENCE.VIEW.ASYNC.S` 1 per tile; no
   `VOTE` / `SHFL` in the rescale (the dup-form register rescale); `LDL` =
   `STL` = 0; converter body byte-identical to today's except the ring
   constants (`toParity<5>` / `<6>` immediates); `UTMALDG` 8 (a16, mixed) / 0.
3. `cuobjdump -res-usage`: STACK 0; `sizeof(SharedMem)` 206 4xx B (207 4xx
   trace); occupancy calculator 1; `launch__grid_size` 132.

Conformance: `python tests/attention/run_xqa_mixed_page_transport.py` = all
cases (incl. `XQA_PERSISTENT_CTAS` = 1 / 3 / 5 and S = 64 / 128 short-item
cases: solo items, 2-tile items, odd first tile, C7 class), fp32 reference at
3 bf16 ulps (already in place).

Timing (locked, `mixedkv_remote_run.sh <checkout> r4p7 sm90 transport_a16 fp8
fp4 mixed`, 5 x 5, min / median / max; q = 4 rows unchanged):

| mode | accept (median) | predicted | reject if |
|---|---|---|---|
| fp8 | <= 63.5 (-6.5 %) | 62.0 | > 66.0 (inside the pair's balanced-split bound -2.6 -> the fusion bought nothing) |
| fp4 | <= 59.0 | 58.1 | > 60.6 |
| mixed | <= 63.0 | 62.8 | > 64.6 |
| a16 | <= 80.5 | ~80 | > 82 |

Trace (`MIXED_KV_TRACE 1`, per-group stamps: K-wait passed, QK done, P
stored, PV issued, item publish / combine; `parse_xqa_ctarec_roles.py`
extended with the two groups' accumulators):

- residency probe 1 on every CTA, 132 records, body unimodal (max - min <= 6 us);
- **tile-SM cadence** (group c's stamp(g) -> stamp(g + 2), halved, medians
  over tiles 4-60) **<= 0.85 us** accept; > 0.90 reject; the value pins the
  IPC of 1.3 for the next round;
- the K / V wait segments of both groups <= 0.12 us (no wait on data); if
  K-wait > 0.2 with the converter `ready` (idle) segment < 15 % of the
  cadence, the converters pace (gate 4 of section 8 for the next round, not a
  reject of this build);
- partial exchange: publish -> combine-done <= 0.3 us per item;
- item boundary: the first owned tile's K-wait -> QK within +0.4 us of other
  tiles (the qBar wait; unchanged mechanism).

ncu (fp8, one launch, `--launch-skip 4`): `smsp__warps_active` ~4-5 per
scheduler (A) / 4 (B), `smsp__issue_active` >= 56 % (if < 50 %: the chains
are not overlapping — check the partial protocol for an unintended per-tile
coupling), `sm__inst_executed` <= 33 M (-10 % vs 36.7 M; the issue model's
input), `dram__bytes_read.sum` unchanged per mode,
`l1tex__data_pipe_lsu_wavefronts_mem_shared` per tile within 5 % of today
minus the X traffic (C11-class read).

## 8. Do not build if

1. **Registers**: `ptxas -v` at `(640, 1)` / `(512, 1)` shows spills in any
   role, or the consumer region's REG > 64 (a fused chain needing more than
   64 is a code-shape defect: runtime-indexed acc access, both rescale forms
   materialised, or the combine's partner O held whole; fix the shape, do not
   raise the cap).
2. **Instruction count (pre-run, from SASS)**: the consumer loop body is not
   at least 300 warp-instructions per tile below today's gemm0 + gemm1 bodies,
   or the per-tile-SM total exceeds 3 600 (A) / 3 450 (B).  Then the issue
   model predicts >= 0.82-0.85 us and the wall band's lower edge moves above
   60 (fp8) — no target is in reach; do not run.
3. **Shared memory**: `sizeof(SharedMem)` at dK = 5, dV = 6 exceeds 232 448 B
   (trace build included) — drop to 5 / 5 with the immediate PV wait (C19
   alternative), never below 4 / 5.
4. **Converter co-criticality**: the SASS count of the converter body per
   tile (expansion + copy issue + glue) x 8 warps exceeds 45 % of the
   per-tile-SM total (today ~1 300 of ~4 000 = 33 %) — the fused design then
   moves the bound to the converters at the cadence it targets, and the
   converter instruction cut ([16]-class, or A16-by-TMA for more pages) must
   come first.  (Not met today: 33 %.)
5. **Occupancy**: the calculator returns 2 (smem < 115 712 B: a sizing bug;
   the design's P = 132 / 66 tiles is then false).
6. **Rescale cost**: the register (dup-form) rescale cannot be expressed
   without `SHFL` / `VOTE` (i.e. the compiler does not keep m in the dup
   layout matching the accumulator columns) — then the fused chain keeps
   today's 0.24 us ballot rescale and T_chain grows by 0.15: cadence band
   0.86-0.96, no target; redesign the layout before building.
7. **Partial protocol**: the conformance matrix has no case with 1-tile and
   2-tile items at both group parities (S = 64 / 128 with `XQA_PERSISTENT_CTAS`
   1 / 3 / 5) — add them first; the protocol's correctness is exercised only
   there.
8. **Numerics**: the reviewer requires bit-identical output to the
   single-chain kernel for multi-tile items — impossible by construction
   (C18: two-partial online softmax); the fp32 reference at 3 bf16 ulps is the
   contract, as for the cross-CTA merge.
9. **Substrate**: step B is started before step A's trace has been read (the
   [15] loader merge has no standalone payoff — [15] doc 8.1 — and its
   value inside [7] is <= 0.03 us per tile; it is built only if step A's IO
   warps show as >= 5 % of issue on the critical path in the ncu per-warp
   breakdown).
10. **Break-even check on the day**: re-read the P = 132 control (`XQA_PERSISTENT
    _CTAS=132` on the production kernel, [15] 1.1: fp8 72.7) — if a merged
    change has moved it below 66, the lone chain has changed and 6.1 must be
    re-derived before the build.

## 9. Go / no-go

**Go, staged, with the pre-run gates of section 7 items 1-2 and section 8
items 1-4 as hard stops.**  Reasons:

- It is the only remaining structural lever on the SM's per-tile work: the
  round-3 record shows the pair within 10 % of the issue floor of today's
  instruction stream (1.3), the pool and the fill cut rejected, and [15]
  standalone null.  [7] removes the X ring protocol and the ballot rescale
  from every tile (-8..-12 % of the SM's instructions), the slot asymmetry
  (-2..-2.6 us) and half the start burst (-1.5 us), and its central
  prediction is fp8 -6, fp4 -2.5, mixed -2 (6.3).
- It is boring in the sense the method requires: the consumer body is one
  parameterised loop (16 HGMMA, 2 bar.sync, 2 stage waits, 2 releases per
  tile), the two groups meet only at item ends through one mbarrier pair and
  4 KB of smem, the converters and loaders are textually unchanged except
  three constants and one record bit, and every invariant in section 4 is an
  ownership or program-order argument with no per-launch global protocol.
- Its weak point is stated, not hidden: **the central case misses fp8 <= 58
  and mixed <= 62** (62.0 / 62.8); the targets are inside the band only at
  the optimistic end (IPC 2.5 with 16-20 warps, or the fill item on top).
  If the confirmation lands at the central value, the next lever is the
  instruction count itself (converter expansion 1 300 of ~3 500 per tile-SM,
  the ~400 cycles of wgmma descriptor arithmetic per GEMM), and this build's
  ncu `inst_executed` and cadence pin the IPC that every later prediction
  needs.
- Step A first: the fused consumer on the existing five-group layout at
  `__launch_bounds__(640, 1)` (consumer branches :1451-2123, SharedMem
  :233-470, barrier init :1342-1398, `finalizedItems` -> `partialDone` in the
  merge warp :2299-2447, one fill bit :3310-3410, host asserts) — roughly a
  third of the [15] + [7] diff, no loader rewrite, no Q TMA, no cursor in
  smem; its trace decides whether step B is worth its own risk list.

Predicted (step A, central): **fp8 62.0, fp4 58.1, mixed 62.8, a16 ~80 us**;
cadence 0.79 us per tile-SM (band 0.70-0.87).
