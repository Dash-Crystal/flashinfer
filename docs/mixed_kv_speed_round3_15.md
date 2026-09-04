# Round 3, lever [15]/[34] — four warp groups, one CTA per SM, loader merged into the converters (design + control measurement)

Kernel: `csrc/xqa/mha_sm90.cu` at `e113026a` (main `claude/mixed-kv-sm90-tma`,
kernel state = [8] merged at `039ba5c7`), the `MIXED_KV_PERSISTENT` q=1 build
(a16 / fp8 / fp4 / mixed).  Bench shape B=17, S=4096, 8 KV heads, GQA 4, D=128.
Line references are into `mha_sm90.cu` at that commit.  No production kernel
edit in this phase; the control measurement (section 1) uses the existing
`XQA_PERSISTENT_CTAS` override on the [8] kernel and the round-3 pair track's
trace build (`/tmp/r3pair_trace132.log`, `/tmp/r3pair_trace.log`,
`/tmp/r3pair_ncu_fp8_P{264,132}.log` on nkcut2; copies and the analysis script
in `/tmp/r3p15/` locally, script committed as
`benchmarks/parse_xqa_ctarec_roles.py`).

State after [8] (nkcut2 H200, locked 5x5, medians): transport_a16 78.8, fp8
67.8, fp4 60.5, mixed 64.4 us.  Targets fp8 <= 58, fp4 <= 36, mixed <= 62.

## 0. Verdict in one paragraph

The lever's premise — "one CTA per SM removes the co-resident pair asymmetry,
so the SM runs at the fast member's rate (-10.6 us fp8 / -9 us fp4)" — is
falsified by the control measurement: the [8] kernel launched with
`XQA_PERSISTENT_CTAS=132` (one 66-tile CTA per SM, residency probe = 1) is
**slower** in production: fp8 67.7 -> 72.7, fp4 60.4 -> 68.0, mixed 64.2 -> 73.4,
a16 79.0 -> 79.5 (interleaved A/B under the lock, section 1.1).  The lone CTA
runs a tile in 0.95 us (fp8) — 1.5x faster than a co-resident fast member
(1.41) — but the pair delivers a tile per SM every 0.865 us because the two
CTAs' dependent consumer chains overlap.  The per-CTA histogram at P = 132 is
unimodal (body spread 5 us, not 14) and every role's per-tile total equals the
cadence with the converters **idle 66 % of the time** and gemm0/gemm1 never
waiting on data (K expansion completes 1800-3000 cycles before gemm0's wait):
the lone CTA is bound by the two GEMM groups' own dependent chains
(kwait -> HGMMA -> colMax -> softmax -> X store -> fence -> arrive; xwait ->
rescale -> HGMMA), 1.17-1.20 us each in the trace build, 0.95 us in production.
Nothing in [15]'s layout — 64/128 registers, deeper rings, merged loader,
per-warp TMA — shortens that chain (P0.5 already classified it latency-bound);
the design below therefore predicts fp8 ~66, fp4 ~62, mixed ~66, a16 ~79 for
[15] as specified, i.e. **do not build it as a standalone lever** (section 8,
condition 1 is met today).  The layout is still the right substrate for the
consumer-side lever [7] (two consumer groups on alternate whole tiles at one
CTA per SM), and sections 3-5 are written so that [7] can reuse them; section
6.3 gives the combined prediction (fp8 ~61, ~57 with the fill item).

## 1. Attribution evidence

### 1.1 Production control: P = 264 vs P = 132 on the [8] kernel (nkcut2 H200, 2026-09-04 09:50 UTC)

`benchmarks/bench_xqa_mixed_page_transport.py --modes transport_a16 fp8 fp4
mixed --q-lens 1 --repeats 5 --trials 5`, checkout
`/home/bigboi/dash-flashinfer-claude-r2p8`, cached workspace `/tmp/mixedkv-r2p8`
(the production build of the merged [8] kernel), under
`flock /tmp/mixedkv-gpu0.lock`, interleaved P = 264 / 132 / 264 / 132 in one
session (repeats x kernel < 0.4 ms).  Median (min / max) us:

| mode | P = 264, run 1 | P = 132, run 1 | P = 264, run 2 | P = 132, run 2 | 132 vs 264 |
|---|---|---|---|---|---|
| transport_a16 | 79.01 (78.73 / 79.28) | 79.47 (78.82 / 79.74) | 79.05 (78.68 / 79.50) | 79.54 (79.16 / 80.20) | +0.5 us (+0.6 %) |
| fp8 | 67.64 (67.58 / 67.94) | 72.62 (72.44 / 73.22) | 67.67 (67.58 / 68.13) | 72.74 (72.24 / 73.03) | **+5.0 us (+7.4 %)** |
| fp4 | 60.45 (60.33 / 60.76) | 68.11 (67.95 / 68.13) | 60.42 (60.28 / 60.76) | 67.96 (67.75 / 68.20) | **+7.6 us (+12.5 %)** |
| mixed | 64.19 (64.10 / 64.76) | 73.36 (72.92 / 74.01) | 64.15 (63.71 / 64.84) | 73.43 (72.95 / 73.69) | **+9.2 us (+14.4 %)** |

Run-to-run spread 0.05-0.15 us; the differences are 30-100x the spread.
Effective KV bandwidth fp8 2240 -> 2085 GB/s, fp4 1327 -> 1179, a16 3610 -> 3588.

Derived lone-CTA tile time (production), using the trace build's fill / tail
for P = 132 (7.2 / 2.8 us fp8, 6.8 / 2.6 fp4, section 1.2):

    T_lone(fp8)   = (72.7 - 7.2 - 2.8) / 66 = 0.95 us      pair SM throughput: 57.1 / 66 = 0.865 us per tile-SM
    T_lone(fp4)   = (68.0 - 6.8 - 2.6) / 66 = 0.89 us      pair: 53.8 / 66 = 0.815
    T_lone(mixed) = (73.4 - 7.2 - 2.8) / 66 = 0.96 us      pair: ~(64.2 - 8.5 - 2.8) / 33 / 2 = 0.80

Break-even for any one-CTA-per-SM design: `T_lone <= 0.865` (fp8) / `0.815`
(fp4); the task's predicted 52-55 us needs `T_lone <= (55 - 7 - 3) / 66 = 0.68`.
The lever's own changes cannot reach either (section 6.1).

### 1.2 Trace build, per-CTA and per-role: where the lone CTA's time goes

Pair-track trace build (r3pair-trace: `MIXED_KV_TRACE 1` plus per-CTA per-role
segment accumulators in the `ctarec` record: `acc[slot]` = sum over all tiles
of `stamp(slot) - previous stamp of the same role`, so `acc / tiles` is the
role's per-tile time in that segment and a role's four segments sum to the
cadence).  The trace build's own wall is 1.15-1.2x production (device stamps
and the 264-record printf); ratios and idle shares are what the table is for.
Analysis: `python3 benchmarks/parse_xqa_ctarec_roles.py <log>` (medians over
the 132 / 264 records, co-tenant outliers = tail > 15 us excluded; none in
these launches).

| build / P | mode | fill med | body med (min / max), per tile | body deciles 0/10/50/90/100 | tail | end med / max |
|---|---|---|---|---|---|---|
| trace P=132 (66 tiles) | fp8 L0 | 7.2 | 87.9 (84.9 / 89.8), **1.333** | 84.9 / 86.4 / 87.9 / 89.0 / 89.8 | 2.7 | 97.7 / 99.9 |
| | fp8 L1 | 7.5 | 88.0 (84.8 / 89.6), 1.333 | 84.8 / 86.7 / 88.0 / 88.5 / 89.6 | 2.8 | 98.3 / 99.4 |
| | fp4 L0 | 6.7 | 84.6 (82.7 / 85.7), **1.282** | 82.7 / 84.0 / 84.6 / 85.2 / 85.7 | 2.6 | 93.8 / 95.7 |
| | fp4 L1 | 6.9 | 85.0 (82.1 / 86.1), 1.289 | 82.1 / 84.0 / 85.0 / 85.7 / 86.1 | 2.6 | 94.8 / 96.1 |
| trace P=264 (33 tiles) | fp8 L0 | 8.3 | 61.0 (53.9 / 67.7), 1.860 | 53.9 / 55.9 / 58.8 / 65.7 / 67.7 (bimodal 56 / 65) | 2.6 | 72.6 / 80.1 |
| | fp4 L0 | 7.6 | 63.5 (54.2 / 68.8), 1.925 | 54.2 / 56.2 / 63.5 / 66.9 / 68.8 | 2.5 | 73.0 / 79.4 |
| | a16 L0 | 10.8 | 67.8 (54.1 / 74.1), 2.054 | 54.1 / 57.7 / 67.8 / 71.6 / 74.1 | 3.1 | 80.2 / 84.9 |

`TRACE cta0 ... residentCtasAtTile4 1 nbIters 66 grid 132 x 1 x 1` on every
P = 132 launch (residency probe).  At P = 132 the body is unimodal: the
dispatch-slot asymmetry is gone by construction, as the lever claimed — and the
wall is 25 % longer.

Per-tile role segments at P = 132 (median over the 132 CTAs of `acc / tiles`,
us at 1.98 GHz; fp8 L0, fp4 L0 in parentheses):

| role | segment 1 | segment 2 | segment 3 | segment 4 | total |
|---|---|---|---|---|---|
| gemm0 | kwait (xarr(t-1) -> K wait passed) 0.212 (0.201) | mma (-> wait_group + consumed.arrive) 0.455 (0.449) | smax (colMax exchange + softmax) 0.233 (0.230) | xarr (X store + xBar.consumed wait + fence + arrive) 0.274 (0.261) | **1.174 (1.141)** |
| gemm1 | vwait 0.234 (0.226) | xwait 0.241 (0.222) | rs (rescale) 0.238 (0.224) | mma (PV HGMMA + wait) 0.484 (0.490) | **1.197 (1.162)** |
| K loader | start (waiting for the stage release) 0.894 (0.881) | iss 0.259 (0.239) | | | 1.153 (1.120) |
| V loader | 0.953 (0.929) | 0.230 (0.218) | | | 1.183 (1.147) |
| K converter | ready (wait_group + stage-release wait, **idle**) 0.765 (0.714) | done (expansion) 0.383 (0.402) | | | 1.149 (1.116) |
| V converter | 0.763 (0.726) | 0.409 (0.415) | | | 1.172 (1.141) |

Same table at P = 264 (fp8 L0): gemm0 1.607 (0.262 / 0.613 / 0.357 / 0.375),
gemm1 1.663 (0.318 / 0.396 / 0.305 / 0.644), K converter 1.552 (0.944 idle /
0.608 expansion), V converter 1.621 (0.956 / 0.664).  Every segment of the GEMM
chains is 1.3-1.4x longer with two CTAs per SM (issue arbitration between the
pair), and the converters' expansion is 1.6x longer — but the SM finishes two
tiles per 1.86 us instead of one per 1.33.

Data readiness at P = 132 (CTA 0, fp8 launch 1, tiles 2-7): K converter
`done(t)` precedes gemm0's `kwait(t)` by 1791 / 2432 / 2458 / 3047 / 2253 / 2339
cycles; V converter `done(t)` precedes gemm1's `vwait(t)` by 1053 / 1181 / 1001 /
1455 / 1223 / 1141; gemm0 `xarr(t)` -> gemm1 `xwait(t)` 83-233 cycles (barrier
round trip).  fp4 identical in shape (K lead 1131-2863, V 1051-1529).  **No
consumer wait is on data**; the 0.21 us "kwait" segment is gemm0's loop tail
(record LDS, fence, arrive) plus an already-complete `arrive_and_wait` (~190
cycles, P0.5).  The cadence is the GEMM groups' dependent chain — the same
conclusion as P0.5 (converters skipped, 1 CTA/SM: 0.84 us), now with the
converters active: 0.95 us production, of which <= 0.1 us is issue contention
with the 8 converter warps.

### 1.3 ncu, production build, fp8 (pair track, `--launch-skip 4`, clock-controlled)

| metric | P = 264 | P = 132 |
|---|---|---|
| gpu__time_duration | 87.7 us | 93.3 us (+6.4 % at base clock; +7.4 % at boost in 1.1) |
| smsp__inst_executed.sum | 36.70 M | 36.16 M (same work) |
| sm__inst_executed per active cycle (SM IPC) | 2.51 | 2.24 |
| issue active (% of peak) | 62.9 | 56.3 |
| warps active / eligible per scheduler | 9.41 / 1.58 | 4.99 / 0.97 |
| warp cycles per issued instruction | 14.97 | 8.86 |
| DRAM throughput % | 43.5 | 40.3 |
| smem wavefronts / bank conflicts | 9.58 M / 0.89 M | 9.66 M / 1.04 M |

Halving the warps per scheduler makes each warp issue 1.7x more often
(14.97 -> 8.86 cycles per instruction) but leaves the scheduler with 0.97
eligible warps on average: the SM is latency-starved, not issue-saturated, at
one CTA per SM.  This is the P0.5 "latency-bound" reading measured on the
production kernel.

### 1.4 What the a16 numbers say about the "A16 lever"

a16 at P = 132 is 79.5 vs 79.0 at P = 264; its P = 264 trace body is 67.8 us for
33 tiles x 32 KB x 264 CTAs = 279 MB, i.e. **4.1 TB/s during the body** — the
probe's achievable range (3.4-4.2, P0.1) — and the loader's TMA issue segment
is 0.65 us of the 2.05 us tile with the loader idle 1.23 us waiting for stage
releases (a16 row of the P = 264 role table: `kl start 1.231 / iss 0.646`,
converters idle 1.72 of 1.85).  The 78.8 us wall is fill 10-11 us (start burst
of 264 x 3 x 32 KB = 25 MB plus the 4-5 us prologue) + body 68 + tail 3.  The
per-warp TMA issue of [15] shortens a segment that is not on the critical
path; the a16 lever is the fill, not the issue.  Prediction for a16 under
[15]: unchanged within 1 us (section 6).

## 2. Current data flow and control flow of the touched roles (as written at `e113026a`)

Five warp groups of 128 threads (`ctaWarpGroups = 5` :186,
`__launch_bounds__(640, 2)` :1151), 2 CTAs/SM by registers (640 x 48 = 30720 of
65536) and shared memory (`sizeof(SharedMem)` = 113 664 B,
`static_assert(smemSize + 1024 <= 233472 / 2)` :465).  Register split
`setmaxnreg.dec 40` (z <= 2) / `.inc 56` (z >= 3) :1445-1449.

- **Barrier init** :1350-1398.  `kBar[s]` produced = 128 gemm0 + 128 K
  converters (+ 32 loader when `mixedLoaderTma`), consumed = 128 + 32 loader
  :1359-1364; `vBar[s]` mirror :1368-1373; `xBar[x]` 256 / 256 :1380; `qBar[b]`
  160 / 160 :1386; `kMetaReady[c]`, `vMetaReady[c]` count 32 :1393-1394.
  Prologue scan by IO warp 3 (`persistentScanWid = 11`) :1401-1405, then
  `__syncthreads` :1414; `nbCtaTiles = sched.x1 - sched.x0` :1419.
- **gemm0** (z = 0) :1451-1703: pre-arrives `qBar[*].consumed`, `kBar[*].consumed`
  :1461-1466; register `runningColMax` :1475; per tile g:
  `kBar[g%3].produced.arrive_and_wait` :1515 -> record word LDS :1523 -> on
  `first`: reset colMax, `qBar[j&1].produced.arrive_and_wait` :1526-1529 -> 2
  parts x 4 HGMMA (Q from `smem.q[idxQBuf]`) :1541-1590 -> `wait_group<0>`,
  `kBar.consumed.arrive` :1594-1595 -> scale, mask from the record :1601-1615 ->
  `computeWarpGrpColMax_sync` (named barrier), softmax, colSum :1620-1630 ->
  `storeGemm0AccToShm` (inside: `xBar[g%2].consumed.arrive_and_wait`) :1650 ->
  xColMax / xColSum :1652-1664 -> `fence.proxy.async`, `xBar.produced.arrive`
  :1687-1689 -> on `last`: `qBar[j&1].consumed.arrive`, j++ :1695-1698.
- **gemm1** (z = 1) :1704-2123: pre-arrives `vBar[*].consumed`, `xBar[*].consumed`
  :1706-1714; register `accColMax / accColSum` :1723-1724, `Gemm1Acc acc{}`
  :1744; per tile: `vBar[g%3].produced.arrive_and_wait` :1761 -> record :1772 ->
  on `first`: acc = 0, running max / sum reset :1777-1782 ->
  `xBar[g%2].produced.arrive_and_wait` :1793 -> rescale -> 8 PV HGMMA, commit,
  wait -> on `last`: publish colMax / colSum, finalize to scratch chunk
  `2c + isCtaLast` or to `output` :2000-2047, thread 0 `st.release.cta
  finalizedItems = j + 1` :2050-2055 -> `xBar.consumed.arrive`,
  `vBar.consumed.arrive` :2103-2107.
- **IO group** (z = 2) :2124-2632.  Warps 0 / 1 = K / V loader :2146-2277:
  `fillTileMeta(chunk 0)`, `metaReady[0].arrive` :2155-2156; per tile (a16 /
  mixed modules :2166-2233): record LDS (pages, formats, head) :2170-2173 ->
  `stageBar[g%3].consumed.arrive_and_wait` :2177 -> elected lane
  `arrive_tx(produced, nbA16 x 2 x 2 KB, 32)` + one `tma::loadAsync` per (part,
  A16 page) :2188-2223 -> chunk refill at `g + MIXED_KV_META_LEAD` (lead 4)
  :2227-2233; static fp8 / fp4 modules :2245-2277 do only the release arrive and
  the refills.  Warp 2 = Q warp :2278-2298 (`QCvt::load` into 16 registers,
  `qBar[j&1].consumed.arrive_and_wait`, `QCvt::store`, fence, `produced.arrive`).
  Warp 3 = merge warp :2299-2447 (poll `finalizedItems`, `atom.acq_rel.gpu.inc`,
  last arriver combines `<= nbPartials` chunks in registers, 8 lanes per head).
- **Converters** (z = 3 K :2633-2712, z = 4 V :2713-2760): warp w owns page w of
  every tile (`static_assert(nbPagesPerTile == convertWarpsPerOperand)` :3141;
  lane cut lanes 0-15 / 16-31 = the page's 16 tokens at head part 0 / 1,
  `makeExpandLane` :583-608).  `issueKCopies(t)` :2660-2676: at `t % 16 == 0`
  `kMetaReady[(t/16)%2].wait_parity(toParity<1>(t/32))`; `kBar[t%3].consumed
  .wait_parity(toParity<3>(t))` (wait, no arrive); `issueCompressedPageCopies`
  (own page, own tag, head from the record) :3137-3195.  Prologue issues tiles
  0..kAhead-1 (`kAhead = nbKBuf - 1 = 2`) :2677-2681; per tile
  `waitGroup<kAhead-1>`, `__syncwarp`, `expandPackedStage` :2690, fence,
  `kBar.produced.arrive` :2695, rotate tags, issue tile `g + kAhead`,
  `commitGroup` :2699-2711.
- **Epilogue** :2762-2810: trace printf, `__syncthreads`, barrier destruction.
- **Host** :4868-4876 `choosePersistentGridSize` (`XQA_PERSISTENT_CTAS`
  override), :4967-4975 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` -> P =
  ctasPerSm x SMs, `ScratchMem{scratch, 2P, 1}`.

## 3. New data flow and control flow (the [15] layout)

### 3.1 Groups and roles

    z = 0  gemm0        128 thr   QK^T HGMMA, mask, colMax, softmax, X store; issues its own Q TMA (elected lane)
    z = 1  gemm1        128 thr   PV HGMMA, rescale, finalize; semaphore + merge on the last tile of a partial item;
                                  warp 3 runs the prologue scan before the __syncthreads
    z = 2  K side       128 thr   warp w owns page w of every K tile: metadata chunk fill (warp 0, pipelined),
                                  cp.async copies + in-place expansion of a compressed page, or expect_tx + 2 TMA
                                  boxes of an A16 page (elected lane), kBar.produced arrive
    z = 3  V side       128 thr   mirror on V

`ctaWarpGroups = 4`, `__launch_bounds__(512, 1)`, no `setmaxnreg` (section 5.1),
`P = multiProcessorCount` (occupancy calculator returns 1 by shared memory,
section 5.2), 66 tiles per CTA for the bench.  The IO group, the Q warp, the
merge warp, `nbIOWarps`, `mixedLoadWarpsPerOperand`, `finalizedItems` and
`qBar[].consumed` disappear.

### 3.2 Data flow

    seqLenList[B] --scan (gemm1 warp 3)--> smem.sched {x0, x1, T, req0, head0, tile0, Lseq0, seqLen0, seqLen1}
    sched --> ItemCursor in smem.cursor[2] (K side warp 0 / V side warp 0; loaded into registers only inside the fill)
    K side warp 0: cursor + page table + page_format --fillTileMeta (3-step pipelined)--> smem.meta[K][chunk][g%16]
                   (TileRecord: pages[4], formats, tile word, {req:16 | head:16}, {nextReq:16 | nextHead:16})
                   --kMetaReady[chunk].arrive (count 32)-->
    K side warp w: meta[K][g] {page w, tag w, head}:
                   compressed  --cp.async--> packed rows / scales --expand--> k stage rows of page w --fence--> produced.arrive
                   A16         --expect_tx(2 x 2 KB) + 2 TMA boxes (parts 0, 1)--> k stage rows of page w ; produced.arrive after
                   past-end    --zero fill--> produced.arrive
    gemm0: kBar[g%d].produced.wait (128 + 128 arrivals + tx bytes) -> record word -> Q(j) from q[j&1] -> X(g%x)
           at first tile of item j-1 (and in the prologue for j = 0): elected lane expect_tx(qBar[j&1].produced, 2 KB)
           + 2 TMA boxes of Q(nextReq, nextHead) into q[j&1]
    gemm1: vBar[g%d].produced.wait -> record -> on last: finalize -> output or scratch[2c + isCtaLast]
           -> if partial: thread 0 atom.acq_rel.gpu.inc semaphores[H*req + head]; bar.sync; if last arriver:
              128 threads __ldcg the partial chunks -> online combine (+ sinks) -> output

### 3.3 Control flow per role (g = CTA-local tile counter, G = x1 - x0, d = K/V ring depth, x = X ring depth)

    gemm0   prologue: elected lane: Q(0) TMA into q[0] (req0, head0 from sched; expect_tx qBar[0].produced 2 KB)
            pre-arrive kBar[*].consumed
            for g: kBar[g%d].produced.arrive_and_wait ; word = LDS meta[K][g].tile
                   if first: runningColMax = -inf ; qBar[j&1].produced.arrive_and_wait ;
                             if not the CTA's last item: elected lane expect_tx(qBar[(j+1)&1].produced) + Q(j+1) TMA
                             (nextReq / nextHead from the record; the buffer was last read by item j-1's wgmma,
                              completed by this group's wait_group<0>, program-ordered before this point)
                   QK HGMMA (q[j&1]) ; wait_group<0> ; kBar[g%d].consumed.arrive
                   mask ; colMax ; softmax ; X(g%x) store (xBar[g%x].consumed.arrive_and_wait inside) ;
                   fence.proxy.async ; xBar[g%x].produced.arrive
                   if last: j++
    gemm1   warp 3: persistentPrologueScan before the __syncthreads (was IO warp 3)
            pre-arrive vBar[*].consumed, xBar[*].consumed
            for g: vBar[g%d].produced.arrive_and_wait ; word = LDS meta[V][g].tile
                   if first: acc = 0, accColMax = -inf, accColSum = 0
                   xBar[g%x].produced.arrive_and_wait ; rescale ; PV HGMMA ; commit ; wait
                   if last: publish colMax / colSum ; finalize -> output or scratch chunk 2c + isCtaLast
                            if partial (record bit): thread 0: old = atom.acq_rel.gpu.inc(sem, nbPartials - 1) ;
                                     smem.mergeOld = old ; gemm1WarpGrpSync ;
                                     if old == nbPartials - 1: all 128 threads merge (4 elements each, section 3.5)
                            j++
                   xBar[g%x].consumed.arrive ; vBar[g%d].consumed.arrive
    K side  warp 0 of the group: fillTileMeta(chunk 0) synchronously in the prologue ; kMetaReady[0].arrive
            every warp w: prologue issue(t) for t < kAhead = d - 1 (as :2677-2681, with the A16 branch below)
            for g: waitGroup<kAhead-1> ; __syncwarp
                   if tag(g) is A16: nothing to expand (the TMA bytes are tracked by produced's tx count)
                   else expandPackedStage(stage g%d, page w) ; fence.proxy.async
                   kBar[g%d].produced.arrive
                   warp 0: fill step for chunk k if g in {16k - L, 16k - L + 1, 16k - L + 2}  (section 3.4)
                   issue(g + kAhead):  if (g+kAhead) % 16 == 0: kMetaReady[..].wait_parity(toParity<1>((g+kAhead)/32))
                                       kBar[(g+kAhead)%d].consumed.wait_parity(toParity<d>(g+kAhead))
                                       tag = meta[K][g+kAhead].formats byte w
                                       A16:        elected lane: mbarrier.expect_tx(kBar[..].produced, 2 x 2 KB) ;
                                                   2 x tma::loadAsync(part p rows of page w, head from the record)
                                       compressed: issueCompressedPageCopies (own page, unchanged :3137-3195)
                                       past-end:   nothing (expansion zero-fills)
                   commitGroup (empty groups keep the wait_group accounting uniform, as today)
    V side  mirror with vBar / meta[V] / vMetaReady / smem.vBufs
    end     __syncthreads ; destroy barriers ; return

Ownership is the property that makes the loader merge free of any new
rendezvous: a page is either A16 or compressed, and warp w owns page w for
both transports, so per tile a warp does **either** the two TMA boxes **or**
the copies + expansion, never both; no lane waits on another warp for a
page's bytes (the TMA bytes are counted by the stage barrier's tx count, the
cp.async bytes by the owner's `wait_group`).

### 3.4 Metadata chunk fill on a converter warp, pipelined

`fillTileMeta` today (:3310-3410) is one warp, 2 entries per lane, three
phases: A (cursor walk, ALU), B (two dependent page-table loads per entry),
C (shfl gather + record stores).  Run synchronously on a converter warp it
would hold that warp for one dependent round-trip pair (0.75-1.25 us under
load) once per 16 tiles.  The lone-CTA converters are idle 66 % of the time
(section 1.2) and the ring gives `d - 2` tiles of slack, so even the
synchronous form is absorbed; the design nevertheless pipelines it so that no
converter warp ever blocks on a global load:

    step 1 (g = 16k - L):     load cursor from smem.cursor[op] ; phase A ; issue the 2 page-index LDGs per lane
                              (kept in 2 registers) ; store cursor back
    step 2 (g = 16k - L + 1): issue the 2 page_format LDGs (2 registers)
    step 3 (g = 16k - L + 2): phase C: gather formats, STS records ; kMetaReady[k%2].arrive

with `L = kAhead + 3 = d + 2` so that step 3 precedes `issue(16k)` in warp 0's
program order (issue(16k) runs at iteration `16k - kAhead`).  Live state
across two expansions: 2 x (page index, tile word, req/head pair) ~ 6
registers in step 1-2, 4 in step 2-3 (section 5.1).  The cursor itself (9
registers) lives in `smem.cursor[op]` (40 B) between fills, so its live range
never overlaps an expansion.  WAR bound for the record slots (C10 restated,
section 4): `L <= 15 - x`; with x = 3 that is `L <= 12`, so d <= 10 — every
depth of section 5.2 qualifies.

### 3.5 Merge on gemm1

The IO merge warp's protocol (:2299-2447) moves into gemm1's `last` branch,
after `finalizeAndWriteOut_sync` (whose internal `warpGrpBar` sync orders all
128 threads' scratch stores before thread 0's atomic; this is the same
`bar.sync -> atom.acq_rel.gpu` cumulativity chain the pre-[8] epilogue used at
:2229-2250 of `fe2e9a33` and the merge warp used through `st.release.cta` /
`ld.acquire.cta`).  `old` is broadcast through one smem word and a
`gemm1WarpGrpSync` (bar.sync orders shared memory).  The combine uses 128
threads x 4 elements (thread `t`: head `t / 32`, elements `4 (t % 32) ..`) with
the same online formula and the same chunk enumeration (`c0`, `c1`, `qP / rP`
recurrence, `lastChunk`) as :2333-2345 / :2385-2432; `__ldcg` loads after the
acquire.  Cost on gemm1: one atomic round trip (~0.5-1 us) per partial item
and, for the last arriver, one load round trip + 4 element-combines; on an
interior item this stalls gemm1 only, and gemm0 continues for `x` tiles
(3 x 0.95 = 2.9 us of X-ring slack against <= 2 us of merge), the converters
for `d - 1`.  On the CTA's last item it is the tail, as today (2.6-3.0 us).
`finalizedItems` and the merge warp's poll go away.

### 3.6 Q by TMA from gemm0

`smem.q[2]` becomes `alignas(1024)` (the 128 B swizzle atom is 8 rows = 1 KB,
`QBuffer::Elem` is `Array2D<LdGrain, 8, 8>` = 1 KB per part, the K part
buffers are already `alignas(1024)` :238).  One more `__grid_constant__`
`CUtensorMap tensorMapQ` over `q` viewed as `[nbReq * nbQHeads][headElems]`
bf16 with `SWIZZLE_128B`, box `{64 elems, ctaNbQHeads = 8 rows}`; the box for
item (req, head) starts at row `headGrpSize (nbKHeads req + head)`, exactly
the address `TinyPtr<IOHead const>{q, headGrpSize * (nbKHeads * req + head)}`
the Q warp uses today :2288.  Rows 4..7 of the box are the next group's heads
(finite values; today those rows of `smem.q` are **never written** —
`QCvt::load` breaks at `idxGrain >= totalGrains = 64` :3013-3017 — so the
padded columns already hold arbitrary data, and columns are independent in
the SWAP_AB softmax; at the tensor's end TMA zero-fills).  Two boxes per
item (parts 0 / 1), 2 KB, `expect_tx` on `qBar[b].produced` whose count
becomes 128 + 1 (the elected lane's `arrive.expect_tx` + gemm0's 128
`arrive_and_wait`).  `needInputCvt == false` is already a `static_assert`
(:2281); the fp8-cache variant would keep the register path and is out of
scope as in [8] 7.8.

## 4. Barrier, parity and ownership invariants

D1-D6, C1-C7 (`mixed_kv_page_transport_dataflow.md`, A1-A7) and C8-C13
([8] design section 3, 8.1-8.11) remain the reference.  Changes and additions:

- **C4 (barrier accounting), restated.**

  | barrier | produced count | tx bytes | consumed count | waiters |
  |---|---|---|---|---|
  | `kBar[s]`, `vBar[s]` (s < d) | 128 GEMM + 128 converters = 256 (was 256 / 288) | `expect_tx` (no arrive) by the elected lane of each converter warp whose page is A16: 2 parts x 16 rows x 128 B = 4 KB per A16 page | 128 (GEMM group only; was 160) | GEMM `arrive_and_wait`; converters `wait_parity` |
  | `xBar[x]` | 256 | - | 256 | unchanged |
  | `qBar[b]` | 129 (gemm0 128 + its elected lane's `arrive.expect_tx`) | 2 KB | **deleted** | gemm0 `arrive_and_wait` |
  | `kMetaReady[c]`, `vMetaReady[c]` | 32 (side warp 0) | - | - | every warp of the side, `wait_parity` at `t % 16 == 0` (warp 0 arrives before its own wait: phase completes on its arrive) |
  | `gemm0WarpGrpBar`, `gemm1WarpGrpBar`, named barriers 3 / 4 | unchanged | | | |

  `mbarrier.expect_tx` without arrive is `mbarrier.expect_tx.relaxed.cta.shared::cta.b64` (sm_90); the phase cannot complete before the 256 arrivals regardless of the order of `expect_tx` and the TMA's `complete_tx`, because the owner warp's 32 arrivals come after its `expect_tx` in program order.
- **C8 (consumer-gated parity) with consumed count 128.**  Phase k+1 of
  `consumed[s]` completes only with the GEMM group's arrive for tile `kd + s`,
  which follows `produced(kd + s)`, which needs every converter warp's arrive
  for that tile, which follows each warp's `issue(kd + s)` wait on phase k.
  The waiter is never two phases behind; the loader's arrive was never part
  of the argument.  `kMetaReady` chain: fill f+1 of chunk c is written at
  iteration `32(f+1) + 16c - L + 2` after warp 0's `issue(32(f+1) + 16c - L +
  2 + kAhead - 1)` wait, i.e. after gemm0 consumed tile `32(f+1) + 16c - L +
  kAhead - d + 1 = 32(f+1) + 16c - L` (kAhead = d - 1), which implies every
  converter warp passed `issue(32f + 16c)` = its wait on phase f.  Holds for
  every `L >= 1`.
- **C10 (record visibility and reuse), restated for converter-side fills.**
  Writer = side warp 0 at step 3 (iteration `g_w = 16k - L + 2`), release =
  `kMetaReady.arrive`; readers: the side's warps at `issue` (acquire through
  their `wait_parity`), gemm0 / gemm1 after the stage `produced` wait (two
  synchronizes-with edges as today).  WAR: the slot rewritten at `g_w` for
  tiles `[16k, 16k + 16)` held `[16k - 32, 16k - 16)`; newest old tile
  `16k - 17 = g_w + L - 19`.  Warp 0's most recent wait at `g_w` is
  `consumed` for tile `g_w + kAhead - 1`, so gemm0 has consumed (and read the
  record of) tile `g_w - 1`; gemm0's `consumed.arrive(g_w - 1)` follows its
  `xBar.consumed` wait for tile `g_w - 2`, which needs gemm1's arrive for tile
  `g_w - 2 - x`, program-ordered after gemm1's read of that record; a sibling
  converter warp has arrived `produced(g_w - 1)`, so it has issued (read) up
  to tile `g_w - 1 + kAhead - 1`.  Safe iff `g_w + L - 19 <= g_w - 2 - x`, i.e.
  **`L <= 17 - x`**; with the fill's own step-1 loads also reading the cursor
  only, the binding reader is gemm1: `L <= 15 - x` (12 at x = 3).  `L = d + 2`
  (3.4) satisfies it for `d <= 10`.
- **C11 (scratch slot rule)** unchanged: `2c + isCtaLast`, `ScratchMem{scratch, 2P, 1}`.
- **C12 (merge hand-off)** is replaced by **C14 (merge in the finalizing group).**
  gemm1's scratch stores -> `warpGrpBar` sync inside finalize -> thread 0
  `atom.acq_rel.gpu` (release cumulative over the group's stores) -> the last
  arriver's `atom` acquire -> `bar.sync` broadcast -> `__ldcg` loads.  No
  counter, no poll, no cross-CTA spin: the atomic never blocks and a CTA that
  is not the last arriver does nothing.  Liveness: gemm1's merge waits on
  nothing inside its own CTA; gemm0 blocks after `x` tiles on `xBar.consumed`,
  the converters after `d - 1` tiles on `consumed` parity, and all resume
  when gemm1 resumes.
- **C15 (Q buffer ownership by program order).**  `q[b]` is written by TMA
  issued by gemm0's elected lane at the first tile of item `j - 1` for item
  `j` (`b = j & 1`) and read by gemm0's wgmma during item `j`.  The previous
  reader of `q[b]` is item `j - 2`, whose last `wgmma.wait_group<0>`
  (warpgroup-collective) precedes the elected lane's TMA issue in program
  order; both are async-proxy operations on the same buffer ordered by
  completion.  `qBar[b].produced` phase m completes with the 2 KB landing plus
  gemm0's 128 waits for item `2m + b`; the elected lane's `arrive.expect_tx`
  for item `j + 1` (buffer `(j+1) & 1`) is issued at item `j`'s first tile,
  after gemm0's `produced` wait for item `j - 1` on that same buffer (program
  order), so the barrier's phases alternate arrive/expect -> wait strictly and
  no phase is skipped.  Items past the CTA's
  last item are not prefetched (`nextReq == 0xFFFF` in the record).
- **C16 (page ownership across transports).**  `static_assert(nbPagesPerTile ==
  convertWarpsPerOperand)` :3141 now also covers the TMA path: page `i` of
  tile `g` is written by warp `i` only (TMA boxes to rows `i * 16 .. + 16` of
  both parts, or `cp.async` + expansion into the same rows).  D3 / D6 / A7
  are unchanged; the mixed module's per-tile `arrive_tx(nbA16 x 4 KB)` by one
  lane (:2212) becomes `nbA16` separate `expect_tx(4 KB)` by `nbA16` lanes.
- **C17 (cursor in shared memory).**  `smem.cursor[op]` is read and written
  only by side warp 0 inside fill steps (LDS / STS of 40 B, once per 16
  tiles); no other warp touches it, so no barrier is involved (D3 style
  single-owner rule).
- **C9 / C13** unchanged: no ring, stage, count or trace slot depends on items;
  the partition is `|R_c| in {floor(T/P), ceil(T/P)}` with P = 132 (66 tiles).
- **C7-class** unchanged: items begin mid-pipeline with copies `kAhead` tiles
  ahead; the matrix keeps the many-items-per-CTA cases (P = 1 / 3 / 5, T < P).

## 5. Budgets

### 5.1 Registers

`__launch_bounds__(512, 1)`: 65 536 / 512 = **128 registers per thread
available**; no `setmaxnreg`, no C7507 question, no pool balance.  The
design's *need* per role (so that the layout would still fit 64 if a future
change wanted two CTAs per SM):

| role | live set | today (ptxas, 0 spill) | new | need |
|---|---|---|---|---|
| gemm0 | `Acc` = 64 x 8 fp32 / 128 thr = 4; `runningColMax` 2; descriptors, addresses, loop | 40 | + record word, elected-lane TMA coordinates (transient) | <= 48 |
| gemm1 | `Gemm1Acc` = 128 x 8 / 128 = 8; `accColMax / accColSum` 4; xvoScale, descriptors | 40 | + merge (8 fp32 acc, sum, max, 2 pointers, transient in the `last` branch; the IO merge warp fit 40) | <= 56 |
| converters | expansion (`LdGrain words[4]` 16, `out[2]` 8, scale words 4, `ExpandLane` 4, `ExpandScales` 6, addresses) | 56 (48 spilled 8-24 B before fe2e9a33) | + fill pipeline state 4-6 across two expansions (cursor in smem), + 2 TMA coordinates (elected lane, transient) | <= 64 |

The SWAP_AB `m64n8k16` chains need no extra live state: the accumulator is
the `GmmaAcc<64, 8>` (4 registers) / `GmmaAcc<128, 8>` (8), and the
register-resident colMax / colSum of [1] are `RegColWiseVec` = 2 floats each.
Any of the three rows exceeding 64 would be a code-shape regression (C2 /
C3), not a need — checked by `ptxas -v` (0 spill) and by reading
`cuobjdump -res-usage` REG; 128 is the hard cap.

### 5.2 Shared memory (one CTA per SM: 232 448 B usable; today 113 664 B)

Bytes as a function of K/V depth `d` and X depth `x` (per-operand scale ring
`nbScaleTiles = d + 1` entries of 512 B; records 2 048; Q 2 x 2 048; col
vectors 672; sched 48 + cursors 80 + mergeOld 4; barriers <= 200):

    smem(d, x) = 32 768 d + 2 048 x + 4 096 + 2 048 + 1 024 (d + 1) + ~1 000

| d | x | bytes | fits 232 448 | forces 1 CTA/SM (> 115 712) |
|---|---|---|---|---|
| 3 | 2 | 113 664 (measured today) | yes | no |
| 4 | 3 | 149 500 | yes | yes |
| **5** | **3** | **183 300** | yes | yes |
| 6 | 3 | 217 100 | yes | yes |
| 6 | 4 | 219 100 | yes | yes |
| 7 | 3 | 250 900 | **no** | - |

Specification: **d = 5, x = 3** (183 KB; +1 032 B trace build).  Why not
deeper: section 1.2 shows no consumer wait on data at d = 3 with one CTA per
SM (K lead 1 800-3 000 cycles), so depth buys nothing measurable; 5 covers
the start-burst copy latency of the fill (3-4 us = 3-4 tiles) and leaves
49 KB for [7]'s per-group P buffers.  `static_assert(smemSize <= 232448)`
replaces the `<= 233472 / 2` assert :465; the host's
`cudaFuncAttributeMaxDynamicSharedMemorySize` is already set from
`hostSmemSize`.

### 5.3 Issue budget per converter warp per tile

Cadence to fit: 0.87-0.95 us = 1 720-1 880 cycles.

| item | warp-instructions | time (one warp, measured or bounded) |
|---|---|---|
| expansion of one compressed page (fp8 / fp4 lane-tile, [16]) | 187 / 188 | 0.38-0.43 us measured at P = 132 (`kc done` / `vc done`) |
| copy issue (`issueCompressedPageCopies`, own page) | 71-90 | 0.16-0.18 us (`kl iss` at P = 132) |
| wait_group, syncwarp, fence, produced.arrive, parity wait, commit | ~50 | ~0.05 us |
| **A16 page instead**: `expect_tx` + 2 x `tma::loadAsync` (elected lane) | ~30 | 2 x 100-200 ns = 0.2-0.4 us, replaces the expansion (a page is one or the other) |
| fill step (warp 0 only, once per 16 tiles per step) | ~120 per step | no exposed latency (pipelined) |
| total, compressed page | ~330 | **~0.6 us of 0.95** |
| total, A16 page | ~120 | ~0.45 us |

Per-warp TMA issue: 2 boxes per warp per tile against 16 per lane today
(:2216-2223) — 8x less serial issue on any one lane, but see 1.4: this
segment is not on a16's critical path.

## 6. Predicted periods and wall

### 6.1 [15] as specified (one CTA per SM, GEMM chains unchanged)

Start from the measured lone-CTA production tile time (1.1): fp8 0.95, fp4
0.89, mixed 0.96.  Effects of the lever's changes on that number:

| change | mechanism | effect on T_lone |
|---|---|---|
| converters at 64-128 registers, more ILP | converters are idle 66 % (1.2); expansion 0.38 of 0.95 | 0 |
| rings d = 5 | no consumer wait on data at d = 3 (K lead 1 800-3 000 cyc) | 0 |
| 16 warps instead of 20 per SM (no IO group) | issue tax of the spinning loader / Q / merge warps; P0.5 bound <= 0.16 us at 10 warps/scheduler, here 4 -> 5 | -0.02 .. -0.08 |
| Q TMA instead of the Q-warp hand-off | +0.4 us per item on gemm0's first tile (section 10 of [8]), 2 items per CTA | -0.8 us per CTA (-0.012 per tile) |
| per-warp A16 TMA issue | a16 body is DRAM-bound at 4.1 TB/s | 0 |
| merge on gemm1 | +0.5-1 us per interior partial item on gemm1, absorbed by x = 3 | 0 |

    T_lone'  = 0.95 - 0.08 (best) = 0.87 fp8 ; 0.89 - 0.08 = 0.81 fp4 ; 0.96 - 0.08 = 0.88 mixed
    wall'    = fill 7 + 66 x T_lone' + tail 3

| mode | today | predicted [15] (best case) | target |
|---|---|---|---|
| fp8 | 67.7 | 7 + 57.4 + 3 = **67.4** (worst case, no issue-tax gain: 72.7) | <= 58 |
| fp4 | 60.4 | 7 + 53.5 + 3 = **63.5** (68.0) | <= 36 |
| mixed | 64.2 | 7 + 58.1 + 3 = **68.1** (73.4) | <= 62 |
| a16 | 79.0 | fill 10 + body 68 (DRAM 4.1 TB/s) + 3 = **~79-81** | parity |

The task's predicted 52-55 / 48 / 52 / 70 would need `T_lone = 0.65-0.7`; the
lever offers 0.87 at best.  Standalone [15] is a null-to-negative change.

### 6.2 Why the pair beats the lone CTA (the model the numbers fit)

Per SM and per tile the pipeline executes ~3 600 warp-instructions (consumer
~2 100 + converters 2 x 4 x 188 + copy issue) — identical at P = 132 and 264
(ncu 36.2 M vs 36.7 M).  The lone CTA issues them at SM IPC 2.24 with 0.97
eligible warps per scheduler; its cadence is the sum of one GEMM group's
dependent latencies (kwait 0.21 + HGMMA 0.46 + colMax/softmax 0.23 + X store /
fence / arrive 0.27 = 1.17 in the trace build; gemm1 1.20), because within a
group tile t+1's HGMMA is not issued before tile t's softmax and X store.
Two co-resident CTAs interleave two such chains on the same schedulers: each
chain stretches 1.35x (1.17 -> 1.61) but two tiles complete per stretched
period -> 0.93 us per tile-SM (trace build), 0.865 in production.  The
dispatch-slot asymmetry (46.5 vs 57.1) is the arbitration *between* the two
chains, not a defect the lone CTA escapes cheaply.

### 6.3 What the layout is for: [7] on top of [15]

With one CTA per SM the two consumer groups can process alternate whole tiles
(FA3 pattern, plan [7]): per group the chain covers QK + softmax + PV of one
tile ~ 1.17 + (0.24 + 0.48) = 1.9 trace / ~1.55 production us, and two groups
alternating give a cadence of ~0.78 us; converters at 0.6 of 0.78 per warp
(5.3) and DRAM (fp8 156 MB / 50 us = 3.1 TB/s) are below it.  Predicted
`wall = 7 + 66 x 0.78 + 3`: fp8 **~61.5**, fp4 ~60 (its chain is the same;
fp4 is consumer-bound), mixed ~62; with the fill item (-4 us, [8] section 10
item 2) fp8 ~57.5, i.e. the target line is reachable only as [15] + [7] +
fill, not as [15].  This is the plan's original order ("[15] before [7]")
with the difference that [15] has no standalone payoff and must not be gated
on one; the two are one build.

## 7. Verification artifacts, accept / reject

Build (each module a16 / fp8 / fp4 / mixed; ptxas recipe
`/tmp/main_ptx/ninja_flags.py` -> `nvcc -ptx` -> `ptxas -arch=sm_90a -v`):

1. `ptxas -v`: 0 bytes stack, 0 spill stores / loads; REG <= 128 by
   construction, **record the value**; no `USETMAXREG` in the SASS
   (`grep -c USETMAXREG` = 0; the A4 count changes from 2 to 0 for this build).
2. `cuobjdump -sass`: `LDL` = `STL` = 0; `UTMALDG` = 2 (Q) in fp8 / fp4, 2 + 4
   (Q + K / V per-warp boxes, two parts each, one code path per side) in a16
   / mixed; `LDGSTS` > 0; exactly one `ATOMG...INC`, no `ATOMS`; HGMMA 8 + 8;
   `SYNCS.ARRIVE.TRANS64.A1T0` count per role: gemm0 K-consumed + X + Q-tx
   only; gemm1 unchanged; converters produced + metaReady only; the merge
   warp's `NANOSLEEP` loop gone (NANOSLEEP count = 0 in the persistent build).
3. `cuobjdump -res-usage`: STACK 0; `sizeof(SharedMem)` 183 3xx B; occupancy
   calculator returns 1; `launch__grid_size` = 132.

Conformance: `python tests/attention/run_xqa_mixed_page_transport.py` = 60 / 60
(exit 0), including the T < P (empty CTAs), P = 1 / 3 / 5 and 1-3-tile-item
tail cases (C7 class: items begin with compressed tiles issued `kAhead` ahead;
Q(j+1) prefetch and merge-on-gemm1 both cross item boundaries).

Timing (locked, `mixedkv_remote_run.sh <checkout> r3p15 sm90 transport_a16 fp8
fp4 mixed`, 5 x 5, min / median / max; q = 4 rows unchanged):

| mode | accept only if | predicted (6.1) | reject if |
|---|---|---|---|
| fp8 | median <= 64.0 (-5 %; anything less is inside the pair track's dynamic-pull gain of -6) | 67.4 | > 67.7 |
| fp4 | median <= 57.5 | 63.5 | > 60.5 |
| mixed | median <= 61.0 | 68.1 | > 64.4 |
| a16 | median <= 80 | 79-81 | > 82 |

Trace (`MIXED_KV_TRACE 1`, `xqa_mixed_trace_once.py --modes fp8 fp4 transport_a16
--q-len 1 --launches 3`; per-CTA `ctarec` histogram with the pair track's
role accumulators, `parse_xqa_ctarec_roles.py`):

- residency probe = 1 on every CTA; 132 records per launch; body unimodal
  (max - min <= 6 us);
- gemm0 / gemm1 per-tile totals <= 1.10 us (trace build) — i.e. the chain
  moved by >= 6 % — **is the only outcome that would justify keeping [15]
  standalone**; predicted 1.15-1.17 (no change);
- converter `done` segment <= 0.40, `ready` (idle) >= 0.6 (the converters are
  not the bound, before and after); `kc_ready(16k)` at steady state (fill
  lead L = d + 2 not exposed);
- item boundary: gemm0 K-wait -> mma on the item's first tile within 10 % of
  other tiles (Q TMA prefetch removes the +0.4 us).

ncu (fp8, one launch): `launch__occupancy_limit_shared_mem` = 1,
`smsp__warps_active` ~4 per scheduler, `sm__inst_executed` within 3 % of
36.2 M; `dram__bytes_read.sum` unchanged per mode.

## 8. Do not build if

1. **(Met today.)**  The one-CTA-per-SM control on the current kernel
   (`XQA_PERSISTENT_CTAS=132`) is slower than P = 264 on every compressed
   mode (fp8 +5.0, fp4 +7.6, mixed +9.2 us, section 1.1) and its trace shows
   the GEMM groups' dependent chains at the cadence with the converters idle
   66 % and no data wait.  None of [15]'s mechanisms acts on that chain, so a
   standalone [15] cannot beat 67.7 / 60.4 / 64.2.  Build [15] only as the
   layout of [7] (section 6.3), with [7]'s design written first.
2. `ptxas -v` at `__launch_bounds__(512, 1)` spills in any role, or REG > 64
   in gemm0 / gemm1 (then [7]'s two consumer groups, which need the extra
   registers for two tiles of state, are the ones squeezed).
3. The converter warp's fill pipeline (3.4) cannot keep its 4-6 live
   registers without spilling **and** the synchronous fallback exposes
   `kc_ready(16k)` in the trace: then the fill needs its own warp again and
   the "loader merged into converters" premise fails.
4. `sizeof(SharedMem)` at d = 5, x = 3 exceeds 232 448 B (trace build
   included) — recompute d.
5. The occupancy calculator returns 2 for the new module (would mean smem <
   115 712 B, a sizing bug: P would be 264 and the design's 66 tiles false).
6. The Q tensor map cannot be built for the deployment's q layout (non-unit
   head stride, or `q_len > 1` shapes routed to this kernel): keep the
   register Q path on a converter lane group instead, at +1 us per item.
7. The pair track lands a dynamic tile-range pull or a priority fix on the
   2-CTA/SM kernel: re-measure the P = 132 control on *that* kernel before
   any one-CTA/SM work (the break-even `T_lone <= 0.865` moves with it).
8. The conformance runner's `XQA_PERSISTENT_CTAS` cases are dropped: the
   merge-on-gemm1 and Q-prefetch paths are exercised only by many-items-per-CTA
   cases.

## 9. Recommendation

- **[15] standalone: no-go** (section 0, 6.1, 8.1).  The measured facts to
  carry forward: lone-CTA tile time 0.95 / 0.89 / 0.96 us (fp8 / fp4 / mixed)
  vs pair throughput 0.865 / 0.815 / 0.80 per tile-SM; GEMM chains 1.17-1.20
  us at the cadence with converters 66 % idle; ncu 0.97 eligible warps per
  scheduler at P = 132.
- The pair asymmetry is a 2-CTA/SM arbitration effect; the cheaper levers
  against it are the pair track's (dynamic tile-range pull: -6 us fp8 / -4.5
  fp4 at unchanged pipeline; or a dispatch-order-aware split of the tile
  space between the two members of an SM pair).
- The fill (7-8.5 us, [8] section 10 item 2) is the next largest item on all
  four modes and is independent of the layout.
- The layout in sections 3-5 stands as the substrate for [7]; the invariants
  C14-C17 and the budgets are written for that build.  Predicted [15] + [7]
  + fill: fp8 ~57.5, fp4 ~56, mixed ~58; fp4 <= 36 remains analytically out of
  reach (plan gate check).
