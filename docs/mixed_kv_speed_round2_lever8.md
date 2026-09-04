# Round 2, lever [8] — persistent, balanced CTA scheduling with cross-item prefetch (design)

Kernel: `csrc/xqa/mha_sm90.cu` as of `fe2e9a33` (main `bfb0eca0`), the
`ENABLE_MIXED_KV_CACHE && !SPEC_DEC` q=1 build (a16 / fp8 / fp4 / mixed static
formats).  Bench shape B=17, S=4096, 8 KV heads, GQA 4, D=128, q=1.  Line
references are into `mha_sm90.cu` at that commit.  No kernel edits in this
phase; this document is the data-flow / control-flow description that precedes
the change (method rule: design first, read SASS/ptxas/trace second, one
confirmation run third).

Round-2 baseline (nkcut2 H200, locked, medians): transport_a16 81.8, fp8 76.9,
fp4 70.7, mixed 79.5 us.  Targets: fp8 <= 58, fp4 <= 36, mixed <= 62.

## 0. Why this lever, in numbers

The grid today is `1 x nbSubSeq x (B*H)` = 1 x 5 x 136 = 680 CTAs on 264
resident slots (2 CTAs/SM x 132 SMs, `__launch_bounds__(640,2)` :959) — 2.58
waves.  Each CTA runs 13 tiles (64 tiles / 5 sub-sequences, :1187) and pays its
own prologue + first-K latency ("fill", ~4.4 us: clk0 -> K-loader start 3864
cyc, first K ready 7828 cyc = 3.95 us on fp4; 7.75 us on a16).  Wall accounting
from the round-2 baseline (backends.md, "Round 2 baseline"):

    wall  =  3 lifetimes x (13 x T_tile + 4.4)  +  tail (~10 us)
    fp8:     3 x (13 x 1.38 + 4.4) = 67.0 + 10.0  = 77.0   (measured 76.9)
    fp4:     3 x (13 x 1.15 + 4.4) = 58.0 + 12.7  = 70.7   (measured 70.7)

i.e. 39 tile-times, three fills and a ~10 us tail for 33 tile-times of useful
work per slot.  A persistent grid of P = 264 CTAs that each take a contiguous
share of the 8704 tiles processes ceil(8704/264) = 33 tiles per CTA, pays one
fill per slot, and the item-boundary latency is prefetched away:

    wall' =  F + 33 x T_tile + finalizes + tail'

with F ~ 3.5 us (one prologue + first-K, minus [36a][36b]), finalizes <= 2 x 0.4
us, tail' ~ 3 us (last merge ~2 + drain ~1).  Section 5 gives the arithmetic
per mode; the headline is fp8 77 -> ~53, fp4 71 -> ~45, mixed 80 -> ~55 at
unchanged role periods, a16 unchanged (DRAM-bound: 285 MB at ~3.5 TB/s = 81 us).

## 1. Current data flow and control flow (per role, as written)

### 1.1 Kernel entry and prologue (all 640 threads)

- `idxReq = blockIdx.z / nbKHeads` :1011, `idxHeadGrp = blockIdx.z % nbKHeads`
  :1023, `cacheSeqLen = getCacheSeqLen(cacheList, idxReq)` :1025 — one
  dependent global load by **every** thread before anything else.
- `nbTiles = divUp(cacheSeqLen, 64)` :1072, `nbTilesInUse = nbTiles -
  nbSkipLeadingTiles` :1075 (sliding window: `nbSkipLeadingTiles`,
  `tile0NbSkipTokens` :1060-1061).
- `maxNbSubSeq = gridDim.y` :1077, `idxSubSeq = blockIdx.y` :1078,
  `isMultiBlockMode`, `idxKTileInit = nbSkipLeadingTiles + idxSubSeq` :1080,
  `nbSubSeq = min(nbTilesInUse, maxNbSubSeq)` :1082; CTAs with `idxSubSeq >=
  nbSubSeq` return :1086.  Tile t of this CTA is sequence tile `idxKTileInit + t
  * nbSubSeq` (strided split, :1256).
- Warp ids :1090-1091; tensor-map prefetch :1092-1101; barrier init by warps 0..8
  :1123-1174 (kBar[3] produced = 128 gemm0 + 128 converters (+32 loader if
  `mixedLoaderTma`), consumed = 128 + 32; vBar[3] likewise for gemm1; xBar[2]
  256/256; qBar 160/160; kMetaReady[2]/vMetaReady[2] count 32); `__syncthreads`
  :1181; `nbIters = divUp(nbTiles - idxKTileInit, nbSubSeq)` :1187;
  `setmaxnreg .dec 40` (z <= 2) / `.inc 56` (z >= 3) :1206-1210.

### 1.2 gemm0 (z = 0, 128 threads) :1212-1410

Pre-arrives `qBar.consumed` :1221 and every `kBar[s].consumed` :1222-1224 (the
"stage is free" phase-0 arrivals), initializes the register-resident
`runningColMax` :1234, waits `qBar.produced` :1243 (token wait), then per tile
`idxIter` :1255: `kBar[idxIter % 3].produced.arrive_and_wait()` :1264 -> 2 parts
x 4 HGMMA :1268-1306 -> `wait_group<0>` + `kBar.consumed.arrive()` :1321-1322 ->
scale, mask (`isFirstTile`, `isLastTile`, `cacheSeqLen % 64`) :1338-1352 ->
`computeWarpGrpColMax_sync(smem.gemm0WarpColMax[idxIter % 2], runningColMax)`
:1356 -> online softmax, colSum -> `storeGemm0AccToShm(xBuf(idxIter % 2),
xBar.consumed, acc)` :1380 (inside: `xBar.consumed.arrive_and_wait()` :3340) ->
xColMax/xColSum :1385-1396 -> `fence.proxy.async`, `xBar.produced.arrive()`
:1407.  After the loop `qBar.consumed.arrive()` :1410 (never observed today:
no second Q).

### 1.3 gemm1 (z = 1) :1411-1744

Pre-arrives every `vBar[s].consumed` and `xBar[s].consumed` :1413-1422;
`accColMax/accColSum` registers :1431-1432; `Gemm1Acc acc{}` :1452.  Per tile:
`vBar.produced.arrive_and_wait()` :1462 -> `xBar.produced.arrive_and_wait()`
:1478 -> `rescaleGemm1AccForNewColMax` :1532 -> 8 PV HGMMA + one `commit/wait`
:1556-1620 -> **if last tile** :1677: publish colMax/colSum to smem :1681-1685,
then either the multi-block save (`ScratchMem{scratch, maxNbSubSeq*H*B, 1}`,
`idxChunk = maxNbSubSeq*idxSeq + idxSubSeq`, rowSumMax + partial tokens via
`finalizeAndWriteOut_sync` into `outSwizzleBuf(idxXBuf)`) :1691-1715 or the
direct output write :1717-1735 -> `xBar.consumed.arrive()` :1738,
`vBar.consumed.arrive()` :1740.

### 1.4 IO group (z = 2, 4 warps) :1745-2076

- Warp 0 is **both** the Q loader (:1756-1811: `QCvt::load` :1802,
  `qBar.consumed.arrive_and_wait` :1804, store, fence, `qBar.produced.arrive`
  :1810) **and** the K loader (`kLoadWarpBeg = 0` :1813).  This is [36b]: the
  first K work of the CTA sits behind the Q round trip.
- K loader :1822-1902: `KVTilePartLoader` holds `idxReq`, `idxHeadGrp`,
  `baseOffset = idxReq * maxNbPagesPerSeq` as consts :2510-2530.  Prologue fills
  metadata chunk 0 **and** chunk 1 :1842-1847 (two dependent-load pairs before
  the first tile; [36a]: chunk 1 is first needed at tile 14).  Per tile:
  `readTileMeta`/`publishPages` (a16 build only) :1853-1855,
  `kBar[idxKStage].consumed.arrive_and_wait()` :1860, elected-lane `arrive_tx` +
  A16 TMA boxes :1864-1885, chunk refill for `ahead = idxIter + 2` when `ahead %
  16 == 0 && ahead >= 32` :1887-1896 (`kMetaReady[chunk].arrive()`).
- V loader (warp 1) :1962-2024 mirrors K with `vBar`, `vMeta`.
- Warps 2 and 3 of the IO group are **idle** in the mixed build.

### 1.5 Converters (z = 3 K, z = 4 V) :2077-2205

`issueKCopies(t)` :2105-2120: at `t % 16 == 0` wait
`kMetaReady[(t/16) % 2].wait_parity(toParity<1>(t / 32))` :2110; wait (no
arrive) `kBar[t % 3].consumed.wait_parity(toParity<3>(t))` :2115;
`issueCompressedPageCopies(cacheList, isK, idxHeadGrp, smem, kMeta, t, ...)`
:2116 (reads `metaFormats/metaPages[op][chunk][entry][warp]` :2594-2603,
payload address uses `idxHeadGrp` :2612-2614 and scale address :2645-2648).
Prologue issues tiles 0..kAhead-1 :2121-2125 (one `commitGroup` per tile, also
for tiles past `nbIters`).  Per tile :2126-2156: `waitGroup<kAhead-1>`,
`__syncwarp`, `expandPackedStage` :2134, `fence.proxy.async`,
`kBar.produced.arrive()` :2139, rotate `kTags`, issue tile `idxIter + kAhead`
:2146-2148, `commitGroup`.  V mirrors :2158-2205.

### 1.6 Epilogue :2229-2410

`__syncthreads` :2229, barrier destruction :2230-2236, then (multi-block only)
one thread's `atom.acq_rel.gpu.inc` on `semaphores[idxSeq]` with limit
`nbSubSeq - 1` :2240-2250, and the last CTA merges through `MultiBlockSMem`
**aliased over `smemByteBuf[0]`** (tokens 4 x 8 x 256 B, rowSumMax, 4 barrier
pairs) :2251-2400.  This aliasing is why the merge cannot stay where it is once
the next item's K/V stages are live in the same shared memory.

### 1.7 Host :4475-4561

`ctasPerSm` from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` :4493-4497,
`nbSubSeqPerSeq = chooseNbSubSeq(...)` :4499 (:4243-4290: minimizes waves x
tiles-per-CTA over `n`; picks 5 for the bench shape), `dimGrid = {1,
nbSubSeqPerSeq, nbKHeads * batchSize}` :4507.  Scratch = `workspace[8 MB:]`,
semaphores = `workspace[:8 MB]` (`flashinfer/decode.py` :3675-3676), sized by
the caller (bench: 256 MB, zero-initialized).

## 2. New data flow and control flow

### 2.1 Work decomposition: static contiguous partition of the linear tile space

Linearize all work as tiles `x` in `(request, head, tile)` order:

    tiles(r)   = divUp(seqLen[r], 64) - nbSkipLeadingTiles(r)        (tiles in use)
    prefix(r)  = sum_{r' < r} tiles(r')            T_req = prefix(B)
    T          = H * T_req                          (H = nbKHeads)
    x(r, h, t) = H * prefix(r) + h * tiles(r) + t

CTA `c` of `P = gridDim.x` owns the half-open range

    R_c = [x_c, x_{c+1}),   x_c = ceil(c * T / P)      (64-bit products)

so `|R_c| in {floor(T/P), ceil(T/P)}` — max load ceil(T/P) tiles, the minimum
possible for any assignment of whole tiles.  The CTA containing tile `x` is
`c(x) = floor(x * P / T)` (proof: `ceil(cT/P) <= x  <=>  cT/P <= x  <=>  c <=
xP/T`).  For sequence `s` with linear extent `[L_s, L_s + tiles(s))`:

    c0 = c(L_s),  c1 = c(L_s + tiles(s) - 1),  nbPartials(s) = min(c1 - c0 + 1, tiles(s))

(Review fix: when T < P a CTA holds at most one tile, so CTAs strictly inside
[c0, c1] can be empty — `x_c == x_{c+1}` — and never arrive on the semaphore;
the count of non-empty CTAs is then exactly tiles(s).  When T >= P no CTA is
empty and the `min` is the identity.  The merger enumerates the i-th non-empty
CTA as `c0 + i` (T >= P) or `c(L_s + i)` (T < P).)

**Why static and not an atomic pull.**  A pull queue of equal items of `s`
tiles gives max load `ceil(N/P) * s`: with today's 13-tile items (680 of them)
that is 3 x 13 = 39 tiles; 8-tile items 5 x 8 = 40; 4-tile items 9 x 4 = 36;
only 1-tile items reach 33 — and a 1-tile item pays an atomic, a Q load and a
finalize + partial write per tile.  The contiguous partition reaches 33 with no
atomics, no queue, and items that are as long as the sequences allow (1-2 per
CTA for the bench).  Balance is exact for uniform tile cost; for mixed-format
streams the per-CTA sum over 33 tiles has spread ~sqrt(33) x sigma_tile ~ 3 us
for a 0.5 us per-tile sigma, against the 6-tile (8 us fp8) quantization loss of
the pull queue.  The per-CTA end histogram (section 6) measures this spread
directly; a dynamic tile-range pull is the fallback only if it exceeds 5 us.

For the bench: T = 8704, P = 264, 256 CTAs get 33 tiles and 8 get 32; every
range lies in at most two sequences (33 < 64), every sequence is covered by 2
or 3 CTAs (~400 partials instead of 680).  `P` is derived at launch from the
occupancy calculator and the SM count (:4493-4497, already there); an env
override `XQA_PERSISTENT_CTAS` (analogous to `XQA_NB_SUB_SEQ`) exists for the
conformance matrix only.  CTAs with `x_c == x_{c+1}` (T < P) do nothing after
the prologue.

**Items.**  A CTA's range decomposes into items = maximal runs inside one
sequence: `item = {req, head, tileBeg, tileEnd, gBeg}` with `g` the CTA-local
tile counter `g = x - x_c in [0, G)`, `G = x_{c+1} - x_c`.  Only a CTA's first
and last items can be partial (an interior item covers a whole sequence, so
`nbPartials = 1`).  Hence every partial is "CTA c's first item" or "CTA c's
last item", and scratch needs exactly **2P chunks**:

    chunk(c, item) = 2c + (item is the CTA's last item ? 1 : 0)

A merger enumerating `c in [c0, c1]` for sequence `s` reads chunk `2c+1` for
`c < c1` (those CTAs' ranges continue past their piece or end with it), and for
`c = c1` chunk `2c1+1` if `x_{c1+1} == L_s + tiles(s)` (the range ends with the
sequence, so the piece is that CTA's last item) else `2c1+0`.  Both sides
compute the same integer formula from `(L_s, tiles(s), T, P)`; no table.
`ScratchMem{scratch, 2P, 1}` replaces `{scratch, maxNbSubSeq*H*B, 1}` (:1692,
:2243, :2353): 528 chunks x 4 heads x 264 B = 557 KB for the bench.
Semaphores stay `semaphores[idxSeq]`, one per (req, head), `atom.inc` limit
`nbPartials - 1` (self-resetting, as today :2247-2250).

### 2.2 The tile stream: items live in four walker warps and nowhere else

The central simplification: **the K/V stage rings, X ring, converters, chunk
ring and all per-tile barriers see only the CTA-local tile counter `g`**.  What
a tile needs from its item is carried in its metadata record.  The per-tile
metadata entry (today `metaPages[2][2][16][4]` + `metaFormats` :344-345, 20 B
per tile) becomes a 32 B record:

    struct TileRecord {                 // one per (operand, chunk, tile), 16 B aligned
      KVCachePageIndex pages[4];        // 16 B; kBAD_PAGE_INDEX past the sequence end
      uint8_t  formats[4];              //  4 B; kMixedBadPageFormat past the end
      uint32_t idxReq;                  //  4 B  (Q / output / page-table base)
      uint16_t idxHeadGrp;              //  2 B  (payload/scale/TMA head coordinate)
      uint8_t  flags;                   //  bit0 firstOfItem  bit1 lastOfItem
                                        //  bit2 itemIsPartial  bit3 itemIsCtaLast
      uint8_t  validBeg, validEnd;      //  mask bounds in tokens (0..64)
      uint8_t  pad[3];
    };
    TileRecord meta[2][nbMetaChunks=2][metaChunkTiles=16];   // 2048 B (+768 B)

`validBeg = tile0NbSkipTokens` on a sequence's first tile (sliding window) else
0; `validEnd = cacheSeqLen % 64` on its last tile (if nonzero) else 64.  gemm0
therefore needs neither `cacheSeqLen` nor `nbTiles`; the converters need neither
`idxHeadGrp` nor `nbIters` beyond `G`; the a16 TMA loader takes the head
coordinate from the record.

Four warps walk the item sequence independently with a register-resident
cursor (warp-uniform, ~8 registers), all starting from the same prologue
result:

    struct ItemCursor {
      uint32_t x, xEnd;          // linear cursor, range end
      uint32_t req, head;        // current sequence
      uint32_t seqLenReq;        // cacheSeqLen(req)
      uint32_t tileInSeq;        // nbSkipLeadingTiles(req) + t
      uint32_t nextSeqLen;       // seqLen[req+1], loaded on entering req (latency hidden)
      uint32_t Lseq;             // linear start of the current sequence
    };
    next item: nb = min(tilesEnd(req) - tileInSeq, xEnd - x);
               emit {req, head, tileInSeq, tileInSeq + nb, gBeg = x - x0};
               x += nb; tileInSeq += nb; Lseq += (tileInSeq == tilesEnd) ? tiles(req) : 0;
               if tileInSeq == tilesEnd: head++; tileInSeq = skip(req);
                   if head == H: req++, head = 0, seqLenReq = nextSeqLen,
                                 issue nextSeqLen = seqLen[req+1]

- **K loader (IO warp 0)** and **V loader (IO warp 1)**: walk items inside
  `fillTileMeta` (which becomes: for chunk tiles `[gBeg, gBeg+16)`, loop over
  the items overlapping the chunk, fill the lanes' entries of each; one
  page-table load pair per entry as today :3030-3047).  A chunk may span many
  items (1-tile sequences) — the loop is warp-uniform over items, lane-parallel
  over entries within an item; no ring lookup, no barrier.
- **Q warp (IO warp 2, new)**: per item `j`: `QCvt::load` for
  `(req, head)` into 16 registers, `qBar.consumed.arrive_and_wait()`,
  `QCvt::store`, `fence.proxy.async`, `qBar.produced.arrive()`.  This is [36b]:
  the K loader warp never touches Q.
- **Merge warp (IO warp 3, new)**: per item `j`: if the item is partial, wait
  for `smem.finalizedItems > j` (see 2.5), then the semaphore protocol and, if
  last, the register-resident merge.

The **prologue scan** (IO warp 3, before the existing `__syncthreads` :1181,
in place of the per-thread `getCacheSeqLen` :1025): pass 1 sums `tiles(r)` over
`r < B` (lane-strided loads, warp inclusive scan; B=17 is one step, one L2/DRAM
round trip — the same round trip the kernel pays today at :1025); computes `T`,
`x_c`, `x_{c+1}`; pass 2 (L1-hot) finds `req0`, `head0`, `tile0`, `L_seq0`;
publishes `{x0, x1, req0, head0, tile0, Lseq0, seqLen[req0], seqLen[req0+1], T}`
and `G = x1 - x0` into `smem.sched` (48 B).  Every walker copies it into its
cursor after the `__syncthreads`; `G` replaces `nbIters` in every loop bound.

### 2.3 gemm0 (z = 0), new per-tile flow

    pre-arrive qBar.consumed, kBar[*].consumed        (once per CTA, as :1221-1224)
    for g in [0, G):
      rec = meta[K][(g/16)%2][g%16]                  (LDS: flags, validBeg, validEnd)
      if rec.firstOfItem:  runningColMax = -inf ; qBar.produced.arrive_and_wait()
      kBar[g%3].produced.arrive_and_wait()            (:1264, unchanged)
      QK HGMMA x8, wait, kBar[g%3].consumed.arrive()  (:1268-1322, unchanged)
      mask if validBeg > 0 || validEnd < 64            (:1338-1352, operands from rec)
      colMax via gemm0WarpColMax[g%2], softmax, X(g%2) store, xBar[g%2].produced.arrive()
      if rec.lastOfItem:   qBar.consumed.arrive()     (:1410 moved into the loop)

Reading `rec` after `kBar.produced` is visible and hazard-free (C10 below).
The per-item cost added to gemm0 is one Q-buffer round trip through the Q warp
(store of pre-loaded registers + fence + arrive; ~0.2-0.4 us), absorbed by the
three K stages and the two X entries; a second Q buffer (2 KB) is the fallback
if the trace shows `g0_kwait` growing at item boundaries (section 6).

### 2.4 gemm1 (z = 1), new per-tile flow

    pre-arrive vBar[*].consumed, xBar[*].consumed     (once, :1413-1422)
    for g in [0, G):
      rec = meta[V][(g/16)%2][g%16]                  (LDS: flags, idxReq, idxHeadGrp)
      if rec.firstOfItem:  acc = 0 ; accColMax = -inf ; accColSum = 0
      vBar[g%3].produced.arrive_and_wait(); xBar[g%2].produced.arrive_and_wait()
      rescale, PV HGMMA x8, commit, wait                 (:1532-1620, unchanged)
      if rec.lastOfItem:
        publish colMax/colSum (:1681-1685)
        if rec.itemIsPartial: chunk = 2*blockIdx.x + rec.itemIsCtaLast;
             rowSumMax[chunk], finalizeAndWriteOut_sync -> scratch tokens[chunk]
        else: finalizeAndWriteOut_sync -> output[headGrpSize*(H*idxReq + idxHeadGrp)] (+sinks)
        gemm1 group sync (inside finalize) ; thread 0: st.release.cta smem.finalizedItems = j+1
      xBar[g%2].consumed.arrive(); vBar[g%3].consumed.arrive()   (:1738-1740)

`acc` re-initialization per item is what the existing comment at :1530
("For persistent CTA, just re-initialize acc") anticipated.  `finalize` still
uses `outSwizzleBuf(g%2)`, which aliases X(g%2); gemm0 may write X(g%2) again
only after gemm1's `xBar[g%2].consumed.arrive()`, which follows finalize in
program order (as today for the last tile).

### 2.5 Merge warp (IO warp 3) and the partial protocol

Per item `j` of its cursor, if `nbPartials > 1`:

1. Poll `ld.acquire.cta smem.finalizedItems` until `> j` (one lane,
   `__nanosleep` back-off; the warp is otherwise idle).
2. One lane: `atom.acq_rel.gpu.inc semaphores[idxSeq], nbPartials - 1` (the
   instruction at :2247-2250, unchanged).  Broadcast `old`.
3. If `old == nbPartials - 1` (last): for `half in {0, 1}` (8 elements per lane,
   lane `l` owns head `l / 8`, elements `16*(l%8) + 8*half ..`): for `c in [c0,
   c1]`: chunk by the rule of 2.1; load `SumMax` (8 B, broadcast within the 8
   lanes of a head) and 16 B of bf16 tokens; online-combine as
   :2302-2311; add sinks (:2315-2321); write 16 B of output.  <= 3 partials x 24
   B per lane per half; ~2 us, off every other role's path.

The counter replaces a barrier deliberately: gemm1 must never wait on the
merge warp, and an mbarrier with only one arriving side has a phase-ambiguity
(a waiter two phases behind mis-reads parity) or an over-arrival (UB) problem.
A monotone `st.release.cta` / `ld.acquire.cta` word has neither.  The memory
chain that makes gemm1's scratch writes visible to another CTA's merger is the
one the code relies on today (:2229 `__syncthreads` then one thread's
`atom.acq_rel.gpu`): gemm1 writes -> group `bar.sync` inside finalize ->
thread 0 `st.release.cta` -> merge lane `ld.acquire.cta` -> `atom.release.gpu`
(release cumulativity); the merger's `atom.acquire.gpu` then orders its loads
after every other CTA's released writes.

The `MultiBlockSMem` block :2251-2400 is deleted for this build (its 4 KB
alias over `k[]` cannot coexist with the next item's stages); `isLastCta` goes.

### 2.6 K/V loaders (IO warps 0, 1)

    fill chunk 0 (tiles 0..15) ; kMetaReady[0].arrive()          (:1842-1843)
    for g in [0, G):
      rec = meta[K][..g..] ; publishPages(rec) (a16 TMA build)      (:1853-1855)
      kBar[g%3].consumed.arrive_and_wait()                         (:1860)
      elected: arrive_tx(txBytes) + A16 TMA boxes, head = rec.idxHeadGrp  (:1864-1885)
      ahead = g + 2 ; if ahead % 16 == 0 && ahead < G:
           fillTileMeta(chunk (ahead/16)%2, tiles [ahead, ahead+16)) ; kMetaReady[..].arrive()

The `ahead >= 32` guard (:1892) goes: chunk 1 is filled at `g = 14`, which is
[36a] (its WAR argument is vacuous: chunk 1 has no earlier contents).  Nothing
in the loop knows about items; the walker runs inside `fillTileMeta`.  V is
the mirror.  Because the chunk fill for tiles `[g+2, g+18)` and the converters'
copies for tile `g+2` are issued exactly as for any interior tile, the next
item's first K/V bytes are in flight two tiles before gemm0 needs them —
requirement (b) holds by construction rather than by a special case.

### 2.7 Converters (z = 3, 4)

Unchanged loop over `g < G` (:2121-2156); `issueCompressedPageCopies` reads
`idxHeadGrp` from the record it already reads the page/tag from (one more
LDS.U16 on the copy-issue path, 3 instructions).  Parity waits use `g`, which
is what `t`/`idxIter` already are — a running counter across items.  The
`commitGroup` per tile, including empty groups, keeps `waitGroup<kAhead-1>`
accounting uniform through item boundaries.

### 2.8 Kernel end, host

After every role's loop: `__syncthreads` :2229 (this also waits for the merge
warp's last merge), destroy barriers :2230-2236, return.  Host: `dimGrid = {P,
1, 1}`, `P = ctasPerSm * multiProcessorCount` (or the env override);
`chooseNbSubSeq` is not called for this build (kept for SPEC_DEC / non-mixed).
The kernel asserts `gridDim.y == 1 && gridDim.z == 1` under the persistent
build.  `ScratchMem` chunk count `2 * gridDim.x`.

### 2.9 Rejected alternatives (recorded so they are not re-proposed)

- Atomic pull of `(seq, subseq)` items: 39-tile max load (2.1).
- Item descriptor ring in smem with ready/free barriers for all roles: a
  16-tile chunk fill can need up to 18 items ahead (1-tile sequences), so the
  ring must be >= 32 deep or it deadlocks against the chunk's ready barrier;
  the per-tile record makes the ring unnecessary.
- Keeping `MultiBlockSMem` and deferring merges to after the loop: adds up to
  2 x 2 us to the wall, and the alias is still live during the last item.
- Merge on the gemm1 warps: 1-2 us of gemm1 stall per merge; the X ring
  (depth 2) does not cover it.
- Second Q buffer: +2 KB for ~0.3 us per item on a non-pacing role; kept as a
  fallback, not in the design.

## 3. Barrier, parity and ownership invariants

Existing: D1-D6, C1-C7 (`mixed_kv_page_transport_dataflow.md`, A1-A6).  The
XQA host's analogue of the A5 hazard ("a wait for item i+1 on a phase that item
i's last arrive can still satisfy") is handled as follows.

**Token waits are immune.**  Every `arrive_and_wait` in the kernel
(`MBarrier::arrive_and_wait` = `wait(arrive())`, barriers.cuh :361-364) waits on
the token of its own arrival, which encodes the phase.  These are: gemm0
`kBar.produced` :1264, `xBar.consumed` :3336, `qBar.produced` :1243; gemm1
`vBar.produced` :1462, `xBar.produced` :1478; loaders `kBar/vBar.consumed`
:1860/:1989; Q warp `qBar.consumed` :1804.  Their phase bookkeeping needs no
counter and cannot alias across items.

**Parity waits are keyed on the CTA-local tile counter `g`.**  Three sites:
converters `kBar[g%3].consumed.wait_parity(toParity<3>(g))` :2115 (V :2178),
`kMetaReady[(g/16)%2].wait_parity(toParity<1>(g/32))` :2110 (V :2175).  Today
`t` is the CTA's tile index; with items it stays the CTA's tile index — no
per-item reset exists anywhere (requirement (a)).

- **C8 (consumer-gated parity).**  A parity wait is unambiguous iff the waiter
  can never be two phases behind the barrier.  For `consumed[s]`: phase k+1
  completes only with the loader's arrive for tile `3(k+1)+s`, which is after
  gemm0's arrive for tile `3k+s`, which is after `produced(3k+s)`, which needs
  the converter's arrive for `3k+s`, which follows that converter's wait on
  phase k.  So the converter's wait on phase k precedes phase k+1's
  completion; the chain is per tile and does not mention items.  For
  `kMetaReady[c]`: fill f+1 of chunk c is issued at tile `32(f+1) + 16c - 2`
  after `kBar.consumed` for that tile, which implies gemm0 consumed tile
  `32(f+1)+16c-5`, which implies the converters expanded it, which is after
  their chunk wait for fill f.  Same shape.
- **C9 (item-agnostic tile stream).**  Item state exists only in the four
  walker warps' cursors and in the `TileRecord` fields they write.  No ring,
  stage, barrier count, `cp.async` group count or trace slot depends on item
  boundaries.  gemm0/gemm1 read item transitions from record flags in tile
  order.  This is the structural statement of requirement (a).
- **C10 (record visibility and reuse).**  A record for tile g is written by
  the loader warp before `kMetaReady.arrive()` (release), read by the
  converters after their `wait_parity` (acquire), and read by gemm0 after
  `kBar[g%3].produced` completes, which the converters' arrive (release, after
  their read) is part of: two mbarrier synchronizes-with edges give
  happens-before from the write to gemm0's read.  gemm1 reads the V record
  through `vBar[g%3].produced` identically.  Reuse (WAR): entry g's slot is
  rewritten for tile g+32 at tile g+30, after `kBar[(g+30)%3].consumed`, which
  implies gemm0 consumed tile g+27, which is after gemm0 read record g (program
  order in gemm0) and after the converter's copy issue for g (C8 chain).  The
  existing chunk refill comment at :1856-1859 (:1857) is this argument for the pages
  and tags; the new fields add readers (gemm0, gemm1) that sit strictly
  inside the same window.
- **C11 (scratch slot rule).**  Every partial is a CTA's first or last item
  (a contiguous range that contains a sequence boundary on both sides of an
  item contains the whole sequence).  Slot `2c + isCtaLast` is therefore
  unique per partial, and the merger's enumeration (2.1) reproduces it from
  `(L_s, tiles(s), T, P)` alone.  Items with `nbPartials == 1` never touch
  scratch or semaphores.  Merges may complete in any order across CTAs; the
  semaphore's `inc` with limit `nbPartials - 1` counts arrivals, not
  identities (as today).
- **C12 (merge hand-off without phases).**  gemm1 -> merge warp is a monotone
  counter (`st.release.cta` / `ld.acquire.cta`), so gemm1 never blocks on the
  merge warp and no phase can be skipped.  Visibility to other CTAs follows
  the existing release chain (2.5).
- **C13 (balanced partition).**  `|R_c| in {floor(T/P), ceil(T/P)}`; every
  role's work per CTA is `G` tiles plus one finalize per item; the number of
  items per CTA is `1 + (number of sequence boundaries inside R_c)`, <= 2 for
  the bench.  The bound is arithmetic, not measured.
- **D3/D6 (ownership) and C6 (issue budget)** are untouched: the copy and
  expansion code is unchanged except for the head-coordinate source.
- **C4** (barrier accounting) unchanged: counts are as at :1123-1174; only the
  number of phases per barrier grows (from ~13/3 to ~33/3 per stage).
- **C7-class (item i+1 begins mid-pipeline).**  With copies two tiles ahead
  and chunks 16 tiles ahead, the first tile of item i+1 is always a
  compressed tile issued before item i finished; the matrix must contain
  cases with many items per CTA (section 6) — the current 54 cases have one
  item per CTA except (2200, sub 1).
- **qBar per item.**  `consumed` phase j completes with gemm0's arrive (the
  pre-arrive for j=0, the end-of-item arrive for j>=1) plus the Q warp's
  arrive for item j; `produced` phase j with the Q warp's arrive plus gemm0's
  wait-arrive.  gemm0's end-of-item arrive precedes its next `produced` wait in
  program order, so the Q warp's `consumed` wait for item j+1 is always
  satisfiable; the Q warp's `produced` arrive for j+1 needs only its own
  `consumed` wait.  No cycle.
- **Liveness across items.**  New inter-role edges: gemm1 -> merge warp (one
  direction, counter), Q warp <-> gemm0 (qBar, per item, acyclic as above).
  Walkers wait on nothing.  Everything else is the existing per-tile pipeline
  with a longer tile stream.

## 4. Register and shared-memory budget

**Registers** (`__launch_bounds__(640,2)`: pool 640 x 48 = 30720; split
3x128x40 + 2x128x56 = 29696, :1206-1210, unchanged):

| role | budget | added live state | expectation |
|---|---|---|---|
| gemm0 | 40 | record LDS (transient), 2 predicates | 0 spill (`ptxas -v`) |
| gemm1 | 40 | record LDS (transient), routing at item end (transient) | 0 spill |
| IO warps 0/1 (loaders) | 40 | `ItemCursor` ~8 regs across the tile loop; `KVTilePartLoader` loses 3 const members | 0 spill; fallback: cursor in smem (32 B per warp, LDS/STS at item boundaries only) |
| IO warp 2 (Q) | 40 | 16 regs Q data + cursor 8 + addresses | ~30 |
| IO warp 3 (merge) | 40 | cursor 8 + 8 fp32 acc + 4 data + sum/max/ptrs | ~28 |
| converters | 56 | head LDS at copy issue (transient) | unchanged (188 / 187 SASS expansion) |

(Outcome, recorded at confirmation in section 10: none of the fallbacks in
this section — smem cursor, IO at 48 — was needed; every module is 40/56 with
0 stack / 0 spill and two `USETMAXREG`.)

If the IO group does not fit 40 with the cursor in registers and the smem
cursor also spills, the pool admits IO at 48 with the GEMM groups at 40:
2x128x40 + 128x48 + 2x128x56 = 30720 exactly; the `.dec 40` at :1207 then
splits into `.dec 40` for z in {0,1} and `.dec 48` for z = 2 (three
`USETMAXREG` in the SASS instead of two — the A4 count changes to 3 in that
case and must be recorded).  Check: `ptxas -v` prints no C7507, 0 bytes stack,
0 spill stores/loads; `cuobjdump -sass | grep -c USETMAXREG` = 2 (or 3 with the
fallback); `LDL`/`STL` = 0.

**Shared memory** (2 CTAs/SM bound: 233472 / 2 - 1024 = 115712 B per CTA;
current `sizeof(SharedMem)` = 110848 B):

| change | bytes |
|---|---|
| `TileRecord meta[2][2][16]` replaces `metaPages` (1024) + `metaFormats` (256) | +768 |
| `smem.sched` (prologue scan result, 48 B) + `nbCtaTiles` + `finalizedItems` | +64 |
| `isLastCta` removed; `MultiBlockSMem` alias removed (no size) | -1 (padding) |
| **total** | **~111680 B**, margin ~4 KB |
| trace build: `MIXED_KV_TRACE_TILES=16` (+8 x 128) + per-CTA globaltimer record (global, not smem) | 113728 B, still 2 CTAs/SM |

`static_assert(sizeof(SharedMem) + 1024 <= 233472 / 2)` is added so a future
field cannot silently drop occupancy to 1; the ncu occupancy row and the trace
residency probe (`mixedKvTraceSmResident`, expect 2) confirm at runtime.

## 5. Predicted per-tile period and wall

Per-tile periods do not change: the pipeline per tile is identical, plus one
LDS in gemm0, one in gemm1, one LDS.U16 per copy issue in each converter warp
(3 instructions against ~2100 per tile).  Inputs (round-2 baseline, CTA 0
tiles 3-7): body-derived `T_tile` fp8 1.38, fp4 1.15, mixed 1.44 (from 79.5 =
3 x (13 T + 4.4) + 10.0), a16 DRAM-bound.  Fill `F`: 3.95 us fp4 today (clk0 ->
first K ready); minus [36a] (one page-table load pair, ~0.3 us) and [36b] (Q
round trip off the K warp, ~0.5-1 us) -> ~3.5 us for fp8/fp4/mixed, ~6.5 us
for a16 (7.75 today).  Tiles per CTA `G = 33`.  Finalizes: <= 2 per CTA at
~0.4 us.  Tail: last CTA's merge (~2 us: one atomic round trip + one load
round trip + write) + drain (~1 us).

    wall' = F + G x T_tile + finalizes + tail

| mode | today | F | 33 x T | fin | tail | predicted | pessimistic (K-conv period 1.62 as T) | target |
|---|---|---|---|---|---|---|---|---|
| fp8 | 76.9 | 3.5 | 45.5 | 0.8 | 3 | **~53** | 60.7 | <= 58 |
| fp4 | 70.7 | 3.5 | 38.0 | 0.8 | 3 | **~45** | — | <= 36 (not reachable by [8] alone; [15]/[7]) |
| mixed | 79.5 | 3.5 | 47.5 | 0.8 | 3 | **~55** | — | <= 62 |
| a16 | 81.8 | 6.5 | DRAM 285 MB / 3.5 TB/s = 81 | 0.8 | 3 | **78-81** (ramp only) | — | parity |

Cross-checks the histogram must confirm: (i) all 264 CTAs start within the
launch ramp (< 5 us spread); (ii) per-CTA body (first K ready -> last tile
done) = 33 x T_tile +- 5 %; (iii) end spread (max - median) <= 5 us; (iv) the
ncu `sm__cycles_active.avg / .max` >= 0.9 (today ~0.6-0.7 from the 2.58-wave
quantization).  If (ii) holds and the wall does not follow, the tail model
(iv) is wrong and the fix is in the merge/drain, not in the pipeline.

## 6. Verification artifacts and accept / reject

Build checks (each module: static a16 / fp8 / fp4 and the mixed module):

1. `ptxas -v` on the TU with the ninja flags (`/tmp/main_ptx/ninja_flags.py`):
   no C7507; 0 bytes stack frame, 0 spill stores, 0 spill loads.
2. `cuobjdump -sass`: `USETMAXREG` = 2 (`DEALLOC 0x28`, `TRY_ALLOC 0x38`); `LDL`
   = `STL` = 0; `UTMALDG` = 8 in the a16 and mixed modules (the K and V loaders
   are one code path with operand-selected addresses, so the 16 sites of the
   baseline halve; the dynamic box count per tile is unchanged) and 0 in the
   fp8/fp4 static modules (`mixedLoaderTma` false); `LDGSTS` > 0; exactly one
   `ATOM...INC` (the semaphore) and no `ATOMS`.  Barrier sites **per role**
   (`xqa_sm90_converter_sass.py`-style split of the SASS by the warp-group
   branch, not kernel totals — the totals fall with the loader merge and the
   deleted `MultiBlockSMem` epilogue and cannot show whether a GEMM role gained
   a site): gemm0 SYNCS.PHASECHK / ARRIVE / BAR.SYNC 1 / 3 / 12 as the baseline;
   gemm1 1 / 7 / 11 plus the in-loop finalize's `warpGrpBar` syncs, executed
   once per item (the finalize moved from the epilogue into the loop; its
   per-tile path gains no site); HGMMA 8 + 8.  Accepted cost: five
   `CALL.REL.NOINC` sites to ptxas' 64-bit division subroutine (prologue scan
   x0 / x1; merge warp c0 / c1 / x_{c1+1}) plus one in the T < P branch of the
   merge enumeration — once per CTA or once per partial item, on no per-tile
   path.
3. `cuobjdump -res-usage`: REG 48 (launch cap), STACK 0; smem = the new
   `sizeof(SharedMem)`; `cudaOccupancyMaxActiveBlocksPerMultiprocessor` = 2.

Conformance (`python tests/attention/run_xqa_mixed_page_transport.py`, exit
code = failures): the existing 54 cases (all q=4/64 cases run the untouched
SPEC_DEC / mha.cu paths and must be byte-identical), plus persistent-specific
cases added to `TAIL_CASES` with an `XQA_PERSISTENT_CTAS` override replacing
the `XQA_NB_SUB_SEQ` override semantics:
- P = 1 (one CTA, everything: 2 x 8 x tiles items, > 2 chunk refills, every
  output direct, no merge);
- P = 3 (multi-item CTAs, partials on both range ends, merges by non-first
  CTAs);
- P = default (T < P: empty CTAs; 64-tile sequences split into up to 17
  partials; seq_len 50 / 100 / 130 -> 1-3 tile items so one chunk spans many
  items and every item after the first begins with a compressed tile — the
  C7 class);
- fp8 / fp4 / mixed each.  Expected 54 + 27 = 81 cases, all PASS.

Timing (locked, `flock /tmp/mixedkv-gpu0.lock bash /home/bigboi/mixedkv_remote_run.sh
<checkout> r2p8 sm90 transport_a16 fp8 fp4 mixed`; 5 x 5, repeats x kernel <
1.5 ms; `/tmp/mixedkv-r2p8.bench.txt` rows `q_len 1` for the four modes;
`q_len 4` rows must equal the baseline within spread):

| mode | accept ([8] gate: -12 % at unchanged T_tile) | predicted | reject if |
|---|---|---|---|
| fp8 | median <= 67.7 | ~53 | > 67.7 or periods changed > 5 % |
| fp4 | median <= 62.2 | ~45 | > 62.2 |
| mixed | median <= 70.0 | ~55 | > 70.0 |
| a16 | median <= 82 (parity) | 78-81 | > 84 |

Trace (`MIXED_KV_TRACE 1`, `python benchmarks/xqa_mixed_trace_once.py --modes fp8
fp4 transport_a16 --q-len 1 --launches 3` under the lock):
- Same-launch role periods (slots g0_kwait g0_mma g0_smax g0_xarr g1_vwait
  g1_xwait g1_rs g1_mma kl_start kl_iss vl_start vl_iss kc_ready kc_done
  vc_ready vc_done, tiles 3-7 of CTA 0): fp8 1.38 / 1.46 / 1.43 / 1.39 / 1.62 /
  1.02, fp4 1.14 / 1.15 / 1.13 / 1.17 / 1.18 / 1.12, a16 2.52 / 2.42 / 2.38 /
  2.68 / 2.88 / 2.51 within +-5 %.
- Item boundary: the trace gains two compile-time knobs, `MIXED_KV_TRACE_CTA`
  (which CTA records) and `MIXED_KV_TRACE_TILE0` (first recorded tile), so CTA
  1 tiles 24..39 (its boundary is at g = 31: x_1 = 33, sequence 0 ends at 64)
  show `g0_kwait(31->32)` and `kc_ready(32)` at the steady-state period, i.e.
  no first-K latency at the boundary; `kl_start` gaps unchanged.
- Per-CTA `%globaltimer` record `g_mixedKvCtaTrace[P][4] = {start, firstKReady,
  lastTileDone, end}` (trace build only; the p03 tooling
  `benchmarks/parse_xqa_trace.py` is extended to print the histogram):
  start spread < 5 us; body = 33 x T +- 5 %; end spread <= 5 us; a per-slot
  idle fraction ((wall - body - F) / wall) <= 10 %.

ncu (one kernel, `--launch-skip` past warm-up): `launch__grid_size` = 264,
`launch__occupancy_limit_registers` = `_shared_mem` = 2,
`sm__ctas_launched.sum` = 264, `sm__cycles_active.avg / .max` >= 0.9,
`dram__bytes_read.sum` unchanged from baseline per mode (bytes do not change),
`gpu__time_duration` consistent with the bench median.

## 7. Do not build if

1. The prologue scan or the walker cannot be expressed without a
   per-tile or per-item smem table of unbounded size — the design depends on
   the cursor (8 registers) and the 32 B per-tile record only.
2. `ptxas -v` at 40/56 spills in any role with the cursor in registers **and**
   with the smem-cursor fallback **and** with IO at 48 (pool exactly 30720):
   the register story then needs [15]'s 64-register layout first.
3. `sizeof(SharedMem)` would exceed 115712 B (2 CTAs/SM) — including the
   trace build at `MIXED_KV_TRACE_TILES=16`.
4. The occupancy calculator returns `ctasPerSm != 2` on H200 for this module
   (then P = 132 or 396 and the model's 33 tiles per CTA is false; re-derive).
5. The same-launch trace on the persistent build shows any role period moved
   by more than 5 % from the baseline table — the lever's premise is
   "unchanged periods"; a moved period means the record LDS or the Q hand-off
   entered a critical path and must be redesigned before timing.
6. main's barrier set changes before this lands (the plan's rebase rule:
   "[8] must rebase onto A+B's final barrier set"): re-derive C8/C10 for the
   new set instead of merging mechanically.
7. The conformance runner cannot take the `XQA_PERSISTENT_CTAS` override:
   without the P = 1 / P = 3 / T < P cases the persistent path's item
   boundaries are untested (the existing matrix has one item per CTA).
8. (Withdrawn for head counts by 8.6; replaced by:) the Q warp is written for
   `needInputCvt == false` (bf16 KV cache, `CACHE_ELEM_ENUM = 5`, the only
   configuration of this matrix): `static_assert(nbQLdWarps == 1 && nbQLdThrds
   == warp_size)` in the Q warp.  An fp8-cache mixed build (`nbQLdWarps` > 1)
   fails to compile rather than misbehave; extending the Q warp to a multi-warp
   converter is out of this round's scope.
9. The semaphore region is not zero at first use in the target deployment —
   the same requirement the current multi-block path has; not new, but with
   P-indexed scratch a stale semaphore now corrupts a different sequence's
   merge count, so confirm the allocation path zeroes it.

Scope: `ENABLE_MIXED_KV_CACHE && !SPEC_DEC` only; SPEC_DEC (Track S) and the
non-mixed sm90 kernel keep `chooseNbSubSeq` and the `1 x n x (B*H)` grid.
Unverified by this change: the sliding-window bookkeeping (`seqSkipTokens`,
`seqTilesInUse`, `validBeg = skip % 64` on `t == 0`) is compiled out in every
module of the matrix (`SLIDING_WINDOW = 0`); it is written to the same
arithmetic as the non-persistent path but has no conformance case.

---

## 8. Amendments after review (resolved before code)

Each item names the reviewer finding it answers.  Where it changes sections
1-7 above, the amendment wins.

### 8.1 Record read ordering (blockers 1 and 3) — C10 restated

The pseudo-code of 2.3/2.4 read the record before the tile's `produced`
wait.  That is wrong at g = 0 (no happens-before edge from the loader's
chunk-0 STS to gemm0's LDS: gemm0 has waited on nothing but `qBar`, which the
Q warp — a different warp from the K loader — arrives) and only accidentally
right for g >= 1.  **Rule: every reader of record g reads it after the wait
that the record's writer chain feeds.**

- gemm0 reads record g (one `LDS.32` of the `tile` word) immediately after
  `kBar[g%3].produced.arrive_and_wait()` and before its `kBar.consumed.arrive`.
  Chain for every g >= 0: K loader STS (fill) -> `kMetaReady[c].arrive`
  (release) -> K converter `wait_parity` (acquire) at `issue(g)` -> (program
  order) converter `kBar[g%3].produced.arrive` (release) for tile g -> gemm0
  wait (acquire).  For g in {0, 1} the converter's `issue(g)` is its prologue,
  which still follows its `kMetaReady[0]` wait.  The Q wait therefore follows
  the K wait: `if (first) { runningColMax = -inf; qBar[j&1].produced.arrive_and_wait(); }`.
  Q is needed only by the HGMMA, which comes after both waits.
- gemm1 reads record g's `tile` word after `vBar[g%3].produced.arrive_and_wait()`
  (same chain through the V loader / V converters) and the `idxReq`/`idxHeadGrp`
  pair only inside `if (last)`.
- Converters read pages/format/head at `issue(g)` after their `kMetaReady`
  wait (unchanged).
- The a16 TMA loader reads record g at its own iteration g, after its own
  fill (program order).

WAR (reuse) with the record read placed *before* the reader's release arrive:
the slot refilled at loader iteration g (section 8.2: tiles `[g+4, g+20)`)
previously held tiles `[g-28, g-12)`.  Last readers of tile g-13: gemm1 (after
`vBar.produced(g-13)`, before its `xBar[(g-13)%2].consumed.arrive`), gemm0
(before its `kBar.consumed.arrive(g-13)`), converters (`issue(g-13)` at
iteration g-15, before `produced.arrive(g-14)`).  The loader's
`kBar[g%3].consumed` wait at iteration g needs gemm0's `consumed.arrive(g-3)`,
which is program-ordered after gemm0's read of g-3 (hence of g-13), after
gemm0's `xBar.consumed` wait for tile g-11 (which needs gemm1's arrive for
tile g-13, program-ordered after gemm1's read of record g-13), and after
gemm0's `produced` wait for g-3 (which needs the converters' arrive for g-3,
program-ordered after their read of g-1).  So a lead L is WAR-safe iff the
refilled slot's newest old tile `g+L-17 <= g-2`, i.e. L <= 15.  L = 4 (8.2).

### 8.2 Metadata chunk fill: exactly one fill per chunk, lead 4 (notes)

The prologue fills chunk 0 only.  Chunk c is filled for tiles `[16k, 16k+16)`
at loader iteration `16k - 4` (`ahead = g + 4; ahead % 16 == 0 && ahead < G`),
so chunk 1's first fill is at g = 12 and every chunk is filled exactly once
per 32 tiles: `kMetaReady[c]` completes phase f on fill f, which is what the
converters' `wait_parity(toParity<1>(t/32))` at `t % 16 == 0` assumes.  (The
old `ahead >= 32` guard existed because the prologue also filled chunk 1; a
second fill at ahead = 16 would have arrived `kMetaReady[1]` twice and put the
parity one phase off — a hang, not a vacuous guard.  With the prologue fill of
chunk 1 removed the guard is removed too; the invariant is stated here.)

Lead 4 instead of 2: the fill's dependent page-table pair (`idxPage ->
page_format[page]`, 0.75-1.25 us under load) is issued at loader iteration
16k-4, gated on gemm0's `consumed(16k-7)`; the converters need it at
`issue(16k)` in their iteration 16k-2.  That is ~2 tile periods of slack
against ~1 us of latency.  C8 for `kMetaReady` with lead 4: fill f+1 of chunk c
is issued at tile 32(f+1)+16c-4 after `kBar.consumed` for that tile ->
gemm0 consumed 32(f+1)+16c-7 -> converters produced it -> they passed
`issue(32f+16c)` (their wait on phase f) — the waiter is never two phases
behind.  The fill loop itself is two-phase so its latency does not scale with
the number of items in the chunk (8.4).

Slack budget, stated: the fill's synchronous page-table pair blocks the loader
for ~0.75-1.25 us at iteration 16k-4, which delays its `consumed.arrive` for
tile 16k-3; the converters' `issue(16k-2)` parity wait needs that arrive, so
the slack is two tile periods (~2.3 us fp8 / fp4) against a 1.25 us worst
case — adequate, not large.  The confirmation trace with `TILE0 = 11` and
`TILE0 = 27` must show `kc_ready(16)` and `kc_ready(32)` at the steady-state
period; if it does not, the fallback is `MIXED_KV_META_LEAD` 6-8 (WAR-safe
for any lead <= 15 by 8.1), not a change of protocol.

### 8.3 `ItemCursor` (blocker 2 and notes)

One `ItemCursor` implementation is used by all four walkers (K loader, V
loader, Q warp, merge warp); the loaders call it with a clip limit (chunk
end), the Q and merge warps with `xEnd`, so the Q/merge item sequences and the
loaders' record flags are the same enumeration by construction.  State:
`x, xEnd, x0, req, head, tileInSeq, seqLen, nextSeqLen, Lseq` (9 registers).
`nextSeqLen = seqLen[req+1]` is loaded **only if `req + 1 < batchSize`**
(else 0) — in the prologue scan and in the cursor.  Requests with zero tiles
in use are skipped when entered (`while (req < B && tiles(seqLen) == 0)`); the
scan's containment test `H*prefix(r) <= x0 < H*prefix(r+1)` skips them
automatically (empty interval).

Per-tile flags derived by the cursor for tile at linear x, in-use tile t of
sequence (req, head) with tiles(req) = tl:

    first    = (x == x0) || (t == 0)
    last     = (x + 1 == xEnd) || (t + 1 == tl)
    partial  = !(Lseq >= x0 && Lseq + tl <= xEnd)     (the item does not cover the whole sequence
                                                       <=> some other CTA holds a tile of it <=> c0 < c1)
    ctaLast  = (Lseq + tl >= xEnd)
    validBeg = (t == 0) ? tile0NbSkipTokens(req) : 0
    validEnd = (t + 1 == tl) ? (seqLen % 64 ? seqLen % 64 : 64) : 64

`Lseq >= x0` identifies "this item started at the sequence start" without a
flag: only the CTA's first item can start mid-sequence.  Entries of the last
chunk past G are written as `kBAD_PAGE_INDEX` / `kMixedBadPageFormat` /
`tile = 0`, so converters zero-fill and nothing stale is read.  gemm1 counts
`last` flags to know the item index j and publishes `finalizedItems = j + 1`.

Hang guard: the merge warp asserts (debug builds) at the end of its item loop,
after the final `__syncthreads`, that `smem.finalizedItems` equals the number
of items it enumerated.

### 8.4 Fill is two-phase; per-fill cost is one dependent round trip

`fillTileMeta(chunk)`: phase A walks the pieces overlapping the chunk with the
cursor (warp-uniform, ALU only, plus one `seqLen[req+1]` load per request
boundary crossed); each lane captures, for its two entries (tile lane/4 and
lane/4 + 8, page lane % 4), the `(req, seqTile, nbPages, tile word, head)` of
the piece that contains its tile.  Phase B issues the two dependent load
pairs of every lane at once.  Phase C gathers the four page formats of a tile
into its lane j = 0 (three `shfl`) and writes the record: lanes write their
page (`STS.32`), lane j = 0 the second 16 B (`STS.128`).  The number of items
in a chunk (up to 16 with 1-tile sequences) changes only the ALU loop, not the
number of exposed memory round trips.

### 8.5 Q is double-buffered (note promoted to design)

`smem.q[2]`, `qBar[2]` (+2 KB, +16 B).  Q warp item j: `QCvt::load(laneId(), ...)`
into registers, `qBar[j&1].consumed.arrive_and_wait()`, `QCvt::store(laneId(),
smem.q[j&1], ...)`, `fence.proxy.async`, `qBar[j&1].produced.arrive()`.  gemm0
pre-arrives `consumed` on both buffers, waits `qBar[j&1].produced` on the first
tile of item j (after the K wait, 8.1), and arrives `qBar[j&1].consumed` after
the last tile of item j.  `qBar[b].consumed` phase m completes with the Q
warp's arrive for item 2m+b and gemm0's arrive for item 2m+b-2 (the pre-arrive
for m = 0); gemm0 finishes item j-2 before it can need Q(j), so the Q warp's
store of Q(j) happens while item j-1 runs and gemm0's `produced` wait at the
item boundary is already complete: no rendezvous on gemm0's path.  Acyclic
(the only edges are gemm0 -> Q warp on `consumed`, Q warp -> gemm0 on
`produced`, both per item).  `QCvt::load/store` take `laneId()` (the
converter is 32-thread; `threadIdx.x` on IO warp 2 is 64..95).

### 8.6 Merge warp (notes)

- Polls `ld.acquire.cta finalizedItems` (all 32 lanes, one broadcast LDS)
  with `__nanosleep(1000)` between polls: <= 1 poll/us against converters
  issuing ~300 instructions/us — negligible issue share.
- Lane 0 executes `atom.acq_rel.gpu.inc`; `old` is broadcast by `shfl`; a
  `__syncwarp()` follows the broadcast so the other lanes' loads are ordered
  after lane 0's acquire (bar.warp.sync orders memory among the participating
  lanes); the scratch loads use `__ldcg` (L2, no L1 allocation).
- Head mapping: lane l owns head `l/8` of a group of 4, elements
  `16*(l%8) .. +16`, processed in two 8-element halves (8 fp32 accumulators);
  the pass loops over head groups of 4 (`divUp(ctaNbValidQHeads, 4)` passes,
  compile-time), so any `headGrpSize` is covered (7.8 is withdrawn), and each
  half is masked by `elem < validElemsPerHead` (isHeadPadded).
- Tail model: if the merge warp is the last arriver for both of its items the
  tail is two merges (~4 us) plus the last finalize; the histogram check (iii)
  reads "end spread <= 5 us" with that in mind.

### 8.7 Kernel entry changes (notes)

Under the persistent build: no `idxReq`/`idxHeadGrp`/`cacheSeqLen` per
thread; `assert(gridDim.y == 1 && gridDim.z == 1)` replaces the
`gridDim.x == nbInputSeqSplit` / `gridDim.z == nbKHeads*batchSize` asserts;
the `idxSubSeq >= nbSubSeq` early return is gone (every CTA reaches barrier
init and the `__syncthreads`); `smem.finalizedItems` is zeroed and
`smem.sched` written (by IO warp 3's scan) before the `__syncthreads`; T = 0
(all sequences empty) gives `x0 = x1 = 0` for every CTA (no division: the
scan tests `T == 0` first) and every role's loop runs zero times.

### 8.8 Converter head coordinate cost (note)

Today `idxHeadGrp * payload_stride.head` and `idxHeadGrp * scale_stride.head`
are kernel-uniform and hoisted.  Per tile they become one `LDS.32` plus two
64-bit multiply-add chains (~6-8 SASS of ~310 per converter lane-tile); the
SASS count of the copy-issue path is part of the build check (+-5 rule on the
`xqa_sm90_converter_sass.py` counts).

### 8.9 Shared memory (note)

Current 110 848 B (re-derived: k 49 152 + X/out 4 096 + v 49 152 + q 2 048 +
col vectors 672 + pages/formats/meta/scales 5 416 + barriers 192 + isLastCta 1
-> 110 729 -> 110 848 at 128 B alignment).  Changes: records +768 (2 048 for
1 280), `sched` 48 + `finalizedItems` 4, second Q buffer +2 048, second `qBar`
+16, minus `pages`/`pageFormats`/`isLastCta` (41).  Total 113 572 -> **113 664 B**
(128-aligned); trace build (`MIXED_KV_TRACE_TILES=8`, +1 032) 114 604 ->
**114 688 B**, both <= 115 712.  The trace does **not** default to 16 tiles:
`MIXED_KV_TRACE_TILE0` moves the 8-tile window (TILE0 = 27 covers the CTA-1
boundary 31 -> 32 and the refill tile 32; TILE0 = 11 covers 12..19 for the
chunk-1 fill at g = 12 / use at 16).  A `static_assert(sizeof(SharedMem) +
1024 <= 233472 / 2)` is in the code; the measured `sizeof` is recorded at
confirmation.

### 8.10 Prediction restated (blocker 4 and note)

A pipeline's steady-state tile period is the maximum of its roles' periods;
1.38 us (gemm0 over tiles 3-7 of a 13-tile CTA, with the converters' 2-tile
lead hiding the difference) is not a steady state when the K converter runs at
1.62.  For a 33-tile CTA the K converter's 1.62 is the expected fp8 period; the
optimistic column stands only if the same-launch trace on the persistent build
shows the K converter period at or below gemm0's.

| mode | period used | wall = F 3.5 + 33 x T + fin 0.8 + tail 3 | target | status |
|---|---|---|---|---|
| fp8 | 1.62 (K converter) | **60.8**; 53.3 if K conv <= 1.38 | <= 58 | pass/fail decided by the trace: fp8 <= 58 needs [15]/[43] (K converter parity) in addition to [8] unless the persistent trace shows kc_done <= 1.38 |
| fp4 | 1.18 | **46.2** | <= 36 | fails as stated; [15]/[7] |
| mixed | 1.44 (body-derived; no same-launch trace exists) | **54.8** | <= 62 | pass |
| a16 | DRAM 2.5 | **78-81** | parity | pass |

The confirmation trace must reproduce, on the persistent build, the
same-launch periods of the baseline table (backends.md "Round 2 baseline")
within +-5 %: fp8 gemm0 1.38 / gemm1 1.46 / K-load 1.43 / V-load 1.39 /
K-convert 1.62 / V-convert 1.02; fp4 1.14 / 1.15 / 1.13 / 1.17 / 1.18 / 1.12.
The role that paces fp8 on the persistent build is whichever period the wall
follows: 33 x 1.62 + 7.3 = 60.8 or 33 x 1.38 + 7.3 = 52.8.  The accept gate
stays at -12 % (67.7 / 62.2 / 70.0) because it tests *this* lever's premise
(unchanged periods, one fill, no wave tail); the targets are a separate line.
Additional trace acceptance: `kc_ready(16)` and `kc_ready(32)` at steady-state
period (window TILE0 = 11 and TILE0 = 27), no `g0_kwait` bump at the item
boundary.

### 8.11 Protocol notes confirmed (no change)

qBar per-item phases acyclic (8.5); kBar/vBar/xBar waits are token waits;
converter parity waits are g-keyed with the C8 bound; scratch slot rule
`2c + isCtaLast` and the merger's enumeration agree (`c < c1` -> `2c+1`;
`c = c1` -> `2c1+1` iff `x_{c1+1} == L_s + tiles(s)` else `2c1`); no CTA
spin-waits on another CTA (co-residency is a performance assumption only);
`outSwizzleBuf(i)` aliases only X(i), so finalize inside the loop is safe.

## 9. As written (kernel state after this change)

Build scope: `MIXED_KV_PERSISTENT = ENABLE_MIXED_KV_CACHE && !SPEC_DEC`; the
non-mixed sm90 kernel and SPEC_DEC keep the `1 x n x (B*H)` grid,
`chooseNbSubSeq`, `MultiBlockSMem`.  Under the persistent build the mixed
multi-block epilogue is not compiled.

**Data flow.**

    seqLenList[B] --scan (IO warp 3)--> smem.sched {x0, x1, T, req0, head0, tile0, Lseq0, seqLen0, seqLen1}
    sched --> ItemCursor (registers) in IO warps 0 (K), 1 (V), 2 (Q), 3 (merge)
    K loader: cursor + page table + page_format --fillTileMeta--> smem.meta[K][chunk][16] (32 B records)
                                                                     --kMetaReady[chunk].arrive-->
    K converters: meta[K][g] {page, format, head} --cp.async--> packed rows / scales --expand--> k stage
                                                                     --kBar.produced.arrive-->
    gemm0: kBar.produced.wait -> meta[K][g].tile {validBeg, validEnd, first, last} ; q[j&1] ; X(g%2)
    Q warp: q[(req, head)] --regs--> smem.q[j&1]  (qBar[j&1])
    gemm1: vBar.produced.wait -> meta[V][g].tile ; on last: meta[V][g].{idxReq, idxHeadGrp}
           -> output[headGrpSize*(H*req + head)]  or  scratch chunk 2*blockIdx.x + ctaLast
           -> st.release.cta smem.finalizedItems = j + 1
    merge warp: finalizedItems > j -> atom.acq_rel.gpu.inc semaphores[H*req + head] (limit c1-c0)
           -> last arriver: __ldcg chunks {2c+1 | c in [c0,c1)} + {2c1 + (x_{c1+1} == Lend)} -> output

**Control flow per role** (g = CTA-local tile counter, G = x1 - x0, j = item
counter):

    gemm0   pre-arrive qBar[0..1].consumed, kBar[*].consumed
            for g: kBar[g%3].produced.arrive_and_wait ; tile = LDS meta[K][g].tile
                   if first: runningColMax = -inf ; qBar[j&1].produced.arrive_and_wait
                   QK HGMMA (Q from q[j&1]) ; wait ; kBar[g%3].consumed.arrive
                   mask if validBeg > 0 || validEnd < 64 ; colMax ; softmax ; X(g%2) ; xBar[g%2].produced.arrive
                   if last: qBar[j&1].consumed.arrive ; j++
    gemm1   pre-arrive vBar[*].consumed, xBar[*].consumed
            for g: vBar[g%3].produced.arrive_and_wait ; tile = LDS meta[V][g].tile
                   if first: acc = 0, accColMax = -inf, accColSum = 0
                   xBar[g%2].produced.arrive_and_wait ; rescale ; PV HGMMA ; commit ; wait
                   if last: publish colMax/colSum ; LDS idxReq/idxHeadGrp ; finalize -> scratch or output ;
                            thread 0: st.release.cta finalizedItems = j + 1 ; j++
                   xBar[g%2].consumed.arrive ; vBar[g%3].consumed.arrive
    K loader (IO 0)   cursor ; fillTileMeta(chunk 0) ; kMetaReady[0].arrive
            for g: [a16/mixed module: LDS meta[K][g] pages/formats/head]
                   kBar[g%3].consumed.arrive_and_wait
                   [a16/mixed module, elected: arrive_tx(nbA16 x 2 parts x 2 KB) + A16 boxes]
                   if (g+4) % 16 == 0 && g+4 < G: fillTileMeta(chunk ((g+4)/16)%2) ; kMetaReady[..].arrive
    V loader (IO 1)   mirror with vBar / meta[V] / vMetaReady (same code; operand selects addresses only)
    Q warp (IO 2)     cursor ; for each item j: load Q(req, head) ; qBar[j&1].consumed.arrive_and_wait ;
                      store q[j&1] ; fence.proxy.async ; qBar[j&1].produced.arrive
    merge warp (IO 3) scan (before __syncthreads) ; cursor ; for each item j:
                      if partial: poll finalizedItems > j (ld.acquire.cta, nanosleep 1 us) ;
                                  lane 0 atom.acq_rel.gpu.inc ; shfl old ; __syncwarp ;
                                  if last arriver: __ldcg chunks -> online combine -> (+ sinks) -> output
                      debug builds: wait finalizedItems == number of items, assert
    converters        unchanged loop over g < G ; issue(g) reads page/format/head from meta[op][g]
    end               __syncthreads ; destroy barriers ; return

Code shape rules applied: the operand-dependent objects of the loader (stage
barriers, metadata barriers, stage base, tensor map) are selected as shared /
param *addresses*, never as struct references; every record access is a
shared-window `ld/st.shared` at `metaBase[op] + (g % 32) * 32 + imm`; the
fill's per-entry state is two named register sets (entries lane/4 and
lane/4 + 8), not an indexed array; all page loops are `#pragma unroll` over
the four pages.  `KVTilePartLoader` keeps only its non-mixed members; the
mixed metadata helpers (`readMixedTileMeta`, `fillTileMeta(idxIterBeg, ...)`,
`publishPages`, `packedPartTxBytes`, `issuePackedPartLoad`,
`loadPackedScales*`) are replaced by `fillTileMeta(gBeg, cursor)`,
`issueCompressedPageCopies(record)` and the loader's inline A16 issue.

Trace additions: `MIXED_KV_TRACE_CTA`, `MIXED_KV_TRACE_TILE0` (8-tile window),
`TRACE ctarec <cta> start firstk last end tiles` per CTA (`%globaltimer`),
parsed by `benchmarks/parse_xqa_trace.py` into the per-CTA histogram (start
spread, fill, body, end spread, idle fraction); the `TRACE tile` parser splits
launches on a non-increasing tile index so a moved window works.

Host: `dimGrid = {P, 1, 1}` with `P = choosePersistentGridSize(SMs, ctasPerSm)`
(`XQA_PERSISTENT_CTAS` override) in both launchers; `ScratchMem{scratch,
2 * gridDim.x, 1}`; semaphores unchanged.  Conformance: `TAIL_CASES` now
carry the `XQA_PERSISTENT_CTAS` override — (50/100/130, default P: T < P,
empty CTAs, 1-3-tile items), (2200, P = 1), (2200, P = 3), (4096, P = 5),
(285, subnormal / maxscale) — 8 x 3 modes = 24 tail cases; the matrix is
32 + 2 + 24 + 2 = 60 cases.

Verification for this change is section 6 plus 8.10's period rule.  Not run in
this phase (review by reading first): `ptxas -v` (no C7507, 0 stack / spill,
IO group at 40 with the cursor), `USETMAXREG` = 2, `LDL`/`STL` = 0, the
60-case matrix, the locked bench, the same-launch trace with TILE0 = 0 / 11 /
27 and the per-CTA histogram.

## 10. Confirmation results (2026-09-04, nkcut2 H200, worktree r2p8 @ 9ce501fe)

Full numbers, tables and the attribution experiments are in
`mixed_kv_page_transport_backends.md`, section "Round 2, lever [8] —
persistent balanced CTA scheduling: confirmation".  Summary against sections
4-8 of this document:

- **Build checks (section 6 items 1-3)**: all met on the four modules — no
  C7507, 0 stack / spill, `USETMAXREG` = 2, `LDL` = `STL` = 0, one `ATOMG.INC`,
  no `ATOMS`, HGMMA 8 + 8, `UTMALDG` 8 (a16, mixed) / 0 (fp8, fp4), gemm0 /
  gemm1 barrier sites unchanged per role (gemm0 PHASECHK 8 / ARRIVE 11 /
  BAR.SYNC 1, gemm1 17 / 13 / 1; the +1 gemm0 arrive is the per-item
  `qBar.consumed`), 5 accepted `div_u64` call sites (68 straight-line SASS
  each).  `sizeof(SharedMem)` 113 664 B (8.9 predicted 113 664), 2 CTAs/SM by
  registers and by shared memory.  Section 4 fallbacks (smem cursor, IO at 48)
  unused.
- **Conformance**: 60 / 60 (the T < P cases exercise the review fix).
- **Timing** (medians, 5 x 5 locked): a16 78.8 (81.8), fp8 67.8 (76.9), fp4
  60.5 (70.7), mixed 64.4 (79.5); q=4 rows unchanged.  Gate (-12 %): fp4 and
  mixed pass, a16 parity, **fp8 misses by 0.1 us** (-11.8 %).  Targets not met.
- **Trace, periods (8.10)**: on the traced CTAs (0 and 1) every role period is
  at or below the baseline (fp8 1.2-1.35 us, fp4 1.2-1.25, tiles 3-7 / 11-18 /
  27-32); `kc_ready(16)`, `kc_ready(32)` at steady state (lead 4 holds, no
  fallback to 6-8); item boundary without first-K latency but +0.4 us on
  gemm0's K-wait -> mma segment on the item's first tile.
- **Trace, histogram (section 6)**: start spread 0.3 us (i); body 33 x T holds
  only for half the CTAs — **every SM has one CTA at 46.5 us and one at 57.1
  us (fp8; fp4 44.7 / 53.8) for identical work**, independent of the tile range
  (`MIXED_KV_TRACE_REVERSE_RANGES`) and of the SM (`smid`); a16 is unaffected.
  (ii) therefore fails for the slow member, (iii) end spread ~7 us > 5 us, (iv)
  `sm__cycles_active.avg/.max` 0.95-0.96 holds.  The section-2.1 fallback
  condition (per-CTA spread > 5 us -> dynamic tile-range pull) is triggered.
- **Fill**: 8.5 / 7.4 / 6.6 us (fp8 / fp4 / a16) against the 3.5 us assumed in
  section 5: a start burst (the first three tiles of all 264 CTAs, copy
  latency 3-4 us for tiles 1-2) plus a 4.3-5.4 us prologue to `kl_start(0)`
  (three dependent round trips; baseline CTA 0 needed 2.1 us for the same
  count) — to be stamped before any change.
- **Tail**: 2.6-3.0 us as modelled.  Co-tenant time-slice outliers (25-280 us
  on 1-8 CTAs per launch) are excluded from the histogram statistics and do not
  appear in the bench.

Wall model as measured (fp8): 8.5 fill + 57.1 slow-member body + 2.8 tail =
68.4 (bench 67.8).  Verdict: correct and structurally as designed; -9 to -15 us
on the compressed modes; accept gate met on fp4 / mixed / a16, missed by 0.1 us
on fp8; the lever's "unchanged periods" premise is broken by the co-resident
CTA pair asymmetry, which is the next item (dynamic pull: -6 us fp8 / -4.5 fp4;
removing the asymmetry: -10.6 / -9), followed by the fill (-4 to -5 us).
