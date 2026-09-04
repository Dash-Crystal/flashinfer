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

    c0 = c(L_s),  c1 = c(L_s + tiles(s) - 1),  nbPartials(s) = c1 - c0 + 1

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
   = `STL` = 0; `UTMALDG` count unchanged per module (A16 K/V boxes in the a16
   and mixed modules; 0 in the fp8/fp4 static modules, `mixedLoaderTma` false);
   `LDGSTS` > 0; exactly one `ATOM...INC` (the
   semaphore) and no `ATOMS`; gemm0/gemm1 SYNCS.PHASECHK / ARRIVE / BAR.SYNC
   counts unchanged from the round-2 baseline (gemm0 1 / 3 / 12, gemm1 1 / 7 /
   11; HGMMA 8 + 8) — the item loop adds no barrier sites to the GEMM roles.
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
8. `ctaNbValidQHeads > 8` or `isHeadPadded` builds are requested on this path:
   the register merge (16 elements per lane) is sized for headGrpSize <= 8 x
   D=128; other configurations keep the existing grid (compile-time
   `MIXED_KV_PERSISTENT` off, `static_assert`).
9. The semaphore region is not zero at first use in the target deployment —
   the same requirement the current multi-block path has; not new, but with
   P-indexed scratch a stale semaphore now corrupts a different sequence's
   merge count, so confirm the allocation path zeroes it.

Scope: `ENABLE_MIXED_KV_CACHE && !SPEC_DEC` only; SPEC_DEC (Track S) and the
non-mixed sm90 kernel keep `chooseNbSubSeq` and the `1 x n x (B*H)` grid.
