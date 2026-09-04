# Round 5 — consumer wgmma-descriptor and ring-address arithmetic on the uniform datapath (design)

Kernel: `csrc/xqa/mha_sm90.cu` at `864479f6` (branch `claude/mixed-kv-sm90-tma`;
production q=1 kernel state = lever [8]; `git diff 530f5d4d 864479f6 -- csrc`
is empty, so the r4p7 SASS artefacts on nkcut2 are this source's SASS).
Bench shape B=17, S=4096, 8 KV heads, GQA 4, D=128, 64-token tiles, 16-token
pages, `HGMMA.64x8x16.F32.BF16` (ctaNbQHeads = 8, bf16 math element for a16 /
fp8 / fp4 / mixed alike).  Line references are into `mha_sm90.cu` of this
worktree (`wt/r5desc`).  **No kernel edit, no build of a modified kernel, no
GPU timing in this phase.**  Measurements used are readings of existing
artefacts: the lineinfo SASS of this source (`nkcut2:/tmp/r4p7_sass/f{1,2,-1}_li.nvdis`,
REG 48 / STACK 0 / 0 spill, no C7507, `f1.res`) and the production ncu
`SourceCounters` reports (`/tmp/r4mixed2_pcs_*.ncu-rep`, r2p8 checkout, line
numbers -17).

State (nkcut2 H200, locked 5x5 medians, q=1 us): transport_a16 79.4, fp8 67.9,
fp4 60.6, mixed 64.6.  Targets fp8 <= 58, fp4 <= 36, mixed <= 62.

## 0. Verdict in one paragraph

The two consumer groups rebuild their wgmma shared-memory descriptors from
scratch on every tile: **400 of the 668 uniform-datapath warp-instructions per
tile-SM are descriptor arithmetic (212 gemm0, 188 gemm1) and ~200 more are
ring / barrier addressing that re-derives the shared-window base with `S2UR
SR_CgaCtaId` and 64-bit generic pointer arithmetic per tile** (section 1;
static SASS count x 4 warps agrees with the ncu-attributed dynamic count to the
instruction).  Of the 400, only **112 are necessary**: the `HGMMA` operand
block is four consecutive uniform registers (`gdesc[UR16]` = A in UR16:17, B in
UR18:19) and both operands move every k-step, so each of the 16 wgmmas per tile
needs two `UIADD3`s (per warp), exactly CUTLASS's `DescriptorIterator` cost.
The other 288 are (i) the 64-bit descriptor *high word* and the swizzle
`baseOffset` rebuilt per tile from `makeMatDesc(nullptr, ...)` because ptxas
does not fold `cvta.to.shared` of a null pointer or of the extern smem symbol
(20 ops per warp per tile in gemm0, 13 in gemm1), (ii) a two-op chain per
operand per wgmma (`UIADD3` byte address, then `ULEA.HI` to fold `>> 4` into the
descriptor) where one `UIADD3` on a pre-shifted per-stage word suffices, and
(iii) `UMOV` shuffles into the UR16-19 block (gemm1: 36 per tile-SM).  The
design (section 3) keeps **one pre-shifted descriptor low word per ring** in a
uniform register, derived once per tile from a rolling stage cursor (`s in
{0,1,2}`, advanced with `UIADD3 / UISETP / USEL`, no division), adds
compile-time immediates per wgmma, and gives the barrier calls of the two loops
32-bit shared addresses from the same cursor.  Predicted (section 6): descriptor
arithmetic **400 -> 128** per tile-SM, ring / barrier addressing **~200 ->
~100**, uniform total **668 -> ~296**; SM total fp8 **4 150 -> ~3 780 (-9 %)**,
fp4 3 881 -> ~3 510, mixed 3 580 -> ~3 210.  Descriptor dependency depth before
the first QK wgmma 7 -> 2, between wgmmas 2 -> 1.  Wall by the issue-bound
model at the measured paired IPC: **fp8 62.8-65.3 (central 64.0), fp4
55.8-58.2 (57.0), mixed 59.1-61.9 (60.5), a16 unchanged**; fp8 <= 58 is not
reached by this lever alone (it is one of the three cuts round 4 [7] rev 2
section 9 named), mixed <= 62 is inside the band.  The change is bit-exact by
construction (same 16 `HGMMA`s, same 64-bit descriptor values, same barrier
protocol), adds no shared memory, holds <= 8 uniform registers per group and
no vector registers.  **Recommendation: go** (section 9), gated by SASS counts
before the single confirmation run (section 7).  The task's "<= 150 for the
668" is met for the descriptor arithmetic proper (400 -> 128) and not for the
whole uniform budget (296): the remaining ~170 are record / mask-bit / colMax
slot addressing whose floor this lever does not touch (section 6.1).

## 1. Before: the per-tile-SM count of the region, by class, with the method

### 1.1 Method

Two independent readings, which must agree (they do):

- **Static.** `nvdisasm --print-line-info` of this source's fp8 module
  (`/tmp/r4p7_sass/f1_li.nvdis`, built by `/tmp/r4p7_sass/build.sh` with the
  production flags of `/tmp/main_ptx/ninja_flags.py`, `-lineinfo -cubin`; its
  address-stripped SASS is identical to the production-flag build).  Script
  `nkcut2:/tmp/r5desc_static.py` (uses `benchmarks/microbench/xqa_sm90_converter_sass.py::parse_nvdis`)
  selects the instructions whose outermost `mha_sm90.cu` frame is in the gemm0
  loop (lines 1504-1699) or the gemm1 loop (1752-2109) and histograms the
  uniform-datapath opcodes (`ULEA UIADD3 ULOP3 USHF UMOV UIMAD UISETP USEL R2UR
  S2UR ULDC`) per source line.  The bodies are straight-line (both GEMMs are
  fully unrolled, 8 `HGMMA` each, no k-loop), so **per tile-SM = static count x
  4 warps** for every line executed once per tile.
- **Dynamic.** ncu `SourceCounters` "Instructions Executed" per PC of the
  production kernel (`/tmp/r4mixed2_pcs_fp8_P132.csv`, P = 132 so one tile-CTA
  = one tile-SM, 8 712 tiles), attributed through the r2p8 lineinfo listing
  (`/tmp/r2p8_ptx/li1.nvdis`, r2p8 line = this worktree's line - 17), capped at
  one execution per tile per warp as in `/tmp/r4p7_body.py`; script
  `nkcut2:/tmp/r5desc_dyn.py`.

### 1.2 Result (fp8; fp4 and mixed modules have the same consumer code: static uniform 96 / 97 per gemm0 warp)

Per warp static and per tile-SM (x4) static / dynamic, uniform-datapath ops
only.  Class of each line: **D** = wgmma descriptor arithmetic, **R** = ring /
barrier / record addressing, **O** = other uniform (mask bits, loop, colMax
slots).

| line (this worktree) | what | class | static / warp | per tile-SM static | per tile-SM dynamic | opcodes (dynamic, per tile-SM) |
|---|---|---|---:|---:|---:|---|
| gemm0 :1504 | loop counter / exit test | O | 2 | 8 | 8 | UIADD3 4, UISETP 4 |
| :1514 | `idxKStage = idxIter % 3` | R | 3 | 12 | 12 | UIMAD.WIDE 4, USHF 4, UIMAD 4 (magic-number modulo) |
| :1515 | `kBar[idxKStage].produced.arrive_and_wait` address | R | 8 | 32 | 32 | S2UR 8 (`SR_CgaCtaId`, `SR_SWINHI`), UIMAD.WIDE 4, ULEA 4, UMOV 16 |
| :1522-1525 | record LDS address, `tileFirst/Last`, `idxQBuf` | O/R | 6 | 24 | 24 | USHF 4, ULOP3 16, R2UR 4 |
| :1528 | `qBar[idxQBuf]` address (hoisted above `if (tileFirst)`) | R | 3 | 12 | 12 | UIMAD.WIDE 4, UMOV 8 |
| **:1572** | **Q descriptor `addAddr` (8 per tile)** | **D** | **16** | **64** | **64** | UIADD3 60, ULEA 4 |
| **:1575** | **K descriptor `addAddr` + base descriptor rebuild (8 per tile)** | **D** | **37** | **148** | **148** | ULEA 68 (ULEA.HI 56 + 12), ULOP3 40, USHF 24, USEL 8, UIMAD 4, UMOV 4 |
| :1595 | `kBar.consumed.arrive` address | R | 3 | 12 | 12 | ULEA 8, UMOV 4 |
| :1614-1615, :1640 | mask bounds from the record word, `idxIter & 1` | O | 4 | 16 | 16 | ULOP3 12, USHF 4 |
| :1639 | colMax slot address | O | 2 | 8 | 8 | ULEA 8 |
| :1663 | X buffer address for `storeGemm0AccToShm` (+ xBar.consumed) | R | 5 | 20 | 20 | UIADD3 8, UIMAD 8, UMOV 4 |
| :1676 | `xColMax / xColSum` store addresses | R | 2 | 8 | 8 | ULEA 8 |
| :1690 | `xBar.produced.arrive` address | R | 4 | 16 | 16 | UIADD3 4, ULEA 4, UMOV 8 |
| :1696-1697 | `tileLast` branch, `idxItem++` | O | 2 | 8 | 8 | UIMAD 4, UIADD3 4 |
| **gemm0 total** | | | **97** | **388** | **376** | (dynamic omits the 12 executed less than once per tile) |
| gemm1 :1752 | loop | O | 2 | 8 | 8 | |
| :1757-1758 | `idxVBuf = idxIter % 3`, `idxXBuf = idxIter & 1` | R | 4 | 16 | 16 | UIMAD 8, USHF 4, ULOP3 4 |
| :1761 | `vBar.produced.arrive_and_wait` address | R | 5 | 20 | 20 | S2UR 4, UIMAD.WIDE 4, UMOV 12 |
| :1770 | record LDS address | O | 2 | 8 | 8 | |
| :1792 | `xBar.produced.arrive_and_wait` address | R | 2 | 8 | 8 | UIMAD 4, UMOV 4 |
| :1846 | `xColMax[idxXBuf] / xColSum[idxXBuf]` slot addresses in the rescale | R | 5 | 20 | 20 | ULEA 20 (predicated `@!UP0`) |
| **:1871** | **X base `ULEA` + first `UIADD3` (attributed to the fence line)** | **D** | **2** | **8** | **8** | |
| **:1923** | **X descriptor `addAddr` (4 distinct per tile) + base descriptor rebuild** | **D** | **42** | **168** | **168** | UIADD3 36, ULEA 48, ULOP3 24, UMOV 36, USHF 16, USEL 4, UIMAD 4 |
| **:1933** | **V descriptor `addAddr` (8 per tile)** | **D** | **3** | **12** | **12** | UIADD3 8, ULEA 4 |
| :1992 | `tileLast` test | O | 1 | 4 | 4 | |
| :2112 | `xBar.consumed.arrive` address | R | 4 | 16 | 16 | UIADD3 4, ULEA 4, UMOV 8 |
| :2114 | `vBar.consumed.arrive` address | R | 2 | 8 | 8 | ULEA 8 |
| (:2011-2034 finalize) | per item, not per tile | - | 9 | - | 0 | |
| **gemm1 total (per tile)** | | | **71** | **284** | **292** | |
| **consumers** | | | **168** | **672** | **668** | |

By class (dynamic, per tile-SM): **D = 400** (gemm0 212, gemm1 188), **R ~ 200**
(gemm0 112: :1514/1515/1528/1595/1663/1676/1690; gemm1 88:
:1757/1758/1761/1792/1846/2112/2114), **O ~ 68**.  Total 668 = 16 % of the
4 150 warp-instructions the SM executes per tile (round 4 [7] rev 2, 2.2).

### 1.3 What the 400 descriptor instructions are (from the listing, gemm0 :42f0-:4730, gemm1 :2aa0-:2f50 of `f1_li.nvdis`)

Per tile, per warp, gemm0:

    UMOV   UR16, URZ                         ; cvta_to_shared(nullptr) — not folded by ptxas
    ULOP3  UR18 = UR16 & 0xffff ; USHF.R UR19 = UR18 >> 6 ; ULOP3 UR23 = UR19 & 0xe ; ULOP3 UP1 = UR16 & 0x3ff ; USEL UR23 = UP1 ? UR23 : 0
                                             ; makeMatDesc baseOffset of the *null* pattern address (always 0)
    ULOP3  UR17 = UR16 & 0x3fff0 ; USHF.R UR18 = UR17 >> 12 ; USHF.R UR22 = UR17 >> 4 ; UIMAD.WIDE UR18 = UR18 * 0x100 ; ULOP3 UR14 = UR18 | 0xff | UR22
                                             ; encode(0) split into addr / dimKOffset fields (low word of the base descriptor)
    ULOP3  UR14 = UR8 & 0xffff ; USHF.R UR14 >>= 6 ; ULOP3 UR14 &= 0xe ; ULOP3 UP0 = UR8 & 0x3ff ; USEL UR14 = UP0 ? UR14 : 0 ; USHF.L UR17 = UR14 << 16
    ULOP3  UR17 = UR17 | 0x40000040 | UR19   ; high word: baseOffset(&smem.k[0]) | SBO 1024 B | SWIZZLE_128B  — same for K and Q, = 0x40000040
    ULEA   UR13 = (idxKStage << 14) + UR8    ; K stage byte address (UR8 = shared window base, itself S2UR + ULEA per tile at :1515)
    ULEA   UR12 = (idxQBuf << 11) + UR8      ; Q buffer byte address
    then per wgmma (k, part):  UIADD3 UR16 = UR13 + imm ; ULEA.HI UR16 = UR14 + (UR16 >> 4)   ; K
                               UIADD3 UR18 = UR12 + imm ; ULEA.HI UR18 = UR14 + (UR18 >> 4)   ; Q
    HGMMA.64x8x16.F32.BF16 R24, gdesc[UR16], R24

i.e. 21 setup + 8 x 4 = 53 per warp (the listing shows 53).  gemm1 is the same
shape with the X operand in UR18:19 and V in UR16:17 (`tnspA`), plus 9 `UMOV`s
per warp because ptxas computes descriptors ahead into UR22-26 and moves them
into the UR16-19 block before each `HGMMA`.

Immediates seen (byte offsets, shared window): Q at `0x19000 + idxQBuf * 0x800`,
part `+0x400`, k-step `+0x20`; K at `idxKStage * 0x4000`, part `+0x2000`, k-step
`+0x20`; X at `0xc000 + idxXBuf * 0x800`, k-step `+0x20`; V at `0xd000 +
idxVBuf * 0x4000`, part (M instruction) `+0x2000`, k-step (16 rows) `+0x800`.
Every buffer the descriptors point at is `alignas(1024)` in `SharedMem`
(:240, :278) and the dynamic smem base is `CgaCtaId << 24 | 0x400` (listing
:4130 `ULEA UR8, UR10, 0x400, 0x18`), so `baseOffset` evaluates to 0 on every
tile; the `USEL` chain computes a constant.

## 2. Current data flow and control flow (as written)

gemm0 (`warpIdx.z == 0`, :1452-1700), per tile `idxIter`:

    idxKStage = idxIter % 3                                          :1514   (UIMAD.WIDE magic, 3 ops)
    kBar[idxKStage].produced.arrive_and_wait()                       :1515   (generic &kBar[s] -> 64-bit -> low word, S2UR x2 per tile)
    tileWord = ldsU32(tileRecordAddr(smem, 0, idxIter) + 20)         :1522   (record slot (g % 32) * 32)
    tileFirst / tileLast / idxQBuf = idxItem & 1                     :1523-1525
    if (tileFirst) { runningColMax reset; qBar[idxQBuf].produced.arrive_and_wait() }   :1526-1529
    for part in 0..1:                                                :1538
      kBuf = smem.k[idxKStage * 2 + part]                            :1541-1546
      matDescKBase = makeMatDesc(nullptr, 0, 1024, &smem.k[0], SW128).raw()   :1549   (INSIDE the loop -> rebuilt per tile)
      for k in 0..3:                                                 :1569
        matDescQ = addAddr(matDescQBase, &smem.q[idxQBuf][part](0, 2k))       :1571   (matDescQBase from :1499, outside the loop, but sunk)
        matDescK = addAddr(matDescKBase, &kBuf(0, 2k))                        :1575
        mma_async_shmA<bf16, 8>(acc, matDescK, matDescQ, accHasVal)           :1577   (SWAP_AB: A = K, B = Q)
      commit_group                                                   :1587
    wait_group<0>; kBar[idxKStage].consumed.arrive()                 :1594-1595
    ... softmax, X store into xBuf(idxIter % 2), xBar[idxIter % 2].produced.arrive   :1602-1690
    if (tileLast) { qBar[idxQBuf].consumed.arrive(); idxItem++ }     :1695-1698

gemm1 (`warpIdx.z == 1`, :1704-2120), per tile:

    idxVBuf = idxIter % 3 ; idxXBuf = idxIter % 2                     :1757-1758
    vBar[idxVBuf].produced.arrive_and_wait()                          :1761
    tileWord = ldsU32(tileRecordAddr(smem, 1, idxIter) + 20) ...      :1769-1780
    xBar[idxXBuf].produced.arrive_and_wait()                          :1792
    rescaleGemm1AccForNewColMax(xColMax[idxXBuf], xColSum[idxXBuf], ...)   :1846
    descXBase = makeMatDesc(nullptr, 0, 1024, SW128).raw()            :1855   (inside the loop)
    descVBase = makeMatDesc(nullptr, 0, 1024, SW128).raw()            :1860   (inside the loop)
    gmma::fence()                                                     :1871
    for k in 0..3:                                                    :1874
      descX = addAddr(descXBase, &xBuf[0](0, 2k))                     :1922
      for m in 0..1:                                                  :1929
        descV = addAddr(descVBase, &vBuf[m](16k, 0))                  :1932
        mma_async_shmA<bf16, 8, transA=true>(acc(m), descV, descX, true)   :1934
    commit_group ; wait_group<0>                                      :1956-1957
    ... item end (publish / finalize)                                 :1992-2110
    xBar[idxXBuf].consumed.arrive() ; vBar[idxVBuf].consumed.arrive()  :2112-2114

`addAddr` (`gmma.cuh` :65-71) is `lo += cvta_to_shared(ptr) >> 4`; `makeMatDesc`
(:73-101) computes `baseOffset` from `patternStartAddr % {1024,512,256}` at run
time.  `MBarrier::addr()` (`barriers.cuh`) is `cvta_to_shared(this)` from a
generic `this` pointer, so every barrier call site pays the generic-address
formation (`UIMAD.WIDE` from the 64-bit smem base held in UR6:7, which ptxas
re-derives from `S2UR SR_CgaCtaId / SR_SWINHI` each tile at :1515 / :1761).

## 3. New data flow and control flow

### 3.1 Objects (all warp-uniform, all in uniform registers; nothing in shared memory changes)

Prologue of each consumer group, once per CTA (after `setmaxnreg`, before the
tile loop):

    smemBase = cvta_to_shared(&smem)                    ; 32-bit; made opaque with asm volatile("" : "+r") so ptxas cannot
                                                        ;   rematerialise it from S2UR per tile (today's :1515 pattern); at most 1 R2UR per tile if it lands in R
    assert(smemBase % 1024 == 0)                        ; debug-build trap; the SW128 pattern of the TMA a16 path already requires it (D1)
    descHi   = 0x40000040u                              ; constexpr: SBO = 1024 B >> 4 = 0x40, LBO 0, baseOffset 0, SWIZZLE_128B = 1 << 30 of the high word
                                                        ;   static_assert(makeMatDescConst(...) == descHi) evaluated at compile time from the Array2D types
    kDescLo0 = (smemBase + offsetof(SharedMem, k))      >> 4     ; stage 0, part 0, k 0
    qDescLo0 = (smemBase + offsetof(SharedMem, q))      >> 4
    vDescLo0 = (smemBase + offsetof(SharedMem, vBufs))  >> 4     (gemm1)
    xDescLo0 = (smemBase + offsetof(SharedMem, reusedXVOutSwizzleBuf)) >> 4   (gemm1)
    kBarBase = smemBase + offsetof(SharedMem, kBar)     ; vBarBase, xBarBase, qBarBase likewise (gemm0: kBar, xBar, qBar; gemm1: vBar, xBar)
    s = 0                                               ; ring stage cursor for K (gemm0) / V (gemm1); nbKBuf == nbVBuf == 3
    x = 0                                               ; X slot cursor (nbXBuf == 2; today's `idxIter & 1` is already one op — kept as is)

Per-tile derived words (gemm0):

    kDescLo = kDescLo0 + (s << 10)                      ; 1 ULEA   (16 KB stage stride >> 4 = 0x400)
    qDescLo = qDescLo0 + (idxQBuf << 7)                 ; 1 ULEA   (2 KB Q buffer stride >> 4 = 0x80)
    kBarAddr = kBarBase + (s << 4)                      ; 1 ULEA   (CtaBarrierPair is 16 B: produced at +0, consumed at +8)

Per wgmma (k, part), gemm0 — compile-time immediates, unrolled as today:

    descK = (uint64(descHi) << 32) | (kDescLo + ((part * 0x2000 + k * 0x20) >> 4))   ; 1 UIADD3 into UR16
    descQ = (uint64(descHi) << 32) | (qDescLo + ((part * 0x400  + k * 0x20) >> 4))   ; 1 UIADD3 into UR18
    HGMMA gdesc[UR16]                                   ; UR17 = UR19 = descHi stay resident across the tile (as they do today)

gemm1 per tile: `vDescLo = vDescLo0 + (s << 10)`, `xDescLo = xDescLo0 + (x << 7)`,
`vBarAddr = vBarBase + (s << 4)`, `xBarAddr = xBarBase + (x << 4)`; per wgmma
(k, m): `descV = vDescLo + ((m * 0x2000 + k * 0x800) >> 4)` (1 UIADD3), `descX =
xDescLo + ((k * 0x20) >> 4)` (1 UIADD3 per k, reused across m = 0, 1: the B
block UR18:19 is untouched between the two M instructions of a k-step).

Cursor advance, at the end of the tile body (off the critical path; the next
tile's first use is after its barrier wait):

    s = (s == 2) ? 0 : s + 1                            ; UIADD3, UISETP, USEL — 3 ops, replaces the 3-op magic modulo of :1514 / :1757
    x ^= 1                                              ; 1 op (today's ULOP3)

### 3.2 Barrier calls with 32-bit shared addresses

`barriers.cuh` gains static overloads of the four operations the two loops use,
taking a `uint32_t` shared-window address instead of `this`:
`MBarrier::arrive(addr)`, `MBarrier::arrive_and_wait(addr)` (arrive + the
existing `try_wait` poll on the returned token), `MBarrier::wait_parity(addr,
parity)` (not needed by the consumers, listed for symmetry), and
`CtaBarrierPair` helpers `producedAddr(pairAddr) = pairAddr`,
`consumedAddr(pairAddr) = pairAddr + 8`.  The PTX is the existing strings with
the `"l"(addr())` operand replaced by `"r"(addr)` and the `.shared::cta`
state-space qualifier (`mbarrier.arrive.release.cta.shared::cta.b64`,
`mbarrier.try_wait.acquire.cta.shared::cta.b64`), which is the form SASS already
emits (`SYNCS.ARRIVE.TRANS64.A1T0 R6, [UR12+0x1baf0]`).  The existing
`this`-based methods stay for every other caller; the converters, loaders and
Q / merge warps are untouched.

### 3.3 Control flow, gemm0 (only the marked lines change; protocol identical)

    prologue: objects of 3.1 ; phase-0 free arrivals as today (:1460-1465)
    for idxIter in 0..nbCtaTiles:
      gmma::fence()
      arrive_and_wait(kBarAddr)                                         [was kBar[idxKStage].produced.arrive_and_wait]
      tileWord = ldsU32(tileRecordAddr(smem, 0, idxIter) + 20)          unchanged (record cursor rejected: see 6.1)
      tileFirst / tileLast / idxQBuf                                    unchanged
      if (tileFirst) { reset; arrive_and_wait(qBarBase + idxQBuf * 16) } [address form only]
      kDescLo, qDescLo                                                  [2 ULEA]
      8 x { UIADD3 UR16 ; UIADD3 UR18 ; HGMMA }  in the same (part, k) order, same accHasVal, same commit per part
      wait_group<0> ; arrive(kBarAddr + 8)                              [was kBar[idxKStage].consumed.arrive]
      softmax / mask / colMax / X store / xBar as today (X-side addresses may take xBarBase + (x << 4); see 6.1 part B)
      if (tileLast) { arrive(qBarBase + idxQBuf * 16 + 8); idxItem++ }
      s = (s == 2) ? 0 : s + 1

gemm1 correspondingly: `arrive_and_wait(vBarAddr)`, record read, `arrive_and_wait(xBarAddr)`,
rescale, `vDescLo / xDescLo`, `gmma::fence`, 8 x { (UIADD3 UR18 per k) ; UIADD3
UR16 ; HGMMA tnspA }, commit, `wait_group<0>`, item end unchanged,
`arrive(xBarAddr + 8)`, `arrive(vBarAddr + 8)`, cursor advance.

### 3.4 Why not "select among three precomputed descriptors"

A `Raw desc[3]` indexed by `s` at run time is a register array indexed by a
runtime value (C2 of the transport design): ptxas would either place it in local
memory (STACK > 0, forbidden) or expand it into `USEL` chains (2 per select, 4
per tile for K, Q-independent) — no better than the one `ULEA` from the cursor,
and it needs three URs per ring instead of one.  The `ULEA` form is chosen.

### 3.5 Why not `%3` by the existing magic multiply

It is three uniform ops either way (`UIMAD.WIDE + USHF + UIMAD` vs `UIADD3 +
UISETP + USEL`); the cursor form is chosen because it is loop-carried and
therefore *available at the top of the tile* — the modulo today sits at the
head of the dependency chain of the K wait (:1514 -> :1515), the cursor moves
that work to the previous tile's tail.

## 4. Invariants affected, and bit-exactness

- **Bit-exactness of the descriptors (the whole correctness argument).**
  Today `lo = baseLo + ((stageAddr + imm) >> 4)` with `baseLo = encode(0) = 0`
  in the address field and the dimK/dimMN fields as constants; new `lo =
  ((smemBase + off_ring) >> 4) + (s << 10) + (imm >> 4)`.  Every term is a
  multiple of 16 bytes before shifting (`alignas(1024)` buffers, immediates
  multiples of 0x20), so `(a + b) >> 4 == (a >> 4) + (b >> 4)` exactly; the
  14-bit address field cannot carry into bit 14 because the largest shared
  address is < 228 KB (`< 2^18`, `>> 4 < 2^14`), the same bound `encode`'s
  mask `0x3FFFF` relies on today.  The high word is `0x40000040` today on every
  tile (1.3) and is the constant `descHi`; a `static_assert` builds it from the
  same `Array2D` types `getSwizzleMode<true>` sees, and the prologue asserts
  the 1024-B alignment that makes `baseOffset` 0.  The 16 `HGMMA`s per tile,
  their order, operands, `accHasVal`, `commit_group` / `wait_group` placement
  and accumulator registers are unchanged, so every accumulator value is
  bit-identical and so is every output.  The verification therefore demands
  **bit-identical outputs** against the production build, not tolerance
  (7.5).
- **D1 (landing layout / 1024-B alignment).**  Already required by the TMA
  `SWIZZLE_128B` a16 path into `k[] / vBufs[]`; the design makes it an explicit
  prologue check.  D2-D6 (packed rows, ownership, tail, proxy order, copy
  ownership): untouched — converters and loaders are not edited.
- **C2 (no runtime-indexed register arrays).**  Respected by construction
  (3.4).
- **C3 (register budgets from live sets).**  Section 5.
- **C4 (barrier accounting).**  Unchanged counts, unchanged arrive / wait
  sites, unchanged phases: only the address operand form changes.  The
  arrival-token `try_wait` poll is the existing one.
- **C10 (record visibility: record read after the K / V wait, before the
  consumed arrive).**  Order of the record `LDS` relative to the barrier
  operations is unchanged.
- **Lever [8] item protocol (qBar phases, `idxItem`, `tileFirst / tileLast`).**
  Unchanged; `idxQBuf` still drives the Q descriptor and the qBar address.
- **Persistent ring depth constants.**  `s` wraps at `nbKBuf - 1`
  (`static_assert(nbKBuf == nbVBuf == 3)` already at :2149); `x` at
  `nbXBuf == 2` (`static_assert`).  A future change of the ring depth changes
  one constant per cursor.
- **Trace build (`MIXED_KV_TRACE`).**  `TRACE_STAMP` sites unchanged.

## 5. Register and shared-memory budgets (ptxas C7507 rule)

- **Shared memory: unchanged** (`sizeof(SharedMem)` identical; no new
  fields).  Gate: `cuobjdump -res-usage` SHARED and the occupancy calculator
  output identical to today's (`f1.res`: `REG:48 STACK:0 SHARED:1024`).
- **Vector registers.**  Every new object is warp-uniform and consumed only by
  uniform instructions or as an `HGMMA` descriptor; ptxas allocates them to
  URs.  The gemm groups run under `setmaxnreg.dec 40` (:1446) with a live set
  (acc 4-8, colMax / softmax state, lane constants) that this design does not
  touch; if ptxas keeps `smemBase` in an R register because of the opaque asm,
  that is one register (today's listing has ~20 R live in the gemm0 loop).
  C7507 fires only if a role's need exceeds its `.dec` value; nothing here
  raises the need.  Gate: `ptxas -v` prints no C7507, `0 bytes spill`, REG 48;
  `cuobjdump -sass` shows the two `USETMAXREG` (DEALLOC 0x28 / TRY_ALLOC 0x38)
  exactly as today.
- **Uniform registers.**  Per group <= 8 loop-invariant URs (`kDescLo0 /
  qDescLo0` or `vDescLo0 / xDescLo0`, `kBarBase`, `qBarBase` / `xBarBase`,
  `descHi`, `s`, plus the existing `idxIter / idxItem / nbCtaTiles`) against 63
  available; the loop today peaks at ~UR26.  ptxas may still choose to
  rematerialise (it does so today for the window base); the SASS gate (7.1)
  is the arbiter, and the fallback for a rematerialised base is the opaque-asm
  form already specified in 3.1.

## 6. Predicted counts, dependency depth, and wall

### 6.1 Counts after (per tile-SM = 4 warps), fp8; the same for fp4 / mixed (identical consumer code)

| region | before (1.2) | after | how |
|---|---:|---:|---|
| gemm0 descriptors (:1572, :1575) | 212 | **72** | 8 x 2 `UIADD3` = 16 / warp + `kDescLo`, `qDescLo` 2 / warp = 18 x 4 |
| gemm1 descriptors (:1871, :1923, :1933) | 188 | **56** | 8 V + 4 X `UIADD3` = 12 / warp + `vDescLo`, `xDescLo` 2 / warp = 14 x 4; the 36 `UMOV`s vanish because each low word is written into its UR16-19 slot directly |
| **descriptor class D** | **400** | **128** | floor 112 (two moving operands per wgmma: same as CUTLASS sm90 mainloops, one `UIADD3` per operand per k-block) + 16 per-tile derivations |
| K ring (:1514, :1515, :1595) | 56 | **16** | cursor advance 3 + `kBarAddr` 1 per warp; `S2UR`, `UIMAD.WIDE`, `UMOV` chains gone |
| V ring (:1757, :1761, :2114) | 40 | **16** | same |
| qBar (:1528) | 12 | **4** | `qBarBase + idxQBuf * 16` |
| X ring, gemm1 (:1758, :1792, :2112, :1846) | 60 | **~28** | `x` 1, `xBarAddr` 1, rescale slot addresses 2-3 (`xColMax / xColSum` bases + `x << 5`), consumed arrive 0 (reuses `xBarAddr`) |
| X ring, gemm0 (:1663, :1676, :1690) — *part B, optional* | 44 | **~28** | `xBufAddr = xBase + (x << 11)`, `xBarAddr`; `storeGemm0AccToShm` takes the 32-bit address (its `STSM` addresses are vector ops and not counted here) |
| **ring / barrier class R** | **~200** | **~92** | |
| other O (loop, record address and bits, mask bounds, colMax slot, tileLast) | ~68 | **~68** | not touched; record slot `(g % 32) * 32` is already 2 ops |
| **uniform total, consumers** | **668** | **~288-296** | -372 to -380 |
| consumer body (all classes) | 1 462 | **~1 085** | |
| **SM total per tile** | fp8 4 150 / fp4 3 881 / mixed 3 580 | **~3 775 / ~3 505 / ~3 205** | -9.0 / -9.7 / -10.5 % |

The task's target "<= 150" is met for the descriptor arithmetic (400 -> 128)
and is **not reachable for the whole uniform budget**: 112 (operand words) +
~50 (cursor advances, five barrier addresses, two slot addresses) + ~68
(record / mask / colMax / loop) = ~230 is the floor of the uniform datapath at
this loop structure, and the record / mask-bit work is not descriptor
arithmetic.  The design predicts ~290 and states the floor so the gate in 7.1
is a real number, not the task's.

### 6.2 Dependency depth (uniform ops on the path to the wgmma issue)

| chain | before | after |
|---|---|---|
| tile top -> first QK `HGMMA` (gemm0) | `UIMAD.WIDE -> USHF -> UIMAD` (modulo) -> `ULEA` (stage addr) feeding the K wait, then high word `UMOV -> ULOP3 -> USHF -> ULOP3 -> USEL -> USHF -> ULOP3` (**depth 7**, 21 ops) and low word `S2UR -> ULEA -> ULEA -> UIADD3 -> ULEA.HI` (depth 5) | `s` (loop-carried) `-> ULEA kDescLo -> UIADD3 UR16` (**depth 2**, 4 ops incl. Q) |
| between consecutive `HGMMA`s | `UIADD3 -> ULEA.HI` per operand (**depth 2**, 4 ops) | `UIADD3` per operand (**depth 1**, 2 ops) |
| tile top -> first PV `HGMMA` (gemm1) | same shape, depth 7 (13 setup ops) + 3 `UMOV` | depth 2 |
| tile top -> K / V wait issue | modulo (3) + generic address (`S2UR x2 -> ULEA -> UMOV -> UIMAD.WIDE -> UMOV`), depth 6 | `ULEA kBarAddr` from the loop-carried `s`, depth 1 |

Issue slots before the first wgmma of a tile drop from ~30 to ~5 per warp in
gemm0.  On the gemm0 chain (P0.3 floor 1.00 us / tile lone) this is worth
25-30 dependent-or-serial issue slots x the per-warp slot interval (2.5-4
cycles at 10 warps per scheduler and IPC 2.48) = **60-120 cycles = 0.03-0.06
us per tile**, i.e. the chain floor moves 1.00 -> ~0.95 us.  The 8 QK wgmmas
themselves (~0.37 us lone, tensor-pipe / smem-read bound) are unaffected: the
uniform ops between them were already hidden behind wgmma execution, so their
removal is an issue-slot effect, not a latency effect.  **The chain effect
alone is worth ~2-4 us of wall; the count effect (6.3) is the larger claim and
rests on the issue-bound reading.**

### 6.3 Wall by the issue-bound model

    cadence' = cadence x N' / N ;   wall = fill 8.5 + 66 x cadence' + tail (2.8 / 2.6 / 2.8)
    N, cadence from round 4 [7] rev 2 2.4: fp8 4 150 / 0.858, fp4 3 881 / 0.75, mixed 3 580 / 0.81 (paired, 1.98 GHz); check: today 67.9 / 60.6 / 64.8 reproduced

| mode | today | N -> N' | cadence -> | **optimistic** (all freed slots convert) | **central** (75 %) | **pessimistic** (50 %, = chain-only effect) | target |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp8 | 67.9 | 4 150 -> 3 775 | 0.858 -> 0.780 | **62.8** | **64.0** | **65.3** | <= 58 (not reached) |
| fp4 | 60.6 | 3 881 -> 3 505 | 0.750 -> 0.677 | **55.8** | **57.0** | **58.2** | <= 36 (plan gate; not reached) |
| mixed | 64.6 | 3 580 -> 3 205 | 0.810 -> 0.725 | **59.1** | **60.5** | **61.9** | <= 62 (inside the band) |
| a16 | 79.4 | DRAM-bound | - | 79-80 | 79-80 | 79-80 | parity |

Assumptions, and where the argument fails:

1. **The SM is issue-slot-bound at 2 CTAs/SM and the uniform-datapath
   instructions consume those slots.**  Evidence: pair doc 1.3-1.4 — 63 %
   issue-active, 1.58 eligible warps per scheduler, every *work* segment of the
   slot-20..39 CTA 16-28 % longer with memory segments equal; uniform
   instructions are counted in `inst_executed` and are issued by the same
   scheduler as vector instructions (they are not free co-issues).  *Fails if*
   the stretch is MIO / LSU-queue arbitration rather than issue arbitration
   (the pair doc could not separate the two): uniform ALU ops do not touch the
   MIO queue, so removing them would free nothing for the converters' LDS /
   STS and the gain collapses to the chain effect (pessimistic column, ~-2.6
   us fp8).  Discriminator after the run: `smsp__issue_active` should fall by
   ~6-9 points at unchanged `l1tex` wavefronts if the slots were the bound;
   if issue-active is unchanged and the wall is too, the MIO reading wins.
2. **IPC constant under the cut.**  Track S step 6 (q=4 kernel) is the
   counter-example: -16 % instructions, issue-active 52 -> 43 %, short
   scoreboard up, wall flat — because the *removed* instructions were on the
   converter warps whose own chains then stalled on scoreboards.  Here the
   removed instructions are on the *consumer* warps, whose chains are
   dominated by wgmma / barrier waits (not scoreboards), and the beneficiaries
   are the converter warps that do not change; the failure mode of step 6 does
   not map one-to-one, but the 50 % column is kept as the floor for it.
3. **Fill and tail unchanged** (8.5 / 2.6-2.8 us): the prologue adds ~40
   uniform ops once; the fill is DRAM / first-K bound.
4. **Mixed's count sensitivity is lower**: its IPC is fetch-limited (mixed
   doc: `no_instruction` +55 %), so the central estimate for mixed should be
   read at the pessimistic end (61.9) until measured.

Predicted for the record: **fp8 64.0 (62.8-65.3), fp4 57.0 (55.8-58.2), mixed
60.5 (59.1-61.9), a16 ~79.5**.

## 7. Verification artefacts with accept / reject numbers (SASS gates before any timing)

Remote checkout `nkcut2:/home/bigboi/dash-flashinfer-claude-r5desc` (rsync of
this worktree, symlinks once, `touch` the `.cu/.cuh` after every rsync),
workspace `/tmp/mixedkv-r5desc` (rm -rf before builds).

### 7.1 Compile-only lineinfo build (no GPU), `/tmp/r5desc_sass/build.sh` = `/tmp/r4p7_sass/build.sh` with the checkout path substituted

| check | accept | reject |
|---|---|---|
| `ptxas -v` (fp8 / fp4 / mixed) | no C7507; `0 bytes stack frame, 0 bytes spill`; `Used 48 registers` | any C7507, any spill, REG != 48 |
| `cuobjdump -sass`: `USETMAXREG` | exactly 2 (DEALLOC 0x28, TRY_ALLOC 0x38) | otherwise |
| `cuobjdump -res-usage` | `REG:48 STACK:0 SHARED:1024`; occupancy 2 CTAs/SM | otherwise |
| `HGMMA` count and forms | 16, `64x8x16.F32.BF16`, 8 plain + 8 `.tnspA`, same `gsb0` / `!UPT` placement as today | otherwise: the wgmma stream changed |
| `/tmp/r5desc_static.py` gemm0 uniform per warp (lines 1504-1699 -> new range) | **<= 50** (today 97); descriptor lines **<= 20** (today 53) | > 60: rematerialisation or the high word rebuilt in the loop — do not run |
| same, gemm1 per tile | **<= 32** (today 71); descriptor lines **<= 16** (today 47) | > 40 |
| `S2UR` inside either loop body | 0 | >= 1: the window base is rematerialised — apply the opaque-asm fallback (3.1) and rebuild |
| `UIMAD.WIDE` inside either loop body | 0 (was 2 + 2 modulo / barrier) | >= 1 |
| `UMOV` between two `HGMMA`s | 0 | >= 1 per wgmma: ptxas is shuffling into UR16-19; investigate before running |
| SASS diff with all uniform-datapath instructions and `SYNCS` address operands stripped | consumer bodies identical to today's (same vector instruction sequence, `STSM`, `F2FP`, `FMUL`, `SHFL`, `BAR.SYNC`) | anything else changed in the consumers, or any change in converter / IO / Q / merge bodies |
| `sizeof(SharedMem)`, barrier init counts | identical | otherwise |

### 7.2 Conformance (before timing)

`tests/attention/run_xqa_mixed_page_transport.py`, the 72-case matrix, and
**bit-identity** of the outputs against the production build's outputs for
every case (dump / compare; tolerance is not the test here — 4.).  Any
mismatch = reject.

### 7.3 One confirmation run

`flock /tmp/mixedkv-gpu0.lock bash /home/bigboi/mixedkv_remote_run.sh
/home/bigboi/dash-flashinfer-claude-r5desc r5desc sm90 transport_a16 fp8 fp4
mixed` (72-case matrix + locked q=1 / q=4 bench + res-usage + ncu occupancy;
repeats x kernel < 1.5 ms per the co-tenant rule; min / median / max).

| mode | accept (median) | reject |
|---|---:|---:|
| fp8 | <= 65.5 | > 67.0 or > today + 0.5 |
| fp4 | <= 58.5 | > 60.0 |
| mixed | <= 62.0 | > 64.0 |
| a16 | 78.5-80.5 | outside |
| q=4 modules (`mha.cu`, untouched) | today +- 1 % | otherwise: the barrier header change leaked |

### 7.4 ncu (after the timing, `--clock-control none`, `--launch-skip 4 --launch-count 1`, fp8 P = 264; clock stated in the table)

`smsp__inst_executed.sum` **<= 33.3 M** (today 36.16 M at P 132 / 36.7 M at P
264; predicted 32.9-33.2 M); `smsp__issue_active` down 5-9 points at unchanged
`l1tex__data_pipe_lsu_wavefronts_mem_shared` — the discriminator of 6.3 (1);
`launch__registers_per_thread` 48, occupancy 2.  `SourceCounters` re-run with
`/tmp/r5desc_dyn.py`: consumer uniform **<= 320** per tile-SM.

### 7.5 Trace (`MIXED_KV_TRACE=1` copy, `benchmarks/xqa_mixed_trace_once.py`, `parse_xqa_ctarec_roles.py`)

gemm0 `stamp 0 -> stamp 1` (K wait -> QK done) shorter by 0.03-0.06 us lone;
paired fast / slow cadence both shorter; the slow / fast ratio unchanged
(1.20-1.23) — the lever does not touch the slot asymmetry.

## 8. Do-not-build-if

1. The compile-only SASS misses any 7.1 accept value (uniform counts, `S2UR`
   / `UIMAD.WIDE` / `UMOV` in the loop, C7507, REG, `HGMMA` stream).
2. The uniform-stripped SASS diff shows any change outside the consumer
   descriptor / barrier-address lines, or any change in the converter, IO, Q
   or merge warps' bodies (the barrier overloads must be additive).
3. `sizeof(SharedMem)` or any barrier init count changes.
4. Bit-identity in 7.2 fails on any case.
5. A fresh ncu of the *production* kernel at P = 264 shows `smsp__issue_active`
   < 50 % or eligible warps < 1.2 per scheduler (the SM is no longer
   issue-bound, e.g. after another lever landed): re-derive 6.3 before
   running.
6. The design is proposed with a runtime-indexed descriptor array, a `%3`
   in the loop, or a `makeMatDesc` call inside the loop (3.4, 3.5).

## 9. Go / no-go

**Go**, as a build gated by 7.1-7.2 and confirmed by one run (7.3), for these
reasons:

1. It is the only consumer-side lever that needs no protocol change (round 4
   [7] rev 2, section 9): the barrier arrive / wait sites, phases, counts and
   the wgmma stream are unchanged; the output is bit-identical by
   construction, so the correctness gate is equality, not tolerance.
2. The count is measured, not estimated: 400 descriptor + ~200 ring uniform
   ops per tile-SM from two agreeing readings (static x 4 = dynamic to the
   instruction), and the after-count has a stated floor (112 + ~50 + ~68) with
   the design 60 above it.
3. It acts on the bottleneck the round-3 / round-4 evidence names (issue
   arbitration at 2 CTAs/SM), removing 9-10 % of the SM's per-tile
   instructions and shortening the consumer chains' descriptor depth 7 -> 2,
   with the failure mode of that argument (MIO vs issue) stated and given a
   post-run discriminator (7.4).
4. Cost: ~60 lines in the two consumer loops, four static overloads in
   `barriers.cuh`, two `static_assert`s and one prologue alignment check; no
   new shared memory, no vector registers, no new invariants beyond D1 made
   explicit.

What it does not do: fp8 <= 58 needs the converter copy-issue (~720) and
expansion (~1 504) cuts as well (round 4 [7] rev 2, section 9); this lever's
central fp8 is 64.0.  mixed <= 62 is inside the band and would be the first
target met on the sm90 q=1 kernel if the optimistic-to-central reading holds.

Predicted for the record: **fp8 64.0 (62.8-65.3), fp4 57.0 (55.8-58.2), mixed
60.5 (59.1-61.9), a16 ~79.5 us**; consumer uniform ops 668 -> ~290 per tile-SM,
descriptor arithmetic 400 -> 128.  Not built in this phase.

## Appendix: artefacts of this design

- Static per-line uniform counts: `nkcut2:/tmp/r5desc_static.py` on
  `/tmp/r4p7_sass/f{1,2,-1}_li.nvdis` (fp8: gemm0 97 / gemm1 80 uniform per
  warp; fp4 96 / mixed 97 for gemm0).
- Dynamic per-line uniform counts: `nkcut2:/tmp/r5desc_dyn.py` on
  `/tmp/r2p8_ptx/li1.nvdis` + `/tmp/r4mixed2_pcs_fp8_P132.csv` (gemm0 376,
  gemm1 292 per tile-CTA, matching round 4 [7] rev 2 2.2).
- Listing excerpts read: `f1_li.nvdis` gemm0 `:40c0-:48b0` (loop head, K wait,
  descriptors, 8 `HGMMA`), gemm1 `:28c0-:2f60` and `:3e90-:3f50` (rescale,
  descriptors, 8 `HGMMA`, consumed arrives).
- No remote checkout created, no workspace built, no GPU job started.
