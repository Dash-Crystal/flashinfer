# Round 5 — converter copy-issue chain: ~950 -> ~320 warp-instructions per tile-SM (copy function proper ~700 -> ~240)

Worktree `wt/r5copy` at `864479f6` (production sm90 q=1 kernel = lever [8]).
Design only: every number below comes from reading existing artefacts (the
r4p7 lineinfo listing of this exact source, the production ncu
`SourceCounters` reports of the r2p8 checkout, the trace tables of earlier
rounds).  No GPU job was run for this document.

Scope: the converter warps' **copy-issue path** of `csrc/xqa/mha_sm90.cu` —
`issueKCopies` / `issueVCopies` (:2661-2673, :2728-2737: parity math, the two
waits, the call) and `issueCompressedPageCopies` (:3137-3212: record reads,
tag decode, 64-bit source arithmetic, five `cp.async`, scale copy) plus the
tag rotation and `commitGroup` (:2701-2708, :2755-2759).  The expansion
(`expandPackedStage`, :3466-3629) is untouched except for the E2M1 landing
read (section 3.4: one `LOP3` more per lane-tile).  The consumer and IO
roles are untouched.

## 0. Verdict in one paragraph

The copy-issue path executes **~119 (K) / ~126 (V) warp-instructions per
warp per tile** in the fp8 module (section 1: 476 + 506 per tile-CTA of 4 + 4
warps, ncu `inst_executed` at P = 132 = per tile-SM), of which the copy
function proper is ~86-90 per warp (the task's "~720 per tile-SM"); the
rest is parity arithmetic, two `try_wait`s and the commit.  The stream is
five `LDGSTS` wrapped in 11 `IMAD.WIDE`, 3 `IADD3.X`, 44 32-bit
address/move ops, 28 uniform-datapath ops, 14 predication/branch ops, 6
constant-bank loads, 3 `LDS`, and 6 ptxas filler `@!PT LDS RZ,[RZ]`; its
dependency depth from the tag `LDS.U8` to the last `LDGSTS` is ~14
dependent instructions across three latency hops (`LDS` -> `R2UR` -> uniform
compare -> `BRA` -> `ULDC` -> `LDS` -> five chained `IMAD.WIDE`/`IADD3.X`),
which the production PC samples confirm (`R2UR` and the first `IMAD.WIDE`
are the top `short_scoreboard` PCs of the path, section 1.4).  The F25-style
form (section 3) — lane-constant 64-bit bases held in registers, the page
and head coordinates from **two `LDS` per warp per tile** (`LDS.32` page,
`LDS.128` record tail), **one `IMAD.WIDE` per copy** (the row copies fan
out from one anchor), destinations `[R + imm]` from two lane bases plus one
stage add, the format as data (`SEL` on base/stride, `@P` on the copies) in
the mixed module, the bad-page test on the page index itself in the static
modules — is **~40 (fp8) / ~35 (fp4) / ~54 (mixed, compressed page)
warp-instructions per warp per tile** for the same path, depth 3 after the
record loads, 7 (static) / 11 (mixed) extra live registers inside the
converters' 56.  Per tile-SM: fp8 4 150 -> **~3 550**, fp4 3 881 -> **~3 460**,
mixed 3 580 -> **~3 180**.  Issue model at the measured paired IPC: fp8
**59.0 us** (band 56-63.5), fp4 **~55** (chain-floored, band 54-58), mixed
**58.6** (band 56-62.5); targets fp8 <= 58 and fp4 <= 36 are not reached by
this lever alone, mixed <= 62 is inside the band.  **Go** (section 9) — the
first count lever on the existing layout that does not need a protocol
change, with SASS gates before any timing.

## 1. Before: SASS-derived per-tile-SM count of the region, by class

### 1.1 Method (reading only)

- Static text: `/tmp/r4p7_sass/f{1,2,-1}_li.nvdis` on nkcut2 — `nvdisasm
  --print-line-info-inline` of a `-lineinfo -cubin` build of **this exact
  source** (`diff` of `mixedkv-wt-r4p7/csrc/xqa/mha_sm90.cu` against this
  worktree: empty; ptxas log `0 bytes stack frame, 0 spill`, REG 48, two
  `USETMAXREG` 0x28 / 0x38).  Parser `/tmp/r5copy_region.py` (nkcut2 and
  local `/tmp`): each SASS instruction is attributed to the innermost
  `mha_sm90.cu` line and to the outermost frame line; the K converter is
  outer lines [2633, 2715), the V converter [2715, 2763).  A bug in a first
  version of the parser (chain reset after every instruction, so
  instructions sharing one `//##` header were dropped) was fixed before any
  number below was taken; the fixed parser reproduces the r4p7 doc's role
  totals.
- Dynamic count: the production ncu reports `/tmp/r4mixed2_pcs_{fp8,fp4,
  mixed}_P132.csv` (r2p8 checkout, `--section SourceCounters`, one launch,
  8 712 tiles, SM clock locked at 1.38 GHz — instruction counts are
  clock-invariant and are the only thing used) joined to the r2p8 lineinfo
  listing `/tmp/r2p8_ptx/li{1,2,-1}.nvdis` by `/tmp/r5copy_dyn.py` (r2p8
  lines = this tree's lines - 17 for lines >= 688; the r4p7 doc established
  that the two sources compile to the same 4 696 SASS at REG 48).  Numbers
  are `inst_executed` / 8 712 = warp-instructions per tile-CTA = per
  tile-SM at P = 132; the K converter has 4 warps, so "per warp" = / 4.  The
  steady-state issue site is the loop instance (outer :2701-2708; the two
  prologue instances at :2679 execute once per CTA); its per-PC count is
  3.9 per tile-CTA (= 4 x 64/66: the loop issues 64 of a CTA's 66 tiles).
- Stall attribution: the same csv's `stall_*` columns summed over the path's
  PCs (`/tmp/r5copy_stall.py`).

### 1.2 Static SASS of the steady-state copy-issue site (K converter, loop instance, per lane-tile = per warp-tile)

| class | fp8 (`f1`) | fp4 (`f2`) | mixed (`f-1`, both format arms) |
|---|---:|---:|---:|
| `LDGSTS` (`.128` packed rows + `.64` scale) | 4 + 1 = **5** | 2 + 1 = **3** | 6 + 1 = 7 (4 + 1 executed on an fp8 page, 2 + 1 on fp4) |
| `LDGDEPBAR` (commit) | 1 | 1 | 1 |
| 64-bit MAC `IMAD.WIDE.U32` / `UIMAD.WIDE` | **11** (10 + 1 parity) | 9 | 13 (+ 2 `IMAD.X`) |
| `IADD3.X` (carry adds of the span pointer) | 3 | 3 | 4 |
| 32-bit address / ALU / moves (`LOP3`, `IMAD`, `IMAD.SHL`, `SHF`, `LEA`, `VIADD`, `IADD3`, `IMAD.U32`, `MOV`) | **44** | 28 | 56 |
| uniform datapath (`U*`, `R2UR`, `S2UR`: span offset `UIMAD`, parity math, tag compare, `USEL`, tag rotation `UPRMT/ULOP3`) | **28** | 22 | 33 |
| predication / branch (`ISETP`, `PLOP3`, `USEL`, `BRA`, `BSSY`, `BSYNC`) | 14 | 19 | 24 |
| constant-bank loads of the span (`LDC.64`, `ULDC`, `ULDC.64`) | 6 | 6 | 8 |
| `LDS` record reads (`LDS.U8` tag, `LDS` page, `LDS` head) | 3 | 3 | 3 |
| ptxas fillers `@!PT LDS RZ, [RZ]` (3 before each `LDGSTS` group) | 6 | 6 | 9 |
| barrier waits (`SYNCS.PHASECHK.TRYWAIT` + `BRA`, meta + consumed) | 2 (+ 8 spin) | 8 | 8 |
| **static total** | **123** | **109** | **168** |

(The 3 static `BSSY/BSYNC` + `@P BRA` pairs are the divergent `if (lane <
tokensPerPage)` around the scale copy; the meta-ready wait's 13
instructions execute on 1 tile in 16.)

### 1.3 Executed warp-instructions per tile-CTA (ncu, fp8 P132; K converter, 4 warps)

| piece (this tree's lines) | per tile-CTA | per warp | what |
|---|---:|---:|---|
| :2701-2703, :2708 loop control, tag rotation, `LDGDEPBAR` | 31.8 | 8 | `UPRMT`/`ULOP3` rotation, loop test, commit |
| :2662-2663 lambda: `t % 3`, `t / 3` parity (magic multiply), `t % 16` test | 34.9 | 8.7 | `IMAD.WIDE.U32 x 0xAAAAAAAB`, `R2UR` x2, `USHF`, `UIMAD`, `PLOP3`, `BRA` |
| :2666-2667 `kMetaReady` wait (1 tile in 16) | 7.1 | 1.8 | |
| :2671 `kBar[stage].consumed.wait_parity` | 57.4 (body ~39, spin ~18) | 14.4 | bar address `UIMAD.WIDE`, parity `USEL/SHF`, `TRYWAIT`, `BRA`, retries |
| :790-793 record address + `LDS` page / head | 19.4 | 4.9 | |
| :3146-3158 tag `LDS.U8`, `R2UR`, `UISETP`, `USEL`, `PLOP3`, `BRA` | 31.0 | 7.8 | tag decode + bad/A16 early-out test |
| :3165-3168 span select (`UIMAD x 0x48`), `ULDC`/`LDC.64` of 4 span words, first `IMAD.WIDE` | 31.0 | 7.8 | |
| :3171-3180 payload address chain + 4 row destinations + 4 `LDGSTS.128` | 166.5 | 41.7 | 10 `IMAD.WIDE`, `IADD3/IADD3.X`, 24 `LOP3`/`IMAD.SHL`/`IMAD`/`VIADD`, 3 fillers |
| :3194-3201 scale copy (divergent branch, 3 `IMAD.WIDE`, `IADD3.X`, fillers, `LDGSTS.64`, `BSYNC`) | 117.1 | 29.3 | |
| **copy-issue path, loop instance** | **476.1** | **119** | |
| of which `issueCompressedPageCopies` proper (:790-793, :3146-3204) | 344.9 | **86** | the task's "~90 per warp, ~720 per tile-SM" |

V converter (:2755-2759 site): **505.6 per tile-CTA = 126 per warp** (same
code; the `consumed` wait spins more).  fp4: K 365.6 (91 per warp), of which
the function proper ~285 (71).  mixed: K 376.1 (94 per warp averaged over
the 2/3 compressed and 1/3 A16 pages; a compressed-page warp runs ~125, an
A16 warp ~30).  Whole K converter per warp per tile: fp8 336, fp4 305,
mixed 253 (expansion 188 / 188 / 2/3 x 188, glue ~30).

**Per tile-SM (K + V, 8 warps), fp8: copy-issue path ~950-980 (~23 % of
4 150); function proper ~700 (17 %).**  fp4: ~730 / ~570.  mixed: ~750 /
~600 (the A16 warps' early-out included).

### 1.4 Stall structure of the path (fp8 K, production csv, 826 samples over the loop site)

`wait` 19.6 %, `long_scoreboard` 15.4 % (+ 8.2 % not-issued), `selected`
12.2 %, `not_selected` 8.7 %, `short_scoreboard` 8.6 % (+ 3.5 %), `math`
3.5 %, `dispatch` 3.3 %, `branch_resolving` 2.9 %.  Top PCs: the `@!P0 BRA`
after the `consumed` `TRYWAIT` (114 samples, `long_sb` — waiting for gemm0's
stage release: a wait, not issue), the first payload `IMAD.WIDE.U32 R4,
R13, R10, R4` (41, `short_sb` 40: waits for the page `LDS` and the
`LDC.64` of the span pointer), `R2UR UR6, R4` after the tag `LDS.U8` (35,
`short_sb` 28), `UPRMT` (28, `branch_resolving`), `UISETP` on the tag (21,
`wait`).  fp4 and mixed: the same shape (mixed's `long_sb` 24.9 %: the
release wait; `R2UR` 38 samples `short_sb`).  Expansion for comparison (fp8
K, 866 samples): `selected` 19.3, `wait` 14.1, `short_sb` 13.5, `no_inst`
13.0.

Reading: the path's own issue cost is ~120 slots per warp-tile, and its
critical path is the dependent chain tag-`LDS` -> `R2UR` -> `UISETP` ->
`PLOP3` -> `BRA` -> `UIMAD` -> `ULDC` -> page-`LDS` -> `IMAD.WIDE` (page) ->
`IMAD.WIDE` (head) -> `IMAD.WIDE` (row) -> `IADD3` -> `IADD3.X` -> `IMAD.WIDE`
(row step) x3 -> `LDGSTS`: **~14 dependent instructions and three latency
hops (two `LDS`, one uniform-datapath round trip), ~200-250 cycles** before
the last row copy leaves, then the divergent scale branch (another `ULDC` ->
3 `IMAD.WIDE` -> `IADD3.X` -> `LDGSTS`).

### 1.5 Where the count comes from (by cause, fp8, per lane-tile)

1. **Per-tile recomputation of lane constants** (r0 = lane / 8, c = lane % 8,
   `c ^ (r % 8)`, `r * 128`, the four destinations, the scale destination):
   ~30 of the 44 32-bit ops.  `ExpandLane` hoisted the expansion's
   equivalents in [16]; the copy path was not hoisted.
2. **Five 64-bit terms per source instead of one**: `page * ps` + `head * hs`
   + `r0 * ts` + span pointer (`IADD3/IADD3.X`) + `c * 16`, then the row
   step as a serial `src += 4 * ts` chain (3 more `IMAD.WIDE`); the scale
   source repeats the pattern (3 `IMAD.WIDE` + carry add).  11 `IMAD.WIDE`
   + 3 `IADD3.X` + ~8 moves.
3. **Span selection at runtime** from the tag (`UIMAD x 0x48`, then `LDC.64`
   / `ULDC` of `k_payload`, `payload_stride.{page,token,head}`,
   `k_scales`, `scale_stride.*`: 6 constant loads per tile, each a
   scoreboard wait) although the format is compile-time in the static
   modules and only two spans exist in the mixed module.
4. **Tag decode on the uniform datapath**: `LDS.U8` -> `R2UR` -> `UISETP` ->
   `USEL` -> `PLOP3` -> `BRA`, and the head read as a third `LDS`, when the
   static modules' bad-page test is `page == kBAD_PAGE_INDEX` (fillTileMeta
   :3370-3372 sets the tag from exactly that test).
5. **Parity math by magic multiply** (`t / 3`, `t % 3`, `t % 16`, `t % 4`)
   per tile instead of loop-carried stage / parity state: ~17 uniform ops.
6. **Divergent scale branch** (`BSSY`, `ISETP`, `@P BRA`, `BSYNC`) instead of
   a predicated `LDGSTS`, and a second filler triple for its `LDGSTS` group.

## 2. Current data flow and control flow (as written; line refs into this tree)

Data flow per compressed page (one converter warp = one page of the tile;
`convertWarpsPerOperand == nbPagesPerTile == 4`, :3142):

```
TileRecord meta[op][chunk][g%16] (32 B: pages[4] +0, formats +16, tile +20, idxReq +24, idxHeadGrp +28; :388-396)
  LDS.U8  tag  = rec + 16 + idxWarp                (:3148)     -> R2UR -> bad/A16 early-out (:3151-3158)
  LDS.32  page = rec + 4*idxWarp                   (:3162)
  LDS.32  head = rec + 28                          (:3163)
span = cacheList.transport.formats[format]  (kernel param, c[0x0]; :3166 -> UIMAD x 0x48 + LDC/ULDC)
payloadPage = span.k/v_payload + page*ps + head*hs (64-bit; :3169-3171)
fp8: lane owns chunk c = lane%8 of rows r0 + 4k, r0 = lane/8, k < 4:
     src_k = payloadPage + (r0+4k)*ts + c*16 ; dst_k = slot + (r0+4k)*128 + ((c ^ (r0+4k)%8))*16   (:3175-3183)
fp4: lane owns chunk c = lane%4 of rows r0 + 8k, r0 = lane/4, k < 2, 80 B landing stride         (:3185-3193)
lane < 16: scales src = span.k/v_scales + page*sps + lane*sts + head*shs -> dstScales[idxWarp*16+lane] (8 B; :3196-3202)
return tag -> kTags byte (kAhead-1)                                                               (:2703)
```

Control flow of a converter warp (K; V mirrors at :2728-2759):

```
prologue: issueKCopies(0), commit; issueKCopies(1), commit                          (:2675-2681)
loop g = 0 .. nbCtaTiles-1:                                                          (:2682)
  cp.async.wait_group<1> ; __syncwarp                                                (:2684-2685)
  expandPackedStage(stage g%3, kScales[g%4], kTags & 0xFF)                           (:2690)
  fence.proxy.async ; kBar[g%3].produced.arrive                                      (:2694-2695)
  kTags >>= 8 ; if (g+2 < nbCtaTiles) kTags |= issueKCopies(g+2) << 8 ; commit      (:2701-2708)
issueKCopies(t):  if (t%16 == 0) kMetaReady[(t/16)%2].wait_parity(t/32 & 1)         (:2663-2667)
                  kBar[t%3].consumed.wait_parity((t/3) & 1)                          (:2671)
                  return issueCompressedPageCopies(..., t, k[(t%3)*2 + 1], kScales[t%4], warpIdx.x)   (:2672)
```

Invariants in force on this path: C6 (issue budget), C10 (record read after
the `kMetaReady` wait, lever-8 doc 8.1), D6 (warp-contiguous copy
ownership), D4 (bad pages: no copy, zero-filled by the expansion), A4
(`ptxas -v` no C7507, two `USETMAXREG`, STACK 0), the `kAhead = 2` group
accounting (one commit per tile, empty groups included).

## 3. New data flow and control flow

### 3.1 Principle

Everything that does not change per tile is computed once per warp and
kept in registers (`CopyLane`, the copy-side twin of `ExpandLane` :554-598);
everything that changes per tile comes from **two `LDS`** and enters the
addresses through **one `IMAD.WIDE` per copy**; the destinations are
`[R + imm]` from lane bases plus one stage add; the only control flow is
the two uniform waits and (mixed module) one uniform early-out for pages
that are not copied.  Nothing about *what* is copied changes: the same
bytes go to the same shared addresses (except the E2M1 landing permutation
of 3.4), so the expansion, its numerics and the stage protocol are as
before.

### 3.2 Per-warp constants (`CopyLane`, computed at converter start next to `makeExpandLane`)

fp8 (static module and the fp8 arm of the mixed body), lane `l`, `r0 = l/8`,
`c = l%8`, `x = c ^ r0` (r0 < 4 so `r0 % 8 == r0`):

```
src64   = span.payload(K|V) + r0*ts + c*16          64-bit, 2 regs   (page and head terms are per tile)
ssrc64  = span.scales(K|V)  + l*sts                 64-bit, 2 regs   (lanes >= 16 hold a harmless value; predicated off)
dstEven = packedBase + idxWarp*2048 + r0*128 + x*16          rows r0, r0+8  (k even):  +0, +1024
dstOdd  = packedBase + idxWarp*2048 + (r0+4)*128 + (x^4)*16  rows r0+4, r0+12 (k odd): +0, +1024
sdst    = (idxWarp*16 + l)*8                                 within a TileScales slot
```

Row `r0 + 4k` has `(r0+4k) % 8 = r0 + 4(k%2)`, so its swizzle term is `x`
for even k and `x ^ 4` for odd k: two destination bases carry the whole
page, and `k = 2, 3` are the same bases `+ 1024` (immediates).  Source rows
are `src64 + 4k*ts`: the anchor plus three `IMAD.WIDE.U32(ts, 4k, anchor)`
(`Rd.64 = Ra * imm + Rc.64`; the token stride `ts` is kernel-uniform and
lives in a `UR` or one `R`).

fp4 static module (compile-time format): native ownership `c = l%4`, `r0 =
l/4` (`r0 < 8`), rows `r0`, `r0 + 8`: one `dst = packedBase + idxWarp*2048 +
r0*128 + ((c ^ r0))*16`, second row at `+1024`; `src64 = payload + r0*ts4 +
c*16`, second row `IMAD.WIDE.U32(ts4, 8, anchor)`.  (Landing layout 3.4.)

Mixed module: `src8/src4`, `ssrc8/ssrc4` (8 regs), the fp8 destination bases
(the fp4 arm uses the fp8 ownership, 3.4), strides `ps8/ps4, ts8/ts4,
hs8/hs4, sps8/sps4, shs8/shs4` as `UR` (kernel-uniform; the constant bank
is the fallback, 1 `ULDC` each per tile).

### 3.3 Per-tile flow (fp8 static; fp4 static differs only in copy count)

```
issue(t):            [uniform, loop-carried: issStage in {0,1,2}, issParity, issSlot = t & 3]
  if (t % 16 == 0) kMetaReady[(t/16)&1].wait_parity(...)                 ULOP3, BRA (+ rare wait)
  kBar[issStage].consumed.wait_parity(issParity)                          UIMAD (bar addr), TRYWAIT, BRA
  rec   = metaBase + (t % 32) * 32                                         ULOP3, ULEA  (or carried: +32 wrap)
  page  = LDS.32 [rec + 4*idxWarp]                                         LDS
  tail  = LDS.128 [rec + 16]  -> formats, tile, idxReq, head              LDS   (head = tail.w)
  P     = (page != kBAD_PAGE_INDEX)                                        ISETP        (static: this IS the tag test)
  tag   = P ? STATIC_FORMAT : 0xFF                                         SEL -> kTags byte
  a     = IMAD.WIDE.U32(page, ps, src64)   ; a = IMAD.WIDE.U32(head, hs, a)    2 IMAD.WIDE (anchor: row r0)
  s1, s2, s3 = IMAD.WIDE.U32(ts, 4|8|12, a)                                3 IMAD.WIDE (independent)
  b     = IMAD.WIDE.U32(page, sps, ssrc64) ; b = IMAD.WIDE.U32(head, shs, b)   2 IMAD.WIDE (scale anchor)
  dE    = dstEven + stageOff ; dO = dstOdd + stageOff ; dS = sdst + slotOff      2 UIMAD (offsets) + 3 VIADD
  @P LDGSTS.128 [dE], [a] ; @P LDGSTS.128 [dO], [s1] ; @P LDGSTS.128 [dE+1024], [s2] ; @P LDGSTS.128 [dO+1024], [s3]
  @P&&(lane<16) LDGSTS.64 [dS], [b]                                        1 ISETP.AND for the scale predicate
  kTags |= tag << 8 ; LDGDEPBAR                                            UPRMT/ULOP3 (as today), commit
  issStage/issParity advance: issStage = issStage == 2 ? 0 : issStage + 1 ; issParity ^= (issStage == 0)   ~4 uniform ops
```

Predicated `cp.async` is written as one `asm` with an internal `setp` and
`@p cp.async...` (the F25 `@leader` form), so no `BSSY/BSYNC` and no
divergence; a predicated-off `LDGSTS` still costs its issue slot (counted).
The scale word of lanes >= 16 is never copied (as today).  Bad pages (`P`
false) copy nothing and the expansion zero-fills them (D4, unchanged).

**Head per tile, not per item.**  The task's per-item bases would fold
`head * hs` into `src64` at item boundaries; that saves 2 `IMAD.WIDE` per
tile but needs the `first` bit test on the tile word (`ISETP` + `BRA`, the
same 2 instructions) plus a rare recompute path and item state in the
converters.  Equal count, more control flow, one more invariant (C10's
record-read ordering would have to cover the recompute): the per-tile
`IMAD.WIDE` on `head` is chosen.  The record's `idxHeadGrp` is already read
per tile today (:3163).

### 3.4 Mixed module: format as data, one landing layout for both compressed formats

Today E2M1 pages land at an 80 B row stride (`packedRowStrideFP4` :370) with
their own lane ownership (`c = l%4`, rows `l/4 + 8k`), so the fp8 and fp4
copy bodies differ in destination formula, row step **and immediates**,
which forces a per-format branch.  New: **E2M1 rows land at 128 B stride
with the E4M3 swizzle** — chunk `c` (`c < 4`) of row `r` at `r*128 + ((c ^
(r%8)) * 16)` inside the page's 2 KB slot (same `packedSlotBytes`, smem
unchanged).  Then in the mixed body the fp4 arm uses the fp8 ownership
(`c = l%8`, rows `r0 + 4k`) with lanes `c >= 4` predicated off: the
destination bases, the immediates (`+1024`) and the copy count (4 + 1) are
identical for both formats and the format enters only as data:

```
fmt  = (tail.x >> 8*idxWarp) & 0xFF                          SHF/PRMT (1)
isBad = page == -1 ; isA16 = fmt == 0 ; isFP8 = fmt == 1     3 ISETP (isBad folds into the early-out)
if (!(isBad || isA16 ... )) uniform early-out (PLOP3 + BRA)  pages the loader TMAs / bad pages: no copies (as today)
base = isFP8 ? src8 : src4 (2 SEL) ; ps/ts/hs = isFP8 ? .8 : .4 (3 SEL) ; sbase (2 SEL) ; sps/shs (2 SEL)   9 SEL
Prow = isFP8 || c < 4  (1 PLOP3 with the lane-constant predicate; +1 ISETP if not kept live)
copies as 3.3 with @Prow on the four row copies, @(lane<16) on the scale copy
```

Static fp4 module: native ownership (3.2) on the same landing layout, 2 + 1
copies.  The expansion's E2M1 read (`:3575-3579`, two `LDS.128` at
`row + c*16`) becomes two `LDS.128` at `row` and `row ^ 16` where the lane
base `l.fp4` carries `((2p) ^ x) * 16` (`x = token % 8`, `p` the head
part): **+1 `LOP3` per lane-tile in the fp4 expansion** (the fp8 read path
already has this form, `:3525-3529`).  Bank behaviour: eight consecutive
tokens read chunk `(2p) ^ x` = eight distinct chunks -> conflict-free
`LDS.128` (as fp8's); `LDGSTS` octets (eight lanes = one row, four active)
land in one 128 B line (A7 rule: ideal wavefronts); the fp4-static octet
(lanes `8m..8m+7` = rows `2m, 2m+1`) spans two lines as today's 80 B stride
does.  Nothing else reads the packed rows.

Format-as-data cost on fp4 pages of the mixed module: 4 + 1 `LDGSTS` and 5
`IMAD.WIDE` where a format-specific body would issue 2 + 1 and 3 (+4 per
lane-tile), bought against one body (the mixed module's `no_instruction`
stalls are +55 %, mixed doc 1.1, and today's copy site is 168 static
instructions with two arms).

### 3.5 What is not changed

`kAhead = 2`, one `commitGroup` per tile including empty groups,
`wait_group<1>`, the `kMetaReady` / `consumed` waits and their parities (only
their arithmetic form), the `produced.arrive`, the fence, the `kTags`
rotation, `TileRecord`, `fillTileMeta`, the loader, gemm0 / gemm1, the
expansion body (numerics, stores, votes), `SharedMem` size and layout,
`MIXED_KV_EXPERIMENT` hooks (bits 2 and 4 keep their meaning: return the
tag before any copy; skip the scale copy).

## 4. Invariants affected, bit-exactness

- **D6 (copy ownership)** — restated, not weakened: E4M3 unchanged (chunk
  `l%8` of rows `l/8 + 4k`); E2M1 in the mixed module: chunk `l%8 < 4` of
  rows `l/8 + 4k` (16 active lanes x 4 copies); E2M1 in the static fp4
  module: chunk `l%4` of rows `l/4 + 8k` (unchanged).  Sources stay
  16 B-aligned rows; sector efficiency unchanged (each active lane reads a
  whole aligned 16 B chunk of a 64 B / 128 B row).
- **D2 / landing layout (sm90 amendment)** — E2M1 packed rows: 128 B stride,
  chunk `c` at `(c ^ (r%8)) * 16` (was 80 B stride, `c * 16`).
  `packedRowStrideFP4` is retired; `static_assert(4 * 16 <= 128)`.  The
  expansion's `l.fp4` formula follows (3.4).  The "reads before
  `__syncwarp` before writes" ordering of the last head-part buffer is
  unchanged (the packed rows still live where the part-1 A16 rows are
  written last).
- **D4 (tail)** — unchanged: bad pages issue no copy (predicate false),
  the expansion zero-fills; the record's `kBAD_PAGE_INDEX` <=> tag `0xFF`
  identity (`fillTileMeta` :3370-3372) is now relied on by the static
  modules' copy predicate — add a `static_assert`-style comment at both
  sites and a host-side check that `page_format[]` never holds `0xFF`.
- **C6 (issue budget)** — restated as the gate of section 7: copy-issue path
  <= 48 (fp8) / 42 (fp4) / 62 (mixed compressed) static SASS per lane-tile.
- **C10 (record visibility)** — unchanged: both `LDS` follow the same
  `kMetaReady` wait as today's three; the record is read once per tile per
  warp (the head is no longer a separate read).
- **C3 / A4 (register split)** — unchanged numbers (40 / 56, 3*128*40 +
  2*128*56 = 29 696 <= 30 720); the converters gain 7 (static) / 11 (mixed)
  loop-invariant registers (section 5).
- **Bit-exactness**: copies move bytes; the E2M1 landing permutation is
  undone by the same-lane read; the expansion's arithmetic, votes and
  stores are untouched; A16 pages are untouched.  Confirmation: the 72-case
  matrix (`mixedkv_remote_run.sh`) bit-exact, including the tail / partial
  / persistent (items > CTAs) cases.

## 5. Register and shared-memory budgets (ptxas C7507 rule)

- Shared memory: **unchanged** (E2M1 rows stay inside their 2 KB slot; no
  new arrays).  The 2-CTAs/SM `static_assert` (:485) is unaffected.
- Registers, converters (`setmaxnreg.inc 56`, :1448): today's steady-state
  live set across the tile loop = `ExpandLane` 4 + `ExpandScales` 6 + `kTags`
  1 + lane 1 + loop / stage state (mostly `UR`) + the expansion's peak (4
  `LDS.128` words 16 + scale word + 8 decoded + store addresses ~4 -> ~30)
  = ~42 at the expansion peak; the copy site's peak today is ~24 (four
  64-bit sources + destinations + span words).  New loop-invariants:
  `CopyLane` 7 (fp8: 2 x 64-bit + 3 x 32-bit), 6 (fp4), 11 (mixed); the
  copy site's transient peak becomes 4 x 64-bit sources (8) + scale source
  (2) + 3 destinations = 13 on top of the invariants.  Static: ~42 + 7 = 49
  <= 56 at the expansion peak (the copy transients are not live there);
  mixed: ~42 + 11 = 53 <= 56 — **tight**.  Strides (5 per format) must sit
  in `UR` (63 available; kernel-uniform values from `c[0x0]`) or be re-read
  by `ULDC` per tile (+5 / +10 issue slots, still inside the band).
- **Fallback if ptxas spills (STACK > 0) or emits C7507**: keep the 64-bit
  bases as span pointer in `UR` pairs + a 32-bit lane offset (`r0*ts +
  c*16`), rebuilt per tile with one `IADD3 / IADD3.X` pair after the anchor
  `IMAD.WIDE` (+2 per source anchor = +4 per lane-tile), saving 2 (static)
  / 4 (mixed) registers.  Not chosen up front because the direct form is
  the F25 form the task names and the gate (section 7) catches the spill
  before any timing.
- Role split, launch bound and the two `USETMAXREG` immediates (0x28 /
  0x38) are unchanged; `ptxas -v` must print no C7507 for the four modules
  (a16 / fp8 / fp4 / mixed).

## 6. Predicted counts after, dependency depth, wall

### 6.1 Copy-issue path per warp per tile (steady state)

| class | fp8 before -> after | fp4 before -> after | mixed (compressed page) before -> after | mixed (A16 page) before -> after |
|---|---:|---:|---:|---:|
| `LDGSTS` | 5 -> 5 | 3 -> 3 | ~5 -> 5 | 0 -> 0 |
| `LDGDEPBAR` | 1 -> 1 | 1 -> 1 | 1 -> 1 | 1 -> 1 |
| `IMAD.WIDE` (incl. parity) | 11 -> **7** | 9 -> **5** | ~11 -> **7** | 1 -> 0 |
| `IADD3.X` | 3 -> **0** | 3 -> 0 | 4 -> 0 | 0 |
| 32-bit addr / ALU / moves | 44 -> **3** (3 `VIADD`) | 28 -> 2 | 56 -> 3 | ~6 -> 1 |
| `SEL` (format as data) | 0 -> 1 | 0 -> 0 | 0 -> 9 | 0 |
| uniform datapath (stage/parity, record addr, offsets, tag rotation) | 28 -> **~12** | 22 -> ~12 | 33 -> ~12 | ~20 -> ~10 |
| predication / branch | 14 -> **5** (`ISETP` x2, `PLOP3`, meta `BRA`, wait `BRA`) | 19 -> 5 | 24 -> 8 (+ early-out `PLOP3/BRA` = 2) | ~12 -> 6 |
| constant-bank loads | 6 -> **0-5** | 6 -> 0-5 | 8 -> 0-10 | 0 |
| `LDS` (+ `R2UR`) | 3 (+3) -> **2** | 3 -> 2 | 3 -> 2 | 1 -> 2 |
| fillers `@!PT LDS RZ` | 6 -> 3-6 | 6 -> 3-6 | 9 -> 3-6 | 0 |
| barrier waits (body) | 2 -> 2 | 2 -> 2 | 2 -> 2 | 2 -> 2 |
| **total per warp-tile** | **~119 -> ~40** (band 36-48) | **~91 -> ~35** (31-42) | **~125 -> ~54** (50-62) | **~30 -> ~24** |
| **copy function proper** | **~86 -> ~30** | ~71 -> ~26 | ~95 -> ~44 | — |

Per tile-SM (8 warps): fp8 copy path **~950 -> ~320** (function proper ~700
-> ~240), fp4 **~730 -> ~280**, mixed (2/3 compressed, 1/3 A16) **~750 ->
~350**.  fp4 expansion +1 `LOP3` per lane-tile (+8 per tile-SM).

### 6.2 Per-role, per tile-SM (warp-instructions; fp8 / fp4 / mixed)

| role | before | after |
|---|---:|---:|
| gemm0 (4 warps) | 927 / 923 / 927 | unchanged |
| gemm1 (4) | 535 / 527 / 535 | unchanged |
| IO (4) | ~75 / ~73 / ~160 | unchanged |
| K converters (4) | 1 314 / 1 189 / ~1 000 | **~1 000** / ~960 / ~800 |
| V converters (4) | 1 318 / 1 193 / ~1 030 | **~1 010** / ~970 / ~810 |
| **total (uninstrumented scale)** | **4 150 / 3 881 / 3 580** | **~3 550 / ~3 460 / ~3 180** (-14.5 % / -11 % / -11 %) |
| converter share | 63 / 62 / 57 % | 57 / 56 / 51 % |

### 6.3 Dependency depth per tile (copy path, after the `consumed` wait)

Before: `LDS.U8` -> `R2UR` -> `UISETP` -> `PLOP3` -> `BRA` -> `UIMAD` -> `ULDC`
-> `LDS` -> `IMAD.WIDE` -> `IMAD.WIDE` -> `IMAD.WIDE` -> `IADD3` -> `IADD3.X` ->
`IMAD.WIDE` -> `IMAD.WIDE` -> `IMAD.WIDE` -> `LDGSTS`: **16 dependent
instructions, 3 latency hops** (~200-250 cycles to the last row copy); the
scale copy adds a divergent branch and a 6-deep chain after it.
After (static): `LDS.32` || `LDS.128` -> `ISETP` (predicate) || `IMAD.WIDE`
(page) -> `IMAD.WIDE` (head) -> {`IMAD.WIDE` x3 in parallel} -> `LDGSTS` x4;
scale: `IMAD.WIDE` -> `IMAD.WIDE` -> `LDGSTS`: **depth 4 after one `LDS`
latency, no uniform-datapath round trip** (~30 + 4 x ~5 = ~50-60 cycles to
the last copy).  Mixed: +2 (`SHF` extract -> `ISETP` -> `SEL`s), ~70 cycles.
The destinations (`VIADD` from bases) are off the chain.

Track-S-step-6 caveat applied: that kernel's cut removed instructions whose
latency was *covered* and exposed the scale-chain scoreboards.  Here the
removed instructions are themselves the exposed chain (1.4: `R2UR` and the
first `IMAD.WIDE` are the path's top `short_sb` PCs; `math`/`dispatch` are
small) and no new long-latency operation is introduced (the two `LDS` were
three), so the path's stall structure can only improve with the count.

### 6.4 K-ready time relative to the gemm0 cadence

The copy issue for tile `t+2` sits between `produced.arrive(t)` and the
expansion of `t+1` (:2694-2708 then :2684-2690), so its duration is part of
the converter's per-tile period and of the K(`t+1`)-ready time.  Converter
period per CTA-tile today (fast / slow pair member, 2.5 of the [7] doc):
329 instr x 5.8 / 7.4 cyc = 0.96 / 1.23 us, of which the copy path ~119 x
5.8 / 7.4 = **0.35 / 0.44 us**; after: 250 instr -> **0.73 / 0.93 us** per
CTA-tile against a CTA tile period of 2 x 0.858 = 1.72 us (paired).  The
converter's duty falls from 56-72 % to 42-54 %, and K(`t+1`).produced
arrives ~0.25-0.3 us earlier relative to gemm0's cadence.  gemm0's K-wait
today (261-262 ns, equal in both members — pair doc 1.4) is therefore the
barrier round trip, not converter lateness; this lever adds slack and does
not shrink that segment.  Landing: the first `LDGSTS` leaves ~90 cycles
after the stage release instead of ~350, and the copies of `t+2` have >= 2
CTA tile periods (3.4 us) to land against a ~2 us loaded latency — the
`wait_group` stays non-blocking.

### 6.5 Wall via the issue model, assumptions stated

    cadence = N / (IPC x 1.98 GHz)      N from 6.2; IPC = the measured paired steady-state SM IPC
    wall    = fill 8.5 + 66 x cadence + tail (2.8 / 2.6 / 2.8)      ([7] doc 2.4, 7.4)

| mode | N before -> after | IPC (paired, measured) | cadence before -> after (us) | wall before -> **predicted** | band | target |
|---|---:|---:|---:|---:|---|---:|
| fp8 | 4 150 -> 3 550 | 2.48 | 0.858 -> 0.723 | 67.9 -> **59.0** | 56.0-63.5 | <= 58 (miss, band edge) |
| fp4 | 3 881 -> 3 460 | 2.65 | 0.750 -> 0.659 -> **floored 0.66-0.69** | 60.6 -> **~55** | 54-58 | <= 36 (miss) |
| mixed | 3 580 -> 3 180 | 2.24 | 0.810 -> 0.717 | 64.6 -> **58.6** | 56-62.5 | <= 62 (inside the band) |
| a16 | — | — | DRAM-bound | 79.4 -> **79.4** | +-1 | parity |

Assumptions and where they fail:

1. **The SM is issue-bound at 2 CTAs/SM** (pair doc 1.4: every work segment
   of the second-dispatched CTA is 16-28 % longer, memory segments equal;
   issue-active 0.56-0.63 with 1.58 eligible warps per scheduler).  Under
   this assumption the cadence scales with the count (central column).  It
   fails if the SM becomes **chain-bound** first: two CTAs' consumer chains
   (1.25-1.37 us per CTA-tile lone, [7] doc 7.3) interleaved give a floor
   of ~0.63-0.69 us per tile-SM — fp8's 0.72 is above it, **fp4's 0.66 is
   at it** (hence the floored row), mixed's 0.72 above.  A result at the
   floor with the counts as predicted means the next lever must shorten
   the chain (the 668 uniform descriptor ops per tile-SM are on it).
2. **IPC constant under the cut.**  Track S step 6 measured issue-active
   falling with the count and time flat (the freed slots became scoreboard
   stalls).  Here the discriminator is the same ncu pair: if
   `smsp__issue_active` falls by the count ratio while duration is flat,
   assumption 1 was wrong for this regime.  The pessimistic edge of the
   band takes half the count cut as time (fp8 0.79 -> 63.5).
3. **Fill and tail unchanged** (fill is 1 CTA/SM ramp at ~0.95-1.0 us per
   tile and is untouched by a converter cut of this size: -0.1 us per tile
   x 8-9 fill tiles is inside the band).
4. **The converter is not the pacing role** after the cut (6.4: duty <= 54 %
   in the slow member).  Holds by construction.

## 7. Verification artefacts, accept / reject (SASS gates before any timing)

Order: compile-only gates on nkcut2 (`/tmp/r4p7_sass/build.sh` recipe with
this checkout), then the standard confirmation script once.

1. **`ptxas -v`, four modules** (`-DMIXED_PAGE_STATIC_FORMAT` = 0, 1, 2, -1):
   `0 bytes stack frame, 0 spill`, no C7507, `Used 48 registers`;
   `cuobjdump -sass`: exactly two `USETMAXREG` (`DEALLOC 0x28`, `TRY_ALLOC
   0x38`); `cuobjdump -res-usage` STACK 0.  Reject otherwise (then the
   section-5 fallback, and re-gate).
2. **Copy-issue site, static SASS per lane-tile** (`/tmp/r5copy_region.py
   f*_li.nvdis 2633 2715 dump`, loop instance = the single steady-state
   outer line of the new `issueKCopies` call; V likewise): fp8 <= 48 with
   `IMAD.WIDE` exactly 7, `IADD3.X` 0, `LDGSTS` 5 (4 `.128` + 1 `.64`),
   `LDS` 2, `LDS.U8` 0, `R2UR` <= 1, `BSSY/BSYNC` 0, `LDC/ULDC` <= 5,
   fillers <= 6, `SEL` <= 1; fp4 <= 42 with `IMAD.WIDE` 5, `LDGSTS` 3; mixed
   <= 62 with one body (`LDGSTS` 5 static, `IMAD.WIDE` 7, `SEL` <= 10, one
   early-out `BRA`), `IMAD.X` 0.  **Reject (do not time) if fp8 > 56 or
   `IADD3.X` > 0** (the F25 lesson: `mad.wide.u32` with a 64-bit register
   addend must come out as one `IMAD.WIDE.U32 Rd, Ra, Rb, Rc.64`; if ptxas
   splits it, write the anchor as the split form deliberately and re-count
   — the budget then is +2 per anchor).
3. **Expansion untouched**: `expandPackedStage` static SASS of the fp8
   module identical in opcode histogram to today's (`3431-3557` region:
   `PRMT` 63, `LOP3` 64, `HMUL2` 96, `STS.128` 24 per K converter), fp4
   module +1 `LOP3` per site, `VOTE.ALL` count unchanged.
4. **a16 module**: converter region unchanged in count +-2 (its copy site
   is the early-out); `transport_a16` bench within +-1 us of 79.4.
5. **Correctness**: the 72-case matrix bit-exact (all four modes, tails,
   partial items, persistent items > CTAs, subnormal / max-scale / tiny-global
   regimes).  Any mismatch rejects.
6. **ncu (one launch per mode, `--clock-control none`, state the clock)**:
   `smsp__inst_executed.sum / 8712` fp8 <= 3 650 (predicted 3 550), fp4 <=
   3 550, mixed <= 3 280; converter body per tile-SM (r4p7 roles script)
   fp8 <= 2 100 (from 2 632); copy-site PCs' `short_sb + long_sb` share
   below today's 24 % (1.4).  Discriminator for 6.5 assumption 2:
   `smsp__issue_active` must not fall by more than half the count ratio
   while duration is flat.
7. **Trace (`MIXED_KV_TRACE=3`, CTA 0, tiles 4-60, medians, 1.98 GHz)**:
   copy-issue segment s9-s8 minus the stage-release wait: fp8 ~700 -> <= 350
   cycles; K converter period 6.4's 0.73-0.93 us band; gemm0 K-wait
   unchanged (barrier floor ~260 ns) or lower.
8. **Bench (the ONE confirmation, `flock ... mixedkv_remote_run.sh <ck>
   r5copy sm90 transport_a16 fp8 fp4 mixed`, co-tenant rule repeats x t <
   1.5 ms, min / median / max)**: accept fp8 <= 62.0, fp4 <= 58.0, mixed <=
   61.0, a16 78.4-80.4; **reject** if fp8 > 66 or fp4 > 60 or mixed > 64
   (no gain outside noise) or a16 > 81.

## 8. Do not build if

1. A compile-only build of the new site shows `IADD3.X` > 0 or `IMAD.WIDE`
   > 7 per lane-tile (fp8) after the deliberate-split rewrite: the anchor
   form is not expressible and the count band (6.1) is not met on paper.
2. `ptxas -v` shows a spill or C7507 in any module **after** the section-5
   fallback: the 56-register converter budget cannot carry the lane bases;
   the lever then needs the [15] 64-register layout first.
3. The 72-case matrix has no case where a compressed tile follows an A16
   tile in the mixed module with page slots past the sequence end (bad
   pages) — it does (tails + mixed); if that case were missing, add it
   first (the new static-module predicate relies on the page/tag identity).
4. The trace shows gemm0's K-wait > 400 ns in the *lone* control today
   (converter pacing in the lone regime): then the converter is the bound
   and the expansion, not the copy path, is the lever (6.4's slack argument
   fails).  The [16] trace has it at 311 cycles = 157 ns: not the case.

## 9. Go / no-go

**Go.**  (1) The region's count is measured, not modelled: ~950 per tile-SM
for the path, ~700 for the function proper, 23 % / 17 % of the SM's
issue slots, with a dependency chain whose scoreboards are the path's top
stalls; (2) the new form is the F25 form that landed as designed on the
fa3 host (`IMAD.WIDE` 24 per pair as predicted, copies 3 instructions each
after the split lesson — folded into gate 2 here), on the existing layout,
protocol and record, with the expansion untouched; (3) the prediction is
inside the issue model that the round-3 attribution established, with the
chain floor and the Track-S counter-example named as the two ways it fails
and with ncu discriminators for both; (4) the mixed target is inside the
band, fp8 at its optimistic edge, fp4 not reachable by any converter-side
cut (its floor is the consumer chain) — this lever is a prerequisite for
the descriptor lever (668 -> ~170 per tile-SM) to reach fp8 <= 58, not a
substitute.  Gates 1-5 (reading) run before any GPU time; one confirmation
run.

## Appendix: artefacts read / produced for this document

- nkcut2: `/tmp/r4p7_sass/f{1,2,-1}_li.nvdis` (this source, lineinfo),
  `/tmp/r4p7_sass/f1.{res,ptxas.log,sass}`; `/tmp/r2p8_ptx/li{1,2,-1}.nvdis`
  + `/tmp/r4mixed2_pcs_{fp8,fp4,mixed}_P132.csv` (production ncu, 2026-09-04
  15:27-15:28 UTC); scripts `/tmp/r5copy_region.py` (static, per
  inner/outer line, dump), `/tmp/r5copy_dyn.py` (executed per tile-CTA per
  line), `/tmp/r5copy_stall.py` (stall shares per region).
- Local copies: `/tmp/r5copy_f{1,2,m1}_kconv.txt` (static dumps of the K
  converter region with line attribution), `/tmp/r5copy_dyn_{fp8,fp4,
  mixed}_k.txt`, `/tmp/r5copy_dyn_fp8_v.txt`.
- No GPU job started; no checkout synced; no code changed in this worktree.
