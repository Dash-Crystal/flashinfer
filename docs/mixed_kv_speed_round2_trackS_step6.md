# Track S step 6 — [44] placement decode + owner-cut expansion + hoisted copy constants for the sm90 SPEC_DEC q=4 build (design, no build)

Tree `claude/mixed-kv-sm90-tma` @ `5cc416fd`, worktree `wt/S6`.  Design only: no kernel edit, no
remote build.  Scope is `csrc/xqa/mha.cu` (SPEC_DEC, `M_TILESIZE 16`, sm90, `CACHE_ELEM_ENUM 5`)
and the mixed copy/expansion helpers in `csrc/xqa/mhaUtils.cuh`.  `csrc/xqa/mha_sm90.cu` (q=1
kernel) and the FA3 headers are read for comparison only.

## 0. Where it stands and the question

After step 5 ([43]; backends.md "Track S step 5"), nkcut2 H200, locked medians, q=4, B=17,
S=4096, 8 KV heads, GQA 4, D=128, bf16: transport_a16 86.1, fp8 114.0, fp4 115.9, mixed 116.1
us.  Targets (targets.md): fp8 <= 94 (0.82x of today), fp4 <= 59 (0.51x), mixed <= 101 (0.87x).
Step 5's ncu on the four modules: 2 CTAs/SM, REG 124-128, STACK 0, no_instruction 0.14-0.51,
issue-active 50-58 %, warp-cycles per issued instruction 5.8-6.6, DRAM 15-28 % (a16 67 %), and
`smsp__inst_executed.sum` per launch **a16 29.85 M, fp8 42.76 M, fp4 50.85 M, mixed 51.47 M**.
Step 5's conclusion: the lever is executed instructions per tile.

The design question: which structural change takes the compressed tile to ~0.6x of its
instruction count.  Answer in one line: **there is no single structural change that does it;
the expansion's decode is the only component with a > 2x lever, and taking it (the bit-placement
decode the sm90 q=1 kernel already has, an owner cut that halves the per-block overhead, and
hoisted copy address constants) gives 0.66x (fp4) / 0.75x (fp8) / 0.78x (mixed) by arithmetic
— fp8 and mixed pass their targets, fp4 does not and cannot in an mma.sync kernel that
materialises A16 operands (section 5.3).**  The register-fragment decode (A/B), the scale fold
(C), the FP8-native MMA (D) and the larger tile (E) are each shown below to be a wash, illegal,
non-bit-exact, or smem-blocked.

Unit used throughout: **U = warp-instructions per 64-token tile summed over the CTA's eight
warps** = `inst_executed / 8704` (8,704 = 17 x 8 x 4096 / 64 tiles per launch), which is also the
instructions one SMSP executes per 256-token CTA tile.  Measured: **a16 3,430 U, fp8 4,913 U,
fp4 5,842 U, mixed 5,913 U** (the per-CTA prologue / merge / epilogue is inside these numbers,
amortised over 12.8 tiles per CTA: 8,704 / 680).  Issue rate implied by the wall time:
`inst / (132 SMs x 4 SMSP x 1.98 GHz x t)` = a16 0.33, fp8 **0.36**, fp4 0.43, mixed 0.42
instructions per SMSP-cycle; fp8 issues 15 % slower than fp4 on 16 % fewer instructions, which is
the short-scoreboard reading of step 5 (1.32 warps per issue cycle: the `LDS -> F2FP -> HADD2 ->
F2FP -> HMUL2` chain of the FP8 expansion).

## 1. Where the instructions go (source count, reconciled to ncu)

Geometry of the build (`mha.cu:100-172`, `:376-395`): `ctaShapeInWarps {4,1,2}`, `warpTile
{64,16}`, `kHeadPartBytes 128` (2 parts per K head), `cacheVTileSeqLen 32`, `grpLoadV` false on
sm90 (`defines.h:214-216`), `gemm1WarpsPerGrp 2`, `gemm1NbWarpGrps 2`, `nbCacheVTilesPerXTile 2`,
`nbXItersPerCtaTile 4`.  Per 256-token CTA tile: each gemm0 warp runs one 64-token tile in two
128 B parts (`runGemm0`, `mha.cu:2546-2660`); each gemm1 warp runs four iterations, each a
32-token x 64-column V tile (`mha.cu:3116-3350`).  So per 64-token tile (U): one gemm0 warp-tile
(2 parts) + four gemm1 warp-iterations.

Per-lane costs below are read from the source of the helpers that step 5 left in the hot loop;
the SASS totals of step 5 (part body 921 SASS with 32 HMMA / 20 LDSM / 10 LDGSTS, V body 888)
bound them but do not split them, so the first artifact of this step is the split itself
(section 6, artifact A0).  "lane-block" = one lane expanding one (token, 16-coefficient) block,
the ownership of `expandMixedPartialHeadsInPlace` (`mhaUtils.cuh:846-851`: `blockInPart = lane %
4`, `headInSpan0 = lane / 4`, rows step by 8 per iteration).

| component | where (lines) | per-lane cost today | count per U | U today, fp8 / fp4 / mixed / a16 |
|---|---|---|---|---|
| FP8 expansion, one lane-block | `expandCompressedBlock16WithScale` `mhaUtils.cuh:751-761` via `convertE4M3x2ToA16` `:502-519` (sm90 takes the `cvt.rn.f16x2.e4m3x2` -> `__half22float2` -> `__float22bfloat162_rn` route: F2FP, 2 HADD2.F32, F2FP per pair) + `mulA16x2` `:703-713`; scale `convertE4M3ScaleToA16Bits` `:593-598` + `broadcastA16Scale`; LDS.128 + LDS.U8 + 2 STS.128 `:874-886` | 8 pairs x (4 cvt + 1 HMUL2) = 40; scale 5; LDS 2; STS 2; address 2; loop/pred 2-4 = **~55-59** | K 16 lane-blocks (2 parts x 8) + V 16 (4 iters x 4) = 32 | fp8 **1,890**; mixed 1/3 of pages: 630 |
| FP4 expansion, one lane-block | same body `:762-771` via `convertE2M1x8ToA16` `:571-590`, which on `__CUDA_ARCH__ < 1000` is `cutlass::detail::_e2m1_to_bf16_x8` (32 SASS per 8 values, P0.4 [12]) | 2 x 32 + 8 HMUL2 + scale 5 + LDS 2 + STS 2 + 4 = **~85-88** | 32 | fp4 **2,800**; mixed 1/3: 930 |
| A16 page skip in the expansion | `:900-912` format select + branch | ~10 per page span | 8 K + 8 V spans | mixed 1/3: 50 |
| K copy, one 16-token page span | `copyMixedPartialHeadsAsync` `:319-424` body: `pageBase` (64-bit, `:337-339`), per iteration `blockInSpan/headInSpan/blockInPart/token/valid/elem/payloadElemOffset/firstSource` `:341-356` + swizzled `dst.at` `:395-397` + 1-2 LDGSTS; page/format `selectByIndex` chains + 3-way branch `:427-449` | ~16 (selects) + 3 + 4 + 2 iterations x ~24 = **~70-75** | 2 parts x 4 pages | **580** (all modes; static modules save the selects: ~-15 per span) |
| K scale copy | `:451-499`, 2 iterations x (page/format selects, 3-term 64-bit stride product, LDGSTS) | 2 x ~20 = 40 | 2 parts | 80 |
| V copy + scales | same helper, `maxNbCopiedHeads 32`, 2 spans + 1 scale iteration | 2 x 72 + 20 = ~165 | 4 iterations | **660** |
| gemm0 round fixed cost | `loadKTilePart` `:2370-2500` (tag broadcast `:2395`, flags `:2427-2431`, `isFullTile`, `nbHeadsAvail`, `loadPages` every other part `:2493-2499` = `getPage` LDG + `mixedPageTagLane` LDG + selects), `waitGroup/commitGroup/syncwarp`, `CircIdx`, rolled-loop control (`idxXTile`, `smemQOffset`, 8 R2UR / 35 UMOV noted in step 5) | ~120 per part | 2 | 240 |
| gemm0 MMA | `smemQKPartGemm` `:1425-1590`: per k16 step 1 LDSM.x4 (Q) + 4 LDSM.x4 (K) + 8 HMMA + ~6 address | 4 steps x 19 = 76 per part | 2 | 152 |
| gemm0 softmax + X hand-off | `rescaleAcc`, mask (last tiles only), `warpTileOnlineSoftmax` `:640-690` (32 FMNMX + shuffles + 32 x (FFMA, MUFU.EX2)), `toFp16` 16 F2FP, `computeRowSum` 4 HMMA + shuffles, `xBar.consumed.wait`, `storeOrderedGemmOutTile` 4 STSM, row max/sum stores, `produced.arrive` | ~220 | 1 | 220 |
| gemm1 iteration fixed cost | `:3131-3260`: `test_wait_parity`, `nextStep` carries, `loadVTilePart` setup + tag broadcast + flags `:2899-2935`, `advanceVPages` `:2836-2872` (+ `loadPages` = 2 LDG + selects), commit, `xBar.produced.wait` + row max/sum LDS + 2 ballots + rescale decision `:3178-3245`, `waitGroup`, `CircIdx`, `consumed.arrive`, loop | ~200-230 | 4 | **880** |
| gemm1 MMA | `smemXVPartGemm` `:1595-1860`: 2 k16 steps x (1 LDSM X + 4 HMUL2 X rescale + 4 LDSM.T V + 8 HMMA + ~5) | 2 x 22 = 44 | 4 | 176 |
| per-CTA prologue / multi-block merge (n = 5 sub-sequences) / epilogue, amortised | `:2130-2370`, `:3440-3600` | - | 1/12.8 CTA | residual |

Sums of the modelled rows: fp8 4,880 (measured 4,913), fp4 5,790 (5,842), mixed 4,590 + A16
spans 50 (5,913: residual 1,270 = the dynamic module's per-page branching/selects being
under-counted plus the amortised per-CTA fixed cost, which the static modules pay too), a16
3,000 (3,430).  The model is within 15 % on every module and the ordering it implies is the point:

- **Expansion is the largest component: 38 % of fp8, 48 % of fp4, ~28 % of mixed** (1/3 of the
  pages are A16), and its per-lane-block cost is 55-88 instructions for 16 values whose useful
  work is 8 packed multiplies and two 16 B stores.  The sm90 q=1 kernel took exactly this
  component from 262 -> 187 (fp8) and 385 -> 188 (fp4) executed SASS per lane-tile (wt/B [16],
  backends.md "Phase 2 [16]"); `mha.cu`'s expansion never received it — the helpers
  `e4m3x4ToBF16x2Pow2m120` / `e2m1x8ToBF16x2Pow2m126` / `e4m3x4ScalesToFloat` sit in
  `mhaUtils.cuh:656-689` and are used only by `mha_sm90.cu:2790-2830`.
- **Copy issue + address math is the second: ~22-25 %** (1,320 U), 5x the LDGSTS count (the
  ideal per FA3's A6 is ~6 per 16 B copy: address add, predicate, LDGSTS).
- **Bookkeeping ~19 %**, three quarters of it in the four gemm1 iterations per 64 tokens.
- **MMA + softmax ~9 %** (550 U) — the irreducible part at M 16.

Targets in U at today's issue rates: fp8 <= 4,050 (-860), fp4 <= 2,980 (-2,860), mixed <=
5,150 (-760).  fp4's target lies **below the a16 module's 3,430 U**, i.e. below a kernel that
copies A16 bytes and expands nothing; that is the arithmetic behind section 5.3.

## 2. Options A-F

Notation for the MMA fragments (PTX ISA m16n8k16 .bf16, `mma.cuh:44-50`): lane `l`, `g = l / 4`,
`q = l % 4`.  B fragment (K^T for gemm0, V for gemm1): `b0 = {B[k=2q][n=g], B[2q+1][g]}`,
`b1 = {B[2q+8][g], B[2q+9][g]}`; A fragment: `a0 = {A[m=g][k=2q..2q+1]}`, `a1 = A[g+8][2q..]`,
`a2 = A[g][2q+8..]`, `a3 = A[g+8][2q+8..]`.  A 16-coefficient scale block is exactly one k16
step of gemm0 (K's d) and exactly two n8 tiles (16 columns) of gemm1 (V's head columns).

### A. Decode K into the mma.sync B fragment (no STS/ldmatrix)

Packed FP8 row of token `t`: byte `d` at offset `d`.  The lane needs `d = 2q, 2q+1, 2q+8, 2q+9` of
tokens `g` and `g+8`.  `ldmatrix.x2` over the 16 packed rows (the form `loadMixedKPageFragment`
already uses, `mha.cu:1129-1130`) delivers bytes `4q..4q+3` of rows `g`, `g+8`, so the existing
code spends **2 SHFL per n-tile** (`:1136-1141`) to re-pair them.  The stock FP8-KV path avoids
the shuffles by permuting Q's columns once (`reorder16bQHeadsToMatch8bKCache`, `mhaUtils.cuh:
965-998`, called at `mha.cu:2530` when `cacheElemSize == 1`), but that Q order is wrong for A16
pages in the same tile; a second Q copy is 4,096 B and the SharedMem has 256 B of headroom under
the 2-CTA cap (step 5 table: 115,456 of 115,712), and permuting A16 K rows in smem is a 4 B-granular
shuffle (LDS.128 x2 + STS.128 x2 per 32 B) that costs more than it saves.

Count per lane per (page, k16 step) = 8 values, 2 HMMA, using the placement decode for both forms:

| | register decode (A) | expansion form with placement decode (this step) |
|---|---|---|
| loads | 1 LDSM.x2 (+2-4 SHFL, or the Q permutation problem) | 0.5 x (LDS.128 + LDS.U16) + 1 LDSM.x4 share |
| decode | 2 words x 6 = 12 | 0.5 x 24 = 12 |
| scale prep | 2 tokens x 1 block: 2 scales -> ~8 | 0.5 x 4 = 2 (the lane's 2 scales serve 32 values) |
| scale multiply | 4 HMUL2 | 4 HMUL2 |
| stores | 0 | 0.5 x 2 STS.128 = 1 |
| total | **~25-29** | **~20-21** |

Register decode doubles the scale preparation per value (a lane holds 2 tokens x 8 values
instead of 1 token x 32) and pays shuffles or the Q permutation, against saving one STS and half an
LDS per 8 values.  It is +20-40 % instructions, not a saving; it also brings back the per-page
format branch inside the unrolled MMA loop that cost 2.7x code / 5x branches on sm120
(`mha.cu:73-87`, backends.md "sm120 state" item 2), and the sm90 q=1 kernel closed the analogous
register-A route negatively ([35], backends.md "RS-decode in the GEMM groups").  **Rejected.**

### B. Decode V into the PV B fragment

V's B fragment needs tokens `k = 2q, 2q+1, 2q+8, 2q+9` of column `n = g`: four different packed
rows, one byte each (`loadMixedVPageFragment`, `mha.cu:1281-1319`: four byte loads + byte
assembly per word).  Per 8 values: 4-8 LDS.U8, 4 PRMT to assemble, 12 decode, scale prep for 4
tokens x 1 block (~12), 4 HMUL2 -> ~40, twice the expansion form.  `ldmatrix.trans` cannot help
(it transposes 16-bit elements, not bytes).  **Rejected**, same reasons as A plus the transpose.

### C. Fold the block scale into P, the accumulator, or Q

K: `S[r][t] = sum_b s_{t,b} (Q_b . k_b)[r][t]`; the scale depends on the token (N index) and on
the k16 block, so it can be folded neither into Q (t-dependent) nor into the accumulator across
blocks (b-dependent).  The legal form is per k16 step into a zeroed partial accumulator followed
by `acc += s * partial`: per (page, k16) per lane 2 n-tiles x 4 FFMA + 4 zero registers + scale
to fp32 = **8+ FFMA in place of 4 HMUL2** — more instructions, and an FP32 product order that is
not the reference's bf16 rounding.  V: `O[r][c] = sum_t P[r][t] s_{t,b(c)} v[t][c]`; the scale
depends on the summed index t, so the accumulator fold is illegal, and folding into P needs one
scaled P copy per 16-column block: 4 copies per 64-column warp tile = 32 HMUL2 per lane per V
tile — the same 32 HMUL2 the per-coefficient form pays (2048 values / 32 lanes / 2 per HMUL2),
plus 32 registers, and `bf16(P) x bf16(s)` rounds differently from the reference's
`bf16(v x bf16(s))` -> not bit-exact.  Exponent-only folding (IADD3 into the bf16 exponent) would
need power-of-two block scales, which R1 (E4M3 scales) freezes out.  **Rejected: wash or
illegal for K, wash and non-bit-exact for V.**

### D. FP8-native mma.sync (m16n8k32 e4m3) with Q quantised to E4M3

`mma.cuh:52-58` (`m16n8k32.row.col.f32.e4m3.e4m3.f32`) spans 32 k = two scale blocks with
different `s_{t,b}`, so the per-block scale cannot be applied after the MMA; the k16 e4m3 shape
(`mmaF8_k16`, `:64-72`) is `.kind::f8f6f4` on sm_120a, not sm90.  Zero-padding the upper k16 of
the B fragment restores per-block scaling at the price of half-empty MMAs: per (page, k16) per
lane 1 LDSM + 1 PRMT (zero half) + 2 MMA + 8 FFMA (post-scale, section C) + scale-to-fp32 prep
~6 + 4 zero regs = **~22**, equal to the placement expansion, with these numerics: Q rounded to
E4M3 (3 mantissa bits; relative error up to 2^-4 per element, SageAttention-style per-row Q
scale), so `S` differs from the A16 reference on **every** compressed page — the harness is
bit-exact (34/34 against A16 expansion) and would have to become a tolerance test for the
compressed modes; A16 pages in a mixed tile would still take the bf16 path (two Q operands live,
+16 registers).  FP4 pages have no sm90 MMA operand type at all.  **Rejected: no instruction
gain, not bit-exact, FP4 excluded.**

### E. Larger effective tiles per warp

Bookkeeping is ~19 % of U (section 1), three quarters in gemm1's four 32-token iterations per 64
tokens.  The M tile is fixed at 16 by `q x GQA = 16`.  A 64-row V tile halves gemm1's iteration
count (`cacheVTileSeqLen 64` -> `nbCacheVTilesPerXTile 1`, `nbXTilesPerXIter 2`) and saves ~440 U,
but the V ring becomes 65,536 B and SharedMem 147 KB (step 5 table: does not fit two CTAs);
paying for it with 64 B K parts (-32 KB) re-adds two gemm0 rounds (+240 U, and step 4 -> 5 measured
the four-round form as the slower one).  Net <= -200 U (3 %) at best.  **Not the lever.**  The
cheaper bookkeeping item — gemm1 re-gathering the page indices and tags that its gemm0 warp already
holds — is analysed in Appendix A (-60..-100 U, needs a 3-slot metadata ring and a copy-issue
reorder; not part of this step).

### F. What `mha_sm90.cu` does that this kernel can take cheaply

1. **Bit-placement decode with the power-of-two folded into the block scale**
   (`mha_sm90.cu:2779-2830` `expandE4M3BlockBF16` / `expandE2M1BlockBF16`, `foldScalesFinite`;
   used at `:2885-2970`; helpers `mhaUtils.cuh:656-689`).  E4M3 byte -> bf16 lane by PRMT + SHF +
   LOP3 per pair = `x * 2^-120` exactly (subnormals included); E2M1 nibble -> `mag * 2^-126`.
   The fold `bf16_rn(s * g * 2^120)` = `bf16_rn(s * g) * 2^120` exactly while finite, so
   `HMUL2(placed, folded)` = `bf16_rn(x * bf16_rn(s*g))` = the reference's single rounding
   (`convertE4M3x2ToA16` exact + `mulA16x2`): **bit-exact**, verified 52/52 on H200 in wt/B.
   Measured there: fp8 lane-tile 262 -> 187, fp4 385 -> 188 executed SASS; K expansion 1490 ->
   816 / 1440 -> 750 cycles.  Per pair: 4 SASS (3 + HMUL2) instead of 5-6 for FP8, 4 (folded) / 5
   instead of 9 for FP4.  This is the >2x lever on the largest component.  **Taken.**
2. **Four block scales prepared from one 32-bit word** (`e4m3x4ScalesToFloat`, `:2901`): 2 F2FP +
   4 HADD2.F32 + 4 FMUL + 2 F2FP pack + 2-4 PRMT for four scales instead of 5 per scale.  In the
   expansion form each lane needs only the scales of its own blocks; with the owner cut below a
   lane holds two adjacent blocks of one token, so one `LDS.U16` + the 2-scale half of this
   routine (~8) replaces 2 x (LDS.U8 + 5).  **Taken (2-scale form).**
3. **Lane cut with all address constants hoisted** (`ExpandLane`, backends.md "Phase 2 [16]"
   item 3: one IADD per stream per tile, the rest `[R+UR+imm]`).  `mha.cu`'s expansion already
   addresses by `tileBase + byteOffset<swizzle>(row, col)` with per-iteration constants
   (`mhaUtils.cuh:870-873`); the cut change below is what makes the swizzle term a lane constant
   too.  **Taken.**
4. The q=1 kernel's separate converter warp groups, its mbarrier protocol and the `setmaxnreg`
   split are the 5-warpgroup structure and are not adopted; `mha.cu` keeps compute-warp expansion
   under `cp.async` groups (`ldgsts::waitGroup<1>` `mha.cu:2579`, `:3554`).
5. Not adoptable: `mha_sm90.cu`'s A16 TMA landing (the K ring here is written by per-lane
   `cp.async`; TMA issue cost was the FA3 A1 finding) and its shared metadata chunk
   (`readTileMeta`, `:600-640`; Appendix A shows the `mha.cu` equivalent is worth ~1-2 %).

## 3. The design: [44] = F1 + F2 + owner cut + hoisted copy constants

### 3.1 Expansion: placement decode, fold vote, half-row owner cut

`expandMixedPartialHeadsInPlace` (`mhaUtils.cuh:813-917`) is rewritten for the sm90 bf16 build
(`__CUDA_ARCH__ == 900 && !INPUT_FP16`; every other build keeps the current body byte-for-byte,
as [43] did with its guard):

- **Owner cut.** A warp instruction still covers one page (the [40] invariant that makes the
  format warp-uniform: `blocksPerSpan = 16 tokens x 4 blocks = 64 = 2 x warp_size`).  Lane `l`
  owns token `l % 16` of the page and blocks `2h, 2h+1` with `h = l / 16` (32 values), instead of
  block `l % 4` of rows `l / 4 + 8i` (two lane-blocks in two iterations).  Its two packed sources
  are grains `4h` and `4h+2` of its row (FP8: 16 B each, `LDS.128` x2; FP4: 8 B each in the low
  half of those grains, `LDS.64` x2), its two scales are bytes `2h, 2h+1` of the row's 4 B scale
  word (`LDS.U16`; `scaleLoadBytes = 4` for a 128 B part or half-row, `mhaUtils.cuh:452`,
  `mha.cu:419-431`), and its outputs are grains `4h..4h+3` (4 x `STS.128`).
  Bank behaviour (`Array2D::byteOffset<swizzle>`, `utils.cuh:346-358`, `rowBytes 128 =
  cacheLineSize` -> physical grain `c ^ (r % 8)`): an 8-lane phase of `LDS.128`/`STS.128` is rows
  `8j..8j+7` at one logical grain -> eight distinct physical grains, conflict-free; since
  `r % 8 = (l % 16) % 8` the swizzle XOR is a **lane constant**, so every address is `laneBase +
  imm` (page `p` at `+ p * 16 * 128`, part/tile base once per call).  Today's cut has the same
  property but pays it per lane-block; the new cut pays the fixed part (scales, address) once per
  32 values.
- **Decode.** FP8 block: 4 x `e4m3x4ToBF16x2Pow2m120` (6 SASS each) -> 8 words, 8 `HMUL2` by the
  folded scale, 2 `STS.128`: **34**.  FP4 block: 2 x `e2m1x8ToBF16x2Pow2m126` (12 each) -> 8
  words, 8 `HMUL2` (folded) or 16 (fallback: x `0x7E807E80` then x scale, as
  `expandE2M1BlockBF16<false>`), 2 `STS.128`: **34 / 42**.
- **Fold decision** per warp per call (= per part for K, per V tile for V), as `foldScalesFinite`
  (`mha_sm90.cu:2832-2836`): fp32 `f_i = float(s_i) * g * 2^k` for the lane's two scales, `FMNMX`,
  `FSETP` against `255.5 * 2^k`, `VOTE.ALL`: ~5 instructions, then one of two loop bodies per
  format.  Host-side facts that decide how often the fallback runs: the sealer caps FP8 block
  scales at 128 (`fp4_kv_quantization.cu:32-34`, `:343-349`), so the FP8 fold holds for every tile
  with `|g| < 2`; FP4 block scales reach 448 (`:145-158`), so the FP4 fold is per tile (`|s g| < 4`,
  true for the bench's unit global scales and for typical activations' block maxima below 24).
  The dynamic module gets 4 bodies (2 formats x fold/fallback) of ~90 SASS each in place of the
  current 2 of ~180/~300; the hot-loop footprint stays at the step-5 level (2,476 dyn -> ~2,600).
- **Per call per lane** (32 values, FP8): 2 LDS.128 + 1 LDS.U16 + ~8 scale prep (2 F2FP + 2 HADD2
  + 2 FMUL + 1 F2FP + 2 PRMT... the 2-scale half of `e4m3x4ScalesToFloat` + `bf16x2BitsFromFloats`
  + 2 broadcast PRMT) + 5 vote + 2 x 34 + 3 address/loop = **~89 -> 44.5 per 16 values** (today
  55-59).  FP4: 2 LDS.64 + LDS.U16 + 8 + 5 + 2 x 34 + 3 = **~87 -> 43.5 per 16 values** (today
  85-88); fallback 51.5.  Cross-check against wt/B's measured lane-tile of 64 values: 187 fp8 /
  188 fp4 = 47 per 16 values including its tile glue — the estimates above are at that level.
- **Ownership / barrier invariants touched: none.**  The expanding warp is the copying warp
  (`loadKTilePart` and the expansion run on the same gemm0 warp, `mha.cu:2370` / `:2594`; the
  gemm1 warp copies and expands its own V tile, `:2899` / `:3272`); the only ordering is
  `cp.async.wait_group` on the warp's own groups before the expansion (`:2579`, `:3554`) and the
  `__syncwarp()` at the end of the helper (`mhaUtils.cuh:916`) before `ldmatrix` reads — both
  unchanged.  Within the new cut a lane reads only grains `4h, 4h+2` of its own row and writes
  grains `4h..4h+3` of its own row: no cross-lane hazard (FA3 D3 restated), so no
  `__syncwarp` is needed between loads and stores.  A16 pages: skipped as today (`kNeedsExpansion`
  and the per-page format branch, `:900-912`).  Zero-filled (invalid) blocks decode to zero as
  today ([29] note at `mhaUtils.cuh:326-328`).

### 3.2 Copy: hoisted per-page and per-lane constants

`copyMixedPartialHeadsAsync` (`mhaUtils.cuh:286-499`) keeps its ownership (lane = block `l % 4`
of rows `l / 4 + 8i`: coalesced 16 B / 8 B lanes per token row, the A2/D6 finding) and its
page-outer dispatch; the change is which quantities are computed where:

- Per lane, once per kernel (registers): `blockInPart = l % 4`, `headInSpan0 = l / 4`,
  `swz = (2 * blockInPart) ^ (headInSpan0 % 8)` (the destination grain, lane-constant by the same
  argument as 3.1), `payloadElemOffset(blockInPart, idxPart)` per format (3 values), the scale-copy
  lane's `token = l % 16`, `localPage = l / 16`.
- Per page span: `pages`/`formats` `selectByIndex` (unchanged, 16), the branch (3), `pageBase +
  token0 * stride_token` (2 IMAD.WIDE), then per iteration `src = pageSpanBase + iteration * (8 *
  stride_token)` with `8 * stride_token` a kernel-wide 64-bit constant (1 IADD3.64 = 2), `valid`
  (2 ISETP/PLOP), 1-2 LDGSTS with `dst = laneDstBase + span * 2048 + iteration * 1024` as an
  immediate: **~7 per iteration, ~37 per span** (today ~70-75).
- Scale copy: the 3-term 64-bit stride product per lane becomes `scaleBaseForPage + token *
  scale_stride.token` (the page/head terms once per page: 2 IMAD.WIDE, the token term 1 IMAD.WIDE);
  ~12 per iteration (today ~20).
- Verification of the shape, not the stopwatch: SASS of one copy body shows no `IMAD.WIDE` inside
  the iteration except the source add, `LDGSTS` with `[R+imm]` destinations, and the FA3 A6 ratio
  (~6-8 instructions per LDGSTS in the body).

Estimated U: K copy 580 -> 300, K scales 80 -> 50, V copy+scales 660 -> 380: **-590 U** on every
mode (the static modules gain ~-450: they had no selects to begin with).  This is the item with the
widest error bar (the step-5 SASS gives 10 LDGSTS in a 921-instruction K part body but not the
executed count per LDGSTS); artifact A0 measures it before the copy change is built (section 6).

### 3.3 What is not changed

K/V ring shapes, `SharedMem` layout (115,456 B; the new code adds no member), the 2-CTA/SM
launch bounds (`mha.cu:3616-3624`), the `nbSubSeqPerSeq` host model (`:3852-3940`; n = 5 stays
optimal because per-tile work shrinks uniformly), the rolled loops of [43], the gemm0/gemm1 MMA
bodies, softmax, the X hand-off, `mha_sm90.cu`, FA3, sm120 modules (guards keep their SASS
byte-identical — the step-5 proof-by-construction is repeated).

### 3.4 Register and smem budgets at 2 CTAs/SM

Registers (cap 128, today 124-128 with STACK 0): the expansion live set becomes 8 packed words
(FP8; 4 for FP4) + 2 scale words + 8 output words + ~6 temporaries + 3 bases = ~27, against
today's cutlass LUT body that P0.4 measured re-materialising its constants for lack of registers
(31 `IMAD.U32 R, RZ, RZ, UR` per lane-tile at 48 registers in the q=1 kernel; at 128 here the
LUT constants were live and are freed).  The copy hoist adds ~6 lane constants held across the
tile loop (3 per-format offsets, 2 dst bases, 1 swizzle) and 2 for `8 * stride_token`.  Net
expectation: REG stays <= 128 with STACK 0; the accept rule is `cuobjdump -res-usage` REG <= 128,
STACK 0, LDL 0, STL 0 on all four sm90 q=4 modules (as step 5).  Smem: unchanged (no new member;
the fold vote and the scale pairs are registers).  Hot-loop SASS: dyn 2,476 -> predicted 2,500-2,800
(two extra fold bodies, smaller copy bodies); no_instruction must stay <= 0.8 (step 5's 0.51).

## 4. Budgets before / after (U) and predicted wall time

| component | fp8 today -> after | fp4 today -> after | mixed today -> after | a16 |
|---|---|---|---|---|
| expansion (32 lane-blocks of 16 values; mixed = 1/3 fp8 + 1/3 fp4 + 1/3 A16 skip) | 1,890 -> **1,424** (44.5 x 32) | 2,800 -> **1,392** (43.5 x 32; fallback tiles 1,648) | 1,610 -> **990** | 0 |
| copy + scales (K 2 parts, V 4 iterations) | 1,320 -> 730 | 1,320 -> 730 | 1,320 -> 730 | 1,320 -> 730 |
| gemm0 rounds + gemm1 iterations fixed | 1,120 | 1,120 | 1,120 | 1,120 |
| MMA + softmax + X hand-off | 550 | 550 | 550 | 550 |
| residual (per-CTA fixed, dynamic dispatch, model error) | 33 | 52 | 1,313 | 440 |
| **total** | **4,913 -> 3,857 (0.79x)** | **5,842 -> 3,844 (0.66x)** | **5,913 -> 4,703 (0.80x)** | 3,430 -> 2,840 |

Wall time.  The kernel is issue-paced with a latency component that does not shrink with the
count (long scoreboard 0.9-1.4 warps per issue cycle = K/V landing waits; the a16 module is
DRAM-side at 67 %).  Two readings bracket the prediction: (i) issue-proportional, `t x U_new /
U_old`; (ii) the step-4 -> 5 calibration, where -5.6 % instructions (fp8) gave -8 % time and +2.5 %
(mixed) gave -16 % time because the stall mix changed — i.e. time tracks `U x cycles-per-issue`,
and the FP8 chain's short-scoreboard stall (1.32) goes away with the chain (the placement decode
is 3 independent INT ops + 1 HMUL2 per pair against a 5-deep F2FP/HADD2 dependency; the [16] trace
on the q=1 kernel measured the expansion segment shrinking with the count, 1440 -> 750 cycles for
385 -> 188 SASS, "count-bound not MIO-bound").

| mode | today | (i) proportional | (ii) with fp8's issue rate recovering to fp4's 0.43 | predicted band | target | verdict |
|---|---|---|---|---|---|---|
| fp8 | 114.0 | 89.5 | 75 | **80-92** | <= 94 | pass (2-14 us margin) |
| fp4 | 115.9 | 76.3 | 76 | **74-84** | <= 59 | **open** (1.25-1.42x) |
| mixed | 116.1 | 92.3 | 88 | **86-96** | <= 101 | pass |
| transport_a16 | 86.1 | 71 (issue) but DRAM floor 67.5 + fixed | - | **80-86** | 135 | pass (unchanged within noise expected: DRAM-side) |

The DRAM check: fp8 at 80 us moves 152 MB -> 1.9 TB/s (38 % of the 4.5 TB/s sustained probe), fp4
at 74 us 80 MB -> 1.1 TB/s, mixed at 86 us 172 MB -> 2.0 TB/s: none approaches the byte floor, so
the prediction is not DRAM-capped.

Accept/reject numbers for the step (three locked rounds, `--repeats 2 --trials 5`, medians;
co-tenant rule 2 x t < 1.5 ms holds): **accept if fp8 <= 94 and mixed <= 101 and fp4 <= 90 and
a16 within +-3 us of 86; reject the step (revert the tree) if any compressed mode is above 1.05x
its step-5 record or a16 regresses > 5 %.**  fp4 in 74-90 is the expected outcome, not a pass of
its target — see 5.3.

## 5. Risks, and what fp4 <= 59 would take

### 5.1 Main risks

1. **The copy estimate (3.2) is the least measured number.**  If A0 shows copy + scales at < 700 U
   today, the hoist is worth < 300 U and the fp8 prediction moves to 88-96 (marginal against 94).
   Mitigation is the order of work: A0 first, then decode, then copy only if A0 confirms >= 1,000 U.
2. **ptxas at the 128 cap.**  Two extra loop bodies per format and six hoisted lane constants; if
   REG hits 128 with STACK > 0 the build is rejected by construction (section 6), and the fallback
   is the single-body form (always two multiplies, 5 per pair for both formats: +8 HMUL2 per FP8
   lane-call, i.e. fp8 1,424 -> 1,552 U) which needs no vote and no second body.
3. **Fold fallback frequency on real caches.**  FP4 tiles with `|s g| >= 4` take +8 HMUL2 per
   lane-call (+256 U on an all-fallback fp4 stream: still 0.70x).  Bit-exactness does not depend
   on the branch; only time does.  The bench (unit global scales) never takes it, so a
   max-scale regime case is in the matrix (as wt/B's 52-case set).
4. **Hot-loop footprint.**  +2 bodies x 2 formats; if no_instruction returns above 0.8, the
   mixed module should drop the fold bodies (fallback-only, risk 2's form) before anything else.
5. **The a16 copy body** takes the same hoist (the dynamic module has no separate A16 copy path,
   [40]); a16 is DRAM-side, so its time should not move — a move of > 5 % either way means the
   LDGSTS coalescing changed (check `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ldgsts` per
   LDGSTS: must stay at the [43] ratio).

### 5.2 Barrier / ownership invariants (restated, all unchanged)

K: gemm0 warp `w` copies (`cp.async`, `commitGroup`) part `p+1` into buffer `next`, waits
`wait_group 1` for part `p`, expands buffer `curr` in place (its own lanes' data, `__syncwarp` at the
end), reads it with `ldmatrix`, advances `CircIdx` (`mha.cu:2570-2645`).  V: gemm1 warp copies
tile `i+1`, `wait_group 1` for tile `i`, expands its own 32 x 128 B tile, `ldmatrix.trans`, arrives
`xBar.consumed` once per X tile (`:3131-3348`).  Formats/flags per buffer are written by lane 0 of
the owning warp before the copy and read by the same warp after the wait (`:2427-2431`,
`:2927-2934`).  The new cut changes which lane writes which grains, never which warp.

### 5.3 fp4 <= 59 is out of reach of this kernel; what would reach it

With the expansion at zero cost, fp4 would be 5,842 - 2,800 = 3,040 U -> 60 us at today's issue
rate: the target requires **removing the expansion entirely and losing nothing else**.  Anything
that materialises bf16 operands in shared memory pays >= 34 instructions per 16 values (8 HMUL2 +
2 STS + the cheapest decode; a PRMT byte-LUT alternative was counted at 16-18 per 8 values against
the placement's 12-16 and is not better).  What would reach 59: (a) a native narrow-operand MMA
consuming packed E2M1 with the block scale applied by the tensor core — `kind::f8f6f4` /
block-scaled MMA on sm120/sm100, not sm90; (b) a consumer that is DRAM-bound at the fp4 byte count,
i.e. one whose per-tile instruction count is <= ~0.5x a16's 3,430 U — the `mha_sm90.cu` wgmma
consumer with converter warps is the closest existing structure and sits at 96 -> 73 us at q=1
after [16] with a 1.09 us/tile consumer floor (P0.3), so a SPEC_DEC port of it (step 3's rejected
route, K3/V2 at 99 KB fits 2 CTAs/SM) would land around 75-95 us, not 59.  The honest statement for
the targets table: **fp4 q=4 on sm90 is bounded near 75 us by the A16-materialising mma.sync
structure; 59 needs a different MMA operand path.**  fp8 and mixed do not have this problem
because their targets are 0.82x / 0.87x.

## 6. Verification artifacts (mechanism first, stopwatch last)

- **A0 (zero-code, before any edit; decides the order and the copy item):** `-lineinfo` build of
  the four sm90 q=4 modules (byte-identical SASS to production, as P0.4 did), ncu
  `--section SourceCounters --import-source yes` on one launch, `smsp__inst_executed` bucketed by
  helper: `expandMixedPartialHeadsInPlace` (K, V), `copyMixedPartialHeadsAsync` payload loop,
  scale loop, `smemQKPartGemm`, `smemXVPartGemm`, softmax, `loadKTilePart`/`loadVTilePart` glue,
  prologue/merge.  Accept the model if expansion is >= 35 % (fp8) / >= 45 % (fp4) and copy+scales
  >= 18 % of `inst_executed`; the budget table in section 4 is then re-based on A0's numbers.
- **A1 (SASS class counts, per module, `cuobjdump -sass` of `xqa_mha.cuda.o`, hot loops delimited
  by back-edges as in step 5):** in the expansion bodies `F2FP` per FP8 lane-call 18 -> <= 3,
  `HADD2.F32` 16 -> <= 4, `PRMT`+`SHF`+`LOP3` = 3 per pair (24 per block), `HMUL2` 8 per block (16
  in the fallback body), `STS.128` 4 per lane-call, `LDS.128` 2 (FP8) / `LDS.64` 2 (FP4) + `LDS.U16`
  1; no `LDL`/`STL`; REG <= 128, STACK 0.  Copy bodies: `LDGSTS` count unchanged (K 2 A16 + 2 FP8
  + 1 FP4 + 1 scale per span-iteration shape as [29]), `IMAD.WIDE` <= 1 per iteration, dyn hot
  loop <= 2,800 SASS.
- **A2 (ncu, one launch, `--launch-skip 1 --launch-count 1`, all four modules):**
  `smsp__inst_executed.sum` fp8 42.8 M -> 32-35 M, fp4 50.9 M -> 32-35 M, mixed 51.5 M -> 40-43 M,
  a16 29.9 M -> 24-27 M; `smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active`
  fp8 1.32 -> <= 0.7; `smsp__inst_executed_pipe_xu` (conversion pipe) fp8 down by >= 4x in the
  expansion region (if this counter does not move, the chain latency was the cause and the
  short-scoreboard drop is the evidence); `no_instruction` <= 0.8; `launch__occupancy_limit_*` 2 /
  2; `launch__shared_mem_per_block_dynamic` 115,456 B unchanged; LDGSTS wavefronts per instruction
  unchanged (5.1 item 5).
- **A3 (correctness):** `tests/attention/run_xqa_mixed_page_transport.py` 34/34 bit-exact on
  nkcut2 (default and `XQA_NB_SUB_SEQ=2`) plus a max-block-scale FP4 case (fold fallback path) and
  an E4M3-subnormal case, both bit-exact against the A16-expansion reference; ws-1 sm120 modules
  SASS byte-identical to a pristine build (guards), 34/34 there too.
- **A4 (bench rows, three locked rounds, `--repeats 2 --trials 5`, q=1 control included):**
  accept/reject as section 4: fp8 <= 94, mixed <= 101, fp4 <= 90, a16 86 +- 3, q=1 control
  unchanged (82 / 77 / 73 / 83 after wt/B).  Predicted rows: a16 80-86, fp8 80-92, fp4 74-84,
  mixed 86-96.

## 7. Do not build if

1. A0 puts the expansion below 30 % of `inst_executed` on fp4 or below 25 % on fp8 — the decode
   lever would then be worth < 0.85x and the fp8/mixed targets would need the copy item alone,
   which the model does not support.
2. A0 puts copy + scales below 12 % — build the decode only (3.1); skip 3.2.
3. The placement decode is not bit-exact against `convertE4M3x2ToA16` + `mulA16x2` on an
   exhaustive (256 codes x scale regimes incl. subnormal inputs and the fold boundary) device
   test — the wt/B exactness argument is for bf16 math; this build is bf16 (`INPUT_FP16 0`), and an
   fp16 build must keep the current body (guard by `!INPUT_FP16`).
4. `cuobjdump -res-usage` of the edited build shows STACK > 0 or LDL/STL > 0 on any sm90 q=4
   module, or REG > 128 (2 CTAs/SM lost) — revert to the single-body (fallback-only) form before
   re-measuring; if that still spills, stop.
5. The sm120 modules' SASS changes (guard leak) — fix the guard before any timing.
6. Another track has changed `expandMixedPartialHeadsInPlace` / `copyMixedPartialHeadsAsync`
   ownership or the K/V ring shapes in `mha.cu` since `5cc416fd` (Track W shares the file);
   re-derive section 1 on the merged tip first.
7. The step-6 result is judged against fp4's 59 us: it is not that step's target (5.3); building
   it "for fp4" alone is not justified — its value is fp8 and mixed passing, and fp4 0.66x.

## Appendix A — gemm1 metadata sharing (analysed, not built)

The gemm1 group `g` of X tile `x` needs pages `4x + 2g, 4x + 2g + 1` of the CTA tile — two of the
four pages gemm0 warp `x` prefetched two tiles ahead (`loadPages`, `mha.cu:2340-2368`; tags one
tile ahead via `mixedPageTagLane` / `broadcastMixedPageTags`, `mhaUtils.cuh:205-226`).  gemm1
re-gathers them (`:2814-2872`): 1 LDG (indices) + 1 LDG (tags) + `selectByIndex` chains + shuffles
per 32-token V tile, ~15-25 instructions x 4 per U (60-100 U, 1-2 %).  Sharing needs: a
`{pages[4], tags}` slot per gemm0 warp written at `loadKTilePart(t-1, part 1)` (inside
`runGemm0(t-2)`, when the tags of tile `t` have landed, before `xBar.produced.arrive(t-2)`), read by
gemm1 after `xBar.produced.wait(t-2)`, so the copy of `V(t)` can still be issued at the start of
gemm1's iteration `t-1` as today (`:3169` before `:3178`).  Lifetime: gemm0 overwrites slot `t % S`
with tile `t+S` inside `runGemm0(t+S-2)`, which follows its `xBar.consumed.wait(t+S-3)`
(`:2727`, waited before storing `X(t+S-2)`), i.e. gemm1 has finished iteration `t+S-3` >= `t-1`
(its last read of slot `t`) iff `S >= 2`; `S = 3` gives one tile of slack: 3 x 20 B x 4 warps =
240 B <= the 256 B headroom.  Deferred: the saving is 1-2 % and it adds a cross-warp-role
ownership rule to a protocol that today has exactly one (the X buffer).

## Appendix B — option D numerics, for the record

E4M3 quantisation of Q per row (`q_scale[r] = max_d |Q[r][d]| / 448`) gives `|dQ| <= 2^-4 |Q|`
per element worst case (3 mantissa bits, round-to-nearest); `S = Q K^T` over D = 128 accumulates
an error of order `2^-4 / sqrt(128) x |S|` typical, `2^-4 |S|` worst case, comparable to the
E2M1 payload's own quantisation error and 8x the bf16 reference's.  No compressed page would be
bit-exact against the A16-expansion reference; the harness would need an fp32-reference tolerance
(as the FA3 `test_fa3_mixed_page_transport.py` A16 cases do not).  FP4 pages have no e2m1 operand
on sm90 (`mma.sync` narrow types are `.e4m3/.e5m2` at k32 for sm_89+; `.e2m1` only under
`.kind::f8f6f4` on sm_120a).  The instruction count is a wash (section 2, D), so the numerics
buy nothing.
