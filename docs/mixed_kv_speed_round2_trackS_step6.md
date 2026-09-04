# Track S step 6 — [44] placement decode + owner-cut expansion + hoisted copy constants for the sm90 SPEC_DEC q=4 build (design rev 2; code as written in section 8)

Tree `claude/mixed-kv-sm90-tma` @ `5cc416fd`, worktree `wt/S6`.  Design (rev 2 after the judge
round) plus the code as written (section 8); no remote build or run from this worktree.  Scope is `csrc/xqa/mha.cu` (SPEC_DEC, `M_TILESIZE 16`, sm90, `CACHE_ELEM_ENUM 5`)
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
hoisted copy address constants) gives 0.69x (fp4) / 0.83x (fp8) / 0.82x (mixed) by arithmetic on
the step-5 SASS segment counts (section 1, revision 2) — mixed passes its target with margin,
fp8 passes only if the FP8 chain's short-scoreboard stall goes with the chain (section 4,
reading (ii)), fp4 does not and cannot in an mma.sync kernel that materialises A16 operands
(section 5.3).**  The register-fragment decode (A/B), the scale fold (C), the FP8-native MMA (D)
and the larger tile (E) are each shown below to be a wash, illegal, non-bit-exact, or
smem-blocked.

Revision 2 (judge round, before code): the section-1 component table is re-based on the step-5
SASS segments (the static fp8/fp4 bodies are straight-line, so static = executed; the earlier
source-read counts were 5-10 % high on the expansion and ~2x low on the bookkeeping); the fold
vote gains the `|g| >= 2^-117` term; the owner cut gains a `__syncwarp()` after the
`cp.async.wait_group` (cross-lane visibility) and a template-bool guard derived from the step-5
predicate; 5.3 is restated as a floor argument.  Every change is marked "rev 2".

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

**Rev 2 — re-based on the step-5 SASS.**  The per-lane costs of revision 1 were read from the
helper source; the step-5 SASS (nkcut2 `/tmp/mixedkv-wtE-s5-v1-sass`, hot loops delimited by
back-edges, bodies split at the `DEPBAR` (wait_group) / first `LDS.128|LDS.64` after it / last
`STS.128` / first `LDSM`) gives the split directly, and for the static fp8/fp4 modules the part
and V bodies are straight-line (fully unrolled spans and iterations, no data-dependent branch), so
**static = executed** there.  "lane-block" = one lane expanding one (token, 16-coefficient) block,
the ownership of `expandMixedPartialHeadsInPlace` (`mhaUtils.cuh:846-851`: `blockInPart = lane %
4`, `headInSpan0 = lane / 4`, rows step by 8 per iteration).

| SASS segment (step-5 modules) | fp8 | fp4 | a16 | dyn (static SASS of the rolled loops) |
|---|---|---|---|---|
| K part body (`runGemm0` inner, 2 per U) = pre-expansion (copy + scales + round fixed) + expansion + MMA | **909** = 390 + 424 + 95 | **1,136** = 387 + 654 + 95 | **578** = 477 + 0 + 101 | 921 = 502 + 268 + 151 |
| V iteration body (gemm1 inner, 4 per U) | **652** = 326 + 214 + 112 | **771** = 323 + 336 + 112 | **443** = 368 + 0 + 75 | 888 = 469 + 297 + 122 |
| gemm0 outer loop minus the part body (static; softmax, mask, X hand-off, tile control) | 682 (32 MUFU.EX2, 32 FFMA, 40 FMNMX, 34 FMUL, 17 F2FP, 16 LDG, 127 IMAD, 112 LOP3, 84 ISETP, ...) | - | 627 | - |
| expansion classes, fp8 K part | HADD2.F32 136, F2FP 72 + 72, HMUL2 64, IMAD 20, LOP3 16, STS.128 16, LDS.128 8, LDS.U8 8, FMUL 8 | PRMT 192, LOP3 144, IMAD 138, HMUL2 64, SHF 48, STS.128 16, LDS.64 8, LDS.U8 8, F2FP 16, HADD2 8, FMUL 8 | - | - |
| copy iteration (one LDGSTS + its address / predicate glue; 3 `@!PT LDS RZ,[RZ]` placeholders per LDGSTS are ptxas-emitted and not hoistable) | K 25, V ~28 | same | ~20 per LDGSTS (2 per block) | per span ~20 common + one 70-84 format body |

Per-lane-block expansion: fp8 424 / 8 = **53** (rev 1 said 55-59), fp4 654 / 8 = **81.75** (rev 1
85-88); V 214 / 4 = 53.5, 336 / 4 = 84.  Cross-check of the method: the fp4 - fp8 expansion delta
per U is 2 x (654 - 424) + 4 x (336 - 214) = 948 U against the measured `inst_executed` delta
(50.85 - 42.76) M / 8,704 = 929 U; the attribution is sound and the revision-1 numbers were 5-10 %
high on the expansion and about 2x low on the bookkeeping.

Components per U (2 K parts + 4 V iterations; "measured - bodies" is the executed outer-loop
share, which the static 682 only bounds because the mask / rescale branches are not all taken):

| component | fp8 | fp4 | a16 | mixed (derived) |
|---|---|---|---|---|
| expansion (32 lane-blocks) | 2 x 424 + 4 x 214 = **1,704 (35 %)** | 2 x 654 + 4 x 336 = **2,652 (45 %)** | 0 | 1/3 fp8 + 1/3 fp4 spans + A16 skip ~50 = **~1,500 (25 %)** |
| copy + scales (K 8 iterations x 25 + 2 scale iterations x 20; V 4 x 28 + 20) | 2 x 240 + 4 x 132 = **1,008 (21 %)** | ~1,000 | a16 pre-expansion is +87 (K) / +42 (V) over fp8's: 1,008 + 342 = **1,350** | copy + per-page format / page selects + branches: 5,913 - 1,500 - 1,084 - 638 - 490 = **~2,200** (the static copy plus ~1,200 U of dynamic dispatch = revision 1's "residual 1,313", now named) |
| round / iteration fixed cost (pre-expansion minus copy: `loadKTilePart` / `loadVTilePart` glue, tag broadcast, flags, `loadPages`, wait_group, CircIdx, rolled-loop control) | 2 x 150 + 4 x 196 = **1,084 (22 %)** | ~1,080 | ~1,086 | ~1,100 |
| MMA segments (LDSM + HMMA + address; 95 per part / 112 per V iteration, not the 76 / 44 of rev 1) | 190 + 448 = **638** | 638 | 202 + 300 = 502 | ~640 |
| outer loops (softmax, X hand-off, gemm1 rescale decisions, tile control) + per-CTA prologue / merge / epilogue amortised over 12.8 tiles = measured - bodies | 4,913 - 4,426 = **487** | 5,842 - 5,356 = **486** | 3,430 - 2,928 = **502** | ~490 |
| **total (measured)** | **4,913** | **5,842** | **3,430** | **5,913** |

The remainder is 486-502 U on all three static modules, i.e. format-independent as it must be;
that is the check that the segment cut is right.  What the table says:

- **Expansion is the largest component: 35 % of fp8, 45 % of fp4, ~25 % of mixed** (1/3 of the
  pages are A16), and its per-lane-block cost is 53-84 instructions for 16 values whose useful
  work is 8 packed multiplies and two 16 B stores.  The sm90 q=1 kernel took exactly this
  component from 262 -> 187 (fp8) and 385 -> 188 (fp4) executed SASS per lane-tile (wt/B [16],
  backends.md "Phase 2 [16]"); `mha.cu`'s expansion never received it — the helpers
  `e4m3x4ToBF16x2Pow2m120` / `e2m1x8ToBF16x2Pow2m126` / `e4m3x4ScalesToFloat` sit in
  `mhaUtils.cuh:656-689` and are used only by `mha_sm90.cu:2790-2830`.
- **Bookkeeping is the second: 1,084 + 487 = ~1,570 U (32 % of fp8)**, 784 of it in the four
  gemm1 iterations per 64 tokens (196 per iteration).  Revision 1 modelled it at 1,340 and put the
  difference into "residual".
- **Copy issue + address math: ~1,000 U (21 %)** in the static modules, ~2,200 in the dynamic one;
  25 SASS per LDGSTS against a floor of ~9 (address add, predicate/select, LDGSTS, 3 placeholders).
- **MMA ~640 U (13 %)** — the irreducible part at M 16, plus the outer loops' ~490.

Targets in U at today's issue rates: fp8 <= 4,051 (-862), fp4 <= 2,974 (-2,868), mixed <=
5,144 (-769).  fp4's target lies below the a16 module's 3,430 U — but a16's count is not a
floor for fp4 (it carries a 342 U larger copy and drops to ~2,840 after the hoist); the floor
argument is made properly in section 5.3.

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
   under `cp.async` groups (`ldgsts::waitGroup<1>` `mha.cu:2579`; V: `syncVTileLoad`
   `mha.cu:3071-3083` = `waitGroup<nbVBuffers-1>`, called at `:3244` — rev 2: not `:3554`, which is
   the multi-block merge loop).
5. Not adoptable: `mha_sm90.cu`'s A16 TMA landing (the K ring here is written by per-lane
   `cp.async`; TMA issue cost was the FA3 A1 finding) and its shared metadata chunk
   (`readTileMeta`, `:600-640`; Appendix A shows the `mha.cu` equivalent is worth ~1-2 %).

## 3. The design: [44] = F1 + F2 + owner cut + hoisted copy constants

### 3.1 Expansion: placement decode, fold vote, half-row owner cut

`expandMixedPartialHeadsInPlace` (`mhaUtils.cuh:813-917`) is left byte-for-byte as it is; a
second helper with the new body is selected at the two call sites by a template bool.

- **Guard (rev 2).**  `mha.cu` defines `MIXED_BF16_PLACEMENT_EXPANSION = ENABLE_MIXED_KV_CACHE
  && MIXED_COMPACT_TILE_LOOPS && !INPUT_FP16 && !(GRP_LOAD_V)` (and `kMixedBF16PlacementExpansion`
  from it, `static_assert`ed to imply `kCompactTileLoops && is_same<InputElem, __nv_bfloat16> &&
  !grpLoadV && !compactMixedPages`), i.e. derived from the step-5 predicate `__CUDA_ARCH__ == 900
  && CACHE_ELEM_ENUM == 5 && SPEC_DEC && M_TILESIZE == 16` (`mha.cu:147-169`,
  `MIXED_COMPACT_TILE_LOOPS`) plus the bf16 requirement of the placement decode; a bare
  `__CUDA_ARCH__ == 900 && !INPUT_FP16` (rev 1) would have handed the new cut to the other sm90
  `mha.cu` instantiations (64 B parts, `cacheVTileSeqLen 64`, `grpLoadV` V paths with
  `headsPerWarp` row offsets), whose geometry it does not fit.  Both call sites (`mha.cu:2594` K,
  `:3272` V) select the new helper under `#if MIXED_BF16_PLACEMENT_EXPANSION` — a preprocessor
  guard rather than `if constexpr`, because the V site is in the kernel body (not a template):
  a discarded `if constexpr` branch there is still instantiated, the new helper's
  `static_assert`s would fire in the other builds, and the preprocessed source of those builds
  must stay identical.  So every other build keeps the old body byte-for-byte (the q=1 sm90 and
  all sm120 modules never instantiate the new helper; `mla_sm120.cu` includes `mhaUtils.cuh` too
  and is covered by the same argument — the SASS comparison of section 6 A3 checks all of them).  The new
  helper `static_assert`s what the cut needs: `partBytes == 128` (`blocksPerPart == 4`,
  `blocksPerSpan == 64 == 2 * warp_size`, one pass per span), `headsPerSpan == 16 ==
  tokensPerPage`, `scaleLoadBytes == 4`, `nbWarps == 1` (no `idxWarp` parameter), `Tile::rowBytes
  == 128` and `swizzle` (the XOR is `c ^ (r % 8)`, `utils.cuh:346-358`), `sizeof(LdGrain) == 16`,
  bf16 `InputElem`; it takes no `dstHeadOffset` (the tile origin is row 0 — both call sites pass
  the literal 0 today, `mha.cu:2594-2599`, `:3272-3278`) and no `idxWarp`.  Any other geometry
  fails to compile rather than silently taking a wrong cut.
- **Owner cut.**  A warp instruction still covers one page (the [40] invariant that makes the
  format warp-uniform: `blocksPerSpan = 16 tokens x 4 blocks = 64 = 2 x warp_size`).  Lane `l`
  owns token `tok = l % 16` of the page and blocks `2h, 2h+1` with `h = l / 16` (32 values),
  instead of block `l % 4` of rows `l / 4 + 8i` (two lane-blocks in two iterations).  Its two
  packed sources are logical grains `4h` and `4h+2` of its row (the copy places compressed block
  `b` in grain `2b`, `mhaUtils.cuh:395-421`; FP8: 16 B each, `LDS.128` x2; FP4: 8 B each in the
  low half of those grains, `LDS.64` x2), its two scales are bytes `2h, 2h+1` of the row's 4 B
  scale word (`LDS.U16`; the copy stores `scaleGroup = idxPart * 4` at row stride 4,
  `mhaUtils.cuh:452-499`, `mixedVScaleBytes 4` in this build, `mha.cu:419-431`), and its outputs
  are logical grains `4h..4h+3` (4 x `STS.128`).  The two lanes sharing a row (`h = 0, 1`) touch
  disjoint grains.
  Addresses (rev 2, corrected): row `r = span * 16 + tok`, so `r % 8 = tok % 8 =: x` is a lane
  constant and the physical grain of logical grain `4h + j` is `(4h + j) ^ x = (4h ^ x) ^ j`
  (`4h` and `j < 4` occupy disjoint bits).  The four physical grains are a permutation of one
  4-aligned group that depends on `x & 3`, so they are **four lane-constant u32 addresses** `a_j =
  tileBase + tok * 128 + ((4h ^ x) ^ j) * 16`, each used as `[a_j + span * 2048]` — not a single
  `laneBase + imm` as rev 1 wrote; the loads use `a_0`, `a_2`.  Cost: 4 LOP3/IMAD once per call
  (hoistable across the unrolled spans of the static modules; in the rolled dynamic loop 4 live
  registers), the scale address `scaleBase + tok * 4 + 2h` one more.  The 2048 / 64 span strides
  are `Tile::rowBytes * 16` / `scaleLoadBytes * 16` template constants (K and V tiles both have
  128 B rows in this build: `KSmemBuffer` 64 x 8 grains, `VSmemBuffer` 32 x 8 grains,
  `mha.cu:389-392`).  Bank behaviour: an 8-lane phase of `LDS.128` / `STS.128` is rows `8j..8j+7`
  at one logical grain -> eight distinct physical grains, conflict-free (as today).
- **Decode.** FP8 block: 4 x `e4m3x4ToBF16x2Pow2m120` (6 SASS each) -> 8 words, 8 `HMUL2` by the
  folded scale, 2 `STS.128`: **34**.  FP4 block: 2 x `e2m1x8ToBF16x2Pow2m126` (13 each: `w << 4`
  + 4 x (PRMT, SHF, LOP3)) -> 8 words, 8 `HMUL2` (folded) or 16 (fallback: x `0x7E807E80` then x
  scale, as `expandE2M1BlockBF16<false>`), 2 `STS.128`: **36 / 44**.
- **Fold decision (rev 2) per warp per page span** (4 per K part, 2 per V tile; each span has its
  own scale word and, in the dynamic module, its own format), inside the warp-uniform format
  branch so `__all_sync` is convergent.  The lane's two scales `s_0, s_1` (E4M3 -> f16 exact ->
  fp32 exact: `cvt.rn.f16x2.e4m3x2` on the u16 + `HADD2.F32`, the same route as
  `e4m3x4ScalesToFloat`) give `f_i = s_i * gFold` with `gFold = g * 2^k` (k = 120 FP8, 126 FP4);
  the vote is `__all_sync(~0, max(|f_0|, |f_1|) < 255.5 * 2^120) && foldOk` with **`foldOk = |g|
  >= 2^-117`** (warp-uniform, from the global scale, as `mha_sm90.cu:492-493`, `:533-534`,
  `:2832-2836`).  Why the second term: the reference scale is `bf16_rn(fl32(s * g))`
  (`convertE4M3ScaleToA16Bits`, `mhaUtils.cuh:593-598`); `bf16_rn(fl32(s * g * 2^k)) =
  bf16_rn(fl32(s * g)) * 2^k` holds only while `fl32(s * g)` is fp32-normal (then the power of two
  commutes with both roundings, and the upper bound keeps the bf16 result finite: bf16 max is
  `255 * 2^120`, the bound is max + half ulp, ties-to-even would round to inf, so strict `<` is
  right).  The smallest nonzero E4M3 scale is `2^-9`, so `|g| >= 2^-117` makes every `s * g`
  normal; below it `fl32(s * g)` is subnormal (fewer than 24 significant bits) and can round
  differently from the scaled product in the last bf16 bit.  `gFold` itself: `g * 2^120` is exact
  for `|g| < 2^8` and overflows to inf otherwise, which fails the vote (inf is not `<`), so large
  globals also take the fallback.  Zero-filled (invalid) rows have `s = 0 -> f = 0`: vote-neutral,
  and their zero payload decodes to `+-0 x S` in both forms.  Cost per span per lane: LDS.U16,
  F2FP, 2 HADD2, 2 FMUL, FMNMX, FSETP, VOTE, PLOP3, F2FP pack, 2 PRMT broadcast = **~12**.  The
  fallback body multiplies the placed value by exactly `2^120` / `2^126` first (`0x7B807B80` /
  `0x7E807E80`, exact: E4M3/E2M1 magnitudes have <= 4 significant bits and `x * 2^-120 * 2^120 =
  x` is a bf16 value < 449), then by `bf16_rn(s * g)` — the reference's single rounding — so
  bit-exactness does not depend on which body runs, only time does.  Host-side facts that decide
  how often the fallback runs: the sealer caps FP8 block scales at 128
  (`BSFP8_A16_SCALE_MAX`, `fp4_kv_quantization.cu:34`, `:343-349`), so the FP8 fold holds for
  every tile with `|g| < 255.5 / 128 = 1.996`; FP4 block scales reach 448 (`:145-158`, no cap),
  so the FP4 fold is per span (`|s g| < 3.99`, true for the bench's unit global scales and for
  block maxima below 24 at `g = 1`).  The dynamic module gets 4 bodies (2 formats x fold /
  fallback) of ~90 SASS each in place of the current 2 of ~180 / ~300; the hot-loop footprint stays
  at the step-5 level (2,476 dyn -> ~2,600).
- **E4M3 NaN codes (rev 2).**  `0x7F` / `0xFF` decode by placement to the finite `480 * 2^-120`
  where `cvt` gives NaN.  The sealer never emits them (payload capped at 448, block scales at 128 /
  448), and `mha_sm90.cu` already made the same choice: NaN payload codes are outside the
  contract, and the exhaustive device test of section 7 item 3 excludes those two codes.
- **Per call per lane** (32 values, FP8, fold body): 2 LDS.128 + ~12 scale/vote + 2 x 34 + 4
  address (amortised over spans) + ~3 loop/branch = **~87 -> ~44 per 16 values** (today 53; wt/B's
  measured lane-tile of 64 values, 187 fp8 / 188 fp4 = 47 per 16 values with its tile glue, is the
  calibration).  FP4: 2 LDS.64 + 12 + 2 x 36 + 7 = **~91 -> ~46**; fallback +16 HMUL2 -> 54.  Rev
  2 budgets (section 4) use **46 per 16 values** for both formats' fold bodies.
- **Ownership / barrier invariants touched: one (rev 2).**  The expanding warp is the copying warp
  (`loadKTilePart` and the expansion run on the same gemm0 warp, `mha.cu:2370` / `:2594`; the
  gemm1 warp copies and expands its own V tile, `:2899` / `:3272`), and the ordering before the
  expansion is `cp.async.wait_group` on the warp's own groups (`:2579`; `syncVTileLoad` `:3081`
  at `:3244`).  **`cp.async.wait_group` completes and makes visible the executing thread's own
  copies only** (`ldgsts.cuh:59-61`, bare `cp.async.wait_group N`); today's payload read is
  same-lane (copy `blockInSpan = 32 * iteration + lane` -> row `lane / 4 + 8 * iteration`, block
  `lane % 4`, `mhaUtils.cuh:341-344`, exactly the expansion's ownership, `:846-851`) and the
  `__syncwarp()` after the wait at both sites runs only under `compactMixedPages` (`mha.cu:2579-
  2583`, `:3245-3249`), which is false here (`:85-87`).  In the new cut lane `l` reads the grains of
  token `tok` written by lanes `4 * (tok % 8) + 2h` and `+ 2h + 1` at iteration `tok / 8`: a
  cross-lane RAW on the payload.  **Fix: the new helper begins with `__syncwarp()`** (`bar.warp.sync
  0xffffffff`: execution barrier with memory ordering among the warp's lanes; each lane has already
  passed its own `wait_group`, so after the barrier every lane's copies of the group are complete
  and visible to all lanes), i.e. it sits between the wait and the first cross-lane LDS at both
  sites, exactly where the compact path has it.  Counted: 1 per part + 1 per V iteration = 6 U per
  64 tokens.  Note (rev 2, pre-existing): the **scale byte** was already read cross-lane by today's
  cut — the scale copy's lane writes row `lane + 32 * iteration` (`mhaUtils.cuh:463-465`), the
  expansion reads row `lane / 4 + 8i + 16 * span` — under the same per-thread `wait_group` without a
  warp barrier.  It is benign on the hardware (the `DEPBAR.LE SB0` that `wait_group` compiles to is
  a per-warp scoreboard), but it is not the PTX contract; the guarded build's barrier closes it,
  the sm120 / q=1 builds keep it (byte-identity rule) and it is listed as an open question for the
  owning tracks.  Within the new cut a lane writes only grains `4h..4h+3` of its own row, which it
  alone reads (its sources `4h`, `4h+2` are among them): no read-after-write against another
  lane's store, so no barrier is needed between the loads and the stores; the `__syncwarp()` at
  the end of the helper before `ldmatrix` stays.  A16 pages: skipped as today (`kNeedsExpansion`
  and the per-page format branch, `:900-912`).

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
  immediate: **~9 per iteration** (rev 2: the 3 `@!PT LDS RZ,[RZ]` placeholders ptxas emits per
  LDGSTS cannot be hoisted, so the floor is ~9, not ~7), **~45 per span** (today 8 x 25 = 200 per
  part in the static modules, 92-104 per span in the dynamic one).  The `span * 2048 /
  iteration * 1024` destination immediates are `Tile::rowBytes * headsPerSpan` /
  `Tile::rowBytes * rowsPerIteration` template constants (128 B rows for both K and V here, but
  written as constants, not numbers).  `swz = (2 * blockInPart) ^ (headInSpan0 % 8)` is a lane
  constant only because `dstHeadOffset == 0` and spans / iterations advance rows by 16 / 8 — the
  guarded body `static_assert`s / asserts that; the source add per iteration is valid because a
  16-token span is contiguous in one page.
- Scale copy: the 3-term 64-bit stride product per lane becomes `scaleBaseForPage + token *
  scale_stride.token` (the page/head terms once per page: 2 IMAD.WIDE, the token term 1 IMAD.WIDE);
  ~12 per iteration (today ~20).
- Verification of the shape, not the stopwatch: SASS of one copy body shows no `IMAD.WIDE` inside
  the iteration except the source add, `LDGSTS` with `[R+imm]` destinations, and the FA3 A6 ratio
  (~6-8 instructions per LDGSTS in the body).

Estimated U (rev 2, from the SASS copy iterations of section 1): static modules K 2 x 240 -> 2 x
(8 x 9 + 2 x 12) = 192, V 4 x 132 -> 4 x (4 x 9 + 12) = 192: 1,008 -> ~384, **-450..-600 U**; the
dynamic module additionally loses the per-iteration share of the select chains: **~-590 U**.  This
remains the item with the widest error bar (ptxas already hoists an unknown part of the C++
per-iteration arithmetic; the SASS says 25 per iteration today, the floor is ~9); artifact A0
measures the executed copy share before the copy change is built (section 6), and section 4 uses
-590 on every mode as the design's number.

### 3.3 What is not changed

K/V ring shapes, `SharedMem` layout (115,456 B; the new code adds no member), the 2-CTA/SM
launch bounds (`mha.cu:3616-3624`), the `nbSubSeqPerSeq` host model (`:3852-3940`; n = 5 stays
optimal because per-tile work shrinks uniformly), the rolled loops of [43], the gemm0/gemm1 MMA
bodies, softmax, the X hand-off, `mha_sm90.cu`, FA3, sm120 modules (guards keep their SASS
byte-identical — the step-5 proof-by-construction is repeated).

### 3.4 Register and smem budgets at 2 CTAs/SM

Registers (cap 128, today 124-128 with STACK 0): the expansion live set becomes 4-8 packed words
(FP8: the second block's LDS.128 can follow the first block's stores; 4 for FP4) + 2 scale words +
8 output words + ~6 temporaries + 4 grain addresses + scale address (rev 2: the swizzle permutes
the four output grains, section 3.1) = ~27-31, against
today's cutlass LUT body that P0.4 measured re-materialising its constants for lack of registers
(31 `IMAD.U32 R, RZ, RZ, UR` per lane-tile at 48 registers in the q=1 kernel; at 128 here the
LUT constants were live and are freed).  The copy hoist adds ~6 lane constants held across the
tile loop (3 per-format offsets, 2 dst bases, 1 swizzle) and 2 for `8 * stride_token`.  Net
expectation: REG stays <= 128 with STACK 0; the accept rule is `cuobjdump -res-usage` REG <= 128,
STACK 0, LDL 0, STL 0 on all four sm90 q=4 modules (as step 5).  Smem: unchanged (no new member;
the fold vote and the scale pairs are registers).  Hot-loop SASS: dyn 2,476 -> predicted 2,500-2,800
(two extra fold bodies, smaller copy bodies); no_instruction must stay <= 0.8 (step 5's 0.51).

## 4. Budgets before / after (U) and predicted wall time

Rev 2: re-based on the section-1 SASS components; expansion after = 46 per 16 values x 32
lane-blocks = 1,472 (fold bodies; FP4 fallback spans 54 x 32 = 1,728), copy hoist -590 on every
mode (3.2), `__syncwarp` +6.

| component | fp8 today -> after | fp4 today -> after | mixed today -> after | a16 |
|---|---|---|---|---|
| expansion (32 lane-blocks of 16 values; mixed = 1/3 fp8 + 1/3 fp4 + 1/3 A16 skip) | 1,704 -> **1,472** (-232) | 2,652 -> **1,472** (-1,180; fallback spans 1,728) | ~1,500 -> **~1,030** (-470) | 0 |
| copy + scales (+ dynamic dispatch for mixed) | 1,008 -> 418 | ~1,000 -> 410 | ~2,200 -> 1,610 | 1,350 -> 760 |
| round / iteration fixed cost (+6 `__syncwarp`) | 1,084 -> 1,090 | 1,080 -> 1,086 | 1,100 -> 1,106 | 1,086 |
| MMA segments | 638 | 638 | 640 | 502 |
| outer loops + per-CTA amortised | 487 | 486 | 490 | 502 |
| **total** | **4,913 -> 4,105 (0.84x)** | **5,842 -> 4,092 (0.70x)** | **5,913 -> 4,876 (0.82x)** | 3,430 -> 2,850 (0.83x) |

The fp8 expansion saving is **~230 U, not the 466 of rev 1** (53 -> 46 per 16 values; the
placement decode removes the F2FP/HADD2 chain but the fixed part of a lane-call — scale prep, vote,
addresses, stores — is already close to wt/B's measured glue), so fp8's count-only prediction is
0.84x -> **95 us on reading (i)**: at or 1 us above its 94 us target.  fp8's pass therefore rests on
reading (ii), the issue-rate recovery, and section 6 A2 ties it to evidence
(`short_scoreboard` 1.32 -> <= 0.7, `pipe_xu` down >= 4x) rather than to the count.

Wall time.  The kernel is issue-paced with a latency component that does not shrink with the
count (long scoreboard 0.9-1.4 warps per issue cycle = K/V landing waits; the a16 module is
DRAM-side at 67 %).  Two readings bracket the prediction: (i) issue-proportional, `t x U_new /
U_old`; (ii) the step-4 -> 5 calibration, where -5.6 % instructions (fp8) gave -8 % time and +2.5 %
(mixed) gave -16 % time because the stall mix changed — i.e. time tracks `U x cycles-per-issue`,
and the FP8 chain's short-scoreboard stall (1.32) goes away with the chain (the placement decode
is 3 independent INT ops + 1 HMUL2 per pair against a 5-deep F2FP/HADD2 dependency; the [16] trace
on the q=1 kernel measured the expansion segment shrinking with the count, 1440 -> 750 cycles for
385 -> 188 SASS, "count-bound not MIO-bound").  `t(ii) = U x 8,704 / (132 x 4 x 1,980 x rate)`
with rate 0.43 (fp4's today) for fp8 and fp4, 0.42 for mixed.

| mode | today | (i) proportional | (ii) at fp4's issue rate 0.43 | predicted band | target | verdict (rev 2) |
|---|---|---|---|---|---|---|
| fp8 | 114.0 | 95.3 | 79.5 | **79-95** | <= 94 | **marginal**: passes only on reading (ii); the count alone gives 0..-1 us of margin |
| fp4 | 115.9 | 81.2 | 79.2 | **77-84** | <= 59 | **open** (1.31-1.42x); accept <= 90 |
| mixed | 116.1 | 95.7 | 93.6 (0.42) | **89-96** | <= 101 | pass (5-12 us margin) |
| transport_a16 | 86.1 | 71.5 (issue) but DRAM floor 67.5 + fixed | - | **80-86** | 135 | pass (unchanged within noise expected: DRAM-side) |

The DRAM check: fp8 at 80 us moves 152 MB -> 1.9 TB/s (38 % of the 4.5 TB/s sustained probe), fp4
at 77 us 80 MB -> 1.0 TB/s, mixed at 89 us 172 MB -> 1.9 TB/s: none approaches the byte floor, so
the prediction is not DRAM-capped.

Accept/reject numbers for the step (three locked rounds, `--repeats 2 --trials 5`, medians;
co-tenant rule 2 x t < 1.5 ms holds): **accept if fp8 <= 94 and mixed <= 101 and fp4 <= 90 and
a16 within +-3 us of 86; reject the step (revert the tree) if any compressed mode is above 1.05x
its step-5 record or a16 regresses > 5 %.**  fp4 in 77-90 is the expected outcome, not a pass of
its target — see 5.3.  fp8 between 94 and 97 with `short_scoreboard` unchanged means reading (ii)
did not materialise: the step is then kept for mixed and fp4 (accept rule above) and fp8's target
is reported open with the A2 counters as the reason.

## 5. Risks, and what fp4 <= 59 would take

### 5.1 Main risks

1. **The copy estimate (3.2) is the least measured number.**  If A0 shows copy + scales at < 700 U
   today, the hoist is worth < 300 U and the fp8 count-only prediction moves to ~4,400 U = 0.90x ->
   102 us on reading (i) (rev 2; fp8 then depends entirely on reading (ii)), mixed to ~99 us
   (marginal against 101).  Mitigation is the order of work: A0 first, then decode, then copy only
   if A0 confirms >= 900 U (the SASS says ~1,000 in the static modules).
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

### 5.2 Barrier / ownership invariants (restated; rev 2: one barrier added)

K: gemm0 warp `w` copies (`cp.async`, `commitGroup`) part `p+1` into buffer `next`, waits
`wait_group 1` for part `p`, **`__syncwarp()` (new, inside the guarded helper: the wait is
per-thread, the new cut reads other lanes' copies)**, expands buffer `curr` in place (`__syncwarp`
at the end), reads it with `ldmatrix`, advances `CircIdx` (`mha.cu:2570-2645`).  V: gemm1 warp
copies tile `i+1`, `wait_group 1` for tile `i` (`syncVTileLoad`, `:3071-3083`, at `:3244`),
**`__syncwarp()` (new, same place)**, expands its own 32 x 128 B tile, `ldmatrix.trans`, arrives
`xBar.consumed` once per X tile (`:3131-3348`).  Formats/flags per buffer are written by lane 0 of
the owning warp before the copy and read by the same warp after the wait (`:2427-2431`,
`:2927-2934`; the lane-0 store is ordered by the `__syncwarp()` that follows it).  The new cut
changes which lane writes which grains, never which warp; the added barrier is the one the
compact path already has at both sites (`:2579-2583`, `:3245-3249`), now also taken by the guarded
expansion build.  Cost 6 U per 64 tokens (section 4).

### 5.3 fp4 <= 59 is out of reach of this kernel; what would reach it

Rev 2 (the rev-1 form — "zero expansion would give 3,040 U -> 60 us, so the target requires
removing the expansion entirely and losing nothing else" — ignored the design's own -590 U copy
hoist, and "fp4's target lies below a16's 3,430 U" is not a floor because a16 carries a 342 U
larger copy and itself drops to ~2,850 after the hoist).  The sound form is a floor:

- **Non-expansion floor after this step (SASS-derived):** fp4 today 5,842 - expansion 2,652 = 3,190
  U; minus the copy hoist 590 = **~2,600 U** (round fixed 1,086 + MMA 638 + outer 486 + copy 410).
- **Expansion floor of any smem-materialising form:** 8 HMUL2 + 2 STS.128 per 16 values are
  unavoidable once bf16 operands are written to shared memory, plus the cheapest decode (placement:
  24 for FP8, 26 for FP4) = **>= 34 per 16 values**, x 32 lane-blocks = **1,088 U** (a PRMT
  byte-LUT alternative was counted at 16-18 per 8 values against the placement's 12-13 and is not
  better; zero scale-prep / vote / address is assumed).
- **Floor: ~3,690 U.**  At today's fp4 issue rate 0.43 that is `3,690 x 8,704 / (1,045,440 x
  0.43)` = **71 us**; at 0.42, 73 us.  59 us at 3,690 U needs **0.52 instructions per SMSP-cycle
  whole-kernel**, i.e. ~90 % of today's 58 % issue-active fraction with **zero** tail on a
  2.58-wave grid and no landing-wait stalls — not a property this kernel has or that this step
  changes (long-scoreboard 0.9-1.4 warps per issue cycle are K/V landing waits, and the grid tail
  is the host `nbSubSeqPerSeq` model's residual).

What would reach 59: (a) a native narrow-operand MMA consuming packed E2M1 with the block scale
applied by the tensor core — `kind::f8f6f4` / block-scaled MMA on sm120/sm100, not sm90; (b) a
consumer that is DRAM-bound at the fp4 byte count, i.e. one whose per-tile instruction count is
<= ~2,600 U with no expansion at all — the `mha_sm90.cu` wgmma consumer with converter warps is the
closest existing structure and sits at 96 -> 73 us at q=1 after [16] with a 1.09 us/tile consumer
floor (P0.3), so a SPEC_DEC port of it (step 3's rejected route, K3/V2 at 99 KB fits 2 CTAs/SM)
would land around 75-95 us, not 59.  The honest statement for the targets table: **fp4 q=4 on sm90
is bounded near 71-73 us by the A16-materialising mma.sync structure at today's issue rate, ~77-84
us after this step; 59 needs a different MMA operand path.**  fp8 and mixed do not have this
problem because their targets are 0.82x / 0.87x.

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
- **A3 (correctness):** `tests/attention/run_xqa_mixed_page_transport.py` 54/54 bit-exact on
  nkcut2 (default and `XQA_NB_SUB_SEQ=2`) plus a max-block-scale FP4 case (fold fallback path), an
  E4M3-subnormal case and a tiny-global-scale case (`|g| < 2^-117`: the `foldOk` fallback), all
  bit-exact against the A16-expansion reference; the exhaustive device test of section 7 item 3
  (256 E4M3 codes minus the two NaN codes x 16 E2M1 codes x scale regimes incl. the fold boundary
  `s g` just below / above 255.5 and `|g|` just below / above `2^-117`); ws-1 sm120 modules
  (`xqa_mha` all formats, q=1 and q=4, and `mla_sm120`) SASS byte-identical to a pristine build
  (guards; the dyn q=1 module differs between two pristine builds by ptxas variation, as step 5
  noted), 54/54 there too.  SASS of the guarded modules shows exactly one `WARPSYNC`/`BAR.SYNC`-class
  instruction between the `DEPBAR` and the first `LDS` of the expansion (the added `__syncwarp`).
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
   exhaustive (254 E4M3 codes — the NaN codes `0x7F`/`0xFF` are outside the sealer's contract,
   rev 2 — x scale regimes incl. subnormal inputs, the `255.5 * 2^120` fold boundary and the
   `2^-117` global boundary) device test — the wt/B exactness argument is for bf16 math; this
   build is bf16, and an fp16 build keeps the current body (the guard requires bf16 `InputElem`).
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

## 8. As written (code state of this worktree; updated per commit)

### 8.1 Item 1 — placement decode + fold vote (`expandMixedPartialHeadsInPlaceBF16Placement`)

Files: `csrc/xqa/mhaUtils.cuh` (new helpers after `expandMixedPartialHeadsInPlace`: `ldsU16`,
`ldsB64`, `e4m3x2ScalesToFloat`, `bf16x2Broadcast`, `MixedFoldTag`, `foldScalePairFinite`,
`expandE4M3Block16BF16Placed<kFold>`, `expandE2M1Block16BF16Placed<kFold>`,
`expandMixedPartialHeadsInPlaceBF16Placement`), `csrc/xqa/mha.cu` (guard after `grpLoadV`; K
site in `runGemm0`, V site in the gemm1 loop, both `#if MIXED_BF16_PLACEMENT_EXPANSION`).  The
stock helper and `mha_sm90.cu` are untouched; the block expanders mirror `mha_sm90.cu:2792-2830`
under new names because that file belongs to another track.  Commit "item 1" carried today's
lane ownership (block `l % 4` of rows `l / 4`, `l / 4 + 8`; two `LDS.U8`); commit "item 2"
replaced it with the half-row cut below — the decode, vote and bodies are unchanged between the
two.

Decode / scale data flow (per lane, per span, both commits):

```
s_i = E4M3 block scale bytes            (item 1: LDS.U8 x2 of rows r0, r1; item 2: one LDS.U16)
r_i = float(s_i)                        cvt.rn.f16x2.e4m3x2 + HADD2.F32 (both exact embeddings)
f_i = r_i * (g * 2^k)                   k = 120 (FP8) / 126 (FP4); g * 2^k exact for |g| < 2^8, inf beyond
fold = __all_sync(|g| >= 2^-117 && max(|f_0|, |f_1|) < 255.5 * 2^120)     (warp-uniform by construction)
fold:     sf2_i = bf16x2{f_i, f_i}      one F2FP.BF16.PACK_AB each (no PRMT broadcast)
fallback: sf2_i = bf16x2{r_i g, r_i g}
FP8 block: LDS.128 -> 4 x e4m3x4ToBF16x2Pow2m120 (PRMT, PRMT, SHF, SHF, LOP3, LOP3 -> x * 2^-120)
FP4 block: LDS.64  -> 2 x e2m1x8ToBF16x2Pow2m126 (SHF + 4 x (PRMT, SHF, LOP3) -> mag * 2^-126)
fallback only: HMUL2 by 0x7B807B80 / 0x7E807E80 (exactly 2^120 / 2^126; E4M3/E2M1 magnitudes have
               <= 4 significant bits and x < 449, so the product is exact)
HMUL2 by sf2_i (8 per block); STS.128 x2 per block
```

### 8.2 Item 2 — half-row owner cut (same helper; commit "item 2")

Lane geometry, one call = one 128 B K part (64 rows = 4 spans) or one 32 x 128 B V half-tile (2
spans):

```
lane l: tok = l % 16 (row of every span), h = l / 16 (blocks 2h, 2h+1 = logical grains 4h..4h+3)
row r = 16 span + tok; x = r % 8 = tok % 8 (lane constant: spans advance rows by 16)
physical grain of logical 4h + j = (4h + j) ^ x = (4h ^ x) ^ j    (4h and j < 4: disjoint bits)
  -> the four grains are a permutation of one 4-aligned group that depends on x & 3, so the lane
     holds FOUR u32 addresses  addr_j = tileBase + Tile::byteOffset<swz>(tok, 4h + j)
     and uses [addr_j + 2048 * span]        (Tile::rowBytes * headsPerSpan, a template constant)
scale pair: [scales + 4 tok + 2h + 64 span]  (LDS.U16: lo = block 2h, hi = block 2h+1)
sources: block 2h in grain addr_0 (FP8 16 B / FP4 low 8 B), block 2h+1 in addr_2
outputs: block 2h -> addr_0, addr_1; block 2h+1 -> addr_2, addr_3
```

Control flow: `__syncwarp()` on entry — each lane has passed its own `cp.async.wait_group`
(`mha.cu` K: `waitGroup<1>` before the part body; V: `syncVTileLoad` = `waitGroup<nbVBuffers-1>`),
but that completes the executing thread's copies only, and the payload grains of row `tok` were
copied by lanes `4 (tok % 8) + 2h`, `+ 2h + 1` at copy iteration `tok / 8`, the scale word by lane
`tok + 16 (span % 2)`; the `bar.warp.sync` makes all lanes' completed copies visible to all lanes
(the compact path has the same barrier at both sites).  Then the span loop
(`#pragma unroll(pageLoopUnroll)`: unrolled in static modules, rolled in the dynamic one) -> the
warp-uniform format branch (dynamic module) -> `LDS.U16`, vote (inside the branch, all 32 lanes:
convergent) -> `body(MixedFoldTag<true/false>)`, straight-line over the lane's two blocks via an
explicit `block(a, b, sf2)` called twice -> `__syncwarp()` before the `ldmatrix` reads.  A lane
writes only grains it alone reads (its sources are among its outputs), so no barrier sits between
its loads and stores.  No runtime-indexed register arrays (`out[2]` is indexed by literals), no
pointer selects (`g` / `pow2k` are `if constexpr`-resolved), every shared address is a
lane-constant u32 plus an immediate.

Static guarantees (`static_assert`): bf16 `InputElem`, 16 B grains, `swizzle`, 128 B rows with 8
grains, `partBytes 128` / `blocksPerPart 4`, `headsPerSpan 16` = `tokensPerPage`, 64 blocks per
span = 2 x `warp_size` (one row, two adjacent blocks per lane), `blocksPerLane 2`, 4 B scale words,
`headsPerSpan % 8 == 0` (the swizzle term is lane-constant), `dstNbHeads >= maxNbCopiedHeads`; the
helper takes no `dstHeadOffset` / `idxPart` / `idxWarp` (row 0 origin, one warp; both call sites
passed the literal 0 / their own warp).  `MIXED_BF16_PLACEMENT_EXPANSION =
ENABLE_MIXED_KV_CACHE && MIXED_COMPACT_TILE_LOOPS && !INPUT_FP16 && !(GRP_LOAD_V)` is
`static_assert`ed to imply `kCompactTileLoops && bf16 && !grpLoadV && !compactMixedPages`.

Instantiations: K `<64, 2, true>` on `KSmemBuffer` (64 x 8 grains), V `<32, 2, true>` on
`VSmemBuffer` (32 x 8 grains); both pass `sourceHeadOffset = 0`.

Expected SASS per lane-call (32 values; the section 6 A1 list applies): FP8 fold body — LDS.U16,
F2FP.E4M3, 2 HADD2.F32, 2 FMUL, FMNMX, FSETP, VOTE, PLOP3, 2 F2FP.PACK, 2 x (LDS.128 + 24 + 8
HMUL2 + 2 STS.128) = 68, ~4 address / loop = **~85** (today 2 x 53 = 106); FP4 — 2 x (LDS.64 +
26 + 8 + 2) = 74 + 13 + 4 = **~91**; fallback bodies +16 HMUL2.  Per 16 values ~43-46, the
section-4 budget's 46.  F2FP per FP8 lane-call 18 -> 3, HADD2.F32 16 -> 2, HMUL2 16 (32 in the
fallback body), STS.128 4, LDS.128 2 / LDS.64 2, LDS.U16 1, one `WARPSYNC` between the `DEPBAR`
and the first LDS, no LDL / STL, REG <= 128.
