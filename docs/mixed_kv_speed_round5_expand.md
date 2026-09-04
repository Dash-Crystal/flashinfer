# Round 5 — converter expansion body: 188 -> 167 (fp8) / 172 (fp4) SASS per lane-tile at unchanged bit-exactness (design)

Worktree `wt/r5expand` at `864479f6` (production sm90 q=1 kernel, lever [8]).
Scope: `expandPackedStage` and its lane constants in `csrc/xqa/mha_sm90.cu`
(the converter warps' in-place E4M3 / E2M1 -> BF16 expansion), for the three
compressed modules (`MIXED_PAGE_STATIC_FORMAT` = 1 fp8, 2 fp4, -1 mixed).  Not in
scope: the copy-issue chain (worktree r5copy), the consumers' wgmma descriptor
arithmetic (r5desc), the A16 module (its converters skip the expansion).

**Headline.**  The task asked for <= 150 SASS per lane-tile.  That number is
below the floor: the [16] audit's "essential 139" was an arithmetic slip — the
essential body is **147 (fp4) / 141 (fp8)** (section 1.3), and the fp32 block
scale route (15) plus the fold decision (4) are bound to the bit-exactness
contract, so the reachable count at unchanged numerics is **167 (fp8) / 172
(fp4) / ~176-181 per compressed page (mixed)**: -21 / -16 per lane-tile, -168 /
-128 / ~-90 per tile-SM (-4.0 % / -3.3 % / -2.5 % of the SM's issue).  Every
removed instruction is address or vote glue; no decode arithmetic changes.  The
one route to <= 150 (move the scale conversion and the fold decision to the IO
warps) is issue-neutral for the SM and adds a producer protocol; it is rejected
in section 3.6.  Predicted walls (issue-bound model, section 6): fp8 67.9 ->
65.6, fp4 60.6 -> 59.0, mixed 64.6 -> 63.3; none reaches a target alone.  Go as
a component of the round-5 combined cut (copy issue + descriptors + this), with
SASS gates before any timing (section 7).

---

## 1. The region today, by SASS class, with the method

### 1.1 Method (reading only; no GPU job)

Inputs: the lineinfo listings `/tmp/r4p7_sass/f{1,2,-1}_li.nvdis` on nkcut2
(compile-only builds of `530f5d4d`'s `mha_sm90.cu`, whose `csrc/` and
`include/` are byte-identical to this worktree's — `git diff --stat 864479f6
530f5d4d -- csrc include` is empty — so the source line numbers below are this
worktree's), and the production ncu `SourceCounters` runs
`/tmp/r4mixed2_pcs_{fp8,fp4,mixed}_P132.csv` (one launch of the locked bench,
8 712 tiles, 4 converter warps per operand; the fp8 listing's text is identical
to the profiled r2p8 binary's, round-4 [7] rev 2 section 2.1).  Tools:
`benchmarks/microbench/xqa_sm90_converter_sass.py --paths / --dump K-expand`
(static classes and the branch-split executed path), and a join of the listing
with the CSV's "Instructions Executed" per address, divided by 8 712 x 4 (=
executions per warp per tile) — script `/tmp/r5expand/dyn.py` (local), outputs
`/tmp/r5expand/f{1,2,-1}_Kexpand.dyn` (per-instruction executed-per-lane-tile,
stall samples, SASS, inline frames).  Static = dynamic inside the fold path
(every listed instruction executes 1.00 per lane-tile for fp8 / fp4; 0.67 / 0.33
in mixed), so the executed count is the branch-split static count, confirmed by
the profile: **fp8 187.8, fp4 187.8, mixed 131.2 per K-warp-tile** (V within 1).

### 1.2 Executed instructions per lane-tile, K converter, steady state (fold path)

| class | fp8 (f1) | fp4 (f2) | what |
|---|---:|---:|---|
| PRMT | 36 | 36 | 32 placement spreads (`mhaUtils.cuh:658-659` / `:673`) + 4 scale broadcasts (`prmtSelfB32(sc, 0x1010/0x3232)`, `:3549`) |
| SHF + IMAD.SHL | 8 + 26 = 34 | 11 + 30 = 41 | fp8: 32 `<< 4` (`:661-662`) + 1 `>> 16` (scale hi half) + 1 spare; fp4: 32 `<< 2` (`:675`) + 8 `<< 4` (`w4`, `:668`) + 1 |
| LOP3 | 42 | 39 | 32 masks `0x87F087F0` / `0x81C081C0`; 7 store XORs `a16Row ^ (0x10..0x70)` (`:3497`); fp8 only: 3 read XORs `row ^ 0x10/0x20/0x30` (`:3525`) |
| HMUL2.BF16_V2 | 32 | 32 | one per pair (`mhaUtils.cuh:709`) |
| STS.128 | 8 | 8 | `[R2+UR11]` once, then `[Rx]` from the XORs |
| LDS | 5 (4 x .128 + scale) | 3 (2 x .128 + scale) | fp8 reads: 1 `[R3+UR11+0x2000]`, 3 from XORed registers |
| F2FP | 4 | 4 | 2 `F16.E4M3.UNPACK_B` (scale bytes -> f16x2), 2 `BF16.F32.PACK_AB` |
| HADD2.F32 | 4 | 4 | f16 -> f32 per scale |
| FMUL.FTZ | 4 | 4 | `f32(s) * gfold` |
| FMNMX | 3 | 3 | fold vote max tree |
| FSETP / VOTE.ALL / NOP | 1 / 1 / 2 | 1 / 1 / 2 | vote compare, vote, `__syncwarp` NOPs (`:3531`, `:3627`) |
| BRA | 3 | 3 | bad-page (`:3504`), fold (`:3557`), trailing jump from the hot body back to the join |
| addressing / test | UISETP, PLOP3, ULEA, VIADD, USHF, ULOP3, IMAD.U32, IADD3 x2, VIADD = 10 | ISETP, ULEA, VIADD, USHF, ULOP3, MOV, IADD3 = 7 | tag test; stage base; `scales[t % 4]` address; fp8 row base + 0x2000 |
| **total** | **188** | **188** | matches ncu 187.8 |

Two-multiply fallback path (vote failed): 223 / 223.  Zero fill (past the
sequence end): 17.  Static `expandPackedStage` region: fp8 391, fp4 399 SASS
per role (K and V each carry a copy); mixed 773 per role (both formats x fold /
fallback, 12.4 KB per role, 24.7 KB of the module's hot code).

Mixed module (f-1), per K-warp-tile: 2 instructions on every page (PLOP3 + BRA
bad-page test), 15 on every compressed page (three uniform dispatch branches
`:3478-3524` with UISETP / PLOP3 each, stage ULEA, `scales` address, scale
LDS, NOP), then **180 (fp4 page) / 178 (fp8 page)** in the format body — the
body includes the per-tile `MOV UR15; FMUL 2^126` recompute of the fold
multiplier (`foldMultiplier`, `:3412-3421`, mixed only, kept out of registers
to avoid the STACK 8 of an earlier variant).  Per compressed page: fp4 197, fp8
195 (static 188): the dual-format cost is +7..+9 instructions of uniform
dispatch per page, plus the 24.7 KB of duplicated hot code (round-4 mixed doc,
footprint ordering of gemm0 `no_inst`).  A16 pages: 2-3.  Average 131.2 =
(195 + 197 + 2) / 3 within 0.3.

### 1.3 Essential vs glue (corrects the [16] audit)

Essential = the instructions any implementation of this lane cut, this decode
and this layout must execute: placement 3 per pair (PRMT, shift, mask: no
2-instruction form exists — section 3.7), one HMUL2 per pair, 8 STS.128, the
packed loads, the scale load.

| | fp8 | fp4 |
|---|---:|---:|
| placement (32 PRMT + 32 SHF + 32 LOP3) | 96 | 96 |
| fp4 `w << 4` per packed word | – | 8 |
| HMUL2 | 32 | 32 |
| STS.128 | 8 | 8 |
| LDS.128 | 4 | 2 |
| LDS scale word | 1 | 1 |
| **essential** | **141** | **147** |
| glue today | 47 | 41 |

The [16] audit summed the fp4 list to 139; the same list sums to 147.  The
glue of 47 / 41 splits into: scale route to bf16x2 broadcasts 17 (13 + 4
PRMT), fold vote 7 (3 FMNMX, FSETP, VOTE, NOP, BRA), store XOR 7, fp8 read XOR
3 + row adds 2, scale-slot address 4, stage/row base 2, tag test 3, trailing
`__syncwarp` NOP 1, trailing BRA 1 (fp4: no read XORs, 1 fewer address op).

Per tile-SM (8 converter warps): expansion 1 504 (fp8) / 1 504 (fp4) / 1 050
(mixed) of 4 150 / 3 881 / 3 580 total (36 % / 39 % / 29 %).

## 2. Current data flow and control flow (line refs)

Lane cut (`ExpandLane`, `:554-559`, `makeExpandLane` `:571-599`): warp w of an
operand's four converter warps owns page w of the tile; lane l = (token
l % 16, head part p = l / 16); it owns 64 values = 4 blocks of 16 and their four
E4M3 block scales (one 4 B word at `TileScales[token][4p]`).  Lane constants
kept in registers across the tile loop: `a16` (the lane's A16 row in part p,
bits [6:4] = token % 8 =: x), `fp8` (chunk (4p) ^ x of the packed E4M3 row),
`fp4` (32 packed bytes at an 80 B row stride), `scale`.  Global scales
(`ExpandScales`, `:561-610`): `g`, `gfold = g * 2^120` (fp8) / `2^126` (fp4),
`foldOk = |g| >= 2^-117`, per operand, computed once (`:2660`, `:2727`).

Per tile, converter loop (`K :2679-2712`, `V :2742-2760`):

```
waitGroup<1>  ; __syncwarp                 (own + warp's copies of tile t landed)   :2684-2685
expandPackedStage(stage, scales[t % 4], tag byte, lane, gs)                        :2690 / :2750
fence.proxy.async ; kBar[stage].produced.arrive                                   :2694-2695
tag rotate ; issueKCopies(t + 2) ; commitGroup                                     :2700-2710
```

`expandPackedStage` (`:3467-3628`), format = tag (mixed) or the static format
unless the tag is `kMixedBadPageFormat`:

```
A16 page      -> __syncwarp ; return                                               :3489-3492
stage = cvta(parts) ; a16Row = stage + lane.a16                                    :3495-3497
bad page      -> __syncwarp ; 8 x STS.128 zero at a16Row ^ (c*16) ; __syncwarp     :3504-3515
scaleWord = LDS.32 [scales + lane.scale]                                           :3516-3518
fp8: row = stage + lane.fp8 ; words[b] = LDS.128 [row ^ (b*16)], b = 0..3          :3525-3530
fp4: row = stage + lane.fp4 ; words[c] = LDS.128 [row + c*16], c = 0..1            :3575-3580
__syncwarp                                    (all reads of the page before any write)  :3531 / :3581
f[4] = f32(scale bytes) * gfold  (cvt.rn.f16x2.e4m3x2 x2, f16->f32 x4, FMUL x4)   mhaUtils.cuh:679-689
fold = VOTE.ALL(max|f| < 255.5 * 2^120) && foldOk                                  :3460-3463
fold:   sc01 = bf16x2(f0,f1) ; sc23 = bf16x2(f2,f3)                                :3540-3541
        for b: sf2 = prmt(sc, 0x1010 / 0x3232) ; out = HMUL2(place(words[b]), sf2) ; STS.128 x2 at a16Row ^ ((2b+g)*16)
!fold:  g[4] = f32(s) * g ; same loop with an extra HMUL2 by 2^120 (fp8) / 2^126 (fp4) per pair
__syncwarp                                                                          :3627
```

Landing layout of the packed rows (written by `issueCompressedPageCopies`,
`:3137-3205`, the converters' own cp.async): E4M3 rows are 128 B with the TMA
swizzle chunk `c ^ (r % 8)` (`:3178`), E2M1 rows 64 B at an 80 B stride
(`:3190`); block scales 8 B per token in `kScales[t % 4]` (`:3195-3204`),
copied by lanes < 16.  Output: A16 rows of parts 0 and 1, 128 B swizzled
(chunk `(2b+g) ^ x`), the part-1 rows overwriting the packed rows of the same
page — hence the mid-body `__syncwarp` (D3-analogue, header comment `:540-543`).

Dependency structure of the hot path (from the fp8 SASS, `f1_Kexpand.dyn`):

- decision chain, 14 dependent instructions incl. one LDS and two XU/FMA-pipe
  stages: `USHF -> ULOP3 -> IMAD.U32 -> IADD3 -> LDS(scale) -> SHF.R -> F2FP.E4M3
  -> HADD2.F32 -> FMUL -> FMNMX -> FMNMX -> FSETP -> VOTE.ALL -> @P0 BRA`; the
  body (all 151 instructions) sits behind the branch.  The profile puts the
  region's largest stall sample on the first instruction after the scale LDS
  (`SHF.R` 82 / 71 samples of ~400 in the region: short scoreboard on the LDS)
  and on the branch (39 / 19: waits for the VOTE).
- data chain per pair after the branch: `[F2FP.PACK -> PRMT bcast]` joins
  `[LDS.128 -> PRMT -> SHL -> LOP3] -> HMUL2 -> [LOP3 XOR] -> STS.128`: 5-6.
- depth entry -> first STS ~ 19-20 dependent instructions; the 4 fp8 LDS.128
  are issued before the vote (good), 3 of them behind a LOP3 each.

## 3. New data flow and control flow

Principle: every per-tile address becomes `[R_const + UR_stage (+ imm)]` with
the lane constants computed once per warp (as the F25 / S6 owner cuts did for
the copy path), the fold decision becomes an integer byte test on the raw
scale word (equivalent to today's fp32 test, section 4.2), and the scale route
packs its broadcasts directly.  Decode arithmetic, layout, barriers, and the
copy path are untouched.

### 3.1 Lane constants (replaces `ExpandLane`)

```
struct ExpandLane {
  uint32_t st[4][2];  // store base of block-register k, half g: p*8192 + token*128 + (((2k+g) ^ x) * 16)
  uint32_t rd8[4];    // fp8 read base of block k:  packedBase + w*2048 + tokenInPage*128 + (((4p+k) ^ x) * 16)
  uint32_t rd4;       // fp4 read base:             packedBase + w*2048 + tokenInPage*80  + p*32   (unchanged)
  uint32_t scale;     // token*8 + p*4                                                            (unchanged)
};
```

All indices are compile-time in the unrolled block loop, so each entry is one
register (no local memory; C2-analogue).  Static fp8 module: 8 + 4 + 1 = 13
live constants (today 3); static fp4: 8 + 1 + 1 = 10; mixed: 8 + 4 + 1 + 1 =
14 (today 4).  Per-operand uniform constants (registers or URs): `gfold`, `g`,
`foldOk` (unchanged), new `voteAdd` (3.4).

### 3.2 Per-tile addressing

```
UR_stage = stage base (existing ULEA)                                  1 uniform
UR_scale = scalesBase + stage * 512   (ULEA from the stage index)      1 uniform   [today: USHF, ULOP3, IMAD.U32, IADD3]
scaleWord = LDS.32 [R_scale + UR_scale]                                            [today: LDS [R4 + 0x1aad0] after 4 ops]
fp8: words[k] = LDS.128 [R_rd8[k] + UR_stage],  k = 0..3               4 LDS       [today: VIADD, IADD3, 3 LOP3, 4 LDS]
fp4: words[c] = LDS.128 [R_rd4 + UR_stage + c*16]                      2 LDS       [today: VIADD, 2 LDS — imm form already]
store (k, g):  STS.128 [R_st[k][g] + UR_stage], v[g]                   8 STS       [today: VIADD a16Row, 7 LOP3, 8 STS]
```

The scale ring is indexed by the **stage** (`nbScaleTiles = nbKBuf = 3`,
`kScales[stage]`, `static_assert(nbScaleTiles == nbKBuf && nbKBuf == nbVBuf)`)
instead of `t % 4`, so its address is a shift of the stage index the loop
already holds in a UR.  Liveness (4.3): the ring is converter-private and its
depth equals the stage depth the copies are already gated on.

### 3.3 Scale route

```
lo16x2, hi16x2 = cvt.rn.f16x2.e4m3x2 (scaleWord), (scaleWord >> 16)       SHF + 2 F2FP          (unchanged)
f[k] = f32(half) * gfold                                                   4 HADD2.F32 + 4 FMUL  (unchanged)
sf2[k] = bf16x2(f[k], f[k])                                                4 F2FP.BF16.PACK_AB   [today 2 PACK_AB + 4 PRMT]
```

`F2FP.BF16.F32.PACK_AB Rd, Ra, Rb` rounds each input independently, so the
pair pack with equal inputs is the broadcast of the same bf16 bits the pair
form produced (`bf16x2BitsFromFloats(f, f)`; the S6 helper `bf16x2Broadcast`).
-2 per lane-tile.  The fallback arm packs `bf16x2(g[k], g[k])` the same way.

### 3.4 Fold decision as a byte threshold

Today: `fold = VOTE.ALL(fmax_k |f32(s_k) * gfold| < 255.5 * 2^120) && foldOk`
(`foldScalesFinite`, `:3460-3463`): 3 FMNMX + FSETP + VOTE + NOP, and it sits
at the end of the fp32 chain.

New, per operand once (converter prologue, next to `makeExpandScales`):
`thr` = the largest E4M3 magnitude code c in [0x00, 0x7E] such that
`fabsf(f32(decode(c)) * gfold) < 255.5f * 0x1p120f` evaluated with the same
`cvt.rn.f16x2.e4m3x2` -> f32 -> `FMUL.FTZ` -> compare the kernel uses today
(seven-step binary search over the code, ~60 instructions once per warp; the
predicate is monotone in c because E4M3 magnitude is monotone in its code and
`mul.rn` by a fixed factor and `fabsf` are monotone), `thr = -1` if no code
passes; `voteAdd = (0x7F - thr) * 0x01010101` (thr = -1 -> 0x80808080).

Per tile:

```
m   = scaleWord & 0x7F7F7F7F           LOP3
t   = m + voteAdd                       IADD3          (byte sums <= 0x7F + 0x7F = 0xFE: no carry across bytes)
P0  = (t & 0x80808080) != 0             LOP3.LUT P0, RZ, ...   (predicate-output form; else LOP3 + ISETP)
fold = VOTE.ALL(!P0) && foldOk          VOTE.ALL (+ NOP)
```

Byte k's sum has bit 7 set iff `mag_k > thr` iff `f32(s_k) * gfold` fails
today's compare (4.2).  4-5 instructions, all ALU-pipe, depending only on the
scale LDS; the FMNMX tree is gone and the F2FP / HADD2 / FMUL chain is no
longer on the branch's critical path.

### 3.5 Branch layout and warp syncs

- Hot arm (fold) as the fall-through, cold arm (two-multiply) out of line
  jumping back to the join: the executed path carries one taken branch at
  most (`@P BRA` to the cold arm not taken) instead of two (today the hot body
  is the branch target and ends with a jump back).  Source: test `!fold` first
  with the cold arm in the `if`; ptxas is not obliged to honour it, so this is
  a gate (7.2), counted as -1 if met.
- The mid-body `__syncwarp` (`:3531 / :3581`) stays: it is the only ordering
  between every lane's packed-row reads and any lane's part-1 stores of the
  same page (the part-1 rows overwrite the packed rows).
- The trailing `__syncwarp` (`:3627`) is removed.  What follows it in program
  order is `fence.proxy.async` and a per-thread `mbarrier.arrive` (each of the
  128 converter lanes arrives; the barrier count is 128), so gemm0's
  acquire-after-completion sees every lane's stores without a warp sync; the
  next tile's reads are of another stage and are preceded by the loop's own
  `waitGroup + __syncwarp` (`:2684-2685`); the copies issued afterwards for
  tile t + 2 target stage (t + 2) % 3 = (t - 1) % 3, whose reads by every lane
  of this warp precede tile t - 1's mid-body sync, which precedes this lane's
  tile-t stores and hence its copy issue.  Zero-fill and A16 paths keep their
  syncs (they are not on the hot path).

### 3.6 Considered and rejected

- **Scale conversion / vote off the converters (IO warps)**: the only way to
  <= 150 (fp8 141 + ~7 = 148; fp4 147 + 7 = 154 still misses).  The IO warp
  would convert the same 128 scale words per tile per operand the four
  converter warps convert today (same ~90 warp-instructions), so the SM's
  issue total — the bound at 2 CTAs/SM (round-3 pair doc) — is unchanged; it
  only moves ~20 x 4 warp-instructions from the converter warps' serial
  per-tile chain to an IO warp, and needs a new producer stage (IO cp.async +
  wait + convert + STS + fence + arrive, a per-tile barrier wait in the
  converters, +2..+12 KB of smem for pre-converted or pre-broadcast scales).
  Rejected: issue-neutral, protocol not boring.
- **No vote via a smaller fold (2^112) compensated in `qScale` / the output
  scale**: the expanded K/V would carry a 2^-8 factor; A16 pages (TMA-landed,
  unscaled) would not — wrong for the mixed module and the a16 path; and it
  needs a bound on the global scale the kernel cannot check without a host
  sync (C9: no host bound).  Rejected.
- **Vote elided when `|g| <= 0.5`** (then `s * g < 255.5` for every E4M3 s):
  correct as a per-operand uniform predicate, but the bench's `g = 1` takes
  the voting path; no gain on the locked bench.  Not designed (could be added
  as a uniform pre-branch later at +2 instructions on the voting path).
- **fp8 read permutation (blocks in `k ^ (x & 3)` register order, scale bytes
  permuted by one PRMT)**: saves the 3 read XORs with 1 PRMT and no extra read
  registers, but the store bases then differ between fp8 (permuted) and fp4
  (natural), which the mixed module would have to carry twice (16 store
  registers).  The 4 read-base registers (3.2) give the same count with one
  layout for all modules.
- **fp16 math** (F2FP.F16.E4M3 + HMUL2.F16 = 2 per pair instead of 4): a
  different math type from the model's bf16 cache; out of contract.
- **Head-dimension permutation of Q and K so pairs {b2:b0} come from one
  shift**: A16 pages land in natural order; V's output order would need an
  epilogue permutation.  Rejected.
- **Zero-fill of bad pages by cp.async `src-size 0` in the copy path** (would
  remove the per-tile tag test: -3): belongs to the copy-issue design
  (r5copy); noted for it, not counted here.

### 3.7 Why the placement stays at 3 per pair

A placed bf16 pair needs, per half, 7 value bits moved by 4 (fp8) / 2 (fp4)
positions and the sign moved to bit 15 with zeros elsewhere.  A 32-bit shift
moves both halves' value bits together only when the two source bytes are 16
bits apart, i.e. for the pairs {b2:b0} / {b3:b1} — the wrong output order; the
PRMT byte spread (with sign replication into the neighbouring byte) is what
makes one shift and one 3-input mask serve both halves of an in-order pair.
Any variant built from SHF / LOP3 alone needs two shifted copies and two masks
per pair (4); the f16-cvt detour (`F2FP` then `SHF`, `LOP3`, `IMAD`) is 4; a
LUT is >= 8 per 8 values (P0.4).  So 96 stays.  HMUL2 32 is one rounding per
pair, the reference's.  Stores: STS.128 is the widest; 8 per lane-tile.

## 4. Invariants and bit-exactness

### 4.1 Items touched

- **Landing layout of packed rows** (E4M3 swizzle `c ^ (r % 8)`, E2M1 80 B
  stride; copy path `:3170-3195`): unchanged.  The read bases `rd8[k]` encode
  the same swizzle as today's `row ^ (k * 16)` (`(4p + k) ^ x` per chunk).
  Coordination with r5copy: if that design changes the landing layout, the
  read constants here follow it (they are a function of the layout only).
- **Output layout** (A16 rows, chunk `(2b + g) ^ x`): unchanged; `st[k][g]`
  encodes exactly `a16Row ^ ((2k + g) * 16)`.  The two-way bank-group
  structure of the 8-lane STS phases is a function of the addresses, not of
  how they are formed: unchanged (8 distinct groups per phase).
- **Read-before-write ordering within the page** (header `:540-543`): the
  mid-body `__syncwarp` stays; the trailing one is removed with the argument
  in 3.5.  The "every read precedes a warp sync and every write follows it"
  statement remains true per tile.
- **Scale ring** (`kScales / vScales`): 4 entries indexed by `t % 4` -> 3
  indexed by the stage.  Readers and writers are the same converter warp
  (`:2675 / :2691`, `:2738 / :2750`; no other role touches them).  Copies
  for tile t + 2 write slot (t + 2) % 3 after `consumed.wait_parity` of that
  stage, i.e. after gemm0 released tile t - 1, whose scales this warp read
  during tile t - 1's expansion — before, in this warp's program order, its
  tile-t expansion and hence its tile-(t + 2) issue.  Same argument as the
  packed rows' (they share the stage).  smem: -1 KB (two rings of 512 B).
- **Fold decision** (C9-analogue in `backends.md` A8): same predicate,
  evaluated as a byte threshold (4.2).  `foldOk` unchanged.
- **Barrier accounting, phases, `TileRecord`, tags** (lever [8] C-items):
  untouched; the tag byte is still read at copy issue and rotated.

### 4.2 Bit-exactness, per format

The produced bf16 bits are `HMUL2.BF16(place(x), sf2)` with the same
`place(x)` and the same `sf2` values as today, so the outputs are identical if
(i) `sf2` is unchanged, (ii) the fold decision is unchanged, (iii) the
fallback arm is unchanged.

(i) `sf2[k] = bf16_rn(f32(s_k) * gfold)` broadcast: F2FP.PACK_AB rounds its
two inputs independently; with equal inputs it yields the bf16 the pair form
gave for that input.  Fallback: `bf16_rn(f32(s_k) * g)` likewise.

(ii) Today: fold iff `foldOk && for all k: |f32(s_k) * gfold| < 255.5 * 2^120`
(fp32, `.ftz` multiply — inputs are fp32-normal on the fold path because
`foldOk` guarantees `|s g| >= 2^-126` and `s >= 2^-9`; the profile's
`FSETP.GEU` is the negation).  `f32(s)` is exact (E4M3 -> f16 -> f32 are
embeddings, subnormal scales included) and monotone non-decreasing in the
7-bit magnitude code; `x -> |x * gfold|` under round-to-nearest is monotone
non-decreasing; so the set of passing codes is a prefix `[0, thr]` and the
per-block predicate equals `mag(s_k) <= thr`.  `thr` is found with the same
instructions (cvt, HADD2.F32 route, `FMUL.FTZ`, `<` compare), so no
arithmetic is re-derived.  Corner cases: `g = 0` -> every finite code passes,
thr = 0x7E, fold as today (products 0); `gfold = +-inf` (|g| >= 2^8) or NaN
-> no code passes, thr = -1, `voteAdd = 0x80808080`, the vote fails as today
(FSETP.GEU is true for inf and NaN); negative g -> `fabsf`, as today; NaN
scale code 0x7F -> `mag > thr` always -> fallback (today: `fmaxf` drops NaN
operands and may fold; the sealer never emits 0x7F and the harness remaps it —
outside the contract, noted as the only behavioural difference, in the
direction of the cvt path).

(iii) Fallback arm: unchanged body, same broadcast argument.

Per format:

- **E4M3 payload, subnormals and +-448**: placement `(prmt << 4) & 0x87F087F0`
  is byte-exact `x * 2^-120` for every finite code (subnormal `m * 2^-9` lands
  as bf16 subnormal `m * 2^-129`; `mul.rn.bf16x2` handles subnormal inputs —
  the [16] exhaustive check on H200); 448 -> `448 * 2^-120 * sf2`.  Unchanged
  instructions, unchanged operands.
- **E4M3 block scales up to 448**: `s g >= 255.5` fails the vote (today and
  new: `thr` at g = 1 is code 0x77 = 240; 256 (0x78) and above fail) -> two-
  multiply arm, exact by [16]'s argument (`2^120` multiply is exact for E4M3
  magnitudes; then one rounding).
- **Tiny global scale** (`|g| < 2^-117`, `tinyglobal` regime): `foldOk` false
  -> the branch takes the fallback regardless of the byte test, as today.
- **E2M1**: placement `(prmt(w4, w) << 2) & 0x81C081C0` = `mag * 2^-126`
  (code 001 -> bf16 subnormal 0.5 x 2^-126); fold iff `|s g| < 3.99` via the
  same threshold construction with `gfold = g * 2^126` and the same compare
  constant (the kernel's `foldScalesFinite` uses the E4M3 bound
  `255.5 * 2^120` for both formats with `gfold` carrying the format's power;
  the threshold search evaluates that exact expression).
- **Zero fill / A16 pages**: untouched paths.

Acceptance is the existing bit-exact matrix (72 cases incl. `subnormal`,
`maxscale`, `tinyglobal`, tails), run once after the SASS gates (7.3).

## 5. Register and shared-memory budgets

- Converter role today: peak `R37` in every role listing (fp8, fp4, mixed; K
  and V), i.e. 38 of the 56 registers `setmaxnreg.inc 56` grants
  (`:1433-1449`); 18 spare.  New lane constants: +10 (static fp8: 13 vs 3),
  +7 (fp4), +10 (mixed), plus `voteAdd` (1, or a UR).  Predicted peak ~48-49
  of 56 in every module; the mixed module keeps its per-tile `foldMultiplier`
  recompute (2 instructions) rather than hoisting two more registers.
- **ptxas C7507 rule** (backends A4, `:1433-1444`): the converter role's need
  must stay <= 56 and no role may need more than its budget, else ptxas drops
  every `setmaxnreg` silently and the kernel runs at 48 with spills.  Gates:
  `ptxas -v` prints no C7507 and `0 bytes spill`, `cuobjdump -res-usage`
  REG 48 STACK 0, SASS has exactly two `USETMAXREG` (DEALLOC 0x28, TRY_ALLOC
  0x38).  The IO (40) and GEMM (40) roles are untouched.
- If ptxas rematerialises the store bases instead of keeping them live (it
  would show as `LOP3 ... 0x10..0x70` in the fold body), the design falls
  back to 4 store bases (blocks) + 4 `LOP3 ^ 0x10` per tile (-3 instead of
  -7); if the mixed module spills, its fp8 reads fall back to the 3 XORs
  (`#if MIXED_PAGE_STATIC_FORMAT < 0`, the existing knob pattern), -1
  register net of the 4 bases.
- Shared memory: `kScales / vScales` 4 -> 3 entries: -1 024 B.  Nothing else
  moves (the TileRecord ring, stage rings and barriers keep their offsets;
  the a16 module does not include the scale rings' users but shares the
  struct — its `SharedMem` shrinks by the same 1 KB, an immediate change only,
  no layout change elsewhere: gate = a16 SASS count within +-8 of 2 488).

## 6. Predicted counts, depth, and wall

### 6.1 Per lane-tile after (fold path)

| class | fp8 before -> after | fp4 before -> after |
|---|---:|---:|
| PRMT | 36 -> 32 | 36 -> 32 |
| SHF + IMAD.SHL | 34 -> 33 | 41 -> 40 |
| LOP3 | 42 -> 34 (32 masks + 2 vote) | 39 -> 34 |
| IADD3 | 2 -> 1 (vote) | 1 -> 1 |
| HMUL2 | 32 -> 32 | 32 -> 32 |
| STS.128 | 8 -> 8 | 8 -> 8 |
| LDS | 5 -> 5 | 3 -> 3 |
| F2FP | 4 -> 6 | 4 -> 6 |
| HADD2 / FMUL | 4 / 4 -> 4 / 4 | same |
| FMNMX / FSETP | 3 / 1 -> 0 / 0 | same |
| VOTE / NOP | 1 / 2 -> 1 / 1 | same |
| BRA | 3 -> 2 (bad-page, fold) | same |
| uniform ops (ULEA x2, UISETP) + PLOP3 | 5 + ... -> 4 | 3 -> 3 (ISETP form) |
| VIADD / IMAD.U32 / MOV address ops | 5 -> 0 | 3 -> 0 |
| **total** | **188 -> 167** | **188 -> 172** |

Mixed, per compressed page: fp8 195 -> ~176, fp4 197 -> ~181 (the 15-
instruction dispatch prefix loses its 4 scale-address ops, keeps the three
uniform branches and the fold-multiplier recompute); A16 pages 2-3
unchanged.  Fallback arm: 223 -> 202 / 207.

Per tile-SM (8 warps): fp8 expansion 1 504 -> 1 336 (-168); fp4 1 504 -> 1 376
(-128); mixed ~1 050 -> ~960 (-90).  SM totals: fp8 4 150 -> 3 982 (-4.0 %),
fp4 3 881 -> 3 753 (-3.3 %), mixed 3 580 -> 3 490 (-2.5 %).  Converter warp
per tile (fp8, with copy issue ~104 and loop glue ~37 unchanged): 329 -> 308
(-6.4 %).

### 6.2 Dependency depth

| chain | before | after |
|---|---|---|
| entry -> fold branch resolved | 14 (4 uniform/address, LDS, SHF, F2FP.E4M3, HADD2, FMUL, 3 FMNMX, FSETP, VOTE, BRA; two XU/FMA-pipe stages) | 7 (ULEA, LDS, LOP3, IADD3, LOP3.P, VOTE, BRA; ALU only) |
| entry -> first sf2 | 12 (address 4, LDS, SHF, F2FP, HADD2, FMUL, F2FP.PACK, PRMT) | 8 (ULEA, LDS, SHF, F2FP, HADD2, FMUL, F2FP.PACK); runs parallel to the vote, not behind it |
| entry -> first STS | ~19-20 | ~12 (vote 7 + PRMT, SHL, LOP3, HMUL2, STS) |
| per store | LOP3 (XOR) -> STS | STS |
| fp8 packed loads | VIADD -> IADD3 -> LOP3 -> LDS (3 of 4) | LDS (all 4, issued at entry) |

Issue-slot effect: the removed instructions were address / vote filler that
was *not* covering latency (the region's stall samples sit on the scale-LDS
consumer and the vote branch, not on the removed ops), so the count cut should
not be repaid in exposed latency; the shorter decision chain removes ~30-40
cycles of serialisation per tile from the converter warp's critical path.
This is the difference from Track S step 6, where the cut was in copy issue
and the freed slots became long-scoreboard waits on the page loads.

### 6.3 Wall, issue-bound model

Model (round-3 pair doc, round-4 [7] rev 2 sections 2.3-2.5): at two CTAs per
SM the cadence is the SM's per-tile issue demand at the pair's IPC, `cadence =
N / (IPC * f)`, with N = 4 150 (fp8), IPC 2.48, f = 1.98 GHz -> 0.845 us
(measured 0.83-0.85).  Assumptions: (a) IPC unchanged by the cut (the freed
slots were filler, 6.2); (b) the fixed (non-body) part of the wall is
unchanged (fp8 67.9 - 56.6 = 11.3 us; fp4 / mixed taken as 11 us); (c) the
consumer chain (~0.79 us) stays below the issue cadence.

| mode | N before -> after | body before -> after (us) | wall before -> after | target |
|---|---|---|---|---|
| fp8 | 4 150 -> 3 982 (-4.0 %) | 56.6 -> 54.3 | **67.9 -> 65.6** (-2.3) | <= 58 (no) |
| fp4 | 3 881 -> 3 753 (-3.3 %) | 49.6 -> 48.0 | **60.6 -> 59.0** (-1.6) | <= 36 (no) |
| mixed | 3 580 -> 3 490 (-2.5 %) | 53.6 -> 52.3 | **64.6 -> 63.3** (-1.3) | <= 62 (no) |

Band: -0.5 .. -3.0 us (fp8).  Where the argument fails: (1) if the converter
warps are stall-bound rather than count-bound at the pair's issue share, the
count cut buys nothing (Track S step 6's failure mode) — mitigated but not
excluded by 6.2; the trace gate in 7.4 decides.  (2) If the cadence is already
chain-bound after the other round-5 cuts (issue cadence below ~0.79 us), this
lever's gain is zero there and the consumer-side levers ([7]) return.  With
the three round-5 cuts together (copy issue ~-360, descriptors ~-500,
expansion -168: ~-1 030 of 4 150, -25 %) the issue cadence drops to ~0.63 us,
below the chain — the combined tree will be chain-bound and the round-4
estimate "fp8 body toward 42-45 us" is the chain's number, not the issue
model's.  (3) IPC may rise (shorter chains) or fall (the descriptor cut
removes uniform-datapath ops that issue cheaply): the band covers +-1 us.

## 7. Verification artifacts, accept / reject (SASS before any timing)

Recipe: rsync the worktree to `nkcut2:/home/bigboi/dash-flashinfer-claude-r5expand/`
(symlinks once, touch the `.cu/.cuh` after every rsync), compile-only with
`/tmp/r4p7_sass/build.sh` (path substituted) for formats 1, 2, -1, lineinfo
and plain.

7.1 **ptxas / res-usage** (all three modules): no C7507; `0 bytes spill`;
REG 48, STACK 0, LOCAL 0; exactly two `USETMAXREG`.  Reject on any miss.

7.2 **SASS, `xqa_sm90_converter_sass.py --paths` and `--dump K-expand`** (K
and V within +-1):
- executed per lane-tile, fold path: fp8 **<= 168** (accept), fp4 **<= 173**;
  reject if fp8 > 175 or fp4 > 180 (bases rematerialised or vote not fused).
- fold body: PRMT = 32, LOP3 = 32 (+2 in the vote prefix), HMUL2 = 32,
  STS.128 = 8 all `[R+UR]`, no `LOP3 ... 0x10|0x20|...|0x70`; fp8 LDS.128 = 4
  all `[R+UR]`, fp4 LDS.128 = 2 `[R+UR(+0x10)]`; scale LDS `[R+UR]`.
- FMNMX = 0, FSETP = 0 in the region; F2FP = 6 (2 E4M3 + 4 PACK_AB); HADD2 =
  4; FMUL = 4; VOTE = 1; BRA on the hot path <= 2 (bad-page, fold); the
  fold's hot arm is the fall-through (else count 168 / 173, still accept).
- static region: fp8 <= 360, fp4 <= 370, mixed <= 720 per role; a16 module
  SASS 2 488 +- 8 and its converter/IO regions text-identical except
  `SharedMem` immediates.
- mixed dispatch prefix unchanged in structure (three uniform branches).

7.3 **Correctness**: the standard remote run's 72-case matrix (bit-exact) for
transport_a16 / fp8 / fp4 / mixed, incl. the `subnormal`, `maxscale`,
`tinyglobal` regimes; 72/72 or reject.

7.4 **One confirmation run** (`flock /tmp/mixedkv-gpu0.lock bash
/home/bigboi/mixedkv_remote_run.sh <checkout> r5expand sm90 transport_a16 fp8
fp4 mixed`; co-tenant rule repeats x t < 1.5 ms; min / median / max):
- ncu `SourceCounters` (`--clock-control none` for anything compared to bench
  time; instruction counts are clock-invariant): `/tmp/r4p7_body.py` method,
  K + V converter body per tile-SM fp8 <= 2 480 (today 2 632; expansion part
  1 336 +- 24); `smsp__inst_executed.sum / 8712` fp8 <= 4 000, fp4 <= 3 780,
  mixed <= 3 510.
- trace (`MIXED_KV_TRACE=1` copy, `parse_xqa_trace.py`): K expansion segment
  (s13 - s12) fp8 lone-CTA 816 -> <= 760 cyc accept (predicted 725); >= 800
  = stall-bound, the count did not buy time (record, keep the tree only if
  7.5 is neutral or better).
- wall (locked bench medians): accept fp8 <= 66.0, fp4 <= 59.5, mixed <= 63.6;
  a16 79.4 +- 0.5.  Reject (revert) if any compressed mode is > +0.5 us over
  its locked median or a16 moves > 1 us.

7.5 State the SM clock in every table (bench ~1.98 GHz under the co-tenant;
ncu `--clock-control none`).

## 8. Do not build if

1. r5copy's design changes the E4M3 landing layout or the scale ring
   ownership (this design's read constants and the stage-indexed scale ring
   assume the current copy path); merge the two designs first.
2. r5desc / r5copy measurements show the production cadence already
   chain-bound (< 0.80 us) — then build this only inside the combined tree,
   never as a standalone timing experiment (its standalone gain would read as
   noise).
3. The compile-only SASS misses 7.1 or the fold body still carries XOR LOP3s
   after the 4-base fallback (7.2 reject line): the register budget argument
   in section 5 is wrong and the design needs the [15]-style budget rethink.
4. The mixed module spills with the fp8 read-XOR fallback applied.

## 9. Go / no-go

**Go, as a component** of the round-5 instruction-stream cut; **no-go as a
route to <= 150**: the target is below the essential floor (fp8 141, fp4 147)
plus the contract-bound scale route (15) and fold decision (4); 167 / 172 is
the reachable count at unchanged bit-exactness, unchanged layout and unchanged
protocol.  The design removes only address and vote glue (store and read
bases as lane constants addressed `[R + UR]`, stage-indexed scale ring,
byte-threshold vote, direct bf16 broadcasts, one warp sync and one branch),
halves the decision chain (14 -> 7) and cuts the SM's issue by 4.0 / 3.3 /
2.5 % — predicted fp8 65.6, fp4 59.0, mixed 63.3 us (band +-1), no target
reached alone.  Every gate is a SASS reading before a single timing run.

Artifacts: `/tmp/r5expand/` (local): `f{1,2,-1}_Kexpand.{sass,dyn}`, `dyn.py`,
`regs.py`; nkcut2 `/tmp/r4p7_sass/f{1,2,-1}_li.nvdis`,
`/tmp/r4mixed2_pcs_*_P132.csv` (read only; no GPU job started).
