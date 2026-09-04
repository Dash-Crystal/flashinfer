# Mixed KV speed, round 5, Track F: lever [27] — the pair as two half-pairs, the top of the pair made branch-light, the store-burst drain priced as pipe time, the class-sorted dynamic module

**Revision 2** (judge blockers answered in section 11; the design changed
where they required it: the MIO-operand pins are withdrawn - an empty `asm
volatile` emits nothing ptxas can see, and the stall they were to remove is
the body's store burst draining through the MIO pipe, not a register
allocation - the arms are re-priced at pipe time, the centre is the former
pessimistic column, the dynamic module's entries carry their slots in the
row's word 6 (no page-index bound), and the decode loops index slots by a
register shift, not a register array.  Section 12 records the code as
written per step.)  Base: `claude/mixed-kv-sm90-tma`
@ `659eacfa` (wt/F26ab merged: F26a + F26b + tests; F26c's per-slot dynamic
bodies NOT merged).  Extends `docs/mixed_kv_speed_round4_fa3_f26.md` (F26
design; section 12 "as measured"), `docs/mixed_kv_speed_round3_fa3_f25.md`,
the Track F [25] / [26] sections and the "F26a+b merged-tree confirmation" of
`docs/mixed_kv_page_transport_backends.md`, and
`docs/mixed_kv_page_transport_dataflow.md` (D1-D6, C1-C19, A1-A10).  Every
measured number below is read from the F26 artefacts on
`nkcut2:/tmp/mixedkv-wtF26-art/` (`ncu_read.log`, `ncu_read_a16.log`,
`ncu_{fp8,fp4,mixed,a16}_source.csv` through `f26_pc.py` / `f26_ncu_read2.py`,
`sf{1,2,-1,0}_mask0.sass` through `f26_seg.py`, `counts.log`,
`loopsite_*.log`, `ptxas_*.log`, `trace_q1.txt`, `bench_final.txt`) and from
one F26ab artefact made for this document by reading an existing object (no
build): `nkcut2:/tmp/mixedkv-wtF27-art/f26ab_sf-1_mask1.sass` =
`cuobjdump -sass` of the merged tree's dynamic module
(`/tmp/mixedkv-wtF26ab/.cache/.../static_format_-1/..._mask_1.cuda.o`,
mtime 2026-09-04 19:24), read with `f25_loopsite.py` / `f25_counts.py`.
Addresses are the mask_0 objects' (the profiled ones) unless stated; 1 % of
fp8 loop-site samples = 26.7 issue cycles (loop pair 2667), fp4 28.8 (2876),
mixed 81.8 (8178).

## 0. Summary

State (nkcut2 H200, q=1 / q=64 us, merged tree): stock FA3 A16 301.4 /
310.4, transport_a16 282.5 / 289.1, **fp8 366.3 / 388.8, fp4 392.5 / 400.3,
mixed 635.1 / 645.2**; target <= 330.  T_c = 282.5 us / 220 pairs = 1.284 us
= 2544 wall cycles; the accept line <= 330 us is a wall pair of 2970 = an
issue pair of ~2590 (F26's x 1.148 convention); consumer-bound needs an issue
pair <= ~2216 (wall 2544).  F26 fp8 measured **2667 issue cycles** for 651
warp-instructions per pair per warp (IPC 0.24): the producer paces, barely
(consumer K-wait 3.8 %, tensor pipe 51 %).

**What the F26 samples say (section 1), in one sentence each:**

1. **Top of pair, 367 cycles for 36 instructions**: 72 are the consumer's
   slack (the two spins, taken on ~4 % of pairs, ~1000 cycles each); the
   other ~295 are five conditional branches (gather test, store test, spin K,
   spin V, and the back-edge) with their predicate waits (~50), branch
   resolution (~50), instruction-fetch bubbles after the three taken ones
   (~65: `no_inst` at `0xa770`, `0xa910`, `0xaae0`), two `BSSY` / `BSYNC`
   pairs around the per-thread spins (~20), and the stage-base
   recomputation chain (~30).
2. **The arm segments, 1414 vs 835**: it is neither the fold / `HMUL2`
   dependency nor the store address chains.  It is **the store burst draining
   through the MIO pipe, surfacing as a write-after-read scoreboard wait**:
   an `STS.128` holds its address and data registers until the MIO reads
   them, all twelve `STS.128` of a body share one read scoreboard (control
   bits in `sf1_mask0.sass`: the K hot arm's stores all set `rd = SB3`, the V
   arm's `SB2`; fp4 25 on `SB1`, 23 on `SB2`), so the first instruction after
   the body that writes *any* of the ~50 registers the burst reads waits for
   the *last* store's operand read - i.e. for the burst to drain at the pipe's
   ~8 cycles per wavefront (16 wavefronts x 8 = 128 at `0xd4e0`; the V arm's
   8 at `0xf380`).  Revision 1 read this as ptxas register reuse curable by
   pinning; revision 2 withdraws that (3.2): the wait is pipe time, it can be
   moved but not removed by allocation, and an empty `asm volatile` pin
   emits nothing (compile-only probe `f27_pin_probe.ptx`, section 11).
   Three PCs carry 305 cycles of the fp8 pair: `IMAD R61 ... 0x6000` at `0xd4e0` (127
   cycles, `short_sb` 98 %; R61 was `STS.128 [R103+0x2000], R60..63` seven
   instructions earlier), `VIADD R12` at `0xf380` (72; R12 was the data of
   `STS.128 [R61+0x800], R12` at `0xecd0`, 107 instructions and >250 cycles
   earlier), `LDS.128 R32` at `0xaf30` (105; R32 was the address of the
   twelve landing `LDS.64 [R32+...]` issued just before).  fp4 has the same
   four (`0xd8e0` 114 on the `STS` *address* register R110, `0xf8f0` 117,
   `0xafb0` 103, `0xd000` 30 = 364 cycles).  The rest of the arm excess is
   the MIO dispatch back-pressure itself (`dispatch` 23-26 % on the
   instruction after every `STS.128` group; the K body at IPC 0.34 behind
   the 72-wavefront copy burst, the V body at 0.46 - F25's bodies ran at
   0.53-0.55 behind a copy block that issued three times slower).  `0xaf30`
   (105) is not a WAR at all: `LDS.128 R32` waits `[SB1, SB2]` and SB1 is the
   *write* barrier of the 24 landing `LDS.64` (ptxas merged wr and rd on
   SB1) - the landing loads' data latency, which the row hoist (3.1) takes
   off this PC.
3. **The octet scale copies at 11.9 wavefronts per instruction** are at the
   floor of the *source layout*: the bench's and the quantizer's scale
   tensor is `[pages, tokens, heads, 8]` (`bench_fa3_mixed_page_transport.py:28`,
   `kv_cache_fp8.py`), so with 8 KV heads a page-head's sixteen 8 B rows
   sit two per 128 B line in eight lines, an octet instruction (six pages x
   four rows) touches 12 lines, and an `LDGSTS` costs one wavefront per
   distinct global line (the payload `.128` at 4.00 = 4 rows in 4 lines,
   the fp4 `.64` at 4.00 = 4 rows in 4 lines with 2 the smem ideal, the
   octet at 11.93 = 12 lines).  Per operand per CTA the six pages' 48 lines
   are each touched exactly once already (4 warps x 12 = 48 = F25's twelve
   leader-lane copies x 4 warps x 1.98).  **No kernel-side form lowers it;
   a host-side scale layout `[pages, heads, tokens, 8]` (token stride 8 B)
   makes a page-head's rows one line** and then two kernel forms exist
   (3.3): in-warp `.128` row-pair copies (6 wavefronts per instruction, 48
   per pair per CTA from 89, "conflicts" 1.99 M -> ~0.7 M) or two-warp
   page ownership behind a per-stage scale mbarrier (3 per instruction on
   two warps, 12 per pair, ~0.1-0.2 M).  The copy block's time is ~8
   issue cycles per wavefront per warp (fp8 581 / 72 = 8.1; a16 773 / 96 =
   8.1), i.e. the LSU pipe's residual capacity with the consumer's 0.39
   wavefronts per cycle - so the octet copies cost ~95 cycles per pair
   today and ~48 / ~24 under the two forms.
4. **The dynamic module** (F26a order, 635 us): its loop site is 2380
   instructions (52 predicated `LDGSTS` + 52 fused `IMAD.WIDE.U32` in the
   copy body, four rolled format loops of 248 / 284 instructions of text
   with 40 `FLO` + 40 `BREV`, 15 `BSSY` / 15 `BRA.DIV`, 0 spills) - the F25
   shape with F26a's addresses.  F26c's per-slot bodies spilled (128 B
   frame) because six copy bases + two masks + both operands' rows were
   live across twelve branched slot bodies.  F27 keeps the predicated
   straight-line copies (their ~200 predicated-off issue slots overlap the
   LSU-paced copy block) and replaces the mask-driven loops by
   **class-sorted entries**: the chunk store writes each tile's six (page,
   slot) entries in format order (fp8, fp4, a16) with the counts n8, n4 in
   the row word (once per 16 pairs, ~15 instructions per lane), so the
   decode is two rolled loops per operand with trip counts n8 / n4, one
   shared hot / cold body per format (~59 instructions per block), the slot
   index as a register offset, software-pipelined loads, no `FLO` / `BREV`,
   no per-slot branch, and one block's state live at a time (~100 registers).
   The destination of every byte is unchanged (the slot travels with the
   entry), so the stage the consumer reads is identical: bit-exact.

**F27 = the F26 pair split into two half-pairs (K half, V half), each of
which is the F26 principle applied to one operand: `wait_group 1` -> the
operand's landing loads -> `__syncwarp` -> its scale loads -> its six + one
copies (the other ring's tile) interleaved with its scale chains -> commit
-> vote -> body**; two commit groups per pair (the K half's commit
unconditional, so the last pair's empty K group keeps the V half's
`wait_group 1` correct) with both landing covers ~1550 cycles (3.1); the
chunk rows read at the very top of the pair; the acquire probes voted
uniform (`__all_sync`, `hasK` folded in) so the spin is one uniform branch
whose body is a warp-uniform `try_wait` loop (no `BSSY` / `BSYNC`, no
`BRA.DIV`); the gather / store tests folded into one rare-taken test in the
tail under the fence drain; **no register pins** (3.2: the post-body wait is
the store burst's pipe time; nothing at 136 registers removes it, so it is
priced, not designed away); the fp4 copy block's ~60 spurious `VIADD Rx,
Rx, URZ` / `UMOV` / `IMAD.MOV` (3.6) hunted by a compile-only probe before
the build (authored in 12, not yet run); the dynamic module as item 4 with
its slots in the row's word 6.  The
static hot / cold bodies, the placement decodes, the E4M3 / E2M1 folds and
votes, the STS pattern, the 12-warp layout at 136 / 184 registers, the
barrier arrival counts, the a16 module and the consumer are untouched.

Instruction count is **not** the lever (fp8 651 -> ~655 per warp per pair;
+2 `DEPBAR`-class, +1 `NOP`, +3 fillers, -8 protocol); the lever is the
stall structure.  Revision 2 states three model columns for the fp8 issue
pair (4.2): **floor ~1985** (every segment at its pipe-time price: a model
output, not a prediction), **centre ~2300** (top, preps and copies at their
F27 prices, the arms at F26's 1414 - the queue they meet is shorter but the
burst drain lands in them), **derivable-only ~2465** (F26's 2667 minus the
top's branch structure, the one segment whose removal is argued PC by PC);
pessimistic ~2620 (the mid-pair `DEPBAR` exposes latency, copies at fp4's
price).  fp4 ~2025 / ~2530 / ~2700 / ~2900; mixed not claimed (~2650 /
~3300).

| mode (us, q=1 / q=64) | now (659eacfa) | F27 centre (derivable-only) | F27 band | accept <= 330 |
|---|---|---|---|---|
| stock_a16 | 301.4 / 310.4 | unchanged | 298-304 / 307-313 | control |
| transport_a16 | 282.5 / 289.1 | unchanged (module untouched) | 281-285 / 287-292 | control |
| fp8 static | 366.3 / 388.8 | **293 / 300** (**315 / 322**) | 290-334 / 297-341 | met at the centre and at the derivable-only column; the pessimistic end (2620 issue = 334 us) is a reject by 4 us - the accept is not robust to the arms getting *worse* than F26 (4.2) |
| fp4 static | 392.5 / 400.3 | **322 / 329** with the copy-block excess kept, **307 / 314** if 3.6's fix lands (derivable-only **344 / 351**) | 300-350 / 307-357 | met at the centre only with the copy block or the arms moving; the top alone does not reach 330 on fp4 (4.2) |
| mixed (dynamic) | 635.1 / 645.2 | **~340 / 348** (8 blocks per pair on the 2/2/2 mix, two post-body drains, loop bubbles) | 300-420 / 308-428 | **not claimed**: the ptxas / SASS gates of section 5 (0 spills, loop site <= 1000, `BSSY` 0 in the loops) decide before any bench; the model's own centre is above 330 |

**Is the consumer-bound floor reachable?**  Not by this design's own model:
the producer's pipe time per pair (4.3: (96 `STS` + 72 `LDGSTS`) x 8 + 68
`LDS` x ~3.7 = ~1590 wavefront-cycles per warp) plus the top (~165, pipe
idle after the fence) and the fence drain (~100) is **~1860 issue cycles
with perfect ALU overlap** - above the 2216 mark only if the overlap fails,
but the floor column (1985) is a sum of segment prices, not a derivation of
overlap, and revision 2 claims nothing below the centre.  The consumer-bound
regime (< 2216 issue) is therefore **not claimed** for any module; what is
claimed is the accept (<= 2590 issue = 330 us) at the fp8 centre and at
the derivable-only column, and for fp4 at the centre.  The floor itself
stays T_c (1 + c), c 2-4 %; nothing in F27 lowers T_c.

## 1. The F26 record re-read per segment (fp8 unless stated; `f26_ncu_read2.py` segments, `f26_pc.py --seg`)

Loop site `[0xa770,0xf520]` (mask_0), 1244 instructions, 627 executed per
warp per pair, 91.9 % of the producer's samples, `selected` 23.5 % -> 2667
issue cycles.  Stall mix: selected 23.5, dispatch 20.2, wait 15.8, short_sb
12.3, not_selected 10.6, long_sb 6.5, math 3.8, no_inst 3.6.

| segment (fp8) | instr text / exec per warp | cycles | IPC | stall mix | what the PCs say |
|---|---|---|---|---|---|
| top + spins `[0xa770,0xad20)` | 91 / 36 | **367** | 0.10 | wait 25, long_sb 18, no_inst 14, not_sel 12, branch_resolving 11 | below (1.1) |
| `DEPBAR` + 24 landing `LDS.64` `[0xad20,0xaec0)` | 26 / 25 | 169 | 0.15 | wait 40, mio 21 | MIO dispatch of 24 loads (~6-7 cycles each) |
| `NOP` + rows + scale `LDS` `[0xaec0,0xaf30)` | 7 / 7 | 17 | | | |
| copy block `[0xaf30,0xb440)` | 82 / 77 | **581** = 41 per `LDGSTS`, **8.1 per wavefront** (72 per warp) | 0.13 | dispatch 27, short_sb 24, wait 17 | `LDS.128 R32` `0xaf30` 18.0 % of the segment (105 cycles, short_sb 96 %: WAR on the landing loads' address register R32); `IMAD.WIDE.U32 R62, R32` `0xb0f0` 5.0 % (RAW on it); every `LDGSTS` and the instruction after it `dispatch` 60-95 % (`0xb1f0` 69 %, `0xb260 PRMT` 95 %) |
| K vote + arms `[0xb440,0xd570)` | 529 / 252 | **810** | 0.31 | selected 31, dispatch 26, short_sb 15, wait 11 | `IMAD R61, R93, 0x6000, R6` `0xd4e0` **15.7 % = 127 cycles**, short_sb 98 %; `VOTE.ALL` + `@P0 BRA` 5.4 % (44); hot-arm entry `PRMT` `0xc790` no_inst 2.3 % (19); the body's `LOP3` / `PRMT` / `SHL` PCs dispatch 30-50 %, not_selected 30-40 % |
| V vote + arms `[0xd570,0xf440)` | 493 / 216 | **604** | 0.36 | selected 35, dispatch 23, short_sb 11, wait 10 | `VIADD R12, R90, 0x1` `0xf380` **12.0 % = 72 cycles**, short_sb 91 %; vote + branch 5.1 % (31); entry `PRMT` `0xe650` no_inst 3.4 % (20) |
| probes + fence + tail `[0xf440,0xf530)` | 15 / 14 | 118 | 0.12 | long_sb 56 | `SYNCS.ARRIVE` `0xf4f0` long_sb 76 % (the fence drain) |

fp4 (`[0xa870,0xfa90]`, 2876): top 342, landing 200 + 13, **copy block 787
= 56 per `LDGSTS`** for 137 executed instructions (fp8: 77), K 795, V 623,
tail 116.  Its extra 60 copy-block instructions are `VIADD R61, R61, UR5` /
`UR6` / `UR7` / `UR8` after `UMOV UR5..8, URZ` (adds of a uniform zero to the
high or low word of a 64-bit source), `IMAD.MOV.U32` copies of the base
pairs and `IADD3 R31, R31, UR5` (`sf2_mask0.sass` `0xb160-0xb490`): a
lowering artefact of the fp4 `page_src` / `cp8` path that fp8 does not show
(3.6; a compile-only probe names it before the build).  fp4's WAR PCs:
`IMAD R110, R85, 0x6000, R10` `0xd8e0` 14.3 % of the K arm = 114 cycles -
R110 is the *address* register of `STS.128 [R110+0x2800], R60` four
instructions earlier; `VIADD R20, R70, 0x1` `0xf8f0` 18.8 % of the V arm =
117; `LDS.128 R20` `0xafb0` 13.1 % of the copy block = 103; `IMAD.SHL.U32
R20, R63` `0xd000` 3.8 % = 30 (R20 = data of `STS.128 [R111], R20` twelve
instructions earlier).  Also `BRA.DIV UR5` `0xb930` 4.1 % of the K arm = 32
cycles long_sb (the `__syncwarp` reconvergence check waiting on the
scoreboard of the landing loads).

### 1.1 The top of the pair, PC by PC (367 cycles; `f26_pc.py ... 0xa770 0xad20`, 5740 samples)

| mechanism | PCs (share of the segment) | cycles |
|---|---|---|
| consumer slack: the two spins, entered on 3726 / 91392 = 4.1 % (V) and 3.6 % (K) of pairs, ~1000 cycles per entry | `@!P0 BRA` `0xac20` 11.8 %, `0xabb0` 7.8 % (long_sb 94 %) | **72** (stays: it is the consumer) |
| instruction fetch after taken branches | `LOP3 P0` `0xa770` 6.6 % (the back-edge), `BSYNC` `0xa910` 5.3 % (target of the skip-gather `@P0 BRA`), `LOP3 P0` `0xaae0` 5.6 % (target of the skip-store `@P0 BRA`) - all no_inst 64-75 % | **65** |
| predicate -> branch waits | `@P0 BRA` `0xa940` 5.1 % (wait 77 %), `0xa7a0` 4.2 %, `0xab70` 2.0 %, `@P1 BRA` `0xabe0` 1.4 % | **46** |
| branch resolution | `ISETP` `0xa920` 5.4 % (branch_resolving 71 %), `BSSY` `0xabd0` 4.6 %, `LDGDEPBAR` `0xac40` 3.9 % | **51** |
| `BSSY` / `BSYNC` of the two per-thread spins | `0xab10` 1.7 % (barrier 78 %), `0xabc0` 2.0 %, `0xac30` 1.9 % | **21** |
| stage / phase / base chain (`VIADD`, `ISETP`, `SEL` x 3, `LOP3` x 5, `IMAD` x 5) | `0xacd0` 2.0 %, `0xad10` 2.0 %, `0xad00` 1.6 %, `0xab60` 1.7 %, `0xab30` 1.3 %, `0xab40` 1.1 % | **36** |
| issue of the rest (36 instructions) and the `LDGDEPBAR` of `cp_async_wait<0>` | remainder | ~75 |

Reading: **the top of the pair is five conditional branches' worth of
fetch / resolve / predicate latency (~180 cycles) plus the slack**, not
instruction count.  Every taken uniform branch costs ~20 cycles of `no_inst`
(the fetch redirect; Hopper has no branch prediction) and every conditional
branch ~6-15 of predicate wait and resolution.  The gather / store rare
paths are laid out inline, so skipping them *takes* the branch.

### 1.2 The arms: the MIO-operand WAR, quantified

The K hot arm `[0xc790,0xd4b0)` (210 instructions) ends with four
back-to-back `STS.128` (`0xd470-0xd4a0`: `[R103+0x2000], R60`,
`[R102+0x2000], R64`, `[R103+0x2800], R68`, `[R102+0x2800], R72` = 16
wavefronts per warp, 64 for the CTA), then `FSETP` x 6 / `PLOP3` x 3 (the V'
vote's chain) and `IMAD R61, R93, 0x6000, R6` (the V' stage base for the V
body's `STS [R61 + imm]`) at `0xd4e0`: **127 cycles, short_sb 98 %**.  The
`IMAD` writes R61 while the `STS.128 [R103+0x2000], R60` (R60-R63) seven
instructions earlier has not yet read R61: the MIO queue at that point holds
the K body's twelve `STS.128` (48 wavefronts per warp) behind whatever of the
copy block's 72 the pipe has not drained.  In the V arm, `STS.128 [R61+0x800],
R12` at `0xecd0` (block 1's second store) is still holding R12 when `VIADD
R12, R90, 0x1` at `0xf380` (107 instructions later, after the V body's twelve
stores) writes it: **72 cycles** - the store queue is > 250 cycles deep at
the end of the V body.  In the copy block, `LDS.128 R32, [R72+0x30870]` at
`0xaf30` writes R32, the address register of `LDS.64 R40, [R32]` ...
`[R32+0x2800]` (six of the twenty-four landing loads issued 20-25 slots
earlier): **105 cycles**.  fp4 repeats the pattern on the store *address*
register (`0xd8e0`, 114).  Total: fp8 **305** cycles per pair on three PCs,
fp4 **364** on four.

Why ptxas does this: it schedules and allocates with the MIO operand read as
an issue-time event (an `STS` frees its sources when issued), and at 136
registers with ~130 live during the copy block the freed sources are the
free pool.  The mechanism is invisible to the count and to the F25 record
because F25's copy block (1160 cycles for 24 `LDGSTS`) issued three times
slower than the LSU could drain it, so its bodies met a near-empty queue
(IPC 0.53-0.55, dispatch 15 %); F26's block issues 72 wavefronts per warp in
581 cycles - **exactly the LSU pipe's residual rate** (4 warps x 72 / 581 =
0.50 wavefronts per cycle + the consumer's 1001 / 2544 = 0.39 -> 0.89 of a
~1.0-1.2 peak; `l1tex` shared pipe 52.8 % of peak over the whole kernel) -
and the K body's 48 wavefronts queue behind it.  **F26 did not remove the
LSU time of the copies, it moved it into the arms**: F25 copy + arms 1160 +
395 + 380 = 1935; F26 581 + 810 + 604 = 1995.

What is *not* the cause: the `STS.128` pattern (4.00 wavefronts per PC, ideal,
unchanged), the fold / `HMUL2` chain (the `HMUL2` PCs sample math 40 % /
dispatch 25 % at ~1 % each, no short_sb), the store address arithmetic (the
`STS` use `[R + imm]` from two per-operand bases).

**Re-read for revision 2 (control bits, `sf1_mask0.sass`; judge's
`ctrl.py`).**  The wait masks say what the PCs alone could not: `IMAD R61`
at `0xd4e0` waits `[SB2, SB3]` and every one of the K hot arm's twelve
`STS.128` sets `rd = SB3`; `VIADD R12` at `0xf380` waits `[SB1, SB2]` and
the V arm's twelve set `rd = SB2`; fp4's 48 stores split 25 on `SB1` / 23 on
`SB2`.  A read scoreboard is a per-index counter shared by every instruction
that set it, so **writing any register read by any store of the body waits
for the body's last store to read its operands** - R12 at `0xf380` is block
1's output, seven stores before the end, and would not have been covered
by an 18-register pin set.  What the wait measures is the trailing burst
draining through the MIO pipe at the same ~8 cycles per wavefront the copy
block (581 / 72) and F25's bodies (395 / 48) show: 16 wavefronts x 8 = 128
at `0xd4e0` (measured 127), 8-9 x 8 at `0xf380` (72).  **Register
allocation cannot remove it**: at the end of a body the ~50 store-operand
registers *are* the free pool, and the next instructions (the V' stage
base, the probe addresses, the next half's 24 landing-load destinations)
must write registers; whichever ptxas picks waits on the burst.  The pins
of revision 1 (a) emit no PTX instruction (`f27_pin_probe.ptx` lines 76-77:
`// begin inline asm` immediately followed by `// end inline asm`; ptxas
computes liveness from PTX instructions and rewrote the `LDGSTS` address
pair 3 instructions after the `LDGSTS` and `STS` data 19 after the `STS`
in the probe SASS) and (b) would not have helped if they had: a use keeps
18 registers out of the pool, the writer takes a nineteenth from the same
scoreboard set.  The 305 / 364 cycles are therefore **pipe time,
relocatable but not removable** at 136 registers: the half-pair order moves
the queue the bodies meet from 72 + 48 wavefronts to ~30 and puts the
K' drain under the V half's `DEPBAR` and landing loads (which wait anyway),
but a body still costs at least its 48 wavefronts x 8 = 384 of pipe plus
its trailing drain (~128) plus the vote / branch (~40) = **~550** (4.2),
and the design prices the arms at F26's 1414 for its centre.

### 1.3 The copy block price and the scale copies

`LDGSTS` wavefronts by PC (`--sts`): payload `.128 BYPASS` 4.00 = 4.00 ideal
(32 lanes x 16 B: 4 rows x 128 B = 4 global lines = 4 smem lines); fp4
payload `.64` 4.00 vs 2.00 smem-ideal (4 rows in 4 global lines, 64 B each);
octet `@!P2 LDGSTS.64` **11.93 vs 1.98 ideal** (24 lanes x 8 B into 192
contiguous smem bytes; sources: page b's row r for six pages and four rows -
with the `[pages, tokens, heads, 8]` layout and 8 heads the token stride is
64 B, so rows r, r+1 share a line and rows r+2, r+3 the next: **12 lines**).
The rule the three PCs obey: **an `LDGSTS` costs max(distinct 128 B global
lines, smem wavefronts)**.  The octet form and F25's leader-lane form both
touch every one of the CTA's 48 lines exactly once per operand (89-96
wavefronts per pair per CTA): the saving F26 booked was instructions (10 per
pair), not pipe time, exactly as the F26 record concluded.  The 1.99 M
"`LDGSTS` bank conflicts" are ncu's wavefronts-minus-ideal of these PCs
((11.93 - 1.98) x 7.5 per pair x 23936 = 1.78 M) plus the payload's 0: the
metric counts the line serialisation, not bank conflicts.

Copy-block price: a16 loop blocks 356 + 417 = 773 cycles for 96 wavefronts
per warp = **8.1 per wavefront**; F26 fp8 581 / 72 = **8.1**; fp4 787 / 72 =
10.9 (the 60 extra instructions and the two `LDS.128` WAR stalls: 787 - 103 -
~120 = 564 / 72 = 7.8).  C19 is restated as: **a copy block costs ~8 issue
cycles per wavefront per warp** - the LSU pipe's residual capacity under
the consumer's gmma reads; the `LDGSTS` count matters only through its
wavefronts.

### 1.4 The dynamic module as merged (F26a order; `f26ab_sf-1_mask1.sass`)

Region 5438 instructions (F26c 5519, F25 5528); `REG 168, STACK 0` (no
spills); loop site `[0x12fd0,0x1c480]` **2380** instructions with 4 `VOTE.ALL`
(F25 2391): `LDGSTS` 52 all predicated (per operand 6 slots x {a16 x 2, fp8,
fp4} + 2 scale) + `IMAD.WIDE.U32` 52 fused (`IADD3.X` / `IMAD.X` 0 - the
F26a address form carried over), `IMAD.MOV.U32` 106 + `VIADD` 85 + `SEL` 41 +
`ISETP` 32 + `SHF.R.U32.HI` 36 (base staging and predicate production for
the predicated copies, F25's "99 add / select" minus the carry adds), `FLO`
40 + `BREV` 40 (the mask-driven loops), `LDS` 86 + `LDS.64` 48, `BSSY` 15,
`BRA.DIV` 15, `NOP` 18 (the per-step `__syncwarp`s), `BRA` 63.  Loop bodies
(text): hot 248 / cold 284 per format loop per operand, two pages per step
(F25: 235 / 268).  Executed per pair is not measured (no ncu of the merged
dyn); bounded from the text: copy body ~200-225 issue slots (all predicated
instructions issue), format loops 4 sites x 1 step x ~200 = ~800 for 8
blocks on the 2/2/2 mix, prep 2 x ~90, protocol ~160 -> **~1200-1400
modelled, ~1000-1100 implied by the bench** (635 us = 2.89 us per pair = 5715
wall cycles; F25's 1211 instructions gave 650 us).  The count model of F27's
dynamic module (3.4) starts from these text counts.

## 2. What F27 changes and what it keeps

Kept: the 12-warp layout, 136 / 184 registers; A9 ownership (thread t =
block t % 8 of row t / 8 of all six pages; lane b < 6 of a row octet copies
page b's scale row); the E4M3 / E2M1 placement decodes, the folds, one vote
per operand (per format in the dynamic module), the two-arm static bodies
(210 / 267, 222 / 279) and the `STS.128` pattern; the a16 copy-address form
(`LEA` / `LEA.HI.X` once, `IMAD.WIDE.U32 Rd, R, UR, R.64` + `LDGSTS` per
copy); the 32-row chunk ring, the single pipeline counter, no pending words
in the static modules; `PipelineAsync` arrival counts 128 / 256; the peel,
K(last) alone, the drain; the a16 module's text (`STATIC_A16` arms, its SASS
= the F25 stream); the consumer, epilogue, scheduler, `sparse_mainloop.cuh`,
`mainloop_mma.cuh`, `named_barrier.cuh`; the chunk-table WAR argument
(`CHUNK_STORE_PAIR = 8 > NUM_STAGES`).

Changed (all in `sparse_mixed_mainloop.cuh` unless stated):

1. **Half-pair order** (3.1): per pair, K half then V half, each `wait_group
   1` -> landing loads -> `__syncwarp` -> scale loads -> copies (+ chains)
   -> `commit_group` -> vote -> body; two commit groups per pair; both
   chunk rows and page words read at the top of the pair before the spins.
2. **Top of the pair** (3.1, 3.5): `ok = __all_sync(okK && okV)` (one
   `VOTE`, `hasK` folded in), one uniform spin branch (both waits inside as
   voted `try_wait` loops), no `BSSY` / `BSYNC`, no `BRA.DIV`; the gather and the store + group barrier under one rare-taken
   test in the tail (after the arrives, under the fence drain); the `hasK`
   test kept (one not-taken uniform branch).
3. **No MIO-operand pins** (3.2, revision 2): the post-body scoreboard wait
   is the store burst's pipe time and no register form removes it at 136
   registers; the arms are priced at pipe time, the control-bit decoder
   (`f27_ctrl.py`, section 5) *reports* where the wait lands after each
   body, and the ncu segment table (6.3) is the arbiter of the arm price.
4. **Scale copies** (3.3): unchanged in the kernel unless the host layout
   changes; the two forms for the `[pages, heads, tokens, 8]` layout are
   specified and priced (3a in-warp `.128` row pairs, 3b two-warp ownership
   + scale mbarrier), with the host change as a separate, gated commit.
5. **Dynamic module** (3.4): class-sorted entries written by `chunk_store`
   (word 7 = `n8 | n4 << 4 | valid << 16 | flags << 24`, word 6 = the six
   entries' slot nibbles `slot_i << 4 i`, words 0-5 = the page indices in
   format order - full 32-bit pages, no page-index bound); predicated
   straight-line copies over the sorted entries (slot offset as a register);
   two rolled decode loops per operand with shared hot / cold bodies,
   software-pipelined loads, ballot-derived uniform counts, the slot of
   entry i from `(w6 >> 4 i) & 0xF` (a register shift, no register array,
   no `LDS`); `expand_format_pages`' `FLO` / `BREV` loops and the F26c
   per-slot bodies both gone; the loop pending state is the two rows
   (K' = this pair's V row, V' = the row at `oV - 32`), not pending words.
6. **fp4 copy block** (3.6): the spurious `VIADD ..., URZ` / `IMAD.MOV` /
   `UMOV` lowering named by a compile-only probe of `copy_compressed_page<FP4>`
   / `cp8` before the build; if it is the `cp8` asm's `"l"` operand formed
   from a `void const*` sum, the fix is the fp8 form (a `uint8_t const*`
   pointer add without the intermediate `void const*`).

## 3. The design

### 3.1 Data flow and control flow of one loop pair (static modules)

Pair t issues (K(t-1), V(t)) into (sK, sV) = (sV + 1, sV) mod 3 of the two
rings and finishes (K', V') = **(K(t), V(t+1))** - the pair the previous
iteration (t+1) issued - in stages (pK, pV) = (sV, sV - 1); K' is the tile
at this pair's V row (entry eV), V' the tile at entry eV - 1.  Two cp.async
commit groups per pair: A(t) = the K copies, B(t) = the V copies.  Loop
pairs are numbered as the code's `t` (the V tile).

```
[top of pair t]
rows:      LDS.128 x 2 (V row at meta_base + oV), LDS.128 x 2 (K row at +32 & 0x3FF), LDS.32 x 2 (octet page words)
                                  12 + 2 registers live to the copies; the LDS.128 latency (~30 + MIO queue) is hidden by
                                  everything below; no MIO address register of these is reused before the copies (3.2)
counter:   sK, phK, pK = sV, pV = sV - 1; the five stage bases (IMAD x 0x6000 / 0xc00, LOP3 swizzle)
ok = __all_sync(~0, (okK | !hasK) & okV)  one VOTE on the two test_wait results carried from pair t+1 (uniform by construction);
                                  hasK folded in so the last pair (V alone) does not enter the spin on a stale K probe
if (!ok) { if (hasK) spin(empty_k[sK], phK); spin(empty_v[sV], phV); }   ONE uniform branch, rare-taken (~4 % of pairs),
                                  out of line.  spin = do { r = mbarrier.try_wait.parity } while (!__all_sync(~0, r)):
                                  the loop exit is a VOTE result, so ptxas emits no BSSY / BSYNC and no BRA.DIV at the later
                                  __syncwarps (PipelineAsync::producer_acquire is a per-thread try_wait loop inside one asm
                                  block: it would keep both).  A phase already complete returns true at once (~40 cycles).
[K half]
cp.async.wait_group 1             DEPBAR.LE SB0, 0x1: A(t+1) (this thread's K copies of the previous pair) landed; B(t+1) may
                                  be in flight; cover of A(t+1) = K' body + mid-pair wait + V' loads + V copies + V' body +
                                  tail + top ~1550 cycles (a model output; 6.2's trace is the arbiter)
K' landing loads: 6 x 2 LDS.64    own landings of K' (fp4: 12 LDS.32); address registers l8a / l8b stay live to the body (no WAR)
__syncwarp                        (1) the six landed K' scale slots of the row are readable (A9), (2) every lane's K' loads precede
                                  every lane's K' stores (A7)
K' scale loads: 6 LDS.32          immediates sc_rd + pK * SCALE_STAGE + j * 8
K copies: 6 x {IMAD.WIDE.U32 ; LDGSTS.128 [land8 + sK * STAGE + j * PAGE]} + {IMAD.WIDE ; @(b<6) LDGSTS.64 [sc_cp + sK * SCALE_STAGE]}
          interleaved by ptxas with the K' scale chains (6 x PRMT, F2FP.E4M3, HADD2, FMUL, FSETP; PLOP3 x 5): 30 wavefronts per warp
          (36 after 3a: 24 + 6), ~240 cycles at 8 per wavefront, which hides the ~90-cycle chain to the vote
cp.async.commit_group             LDGDEPBAR: group A(t) - UNCONDITIONAL, outside `if (hasK)`: on the last pair (V alone) it
                                  is an empty group, so the V half's wait_group 1 still completes B(t+1) (PTX: "all prior
                                  groups complete"; an empty group is complete at once but keeps its position)
K' vote (VOTE.ALL) ; uniform BRA ; K' body: 6 x {F2FP.PACK ; 8 PRMT, 8 SHL, 8 LOP3 ; 8 HMUL2 ; 2 STS.128} into stage pK
                                  [the body's trailing store burst drains during the V half's DEPBAR / landing loads: the
                                   first register write after the body waits on the body's rd scoreboard - priced in 4.2]
[V half]
cp.async.wait_group 1             B(t+1) landed (A(t) may be in flight); cover of B(t+1) = V' body + tail + top + K' loads +
                                  K copies + K' body of pair t ~1550 cycles
V' landing loads ; __syncwarp ; V' scale loads
V copies: 6 + 1 into sV (V ring), chains interleaved
cp.async.commit_group             group B(t)
V' vote ; V' body into stage pV
[tail]
okK = test_wait(empty_k[sK + 1 mod 3], phase) ; okV = test_wait(empty_v[sK], phK)     non-blocking probes for pair t-1
fence.proxy.async ; arrive full_k[pK] ; arrive full_v[pV]
if ((jrow & ~0x100) == 0) { gather + countdown (jrow == 0) / store + BAR.SYNC (jrow == 0x100) }   ONE rare-taken uniform
                                  test (1 / 8 of pairs; META_J_MASK = 0x1E0 so the test selects j == 0 and j == 8 exactly), its
                                  predicate wait overlapped with the fence drain; out of line; before the advance of oV
sV = sK ; phV = phK ; oV = (oV + 32) & 0x3FF ; loop (the one taken branch of the common path)
```

Why this order:

- **Every latency is issued before a copy block and consumed after it, per
  operand** - the F26 principle, kept: the K' chain (~90 cycles to the vote)
  under the K copies (~240), the V' chain under the V copies; the landing
  loads ~300 cycles before their first use.
- **The LSU demand is spread**: per warp [rows 6 wf] [K' loads 23] [K copies
  30] [K' stores 48] [V' loads 23] [V copies 30] [V' stores 48] over ~1800
  cycles instead of [loads 60] [copies 72] [stores 96] over ~1400 with the
  copies as one saturating burst (0.89 of the pipe with the consumer's 0.39).
  The bodies' stores meet a queue of ~30 wavefronts (this half's copies), not
  72 + the other body's 48.  This is what buys the arms' IPC back (F25's
  0.53-0.55 was measured behind a slow copy block; F26's V body at 0.46
  behind a heavy one): budget 0.46-0.50 (section 4).
- **Both landing covers are ~1550 cycles by the segment model** (F26's
  measured-safe cover: `wait` 0.023-0.030 us, first post-`DEPBAR` PC 0.96 %);
  they are model outputs, longer in the pessimistic column, and 6.2's `wait`
  <= 0.05 us at *both* `DEPBAR`s is the hard stop.  Group accounting,
  verified against the loop as merged (`sparse_mixed_mainloop.cuh:2055-2226`):
  at the top of pair t the outstanding groups are {A(t+1), B(t+1)},
  `wait_group 1` completes A(t+1) (groups complete in order); mid-pair they
  are {B(t+1), A(t)}, `wait_group 1` completes B(t+1).  Two requirements the
  code must meet: (a) the peel (`pair_step(kv_tile_idx, Full, Partial)`,
  today one `cp_async_fence` at the end of `produce_pair`) commits K then V
  as two groups - in the compressed modules only (`if constexpr
  (HAS_COMPRESSED)`), so the a16 module's text is unchanged; K(last) alone
  then commits one group plus an empty one, finished by `wait_all`; (b) the K
  half's `commit_group` is unconditional (above).  K(last) alone
  (`finish_one(K)`: `wait_all` before `barrier_O.wait`, C7) and the drain
  (`finish_one_stage(V)`, `wait_all`) are unaffected; 1-tile items
  (`n_loop_pairs = 0`: the peel with tK = -1, A empty, B = the V copies, then
  the drain's `wait_all`) are fine.
- **The chunk rows at the top**: their `LDS.128` no longer follows the 24
  landing loads in the MIO queue (F26: 105 cycles of WAR at `0xaf30` + 29 of
  RAW at `0xb0f0`), and the K row's loads are no longer sunk into the `hasK`
  branch (F26 gate note: 6 fillers).  The rows' 14 registers are live
  across the spins (where liveness is at its lowest) and the K half.
- **One uniform spin branch**: `(okK | !hasK) & okV` per thread, voted with
  `__all_sync` (the F26 gate's "written down, not tuned" alternative).  When
  false the block re-waits both barriers with a warp-uniform `try_wait` loop
  (`mixed_detail::spin_uniform`: `do { r = try_wait(phase) } while
  (!__all_sync(~0, r))`; an already complete phase returns true at once, ~40
  cycles, on the rare path only; mixed okK / okV across lanes are safe
  because every lane runs both waits).  `PipelineAsync::producer_acquire` is
  not used here: its `wait` is a per-thread `try_wait` loop inside one asm
  block, which ptxas wraps in `BSSY` / `BSYNC` and follows with `BRA.DIV`
  reconvergence checks at the next `__syncwarp`s - the F26 3 `BRA.DIV`.
  The probes keep (sKn, phKn) / (sVn, phVn) exactly as the acquires did.
  `hasK` stays a not-taken uniform branch (~8 cycles); peeling the last pair
  to remove it would duplicate ~1200 instructions (do-not-build 10 stands).
- **The gather / store in the tail**: the two 1/16 tests become one 1/8 test
  `(jrow & ~0x100) == 0` whose taken path selects gather (+ the chunk
  countdown) or store + group barrier; the common path is one not-taken
  conditional branch whose predicate wait overlaps the fence's drain
  (`SYNCS.ARRIVE` long_sb 76 %).  The WAR argument of the chunk table is
  unchanged: the store of chunk c+1 (buffer (c+1) & 1 = (c-1) & 1) at the
  tail of pair 8 of chunk c trails the last read of that buffer (the V row
  of pair 15 of chunk c-1; the dynamic module's V' row read at pair 0 of
  chunk c) by 8-9 pairs > NUM_STAGES = 3, and moving the store from the top
  to the tail changes that distance by less than one pair.  The gather at
  the tail of pair 0 fills registers `cr` consumed by the store 8 pairs later
  (later-not-earlier; the ring argument is untouched).  The group barrier
  after the arrives is a named barrier of the 128 producer threads only.
  The two taken branches and their fetch bubbles (~85 cycles) leave the
  common path.

Hazards restated for the half-pair order (A7 / A9 / D5): the K' body's
stores go to K-ring stage pK = sV; the V copies of the same pair go to
V-ring stage sV (a different buffer); the V' landing loads read V-ring pV;
the K copies write K-ring sK.  Within a half, every lane's landing loads
precede the half's `__syncwarp`, every lane's stores follow it (A7); the
half's scale slots were landed by each lane's own `wait_group` before the
`__syncwarp` (A9).  The slot the K copies' octet overwrites (K-ring sK) was
last read two pairs ago; the acquire of sK (after all 128 arrivals of that
finish) orders those reads before these writes (F25's argument).  One
`fence.proxy.async` per pair after both bodies, then both arrives (D5): the
K' arrive waits for the V' body - the F26 position, kept (the consumer's K
wait is the pacing wait; moving the K arrive before the V half would need a
second fence, ~60 cycles of drain on the critical path, for a ~450-cycle
earlier K commit: a measured trade for a follow-up, not F27).

The trace stamps (`MIXED_FA3_TRACE`): `acq` = the spin block, `wait` =
the K half's `DEPBAR`, `fcK` = the V half's `DEPBAR` (the two waits are the
6.2 hard stop), `barB` = both halves' loads + barrier + scale loads, `iss` =
both halves' copies to their commits, `expK` / `expV` = vote + body, `fcV` =
probes + fence + arrives (the tail test follows the stamp).  The print format
is unchanged (`fcK` is re-purposed for the second wait); ratios only (F25
do-not-build 10).

### 3.2 The post-body scoreboard wait: pipe time, priced (revision 2; the pins of revision 1 withdrawn)

The mechanism (1.2, re-read): an `STS.128` holds its A and D registers
until the MIO reads them; all twelve stores of a body set the same read
scoreboard; the first instruction after the body that writes any register
the burst read waits until the burst has drained through the pipe at ~8
cycles per wavefront.  What the wait costs is the trailing burst's pipe
time - 16 wavefronts x 8 = 128 cycles behind the K body's last four stores,
~70 behind the V body's - and it lands on whichever instruction ptxas
schedules first with a register from the dead store operands, which at 136
registers is the next instruction that writes anything (the F26 victims:
the V' stage base `IMAD` at `0xd4e0`, the probe address `VIADD` at `0xf380`;
in F27 it will be the V half's `DEPBAR` successor or the first landing
`LDS.64` destination).

**Revision 1's pins are withdrawn**, for two independent reasons, both
verified before any build:

1. *They emit nothing.*  `asm volatile("" :: "r"(x)...)` lowers to PTX as
   `// begin inline asm` / `// end inline asm` with no instruction naming
   the operands (compile-only probe `nkcut2:/tmp/mixedkv-wtF27-art/
   f27_pin_probe.cu` -> `.ptx` lines 76-77, nvcc 13.0 sm_90a; the judge's
   `/tmp/mixedkv-wtF27-judge/pins.ptx` the same).  NVVM keeps the values
   alive to the asm and fixes source order relative to the other volatile
   asms; ptxas computes liveness from PTX instructions and sees neither: in
   the probe SASS it rewrote the `LDGSTS` address pair 3 instructions after
   the `LDGSTS` and the `STS` data registers 19 after the `STS` despite the
   pin.  Remote action 2 of revision 1 (the probe) was therefore run for
   this revision and refutes the mechanism.
2. *They would not have removed the wait if they had emitted a use.*  The
   scoreboard is a counter shared by the body's twelve stores, so any write
   to any of the ~50 registers the burst reads waits for the last store (R12
   at `0xf380` is block 1's output, outside any 18-register set).  Keeping
   18 live moves the first post-body write to a nineteenth register of the
   same set; keeping all ~50 live (+50 registers) is not available at 136,
   and the values that must be written next (24 landing destinations, the
   stage bases, the probe words) have to go somewhere.

**An instruction-emitting sink was priced and is not built either**: a
`LOP3` fold of the sixteen outputs and two bases into one word consumed by a
`st.shared` (~16 issue slots per half + one wavefront) would pin 18
registers at the PTX level, but by reason 2 the wait would move, not vanish;
and any sink that reads the store operands *after* the stores is itself
scheduled by ptxas relative to the stores it does not see as MIO-ordered.
Do-not-build 12.

**What the design does instead.**

- *Price it*: the arms are budgeted at pipe time - 48 wavefronts x 8 = 384
  of store pipe per body, + the trailing drain ~128 (relocatable), + the
  vote / branch / fetch ~40 = **~550 per arm as the floor**; the centre keeps
  F26's measured 1414 for the two arms (4.2: the half-pair order shortens the
  queue the bodies meet from 72 + 48 wavefronts to ~30, which raises the
  body IPC toward F25's 0.53, but the drain that F26 paid at `0xd4e0` /
  `0xf380` is paid at the V half's `DEPBAR` successor instead).
- *Put the drain where a wait already is*: in the half-pair order the K
  body is followed by the V half's `cp.async.wait_group 1` and its 24
  landing loads, the V body by the probes, the fence (`FENCE.VIEW.ASYNC`
  itself waits for the stores to perform, long_sb ~60) and the arrives.  A
  scoreboard wait that overlaps a `DEPBAR` or a fence drain costs nothing
  extra; one that lands on a landing load's destination delays that load by
  the drain.  Which happens is a ptxas outcome the design reports (gate 5,
  `f27_ctrl.py`) and 6.3 measures (the first post-body PC's short_sb share);
  neither is a stop.
- *The row hoist* (3.1) removes `0xaf30`'s 105 cycles by construction: the
  chunk rows' `LDS.128` no longer queue behind the 24 landing loads whose
  write scoreboard (SB1, merged with the read side) they waited on, and the
  landing bases `l8a` / `l8b` are live to the body (they are the `STS`'
  companions in `ExpandBases`), so no write of theirs follows the loads.
  Merged wr / rd scoreboards can recur anywhere: the control-bit decoder
  is part of the gate tooling so that the next reading names them.

Gate (section 5, revision 2): `f27_ctrl.py` decodes the control words of
the loop site (wait mask, rd / wr barrier index per instruction) and reports,
for each body, the rd scoreboard its stores set and the first instruction
after the last store that waits on it, with the distance in instructions;
the F26 numbers (7 and 107 instructions, 127 and 72 cycles) are the
baseline.  No distance threshold is gated - the wait is expected - but a
wait that lands on a *landing load* (the V' `LDS.64`s) is reported as the
case the pessimistic column prices.

### 3.3 Scale copies: the kernel is at the layout's floor; the two forms behind a host layout change

Today (`[pages, tokens, heads, 8]`, 8 heads): a page-head's rows are 8 B in
every 64 B, sixteen rows in eight lines; the CTA's six pages x 8 lines = 48
lines per operand are each touched once per pair (four warps x 12 lines):
89-96 wavefronts per pair per CTA, ~95 issue cycles per warp per pair at 8
per wavefront, "conflicts" 1.99 M (1.3).  No permutation of lanes, rows or
copy width changes the line count: the bytes a CTA needs are 1 / 8 of each
line it touches.  **Kernel side, the octet form stays as is** (it is the
fewest instructions for the floor).

**Host prerequisite (a separate commit, gated by the same 118 tests): the
scale tensors as `[pages, heads, tokens, 8]`**, i.e. `scale_stride.token = 8`,
`.head = 128`, `.page = 128 x heads` - the kernel takes the strides at run
time (`KVPageByteStrides`, `compressed_base`, `page_src`), so this is a
change to `kv_cache_fp8.py` (the scale allocation and the `reshape`s at
`:97`, `:167`, `:210-220`, the shape checks at `:270-280`, `:363`, `:397`),
`csrc/fp4_kv_quantization.cu` (the writer's index), the bench's
`make_transport` (`scale_shape`) and the test's transport builder; the XQA
decode hosts read the same cache and must be re-pointed the same way
(`docs/mixed_kv_page_transport_backends.md` lists them).  R1 freezes the
tag, not the scale layout; this is still a project-wide layout decision and
F27 does not make it - it states what the kernel gains if it is made:

- **3a, in-warp `.128` row pairs** (A9 kept, no barrier change): lane b < 6
  of every *even* row copies rows (r, r + 1) of page b = 16 contiguous,
  16 B-aligned bytes (token stride 8, head / page strides multiples of 16)
  with one `cp.async.cg ... 16` into the slot layout `slot(r, j) = (r >> 1) x
  96 + j x 16 + (r & 1) x 8` (`SCALE_ROW_STRIDE` -> a row-pair stride of 96;
  768 of 3072 B used); readers read page j's word at `(r >> 1) x 96 + j x 16 +
  (r & 1) x 8 + 4 (k >> 2)` - an immediate `j x 16` from a per-thread base;
  the warp's four rows hit bank words {0, 2, 24, 26} + 4 j + (k >> 2) mod 32:
  eight distinct banks, conflict-free.  12 active lanes, 6 pages = **6 lines
  = 6 wavefronts per instruction (from 12)**, 48 per pair per CTA, "conflicts"
  (6 - 2) x 7.5 x 23936 = **~0.7 M**.  Predicate: `(r & 1) == 0 && b < 6`;
  partial arm src-size `16 b + r + 1 < valid ? 16 : (16 b + r < valid ? 8 :
  0)` - a 16 B copy with src-size 8 zero-fills the odd row (the D4 form).
  Copy count unchanged (one instruction per operand).
- **3b, two-warp page ownership + scale mbarrier** (~0.2 M): lanes 8 p + r' of
  warps 0 and 1 (p = 0..2 -> pages 3 w + p, r' = 0..7 -> row pair 2 r') copy
  each page's 128 B line as eight 16 B copies: 24 active lanes, 3 lines = **3
  wavefronts per instruction, 6 per operand per CTA (12 per pair, from 89)**,
  wavefronts = ideal, "conflicts" ~0.1-0.2 M (the residual from other PCs).
  Visibility across warps needs a barrier the current design does not have:
  per ring a `scale_full[3]` mbarrier (`SharedStorage` + `kernel_traits.cuh`,
  48 B), init count 64; warps 0 / 1 issue `cp.async.mbarrier.arrive.noinc
  [scale_full[stage]]` after their scale `LDGSTS` (one instruction; it fires
  when that thread's prior cp.asyncs complete); every warp probes
  `scale_full[pending stage].test_wait(phase)` with the acquire probes (the
  phase is the stage's fill parity the producer already tracks) and includes
  it in the top-of-pair vote; the rare spin waits on it.  Warps 2 / 3 thus
  depend on warps 0 / 1's copies one pair old - the pace is set by the slowest
  warp already (the consumer's release needs all 128 arrivals), so the
  coupling costs nothing in steady state; it is a protocol change (C4:
  a third barrier class, +1 arrive / +1 probe per pair) and is **F27b, not
  F27**: the saving over 3a is 36 wavefronts per pair per CTA (~2 % of the
  pipe, ~36 issue cycles per warp).

Both forms keep A9's reader rule (a lane reads only slots its own row's
copies filled, after the `__syncwarp`; 3b adds the mbarrier for the copying
warp's lanes).  The kernel-side wavefront saving (3a: 41 per pair per CTA,
~2 % of 1899; ~50 issue cycles per warp) is real but small; it is not where
the F26 time went, and section 4 prices F27 without it.

### 3.4 Dynamic module: class-sorted entries with their slots in word 6, predicated copies, two rolled loops with shared bodies (revision 2)

Requirement (the F26c lesson): the per-format dispatch must add **no live
state on the copy path** beyond the six bases (12 registers: `a16_src0/1`,
`p8`, `p4`, `s8`, `s4`) the predicated F25d / F26a body already holds with 0
spills, and no branch tree over slots.

**Chunk-table row (dynamic module), revision 2 layout.**  Words 0-5: the
six page indices **in class order** (fp8 entries first, then fp4, then a16;
full 32-bit `IdType` values - no page-index bound, blocker 3 of section 11);
word 6: **the entries' slot nibbles**, `slot_i << 4 i` for i = 0..5 (bits
24-31 zero); word 7: `n8 | n4 << 4 | valid << 16 | flags << 24` (`n8`, `n4`
<= 6 fit a nibble each).  The masks `m8 | m4 << 8` of C10 are gone (no
reader remains: the copies and the decode consume counts and slots; the
tests check the output, not the table).

**Chunk store (once per 16 pairs; `chunk_store`, dynamic arm).**  Lane (row,
slot) has `(page, tag)`.  `m8`, `m4` by the existing `REDUX.OR`
(`__reduce_or_sync` over the row's octet; idle slots 6, 7 and A16 pages
contribute 0); `n8 = popc(m8)`, `n4 = popc(m4)`; `below = (1 << slot) - 1`;
`rank = is8 ? popc(m8 & below) : is4 ? n8 + popc(m4 & below) : n8 + n4 +
popc(~(m8 | m4) & 0x3F & below)` (two `SEL`s on 32-bit values); lanes with
`slot < 6` store `page` at `row + 4 rank` (`STS.32` with a register address -
a shared-memory store, not a register-array index) and contribute `slot << 4
rank` to a third `REDUX.OR`; lane 0 writes words 6 and 7 (one `STS.64` at
`row + 24`).  rank is a permutation of 0..5 over the six slots (the a16 class
absorbs out-of-range tail slots, tagged A16 by `chunk_load`), so no two
lanes collide and every entry i < 6 has a slot.  ~18 instructions per lane,
~1 per pair amortised.  The static modules' store is untouched (the a16 SASS
is the control).

**Rows read per loop pair (top of the pair, before the spins).**  K row
(entry eV + 1: this pair's K copies; 2 `LDS.128`), V row (entry eV: this
pair's V copies **and the pending K'**: 2 `LDS.128`), and **words 6-7 of the
V' row** (entry eV - 1, address `meta_base + ((oV - 32) & 0x3FF)`: one
`LDS.64` at `+ 24`), plus the two lane-octet page words.  The pending state
of the loop is thus the rows, not pending words: K' is pending iff `(mV.w7 &
0xFF) != 0`, V' iff `(w7' & 0xFF) != 0` (an all-a16 tile arrived by
`cpasync_barrier_arrive` when it was issued and must not be arrived again;
the static modules compile no such test).  Ring WAR of the V' row: it is
read at most 9 pairs before the chunk store overwrites its buffer (3.1).
The single-operand sites (K(last) alone, the drain) keep a pending record:
the F25 word `w7 & 0x03FFFFFF | stage << 30` plus the slots word `w6`
(`Operand::pending_slots` / `staged_slots`), set by `issue_operand` and
rotated as the word is.

**Copies (per operand, straight-line, predicated; the F26a body over sorted
entries).**  For entry i = 0..5 (unrolled): `page_i = pages[i]` (a register
from the row), `slot_i = (w6 >> 4 i) & 0xF` (a constant shift: `SHF` +
`LOP3`, or one `LOP3` + `SHF` for `off_i = slot_i << 11`), `is8 = i < n8`,
`is4 = !is8 && i < n8 + n4`, `isA = i >= n8 + n4` (compares on the row's
counts - value-uniform data; the copies are predicated, not branched, so the
ballot-derived counts are needed only by the decode loops); then the F26a predicated
copies with `off_i` added to the destination (`land8 + stage x STAGE +
off_i` - one `IADD3` per destination, three per entry), the a16 pair's two
rows likewise: `@is8 LDGSTS.128`, `@is4 LDGSTS.64`, `@isA LDGSTS.128` x 2,
one `IMAD.WIDE.U32` each (4).  Scale rows: lane k copies entry k's row:
`page_k = LDS.32 [row + 4 k]` (the existing `scale_page_word`), `slot_k =
(w6 >> 4 k) & 0xF` (a register shift by `4 k`), `dst = sc_cp - 8 k + 8
slot_k + stage x SCALE_STAGE`, `@(k < n8) LDGSTS.64 <- s8 + page_k x s8_ps`,
`@(n8 <= k < n8 + n4) <- s4 + page_k x s4_ps` (per-lane predicates on the
uniform counts; lanes 6, 7 have k >= 6 > n8 + n4 and are off).  Per operand:
6 x (2 slot + 3 address + 4 `IMAD.WIDE` + 4 predicated `LDGSTS` + 3
predicates) + scale ~10 ~= **~106 issue slots, of which 12 + 2 `LDGSTS`
issue wavefronts** (F26a ~110 by the text count; F25 246).  The
predicated-off slots cost issue slots inside a block that is LSU-paced at ~8
cycles per wavefront (24 + 16 (a16 rows) + 2 x 6 scale = ~52 wavefronts per
warp on the 2/2/2 mix, ~420 cycles), where ~150 issue slots are free: **the
predication is priced at 0-100 cycles, not at its count** (unverified until
the ncu executed count exists).  Live state: the six bases, `n8u`, `n4u`, the
row (8 words), stage bases: the F26a set.  Partial arms (peel V, K(last)):
src-size predicates `slot_i x 16 + r < valid` with `slot_i` a register (one
`SHF` + `IADD3` + `ISETP` per entry).

**Decode (per operand, after the half's copies): two rolled loops with
shared bodies.**  Before the vote: for i = 0..5 (unrolled) the sorted
entry's scale word `s_i = LDS.32 [sc + 8 slot_i]` (a register offset: `LEA`
per entry; the slot is `(w6 >> 4 i) & 0xF`, constant shift), `f_i = f32(s_i)`,
`v8_i = f_i x gs8`, `v4_i = f_i x gs4`, `ok8 &= (i >= n8u) | fold_ok(v8_i)`,
`ok4 &= (i < n8u | i >= n8u + n4u) | fold_ok(v4_i)` (bitwise; 12 `FMUL`, 12
`FSETP`, `PLOP3` trees; a16 entries' stale slots are read harmlessly - the
slot is always < 6, in the stage); `hot8 = __all_sync(ok8)`, `hot4 =
__all_sync(ok4)`.  Then:

```
if (hot8) loop<FP8, hot>(w6, 0, n8u)      else loop<FP8, exact>(w6, 0, n8u)         // uniform branch, once per format
if (hot4) loop<FP4, hot>(w6, n8u, n8u+n4u) else loop<FP4, exact>(w6, n8u, n8u+n4u)
```

`loop<FP8, EXACT>(w6, lo, hi)` is `#pragma unroll 1` over i in [lo, hi)
(uniform trip count; `if (lo == hi) return` uniform), **the entry's slot is
`(w6 >> (4 i)) & 0xF` - a register shift by a register amount (`SHF.R.U32`
with a register shift count) and a mask, then `<< 11` for the landing /
output offset and `<< 3` for the scale slot: 4 ALU per entry, no `LDS` of an
entry word, no register array (C2)** - blocker 2 of section 11 asked for the
form; revision 1's `entry_i` registers were indeed C2.  Software-pipelined
as F25's loops: iteration i issues entry i+1's packed loads (2 `LDS.64` /
2 `LDS.32`; the last iteration re-reads entry i, harmless), then entry i's
scale word (`LDS.32`, readable after the half's `__syncwarp`, A9), its `sf2`
(`PRMT`, `F2FP.E4M3`, `HADD2`, `FMUL`, `F2FP.PACK`), `__syncwarp`, entry i's
body (35 hot / 43 cold) and two store-address `IADD3`; the prologue issues
entry lo's loads before the first `__syncwarp`.  A7 per entry: every lane's
loads of entry i were issued in iteration i-1 (or the prologue) before that
iteration's `__syncwarp`; every lane's stores of entry i follow it.  Per
iteration: 4 (slot) + 3 `LDS` + 5 (scale) + 35 (body) + 2 + loop 3 (`VIADD`,
`ISETP`, `BRA`) = **~54 hot / ~62 cold per block** (revision 1: 52 / 60);
per operand on the 2/2/2 mix 4 blocks -> ~220 + 2 loop setups ~16; taken
branches per operand: 2 back-edges + 2 loop exits + the two hot / cold
selections = ~4-6 (~100 cycles); code per site per operand 2 x (54 + 62) +
setup ~40 = **~275** (F26a: 4 loops x ~250 + `FLO` selection = ~1100).

**Per pair on the 2/2/2 mix (modelled; the SASS gate reads the text, ncu the
executed count):** protocol ~42 (static ~35 + `n8u` / `n4u` ballots 4 + the
two pending tests) + rows 7 + copies 2 x 106 + landing loads in the loops 2
x 8 + scale words 2 x 6 + chains 2 x (12 + 12 + 6) + votes 4 + decode 2 x
236 + tail 16 = **~790 issue slots** (F26a ~1000-1100 by the text model -
the F26 ncu "mixed" row, 1321 per warp, is F26c's per-slot form, not the
merged F26a - F25 1211), of which ~150 predicated-off copy slots overlap the
LSU-paced block.  Region: loop site ~2 x (106 + 60 + 275) + top / tail
~130 = **~1010 instructions**; the two single-operand sites (exact bodies
only, no vote: one loop per format, ~190 each) + peel (~260) + prologue
(~450) -> **~2100 (34 KB) from 5438 (87 KB)**: `no_inst` (F25 / F26 10-11 %)
expected at the static modules' 2-4 %.  Registers: one block in flight
(packed 4 + next 4, outputs 8, sf2, offsets 4, loop 3) + bases 12 + counts 2
+ rows 8 + `w6'` / `w7'` 2 + protocol ~40 = **~92** (no pins).
Bit-exactness: every entry's bytes land at `slot x PAGE` / `slot x 8` (the
same addresses as the slot-ordered form), the hot / cold choice is per format
per operand over the same set of pages (order-independent AND), the cold
path is F25's term for term; A16 slots copy as before; stale slots (a16
pages' scale slots) are never decoded (the loops run over compressed entries
only).

Why not the other forms: F26c's per-slot branched bodies (spilled; 12 taken
branches per pair, 88 KB); a unified fp8 / fp4 body with runtime `PRMT`
selectors / shift / mask (+~26 `SEL` per block, +60 %, the same 6 slots x
one body per operand: the same instruction count as the two loops without
the loop bubbles but +50 % on every block - priced at ~+400 issue cycles per
pair, worse than ~100 of bubbles); a 6-slot rolled loop with a per-slot
format switch (12 taken branches per pair); the F25 mask-driven loops
(`FLO` / `BREV` / `SEL` per step, 80-125 per block); revision 1's `page |
slot << 28` entry word (needs `IdType` pages < 2^28, which no host check
enforces - `mixed_page_prefill.py` bounds `head_dim`, dtypes and the
transport shapes only - and reads the entry word back from smem in the
loop).  The "pages cannot be permuted" objection of F26 3.6 is void for a
*processing* order that carries the slot with the entry: the bytes land
where the slot says.

### 3.5 Protocol (C14) restated, per warp per pair

| item | F26 (SASS) | F27 | how |
|---|---|---|---|
| chunk rows + page words | 4 `LDS.128` + 2 `LDS.32` in two places (K's inside `hasK`), 6 fillers | same 6 `LDS`, at the top, one group; fillers 3 + 3 (two `LDGSTS` groups per pair now) | 3.1 |
| acquires | 2 x (`BSSY`, test, `@P BRA`, spin: `IMAD`, `SHF`, `TRYWAIT`, `@!P BRA`, `BSYNC`) ~14 + 2 `SEL` of the probes | `PLOP3 P` ((okK \| !hasK) & okV), `VOTE.ALL`, `@!P BRA` -> **3** on the common path; the spin block out of line, itself a `TRYWAIT` + `VOTE.ALL` + `@!P BRA` loop per barrier | uniform: no `BSSY` / `BSYNC`, no `BRA.DIV` guards at the votes and the `__syncwarp`s (their reason - per-thread spin branches - is gone) |
| pipeline counter, stage bases | ~20 | ~18 (`sK`, `phK`, `pV`: `VIADD`, `ISETP`, `SEL` x 3, `LOP3`; five `IMAD` bases + 4 `LOP3` swizzle) | unchanged in kind |
| `wait_group`, `commit_group`, `__syncwarp` | `LDGDEPBAR` x 2 (one is `wait_all`'s empty group) + `DEPBAR` + `NOP` + `BRA.DIV` = 5 | `DEPBAR.LE SB0, 0x1` x 2 + `LDGDEPBAR` x 2 + `NOP` x 2 = **6** (`cp_async_wait<1>` emits no extra `LDGDEPBAR`) | |
| gather / store tests | 2 tests + 2 taken branches + `BSSY` / `BSYNC` (~8) | 1 test + 1 not-taken branch (2), the rare block out of line | 3.1 |
| probes, fence, arrives | 2 `PHASECHK` + 4 address + 2 `SEL` + `MEMBAR` + `FENCE` + 2 `ARRIVE` = 12 | same 12 (3b: +1 `PHASECHK`, +1 `ARRIVE` on warps 0 / 1) | |
| loop | 3 | 3 | |
| **protocol total** | ~161 (F26 gate) | **~100** (the gather / store text is still counted once per 16: ~3) | the F26 budget "<= 70" was set for a protocol without the stage-base recomputation; F27 keeps the bases (the 3x unroll that would make them immediates is do-not-build 10) and budgets **<= 105** |

Barrier protocol (C4, C7, A5): `PipelineAsync` producer arrival 128 /
consumer 256 unchanged; per pair one acquire per ring (as a probe + rare
spin), two `wait_group 1`, two `commit_group`, two `__syncwarp`, one fence,
two arrives; K(last) finished before `barrier_O.wait`; `kQueryEmpty` 384;
chunk-table group barrier 128 every 16 pairs (in the tail).  3b adds
`scale_full[2][3]` (count 64; `cp.async.mbarrier.arrive.noinc` by warps 0 /
1; `test_wait` by all).

### 3.6 fp4: the copy block's spurious instructions

`sf2_mask0.sass` `[0xafb0,0xb8b0)` executes 137 instructions for the same 14
`LDGSTS` fp8 issues in 77: `UMOV UR5..UR8, URZ` (8), `VIADD Rx, Rx, UR5..8`
(12 - adding zero to one word of a 64-bit source), `IMAD.MOV.U32` of base
pairs (~10), `IADD3 R31, R31, UR5` / `IADD3 R29, R29, UR8`, `MOV`.  fp8's
block has none.  The two modules differ in `copy_compressed_page` only by
`cp8` vs `cp16` (`"l"(gmem)` of a `void const*` from `page_src`) and by
`land4`'s `+ 8 (k & 1)`.  The design does not guess: **a compile-only probe
(`f27_fp4_copy.cu`, `nvcc -arch=sm_90a -O3 -cubin`) of `copy_compressed_page<FP4,
true>` x 6 in the a16 context** (a `DTypeKV`-typed base built in a separate
function, carried across a `bar.sync`, six `page_src` + `cp8`) reproduces or
refutes the `VIADD ..., URZ` lowering and tries the two candidate forms
(`uint8_t const*` arithmetic without the `void const*` step; `cp8` taking
`uint8_t const*`).  If reproduced and fixed: -60 issue slots and ~-120
cycles in the fp4 copy block (787 -> ~600 before the WAR removal).  Not
counted in the fp4 centre of section 4 (counted in the band).

## 4. Budgets and prediction

### 4.1 Per-warp per-pair instruction budget (executed; fp8 / fp4 / dyn on the 2/2/2 mix)

| class | fp8 F26 (`loopsite_fp8.log`, 627 loop + 24 outside) | fp8 F27 | fp4 F27 | dyn F27 |
|---|---|---|---|---|
| top: rows 6, counter / bases ~18, ok vote 3, `hasK` 1, fillers 0 | 36 (+ 6 rows in the copy block) | **~30** | ~30 | ~33 (+ 1 `LDS.64` V' words, 2 pending tests) |
| per half: `DEPBAR`, 12 landing `LDS`, `NOP`, 6 scale `LDS.32` | 25 + 7 (both) | 2 x 20 = **40** | 40 | 2 x 8 (loads in the loops) + 2 x 6 |
| scale chains + `PLOP3` + `VOTE` + `BRA` | 68 | 68 | 68 | ~108 (+ 2 x (2 ballots + 2 popc)) |
| copies (7 `IMAD.WIDE` + 7 `LDGSTS` + 3 fillers + `LDGDEPBAR` per half) | 28 + 6 + 1 | 2 x 18 = **36** | 36 (+ 0 after 3.6; today + 60) | 2 x 106 = 212 (12 + 2 `LDGSTS` issue wavefronts) |
| bodies | 420 (2 x 210) | **420** | 444 | 8 x ~54 + 2 x 16 = ~465 |
| tail: probes, fence, arrives, chunk test, loop | 14 + ~10 | **~16** | 16 | 16 |
| gather / store (amortised) | ~3 | ~3 | ~3 | ~4 |
| **total** | **651** (ncu 2603 / 4) | **~655** | ~680 (~740 with the fp4 excess) | **~790** |

The counts are flat: F27 is a stall design.  Pipe mix at parity (per SMSP
per 2544 cycles, fp8): unchanged from F26's 3.7 (ALU ~42 %, FMA ~35 %, XU
~55 %, issue ~57 %); nothing saturates but the shared-memory data pipe
(4.3).

### 4.2 Per-warp per-pair cycle budget by segment (issue cycles; fp8; revision 2 columns)

Method: F26's measured segments (section 1) with each mechanism's removal
argued from its PCs; IPC per segment argued from measured neighbours.  Four
columns: **floor** = every segment at its pipe-time price (a model output:
it assumes the ALU issue of the bodies overlaps the pipe completely and is
*not* a prediction); **centre** = the top, the preps and the copies at their
F27 prices and the arms at F26's measured 1414 (the half-pair order shortens
the queue the bodies meet but the burst drain lands in them; nothing in 3.2
removes it); **derivable-only** = F26's 2667 minus the top's branch structure,
the one segment whose removal is argued PC by PC (1.1) and needs no
scheduling outcome; **pessimistic** = the top partly moved, the preps at
F26's price, copies at fp4's price, the arms 6 % worse than F26 (the
mid-pair `DEPBAR` exposing latency on top of the drain).

| segment | F26 measured | floor | **centre** | derivable-only | pessimistic | argument |
|---|---|---|---|---|---|---|
| top: rows, counter, ok vote, spin test, `hasK` (~30 instr) | 367 | 165 | **165** = slack 72 + fetch 24 (the back-edge) + waits ~35 + issue ~34 | 165 | 220 | 1.1: three taken branches (65) and their resolution / predicate waits (~90) and the `BSSY` / `BSYNC` (21) leave the common path; the slack stays; the rows' `LDS.128` issue 6 slots here |
| K half prep: `DEPBAR`, 12 `LDS.64`, `NOP`, 6 `LDS.32` (20) | 169 + 17 for both = 186 | 70 | **70** | 93 (half of F26's) | 100 | 12 + 6 MIO instructions at ~3-4 cycles of MIO dispatch each (F26: 24 loads in 169 = 7 each behind the `wait_all` group and the row WAR); the `DEPBAR` itself ~0 (first post-`DEPBAR` PC 0.96 %); the K' burst drain may land here (3.2) - it overlaps the `DEPBAR` |
| K copies + chains (18 + 36) | 581 for both | 240 | **240** = 30 wavefronts x 8 | 290 (half of F26's) | 290 | 1.3: the LSU residual price; the chains fill the issue gaps (F26: `PRMT` / `F2FP` interleaved at dispatch 40-95 %) |
| K vote + arm (213) | 810 | 550 = 384 pipe + ~128 drain + ~40 vote / branch / fetch | **707** (F26's 1414 / 2) | 810 | 750 | 3.2: the body's 48 wavefronts x 8 are the floor; the drain is relocatable, not removable; the centre does not claim the body IPC moves (F26 0.34 / 0.46, F25 0.53-0.55 behind a slow copy block) |
| V half prep (20) | (in the 186) | 70 | **70** | 93 | 100 | as K |
| V copies + chains | (in the 581) | 240 | **240** | 290 | 290 | as K |
| V vote + arm | 604 | 550 | **707** | 604 | 750 | as K |
| tail: probes, fence, arrives, chunk test, loop (16) | 118 | 100 | **100** | 118 | 120 | the fence drain (long_sb ~60) stays; the chunk test's predicate wait overlaps it |
| **pair (issue)** | **2667** (IPC 0.24) | **~1985** (IPC 0.33) | **~2300 (IPC 0.28)** | **~2465** | **~2620** | |
| **pair (wall, x 1.148)** | 3060 (paces) | ~2280 (< 2544: not claimed) | **~2640 -> 293 us** | **~2830 -> 315 us** | **~3010 -> 334 us** | accept iff issue <= ~2590 (330 us) |

Reading: the accept holds at the centre and at the derivable-only column;
it fails at the pessimistic end by 4 us.  The consumer-bound regime (issue
<= 2216) is reached by no column the design stands behind (the floor is a
sum of prices, not a derivation of overlap; 4.3 puts the pipe-time floor of
the producer's own wavefronts at ~1860 with perfect overlap).  **The reject
cases are a segment getting worse than F26**: a cover failure at the
mid-pair `DEPBAR` (6.2's second wait), a spill (ptxas -v), the arms above
1500 - the first two are gated before the bench, the third is what 6.3's
segment table names.

fp4: + 24 landing `LDS.32` issue slots (~+30 cycles), bodies 222 (+~25 each),
the `BRA.DIV` long_sb (32) gone with the uniform spin, the row hoist takes
`0xafb0`'s 103 off the copy block, the copy block's 60 extra instructions
kept in the centre (3.6 unresolved: +~120 cycles at the copy IPC) -> floor
~2025, **centre ~2530 -> wall 2900 -> 322 us** (~2410 -> 307 us if 3.6's fix
lands), derivable-only ~2700 -> 344 us (**reject**: the top alone does not
carry fp4), pessimistic ~2900 -> 358 us.  fp4's accept therefore needs the
copy block or the arms to move as well as the top.

mixed (dynamic, 3.4): top 165 + 30 (ballots, V' words) ; two halves: prep 2
x 60 (loads in the loops), copies 2 x 52 wavefronts x 8 = **830** (the a16
slots' two 16 B row copies add 16 wavefronts per operand on the 2/2/2 mix;
the ~150 predicated-off slots overlap), chains 2 x 30, votes 4 x 20, decode 8
x 54 / 0.45 = **960** + loop bubbles ~6 taken branches x 20 = 120 + two body
drains 2 x ~64 (8 blocks' bursts are half a static body's) = 128, tail 100 ->
**~2650 issue (IPC 0.30) -> wall 3040 -> ~338 us**; pessimistic (bubbles
250, decode IPC 0.38, copies 1000, the a16 rows' copies at 10 per wavefront)
~3300 -> wall 3790 -> **420 us**.  Above 330 at the model's own centre: **not
claimed**; the ptxas / SASS gates decide whether it is built for timing at
all, and the bench reads the loop price.

### 4.3 Shared-memory wavefront budget (C11), per pair per SM, fp8

gmma 1001 (unchanged); `STS.128` 96 x 4.00 = 384; landing `LDS.64` 96 x 1.92
= 184; scale `LDS.32` 48; rows `LDS.128` 16 x 2 + words 8 = 40; payload
`LDGSTS` 48 x 4 = 192; octet 8 x 11.9 = **95** (3a: 48; 3b: 12).  **Per
warp, as issue time at C19's price (revision 2, blocker 6):** (96 `STS.128`
+ 72 `LDGSTS`) wavefronts x 8 + (46 landing + 12 scale + 10 row `LDS`) x ~3.7
= 1344 + ~250 = **~1590 pipe cycles**, plus the top (~165: the pipe is idle
after the fence) and the fence drain (~100) = **~1860 per pair with perfect
ALU overlap** - the floor the segment sums of 4.2 must not go below (the
1985 floor column is above it; revision 1's 1805 centre was below it and is
withdrawn); `op_ld` and
`op_st` unattributed residual ~170 + ~130 (F25 / F26, unchanged) -> **~2190
measured-equivalent, of which the attributable producer set is ~940; in F26's
metric convention (`op_ld` 388 + `op_st` 696 + gmma 1001) 1899 -> F27 1899
(same bytes), 1852 (3a), 1816 (3b)** = 73-75 % of a 2544-cycle pair.  The
pipe is the binding shared resource: at ~1.0-1.2 wavefronts per cycle peak
and the consumer's 0.39 per cycle, the producer's ~900 per pair need ~1500
cycles of residual capacity - fits a 2544 pair only if spread (3.1), which
is the design's central claim and the segment IPCs' basis.

### 4.4 Dependency depth (critical chains per half)

- To the K' `VOTE`: `DEPBAR` (~0-20) -> 12 `LDS.64` issue (~40) -> `NOP` -> 6
  `LDS.32` issue (~20) + latency (~30) -> `PRMT`, `F2FP`, `HADD2`, `FMUL`,
  `FSETP` (~25) -> `PLOP3` x 5 (~15) -> `VOTE` (~10) -> `BRA`: ~160 from the
  `DEPBAR`, of which the copy block (~240) covers everything after the
  scale loads: exposed **~0**; `VOTE` -> `BRA` ~10 remains.
- To the first K' `STS`: `VOTE` -> `F2FP.PACK` (~8) -> `HMUL2` (4) -> `STS`;
  block 0's decode (~12) runs in the shadow of the vote: **~15**.
- Copy block: rows (loaded at the top, ~600 cycles earlier) -> `IMAD.WIDE`
  (~6) -> `LDGSTS`: ready; the block's length is the LSU's service of 30
  wavefronts (~240).
- Landing: `wait_group 1` with ~1550 cycles of cover for both groups (3.1);
  exposed 0 at F26's measured latency (`wait` 0.023-0.030 us at 1600);
  gated (6.2).
- Acquire: probes issued in the previous tail (~120-cycle round trip
  overlapped with the fence drain and the top's loads); the vote at the
  top reads the two predicates (~6); the spin only when not ready.
- Tail: `FENCE.VIEW.ASYNC` waits for the V' body's 12 `STS.128` to perform
  (~60, long_sb): kept; the probes' issue and the chunk test overlap it.

### 4.5 Registers (C3)

Producer 136.  Widest point (during the K copies of pair t), counted as F26
3.10 did: item / protocol constants ~44 (bases 2 x 10 with strides in UR;
`sV`, `phV`, `oV`, `chunks_left`, `t`, `okK`, `okV`, scheduler ~3; per-pair
`sK`, `phK`, `pK`, `pV`, `hasK`, stage bases 5), rows 12 + words 2, K' `pk`
24 + `sw` 6 + `v` 6 + `e` 7, copy sources in flight ~8 = **~109** (revision
1: ~127 with 18 pins; F26: ~135 at its widest, 0 spills); during the K'
body: K' packed (dying) + outputs 16-32 + `e` 7 + constants 44 + rows (V) 8
= ~97; during the V copies: V' 43 + constants 44 + sources 8 = ~95.  The
half-pair order genuinely lowers the widest live set (one operand's 24
packed words across the copies instead of two), so F27b's register risk is
lower than F26's.  ptxas -v gate: 0 / 0 / 0, no C7507; no fallback is needed
or named (the F26 3.10 options stay available as F26 states them).  Dynamic
module ~92 (3.4).  Consumer 184, untouched.

## 5. SASS gates before any timing (F27a-d builds; `cuobjdump -sass` of the mask_1 objects and the mask_0 objects the bench runs; producer region `USETMAXREG.DEALLOC .. EXIT`; loop site = the back-edge body with the two / four loop `VOTE.ALL`s; `f25_counts.py`, `f25_loopsite.py`, `f26_seg.py`, plus `f27_war.py` = the operand-hold distance check)

| gate | accept | reject -> action |
|---|---|---|
| `ptxas -v` | static: 0 B frame / 0 / 0, no C7507, `USETMAXREG 0x88 / 0xB8`; dynamic: **0 / 0 / 0** (F26c 128 / 408 / 608) | static: 4.5's fallback, one step; dynamic: stop (the design's register claim is wrong); no ad hoc tuning |
| order (3.1) | loop site in program order: 6 row `LDS` -> counter -> `VOTE.ALL` (ok) -> `@!P BRA` (spin, out of line) -> `DEPBAR.LE SB0, 0x1` -> 12 `LDS.64` -> `NOP` -> 6 `LDS.32` -> 7 `IMAD.WIDE` / 7 `LDGSTS` (chains interleaved) -> `LDGDEPBAR` -> `VOTE.ALL` -> K arm -> `DEPBAR.LE SB0, 0x1` -> 12 `LDS.64` -> `NOP` -> 6 `LDS.32` -> copies -> `LDGDEPBAR` -> `VOTE.ALL` -> V arm -> 2 `SYNCS.PHASECHK` -> `MEMBAR` + `FENCE` -> 2 `SYNCS.ARRIVE` -> chunk test `@P BRA` (out of line) -> loop | a `DEPBAR.LE SB0, 0x0` in the loop; a `DEPBAR` before the spin branch; the V' loads before the K arm; any `LDS` of a half after its first `LDGSTS`; `LDGDEPBAR` != 2 per pair |
| branches (1.1) | on the common path of the loop site exactly: 1 taken (`BRA` back-edge) + 3 not-taken conditionals (spin, `hasK`, chunk test) + 2 vote branches; **`BSSY` / `BSYNC` 0 in the loop site outside the chunk-store block** (the spin block is a `TRYWAIT` / `VOTE.ALL` / `@!P BRA` loop - uniform, no `BSSY`; if `BSSY` appears there the spin was not written as the voted loop); **`BRA.DIV` 0** (F26: 3); hot arm as fall-through reported, not gated | `BSSY` > 0 at the spins -> the ok vote or the spin loop did not reach ptxas as uniform (one `VOTE` per exit predicate); `BRA.DIV` > 0 -> a per-thread branch precedes a `__syncwarp` |
| **post-body scoreboard wait (3.2, revision 2; report, not a stop)** | `f27_ctrl.py` (`benchmarks/mixed_kv_f27_probes/`) decodes the control words of the loop site (stall, yield, wr / rd barrier index, wait mask); per body it reports the rd scoreboard set by its `STS.128`s (all twelve on one index, as F26), the first later instruction whose wait mask names it and the distance in instructions; the same for the copy blocks' `LDGSTS` and the landing `LDS.64`s' write barriers | nothing rejects: the wait is expected (F26 baseline 7 / 107 instructions, 127 / 72 cycles); a wait landing on a V' landing load is the pessimistic column's case and is named in the report; the arbiter of the arm price is 6.3's segment table |
| copy blocks | per half 7 `IMAD.WIDE.U32 Rd, R, UR, R.64` + 7 `LDGSTS` (6 `.128` / `.64` payload + 1 `@P .64` scale), `IADD3.X` / `IMAD.X` 0, `SEL` 0, fillers 3 per half; `LDGDEPBAR` exactly 2 per pair (the K half's unconditional); **fp4: `VIADD ..., UR` with a `UMOV UR, URZ` source 0, `IMAD.MOV` of base pairs <= 2 per half** (3.6) | fp4 spurious adds > 0 after the probe's fix -> report the form and use the band's centre with the excess kept (322 us) |
| protocol (3.5) | loop-site count - 24 - 12 - 68 - 420 - 36 - 6 rows <= **105**; `SHF.R.S32.HI` / `LEA.HI` division 0 outside the chunk block | over -> list the opcodes |
| bodies (C12) | hot 210 / cold 267 (fp8), 222 / 279 (fp4) unchanged; `VOTE.ALL` 3 in the loop site (ok + 2 votes), 0 in bodies; `UMOV` 0 in bodies | any body change is a mistake |
| region | fp8 <= 2500, fp4 <= 2550 (F26 2543 / 2582: the out-of-line spin and chunk blocks add ~20); **dyn <= 2300** (from 5438) | dyn > 2600 -> the loops were unrolled or the bodies duplicated per site: read the loop site |
| dyn (3.4) | loop site <= 1050 with 4 `VOTE.ALL` + 4 `VOTE.ANY` + 4 `POPC`; per operand 4 rolled loop bodies (hot8, cold8, hot4, cold4) each with one back-edge, `FLO` / `BREV` 0, `NOP` 1 per body (the per-step barrier), **`LDL` / `STL` 0 anywhere in the region**, the slot of a loop entry as `SHF.R.U32 R, R, R` (register shift count) + `LOP3 ... 0xF` + `SHF.L` / `LEA`; copy body per operand 26 `LDGSTS` predicated + 26 `IMAD.WIDE` + 6 slot offsets, `SEL` on an address 0; `chunk_store` (dyn) with 3 `REDUX` + 2 `POPC` + one `STS` with a register address (the ranked page) + one `STS.64` (words 6-7); one `LDS.64` at the top (the V' words) and two `ISETP` pending tests; `BSSY` / `BSYNC` 0 in the loop site outside the chunk block | `BSSY` around a decode loop -> the trip count reached ptxas non-uniform (the ballot is missing); `LDL` / `STL` > 0 -> stop |
| a16 module and stock | a16 SASS identical to F25's / F26's stream (mask_1 3832 / 3880); stock byte-identical | any change: the `STATIC_A16` arms or shared text were touched |
| host layout commit (3.3, if made) | strides in the bench's `make_transport` and the quantizer: token 8, head 128; the kernel object unchanged (byte-identical for the layout commit alone); 3a's build: octet `LDGSTS` -> one `@P LDGSTS.E.BYPASS.LTC128B.128` per half, readers' immediates `j x 16` | |

Nothing in SASS gates the landing cover: **the trace (6.2) is a hard stop
before any bench**, as in F26.  Gate tooling (`benchmarks/mixed_kv_f27_probes/`):
`f27_ctrl.py` (control-word decoder; wait masks and barrier indices per
instruction, the per-body report above) and `f27_fp4_copy.cu` (the 3.6
compile-only probe); the F25 / F26 scripts (`f25_counts.py`,
`f25_loopsite.py`, `f26_seg.py`) stay on nkcut2 as before.

## 6. Verification (after the gates; confirmation, not tuning)

- **6.1 tests**: `tests/attention/run_fa3_mixed_page_transport.py`, 118 / 118
  bit-exact on each of F27a, F27b, F27c, F27d (the 33-tile ring wrap, the
  22-tile buffer 1, the dynamic-uniform extremes are in the matrix since
  F26).  **Six cases added for the dynamic module's sorted entries**
  (`_run_dynamic_pattern`, three per-tile patterns x q = 1 / 64, 18 pages =
  3 tiles per item with a partial tail page): every tile (fp4, a16, fp8, fp8,
  a16, fp4); every tile (a16, a16, fp4, fp8, a16, a16) - ranks that are
  permutations the 2/2/2 cycle (fp8, fp4, a16 repeating) does not exercise
  (a16 entries before compressed ones, both classes split across the tile);
  and tiles (mixed, all-a16, mixed) so that a pair has one pending and one
  not-pending operand in both orders (K' pending / V' not, V' pending / K'
  not) and K(last) alone finishes a tile whose predecessor is all-a16.  The
  reference is the a16 module on the host-decoded pages (the extremes
  helper's decode with g = 1).  Run first on the merged tree (expected pass:
  they are layout-independent), then on F27c.  124 cases.
- **6.2 trace** (`MIXED_FA3_TRACE`, q=1, CTA 0 items 0 / 1, ratios): `wait` +
  `fcK` (the K half's and the V half's `DEPBAR`) <= 0.05 us per pair (hard
  stop; the two covers of 3.1) - reject
  > 0.1 -> the covers are shorter than modelled: read `issK` / `issV` and
  `expK` / `expV` (a body that ran long is the cover's friend, not its
  enemy: a long `wait` with short bodies means the loaded latency is above
  1500 cycles; then the fallback is **F27' = F27a alone** (F26's order with
  the top / tail changes; the F27b commit reverted), priced as the
  derivable-only column: ~2465 issue -> **~315 us**); fp8 `iss` <= 0.30 us
  (2 x 240 cycles + loads); `expK` == `expV` within 0.03 (fp8 and fp4);
  `acq` >= 0.03 (the slack); mixed `iss` <= 0.5.
- **6.3 ncu** (fp8 / fp4 / mixed / a16, q=1, third launch, `f26_run_ncu.sh`
  + `f26_ncu_read2.py` with the F27 segment markers: top = loop start to the
  first `DEPBAR`, K half = to the second `DEPBAR`, V half = to the first
  `PHASECHK`, tail): loop-site issue cycles per pair <= 2590 (accept; the
  centre 2300, the derivable-only 2465); per segment vs 4.2 - **the arm
  segments are the arbiter of 3.2's price**: <= 1414 for the two = the
  centre held, 1100-1200 = the half-pair order bought body IPC, > 1500 =
  the pessimistic case (name the post-body PC that carries the drain and
  its wait mask); **the first write after each body's last `STS.128`
  reported with its short_sb share** (F26: 4.78 % + 2.74 %; the design
  expects it to move to the V half's `DEPBAR` successor, not to vanish), the
  first `LDS.128` of the top <= 1 % of loop samples (the `0xaf30` removal
  is claimed); short_sb of the loop site reported (F26 12.3); dispatch <=
  15 % (20.2); no_inst <= 2 % (3.6; dyn <= 4 % from 11); `branch_resolving`
  <= 3 %; the spin branches' share reported as the slack; consumer K-wait
  read by F26 6.3's rule (K-wait 3-8 % with the pair < T_c = the contention
  term c; K-wait > 8 % or <= 3 % with the wall > 300 = the producer paces:
  the segment table names the segment; consumer-bound is not claimed);
  tensor pipe within 5 % of a16's 67 %; `l1tex` shared wavefronts per pair
  1850-1950 by class (`op_ld` ~390, `op_st` ~700, gmma 1001; the octet PCs
  at 11.9 unless the layout commit landed: then 6 (3a) / 3 (3b) and
  "conflicts" ~0.7 M / ~0.2 M); dyn: producer `inst_executed` per pair
  3000 +- 10 % (750 per warp), the loops' back-edge `BRA` PCs' share (the
  bubble price), `no_inst`.
- **6.4 bench** (`bench_fa3_mixed_page_transport.py --q-lens 1 64 --repeats
  1 --trials 5`, nkcut2 lock, co-tenant rule): the section 0 rows, min /
  median / max; transport_a16 must reproduce 281-285 / 287-292 or the
  session is offset.  Accept / reject: fp8 <= 300 / <= 307 = the centre
  (the arms moved too); 301-320 = the derivable-only column (accept on the
  target; 6.3 names the arm price); 321-330 = accept, marginal; > 330 reject
  (6.3: the arms above 1500 or a cover failure).  fp4 <= 315 / 322 = the
  copy-block fix or the arms moved; 316-330 accept; > 330 = as the model's
  derivable-only column says (the top alone does not carry fp4): report,
  do not tune.  mixed <= 330 accept (better than the model); 331-420 = the
  model's own band (report the dyn segment table; build nothing new); > 420
  = the count model wrong again (read the dyn SASS executed counts before
  anything else).

## 7. Invariant amendments (dataflow document, when F27 lands)

- **A10 (pair order), replaced.**  Per loop pair: chunk rows and page words;
  counter; one uniform vote on the two probes (`hasK` folded in) and one
  rare spin block (a warp-uniform `try_wait` loop per barrier); K half =
  `wait_group 1`, K' landing loads, `__syncwarp`, K' scale loads, K copies
  (+ K' chains), unconditional `commit_group`, K' vote + body; V half
  likewise into the V ring with the V' operand; probes; fence; arrives; the
  chunk gather / store under one test in the tail.  Two commit groups per
  pair (the peel commits two in the compressed modules); both covers ~1550
  cycles by the model, 6.2 the arbiter.
- **C14 (protocol), restated**: <= 105 per pair per warp, itemised as 3.5;
  one uniform spin branch, no `BSSY` / `BSYNC` / `BRA.DIV` in the loop site
  outside the chunk block.
- **C19 (copy-block price), restated**: ~8 issue cycles per wavefront per
  warp = the LSU pipe's residual capacity under the consumer's gmma reads;
  an `LDGSTS` costs max(distinct 128 B global lines, smem wavefronts); the
  lever on a copy block is its wavefronts, and the lever on the bodies
  behind it is the queue it leaves (spread the copies).
- **C20 (new, post-body scoreboard drain)**: a body's `STS.128` burst sets
  one read scoreboard; the first later write of any register the burst read
  waits for the burst to drain through the MIO pipe (~8 cycles per trailing
  wavefront: ~128 behind a six-block body).  It is pipe time - relocatable
  by placing a `DEPBAR` / fence / `__syncwarp` there, not removable by
  register allocation at 136 registers (an empty `asm volatile` use emits
  nothing ptxas sees; a real sink moves the writer, not the wait).  Priced
  in every arm budget; reported by the control-word decoder (gate 5);
  measured by 6.3's first-post-body PC.
- **C11**: bound unchanged (<= ~85 %); F27 expects 1850-1950 per pair (73-77
  %); the octet PCs at the layout's floor (11.9) until the host layout
  changes.
- **C17 (dynamic copies) / C10 (dynamic decode), restated**: the dynamic
  chunk row is words 0-5 = the page indices in class order (fp8, fp4, a16;
  full 32-bit), word 6 = the entries' slot nibbles (`slot_i << 4 i`), word 7
  = `n8 | n4 << 4 | valid << 16 | flags << 24` (the C10 masks are gone);
  copies are predicated straight-line over the entries with the slot offset
  as a register; the decode is two rolled loops per operand with shared hot
  / cold bodies, uniform trip counts by ballot, the entry's slot by a
  register shift of word 6 (no register array, C2; no `LDS` of an entry
  word), software-pipelined loads, one `__syncwarp` per step; no `FLO` /
  `BREV`, no per-slot branch; the loop's pending state is the K' row (= the
  pair's V row) and words 6-7 of the V' row (`oV - 32`), pending iff `w7 &
  0xFF != 0`; the single-operand sites carry the pending word plus the
  slots word.  "Pages cannot be permuted" is withdrawn for processing order:
  the slot travels with the entry.
- **C21 (new, scale layout, conditional)**: if the cache's scale tensors
  become `[pages, heads, tokens, 8]`, the scale copy is one `.128` row-pair
  copy per operand into `(r >> 1) x 96 + j x 16 + (r & 1) x 8` slots (3a);
  the two-warp form (3b) adds `scale_full[2][3]` mbarriers.

## 8. Do not build

1. Any change to the decode bodies, the `STS.128` pattern or the fold /
   vote (F26 do-not-build 1, 6): they are measured ideal and unchanged.
2. **An `LDS.128` landing load** (one per page instead of two `LDS.64`): the
   [24b] store swap is per lane (`swap = ((b >> 2) ^ r) & 1`), so the
   physical half order would need per-lane `SEL`s on the data (+4 per
   block) or giving up the swap (2-way `STS.128` conflicts, +96 wavefronts
   per pair).  Wavefront-neutral as it is; F25 had it right.
3. A blocking `try_wait` before the commits; a `DEPBAR.LE 0` mid-pair
   (F26 13; the covers).
4. Moving the K' arrive before the V half without measuring the second
   fence's drain (3.1): a follow-up trade, not F27.
5. Peeling the last pair to remove `hasK`; a 3x loop unroll for immediate
   stage bases (F26 10: code size).
6. A second producer warp group, role-split producers, consumer-side
   decode, SWAP_AB, fp8 wgmma (F25 2A-2E, 7): the F27 reading puts the pair
   under T_c with the present warps.
7. A kernel-side scale copy form under the present `[pages, tokens, heads,
   8]` layout: it is at the line floor (3.3).
8. The F26c per-slot dynamic bodies; a unified fp8 / fp4 body with runtime
   selectors; a 6-slot loop with a per-slot format switch (3.4).
9. Trace-segment absolutes as inputs; any GPU timing before section 5 and
   6.2 pass.
10. Register-pressure games (4.5 names no fallback; F26 3.10's options stay
    as F26 states them).
12. **Register pins of any form** against the post-body scoreboard wait
    (3.2): an empty `asm volatile` use emits nothing; an instruction-emitting
    sink moves the writer to another register of the same scoreboard set;
    the wait is the store burst's pipe time.  Price it, place a wait under
    it, do not "fix" it.
13. `PipelineAsync::producer_acquire` in the loop's spin block (a per-thread
    `try_wait` loop inside one asm block: `BSSY` / `BSYNC` + `BRA.DIV`);
    the spin is the voted `try_wait` loop of 3.1.
11. **Building the host layout change and the kernel's 3a / 3b in the same
    commit as F27a-c**: the layout commit is its own gated step (the kernel
    object must be byte-identical across it).

## 9. Files touched (build order; each step bit-exact on its own; a16 and stock untouched throughout)

- **Tests first** (`tests/attention/run_fa3_mixed_page_transport.py`): the
  three dynamic-permutation patterns of 6.1 at q = 1 / 64 (124 / 124
  expected on the merged tree).
- **Probes** (`benchmarks/mixed_kv_f27_probes/`): `f27_fp4_copy.cu` (3.6,
  compile-only, authored here, run as the first remote action) and
  `f27_ctrl.py` (the control-word decoder of gate 5).
- **F27a - top and tail** (`sparse_mixed_mainloop.cuh`): rows and page words
  at the top; `ok = __all_sync(~0u, (okK | !hasK) & okV)`, one spin block of
  voted `try_wait` loops; the chunk gather / store under one test after the
  arrives; the trace stamps.  F26's half order otherwise.  Gate: branches
  row (`BSSY` 0, `BRA.DIV` 0), order row (rows first), protocol <= 105;
  tests.
- **F27b - half pairs** (`sparse_mixed_mainloop.cuh`): the static loop as
  3.1 (`pending_loads` / `pending_scales` / copies / `pending_body` per
  operand with `cp_async_wait<1>` before each; the K half's commit
  unconditional; the peel commits two groups in the compressed modules);
  **no pins** (3.2); K(last) alone and the drain unchanged (`wait_group 0`).
  Gate: order row, `LDGDEPBAR` 2 per pair, the control-word report, `ptxas
  -v`; tests; **then the 6.2 trace (both waits) before anything else**.
- **F27c - dynamic module** (`sparse_mixed_mainloop.cuh`): `chunk_store`'s
  sorted entries + slot nibbles (dyn arm), `issue_tile_copies` (dyn) over
  entries with slot offsets and count predicates, `copy_scale_rows_dyn` with
  count predicates and the lane's slot, `expand_operand` (dyn) ->
  `dyn_scales` over sorted slots + two votes + `expand_class_loop<FP8 / FP4,
  EXACT>` (the software-pipelined loop over entries [lo, hi) with the slot
  by register shift); `expand_format_pages`, the `pending_mask` readers and
  the `FLO` selection removed; the dynamic loop in the half-pair order (its
  pending state the K' row = this pair's V row and the V' row's words 6-7).
  Gate: dyn rows; `ptxas -v` 0 spills (stop otherwise); tests incl. the
  permutation cases.
- **F27d - measurements**, in this order: the fp4 compile-only probe (3.6)
  and its fix if reproduced (a source-form change in `copy_compressed_page`
  / `cp8`, gated by the fp4 copy-block row); section 5 gates on each build;
  6.2 trace (hard stop); 6.4 locked bench; 6.3 ncu; the results section of
  this document and the amendments of section 7.
- **Separately, if the project decides it**: the scale layout commit
  (`kv_cache_fp8.py`, `fp4_kv_quantization.cu`, the bench and test
  transports, the XQA hosts' strides) with the kernel object byte-identical;
  then 3a (`SCALE_ROW_STRIDE` -> the row-pair layout, `copy_scale_rows` as a
  `cp16_pred` by even rows, `sc_rd` / `sc_cp` / `PagePos::sc_off` = `j x 16`)
  under its own gate row; 3b only after 3a's ncu shows the copy block still
  above 8 cycles per wavefront.

## 10. Remote actions, in order (none started here)

1. `benchmarks/mixed_kv_f27_probes/f27_fp4_copy.cu` compile-only probe (3.6)
   on nkcut2 with the F26 env (`/tmp/mixedkv-wtF26-env.sh` for `nvcc` on
   `PATH`; `nvcc -arch=sm_90a -O3 -cubin` + `cuobjdump -sass`), read with
   `f26_seg.py`: does `copy_compressed_page<FP4>` reproduce `VIADD ..., URZ`?
   Which of the two source forms (kernels `probe_fp4_void`, `probe_fp4_u8`,
   `probe_fp4_cp8u8`) removes it?  Authored in this worktree, **not run**
   (no local nvcc; remote runs excluded from this step).
2. ~~`f27_pins.cu`~~ **done for revision 2**: `f27_pin_probe.cu` ->
   `.ptx` lines 76-77 (empty asm block, no operand references), SASS: the
   `LDGSTS` address pair rewritten 3 instructions later, the `STS` data 19
   later.  Result: the pins are withdrawn (3.2, section 11).
3. Builds F27a, F27b, F27c with gates (`f27_ctrl.py` report on each); 6.2
   (both waits); 6.4; 6.3.

## 11. Judge blockers on revision 1 and their resolutions (revision 2)

Each item names the rev 1 text, the defect and what rev 2 does instead; the
design changed where the blocker required it.

1. **The MIO-operand pins were a no-op at the ptxas level** (rev 1 3.2, the
   arm budget 1414 -> 2 x 460, the +18-register budget of 4.5, the "pins
   emit no instruction, count within +6" gate row, the F27' fallback).  An
   empty `asm volatile("" :: "r"...)` lowers to PTX as `// begin inline asm`
   / `// end inline asm` with no instruction referencing the operands
   (`nkcut2:/tmp/mixedkv-wtF27-art/f27_pin_probe.ptx` lines 76-77 and the
   judge's `/tmp/mixedkv-wtF27-judge/pins.ptx`); ptxas computes liveness from
   PTX instructions, so the pinned values die at their last real use and the
   probe SASS shows the `LDGSTS` address pair rewritten 3 instructions after
   the `LDGSTS` and `STS` data 19 after the `STS` despite the pin.  **Fix:
   the pins are withdrawn** (3.2, do-not-build 12); the arm budget is
   re-derived (item 2); the gate row is replaced by the control-word report
   (item 4); 4.5 loses the 18 registers (widest ~109); the F27' fallback is
   restated as F27a alone at the derivable-only price (6.2).  The
   instruction-emitting sink the blocker offered as the alternative was
   priced (~16 issue slots + 1 wavefront per half) and rejected for the
   reason in item 2: it moves the writer, not the wait.
2. **The WAR reading was right in kind but wrong in mechanism, and the arm
   budget did not follow.**  Control bits: all 12 `STS.128` of the K hot arm
   set `rd = SB3`, the V arm's `SB2` (fp4 25 on `SB1` / 23 on `SB2`); `IMAD
   R61 @0xd4e0` waits `[SB2, SB3]`, `VIADD R12 @0xf380` waits `[SB1, SB2]`.
   A scoreboard is a per-index counter shared by the whole body, so writing
   *any* of the ~50 store-operand registers waits for the *last* store's
   operand read; `0xf380`'s R12 is block 1's output, outside the pin set.
   The stall equals the trailing burst's pipe time (16 wavefronts x 8 = 128
   at `0xd4e0`; ~8 at `0xf380`) at the ~8 cycles per wavefront the doc
   itself measures for `LDGSTS` (581 / 72) and F25's bodies (395 / 48): pipe
   time, relocatable, not removable by allocation.  **Fix: "WAR 305 removed"
   is withdrawn** (1.2 re-read, 3.2); the arms are priced at >= 384 pipe +
   ~128 trailing drain + ~40 vote / branch = ~550 each as the *floor* and at
   F26's 1414 for the centre; the pair centre is re-derived (item 3).
3. **The 1805 centre contradicted the design's own C19 rate.**  Producer
   wavefronts per warp per pair ~(96 `STS` + 72 `LDGSTS`) x 8 + (46 + 12 +
   10 `LDS`) x ~3.7 = ~1590 pipe cycles, + top 165 (pipe idle after the
   fence) + fence drain ~100 = ~1860 with perfect ALU overlap, above 1805.
   **Fix: 4.2 is rewritten with four columns** - floor ~1985 (segment
   prices, labelled a model output), **centre ~2300 -> 293 us** (the former
   "arms unmoved" sensitivity; the rev 1 pessimistic 2240 sat below the
   centre's arms and is replaced), derivable-only ~2465 -> 315 us (the top's
   branch structure alone, the one PC-by-PC argument), pessimistic ~2620 ->
   334 us (a reject by 4 us); the "<= 2216 consumer-bound" claim is dropped
   for every module (0 and 4.2), 4.3 states the ~1860 pipe-time floor, the
   fp8 band is 290-334 (the 288-302 band is withdrawn), fp4's derivable-only
   column (344 us) is stated as a reject so that fp4's accept is seen to
   depend on the copy block or the arms, mixed's own centre is above 330 and
   stays not claimed.
4. **The SASS gate set could not stop a bad arm build** (`f27_war.py`'s
   distance check meaningless without a real pin; its "blocks 0-3 reported,
   not gated" clause exactly where the relocated wait lands - the next half's
   12 `LDS.64` destinations are allocated from the dead body outputs).
   **Fix: the row is replaced by a control-word decoder report**
   (`benchmarks/mixed_kv_f27_probes/f27_ctrl.py`: stall / yield / wr / rd
   barrier index / wait mask per instruction; per body the rd scoreboard its
   stores set, the first later instruction that waits on it and the
   distance; the same for the `LDGSTS` and the landing loads' write
   barriers), explicitly a report and not a stop because the wait is
   expected at 136 registers; the ncu segment table (6.3) is the arbiter of
   the arm price, with the arms priced at F26's value beforehand (item 3).
5. **The dynamic decode loops indexed a register array by a runtime trip
   variable** (`entry_i`, `slot_i`, `page_i` for i = 0..n8u-1 against "the
   six entries (rows: 6 + 2, as static)" = C2 local memory, caught only
   after the build), and the per-iteration count had no load of the entry
   word.  **Fix (3.4): the entry's slot is a nibble of the row's word 6**,
   read in the loop as `(w6 >> (4 i)) & 0xF` - a register shift by a
   register amount plus a mask, no register array, no `LDS` of an entry word
   (the blocker's `LDS.32 [row + 4 i]` form is not needed once the slot
   leaves the page word); the decode needs no page index at all (only the
   copies do, and they are unrolled over registers).  +4 ALU per iteration
   (54 / 62 per block), the register model ~92, the counts in 3.4 / 4.1 /
   4.2 updated; gate 5's dyn row names the `SHF.R.U32 R, R, R` form and adds
   `LDL` / `STL` 0 over the whole region.
6. **The entry encoding `page | slot << 28` needed `IdType` pages < 2^28 and
   the doc claimed a host check that does not exist** (`mixed_page_prefill.py`
   bounds `head_dim`, dtypes and shapes only).  **Fix: the slot nibbles move
   to word 6 of the dynamic row** (6 x 4 bits); the page words stay full
   32-bit; no host check is added; the `m8 | m4 << 8` masks of C10 go (no
   reader remains: copies and decode consume counts and slots; the tests
   read the output, not the table).  C10 / C17 restated (section 7); the
   `pending_mask8 / 4` readers are removed with F27c.
7. **Index typo in 3.1**: pair t "finishes (K', V') = (K(t-2), V(t-1))"
   should read (K(t), V(t+1)) - the previous pair t+1 issued (K(t), V(t+1));
   K' = entry eV (this pair's V row), V' = entry eV - 1, matching the code
   (pK = sV, pV = sV - 1).  **Fixed** in 3.1; it matters for the dynamic
   module's pending rows (K' row = this pair's V row already loaded; V' row
   = `(oV - 32) & 0x3FF`).

Notes taken into the design (not blockers): the group accounting is stated
with its two code requirements (the peel commits two groups, in the
compressed modules only; the K half's commit is unconditional so the last
pair's empty A(t) keeps B(t+1) completable) and the 1-tile / K(last) / drain
cases (3.1); A9 / A7 per half and per loop iteration (3.1, 3.4); the spin is
a warp-uniform `try_wait` loop with `hasK` folded into the predicate, and
`PipelineAsync::producer_acquire` is do-not-build 13 (its per-thread asm
loop is the source of F26's `BSSY` / `BRA.DIV`); the folded 1/8 tail test
`(jrow & ~0x100) == 0` is checked against `META_J_MASK = 0x1E0` and the
chunk-table WAR distance (8-9 pairs > 3) restated with the gather's
later-not-earlier register fill; `0xaf30` re-read as the landing loads' data
latency on a merged wr / rd SB1 (the row hoist removes the PC; the decoder is
gate tooling); the ~1550 covers labelled model outputs with 6.2 the arbiter;
the F26 ncu "mixed" row identified as F26c (1321 per warp), not the merged
F26a, and the "~1000-1100 implied by the bench" labelled as resting on an
assumed IPC; the class-sorted rank shown to be a permutation of 0..5 (the
a16 class absorbs tail slots); the not-pending operand of the dynamic loop
stated as `(w7 & 0xFF) != 0` of the K' / V' rows; the arm floor's trailing
drain (~128) placed after each body, not inside the 460 that rev 1 had; the
realistic expectation under the corrected model - fp8 ~2450-2500 issue ->
~310-320 us, accept on the target, outside the withdrawn band - is the
derivable-only column; the recommended build order (F27a and F27c first,
F27b's arm claim re-designed around pipe time) is section 9's.

## 12. As written (filled per step; tests, probes, F27a-c in this worktree; nothing built or run here)

### As written: tests (`tests/attention/run_fa3_mixed_page_transport.py`)

Not run in this worktree (no GPU; review by reading).  `_pattern_transport(shape,
dtype, dev, pattern)` = `_make_transport(.., "mixed")` re-tagged with the
pattern cycled over the physical pages and the A16 reference recomputed with
the extremes helper's host decode (`_apply_decoded_reference`, shared with
`_extreme_transport`; g = 1, block scales {0.5, 1, 2}: every block folds);
`_run_dynamic_pattern(name, pattern, q_len)`: B = 2 x H = 2, 18 pages = 3
tiles per request, kv_len 285, dynamic module against the a16 module,
`torch.equal`.  Patterns: `(2, 0, 1, 1, 0, 2) x 3`, `(0, 0, 2, 1, 0, 0) x 3`,
`(1, 2, 0, 1, 2, 0) + (0,) x 6 + (2, 1, 0, 2, 1, 0)`; q = 1 and 64 causal.
124 cases; run first on the merged tree (layout-independent), then per step.

### As written: probes (`benchmarks/mixed_kv_f27_probes/`)

`f27_fp4_copy.cu`: four `extern "C"` kernels sharing one six-copy block in
the kernel's context (a `__noinline__` `compressed_base` from halved byte
strides on a `__nv_bfloat16 const*`, the base carried across a `__syncthreads`,
six page words in registers, `dst = land + j x 2048`, the fp4 landing's `+ 8
(k & 1)`): `probe_fp8_ctrl` (`cp16(page_src(...))`, the fp8 form, control),
`probe_fp4_void` (`cp8(page_src(...))`, the F26 fp4 form under test),
`probe_fp4_u8` (`uint8_t const*` arithmetic, no `void const*` step),
`probe_fp4_cp8u8` (`cp8_u8` with a `uint8_t const*` asm operand).  Compile
with `nvcc -arch=sm_90a -O3 -cubin`, read with `cuobjdump -sass`: count
`VIADD .., UR` / `UMOV UR, URZ` / `IMAD.MOV.U32` between the first
`IMAD.WIDE.U32` and the `LDGDEPBAR` of each kernel.  **Not compiled here**
(no local nvcc; remote runs excluded from this step): the fp4 centre of 4.2
keeps the excess.  `f27_ctrl.py`: the control-word decoder of gate 5
(`--all` prints every instruction's stall / yield / wr / rd / wait bits; the
default prints, per `STS.128` body (>= 8), `LDGSTS` block (>= 4) and `LDS`
group (>= 6) in `[--lo, --hi]`, the scoreboard set and the first later
instruction whose wait mask names it, with the distance).  Layout: control
= second qword >> 41; stall [0:4), yield [4], wr [5:8), rd [8:11), wait
[11:17), reuse [17:21) (7 = no barrier).  Checked on a two-instruction
synthetic input only.

### As written: F27a (top and tail; `sparse_mixed_mainloop.cuh`)

Not built or run here.  Changes, all inside the compressed modules' loop
(`if constexpr (STATIC_A16)` takes the untouched `pair_step` loop; `produce_
pair`, `pair_step`, the peel, K(last) alone and the drain are unchanged):

- `mixed_detail::spin_uniform(ClusterBarrier const&, phase)`: `do { done =
  __all_sync(~0, bar.try_wait(phase)); } while (!done);` - the exit predicate
  is a `VOTE.ALL` result.
- Top of the pair, in program order: `hasK`, `jrow`, the scheduler prefetch
  test, the counter (`sK`, `phK`, `pK`, `pV`, `sVn`, `phVn`, `sKn`, `phKn`),
  **the rows** (`read_meta_row(rowK)`, `read_meta_row(rowV)`, both
  `scale_page_word`s - now for both the static and the dynamic branch),
  then `ok = __all_sync(~0u, (okK | !hasK) & okV)` and `if (!ok) { if (hasK)
  spin_uniform(empty_k[sK], phK); spin_uniform(empty_v[sV], phV); }`.  The
  F26 `if (hasK && !okK) producer_acquire(..)` / `if (!okV)
  producer_acquire(..)` pair is gone from the loop (it stays in
  `produce_pair` for the peel and K(last) alone, shared with the a16
  module).
- The static branch keeps the F26b order (`cp_async_wait<0>` -> both
  operands' `pending_loads` -> `__syncwarp` -> both `pending_scales` -> K
  copies (`hasK`) -> V copies -> one commit -> K' body -> V' body -> probes
  -> fence -> arrives) minus the rows, which moved up; the dynamic branch
  keeps the F26a order minus the rows.
- Tail, after the arrives and before the advance: `if ((jrow & ~0x100u) ==
  0u) { if (jrow == 0u) { --chunks_left; if (chunks_left > 0) chunk_load(..); }
  else if (chunks_left > 0) { chunk_store(..); group_barrier(); } }` - the two
  F26 top-of-pair tests removed.  `chunks_left` is decremented at the same
  pair as before (j == 0), so the store's chunk index at j == 8 is unchanged.
- Trace: `acq` = the spin block (`acqK` = the K wait, `acqV` = the V wait;
  `tr0b = tr0` when the test passed), `gat` / `bar` stamped inside the tail
  block; the other buckets as F26b.

Expected SASS (gate 5): loop site in order rows (`LDS.128` x 4, `LDS.32` x
2) -> counter -> `PLOP3` + `VOTE.ALL` + `@!P BRA` (the spin block out of line:
`TRYWAIT`, `VOTE.ALL`, `@!P BRA` per barrier, no `BSSY`) -> `LDGDEPBAR` +
`DEPBAR.LE SB0, 0x0` -> 24 `LDS.64` -> `NOP` (no `BRA.DIV`) -> 12 `LDS.32`
-> copies -> `LDGDEPBAR` -> `VOTE.ALL` -> arm -> `VOTE.ALL` -> arm -> 2
`PHASECHK` -> `MEMBAR` + `FENCE` -> 2 `ARRIVE` -> `LOP3 P` + `@P BRA` (the
tail block out of line) -> loop; `BSSY` / `BSYNC` 0 outside the chunk-store
block, `BRA.DIV` 0; protocol <= 105.  Bodies, copy blocks, the a16 and stock
objects unchanged.

### As written: F27b (half pairs, static modules; `sparse_mixed_mainloop.cuh`; no pins)

Not built or run here.  The static branch of the loop (`if constexpr
(DYNAMIC) {..} else {..}`) is the 3.1 order; the dynamic branch keeps F26a's
until F27c; `produce_pair` gains one `cp_async_fence()` between the K and the
V issue under `if constexpr (HAS_COMPRESSED)` (the a16 module's text is
unchanged).

**Static loop pair, in program order** (after F27a's rows, counter and spin
test): `{ cp_async_wait<1>(); PendingStatic PK; pending_loads(PK, K.bases,
pK); __syncwarp(); pending_scales(PK); if (hasK) issue_loop(K, mK, spK, sK);
cp_async_fence(); pending_body(K'); }` then `{ cp_async_wait<1>(); PendingStatic
PV; pending_loads(PV, V.bases, pV); __syncwarp(); pending_scales(PV);
issue_loop(V, mV, spV, sV); cp_async_fence(); pending_body(V'); }` then the two
`test_wait` probes, `fence_view_async_shared()`, the two `producer_commit`s,
F27a's tail test and the advance.  The K half's `cp_async_fence()` is outside
`if (hasK)`.  The two brace blocks state the register claim of 4.5 (one
`PendingStatic` live at a time); ptxas -v decides.

**Group accounting as written.**  K(last) alone (`produce_pair(Partial, Full,
last, -1)`): K copies, fence (A), no V, fence (empty) -> `finish_one(K)`'s
`wait_all`.  Peel (`pair_step(kv_tile_idx, Full, Partial)` -> `produce_pair(
Full, Partial, last-1, last)`): K(last-1) copies, fence (A0), V(last) copies,
fence (B0).  Loop pair 1: K half `wait_group 1` -> A0 complete (B0 may pend);
fence (A1); V half `wait_group 1` -> outstanding [B0, A1] -> B0 complete.  Pair
t: [A(t+1), B(t+1)] -> A(t+1); [B(t+1), A(t)] -> B(t+1).  Last pair (hasK
false): A(t) empty; the V half's `wait_group 1` over [B(t+1), A(t) empty]
completes B(t+1) ("all prior groups complete").  Drain (`finish_one_stage(V)`):
`wait_all`.  1-tile items (`n_loop_pairs = 0`): the peel with tK = -1 commits
an empty A0 and B0 = the V copies; the drain's `wait_all` finishes it.
Dynamic module in this commit: its single-group loop pairs still complete
after the peel's two groups (`wait_group 1` at pair 1 completes A0 and B0).

**Hazards (A7 / A9 / D5) per half.**  K' loads read K-ring stage pK = sV,
the K copies write K-ring sK = sV + 1, the V copies write V-ring sV, the V'
loads read V-ring pV = sV - 1: the pending stages are the ones this pair's
copies do not touch.  Per half every lane's landing loads precede the half's
`__syncwarp`, every lane's stores follow it; the half's scale slots landed at
each lane's own `wait_group` before that `__syncwarp`.  The octet WAR (K-ring
sK's slots last read two pairs ago; the acquire of sK orders those reads
before these writes) is F25's argument unchanged.  One fence, both arrives
after both bodies (D5; the K' arrive after the V' body is the F26 position,
do-not-build 4).

**Trace (6.2).**  `wait` = the K half's `DEPBAR` (`trw - tr1`), `fcK` = the V
half's (`trw2 - trk`): the two covers; `barB` = both halves' loads + barrier
+ chains; `iss` = both copy blocks to their commits; `expK` / `expV`; `fcV` =
probes + fence + arrives; `fin` = bodies + `fcV`.  Print format unchanged.

**Expected SASS (gate 5).**  Loop site in order: rows -> counter -> `PLOP3` +
`VOTE.ALL` + `@!P BRA` -> `DEPBAR.LE SB0, 0x1` -> 12 `LDS.64` -> `NOP` -> 6
`LDS.32` -> 7 `IMAD.WIDE.U32` / 7 `LDGSTS` (chains interleaved) ->
`LDGDEPBAR` -> `VOTE.ALL` -> K arm -> `DEPBAR.LE SB0, 0x1` -> 12 `LDS.64` ->
`NOP` -> 6 `LDS.32` -> copies -> `LDGDEPBAR` -> `VOTE.ALL` -> V arm -> 2
`PHASECHK` -> `MEMBAR` + `FENCE` -> 2 `ARRIVE` -> tail test -> loop;
`LDGDEPBAR` exactly 2 per pair (no `wait_all` empty group in the loop),
`DEPBAR.LE SB0, 0x0` 0 in the loop; `BSSY` / `BRA.DIV` 0; bodies unchanged;
`f27_ctrl.py` report: the K body's rd scoreboard first waited on by the V
half's `DEPBAR` successor or a V' `LDS.64` (the design expects the wait
there; distance reported), the V body's by the probes / fence.  `ptxas -v`:
0 / 0 / 0, no C7507.  Then tests, then the 6.2 trace at both waits before
anything else.

### As written: F27c (dynamic module; `sparse_mixed_mainloop.cuh`)

Not built or run here.  Files: `sparse_mixed_mainloop.cuh` only (the host
Python, `kernel_traits.cuh` and the shared Hopper files untouched; `TileMeta`
keeps its 32 B shape - words 0-5 `pages[]`, word 6 `tags[0..3]`, word 7
`tags[4..5] | valid | flags` - the dynamic module re-interprets words 6-7).

**Chunk store (dynamic arm).**  `is8 / is4` from the tag; `m8`, `m4` by the
two existing `__reduce_or_sync`s; `n8 = popc(m8)`, `n4 = popc(m4)`; `below =
(1 << slot) - 1`; `rank = is8 ? popc(m8 & below) : is4 ? n8 + popc(m4 &
below) : n8 + n4 + popc(~(m8 | m4) & 0x3F & below)`; `own = slot < 6`;
`nibbles = __reduce_or_sync(octet, own ? slot << 4 rank : 0)` (a third
`REDUX.OR`); `if (own) sts32(row + 4 rank, page)`; lane 0: `sts64(row + 24,
{nibbles, n8 | n4 << 4 | valid << 16 | flags << 24})` with `flags =
kFlagFilled | (n8 | n4 ? kFlagCompressed : 0)`.  New `mixed_detail::sts64`
(`st.shared.v2.b32`).  `static_assert(PAGES_PER_TILE <= 15 && 4 x
PAGES_PER_TILE <= 32)`.  The static arm is untouched.

**Row accessors.**  `TileRegs::n8() = w7 & 0xF`, `n4() = (w7 >> 4) & 0xF`,
`slot(i) = (w6 >> 4 i) & 0xF` (i an unrolled constant); `mask8 / mask4` and
`pending_mask8 / 4` removed; `pending_n8 / n4` read the pending word's bits
0-7; `entry_slot(slots, i)` is the same shift with `i` a register (the loops
and the lane-indexed scale copy).

**Copies.**  `copy_dynamic_entry<FULL>(b, page, slot, p8, p4, stage, valid,
t)` = `copy_dynamic_page` with `PagePos pp = dynamic_page(slot)` (the slot a
register: `off = slot x 2048`, `sc_off = slot x 8`, `tok0 = slot x 16`);
`issue_tile_copies` (dyn): `n8 = m.n8()`, `n4 = m.n4()`, for i = 0..5
unrolled `is8 = i < n8`, `is4 = !is8 & (i < n8 + n4)`,
`copy_dynamic_entry(b, m.page(i), m.slot(i), is8, is4, ..)`, then
`copy_scale_rows_dyn<FULL>(b, spage, m.w6, n8, n4, stage, valid, t)`: lane k
copies sorted entry k's row (`spage` = word k of the row, as F26) with `p8 =
k < n8`, `p4 = !p8 & (k < n8 + n4)`, `slot = entry_slot(w6, k)` (register
shift by 4 k), `dst = sc_cp + stage x SCALE_STAGE + 8 slot - 8 k`, partial
src-size `16 slot + r < valid`.  No ballot at the copy sites (the copies are
predicated on value-uniform counts; 3.4 amended).

**Decode.**  `PendingDyn {e, slots, n8u, n4u, ok8, ok4}`; `dyn_scales(pd, b,
slots, n8, n4, stage, t)`: `expand_bases`, `n8u = popc(ballot(lane < n8))`,
`n4u` likewise, six `lds32(e.sc + dynamic_page(entry_slot(slots, i)).sc_off)`
(constant shifts), `f_i`, `ok8 &= !(i < n8u) | fold_ok(f x gs8)`, `ok4 &=
!(n8u <= i < n8u + n4u) | fold_ok(f x gs4)` bitwise; `dyn_bodies<VOTE>`: with
FOLDS && VOTE two `__all_sync`s selecting `expand_class_loop<FP8, hot / exact>
(.., slots, 0, n8u)` and `<FP4, ..>(.., slots, n8u, n8u + n4u)`; FOLDS alone
(the single-operand sites) the two exact loops; f16 the two hot loops.
`expand_class_loop<FP8, EXACT>(prm, isK, e, b, slots, lo, hi, t)`: `if (lo ==
hi) return;` (uniform), `g` (EXACT), `i = lo`, `cur = dynamic_page(entry_slot(
slots, i))`, `c = load_packed(e, cur.off)`; `for (;;)` (`unroll 1`): `nxt =
i + 1 < hi ? i + 1 : i` (a `SEL`), `np = dynamic_page(entry_slot(slots,
nxt))` (register shift), `x = load_packed(e, np.off)`, `sw = lds32(e.sc +
cur.sc_off)`, `sf`, `__syncwarp()`, `expand_block<FP8, EXACT>(e, c, sf,
cur.off)`, `if (nxt == i) break; i = nxt; cur = np; c = x;`.
`expand_operand<VOTE>(prm, isK, b, slots, n8, n4, stage, t)` (dyn arm):
`__syncwarp(); dyn_scales; dyn_bodies` - the single-operand sites;
`expand_format_pages` removed.

**Pending state.**  `Operand` gains `pending_slots / staged_slots`
(`issue_operand` stores `m.w6` beside the pending word; `rotate_pending`
copies both; `expand_pending` passes `pending_slots`, `pending_n8 / n4` and
the stage) - the single-operand sites (K(last) alone, the drain).  The loop
keeps no pending words: `issue_loop` (dyn) only issues the copies and the
`cpasync_barrier_arrive` of an all-a16 tile; the `finish_pending_pair` lambda
is removed (`produce_pair`'s FINISH arm now holds `static_assert(!HAS_
COMPRESSED || decltype(kpart)::value)`: only the a16 module's `pair_step`
reaches a full / full `produce_pair`).

**Dynamic loop pair** (after F27a's rows and spin test): `rowVp = meta_base +
((oV - 32) & 0x3FF)`, `wVp = lds64(rowVp + 24)` (w6', w7'), `pendK = (mV.w7 &
0xFF) != 0`, `pendV = (wVp.y & 0xFF) != 0`; K half: `cp_async_wait<1>();
__syncwarp(); PendingDyn DK; dyn_scales(DK, K.bases, mV.w6, mV.n8(), mV.n4(),
pK); if (hasK) issue_loop(K, mK, spK, sK); cp_async_fence(); dyn_bodies<true>
(.., DK, K.bases)`; V half: `cp_async_wait<1>(); __syncwarp(); PendingDyn DV;
dyn_scales(DV, V.bases, wVp.x, wVp.y & 0xF, (wVp.y >> 4) & 0xF, pV);
issue_loop(V, mV, spV, sV); cp_async_fence(); dyn_bodies<true>(.., DV,
V.bases)`; probes; fence; `if (pendK) producer_commit(pK); if (pendV)
producer_commit(pV)`.  A not-pending operand's `dyn_scales` reads stale slots
(harmless: every slot < 6, inside the stage) and its loops run zero steps
(n8 = n4 = 0 in its row word), so nothing is branched but the arrive.  Drain:
`rowL = meta_base + ((oV - 32) & 0x3FF)`, `wL = lds64(rowL + 24)`, `V.pending
= (wL.y & 0xFF) ? (wL.y & 0x03FFFFFF) | (sV - 1) << 30 : 0`, `V.pending_slots
= wL.x`, `finish_one(V)` (wait_all, exact bodies, conditional arrive).

**Bit-exactness argument.**  Every entry's bytes land at `slot x 2048` /
`slot x 8` of the stage, the addresses the slot-ordered form used; the copies
are the same four predicated `cp.async` per entry with the same D4 src-size
rule (`16 slot + r < valid`); the decode of an entry reads its own landing
(`e.l8a / l8b + slot x 2048`) and its slot's scale word, and stores to `e.d0
/ d1 + slot x 2048` - `expand_block` unchanged; the per-format vote is an
AND over the same set of blocks in another order (order-independent); the
cold body is `expand_block<.., true>` as before; a16 slots copy as before;
the class order of the row is a permutation of the six slots, so every slot
is copied and decoded exactly once.

**Expected SASS (gate 5, dyn row).**  Copy body per operand: 6 x (`LOP3` /
`SHF` slot, 3 `IADD3` destinations, 4 `IMAD.WIDE.U32`, 4 `@P LDGSTS`) + 2
`@P LDGSTS.64` scale = 26 predicated `LDGSTS`; `SEL` on an address 0;
`chunk_store` 3 `REDUX` + 2 `POPC` + `@P STS` (register address) + `@P
STS.64`; per operand `dyn_scales` 2 `VOTE.ANY` + 2 `POPC` + 6 `LDS.32`, 2
`VOTE.ALL`, four rolled loop bodies (hot8, cold8, hot4, cold4) each with one
back-edge, one `NOP` (the per-step barrier), `SHF.R.U32 R, R, R` + `LOP3 ..
0xF` for the slot, `FLO` / `BREV` 0; one `LDS.64` at the top (V' words), two
`ISETP` pending tests, `@P SYNCS.ARRIVE` x 2; `BSSY` / `BSYNC` 0 in the loop
site outside the chunk block; `LDL` / `STL` 0 in the region; `ptxas -v` 0 /
0 / 0 (stop otherwise); loop site <= 1050, region <= 2300.  Tests incl. the
six pattern cases.


### As measured: F27 (2026-09-04, nkcut2 H200; wt/F27 @ 27c27216 = F27a-c + review fixes; full record in docs/mixed_kv_page_transport_backends.md, Track F [27])

**Build.**  One build of 27c27216 (`/tmp/mixedkv-wtF27`), one trace build
(`/tmp/mixedkv-wtF27-trace`, `-DMIXED_FA3_TRACE=1`), one locked bench, one
ncu pass per mode; artifacts `nkcut2:/tmp/mixedkv-wtF27-art/`.  `f27_ctrl.py`
validated on F26's `sf1_mask0.sass` first (0xd4e0 waits [SB2, SB3]; K hot
`STS.128` rd = SB3; V arm rd = SB2, 0xf380 waits [SB1, SB2]).

**6.1 tests: 124 / 124 bit-exact** (the six dynamic-pattern cases included;
`.o` mtimes after the run's start; the import path printed in-process).

**Gate 5.**  ptxas 0 / 0 / 0 on all four modules, no C7507, `USETMAXREG 0x88 /
0xB8`, STACK 0, `LDL` / `STL` 0 - **the dynamic module's register claim
(3.4) holds**.  Order as 3.1 exactly (rows at the top -> ok `VOTE.ALL` ->
spin -> `DEPBAR.LE SB0, 0x1` -> 12 `LDS.64` -> `NOP` -> 6 `LDS` -> 7 + 7 ->
`LDGDEPBAR` -> vote -> arm, twice -> probes -> fence -> arrives -> chunk test);
two commit groups per pair.  Bodies unchanged (480 / 504 per arm incl. vote +
branches; `MOV` 0).  a16 identical to F26's stream, stock byte-identical.
**Missed:** `BSSY` 2 + `BRA.DIV` 4 on the common path (ptxas' convergence
guards; the ok and chunk tests laid out as taken branches over inline blocks;
the K half's scale copy as a per-lane `@P4 BRA`); region 2525 / 2602 / 3313
(<= 2500 / 2550 / 2300); dyn loop site 1625 with `BSSY` 13 (<= 1050, 0); the
3.6 spurious `UMOV URZ` + `VIADD` present on fp8 and fp4 alike (~13 per half;
the probe was not run).  `f27_ctrl.py`: K hot body rd SB2 -> first wait **+1**
(`IMAD R25, R101, 0x6000`), V hot body rd SB1 -> **+2** (`VIADD R18`): the
drain lands on the first register write after the body, as in F26.

**6.2 trace (us per pair, traced build).**  `wait` / `fcK` = **fp8 0.012-0.027
/ 0.002-0.006, fp4 0.004-0.015 / 0.001-0.006, mixed 0.010-0.017 /
0.002-0.009: the hard stop passes at both `DEPBAR`s**; F27' not indicated.
fp8 `iss` 0.28-0.42 (<= 0.30 marginal), `expK` / `expV` 0.32 / 0.32, `acq`
0.40-0.58; fp4 `expV - expK` 0.07 (> 0.03); mixed `iss` 1.2-1.45 (<= 0.5
miss).  Traced pairs fp8 2.20-2.25, fp4 2.13-2.16, mixed 3.4-3.8 (F26 2.4-2.8,
2.5-2.7, 5.2-6.1); a16 2.02-2.04 unchanged.

**6.4 bench (locked, 5 trials, min / median / max; q=1 and q=64).**  stock
298.8 / 299.7 / 301.2 and 308.7 / 309.4 / 311.1; transport_a16 281.4 / 282.4 /
282.9 and 287.1 / 287.9 / 288.2 (controls hold).  **fp8 373.1 / 373.4 / 373.6
and 380.9 / 382.2 / 383.6; fp4 390.2 / 390.8 / 391.6 and 395.0 / 396.5 /
396.9; mixed 640.4 / 644.4 / 646.8 and 647.2 / 649.2 / 650.1.**  Against
section 0: fp8 above the band's pessimistic end (334) -> **reject**; fp4 above
350 -> **reject**; mixed above 420 -> **reject** ("count model wrong again":
read below).  Versus F26 (659eacfa): fp8 +1.9 % / -1.7 %, fp4 -0.4 % / -0.9 %,
mixed +1.5 % / +0.6 %.

**6.3 ncu (q=1, third launch; segments per the F27 markers).**

| segment (fp8, issue cycles per pair) | 4.2 budget | F26 | **F27** |
|---|---|---|---|
| top (rows + ok vote + spin test) | 165 | 367 | **211** |
| K `DEPBAR` + landing | 70 | 186 (both preps) | **98** |
| K scales + copies | 240 | 581 (both copy blocks) | **517** (74 / `LDGSTS`) |
| K vote + arm | 460 | 810 | **745** |
| V `DEPBAR` + landing | 70 | - | **123** |
| V scales + copies | 240 | - | **395** (56 / `LDGSTS`) |
| V vote + arm | 460 | 604 | **477** |
| probes + fence + tail | 100 | 118 | **144** |
| **pair** | **<= 2590 accept (centre 2300)** | **2667** | **2711** |
| executed per warp per pair | ~655 | 651 | **689** |

The top (-156) and the arms (-192) moved as designed; the preps + copies
(+366) and the tail (+26) ate it.  The mechanism is 1.2's own: in each half
the first instruction after the landing loads writes the landing loads'
*address* register and waits for the twelve `LDS.64` to read it (`LDS R39,
[R44]` **5.2 %** short_sb 97 %, `IMAD.WIDE.U32 R32` **4.4 %**: F26's 0xaf30
once per half, 105 -> ~260 cycles), and the two post-body drains land at +1 /
+2 (`IMAD R25` **3.55 %**, `VIADD R18` **1.96 %**).  Four WAR PCs = ~410
cycles (F26 three = 305).  Spin branches 0.03 + 0.04 % (F26 1.6 + 1.1): the
uniform spin is almost never taken.  Consumer K-wait 6.0 % (F26 3.8) with the
pair > T_c: the producer paces.  Tensor 50.5 %.  Wavefronts 1951 (496 / 691 /
1001); octet copies 12.0 per instruction (the layout floor, 3.3).  fp4: pair
2848 (F26 2876), K arm 968 at dispatch 34.5 %, the same four PCs.  **mixed:
4647 issue cycles at 1232 executed per warp (design 750; F26 8178 / 1321):
the class loops 284 + 287 per operand (design ~220), the predicated copy
bodies 278 + 274 per half (design ~70: 52 predicated `LDGSTS` + 52 `IMAD.WIDE`
+ ~110 `@P IMAD.MOV` entry selects) at `wait` 37-47 %; consumer K-wait 47.8 %.**

**Verdict.**  Hard stop met, structure met, **6.4 reject on all three
compressed modes**; target <= 330 remains open.  What the design got right:
the half-pair covers (both `DEPBAR`s < 0.03 us), the top's price (-156 for
the removed branch structure), the arms' response to the order (-192), the
dynamic module's registers (0 spills at 168), bit-exactness with the sorted
table.  What it got wrong: the belief that the row hoist removes the
landing-address WAR (ptxas re-creates it wherever a dead MIO base register
meets the next MIO destination - now in both halves), the copy blocks' price
behind the other half's store burst (74 / 56 per `LDGSTS`, not 41), the flat
instruction count (+38 per warp per pair: the second landing group's
addresses, 4 `BRA.DIV`, 2 `BSSY` / `BSYNC`, the K half's split scale copy),
and the dynamic copy bodies' issue cost (predicated-off slots are not free
when the block is issue-bound).  Follow-ups 1-5 in the backends record.
