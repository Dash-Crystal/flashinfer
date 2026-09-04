# FA3 mixed-page producer: what the ptxas register schedule is doing (SASS/PTX/ncu review, no builds)

Read-only review of F26a+b (`main`, `sf1_mask0.sass` in `/tmp/mixedkv-wtF26-art`) and F27
(`wt/F27`, `/tmp/mixedkv-wtF27-art`), plus 17 compile-only probes (`nkcut2:/tmp/frev/probes*.cu`,
CUDA 13.0 `nvcc -arch=sm_90a -cubin`, control words decoded with the F27 `f27_ctrl.py`: `wr`/`rd` =
scoreboard set by the instruction, `wait[..]` = scoreboards it blocks on).  Question: why did five
rounds of source restructuring land their instruction counts and leave fp8 at 366-373 us
(target 330, consumer-bound floor 283), and what, if anything, ptxas can be made to honour.

## A. Hardware model: what an MIO operand scoreboard is and what it costs

LDS / STS / LDGSTS / SYNCS are dispatched to the SM's MIO queue and read their register operands
when the LSU dequeues them, not at issue.  ptxas therefore attaches a *read* scoreboard (`rdN`) to
the instruction's source registers; the counter is released when the LSU has read them.  Any later
instruction that *writes* one of those registers carries `wait[N]`.  Two properties make this
expensive here: (1) one counter is shared by every instruction that set it, so a write of *any*
register read by *any* STS of a body waits for the *last* STS of the body to be dequeued; (2) for
LDS, ptxas usually merges the read barrier with the load's write barrier (`wr1 rd1`), so a WAR on a
load's *address* register waits for the load's *data return*, i.e. the full queue round trip.
Queue depth = wavefronts ahead in the SM's shared pipe (producer STS 4/instr, LDGSTS 4-12/instr,
plus the consumer's 1001 gmma operand wavefronts per pair); F26 measured ~8 issue cycles per
producer wavefront, so a 48-wavefront body burst is ~380 cycles of queue.

F26 (`sf1_mask0.sass`), the three PCs the record priced at 305 cycles/pair:
```
0x0d470 st4 rd3 wait[-]   STS.128 [R103+0x2000], R60          <- K hot body: 12 STS.128, all rd3
0x0d480 st4 rd3           STS.128 [R102+0x2000], R64
0x0d490 st4 rd3           STS.128 [R103+0x2800], R68
0x0d4a0 st2 rd3           STS.128 [R102+0x2800], R72
0x0d4b0                   FSETP.GEU.FTZ.AND P5, PT, |R110|, 3.39e+38, PT
0x0d4d0                   FSETP.GEU.FTZ.AND P1, PT, |R101|, ...
0x0d4e0 st1 wait[2,3]     IMAD R61, R93, 0x6000, R6          <- V' stage base written into a K store operand: 127 cyc, short_sb 98 %
0x0ecd0 st1 rd2           STS.128 [R61+0x800], R12           <- V arm block 1 ...
0x0f380 st1 wait[1,2]     VIADD R12, R90, 0x1                <- ... phase word written into R12, 107 instr and 11 STS later: 72 cyc
0x0ad90 st4 wr1 rd1       LDS.64 R40, [R32]                  <- landing loads, base R32/R33/R34, wr and rd merged on SB1
0x0aea0 st1 wr1 rd2       LDS.64 R78, [R34+0x2800]
0x0aeb0 st5 wait[3,4]     BRA.DIV UR8, 0x11560               <- the __syncwarp guard waits on the first two landing loads
0x0af30 st1 wr1 wait[1,2] LDS.128 R32, [R72+0x30870]         <- chunk row into the landing *address* register: 105 cyc
```
F27 (`sf1_mask0.sass`) moved every burst and ptxas re-created the pattern at four PCs (~410 cycles):
```
0x0aa90 st1 wr2 rd1   LDS.64 R30, [R38]          K' landing group, bases R38 / R39 (= R38 ^ 8)
0x0ab50 st1 wr1 rd1   LDS.64 R66, [R39+0x2800]
0x0ab60 st5           BRA.DIV UR5, 0x11760
0x0ab70 st1           NOP                        (the __syncwarp)
0x0ab80 st1 wr1 wait[1]  LDS R39, [R44]          first scale word -> landing base register: 5.20 % of loop samples, short_sb 97 %
0x0cea0 st1 rd2       STS.128 [R117+0x2800], R64  K hot body, 12 x rd2
0x0ced0 st2 rd2       STS.128 [R116+0x2800], R68
0x0cee0 st1 wait[1,2] IMAD R25, R101, 0x6000, R7  V' stage base, +1 after the body: 3.55 %, short_sb 96 %
0x0cef0 st4           DEPBAR.LE SB0, 0x1
0x0cf40 st1 wr2 rd1   LDS.64 R24, [R32]           V' landing group, bases R32 / R33
0x0d000 st1 wr1 rd1   LDS.64 R62, [R33+0x2800]
0x0d010 st5 wait[2,3] BRA.DIV UR5, 0x11840        guard waits on the first two landing loads (1.13 %)
0x0d020 st1           NOP
0x0d0a0 st1 wait[1]   IMAD.WIDE.U32 R32, R20, UR5, R34   first copy address -> landing base register: 4.40 %, short_sb 94 %
0x0f300 st2 rd2       STS.128 [R68+0x2800], R64   V hot body, 12 x rd2
0x0f320 st1 wait[1,2] VIADD R18, R87, 0x1         phase word, +2 after the body: 1.96 %, short_sb 95 %
```
`f27_ctrl.py` on the F27 fp8 mask_1 object gives the same reading independent of ncu: K hot body
`rd SB[2]` first waited `+1` (`IMAD R25`), V hot body `rd SB[1]` first waited `+2` (`VIADD R18`), K
landing group `wr SB[1,2,3,4]` first waited `+3`.

What the excerpts say about the mechanism, beyond the F26/F27 records: the writer is always an
ALU instruction whose value comes from *loop state* (stage index x 0x6000, phase `+1`, ring offset)
or from *an earlier* MIO result (copy addresses from the row words loaded at the top), and its
destination is the register ptxas freed most recently -- an MIO operand.  The stall is real but it
is mostly *re-attributed queue latency*: at `0xab80` the scale word is consumed 10 instructions
later (`PRMT R60, R39` at `0xabf0`, 1.17 % short_sb), at `0xd0a0` the copy block is issue-bound
behind the same queue (74 cyc/LDGSTS, short_sb 33 %).  Removing the four WARs recovers the issue
slots between the WAR writer and the true consumer (10-30 instructions per PC), not the 410 cycles.

## B. Control-flow model: where BSSY / BRA.DIV come from and what ptxas proves uniform

`BRA.DIV URn, target` (preceded by `UMOV URn, 0xffffffff` = the `__all_sync` membermask) is ptxas'
convergence guard before a collective: if the active mask differs from URn it jumps to a
`BSSY; WARPSYNC.COLLECTIVE R69; VOTE.ALL / NOP; ENDCOLLECTIVE; BSYNC; BRA back` block (F27
`0x11760-0x117c0`, `0x11840-0x118a0`).  The fallback's `WARPSYNC.COLLECTIVE` waits on all six
scoreboards, and the guard itself inherits waits (`0xd010 wait[2,3]`, F26 `0xaeb0 wait[3,4]`) --
each guard exposes the first landing loads' latency (~30 cycles each, 1.1 % of samples) on top of
its two issue slots.  F27 fp8 executes four per pair plus `BSSY/BSYNC` at `0xab90/0xaec0` (around the
`hasK` + per-lane `@P4` scale-copy region) and `0xf380/0xf770` (chunk block, `BSYNC` 0.86 % no_inst).

Probes (`probes.cu`, `probes3.cu`, `probes4.cu`; all `-O3`, counts of `BSSY` / `BRA.DIV`):

| shape | guards |
|---|---|
| per-thread `if (v&1)` from LDS, then `__syncwarp` (P3a) | 0 (if-converted to `@!P0 STS`) |
| `if (__all_sync(...))` region, `__syncwarp` (P3b) | 0 |
| voted `try_wait` spin (`spin_uniform`), `__syncwarp`, vote (P3d, P11) | 0 |
| cutlass-style per-thread `try_wait` loop inside one asm, `__syncwarp` (P13) | 0 |
| per-lane `(lane&7) < 6` around a `cp.async`, `__syncwarp` (P17) | 0 (predicated) |
| **F27 top shape with the trip count read from smem by each thread (P14)** | **BSSY 2, BRA.DIV 3, WARPSYNC.COLLECTIVE 4** |
| same, trip count broadcast by `__shfl_sync(.., 0)` (P15) | 0 |
| same, loop exit voted `while (__any_sync(~0, it < n))` (P16) | 0 |

So ptxas' analysis is value-based, not construct-based: a branch is uniform iff its predicate is
built from kernel parameters, uniform-datapath values, `SHFL` broadcasts or `VOTE` results; a value
that went through a per-thread `LDS` / `LDG` is divergent, and every `VOTE.ALL` / `NOP(__syncwarp)`
reachable from a branch on it gets a guard until the next `BSYNC`.  In the kernel the loop bound
`kv_tile_idx`, `swa_begin`, `num_chunks`, hence `hasK`, the chunk test and the back-edge, descend
from `block_coord` (`kv_len` via the scheduler's per-thread loads: `sparse_mixed_mainloop.cuh:1740-
1761`), which is why the guards sit at the loop head (`0xa890`), both `__syncwarp`s and the K vote,
and why the V vote (after the V-half `NOP`, a reconvergence point) has none.  The `@P4 BRA` per-lane
branch (`k < 6` in `copy_scale_rows`, lanes 6-7 predicated off) is a real divergence and costs the
`BSSY/BSYNC` pair; in the probe the same predicate was if-converted because the block was one
`LDGSTS` -- in the kernel it guards `IMAD.WIDE.U32 + LDGSTS + fillers`, past ptxas' if-conversion
size.  The dynamic module's 13 `BSSY` (ballot-derived trip counts) are the P14 case again: `POPC` of
a `VOTE` result is uniform in value, but ptxas only tracks the `VOTE` predicate itself, not
integer arithmetic on it.

## C. Why each source-level control was invisible

1. **Empty pin `asm volatile("" :: "r"(v))`.**  PTX has no instruction for it (`f27_pin_probe.ptx`:
   `// begin inline asm` immediately followed by `// end inline asm`); ptxas computes liveness from
   PTX instructions.  A real consumer (`lop3` into a stored sink, P1) does extend the live range --
   and P2 shows the consequence when it is placed after the next burst: ptxas hoisted the four
   `LOP3` above the second store pair and allocated its result into the freed store operand
   (`STS.128 [UR5-0x4800], R4 ... rd1` -> `LOP3.LUT R4, R4, R5, R6` at +8, `wait[1]`).  The pin moves
   the WAR onto its own destination.
2. **Source order under `asm volatile`.**  Volatile orders PTX emission only.  ptxas keeps the
   relative order of shared-memory instructions (P2: `STS, STS, LDGSTS, LDS, STS, STS` survive in
   order) and freely moves everything else: F26 computed `expand_bases(pV)` in `pending_loads(PV)`
   before the copies; SASS placed `IMAD R61/R60, R93, 0x6000` at `0xd4e0/0xd500`, after the K body,
   because the value is one IMAD from live state and sinking it saves two registers across 400
   instructions.  P7 (20 registers live) keeps a hoisted base above the burst; F26 (130 live) sinks
   it.  Hoisting is honoured only when pressure allows -- not a property of the source.
3. **`if constexpr` / unrolling.**  Produces straight-line PTX; ptxas' list scheduler then reorders
   within the basic block by dependence and pressure.  Unrolled bodies get exactly the allocation
   above: the six blocks' outputs (48 regs) are distinct only because they are all live to the
   burst; the next writer after the burst takes the first freed one.
4. **`__syncwarp` / voted spins as convergence hints.**  ptxas does not read them as such (B): the
   guard count is decided by the *predicates* that dominate the region, so replacing cutlass's
   per-thread spin with `spin_uniform` removed nothing when the loop bound stayed divergent.
5. **`-Xptxas -O1/-O3`, `-maxrregcount 40`** (P0/P1/P2 rebuilt at each): identical scoreboard
   placement.  The reuse-just-freed policy is not a pressure heuristic; P10 (a 176-instruction
   F27-shaped half at ~50 registers) still writes body outputs into `LDGSTS` address pairs at
   +10..+16 and block j+1's outputs into block j's store operands at +15/+16.

## D. What does control the SASS schedule -- and the constructions ptxas honours

**D1. The one rule that removes a read scoreboard.**  P5 vs P6 vs P9:
```
P5  next writer is ALU from loop state:   STS.128 [UR4-0x800], R4  rd0  -> +3 IMAD R4, ... wait[0]
P6  next writer depends on a LATER LDS:    STS.128 [R14-0x800], R4  rd-  (no read barrier at all)
P9  two bodies, A's outputs consumed by predicate-only compares (setp -> @p add sink) after B's stores:
    STS.128 [UR5], R12  rd-   STS.128 [UR7-0x4000], R4  rd-   (both bursts barrier-free)
```
ptxas knows the MIO queue is in order: if every later writer of a store's operands is data-
dependent on an MIO instruction issued *after* the store, the WAR needs no barrier and it emits
none.  The a16 module shows the same (`0x84a0 LDGSTS ... [R60.64]` `rd-`: R60 is next written by a
`SEL` behind the next tile's row `LDS`).  This is the only mechanism in the probe set that ptxas
honours regardless of pressure, register cap or placement.  Its price: the value that overwrites a
burst operand must itself come from a load issued after the burst, so it arrives one queue round
trip later -- fine for the *next half's decode* (its inputs are the next landing loads anyway),
not for addresses and loop state, which are needed *to issue* those loads.

**D2. Rotating pool by overlapping live ranges.**  Works exactly as far as D1 lets it: P9's bodies
are barrier-free because their next writers are the next iteration's LDS-derived values, not
because the pool rotates.  A consumer that produces a vector register (P2) re-creates the WAR on
itself; a consumer that produces only predicates or updates a loop-carried accumulator (P9) does
not.  The kernel's `fold_ok` chain (`FSETP` -> `PLOP3`) is already of the harmless kind.

**D3. Uniform datapath.**  `UR`/`UP` cannot be requested from PTX.  P8 (stage index from a kernel
parameter, rotated mod 3) stays in vector registers (`IMAD R14, R13, 0x6000, R12`; `SEL R13`);
only the trip counter was promoted (`UIADD3 UR4`).  P4b (warp-uniform page index) still emits
`IMAD.WIDE.U32 R, R, UR, R` + `LDGSTS [R+imm], [R.64]` -- the a16 module's form is not "uniform-
register addresses", it is the same per-thread pair with a `UR` stride; its 27 cycles/LDGSTS come
from a shallow queue (no STS bursts in that warp), not from the address form.  Not a lever.

**D4. What is in the free pool at the burst's end** is decided by the register budget, not by
ptxas: with `PK.pk` 24 + outputs 48 + `sw`/`v` 12 + bases/protocol ~50 live, the only registers
that die inside a body are its store operands, so *whatever* ptxas writes next lands on them.
The F26 order (both operands' landing sets live across the K body) is where 136 bites -- the
dynamic module spilled (408/608 B) and the static ones sat at 130 live.  At 136 the producer
cannot hold half h's working set and half h+1's inputs; therefore each half issues its landing and
scale loads *into* the queue its own copies and the other half's stores just filled, and pays one
round trip per half.  That is the `prep + copies` segment: 1133 cycles for ~133 instructions per
pair (F27), 767 in F26 -- the four WAR PCs are where ptxas makes it visible.

**D5. Boring-by-construction design, scored against the probes.**
- Loop bounds and branch predicates broadcast once per item through `__shfl_sync` (or the loop
  exit voted) -> P15/P16: all `BRA.DIV`/`BSSY`/`WARPSYNC.COLLECTIVE` gone, ~80-100 cycles/pair
  (4 guards x (UMOV + BRA.DIV + inherited wait ~30) + 2 BSSY/BSYNC + branch_resolving samples).
  Honoured by construction (value-based analysis).  Same fix for the dynamic module's 13 `BSSY`:
  vote the trip predicates, do not `POPC` them.
- The K-half scale copy as an unconditional `LDGSTS` with src-size 0 for lanes 6-7 (already the
  `@!P2 .64` form in F26) instead of the `@P4 BRA` region: removes one `BSSY/BSYNC` and the
  filler split.  Honoured (P17 predication).
- Post-burst writers that are MIO-derived (D1): compute the *next* half's stage base, phase word
  and ring offset from values that are loaded (or re-loaded) after this half's stores -- e.g. the
  chunk-table row already read per pair -- rather than from loop-carried registers.  ptxas then
  emits no `rd` barrier on the body and the first scale/landing `LDS` issues without a wait.  Saves
  the issue slots, *not* the round trip (D1's price); estimate 60-120 cycles/pair.
- Hoisted loop-state (P7) is not robust at 136 (F26 evidence) -- do not rely on it.
- The 3.6 spurious `UMOV URZ` + `VIADD R, R, UR` (~13 per half on fp8 and fp4) are the only
  instruction-count lever left in the copy blocks: ~26 slots, ~40 cycles.
Sum of what is honoured by construction: ~200-260 issue cycles per pair on fp8.

## E. Verdict

fp8 is producer-paced at 2711 issue cycles/pair (373 us; consumer T_c ~ 2544 wall = ~2216 issue).
330 us needs ~2396: **-315**.  The constructions ptxas provably honours (D5) are worth ~200-260,
with the WAR stalls counted honestly as re-attributed queue latency, not as 410 removable cycles.
The remaining ~900 cycles of `prep + copies` are the one-round-trip-per-half structure forced by
136 registers (D4); the arms' 1222 cycles are 48 wavefronts x 8 of pipe per body plus dispatch
contention with the two consumer warps per SMSP.  Neither moves with allocation or ordering.  So:
fp8 at 330 on the producer-expansion architecture is *marginal at best* (a full D5 round lands
~340-350 by these numbers, inside the F26/F27 error band of what "flat" has meant twice); fp4
(2848, +20 us behind) and mixed (4647) are out of reach on it.

Role change: the consumer WGs issue at 44-50 % and have 184 registers; the 420 body instructions
x 4 producer warps = 210 per consumer warp per pair fit the issue headroom (round2 2A budgeted 427
on the 12-warp form; the 8-warp form halves it).  What does *not* move is the pipe: `wgmma` takes
B (K and V here) from smem only (round2 2B), so a consumer-side decode still writes 384 STS.128
wavefronts per pair into the same 74 %-utilised shared pipe -- but from warps whose issue is
otherwise waiting on tensor completion, and whose landing loads do not queue behind their own
copies.  Register-operand decode exists only for K via SWAP_AB (K as the A operand, softmax across
warps; 2B priced it at a new ~2100-instruction consumer skeleton) and never for V.  The honest
statement of the remaining path: (1) one cheap D5 round to bank the ~200 cycles and confirm the
`prep + copies` floor with the WARs gone; then (2) consumer-side decode of V at least (its half of
the STS moves to warps with slack, the producer drops to K only and fits its landing set under
136), gated on the consumer `%globaltimer` trace 2A asked for and never got.  Nothing else in the
producer's register schedule is left to control from source.

Probe sources and decoded SASS: `nkcut2:/tmp/frev/{probes,probes2,probes3,probes4}.cu`,
`probes*_O3.sass`, `probe_ctrl.py` (wraps `/tmp/mixedkv-wtF27-art/f27_ctrl.py`).
