# Mixed KV speed, round 4, Track F: lever [26] — the pair re-ordered around the copy issue (revision 2)

Design only (no kernel edits, no GPU timing).  Base: `claude/mixed-kv-sm90-tma`
@ `40dcdc32` (merge of wt/F25).  Extends `docs/mixed_kv_speed_round3_fa3_f25.md`
(F25 design, results in its section 11 and in
`docs/mixed_kv_page_transport_backends.md` "Track F [25]") and
`docs/mixed_kv_page_transport_dataflow.md` (D1-D6, C1-C17, A1-A9).  Every
measured number below is read from the F25 artefacts on
`nkcut2:/tmp/mixedkv-wtF25-art/` (`sf{1,2,-1}_mask1.sass`, `sf0_mask1.sass`,
`ncu_{fp8,fp4,mixed,a16}_source.csv`, `*_raw.csv`, `trace_q1.txt`,
`loopsite_*.log`, `counts.log`) with two reading scripts written for this
document and left next to them: `f26_seg.py` (SASS window by address) and
`f26_pc.py` (pc-sample aggregation of an ncu source CSV by address range,
segment and opcode class; the ncu listing is 7 instructions shorter than
cuobjdump's before the producer region, so its addresses are the cuobjdump
addresses minus 0x70).  One compile-only probe was built
(`f26_addr_forms.cu` -> `f26_addr_forms.sass`, `nvcc -arch=sm_90a -O3 -cubin`,
CUDA 13.0, no launch) to settle the copy-address form.  No timing of new code
was made.

## 0. Summary (revision 2)

Revision 2 answers the judge's four blockers on revision 1 (section 11) and
changes the design where they required it: the copy-block cost model is
re-derived from the a16 module's own loop-site samples (the fused address form
is *not* what makes a copy block cheap: the block's time is the LSU pipe's
share of its `LDGSTS` wavefronts, ~30 cycles per `LDGSTS` in the a16 module
and in F25 alike), the acquire spin now precedes the `wait_group` so that the
landing cover absorbs the consumer slack, the copy address form is prescribed
as the a16 module's *actual* construction (a `DTypeKV`-typed pointer add that
lowers to `LEA / LEA.HI.X`, which ptxas cannot reassociate into the page term)
with the compile-only probe named as the first remote action and the band no
longer hinging on the outcome, the scale-slot layout is made row-contiguous so
the lane-octet scale copy coalesces by A7's measured rule, and the test matrix
gains multi-chunk items stated in tiles (33 tiles: buffer 1 and the ring wrap;
22 tiles: buffer 1 without wrap) that are run on the F25 kernel first.

F25 left the compressed modes at fp8 402.8 / 414.9, fp4 422.2 / 429.7, mixed
650.0 / 664.4 us (q=1 / q=64) against transport_a16 282.8 / 289.5 and a target
of <= 330.  The per-segment pc samples (section 1) say: **the two decode
bodies (420 of the 716 instructions) run at IPC 0.52-0.59 and take 775 of the
pair's 3113 cycles; the other 296 instructions take 2340 cycles.**  Four
mechanisms, each visible at named PCs, account for that 2340:

1. the copy blocks (36 `LDGSTS` per pair: 24 payload `.128` + 12 four-lane
   `@P .64` scale copies, 120 wavefronts) cost 1160 cycles = **32 cycles per
   `LDGSTS`, 9.7 per wavefront** - the same per-`LDGSTS` cost as the a16
   module's zero-add loop copy blocks (751 cycles for 24 `LDGSTS.128`, 96
   wavefronts: 31 per `LDGSTS`, 7.8 per wavefront; sf0 `[0xd5e0,0xd8a0)` +
   `[0xdf00,0xe1c0)`, 29.7 % of the a16 loop-site samples).  The block's time
   is the LSU pipe's service of its wavefronts (75-79 % pipe utilisation per
   C11), not the address arithmetic; revision 1's "carry-predicate
   serialisation at 9 cycles per instruction" is withdrawn (section 11.2);
2. the acquire `TRYWAIT`s, the chunk-table `LDS.128`s and the chunk-store
   `BAR.SYNC` are waited on where they are issued (~250 cycles);
3. the pre-vote chain (scale `LDS.32` -> `PRMT` -> `F2FP` -> `HADD2` -> `FMUL`
   -> `FSETP` -> `PLOP3` -> `VOTE`) is exposed once per operand, and the
   landing loads of the pair's first operand sit in the MIO queue behind the
   `LDGSTS` just issued (fp8 ~380 cycles; fp4 ~560, which is the "expK 0.68
   vs expV 0.38" asymmetry);
4. the dynamic module's predicated copy body (492 instructions per pair, 71
   predicated `LDGSTS`, 72 chains) is dispatch-bound at IPC 0.16 (2900
   cycles) and its rolled format loops execute 640 instructions for 8 blocks.

**F26 = the same bodies, the same layout of the K / V stages, the same barrier
protocol, with the pair re-ordered so that every latency of (2) and (3) is
issued before the copy block and consumed after it (the acquire spin first,
then `wait_group 0`, then both pending operands' landing loads, one
`__syncwarp`, both operands' scale loads, the chunk rows, the scale chains,
the copies); eight fewer `LDGSTS` per pair (the six scale rows of an operand by
one lane-octet instruction into a row-contiguous slot layout: 36 -> 28 per
pair, 120 -> ~100 wavefronts); the copy addresses in the a16 module's
construction; the chunk-table row addressed as a 32-row ring; the static
modules' pending words removed; the acquires probed with a non-blocking
`test_wait` at the end of the previous pair and spun only if not ready; and
the dynamic module's copies and decode unrolled per page slot under
warp-uniform branches on ballot-derived masks.**  Per warp per pair the fp8
count goes 716 -> ~612, fp4 740 -> ~636, mixed 1211 -> ~800; the producer's
pair is predicted (section 4, issue-based cycles as section 1 counts them) at
**fp8 ~1985 centre / ~2640 pessimistic** = wall-equivalent 2280 / 3030 against
the consumer's T_c = 2544 (1.285 us): the centre is consumer-bound with 10 %
of margin, the pessimistic column paces at 336 us.  The accept (<= 330) is
therefore **not** "met with margin at both ends of the band" as revision 1
said; it is met at the centre and decided by one measured number, the
producer's pair in 6.3 (accept iff <= ~2590 issue cycles).

| mode (us, q=1 / q=64) | F25 @ 40dcdc32 | F26 rev 2 centre | F26 rev 2 band | accept <= 330 |
|---|---|---|---|---|
| stock_a16 | 300.9 / 310.9 | unchanged | 297-304 / 307-314 | control |
| transport_a16 | 282.8 / 289.5 | unchanged (module untouched) | 281-286 / 287-293 | control |
| fp8 static | 402.8 / 414.9 | **297 / 304** (consumer-bound, c ~5 %) | 289-336 / 296-344 | met at the centre; the pessimistic column (copy block 1100, half the latencies still exposed) misses by 6 us |
| fp4 static | 422.2 / 429.7 | **298 / 305** | 290-339 / 297-347 | as fp8 |
| mixed (dynamic) | 650.0 / 664.4 | **~305 / 313** (at parity: producer pair ~= T_c) | 295-357 / 303-365 | undecided: the centre passes with 15 % of pair margin, the pessimistic column (branches + copy body) misses; 6.3 decides |

**Is <= 330 reachable by F26 alone?**  For fp8 and fp4: yes if the pair lands
within ~30 % of the centre, i.e. if the copy block costs what the a16 module's
does (~850 cycles for 100 wavefronts) and the loads' latencies are hidden as
3.1 argues; the two things that would push it past 2590 are a copy block above
~1450 cycles (the pipe contention is worse than the a16 calibration) or the
landing cover failing (3.1: `wait` > 0.05 us - gated, with a priced
fallback).  For mixed: at parity at the centre; the per-slot branch count and
the copy body's issue rate (3.6) decide it, and the prediction is honest about
that: the mixed accept is not claimed.  The floor for every compressed mode is
transport_a16's 283 / 290 plus the shared-memory-pipe contention term c
(2-10 %, unmeasured; 6.3 bounds it): **~289-311 us at q=1**.  Nothing in F26
lowers T_c.

## 1. The F25 record re-read per segment (fp8 unless stated)

Per-pair cycles at 1.98 GHz: fp8 ncu 397.2 us / 220 pairs per SM = 1.805 us =
3574 cycles of wall per pair, of which the producer's loop site carries 45,158
pc samples over 91,392 executions (2.15 per 16 pairs); the trace's 2.2 us pair
and the ncu `selected` 22.7 % put the producer's own pair at **~3113 cycles**
(716 issue slots / 0.23).  One percent of loop-site samples = ~31 cycles.
Segments are the F25 SASS ranges of `sf1_mask1.sass` (cuobjdump addresses;
`f26_pc.py ncu_fp8_source.csv 0xe9c0 0x13a90 --dealloc 0xb3b0 --seg ...`):

| segment (fp8 loop site `[0xe9c0,0x13a90]`) | instr | executed / pair | samples | cycles | IPC | stall mix |
|---|---|---|---|---|---|---|
| A: loop/chunk index, gather (1/16), chunk store + `BAR.SYNC` (1/16), 2 acquires, K meta `LDS.128` x2, **K copies** | 138 | ~110 | 31.0 % | 965 | 0.11 | wait 23, long_sb 20, dispatch 19 |
| B: V meta `LDS.128` x2, **V copies**, pending words, state advance | 65 | 65 | 18.7 % | 582 | 0.11 | **dispatch 39**, wait 22, long_sb 12 |
| `LDGDEPBAR`, `DEPBAR.LE 1` | 2 | 2 | 0.4 % | 12 | | |
| K landing `LDS.64` x12 | 12 | 12 | 2.5 % | 78 | | wait 53, mio 16 |
| `NOP` (the `__syncwarp`) + first scale `PRMT` | 1 | 1 | **3.8 %** | 118 | | short_sb 96 (the scale `LDS.32` latency) |
| K vote prep (6 `LDS.32`, 6 chains, `PLOP3`, `VOTE`, `BRA`) | 44 | 44 | 5.6 % | 174 | 0.25 | wait 28, not_sel 24 |
| K cold arm (not executed) | 267 | 0 | 1.0 % | 32 | | no_inst 46 (fetch of the taken hot branch) |
| **K hot arm** | 210 | 210 | 12.7 % | **395** | **0.53** | selected 52, dispatch 15 |
| V prep (12 `LDS.64`, `NOP`, 6 `LDS.32`, chains, vote) | 59 | 59 | 8.8 % | 274 | 0.22 | wait 38, short_sb 17 |
| V cold arm (not executed) | 267 | 0 | 1.1 % | 34 | | no_inst |
| **V hot arm** | 210 | 210 | 12.2 % | **380** | **0.55** | selected 59, long_sb 10 |
| tail: 4 fillers, `MEMBAR`, `FENCE.VIEW.ASYNC`, 2 `SYNCS.ARRIVE`, loop | 14 | 14 | 2.2 % | 68 | | long_sb 68 (the fence drains the `STS`) |

Named PCs behind the mix (share of loop-site samples): `PRMT R27, R32, R7`
after the first scale `LDS.32` 3.76 % (short_sb); V meta `LDS.128` 2.35 %
(long_sb); the two acquire `@!P0 BRA` 2.21 + 1.92 % (long_sb: the `TRYWAIT`
result round trip, not a consumer wait - the producer is the pacing side);
`@!P1 BRA` around the chunk store 1.62 % (barrier 47 %, no_inst 21 %: the
`BAR.SYNC` every 16 pairs); K meta `LDS.128` -> `IMAD.WIDE` 1.61 % (short_sb);
`SYNCS.ARRIVE` 1.58 % + `FENCE` 1.47 % (long_sb); the copy adds: every
`IADD3 Rx, P0, ...` PC shows `dispatch` at 93-97 % of its samples
(`0xef50` 281/295, `0xf2d0` 269/285, `0xefb0` 208/227, `0xeed0` 205/221), the
`IMAD.X` PCs wait 50 % / dispatch 30 %, the `IMAD.WIDE.U32` PCs wait / dispatch
/ short_sb.  **Revision 1 read this as a carry-chain serialisation (P0 / P1 / P4 recycled
through the 12 chains, ~9 cycles per instruction); revision 2 withdraws that
reading (section 4, blocker 11.2).**  The a16 module's *loop* copy blocks
(sf0 `[0xd5e0,0xd8a0)` and `[0xdf00,0xe1c0)`, loop site `[0xcb00,0xeab0]`)
have no adds at all - `IMAD.WIDE.U32 R30, R4, UR8, R16` (page x uniform
stride + the per-thread 64-bit base as the accumulator) then `LDGSTS`, one
instruction per copy - and still take 29.7 % of the a16 loop-site samples
(31,943; ~750 cycles per pair for 24 `LDGSTS`: 31 cycles per `LDGSTS`, 7.8 per
wavefront; stall mix wait 27-30, dispatch 24-43, not_selected 13-22).  F25's
1160 cycles for 36 `LDGSTS` (120 wavefronts) is the same price per `LDGSTS`
(32) and per wavefront (9.7): **a copy block costs its `LDGSTS` count times
the LSU pipe's price, whichever address form feeds it**; the `dispatch`
stalls at the `IADD3 P0` PCs are the back-pressure of the `LDGSTS` stream
surfacing at the issue slot ahead of each copy (the a16 blocks' predicate-free
`VIADD` / `IMAD.MOV` show dispatch 30-40 % the same way).  The item-prologue
block `0xc6c0-0xc870` revision 1 cited is the K(last)-alone copy (exec 19,584
per run, once per item), same form.

**fp4 (`sf2_mask1.sass`, 48,181 samples; 1 % = ~33 cycles).**  Same shape with
one difference: K prep `[0xf6d0,0xfaf0)` (12 landing `LDS` + 6 scale `LDS` +
chains) is **17.7 %** of the samples (short_sb 42, wait 20, long_sb 16) against
V prep's 6.9 %, and the K hot arm 13.6 % against V hot 9.8 %.  Top PCs:
`VIADD R21, R78, 0x1` at `0xf890` 2.45 % short_sb - a WAR on R21, the address
register of the six `LDS [R21+...]` landing loads issued just before it: the
`VIADD` cannot retire the register until the MIO has read it, and the MIO is
processing the pair's 24 `LDGSTS.64` (`.ca`, 3.98 wavefronts each = ~96
wavefronts) that were issued immediately ahead of the loads; `LDS R40,
[R20+0x1000]` 2.04 % long_sb (the `DEPBAR` completing behind the same queue);
`SHF.R.U32.HI R109, RZ, 0x1e, R107` 1.47 % short_sb (V prep's pending-stage
extraction behind the K body's `STS`).  **The asymmetry is MIO queue order:
the K operand's 18 loads are issued directly behind the pair's `LDGSTS` burst
(fp4: 24 x 8 B `.ca`, two passes through L1TEX), the V operand's loads are
issued after the K body when the queue has drained.**  fp8's K loads sit
behind 12 x `.128 .BYPASS` + 12 x `.64` and pay less (the `PRMT` 3.8 % and the
landing `LDS.64` 2.5 % are its share).  The 0.3 us trace difference (expK
0.66-0.71 vs expV 0.37-0.39) is ~600 cycles = the fp4 K-prep and K-hot excess
(17.7 + 13.6 - 6.9 - 9.8 = 14.6 % = ~480 cycles) plus the trace stamp's own
position (the `wait` stamp reads 0.01 us in both modes because ptxas schedules
the `S2R` before the `DEPBAR` completes; the landing wait is booked to expK).

**Dynamic (`sf-1_mask1.sass`, loop site 2391 instructions, 74,550 samples; the
dyn pair is ~6300 cycles, 1 % = ~63 cycles).**  Segments: prologue 3.9 %
(no_inst 45); **copy body `[0x13d90,0x15c50)` 492 instructions, all executed
(predicated), 48.7 % = ~2900 cycles, dispatch 45 %, IPC 0.16** (72
`IMAD.WIDE.U32`, 46 `IMAD.X`, 53 `IADD3`, 71 `VIADD`, 26 + 24 `ISETP`, 40
`SEL`, 71 predicated `LDGSTS`; the `ISETP` / `LOP3 P` predicate producers are
dispatch-stalled at 95 % - `0x14710` 580/588, `0x147b0` 533/551 - the same
predicate-port serialisation as the static carry chains, now with 5-6
predicate producers per page); DEPBAR + vote prep 88 instructions 7.8 %; K
format loops 893 instructions of code, 355 executed per pair, 21.3 % (wait 31,
short_sb 18); V format loops 824 / 286 executed, 17.3 % (no_inst 22).  Per pair
the loops execute 640 instructions for 8 blocks = 80 per block against the
static hot body's 35 + 8 (`FLO`/`BREV`/`SEL` page selection 5 per step, the
per-step scale-word reloads, two `__syncwarp` `NOP`s, the loop branch); the
copy body executes 492 instructions to issue 12 copies.  `no_inst` 10.4 % of
the region.  The F25 model (~790) missed because it counted the copy body at
~180 (the predicated-off instructions and the predicate producers were not
counted) and the loops at 40 per block.

**Shared-memory stores (item 5 of the brief).**  The fp8 hot arms' 24
`STS.128` PCs each show `L1 Wavefronts Shared` 365,568 = 4.00 per instruction =
ideal, N-way 4.00, excess 0 (`f26_pc.py ... --sts`); the two exact-arm sites
are the same form.  Summing the source view: `STS.128` 8,982,528 wavefronts
(375.3 per pair, all ideal), `STS`/`.U8`/`.U16` of the chunk store 19,584.  The
raw `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum` is 12,259,600
(512 per pair), so 136 wavefronts per pair of `op_st` traffic come from PCs
that are not `STS` instructions (fp4: 9,762,513 - 8,982,528 = 32 per pair;
dyn 12; a16 12 with no `STS.128` at all: the `SYNCS.ARRIVE` traffic).  The
"5.41 wavefronts per `STS.128`" of the F25 record was `op_st.sum / STS.128
count`; **the store pattern is conflict-free at every `STS.128` PC and no store
order or layout change is warranted** (do-not-build item 6).  The fp8-only
residual (124 per pair above fp4's) is unattributed and does not sit on the
producer's issue path; it is listed as an ncu follow-up (6.3).

**Consumer.**  K-wait `@P0 BRA` 10.2 / 18.0 / 37.3 % (fp8 / fp4 / dyn), 2.5 %
a16; consumer `inst_executed` 3301 per pair in every module (184 registers
cost nothing).  T_c = 282.8 us / 220 = 1.285 us = **2544 cycles**.

## 2. What F26 changes and what it keeps

Kept verbatim from F25: the 12-warp layout and 136 / 184 registers; A9
ownership (thread t = block t % 8 of row t / 8 of all six pages; lane 0 copies
the row's 8 B scale slot - amended to the octet gather in 3.3); the E4M3 /
E2M1 placement decodes and the per-operand fold votes (C9, C12, C16); the
two-arm (hot / cold) operand bodies of the static modules; the peel, the
unconditional loop finish and the two single-operand sites (C13, C14); the
chunk table and its 16-pair gather / store cadence; the a16 module's text
(`STATIC_A16` arms untouched; its SASS stays the F25 stream); the consumer,
epilogue, scheduler, `named_barrier.cuh`, `sparse_mainloop.cuh`,
`mainloop_mma.cuh`; `kernel_traits.cuh` and the `prefill_sm90.cuh` hook.

Changed (all inside `sparse_mixed_mainloop.cuh`; the host gains one
even-stride check in `to_underlying_arguments`, no `Params` change):

1. **Pair order** (3.1): the acquire spins first, then `wait_group 0`, both
   pending operands' landing loads, one `__syncwarp`, both operands' scale
   loads, the chunk-table rows and the scale chains before the copy block;
   the votes and bodies follow it; the next pair's acquires are probed
   (non-blocking) after the bodies.
2. **Copy address form** (3.2): the a16 module's actual construction - a
   `DTypeKV`-typed element-offset base built once per item in `make_bases`
   (the `LEA / LEA.HI.X` pair ptxas keeps as the `IMAD.WIDE.U32` accumulator),
   `page x stride` as C++ arithmetic.  Unprobed; the form is reported by the
   SASS gate, and the band does not hinge on it (section 4).
3. **Scale rows by one lane-octet instruction** (3.3): lane b < 6 of each row
   octet copies page b's 8 B scale row (one `LDS.32` of the row's page index,
   one `IMAD.WIDE`, one `@P LDGSTS.64` per operand) into a row-contiguous
   slot layout (`r x 48 + b x 8`) so the octet coalesces: 36 -> 28 `LDGSTS`
   and 120 -> ~100 wavefronts per pair.
4. **Protocol** (3.4): 32-row ring addressing of the chunk table (no `/ 16`,
   `% 16`), one pipeline counter for both rings, no pending words in the
   static modules (the pending stages are functions of the counter),
   acquires probed with a non-blocking `test_wait` at the end of the previous
   pair and spun only if not ready (before the `wait_group`), one `__syncwarp`
   per pair.
5. **fp4 asymmetry** (3.5): removed by (1); no fp4-specific code.
6. **Dynamic module** (3.6): copies per page slot under a warp-uniform format
   branch (ballot-derived masks), decode per page slot under the same
   branches with one body per format whose exact (cold) form is 8 predicated
   `HMUL2` and a selected scale; no rolled loops, no `__ffs`.

## 3. The design

### 3.1 Data flow of one loop pair (static modules; dynamic in 3.6) - revision 2 order

Pair t of the loop issues (K(t-1), V(t)) into stages (sK, sV) and finishes the
pair issued one iteration earlier, which sits in stages (sK', sV') = (sV,
sV - 1 mod 3) (3.4).  Program order per thread:

```
[top of pair t]
[if !okK spin K] [if !okV spin V]   tokens from the two test_wait probes issued at the end of pair t-1 (or by the peel);
                                    the spin (mbarrier.try_wait loop) is where the consumer's slack is absorbed
cp.async.wait_group 0               DEPBAR.LE SB0, 0x0: this thread's copies of the pending pair (committed mid pair t-1) have landed;
                                    cover = K'/V' bodies + tail + spin of pair t-1 (see "Landing cover" below)
K' landing loads: 6 x 2 LDS.64      own landings of the pending K (fp4: 12 LDS.32)
V' landing loads: 6 x 2 LDS.64      own landings of the pending V
__syncwarp                          the pair's ONE warp barrier: every lane is past its wait, so (a) the six landed scale
                                    slots of each operand (lane b's copies, 3.3) are readable by the row and (b) every lane's
                                    landing loads of both operands are ordered before any lane's stores below (A7)
scale loads: 6 LDS.32 (K'), 6 LDS.32 (V')      immediate offsets from sc_rd + stage x SCALE_STAGE (row-contiguous slots, 3.3)
chunk rows: 4 LDS.128 + 2 LDS.32    pages of tiles t-1 (K) and t (V) from the 32-row ring (3.4); the two LDS.32 are the
                                    lane-octet page words of the scale copies (hoisted here so one LDGSTS group follows one LDS group)
scale chains (both operands): 12 x {PRMT, F2FP.E4M3, HADD2, FMUL, FSETP}, PLOP3 trees     (ptxas interleaves them with the copies)
K copies: 6 x {IMAD.WIDE.U32 src.64 = page_j * UR ps + pbase.64 ; LDGSTS.128 [land8 + sK*STAGE + j*PAGE], [src.64], 16}
          1 x {IMAD.WIDE.U32 ; @(b<6) LDGSTS.64 [sc_cp + sK*SCALE_STAGE], [ssrc.64], 8}                          (3.3)
V copies: the same 6 + 1 into sV
cp.async.commit_group               LDGDEPBAR
K' vote: VOTE.ALL over the 6 FSETPs ; uniform BRA (hot / cold arm)
K' body: 6 x {F2FP.PACK sf2 ; 8 PRMT, 8 SHL, 8 LOP3 ; 8 HMUL2 ; 2 STS.128}
V' vote ; V' body
okK = test_wait(empty_k[sK(t+1)], phK(t+1)) ; okV = test_wait(empty_v[sV(t+1)], phV(t+1))   non-blocking probes (mbarrier.test_wait)
fence.proxy.async ; arrive full[sK'] ; arrive full[sV']
pair counter advance ; ring offset advance ; loop
```

Why this order and not F25's (copies -> `LDGDEPBAR` -> `DEPBAR.LE 1` -> loads
-> `__syncwarp` -> scale loads -> vote), and not revision 1's (`DEPBAR` at
the very top, acquires as `try_wait` after the loads):

- Every load whose latency F25 exposed (scale `LDS.32` -> vote chain, landing
  `LDS.64`, chunk-row `LDS.128`, acquire round trip) is issued before the copy
  block and consumed after it.  The copy block is 28 `IMAD.WIDE.U32` + 28
  `LDGSTS` whose issue takes ~850 cycles (section 4: the LSU pipe's service of
  ~100 wavefronts) - far more than the ~30 (`LDS`), ~70 (scale chain to
  `VOTE`) and ~120 (barrier probe round trip) it has to hide.
- Both pending operands' landing loads are issued while the MIO queue holds
  no `LDGSTS` (the previous pair's completed at the `DEPBAR`; this pair's are
  not yet issued): that removes the fp4 K-operand excess (3.5) for K' and
  never creates it for V'.  Revision 1 issued the V' loads inside the K' body
  (after block 3) behind the 28 `LDGSTS` just issued, and needed a second
  `__syncwarp` after the K' vote's branch join - the placement F24 measured as
  `BRA.DIV` guards; both are gone.  Cost: the 24 V' packed words are live
  across the copy block and the K' body (+24 registers, 3.10).
- **The spin precedes the `wait_group`** (revision 1 had them the other way
  round).  With the acquire spin *after* the `DEPBAR`, the cover for the
  pending group would have been bodies + tail only, ~1100-1300 cycles
  regardless of the consumer's pace (the slack was absorbed after the wait).
  With the spin first, the slack is absorbed before the wait.
- The acquire probes are `mbarrier.test_wait` (non-blocking by PTX semantics),
  not `try_wait` (which may suspend the thread up to a system-dependent
  limit): a blocking probe issued before this pair's commits would hold the
  pending pair's `full` arrives hostage to the consumer's release of a stage
  two pairs older and open a bubble on the consumer's side.  `PipelineAsync`
  does not expose a producer `test_wait`, so the probe reads the pipeline's
  `empty_barrier_[stage].test_wait(phase)` from `SharedStorage::pipeline_k /
  pipeline_v` directly (the same barrier and phase `producer_acquire` waits
  on); the spin is `producer_acquire(PipelineState(stage, phase, 0))`.  The
  probes sit after the V' body so that their ~120-cycle round trip overlaps
  the fence (long_sb 68 % of its samples: it drains the `STS`), the two
  arrives and the loop tail.
- The brief's variant (`__syncwarp` -> 6 scale `LDS.32` -> 12 landing `LDS.64`
  -> vote, F25 order otherwise) hides the scale chain under 12 issue slots
  (~24-30 cycles of ~70) and keeps the loads behind the `LDGSTS` burst.

**Landing cover (blocker 11.3).**  The group committed at the `LDGDEPBAR` of
pair t-1 is waited at the `DEPBAR` of pair t, after the spin.  Cover =
(K' vote + body + V' vote + body + probes + fence + arrives + loop) of pair
t-1 + spin at the top of pair t = **the pair's wall time minus the part of
the pair that precedes the `LDGDEPBAR`** (wait + loads + chains + copy issue
~= 950 cycles at the centre of section 4):

- consumer-bound (the predicted regime): the pair's wall time is T_c = 2544,
  cover ~= **1600 cycles = 0.81 us**, of which the spin is the slack (~250 at
  the fp8 centre);
- producer-paced: the pair is P > 2544 cycles, cover = P - 950 >= 1600.

F25's cover was one full pair (its `DEPBAR.LE 1` followed the copies): 3113
cycles at F25's pace, and its traced `wait` was 0.01-0.02 us; F24's 0.03 us
was at a 2.6 us pair.  No artefact measures the loaded landing latency at a
~0.8 us cover, and F25 design 3.3 puts the loaded HBM / L2 latency at
1200-2000 cycles: the F26 cover sits inside that range, i.e. **between 0 and
~400 cycles per pair may be exposed at the `DEPBAR`**, on the producer's
critical path.  This is stated, not argued away; it is handled as a hard gate
with a priced fallback:

- gate 6.2: `wait` (the stamp across the `DEPBAR`, after the spin) <= 0.05 us
  per pair and the first post-`DEPBAR` PC's long_sb <= 2 % of loop-site
  samples; the trace is read **before** any bench;
- fallback F26b' if it fails: the `DEPBAR` goes back to F25's position
  (`wait_group 1` after this pair's `LDGDEPBAR`, cover = one full pair) and
  the pending pair's loads follow it (F25's load order), everything else of
  F26 kept (a16 address form, octet scale copies, ring, counter, early probe
  / late spin, one `__syncwarp`).  Priced in section 4 as its own column:
  mechanism (3) returns (~380 fp8 / ~560 fp4 cycles), the fp8 centre moves
  to ~2265 issue cycles (wall-equivalent ~2600, at parity, ~300-305 us), fp4
  to ~2445 (~312 us).  It is a different design with a different band, and
  the band is written down here so that choosing it is not tuning.

The single-operand sites (K(last) after its issue, the drain V) keep F25's
form: `wait_group 0`, 12 landing loads, `__syncwarp`, 6 scale loads, exact
body, fence, arrive; the peeled pair issues copies only and performs the
blocking acquires of F25 (two per item).

### 3.2 Copy address form (item 2 of the brief) - probe results, the a16 mechanism, the prescription

What the hardware offers: `IMAD.WIDE.U32 Rd.64, Ra, URb, Rc.64` - a 32 x 32 ->
64 multiply-add with a 64-bit register-pair accumulator and a uniform-register
multiplier.  The a16 module's loop copies use exactly this form (`sf0_mask1.sass`
loop site `[0xcb00,0xeab0]`, copy blocks `[0xd5e0,0xd8a0)` and `[0xdf00,0xe1c0)`:
12 x `IMAD.WIDE.U32 R30, R4, UR8, R16` + `LDGSTS ... [R30.64]` each, zero adds;
the item-prologue block `0xc6c0-0xc870` revision 1 cited is the K(last)-alone
copy, exec 19,584 per run = once per item, same form).  The compressed
modules' `page_src` (`mad.wide.u32 d, page, stride, base64` in PTX) compiles to
`IMAD.WIDE.U32 Rd, Rpage, UR, RZ` + `IADD3 lo, P, lo, base_lo` + `IADD3.X`/`IMAD.X`.

Probe (`f26_addr_forms.cu`, fifteen kernels, compile-only; adds per copy in
the copy block of each):

| kernel | base form | per copy |
|---|---|---|
| kA | today's: two chained asm `mad.wide.u32` from `ptr + b*16` | `IMAD.WIDE.U32 ..., RZ` + `IADD3` + `IADD3.X` (3) |
| kB | 32-bit page offset (`page * stride` u32) + 64-bit add | `IMAD` + `IADD3` + `IMAD.X` (3) - the carry predicate stays |
| kC | shfl-broadcast page term + per-thread 32-bit offset | `IMAD.WIDE.U32 Rd, Rpage, UR, R12.64` fused + 2 adds for the param pointer (3) |
| kD | scale-row copy by one lane-octet (3.3) | 1 `LDS.32` + 1 chain + 1 `@P LDGSTS.64` for six rows (the chain itself as kA) |
| kJ | uniform 64-bit pointer + per-thread 32-bit offset | fused + `IADD3` + `IADD3.X`: `LDGSTS` has no `[R.U32 + UR.64]` global form |
| kM | base re-formed per pair by asm `mad.wide.u32 (rt, 1, hp)` | 3 (the asm result is split) |
| kY | base made opaque through `st.shared.v2.b64` / `ld.shared.v2.b64` | `IMAD.WIDE.U32 ..., RZ` + 2 adds although the base is an adjacent pair |
| kT, kX, kZ | `uint8_t const*` base, `ptr + head*hs + row*ts + blk*16` (kX / kZ int64 strides, kZ under pressure) | fused with **one** loop-invariant 64-bit term, then `IADD3` + `IADD3.X` of a **second** loop-invariant term (3): 36 `IADD3(P)` + 36 `IADD3.X` per 36 `LDGSTS` |
| kF, kR, kW ... | other orderings / pressure variants | >= 1 carry add per copy (kR: 60 `IADD3` per 30 `LDGSTS`) |

**Reading (revision 2, blocker 11.4).**  Not one of the fifteen kernels
reproduces the a16 module's zero-add loop; revision 1's prescription (kT / kX
/ kZ: pointer-typed base, int64 strides, C++ `page * stride`) is exactly the
form that emits 3 instructions per copy.  What the a16 module does that no
probe did: its base pair `R16:R17` is formed **once, by `LEA` / `LEA.HI.X`**
(sf0 `0xc130` / `0xc180`) - the x2 scaling of the `DTypeKV const*` element
arithmetic in `make_bases` (`base + int64_t(a_r) * ts + a_c * CHUNK_ELEMS` on a
2-byte pointer lowers to `mul.lo.s64` + a shift-add) - and a shifted pair is a
term ptxas cannot reassociate into the per-copy multiply-add tree, so the
whole pair stays the `IMAD.WIDE.U32` accumulator.  A byte-typed pointer sum of
two 64-bit products is a tree ptxas *can* split, and it keeps two terms.

**Prescription (F26a): build the compressed bases exactly as the a16 base is
built.**  `compressed_base(ptr, strides, head, row, blk_off)` casts the span
pointer to `DTypeKV const*` and adds *element* offsets - `int64_t(head) *
int64_t(hs / 2) + int64_t(row) * int64_t(ts / 2) + blk_off / 2` - then casts
back to `uint8_t const*`; every byte term is even by the host's alignment
checks (payload rows block-aligned 16 / 8, scale rows 8 B; `to_underlying_
arguments` adds the explicit even-stride check), so the halving is exact.
`page_src(base, page, ps)` is `base + uint64_t(page) * uint64_t(ps)` in C++
(the a16 `copy_a16_page` line, byte stride in the `UR`).  No asm, no
`Params` change (the strides stay `KVPageByteStrides` u32; the widening is
device-side, once per item).  The bases are members of `OperandBases` as
before (`uint8_t const*`), built in `make_bases` after the chunk-0 group
barrier as the a16 base is, and shared by the FULL loop arm and the two
partial-arm calls.

**This is unprobed** (no compiler on the design host; revision 2 was written
without a build).  Two consequences are drawn instead of an assumption:

1. **The compile-only probe is the first remote action of F26d**, before the
   F26a build is read: kernel kV in `f26_addr_forms.cu` = the a16 context
   replicated (a `DTypeKV`-typed element-offset base built in a separate
   function, carried across a `bar.sync`, used by a FULL arm and a partial
   arm with the a16 `v ? s : base` 64-bit select), plus kV2 = the same with
   the base halved-byte-typed as prescribed above.  Expected: `LEA` /
   `LEA.HI.X` once, 12 x `IMAD.WIDE.U32 ..., R.64` + `LDGSTS`, zero adds.  If
   kV fuses and kV2 does not, the F26a text takes kV's exact construction
   (`DTypeKV const*` members, cast at the copy) before the build is read.
2. **The count and the cost are stated for both outcomes**, and section 4
   shows the band does not hinge on it: a fused copy block is 28
   `IMAD.WIDE.U32` + 28 `LDGSTS`; the 3-instruction form is 28 + 56 + 28 =
   112 issue slots.  The block's cost is the LSU pipe's service of its ~100
   wavefronts (~850 cycles, section 4), during which the ALU is idle for
   most of the time: 56 extra `IADD3` / `IADD3.X` fill issue gaps and add
   ~60-150 cycles, which is inside the pessimistic column.  Gate 6.1 then
   reports the form as a fact of the build, not as an accept / reject.

The predicate-port attribution of revision 1 (the 93-97 % `dispatch` at the
`IADD3 P0` PCs read as a P0 WAR chain) is withdrawn: the a16 module's
predicate-free `IMAD.MOV` / `VIADD` in its copy blocks show `dispatch` 30-40 %
too, in-order issue does not stall an ALU op on a predicate WAR, and the
`dispatch` stalls of both kernels sit at the instruction ahead of each
`LDGSTS` - the LSU / MIO back-pressure of the `LDGSTS` stream surfacing at
the issue slot before it.  Fewer address instructions do not remove it;
fewer `LDGSTS` (3.3) do.

The 32-bit page-offset form of the brief is **not** a fallback: it keeps the
carry predicate (kB) and adds a host constraint (page x stride < 2^32 per
span) for nothing.  Revision 1's fallbacks (i) "try the two orders" and (ii)
"struct member of pointer type" are withdrawn as guess-and-check; (iii)
"interleave the chains in groups" is a ptxas scheduling outcome the source
does not control and is withdrawn too.

### 3.3 Scale rows by one lane-octet instruction into a row-contiguous slot layout

Today each of the six pages costs one address chain and one `@leader
LDGSTS.64` (lanes b == 0 of the four rows of a warp: 4 active lanes writing 32
contiguous bytes, 1.98 wavefronts per instruction measured).  F26: lane b < 6
of every row octet copies **page b's** scale row of its row r: source =
`sbase.64 + page_b x UR sts` where `page_b` is read by that lane directly from
the tile's chunk-table row (`LDS.32 [row + 4 b]`: 6 distinct words per octet,
broadcast within, 1 wavefront; lanes 6, 7 read `w6` / `w7` and are predicated
off), destination `sc_cp + stage x SCALE_STAGE` with `sc_cp` a per-thread
constant.  Per operand: 1 `LDS.32` (hoisted next to the chunk rows), 1
`IMAD.WIDE.U32`, 1 `@P LDGSTS.64` (24 active lanes) instead of 6 chains + 6
`LDGSTS` (F25: 6 x (3 + 1) = 24 issue slots and 6 `LDGSTS`; F26: 3 and 1).

**Slot layout (revision 2; the judge's note on A7's rule).**  With F25's slot
layout (page j's 16 rows x 8 B at `j x 512`), the six destinations of an octet
lie in six different 128 B lines, and A7's measured rule - an octet coalesces
only when its destinations lie in one 128 B line, otherwise one wavefront per
lane - prices the instruction at 24 wavefronts, 48 per pair against F25's 24:
the instruction saving would have bought a wavefront loss.  F26 therefore lays
the scale slots out **row-contiguously**: row r's six 8 B slots at `r x 48 +
j x 8` (768 B of the 3 KB stage used; `SCALE_ROW_STRIDE = 48`,
`SCALE_SLOT_BYTES = 8`).  The octet's six destinations are then 48 contiguous
bytes (one line for 6 of every 8 rows, a line pair for rows r mod 8 in {2, 5}):
~2 wavefronts per instruction by analogy with the 32-contiguous-byte form's
1.98, i.e. **~4 per pair (from 24)**; 6.3 reports the measured value.  Readers:
lane (r, k) reads word `r x 48 + j x 8 + 4 (k >> 2)` for page j - the same six
`LDS.32` at immediate offsets (`j x 8` instead of `j x 512`); the warp's four
rows hit bank words `12 r + 2 j + (k >> 2)` = offsets {0, 12, 24, 4} + 2 j +
h mod 32, eight distinct banks per instruction (conflict-free, 1 wavefront,
as today).  The shared-storage size and the a16 module are untouched
(`kMixedScaleStageBytes` unchanged; the a16 module never touches the scale
arrays).

A9 ordering is unchanged in kind: every lane waits for its own groups, the
pair's `__syncwarp` orders lane b's landed slot of page b before the row's
eight lanes read it (F25 relied on the same barrier for lane 0's slot); the
D4 partial arm predicate becomes per lane `16 b + r < valid` (page b's row r)
carried as the src-size operand of the `@(b < 6)` copy.  kD shows the form
compiles as written (`LDS R9, [R13+UR6]` -> `@!P1 IMAD.WIDE.U32` -> `@!P1
LDGSTS.E.LTC128B.64`).  In the dynamic module lane b's page may be FP8, FP4 or
A16: two chains (`s8`, `s4` bases), two predicated `LDGSTS.64` (`@(p8_b & b<6)`,
`@(p4_b & b<6)`), predicates from the row's mask word read by the same lane
(`LDS.32 [row + 28]` broadcast, per-lane shift) - 8 instructions per operand.

### 3.4 Protocol (C14 restated) - back to <= 70 with the items named

F25's loop-site remainder was ~99 executed per pair (716 - 104 prep - 420 hot
- 93 copies); the F25 record's "~143" counted the 1/16-amortised gather /
store text as executed.  F26 items, per warp per pair:

| item | F25 (SASS) | F26 | how |
|---|---|---|---|
| chunk-table addressing | `entry / 16`, `entry % 16` three times (`SHF.R.S32.HI`, `LEA.HI`, `LOP3`, `SHF`, `IMAD.IADD` chains: ~25) + 2 x 2 `LDS.128` | ring offset `oV = (oV + 32) & 0x3FF` (2), K row = `(oV + 32) & 0x3FF` (2), 4 `LDS.128 [R + UR + imm]` + 2 `LDS.32` (octet page words) -> **10** | entry e of chunk c lives in buffer c & 1, slot e % 16, i.e. at row `e & 31` of the 1 KB table (`TileMeta` is 32 B, `meta[2][16]` contiguous: `read_meta`'s `meta[(e/16)&1][e%16]` is row `e & 31` for every e; the a16 module keeps the F25 text and its immediates); `j == 0` is `(oV & 0x1E0) == 0`, `j == 8` is `== 0x100`; `next_chunk` is a countdown of chunks left, decremented when `j` wraps to 0 |
| acquires | 2 x (`LEA`, `SHF`, `TRYWAIT`, `@!P BRA` to the spin) = ~10, latency exposed | two non-blocking `mbarrier.test_wait` probes (`SYNCS.PHASECHK...`) issued after the V' body of the previous pair, their predicates tested at the top of the pair (`@!P BRA spin`) -> **8**, latency hidden under fence + arrives + loop | 3.1: `test_wait`, not `try_wait` (a blocking probe before the commits would hold the arrives hostage); the spin is `PipelineAsync::producer_acquire(PipelineState(stage, phase, 0))` = `mbarrier.try_wait` loop, the same wait F25 performed |
| pipeline state | two `PipelineState` advances (`VIADD`, `ISETP`, `SEL`, `SEL`, `LOP3` x 2 = 10) | one counter (sV, phase_V): `VIADD`, `ISETP`, `SEL`, `LOP3`; `sK = sV + 1 mod 3` (`SEL`), `phase_K = sV == 2 ? phase_V ^ 1 : phase_V` (`SEL`) -> **6** | K's ring is exactly one stage ahead of V's for the whole loop (K(last) alone advanced it once; the last pair, tK = -1, advances V alone and restores sK == sV, phase_K == phase_V for load_tail and the next item); the two `PipelineState` objects are written back from the counter at the item's end |
| pending words | `LOP3 0x3FFFFFF`, `IMAD.SHL 0x40000000`, `LOP3` merge per operand + `SHF.R.U32.HI` extraction x 2 = ~8 | static: none - pending K stage = sV, pending V stage = sV - 1 mod 3 (one `SEL`) -> **1**; dynamic keeps F25's words (masks) -> 8 | the static finish needs only the stage; `TileRegs::pending_word` stays for the dynamic module |
| expand bases | 2 x (`IMAD` x 3 stage bases, `LOP3` x 2 halves) = ~12 | same **10** | |
| copy destination bases | 2 x 2 `IMAD` | **4** | |
| `LDGDEPBAR`, `DEPBAR`, `__syncwarp` (`NOP`) | 3 | **3** (one barrier per pair, 3.1) | |
| fence + commits | `MEMBAR.ALL.CTA` + `FENCE.VIEW.ASYNC` + 2 `LEA` + 2 `SYNCS.ARRIVE` = 6 | **6** (unchanged) | |
| loop, chunk gather / store amortised | ~10 + 35/16 | `t` decrement, compare, `BRA`, two `j` tests, countdown = 7 + 3 -> **10** | |
| `@!PT LDS RZ, [RZ]` fillers | 6 (two `LDGSTS` groups per pair) | **3** (one group: all `LDS` of the pair, including the octet page words, precede the first `LDGSTS`) | ptxas emits three predicated-off `LDS` before the first `LDGSTS` that follows an `LDS`; they occupy issue slots and count in `inst_executed` |
| **total** | ~99 executed (143 in text) | **~61** | budget <= 70 |

The chunk store and its `BAR.SYNC` (every 16 pairs, ~50 cycles per pair
amortised, `barrier` stalls 47 % at its `@!P1 BRA`) are kept; moving the store
to the pair after the K body would hide it but changes the [21] WAR argument
(`CHUNK_STORE_PAIR`) and is not needed for the budget.  The ring changes only
*where within a pair* the rows are read; the WAR argument (`CHUNK_STORE_PAIR =
8 > NUM_STAGES`) is unchanged.

**Barrier protocol (C4, C7, A5) unchanged**: `PipelineAsync` with producer
arrival count 128 / consumer 256, `full_barrier` arrive per thread after the
fence, K(last) finished before `barrier_O.wait`, `kQueryEmpty` 384, chunk-table
group barrier 128.  What moves is only *when* the probe is issued and *where*
the `wait_group` sits; the set of waits per pair is the same (one
`wait_group`, two acquires, two arrives, one fence).  cp.async group
accounting: `wait_group 0` at the top of pair t waits exactly the set F25's
`wait_group 1` after the copies waited (only the previous pair's group is
outstanding; this pair's is not yet committed); K(last) (`finish_one`,
`wait_group 0`, before `barrier_O.wait`) and the drain keep their form.

### 3.5 fp4 expK / expV (item 4 of the brief)

Attribution in section 1: MIO queue order plus a WAR on the landing-load
address register.  F26 issues both pending operands' landing loads (fp4: 24
`LDS.32`) and both operands' scale words before this pair's `LDGSTS`, right
after the `DEPBAR` that completed the previous pair's, so no load of the pair
waits behind a `LDGSTS` burst (re-checked for the revision 2 order: the
`DEPBAR` precedes the loads and the copies follow them, in both regimes of
3.1).  Predicted: fp4 expK = expV = 0.37-0.39 us on the trace, fin 0.76-0.80
(from 1.14-1.20); the fp4 pair loses ~560 cycles by this alone.  No
fp4-specific code; the fp4 landing (8 B `.ca` copies, `LDS.32` halves) is A7's,
unchanged - a 16 B fp4 copy form remains impossible (8 B alignment, F25 3.4).

### 3.6 Dynamic module (item 6 of the brief)

Count model re-derived from the loop-site SASS (section 1): copies 492
executed per pair (not ~180), loops 640 executed for 8 blocks (not 320 + 140),
prep 2 x 88, protocol ~100 = ~1400 modelled vs 1211 measured (the predicated
`LDGSTS` of the format that is not present issue but the ncu
`inst_executed` counts them; the loops' odd-page duplicate decode does not
happen on the bench).  Both F25 follow-up options were priced:

- *One loop body per format with the hot / exact choice predicated* (F25
  follow-up 3): halves the loop code (4 loops instead of 8: 5528 -> ~4500
  instructions) but keeps the copy body (492) and the per-block loop overhead
  (~30 of the 80 per block: `FLO`, `BREV`, `SEL`, reloads, back-edge, two
  `NOP`s): executed ~1050 per pair, still 1.7x the static.
- *Sorted-page table with unrolled per-class bodies* (F25 6.4): a tile's
  pages cannot be permuted (bit-exact accumulation order, causal masks), so
  "sorted" can only mean bodies per (n8, n4) class with the slot order kept:
  28 classes x 2 formats' bodies, or a class dispatch that is itself a branch
  tree; the copy body is unchanged.  Rejected on code size.

**Chosen: per-slot unrolled bodies with the page format as a warp-uniform
branch.**  The chunk-table row's mask word (`m8 | m4 << 8 | valid << 16 |
flags << 24`, F24c) is read by every lane; `m8u = __ballot_sync(~0, (w7 >> lane)
& 1) & 0x3F` and `m4u` likewise (two `VOTE.ANY`) give nvcc / ptxas masks it
knows are warp-uniform, so the per-slot tests `(m8u >> j) & 1` compile to
uniform branches without `BSSY` / `BSYNC` (probe kW: 6 slots x 3 forms, `BSSY`
0, 13 `BRA`, ptxas turned the smallest blocks into predication).  **At the
finish site the masks are re-balloted** (revision 2): the pending word is
carried one pair in a register and is LDS-derived per lane, so without two
more `VOTE.ANY` per operand ptxas would not know the per-slot branches are
uniform and would emit `BSSY` / `BSYNC` around the 12 bodies.  Per operand:

- copies (`issue_tile_copies`, dynamic arm): for each slot j (unrolled): `if
  (m8u bit j) {fp8: IMAD.WIDE + LDGSTS.128 [land8 + j*PAGE]} else if (m4u bit
  j) {fp4: IMAD.WIDE + LDGSTS.64 [land4 + j*PAGE]} else {a16: 2 x (IMAD.WIDE +
  LDGSTS.128) [a16_dst + j*PAGE (+ATOM)]}`; six bases (`a16_src0/1`, `p8`,
  `p4`, `s8`, `s4`) in the a16 form of 3.2; then the octet scale copy of 3.3
  (8 instructions).  **Accepted forms**: the fp8 / fp4 slot blocks are two
  instructions and ptxas will most likely if-convert them (`@P IMAD.WIDE`, `@P
  LDGSTS`) rather than branch; the gate accepts either (predicated blocks: up
  to 6 x (2 + 2) = 24 predicated + 12 a16 instructions per operand executed;
  branched: ~6 x 3 tests / branches + the taken blocks).  Executed per operand
  on the bench mix (2 / 2 / 2): **~50-75** (F25 246).  The partial arm (two
  per-item calls) adds the D4 src-size register per copy as F25.
- decode (`expand_operand`, dynamic arm), in the 3.1 order: landing loads per
  slot under the same (re-balloted) branches (fp8 2 `LDS.64`, fp4 2 `LDS.32`,
  a16 none) -> the pair's `__syncwarp` -> 6 scale `LDS.32` -> per page one
  `f32(s_j)`, `gsel_j = in8_j ? gs8 : gs4` (`SEL`), one `FMUL`, one `FSETP`,
  folded into `ok8` / `ok4` by the format bit (2 `PLOP3`) -> two `VOTE.ALL` ->
  `Pc8 = !hot8`, `Pc4 = !hot4`; `g8 = Pc8 ? g8_plain : gs8`, `g4` likewise (2
  `SEL`; the plain globals are loaded once per item into two registers) ->
  per slot j: `if (m8u bit j) {fp8 block: sf2 = bf16x2(mul_rn_f32_denorm(f32(s_j), g8)) ;
  decode ; @Pc8 8 x HMUL2 by 2^120 ; 8 HMUL2 by sf2 ; 2 STS.128} else if (m4u
  bit j) {fp4 block likewise with Pc4, 2^126}`.  One body per format per slot
  (12 bodies per operand instead of 8 rolled loop variants per site); the
  exact form costs 8 predicated-off `HMUL2` issue slots per block on the hot
  path (+16 % of a block) and is bit-identical to F25's cold arm term for
  term (`@Pc` un-fold by 2^k, then `bf16x2(mul_rn_f32_denorm(f32(s), g))` -
  `mul_rn_f32_denorm` kept, the F24 FTZ lesson; `fold_ok` uses one constant
  for both formats, 255.5 x 2^120 == 3.9921875 x 2^126, so one `FSETP` per
  page is right).  Per operand on the bench mix: 4 blocks x (35 + 8 + 5) + 12
  tests / branches ~= **~200**; prep ~60.
- protocol: F25's per-operand pending tests (4 warp-uniform `if
  (op.pending)`), pending words with masks (8), 4 re-ballot `VOTE.ANY`,
  everything else as 3.4: ~75.

Per pair on the bench mix: copies ~130 + landing loads ~16 + prep ~120 +
decode ~400 + protocol ~75 + fillers ~40 = **~800** (F25 1211), plus ~12-18
taken uniform branches (the per-slot decode dispatch; the copy blocks are
expected predicated) at ~10-20 cycles each, which section 4 prices.  Code:
loop site ~2 x (75 + 60 + 12 x 52) + 70 ~= 1700; the two single-operand sites
~700 each (exact bodies only, `Pc` = true, no vote); prologue ~250 -> **region
~3400 instructions (54 KB) from 5528 (88 KB)**; hot footprint per pair ~800
instructions (13 KB) from ~1200 (19 KB) - the `no_inst` 10 % is expected to
fall toward the static modules' 2-3 %.  The mask-driven table build
(`chunk_store`'s `REDUX.OR`, 1 `BRA.DIV` per 16 pairs) is unchanged.

Why not predication for the decode (C17's letter): a 52-instruction body
predicated per slot would execute 12 bodies for 4 present pages; the branch
form executes the 4.  For the copies both forms are admissible and ptxas
chooses (above).  C17 is amended (section 7): "no per-page branch" -> "no
*divergent* per-page branch; uniform branches on ballot-derived masks are the
form".

### 3.7 Per-warp per-pair instruction budgets by class (executed; from the F25 SASS)

| class | fp8 F25 (`loopsite_fp8.log`, executed) | fp8 F26 | fp4 F26 | dyn F26 (bench mix) |
|---|---|---|---|---|
| landing loads (`LDS.64` / `LDS.32`) | 24 | 24 | 24 | 16 (+12 slot tests) |
| scale loads (`LDS.32`) | 12 | 12 | 12 | 12 |
| scale chains (`PRMT`, `F2FP.E4M3`, `HADD2`, `FMUL`, `FSETP`) + `PLOP3` + `VOTE` + `BRA` | 68 | 68 | 68 | ~100 (`SEL` per page, two accumulators, 4 `VOTE.ALL`, 4 `VOTE.ANY`, 4 `SEL`) |
| hot bodies (`F2FP.PACK`, 8 `PRMT`, 8 `IMAD.SHL`/`SHF`, 8 `LOP3`, 8 `HMUL2`, 2 `STS.128` per block) | 420 (2 x 210) | 420 | 444 (2 x 222) | 8 blocks x 43-44 + 64 `@P HMUL2` + 24 tests / `BRA` = ~440 |
| payload copies (`IMAD.WIDE.U32` + `LDGSTS`; fused form - 3-instruction form +48) | 24 + 24 + 45 adds = 93 (incl. scale copies) | 12 + 12 = 24 | 24 | 8 x 2 + 4 x 2 (a16 rows) + 12 tests (+ predicated-off slots up to 24) = ~48-72 |
| scale-row copies (3.3) | (in the 93) | 2 x 2 = 4 (+ 2 `LDS.32` counted under meta rows) | 4 | 2 x 6 = 12 |
| fillers `@!PT LDS RZ` | 6 | 3 | 3 | ~36 |
| meta rows (`LDS.128` x 4 + 2 `LDS.32` + ring) | ~25 | 10 | 10 | 10 |
| acquires, state, pending, bases, `DEPBAR`s, fence + commits, loop | ~74 | ~51 | ~51 | ~66 |
| **total** | **716** (ncu 716) | **~612** (fused) / ~660 (3-instruction chains) | **~636** / ~684 | **~800** |

Pipe mix at parity (per SMSP per pair of 2544 cycles), fp8 F26: producer ALU
(`PRMT`, `LOP3`, `FSETP`, `PLOP3`, `ISETP`, `SEL`, `IADD3`) ~440 x 2 cycles =
880; FMA (`HMUL2` 96, `IMAD.SHL` 96, `IMAD.WIDE` 28 x 2, `IMAD` ~20, `FMUL` 12)
~280 x 2 = 560; XU (`F2FP` 24) ~200; LSU issue ~60 x 4 = 240.  Consumer (2
warps, 826 instructions): FMA ~330, ALU ~200, XU ~1200.  Totals: ALU 42 %,
FMA 35 %, XU 55 %, issue (612 + 826) / 2544 = 57 %.  Unchanged in kind from
F25 (the decode is the same); nothing saturates.  The shared-memory data pipe
(3.9) at 75-79 % is the one resource near its limit, and it is what prices
the copy block in section 4.

### 3.8 Dependency depth of the pair (what is on the critical path)

- To the K' `VOTE`: `DEPBAR` (~20 when the group is complete) -> `LDS.32` scale
  (~30) -> `PRMT` (4) -> `F2FP.E4M3` (~8) -> `HADD2` (4) -> `FMUL` (4) ->
  `FSETP` (4) -> `PLOP3` x 2 (8) -> `VOTE.ALL` (~10) -> `BRA`: **~90 cycles of
  chain**, of which F25 exposed ~70 (the `PRMT` PC alone 118 cycles as
  short_sb + the chain's `wait`).  In F26 the copy block (~850 cycles) is
  between the `LDS.32` and the `VOTE`: exposed 0; the `VOTE` -> `BRA` (~10)
  remains.
- To the first K' `STS`: `VOTE` -> `F2FP.PACK sf2` (~8) -> `HMUL2` (4) -> `STS`
  ~= 15 after the vote; the decode chain of block 0 (`LDS.64` landed long ago
  -> `PRMT` -> `SHL` -> `LOP3`, 12) runs before the vote resolves.
- To the first V' `STS`: the V' landing words and scale chain were loaded /
  computed before the copies: exposed 0 (F25 exposed the 12 `LDS.64` + 6
  `LDS.32` + chain: 274 cycles in the V prep segment); the V' `VOTE` -> `BRA`
  (~10) remains.
- Copy block: chunk rows `LDS.128` (~30) -> `IMAD.WIDE.U32` (~6) -> `LDGSTS`:
  the rows are read before the scale chains are issued, so the first
  `IMAD.WIDE` finds its page index ready.  The 14 chains of an operand are
  independent; the block's length is the LSU pipe's service of its 28
  `LDGSTS` (~30 cycles each, section 4) - the one segment F26 cannot shorten
  below the wavefront count.
- Acquire: `test_wait` (~120 round trip) issued after the V' body, `@!P BRA
  spin` at the top of the next pair: exposed 0 when the stage is free; when
  the consumer has not released, the spin is the consumer wait (desired, and
  it now precedes the `DEPBAR`, 3.1).
- Landing: the `DEPBAR` after the spin; cover ~1600 cycles at parity (3.1);
  exposed 0-400 depending on the loaded latency - gate 6.2.
- Tail: `FENCE.VIEW.ASYNC` waits for the 12 V' `STS.128` to perform (~50-70
  cycles, long_sb 68 % of its samples): kept (one fence per pair); the two
  probes are issued ahead of it so their round trip overlaps this drain.

### 3.9 Shared-memory wavefront budget (C11)

Per pair per SM, fp8: wgmma operand reads 1001 (unchanged); `STS.128` 96 x 4.00
= 384 (unchanged, ideal at every PC); landing `LDS.64` 96 x 2 = 192, scale
`LDS.32` 48 x 1 = 48, chunk-row `LDS.128` 16 x 2 = 32, octet page-index
`LDS.32` 8 x 1 = 8: attributable `op_ld` ~280 - but F25 measured `op_ld` 441
with the same attributable set (~272), so **~170 wavefronts per pair of
`op_ld` are unattributed** (`TRYWAIT` / `SYNCS` / barrier traffic booked to
the class) and will not vanish: F26 states **`op_ld` ~450**, not ~280
(revision 2).  `LDGSTS.128` 48 x 4 = 192, scale-row `LDGSTS.64` 8 x ~2 = 16
(F25 96 x ~2 = 190 -> 16; total `LDGSTS` 281 -> ~210); non-`STS` `op_st`
residual ~136 (unattributed, unchanged).  **Total ~2000-2050 wavefronts per
pair = 79-81 % of 2544** - C11's <= ~85 % holds but with less room than
revision 1 stated; fp4 ~2000 (landing `LDS.32` 96 x 1, payload `LDGSTS.64` 48
x 3.98); dyn ~2000.  This share is what the consumer sees as contention on
its 1001 gmma-read wavefronts (the term c of section 4, 2-10 %) and what
prices the producer's own `LDGSTS` at ~30 cycles each.

### 3.10 Registers (C3)

Producer 136.  Live set at the widest point of the revision 2 order (during
the copy block): K' packed words 24 + V' packed words 24 + K' / V' scale words
12 + K' / V' scale products 12 + copy sources in flight (ptxas keeps ~4-6 pairs
= ~10) + chunk pages 12 + octet page words 2 + bases (`p8`, `s8` pairs per
operand 8; `out0/1`, `land8`, `sc_rd`, `sc_cp` 10; expand bases 2 x 5 = 10) +
strides in UR + counter / phase / ring / loop / countdown ~8 + item constants
~10 + `gs8` x 2 = **~130**.  This is close to 136 and is the design's one
register risk (revision 1 had ~104 with the V' loads deferred into the K'
body).  ptxas -v gate: no C7507, `STACK 0`, `LDL` / `STL` 0.  Fallback if the
gate fails (structural, priced): issue the V' landing loads after the
`LDGDEPBAR` and before the K' vote (still straight-line, no branch join, one
`__syncwarp` before the K' vote instead of before the scale loads - the
barrier then orders both operands' loads before both bodies as before);
their latency then sits behind the 28 `LDGSTS` in the MIO queue and is
consumed ~450 cycles later after the K' body: exposed ~0-100 cycles (the fp4
mechanism of 3.5 for V' only, at a longer distance).  During the K' body
after block 3: K' remaining packed 12 + decoded in flight ~24 + `sf2` 6 + V'
packed 24 + V' `sf2` 6 + bases ~30 + protocol ~10 = ~112.  Dynamic module:
the per-slot branched bodies hold one block's decode at a time; the six
scale words + six `f32(s_j)` + 2 masks + 8 bases (a16 pairs 2 x 2, compressed
pairs 4 x 2) ~= 30 more than static's constants -> ~120; same gates.
Consumer 184, untouched.

## 4. Predicted producer time and walls (revision 2: re-derived from the a16 loop-site segments)

**Calibration (blocker 11.2).**  The reference for the copy block is the a16
module's *loop-site* copy blocks, not its item prologue: `f26_pc.py
ncu_a16_source.csv 0xcb00 0xeab0 --dealloc 0xb3e0 --seg ...` gives the two
blocks sf0 `[0xd5e0,0xd8a0)` and `[0xdf00,0xe1c0)` - each 12 `IMAD.WIDE.U32`
with an `R.64` accumulator + 12 `LDGSTS.128` + ~20 `VIADD` / `MOV`, zero adds -
at 14.0 % + 15.7 % = 29.7 % of the a16 loop-site samples (31,943; a16 pair
~2530 cycles) = **~750 cycles per pair for 24 `LDGSTS` = 31 per `LDGSTS`, 8.5
per instruction, 7.8 per wavefront (96)**; stall mix wait 27-30, dispatch
24-43, not_selected 13-22, selected 13-16 (excluding not_selected: ~600).  F25
fp8's copy blocks: 1160 cycles for 36 `LDGSTS` (120 wavefronts) = 32 per
`LDGSTS`, 9.7 per wavefront - the same price per `LDGSTS` with a 3-instruction
address chain.  **Reading: a copy block costs its `LDGSTS` count times ~30
cycles (its wavefront count times ~8-10), whichever address form feeds it**;
the price is the LSU pipe's service under the 75-80 % utilisation of 3.9.
Revision 1 priced the F26 block at 180 (centre) / 600 (pessimistic) from an
issue-slot model; that is refuted by these samples and replaced.

**Conversion.**  Segment cycles below are issue-based as section 1 counts them
(F25 fp8 pair 3113 = 716 / 0.23); the wall-based pair is 397.2 us / 220 / 1.98
GHz = 3574 (the item prologue, Q wait, drain and sampling outside the loop
site), a factor **1.148** that is applied to compare a producer pair with T_c
= 2544 (which is a wall-based number: 282.8 us / 220 pairs).  The accept
threshold <= 330 us is a wall pair of 330 / 220 x 1980 = 2970 cycles = **an
issue pair of ~2590**.

**Method**: F25's per-segment cycles (section 1) carried forward with each
mechanism's removal argued from its PCs, the copy block priced by the a16
calibration:

| segment | F25 fp8 | F26 centre | F26 pessimistic | F26b' fallback (3.1) | argument |
|---|---|---|---|---|---|
| protocol outside the copies (index arithmetic, acquire tests, chunk rows, chunk barrier, state) | ~385 (A minus its copy part) | 120 | 220 | 120 | probe and `LDS.128` latencies hidden; ring addressing removes the division chains; chunk `BAR.SYNC` amortised 50 stays |
| copy blocks (K + V) | 1160 (36 `LDGSTS`, 120 wf) | **850** (28 `LDGSTS`, ~100 wf, at the a16 price 31 / `LDGSTS`; 100 x 7.8-9.7 = 780-970) | **1100** (3-instruction chains: +56 ALU slots, ~+100; pipe contention above the a16 calibration) | 850 | the block is pipe-bound: the saving is the eight `LDGSTS` and ~20 wavefronts of the octet scale copies, ~250 cycles, not the address form |
| `DEPBAR`, 24 landing + 12 scale loads, pre-vote chains, K' `VOTE` | 382 | 100 | 200 | 380 | loads' issue ~50 + `DEPBAR` ~20 + `VOTE` -> `BRA` ~20; pessimistic: half the scale chain and the landing exposed (3.1 cover); F26b': F25's exposure returns |
| K' hot body | 395 | 395 | 430 | 395 | unchanged code, measured IPC 0.53 |
| V' prep (vote + branch; loads and chain done before the copies) | 274 | 40 | 100 | 274 | F26b': F25's V prep returns |
| V' hot body | 380 | 380 | 420 | 380 | unchanged, IPC 0.55 |
| cold-arm fetch after the taken branch | 66 | 20 | 66 | 20 | hot arm as fall-through is a ptxas layout outcome: reported, not gated |
| tail (2 probes, fence, arrives, loop) | 68 | 80 | 100 | 80 | probes overlap the fence drain |
| **pair (issue)** | **3113** (IPC 0.23) | **~1985 (IPC 0.31)** | **~2640 (IPC 0.23)** | **~2500** | |
| **pair (wall-equivalent x 1.148)** | 3574 | **~2280 < 2544: consumer-bound (10 % margin)** | **~3030 > 2544: paces at 336 us** | ~2870: paces at ~318 us | accept iff issue pair <= ~2590 |

The IPC is derived, not assumed: the bodies' 0.53-0.55 is measured and
unchanged; the copy block is the a16 module's measured price for its
wavefronts; the non-body, non-copy part goes from 296 - 130 instructions in
1180 cycles to ~150 instructions in ~340 cycles because its stall mechanisms
(exposed load latencies, MIO queue order, division chains, the acquire round
trip) are each moved ahead of a ~850-cycle block that hides them; the
pessimistic column keeps half of those latencies exposed and a 3-instruction
copy chain.  What the centre needs from the build: the copy block at the a16
price (a copy block up to ~1450 cycles still passes the accept if the other
rows hold), and the landing cover holding (6.2).

fp4: + 24 issue slots (`LDS.32` landings, 222-instruction bodies) and the
same wavefront count (`LDGSTS.64 .ca` 3.98 wavefronts): **~2010 / ~2660** ->
wall 2310 / 3050 -> consumer-bound / paces at 339 us.

Dynamic (bench mix 2 / 2 / 2 per tile): copies 20 `LDGSTS` (8 payload + 2
scale per operand, ~68 wavefronts) at the a16 price = **600** + per-slot
uniform branches ~12-18 taken at ~15 = **230** + bodies 8 blocks x 52 (44 + 8
`@Pc HMUL2`) at IPC 0.5 = **830** + prep (16 landing + 12 scale loads, 12 chains
with `SEL`, 4 `VOTE.ALL`, 4 `VOTE.ANY`) ~120 at 0.35 = **340** + protocol
(pending words, per-operand tests) ~70 at 0.35 = **200** + tail 80 = **~2280
issue (IPC 0.35) -> wall ~2620: at parity with T_c** (+3 %; the mixed centre
is a producer-paced 300-310 us, not a consumer-bound 300).  Pessimistic (copy
body 800 with predicated-off slots issuing, branches 400, bodies 900, prep
400, protocol 250, tail 100) ~2850 -> wall 3270 -> **357 us**.  The F25 model
(~790) missed the dynamic module by a factor of 1.5 on the same kind of
count; this one is checked against the a16 price for the copies and the
measured body IPC, and its one unpriced term is the taken-branch cost (~15
cycles each assumed; 6.3's `no_inst` + `branch_resolving` rows measure it).

**Walls**: consumer-bound modes land at `T_c (1 + c) x 220` where c is the
smem-pipe contention the compressed operands add to the consumer's gmma reads
(80 % vs 44 % pipe share).  **c is unmeasured** (no run has had a non-pacing
compressed producer); revision 1's 2-4 % is widened to **2-10 %** (queueing at
80 % utilisation is not 2-4 % by default; the target survives c up to ~16 %),
and 6.3 bounds it by the consumer K-wait rule.  fp8: 283 x 1.02-1.10 = **289-311**
consumer-bound, centre (c 5 %) **297 / 304**; the pessimistic column paces at
336 / 344; band 289-336 / 296-344.  fp4: centre 298 / 305, band 290-339 /
297-347.  mixed: centre ~305 / 313 (parity, c ~6 % incl. the A16-tile
commits), band 295-357 / 303-365.  transport_a16 must reproduce 281-286 /
287-293 or the session is offset.

**Accept / reject rows (6.4)**: fp8 <= 311 / <= 319 = consumer-bound as
predicted (accept, centre); 312-330 = the producer paces or c > 10 % (accept
on the target; 6.3 says which: K-wait <= 3 % with the producer pair > T_c =
paces, K-wait 3-8 % with the pair < T_c = c); > 330 reject (6.3's segment
table names the segment that did not move: copy block > 1450 = the pipe price
is above the a16 calibration; prep > 250 = the cover or the MIO order; body
rows moved = a mistake, F26 does not touch them).  fp4 likewise with 312 /
320 and 340.  mixed <= 330 accept, 331-357 = the per-slot branch / copy-body
price materialised (build nothing new: report the measured dyn segment
table and stop), > 357 reject (count model wrong again: re-read the dyn
SASS).  F26b' (fallback) has its own rows: fp8 <= 318 / <= 326 accept.

## 5. SASS gates before any timing (F26a-c builds; `cuobjdump -sass`, producer region `USETMAXREG.DEALLOC .. EXIT`, loop site = the back-edge body with the two / four loop `VOTE`s; `f25_counts.py`, `f25_loopsite.py`, `f26_seg.py`)

| gate | accept | reject -> action |
|---|---|---|
| `USETMAXREG` `DEALLOC 0x88` / `TRY_ALLOC 0xB8`; `ptxas -v` no C7507; `STACK 0`; `LDL`/`STL` 0 | as F25 for fp8, fp4, dyn; a16 `0x48` / `0xD8` | any: the 3.10 fallback (V' landing loads after the `LDGDEPBAR`, before the K' vote) - one structural change, then re-read; no other register tuning |
| **copy block** `[first LDS.128, LDGDEPBAR]` (static loop site) | 28 `IMAD.WIDE.U32` and 28 `LDGSTS` per pair (24 `.128` / `.64` payload + 2 `@P .64` scale), `@!PT LDS RZ` fillers 3, `SEL` 0; **form reported**: fused (`IMAD.WIDE.U32 Rd, R, UR, R.64`, `IADD3`/`IADD3.X`/`IMAD.X` with carry 0, `MOV` into address pairs 0 - the probe kV's opcode sequence) or 3-instruction chains | the form is a fact of the build (3.2), not a stop: record it and use the matching section 4 column; a `LDGSTS` count != 28 or `SEL` on an address = a code mistake |
| **order** (3.1, F26b) | in the loop site, in program order: `@!P BRA` x 2 to the spins -> `DEPBAR.LE SB0, 0x0` -> 24 landing `LDS` -> `NOP`/`WARPSYNC` (the one barrier) -> 12 scale `LDS.32` -> 4 `LDS.128` + 2 `LDS.32` -> (chains interleaved) -> copies -> `LDGDEPBAR` -> `VOTE.ALL` -> K arm -> `VOTE.ALL` -> V arm -> 2 `SYNCS.PHASECHK` (test) -> `MEMBAR` + `FENCE` + 2 `SYNCS.ARRIVE` | any `LDS` of the pending pair after the first `LDGSTS`; a `DEPBAR.LE SB0, 0x1`; a `DEPBAR` before the spin branches; a second `NOP` barrier; `BRA.DIV` > 0 -> restructure (the order is the design) |
| scale-row copies (3.3) | per operand exactly one `LDS.32 [R + UR + imm]` of the row's page word (lane-indexed) hoisted before the first `LDGSTS`, one `IMAD.WIDE.U32`, one `@P LDGSTS.E.LTC128B.64`; no `@P` on payload `LDGSTS` in the loop | six `LDGSTS.64` per operand -> the gather did not fold; the `LDS.32` inside the copy block -> fillers 6, hoist it |
| protocol (3.4) | loop-site count - 24 - 12 - 68 - 420 - (24 or 72) - 4 - 3 <= 70; `SHF.R.S32.HI` / `LEA.HI` division chains 0; `SYNCS.PHASECHK` 2 per pair after the V arm and before the `MEMBAR` | over -> list the opcodes |
| bodies (C12) | hot 210 / cold 267 (fp8), 222 / 279 (fp4) unchanged; `VOTE.ALL` 2 in the loop site, 0 in bodies; `BRA.DIV` 0; `UMOV` 0 in bodies; `MOV` / `IMAD.MOV` inside a body 0 - **at the K arm join up to 24** are accepted (the V' packed words live across the join, revision 2) and reported | any change to the bodies is a mistake (F26 does not touch them) |
| hot arm as fall-through | reported (`@P BRA` after the vote targets the cold arm) | not a stop: ptxas layout (~46 cycles per pair either way) |
| region | fp8 <= 2450, fp4 <= 2550 (F25 2584 / 2664 minus the copy adds and protocol; with 3-instruction chains + 100); dyn **<= 3600** (from 5528) | dyn > 4000 -> the per-slot bodies were not shared per format or the single-operand sites carry votes |
| dyn copy body | per operand 6 slot blocks on `VOTE.ANY`-derived masks, each either `ISETP`/`LOP3 P` + `BRA` (branched) or `@P IMAD.WIDE.U32` + `@P LDGSTS` (predicated) - both accepted and reported; `BSSY`/`BSYNC` 0 in the copy and decode bodies; `FLO`/`BREV` 0 in the loop site; `LDGSTS` sites 6 x (1 + 1 + 2) + 2 = 26 per operand; `VOTE.ANY` 2 per operand at the copy site and 2 at the finish site | `BSSY` > 0 -> a mask reached a branch without a ballot (the finish-site re-ballot is missing) |
| dyn decode | 12 bodies per operand in the loop site (fp8 / fp4 per slot), each with 8 `@P HMUL2` + 8 `HMUL2`; `VOTE.ALL` 4 in the loop site; the single-operand sites: 12 bodies with 16 unpredicated `HMUL2`, `VOTE` 0 | rolled loop (`BRA` back-edge inside a body) -> C10 violated |
| a16 module and stock kernel | a16 SASS identical to the F25 a16 stream (`sf0_mask1.sass`, 3832 / 3880 instructions; the four-UR permutation against `5cc416fd` is accepted as in F25), in particular its loop copy blocks `[0xd5e0,0xd8a0)` / `[0xdf00,0xe1c0)`; stock byte-identical | any a16 change: the `STATIC_A16` arms or the shared loop text were touched |

Gate 6.0 (a16 `LDGSTS` PCs) is not repeated: it passed in F25 and F26 does not
change the copy form the a16 module already has.  Nothing in SASS can gate
the landing cover: **the trace of 6.2 is a hard stop before any bench.**

## 6. Verification (after the gates; confirmation, not tuning)

- **6.1 tests** (`tests/attention/run_fa3_mixed_page_transport.py`; exit code
  = failures; plain runner, no test framework).  The F25 104 cases (the
  NaN-tail cases exercise the partial arms of the two per-item calls, whose
  copy forms change too) **+ 14 new cases, stated in tiles** (blocker 11.1:
  CTA_KV = 96, a chunk is 16 tiles, the ring 32 rows; every existing case has
  `pages_per_req <= 18` = <= 3 tiles per item, so buffer 1 of the chunk table,
  the wrap, the `j == 0` gather, the `j == 8` store + `BAR.SYNC` and the
  countdown had never been bit-checked - only the bench's 43-tile items ran
  them, unchecked):
  - `multi-chunk` 33 tiles: `pages_per_req = 193`, `kv_len = 193 x 16 - 3 =
    3085` -> 33 tiles (entries 0..32): chunk 0 rows 0-15 (buffer 0), chunk 1
    rows 16-31 (buffer 1), chunk 2 = entry 32 -> **row 0 (the wrap)**; the
    pair whose V is entry 31 reads its K row at `(31 + 1) & 31 = 0`; the
    countdown reaches 0 at entry 32 with `next_chunk` false (no gather, no
    store); two gathers (at entries 0 and 16), two stores + `BAR.SYNC` (at 8
    and 24);
  - `multi-chunk` 22 tiles: `pages_per_req = 130`, `kv_len = 130 x 16 - 5 =
    2075` -> 22 tiles (entries 0..21): buffer 1 rows 16-21 **without** a wrap,
    one gather / store, countdown 0 at entry 16;
  - both for fp8, fp4 and mixed (the dynamic module's masks through buffer 1
    and the wrap), q_len 1 and 64 (causal: with CTA_Q = 128 and qo_len 64 the
    tile count is unchanged) = 12 cases, B = 2, H = 2, tail page partial;
  - `dynamic-uniform-extremes` fp8 with `g = 1` and fp4 with `g = 0.5` (q =
    1): six pages of one format per tile through the dynamic module with the
    extremes payload / scale set, so the 448-scale blocks (fp8: and 256) fail
    the per-operand vote while the others pass - the per-slot `@Pc`
    predicated exact form (3.6) is taken by whole operands on both formats =
    2 cases.
  **The 14 cases are run on the F25 kernel first** (the tests commit precedes
  F26a; the F25 tip must pass 118 / 118): that is the baseline of the tests
  themselves, so that a ring / countdown / octet bug in F26a cannot pass as
  "106 / 106" and ship into the bench as silently wrong numerics.  Then 118 /
  118 on F26a, F26b, F26c each.
- **6.2 trace** (`MIXED_FA3_TRACE`, q=1, CTA 0 items 0/1; ratios, not
  absolutes; the stamps follow the revision 2 order: `acq` = the spins at the
  top, `wait` = across the `DEPBAR` after them, `iss` = loads + chains + copies
  to the `LDGDEPBAR`, `expK` / `expV` = vote + body, `fc` = probes + fence +
  commits): **`wait` <= 0.05 us per pair (hard stop before any bench; 3.1);
  reject > 0.1 -> F26b' (3.1), whose own accept is `wait` <= 0.05 at the F25
  position**; fp8 `iss` <= 0.50 us (the copy block at the a16 price, ~850
  cycles = 0.43 us, plus the loads; from 0.38 + the F25 prep); **expK == expV
  within 0.03 us in fp4** (from 0.68 / 0.38), fp8 expK + expV <= 0.45 (the
  bodies alone; from 0.88); mixed `iss` <= 0.55 (from 1.63-1.78).
- **6.3 ncu** (fp8 / fp4 / mixed, q=1, third launch, `f25_run_ncu.sh` +
  `f26_pc.py --seg`): producer `inst_executed` per pair 2450 +- 5 % (fp8;
  2640 with 3-instruction chains), 2540 / 2730 (fp4), 3200 +- 8 % (dyn); the
  segment table of section 4 filled from the samples - **the copy-block
  segment's cycles per `LDGSTS` (accept 25-40, the a16 calibration)**, the
  first post-`DEPBAR` PC's long_sb (<= 2 %), the `PRMT`-after-scale-`LDS` PC (<
  0.5 %), the two `@!P BRA` spin branches (their long_sb is now the consumer
  wait: report), `dispatch` <= 15 % of loop-site samples, `no_inst` <= 3 %
  (dyn), `branch_resolving` (dyn: the taken-branch price); **consumer K-wait
  PC <= 3 %** (the consumer-bound proof); consumer `inst_executed` 3301;
  tensor-pipe active within 5 % of a16's 67.1 %; `l1tex` shared wavefronts
  per pair 1950-2100 by class (`op_ld` ~450, `op_st` ~520 with `STS.128` at
  4.00 per PC, `LDGSTS` ~210, **the octet scale `LDGSTS.64` PCs' wavefronts
  per instruction reported** - expected ~2, A7's rule says 1-2 for 48
  contiguous bytes); the non-`STS` `op_st` residual attributed by PC
  (follow-up).  Reading rule: K-wait 3-8 % with the producer pair < T_c in
  the trace = the contention term c is larger than 4 % (report it; nothing
  on the producer side fixes it); K-wait > 8 % or K-wait <= 3 % with the
  wall above 311 = the producer paces: the segment table names the segment.
- **6.4 bench** (`bench_fa3_mixed_page_transport.py --q-lens 1 64 --repeats 1
  --trials 5`, nkcut2 lock, co-tenant rule): the section 4 rows; min / median
  / max reported.

## 7. Invariant amendments to write into the dataflow document when F26 lands

- **A10 (pair order; supersedes A9's ordering sentence).**  Per loop pair:
  the acquire spins (on the probes of the previous pair); `cp.async.wait_group
  0`; both pending operands' landing loads; one `__syncwarp`; both pending
  operands' scale-slot loads; the two chunk rows and the two octet page words;
  the copies; `commit_group`; K vote and body; V vote and body; two
  non-blocking `mbarrier.test_wait` probes for the next pair's stages; fence;
  commits.  Ownership unchanged (A7 / A9).  Scale slots: lane b < 6 of a row
  octet copies page b's row (one instruction per operand) into the
  row-contiguous slot `r x 48 + b x 8`; the row's eight lanes read all six
  slots after the barrier.
- **C13 (copy addressing), restated.**  Every compressed source is `base +
  page x stride` with the base built as the a16 module's (`DTypeKV`-typed
  element arithmetic in `make_bases`, once per item) and the page term in C++;
  no asm.  The SASS form (fused `IMAD.WIDE.U32` with an `R.64` accumulator, or
  a 3-instruction chain) is reported per build, not gated: the copy block's
  cost is its `LDGSTS` count times the LSU pipe's price (~30 cycles), so the
  design's lever on the copy block is the `LDGSTS` count (28 per pair).
- **C14 (protocol), restated.**  <= 70 per pair per warp, itemised as 3.4;
  chunk-table rows addressed as a 32-row ring (`row = e & 31`); one pipeline
  counter for both rings (`sK = sV + 1 mod 3`); static modules carry no
  pending word (pending stages = `sV`, `sV - 1`); acquires as a non-blocking
  probe at the end of the previous pair + a spin at the top of the pair, the
  spin before the `wait_group`.
- **C17 (dynamic copies), amended.**  Per page slot, a warp-uniform branch on
  ballot-derived format masks (re-balloted at the finish site) selects one of
  three straight-line copy forms - ptxas may if-convert the two-instruction
  compressed forms; no rolled loop, no divergent branch, no `SEL` on an
  address; the scale rows by the lane-octet form with per-lane format
  predicates.  **C10**'s rolled decode loops are withdrawn: per-slot bodies,
  one per format, exact form by predicate.
- **C11.**  Unchanged bound (<= ~85 % of T_c); F26 reports `LDGSTS` ~210 and
  `op_ld` ~450 (the ~170 unattributed wavefronts stay) as the new expected
  values: ~2000-2050 per pair, 79-81 % of 2544.
- **C18 (new, store pattern).**  `STS.128` at 4.00 wavefronts per instruction
  at every PC (ncu source view) is the acceptance; the aggregate `op_st /
  STS.128` ratio is not a gate (it counts non-`STS` traffic).
- **C19 (new, copy-block price).**  A copy block costs its `LDGSTS` count times
  the LSU pipe's price under the module's shared-memory utilisation (a16 and
  F25 both ~30 cycles per `LDGSTS`, 8-10 per wavefront); a design that wants a
  cheaper copy block must remove `LDGSTS` or wavefronts, not address
  instructions.

## 8. Do not build

1. **Any change to the decode bodies** (placement decodes, `HMUL2` count,
   store order, chunk permutation): measured at IPC 0.53-0.59 and 4.00
   wavefronts per `STS.128`; they are 25 % of the pair and not the problem.
2. **The 32-bit page-offset copy form** (brief item 2, first option): kB shows
   it keeps the carry predicate (`IMAD` + `IADD3` + `IMAD.X`) and it needs a
   host bound for no gain.
3. **Asm `mad.wide.u32` for the bases or the page term** (F25e's form): kA / kM
   show ptxas splits an asm-formed 64-bit accumulator; `ld.shared.b64`-opaque
   bases likewise (kY); byte-typed pointer sums of two 64-bit products (kT /
   kX / kZ) are split too - only the a16 construction (3.2) is written.
4. **A uniform-datapath page term** (`R2UR` / `UIMAD.WIDE`): ptxas does not
   treat shfl-broadcast values as uniform for addressing (kC) and `LDGSTS`
   has no `[R.U32 + UR.64]` form (kJ).
5. **A second producer warp group, role-split producers (2E / 2G),
   consumer-side decode, SWAP_AB, fp8 wgmma** (F25 sections 2A-2E, 7): the
   F26 reading shows the producer pair can be brought under T_c without new
   warps; the F25 rejections stand.
6. **A store-order or landing-layout change for the "5.41 wavefronts"**: not
   the store pattern (section 1).
7. **The sorted-page table / per-class unrolled bodies** for the dynamic
   module (F25 6.4): pages cannot be reordered; the class dispatch is the
   per-slot branch by another name with 28 bodies.
8. **Rolled per-format loops with a predicated hot / exact body** (F25
   follow-up 3 as stated): keeps 80 instructions per block and the 492
   copy body; the per-slot form replaces both.
9. **Moving the chunk store / `BAR.SYNC`** out of its pair: 50 cycles per pair
   amortised, and its WAR argument (`CHUNK_STORE_PAIR > NUM_STAGES`) is
   settled.
10. **A three-fold loop unroll to make the stage an immediate**: 3 x 1294
    instructions in the loop site alone (62 KB) against the instruction
    cache that already costs the dynamic module 10 %.
11. **Trace-segment absolute values as inputs** (F25 do-not-build 10 stands):
    the fp4 `wait` stamp reads 0.01 us while ncu shows the landing wait on
    the first post-`DEPBAR` PC.
12. **Any GPU timing before the section 5 gates and the 6.2 trace pass**; in
    particular no bench before the `wait` stamp has been read (the landing
    cover, 3.1), and no "tuning" of the copy form on the kernel: the form is
    settled by the compile-only probe of 3.2, then reported.
13. **A blocking `try_wait` probe before the pair's commits** (3.1): it can
    suspend the warp until the consumer releases a stage two pairs older and
    would hold the pending pair's arrives hostage; `test_wait` only.

## 9. Files touched (build order; each step bit-exact on its own; a16 and stock untouched throughout)

- **Tests first** (`tests/attention/run_fa3_mixed_page_transport.py`): the 14
  cases of 6.1 (`_run_multi_chunk(mode, q_len, pages_per_req, tiles)`,
  `_run_dynamic_uniform(..., extremes_gs=0.5)`), so that the F25 kernel is the
  first thing they run against (118 / 118 expected on the F25 tip).
- **F26a - copies and protocol** (`sparse_mixed_mainloop.cuh`; host check for
  even strides in `to_underlying_arguments`): `OperandBases` bases as
  `uint8_t const*` built in the a16 form (`compressed_base` = `DTypeKV`-typed
  element arithmetic, `page_src` in C++, asm forms removed); the lane-octet
  scale copy (`copy_scale_rows`) into the row-contiguous slot layout
  (`SCALE_ROW_STRIDE = 48`, `sc_rd` / `sc_cp` / the `j x 8` immediates);
  `read_meta_row(row_smem)` by ring offset; the compressed loop (`pair_step`
  / `produce_pair` of the compressed modules) with `oV`, the `j` tests as
  masks, the chunk countdown, one `(sV, phase)` counter with the K state
  derived and both `PipelineState`s written back at the item's end, static
  pending stages from the counter (`Operand::pending` kept for the dynamic
  module only), `test_wait` probes after the finish and the spins at the top.
  F25's pair order kept (copies -> `LDGDEPBAR` -> `wait_group 1` -> finish).
  The a16 module keeps the F25 loop text verbatim (`if constexpr
  (STATIC_A16)`).  Gate: section 5 copy / protocol / a16 rows; tests 118.
- **F26b - pair order** (`finish_pending_pair` split into `finish_loads`
  (`wait_group 0`, both operands' landing loads, `__syncwarp`, scale loads,
  chains) before the copies and `finish_bodies` (votes, bodies) after them,
  then the probes, fence, commits; the spins before the `wait_group`; the
  single-operand sites keep `finish_one`).  The dynamic module keeps F25's
  finish order in this step (`if constexpr (DYNAMIC)`), F26c moves it.  Gate:
  order row; trace 6.2 (`wait` hard stop).
- **F26c - dynamic module** (`issue_tile_copies` dynamic arm per slot with
  ballot masks; the dynamic finish per slot with re-balloted masks in the
  F26b order; `expand_block_pred<FP8>(..., Pc)` for the dynamic bodies - the
  static modules keep the compile-time `EXACT`; `expand_format_pages`
  removed).  Gate: dyn rows; tests incl. the two extremes cases.
- **F26d - measurements** in the section 5 / 6 order, beginning with the
  compile-only probe kV / kV2 of 3.2; results appended here and to the
  backends document; dataflow amendments of section 7.

## 10. The floor, stated

With the bodies unchanged, the producer's pair is bounded below by the two hot
bodies (775 cycles at their measured IPC) plus the copy block at the LSU
pipe's price for 28 `LDGSTS` (~850) plus the loads, votes, probes and commits
(~350) = **~1985 issue cycles = ~2280 wall-equivalent**; the consumer's pair
is 2544.  F26 therefore does not need the producer to be fast; it needs it to
stop pacing, and revision 2's arithmetic says it does so with 10 % of margin
at the centre and not at the pessimistic end - the accept is decided by the
measured pair, not by this document.  Once the producer is under T_c the wall
is transport_a16 (283 / 290) plus the compressed operands' share of the
shared-memory pipe (~80 % vs 44 %), estimated at +2-10 % and measured by
6.3's tensor-pipe and K-wait rows; **~289-311 us at q=1 is the floor of every
compressed mode on this kernel**.  What F26 cannot do: lower T_c (the
consumer is FA3's, C5), remove the smem-pipe share (BF16 materialisation, F25
section 9), cut the copy block below its wavefront count (C19), or make the
mixed mode faster than the static ones (its per-slot branches and predicated
exact form cost ~+10 % of issue slots over fp8 at equal blocks).  If 6.3 shows
the producer under T_c and the wall above 311, the residual is c (the
consumer's smem contention), and the next lever is on the consumer's side of
C5, not on the producer.

## 11. Judge blockers on revision 1 and their resolutions (revision 2)

Each item names the rev 1 text, the defect and what rev 2 does instead; the
design changed where the blocker required it.

1. **The tests could not see the ring-addressed chunk table** (6.1's "kv_len =
   16 x 32 + 1").  CTA_KV = 96, so 513 tokens are 6 tiles: rows 0..5 of buffer
   0 - never buffer 1, never the wrap, never the `j == 0` gather / `j == 8`
   store / countdown that F26 rewrites; and every existing case has <= 3 tiles
   per item, so chunk >= 1 had never been bit-checked in any round (the bench's
   43-tile items are unchecked).  **Fix: 6.1 states the cases in tiles** - 33
   tiles (`pages_per_req = 193`, `kv_len = 3085`: buffer 1, the wrap of entry
   32 to row 0, the K row of the pair with V at entry 31 read at row 0, the
   countdown reaching 0 with `next_chunk` false at chunk 2) and 22 tiles
   (`pages_per_req = 130`, `kv_len = 2075`: buffer 1 without a wrap), each for
   fp8 / fp4 / mixed at q = 1 / 64, plus two dynamic-uniform extremes cases;
   the tests commit precedes F26a so that **the F25 kernel is their first
   subject** (118 / 118 there is the baseline of the tests themselves).
2. **The copy-block cost model was refuted by the a16 module's own samples.**
   Rev 1 priced the F26 block at 180 / 600 cycles by attributing F25's 1160 to
   carry-predicate serialisation, and cited the a16 item-prologue block
   (`0xc6c0-0xc870`, exec 19,584 per run) as the fused reference without
   segmenting the a16 loop site.  The a16 loop-site copy blocks
   (`[0xd5e0,0xd8a0)` + `[0xdf00,0xe1c0)`, zero adds) take 29.7 % of 31,943
   samples = ~750 cycles per pair = ~31 cycles per `LDGSTS` - the same price
   as F25's 32 per `LDGSTS` with 3-instruction chains.  **Fix: section 4 is
   re-derived** with the copy block priced by `LDGSTS` count at the a16
   calibration (F26: 28 `LDGSTS`, ~850 centre / 1100 pessimistic); the fp8
   pair is 1985 / 2640 issue cycles (wall-equivalent 2280 / 3030 against
   2544), i.e. consumer-bound at the centre with 10 % of margin and pacing
   at 336 us at the pessimistic end - "consumer-bound at both ends" is
   withdrawn, the bands are widened (fp8 289-336 / 296-344), the mixed centre
   is restated at parity (~305 / 313, band to 357) and its accept is not
   claimed.  The predicate-port attribution is withdrawn (3.2): the
   `dispatch` stalls are the LSU / MIO back-pressure of the `LDGSTS` stream,
   which fewer address instructions do not remove and fewer `LDGSTS` (3.3:
   36 -> 28) do.  New invariant C19 records the price.
3. **The landing-cover arithmetic was wrong.**  Rev 1 claimed a cover of "one
   pair minus the copy issue"; with its `DEPBAR` at the very top and the
   acquire spin after it, the cover was bodies + tail only (~1100-1300
   cycles) regardless of the consumer's pace, against a loaded latency the F25
   design puts at 1200-2000 - 0-900 cycles exposed per pair on the critical
   path.  **Fix: the spin precedes the `DEPBAR`** (3.1: the probes are issued
   at the end of the previous pair as non-blocking `mbarrier.test_wait` on
   the pipeline's empty barriers - `try_wait` may suspend and would hold the
   pending pair's arrives hostage - and the spins run first at the top), so
   the slack is absorbed before the wait: cover = the pair's wall time minus
   the pre-`LDGDEPBAR` part, **~1600 cycles = 0.81 us at parity, >= 1600 when
   the producer paces**; 3.1 states that this sits inside the 1200-2000
   range (0-400 cycles may be exposed), that no artefact measures latency at
   that cover, and makes trace 6.2 `wait` <= 0.05 us a hard stop with the
   fallback F26b' (F25's `DEPBAR` position, loads after it) priced as its own
   column in section 4 (fp8 ~2500 issue = ~318 us, fp4 ~312 us).  The fp4
   asymmetry argument re-checked for the new order (3.5): both operands'
   loads precede the pair's `LDGSTS` in both regimes.
4. **The copy address form was prescribed without a positive probe.**  All
   fifteen kernels in `f26_addr_forms.sass` carry >= 1 carry add per copy,
   including kT / kX / kZ - the exact rev 1 prescription; the a16 module's
   zero-add loop has its base pair formed once by `LEA / LEA.HI.X` from the
   `DTypeKV`-typed pointer arithmetic (`0xc130 / 0xc180`), a term ptxas cannot
   reassociate into the loop, which rev 1 did not identify; the fallbacks (i)
   / (ii) were guess-and-check and (iii) a ptxas scheduling outcome.  **Fix
   (3.2)**: the mechanism is named; the prescription is the a16 construction
   itself (`DTypeKV`-typed element offsets from the halved byte strides, host
   check for even strides, `page x stride` in C++); it is declared unprobed;
   the compile-only probe kV / kV2 replicating the a16 context is the first
   remote action of F26d; the count and the cost are stated for both outcomes
   (28 + 28 vs 28 + 56 + 28 issue slots, ~+100 cycles inside a pipe-bound
   block) so the band does not hinge on the fusion; the rev 1 fallbacks are
   withdrawn; gate 5 reports the form instead of rejecting on it.

Notes taken into the design (not blockers): the `__syncwarp` after the K'
vote's branch join is gone (both operands' landing loads precede the one
barrier and the copies; +24 registers, 3.10 with its fallback); the finish
site re-ballots the dynamic masks (3.6); the octet page-word `LDS.32` is
hoisted next to the chunk rows (3 fillers, 3.4); the octet scale copy's
wavefront claim is replaced by the row-contiguous slot layout whose count A7's
rule supports (3.3) and 6.3 reports the measured value; `op_ld` is stated at
~450 (3.9); the contention term c is widened to 2-10 % with the 6.3 reading
rule (section 4); the dyn copy blocks are accepted predicated or branched
(3.6, section 5); the hot-arm fall-through is reported, not gated; the C12
`IMAD.MOV` gate accepts up to 24 at the K arm join; the a16 loop site is
`[0xcb00,0xeab0]` and its loop copy blocks are the cited reference
throughout.

## 12. As written (filled per step; tests, F26a-c in this worktree, F26d not run)

### As written: tests (`tests/attention/run_fa3_mixed_page_transport.py`)

Not run in this worktree (no GPU; review by reading).  Two helpers and 14
cases appended to `main()` after the F25 matrix (exit code = failures; the
plain runner, no test framework):

- `_run_multi_chunk(mode, q_len, pages_per_req, kv_len, tiles)`: B = 2 items
  x H = 2 KV heads, one request per item of `pages_per_req` pages and `kv_len`
  tokens (partial tail page; both facts asserted in the helper: `(kv_len +
  95) // 96 == tiles`, `(pages_per_req - 1) x 16 < kv_len <= pages_per_req x
  16`), static fp8 / fp4 module or the dynamic module on the `mixed` cycle
  (page formats 0, 1, 2 repeating, so every tile carries 2 / 2 / 2), against
  the a16 module on the expanded reference, `torch.equal`.  Cases: (193 pages,
  3085 tokens, 33 tiles) and (130 pages, 2075 tokens, 22 tiles) x {fp8, fp4,
  mixed} x {q = 1, q = 64 causal} = 12.  In tiles (CTA_KV = 96, chunk = 16
  tiles, ring = 32 rows): 33 tiles = entries 0..32 -> chunk 0 (rows 0-15),
  chunk 1 (rows 16-31 = buffer 1), entry 32 -> row 0 (the wrap; the pair with
  V at entry 31 reads its K row at row 0), gathers at entries 0 / 16, stores +
  `BAR.SYNC` at 8 / 24, countdown 0 at 32; 22 tiles = entries 0..21 -> buffer 1
  rows 16-21 without a wrap, one gather / store, countdown 0 at 16.  For q =
  64 the causal tile count is unchanged (CTA_Q = 128 >= qo_len).
- `_run_dynamic_uniform(mode, q_len, extremes_gs=None)`: the F25 helper with an
  optional extremes transport (`_extreme_transport(..., gs)`); cases (fp8, g =
  1) and (fp4, g = 0.5) at q = 1, 18 pages (3 tiles), so that within one
  operand some warps' votes fail (448 / 256 scales at g = 1 for fp8; 448 x 0.5
  for fp4) and others pass while every slot of every tile runs that format's
  body: the per-slot predicated exact form of F26c is exercised on whole
  operands of both formats; on the F25 kernel it exercises the rolled cold
  loop the same way.

Total 118 cases (104 + 14).  **Order of running on nkcut2 (F26d): this commit
first, against the F25 kernel it ships with (expected 118 / 118: the F25
`entry / 16`, `entry % 16` addressing is the reference for the ring), then
F26a, F26b, F26c.**
