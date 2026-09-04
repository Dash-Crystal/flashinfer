# Mixed KV speed, round 4, Track F: lever [26] — the pair re-ordered around the copy issue

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

## 0. Summary

F25 moved the producer from IPC 0.10 to 0.23 and left the compressed modes at
fp8 402.8 / 414.9, fp4 422.2 / 429.7, mixed 650.0 / 664.4 us (q=1 / q=64) against
transport_a16 282.8 / 289.5 and a target of <= 330.  The F25 record read the
miss as "count 716 vs 655 and IPC 0.23 vs 0.27".  The per-segment pc samples
(section 1) say something sharper: **the two decode bodies (420 of the 716
instructions) run at IPC 0.52-0.59 and take 775 of the pair's 3113 cycles;
the other 296 instructions - copies, meta, acquires, vote preparation,
protocol - take 2340 cycles (IPC 0.13).**  Four mechanisms, each visible at
named PCs, account for that 2340:

1. the copy blocks (24 `IMAD.WIDE.U32` + 45 carry-chained `IADD3` / `IMAD.X` +
   24 `LDGSTS`, 12 chains per operand through 3-4 recycled predicate
   registers) cost ~9 cycles per instruction: 1160 cycles per pair;
2. the acquire `TRYWAIT`s, the chunk-table `LDS.128`s and the chunk-store
   `BAR.SYNC` are waited on where they are issued (~250 cycles);
3. the pre-vote chain (scale `LDS.32` -> `PRMT` -> `F2FP` -> `HADD2` -> `FMUL`
   -> `FSETP` -> `PLOP3` -> `VOTE`) is exposed once per operand, and the
   landing loads of the pair's first operand sit in the MIO queue behind the
   24 `LDGSTS` just issued (fp8 ~380 cycles; fp4 ~560, which is the
   "expK 0.68 vs expV 0.38" asymmetry);
4. the dynamic module's predicated copy body (492 instructions per pair, 72
   chains, 71 predicated `LDGSTS`) is dispatch-bound at IPC 0.16 (2900
   cycles) and its rolled format loops execute 640 instructions for 8 blocks.

**F26 = the same bodies, the same layout, the same barrier protocol, with the
pair re-ordered so that every latency of (2) and (3) is issued *before* the
copy block and consumed *after* it, the copy block reduced to one
`IMAD.WIDE.U32` (64-bit accumulator, no carry predicate) + one `LDGSTS` per
copy with the six scale-row copies of an operand issued by one lane-octet
instruction, the chunk-table row addressed as a 32-row ring (no divisions),
the static modules' pending words removed (the pending stages are the pair
counter), and the dynamic module's copies and decode unrolled per page slot
with the page format as a warp-uniform branch (ballot-derived masks) and the
hot / exact choice as a predicate on the 8 extra `HMUL2`.**  Per warp per pair
the fp8 count goes 716 -> ~612, fp4 740 -> ~636, mixed 1211 -> ~785; the
producer's time per pair is predicted at 1330-2040 cycles (fp8; 0.67-1.03 us)
against the consumer's 2544 (1.285 us), i.e. **consumer-bound at both ends of
the band**, so the compressed walls converge on transport_a16 plus the
shared-memory-pipe contention the compressed operands add (~+2-4 %):

| mode (us, q=1 / q=64) | F25 @ 40dcdc32 | F26 predicted (centre) | F26 band | accept |
|---|---|---|---|---|
| stock_a16 | 300.9 / 310.9 | unchanged | 297-304 / 307-314 | control |
| transport_a16 | 282.8 / 289.5 | unchanged (module untouched) | 281-286 / 287-293 | control |
| fp8 static | 402.8 / 414.9 | **293 / 300** | 288-312 / 295-320 | <= 330 |
| fp4 static | 422.2 / 429.7 | **295 / 302** | 289-318 / 296-325 | <= 330 |
| mixed (dynamic) | 650.0 / 664.4 | **300 / 308** | 290-338 / 297-346 | <= 330 at the centre; the upper band (copy body still dispatch-bound) misses |

**Is <= 330 reachable by F26 alone?**  For fp8 and fp4: yes, with margin - the
pessimistic end of the producer band (2040 cycles) is still 20 % under T_c, so
the wall is set by the consumer, and the accept row fails only if a mechanism
not in the F25 samples appears.  For mixed: yes at the centre, not at the
upper band; its risk is one item (the per-slot copy body's issue rate, section
3.7), decidable from the F26a SASS and one ncu run.  The floor for every
compressed mode is transport_a16's 283 / 290 plus the smem-pipe contention
(the compressed pair puts 1870 wavefronts on the LSU pipe against a16's 1118;
at T_c that is 74 % vs 44 %): **~288-295 us at q=1**.  Nothing in F26 lowers
T_c; nothing on the producer side can take a compressed mode below
transport_a16.

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
/ short_sb.  **The 12 carry chains of a copy block pass through three or four
predicate registers (P0, P1, P4), so an `IADD3` that writes P0 waits for the
`IADD3.X` that last read it: the chains are serialised 3-4 at a time and the
block issues one instruction per ~9 cycles.**  The a16 module's copy block
(sf0 `0xc6c0-0xc870`) has no adds at all: `IMAD.WIDE.U32 R30, R4, UR8, R16`
(page x uniform stride + the per-thread 64-bit base as the accumulator) then
`LDGSTS` - one instruction per copy - and its `LDGSTS` PCs show dispatch 42 %
too (gate 6.0), which is the `LDGSTS` issue cost itself, not a chain.

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

Changed (all inside `sparse_mixed_mainloop.cuh` except one host-side stride
widening in `Params`):

1. **Pair order** (3.1): the pending pair's own-landing loads, both operands'
   scale loads, the chunk-table rows and the two acquires are issued before
   the copy block; the votes and bodies follow it.
2. **Copy address form** (3.2): the a16 module's form - a pointer-typed
   per-item base built from 64-bit strides in `make_bases`, `page + stride`
   as C++ 64-bit arithmetic - so that every copy is `IMAD.WIDE.U32 Rd, Rpage,
   URstride, Rbase.64` + `LDGSTS` with no carry predicate.  Gated on SASS.
3. **Scale rows by one lane-octet instruction** (3.3): lane b < 6 of each row
   octet copies page b's 8 B scale row (one `LDS.32` of the row's page index,
   one `IMAD.WIDE`, one `@P LDGSTS.64` per operand).
4. **Protocol** (3.4): 32-row ring addressing of the chunk table (no `/ 16`,
   `% 16`), one pipeline counter for both rings, no pending words in the
   static modules (the pending stages are functions of the counter),
   acquires issued as `try_wait` early and spun only if not ready, the hot arm
   as the fall-through of the vote branch, the second `__syncwarp` placed
   before the V body.
5. **fp4 asymmetry** (3.5): removed by (1); no fp4-specific code.
6. **Dynamic module** (3.6): copies per page slot under a warp-uniform format
   branch (ballot-derived masks), decode per page slot under the same
   branches with one body per format whose exact (cold) form is 8 predicated
   `HMUL2` and a selected scale; no rolled loops, no `__ffs`.

## 3. The design

### 3.1 Data flow of one loop pair (static modules; dynamic in 3.6)

Pair t of the loop issues (K(t-1), V(t)) into stages (sK, sV) and finishes the
pair issued one iteration earlier, which sits in stages (sK', sV') = (sV,
sV - 1 mod 3) (3.4).  Program order per thread:

```
[top of pair t]
cp.async.wait_group 0               DEPBAR.LE SB0, 0x0: this thread's copies of pair t+1 (issued one pair ago) have landed
K' landing loads: 6 x 2 LDS.64      own landings of the pending K (fp4: 12 LDS.32)
__syncwarp                          (#1) every lane past its wait: lane b's landed scale slots (3.3) are readable by the row;
                                    every lane's K' landing loads are ordered before any lane's K' stores below (A7)
scale loads: 6 LDS.32 (K'), 6 LDS.32 (V')      both operands' scale words, immediate offsets from sc + stage x SCALE_STAGE
chunk rows: 4 LDS.128               pages of tiles t-1 (K) and t (V) from the 32-row ring (3.4); w7 unused in the loop
acquire K, acquire V                SYNCS.PHASECHK.TRYWAIT x 2 issued here; predicates consumed at the copies (3.4)
scale chains (both operands): 12 x {PRMT, F2FP.E4M3, HADD2, FMUL, FSETP}, PLOP3 trees     (issue interleaved with the copies)
[if !okK spin K] [if !okV spin V]   normally not taken: the try_wait results are ~120 cycles old by now
K copies: 6 x {IMAD.WIDE.U32 src.64 = page_j * UR ps + pbase.64 ; LDGSTS.128 [land8 + sK*STAGE + j*PAGE], [src.64], 16}
          1 x {LDS.32 page_b ; IMAD.WIDE.U32 ; @(b<6) LDGSTS.64 [sc_dst_b + sK*SCALE_STAGE], [ssrc.64], 8}     (3.3)
V copies: the same 6 + 1 into sV
cp.async.commit_group               LDGDEPBAR
K' vote: VOTE.ALL over the 6 FSETPs ; uniform BRA to the cold arm (hot arm = fall-through)
K' body: 6 x {F2FP.PACK sf2 ; 8 PRMT, 8 SHL, 8 LOP3 ; 8 HMUL2 ; 2 STS.128}          (after block 3: V' landing loads, 12 LDS.64)
__syncwarp                          (#2) every lane's V' landing loads before any lane's V' stores
V' vote ; V' body
fence.proxy.async ; arrive full[sK'] ; arrive full[sV']
pair counter advance ; ring offset advance ; loop
```

Why this order and not F25's (copies -> `LDGDEPBAR` -> `DEPBAR.LE 1` -> loads
-> `__syncwarp` -> scale loads -> vote):

- Every load whose latency F25 exposed (scale `LDS.32` -> vote chain, landing
  `LDS.64`, chunk-row `LDS.128`, acquire `TRYWAIT`) is issued before the copy
  block and consumed after it.  The copy block is 14 `IMAD.WIDE.U32` (FMA
  pipe, 2 issue cycles each), 14 `LDGSTS` (MIO, ~4-5 issue cycles each) and
  the interleaved scale chains: **>= 150 issue cycles of independent work**
  between the last load and the first consumer of any of them, against
  latencies of ~30 (`LDS`), ~70 (scale chain to `VOTE`), ~120 (`TRYWAIT`
  round trip), ~30 (`LDS.128` -> `IMAD.WIDE`).
- The pending pair's loads are issued while the MIO queue is empty (the
  previous pair's `LDGSTS` completed at the `DEPBAR`; this pair's have not been
  issued), which is what removes the fp4 K-operand excess (3.5).
- The brief's variant (`__syncwarp` -> 6 scale `LDS.32` -> 12 landing `LDS.64`
  -> vote, F25 order otherwise) hides the scale chain under 12 issue slots
  (~24-30 cycles of ~70) and keeps the loads behind the `LDGSTS` burst; and
  with the landing loads *after* the barrier a second `__syncwarp` is needed
  before the K stores anyway (A7: every lane's loads of a page before any
  lane's stores of it).  The order above puts the landing loads first, the
  barrier second, and hides everything under the copies.  Two `__syncwarp`
  per pair either way (F25 had two; both compiled to `NOP`).
- `wait_group 0` at the top of the pair instead of `wait_group 1` after the
  copies: the landing cover for pair t+1's copies is one pair minus the copy
  issue (~2544 - ~200 cycles at parity) instead of one pair plus it.  F25's
  trace `wait` is 0.01-0.02 us and the post-`DEPBAR` long_sb is 0.9 % (fp8) /
  2.0 % (fp4) with the longer cover; the loaded HBM/L2 latency the F25 design
  put at 1200-2000 cycles fits under 2300.  Gate 6.2 (`wait` <= 0.05 us, first
  post-`DEPBAR` PC long_sb <= 2 %); fallback if it fails: keep `wait_group 1`
  after the copies and use the brief's order (scale loads first, then landing
  loads, second barrier before the stores) - the acquires and chunk rows still
  move ahead of the copies.

The single-operand sites (K(last) after its issue, the drain V) keep F25's
form: `wait_group 0`, 12 landing loads, `__syncwarp`, 6 scale loads, exact
body, fence, arrive; the peeled pair issues copies only.

### 3.2 Copy address form (item 2 of the brief) - probe results and the prescription

What the hardware offers: `IMAD.WIDE.U32 Rd.64, Ra, URb, Rc.64` - a 32 x 32 ->
64 multiply-add with a 64-bit register-pair accumulator and a uniform-register
multiplier.  The a16 module's copies use exactly this form today (`sf0_mask1.sass
0xc6c0-0xc870`: `IMAD.WIDE.U32 R30, R4, UR8, R16` then `LDGSTS ... [R30.64]`,
twelve times, zero adds).  The compressed modules' `page_src` (`mad.wide.u32
d, page, stride, base64` in PTX) compiles to `IMAD.WIDE.U32 Rd, Rpage, UR, RZ`
+ `IADD3 lo, P, lo, base_lo` + `IADD3.X`/`IMAD.X hi` because ptxas never has
the base in an aligned pair.

Probe (`f26_addr_forms.cu`, twelve kernels, compile-only; the per-copy adds in
the copy block of each):

| kernel | base form | per copy |
|---|---|---|
| kA | today's: two chained asm `mad.wide.u32` from `ptr + b*16` | `IMAD.WIDE.U32 ..., RZ` + `IADD3` + `IADD3.X` (3) |
| kB | 32-bit page offset (`page * stride` u32) + 64-bit add | `IMAD` + `IADD3` + `IMAD.X` (3) - the carry predicate stays |
| kC | shfl-broadcast page term + per-thread 32-bit offset | `IMAD.WIDE.U32 Rd, Rpage, UR, R12.64` **fused** (the accumulator {toff, 0} is a wide-mad result) + 2 adds for the param pointer (3); no `R2UR`, no uniform datapath |
| kD | scale-row copy by one lane-octet (3.3) | 1 `LDS.32` + 1 chain + 1 `@P LDGSTS.64` for six rows (the chain itself as kA) |
| kJ | uniform 64-bit pointer + per-thread 32-bit offset | `IMAD.WIDE.U32 Rd, Rpage, R, R10.64` fused + `IADD3` + `IADD3.X`: **`LDGSTS` has no `[R.U32 + UR.64]` global form** |
| kM | base re-formed per pair by asm `mad.wide.u32 (rt, 1, hp)` | 3 (the asm result is split) |
| kY | base made opaque through `st.shared.v2.b64` / `ld.shared.v2.b64` | `IMAD.WIDE.U32 ..., RZ` + 2 adds although the base *is* an adjacent pair (R8:R9): ptxas fuses only accumulators that are its own wide-mad results |
| kT, kX, kZ | the a16 module's C++ pointer form (kX / kZ with int64 strides, kZ under 48 live registers) | `IMAD.WIDE.U32 Rd, Rpage, UR7, R22.64` **fused with one loop-invariant term**, then `IADD3` + `IADD3.X` of a **second** loop-invariant 64-bit term (3): ptxas keeps `ptr + head*hs` and `r*ts + b*16` as two 64-bit values and never pre-adds them; under pressure (kZ) it also copies the split pair into an aligned pair with two `MOV`s before the mad |

Reading: ptxas fuses a 64-bit accumulator into `IMAD.WIDE.U32` only when that
accumulator is the result of its own wide multiply-add tree; a base that
reaches the loop as an asm result, a loaded value or a sum of two such trees
is added in two halves with a carry predicate.  The a16 module's base is one
tree because its pointer term is folded into the per-thread multiply: nvcc
emits `mul.wide.s32 / mad.lo.s64` on the pointer from the `DTypeKV const*`
arithmetic in `make_bases` (`base + int64_t(a_r) * ts + a_c * CHUNK_ELEMS`,
strides int64 from `Params`), and `copy_a16_page` adds `uint64_t(page) *
uint64_t(b.a16_ps)` in C++.  **Prescription (F26a):** build `p8 / s8 / p4 /
s4` exactly as `a16_src0` is built - pointer-typed (`uint8_t const*`), the
span pointer plus `int64_t(head) * int64_t(hs)` plus `int64_t(row) *
int64_t(ts)` plus `blk * 16` in one expression, with the three strides widened
to `int64_t` in `Params::transport` by the host (`KVPageByteStrides` stays u32
on the device-facing API; `to_underlying_arguments` widens into a new
`int64_t` triple per span, four spans; the u32 page stride is kept as the
`UR` multiplier) - and `page_src` as `base + uint64_t(page) * uint64_t(stride)`
in C++, no asm.  Remove `compressed_base` and the asm `page_src`.  Then gate
(6.1): copy block = 14 `IMAD.WIDE.U32 ..., R.64` + 14 `LDGSTS` per operand pair,
**`IADD3` with predicate output = 0, `IADD3.X` = 0, `IMAD.X` = 0, `MOV` into
address pairs = 0** in `[first LDS.128, LDGDEPBAR]`.  If the gate fails on the
first build, the listed fallbacks in order: (i) form the base with the
per-thread part as the *innermost* term (`(ptr + blk*16) + row*ts + head*hs`)
and try the two orders; (ii) `__builtin_assume`-free trick that has worked in
the a16 module: keep the base in a struct member of pointer type and
dereference-cast only at the copy; (iii) accept the 3-instruction chain but
issue the 12 chains of an operand in three groups of four separated by the
scale-chain instructions (breaks the predicate-register serialisation:
`IADD3.X` reads P before the next group's `IADD3` writes it) - counted as +48
per pair and ~+250 cycles in the band.  The 32-bit page-offset form of the
brief is **not** the fallback: it keeps the carry predicate (kB) and adds a
host constraint (page x stride < 2^32 per span) for nothing.

### 3.3 Scale rows by one lane-octet instruction (protocol item)

Today each of the six pages costs one address chain and one `@leader
LDGSTS.64` (lanes b == 0 of the four rows of a warp: 4 active lanes, 1.98
wavefronts per instruction measured).  F26: lane b < 6 of every row octet
copies **page b's** scale row of its row r: source = `sbase.64 + page_b x
UR sts` where `page_b` is read by that lane directly from the tile's chunk-table
row (`LDS.32 [row + 4 b]`: 6 distinct words per octet, broadcast within, 1
wavefront), destination `sc_base + b x SCALE_PAGE + r x 8 + stage x
SCALE_STAGE` (a per-thread constant + immediates).  Per operand: 1 `LDS.32`, 1
`IMAD.WIDE.U32`, 1 `@P LDGSTS.64` (24 active lanes) instead of 6 chains + 6
`LDGSTS` (F25: 6 x (3 + 1) = 24; F26 fused: 6 x 2 = 12 -> 3).  Wavefronts: the
24 lanes write six 32 B segments in six 128 B lines (one per page slot) = 6
wavefronts against 6 x 1.98 today; global side unchanged (4 rows x 8 B per
page = one sector per page).  A9 ordering is unchanged in kind: every lane
waits for its own groups, the operand's `__syncwarp` #1 orders lane b's landed
slot of page b before the row's eight lanes read it (F25 relied on the same
barrier for lane 0's slot); the reads are the same six `LDS.32` at immediate
offsets.  kD shows the form compiles as written (`LDS R9, [R13+UR6]` ->
`@!P1 IMAD.WIDE.U32` -> `@!P1 LDGSTS.E.LTC128B.64`).  In the dynamic module lane
b's page may be FP8, FP4 or A16: two chains (`s8`, `s4` bases), two predicated
`LDGSTS.64` (`@(p8_b & b<6)`, `@(p4_b & b<6)`), predicates from the row's mask
word read by the same lane (`LDS.32 [row + 28]` broadcast) - 8 instructions per
operand.

### 3.4 Protocol (C14 restated) - back to <= 70 with the items named

F25's loop-site remainder was ~99 executed per pair (716 - 104 prep - 420 hot
- 93 copies); the F25 record's "~143" counted the 1/16-amortised gather /
store text as executed.  F26 items, per warp per pair:

| item | F25 (SASS) | F26 | how |
|---|---|---|---|
| chunk-table addressing | `entry / 16`, `entry % 16` three times (`SHF.R.S32.HI`, `LEA.HI`, `LOP3`, `SHF`, `IMAD.IADD` chains: ~25) + 2 x 2 `LDS.128` | ring offset `oV = (oV + 32) & 0x3FF` (2), K row = `(oV + 32) & 0x3FF` (2), 4 `LDS.128 [R + UR + imm]` -> **8** | entry e of chunk c lives in buffer c & 1, slot e % 16, i.e. at row `e & 31` of the 1 KB table (the [21] layout unchanged - the a16 module's immediates are untouched); `j == 0` is `(oV & 0x1E0) == 0`, `j == 8` is `== 0x100`; `next_chunk` a countdown decremented at `j == 0` |
| acquires | 2 x (`LEA`, `SHF`, `TRYWAIT`, `@!P BRA` to the spin) = ~10, latency exposed | the two `TRYWAIT`s issued after the loads, their predicates tested right before the copies (`@!P BRA spin`) -> **8**, latency hidden | `PipelineAsync::producer_acquire` is `barrier.try_wait` in a loop; F26 calls `producer_try_acquire` (CUTLASS has it) early and `producer_acquire` (the spin) only on failure |
| pipeline state | two `PipelineState` advances (`VIADD`, `ISETP`, `SEL`, `SEL`, `LOP3` x 2 = 10) | one counter (sV, phase_V): `VIADD`, `ISETP`, `SEL`, `LOP3`; `sK = sV + 1 mod 3` (`SEL`), `phase_K = sV == 2 ? phase_V ^ 1 : phase_V` (`SEL`) -> **6** | K's ring is exactly one stage ahead of V's for the whole item (K(last) alone advanced it once); the two `PipelineState` objects stay for `producer_commit` / `producer_tail` but are rebuilt from the counter at the commits |
| pending words | `LOP3 0x3FFFFFF`, `IMAD.SHL 0x40000000`, `LOP3` merge per operand + `SHF.R.U32.HI` extraction x 2 = ~8 | static: none - pending K stage = sV, pending V stage = previous sV (one `MOV`) -> **1**; dynamic keeps F25's words (masks) -> 8 | the static finish needs only the stage; `TileRegs::pending_word` stays for the dynamic module |
| expand bases | 2 x (`IMAD` x 3 stage bases, `LOP3` x 2 halves) = ~12 | same **10** | |
| copy destination bases | 2 x 2 `IMAD` | **4** | |
| `LDGDEPBAR`, `DEPBAR`, 2 `__syncwarp` (`NOP`) | 3 | **4** | |
| fence + commits | `MEMBAR.ALL.CTA` + `FENCE.VIEW.ASYNC` + 2 `LEA` + 2 `SYNCS.ARRIVE` = 6 | **6** (unchanged; the `MEMBAR` is what `fence.proxy.async` lowers to with the fence) | |
| loop, chunk gather / store amortised | ~10 + 35/16 | `t` decrement, compare, `BRA`, two `j` tests, countdown = 7 + 3 -> **10** | |
| `@!PT LDS RZ, [RZ]` fillers | 6 (two `LDGSTS` groups per pair) | **3** (one group: all chunk rows are read before the first `LDGSTS`) | ptxas emits three predicated-off `LDS` before the first `LDGSTS` that follows an `LDS`; they occupy issue slots and count in `inst_executed` |
| **total** | ~99 executed (143 in text) | **~60** | budget <= 70 |

The chunk store and its `BAR.SYNC` (every 16 pairs, ~50 cycles per pair
amortised, `barrier` stalls 47 % at its `@!P1 BRA`) are kept; moving the store
to the pair after the K body would hide it but changes the [21] WAR argument
(`CHUNK_STORE_PAIR`) and is not needed for the budget.

**Barrier protocol (C4, C7, A5) unchanged**: `PipelineAsync` with producer
arrival count 128 / consumer 256, `full_barrier` arrive per thread after the
fence, K(last) finished before `barrier_O.wait`, `kQueryEmpty` 384, chunk-table
group barrier 128.  What moves is only *when* the `try_wait` is issued and
*where* the `wait_group` sits; the set of waits per pair is the same (one
`wait_group`, two acquires, two arrives, one fence).

### 3.5 fp4 expK / expV (item 4 of the brief)

Attribution in section 1: MIO queue order plus a WAR on the landing-load
address register.  F26 issues the pending pair's 18 (fp4) / 18 (fp8) loads of
both operands' scale words and the K operand's landing words before this
pair's `LDGSTS`, so no load of the pair waits behind a `LDGSTS` burst, and the
V operand's landing loads are issued during the K body (after block 3), when
the pair's `LDGSTS` have long left the queue.  Predicted: fp4 expK = expV =
0.37-0.39 us on the trace, fin 0.76-0.80 (from 1.14-1.20); the fp4 pair loses
~560 cycles by this alone.  No fp4-specific code; the fp4 landing (8 B `.ca`
copies, `LDS.32` halves) is A7's, unchanged - a 16 B fp4 copy form remains
impossible (8 B alignment, F25 3.4).

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
0, 13 `BRA`, ptxas turned the smallest blocks into predication).  Per operand:

- copies (`issue_tile_copies`, dynamic arm): for each slot j (unrolled): `if
  (m8u bit j) {fp8: IMAD.WIDE + LDGSTS.128 [land8 + j*PAGE]} else if (m4u bit
  j) {fp4: IMAD.WIDE + LDGSTS.64 [land4 + j*PAGE]} else {a16: 2 x (IMAD.WIDE +
  LDGSTS.128) [a16_dst + j*PAGE (+ATOM)]}`; six bases (`a16_src0/1`, `p8`,
  `p4`, `s8`, `s4`) in the a16 form of 3.2; then the octet scale copy of 3.3
  (8 instructions).  Executed per operand on the bench mix (2 / 2 / 2): 6 x
  (2 tests + 1-2 `BRA`) + 2 x 2 + 2 x 2 + 2 x 4 + 8 + fillers (3 per
  `LDGSTS` block that follows an `LDS` on its path: up to 3 x 6) ~= **~60-75**
  (F25: 246).  The partial arm (two per-item calls) adds the D4 src-size
  register per copy as F25.
- decode (`expand_operand`, dynamic arm), in the 3.1 order: landing loads per
  slot under the same branches (fp8 2 `LDS.64`, fp4 2 `LDS.32`, a16 none) ->
  `__syncwarp` -> 6 scale `LDS.32` -> per page one `f32(s_j)`, `gsel_j = in8_j ?
  gs8 : gs4` (`SEL`), one `FMUL`, one `FSETP`, folded into `ok8` / `ok4` by the
  format bit (2 `PLOP3`) -> two `VOTE.ALL` -> `Pc8 = !hot8`, `Pc4 = !hot4`;
  `g8 = Pc8 ? g8_plain : gs8`, `g4` likewise (2 `SEL`; the plain globals are
  loaded once per item into two registers - the cold path no longer reloads
  them) -> per slot j: `if (m8u bit j) {fp8 block: sf2 = bf16x2(f32(s_j) *
  g8) ; decode ; @Pc8 8 x HMUL2 by 2^120 ; 8 HMUL2 by sf2 ; 2 STS.128} else if
  (m4u bit j) {fp4 block likewise with Pc4, 2^126}`.  One body per format per
  slot (12 bodies per operand instead of 8 rolled loop variants per site);
  the exact form costs 8 predicated-off `HMUL2` issue slots per block on the
  hot path (+16 % of a block) and is bit-identical to F25's cold arm
  (`bf16x2(f32(s) g)` after the 2^k un-fold).  Per operand on the bench mix: 4
  blocks x (35 + 8 + 5) + 12 tests / branches ~= **~200**; prep ~60.
- protocol: F25's per-operand pending tests (4 warp-uniform `if
  (op.pending)`), pending words with masks (8), everything else as 3.4: ~70.

Per pair on the bench mix: copies ~130 + landing loads ~16 + prep ~120 +
decode ~400 + protocol ~70 + fillers ~40 = **~785** (F25 1211).  Code: loop site
~2 x (75 + 60 + 12 x 52) + 70 ~= 1700; the two single-operand sites ~700 each
(exact bodies only, `Pc` = true, no vote); prologue ~250 -> **region ~3400
instructions (54 KB) from 5528 (88 KB)**; hot footprint per pair ~800
instructions (13 KB) from ~1200 (19 KB) - the `no_inst` 10 % is expected to
fall to the static modules' 2-3 %.  The mask-driven table build
(`chunk_store`'s `REDUX.OR`, 1 `BRA.DIV` per 16 pairs) is unchanged.

Why not predication for the copies (C17's letter): with fused chains,
predication per slot is 4 `IMAD.WIDE` + 4 `@P LDGSTS` + 3 predicate producers
= 11 issue slots and 3 predicate writes per slot; the branch form is ~6 issue
slots and 2 predicate writes.  Both are admissible; the branch form is chosen
because the F25 dyn copy body's dispatch stalls sit on the predicate producers
(section 1), and it is what the decode needs anyway.  C17 is amended (section
7): "no per-page branch" -> "no *divergent* per-page branch; uniform branches
on ballot-derived masks are the form".

### 3.7 Per-warp per-pair instruction budgets by class (executed; from the F25 SASS)

| class | fp8 F25 (`loopsite_fp8.log`, executed) | fp8 F26 | fp4 F26 | dyn F26 (bench mix) |
|---|---|---|---|---|
| landing loads (`LDS.64` / `LDS.32`) | 24 | 24 | 24 | 16 (+12 slot tests) |
| scale loads (`LDS.32`) | 12 | 12 | 12 | 12 |
| scale chains (`PRMT`, `F2FP.E4M3`, `HADD2`, `FMUL`, `FSETP`) + `PLOP3` + `VOTE` + `BRA` | 68 | 68 | 68 | ~96 (`SEL` per page, two accumulators, 4 `VOTE`, 4 `SEL`) |
| hot bodies (`F2FP.PACK`, 8 `PRMT`, 8 `IMAD.SHL`/`SHF`, 8 `LOP3`, 8 `HMUL2`, 2 `STS.128` per block) | 420 (2 x 210) | 420 | 444 (2 x 222) | 8 blocks x 43-44 + 64 `@P HMUL2` + 24 tests / `BRA` = ~440 |
| payload copies (`IMAD.WIDE.U32` + `LDGSTS`) | 24 + 24 + 45 adds = 93 (incl. scale copies) | 12 + 12 = 24 | 24 | 8 x 2 + 4 x 2 (a16 rows) + 12 tests + 12 `BRA` = ~48 |
| scale-row copies (3.3) | (in the 93) | 2 x 3 = 6 | 6 | 2 x 8 = 16 |
| fillers `@!PT LDS RZ` | 6 | 3 | 3 | ~36 |
| meta rows (`LDS.128` x 4 + ring) | ~25 | 8 | 8 | 8 |
| acquires, state, pending, bases, `DEPBAR`s, fence + commits, loop | ~74 | ~52 | ~52 | ~64 |
| **total** | **716** (ncu 716) | **~612** | **~636** | **~785** |

Pipe mix at parity (per SMSP per pair of 2544 cycles), fp8 F26: producer ALU
(`PRMT`, `LOP3`, `FSETP`, `PLOP3`, `ISETP`, `SEL`, `IADD3`) ~440 x 2 cycles =
880; FMA (`HMUL2` 96, `IMAD.SHL` 96, `IMAD.WIDE` 14 x 2, `IMAD` ~20, `FMUL` 12)
~250 x 2 = 500; XU (`F2FP` 24) ~200; LSU issue ~60 x 4 = 240.  Consumer (2
warps, 826 instructions): FMA ~330, ALU ~200, XU ~1200.  Totals: ALU 42 %,
FMA 33 %, XU 55 %, issue (612 + 826) / 2544 = 57 %.  Unchanged in kind from
F25 (the decode is the same); nothing saturates.

### 3.8 Dependency depth of the pair (what is on the critical path)

- To the K' `VOTE`: `DEPBAR` (~20 when the group is complete) -> `LDS.32` scale
  (~30) -> `PRMT` (4) -> `F2FP.E4M3` (~8) -> `HADD2` (4) -> `FMUL` (4) ->
  `FSETP` (4) -> `PLOP3` x 2 (8) -> `VOTE.ALL` (~10) -> `BRA`: **~90 cycles of
  chain**, of which F25 exposed ~70 (the `PRMT` PC alone 118 cycles as
  short_sb + the chain's `wait`).  In F26 the copy block (>= 150 issue cycles)
  is between the `LDS.32` and the `VOTE`: exposed 0; the `VOTE` -> `BRA` (~10)
  remains.
- To the first K' `STS`: `VOTE` -> `F2FP.PACK sf2` (~8) -> `HMUL2` (4) -> `STS`
  ~= 15 after the vote; the decode chain of block 0 (`LDS.64` landed long ago
  -> `PRMT` -> `SHL` -> `LOP3`, 12) runs before the vote resolves.
- To the first V' `STS`: V' landing `LDS.64` issued after K' block 3 (~200
  cycles before the V body) -> `__syncwarp` #2 (`NOP`) -> `PRMT`...; V' scale
  chain done under the copies: exposed 0 (F25 exposed the 12 `LDS.64` + 6
  `LDS.32` + chain: 274 cycles in the V prep segment).
- Copy block: chunk rows `LDS.128` (~30) -> `IMAD.WIDE.U32` (~6) -> `LDGSTS`:
  the rows are read before the acquires' predicates are tested, so the first
  `IMAD.WIDE` finds its page index ready.  The 14 chains of an operand are
  independent (no predicate coupling in the fused form); their issue is the
  block's length.
- Acquire: `TRYWAIT` (~120 round trip) -> `@!P BRA spin`: issued before the
  scale chains, tested after them: exposed 0 when the stage is free; when the
  consumer has not released, the spin is the consumer wait (desired).
- Tail: `FENCE.VIEW.ASYNC` waits for the 12 V' `STS.128` to perform (~50-70
  cycles, long_sb 68 % of its samples): kept (one fence per pair); a
  per-operand fence + K commit right after the K body would give the consumer
  K ~500 cycles earlier at +1 `FENCE` and ~+40 exposed cycles - listed as an
  optional item, built only if 6.3's consumer K-wait stays > 3 % with the
  producer pair under T_c.

### 3.9 Shared-memory wavefront budget (C11)

Per pair per SM, fp8: wgmma operand reads 1001 (unchanged); `STS.128` 96 x 4.00
= 384 (unchanged, ideal at every PC); landing `LDS.64` 96 x 2 = 192, scale
`LDS.32` 48 x 1 = 48, chunk-row `LDS.128` 16 x 2 = 32, octet page-index
`LDS.32` 8 x 1 = 8 (F25 `op_ld` 441 -> ~280); `LDGSTS.128` 48 x 4 = 192,
scale-row `LDGSTS.64` 8 x 6 = 48 (F25 96 x ~2 = 190 -> 48; total `LDGSTS` 281 ->
~240); non-`STS` `op_st` residual ~136 (unattributed, unchanged).  **Total
~1900-2000 wavefronts per pair = 75-79 % of 2544** - the same pipe share as
F25 / F24 (C11 <= ~2000 holds); fp4 ~1960 (landing `LDS.32` 96 x 1, payload
`LDGSTS.64` 48 x 3.98); dyn ~1950.  This share is what the consumer sees as
contention on its 1001 gmma-read wavefronts and is the reason the compressed
walls are predicted 2-4 % above transport_a16 rather than equal to it.

### 3.10 Registers (C3)

Producer 136.  Live set at the widest point of the F26 order (during the copy
block): K' packed words 24 + K' scale words / products 12 + V' scale words /
products 12 + copy sources in flight (up to 14 pairs, but ptxas will keep ~4-6
= 12) + chunk pages 12 + bases (`p8`, `s8` pairs 4; `out0/1`, `land8`, `sc_rd`,
`a16`-none 4; expand bases 2 x 5 = 10) + strides in UR + counter / phase /
ring / loop ~8 + item constants ~10 = **~104**.  During the K' body after block
3: K' remaining packed 12 + decoded in flight ~24 + `sf2` 6 + V' packed 24 + V'
`v_j` 6 + bases ~30 + protocol ~10 = ~112.  Under 136 in both; ptxas -v gate:
no C7507, `STACK 0`, `LDL`/`STL` 0.  Dynamic module: the per-slot branched
bodies hold one block's decode at a time; the six scale words + six `f32(s_j)`
+ 2 masks + 8 bases (a16 pairs 2 x 2, compressed pairs 4 x 2) ~= 30 more than
static's constants -> ~120; same gates.  Consumer 184, untouched.

## 4. Predicted producer time and walls

**Method**: F25's per-segment cycles (section 1) carried forward with each
mechanism's removal argued from its PCs, not from an assumed IPC:

| segment | F25 fp8 cycles | F26 fp8 (centre) | F26 (pessimistic) | argument |
|---|---|---|---|---|
| protocol outside the copies (index arithmetic, acquires, chunk rows, chunk barrier, state, pending) | ~385 (A minus its copy part) | 120 | 220 | acquire and `LDS.128` latencies hidden (issue before the loads' consumers); ring addressing removes the division chains; chunk `BAR.SYNC` amortised 50 stays |
| copy blocks (K + V) | 1160 (2 x 580 for 2 x 65 instructions) | 180 | 600 | fused form: 28 `IMAD.WIDE` + 28 `LDGSTS` + 6 with no predicate coupling, issue-bound at ~4-5 cycles per `LDGSTS`; pessimistic = the 3-instruction chain kept but de-serialised (3.2 fallback iii) |
| `DEPBAR`, loads, pre-vote chains, `VOTE` (K') | 382 | 80 | 160 | 36 loads' issue ~40 + `DEPBAR` ~20 + `VOTE` -> `BRA` ~20; pessimistic: half the scale chain still exposed |
| K' hot body | 395 | 395 | 430 | unchanged code, measured IPC 0.53 |
| V' prep | 274 | 90 | 150 | 12 landing loads issued under the K body; the vote's ~20 + `__syncwarp` |
| V' hot body | 380 | 380 | 420 | unchanged, IPC 0.55 |
| cold-arm fetch after the taken branch | 66 | 20 | 40 | hot arm as fall-through |
| tail (fence, arrives, loop) | 68 | 68 | 80 | unchanged |
| **pair** | **3113** (IPC 0.23) | **~1330 = 0.67 us (IPC 0.46)** | **~2100 = 1.06 us (IPC 0.29)** | both under T_c = 2544 |

The IPC is not assumed: the bodies' 0.53-0.55 is measured and unchanged; the
non-body part goes from 296 instructions in 2340 cycles (0.13) to ~192
instructions whose four stall mechanisms are each removed by construction
(carry-predicate serialisation, exposed load latencies, MIO queue order,
division chains); at the bodies' issue efficiency the 192 cost ~380 cycles
plus the latencies that stay exposed (~200: `DEPBAR`, two votes, fence, chunk
barrier), which is the centre; the pessimistic column keeps the copy chain at
3 instructions and half of the exposed latencies.  fp4: + 24 issue slots and
the same structure -> 1370 / 2160 cycles.  Dynamic: bodies 8 x ~50 at 0.5 =
800 + slot branches ~100 + prep 120 at 0.35 = 340 + copies ~130 at 0.3 = 430 +
protocol 200 = **~1870 centre** (0.94 us); pessimistic (copy body at F25's
dispatch-bound 0.16: 800 cycles; no_inst 8 %) ~2800 = 1.41 us -> **above T_c**:
that is the mixed upper band.

**Walls**: consumer-bound modes land at `T_c(1 + c) x 220 + fixed` where c is
the smem-pipe contention the compressed operands add to the consumer's gmma
reads (74 % vs 44 % pipe share; the F24 / F25 records show tensor-pipe active
falling with the producer's LSU traffic, but no run has measured c with a
non-pacing compressed producer; 2-4 % is the design's allowance and gate 6.3
measures it): fp8 **288-300 / 295-308**, centre 293 / 300; fp4 289-302 /
296-310, centre 295 / 302; mixed: centre 300 / 308 (c ~4 %, plus ~1 % for the
A16-tile commits), upper band 338 / 346 where the producer paces at 1.41 us.
If the producer's pessimistic fp8 column paced (it does not: 2100 < 2544), the
wall would be 283 x 2100 / 2544 -> still consumer-bound; the first wall that
would show producer pacing is a pair >= 2544 cycles, i.e. the F25 non-body
part shrinking by less than 44 % - which would mean the carry-chain removal
and the load re-ordering both failed, and 6.1 would have said so.

**Accept / reject rows (6.4)**: fp8 <= 312 / <= 320 accept (band), 313-330
accept on the target but reject the centre (re-read 6.3: which segment did not
move), > 330 reject; fp4 <= 318 / <= 325 likewise; mixed <= 330 accept, 331-346
= the copy-body IPC risk materialised (build the predicated-copy alternative
of 3.6, no other change), > 346 reject (count model wrong again: re-read the
dyn SASS).  transport_a16 must reproduce 281-286 / 287-293 (module untouched)
or the session is offset.

## 5. SASS gates before any timing (F26a build; `cuobjdump -sass`, producer region `USETMAXREG.DEALLOC .. EXIT`, loop site = the back-edge body with the two / four loop `VOTE`s; `f25_counts.py`, `f25_loopsite.py`, `f26_seg.py`)

| gate | accept | reject -> action |
|---|---|---|
| `USETMAXREG` `DEALLOC 0x88` / `TRY_ALLOC 0xB8`; `ptxas -v` no C7507; `STACK 0`; `LDL`/`STL` 0 | as F25 for fp8, fp4, dyn; a16 `0x48` / `0xD8` | any: reduce live ranges (V' scale chains after the copies) before timing |
| **copy block** `[first LDS.128, LDGDEPBAR]` (static loop site) | `IMAD.WIDE.U32 Rd, R, UR, R.64` 14 per operand (12 payload + 1 scale + 1 for the other operand's... total 28 per pair), `LDGSTS` 28 (12 `.128` / `.64` payload + 2 `@P .64` scale per pair), **`IADD3`/`IADD3.X`/`IMAD.X` with predicate carry = 0, `MOV`/`IMAD.MOV` into address pairs = 0**, `@!PT LDS RZ` fillers 3, `SEL` 0 | adds present -> 3.2 fallbacks (i), (ii) in order, then (iii) with the count re-stated as +48 and the band's pessimistic column |
| **order** (3.1) | in the loop site, in program order: `DEPBAR.LE SB0, 0x0` -> 12 landing `LDS` -> `NOP`/`WARPSYNC` (#1) -> 12 scale `LDS.32` -> 4 `LDS.128` -> 2 `SYNCS.PHASECHK.TRYWAIT` -> (chains interleaved) -> copies -> `LDGDEPBAR` -> `VOTE.ALL` -> hot arm as fall-through (the `@P BRA` after the vote targets the cold arm) -> 12 `LDS.64` inside the K arm -> `NOP` (#2) -> `VOTE.ALL` -> V arm -> `MEMBAR` + `FENCE` + 2 `SYNCS.ARRIVE` | any `LDS` of the pending pair after the first `LDGSTS`; a `DEPBAR.LE SB0, 0x1`; the cold arm as fall-through -> restructure |
| scale-row copies (3.3) | per operand exactly one `LDS.32 [R + UR + imm]` of the row's page word (lane-indexed) and one `@P LDGSTS.E.LTC128B.64` with 24 active lanes; no `@P` on payload `LDGSTS` in the loop | six `LDGSTS.64` per operand -> the gather did not fold |
| protocol (3.4) | loop-site count - 24 - 12 - 68 - 420 - 30 - 3 <= 70; `SHF.R.S32.HI` / `LEA.HI` division chains 0; `SYNCS.PHASECHK.TRYWAIT` 2 issued before the first `IMAD.WIDE.U32` of the pair | over -> list the opcodes |
| bodies (C12) | hot 210 / cold 267 (fp8), 222 / 279 (fp4) unchanged; `VOTE.ALL` 2 in the loop site, 0 in bodies; `BRA.DIV` 0; `UMOV` 0 in bodies | any change to the bodies is a mistake (F26 does not touch them) |
| region | fp8 <= 2450, fp4 <= 2550 (F25 2584 / 2664 minus the copy adds and protocol); dyn **<= 3600** (from 5528) | dyn > 4000 -> the per-slot bodies were not shared per format or the single-operand sites carry votes |
| dyn copy body | per operand 6 slot blocks, each `ISETP`/`LOP3 P` on a `VOTE.ANY`-derived register + `BRA`; `IMAD.WIDE.U32 ..., R.64` fused inside; `BSSY`/`BSYNC` 0 in the copy and decode bodies; `FLO`/`BREV` 0 in the loop site; `LDGSTS` sites 6 x (1 + 1 + 2) + 2 = 26 per operand; `VOTE.ANY` 2 per operand | `BSSY` > 0 -> the masks were not ballot-derived (ptxas does not know they are uniform) |
| dyn decode | 12 bodies per operand in the loop site (fp8 / fp4 per slot), each with 8 `@P HMUL2` + 8 `HMUL2`; `VOTE.ALL` 4 in the loop site; the single-operand sites: 12 bodies with 16 unpredicated `HMUL2`, `VOTE` 0 | rolled loop (`BRA` back-edge inside a body) -> C10 violated |
| a16 module and stock kernel | a16 SASS identical to the F25 a16 stream (`sf0_mask1.sass`, 3832 / 3880 instructions; the four-UR permutation against `5cc416fd` is accepted as in F25); stock byte-identical | any a16 change: the `STATIC_A16` arms were touched |

Gate 6.0 (a16 `LDGSTS` PCs) is not repeated: it passed in F25 and F26 does not
change the copy form the a16 module already has.

## 6. Verification (after the gates; confirmation, not tuning)

- **6.1 tests**: `tests/attention/run_fa3_mixed_page_transport.py`, the F25
  104 cases (the NaN-tail cases exercise the partial arms of the two per-item
  calls, whose copy forms change too) + 2 new: a dynamic tile whose six pages
  are all FP4 with `g = 0.5` (the `Pc4` predicated exact form through the
  per-slot bodies) and the many-items case with `kv_len = 16 x 32 + 1` (33
  rows of the ring: the wrap at `oV = 0x3E0 -> 0`).  106 / 106 bit-exact,
  pytest not used.
- **6.2 trace** (`MIXED_FA3_TRACE`, q=1, CTA 0 items 0/1; ratios, not absolutes):
  fp8 `iss` (now the copy block only) <= 0.15 us (from 0.38); `fin` = wait /
  expK / expV: **expK == expV within 0.03 us in fp4** (from 0.68 / 0.38), fp8
  expK + expV <= 0.80 (from 0.88); `wait` <= 0.05 (the `wait_group 0` cover) -
  reject > 0.1: fall back to the brief's order with `wait_group 1` (3.1);
  mixed `iss` <= 0.45 (from 1.63-1.78).
- **6.3 ncu** (fp8 / fp4 / mixed, q=1, third launch, `f25_run_ncu.sh` +
  `f26_pc.py --seg`): producer `inst_executed` per pair 2450 +- 5 % (fp8),
  2540 (fp4), 3140 +- 8 % (dyn); copy-block segment <= 8 % of loop-site
  samples (from 31 + 19); the `PRMT`-after-scale-`LDS` PC < 0.5 %; `dispatch`
  <= 12 %; `no_inst` <= 3 % (dyn); **consumer K-wait PC <= 3 %** (the
  consumer-bound proof); consumer `inst_executed` 3301; tensor-pipe active
  within 5 % of a16's 67.1 %; `l1tex` shared wavefronts per pair 1850-2000 by
  class (`op_ld` ~280, `op_st` ~520 with `STS.128` at 4.00 per PC, `LDGSTS`
  ~240); the non-`STS` `op_st` residual attributed by PC (follow-up).  Reject:
  K-wait 3-8 % with the producer pair < T_c in the trace = the contention
  term c is larger than 4 % (report it; nothing on the producer side fixes it);
  K-wait > 8 % = the producer paces: the segment table names the segment.
- **6.4 bench** (`bench_fa3_mixed_page_transport.py --q-lens 1 64 --repeats 1
  --trials 5`, nkcut2 lock, co-tenant rule): the section 4 rows.

## 7. Invariant amendments to write into the dataflow document when F26 lands

- **A10 (pair order; supersedes A9's ordering sentence).**  Per loop pair:
  `cp.async.wait_group 0`; the pending K's landing loads; `__syncwarp` (#1);
  both pending operands' scale-slot loads; the two chunk rows; the two
  `try_wait`s; the copies; `commit_group`; K vote and body (the pending V's
  landing loads issued inside it); `__syncwarp` (#2); V vote and body; fence;
  commits.  Ownership unchanged (A7 / A9).  Scale slots: lane b < 6 of a row
  octet copies page b's row (one instruction per operand); the row's eight
  lanes read all six slots after barrier #1.
- **C13 (copy addressing), restated.**  Every compressed source is one
  `IMAD.WIDE.U32` with a 64-bit register-pair accumulator (the a16 module's
  form): pointer-typed per-item bases from int64 strides in `make_bases`,
  `page x stride` in C++; no asm, no carry-predicate adds in the copy block
  (SASS-gated).
- **C14 (protocol), restated.**  <= 70 per pair per warp, itemised as 3.4;
  chunk-table rows addressed as a 32-row ring (`row = e & 31`); one pipeline
  counter for both rings (`sK = sV + 1 mod 3`); static modules carry no
  pending word (pending stages = `sV`, previous `sV`); acquires as early
  `try_wait` + late spin.
- **C17 (dynamic copies), amended.**  Per page slot, a warp-uniform branch on
  ballot-derived format masks selects one of three straight-line copy forms;
  no rolled loop, no divergent branch, no `SEL` on an address; the scale
  rows by the lane-octet form with per-lane format predicates.  **C10**'s
  rolled decode loops are withdrawn: per-slot bodies, one per format, exact
  form by predicate.
- **C11.**  Unchanged (<= ~2000 wavefronts per pair); F26 reports `LDGSTS` ~240
  and `op_ld` ~280 as the new expected values.
- **C18 (new, store pattern).**  `STS.128` at 4.00 wavefronts per instruction
  at every PC (ncu source view) is the acceptance; the aggregate `op_st /
  STS.128` ratio is not a gate (it counts non-`STS` traffic).

## 8. Do not build

1. **Any change to the decode bodies** (placement decodes, `HMUL2` count,
   store order, chunk permutation): measured at IPC 0.53-0.59 and 4.00
   wavefronts per `STS.128`; they are 25 % of the pair and not the problem.
2. **The 32-bit page-offset copy form** (brief item 2, first option): kB shows
   it keeps the carry predicate (`IMAD` + `IADD3` + `IMAD.X`) and it needs a
   host bound for no gain.
3. **Asm `mad.wide.u32` for the bases or the page term** (F25e's form): kA / kM
   show ptxas splits an asm-formed 64-bit accumulator; `ld.shared.b64`-opaque
   bases likewise (kY).
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
12. **Any GPU timing before the section 5 gates pass** on the F26a build; in
    particular no bench of a build whose copy block still carries carry
    adds "to see": the F25 band was missed by exactly that mechanism.

## 9. Files touched (F26a-d; each step bit-exact on its own; a16 and stock untouched throughout)

- **F26a - copies and protocol** (`sparse_mixed_mainloop.cuh`; `Params` /
  `to_underlying_arguments` for the int64 stride triples): `OperandBases`
  bases as `uint8_t const*` built in the a16 form; `page_src` in C++;
  `compressed_base` removed; the lane-octet scale copy
  (`copy_scale_rows(b, row_addr, stage)`); `read_meta` by ring offset
  (`meta_row(o)`), `pair_step` with `oV`, the `j` tests and the chunk
  countdown; one `(sV, phase)` counter with `PipelineState` rebuilt for
  `producer_commit` / `producer_try_acquire` / `producer_tail`; static pending
  stages from the counter (`Operand::pending` kept for the dynamic module
  only); `try_wait` early / spin late.  Gate: section 5 copy / protocol rows;
  tests 106.
- **F26b - pair order** (`finish_pending_pair` folded into `produce_pair` as
  `finish_begin` (wait, K' loads, barrier, scale loads, chains) before the
  copies and `finish_end` (votes, bodies, V' loads, barrier, fence, commits)
  after them; the single-operand sites keep `finish_one`).  Gate: order row;
  trace 6.2.
- **F26c - dynamic module** (`issue_tile_copies` dynamic arm per slot with
  ballot masks; `expand_operand` dynamic arm per slot; `expand_block<FP8,
  EXACT>` gains a runtime-predicate form `expand_block_pred<FP8>(..., Pc)` for
  the dynamic bodies - the static modules keep the compile-time `EXACT`;
  `expand_format_pages` removed).  Gate: dyn rows; tests incl. the two new
  cases.
- **F26d - measurements** in the section 5 / 6 order; results appended here
  and to the backends document; dataflow amendments of section 7.

## 10. The floor, stated

With the bodies unchanged, the producer's pair is bounded below by the two
hot bodies (775 cycles at their measured IPC) plus the loads, votes, copy
issue and commits (~550 at the bodies' issue efficiency) = ~1330 cycles; the
consumer's pair is 2544.  F26 therefore does not need the producer to be
fast; it needs it to stop pacing, which the F25 samples say is a matter of
four named mechanisms outside the bodies.  Once the producer is under T_c
the wall is transport_a16 (283 / 290) plus the compressed operands' share of
the shared-memory pipe (~75 % vs 44 %), estimated at +2-4 % and measured by
6.3's tensor-pipe row; **~288-300 us at q=1 is the floor of every compressed
mode on this kernel**, and <= 330 is met with margin for fp8 / fp4 and at the
centre for mixed.  What F26 cannot do: lower T_c (the consumer is FA3's, C5),
remove the smem-pipe share (BF16 materialisation, F25 section 9), or make
the mixed mode faster than the static ones (its per-slot branches and
predicated exact form cost ~+10 % of issue slots over fp8 at equal blocks).
If 6.3 shows the producer under T_c and the wall above 300, the residual is
c (the consumer's smem contention), and the next lever is on the consumer's
side of C5, not on the producer.
