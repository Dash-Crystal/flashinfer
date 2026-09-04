# Mixed KV speed, round 3, Track F: lever [25] — what reaches <= 330 us with smem-materialised BF16 operands

Design (revision 2 after the judge review, section 10) and the implementation
as written (section 11).  Base: `claude/mixed-kv-sm90-tma` @ `64a70b9c` (merge
of wt/F24).  Line numbers in sections 1-9 are those of `64a70b9c`.  This
document extends `docs/mixed_kv_speed_round2_fa3_consumer_decode.md` (the F24
design and its results section), `docs/mixed_kv_page_transport_backends.md`
("Track F [24]", lines 1848-2020) and `docs/mixed_kv_page_transport_dataflow.md`
(D1-D6, C1-C11, A1-A8).  Every measured number below is read from the existing
nkcut2 artifacts (`/tmp/mixedkv-wtF24-art/ncu_fp8_source.csv`,
`/tmp/mixedkv-wtF23/ncu_fp8e_source.csv`, `trace_q1.txt`, `sf1_mask1.sass`);
the aggregation script written for this reading is
`nkcut2:/tmp/mixedkv-wtF24-art/f25_ncu_classes.py` (outputs
`f25_classes_f24_fp8.txt`, `f25_classes_f23_fp8e.txt`).  No new GPU run was made.

## 0. Summary

**Chosen: F25 = the 12-warp layout (one producer warp group) with (1) the
producer's register budget raised 72 -> 136 by taking the consumer to 184
(proven spill-free by F24b), (2) branch-free block bodies with one fold vote
per operand per pair (loads first, one `__syncwarp` per operand between the
landing loads and the scale-slot loads), (3) the copy issue collapsed to one
`IMAD.WIDE` + one `LDGSTS` per copy from per-item 64-bit bases, with the
`FULL` arm hoisted per work item by **peeling**: K(last) alone and the first
pair (K(last-1), V(last)) are the only two calls that compile the partial arm,
the loop compiles the `FULL` arm only, (4) the per-warp protocol cut from ~200
to <= 70 warp-instructions per pair (32-bit pending words, an unconditional
finish in the loop of the static modules - valid *because of* the peel - and
single-operand finish sites for K(last) and the drain), (5) E2M1 by bit
placement with a 2^126 fold under the same per-operand vote (fp4 then costs
what fp8 costs; the fp4 landing keeps today's 8 B `cp.async`), (6) the dynamic
module's copy path as six unrolled pages with **predicated per-format copies**
(six `@P LDGSTS` sites per page, two execute; no rolled loop, no per-page
branch, no `SEL` chain on the address).**  Rev 1's "one `LDGSTS.128` form with
src-size 8" for fp4 blocks and scale rows is withdrawn: those sources are only
8 B aligned under the frozen R1 layout (section 3.4, 10.5).  F24b's second
producer warp group is reverted: `named_barrier.cuh`, `epilogue.cuh`,
`sparse_mainloop.cuh` and the warp-group plumbing of `prefill_sm90.cuh` go back
to the `5cc416fd` text; `prefill_sm90.cuh` keeps only the register hook and
`kernel_traits.cuh` the static-format / register constants (per-file list in
3.2, C4).  A role-split 16-warp layout (transport WG + decode WG) is designed
as the only admissible 16-warp fallback and is built only if the a16 ncu probe
(gate 6.0) or the F25 trace (gate 6.2) shows the copy phase is SM-serialised.

| mode (us, q=1 / q=64) | today (F24 @ 64a70b9c) | F25 predicted (centre) | F25 band | accept |
|---|---|---|---|---|
| stock_a16 | 299.8 / 310.5 | unchanged | 297-303 / 306-312 | control |
| transport_a16 | 282.5 / 288.4 | unchanged (a16 module byte-identical) | 281-290 / 284-292 | control |
| fp8 static | 460.2 / 475.6 | **300 / 305** | 288-318 / 292-322 | <= 330 |
| fp4 static | 495.7 / 512.4 | **305 / 310** | 288-325 / 292-330 (hot fold path); +3 % where the fold vote fails | <= 330 |
| mixed (dynamic) | 718.1 / 728.2 | **~340 / ~345** (rev 2: ~790 instr per warp per pair, section 3.5) | 320-370 / 325-375 | <= 330 **not met at the centre**; needs IPC >= 0.29 (section 4) |

The prediction rests on one measured quantity that F25 does not change by
construction: the producer warp's realised issue rate.  Per producer warp the
fp8 pair is **~645 warp-instructions** (section 4); at the [23] *decode-phase*
IPC of 0.27 that is 2390 cycles = 1.21 us + ~0.08 us of acquire round trips =
**1.29 us against T_c = 1.28 us** (the consumer's pair time, from transport_a16
282.5 us / 220 pairs at the 1.98 GHz the runs report).  At the [23] *blended*
IPC of 0.218 (which included 96 XU conversions and 64-bit address chains per
warp per pair that F25 removes) it is 1.56 us -> 340 us; at the 0.35 the F24
SASS shows inside its block bodies it is 1.0 us -> consumer-bound at 285-290.
The band above is that bracket.  Main risks, in order: (1) the per-warp IPC of
a branch-free 6-block body at 136 registers is inferred (from the [23]
decode-phase rate and the F24 body-internal rate), not measured; (2) the copy
phase's residual time after the address chain is gone is not derivable if the
LDGSTS dispatch is SM-serialised: the a16 module - no address chain to speak
of, stock copy form, the same 96 LDGSTS per SM per pair - traces `iss`
0.73-0.75 us (F24 doc, trace table), [23] and F24 0.62-0.85.  If `iss` stays
~0.7 us the fp8 pair is 0.7 + 1.0 (fin at IPC 0.27) + 0.08 = ~1.8 us -> ~390
us, outside the band, and 2E (or 2G, copies interleaved into the decode body)
becomes the build.  This is decidable from the existing a16 build with one ncu
run (gate 6.0, ordered before any F25 timing; **not run in this worktree** -
no remote runs were made); (3) the mixed module's decode keeps rolled
format-outer loops and its copy body carries six predicated copy sites per
page: its centre is ~340 us, above the accept row; (4) fp4's fold vote fails
on operands whose block scales exceed |s g| >= 3.99, adding 8 `HMUL2` per
block (+8 %) — a quantizer-side knob (g).

**What is not reachable.**  With BF16 materialised in shared memory the pair
cannot go below the shared-memory pipe's 1974 wavefront-cycles (1.0 us, ~230 us
kernel) and the compressed modes cannot beat transport_a16 (the consumer sets
T_c); parity (283-290) uses 78 % of that pipe.  Consumer-side decode (2A),
register-operand K (2B, SWAP_AB), fp8 wgmma (2C) and a second producer warp
group by page parity (F24b) are each rejected below with the arithmetic that
rejects them; none reaches 330.

## 1. The [24] record restated as per-SMSP budgets

Benchmark shape B=17, S=4096, 8 KV heads, GQA 4, D=128, bf16, q=1: 544 work
items x 44 `produce_pair` calls = 23,936 pairs per kernel, ~220 pairs deep per
SM.  Clock during the F24 runs 1980 MHz (`nvidia-smi`; ncu `--clock-control
none`).  Cycles below are at 1.98 GHz.

| quantity | value | source |
|---|---|---|
| T_c (consumer pair time) | 282.5 us / 220 = **1.284 us = 2540 cycles** | transport_a16 bench |
| target 330 us | 1.50 us = 2970 cycles per pair | |
| fp8 pair today | 460.2 / 220 = 2.09 us = 4140 cycles (ncu 456.9 us -> 4110) | F24 bench / ncu |
| tensor floor | QK^T + PV = 2 x 2 x 128 x 96 x 128 = 6.29 MFLOP per pair; 4096 FLOP/cycle/SM -> **1536 cycles = 0.78 us** (ncu: tensor active 41.1 % of 4110 = 1690) | ncu `sm__pipe_tensor` |
| smem pipe per pair (SM-wide, 1 wavefront/cycle) | op_gmma 1001 + op_st 418 + op_ld 455 + ldgsts 101 = **1974** (C11) | F24 ncu |
| producer warp-instructions per pair per SM | [23] 3417 (854 per warp, 4 warps); F24 4432 (554 per warp, 8 warps) | ncu region sums |
| consumer warp-instructions per pair per SM | 79.0 M / 23,936 = **3301 (413 per warp)**, both rounds | ncu region sums |
| SMSP issue utilisation | [23] 44.6 %, F24 52.9 % | `smsp__issue_active` |

**Per-SMSP demand is layout-independent.**  Each of the four schedulers hosts
two consumer warps and one (12-warp) or two (16-warp) producer warps; the
work per SMSP per pair is the same in both layouts: 413 x 2 consumer
instructions plus one quarter of the producer's total.  What differs is only
how many warps present that producer quarter and how much protocol is
duplicated to do so.  This is the arithmetic that decides (D) in section 2D.

**Where the producer's time goes (F24 fp8, per-PC samples, producer region =
`USETMAXREG.DEALLOC .. EXIT`, 85,068 samples).**  By opcode class, executed
share / sample share: `IMAD` 24.0 / 20.2 (133 per warp per pair: 48 are the
decode's `IMAD.SHL`, ~85 are addresses and register moves), `LOP3` 11.3 / 9.9,
`PRMT` 9.5 / 4.6, `HMUL2` 8.5 / 4.3, `LDS` 6.7 / 6.4, `BRA` 5.2 / 10.2, `IADD3`
4.6 / 7.0, `ISETP` 3.3 / 6.2, `VIADD` 1.3 / 5.6, `LDGSTS` 2.1 / 4.2, `STS` 2.1 /
0.6, `F2FP` 2.1 / 2.5, `BSSY`+`BSYNC` 3.6 / 3.3, `UMOV` 2.0 / 2.4, `VOTE` 1.1 /
1.2.  Grouped: **decode bodies ~282 of 554 per warp (51 %) but only ~30 % of
the samples; copy issue + protocol ~272 (49 %) and ~60 % of the samples.**  All
28 top PCs are in the copy-issue / protocol path: the meta `valid` extraction
(`LOP3 R, R19, 0xff0000`, `short_sb` behind the row's `LDS.128`, 1751 + 1125
samples), the 64-bit source chain (`ISETP.NE`, `IADD3.X`, `VIADD R, UR`,
`IMAD.WIDE.U32`, `IMAD.MOV`: 700-1450 samples each, `dispatch` 45-60 % and
`math` 15-25 % — integer-pipe dispatch conflicts, no `lg_throttle` /
`mio_throttle` anywhere in the region), the acquire spins (`@!P0 BRA`,
`long_sb`), the commit (`SYNCS.ARRIVE`, `long_sb`).  The [23] record has the
same shape with `IMAD` at 29 % of samples and `F2FP` at 8 % (the 96 XU
conversions per warp per pair that F24a removed).

**The block body itself (F24 `sf1_mask1.sass`, rows 3262-3322 of the source
view = one FP8 block, ~47 instructions, ~2900 samples).**  Per block the
`FSETP -> VOTE.ALL -> IMAD.MOV x3-4 (register join of the hot / cold arms) ->
@P BRA -> F2FP.PACK -> UMOV UR, -1 -> 8 HMUL2 -> BRA.DIV` cluster carries ~1000
of the ~2900 samples: `VOTE.ALL` 285 (`wait` 196), the joins 368 / 248 / 170
(`wait`), `F2FP.PACK` after the branch 175-262 (`branch_resolving` +
`no_inst`: instruction fetch after the taken branch), `UMOV` 203-248 (`wait`),
and the next block's scale `PRMT` 149-152.  The straight-line decode
instructions (`PRMT`, `IMAD.SHL`, `LOP3`, `HMUL2`, `STS`) sample at 20-30 each,
i.e. they issue nearly every time they are sampled.  **Inside the body the
warp issues on ~35 % of its sampled cycles; one third of the body's time is
the per-block vote/branch/join structure that F24a introduced.**  The [23]
body (no vote, `F2FP` form) issued at 0.27 over its expansion phase (588
instructions in the trace's 1.1 us).

**The consumer has no slack for decode work (F24 fp8, consumer region 84,906
samples).**  `@P0 BRA` at the K full-barrier wait (`consumer_wait`,
`mainloop_mma.cuh:59-62`, `:233`) 14,031 samples = **16.5 %** (the producer
pacing; [23]: 23.1 %); the ping-pong `HGMMA.64x96x16` behind
`WarpScheduler::barrier_sync` 8287 = 9.8 %; `WARPGROUP.DEPBAR` 1.3 %; the
per-item `barrier_Q` / `barrier_O` 1.7 %; `selected` 11.3 %; `wait` 23.8 %,
`dispatch` 14.8 %, `not_selected` 12.2 %, `math` 3.8 %.  Non-producer waits
are therefore ~13 % of 4110 cycles = ~530 cycles per pair per warp today and
<= ~400 at parity (the ping-pong wait is the other WG's gemm).  By opcode the
consumer is `FMUL` 16.5 % (the unconditional `rescale_o`, 64 per tile per
thread), `FMNMX` 12.3 %, `MUFU` 11.9 %, `FFMA` 11.8 %, `FADD` 11.4 %, `F2FP`
5.9 %, uniform-datapath ops ~12 %, `HGMMA` 3.3 %.  Its XU load (98 `MUFU.EX2`
per SMSP per pair at 4-8 cycles each) is the most loaded pipe at parity
(~35-40 %), which is why the producer's remaining `F2FP` count matters.

**Pipe budgets at parity (per SMSP per pair, 2540 cycles), F25 fp8 12-warp.**
Consumer: ALU (FMNMX, masks) ~300 cycles, FMA (FMUL/FFMA/FADD, 1 cycle per
32-wide warp op) ~330, XU (MUFU, F2FP) ~900, issue 826.  Producer (section 4):
ALU (9 PRMT + 8 LOP3 + FSETP per block x 12 + ~40 protocol) ~280 ops x 2
cycles = 560, FMA (8 IMAD.SHL x 12 + 8 HMUL2 x 12 + 24 IMAD.WIDE + 24
FMUL/HADD2) ~360, XU (2 F2FP x 12) ~100, LSU-issue ~90, issue ~645.  Totals:
**issue 1470 / 2540 = 58 %, ALU 34 %, FMA 27 %, XU 39 %, smem pipe (SM) 78 %.**
No pipe is saturated; the binding question is the single producer warp's
realised issue rate against a stream that is 60 % ALU (2-cycle) ops, i.e. its
latency tolerance — which is a register/ILP property (section 3.2), not a
warp-count property.

## 2. Alternatives

### 2A. Consumer-side decode (expansion by the consumer warps into smem)

Requires >= ~900 cycles per pair per consumer warp of non-producer wait to hide
427 extra instructions per warp (3417 / 8).  Measured (section 1): ~530 today,
<= ~400 at parity, and the extra instructions are 60 % ALU at 2 cycles each
(>= 850 cycles of pipe time per warp).  **Rejected by the F24 ncu consumer
data; the F24 doc's own trace condition (2A) is failed without a trace build.**
It also breaks C5.

### 2B. Register-operand K via SWAP_AB (S^T = K Q^T, K as the wgmma A operand)

Data flow.  `wgmma.mma_async` takes A from registers or smem and B from smem
only (`kernel_traits.cuh:138-143`: K is B of QK^T, V is B of PV).  K in
registers forces the transposed product with M = tokens: M is 64 per wgmma
per warp group, so with two consumer WGs **CTA_KV becomes 128** (stage 32 KB
per operand; S = 3 stages -> 192 KB + Q 32 KB + scales/meta/barriers ~12 KB =
~236 KB > 227 KB, so S = 2, which A1 rejected for the compressed path).  N =
128 q rows (the q=1 padding stays: the XQA form with N = 8 heads is a
different kernel and is the XQA host's domain by A3).

A-fragment decode (m64nNk16, K = D, one k16 step = one 16-coefficient scale
block).  Lane l of warp w holds rows r0 = 16w + l/4 and r0 + 8, columns 2c,
2c+1, 2c+8, 2c+9 (c = l % 4) of each k-step: two scale bytes (rows r0, r0+8)
and 8 E4M3 bytes from two 16 B blocks.  Per k-step per lane: 2 `LDS.128`
(4-lane broadcast), 4 `PRMT` + 4 `SHL` + 4 `LOP3`, two scale chains (10), 4
`HMUL2` = **28 for 8 values; 8 k-steps = 224 per lane per K tile of 128
tokens** -> 1792 warp-instructions per SM per K tile, vs the producer's
768 x 43 / 32 x 128/96 = **1376 per 128 tokens today**: +30 % (the scale
chain and the loads are replicated over the 4 lanes that share a row), all of
it on the consumer warps, whose slack is ~400-530 cycles (2A).  The 32 A
registers per tile are pinned from issue to `wgmma.wait_group`.

Softmax with S transposed.  The row max of S is the *column* max of S^T =
a reduction over M = tokens, i.e. across the 2 rows a thread holds, the 8
lanes l/4 = 0..7 (3 `SHFL` + `FMNMX` steps), the 4 warps of the WG (smem
exchange + named barrier) and the two WGs (both hold all 128 q columns; their
running maxima must agree: a second exchange + a 256-thread barrier per
tile).  Per thread per tile for m64n128 (64 accumulator registers = 2 rows x
32 columns): max 32 + 3 x 32 x 2 + ~120 (cross-warp) + ~60 (cross-WG) ~= 400;
sum the same ~400; exp 64 x (`FFMA` + `MUFU`) = 128; and P^T is the *B*
operand of O^T = V^T P^T, so it must be **stored to smem** (64 bf16 per thread
per tile: 16 `STS.64` + fence) — **~1100-1300 instructions per thread per
tile against FA3's ~300** (in-thread row max over 48 values + 2 `SHFL`, exp 48,
sum, `rescale_o` 64, `F2FP` 24).  That is +900 x 8 warps = **+7200
warp-instructions per pair on the consumer, to remove ~1000-1400 from the
producer.**

V as the A operand (O^T = V^T P^T).  The lane's fragment holds (d = 16w +
l/4, +8) x (tokens 2c, 2c+1, 2c+8, 2c+9): 8 values from 4 different token rows,
**each with its own block scale** (the scale is per (token, D-block)), so each
`HMUL2` needs a packed pair of two different scales: 8 scale-byte extractions
per k-step per lane instead of 2, and the packed bytes are gathered from 4
rows (4 `LDS.32` + `PRMT`): ~2x the K cost, ~450 per lane per tile.  The
accumulator O^T (128 D x 128 q) is **128 registers per thread per WG**, both
WGs hold partial sums over their 64 tokens, reduced through smem once per item
(64 KB).  Register total S^T 64 + O^T 128 + A fragments 32 + softmax state
>= 240: **does not fit 184, does not fit 256.**

**Verdict: rejected** on all four counts (stage depth, consumer instruction
count 4x, V's per-element scales + P^T store, registers).  The only saving it
offers — K's 192 `STS` wavefronts and ~380 of the 1001 `op_gmma` wavefronts —
addresses a pipe that is at 78 % and not binding.

### 2C. FP8 wgmma with e4m3 operands (K / V never expanded)

`wgmma` e4m3 x e4m3 -> f32 requires **K-major A and B** (no transposed
operands for fp8).  QK^T: B = K^T is K-major (D contiguous) — admissible.
PV: B = V with K = tokens must be token-contiguous; the V rows are
D-contiguous (MN-major), so V needs an in-smem transpose (what FA3's own fp8
path pays with `transposeVTile`).

Numerics.  The block scale s[n, b] is indexed by token n and **D-block b**.
QK: S[m,n] = sum_b s[n,b] sum_{d in b} q[m,d] k'[n,d] — the scale varies along
the reduction axis D, so it cannot be applied to S after the MMA; it would
need one accumulator per D-block (8 x 48 registers) or 8 serialised k16
wgmmas each followed by a per-column `FFMA` pass (8 x 48 `FFMA` + 8 `wgmma.wait`
per tile per thread: more consumer work than 2A).  PV: O[m,d] = sum_n p[m,n]
s[n,b(d)] v'[n,d] — the scale varies along the reduction axis n, so it cannot
be applied to O either; it would need one scaled copy of P per D-block (8 x
24 registers, 8 x 24 `F2FP` per tile).  **The SageAttention dual does not
apply**: SageAttention's K scales are per token block (along N, commuting
with the reduction over D) and its V uses per-channel (per-d) scales
(commuting with the reduction over n); R1's frozen format puts both scales on
the (token, D-block) axis, which is the reduction axis for K and the reduction
axis for V.  Neither operand qualifies.  Independently, quantising Q to e4m3
per tile changes S by up to 2^-4 relative per element: **not bit-exact
against the BF16 reference the tests pin**, whatever the scale placement.

**Verdict: rejected** (reduction-axis scales, V transpose, bit-exact contract).

### 2D. Can 12 vs 16 warps be decided by arithmetic alone?

What arithmetic decides: (i) per-SMSP demand is identical in both layouts
(section 1); (ii) the second warp group adds, per pair per SMSP, one more copy
of the per-warp protocol (F24 measured +1015 warp-instructions per pair per SM
= +30 %, i.e. ~130 per extra warp even after halving copies and blocks) and a
256-thread barrier + gather that trace at 3x the 12-warp module's `bar + gat`;
(iii) at parity the issue slots are 58 % used at 12 warps and ~63 % at 16 —
neither saturates, so more warps cannot be *needed* for slots; (iv) the pipes
(ALU 34 %, FMA 27 %, XU 39 %) are shared per SMSP and gain nothing from a
second producer warp.  What arithmetic does not decide: whether a single
producer warp per SMSP can realise IPC >= 0.26 on a 60 %-ALU stream (F24's
second warp bought +15 % producer throughput per SMSP for +30 % instructions:
the marginal latency-hiding value of a second warp *in that code shape* was
smaller than its protocol cost).  **Conclusion: the page-parity second WG is
dominated (never better than 12 warps at equal per-SMSP demand) and is not
built again; the 12-warp layout's IPC question is answered structurally (2F,
3.2) and measured once (6.1).  The only 16-warp form with a structural
argument is the role split of 2E, whose gain is phase overlap, not slots.**

### 2E. Role-split 16 warps (transport WG + decode WG) — the designed fallback

Data flow.  WG0 = the a16 module's producer plus the scale copies: chunk
table, acquires, all `LDGSTS` of pair t, `cp.async.commit_group`, then
`cp.async.wait_group 1` (its own copies of pair t-1 have landed), then
`bar.arrive(kLanded, 256)`.  WG1 = decode only: `bar.sync(kLanded, 256)`,
expand pair t-1 (12 blocks per thread, the same (row, block) ownership
formula applied to its own thread index, reading landing chunks and scale
slots written by WG0's threads), `fence.proxy.async`, `producer_commit(t-1)`
on the full barrier.  Visibility: WG0's `cp.async.wait_group` followed by the
named barrier (bar.sync orders WG0's completed writes before WG1's reads —
the stock `cp_async_wait + __syncthreads` pattern).  WAR on a stage: WG0
acquires stage s for pair t+3 only after the consumer's release, which
follows WG1's commit, which follows WG1's reads: ordered without a new
barrier.  Full barrier arrival count stays 128 (WG1 arrives for compressed
tiles; in the dynamic module WG0 arrives with `cp.async.mbarrier.arrive` for
A16-only tiles — exactly one WG per stage).

Budgets.  Registers 72 / 72 / 184 (128 x 72 x 2 + 256 x 184 = 65536): the
decode WG is at [23]'s 72, i.e. two pages per step, not six.  Per pair: WG0
~54 copies + ~60 protocol + ~10 signal = ~125 warp-instructions (<= 0.3 us
even at 0.2 IPC), WG1 12 x 44 + 20 (votes) + ~30 = ~580 -> at 0.27 IPC 2150
cycles = 1.09 us + one `bar.sync` wait (~0.05-0.1 us) -> **1.15-1.2 us <
T_c**; at 0.22 IPC 1.33 + 0.1 = 1.43 us -> 315-330 us.  Same IPC bracket as
F25's 12 warps, minus the copy phase (overlapped in WG0), plus the signal and
two more warps of contention per SMSP.  It wins over F25 only if the copy
phase is *SM-serialised* (its time does not fall when the address chain is
removed); that is exactly gate 6.2.  It reintroduces a per-pair group
barrier, but as an arrive/sync pair between roles, not a sync among the
threads doing both (which is what [23]'s "barrier B" measurement rejected).
**Not built unless 6.2 fails.**

### 2F. The counter-named fixes on the 12-warp layout — chosen, with the arithmetic

Per producer warp per pair, fp8, after the changes of section 3 (counts read
off the F24 SASS, 3.1):

| component | instructions | derivation |
|---|---|---|
| decode, 12 blocks x 43 | 516 | per block: 2 `LDS.64`, `LDS.32` (scale word), `PRMT`, `F2FP.E4M3`, `HADD2`, `FMUL`, `FSETP`, `F2FP.PACK`, 8 `PRMT`, 8 `IMAD.SHL`, 8 `LOP3`, 8 `HMUL2`, 2 `STS.128` (F24 body minus `VOTE`, `BRA`, `UMOV`, `BRA.DIV`, 3-4 `IMAD.MOV`, minus the per-block `WARPSYNC`) |
| fold vote, 2 per pair | 14 | per operand: 5 `PLOP3` (AND tree over 6 `FSETP` already counted), `VOTE.ALL`, uniform `BRA` |
| `__syncwarp`, 2 per pair | 2 | one per operand, between the operand's 12 landing loads and its 6 scale-slot loads (3.1); the pair-level `__syncwarp` after `DEPBAR` is removed |
| copies, 12 pages | 48 | per page: `IMAD.WIDE.U32` (payload), `LDGSTS.128` (fp8) / `LDGSTS.64` (fp4), `IMAD.WIDE.U32` (scales), `@leader LDGSTS.64` (8 B scale row) |
| meta read | ~8 | 2 x 2 `LDS.128` + row address (2 x 2); `valid` is not read in the loop (FULL arm) |
| protocol | <= 70 | 2 acquires (~10), `PipelineState` updates (6), `LDGDEPBAR` + `DEPBAR` + `FENCE.VIEW.ASYNC` (3), 2 commits (~6), pending words (4), `expand_bases` stage offsets (2 x 8), loop and chunk-index arithmetic (~8), chunk gather amortised over 16 pairs (~3); no pending test, no `FULL` test in the loop (peeled) |
| **total** | **~655** | [23]: 854; F24: 554 x 2 warps per SMSP = 1108 per SMSP |

Time per pair = acquire round trips (~0.08 us: two `try_wait` on already
complete barriers, section 1 trace `acq` 0.10) + 645 / IPC cycles:

| IPC per producer warp | cycles | pair (us) | fp8 kernel (us) | basis |
|---|---|---|---|---|
| 0.218 | 2960 | 1.58 | ~345 | [23] blended (F2FP-heavy, address chains) — the pessimistic corner |
| 0.27 | 2390 | 1.29 | ~290-300 (consumer-bound, +1 %) | [23] decode-phase rate on a straight-line body |
| 0.35 | 1840 | 1.01 | 285-290 (consumer-bound) | F24 body-internal `selected` share |

The requirement "producer <= T_c" is met from IPC 0.26 upward.  The design
therefore stands on (a) the count (645, verifiable in SASS before any timing)
and (b) three structural properties that move the realised IPC from the
[23] blended figure toward the body-internal one: no XU conversion in the
decode except two per block, no branch or warp-wide op inside the block
bodies, and 48 independent decode chains in flight per operand (136
registers).  These are properties of the SASS (6.1), checked before timing.

## 3. The chosen design (F25)

### 3.1 Data flow (per pair, fp8 static module; fp4 in 3.4, dynamic in 3.5)

```
acquire K stage, acquire V stage                         (unchanged, PipelineAsync)
read meta rows of K(t-1), V(t): 2 x 2 LDS.128 -> pages[6], w7   (unchanged, :703-713)
issue copies K then V: per page j (immediate offsets), FULL arm (loop):
   src   = IMAD.WIDE.U32(pages[j], UR page_stride, pbase64_op)   pbase64 per item (3.2)
   LDGSTS.128 [land8 + stage*STAGE + j*PAGE_REGION], [src], 16   payload block (fp8; fp4: LDGSTS.64 [land4 + ...], 8)
   ssrc  = IMAD.WIDE.U32(pages[j], UR scale_stride, sbase64_op)
   @leader LDGSTS.64 [sc_rd + stage*SCALE_STAGE + j*SCALE_PAGE], [ssrc], 8   row's 8 B of scales (unchanged form)
LDGDEPBAR (commit group of this pair)
DEPBAR.LE 1 (pair t-1's group landed: this thread's own copies)
expand K(t-2) then V(t-1) (both always pending in the loop, 3.2), each as ONE straight-line body:
   6 x 2 LDS.64 packed halves (own landings: no barrier needed)   (all landing loads first)
   __syncwarp        (every lane is past its DEPBAR: lane 0's landed scale slot is ordered before
                      the other seven lanes' LDS.32 of it - D3/A9; the same barrier orders every
                      lane's landing loads before any lane's stores below)
   6 x LDS.32 scale words
   6 x {PRMT, F2FP.E4M3, HADD2, FMUL} -> v_j ; 6 x FSETP |v_j| < 255.5*2^120 ; PLOP3 AND-tree
   VOTE.ALL ; uniform BRA hot/cold
   hot:  6 x {F2FP.PACK sf2_j ; 8 PRMT, 8 IMAD.SHL, 8 LOP3 ; 8 HMUL2 ; 2 STS.128 [d0 + j*PAGE_REGION], [d1 + ...]}
   cold: the same with 8 HMUL2 by 2^120 per block first and sf2_j = bf16(f32(s_j) * g) (g reloaded, LDC+LDG)
FENCE.VIEW.ASYNC ; commit K(t-2), V(t-1) full barriers   (unchanged)
```

Ownership is [23]'s (A7), applied to `u = t` (one warp group): thread `t` owns
block `t % 8` of row `t / 8` of every page; output chunks `2b + swap`,
`2b + 1 - swap` with `swap = ((b >> 2) ^ r) & 1` (A8, C11: `STS.128` at 4
wavefronts); landing halves read in store order (two `LDS.64`).  D3 holds
verbatim (a thread reads only what its own `cp.async` wrote, plus the row's
scale slot after the issuing lane's wait and the operand's `__syncwarp`; the
`__syncwarp` sits *between* the landing loads and the scale loads, so one
barrier serves both orderings - rev 1 listed the scale loads before the
barrier, which was wrong, section 10.3).  The exposed cost of that order is
one `LDS` latency (~30 cycles) per operand before the scale chain, paid once,
not per block.  Shared-memory layout: unchanged (~199.7 KB; the row's scale
slot stays 8 B, `kernel_traits.cuh:89` unchanged).  Wavefronts per pair:
unchanged from F24's measured 1974 (`op_st` 418 at 4.41 per `STS.128`, `op_ld`
455, `LDGSTS` ~101: 4 per `LDGSTS.128`, 3.98 per fp4 `LDGSTS.64`, <= 4 per
leader-only scale copy).

Per work item the finish sites are three (3.2, C14): K(last) alone (K only,
`cp.async.wait_group 0`, before `barrier_O.wait`, C7), the loop pair (K and V,
`wait_group 1`, unconditional in the static modules) and the drain (V only,
`wait_group 0`).  The single-operand sites compile the **exact body only** (no
vote, 8 `HMUL2` by 2^120 then `sf2 = bf16(f32(s) g)`: reference-exact for every
finite input, +~100 instructions per work item, half the static code of a
hot + cold site).

### 3.2 Control flow, registers, and why the block bodies are branch-free

**Vote hoisting (C12).**  The fold test needs all six scale words of the
operand, which `expand_operand` already loads up front (`:1250-1254`); the
six products v_j and the six `FSETP` are computed before any decode, ANDed
(`PLOP3`), voted once (`VOTE.ALL`) and branched once per operand.  Both arms
are the full six-block body; nothing but addresses is live across the join, so
ptxas has no registers to unify with `IMAD.MOV` (the 3-4 per block in F24).
The `__syncwarp` moves to *before* the vote, in straight-line code between the
operand's landing loads and its scale loads: ptxas proves convergence and
emits at most one `WARPSYNC` (the [23] SASS had none) — no `UMOV UR, -1;
BRA.DIV` guards (22 in F24).  The cold arm is the out-of-line copy.  Code size
(rev 2, per inlined site): the loop site carries two operands x (hot 258 +
cold 258 + vote 14) ~= 1060 decode instructions plus copies and protocol; the
K(last) and drain sites carry one operand's exact body each (6 x 51 = 306);
with the item prologue / epilogue (~250) the fp8 producer region is **~2100-2300
instructions** (F24: 2093; [23]: 2687) - rev 1's "<= 1500" counted one site.
The hot-path footprint per loop pair is ~650 instructions (10.4 KB).  Numerics unchanged (C9): the vote covers the same predicate per block,
only its granularity is per operand-per-warp instead of per block-per-warp;
an operand with one over-range block takes the exact path for all 24 blocks of
that warp (the extremes tests exercise this: `run_fa3_mixed_page_transport.py`
scales 448 / 256 with g = 1).

**Registers (C3 restated).**  `__launch_bounds__(384, 1)` -> 168 at launch;
`setmaxnreg.dec 136` (producer), `setmaxnreg.inc 184` (consumers): 128 x 136 +
256 x 184 = 64,512 = 384 x 168 exactly (the pool the CTA owns; there is no
slack).  Admissible 12-warp splits satisfy P + 2 C <= 504: 136/184, 120/192,
152/176, 104/200, 72/216; **128/192 (= 65,536) is not one of them** - the
CTA never holds those registers and `setmaxnreg.inc 192` would wait forever
(rev 1's fallback row, corrected in 6.1).  The consumer at 184 is proven
spill-free (F24b: `ptxas -v` no C7507, `STACK 0`, same consumer code), but
T_c = 1.284 us was measured at 216 in a producer-bound regime: an a16 control
build at 72/184 (cheap, not merged) is listed in 6.4 to measure T_c(184)
before any shortfall is attributed to the producer.  Producer live set for a whole
operand in flight: 6 x 4 packed words + 6 scale words + 6 x 8 decoded words +
6 sf2 + 2 output bases + 2 landing bases + scale base + item constants (two
operands' 64-bit payload and scale bases: 8, strides in UR) + pipeline state
(~8) + loop (~6) ~= 120 < 136.  The a16 module keeps 72 / 216 (its SASS stays
byte-identical: the dealloc / alloc immediates are the only register-split
text and they come from `PRODUCER_REGS / CONSUMER_REGS` selected by
`kMixedStaticFormat`, `kernel_traits.cuh:229-230`).

**ILP that the registers buy.**  Six blocks per operand in one basic block =
48 independent 3-op decode chains + 6 scale chains; the `LDS` latency (~30
cycles) is paid once per operand instead of once per 2-page step; the F2FP.E4M3
-> HADD2 -> FMUL -> FSETP -> PLOP3 -> VOTE chain (~70 cycles) is paid once per
operand instead of once per block (12 x ~120 cycles of exposed chain per pair
in F24's serialised bodies).  Issue-bound estimate per operand: ALU 6 x 18 x
2 = 216 cycles, FMA ~170, XU 6 x 2 x 4 = 48, LSU ~30, overlapping -> ~250-350
cycles per operand if the scheduler interleaves pipes; 2 operands -> 0.25-0.35
us of expansion per pair against F24's traced 1.0-1.16 (which included the
per-block chains) and [23]'s 1.1.  This is the mechanism behind the IPC bracket
of 2F; it is not quoted as a number in the prediction.

**Copy issue (C13).**  Per operand per item: `pbase64 = payload + head *
hs + row * ts + blk * 16`, `sbase64 = scales + head * shs + row * sts` (two
64-bit registers each, computed once in `make_bases`, `:802-860`; today
`compressed_src`, `:870-886`, recomputes them per tile — the `IADD3.X / VIADD
R, UR` top-stalled PCs).  Per page: `IMAD.WIDE.U32 R, pages[j], UR_ps,
pbase64` then `LDGSTS` with the immediate destination; the twelve page
addresses of a pair are independent (12 `IMAD.WIDE` back-to-back on the FMA
pipe, ~50 cycles, then 24 `LDGSTS`).  `FULL` is hoisted per work item **by
peeling the first loop pair**: the only tile that can be partial is
`kv_tile_idx` (`valid = min(CTA_KV, kv_len - tile * CTA_KV)` and
`kv_tile_idx * CTA_KV < kv_len` because `num_kv_tiles <= ceil(kv_len /
CTA_KV)`, so every tile below it has `valid = CTA_KV`), and that tile's two
operands are issued by two different calls: K(last) alone (`produce_pair(
kv_tile_idx, -1)`) and V(last) in the loop's *first* iteration
(`produce_pair(kv_tile_idx - 1, kv_tile_idx)`, `:1649` at `t = kv_tile_idx`).
Rev 1 gave the partial arm to the K(last) call only, so V(last)'s rows past
`valid` would have been copied with src-size 16 from the page's stale rows /
page 0 (D4 violated: garbage scale bytes -> NaN `sf2` -> NaN x P(= 0) = NaN
in O; section 10.1).  Rev 2: the first loop iteration is peeled out of the
loop (`pair_step(kv_tile_idx, FULL_K, PARTIAL_V)`), the K(last) call is
`(PARTIAL_K, -)`, and the loop `for t = kv_tile_idx - 1 .. swa_begin` compiles
`(FULL_K, FULL_V)` only: `issue_operand<PARTIAL>` is a compile-time tag, the
loop loses the per-pair `valid == CTA_KV` branch and its `BSSY/BSYNC`, and
`valid` is read from `w7` only in the two partial calls.  The partial arm no
longer selects a fallback source pointer: `src = pbase64 + page x stride` is a
valid in-bounds address for every (page, row) of the transport tensor (pages
past `kv_len` are page 0), and with src-size 0 no byte is read (CUTLASS's
`cp_async_zfill` passes the unmodified pointer with `src_in_bytes = 0`,
`memory_sm80.h:155-170`).  The a16 module keeps its runtime `valid == CTA_KV`
branch and its `v ? s : base` selects textually (byte-identity).  The scale
copy stays one predicated 8 B `cp.async` per page by lane `b == 0` into the 8 B
row slot (rev 1's 16 B form is withdrawn, 3.4).

**Protocol (C14).**  Rev 1's "both operands are always pending in the
compressed static modules" was false for the loop's first iteration: K(last)
is finished before `barrier_O.wait` (C7), so at `produce_pair(kv_tile_idx - 1,
kv_tile_idx)` both pending words are 0, and an unconditional finish would have
decoded stage 0 (which the consumer is reading K(last) from) and arrived a
second time on `full_barrier[0]` (section 10.2).  The peel fixes this by
construction: the peeled first pair has **no finish call at all** (nothing is
pending: K(last) was finished by its own single-operand site), and every loop
iteration finishes the pair issued one iteration earlier, which is always a
(K, V) pair - so in the static compressed modules the loop's
`finish_pending_pair` is unconditional (no pending test, no `BSSY/BSYNC`),
and the two single-operand sites (`finish_one(K)` after K(last), `finish_one(V)`
at the drain: the last pair always has `tK = -1`, so K is never pending at the
drain) are unconditional too.  The dynamic module keeps one warp-uniform `if
(op.pending)` per operand at every site (A16-only tiles are not pending); it
is data from one smem word, so it cannot diverge.  The pending record is a
32-bit word `(w7 & 0x03FFFFFF) | stage << 30` (`static_assert(NUM_STAGES <=
4)`: bits 30-31 alias the flags byte's bits 6-7, which are masked; NUM_STAGES
is 3).  Budget: <= 70 warp-instructions per pair per warp (itemised in 2F),
checked as the loop-site SASS count minus 12 x 43 - 14 - 2 - 48 - 8 (6.1).
The back-to-back acquire of rev 1 is not built (not needed for the budget).

**Barrier protocol and shared Hopper files (C4).**  Back to [23]: producer
arrival count 128, `kQueryEmpty` 384, `kProducerWG` 128, Q TMA by warp 0 of
WG0, ping-pong ids 2 and 3, chunk gather by all 128 producer threads.  Per
file (rev 2 replaces rev 1's contradictory "revert" / "unchanged in text"):

| file | reverts to `5cc416fd` text | stays (F24 text) | why |
|---|---|---|---|
| `named_barrier.cuh` | everything: `producer_warp_groups_v`, `kFirstConsumerWG` template parameters, the `WarpScheduler` remaps (`:30-40, :46-63, :81-86, :119-130`) | nothing | one producer WG: the stock derivations are exact |
| `epilogue.cuh` | `NUM_COPY_THREADS = cutlass::NumThreadsPerWarpGroup` (`:77-79`) | nothing | same |
| `sparse_mainloop.cuh` | the `static_assert` (`:89-95`) | nothing | same |
| `prefill_sm90.cuh` | `NUM_PRODUCER_WGS` / `NUM_COPY_THREADS` derivation (`:63-67`), `is_producer_wg` (`:94-97`, `:158`) | the register hook `else if constexpr (Ktraits::kMixedTraits) warpgroup_reg_dealloc<Ktraits::PRODUCER_REGS>() / _alloc<Ktraits::CONSUMER_REGS>()` (`:160-161`, `:226-227`) | `5cc416fd` hard-codes `dealloc<72>` / `alloc<216>` for the non-TMA path; 136/184 needs the hook.  Stock traits have `kMixedTraits = false`: the branch is discarded at compile time and the stock kernel's SASS is unchanged (gate 6.1) |
| `kernel_traits.cuh` | `NUM_PRODUCER_WGS`, the `NUM_WARPS` / `NUM_THREADS` / `NUM_PRODUCER_THREADS` overrides (`:225-228`) | `mixed_variant_static_format`, `kMixedStaticFormat`, `PRODUCER_REGS = kMixedStaticFormat == 0 ? 72 : 136`, `CONSUMER_REGS = 216 : 184`, the pool `static_assert` (rewritten for 384 threads), the `MixedTileMeta` mask comment (masks are kept, 3.5) | mixed-only struct; stock traits untouched |
| `variants.cuh`, `mainloop_mma.cuh` | - | untouched | |

Acceptance: stock paged kernel and a16 module byte-identical to `5cc416fd`
(gate 6.1).  The F24 record's "user sign-off owed" on shared-file edits now
covers exactly two items: the register hook in `prefill_sm90.cuh` and the
mixed-only constants in `kernel_traits.cuh`.  `barrier_O` / C7 unchanged.

### 3.3 Landing latency cover (why the pending scheme survives a fast copy phase)

The copies of pair t-1 are issued at the start of iteration t-1 and waited for
(`DEPBAR.LE 1`) after the copies of pair t are issued in iteration t: cover =
one full pair time minus nothing (the issue of pair t is *before* the wait).
At parity the pair is 2540 cycles; loaded HBM/L2 latency under the compressed
modes' 1.3-2.5 TB/s is ~1200-2000 cycles; F24's traced `wait` is 0.03 us at a
2.6 us pair.  If `wait` grows above ~0.1 us at the shorter pair, the exposed
part is L - 2540 and bounds the pair from below at ~L; a 4-stage ring is not
available (192 KB of K/V stages + Q 32 KB > 227 KB), so the remedy would be to
issue the *scale* copies (the small, latency-critical ones) first and the
payload after — no structural change.  Recorded as gate 6.3.

### 3.4 fp4: E2M1 by placement with a 2^126 fold (C16)

Per 8 nibbles (XQA `csrc/xqa/mhaUtils.cuh:664-676`): `w4 = w << 4`; for k =
0..3: `a = prmt(w4, w, sel_k)` = [rep(sign w.byte_k), w.byte_k, rep(sign
w4.byte_k), w4.byte_k], `out_k = (a << 2) & 0x81C081C0` -> 13 instructions;
the half `s | 000000 ee | m 000000` is the E2M1 value x 2^-126 exactly (ee at
the bf16 exponent bits [8:7], m at bit 6; code 001 -> the bf16 subnormal 0.5 x
2^-126).  Per block: 2 `LDS.32` (halves) + `LDS.32` scale + 5 scale + 26 + 8
`HMUL2` + 2 `STS` + 0 = **44 hot (vs 57 with the LUT, `:133-157`)**.  The fold:
`gs4 = g * 2^126` (finite for |g| < 4; else +inf sentinel), v_j = f32(s_j) *
gs4, **fold_ok iff |v_j| < 3.9921875 * 2^126** (bf16 max / 2^126 = 3.984375 plus
half an ulp of [2, 4): the mirror of C9's 255.5 x 2^120) and, per operand,
|g| >= 2^-117 (the same lower-bound sentinel as fp8: a bf16-subnormal
bf16(s g) would round differently after scaling); hot path `sf2 = bf16x2(v)`,
one rounding multiply; cold path 8 `HMUL2` by `0x7E807E80` (2^126, exact:
the placed halves times 2^126 are the E2M1 magnitudes exactly, subnormal 0.5
included) then `sf2 = bf16x2(f32(s) * g)` with g reloaded — the reference's
arithmetic for every finite s, g.  The vote fails whenever any block of the
warp's 24 has |s g| >= 3.99; FP4 block scales are amax / 6, so operands with
|x| >= 24 g take the cold path for that warp (+8 per block, +8 %): the
quantizer's global scale g is the knob that keeps typical caches on the hot
path (host-side, no kernel bound).  The bench payload (`make_transport`, scale
byte 0x38 = 1.0, g = 1) is on the hot path.

fp4 landing: **unchanged from [23]/A7** (rev 1's 16 B form is withdrawn).
Block b's 8 B is copied with `cp.async.ca ... 8` (`cp8` / `cp8_zfill`) as the
8 B half `b & 1` of chunk `(b/2 + 4 (r & 1)) ^ (r & 7)` of the row's D-block-1
line, and the two 4 B halves are read at `land4 + 4 swap` and `^ 4`
(`expand_bases`, `:1120-1129`), 3.98 wavefronts per warp instruction as
measured.  Why: FP4 blocks sit at `blk * 8` in a 64 B row and scale rows are
`[pages, tokens, heads, 8]` - the host guarantees 8 B alignment only
(`check_span`, `:521-536`: `block_align = 8` for FP4, `kScaleAlign = 8`), so
odd blocks and odd heads are 8 B aligned and not 16 B aligned.  cp-size is an
immediate that fixes the access size and its alignment requirement; the PTX
ISA text could not be fetched in this session (the single-page document
exceeds the fetch tool's limit) and the design does not rely on a permissive
reading: a 16 B `cp.async` is used only where the host check guarantees 16 B
(A16 rows, FP8 blocks), 8 B elsewhere, which is what CUTLASS does
(`cp_async_zfill<SizeInBytes>` only ever issues src-size in {0, cp-size} with
`SizeInBytes` matching the pointer's alignment, `memory_sm80.h:151-180,
358-378`).  Consequence for the dynamic module: two `LDGSTS` sizes exist, so
the "one copy form with a `SEL`'d size" argument of rev 1 is void and the
dynamic body dispatches by predication (3.5).  The rev 1 landing's 2-way
`LDS.32` bank conflict noted by the review is moot with the landing unchanged.

### 3.5 Mixed (dynamic module): six unrolled pages with predicated per-format copies; format-outer decode with two pages per step

Chunk table.  The dynamic module **keeps** F24c's row shape (`chunk_store`,
`:626-640`: `tags[4], tags[5]` = the two 6-bit page masks; `w7 = m8 | m4 << 8
| valid << 16 | flags << 24`) - rev 1's "back to tag bytes" is withdrawn
because the decode loops are mask-driven and the masks are also the cheapest
per-page predicates for the copies.  `read_meta` reads the full row (two
`LDS.128`, the static arm) so the six page indices are in registers indexed by
the unrolled constant `j`; `page_at` / `row_addr` (the per-page `LDS.32`) go
away.  The pending word is `(w7 & 0x03FFFFFF) | stage << 30`: masks at bits
0-5 / 8-13, `valid` at 16-23 (unused by the expansion), flags at 24-25.

Copies (C17 restated).  Per page `j` (unrolled, immediate offsets), per
thread: three predicates from the masks (`p8 = m8 >> j & 1`, `p4 = m4 >> j &
1`, `pa = !(p8 | p4)`; the tile is the same for the warp, so they are
warp-uniform data), six source addresses by `IMAD.WIDE.U32` from the six
per-item 64-bit bases (A16 rows `u/16`, `u/16 + 8`; FP8 block; FP4 block; FP8
scale row; FP4 scale row) and **six predicated `LDGSTS` sites** of which
exactly two execute: `@pa LDGSTS.128 [a16_dst + j*PAGE], [srcA0], 16`, `@pa
LDGSTS.128 [a16_dst + ATOM + j*PAGE], [srcA1], 16`, `@p8 LDGSTS.128 [land8 +
j*PAGE], [src8], 16`, `@(p8 & leader) LDGSTS.64 [sc_rd + j*SCALE_PAGE],
[ssrc8], 8`, `@p4 LDGSTS.64 [land4 + j*PAGE], [src4], 8`, `@(p4 & leader)
LDGSTS.64 [sc_rd + j*SCALE_PAGE], [ssrc4], 8`.  Predicated-off lanes issue
nothing to the LSU (no wavefronts, no zero-fill: rev 1's "src-size 0 on the
other lanes" would have zero-filled the leader's slot in the same warp
instruction, section 10.4).  ~15 instructions per page (3 predicate ops, 6
`IMAD.WIDE`, 6 `LDGSTS`) -> **~180 per pair per thread**, all twelve pages
independent, no `SEL` on any address, no rolled loop, no per-page branch.
The partial arm (the two per-item calls only) adds the D4 predicate `v = tok0_j
+ row < valid` per copy as the src-size register (`v ? size : 0`), with the
A16 rows using their own row indices.  This replaces F24c's format-outer copy
loops (`:992-1045`), whose per-pair `iss` traced at 1.7-2.1 us on 8 warps.

Decode.  Format-outer as F24c (`expand_format_pages`, `:1303-1324`) with two
pages per step, at 136 registers.  Per operand: the six scale words are loaded
up front (after the operand's `__syncwarp`, which follows the first step's
landing loads), each scale byte's `f32(s_j)` is formed once and multiplied by
`gs8` and `gs4` (two `FMUL`, two `FSETP`), the per-format vote is `VOTE.ALL`
over `AND_j (!m_j | ok_j)` (A16 pages and the other format's pages do not
vote), and the format's rolled loop runs the **hot or the cold body chosen by
one uniform branch outside the loop** (`loop<HOT>` / `loop<COLD>`: no branch
inside the loop body but the back-edge).  A step decodes two pages (the
second page's stores predicated by "a second page exists" - an odd page count
re-decodes its last page idempotently rather than branching), issues the next
step's landing loads before this step's stores (F24c's one-ahead pipelining),
and meets one `__syncwarp` before its stores (every lane's loads of the step's
pages before any lane's stores of them, D3/A7).  The six scale words are loaded once at immediate offsets for the vote and
**reloaded per page inside the format loops** (two `LDS.32` per step, `sw0` /
`sw1` at `dynamic_page(i).sc_off`): the loop's page index is a runtime value
(`__ffs` of the mask), so indexing the vote's `sw[]` by it would send the array
to local memory (`LDL`, the C2 violation); the reload is the correct shape and
is counted here, not a miss against the 6.1 dyn row.  Bench mix (page `p`
tagged `p % 3` -> 2 A16 + 2 FP8 + 2 FP4 per tile): per thread per pair the
votes ~70 per operand (6 x (3 + 2 + 2 + 2) + 2 x 5 + 4), 8 blocks x ~40 = 320,
scale-word reloads ~8 per operand (2 `LDS.32` per step x 2 steps + their
`IADD`/`LEA` address forms; up to 6 per operand for a single-format tile),
loop overhead ~80 -> **~550 decode**, copies ~180, meta / protocol ~70 (+ 4
warp-uniform pending tests) -> **~800 per warp per pair**: at IPC 0.27 ->
2960 cycles = 1.50 + 0.08 = 1.58 us -> **~340 us**; band 320-370 (IPC
0.25-0.30).  The
accept row (<= 330) needs IPC >= 0.29 on this count.  The tile-uniform fast
path of F24c is not built (the bench has none); a sorted-page table with
unrolled per-class bodies is the only listed follow-up if the module lands at
331-370 (6.4).

## 4. Per-pair arithmetic, all modes (12 warps, per producer warp = per SMSP)

| mode | decode | votes/sync | copies | meta + protocol | total | at 0.27 IPC (+0.08 acq) | at 0.218 | at 0.35 |
|---|---|---|---|---|---|---|---|---|
| fp8 | 12 x 43 = 516 | 16 | 48 | ~78 | **~658** | 1.31 us -> 290-300 | 1.60 -> 350 | 1.03 -> 285 |
| fp4 (hot) | 12 x 44 = 528 | 16 | 48 | ~78 | **~670** | 1.33 -> 292-305 | 1.63 -> 355 | 1.05 -> 285 |
| fp4 (cold everywhere) | 12 x 52 | 16 | 48 | ~78 | 766 | 1.51 -> 332 | — | 1.19 -> 292 |
| mixed (bench mix) | 320 + 140 votes + ~80 loop | (in decode) | ~180 | ~74 | **~790** | 1.56 -> **~340** | 1.91 -> 415 | 1.22 -> 295 |
| a16 (unchanged module) | 0 | 0 | ~150 (stock form) | ~120 | ~270 | producer never paces (`acq` 1.0 us) | | |

Smem pipe (C11), all compressed modes: 1001 + 418 + 455 + ~101 = **1974 per
pair = 78 % of 2540** (fp4's smaller landings: `op_ld` -96, `LDGSTS` -24; mixed:
A16 pages have no `STS`: -140).  Consumer unchanged: 3301 warp-instructions per
pair per SM, T_c = 2540 cycles.

Fixed per-item costs (Q TMA, `barrier_O`, chunk-0 gather, epilogue) are in the
transport_a16 wall already and unchanged; the kernel prediction is
`283 x max(1, pair / 1.284) + ~3` us for q=1 and `288 x ...` for q=64.

## 5. Invariant changes (dataflow doc amendments to write when F25 lands)

- **A9 (supersedes A8's two-warp-group ownership).** One producer warp group
  for every module; thread `t` owns block `t % 8` of row `t / 8` of all six
  pages of each operand (A7 verbatim); the row's 8 B scale slot is copied by
  lane `b == 0` (8 B `cp.async`, as A8) and read by the row's eight lanes
  after every lane's own `cp.async.wait_group` **and** the operand's single
  `__syncwarp`, which is placed after the operand's own-landing loads and
  before its scale-slot loads (one barrier orders both: lane 0's completed
  copy before the other lanes' `LDS.32`, and every lane's landing loads
  before any lane's `STS`).  In the dynamic module the scale-slot reads follow
  the operand's first `__syncwarp` and every step's stores follow that step's
  `__syncwarp`.  fp4 landing: A7's, unchanged.  Dynamic copies: predicated-off
  lanes issue nothing (never src-size 0 to a slot another lane fills).
- **C3 (budgets).** 12 warps, `setmaxnreg` 136 / 184 (pool 64,512); a16
  module 72 / 216.  Build check (A4): no C7507, exactly two `USETMAXREG`
  (`DEALLOC 0x88`, `TRY_ALLOC 0xB8`), `STACK 0` in both regions, for fp8, fp4
  and dynamic; a16 byte-identical.
- **C4.** Back to [23]'s counts (128 / 384 / 128; Q by warp 0 of WG0; ping-pong
  ids 2, 3).  Recorded as a revert of A8's C4 line; the trait hooks
  (`producer_warp_groups_v`, `kFirstConsumerWG`) remain with value 1.
- **C6 (issue budget), restated per warp.** <= ~660 warp-instructions per
  pair per producer warp for fp8 / fp4 (decode 516-528 + 130), <= ~700 for the
  dynamic module; protocol share <= 60.
- **C9 (fold exactness), granularity.** The upper-bound test is voted once per
  operand per warp (all blocks of the warp's six pages) instead of per block;
  the predicate per block is unchanged; the cold path applies to the whole
  operand of that warp.
- **New C12 (branch-free bodies).** No branch, `VOTE`, `WARPSYNC` guard,
  uniform-register move or register join inside a block body; the only
  per-operand control is one uniform branch on the hoisted vote.  SASS check:
  `BRA.DIV` 0, `UMOV` 0 and `IMAD.MOV` <= 2 between the first `PRMT` and the
  last `STS.128` of an operand body.
- **New C13 (copy addressing).** Every compressed source address is one
  `IMAD.WIDE.U32` from a per-item 64-bit base and a uniform-register stride;
  `IADD3.X` / `VIADD` count in the pair loop <= 4; the partial arm is compiled
  only in the two per-item calls that can see tile `kv_tile_idx` - K(last)
  alone and the peeled first pair's V - and the loop compiles the `FULL` arm;
  the partial arm passes the unmodified in-bounds source with src-size 0 (no
  pointer select).
- **New C14 (protocol budget).** Loop-site SASS minus the decode, vote, sync,
  copy and meta counts <= 70 per pair per warp; pending records are 32-bit
  (`stage << 30`, `static_assert(NUM_STAGES <= 4)`, flags masked to bits
  24-25); the static modules' loop finish and their two single-operand finish
  sites are unconditional (the peel makes every finished pair a (K, V) pair
  and K(last) / the drain's V the only single operands); the dynamic module
  keeps one warp-uniform pending test per operand per site.
- **New C16 (E2M1 placement fold).** bf16 E2M1 decode by placement (x 2^-126)
  folds 2^126 into the block scale; exact iff `2^-126 <= |s g| < 3.9921875`,
  tested per operand per warp (vote) with the same +inf sentinel for |g| <
  2^-117; otherwise 8 `HMUL2` by 2^126 then the two-multiply exact form.  Tests:
  fp4 extremes with block scales 448 / 256 / 4 / 3.5 at g = 1 (cold and hot
  in the same operand of different warps), g = 0.5, g = 1.1 x 2^-118.
- **New C17 (dynamic copy body).** The dynamic module's copy body is six
  unrolled pages with six predicated `LDGSTS` sites per page per thread (A16
  rows x2 at 16 B, FP8 block 16 B, FP4 block 8 B, FP8 / FP4 scale row 8 B by
  the leader lane) of which exactly two execute; predicates from the row's
  page masks; sources by `IMAD.WIDE.U32` from six per-item bases; no loop, no
  per-page branch, no `SEL` on an address, no `LDS` on the address chain.
  C10's masks stay (they are the predicates and drive the decode loops);
  C10's rolled *copy* loops are withdrawn; C10's decode loops stay (two pages
  per step, hot / cold body chosen once per format per operand).
- **C11.** Unchanged (<= ~2000 wavefronts per pair; the ncu confirmation
  reports the four classes).

## 6. Verification artifacts (confirmation, not tuning; each with its accept / reject)

**Gate reconciliation (rev 2 numbers are the gates; decided here, before any
timing).**  Revision 1 of this document and the F25 brief carried three
numbers that revision 2 re-derived from one model (section 4's IPC 0.27
centre, sections 3.2 / 10.13): producer region `<= 1500` instructions,
protocol `<= 60` per pair, trace `fin <= 0.60` us.  Those were rev 1's
single-arm count (one operand body per site, no hot / cold pair, no peeled
first pair) and an implied body IPC of 0.45.  **The F25e gates are the rev 2
rows below and nowhere else**: region **2100-2400** total (loop site = hot +
cold x two operands + copies + protocol, plus the two exact single-operand
sites K(last) and drain), protocol **<= 70** per pair (loop-site count - 12 x
43 - 14 - 2 - 48 - 8), `fin` **<= 1.1** us (centre 1.0), `iss <= 0.30`, `acq
>= 0.25`, `IMAD.WIDE.U32` 24, `BRA.DIV` 0, `STACK 0`, `LDL/STL` 0, `VOTE.ALL`
2 in the loop site (4 per finish site in the dynamic module), `USETMAXREG`
`DEALLOC 0x88` / `TRY_ALLOC 0xB8`, ptxas -v without C7507, a16 module and
stock kernel byte-identical to `5cc416fd`.  A build that meets the rev 2 rows
and misses a rev 1 number is accepted; a build that meets a rev 1 number is
not thereby accepted.  Any later change to a gate is a change to this
paragraph and the 6.1-6.4 tables together, made before the run it judges.

**6.0 The copy phase, decided from the existing a16 build before any F25
timing (one ncu run, no new kernel code).**  The a16 module has no address
chain, the stock copy form and the same 96 `LDGSTS` per SM per pair, yet its
trace `iss` is 0.73-0.75 us.  Run the F24 metric set + pc sampling on
`transport_a16` (q=1, third launch) and read, at the `LDGSTS` PCs of the
producer region: `mio_throttle`, `lg_throttle`, `short_scoreboard` shares,
`sm__inst_executed_pipe_lsu` % of peak and warps issuing.  Accept (keep 12
warps): the `LDGSTS` PCs stall on `long_scoreboard` / `wait` (per-warp
latency) with `mio_throttle + lg_throttle <= 5 %` -> the copy phase is
per-warp latency that the 12-warp design overlaps by issue order.  Reject
(build 2E, or 2G = the copies interleaved into the decode body, which the
FULL-hoisted per-item bases make possible without new warps): `mio_throttle +
lg_throttle >= 15 %` at those PCs or LSU pipe >= 60 % during the pair -> the
96 `LDGSTS` are pipe-serialised behind the 1001 gmma-read wavefronts and the
fp8 centre is ~390 us, not 300.  **Not run in this worktree** (no remote runs
were made for rev 2); it is the first step of F25e.

**6.1 SASS (`cuobjdump -sass`, `*_paged_sm90_kernel_mask_1`, producer region
`USETMAXREG.DEALLOC .. EXIT`), before any timing:**

| item | accept | reject -> action |
|---|---|---|
| `USETMAXREG` | exactly two: `DEALLOC 0x88`, `TRY_ALLOC 0xB8`; `ptxas -v` no C7507; `STACK 0` (fp8, fp4, dyn) | C7507 / STACK > 0 in the **consumer** at 184 -> 120 / 192 (128 x 120 + 256 x 192 = 64,512) and the producer at four pages per step; in the **producer** at 136 -> four pages per step at 136 (152 / 176 only with a consumer proof at 176).  Never 128 / 192: 65,536 > the CTA's 64,512 |
| fp8 region count | **2100-2400** total; per site: loop site = 2 operands x (hot 258 + cold 258 + 14) + 48 copies + <= 70 protocol; K(last) and drain sites = one exact operand body (6 x 51) + copies + protocol; per operand body: 12 `LDS.64`, 6 `LDS.32`, 6 `F2FP.E4M3`, 6 `FSETP`, 5 `PLOP3`, 1 `VOTE.ALL`, 1 `BRA`, then 6 x {`F2FP.PACK`, 8 `PRMT`, 8 `IMAD.SHL`/`SHF`, 8 `LOP3`, 8 `HMUL2`, 2 `STS.128 [R+imm]`}; `VOTE.ALL` exactly 2 in the region (the loop site's two operands), 0 in the single-operand sites | any `VOTE` inside a body, `BRA.DIV` > 0, `UMOV` in a body, `IMAD.MOV` > 2 per body -> C12 violated; restructure before timing |
| scale-slot order (C12 addendum) | in each operand body the `WARPSYNC` (if emitted) or the program point of the `__syncwarp` lies after the 12 landing `LDS.64` and before the first `LDS.32` of the scale slot; no `LDS.32` of a scale slot precedes it | otherwise the ordering of A9 is not what was compiled |
| copy path | `IMAD.WIDE.U32` 24 per loop pair, `IADD3.X` <= 4, `VIADD` <= 4 in the loop; `LDGSTS` 24 per pair per thread (12 + 12 predicated) in the loop site; `BSSY/BSYNC` 0 in the loop site (no pending test, no `FULL` test) | more -> the per-item bases were not hoisted (check `make_bases` live set) or the peel did not fold |
| protocol | loop-site count - 12 x 43 - 14 - 2 - 48 - 8 <= 70 | over -> list the extra opcodes before timing |
| fp4 | body as fp8 with `SHF`/`IMAD.SHL` 2 + 4 `PRMT` + 4 `SHF` + 4 `LOP3` per word pair, `HMUL2` 8 hot / 16 cold; payload `LDGSTS.64` (8 B, unchanged form) | |
| dyn | 36 predicated `LDGSTS` sites per operand copy body (six per page), `SEL` 0 on the address chain, no `FLO`/`POPC` in the copy path, `LDL/STL` 0 in the pair loop; 4 `VOTE.ALL` per finish site (two formats x two operands); the format loops reload the scale word per page (2 `LDS.32` per step, section 3.5) - expected, not a miss; `FLO` (`__ffs`) only in the format loops' page selection | rolled copy loop or LDL present -> C17 / C2 violated |
| a16 module, stock paged kernel | byte-identical to `5cc416fd` | any diff = the shared-file revert is incomplete |

**6.2 Trace (`MIXED_FA3_TRACE`, fp8 q=1, CTA 0 items 0/1).**  The traced pairs
are CTA 0's first two items (kernel start, all 132 SMs streaming: the a16 trace
pair is 2.02 us against the 1.28 us bench average) and carry the stamp overhead
the F24 record books as `gap` (0.54-0.67 us), so segments are compared with
each other and with the F24 trace, not with T_c.  Thresholds are derived from
**one** model, the section 4 centre (IPC 0.27): decode + votes 532
instructions = 1970 cycles = 1.0 us of `fin`; copies + meta + protocol ~130 =
~480 cycles = 0.24 us of `iss` (rev 1's `fin <= 0.60` implied IPC 0.45 and
would have rejected a build that lands exactly on the prediction):

| segment | accept | reject -> action |
|---|---|---|
| `iss` | <= 0.30 us (from 0.62-0.85) | > 0.5 with the C13 counts met: LDGSTS dispatch is SM-serialised -> 2E (role split) or 2G (copies interleaved into the decode body); 6.0 should have said so first |
| `fin` (`expK + expV`) | **<= 1.1 us** (centre 1.0; from 1.0-1.2 on F24's 6 blocks per thread - i.e. the same time for twice the blocks) | > 1.3 with C12 met: the body IPC did not follow the structure -> pc-sample the body (6.3) before any restructuring |
| `acq` | >= 0.25 us (the producer waits on the consumers' release = consumer-bound) | ~0.1 with `iss + fin` <= 1.3: trace overhead masks it; decide on the bench row |
| `wait` | <= 0.05 us | > 0.1: landing latency exposed (3.3) -> scale copies first, then re-measure |

**6.3 ncu (fp8 q=1, `--repeats 1`, third launch, `f23_run_ncu.sh` metric set +
`f25_ncu_classes.py`):**

| metric | accept | reject |
|---|---|---|
| producer `inst_executed` per pair | 2500-2750 (4 x 655 +- 5 %) | > 2950: count model wrong, re-read the SASS |
| producer per-warp `selected` share | >= 25 % | < 22 % with the count met: IPC did not move; pc-sample by opcode, decide 2E vs four-pages-per-step |
| producer stall mix | `branch_resolving` <= 1 %, `no_inst` <= 2 %, `dispatch` <= 12 %, **`mio_throttle` + `lg_throttle` <= 5 %, `short_scoreboard` <= 10 %** | `mio_throttle` / `lg_throttle` above -> the smem pipe at 78 % is the binding side (C11's average hides the gmma bursts): 2E / 2G |
| LSU pipe % of peak | within 5 % of transport_a16's (43.8 % at F24's cadence; expect ~60 % at parity) | |
| consumer K-wait PC (`@P0 BRA` at `consumer_wait`) | <= 3 % of consumer samples (from 16.5 %) — the consumer-bound proof | 8-16 %: producer still paces |
| consumer region `inst_executed` per pair | == F24b's 3301 per SM (the consumer at 184 rematerialises nothing) | more: the 184 budget costs the consumer; measure T_c(184) with the a16 control build (6.4) |
| tensor pipe active | within 3 % of transport_a16's | |
| smem wavefronts per pair, by class | total <= 2050; `op_st` <= 450, `STS.128` <= 4.5 per instruction; `LDGSTS.128` 4.0, fp4 `LDGSTS.64` <= 4.0 | |
| `smsp__issue_active` | 55-65 % | |

**6.4 Bench (`benchmarks/bench_fa3_mixed_page_transport.py --q-lens 1 64
--repeats 1 --trials 5`, nkcut2 lock, co-tenant rule: bursts < 1.5 ms; min /
median / max):**

| row | accept | reject / re-derive |
|---|---|---|
| stock_a16 | 297-303 / 306-312 | drift > 3 % -> session offset, rerun |
| transport_a16 | 281-290 / 284-292 | any change on a byte-identical module = machine |
| transport_a16 control at 72 / 184 (cheap build, not merged: `CONSUMER_REGS` 184 for the a16 module) | within 2 % of transport_a16 -> T_c(184) = T_c(216) | slower -> the consumer lost time to 184; the fp8 shortfall below is bounded by this before the producer is blamed |
| fp8 | **<= 330 / <= 330** (band 288-318 / 292-322) | 331-345 with 6.3's `selected` < 22 %: IPC; with `selected` >= 25 % and `iss` > 0.5: 2E / 2G; > 345: count model wrong |
| fp4 | **<= 330 / <= 330** (band 288-325 / 292-330) | as fp8; additionally check the vote path taken (cold-path `HMUL2` executed count via ncu source view = 0 on the bench payload) |
| mixed | target <= 330; **centre ~340** (band 320-370) | 331-370 as predicted: the count (790) is the cause; (only then) a sorted-page table with unrolled bodies per (n8, n4) class; > 370: re-read the dyn SASS row |

**6.5 Correctness (`tests/attention/run_fa3_mixed_page_transport.py`; pytest is
banned):** the 88 cases of F24 bit-exact (parity-tail cases kept: kv_len 149
exercises the partial page of K(last) *and* V(last), i.e. both partial calls of
the peel) + **NaN-pattern tail cases** (kv_len 149, page 0 unreferenced by any
request and filled with E4M3 NaN codes in payload and scales, rows past
`kv_len` of the partial page filled the same way: a `FULL` arm misused on
V(last) or a src-size-16 copy of a page-0 row produces NaN `sf2` and NaN in O,
which zeros could not show; fp8, fp4, mixed x q 1, 64) + fp4 extremes per C16
(the FP4 global scale `g` in {1, 0.5, 1.1 x 2^-118}; block scales {448, 256,
2^-9, 2^-7, 2^-6, 1}: 448 and 256 fail the 2^126 fold, the rest pass, within
one operand of different warps) + a dynamic case whose tile has 0 FP8 pages
and one with 6 FP8 pages (mask edge cases of the two-page steps) + the
many-items case (C7).  Also the E4M3 / E2M1 NaN-code note of C9 stands.

## 7. Do not build

1. **Consumer-side decode (2A).**  The F24 ncu consumer region shows <= ~530
   cycles per pair per warp of non-producer wait against >= 850 cycles of
   added pipe time.
2. **SWAP_AB / register-operand K or V (2B).**  4x softmax instructions, P^T
   store, 128-register O^T per WG, CTA_KV 128 -> S = 2.
3. **fp8 wgmma (2C).**  Block scales lie on the reduction axis for both K
   and V under R1; not bit-exact.
4. **A second producer warp group by page parity (F24b).**  Measured: +30 %
   instructions, +15 % throughput, IPC 0.13 -> 0.10 per warp.  The traits keep
   `NUM_PRODUCER_WGS` but every module sets it to 1.
5. **Per-block fold votes (F24a's form).**  Measured slower than [23]; the
   body-internal sample cluster (section 1) is one third of the body's time.
6. **Three producer warp groups** (48 registers each) or **S = 4 stages**
   (smem).
7. **The E2M1 LUT** in the bf16 modules once C16 passes its extremes (keep
   the LUT for the f16 instantiation only).
8. **F24c's rolled, mask-driven copy loops** (C17 replaces them); the F24c
   decode loops stay.
9. **Any host-side bound on g** (the .item() sync); the vote is exact for
   every finite input.
10. **Trace-segment absolute values as design inputs**: the trace covers CTA
    0's first two items under kernel-start contention and carries ~0.5-0.7 us
    of stamp overhead per pair (`gap`); use ratios, SASS counts and ncu.
11. **16 B `cp.async` from 8 B-aligned sources** (fp4 blocks, scale rows) and
    any "one copy form" that depends on it; **src-size 0 as a lane
    predicate** when another lane of the same instruction fills the
    destination.
12. **An unconditional finish on a pair that can have nothing pending** (the
    first loop iteration without the peel; any dynamic-module site).

## 8. Files touched (implementation order; each step is independently correct and bit-exact; the a16 and stock kernels byte-identical throughout)

- **F25a — layout, registers, shared-file revert.** `kernel_traits.cuh`: drop
  `NUM_PRODUCER_WGS` and the thread-count overrides (12 warps for every
  module), `PRODUCER_REGS = kMixedStaticFormat == 0 ? 72 : 136`,
  `CONSUMER_REGS = 216 : 184`, pool assert for 384 threads; keep the mask
  comment.  `named_barrier.cuh`, `epilogue.cuh`, `sparse_mainloop.cuh`: the
  `5cc416fd` text.  `prefill_sm90.cuh`: `NUM_COPY_THREADS` / role test back to
  the `5cc416fd` text, register hook kept.  `sparse_mixed_mainloop.cuh`: the
  [24b] two-warp-group text folded to one warp group (`PAGES_PER_THREAD` 6,
  `own_u / own_h`, `parity_mask`, `TileRegs::page`, `gather_thread`,
  `q_issuer`), `static_assert(NUM_STAGES <= 4)` staged.  Bodies, copies and
  protocol unchanged (F24 shape at 136 registers, six pages per thread).
  Gate: a16 + stock byte-identical; fp8 `USETMAXREG 0x88 / 0xB8`, no C7507.
- **F25b — bodies.** `expand_operand<VOTE>`: landing loads first, one
  `__syncwarp`, scale loads, hoisted per-operand vote, two six-block arms
  (`VOTE = true`) or the exact body (`VOTE = false`, single-operand sites);
  `expand_block` loses the vote and the `__syncwarp`; `e2m1x8_to_a16` bf16
  arm becomes the 2^-126 placement; `make_bases` `gs4 = g * 2^126` with the
  sentinel; `kFp4Fold*` constants; the pair-level `__syncwarp` after `DEPBAR`
  removed.  fp4 landing and all copy forms unchanged.  Gate: 6.1 body rows;
  tests 88 + fp4 extremes.
- **F25c — copies and protocol.** `OperandBases` gains the 64-bit per-item
  bases (`compressed_src` removed); `copy_compressed_page` one `IMAD.WIDE` per
  copy, no pointer select; `issue_operand<PARTIAL>`; `pair_step` lambda with
  the peeled first pair and the `(FULL, FULL)` loop; `finish_pending_pair`
  unconditional (static) / per-operand tests (dynamic); `finish_one` for
  K(last) and the drain; 32-bit pending words.  Tests: NaN-pattern tail cases.
  Gate: 6.1 copy / protocol rows; trace 6.2.
- **F25d — dynamic module.** `read_meta` full row for every module;
  `issue_tile_copies` dynamic arm -> six unrolled pages with predicated
  per-format copies; `expand_operand` dynamic arm -> up-front per-format votes,
  `expand_format_pages<FP8, HOT>` two pages per step.  Gate: 6.1 dyn row,
  tests with 0 / 6 FP8 pages per tile.
- **F25e — measurements**, in this order: 6.0 (a16 ncu probe), 6.1 SASS gates,
  6.2 trace, 6.3 ncu, 6.4 bench (incl. the 72/184 a16 control), then this
  document's results section and the dataflow amendments of section 5.

## 9. The floor, stated

With BF16 operands materialised in shared memory the compressed pair cannot be
shorter than the shared-memory pipe's 1974 wavefront-cycles (1.0 us; ~230 us
kernel including fixed costs) nor, in this kernel, shorter than the consumer's
1.28 us; the tensor floor is 0.78 us.  Parity (283-290 us) is therefore the
best the compressed modes can do here and needs 78 % of the smem pipe, 58 % of
the issue slots and <= 39 % of any pipe — it is not resource-bound; it is
bound by whether one producer warp per SMSP issues ~645 instructions at
>= 0.26 IPC.  If 6.3 shows the count met and the rate not, the residual is
latency tolerance, and the fallback that adds a second warp per SMSP *without*
duplicating protocol is 2E; no further lever exists on the producer side
short of changing the operand format (per-token scales, which would let the
scale ride on P and Q and remove the `HMUL2` and half the decode) or the
consumer (C5).

## 10. Judge blockers on revision 1 and their resolutions (revision 2)

Each item names the rev 1 text, the defect and what rev 2 does instead; the
design changed where the blocker required it.

1. **`FULL` hoist missed V(last)** (3.2 "Copy issue", 8 F25c).  The loop's
   first iteration is `produce_pair(kv_tile_idx - 1, kv_tile_idx)`: V(last)
   carries the partial tile's `valid`, and the `FULL` arm would have copied
   its rows past `valid` with src-size 16 (D4 violated, NaN-pattern garbage
   reaching O through P = 0 x NaN).  **Fix: the first loop pair is peeled**
   (`pair_step(kv_tile_idx, FULL_K, PARTIAL_V)`), K(last) alone is
   `(PARTIAL_K, -)`, the loop `t = kv_tile_idx - 1 .. swa_begin` is `(FULL,
   FULL)`; `issue_operand<PARTIAL>` is a compile-time tag.  The parity-tail
   cases stay in 6.5 and a NaN-pattern tail case is added (page 0
   unreferenced and NaN-filled, rows past `kv_len` NaN-filled) so that a
   `FULL` misuse cannot pass on zeros.
2. **"Unconditional finish" was unsafe on the first loop iteration** (C14):
   with K(last) finished before `barrier_O.wait`, the first steady-state
   finish had `K.pending == V.pending == 0` and would have decoded stage 0
   under the consumer and arrived twice on `full_barrier[0]`.  **Fix: by the
   peel, the peeled pair has no finish call**, every loop finish handles a
   (K, V) pair issued one iteration earlier (both pending in the static
   modules: unconditional is now correct), and the two single-operand sites
   (`finish_one(K)` after K(last), `finish_one(V)` at the drain) are
   unconditional because the last pair always has `tK = -1`.  The dynamic
   module keeps one warp-uniform `if (op.pending)` per operand per site (A16
   tiles).  Protocol budget restated at <= 70 with its items (2F).
3. **Scale-slot read order raced** (3.1 listed the scale `LDS.32` before the
   `__syncwarp`, contradicting A9).  **Fix: order = own landing loads ->
   `__syncwarp` -> scale `LDS.32` -> chains -> vote -> decode -> `STS`**; one
   barrier per operand serves both orderings; the pair-level `__syncwarp`
   after `DEPBAR` is removed; a C12 SASS check on the position of the barrier
   relative to the first scale `LDS.32` is added (6.1).  The dynamic module's
   steps each carry a `__syncwarp` before their stores; their scale reads
   follow the operand's first barrier.
4. **Dynamic copy B with "src-size 0 on the other lanes" was a WAW race**
   (a `cp.async` with src-size 0 still writes cp-size zeros; seven lanes
   would have zero-filled the slot lane 0 fills in the same instruction).
   **Fix: the scale copies are predicated on `leader` (and the format
   predicate)**: a predicated-off `LDGSTS` issues nothing; no lane targets
   another lane's destination (A9, C17).
5. **16 B cp-size from 8 B-aligned sources** (fp4 blocks at `blk * 8`, scale
   rows at 8 B head stride) was asserted, not shown.  The PTX ISA text could
   not be fetched here (single-page document, fetch limit); the design no
   longer needs the permissive reading: **fp4 blocks and scale rows keep the
   8 B `cp.async`** (the forms that exist today and measured 3.98 wavefronts),
   16 B is used only where the host check guarantees 16 B (A16 rows, FP8
   blocks) - CUTLASS's own `cp_async_zfill` discipline (src-size in {0,
   cp-size}, `memory_sm80.h`).  C17's "one `LDGSTS.128` form" is withdrawn and
   re-derived as six predicated sites per page (two `LDGSTS` sizes); the
   wavefront counts are the measured ones (4.0 / 3.98 / <= 4).  Do-not-build
   item 11.  Consequence: the rev 1 fp4 landing change (and its 2-way
   `LDS.32` conflict flagged by the review) is dropped; the landing is A7's.
6. **Dynamic pending record undefined after F25d** (masks dropped from the
   chunk table while the decode stayed mask-driven).  **Fix: the masks
   stay** in `tags[4], tags[5]` / `w7` bits 0-5, 8-13 (`chunk_store`
   unchanged); the pending word is `(w7 & 0x03FFFFFF) | stage << 30`
   (masks, `valid`, flags bits 0-1; `static_assert(NUM_STAGES <= 4)`); the
   masks are also the per-page copy predicates (no tag-byte derivation, ~0
   instructions).  `read_meta` reads the full row for the dynamic module too
   (page indices in registers at unrolled `j`).
7. **Shared Hopper files: "revert" vs "unchanged in text", and the register
   hook is not in `5cc416fd`.**  **Fix: per-file table in 3.2 (C4)**:
   `named_barrier.cuh`, `epilogue.cuh`, `sparse_mainloop.cuh` and the
   warp-group plumbing of `prefill_sm90.cuh` revert to the `5cc416fd` text;
   `prefill_sm90.cuh` keeps the `kMixedTraits` register hook (discarded at
   compile time for stock traits), `kernel_traits.cuh` keeps the mixed-only
   static-format / register constants; byte-identity gate for a16 + stock
   kept; the two residual shared-file items are named for the user's
   sign-off.
8. (= 1, second statement.)  Same fix; the NaN-pattern test is 6.5's.
9. (= 5, second statement.)  Same fix: 8 B `cp.async` kept; the dynamic body
   is two `LDGSTS` sizes under predication; the "exactly two `LDGSTS.128`"
   and 4-wavefront claims are replaced by the measured per-form counts.
10. (= 4, second statement.)  Same fix: predicate `p8 & leader` / `p4 &
    leader`; stated in A9 / C17.
11. (= 3, second statement.)  Same fix; two `__syncwarp` per pair (one per
    operand, between the landing loads and the scale loads); C12 SASS check.
12. **Register fallback 128 / 192 would hang** (65,536 > the CTA's 384 x 168
    = 64,512).  **Fix: 6.1's reject action lists the admissible splits (P +
    2 C <= 504: 136/184, 120/192, 152/176, 104/200, 72/216)**; 136/184 has
    zero slack: a consumer C7507 means 120/192 with four pages per step, a
    producer C7507 means four pages per step at 136 - never raising the
    consumer without lowering the producer.
13. **6.1 / 6.2 gates inconsistent with the model** (`fin <= 0.60` implied
    IPC 0.45; "region <= 1500" counted one site; the protocol formula mixed
    dynamic and static counts).  **Fix: 6.2 thresholds from the section 4
    centre** (`fin <= 1.1` accept, `> 1.3` reject; `iss <= 0.30` / `> 0.5`);
    6.1's region row is per site with the three inlined finish sites named
    (loop: hot + cold per operand; K(last) and drain: one exact operand
    body), total 2100-2400; the protocol row is the loop-site count minus
    the named static counts.
14. **The copy-phase assumption (iss -> 0.2 us) is contradicted by the a16
    module's own 0.73 us** with no address chain.  **Not resolvable by
    reading**: it needs one ncu run of the existing a16 build (LDGSTS PCs:
    `mio_throttle` / `lg_throttle` / `short_sb`, LSU pipe %, warps issuing).
    Rev 2 makes it **gate 6.0**, ordered before any F25 timing, states both
    outcomes (per-warp latency -> keep 12 warps; pipe-serialised -> fp8 centre
    ~390 us and 2E or 2G is the build) and records that no remote run was
    made in this worktree.  2G (copies interleaved into the decode body; no
    new warps) is named as the cheaper alternative to 2E because the
    FULL-hoisted per-item bases make the copy issue position-independent.

Review notes acted on: the fp4 landing bank-conflict note (moot, landing
unchanged); the consumer-at-184 T_c note (a16 control build at 72/184 in 6.4,
consumer region count gate in 6.3); the `mio_throttle` / `lg_throttle` /
`short_sb` stall shares and LSU pipe % (6.3); `static_assert(NUM_STAGES <= 4)`
and flags masking (C14); the per-operand pending tests counted in the dynamic
budget (3.5).  The mixed module's re-derived count (~790) puts its centre at
~340 us, above the accept row; the table in section 0 says so.

## 11. As written (filled per step; F25a-d in this worktree, F25e not run)

### As written: F25a (layout, registers, shared-file revert)

Files: `include/flashinfer/attention/hopper/{kernel_traits,prefill_sm90,
named_barrier,epilogue,sparse_mainloop,sparse_mixed_mainloop}.cuh`.  Not built
or run in this worktree (review by reading; the remote gates are F25e's).

**Shared files (3.2, C4).**  `named_barrier.cuh`, `epilogue.cuh`,
`sparse_mainloop.cuh`: `git checkout 5cc416fd --` (byte-identical to the [23]
text; `producer_warp_groups_v`, `kFirstConsumerWG` and the relaxed
`static_assert` are gone).  `prefill_sm90.cuh`: `NUM_COPY_THREADS =
cutlass::NumThreadsPerWarpGroup`, `pipeline_params.role` by `warp_group_idx ==
0` and `if (warp_group_idx == 0)` are the `5cc416fd` text again; the only
remaining difference against `5cc416fd` is the register hook (`else if
constexpr (Ktraits::kMixedTraits) warpgroup_reg_dealloc<Ktraits::PRODUCER_REGS>()
/ warpgroup_reg_alloc<Ktraits::CONSUMER_REGS>()`) plus its comment - 10 lines,
discarded at compile time for stock traits (`kMixedTraits = false` in
`AttentionKernelTraits`, unchanged).  `kernel_traits.cuh`
(`MixedAttentionKernelTraits`, mixed-only): `NUM_PRODUCER_WGS` and the
`NUM_WARPS` / `NUM_THREADS` / `NUM_PRODUCER_THREADS` overrides removed (the base
traits' 12 warps / 128 producer threads apply); `PRODUCER_REGS =
kMixedStaticFormat == 0 ? 72 : 136`, `CONSUMER_REGS = kMixedStaticFormat == 0 ?
216 : 184`; `static_assert(BaseTraits::NUM_WARPS == 12)`; the pool assert is
against `(65536 / NUM_THREADS) * NUM_THREADS = 64512`, which is what
`__launch_bounds__(384, 1)` gives the CTA (128 x 136 + 256 x 184 = 64512
passes; 128 x 128 + 256 x 192 = 65536 would fail to compile).  The
`MixedTileMeta` mask comment stays (the masks stay, 3.5).

**`sparse_mixed_mainloop.cuh` (one warp group).**  `NUM_COPY_THREADS =
cutlass::NumThreadsPerWarpGroup`, `PAGES_PER_THREAD = PAGES_PER_TILE` (6),
`PAGE_STEP_BYTES` / `SCALE_PAGE_STEP_BYTES` removed (page `j` is at `j *
PAGE_REGION_BYTES` / `j * SCALE_PAGE_BYTES`, immediates), `own_u` / `own_h` /
`parity_mask` / `page_tok0` removed (`u = t`, `h = 0` everywhere:
`make_bases` has no parity offsets, `static_page(j)`, `TileRegs::page(j)`),
the chunk-table gather and the Q issue are the [23] text again (all 128
threads gather; `warp_idx_in_warpgroup == 0` issues Q).  The dynamic arms
keep F24c's shape with the parity restriction removed (`ma = 0x3F & ~(m8 |
m4)`, `page_of(i)` selects `pg[i]`): F25d replaces them.
`static_assert(NUM_STAGES <= 4)` is staged for the 32-bit pending word (F25c).
Bodies, copies, pending records and finish protocol are F24's, now running six
pages per thread at 136 registers - this intermediate state is bit-exact by
construction (the same code that ran three pages per thread per warp group).

**Expected artifacts.**  a16 module and stock paged kernel byte-identical to
`5cc416fd`: the a16 module's `load()` text after constant folding is the [23]
text (no `own_u`, no gather predicate, no Q-issuer select), its `USETMAXREG`
immediates are `0x48 / 0xD8` from `PRODUCER_REGS / CONSUMER_REGS = 72 / 216`;
fp8 / fp4 / dynamic `USETMAXREG 0x88 / 0xB8`, `ptxas -v` no C7507, `STACK 0`
(fp8, fp4; the dynamic module's F24 32 B frame may persist until F25d).
`BAR.SYNC` counts back to [23]'s: `kProducerWG` 128, `kQueryEmpty` 384.

### As written: F25b (bodies: loads-first, one barrier per operand, hoisted vote, E2M1 placement)

Files: `include/flashinfer/attention/hopper/sparse_mixed_mainloop.cuh`,
`tests/attention/run_fa3_mixed_page_transport.py`.  Not built or run here.

**Data flow (static modules, `expand_operand<VOTE>`).**  In program order:
(1) `pk[j] = load_packed<FP8>(e, static_page(j).off)` for the six pages (two
`LDS.64` fp8 / two `LDS.32` fp4 each, this thread's own landings); (2)
`__syncwarp()`; (3) `sw[j] = lds32(e.sc + j * SCALE_PAGE_BYTES)`, `v[j] =
scale_product<FP8>(b, sw[j], t)` = `mul.rn.f32(f32(s_j), gs)`; (4) bf16 with
`VOTE`: `ok = AND_j fold_ok(v[j])` (`|v| < kFp8FoldMax`, the one constant for
both formats), `__all_sync` -> `expand_static_arm<FP8, false>` (hot: `sf2_j =
a16x2(v[j])`) or `expand_static_arm<FP8, true>` (cold: `g = global_plain<FP8>`
once, `sf2_j = a16x2(f32(s_j) * g)`, `expand_block<FP8, true>` multiplies the
placed halves by `kTwoPow120Bf16x2` / `kTwoPow126Bf16x2` first); bf16 without
`VOTE`: the cold arm only; f16: the hot arm with the plain scale, no vote.
`expand_block<FP8, EXACT>(e, p, sf2, off)` is `static`, has no vote, no
barrier and no branch: decode (4 x `e4m3x4_to_a16` or 2 x `e2m1x8_to_a16`),
[`EXACT`: 8 `HMUL2` by 2^k], 8 `HMUL2` by `sf2`, 2 `STS.128 [d + imm]`.

**E2M1 placement (bf16).**  `e2m1x8_to_a16<bf16>(src, out)`: `w4 = src << 4`;
`out[k] = (prmt(w4, src, sel_k) << 2) & 0x81C081C0` with `sel_k` = `0xC480,
0xD591, 0xE6A2, 0xF7B3` (selector nibbles low to high: `w4.byte_k`, sign-rep
of `w4.byte_k`, `src.byte_k`, sign-rep of `src.byte_k`); the half `s | 000000
ee | m 000000` is the E2M1 value x 2^-126 (code 001 -> the bf16 subnormal
2^-127).  13 instructions per 8 nibbles.  The f16 arm keeps CUTLASS's LUT.
`make_bases`: `gs4 = |g| >= 2^-117 ? g * 2^126 : +inf` (`kFp4Fold`,
`kFp8FoldMinGlobal`, `kFp8FoldSentinelBits`); f16: `gs4 = g`.  Constants
`kFp4Fold = 0x1p126f`, `kFp4FoldMax = 3.9921875 * 2^126` (== `kFp8FoldMax`),
`kTwoPow126Bf16x2 = 0x7E807E80`.  Landing, copy forms, `expand_bases`,
`load_packed`: unchanged (fp4 blocks stay 8 B `cp.async` into half `b & 1` of
chunk `(b/2 + 4 (r & 1)) ^ (r & 7)`).

**Control flow.**  `finish_pending_pair` loses the pair-level `__syncwarp`
after `cp_async_wait` (the barrier is inside each operand: (2) above);
`expand_pending` calls `expand_operand<true>` at every site in this step (the
single-operand `VOTE = false` sites arrive with F25c's protocol change).  The
dynamic module keeps F24c's format-outer one-page-ahead loops with an interim
per-block vote (`expand_block_voted<FP8>`: `scale_product`, `__all_sync`,
`__syncwarp`, `expand_block<FP8, hot|cold>`) and one `__syncwarp` at the top of
its `expand_operand` arm so that the first pages' scale-slot loads follow a
warp barrier; F25d replaces this arm.

**Tests.**  `_extreme_transport(..., gs)` sets the FP4 global scales to `gs` as
well and `dec4` models `A16(x * A16(float(s) * g))`; `_run_extremes` runs
`("fp8", "fp4", "mixed") x (1, 64) x (1, 0.5, 1.1 x 2^-118)` = 18 cases (was
12; the fp4 static module is `static_format = 2`).  Matrix: 64 + 2 + 6 + 4 + 18
= 94.

**Expected artifacts (6.1 body rows).**  fp8 loop-site operand body: 12
`LDS.64`, `WARPSYNC` (or none) after them, 6 `LDS.32`, 6 `F2FP.F16.E4M3`, 6
`HADD2.F32`, 6 `FMUL` (no `.FTZ`), 6 `FSETP`, `PLOP3` tree, one `VOTE.ALL`, one
`BRA`; hot arm 6 x {`F2FP.BF16.PACK_AB`, 8 `PRMT`, 8 `SHF`/`IMAD.SHL`, 8 `LOP3`,
8 `HMUL2.BF16`, 2 `STS.128 [R+imm]`}; cold arm the same with 8 more `HMUL2`
per block and one `LDC`/`LDG` of the global scale; `BRA.DIV` 0, `UMOV` 0 in the
bodies.  fp4 body: 12 `LDS.32` + 6 `LDS.32`, 26 integer ops per block
(`SHL`, 4 x {`PRMT`, `SHL`, `LOP3`} x 2), `HMUL2` 8 hot / 16 cold, no `PRMT`
LUT constants (`0xC0800000`, `0x3F3F3F00` gone from the bf16 module).

### As written: F25c (copies and protocol: per-item bases, the peel, unconditional loop finish, 32-bit pending words)

Files: `include/flashinfer/attention/hopper/sparse_mixed_mainloop.cuh`,
`tests/attention/run_fa3_mixed_page_transport.py`.  Not built or run here.

**Data flow (copies, C13).**  `OperandBases` gains `p8, s8, p4, s4` (64-bit:
span pointer + `head * hs + row * ts + blk * BLOCK_BYTES`, computed once per
work item by `compressed_base`) and `p8_ps, s8_ps, p4_ps, s4_ps` (32-bit page
strides); `CompressedSrc` / `compressed_src` (per-tile recomputation) are
gone.  `copy_compressed_page<FORMAT, FULL>(b, page, pp, stage, valid, t)`:
`src = page_src(base, page, stride)` = `base + page * stride` (one
`IMAD.WIDE.U32`) for the block and for the scale row; FULL: `cp16` (fp8) /
`cp8` (fp4) + `@leader cp8`; partial: `cp16_zfill(land, src, v)` /
`cp8_zfill` + `@leader cp8_zfill(sdst, ssrc, v)` with `v = tok0_j + row <
valid` and the **unmodified** source (no `v ? src : base` select: the address
is in bounds for every (page, row) and src-size 0 reads nothing).  All copy
forms are the existing ones (16 B fp8 blocks, 8 B fp4 blocks, 8 B scale rows).
`copy_a16_page` is untouched (a16 byte-identity).

**Control flow (C13 / C14).**  `issue_operand<PARTIAL>`: `STATIC_A16` keeps the
[23] runtime `valid == CTA_KV` test; the compressed and dynamic modules compile
`issue_tile_copies<!PARTIAL>` only.  `produce_pair(kpart, vpart, tK, tV)` is a
generic lambda over `FullTag` / `PartialTag` (`std::integral_constant<bool>`):
`issue_operand<PARTIAL_K>` / `<PARTIAL_V>`, and `if constexpr (FINISH)
finish_pending_pair()` with `FINISH = !(PARTIAL_K || PARTIAL_V)` - the two
partial calls compile no finish.  Call sites: `produce_pair(PartialTag{},
FullTag{}, kv_tile_idx, -1)` (K(last) alone) then `finish_one(K)` (C7);
`pair_step(t, kpart, vpart)` carries the chunk-table gather / store / barrier
and the prefetch and calls `produce_pair(kpart, vpart, t-1 | -1, t)`; the
compressed modules run `pair_step(kv_tile_idx, FullTag{}, PartialTag{})`
(peeled: V(last) partial, K(last-1) full) and then `for t = kv_tile_idx - 1 ..
swa_begin: pair_step(t, FullTag{}, FullTag{})`; the a16 module runs `for t =
kv_tile_idx ..` with the same body (its text after folding is the [23] loop);
drain `finish_one(V)`.  `finish_pending_pair()`: `cp_async_wait<1>`,
`expand_pending<true, DYNAMIC>(K)`, `(V)`, `fence_view_async_shared`,
`commit_pending<DYNAMIC>(K)`, `(V)` - no pending test in the static modules
(every pair it finishes is a (K, V) pair issued one iteration earlier), one
warp-uniform `if (op.pending == 0) return` per operand in the dynamic module.
`finish_one(op)`: `cp_async_wait<0>`, `expand_pending<false, DYNAMIC>` (the
exact body, no vote), fence, `commit_pending<DYNAMIC>`.  Pending words:
`TileRegs::pending_word(stage) = (w7 & 0x03FFFFFF) | stage << 30`
(`Operand::pending / staged` are `uint32_t`; `pending_stage`, `pending_mask8 /
4` read the fields; `static_assert(NUM_STAGES <= 4)` from F25a).

**Why the loop finish is safe without a test (10.2).**  K(last) is finished by
`finish_one(K)` before `barrier_O.wait`; the peeled pair issues (K(last-1),
V(last)) and finishes nothing; loop iteration `t` finishes the pair issued at
`t + 1` (or the peeled pair), whose `tK = t >= swa_begin` and `tV = t + 1` were
both issued, so both pending words are nonzero in the static compressed
modules (`kFlagFilled`).  A single-tile item (`kv_tile_idx == swa_begin`) has
the peeled pair with `tK = -1`, zero loop iterations and `finish_one(V)` at
the drain; K is never pending at the drain because the last pair always has
`tK = -1`.

**Tests.**  `_run_parity_tail(mode, q_len, nan_tail=True)`: shape gains one
physical page, `kv_indices = arange(1, ...)` (page 0 unreferenced), page 0 and
rows 5..15 of every request's last page are filled with `0xFF` payload bytes
(E4M3 NaN), `0x7F` scale bytes (E4M3 NaN) for both formats and `0x7FC0` (bf16
NaN) in the A16 reference rows; fp8 / fp4 / mixed x q 1, 64 = 6 cases.  Matrix:
94 + 6 = 100.

**Expected artifacts (6.1 copy / protocol rows).**  fp8 loop site: 24
`IMAD.WIDE.U32`, 12 `LDGSTS.E.128` + 12 `@P LDGSTS.E.64`, no `ISETP` on
`valid`, no `BSSY/BSYNC` around the finish, `LDGDEPBAR`, `DEPBAR.LE SB0, 0x1`,
one `FENCE.VIEW.ASYNC.S`, two `SYNCS.ARRIVE`; K(last) site and drain site:
`DEPBAR.LE SB0, 0x0`, one operand's exact body (6 x 51), one `SYNCS.ARRIVE`;
`VOTE.ALL` 2 in the region; `IADD3.X` / `VIADD` <= 4 in the loop.

### As written: F25d (dynamic module: predicated per-format copies, format-outer decode with hoisted votes)

Files: `include/flashinfer/attention/hopper/sparse_mixed_mainloop.cuh`,
`tests/attention/run_fa3_mixed_page_transport.py`.  Not built or run here.

**Chunk table.**  `chunk_store`'s dynamic arm is F24c's (masks in `tags[4],
tags[5]`, `w7 = m8 | m4 << 8 | valid << 16 | flags << 24`).  `read_meta` is one
arm for every module (two `LDS.128`: `pages[6], w6, w7`); `TileRegs::row_addr`
/ `page_at` are gone.  The pending word is F25c's `(w7 & 0x03FFFFFF) | stage
<< 30` - masks at bits 0-5 / 8-13.

**Copies (C17).**  `copy_dynamic_page<FULL>(b, page, j, p8, p4, stage, valid,
t)` per unrolled page `j` (`p8 = (m8 >> j) & 1`, `p4 = (m4 >> j) & 1`, `pa =
!(p8 || p4)`, `leader = blk % 8 == 0`): six sources (`a16_src0/1 + page *
a16_ps`, `page_src(p8 / s8 / p4 / s4, page, stride)`: six `IMAD.WIDE.U32`), four
destinations (`a16_dst`, `land8`, `land4`, `sc_rd` + `stage * bytes` + `j *
PAGE_REGION_BYTES` / `j * SCALE_PAGE_BYTES`), six predicated copies through the
new `mixed_detail::cp16_pred / cp8_pred(smem, gmem, pred, src_size)` (PTX
`setp.ne` + `@p cp.async ... cp-size, src-size`: a predicated-off lane issues
nothing): `@pa` two 16 B A16 rows, `@p8` 16 B block, `@(p8 && leader)` 8 B
scale row, `@p4` 8 B block, `@(p4 && leader)` 8 B scale row.  FULL: src-sizes
are the immediates 16 / 8; partial (the two per-item calls): `n = v ? size : 0`
per copied row (`tok0 + a_r`, `tok0 + a_r + 8`, `tok0 + r` against `valid`),
sources unmodified.  `issue_tile_copies`'s dynamic arm is `for j in 0..5:
copy_dynamic_page<FULL>(b, m.page(j), j, ...)` - no loop over set bits, no
`__ffs`, no select on an address.

**Decode.**  `expand_operand<VOTE>`'s dynamic arm: `__syncwarp()` first (every
lane past its wait: the six scale slots are readable), `sw[j]` for the six
pages at immediate offsets, per page `f = scale_byte_f32(sw[j])`, `ok8 &&=
!in8 || fold_ok(f * gs8)`, `ok4 &&= !in4 || fold_ok(f * gs4)`, `hot8 =
__all_sync(ok8)`, `hot4 = __all_sync(ok4)`, then `expand_format_pages<true,
!hot8>(m8)` and `expand_format_pages<false, !hot4>(m4)` (each a uniform branch
between two instantiations outside the loop; f16: the hot instantiations, no
vote).  `expand_format_pages<FP8, EXACT>(prm, isK, e, b, m, t)`: `if (m == 0)
return`; `i0, i1` = the first two set bits (`i1 = i0` if only one), `c0, c1` =
their landings; loop: `n0, n1` = the next two set bits (`n0 = i0` if none, `n1
= n0` if one), `sw0, sw1` (`LDS.32` at `i * 512`), `x0, x1` = the next pages'
landings (issued before this step's stores), `sf0, sf1` (hot: `a16x2(f32(s) *
gs)`; exact: `a16x2(f32(s) * g)` with `g` loaded once before the loop),
`__syncwarp()`, `expand_block<FP8, EXACT>` for `i0` and for `i1`, `if (!more)
break`, rotate.  An odd page count decodes its last page twice (idempotent: same
values to the same chunks) instead of branching; when no page follows, the
next-step loads re-read the current pages (before their stores).  The interim
`expand_block_voted` of F25b is gone; `VOTE` is ignored by the dynamic arm (the
votes are per format per operand at every site).

**Tests.**  `_run_dynamic_uniform(mode, q_len)`: a pure fp8 / fp4 transport run
through the dynamic module (`static_format = None`): 6 pages of one format per
tile, the other mask 0, kv_len 285 (partial last page); 4 cases.  Matrix: 100 +
4 = 104.  `a16_fp8_runs` (all-A16 tiles next to all-FP8 ones) and `a16_fp4`
(0 FP8 pages with FP4 present) were already in the 64-case matrix.

**Expected artifacts (6.1 dyn row).**  Per copy site 36 `LDGSTS` (six per
page: 3 x `.128`, 3 x `.64`) each under a predicate, 36 `IMAD.WIDE.U32`, no
`FLO` / `POPC` / `BRX` in the copy path, no `SEL` on an address register,
`LDL / STL` 0 in the pair loop (the F24 32 B frame came from the rolled copy
loops' hoisted bases; the unrolled body has none); per finish site 4
`VOTE.ALL` (two formats x two operands) in the loop site, 2 in each
single-operand site (the exact-only `VOTE = false` static path does not apply
to the dynamic module: its votes stay), the format loops with one back-edge
`BRA` each and their `WARPSYNC` (if emitted) before the two `STS.128` pairs.
Count per pair per warp (bench mix): ~790 (3.5).

### Open items for F25e (nothing below was built or run in this worktree)

1. Gate 6.0 (the a16 ncu probe at the `LDGSTS` PCs) decides between the 12-warp
   centre and 2E / 2G before any F25 timing; the rev 2 prediction for fp8 / fp4
   stands only if it passes.
2. Byte-identity of the a16 module and the stock paged kernel against
   `5cc416fd` is asserted from the text (the a16 `load()` folds to the [23]
   text through the `pair_step` / `produce_pair` lambdas and the
   `STATIC_A16` arm of `issue_operand`); ptxas's scheduling of the same code
   through a different inlining order is the one thing reading cannot settle -
   the SASS diff is the first 6.1 row.
3. `ptxas -v`: no C7507 and `STACK 0` for fp8 / fp4 / dynamic at 136 / 184; the
   dynamic module's F24 32 B frame is expected to vanish with the unrolled
   copy body (no hoisted rolled-loop bases), to be confirmed.
4. The predicated `cp.async` helpers (`cp16_pred`, `cp8_pred`) must appear as
   `@P LDGSTS` with no branch in the dynamic copy body (6.1 dyn row); if ptxas
   turns the `setp` + predicate into a branch, the alternative is the mask bit
   as a `src-size` register with the destination kept per lane (never another
   lane's slot) - not the withdrawn src-size-0-to-the-leader's-slot form.
5. Tests: 104 cases (`run_fa3_mixed_page_transport.py`; exit code = failures):
   64 matrix + 2 many-items + 6 parity-tail + 4 extremes-tail + 18 extremes + 6
   NaN-tail + 4 dynamic-uniform.

### F25e results (2026-09-04, nkcut2 H200, wt/F25 @ dd583e36; full tables in docs/mixed_kv_page_transport_backends.md, "Track F [25]")

Run order as in section 6 and the open items above: 104 / 104 bit-exact
(three builds), SASS + `ptxas -v` gates, gate 6.0, bench, trace, ncu.  Builds
2 and 3 changed only source forms after reading build 1's SASS (commit
dd583e36: `page_src` / `compressed_base` as PTX `mad.wide.u32`, vote chains
bitwise): the C++ 64-bit address form had compiled to `IMAD.WIDE.U32` + a
high-word `VIADD` of a materialised zero + `LDC` + `MOV`s (~7 per copy), and
the `&&` chains to `BSSY` / `@P BRA` / `BSYNC`.

| gate (rev 2, section 6) | measured | verdict |
|---|---|---|
| 6.5 tests | 104 / 104, every build | met |
| `USETMAXREG` 0x88 / 0xB8; ptxas no C7507; STACK 0 | as designed for fp8, fp4, dyn (a16 0x48 / 0xD8); 0 B frame, 0 spills, 168 launch regs, all four modules | met |
| region 2100-2400 | fp8 **2584**, fp4 2664, dyn 5528 | miss (+184 / +264): copies 3 instr each (below) + protocol |
| body (C12): `VOTE.ALL` per operand, no `BRA.DIV`, no `UMOV` / `IMAD.MOV` in bodies, hot 258 / cold 258 | hot 210 / cold 267 / vote prep 52 (fp8); `VOTE.ALL` 2 (loop), 0 in bodies; `BRA.DIV` 0, `WARPSYNC` 0 (a `NOP` at the `__syncwarp` point), `UMOV` 0 in bodies | met |
| C12 addendum (landing `LDS.64` x 12 -> `__syncwarp` -> scale `LDS.32` x 6) | read at all four operand sites | met |
| copy path: `IMAD.WIDE.U32` 24, `IADD3.X` <= 4, `VIADD` <= 4, `LDGSTS` 24 (12 predicated), `BSSY` 0 | 24 / 3 / 7 / 24 (12) / **1** (the [23] gather); **plus `IADD3` 29 + `IMAD.X` 21**: ptxas lowers `mad.wide.u32 d, page, stride, base64` as `IMAD.WIDE.U32 d, page, UR, RZ` + `IADD3` + `IMAD.X` - the per-item base never sits in an aligned pair (two source forms tried); peel folded (loop `LDGSTS` without src-size predicates) | `IMAD.WIDE` met; adds miss (+48 per pair) |
| protocol <= 70 | ~143 (loop site 1294 - 2 x 529 bodies - 48 copies - 45 split adds) | miss |
| dyn: 36 predicated `LDGSTS` per operand, `SEL` 0 on addresses, no `FLO` in copies, `LDL/STL` 0, 4 `VOTE.ALL` per finish site | 36 (71 / 72 predicated per pair) / 0 / 0 / 0 / 4; `BRA.DIV` 1 + `WARPSYNC` 1 per pair in `chunk_store`'s `REDUX.OR` ([24c] table build) | met except the table-build `BRA.DIV` |
| a16 module, stock kernel byte-identical to 5cc416fd | stock: identical (2 kernels x 4 objects); **a16: 56 / 3832 instructions differ, all a permutation of four uniform registers** (same opcodes, same order); PTX alpha-equivalent up to one swapped `selp` / `xor` pair (pipeline-state advance) at 8 sites - nvcc's schedule under the peeled loop; the shared-file diff vs 5cc416fd is `kernel_traits.cuh` + the `constexpr` hook in `prefill_sm90.cuh` only (md5 of the other five files equal) | stock met; a16 miss on the letter, identical instruction stream; `transport_a16` bench unchanged (282.8 / 289.5) |
| 6.0 (a16 `LDGSTS` PCs: mio + lg throttle <= 5 %) | 0.0 % (dispatch 42, wait 23, not_selected 23) | met: keep 12 warps |
| 6.2 trace fp8 (q=1): `iss` <= 0.30, `fin` <= 1.1, `acq` >= 0.25, `wait` <= 0.05 | **0.38** / 0.96-1.01 / 0.57 / 0.02 (F24: 0.66 / 1.0-1.16 / 0.10 / 0.03); fp4 `fin` 1.14-1.20 (expK 0.68, expV 0.38); mixed `iss` 1.63-1.78 unchanged from F24c | `fin`, `acq`, `wait` met; `iss` miss; the `acq` is a kernel-start reading (ncu below) |
| 6.3 ncu fp8: producer `inst_executed` per pair 2500-2750 | **2865** (716 / warp; F24 4432; [23] 3417); fp4 2959; **dyn 4842** (1211 / warp vs ~790 modelled) | fp8 +4 % over the band; dyn count model wrong |
| producer `selected` >= 25 % | fp8 22.7 (F24 15.2), fp4 22.3, dyn 23.9 | miss by 2.3 points |
| stall mix: branch_resolving <= 1, no_inst <= 2, dispatch <= 12, mio+lg <= 5, short_sb <= 10 | fp8: < 1 / 2.7 / **17.1** / 0 / 8.8; dyn: `no_inst` **10.2** (+ 8.8 not-issued) | dispatch miss; dyn instruction-cache miss |
| consumer K-wait PC <= 3 % | fp8 **10.2 %** (F24 16.5), fp4 18.0 %, dyn 37.3 %, a16 2.5 % | miss: the producer paces |
| consumer `inst_executed` == 3301 per pair | 3301 / 3301 / 3302 (a16 3316) | met (184 registers cost the consumer nothing) |
| LSU shared wavefronts per pair <= 2050; op_st <= 450; `STS.128` <= 4.5 / instr | fp8 1871 / **512** / **5.41** (fp4 1961 / 408 / 4.31; dyn 1945 / 263 / 4.16) | total met; fp8 st miss (same store pattern as fp4: not the addresses) |
| tensor pipe within 3 % of a16 (67.1 %) | 47.2 / 44.8 / 29.2 % | miss |
| 6.4 bench (us, q=1 / q=64 medians): stock 297-303 / 306-312; a16 281-290 / 284-292; fp8, fp4 <= 330; mixed centre 340 | stock 300.9 / 310.9; a16 282.8 / 289.5; **fp8 402.8 / 414.9; fp4 422.2 / 429.7; mixed 650.0 / 664.4** (F24 460 / 476, 496 / 512, 718 / 728) | controls hold; **fp8 / fp4 / mixed reject** (-12.5 / -14.8 / -9.5 % vs F24) |

Reading of the miss (section 9's model was count x IPC): the count is 716 per
warp against 655 (+9 %: +48 split adds, ~+70 protocol, vote prep 52 vs 14) and
the IPC 0.23 against 0.27 (-15 %): 716 / 0.23 = 3113 cycles = 1.57 us per pair
(model 655 / 0.27 = 1.22 us), against a consumer tile of ~1.35 us -> the
producer paces (K-wait 10 %).  The IPC shortfall sits at two chains the pc
samples name: the scale-load -> `PRMT` -> `F2FP` -> `HADD2` -> `FMUL` ->
`FSETP` -> `VOTE` chain (the top producer PC, 5.4 % of samples, `short_sb`
behind the first scale `LDS.32`: the C12-addendum order puts six scale loads
and their latency between the landing loads and the vote, once per operand)
and the copy address chain (`IMAD.WIDE` / `IADD3` / `IMAD.X`, `dispatch` /
`wait`, ~6 %).  The dynamic module additionally misses the instruction cache
(`no_inst` 10 %: eight rolled loop variants over three sites = 5528
instructions) and its `iss` did not move (1.6-1.8 us).  Follow-ups (backends
doc, numbered 1-5): scale loads under the landing loads; a 32-bit page offset
when the host bounds `page x stride`; one loop body per format with the
hot / exact choice predicated (or the 6.4 sorted-page table) for the dynamic
module; read the fp8 `STS.128` PCs; the a16 letter-identity via the [23] loop
text under `STATIC_A16` if required.
