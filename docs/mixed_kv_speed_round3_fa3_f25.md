# Mixed KV speed, round 3, Track F: lever [25] — what reaches <= 330 us with smem-materialised BF16 operands

Design only (no kernel edits in this worktree).  Base: `claude/mixed-kv-sm90-tma`
@ `64a70b9c` (merge of wt/F24).  Line numbers are those of `64a70b9c`.  This
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
per operand per pair, (3) the copy issue collapsed to one `IMAD.WIDE` + one
`LDGSTS` per copy from per-item 64-bit bases with the `FULL` arm hoisted per
work item, (4) the per-warp protocol cut from ~200 to <= 60 warp-instructions
per pair, (5) E2M1 by bit placement with a 2^126 fold under the same
per-operand vote (fp4 then costs what fp8 costs), (6) the dynamic module's copy
path as one format-as-data body over 6 pages (no rolled loops, no per-page
branch).**  F24b's second producer warp group is reverted (its shared-file
edits go back to the `5cc416fd` text; the register-split hook stays).  A
role-split 16-warp layout (transport WG + decode WG) is designed as the only
admissible 16-warp fallback and is built only if the F25 trace shows the copy
phase is SM-serialised (section 2E, gate 6.2).

| mode (us, q=1 / q=64) | today (F24 @ 64a70b9c) | F25 predicted (centre) | F25 band | accept |
|---|---|---|---|---|
| stock_a16 | 299.8 / 310.5 | unchanged | 297-303 / 306-312 | control |
| transport_a16 | 282.5 / 288.4 | unchanged (a16 module byte-identical) | 281-290 / 284-292 | control |
| fp8 static | 460.2 / 475.6 | **300 / 305** | 288-318 / 292-322 | <= 330 |
| fp4 static | 495.7 / 512.4 | **305 / 310** | 288-325 / 292-330 (hot fold path); +3 % where the fold vote fails | <= 330 |
| mixed (dynamic) | 718.1 / 728.2 | **315 / 320** | 295-338 / 300-342 | <= 330 (marginal) |

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
LDGSTS dispatch is SM-serialised (a16, [23] and F24 all show 0.62-0.85 us for
96 LDGSTS per SM per pair in three different layouts) — the fallback (2E)
overlaps it with the expansion; (3) the mixed module's decode keeps rolled
format-outer loops (data-dependent page counts) and lands inside the band, not
below it; (4) fp4's fold vote fails on operands whose block scales exceed
|s g| >= 3.99, adding 8 `HMUL2` per block (+8 %) — a quantizer-side knob (g).

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
| `__syncwarp`, 2 per pair | 2 | one per operand, between the operand's loads and its first store (3.2) |
| copies, 12 pages | 48 | per page: `IMAD.WIDE.U32` (payload), `LDGSTS`, `IMAD.WIDE.U32` (scales), `@leader LDGSTS` |
| meta read | 6 | 2 x 2 `LDS.128` + 2 (`valid` used only by the partial arm) |
| protocol | <= 60 | 2 acquires (~10), stage/phase updates (6), `LDGDEPBAR` + `DEPBAR` + `FENCE.VIEW.ASYNC` (3), 2 commits (4), pending rotate (2), loop and chunk-index arithmetic (~8), chunk gather amortised over 16 pairs (~3), `FULL` hoisted (0 per pair) |
| **total** | **~645** | [23]: 854; F24: 554 x 2 warps per SMSP = 1108 per SMSP |

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

### 3.1 Data flow (per pair, fp8 static module; fp4 and dynamic in 3.5)

```
acquire K stage, acquire V stage                         (unchanged, PipelineAsync)
read meta rows of K(t-1), V(t): 2 x 2 LDS.128 -> pages[6], w7   (unchanged, :703-713)
issue copies K then V: per page j (immediate offsets):
   src   = IMAD.WIDE.U32(pages[j], UR page_stride, base64_op)   base64 per item (3.3)
   LDGSTS.128 [land8 + stage*STAGE + j*PAGE_REGION], [src]     payload block (16 B)
   ssrc  = IMAD.WIDE.U32(pages[j], UR scale_stride, sbase64_op)
   @leader LDGSTS.128 [sc_rd + ...], [ssrc], 16, 8               row's 8 B of scales, zero-filled to 16 B (3.5)
LDGDEPBAR (commit group of this pair)
DEPBAR.LE 1 (pair t-1's group landed)                     (unchanged: wait one pair later)
expand K(t-2)... i.e. the pending operands, each as ONE straight-line body:
   6 x LDS.32 scale words; 6 x 2 LDS.64 packed halves      (all loads first)
   __syncwarp                                              (orders every lane's loads before any lane's stores)
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
scale slot after the issuing lane's wait and the operand's `__syncwarp`).
Shared-memory layout: unchanged (~199.7 KB; the scale slot per row grows from
8 B to 16 B inside the 512 B page slot: 16 x 16 = 256 B, `kernel_traits.cuh:89`
unchanged).  Wavefronts per pair: unchanged from F24's measured 1974
(`op_st` 418 at 4.41 per `STS.128`, `op_ld` 455, `LDGSTS` ~101; the 16 B scale
copy with src-size 8 is 4 wavefronts per warp instruction like the payload's).

### 3.2 Control flow, registers, and why the block bodies are branch-free

**Vote hoisting (C12).**  The fold test needs all six scale words of the
operand, which `expand_operand` already loads up front (`:1250-1254`); the
six products v_j and the six `FSETP` are computed before any decode, ANDed
(`PLOP3`), voted once (`VOTE.ALL`) and branched once per operand.  Both arms
are the full six-block body; nothing but addresses is live across the join, so
ptxas has no registers to unify with `IMAD.MOV` (the 3-4 per block in F24).
The `__syncwarp` moves to *before* the vote, in straight-line code after the
operand's loads: ptxas proves convergence and emits at most one `WARPSYNC`
(the [23] SASS had none) — no `UMOV UR, -1; BRA.DIV` guards (22 in F24).  The
cold arm is the out-of-line copy (code size: two six-block bodies per operand
per inlined site; the fp8 producer region stays under ~1500 instructions, from
2093).  Numerics unchanged (C9): the vote covers the same predicate per block,
only its granularity is per operand-per-warp instead of per block-per-warp;
an operand with one over-range block takes the exact path for all 24 blocks of
that warp (the extremes tests exercise this: `run_fa3_mixed_page_transport.py`
scales 448 / 256 with g = 1).

**Registers (C3 restated).**  `__launch_bounds__(384, 1)` -> 168 at launch;
`setmaxnreg.dec 136` (producer), `setmaxnreg.inc 184` (consumers): 128 x 136 +
256 x 184 = 64,512 <= 65,536.  The consumer at 184 is proven (F24b: `ptxas -v`
no C7507, `STACK 0`, same consumer code).  Producer live set for a whole
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
pipe, ~50 cycles, then 24 `LDGSTS`).  `FULL` is hoisted per work item: only
the item's first pair (K(last), tile `kv_tile_idx`) can be partial; the loop
body compiles the `FULL = true` arm only (`issue_operand`, `:1346-1362`, loses
its per-pair `valid == CTA_KV` branch and the `BSSY/BSYNC` pair); `valid` is
read from `w7` only inside the K(last) call.  The scale copy stays one
predicated `LDGSTS` per page (lane `b == 0`), now `cp.async.cg ... 16, 8`
(16 B cp-size, 8 B src-size, upper 8 B zero-filled) into a 16 B row slot so
that the static and dynamic modules share one copy form (3.5).

**Protocol (C14).**  `finish_pending_pair` (`:1459-1491`) loses `if (K.pending
!= 0 || V.pending != 0)` in the steady-state loop (both operands are always
pending in the compressed static modules; the K(last)-alone and V(0)-alone
calls keep `if (op.pending == 0) return` inside `expand_pending`); the pending
record shrinks to a 32-bit word (stage index in bits 30-31, `w7` low bits);
`PipelineState` increments stay.  Budget: <= 60 warp-instructions per pair per
warp, checked as the SASS count of the producer region minus 12 x 43 - 14 - 54
- 6 (6.1).  Acquire: the two `try_wait` round trips are issued back-to-back
before either is tested (both `SYNCS.PHASECHK` then both `BRA`), overlapping
the two ~100-cycle round trips (0.10 -> ~0.05 us); optional.

**Barrier protocol (C4).**  Back to [23]: producer arrival count 128,
`kQueryEmpty` 384, `kProducerWG` 128, Q TMA by warp 0 of WG0, ping-pong ids 2
and 3 with `kFirstConsumerWG = 1`, chunk gather by all 128 producer threads.
The F24b edits in `prefill_sm90.cuh:66-67, :95-96, :103`, `epilogue.cuh:77-79`,
`named_barrier.cuh:30-40, :46-63, :81-86`, `sparse_mainloop.cuh` (the relaxed
`static_assert`) fold to the `5cc416fd` text for `NUM_PRODUCER_WGS = 1`; the
traits set `NUM_PRODUCER_WGS = 1` for every module (`kernel_traits.cuh:225`)
and the generic hooks are kept textually so that stock traits are untouched
(acceptance: stock paged kernel and a16 module byte-identical to `5cc416fd`, as
F24 verified).  `barrier_O` / C7 unchanged.

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

fp4 landing (D2 amended): block b's 8 B lands in the row's D-block-1 line at
chunk `b ^ (r & 7)`, low 8 B, upper 8 B zero-filled by `cp.async.cg ... 16, 8`
(src-size 8 into a 16 B chunk).  Same chunk formula as fp8, one `LDGSTS.128`
form for both compressed formats, 4 wavefronts per warp instruction (today's
8 B `cp.async` measured 3.98); the two 4 B halves are read at `land + 4 swap`
and `^ 4` as today (`expand_bases`, `:1120-1129`).  The chunk is another lane's
D-block-1 output chunk, written after the operand's `__syncwarp` — the same
discipline as fp8's landing (A7).

### 3.5 Mixed (dynamic module): one copy body, format as data; format-outer decode with two pages per step

Copies.  Per page, per thread, exactly two `LDGSTS.128` with a src-size
register, no per-page branch, no rolled loop, six pages unrolled with
immediates: copy A = {A16: 16 B of row u/16 at `a16_dst`; FP8: 16 B block at
`land8`; FP4: 8 B block at `land4` (src-size 8)}; copy B = {A16: 16 B of row
u/16 + 8 at `a16_dst + ATOM_BYTES`; FP8 / FP4: the row's 8 B scales into the
16 B slot by lane b == 0 (src-size 8), src-size 0 on the other lanes}.  The
tag byte selects (2-3 `SEL` each) the 64-bit base and the 32-bit stride for
`IMAD.WIDE.U32`, the destination base, and the src-size: ~14 instructions per
page (`PRMT` tag extract 1, `SEL` ~7, `IMAD.WIDE` 2, `LDGSTS` 2, misc 2) ->
**~170 per pair per thread**, all twelve pages independent.  The zero-fill
src-size register form is already what the partial-tile arm uses
(`cp16_zfill`, `:253-257`); the A16 arm stays `cp16` in the static a16 module
(byte-identical).  Tags come from the meta row's tag bytes (`w6`, `w7`; the
F24c masks `tags[4], tags[5]` are dropped, `kernel_traits.cuh:54-61` comment
reverts; `chunk_store` `:626-640` back to the byte stores).  This replaces
F24c's format-outer copy loops (`:992-1045`), whose per-pair `iss` traced at
1.7-2.1 us on 8 warps (would be ~2x that on 4).

Decode.  Format-outer over the pending word's page masks as F24c
(`expand_format_pages`, `:1303-1324`), but two pages per step with the next
two pages' loads issued before the current stores (the static path's shape),
the fold vote once per format per operand (over the format's pages, scale
words loaded first), and 136 registers.  Bench mix (page p tagged p % 3 -> 2
A16 + 2 FP8 + 2 FP4 per tile): per thread per pair 4 FP8 + 4 FP4 blocks = 4 x
43 + 4 x 44 = 348 decode + 4 steps x ~25 loop/branch/exposed-chain ~= 100 ->
**~450**, plus copies 170, meta/protocol ~70 (32-bit pending words with
masks) -> **~690 per warp per pair**, i.e. fp8's count +7 %: at IPC 0.27 ->
1.29 + 0.08 = 1.37 us -> ~315 us; band 295-338.  The tile-uniform fast path of
F24c is not needed (the bench has none) and not built.

## 4. Per-pair arithmetic, all modes (12 warps, per producer warp = per SMSP)

| mode | decode | votes/sync | copies | meta + protocol | total | at 0.27 IPC (+0.08 acq) | at 0.218 | at 0.35 |
|---|---|---|---|---|---|---|---|---|
| fp8 | 12 x 43 = 516 | 16 | 48 | 66 | **646** | 1.29 us -> 290-300 | 1.58 -> 345 | 1.01 -> 285 |
| fp4 (hot) | 12 x 44 = 528 | 16 | 48 | 66 | **658** | 1.31 -> 292-305 | 1.60 -> 350 | 1.03 -> 285 |
| fp4 (cold everywhere) | 12 x 52 | 16 | 48 | 66 | 754 | 1.49 -> 330 | — | 1.17 -> 290 |
| mixed (bench mix) | 348 + ~100 loop | 32 | 170 | 70 | **~690** | 1.37 -> 315 | 1.68 -> 365 | 1.08 -> 288 |
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
  pages of each operand (A7 verbatim); the row's 16 B scale slot is copied by
  lane `b == 0` (`cp.async.cg` 16 B with src-size 8) and read by the row's
  eight lanes after every lane's own `cp.async.wait_group` and the operand's
  single `__syncwarp` (before the vote, before any store).  fp4 landing: block
  b at chunk `b ^ (r & 7)` of the row's D-block-1 line, low 8 B.
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
  `IADD3.X` / `VIADD` count in the pair loop <= 4; the `FULL` predicate is
  evaluated once per work item (only the K(last) call carries the partial arm).
- **New C14 (protocol budget).** Producer-region SASS minus the decode, vote,
  copy and meta counts <= 60 per pair per warp; pending records are 32-bit.
- **New C16 (E2M1 placement fold).** bf16 E2M1 decode by placement (x 2^-126)
  folds 2^126 into the block scale; exact iff `2^-126 <= |s g| < 3.9921875`,
  tested per operand per warp (vote) with the same +inf sentinel for |g| <
  2^-117; otherwise 8 `HMUL2` by 2^126 then the two-multiply exact form.  Tests:
  fp4 extremes with block scales 448 / 256 / 4 / 3.5 at g = 1 (cold and hot
  in the same operand of different warps), g = 0.5, g = 1.1 x 2^-118.
- **New C17 (dynamic copy body).** The dynamic module issues exactly two
  `LDGSTS.128` per page per thread with tag-selected base / stride / size /
  destination; no loop, no per-page branch, no `LDS` on the address chain.
  C10's masks and rolled copy loops are withdrawn; C10's decode loops stay
  (two pages per step).
- **C11.** Unchanged (<= ~2000 wavefronts per pair; the ncu confirmation
  reports the four classes).

## 6. Verification artifacts (confirmation, not tuning; each with its accept / reject)

**6.1 SASS (`cuobjdump -sass`, `*_paged_sm90_kernel_mask_1`, producer region
`USETMAXREG.DEALLOC .. EXIT`), before any timing:**

| item | accept | reject -> action |
|---|---|---|
| `USETMAXREG` | exactly two: `DEALLOC 0x88`, `TRY_ALLOC 0xB8`; `ptxas -v` no C7507; `STACK 0` (fp8, fp4, dyn) | C7507 or STACK > 0 -> try 128 / 192 (128 x 128 + 256 x 192 = 65,536); if the producer needs > 128, four pages per step instead of six |
| fp8 region count | <= 1500 (from 2093); per operand body: 12 `LDS.64`, 6 `LDS.32`, 6 `F2FP.E4M3`, 6 `FSETP`, 5 `PLOP3`, 1 `VOTE.ALL`, 1 `BRA`, then 6 x {`F2FP.PACK`, 8 `PRMT`, 8 `IMAD.SHL`/`SHF`, 8 `LOP3`, 8 `HMUL2`, 2 `STS.128 [R+imm]`} | any `VOTE` inside a body, `BRA.DIV` > 0, `UMOV` in a body, `IMAD.MOV` > 2 per body -> C12 violated; restructure before timing |
| copy path | `IMAD.WIDE.U32` 24 per pair, `IADD3.X` <= 4, `VIADD` <= 4 in the loop; `LDGSTS` 24 per pair per thread (12 + 12 predicated); `BSSY/BSYNC` <= 2 in the loop | more -> the per-item bases were not hoisted (check `make_bases` live set) |
| protocol | region count - 12 x 43 - 16 - 54 <= 60 x (inlined sites) | over -> list the extra opcodes before timing |
| fp4 | body as fp8 with `SHF`/`IMAD.SHL` 2 + 4 `PRMT` + 4 `SHF` + 4 `LOP3` per word pair, `HMUL2` 8 hot / 16 cold; `LDGSTS.128` (not `.64`) | |
| dyn | 12 `LDGSTS.128` sites per operand (two per page), `SEL` ~80 per pair, no `FLO`/`POPC` in the copy path, `LDL/STL` 0 in the pair loop | rolled loop or LDL present -> C17 / C2 violated |
| a16 module, stock paged kernel | byte-identical to `5cc416fd` | any diff = the shared-file revert is incomplete |

**6.2 Trace (`MIXED_FA3_TRACE`, fp8 q=1, CTA 0 items 0/1).**  The traced pairs
are CTA 0's first two items (kernel start, all 132 SMs streaming: the a16 trace
pair is 2.02 us against the 1.28 us bench average) and carry the stamp overhead
the F24 record books as `gap` (0.54-0.67 us), so segments are compared with
each other and with the F24 trace, not with T_c:

| segment | accept | reject -> action |
|---|---|---|
| `iss` | <= 0.30 us (from 0.62-0.85) | > 0.5 with the C13 counts met: LDGSTS dispatch is SM-serialised -> build 2E (role split), whose WG0 overlaps it with WG1's decode |
| `fin` (`expK + expV`) | <= 0.60 us (from 1.0-1.2) | > 0.8 with C12 met: the body IPC did not follow the structure -> pc-sample the body; then four pages per step / two operands interleaved |
| `acq` | >= 0.25 us (the producer waits on the consumers' release = consumer-bound) | ~0.1 with `iss + fin` <= 1.0: trace overhead masks it; decide on the bench row |
| `wait` | <= 0.05 us | > 0.1: landing latency exposed (3.3) -> scale copies first, then re-measure |

**6.3 ncu (fp8 q=1, `--repeats 1`, third launch, `f23_run_ncu.sh` metric set +
`f25_ncu_classes.py`):**

| metric | accept | reject |
|---|---|---|
| producer `inst_executed` per pair | 2500-2700 (4 x 645 +- 5 %) | > 2900: count model wrong, re-read the SASS |
| producer per-warp `selected` share | >= 25 % | < 22 % with the count met: IPC did not move; pc-sample by opcode, decide 2E vs four-pages-per-step |
| producer stall mix | `branch_resolving` <= 1 %, `no_inst` <= 2 %, `dispatch` <= 12 % | |
| consumer K-wait PC (`@P0 BRA` at `consumer_wait`) | <= 3 % of consumer samples (from 16.5 %) — the consumer-bound proof | 8-16 %: producer still paces |
| tensor pipe active | within 3 % of transport_a16's | |
| smem wavefronts per pair, by class | total <= 2050; `op_st` <= 450, `STS.128` <= 4.5 per instruction; `LDGSTS.128` 4.0 | |
| `smsp__issue_active` | 55-65 % | |

**6.4 Bench (`benchmarks/bench_fa3_mixed_page_transport.py --q-lens 1 64
--repeats 1 --trials 5`, nkcut2 lock, co-tenant rule: bursts < 1.5 ms; min /
median / max):**

| row | accept | reject / re-derive |
|---|---|---|
| stock_a16 | 297-303 / 306-312 | drift > 3 % -> session offset, rerun |
| transport_a16 | 281-290 / 284-292 | any change on a byte-identical module = machine |
| fp8 | **<= 330 / <= 330** (band 288-318 / 292-322) | 331-345 with 6.3's `selected` < 22 %: IPC; with `selected` >= 25 % and `iss` > 0.5: 2E; > 345: count model wrong |
| fp4 | **<= 330 / <= 330** (band 288-325 / 292-330) | as fp8; additionally check the vote path taken (cold-path `HMUL2` executed count via ncu source view = 0 on the bench payload) |
| mixed | **<= 330 / <= 330** (band 295-338 / 300-342) | 331-345: the decode loops' step overhead -> (only then) a sorted-page table with unrolled bodies per (n8, n4) class |

**6.5 Correctness (`tests/attention/run_fa3_mixed_page_transport.py`; pytest is
banned):** the 88 cases of F24 bit-exact (parity-tail cases kept: they now
exercise the single-WG partial page) + fp4 extremes per C16 (hot / cold mixed
across warps, g in {1, 0.5, 1.1 x 2^-118}) + a dynamic case whose tile has 0
FP8 pages and one with 6 FP8 pages (mask edge cases of the two-page steps) +
the many-items case (C7).  Also the E4M3 / E2M1 NaN-code note of C9 stands.

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

## 8. Files touched (implementation order; each step builds and passes 6.1 before the next)

- **F25a — layout and registers.** `kernel_traits.cuh:225-230`
  (`NUM_PRODUCER_WGS = 1` for all formats; `PRODUCER_REGS = kMixedStaticFormat
  == 0 ? 72 : 136`; `CONSUMER_REGS = 216 : 184`; pool assert), `:54-61`
  (comment; masks withdrawn); `sparse_mixed_mainloop.cuh` constants `:316-345`
  (`PAGES_PER_THREAD` 6, `SCALE_ROW_BYTES` 16), `own_u / own_h` fold (`:747-752`),
  `TileRegs::page` (`:673-679`), `parity_mask` (`:699-701`), `load` gather /
  Q-issuer predicates (`:1512`, `:1594-1595`).  `prefill_sm90.cuh`,
  `epilogue.cuh`, `named_barrier.cuh`, `sparse_mainloop.cuh` unchanged in text
  (they fold).  Gate: a16 + stock byte-identical; fp8 `USETMAXREG 0x88 / 0xB8`.
- **F25b — bodies.** `expand_operand` (`:1237-1298`): loads-first, one
  `__syncwarp`, hoisted vote, two six-block arms; `expand_block` (`:1168-1232`)
  loses the vote and the `__syncwarp`; `e2m1x8_to_a16` (`:132-157`) bf16 arm
  becomes the placement; `make_bases` (`:838-858`) `gs4 = g * 2^126` with the
  sentinel; new `kFp4Fold*` constants beside `:202-208`; `copy_compressed_page`
  (`:936-963`) fp4 landing at chunk `b ^ (r & 7)` with `cp16_zfill(…, 8)`
  semantics (a `cp16_src8` helper beside `:253-257`), scale copy 16 B / src 8;
  `expand_bases` (`:1120-1129`) `l4a = (land4 + so) | 4 swap`.  Gate: 6.1 body
  rows; tests 88 + fp4 extremes.
- **F25c — copies and protocol.** `OperandBases` (`:765-776`) gains
  `pbase64, sbase64`; `compressed_src` (`:864-886`) removed; `issue_tile_copies`
  static arm (`:972-991`) one `IMAD.WIDE` per copy; `issue_operand`
  (`:1346-1362`) `FULL` from a per-item flag; `produce_pair` / `load`
  (`:1529-1650`) K(last) call with the partial arm, loop with the full arm;
  `finish_pending_pair` (`:1459-1491`) unconditional in the loop; 32-bit pending
  words (`TileRegs::pending_word`, `:690-692`, `Operand`, `:1336-1344`).  Gate:
  6.1 copy / protocol rows; trace 6.2.
- **F25d — dynamic module.** `chunk_store` (`:626-640`) back to tag bytes;
  `issue_tile_copies` dynamic arm (`:992-1045`) -> the format-as-data body;
  `expand_format_pages` (`:1303-1324`) two pages per step, per-format vote;
  `read_meta` dynamic arm (`:705-707`) reads the full row.  Gate: 6.1 dyn row,
  tests with 0 / 6 FP8 pages per tile.
- **F25e — measurements** (6.2-6.4), then this document's results section and
  the dataflow amendments of section 5.

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
