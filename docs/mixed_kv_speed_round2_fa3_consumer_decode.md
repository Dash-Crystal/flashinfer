# Mixed KV speed, round 2, Track F: lever [24] — how the compressed FA3 prefill reaches <= 330 us

Design only (no kernel edits in this worktree).  Base: `claude/mixed-kv-sm90-tma`
@ `5cc416fd` (merge of [23], commit `fa13ad89`).  Paths are relative to the
repository root; line numbers are those of `5cc416fd`.  Prior documents this
extends: `docs/mixed_kv_page_transport_dataflow.md` (D1-D6, C1-C7, A1-A7),
`docs/mixed_kv_speed_plan.md` Track F ([20]-[24], lines 214-241),
`docs/mixed_kv_speed_round1_synthesis.md` (Track F paragraph, line 73),
`docs/mixed_kv_page_transport_backends.md` (P0.3 consumer trace and RT
constants, lines 528-619; round-2 baseline / setmaxnreg rule, lines 1286-1357).

## 0. Summary

**Chosen: a sequence F24a -> F24b -> F24c = (D) decode floor, then (C) a second
producer warp group, then (E) the dynamic module's per-page dispatch.**  (B) is
rejected on the wgmma operand rule; (A) is kept as the fallback that is built
only if a consumer-side trace shows the slack it needs.

| mode (us, q=1 / q=64) | today ([23]) | after F24a (D) | after F24a+b (C) | after F24a+b+c (E) | accept |
|---|---|---|---|---|---|
| stock_a16 | 299.8 / 309.4 | unchanged | unchanged | unchanged | control |
| transport_a16 | 283.4 / 286.9 | unchanged (module untouched) | unchanged (a16 module keeps 12 warps) | unchanged | 283-290 |
| fp8 static | 474.0 / 483.0 | 415-440 | **290-320 / 295-325** | same | <= 330 |
| fp4 static | 507.2 / 517.5 | 470-495 | **295-325 / 300-330** | same | <= 330 |
| mixed (dynamic) | 880.1 / 906.9 | 850-880 | 340-370 | **300-345** | <= 330 (marginal; bench row decides) |

Main risks, in order: (1) the consumer must give up 24-40 registers
(216 -> 184/192) for the second producer warp group; if it spills at every
admissible split the pool arithmetic fails and the sequence stops at F24a;
(2) the per-producer-warp IPC of 0.20-0.22 is assumed to be per-warp dependent-
issue latency (stall mix wait 27 / short_sb 9 / long_sb 9), not a shared-pipe
bound — if a pipe is the bound, doubling the warps does not double the rate;
(3) the mixed module ends inside its acceptance band only after (E), and (E)'s
prediction rests on [40]'s sm90 result, not on an FA3 measurement.

## 1. Where the time goes (restating the [23] record as a per-pair budget)

Benchmark shape B=17, S=4096, 8 KV heads, GQA 4, D=128, bf16.  At q=1 every
(q-head, request) is a work item padded to a 128-row q tile: 17 x 32 = 544
items on 132 SMs = 4 full scheduler rounds + a 16-item fifth round (plan line
216: "tail (5th scheduler round, 16 items)"); each item is 43 KV tiles
(4096/96 -> 43), so the kernel is ~5 x 43 = 215 producer/consumer pairs deep.
Dividing the bench numbers by 215:

| mode | wall us | us per pair | cycles per pair @ 1.98 GHz |
|---|---|---|---|
| stock_a16 | 299.8 | 1.39 | 2760 |
| transport_a16 | 283.4 | 1.32 | 2610 |
| fp8 | 474.0 | 2.20 | 4370 |
| fp4 | 507.2 | 2.36 | 4670 |
| mixed | 880.1 | 4.09 | 8100 |

So the consumer's pair time is **T_c = 1.32-1.39 us (2600-2760 cycles)**; the
target 330 us is a pair of **<= 1.53 us (3040 cycles)**; the producer's pair
must be **<= ~1.4 us** so that the consumer, not the producer, sets the cadence
(dataflow section 5: `T = max(T_c, T_i + T_x)`).  The fp8 producer today is
`iss 0.85 + fin 1.2 + acq/gap ~0.15 = ~2.2 us` (commit `fa13ad89` trace; the
0.7 us "gap" is trace overhead and is absent from the plain build, which is
why the plain build's 2.20 us equals iss + fin + 0.15).

The instruction side is fixed by the [23] SASS and ncu record: **3417
warp-instructions per pair on 4 producer warps = 854 per warp**, issued at
**0.196-0.22 IPC per warp** (3417 / 4370 cycles = 0.78 aggregate).  Per FP8
block the producer executes exactly `LDS.128 + LDS.32 + 5 (scale) + 8 x {F2FP,
SHF, LOP3, IMAD} + 8 HMUL2 + 2 STS.128 = 49`
(`sparse_mixed_mainloop.cuh:154-169` decode, `:832-847` scale, `:849-863`
block, `:907-919` store), 12 blocks per thread per pair (6 pages x 2 operands,
`:924-956`); per FP4 block ~57 (`:117-149` LUT, `:865-878`).  The copies are
12 payload + 12 scale-word `LDGSTS` per thread per pair (`:753-780`), the
chunk-table reads four `LDS.128` per pair (`:597-602`), the protocol two
acquires, one `cp.async` fence, one `fence.proxy.async`, two commits
(`:1104-1129`, `:1161-1209`).  Not the bound: smem port 49 % (2570 wavefronts
per pair incl. the 2-way STS), XU 3 %, DRAM (fp8 moves half of A16's bytes).

The design question is therefore purely one of **issue rate**: 854 dependent
instructions per warp per pair at 0.2 IPC is 4300 cycles; parity needs either
the count per warp below ~520 (at 0.2 IPC -> 2600 cycles) or the IPC above
0.33 at today's count.  The alternatives below are scored on exactly that.

## 2. Alternatives

### 2A. Consumer-side decode (the two consumer warp groups expand K and V)

**Data flow.**  Producer: copies compressed blocks and scale words exactly as
[23] (`:753-780`, landing chunk `b ^ (r&7)` of the row's D-block 1 line, scale
slot `scales[stage][page][t]`), then commits the stage **at issue** with
`cp.async.mbarrier.arrive` — the A16 completion mode of `issue_operand`
becomes the only mode (`:994-1010`); `finish_pending_pair`, the pending
records, `cp.async.wait_group` and the producer's `fence.proxy.async` go away
(`:984-1046`, `:1104-1129`).  Consumer WG w (w = 0, 1) after `consumer_wait`
on the full barrier (`mainloop_mma.cuh:233`, `:238`) expands half the tile —
rows `48w .. 48w+47`, thread `u` of the WG owning block `u%8` of rows
`48w + u/8 + 16j`, j = 0..2 (three blocks per operand, the row's eight lanes
in one warp as A7 requires) — into the wgmma operand layout in place, issues
`fence.proxy.async`, and both WGs meet on a **256-thread named barrier**
before either issues the gemm on that tile.

**Control flow per pair per WG** (against `mainloop_mma.cuh:231-274`):
`consumer_wait(K) ; expand K half ; fence ; bar(K expanded, 256) ;
barrier_sync (ping-pong) ; QK gemm issue ; rescale_o ; consumer_wait(V) ;
expand V half ; fence ; bar(V expanded, 256) ; PV gemm issue ; barrier_arrive
; wait<1> ; release K ; softmax ; wait<0> ; release V`.  The V expansion can
be slid under the WG's own QK gemm (issued asynchronously two lines earlier),
the K expansion under the previous PV only if it is moved before `wait<1>` —
i.e. into the softmax slot, which is the slot that already makes the consumer
1.7x the tensor floor (P0.3-style accounting: 6.3 MFLOP per pair / 3787
FLOP per cycle per SM = 1660 tensor cycles against a 2700-cycle cadence).

**Budget.**  The 3417 warp-instructions per pair move to 8 warps: **427 per
consumer warp per pair**, on top of the consumer's own ~300 (0.11 IPC x 2700
cycles).  Registers: the expansion's live set (packed 4 + outputs 8 + scale 2 +
three bases) is ~20-30 on top of O (64) + S (48) + P (24) + softmax state; fits
216 (and would still fit at [20]'s 200).  Smem: none (in place).  Barriers:
+2 x 256-thread `bar.sync` per pair (21 cycles isolated, 120-200 under load
per P0.3(d), backends lines 604-619) plus the two `fence.proxy.async` per
thread per pair.

**Verdict: conditional, not chosen.**  Whether 427 extra instructions per warp
(600-850 cycles at the 0.5-0.7 IPC a 16-chain decode reaches on an otherwise
idle warp) are free depends on how much of the per-WG loop is spent *waiting*
(`barrier_sync`, `consumer_wait`, `warpgroup_wait`) rather than issuing; the
tensor pipe is idle 40 % of the cadence, so slack exists somewhere, but the
FA3 consumer has never been traced (P0.3 traced XQA).  If the slack is under
~900 cycles per pair per WG the cadence grows by the difference and the target
is missed (2700 + 400 = 3100 cycles = 1.57 us -> 337 us).  It also breaks C5
(`mainloop_mma.cuh` gains a third loop body; the a16 module must keep the
stock one) and moves the D3 ownership proof into the consumer.  **Build (A)
only if** a `%globaltimer` consumer trace of the fp8 module (six stamps per
pair per WG: before/after `barrier_sync`, before/after each `consumer_wait`,
after `wait<0>`) shows >= 900 cycles per pair per WG of wait that is not the
producer's `full` barrier — i.e. slack that (C) does not also consume.

Invariant changes if built: D3 -> "block (r, b) is expanded by consumer
thread (w, u) after the full barrier; it reads only bytes the producer's
`cp.async` wrote (async proxy, ordered by the mbarrier) and writes only its
own two output chunks"; new **C8**: "a compressed stage is consumed by wgmma
only after both consumer warp groups have passed the expanded-barrier for
that stage; `fence.proxy.async` precedes the arrive"; C4: producer
`cp.async.mbarrier.arrive` for every tile; C5 amended.

### 2B. Register-operand (RS) wgmma for the expanded K^T / V

`wgmma.mma_async` takes **A from registers or a smem descriptor and B from a
smem descriptor only**.  In this kernel K is the **B** operand of QK^T
(`kernel_traits.cuh:138-139`, `ss_op_selector`) and V is the **B** operand of
PV (`:140-143`, `rs_op_selector` with `GMMA::Major::MN` for B; P is already the
register A operand, `mainloop_mma.cuh:175-176`, `:194-195`).  So:

- **V can never be a register operand**; half the expansion stays in smem
  regardless.
- K in registers requires the transposed product `S^T = K Q^T` with K as A:
  `M` = tokens (64 per wgmma, so `CTA_KV` = 96 is 1.5 M-tiles -> 64 or 128),
  `N` = 128 q rows, softmax reductions along `M` = across the four warps of a
  WG (smem or shuffles) — the XQA SWAP_AB skeleton the dataflow doc already
  costed at ~2100 instructions per 64 tokens (section 6).  The A fragment of
  `m64k16` gives each thread 2 rows x {2, 2} columns per k-step, so one
  16-coefficient block is spread over four threads' fragments across two
  k-steps: the block-owner decode would need the scale word in four threads
  and a shuffle stage, or a per-thread decode of two values at a time with the
  scale reloaded — more instructions than today, in the consumer.

**Rejected.**  No data-flow or budget table is needed: it removes at most the
K half of the smem writes (2 of ~49 instructions per block) at the price of a
new consumer skeleton.

### 2C. A second producer warp group (8 producer warps; expansion split by page parity)

**Data flow.**  16 warps per CTA (`NUM_WARPS` 12 -> 16; `kernel_traits.cuh:125`
gains a `NUM_PRODUCER_WGS_` knob on `MixedAttentionKernelTraits`, `:187-205`).
Producer thread `t` in [0, 256): half `h = t >> 7`, within-half index
`u = t & 127`; **every [23] ownership formula applies to `u`, restricted to
pages `i ≡ h (mod 2)`**: block `u%8` of row `u/8` (`:634-635`), landing
chunks and scale slot as `make_bases` computes them from `u` (`:668-706`,
slot `sc_base + 4u`, so `SCALE_PAGE_BYTES` = 512 B is unchanged because only
128 threads own any one page); A16 pages (dynamic module) chunk `u%16` of
rows `u/16`, `u/16+8` (`:735-750`).  Each thread copies and expands **3 pages
per operand per tile instead of 6**.  D2, D3 (a thread reads only what its own
`cp.async` wrote; the landing chunk it reads is an output chunk of a lane of
its own warp -> `__syncwarp` at `:916` still suffices), D4 (src-size-0
zero-fill per thread), D5, D6 and A7's wavefront rules carry over verbatim
because the eight lanes of a row stay in one warp and a warp still copies one
128 B global line per page.  Smem: unchanged (~199.7 KB: Q 32 KB, K/V 3 x
24 KB each, scales 18 KB, chunk table 1 KB).

**Control flow.**  Unchanged per thread (`load`, `:1049-1303`), with these
count changes: `PipelineAsync` `producer_arv_count` 128 -> 256
(`prefill_sm90.cuh:96`, follows `NUM_COPY_THREADS`); `kQueryEmpty` count
`NUM_MMA_THREADS + NUM_PRODUCER_THREADS` -> 512 (`:1219-1221`,
`mainloop_mma.cuh:315-316`); `kProducerWG` group barrier 128 -> 256
(`:1086-1089`); the Q TMA must be issued by **warp 0 of WG 0 only** — today
`warp_idx_in_warpgroup == 0` (`:1222`) is true in both producer WGs and would
issue the box twice against a transaction count of one; the chunk-table
gather keeps 16-tile chunks with threads `t < 128` (`static_assert` `:312`
becomes `CHUNK_TILES * 8 == NUM_COPY_THREADS / 2`), or moves to 32-tile
chunks (+1 KB, half the group barriers) — the former is the smaller change.
Role selection `warp_group_idx == 0` -> `< NUM_PRODUCER_WGS`
(`prefill_sm90.cuh:88-90`, `:155`); consumer thread index
`threadIdx.x - NUM_COPY_THREADS` (`:285`, `:296`) follows automatically; the
`barrier_O` arrive by the last warp (`mainloop_mma.cuh:83`) is still the last
consumer warp.  The ping-pong barrier indices assume consumer WGs are 1 and 2
(`named_barrier.cuh:29-31`, `:33-44`, `:46-58`, `mma_init` `:88-107` tests
`> 1`): parametrize by `kFirstConsumerWG = NUM_PRODUCER_WGS` (WGs 2, 3).
`mainloop_mma.cuh` itself is untouched (C5 holds textually).  `MIXED_FA3_TRACE`
stays thread 0.

**Register pool (the load-bearing arithmetic).**  `__launch_bounds__(512, 1)`
(`prefill_sm90.cuh:47`) -> 128 registers per thread at launch; `setmaxnreg`
must satisfy `256 P + 256 C <= 65536`, i.e. **P + C <= 256**, both multiples of
8.  Admissible splits: **72 / 184** (producer code shape unchanged — it fits 72
with STACK 0 today, `prefill_sm90.cuh:153-159`; the live set shrinks with 3
pages per operand), 64 / 192, 80 / 176, 56 / 200.  The consumer runs at 216
today (`:216`) with an unmeasured live set; the estimate O 64 + S 48 + P 24 +
row statistics 4 + descriptors/addresses ~20 = ~160 says 184 has margin, but
this is the item the build must prove first (A4 rule: `ptxas -v` free of
C7507, exactly two `USETMAXREG` in the SASS — `DEALLOC 0x48`,
`TRY_ALLOC 0xB8` for 72/184 — STACK 0 in both regions).  The a16 static
module keeps 12 warps and 72/216 (its SASS must stay byte-identical: it is at
parity and is the control).

**Issue budget.**  Per pair: 3417 warp-instructions of work + the per-warp
protocol duplicated on 4 more warps (acquire, `read_meta`, fences, commits,
loop control: ~90 per warp -> +360) = ~3780 total, **~470 per producer warp**
(after F24a's decode floor, ~425).  At today's 0.196-0.22 IPC per warp that is
2150-2400 cycles = **1.09-1.21 us < T_c 1.32-1.39** — the producer stops being
the bound and the pair is the consumer's.  SMSP occupancy: each of the four
schedulers hosts 2 producer + 2 consumer warps; issue-active today 44.6 %
(producer 0.22 + consumers 2 x 0.11), predicted 0.44 + 0.22 = 0.66 — below
saturation, so `not_selected` grows but does not dominate.  The assumption
that must hold: the producer's stall mix is per-warp latency (wait 27 %,
short_sb 9 %, long_sb 9 % = 45 %) and not a pipe shared by the two producer
warps of a scheduler (dispatch 15 % is the ambiguous slice); F24a removes the
one candidate shared pipe (96 `F2FP.E4M3` per thread per pair) before (C) is
measured, so (C)'s result is interpretable either way.

**What the ping-pong loses.**  Nothing structurally: the two consumer WGs
and their barrier protocol are unchanged.  It loses registers (216 -> 184) and
issue slots (2 -> 4 co-resident warps per scheduler).  The consumer-side
check is the transport_a16-style control inside the fp8 module: a build flag
that skips `expand_operand` (garbage values, timing only) on the 16-warp
layout must show the consumer cadence within 5 % of 12-warp transport_a16
(283 us); if it does not, no producer-side lever can reach parity and (A) is
dead too.

### 2D. The producer's per-pair instruction floor

Per 16 coefficients, bf16 output, exact:

*E4M3 today* (`:154-169`): `cvt.rn.f16x2.e4m3x2` (F2FP) + `SHF` + `LOP3` +
`IMAD` per pair = 4 per 2 values -> **32**, + 8 `HMUL2` = 40 ALU per block.

*E4M3 bit placement* (`csrc/xqa/mhaUtils.cuh:633-662` [16]): per 4 values two
`PRMT` (byte spread with sign replicate), two `SHF` (<< 4), two `LOP3`
(mask `0x87F087F0`) = 6 per 4 values -> **24**, + 8 `HMUL2` = 32 per block,
**no F2FP**.  The placed value is `x * 2^-120` (eeee in the bf16 exponent field
unbiased; subnormals `m * 2^-9` land as bf16 subnormals `m * 2^-129`,
`mul.rn.bf16x2` exact on them, verified exhaustively on H200 per the same
comment), so the fold becomes **2^120** instead of [23]'s 2^112
(`:697-698` `gs8`), exact while `block_scale * global < 2^8` — with the
sealer's cap of 128 (`flashinfer/quantization/kv_cache_fp8.py:15-20`) that
is `global < 2`; the quantizer sets `global = amax / (448 * 128)`
(`:95`), so the bound is `amax < 114688` for the tensor, i.e. always in
practice.  The host check `mixed_page_prefill.py:178-181` changes to
`gs * 128 * 2^120 < bf16 max`.  Per block: `LDS.128 + LDS.32 + 5 + 24 + 8 +
2 STS.128 = 41` (from 49, **-16 %**), and 96 F2FP per thread per pair
become 6 (the scale chain's).

*E2M1 today* (`:117-149`): 20 per 8 nibbles (two-byte LUT via `PRMT`,
interleave, sign) -> 40, + 8 `HMUL2` = 48; block 57.
*E2M1 bit placement* (`mhaUtils.cuh:664-676`): 1 `SHF` + 4 x (`PRMT` +
`SHF` + `LOP3`) = 13 per 8 nibbles -> 26, value `x * 2^-126`.  The 2^126
**cannot** be folded (E2M1 block scales reach 448: `448 * g * 2^126` overflows
bf16), an exponent add breaks the zero/subnormal lanes, so it costs a second
`HMUL2` per word: 26 + 16 = 42 -> block **52** (from 57, -9 %).  The LUT stays
acceptable; the placement variant is optional margin.

*Scale chain* (`:832-847`, 5 instructions per block): irreducible per (row,
block) — the scale is per 16 coefficients along D for both K and V, so it
cannot be factored into Q, P or `sm_scale` (and doing so would change the
rounding the bit-exact tests pin).

**Floor per thread per pair (fp8):** 12 x 41 + copies (12 `LDGSTS` + ~24
address) + chunk reads/protocol (~90) = **~620**, vs ~854 today (-27 %).
**IPC the floor would need for parity on 4 warps: 620 / 2600 = 0.24**
(today 0.196-0.22).  (D) alone therefore lands at ~1.7-1.9 us per pair
(fp8 415-440 us) — **necessary margin, not sufficient**.  Its second effect
(no F2FP) is what makes (C)'s IPC assumption testable.

### 2E. Mixed: the dynamic module's copy issue

The bench's mixed stream tags page `p` with `p % 3`
(`benchmarks/bench_fa3_mixed_page_transport.py:30-31`): every tile is 2 A16 +
2 FP8 + 2 FP4 pages, so a tile-uniform fast path does nothing.  The dynamic
module pays `iss 2.0-2.2 us` per pair against 0.85 static (`fa13ad89`) with
the same executed copies, and its SASS was 5263 instructions before [22]
(synthesis line 73) — the [40] diagnosis applies (backends "Track S step 3":
predicated copy variants x unrolled pages, two expansion bodies x pages,
i-fetch stalls; dyn SASS 17.9 K -> 8.8 K, warp-cycles per issued instruction
10.0 -> 4.5).  Here the unrolled `issue_tile_copies` carries 6 pages x 3 arms
(`:782-817`) plus both `compressed_src` set-ups (`:787-793`), and the dynamic
`expand_operand` arm 6 pages x (2 load arms + 2 expand arms) one page ahead
(`:958-979`).

**Design ([40]'s pattern for FA3).**  (E1) `chunk_store` (`:556-580`) reduces
the row's eight tags into **two 6-bit masks** `m8`, `m4` (via the existing
`__reduce_or_sync`, `:565`) stored in the `TileMeta` tag bytes; `TileRegs`
(`:582-594`) exposes them.  (E2) **format-outer, page-rolled loops**: for each
of FP8, FP4, A16, `for (m = mask; m; m &= m - 1) { i = ffs(m) - 1; ... }` with
one copy body per format; the page index is read from the chunk table by
`LDS.32` at `row_base + 4 i` (a runtime index into the `pages[6]` register
array would be C2's local-memory violation), destinations become
`base + i * PAGE_REGION_BYTES` by one `IMAD` (immediates are lost for the
dynamic module only).  (E3) the expansion likewise: two format loops over the
masks, **two pages per step** from consecutive set bits (second body
predicated off on an odd count) so the load-ahead pipelining of the static
path (`:933-949`) is kept.  Per page ~10 instructions of issue vs ~7 static;
predicted **iss 2.1 -> ~1.2 us on 4 warps, ~0.65 on 8**; `fin` for a
2-FP8 + 2-FP4 tile is two thirds of the static expansion.  Mixed pair after
F24a+b+c: 0.65 + 0.5-0.6 + 0.15 = **1.3-1.4 us ~ T_c -> 300-345 us**; the
acceptance row is inside the band, so (E) is "build and measure", not
"predicted pass".

## 3. The chosen sequence, with arithmetic

**F24a (D): decode floor in the 12-warp layout.**  fp8 per thread per pair
854 -> ~758 (only the decode changes; copies and protocol untouched) at
unchanged IPC -> pair 2.20 -> 1.95 us -> **415-440 us**; fp4 with the LUT kept
unchanged, with the placement variant 507 -> 470-495.  This step is cheap
(one decode routine and the fold constant), and it isolates the IPC
question: if the pair falls *more* than the count ratio (i.e. below 1.9 us),
F2FP was a pipe bound and (C)'s IPC assumption is confirmed from the other
side; if exactly the ratio, the bound is per-warp latency and (C) doubles the
rate as modelled.

**F24b (C): two producer warp groups (fp8 / fp4 / dynamic modules only).**
Per producer warp ~425 warp-instructions per pair at 0.196-0.22 IPC ->
1930-2170 cycles = **0.98-1.10 us <= T_c 1.32-1.39** (margin 20-30 %).  The
pair becomes the consumer's: **fp8 290-320 / 295-325, fp4 295-325 / 300-330**
(the band's top allows the consumer to lose up to 5 % to 184 registers and
issue-slot sharing; the bottom is transport_a16 itself).  Both inside <= 330.

**F24c (E): dynamic dispatch.**  Mixed **300-345** as derived in 2E.

If F24b's consumer control (2C, last paragraph) fails, i.e. the 16-warp
consumer is > 5 % slower than the 12-warp one with expansion skipped, stop:
neither (C) nor (A) can reach parity on this consumer, and the requirement is
re-stated from that number as A7 was.

## 4. Invariant changes (dataflow doc amendments to write when F24b lands)

- **A8 / D6, D2, D3 (ownership).** Producer thread `t` of 256 owns block
  `u%8` of row `u/8` (`u = t & 127`) of pages `i ≡ t>>7 (mod 2)`; A16 pages
  chunk `u%16` of rows `u/16`, `u/16+8` of those pages.  The row's eight lanes
  are one warp; D3's "reads only what its own `cp.async` wrote" and the
  `__syncwarp` landing-chunk rule are unchanged.  The a16 static module keeps
  128 producer threads and A7's formulas.
- **C3 (budgets).** `setmaxnreg` 72 / 184 at `__launch_bounds__(512, 1)`
  (pool 256 x 72 + 256 x 184 = 65536); a16 module 72 / 216 at 384 threads.
  Build check per A4: no C7507, two `USETMAXREG`, STACK 0 in both regions,
  for every module.
- **C4 (barrier accounting).** `PipelineAsync` producer arrival count 256
  (consumer 256 unchanged); `kQueryEmpty` 512; `kProducerWG` 256; Q TMA
  issued by warp 0 of WG 0 only; chunk-table gather by threads `< 128`.
- **C5 (consumer unchanged).** Holds for `mainloop_mma.cuh`; `named_barrier.cuh`
  gains the first-consumer-WG offset; `prefill_sm90.cuh` gains the role test
  `< NUM_PRODUCER_WGS` and the split constants.  Record as C5's second
  exception (the first was the mixed shared storage).
- **C6 (issue budget).** Restated per warp: <= ~430 warp-instructions per
  pair per producer warp for fp8 at the decode floor, <= ~470 fp4.
- **C7.** Unchanged (K(last) finished before `barrier_O.wait`, `:1211-1217`,
  `:1233`); every producer thread of both WGs waits `barrier_O`.
- **New C9 (fold bound).** bf16 FP8 decode by placement folds `2^120`:
  host rejects `fp8_global_scale * 128 >= 2^8`.
- **New C10 (dynamic dispatch, F24c).** The dynamic module's copy and
  expansion loops are format-outer over per-tile page masks with one body per
  format; page indices are read from the chunk table by index, never from a
  runtime-indexed register array (C2).

## 5. Verification artifacts (confirmation, not tuning)

SASS (`cuobjdump -sass`, producer region = `USETMAXREG.DEALLOC .. EXIT` of
`*_paged_sm90_kernel_mask_1`), per module:
- F24a fp8: `F2FP.E4M3` **6 per pair per thread** (was 8 per block: region
  count ~30, from 270); `IMAD` sign-fix 0; per block exactly `LDS.128, LDS.32,
  5, 8 PRMT, 8 SHF, 8 LOP3, 8 HMUL2, 2 STS.128 [R+imm]`; `STACK 0`; `BAR.SYNC`
  4, `FENCE.VIEW.ASYNC` 3, `LDGSTS` 36+36, `LD.E/ST.E` 0 (unchanged protocol
  counts).  fp4 with the placement variant: `PRMT` per block 4 -> ~13 + sign,
  `HMUL2` 16.
- F24b: exactly two `USETMAXREG` (`DEALLOC 0x48`, `TRY_ALLOC 0xB8`); `ptxas -v`
  without C7507; `STACK 0` both regions; per thread per pair `LDGSTS` 6 + 6,
  `STS.128` 12, `LDS.128` 6 (+4 chunk table); a16 module SASS byte-identical
  to `5cc416fd`; one `UTMALDG` site under a WG-0 predicate.
- F24c dyn: SASS <= ~3000 in the producer region (from 5263 pre-[22]); one
  copy body and one expansion body per format; `LDL/STL` 0 (the 16 B stack of
  today's dyn module gone or unchanged, never in the pair loop).

ncu (fp8 module, q=1, `--repeats 1`):
- producer `inst_executed` per pair: F24a ~3030 +-3 %; F24b ~3780 +-5 %
  (work conserved + protocol duplication), **per producer warp <= 480**;
- `smsp__issue_active` 44.6 % -> 55-70 %; producer per-warp IPC >= 0.18
  (`smsp__inst_executed` by warp); stall mix: `dispatch` down after F24a,
  `not_selected` up after F24b but `selected` >= 18 % per producer warp;
- `l1tex__data_pipe_lsu_wavefronts_mem_shared` per pair unchanged (~2570) —
  the layout is untouched; `LDGSTS.128` 4.00 wavefronts per instruction,
  `STS.128` 8.0, `LDS.128` 3.5 as in A7;
- tensor pipe active equal to transport_a16 within 3 % (consumer-bound proof);
- 16-warp consumer control (expansion skipped): duration within 5 % of the
  12-warp transport_a16 run.

Trace (`MIXED_FA3_TRACE`, us per pair, CTA 0 item 0/1):
- F24a fp8: `fin` 1.2 -> <= 1.0 (`expK` + `expV` <= 0.95); `iss` unchanged 0.85.
- F24b: `iss` <= 0.45 static (half the copies per thread), `fin` <= 0.6,
  **`acq` >= 0.25** (the producer waits on the consumers' release = consumer
  bound; today 0.2 and falling would mean the producer still paces);
  `wait` <= 0.05.
- F24c dyn: `iss` <= 0.7, `fin` <= 0.6.

Bench (`benchmarks/bench_fa3_mixed_page_transport.py --q-lens 1 64 --repeats 1
--trials 5`, nkcut2 lock, co-tenant rule: bursts < 1.5 ms):

| row | accept | reject / re-derive |
|---|---|---|
| stock_a16 | 297-303 / 306-312 (control) | drift > 3 % -> session offset, rerun |
| transport_a16 | 281-290 / 284-292 | any change on a byte-identical a16 module = machine, not code |
| fp8 after F24a | <= 445 / <= 455 | > 460: count fell but time did not -> pipe/port, re-derive |
| fp8 after F24b | **<= 330 / <= 330** (predict 290-320) | 331-360: producer still paces (trace `acq` ~0) -> F24c's mask loops on the static path buy nothing; consider (A) with the trace of 2A |
| fp4 after F24b | **<= 330 / <= 330** (predict 295-325) | same |
| mixed after F24c | **<= 330 / <= 330** (predict 300-345) | 331-350: record, attribute `iss` vs `fin` from the trace; > 350: [40]-pattern did not transfer, re-state |

Correctness (`tests/attention/run_fa3_mixed_page_transport.py`, pytest is
banned): 70/70 bit-exact, plus (a) a kv_len whose last tile has an odd number
of valid pages and a partial page (e.g. `96 k + 3 * 16 + 5`) so page-parity
ownership and the src-size-0 tail meet; (b) the many-items case at 16 warps
(C7); (c) fp8 block scale 128 with a global just below the 2^8 / 128 fold
bound, and one above it expecting the host `ValueError`; (d) E4M3 subnormal
payload with the placement decode (the `m * 2^-129` path).

## 6. Do not build if

1. `ptxas -v` prints C7507 or `STACK > 0` for the consumer at **184, 192 and
   176** (all three admissible splits with the producer at 72/64/80): the
   consumer needs more than the pool can give at 16 warps; stop after F24a and
   evaluate (A) by its trace condition (2A).
2. The 16-warp consumer control (expansion skipped) is > 5 % slower than
   12-warp transport_a16: issue-slot sharing alone breaks parity; no
   producer-side lever remains, (A) is excluded by the same measurement.
3. F24a's fp8 pair time falls by less than 60 % of the instruction-count
   ratio (i.e. stays > 2.05 us) *and* ncu shows `dispatch` unchanged: the
   producer is bound by something other than issue count or F2FP (e.g. an
   MIO/LSU queue on `LDGSTS`), and doubling warps would not help — re-attribute
   with PC sampling before F24b.
4. Production fp8 global scales violate `gs * 128 < 2^8`: keep the F2FP
   decode (F24a's fp8 part is dropped; F24b is unaffected).
5. For F24c: the dynamic module's `iss` on 4 warps stays > 1.5 us after the
   mask loops: the excess is not i-fetch/branch structure, and the mixed
   target must be re-stated from the trace rather than pursued with more
   dispatch work.
6. Any design that reintroduces a group barrier inside the pair body
   (barrier B), a runtime-indexed register array (C2), or a per-token
   ownership (A2): [23]'s measurements already rejected each.

## 7. Files touched by each step (for the implementer; no edits here)

- F24a: `include/flashinfer/attention/hopper/sparse_mixed_mainloop.cuh`
  `e4m3x4_to_a16` (`:154-169`), `make_bases` fold (`:697-698`), optional
  `e2m1x8_to_a16` placement variant + `decode_fp4_block` (`:117-149`,
  `:865-878`); `flashinfer/mixed_page_prefill.py:174-181` host bound;
  dataflow doc C9.
- F24b: `kernel_traits.cuh:125-130`, `:187-205` (producer WG knob, a16 module
  = 1); `prefill_sm90.cuh:47`, `:88-90`, `:96`, `:153-159`, `:214-216`;
  `named_barrier.cuh:29-58`, `:88-107`; `sparse_mixed_mainloop.cuh:269`,
  `:312`, `:634-635` (`u = t & 127`, page parity in `issue_tile_copies`
  `:782-817` and `expand_operand` `:924-956`), `:1222` (TMA by WG 0 only);
  `flashinfer/mixed_page_prefill.py` unchanged (same URIs); dataflow doc A8,
  C3-C7 amendments.
- F24c: `sparse_mixed_mainloop.cuh` `chunk_store` (`:556-580`), `TileRegs`
  (`:582-594`), dynamic arms of `issue_tile_copies` (`:806-815`) and
  `expand_operand` (`:958-979`); dataflow doc C10.
