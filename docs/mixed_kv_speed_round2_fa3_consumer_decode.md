# Mixed KV speed, round 2, Track F: lever [24] — how the compressed FA3 prefill reaches <= 330 us

Design, then implementation in three commits (F24a, F24b, F24c; "as written"
sections at the end).  Base: `claude/mixed-kv-sm90-tma` @ `5cc416fd` (merge of
[23], commit `fa13ad89`).  Paths are relative to the repository root; line
numbers are those of `5cc416fd` unless a section says otherwise.  Prior
documents this extends: `docs/mixed_kv_page_transport_dataflow.md` (D1-D6,
C1-C7, A1-A7), `docs/mixed_kv_speed_plan.md` Track F ([20]-[24], lines
214-241), `docs/mixed_kv_speed_round1_synthesis.md` (Track F paragraph, line
73), `docs/mixed_kv_page_transport_backends.md` (P0.3 consumer trace and RT
constants, lines 528-619; round-2 baseline / setmaxnreg rule, lines 1286-1357;
[40] dynamic dispatch, lines 786-932).

Revision 2 (this file) answers the review of revision 1 (`495549a1`).  The
changes of substance: (1) the FP8 fold bound was wrong for the payload the
kernel accepts — the block-scale byte reaches 448, so `s * g * 2^120` overflows
for the existing extremes cases; F24a now carries XQA's exact per-block fallback
(2.4, C9) and the host bound is gone; (2) F24b's control flow omitted the
hard-coded 128-thread constants in `prefill_sm90.cuh:63` and `epilogue.cuh:77-78`
(a deadlock on the first run), the ping-pong barrier remap, the runtime page
index of the parity split, and the scale-slot size (2.3, 7); (3) the shared-
memory LSU pipe is added as a first-class budget (C11) — at the target cadence
the [23] layout saturates it, so F24b carries two wavefront reductions with
derived counts (2.3, 2.6); (4) F24c's time prediction is withdrawn as a
derivation — the commit stands on its structural properties, and one ncu
pc-sampling run of the mixed module is the artifact that decides whether its
number is kept (2.5, 6); (5) cycle budgets are restated at the measured
1.84 GHz and the per-pair accounting at 44 producer calls per item (1).

## 0. Summary

**Chosen: a sequence F24a -> F24b -> F24c = (D) decode floor with an exact
fallback, then (C) a second producer warp group plus the smem-pipe reductions
it needs, then (E) the dynamic module's per-page dispatch.**  (B) is rejected on
the wgmma operand rule; (A) is kept as the fallback that is built only if a
consumer-side trace shows the slack it needs.

| mode (us, q=1 / q=64) | today ([23]) | after F24a (D) | after F24a+b (C) | after F24a+b+c (E) | accept |
|---|---|---|---|---|---|
| stock_a16 | 299.8 / 309.4 | unchanged | unchanged | unchanged | control |
| transport_a16 | 283.4 / 286.9 | unchanged (module byte-identical) | unchanged (a16 module keeps 12 warps, byte-identical) | unchanged | 283-290 |
| fp8 static | 474.0 / 483.0 | 440-460 | **290-345 / 295-350** (band straddles the target; see 3) | same | <= 330 |
| fp4 static | 507.2 / 517.5 | unchanged (LUT kept) | **295-350 / 300-355** | same | <= 330 |
| mixed (dynamic) | 880.1 / 906.9 | 860-880 | 340-380 | not derived (2.5): bounded below by fp8+fp4's row, gate = ncu artifact | <= 330 (marginal) |

Main risks, in order: (1) the consumer must give up 32 registers (216 -> 184)
for the second producer warp group; if it spills at every admissible split the
pool arithmetic fails and the sequence stops at F24a; (2) the shared-memory LSU
pipe: 2571 wavefront-cycles per producer call today against a target cadence
of 2370-2760 cycles — F24b's two reductions bring it to ~1850-2000 (67-84 %),
which is inside the pipe but not comfortably (C11); (3) the per-producer-warp
IPC of 0.196-0.22 is assumed to be per-warp dependent-issue latency and to hold
with two producer warps per scheduler — at the corrected count (~480-495
warp-instructions per producer warp per call) F24b's producer lands at
1.16-1.37 us against T_c 1.29-1.36 us, i.e. the pass is by the middle of the
band, not its top; (4) the mixed module ends inside its acceptance band only
after (E), whose effect is not derived (2.5).

## 1. Where the time goes (restating the [23] record as a per-call budget)

Benchmark shape B=17, S=4096, 8 KV heads, GQA 4, D=128, bf16.  At q=1 every
(q-head, request) is a work item padded to a 128-row q tile: 17 x 32 = 544
items on 132 SMs = 4 full scheduler rounds + a 16-item fifth round; each item
is 43 KV tiles (4096/96 -> 43), which the producer serves in **44
`produce_pair` calls** (K(last) alone, 42 pairs (K(t-1), V(t)), V(0) alone;
`sparse_mixed_mainloop.cuh:1211`, `:1275`) — the [23] ncu record's 61.54 M
shared wavefronts / 2571 per call = 23,930 = 544 x 44 confirms the count.  The
kernel is ~5 x 44 = 220 calls deep; the wall also contains per-item fixed costs
(Q TMA, `barrier_O`, chunk-0 gather round trip, epilogue), so wall / 220 is an
**upper bound** on the per-call cadence.

**Clock.**  The [23] ncu run shows `gpc__cycles_elapsed` at 1.79 GHz and
`sm__cycles_elapsed.max` 859 k over 468 us = **1.84 GHz** under the co-tenant,
not the 1.98 GHz nominal used by revision 1.  All cycle figures below are at
1.84 GHz (8 % fewer cycles per us than revision 1).

| mode | wall us | us per call (wall / 220) | cycles per call @ 1.84 GHz |
|---|---|---|---|
| stock_a16 | 299.8 | 1.36 | 2510 |
| transport_a16 | 283.4 | 1.29 | 2370 |
| fp8 | 474.0 (ncu 468) | 2.15 (2.13) | 3960 (ncu 3915) |
| fp4 | 507.2 | 2.31 | 4240 |
| mixed | 880.1 | 4.00 | 7360 |

So the consumer's cadence is **T_c <= 1.29-1.36 us (2370-2510 cycles)**; the
target 330 us is a call of **<= 1.50 us (2760 cycles)**; the producer's call
must be **<= ~1.25 us** so that the consumer, not the producer, sets the
cadence (dataflow section 5: `T = max(T_c, T_i + T_x)`).  The fp8 producer
today is `iss 0.85 + fin 1.2 + acq/gap ~0.1 = ~2.15 us` (commit `fa13ad89`
trace; the trace's 0.7 us "gap" is trace overhead, absent from the plain build).

**Instruction side** ([23] SASS and ncu): **3417 warp-instructions per call on
4 producer warps = 854 per warp**, at **0.196-0.22 IPC per warp** (3417 / 3915
= 0.87 aggregate).  Per FP8 block the producer executes exactly `LDS.128 +
LDS.32 + 5 (scale: PRMT, F2FP.E4M3, HADD2.F32, FMUL, F2FP.PACK) + 8 x {F2FP,
SHF, LOP3, IMAD} + 8 HMUL2 + 2 STS.128 = 49` (`:154-169`, `:832-847`,
`:849-863`, `:907-919`), 12 blocks per thread per call (6 pages x 2
operands); per FP4 block ~57.  Copies: 12 payload + 12 scale-word `LDGSTS`
per thread per call (`:753-780`); chunk table four `LDS.128`; protocol two
acquires, one `cp.async` fence, one `fence.proxy.async`, two commits.  Blocks
are 588 of the 854; the remaining 266 (copies, addresses, chunk table,
protocol, loop) are "the rest" below.  The consumer executes 3297
warp-instructions per call = 412 per consumer warp.

**Pipe side (corrected).**  Revision 1 wrote "XU 3 %": that is the
`math_pipe_throttle` stall share (5798 / 175,602 samples).  The pipe
utilisations in the same record are `sm__inst_executed_pipe_xu` 20.8 % of peak
sustained active, `pipe_alu` 30.9 %, `pipe_fma` 17.0 %, `pipe_lsu` 10.8 %,
tensor 39.9 %; `dispatch_stall` is 13 % of samples (22,939).

**Shared-memory LSU pipe (new, C11).**  `l1tex__data_pipe_lsu_wavefronts_mem_shared`
= 61.54 M over 23,930 calls = **2571 wavefront-cycles per call per SM**, by op
class: wgmma operand reads (`op_gmma`) 23.95 M = 1001 per call; `op_st` 19.28 M =
805 (the 2-way `STS.128`: 96 warp-instructions x 8 wavefronts = 768, plus
chunk-table stores); `op_ld` 8.70 M = 363 (landing `LDS.128` at 3.5, scale
`LDS.32` at 1, chunk-table `LDS.128`, barrier reads); remainder (`LDGSTS`
smem side) 9.6 M = 401.  At today's 3915-cycle call that is 66 % of the pipe —
"not the bound", as revision 1 said — but **at the target cadence of 2370-2760
cycles the same 2571 wavefronts are 93-108 % of the pipe**, the wgmma reads are
in it, and doubling the producer warps adds no pipe capacity.  transport_a16 at
parity uses ~1400 per call (57 %).  Hence C11: the layout must be brought to
**<= ~2000 wavefront-cycles per call (<= ~75-85 % at the target cadence) before
or inside F24b**, with the count derived, not tuned (2.3, "smem-pipe
reductions").

The design question is therefore twofold: **issue rate** (854 dependent
instructions per warp per call at 0.2 IPC is 4270 cycles; parity needs the
per-warp count below ~500 at 0.2 IPC, or the IPC above 0.35 at today's count)
**and pipe occupancy** (the wavefront budget above).  The alternatives are
scored on both.

## 2. Alternatives

### 2A. Consumer-side decode (the two consumer warp groups expand K and V)

**Data flow.**  Producer: copies compressed blocks and scale words exactly as
[23], then commits the stage **at issue** with `cp.async.mbarrier.arrive` — the
A16 completion mode of `issue_operand` becomes the only mode (`:994-1010`);
`finish_pending_pair`, the pending records, `cp.async.wait_group` and the
producer's `fence.proxy.async` go away.  Consumer WG w (w = 0, 1) after
`consumer_wait` on the full barrier (`mainloop_mma.cuh:233`, `:238`) expands
half the tile — rows `48w .. 48w+47`, thread `u` of the WG owning block `u%8`
of rows `48w + u/8 + 16j`, j = 0..2 — into the wgmma operand layout in place,
issues `fence.proxy.async`, and both WGs meet on a **256-thread named
barrier** before either issues the gemm on that tile.

**Control flow per call per WG** (against `mainloop_mma.cuh:231-274`):
`consumer_wait(K) ; expand K half ; fence ; bar(K expanded, 256) ;
barrier_sync (ping-pong) ; QK gemm issue ; rescale_o ; consumer_wait(V) ;
expand V half ; fence ; bar(V expanded, 256) ; PV gemm issue ; barrier_arrive
; wait<1> ; release K ; softmax ; wait<0> ; release V`.  The V expansion can
be slid under the WG's own QK gemm, the K expansion under the previous PV only
if it is moved before `wait<1>` — into the softmax slot, which already makes
the consumer 1.7x the tensor floor.

**Budget.**  The 3417 warp-instructions per call move to 8 warps: **427 per
consumer warp per call**, on top of the consumer's own 412.  Registers: the
expansion's live set (~20-30) on top of O (64) + S (48) + P (24) + softmax
state fits 216.  Smem: none (in place); the STS wavefronts move to the
consumer warps (C11 unchanged in total).  Barriers: +2 x 256-thread `bar.sync`
per call (120-200 cycles under load per P0.3(d)) plus two `fence.proxy.async`
per thread per call.

**Verdict: conditional, not chosen.**  Whether 427 extra instructions per warp
(600-850 cycles at the 0.5-0.7 IPC a 16-chain decode reaches on an otherwise
idle warp) are free depends on how much of the per-WG loop is spent *waiting*
rather than issuing; the FA3 consumer has never been traced.  If the slack is
under ~900 cycles per call per WG the cadence grows by the difference and the
target is missed.  It also breaks C5 (`mainloop_mma.cuh` gains a third loop
body) and moves the D3 ownership proof into the consumer.  **Build (A) only
if** a `%globaltimer` consumer trace of the fp8 module (six stamps per call
per WG) shows >= 900 cycles per call per WG of wait that is not the producer's
`full` barrier.

### 2B. Register-operand (RS) wgmma for the expanded K^T / V

`wgmma.mma_async` takes **A from registers or a smem descriptor and B from a
smem descriptor only**.  K is the **B** operand of QK^T (`kernel_traits.cuh:
138-139`), V the **B** operand of PV (`:140-143`; P is already the register A
operand).  V can never be a register operand; K in registers requires the
transposed product with K as A (`M` = tokens, softmax reductions across the
warps of a WG) — the XQA SWAP_AB skeleton at ~2100 instructions per 64 tokens,
with the block-owner decode spread over four threads' fragments.
**Rejected**: it removes at most the K half of the smem writes at the price of
a new consumer skeleton.

### 2C. A second producer warp group (8 producer warps; expansion split by page parity)

**Data flow.**  16 warps per CTA.  `MixedAttentionKernelTraits`
(`kernel_traits.cuh:187-205`) gains `NUM_PRODUCER_WGS` (2 for the fp8 / fp4 /
dynamic modules, **1 for the a16 module**, derived from the variant's
`kMixedStaticFormat`) and overrides `NUM_WARPS`, `NUM_THREADS`,
`NUM_PRODUCER_THREADS` (= 256), `PRODUCER_REGS` / `CONSUMER_REGS`; the base
`AttentionKernelTraits` is untouched, and every other consumer of the thread
count reads it through a detection trait `producer_warp_groups_v<Ktraits>`
(default 1) so that stock traits and the quantization traits compile as before.
Producer thread `t` in [0, 256): half `h = t >> 7`, within-half index
`u = t & 127`; **every [23] ownership formula applies to `u`, restricted to
tile pages `i = h + 2j`, j = 0..2**: block `u%8` of row `u/8` (`:634-635`),
landing chunks and scale slot as `make_bases` computes them from `u`
(`:668-706`); A16 pages (dynamic module) chunk `u%16` of rows `u/16`,
`u/16+8` (`:735-750`).  Each thread copies and expands **3 pages per operand
per tile instead of 6**.  D2, D3 (a thread reads only what its own `cp.async`
wrote — amended below for the scale word), D4 (src-size-0 zero-fill per
thread), D5, D6 and A7's wavefront rules carry over because the eight lanes of
a row stay in one warp and a warp still copies one 128 B global line per page.

*Page index without a runtime-indexed register array (C2).*  The unrolled
per-thread page index is `i = h + 2j` with `h` runtime, so `m.pages[i]`
(`TileRegs::pages[6]`, `:583`, `:799`) would become a local-memory array.
Instead `TileRegs::page(h, j)` selects between the two words of the row's
`LDS.128` pair with one `SEL` (`h ? pages[2j+1] : pages[2j]`; for the a16
module, `NUM_PRODUCER_WGS == 1`, it is `pages[j]` verbatim), and
`h * PAGE_REGION_BYTES` / `h * SCALE_PAGE_BYTES` are folded **once per work
item in `make_bases`** into `out0 / out1 / land8 / land4 / sc_rd / a16_dst`,
so every `STS / LDS / LDGSTS` keeps its `[R + imm]` form with the per-page
immediate `j * 2 * PAGE_REGION_BYTES` (verify: `STACK 0`, all `STS.128`
`[R+imm]` as today).  The tail predicate `i * 16 + row < valid` becomes
`h * 16 + (32 j + row) < valid` (one `IADD` per page in the partial-tile arm).

*Scale slot size.*  `SCALE_PAGE_BYTES = NUM_COPY_THREADS * 4` (`:283`) is
decoupled from the producer thread count: only 128 threads own any one page.
With the leader copy below the slot becomes **8 B per row, 128 B per page**
(`kMixedScaleStageBytes` = 6 x 128 = 768 B per stage per operand, from 3 KB;
smem shrinks by 13.5 KB to ~186 KB).

*Smem-pipe reductions (C11), both derived from A7's measured rules.*

(i) **`STS.128` 2-way conflict: 8 -> 4 wavefronts per instruction (805 ->
~420 per call).**  A row's eight lanes write, in the first store, chunks `2b`
(positions `{0,2,4,6} ^ (r&7)` in *both* the D-block-0 line (lanes b < 4) and
the D-block-1 line (b >= 4), 12,288 B = 0 mod 128 B apart) — the same four
bank groups twice.  Fix: per thread `swap = ((b >> 2) ^ r) & 1`; the first
store goes to chunk `2b + swap`, the second to `2b + 1 - swap`.  Then within
one row the first store's positions are `{0,2,4,6} ^ x` in line A and
`{1,3,5,7} ^ x` in line B (or the reverse for odd rows) — eight distinct bank
groups, 4 wavefronts.  The *data* order must follow: the first store carries
the decode of the packed half that holds values `8 swap .. 8 swap + 7`, so the
landing is read as **two `LDS.64` at `landing + 8 swap` and `landing + 8 (1 -
swap)`** (FP4: two `LDS.32` at `+ 4 swap`, `+ 4 (1 - swap)`) instead of one
`LDS.128` / `LDS.64` — one more `LDS` per block, wavefront-neutral: for a
16-lane `LDS.64` phase (rows r, r+1 = same octet pair) row r's b < 4 lanes read
the low halves of chunks `C = {0..3} ^ x` and its b >= 4 lanes the high halves
of the complement; row r+1 (`x | 1`, same chunk sets) reads the opposite
halves — 32 distinct banks, 2 wavefronts per `LDS.64`, 4 per block as today's
`LDS.128` (3.5 measured).  FP4 `LDS.32`: rows r, r+2 share a chunk set but use
complementary (`b&1`, swap) bank pairs; rows r+1, r+3 the other set — 32
distinct banks, 1 wavefront each, 2 per block as today's `LDS.64`.  Thread
constants only: `out0 / out1` absorb the swap in `make_bases`; the two landing
half addresses are `l8a = land8 + stage + 8 swap`, `l8b = l8a ^ 8` in
`expand_bases` (once per operand per call).  D3 is unchanged (same bytes, same
owner).

(ii) **Scale words: one 8 B `cp.async` per row by the row's lane 0 (7.95 ->
<= 4 wavefronts per instruction, 382 -> <= 192 per call, possibly 48).**  Today
each of a row's eight lanes copies a 4 B word (four lanes the same source) into
its own 4 B slot; A7 measured 7.95 wavefronts per warp instruction for this
shape.  Now lane `b == 0` of each row copies the row's 8 B (both words) into
the row's 8 B slot `scales[stage][page][row]`; the other lanes' `LDGSTS` are
predicated off (the instruction count per thread is unchanged: 12 per call).
Four active lanes per warp, four distinct 8 B sources, 32 contiguous
destination bytes: by A7's line rule one wavefront per active lane at most
(<= 4), and 1 if the replay was the same-source sharing.  Each lane reads its
4 B word with `LDS.32` at `slot + 4 (b >> 2)` (32 lanes, 32 contiguous bytes
with 4-lane broadcast: 1 wavefront, as today).  **D3 amendment**: the reading
lane did not issue the copy; visibility is by the issuing lane's
`cp.async.wait_group` followed by **one `__syncwarp()` per call** placed in
`finish_pending_pair` right after the wait (all lanes execute the wait, then
the warp barrier orders lane 0's completed writes before every lane's reads) —
the same ordering already used for the landing chunks (D3, `:916`).  Host
check: scale rows 8 B aligned (was 4 B; every supported layout has token stride
= 8 x heads bytes).

*Budget after (i) + (ii)*: gmma 1001 + st ~420 + ld ~365 (unchanged: the split
loads are wavefront-neutral) + ldgsts 240-384 (payload 192 + scales 48-192) =
**~2030-2170 by the per-class arithmetic; ~1850-2000 by scaling the measured
buckets** (the measured `LDGSTS` bucket, 401, is below the per-class sum, so the
class model over-counts it).  At the target 2370-2760 cycles: **67-90 %** of the
pipe.  The <= ~70 % the review asked for is reached only at the optimistic end;
the residual is risk (2) in section 0 and is measured per op class (5).  No
further reduction is available without changing the tile layout: payload
`LDGSTS.128` (4.0) and the wgmma reads are at their ideal.

**Control flow.**  Unchanged per thread (`load`, `:1049-1303`), with these
count and site changes (revision 1 missed the first three):
- `prefill_sm90.cuh:63` hard-codes `NUM_COPY_THREADS = NumThreadsPerWarpGroup`
  and uses it for the pipeline `producer_arv_count` (`:96`), the consumer
  thread index (`:254`, `:287`, `:291`) and the role test (`:90`, `:155`).  It
  becomes `producer_warp_groups_v<Ktraits> * 128`; role test `wg <
  NUM_PRODUCER_WGS`; the register split reads the mixed traits'
  `PRODUCER_REGS / CONSUMER_REGS` (`:159`, `:216`); stock traits keep `wg == 0`
  and 72/216 textually (byte-identical SASS for stock and a16 modules).
- `epilogue.cuh:77-78` derives `NUM_MMA_THREADS = NUM_THREADS -
  NumThreadsPerWarpGroup` = 384 for a 16-warp CTA: `NamedBarrier::sync(384,
  kValueEmpty)` (`:167`) and the `EpilogueBarrier` (`:171`, `:199`) would wait
  for 384 arrivals from 256 consumer threads (hang), and `write_O` /
  `TiledCopyO` would partition with the wrong count.  `NUM_COPY_THREADS` there
  also becomes `producer_warp_groups_v<Ktraits> * 128`.
- `PipelineAsync` `producer_arv_count` 128 -> 256 (every thread of both WGs
  commits every tile: compressed via `commit_pending`, whose pending word is
  CTA-uniform; A16 via per-thread `cp.async.mbarrier.arrive`); `kQueryEmpty`
  count `NUM_MMA_THREADS + NUM_PRODUCER_THREADS` -> 512 (`:1219-1221`,
  `mainloop_mma.cuh:315-316`, `named_barrier.cuh:90` — all follow from the trait
  override; `mainloop_mma.cuh` is not edited); `kProducerWG` group barrier
  128 -> 256 (`:1086-1089`).
- The Q TMA is issued by **warp 0 of the CTA only** — today `warp_idx_in_
  warpgroup == 0` (`:1222`) is true in both producer WGs and would issue the box
  twice against a transaction count of one.  The predicate is `thread_idx < 32`
  for two WGs and stays `warp_idx_in_warpgroup == 0` textually for one.
- Chunk-table gather by threads `t < 128` only: `chunk_store` for `t >= 128`
  would write `meta[chunk&1][row 16..31]`, aliasing the other buffer and running
  past the array; `chunk_load` would fetch valid but useless indices.  The
  predicate is warp-uniform (whole WG 0), so `__reduce_or_sync` (`:565`) stays
  well-formed; the `static_assert` `:312` becomes `CHUNK_TILES * 8 == 128`.
- Ping-pong barriers (`named_barrier.cuh:29-58`, `mma_init` `:88-107`) assume
  consumer WGs 1 and 2: `get_next_consumer_warp_group_idx<2>` returns `3 - wg`
  (1 / 0 for WGs 2, 3) and `mma_init`'s `> 1` makes both consumer WGs arrive on
  WG 1's barrier.  Parametrize by `kFirstConsumerWG = producer_warp_groups_v<
  Ktraits>`: barrier id `kWarpSchedulerWG1 + (wg - kFirstConsumerWG)` (hardware
  ids 2, 3 as today; no collision with `kQueryEmpty` 0, `kValueEmpty` 1,
  `kPrefetchIndices` 5, `kProducerWG` 6; CUTLASS offsets user ids by its
  reserved count, so no new id is added), next WG `2 kFirst + 1 - wg`,
  `mma_init` test `> kFirst`.  For `kFirst = 1` every expression folds to
  today's constants (a16 SASS unchanged).
- `barrier_O` arrive by `canonical_warp_idx == NUM_WARPS - 1`
  (`mainloop_mma.cuh:83`) is still the last consumer warp; C7 holds (both
  producer WGs finish K(last) before `barrier_O.wait`; 256 arrivals on the full
  barrier).  Tile scheduler: no thread-count assumptions (prefetch / broadcast
  are no-ops).  `cp.async` group accounting is per thread and unchanged.
  `MIXED_FA3_TRACE` stays thread 0.

**Register pool (the load-bearing arithmetic).**  `__launch_bounds__(512, 1)`
-> 128 registers per thread at launch; `setmaxnreg` must satisfy `256 P + 256 C
<= 65536`, i.e. **P + C <= 256**, both multiples of 8 (necessary and
sufficient).  Admissible splits: **72 / 184** (producer code shape unchanged —
it fits 72 with STACK 0 today; the live set shrinks with 3 pages per operand),
64 / 192, 80 / 176.  The consumer runs at 216 today with an unmeasured live
set; the estimate O 64 + S 48 + P 24 + row statistics 4 + descriptors ~20 =
~160 says 184 has margin, but this is the item the build must prove first (A4
rule: `ptxas -v` free of C7507, exactly two `USETMAXREG` — `DEALLOC 0x48`,
`TRY_ALLOC 0xB8` for 72/184 — STACK 0 in both regions).  The stock consumer's
live set is readable from `ptxas -v` of the a16 module (same `mma_f16` code);
the epilogue (O convert + `STSM`) is the likely peak.  A third producer WG is
not admissible (3 x 128 x 64 + 256 x 160: the consumer at 160 is at its
estimate with no margin).

**Issue budget (corrected).**  Work conserved: 3417 -> 3177 after F24a (-60
per thread) -> 3225 after the split loads (+12 per thread), spread over 8
warps = 403, plus the per-warp protocol duplicated on 4 more warps (acquire,
`read_meta`, fences, commits, loop control: ~90) = **~480-495 per producer warp
per call**.  At today's 0.196-0.22 IPC per warp that is 2180-2520 cycles =
**1.19-1.37 us against T_c 1.29-1.36** — the middle of the band (0.21 IPC,
485) gives 1.25 us and a consumer-bound pair; the pessimistic corner does not.
SMSP occupancy: each of the four schedulers hosts 2 producer + 2 consumer
warps; issue-active today 44.6 %, predicted `2 x 0.21 + 2 x (412 / 2450)` =
**0.75-0.80** at parity (revision 1 said 0.66 from a 300-per-warp consumer
count) — below saturation but `not_selected` will grow.  **ALU pipe**: F24a
moves the decode onto the ALU pipe (per block 9 PRMT + 8 SHF + 8 LOP3 + FSETP
= 26 vs 17 today, with F2FP on XU and IMAD on FMA gone); per producer warp per
call ~12 x 26 + ~100 = ~410 ALU instructions x 2 cycles (16 lanes) x 2 warps
per SMSP = ~1640 of 2370-2510 cycles = **65-70 % of the ALU pipe** before the
consumers' own ALU work — the second shared resource to watch (5:
`pipe_alu`, `pipe_xu` per call).  The assumption that must hold: the producer's
stall mix is per-warp latency (wait 27 %, short_sb 9 %, long_sb 9 %) and not a
pipe shared by the two producer warps of a scheduler; F24a removes the F2FP
(XU) candidate before (C) is measured, and C11 removes the smem-pipe candidate.

**Consumer controls.**  Two build flags inside the fp8 module, timing only:
`MIXED_FA3_CONTROL_SKIP_EXPAND` (no decode, no stores: isolates issue-slot
sharing and the 184-register consumer) and `MIXED_FA3_CONTROL_RAW_STS` (the
packed words are stored unchanged: keeps the STS wavefronts, so the two effects
are separable).  The 16-warp consumer with the first flag must run within 5 %
of 12-warp transport_a16; if it does not, no producer-side lever can reach
parity and (A) is dead too.

### 2D. The producer's per-call instruction floor (F24a)

Per 16 coefficients, bf16 output, exact:

*E4M3 today* (`:154-169`): `cvt.rn.f16x2.e4m3x2` (F2FP) + `SHF` + `LOP3` +
`IMAD` per pair = 4 per 2 values -> **32**, + 8 `HMUL2` = 40 ALU/FMA/XU per
block.

*E4M3 bit placement* (`csrc/xqa/mhaUtils.cuh:633-662` [16]): per 4 values two
`PRMT` (selectors `0x9180` / `0xB3A2`: byte spread with the sign replicated
into the neighbouring byte), two `SHF` (<< 4), two `LOP3` (mask `0x87F087F0`)
= 6 per 4 values -> **24**, + 8 `HMUL2` = 32 per block, **no F2FP**.  The
placed value is `x * 2^-120`: `s` at bit 15, `eeee` at [10:7], `mmm` at
[6:4]; normal `1.mmm * 2^(E-7)` -> bf16 `1.mmm * 2^(E-127)`, subnormal
`mmm * 2^-9` -> bf16 subnormal `mmm * 2^-129` (`mul.rn.bf16x2` is exact on
subnormal inputs; verified exhaustively on H200 for [16]).  E4M3 NaN codes
`0x7F / 0xFF` decode to the finite `1.111 * 2^-112` under placement instead of
NaN under `cvt`: the quantizer never emits NaN and the tests exclude it, but a
NaN payload would silently become a finite value (recorded in C9).

*The fold and its bounds (C9).*  The 2^120 is folded into the block scale:
`sf2 = bf16_rn(f32(s) * g * 2^120)`, one `HMUL2.BF16` then rounds `x * sf`
once as the reference does.  This equals the reference `bf16_rn(x *
bf16_rn(f32(s) * g))` **iff** (upper) `|s * g| < 255.5` — bf16 max is
`255 * 2^120`, and `[255.5, 256) * 2^120` rounds to +inf under round-to-nearest-
even, so `x * inf = inf`, `0 * inf = NaN`; and (lower) `|s * g| >= 2^-126` —
below it the unfolded scale is a bf16 subnormal with fewer mantissa bits and
rounds differently from the folded normal value; with `s >= 2^-9` this is
`|g| >= 2^-117` (XQA's `fp8FoldOk`, `mha_sm90.cu:533`).  **The kernel decodes
any E4M3 scale byte (up to 448)**, the quantizer's 128 cap
(`flashinfer/quantization/kv_cache_fp8.py:20`) is not a property of the
payload the kernel receives, and `tests/attention/run_fa3_mixed_page_transport.py:70`
draws block scales `0x7C = 256` and `0x7E = 448` with `g = 1` in four of the
70 cases — so no host bound on `g` alone can make the fold safe (revision 1's
`gs * 128 < 2^8` would have passed `g = 1` and poisoned those cases).
**F24a therefore carries XQA's exact fallback (`mha_sm90.cu:2791-2830`), per
block:** with `v = f32(s) * gfold` (`gfold = g * 2^120`, or **+inf when
`|g| < 2^-117`** so that every block of that operand fails the test — `0 * inf`
is NaN and also fails), `ok = |v| < 255.5 * 2^120` (`FSETP` with the abs
modifier; false for inf and NaN), and a **warp-uniform vote `__all_sync(ok)`**
(one page's blocks of four rows per warp — the same granularity as XQA's per
operand / page vote): if it passes, `sf2 = bf16x2(v)` and the folded path; if
not, the cold path multiplies the placed halves by `2^120` first (8 `HMUL2`
by the constant `0x7B807B80`, exact: `x * 2^-120 * 2^120 = x`), reloads `g`
from the grid-constant parameters (`LDC + LDG`, cold) and uses `sf2 =
bf16x2(f32(s) * g)` — the reference's rounding exactly, for every finite scale
and global.  The vote is per block (12 per call per thread: `FSETP + VOTE.ALL
+ BRA`, +3 instructions); ptxas must emit the branch as a uniform `BRA` without
`BSSY / BSYNC` (a verification item, 5).  **The host bound of [23]
(`mixed_page_prefill.py:174-181`) is removed**: the kernel is bit-identical to
the reference for every finite input, so there is nothing for the host to
reject, and the per-call `.item()` sync goes with it.  Bench and tests: the
bench's `make_transport` uses `g = 1`, scale byte `0x38 = 1.0` -> `v = 2^120`,
fold path (the fp8 rows measure the fast path); the extremes cases (`g = 1`,
scales 448 / 256 mixed with 1 / 2^-9 / 2^-7 / 2^-6 in one page) exercise the
vote failing on some pages and passing on others; new cases add `g = 0.5`
(every scale folds, up to `224 * 2^120`) and `g = 1.1 * 2^-118` (below the lower
bound: sentinel path, reference scale in the subnormal range).

*Per block after F24a*: `LDS.128 + LDS.32 + 8 (PRMT, F2FP.E4M3, HADD2, FMUL,
FSETP, VOTE, BRA, F2FP.PACK) + 24 + 8 HMUL2 + 2 STS.128 = 44` (from 49,
**-10 %**); `F2FP.E4M3` **12 per call per thread** (one per block, the scale
byte; region count ~30 = one per inlined block site, from [23]'s 270 = 9 x 30),
`F2FP.PACK` 12, `PRMT` 9 per block, `IMAD` sign-fix 0.  Per thread per call on
the measured basis: **854 - 12 x 5 = 794** (blocks 528 + rest 266); the
idealised count is 12 x 44 + 48 + 90 = 666 (revision 1 mixed the two bases).
**Parity IPC on 4 warps: 794 / 2370-2510 = 0.32-0.34** (today 0.196-0.22).

*E2M1 today* (`:117-149`): 20 per 8 nibbles (two-byte LUT via `PRMT`,
interleave, sign) -> 40, + 8 `HMUL2` = 48; block 57.  *E2M1 bit placement*
(`mhaUtils.cuh:664-676`): 13 per 8 nibbles -> 26, value `x * 2^-126`; the
2^126 cannot be folded (block scales reach 448) and costs a second `HMUL2` per
word (undo 2^126 first — exact, `0.5 * 2^-126 = 2^-127` is a bf16 subnormal
that times 2^126 is exactly 0.5, on the same non-flushing-multiply premise as
E4M3 — then one rounding multiply by `sf2`, XQA's order at `mha_sm90.cu:
2818-2821`): 26 + 16 = 42 -> block **52** (-9 %).  **The LUT is kept** in
F24a (fp4 unchanged); the placement variant is optional margin.

*Scale chain*: irreducible per (row, block) — the scale is per 16 coefficients
along D for both K and V, so it cannot be factored into Q, P or `sm_scale`
(that would change the rounding the bit-exact tests pin).

**F24a alone**: only `fin` scales (`expK + expV` ~1.1 us x 44/49 -> ~0.99), so
the call goes 2.15 -> ~2.04 us -> **fp8 440-460 us** — necessary margin, not
sufficient.  Its second effect (no F2FP) is what makes (C)'s IPC assumption
testable: if the call falls *more* than the count ratio, F2FP was a pipe bound
and (C)'s assumption is confirmed from the other side.

### 2E. Mixed: the dynamic module's copy issue (F24c)

The bench's mixed stream tags page `p` with `p % 3`: every tile is 2 A16 + 2
FP8 + 2 FP4 pages, so a tile-uniform fast path does nothing.  The dynamic
module pays `iss 2.0-2.2 us` per call against 0.85 static (`fa13ad89`) with
the same executed copies; before [22] its SASS was 5263 instructions.  The
static `iss` of 0.85 us is ~1560 cycles for ~60-80 issue-path instructions per
warp: `iss` is **not** a count-scaled quantity (it contains the acquire-to-
fence window's dependent latencies), so revision 1's "~10 vs ~7 instructions
per page -> 2.1 -> 1.2 us" was not a derivation and is withdrawn.

**What can be read from the code.**  Today's dynamic `issue_tile_copies`
(`:782-817`) is six unrolled pages x (tag extract 2 + two compares and
branches 2-4 + one executed arm ~6) with both `compressed_src` set-ups
(`:787-793`), and `expand_operand`'s dynamic arm (`:958-979`) six pages x (2
load arms + 2 expand arms) — the per-page branches are on data just loaded
from smem (`branch_resolving`), and the fully unrolled 3-format bodies make the
hot footprint the largest of the four modules (the [40] diagnosis on the XQA
side: dyn SASS 17.9 K, 6.3 `no_instruction` stalls per issue-active cycle,
fixed to 0.52 by format-outer page-rolled loops — backends lines 786-932).
After F24c the executed count per page is about the same (FP8 arm: page
`LDS.32`, `ffs` + clear 3, two `IMAD.WIDE` sources, two `IMAD` destinations,
two `LDGSTS`, loop 2 = ~12; A16 ~11): **F24c is not a count lever; its only
mechanism is the removal of data-dependent branches and i-fetch pressure**, and
whether that mechanism is active in the FA3 dynamic module has never been
measured — `nkcut2:/tmp/mixedkv-wtF23` holds `ncu_fp4*` / `ncu_fp8*` only.

**Design ([40]'s pattern for FA3; C2 and C10).**  (E1) `chunk_store`
(`:556-580`) reduces the row's eight tags into **two 6-bit masks** `m8`, `m4`
(two `__reduce_or_sync` over the octet) stored in the `TileMeta` row in place of
tags 4, 5 (`w7` = `m8 | m4 << 8 | valid << 16 | flags << 24`; tags 0..3 stay in
`w6` for the static modules' pending word and are unused by the dynamic one).
(E2) **format-outer, page-rolled copy loops** (`#pragma unroll 1`) in the
dynamic module only: for FP8, FP4, A16 in turn, `for (m = mask_h; m; m &= m -
1) { i = ffs(m) - 1; ... }` with one copy body per format, `mask_h = mask &
(h ? 0x2A : 0x15)` for the parity split; the page index is read from the chunk
table by `LDS.32` at `row_addr + 4 i` (`TileRegs` carries the row's smem
address and `w7` only — one `LDS.32` instead of two `LDS.128` per tile — never a
runtime-indexed register array), destinations become `base + i *
PAGE_REGION_BYTES` by one `IMAD` (immediates are lost for the dynamic module
only).  (E3) The expansion likewise: two format loops over the masks from the
pending word, **two pages per step** from consecutive set bits (the second
body under a warp-uniform `if` on an odd count; `__syncwarp` inside it is
legal because the mask is warp-uniform data) with the next step's two packed
loads issued before this step's stores, keeping the static path's load-ahead.
Static modules keep their unrolled immediate-offset bodies (a16 byte-identical).

**Prediction — withdrawn as a number; stated as a bound and a gate.**  The
dyn-vs-static `iss` gap (2.0-2.2 vs 0.85 us on 4 warps) is the most F24c can
recover; after F24b halves the per-thread copies the dyn `iss` is bounded
below by the static module's (~0.45 us) and above by today's shape (~1.1 us on
8 warps).  The mixed row after F24a+b+c therefore lies in **[fp8+fp4's row,
340-380]**; whether it is <= 330 is decided by the artifact, not predicted.
**Gate before F24c's number is quoted or its commit merged**: one ncu run of the
mixed module with the existing `f23_ncu_region.py` tooling reporting the
producer-region pc-sampling mix (`no_instructions`, `branch_resolving`,
`mio_throttle`, `lg_throttle`, `imc_miss`) and the dyn issue-path SASS count
per page per arm, taken on the F24b build (before F24c) and on the F24c build.
If `no_instructions + branch_resolving` is < 10 % of the producer's samples on
the F24b build, F24c's mechanism is absent and its commit is dropped from the
merge regardless of the bench row.  F24c is implemented as the third commit
because its structural properties (one body per format, C2, no per-page
branch on smem data) stand on their own reading; it is not admitted to the
*plan's arithmetic*.

## 3. The chosen sequence, with arithmetic

**F24a (D): decode floor + exact fallback in the 12-warp layout.**  fp8 per
thread per call 854 -> 794 at unchanged IPC; only `fin` changes -> call 2.15 ->
~2.04 us -> **440-460 us**; fp4 unchanged (LUT kept); mixed 860-880.  Cheap
(one decode routine, the fold constant with its sentinel, the vote) and it
isolates the IPC question as in 2D.  Correctness widens: exact for every finite
scale / global (C9), tested by the extremes cases with `g` in {1, 0.5,
1.1 x 2^-118}.

**F24b (C): two producer warp groups + C11 reductions (fp8 / fp4 / dynamic
modules only).**  Per producer warp ~480-495 warp-instructions per call at
0.196-0.22 IPC -> 2180-2520 cycles = **1.19-1.37 us vs T_c 1.29-1.36**.  The
consumer sets the cadence at the band's middle: **fp8 290-345 / 295-350, fp4
295-350 / 300-355** (bottom = transport_a16 + the consumer's loss to 184
registers and issue-slot sharing, <= 5 %; top = the producer still pacing at
0.196 IPC and 495 per warp).  The acceptance <= 330 lies inside the band, not
at its bottom; the trace's `acq` segment (>= 0.25 us = the producer waiting on
the consumers' release) is the discriminator.  Smem pipe 67-90 % (C11).

**F24c (E): dynamic dispatch.**  Bounded, gated (2E).

If F24b's consumer control (2C, last paragraph) fails, i.e. the 16-warp
consumer is > 5 % slower than the 12-warp one with expansion skipped, stop:
neither (C) nor (A) can reach parity on this consumer, and the requirement is
re-stated from that number as A7 was.

## 4. Invariant changes (dataflow doc amendments to write when F24b lands)

- **A8 / D6, D2, D3 (ownership).** Producer thread `t` of 256 owns block
  `u%8` of row `u/8` (`u = t & 127`) of tile pages `i = (t >> 7) + 2j`; A16
  pages chunk `u%16` of rows `u/16`, `u/16+8` of those pages.  The row's eight
  lanes are one warp; D3's "reads only what its own `cp.async` wrote" is
  amended for the scale word: **the row's 8 B scale slot is copied by the
  row's lane 0 and read by all eight lanes after the issuing lane's
  `cp.async.wait_group` and one `__syncwarp()` per call** (the landing-chunk
  rule of A7 applied to a copy).  Output chunk order per thread: first store to
  chunk `2b + swap`, `swap = ((b >> 2) ^ r) & 1`, second to `2b + 1 - swap`;
  landing read as two half loads in the same order.  The a16 static module
  keeps 128 producer threads and A7's formulas verbatim.
- **C3 (budgets).** `setmaxnreg` 72 / 184 at `__launch_bounds__(512, 1)`
  (pool 256 x 72 + 256 x 184 = 65536); a16 module 72 / 216 at 384 threads.
  Build check per A4: no C7507, two `USETMAXREG`, STACK 0 in both regions,
  for every module.
- **C4 (barrier accounting).** `PipelineAsync` producer arrival count 256
  (consumer 256 unchanged); `kQueryEmpty` 512; `kProducerWG` 256; Q TMA
  issued by warp 0 of the CTA only; chunk-table gather by threads `< 128`;
  ping-pong barrier ids 2, 3 indexed by `wg - kFirstConsumerWG`.
- **C5 (consumer unchanged).** Holds for `mainloop_mma.cuh`; `named_barrier.cuh`
  gains the first-consumer-WG offset; `prefill_sm90.cuh` and `epilogue.cuh` read
  the producer thread count from the traits (`producer_warp_groups_v`).  Record
  as C5's second exception (the first was the mixed shared storage).
- **C6 (issue budget).** Restated per warp: <= ~500 warp-instructions per
  call per producer warp for fp8 at the decode floor, <= ~540 fp4.
- **C7.** Unchanged (K(last) finished before `barrier_O.wait`); every producer
  thread of both WGs waits `barrier_O`.
- **New C9 (fold exactness).** bf16 FP8 decode by placement folds `2^120`
  into the block scale.  The fold is exact iff `2^-126 <= |s g| < 255.5`; the
  kernel tests the upper bound per block (`|f32(s) gfold| < 255.5 x 2^120`,
  warp vote) and the lower bound per operand (`gfold = +inf` when `|g| <
  2^-117`), and takes the exact two-multiply form otherwise.  Failure mode if
  the fold were taken blindly: `+-inf` K/V values and `0 x inf = NaN` in the
  softmax row.  No host bound.  E4M3 NaN payload codes decode to finite
  `1.111 x 2^-112 x scale` (the reference gives NaN; tests exclude NaN).
- **New C10 (dynamic dispatch, F24c).** The dynamic module's copy and
  expansion loops are format-outer over per-tile page masks with one body per
  format; page indices are read from the chunk table by `LDS.32`, never from a
  runtime-indexed register array (C2).
- **New C11 (shared-memory LSU pipe budget).** Per producer call per SM the
  wavefront-cycles of wgmma operand reads + producer `STS` + `LDS` + `LDGSTS`
  must stay <= ~85 % of the target cadence in cycles (<= ~2000 at 2370 cycles);
  every layout change states its per-class wavefront count from A7's measured
  rules before it is written, and the ncu confirmation reports the four classes
  per call.

## 5. Verification artifacts (confirmation, not tuning)

SASS (`cuobjdump -sass`, producer region = `USETMAXREG.DEALLOC .. EXIT` of
`*_paged_sm90_kernel_mask_1`), per module:
- F24a fp8: `F2FP.E4M3` **12 per call per thread** (region count ~30, from
  270); `IMAD` sign-fix 0; per block exactly `LDS.128, LDS.32, 8 (PRMT, F2FP,
  HADD2, FMUL, FSETP, VOTE.ALL, BRA, F2FP.PACK), 8 PRMT, 8 SHF, 8 LOP3, 8
  HMUL2, 2 STS.128 [R+imm]`; **no `BSSY` / `BSYNC` in the pair body** (the
  vote's branch is uniform); the cold body (8 `HMUL2` by `0x7B807B80`, `LDC`,
  `LDG`, `FMUL`, `F2FP.PACK`) present once per block site and off the fall-
  through path; `STACK 0`; `BAR.SYNC` 4, `FENCE.VIEW.ASYNC` 3, `LDGSTS` 36+36,
  `LD.E/ST.E` 0 (unchanged protocol counts).  a16 module byte-identical to
  `5cc416fd`.
- F24b: exactly two `USETMAXREG` (`DEALLOC 0x48`, `TRY_ALLOC 0xB8`); `ptxas -v`
  without C7507; `STACK 0` both regions; per thread per call `LDGSTS` 6 + 6
  (the 6 scale copies predicated on lane 0 of the row), `STS.128` 12, `LDS.64`
  12 (+ `LDS.32` 6 scales + 4 chunk table); one `UTMALDG` site under a warp-0
  predicate; `BAR.SYNC` counts 256 / 512; `WARPSYNC` (or `BAR.WARP.SYNC`) one
  more per call than [23]; a16 module SASS byte-identical to `5cc416fd`; stock
  paged kernel byte-identical.
- F24c dyn: SASS <= ~3000 in the producer region; one copy body and one
  expansion body per format; `LDL/STL` 0 in the pair loop; `LDS.32` page-index
  reads inside the rolled loops.

ncu (fp8 module, q=1, `--repeats 1`, clocks reported with every number):
- producer `inst_executed` per call: F24a ~3180 +-3 %; F24b ~3600 +-5 %
  (work conserved + protocol duplication), **per producer warp <= 500**;
- `smsp__issue_active` 44.6 % -> 70-80 %; producer per-warp IPC >= 0.18
  (`smsp__inst_executed` by warp); stall mix: `dispatch` down after F24a,
  `not_selected` up after F24b but `selected` >= 18 % per producer warp;
- **`l1tex__data_pipe_lsu_wavefronts_mem_shared` per call by op class**:
  `op_gmma` ~1000 (unchanged), `op_st` <= ~450 (from 805), `op_ld` ~365,
  `LDGSTS` <= ~250 (from 401); total <= ~2050 (C11); per instruction:
  `STS.128` 4.0 (from 8.0), `LDS.64` 2.0, `LDGSTS.128` 4.0, scale `LDGSTS` <= 4
  (from 7.95);
- `sm__inst_executed_pipe_alu` and `pipe_xu` per call (ALU <= ~70 % of the
  producer's share as derived in 2C; XU near zero for the producer);
- tensor pipe active equal to transport_a16 within 3 % (consumer-bound proof);
- 16-warp consumer controls: `SKIP_EXPAND` within 5 % of 12-warp transport_a16;
  `RAW_STS` minus `SKIP_EXPAND` = the STS wavefront cost alone.
- mixed module (gate for F24c, 2E): pc-sampling mix of the producer region on
  the F24b build and the F24c build; dyn issue-path SASS per page per arm.

Trace (`MIXED_FA3_TRACE`, us per call, CTA 0 item 0/1):
- F24a fp8: `fin` 1.2 -> <= 1.05 (`expK` + `expV` <= 1.0); `iss` unchanged 0.85.
- F24b: `iss` <= 0.45 static (half the copies per thread), `fin` <= 0.6,
  **`acq` >= 0.25** (the producer waits on the consumers' release = consumer
  bound; ~0 means the producer still paces); `wait` <= 0.05.
- F24c dyn: `iss` and `fin` recorded; not predicted.

Bench (`benchmarks/bench_fa3_mixed_page_transport.py --q-lens 1 64 --repeats 1
--trials 5`, nkcut2 lock, co-tenant rule: bursts < 1.5 ms; min / median / max):

| row | accept | reject / re-derive |
|---|---|---|
| stock_a16 | 297-303 / 306-312 (control) | drift > 3 % -> session offset, rerun |
| transport_a16 | 281-290 / 284-292 | any change on a byte-identical a16 module = machine, not code |
| fp8 after F24a | <= 465 / <= 475 | > 480: count fell but time did not -> pipe/port, re-derive with pc sampling |
| fp8 after F24b | **<= 330 / <= 330** (band 290-345) | 331-360 with trace `acq` ~0: producer still paces -> per-warp IPC did not hold, re-attribute (pipe classes, ALU) before any further step; with `acq` >= 0.25: consumer lost > 5 % to 184 registers / issue sharing -> control flags decide |
| fp4 after F24b | **<= 330 / <= 330** (band 295-350) | same |
| mixed after F24c | recorded against the gate (2E) | if the F24b-build pc-sampling shows no i-fetch / branch component, drop F24c's commit |

Correctness (`tests/attention/run_fa3_mixed_page_transport.py`, pytest is
banned): the 70 cases of [23] bit-exact, plus (a) a kv_len whose last tile has
an odd number of full pages and a partial page (`96 + 3 x 16 + 5 = 149`) so
page-parity ownership and the src-size-0 tail meet on an odd page (fp8 and
mixed); (b) the many-items case at 16 warps (C7); (c) the extremes cases with
`g` = 0.5 (fold path up to `224 x 2^120`) and `g = 1.1 x 2^-118` (sentinel:
every block on the exact path, reference scale subnormal) in addition to
`g = 1` (vote fails on the 448 / 256 pages, passes on the others); (d) the E4M3
subnormal payload (`mmm x 2^-129` placement path) — already in the extremes
payload set.

## 6. Do not build if

1. `ptxas -v` prints C7507 or `STACK > 0` for the consumer at **184, 192 and
   176** (all three admissible splits with the producer at 72/64/80): the
   consumer needs more than the pool can give at 16 warps; stop after F24a and
   evaluate (A) by its trace condition (2A).
2. The 16-warp consumer control (`SKIP_EXPAND`) is > 5 % slower than 12-warp
   transport_a16: issue-slot sharing alone breaks parity; no producer-side
   lever remains, (A) is excluded by the same measurement.
3. F24a's fp8 call time falls by less than 60 % of the instruction-count
   ratio (i.e. stays > 2.10 us) *and* ncu shows `dispatch` unchanged: the
   producer is bound by something other than issue count or F2FP, and doubling
   warps would not help — re-attribute with PC sampling before F24b.
4. The F24a SASS shows `BSSY / BSYNC` around the fold vote or the cold body on
   the fall-through path: the guard is then not the +3 instructions it is
   budgeted as; restructure (per-operand vote hoisting) before measuring.
5. F24b's per-class wavefronts exceed the C11 budget (total > ~2100 per call)
   or `STS.128` is not at 4.0: the permutation analysis is wrong for the real
   swizzle; stop and re-derive from the source counters before timing.
6. For F24c: the F24b-build pc-sampling gate (2E) shows no i-fetch / branch
   component: F24c's mechanism is absent; the commit is dropped and the mixed
   target re-stated from the trace.
7. Any design that reintroduces a group barrier inside the pair body
   (barrier B), a runtime-indexed register array (C2), or a per-token
   ownership (A2): [23]'s measurements already rejected each.

## 7. Files touched by each step

- F24a: `include/flashinfer/attention/hopper/sparse_mixed_mainloop.cuh`
  `e4m3x4_to_a16` (`:154-169`, placement for bf16), `block_scale_a16x2` split
  into the scale-byte-to-f32 step and the pack (`:832-845`), `expand_block`
  (`:907-919`: vote, fold path, cold path), `expand_pending` /
  `expand_operand` (pass `Params`, `isK` for the cold reload), `make_bases`
  fold + sentinel (`:697-698`); `flashinfer/mixed_page_prefill.py:174-181` host
  bound removed; `tests/attention/run_fa3_mixed_page_transport.py` extremes
  with a `gs` parameter and two new `g` values; this document (as written).
- F24b: `kernel_traits.cuh:77-83` (scale slot 8 B per row), `:185-205`
  (`NUM_PRODUCER_WGS`, thread counts, register split, pool static_assert);
  `named_barrier.cuh` (`producer_warp_groups_v`, `kFirstConsumerWG` in the
  index helpers and `WarpScheduler`); `prefill_sm90.cuh:63`, `:88-91`, `:96`,
  `:153-160`, `:213-217`; `epilogue.cuh:77-78`; `sparse_mixed_mainloop.cuh`
  constants (`:269`, `:283`, `:312`, `:318`), `TileRegs::page`, `make_bases`
  (parity offsets, swap), `expand_bases` (half addresses), `load_packed` (two
  half loads), `copy_compressed_page` (leader 8 B scale copy),
  `issue_tile_copies` / `expand_operand` (pages per thread, odd count), `load`
  (gather predicate, Q issuer, `__syncwarp` after the wait), host alignment
  check (scales 8 B), control flags; `tests/attention/run_fa3_mixed_page_
  transport.py` parity-tail case; `flashinfer/mixed_page_prefill.py` unchanged
  (same URIs); this document (as written).
- F24c: `kernel_traits.cuh` `MixedTileMeta` (masks); `sparse_mixed_mainloop.cuh`
  `chunk_store`, `TileRegs` / `read_meta` (dynamic: `w7` + row address),
  dynamic arms of `issue_tile_copies` and `expand_operand` (format-outer rolled
  loops); this document (as written).

---

## As written: F24a (decode floor + exact fold fallback)

Files: `include/flashinfer/attention/hopper/sparse_mixed_mainloop.cuh`,
`flashinfer/mixed_page_prefill.py`, `tests/attention/run_fa3_mixed_page_transport.py`.
Not built or run in this worktree (review by reading; the confirmation run is
the remote step listed in section 5).

**Data flow (bf16 modules).**
- `mixed_detail::e4m3x4_to_a16<bf16>(w)`: `a = prmt(w, w, 0x9180)`, `b = prmt(w,
  w, 0xB3A2)`, `p01 = (a << 4) & 0x87F087F0`, `p23 = (b << 4) & 0x87F087F0` —
  values `x * 2^-120`.  The f16 instantiation keeps the two `cvt.rn.f16x2.e4m3x2`.
- `make_bases`: `gs8 = |g| >= 2^-117 ? g * 2^120 : +inf` (`kFp8FoldMinGlobal`,
  `kFp8Fold`, `kFp8FoldSentinelBits`); f16: `gs8 = g`.
- `expand_block<FP8>(prm, isK, e, b, packed, sw, i, t)`: `v = scale_byte_f32(sw,
  sel) * gs8`; placed decode of the four words; `fold_ok = |v| <
  kFp8FoldMax (255.5 * 2^120)`; `if (__all_sync(~0, fold_ok))` -> `sf2 =
  a16x2_broadcast(v)`; else -> eight `mul.rn.bf16x2` by `0x7B807B80` (2^120),
  `sf2 = a16x2_broadcast(scale_byte_f32(sw, sel) * fp8_global_plain(prm, isK))`
  where `fp8_global_plain` dereferences the operand's global-scale pointer from
  the grid-constant `Params` (cold `LDC + LDG`; no register held across the
  loop).  Then the eight rounding multiplies by `sf2`, `__syncwarp`, the two
  `STS.128` — unchanged.  FP4 arm: LUT decode into `lo, hi`, `sf2 =
  a16x2_broadcast(v)` with `v = f32(s) * gs4`, same tail.
- `block_scale_a16x2` is split into `scale_byte_f32` (PRMT, F2FP.E4M3, HADD2)
  and `a16x2_broadcast` (F2FP.PACK) so the cold path can reuse the byte step.
- `decode_fp8_block` / `decode_fp4_block` are removed (the vote sits between
  decode and multiply, so the decode is inlined in `expand_block`).

**Control flow.**  `expand_pending(prm, op, t)` -> `expand_operand(prm, op.isK,
...)` -> `expand_block(prm, isK, ...)`; the K / V call sites in
`finish_pending_pair` stay explicit (`op.isK` is a constant after inlining, used
only to pick the pointer on the cold path).  Everything else in `load` is
untouched: copies, pending records, waits, fences, commits, chunk table.

**Host.**  `mixed_page_prefill_run_args` no longer inspects the FP8 global
scales (the `.item()` sync per call is gone); the comment records C9.

**Tests.**  `_extreme_transport(shape, dtype, dev, mode, gs)` sets both FP8
global scales to `gs` and recomputes the reference as `A16(x * A16(float(s) *
g))`; `_run_extremes(mode, q_len, gs)` runs for `gs` in {1, 0.5, 1.1 x 2^-118}
x {fp8, mixed} x {q 1, 64} = 12 cases (was 4); the matrix is 64 + 2 + 12 = 78.

**Expected artifacts (section 5, restated for what was written).**  fp8 module
producer region: `F2FP.E4M3` ~30 (from 270), `PRMT` 9 per block, `IMAD` sign-fix
0, `VOTE.ALL` one per block site, no `BSSY/BSYNC` in the pair body, `STACK 0`,
`LDGSTS 36+36`, `BAR.SYNC 4`, `FENCE.VIEW.ASYNC 3`; a16 module byte-identical to
`5cc416fd` (no code path of the a16 module touches `expand_block`, `make_bases`'
FP8 arm or the removed helpers).  Bench fp8 440-460 / 450-470 us; 78/78
bit-exact.

**Deviation from revision 1.**  The E2M1 placement variant is not built (LUT
kept); fp4 is unchanged by this commit.

---

## As written: F24b (second producer warp group + C11 reductions)

Files: `include/flashinfer/attention/hopper/{kernel_traits,named_barrier,
prefill_sm90,epilogue,sparse_mainloop,sparse_mixed_mainloop}.cuh`,
`tests/attention/run_fa3_mixed_page_transport.py`.  `mainloop_mma.cuh`,
`variants.cuh`, `mixed_page_prefill.py` untouched.  Not built or run here.

**Traits and thread counts (data flow of the constants).**
`MixedAttentionKernelTraits` reads `kMixedStaticFormat` from the variant
(`mixed_variant_static_format`) and defines `NUM_PRODUCER_WGS` (1 for format 0,
2 otherwise), `NUM_WARPS = (CTA_Q/64 + NUM_PRODUCER_WGS) * 4`, `NUM_THREADS`,
`NUM_PRODUCER_THREADS = 128 * NUM_PRODUCER_WGS`, `PRODUCER_REGS = 72`,
`CONSUMER_REGS = 184 | 216`, and static-asserts the pool.  `named_barrier.cuh`
gains `producer_warp_groups_v<Ktraits>` (1 when the traits do not define
`NUM_PRODUCER_WGS`: stock and quantization traits), which `prefill_sm90.cuh:63`
(`NUM_COPY_THREADS`, hence `producer_arv_count` and the consumer thread index),
`epilogue.cuh:77` (`NUM_COPY_THREADS`, hence `NUM_MMA_THREADS` = 256 for the
barriers and the O store partition) and `WarpScheduler` (`kFirstConsumerWG`)
consume.  `prefill_sm90.cuh` role test `is_producer_wg` (`wg == 0` textually
for one WG), register split from `PRODUCER_REGS / CONSUMER_REGS` under
`if constexpr (kMixedTraits)`.  Ping-pong ids: `get_warp_group_barrier_idx<F>`
= `kWarpSchedulerWG1 + wg - F`, next WG `2F + 1 - wg`, `mma_init` test `> F`
(F = 1 folds to the stock text).  `sparse_mainloop.cuh:90`'s static_assert
(`NUM_PRODUCER_THREADS == 128`) is relaxed for mixed traits: the mixed mainloop
derives from it for its Q TMA / layout types only.  (This file was not in
revision 2's list; it was found because the assert is evaluated when the base
class is instantiated.)

**Mainloop.**  `NUM_COPY_THREADS = 128 * NUM_PRODUCER_WGS`, `OWNER_THREADS =
128`, `PAGES_PER_THREAD = 6 / NUM_PRODUCER_WGS`, `PAGE_STEP_BYTES =
NUM_PRODUCER_WGS * PAGE_REGION_BYTES`, `SCALE_PAGE_STEP_BYTES`; `own_u(t) = t &
127`, `own_h(t) = t >> 7` (both fold for one WG); `TileRegs::page(h, j)` = SEL
between `pages[2j]`, `pages[2j+1]`; `TileRegs::tag_of(h, j)` (dynamic module,
F24b only) picks the word at compile time and the byte by `8 (base + h)`.
`make_bases`: `page_off = h * PAGE_REGION_BYTES` folded into `out0 / out1 /
land8 / land4 / a16_dst`, `sc_rd = sc_base + h * 512 + r * 8 + 4 (k >> 2)`,
`out0 / out1` at chunks `2k + swap`, `2k + 1 - swap` with `swap = ((k >> 2) ^
r) & 1` (`out_swap(u)`).  `expand_bases`: `l8a = land8 + so + 8 swap`, `l8b =
l8a ^ 8`, `l4a / l4b` likewise (+4 / ^4).  `load_packed`: two `LDS.64` (FP8) /
two `LDS.32` (FP4) in store order.  `copy_compressed_page`: payload copy by
every lane; the row's 8 B scale copy (`cp8` / `cp8_zfill`) by `blk_blk(u) == 0`
only, destination `sc_rd` (the leader's word offset is 0), source
`compressed_src().scales` = the row's scale bytes (the `+ 4 (blk / 4)` is gone).
`issue_tile_copies` / `expand_operand` iterate `j < PAGES_PER_THREAD` with
immediates `j * PAGE_STEP_BYTES` / `j * SCALE_PAGE_STEP_BYTES`; the static
expansion handles the odd count 3 (`if (j + 1 < N)` on constants); the dynamic
expansion's tag is `(tv >> 8 (h + 2 j)) & 0xFF`.  `load()`: `gather_thread =
NUM_PRODUCER_WGS == 1 || t < 128` guards `chunk_load` / `chunk_store` (the
group barrier is met by all 256); `q_issuer = t < 32` for two WGs
(`warp_idx_in_warpgroup == 0` textually for one); `__syncwarp()` after
`cp_async_wait` in `finish_pending_pair` (the A8 ordering for the scale slot);
host check: scale rows 8 B aligned.  Control flags `MIXED_FA3_CONTROL_SKIP_EXPAND`
and `MIXED_FA3_CONTROL_RAW_STS` in `expand_block` (off by default).

**Deviation from revision 2: the scale slot size is not shrunk.**
`kMixedScaleStageBytes` stays `6 x 512` B per stage per operand and the rows
use the first 128 B of each page slot (`SCALE_ROW_BYTES = 8`, `SCALE_PAGE_BYTES
= 512`).  Shrinking it would move `mixed_meta` in the shared storage and change
the chunk-table immediates of the a16 module, which must stay byte-identical;
smem is therefore unchanged (~199.7 KB), not 13.5 KB smaller.  The
wavefront arithmetic of 2C is unaffected (the LDS.32 of 32 lanes still reads
32 contiguous bytes with 4-lane broadcast).

**Tests.**  `_run_parity_tail(mode, q_len)`: `kv_len = 149` (`96 + 3 x 16 + 5`)
for fp8, fp4, mixed at q 1 and 64 — the last tile's partial page 3 is owned by
the odd-parity warp group; matrix 64 + 2 + 6 + 12 = 84.

**Expected artifacts.**  fp8 / fp4 / dynamic modules: `USETMAXREG.DEALLOC
0x48` + `TRY_ALLOC 0xB8`, `ptxas -v` without C7507, STACK 0; per thread per
call `LDGSTS` 6 payload + 6 predicated scale, `STS.128` 12 (all `[R+imm]`),
`LDS.64` 12 + `LDS.32` 6 + chunk table 4 `LDS.128`; one `UTMALDG` under the
warp-0 predicate; `BAR.SYNC` counts 256 (kProducerWG) and 512 (kQueryEmpty);
one `WARPSYNC`/`BAR.WARP.SYNC` more per call than [23]; a16 module byte-identical
to `5cc416fd`; stock paged kernel byte-identical.  ncu: `STS.128` 4.0
wavefronts, scale `LDGSTS` <= 4, `op_st` <= ~450, total <= ~2050 per call.
Bench fp8 290-345, fp4 295-350 (section 3); trace `acq` >= 0.25 us.
