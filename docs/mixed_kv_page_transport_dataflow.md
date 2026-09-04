# Mixed KV page transport: data flow and control flow specification

This document fixes the design of the ragged-format streaming dequantization
prologue by construction, from CUTLASS's documented warp-specialized pipeline
model, so that its bandwidth and stall properties follow from structural
invariants rather than from measurement. Every invariant below is checkable by
reading the code (or a `static_assert`); none requires a benchmark. Measurements
are used only to confirm that an implementation obeys the specification.

Paths are relative to the repository root; CUTLASS is pinned at
`3rdparty/cutlass` (`b46b16d`).

## 1. Requirements

- **R1 (settled, frozen).** Every 16-token KV page carries a one-byte format tag
  chosen by the page sealer from the page's outlier statistics:
  `kA16`, `kBlockScaledFP8` (E4M3, one E4M3 scale per 16 coefficients),
  `kBlockScaledFP4` (E2M1, same scaling), extensible to further tags.
  `include/flashinfer/attention/page_transport.cuh`,
  `csrc/fp4_kv_quantization.cu`.
- **R2.** Attention over such a cache streams pages of any mix of tags into a
  tile mainloop, expanding compressed pages to the A16 math type on the way in,
  for all four inference shapes — naive prefill, continued prefill, draft
  verification (q ≥ 2, causal-in-suffix), and batched auto-regressive decode
  (q = 1) — with one attention implementation that differs only in mask, and
  with wall-clock bounded by bytes moved rather than by the expansion.

## 2. The kernel model (CUTLASS)

Reference: `media/docs/cpp/pipeline.md`, `media/docs/cpp/efficient_gemm.md`
("warp specialization", "persistent kernels"),
`include/cutlass/pipeline/sm90_pipeline.hpp` (`PipelineAsync`,
`PipelineTmaAsync`), `include/cutlass/arch/barrier.h`
(`ClusterTransactionBarrier`, `NamedBarrier`, `fence_view_async_shared`),
`include/cute/arch/copy_sm90_tma.hpp` (`SM90_TMA_LOAD_4D`),
`include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp`
(the mixed-input collective: separate producer/consumer warp groups, narrow
operand converted between load and MMA, scale tensor with its own TMA
descriptor and transaction bytes).

The kernel is a CTA of one **producer warp group** (128 threads, register-
deallocated with `setmaxnreg`) and two **consumer warp groups** (MMA +
softmax, register-allocated), connected by two stage rings (K, V) of depth `S`
whose full/empty barriers are `PipelineAsync` mbarriers. A persistent tile
scheduler hands (q-tile, head, request) work items to both roles. This is the
FA3 kernel in `include/flashinfer/attention/hopper/prefill_sm90.cuh`; the mixed
transport replaces only its producer (`sparse_mixed_mainloop.cuh`).

## 3. Data flow

Per KV tile of `CTA_KV = 96` tokens (6 pages) of one KV head, for each of K and V:

```
 global                         producer WG (128 thr)                         stage smem (SW128 K-major, 96 x 128 A16)
 kv_indices[6] ---------------> lanes 0..5: page index, format tag  --------> meta[stage] (smem)
 page_format[6] --------------/
                                thread 0: expect_tx(bytes) ; 6..12 TMA boxes
 A16 page (2 x 16x64 elems) --- TMA (SW128) ---------------------------------> rows of the page, both D-blocks   (final)
 E4M3 page (16 x 128 B rows) -- TMA (SW128, byte-typed) ---------------------> rows of the page, D-block 1      (packed)
 E2M1 page (16 x  64 B rows) -- TMA (SW64,  byte-typed) ---------------------> rows of the page, D-block 1      (packed)
 block scales (8 B / token) --- cp.async by thread t for token t ------------> scales[stage][t]
                                all threads: wait(tx barrier), wait(cp.async), group barrier
                                thread t (t < 96): expand token t in place:
                                   read its packed row (4 or 8 x 16 B, swizzle-decoded),
                                   decode + block-scale, write 16 x 16 B A16 chunks
                                   (D-block 0 first; D-block 1 last, over the packed row)
                                fence.proxy.async ; producer_commit(stage)
 consumers: consumer_wait(stage) ; wgmma over the A16 stage ; consumer_release(stage)
```

Invariants:

- **D1 (landing layout).** TMA `SWIZZLE_128B` over 128 B rows is the same bit
  permutation as CuTe's `SW128` K-major atom (address bits [4,7) ^= bits
  [7,10)); the stage tensor is `tile_to_shape(SW128 atom, (CTA_KV, D, S))`, so a
  page's 16 rows of one 64-element D-block are one contiguous 2 KB region.
  A16 boxes therefore land in final position; no copy. `static_assert` the
  region size and 1024 B alignment (`kernel_traits.cuh`, `SharedStorage`).
- **D2 (packed rows).** A compressed page's whole-head rows (16 × 128 B E4M3,
  16 × 64 B E2M1) are exactly the size of one page-region of one D-block and are
  delivered swizzled (`SWIZZLE_128B`/`SWIZZLE_64B`) so that a warp reading one
  16 B chunk per token across 8 tokens hits 8 distinct bank groups.
- **D3 (ownership).** Token t of the tile is expanded by producer thread t, and
  every byte that thread reads (its packed row, its 8 B scale word) is read
  before any byte it writes; the only overlap between reads and writes is its
  own packed row, overwritten last. Hence **no cross-thread hazard exists and
  no barrier is needed inside the expansion.**
- **D4 (tail).** Tokens ≥ `kv_len` and page slots past the sequence are written
  as zeros by the same thread so a masked `P = 0` never multiplies stale shared
  memory (FA3's gather producer achieves this with `cp.async` zero-fill).
- **D5 (proxy order).** Generic-proxy writes (expansion, zero fill) are followed
  by `fence.proxy.async.shared::cta` before the stage is committed; TMA writes
  (async proxy) are ordered by the transaction barrier the producer waits on.

## 4. Control flow

Per stage, the producer warp group executes exactly this sequence; nothing
else waits inside it:

1. `producer_acquire(stage)` — waits for the consumers' release of tile t−S.
2. metadata gather (6 lanes, global loads) ; **group barrier A** (128 threads,
   `NamedBarriers::kProducerWG`).
3. thread 0: `arrive_and_expect_tx(bytes(stage))`, issue all TMA boxes;
   thread t: one `cp.async` for its scale word.
4. `cp.async.wait_all` ; **group barrier B** ; `wait(tx barrier[stage], phase)`.
5. expansion (thread-local, no barrier) ; `fence.proxy.async` ;
   `producer_commit(stage)`.

Invariants:

- **C1 (one dependent round trip).** The only DRAM latencies on the producer's
  critical path per tile are the metadata gather (indices → tags, two dependent
  loads) and the TMA/scale landing. They are not serialized with each other
  beyond that: TMA and scales are issued together and waited once.
- **C2 (no register-array indexing by runtime values).** Page indices, tags
  and the valid-token count live in shared memory (`mixed_pages_*`,
  `mixed_formats_*`, `mixed_valid_*`). A register array indexed by a
  thread-dependent value is placed in local memory by the compiler; the first
  implementation violated this (259 MB of local stores per call) and was fixed
  by construction, not by tuning.
- **C3 (register budgets from live sets).** Producer live set: 4 × 16 B words
  (one D-block half of a packed row), 4 scale words, addresses, loop state
  ≈ 60 registers → `setmaxnreg 104`; consumers ≈ 150 (O accumulator 64,
  S accumulator 48, P fragments, softmax state) → `setmaxnreg 200`;
  `104·128 + 200·256 = 64 512 ≤ 65 536`. Declared as
  `SparseMixedCollectiveMainloop::kProducerRegs/kConsumerRegs`, applied in the
  kernel. A spill is a specification violation, visible as `STACK > 0` in
  `cuobjdump -res-usage`.
- **C4 (barrier accounting).** Transaction barrier arrive count 1 (thread 0),
  transaction bytes = Σ over pages of the bytes each TMA box delivers
  (`tile_tx_bytes`), phase tracked per stage in shared memory across work
  items; `PipelineAsync` producer arrival count 128, consumer arrival count
  256 — unchanged from FA3's gather producer.
- **C5 (consumer unchanged).** The consumers, epilogue, scheduler, masks and
  sliding-window logic are FA3's; the mainloop's tile order (K last-tile first,
  then per tile K(t−1), V(t), finally V(0)) is preserved verbatim because
  `mma_f16` waits in that order.

## 5. Stall-freedom and bandwidth conditions

Let `T_c` be the consumers' time per tile, `L` the producer's issue→landing
latency (metadata gather + TMA), `T_x` the expansion time per tile, `T_i` the
producer's issue time per tile, and `S` the stage depth.

- **No consumer stall:** `L + T_x + T_i ≤ (S − 1)·T_c`. With `S = 2` this is
  `L + T_x + T_i ≤ T_c`.
- **No producer stall (throughput):** `T_i + T_x ≤ T_c` (the producer's own
  work fits in one consumer tile).
- **Cadence:** `T = max(T_c, T_i + T_x)`; when the consumer condition fails,
  `T = (L + T_x + T_i)/(S − 1)`.

Quantities for D = 128, `CTA_KV = 96`, H100/H200 at ~1.98 GHz:

| term | prefill (q ≥ 64) | decode-shaped (q = 1, padded to 128 rows) |
| --- | ---: | ---: |
| `T_c` (2 consumer WGs, QKᵀ+PV 128×96×128 ×2 ≈ 6.3 MFLOP, softmax) | ~1000–1200 cycles | ~800 cycles (tensor-bound on padding) |
| `T_x` E2M1 (96 threads × ~300 instr, 4 warps, ~1 IPC/warp) | ~350 cycles | ~350 |
| `T_x` E4M3 (~400 instr) | ~450 | ~450 |
| `T_i` (12 TMA issues + scales + 2 group barriers) | ~150 | ~150 |
| `L` (two dependent loads + TMA landing, loaded machine) | ~1500–2500 | same |

Consequences: `S = 2` does **not** satisfy the consumer condition for
decode-shaped work (`L + T_x + T_i ≈ 2000–3000 > T_c ≈ 800`); `S = 4` does
(`≤ 3·800`). Stage bytes at `S = 4`: K and V 4 × 24 KB each = 192 KB, plus Q
32 KB — over the 227 KB budget with Q; `S = 3` (144 + 32 + O/scales ≈ 180 KB)
satisfies `L + T_x + T_i ≤ 2·T_c = 1600` only at the low end of `L`. The
metadata gather can be taken off the critical path by resolving it one tile
ahead (its two loads then overlap the previous tile), leaving `L ≈ TMA landing
≈ 1000–1500`, which `S = 3` covers. **Specification: `S = 3`, metadata
gathered one tile ahead into a 2-entry shared ring.** For prefill, `S = 2`
already satisfies the condition and the stock FA3 choice stands; the traits
therefore select `S` by whether the work item is decode-shaped (q-tile
padding ratio), a host-side decision.

Bandwidth that follows: at cadence `T = T_c` (no stall), tokens/s per GPU =
`132 · 1.98e9 · 96 / T_c`; at `T_c = 800` that is 31 G tokens/s, i.e. for
E2M1 (128 B per token, K+V) 4.0 TB/s-equivalent of A16 traffic and ~33 µs for
the B=17, S=4096, Hkv=8 benchmark shape; for E4M3 (272 B/token) the same
token rate at 8.4 TB/s-equivalent — both above the A16 kernel's DRAM-bound
12.5 G tokens/s (3.2 TB/s over 256 B/token). Any implementation that meets
D1–D5 and C1–C5 with `S = 3` is bounded by these numbers; one that does not
meet them is out of specification regardless of its timing.

## 6. What this rules out

- Tuning stage depth, warp counts, or converter cuts by benchmark; the depth
  is derived in §5 and the cut in D3.
- Any producer design that waits on DRAM twice per tile in series (C1), holds
  per-tile metadata in registers indexed at runtime (C2), or leaves register
  budgets at FA3's defaults for a producer that converts (C3).
- Retuning the consumers: they are FA3's; the compressed path must not require
  changes to `mainloop_mma.cuh`.
- A second attention implementation per shape. The XQA `SWAP_AB` decode kernel
  (`csrc/xqa/mha_sm90.cu`) avoids the q-row padding FLOPs but pays a per-tile
  softmax/hand-off skeleton it cannot amortize (~2100 instructions per 64
  tokens, measured); at decode shape the padded FA3 consumer is tensor-bound
  at `T_c ≈ 800` for 96 tokens, which is ~2.5× the XQA host's token rate. The
  XQA mixed path is therefore retired to reference status once the FA3 path
  passes its matrix; R2's "one implementation differing only in mask" is met
  by the FA3 host.

## 7. Verification (confirmation, not tuning)

1. `cuobjdump -res-usage`: `STACK:0` for the mixed kernel (C2, C3).
2. `tests/attention/test_fa3_mixed_page_transport.py`: bit-exact against the
   same kernel with every page tagged A16 over the expanded cache, for
   q ∈ {1, 4, 64, 130} × {NHD, HND} × 8 page mixes (D1–D5).
3. ncu on `transport_a16` vs the stock FA3 paged kernel: same DRAM throughput
   within noise and zero local-memory bytes (the A16 path of the mixed producer
   is a pure TMA relanding of the stock gather).
4. ncu on E2M1: issue-active and stall mix consistent with `T_c`-bound
   operation; duration within 15% of §5's figure for the benchmark shape.

---

## Amendments (after the first two implementations)

These change the specification above; where they conflict, the amendment wins.

### A1. Transport is `cp.async` by the producer warp group, not per-page TMA

Measured on H200 with a per-pair `%globaltimer` trace of the producer: a TMA
operation costs ~100–200 ns of issue on sm90 regardless of box size (2 KB A16
page box ≈ 115 ns, 1 KB FP4 box ≈ 190 ns), so 24 boxes per pair from one thread
made the producer the critical path at ~2.3 µs of issue per pair — twice the
consumer's time — while `acquire` never waited and DRAM sat at 10 % of peak.
This is why stock FA3 uses `cp.async` (`USE_TMA_LOAD_KV=false`) for paged KV.

Consequently:

- **D1** is now purely CuTe's SW128 stage layout; nothing in the design depends
  on TMA swizzle equivalence.  The copies and the expansion address the stage
  through the same CuTe tensor, so no swizzle formula appears in the kernel.
- **C4** becomes the stock protocol: A16 tiles are committed with
  `cp.async.mbarrier.arrive` (the producer never waits); compressed tiles form one
  `cp.async` commit group per pair and are waited for with `cp.async.wait_group 1`
  *one pair later*, so their landing latency is covered by a full pair time.
  The producer therefore holds two pairs (pending + current) → **S ≥ 3**, and the
  page-metadata ring has **4 slots** (pending pair, current pair sharing one tile,
  one tile ahead).
- The six tensor maps, the private transaction barriers and their phase words
  are gone (`page_transport_tma.cuh` removed).

### A2. D6 — warp-contiguous copy ownership

Copy ownership is *not* per token.  Within a page, consecutive threads own
consecutive 16 B chunks of a row: a warp instruction covers whole contiguous rows
in global memory (full 32 B sectors, one request per 128 B) and a permutation of
one row's 16 slots in the swizzled stage (conflict-free).  A16: thread t owns
chunk t%16 of rows t/16 and t/16+8 (two copies per page); FP8: chunk t%8 of row
t/8; FP4: threads < 64, chunk t%4 of row t/4; threads < 16 copy the 8 B of
scales.  Per-token ownership (D3) remains the rule for the *expansion*, whose
accesses are all within the owner's row.  The first `cp.async` implementation
used per-token copy ownership and ran at 1.9× stock time from sector over-fetch
and bank conflicts alone.

### A3. §6 revised — hosts by query shape

The FA3 host is bound by its 128-row consumer skeleton at q=1 (stock 296 µs for a
stream the XQA host moves in 84 µs), so the XQA mixed path is **not** retired:
XQA hosts batched AR decode and short draft verification (q ≤ 4 in the current
dispatch); FA3 hosts continued prefill, naive prefill and long verification.
On sm120 only the XQA host exists; see `mixed_kv_page_transport_backends.md`.

### A4. Verification additions

- Every confirmation run is bounded (`timeout`) — a protocol bug shows up as a
  hang, and a hang must cost minutes, not an hour.
- A hang is diagnosed by reading the commit protocol, not by instrumenting: the
  first `cp.async` producer overwrote the pending tile's record when issuing the
  next pair (fix: two records, `staged` → `pending`, rotated after the finish).
- `cuobjdump -sass` counts: `UTMALDG` = Q only, `LDGSTS` > 0, `LDL`/`STL` = 0.
- sm90 XQA (`mha_sm90.cu`): `ptxas -v` on the TU must not print C7507
  (`'setmaxnreg' ignored to maintain minimum register requirements`) — one
  `.dec` below its role's need drops every setmaxnreg in the kernel silently,
  and `cuobjdump -res-usage` still reports the launch cap (REG 48).  The SASS
  must contain exactly two `USETMAXREG` (DEALLOC 0x28, TRY_ALLOC 0x38) and the
  role budgets must balance inside 640 x 48 = 30720.

### A5. C7 — the first K tile of a work item is committed before `barrier_O.wait`

The consumer arrives on `barrier_O` for `work_idx > 0` only after it has
received K(last) of the new item and issued the first QK GEMM
(`mainloop_mma.cuh`).  The producer waits on `barrier_O` before it may write
V (smem_v aliases smem_o).  A K(last) left *pending* across that wait therefore
deadlocks: consumer ← K(last) ← producer ← `barrier_O` ← consumer.  Stock never
hits this because its K(last) is committed at issue.  With deferred expansion
the K(last) tile is finished (waited, expanded, committed) immediately after
issue — one exposed copy latency per work item — and the conformance matrix must
contain a case with more work items than CTAs (persistent scheduler) so that a
CTA's *second* item begins with a compressed tile; the original 64-case matrix
(one item per CTA) could not see this.

Found by running the hung reproducer under `cuda-gdb`: all consumer warps on
`full_k`, all producer warps on `barrier_O`.  Symptom before diagnosis: the
trace build "worked" (timing) and the plain build hung only at benchmark scale.

### A6. C6 — producer issue budget

The producer warp group shares the SM's four schedulers with eight consumer
warps, so its executed instructions per tile must stay near stock's (~6 per
16 B copy: one address add, one predicate, one `LDGSTS`) for the consumer to
remain the bound.  Every earlier producer violated this differently — TMA
acceptance cost (~100–200 ns per box), per-copy 64-bit address arithmetic,
per-copy CuTe address evaluation, rolled-loop control — and none of the fixes
moved the total until the per-tile instruction count fell.  Copy ownership
constants (D6) make every source `thread_const + page_term` and every
destination `thread_const + i * PAGE_REGION_BYTES (+ ATOM_BYTES)`.  Checked with
the per-item `%globaltimer` trace (`iss` segment) and PC sampling of the
producer region, not by timing the kernel.

Code-shape rules that fell out of the same measurements: one inlined copy of
the pair routine (a lambda inlines at every call site); no runtime-selected
`Operand` reference (forces both structs into local memory); decode writes
`uint4` outputs per branch (an array assigned in two branches of a runtime
format switch is materialised in local memory); the expansion is one body with
the format as data (a fully unrolled block loop with a format branch inside is
unswitched into two bodies).

### A7. [23] Block-granular copy-owner expansion (amends D2, D3, A2/D6, C3)

The unit of ownership for compressed pages is one **scale block** (16
coefficients) of one row, for copy and expansion alike; the expansion is done by
the copy owner, so no thread waits on another warp and the pair body has no
group barrier.  Where this section conflicts with D2/D3/A2/C3 above, it wins.

- **D2 (packed rows) is replaced.** Thread `t` owns block `b = t%8` of row
  `r = t/8` of every page (eight consecutive lanes = one row).  Its packed bytes
  (FP8 16 B, FP4 8 B) land in the **row's D-block 1 line**: FP8 block `b` at
  physical chunk `b ^ (r&7)`, FP4 block `b` as the 8 B half `b&1` of chunk
  `(b/2 + 4*(r&1)) ^ (r&7)`; each thread also copies the 4 B word of the page's
  16 x 8 B block scales that holds its block into its own slot
  (`scales[stage][page][t]`, 512 B per page).  Rows past `kv_len` are copied with
  `cp.async` src-size 0 (payload and scale word zero-filled, D4), so the
  expansion has no tail case.
  *Measured rules behind the layout (ncu source counters, `L1 Wavefronts
  Shared` per instruction):* a `cp.async` lane octet coalesces into the ideal
  wavefront count only when its eight destinations lie in one 128 B smem line -
  the A16 copies and this landing run `LDGSTS.128` at 4.00 wavefronts per warp
  instruction; two earlier layouts that landed each block in one of its *own*
  output chunks (spread over the row's two D-block lines, in either lane order)
  ran at 31.9, one wavefront per lane, i.e. +1300 wavefronts per pair.  Sub-16 B
  `cp.async` is intrinsically expensive: 4 B scale words cost 7.95 wavefronts per
  warp instruction (ideal 1), FP4's 8 B blocks 3.98 (ideal 2).  `LDS.128`/`LDS.64`
  of the landings are at their ideal (the row's eight chunks are eight bank
  groups; FP4's odd rows use chunks 4..7 so 16-lane 64-bit phases are disjoint);
  the two `STS.128` of a block are 2-way conflicted (8.0 per instruction, ideal
  4: a row's output chunks `2b ^ (r&7)` are four groups twice, D-block 1 being
  12288 B = 0 mod 128 B away) - the cheaper side of the trade by an order of
  magnitude.
- **D3 (ownership).** Block `(r, b)` is expanded by the thread that copied it: it
  reads its landing chunk and its scale slot (both written by its own `cp.async`,
  so `cp.async.wait_group` on its own groups is the only wait; **barrier B is
  gone**), decodes, and writes chunks `2b`, `2b+1` of row `r` with `STS.128` at
  immediate offsets.  The landing chunk it reads is the output chunk of block
  `4 + b/2` of the same row, i.e. of a lane of its own warp: every lane's loads
  of a page precede any lane's stores of it in program order, made explicit with
  `__syncwarp()` before the stores (no CTA barrier).  One `fence.proxy.async`
  per thread per pair precedes the two commits.
- **A2/D6 (copy ownership).** FP8/FP4: block `t%8` of row `t/8`, 16 B / 8 B per
  thread per page plus its 4 B scale word; A16 ownership unchanged.  Compressed
  sources are 16 B (FP8) / 8 B (FP4) / 4 B (scales) aligned across page, token
  and head strides (host check).
- **C3 (register budget / code shape).** All expansion addresses are
  `[R + imm]` with 32-bit shared-window registers: per operand and stage two
  output bases (`chunk 2b`, `chunk 2b+1`; the second is base +-16 by row parity,
  so it is its own register), the landing base and the scale-slot base, each
  `thread_const + stage * bytes`, page `i` at `+ i * PAGE_REGION_BYTES`
  (`+ i * 512` for scales).  No generic `LD.E`/`ST.E`, no CuTe address evaluation
  in the pair body.  The static modules decode two pages per step (16 independent
  chains) with the next two pages' loads issued before the step's stores (the
  dynamic module pipelines one page ahead); the six scale words are loaded first.
  Decode: FP8 `cvt.rn.f16x2.e4m3x2`, then `a = h >> 3; a + 7 * (a & 0x10001000)`
  (the f16 pattern read as bf16 is the value x 2^-112 exactly, since an E4M3
  value has <= 3 significant mantissa bits; the sign moves from bit 12 to 15)
  with 2^112 folded into the block scale (`bf16_rn(f32(scale) * global * 2^112)`,
  exact while `block_scale * global < 2^16`; the sealer caps FP8 block scales at
  128, the host checks the global); one `HMUL2.BF16` rounds the exact product
  once, as the reference does.  FP4: a 20-instruction prmt LUT per 8 nibbles.
  Per FP8 block 49 instructions (LDS.128, LDS.32, 5 scale, 8 x {F2FP, SHF, LOP3,
  IMAD}, 8 HMUL2, 2 STS.128), per FP4 block ~57; 12 blocks per thread per pair.
  SASS (fp8 module, persistent kernel, producer region): 2687 instructions
  (from 3879), BAR.SYNC 4 (chunk table + item boundaries only), FENCE.VIEW.ASYNC
  3 (one per inlined finish site), LD.E/ST.E 0, STACK 0, LDGSTS 36+36, STS.128
  60 all `[R+imm]`.  The dynamic module keeps a 16 B stack (four chunk-table
  store addresses spilled across the item prologue's gather; 8 instructions per
  work item, none in the pair loop).
- **Result and re-stated requirement (P0.7 branch "fin > 1 us with counts as
  predicted").** Counts are as predicted (ncu producer `inst_executed` 3417
  warp-instructions per pair vs 3320 predicted; landing wait 0.02 us; fence +
  commits 0.06 us) but the expansion takes 0.57-0.63 (K) + 0.48-0.53 (V) us per
  pair on the trace build, not 0.5-0.65 for both: the producer's four warps issue
  at ~0.22 IPC each (stall mix in the producer region: wait 27 %, selected 23 %,
  dispatch 15 %, not_selected 10 %, short_sb 9 %, long_sb 9 %, math 3 %), so a
  pair costs ~2.6 us against the consumer's ~1.55 and FP8/FP4 land at 1.6-1.7x
  stock (see targets: FP8 474/483, FP4 507/517, mixed 880/907 us at q=1/64;
  ncu fp8 468 us, fp4 502 us).  The bound is neither the smem port (raw
  `l1tex__data_pipe_lsu_wavefronts_mem_shared` 61.5M = 2570 per pair incl. the
  2-way STS, ~49 % of the port at the measured cadence)
  nor the XU pipe (F2FP; `math` 3 %): it is dependent-issue latency on one
  producer warp per SMSP sharing issue with two consumer warps.  Parity therefore
  requires either ~2x the per-warp IPC of the expansion (more independent work
  per warp than the 72-register budget admits) or moving the decode off the
  producer warp group (the consumer warps issue at ~35 % and are starved; cf.
  Track W's [29], where the compute warps expand).  The dynamic (mixed) module
  is additionally bound by its copy-issue phase (2.0-2.2 us per pair of per-page
  format branches and three formats' address setup), a [21]/[22]-side cost.
