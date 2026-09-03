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
