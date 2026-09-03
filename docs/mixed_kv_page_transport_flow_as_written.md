# Mixed KV page transport: required flow vs. the kernel as written

Companion to [`mixed_kv_page_transport_dataflow.md`](mixed_kv_page_transport_dataflow.md)
(the binding specification). That document states the requirements; this one lays the
requirement and the implementation (`include/flashinfer/attention/hopper/sparse_mixed_mainloop.cuh`)
side by side in one notation so the two can be diffed by reading.

Notation: `→` data moves; `⊢` a wait; `[async]` the TMA/async proxy does the work and no
thread is occupied.

## Part 1 — What the kernel needs to be

### Actors

One CTA per (Q tile of 128 rows, KV head). Three warpgroups:

- **P** — producer WG (128 threads, 104 registers): owns all KV movement and expansion.
- **C0, C1** — consumer WGs (256 threads, 200 registers): QKᵀ → online softmax → PV over the
  128 Q rows. **Unchanged stock FA3.** They see only "a 96×128 bf16 K tile and a 96×128 bf16
  V tile in a stage of shared memory, in the SW128 layout wgmma wants."

Everything about mixed formats lives inside P. The consumer's contract is *a bf16 tile in
smem*; P's job is to satisfy that contract for the same wall-clock cost regardless of the
format the pages were stored in.

### Data flow — required

```
HBM                                                 SMEM (stage s of 3)                    Tensor cores
────────────────────────────────────────            ──────────────────────────────         ───────────
page p (16 tokens), tag ∈ {A16, FP8, FP4}

 A16 page: 16 × 128 bf16 = 4 KB    ── TMA [async] ──▶  K[s] rows 16i..16i+15, both      ─▶ wgmma reads
                                                       D-blocks (2 × 2 KB, SW128)           K[s] directly
 FP8 page: 16 × 128 B packed = 2 KB ── TMA [async] ──▶  K[s] rows 16i.., D-block 1 only
           + 16 × 8 scales (E4M3)   ── cp.async ─────▶  scales[s][tok][8]
                                                            │  P thread t (t = token):
                                                            │  read 128 B packed row,
                                                            │  scale × cvt → 256 B bf16
                                                            ▼
                                                       K[s] rows 16i.., both D-blocks   ─▶ wgmma
 FP4 page: 16 × 64 B packed = 1 KB  ── TMA [async] ──▶  K[s] rows 16i.., D-block 1, low half
           + scales                 ── cp.async ─────▶  scales[s]
                                                            │  same, 64 B → 256 B
                                                            ▼
                                                       K[s] full                        ─▶ wgmma
 (V identical, into V[s])
```

Invariants on this flow:

- **D1 (landing layout).** A page's 16 rows of one 64-element D-block are one contiguous,
  1024 B-aligned 2 KB region in CuTe's `tile_to_shape(SW128 atom)` layout, so a TMA
  `SWIZZLE_128B` box lands exactly where wgmma will read it. A16 pages therefore cost
  **zero** producer instructions per byte.
- **D2 (packed rows live in the tail of their own destination).** Packed FP8/FP4 rows land
  in D-block 1 of the page region they expand into. Expansion writes D-block 0 (never
  overlaps the packed row) and then D-block 1 (each chunk read before it is overwritten).
  No staging buffer, no extra smem, no copy.
- **D3 (ownership).** Thread `t` of P owns token `t` of the tile (96 ≤ 128). Expansion has no
  cross-thread hazard and needs **no barrier** inside it.
- **D4 (tail).** Rows past `kv_len` in the last tile are zero-filled by their owners: V rows
  beyond the end are multiplied by P = 0 and stale packed bytes decode as Inf/NaN.
- **D5 (proxy order).** Generic-proxy writes (expansion) are followed by `fence.proxy.async`
  before the stage is handed to wgmma and before the stage is later refilled by TMA.

### Control flow — required

The producer loop has the *shape of a stock TMA pipeline*, with expansion folded in only
when a tile actually has compressed pages:

```
P, per tile pair (K(t−1), V(t)):

  1  acquire  K stage, V stage                 ⊢ empty barriers (consumer released them)
  2  group barrier (P only)                    — publishes page metadata gathered last iteration
  3  gather page ids/tags for the pair AFTER NEXT   ← the only dependent global loads;
     into a 3-slot smem ring (one pair ahead)        issued now, consumed two steps later
  4  for each operand (K then V):
       if tile is all-A16 and full:
            expect_tx(full barrier, 24 KB); issue 12 TMA boxes            [async]
            producer_commit                     — P never waits; TMA completes the barrier
       else:
            arrive_and_expect_tx(private barrier, Σ box bytes); issue TMA [async]
            cp.async scales
  5  if any expansion needed:
       cp.async wait; ⊢ private barrier (both operands: ONE wait for the pair)
       expand K rows (thread t → token t), zero-fill tail
       fence.proxy.async; producer_commit(K)
       expand V; fence; producer_commit(V)
  6  next pair
```

Properties:

- **C1 — one dependent round trip per pair**, for the pair *two* ahead, so its latency
  overlaps two full tile times.
- **C2 — no runtime-indexed register arrays** (`arr[stage]` with dynamic `stage` lands in
  local memory).
- **C3 — register budgets from live sets**: producer ≤ 104, consumer 200
  (104·128 + 200·256 = 64 512 ≤ 65 536).
- **C4 — barrier accounting**: the full barrier has 128 producer arrivals; the A16 path adds
  `expect_tx` (no arrive) before its own arrive, so TMA bytes + 128 arrives complete the
  phase; the compressed path arrives once on a *private* count-1 barrier.
- **C5 — consumer unchanged.**

### Stall/bandwidth condition

With S = 3 stages and metadata one pair ahead the pipeline does not stall if
`L_tma + T_expand + T_issue ≤ (S−1) · T_consume`. For A16 tiles `T_expand = 0` and this is
the stock FA3 condition. For FP4, `T_expand` is ≈ 96 threads × (4 chunks → 8 × ~12
instructions) — a few hundred producer instructions per tile against ~2 000 consumer
instructions per tile. Bytes per tile: 24 KB (A16) / 12 KB + 0.75 KB (FP8) / 6 KB + 0.75 KB
(FP4): the same 96 tokens per tile for a quarter of the HBM traffic.

## Part 2 — The kernel as written

### Data flow — as written

```
Host (to_underlying_arguments):
  6 CUtensorMaps: a16K/a16V (bf16, box {64 elems,1 head,16 tok}, SWIZZLE_128B)
                  fp8K/fp8V (u8,  box {128 B,1,16},   SWIZZLE_128B)
                  fp4K/fp4V (u8,  box {64 B,1,16},    SWIZZLE_64B)
  KVPageTransport: page_format[] (u8 tag per page), scale pointers + byte strides
  asserts: page_size == 16; V strides == K strides; static_format ⇒ its payload non-null

SMEM (SharedStorageQKVOMixed), per stage s ∈ {0,1,2}:
  smem_k[s], smem_v[s]      96×128 bf16, tile_to_shape(SW128 atom)   ← static_assert: 1024-aligned,
                                                                       (rows/8 × D/64) atoms per stage
  mixed_scales_k/v[s][96][8] u8                                       ← cp.async destination
  mixed_tma_bar_k/v[s]      ClusterTransactionBarrier, count 1        ← compressed-tile landing
  mixed_phase_k/v           ONE uint32 each: bit s = parity of bar s  ← persisted across work items
  mixed_meta_pages[3][6], mixed_meta_formats[3][6], mixed_meta_valid[3]  ← 3-slot metadata ring

Device, per operand X ∈ {K,V}, tile t, stage s = state.index():
  gather_tile_meta(t → slot t%3): lanes 0..5 load kv_indices[...] and page_format[page]
                                   (0xFF if the page id is invalid); thread 0 writes valid_tokens
  issue_tile_tma:  for page i in 0..5:
      A16 : tma_load_page(a16X, bar, &sX(16i, 0,  s), crd0=0,  head, page)   2 KB
            tma_load_page(a16X, bar, &sX(16i, 64, s), crd0=64, head, page)   2 KB
      FP8 : tma_load_page(fp8X, bar, &sX(16i, 64, s), 0, head, page)         2 KB → D-block 1
      FP4 : tma_load_page(fp4X, bar, &sX(16i, 64, s), 0, head, page)         1 KB → D-block 1, low half
      bad/beyond : nothing
  issue_tile_scales: cp_async_zfill<8>(scales[s][tok], transport.scales + page/tok/head strides)
  expand_token (thread t = token t, page i = t/16, r = t%16):
      A16    : if t ≥ valid_tokens → zero 256 B (D4); else nothing
      FP8    : for d in {0,1}: read 4 × 16 B chunks at ((4d+c) ^ (r%8))·16 of the packed row,
               E4M3×4 scales → bf16×2 pairs, cvt.rn.f16x2.e4m3x2 · scale → write row d·64..d·64+63
               (D-block 0 written first; D-block 1 chunks read before overwritten — D2)
      FP4    : read 4 × 16 B chunks at (c ^ ((r/2)%4))·16, e2m1x8→bf16x8 LUT, scale, write 128 elems
      0xFF   : zero row
  wgmma consumes sX[s] unchanged (C5)
```

### Control flow — as written

```
load():
  gather(t_last → slot), gather(t_last−1 → slot)          — initial two tiles
  K(t_last): acquire; group_barrier; issue_operand(K); if needs_expand: cp.async wait; finish(K)
  Q: TMA + barrier_O wait (stock)
  for t = t_last down to swa_begin+1:  produce_pair(tK = t−1, tV = t)
  prefetch_next_work; produce_pair(tK = −1, tV = swa_begin)
  group_barrier; thread 0 writes the phase words back to smem  — trailing barrier
  broadcast_next_work

produce_pair(tK, tV):
  if tK ≥ 0: pipeline_k.producer_acquire(state_k)         ⊢ empty_k
  pipeline_v.producer_acquire(state_v)                    ⊢ empty_v
  group_barrier()                                         — barrier A: publishes slots tK%3, tV%3;
                                                            every reader of slot (tK−1)%3 is past it
  gather(tK − 1)                                          — one pair ahead into the freed slot (C1)
  issue_operand(K, tK), issue_operand(V, tV):
      needs_expand = tile_has_compressed(slot) || valid_tokens < 96
      if !needs_expand: thread 0: expect_transaction(full[s], 24 576); issue_tile_tma → full[s]
                        (no wait — commit happens in finish)
      else:             thread 0: arrive_and_expect_tx(tma_bar[s], Σ bytes); issue_tile_tma → tma_bar[s]
                        all: issue_tile_scales (cp.async)
  if K.needs_expand || V.needs_expand:
      cp_async_wait<0>(); group_barrier()                 — barrier B: scales visible to all
  finish_operand(K); finish_operand(V):
      if needs_expand: tma_bar[s].wait((phase_bits >> s) & 1); phase_bits ^= 1 << s   (C2: bitmask)
                       expand_token(...); fence_view_async_shared()                   (D3, D5)
      producer_commit(state); ++state                                                 (C4: 1 arrive/thread)
```

### Requirement vs. implementation

| Requirement | As written | Verified by |
|---|---|---|
| D1 landing layout | TMA boxes → `sX(16i, 64d, s)`; static_asserts on atom count and 1024 B alignment | reading (offsets hand-verified in review) + compiler |
| D2 in-place packed rows | packed → D-block 1; D-block 0 written first; chunks read per D-block before overwrite | reading |
| D3 no barrier inside expansion | thread t ↔ token t; barriers A and B both precede expansion | reading |
| D4 tail zero-fill | `t ≥ valid_tokens → zero`, from `kv_len` on the last tile only | reading; bit-exact matrix |
| D5 proxy fence | `fence_view_async_shared()` after expansion, before commit | reading |
| C1 one round trip, one pair ahead | `gather(tK−1)` after barrier A | reading (moved across the barrier to close a WAR race) |
| C2 no runtime-indexed register arrays | phase bitmask; metadata in smem | `cuobjdump -res-usage`: `STACK:0`; SASS has no `LDL`/`STL` |
| C3 budgets | 104/200; FP8 payload live set 16 registers | `cuobjdump -res-usage`: `REG:168`, `STACK:0`; SASS has `USETMAXREG` 104/200 |
| C4 barrier accounting | `expect_transaction` (no arrive) on the A16 path; private count-1 barrier on compressed | reading (traced through `barrier.h`) |
| C5 consumer unchanged | `mainloop_mma.cuh` untouched | `git diff` |
| A16-only tiles cost no producer wait | `expect_transaction` + TMA + commit; no wait | `transport_a16 ≈ stock` in the benchmark |
| Stall condition with S = 3 | S = 3, metadata one pair ahead | benchmark |

Two deliberate boundary deviations from the steady-state picture in Part 1: (a) K(t_last) is
produced alone before Q (stock order, so the consumer can start QKᵀ on the first tile while
Q lands); (b) when *either* operand of a pair needs expansion both wait at barrier B — the
pair is the unit, which is what makes it one wait per pair rather than two.

## What static inspection can and cannot confirm

See [`mixed_kv_page_transport_cutlass_references.md` §7](mixed_kv_page_transport_cutlass_references.md)
for measurement discipline. For the compiled object (`*_paged_sm90_kernel_mask_*.cuda.o`):

| Question | Tool | Answer form |
|---|---|---|
| C2/C3: any local memory? | `cuobjdump -res-usage` | `STACK:0 LOCAL:0`; the stock kernel is the baseline (also 0) |
| C2: which code path spills? | `cuobjdump -sass` \| `grep -c 'LDL\|STL'` | 0 for the mixed kernel; nonzero pinpoints the function by surrounding labels |
| C3: register split really applied? | `cuobjdump -sass` \| `grep USETMAXREG` | `USETMAXREG.DEALLOC.SYNC 104` and `USETMAXREG.ALLOC.SYNC 200` |
| A16 path issues TMA, not ld/st? | `cuobjdump -sass` \| `grep -c UTMALDG` | ≥ 6 (per-format boxes, per operand) |
| mbarrier protocol emitted as expected? | `grep -c 'SYNCS.ARRIVE.TRANS64\|SYNCS.PHASECHK'` | arrive/expect-tx and try_wait sites present |
| Proxy fence present? | `grep -c FENCE.VIEW.ASYNC` | ≥ 1 |
| Spill *counts* (not just presence)? | `ptxas -v` (`FLASHINFER_EXTRA_CUDAFLAGS="-Xptxas -v"`) | `N bytes spill stores, M bytes spill loads` per kernel |
| Actual stall reasons / issue utilisation? | Nsight Compute (`ncu`), not cuobjdump | `smsp__average_warps_issue_stalled_*`, `sm__issue_active`, local-memory bytes |

`cuobjdump` answers the *structural* questions (local memory, register split, which
instruction classes exist). It cannot say whether the pipeline stalls; that is the
benchmark's and `ncu`'s job, and it is a confirmation, not a tuning loop.
