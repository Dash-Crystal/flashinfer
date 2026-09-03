# Mixed KV page transport: one backend per architecture

This document is normative for `csrc/xqa/` and `include/flashinfer/attention/` on the
mixed-KV-page branch. It exists because of the following review finding, quoted exactly:

> **sm90 ≠ sm120, and it keeps trying to unify them.** On sm120 `mma.sync` takes
> register operands and `cvt.rn.bf16x2.e2m1x2` is native → in-fragment decode is
> correct and *already works* (79.6 vs 120 µs). On sm90 `wgmma` reads smem for free
> and FP4 decode is software → expand-to-A16-smem in dedicated converter warps and
> leave the baseline SS-GMMA consumer byte-for-byte untouched. Its current RS-GMMA
> sm90 path puts ~14 instr/lane of shuffle+cvt+FMA in front of each ~8-cycle
> `m64n16k16` wgmma — the decode is 2–3× longer than the MMA it feeds, and the FP4
> arm of `loadMixedKTileFragment` **still reads smem byte-by-byte with the scalar x2
> cvt** despite the "official x8 substitution" claim (the x8 helper exists in
> `mhaUtils.cuh`; this path doesn't call it).

## Design specification

The producer/consumer data flow, its stall-freedom conditions, and the
invariants an implementation must satisfy are fixed in
`mixed_kv_page_transport_dataflow.md`. Kernels are written once from that
specification and reviewed by reading against it; measurement confirms
conformance, it does not tune.

## Rule

Every CUDA compute capability that differs in any instruction used by the attention
prologue, mainloop, or epilogue is a **separate backend**. sm80, sm86, sm89, sm90,
sm100, sm103, sm120, and any later revision are distinct where their ISA, tensor-core
operand sourcing, copy engines, or conversion instructions differ. A kernel is shared
between two architectures only when the *identical* source compiles to the intended
instruction sequence on both; that is a property to be checked, not assumed.

Concretely, for this branch:

| Concern | sm90 (Hopper) | sm120 (Blackwell consumer) |
| --- | --- | --- |
| MMA operand source | `wgmma` A from **smem** (SS) or registers (RS); B from smem | `mma.sync` A and B from **registers** |
| Bulk copy | TMA tensor maps (`cp.async.bulk.tensor`) — one elected lane per copy | `cp.async.bulk` (non-tensor) — one elected lane per copy |
| E4M3 → A16 | `cvt` via f16 pair | native packed `cvt.rn.bf16x2.e4m3x2` |
| E2M1 → A16 | **software** (`prmt` LUT, x8 per 32-bit word) | native `cvt.rn.bf16x2.e2m1x2` |
| Correct place to dequantize | dedicated converter warps → A16 smem stage → unchanged SS-GMMA consumer | inside the consumer's register-fragment load (the path that already wins) |
| Source file | `include/flashinfer/attention/hopper/sparse_mixed_mainloop.cuh` (FA3 host, all query shapes); `csrc/xqa/mha_sm90.cu` (reference) | `csrc/xqa/mha.cu` |
| Bandwidth-side reference in tree | `3rdparty/cutlass/.../sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp` | `include/flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh` |

The two right-hand columns are not converging. Do not port the sm120 register-fragment
decode into the sm90 GMMA consumer; do not port sm90 smem-expansion into sm120.
Shared code between the two is limited to: page-format metadata gather
(`gatherMixedPageFormats`), scale arithmetic (`scaleA16x2`), the conversion helpers in
`mhaUtils.cuh` that are themselves `__CUDA_ARCH__`-guarded, and the host API.

## Dispatch must be explicit and asserted

`flashinfer/jit/xqa.py` chooses which source enters the module. Which kernel executes
for a given (architecture, page format) pair is a fact the test suite must assert, not
infer from timings. Requirements:

1. Every JIT module exports the architecture it was compiled for and the kernel family
   (`mha_sm90` GMMA vs. architecture-neutral `mha`) it will launch; tests read this and
   fail if a measurement is attributed to the wrong kernel.
2. Excluding `mha_sm90.cu` for a format (as `jit/xqa.py` currently does for
   `block_scaled_fp8`) is a **dispatch decision** that must be visible in the module
   URI and in benchmark output. Silent fallback to `mha.cu` on an sm90 device is a
   failure, not a fallback.
3. A performance number for "sm90 FP8/FP4" is only admissible when the asserted kernel
   family is `mha_sm90`.

## What this rules out

- Any change to `mha_sm90.cu` motivated by an sm120 measurement, or vice versa.
- Any single "unified" mixed-input consumer that must be correct on both architectures.
- Any topology argument ("N load warps per operand", "converter warps in the consumer")
  that is not stated per architecture with the instruction it is balancing against.

## What this does not rule out

- Identical **host** APIs, page layouts, scale layouts, and `page_format` semantics
  across architectures. The storage contract in `include/flashinfer/attention/page_transport.cuh`
  is architecture-independent by design.
- Identical **acceptance tests**: bit-exactness against explicit A16 expansion, and the
  roofline targets in `mixed_kv_page_transport_cutlass_references.md`, apply to both.

## sm120 (RTX 5090) state: mixed streams through the XQA host

Measured on an RTX 5090 (B=17, S=4096, 8 KV heads, GQA 4; bursts; 34/34 cases
bit-exact against explicit A16 expansion):

| q per sequence | A16 stock | A16 transport | FP8 | FP4 | mixed |
|---|---|---|---|---|---|
| 1 (batched AR decode) | 179.9 µs | 179.8 | 139.8 | 83.7 | **145.3** |
| 4 (draft verification) | 183.5 | 184.6 | 153.7 | 100.6 | **156.9** |
| 64 (prefill; XQA is not the host for this) | 437.8 | 460.4 | 785.0 | 689.3 | 905.4 |

Two structural changes got the dynamically mixed stream from 1.3× *slower* than
A16 (237 µs) to 1.24× faster:

1. **Page tags are gathered one tile ahead** with a two-deep page-index prefetch
   (`loadPages` rotates `pageIdx ← pageIdxNext`, issues the tag load for the
   landed indices, and requests the indices two tiles ahead; the tag is
   broadcast with shuffles a tile later).  Before, the loading warp — which on
   sm120 is the compute warp — waited on a dependent `page_format[pageIdx]` load
   at every tile part.  Alone this did not change the mixed time (237 → 233 µs);
   it removed a latency chain, not the bottleneck.
2. **Tiles with compressed pages are expanded to A16 in shared memory and fed to
   the stock A16 GEMM** (`MIXED_COMPACT_PAGES=0`, now the default).  The
   register-side "compact" form dispatched on the page tag per (block, page)
   inside the unrolled MMA loop, instantiating all three fragment converters:
   `cuobjdump` showed 17 688 instructions, 647 branches and 338 local loads for
   the dynamic kernel against 6.5–8.3 K instructions, 130–190 branches and
   118–168 local loads for the static-format kernels.  With expansion the dynamic
   kernel is 12 064 instructions / 262 branches, and the mixed stream moves
   172 MB at 1.19 TB/s.  Pure FP8 at q=4 pays 13 % for this (136 → 154 µs);
   mixed streams are the product.

Open on sm120: no prefill-shaped host exists for mixed pages (the FA3 host is
sm90-only); q ≥ 64 through XQA is 1.6–2× slower than A16 and should route to the
sm12x prefill backend once one carries the page-transport contract.

## sm90 (H200) XQA decode host: where the compressed-format time goes

Measured on H200 (B=17, S=4096, 8 KV heads, GQA 4, q=1; bursts; 34/34 bit-exact).
Byte roofline at 4.8 TB/s: A16 59 µs, FP8 32, FP4 17, mixed 36.

### Attribution with the kernel's built-in experiment bits

| mode | production | converters skipped | no compressed TMA | no TMA, no scales |
|---|---|---|---|---|
| A16 transport | 83.7 | 83.6 | 83.8 | 83.7 |
| FP8 | 90.6 | 65.2 | 80.9 | 75.0 |
| FP4 | 94.9 | 61.8 | 86.0 | 80.6 |
| mixed | 108.5 | 73.8 | (n/a) | (n/a) |

Read as a stack for FP4: ≈47 µs is the consumer chain itself (gemm0 → softmax →
P hand-off → gemm1 at 2 CTAs/SM; 45 % of PC samples are barrier spins), ≈15 µs
exposed compressed loads, ≈33 µs exposed conversion.  Mixed-dtype-ness adds
nothing beyond its share of conversion.  E2M1/E4M3 arithmetic is ~10 % of samples.

### Per-tile timeline (MIXED_KV_TRACE, CTA 0, steady state)

| | FP4 | A16 |
|---|---|---|
| consumer cadence per 64-token tile | 1.95 µs | 1.6 µs (bandwidth-bound: 5–7 µs landing) |
| loader TMA issue per tile (8 boxes) | 1.0–1.7 µs | 1.0 µs |
| TMA issue → bytes landed | ~1.7 µs | 5–7 µs |
| converter ready → done | 1.0–1.6 µs | 0.1 µs |

With two K stages the compressed chain per stage is issue + landing + conversion
≈ 4.3 µs → one tile every ~2 µs: pipeline-depth-bound, not ALU- or bandwidth-bound.

### Structural changes made, and what each measured

1. **Converter warps copy compressed pages themselves** (cp.async, one warp per
   page, warp-contiguous chunk ownership, scales included); the loader's TMA is
   A16-only.  Loader issue per tile fell 1.0–1.7 → 0.2 µs.  cp.async landing
   under the saturated memory system measured **~1.9 µs**, so with two stages the
   cadence became landing + expansion = 2.8 µs — worse.  Landing latency at
   3.4 TB/s of aggregate traffic is a fact to cover with depth, not to remove.
2. **Copies two tiles ahead → three K and V stages.**  +32 KB of shared memory
   (84 → 114 KB) cost the second CTA per SM (limit 112.5 KB), which cost more
   than it gained; with the P ring 4 → 2 (110.6 KB) two CTAs fit again — but
   only after the register budget was fixed (below).  Converter period is then
   ≈ (landing + expansion) / 2 ≈ 1.5 µs, at gemm0's floor.
3. **Register budget.** The kernel has no `setmaxnreg`; every warp lives with the
   launch allocation, so `__launch_bounds__` minBlocksPerSM *is* the register
   budget.  The mixed build declared 1 (and `OPTIMIZE_FOR_LATENCY=1` drops the
   floor entirely), so the allocation floated to 58 registers/thread and two
   640-thread CTAs (≤ 51) no longer fit; the driver then also chose a one-block
   shared-memory carveout.  Declaring `(640, 2)` unconditionally gives 48
   registers and both occupancy limits = 2.  The SPEC_DEC (q > 1) variant wants
   226 registers statically and spills heavily (`LDL` hot in PC sampling; mixed
   q=4 935 µs vs FP4 276) — a separate item.
4. **Converter hazard barrier**: the read-before-write hazard of in-place
   expansion is intra-warp (a warp covers one page's (token, head-part) lanes),
   so the 128-thread named barrier per tile per operand is a `__syncwarp`.

Shared-memory budget (2 CTAs/SM ⇒ ≤ 112.5 KB): K/V stage 16 KB each, P entry
4 KB, Q 8 KB, scales ring 1 KB/entry.  (K3, V3, X2) and (K3, V2, X4) both fit;
(K3, V3, X4) does not.  The P ring was sized 4 because at 2 the two consumer
groups run in lock step; which trade wins is measured, not assumed.
