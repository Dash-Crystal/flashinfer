/*
 * Copyright (c) 2026 by FlashInfer contributors.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Paged-KV producer for FA3 (Hopper) with ragged page formats.  Specification:
 * docs/mixed_kv_page_transport_dataflow.md (invariants D1-D6, C1-C7) and
 * docs/mixed_kv_page_transport_flow_as_written.md.
 *
 * Pages carry a one-byte tag: A16, block-scaled E4M3 or block-scaled E2M1
 * (one E4M3 scale per 16 coefficients).  Every KV byte moves the way the stock
 * FA3 paged producer moves it: the producer threads issue 16 B cp.async copies
 * straight into the consumer's SW128 K-major smem stage (a 16-token page is
 * too small a TMA box on sm90 - each TMA operation costs ~100-200 ns of issue
 * regardless of size, which made per-page TMA the critical path).
 *
 * Producer threads: one producer warp group (128 threads) for every module
 * ([25]; [24b]'s second warp group by page parity was measured and rejected:
 * +30 % instructions for +15 % throughput).  Thread t applies every ownership
 * rule below to all six pages of a tile, so each thread copies and expands six
 * pages per operand per tile; the compressed modules run the producer at 136
 * registers (consumer 184) so that a whole operand's decode is in flight per
 * warp.  A row's eight lanes stay in one warp.
 *
 *  * A16 pages: thread u copies chunk u%16 of rows u/16 and u/16+8 of its
 *    pages (D6) and the stage is committed with cp.async.mbarrier.arrive - the
 *    producer never waits.
 *  * Compressed pages ([23]): thread u owns scale block b = u%8 of row r = u/8 of
 *    its pages (eight consecutive lanes copy one row's global line).  It copies
 *    the block's packed bytes (FP8 16 B, FP4 8 B) into the row's D-block 1 line
 *    (chunk b ^ (r&7): one 128 B smem line per lane octet, which is what the
 *    cp.async path needs to coalesce); the row's lane b == 0 copies the row's
 *    8 B of block scales into the row's slot ([24b]).  The pair's copies form
 *    one cp.async commit group.  One pair later each thread waits for its own
 *    copies (cp.async.wait_group), the warp meets one __syncwarp (which orders
 *    lane 0's landed scale slot before the other lanes' read of it; no group
 *    barrier), decodes its blocks (FP8 bf16: bit placement x 2^-120 with 2^120
 *    folded into the block scale, exact fallback by a per-block warp vote
 *    ([24a], C9); FP4: prmt LUT) and stores them to chunks 2b + swap, 2b + 1 -
 *    swap (swap = ((b>>2) ^ r) & 1, [24b]: conflict-free STS.128) with STS.128
 *    at immediate offsets from per-stage 32-bit bases; one fence.proxy.async per
 *    pair, then the commits.  No byte is read by a thread that did not copy it,
 *    except the row's scale slot, read by the row's own warp after the barrier.
 *
 * Page metadata ([21]): a two-buffer chunk table of 16 tiles (kernel_traits.cuh
 * MixedTileMeta).  Threads 0..127 each load one (kv_index, tag) of the next
 * chunk at the first pair of the current chunk, store them at its ninth pair
 * and synchronise the group once; every pair then reads its two tiles' rows
 * (static modules: four LDS.128; dynamic module: the two rows' last words) and
 * the pending record carries what the expansion needs, so the pair body has no
 * group barrier and no exposed global load.
 *
 * Static format ([22]): when the module's AttentionVariant carries
 * kMixedStaticFormat (MixedPageAttention<N>, module URI ..._static_format_N),
 * every page has that format at compile time: the a16 module compiles no tag
 * loads, no format branches and no expansion; the fp8/fp4 modules compile one
 * copy arm.  N = -1 is the dynamic module ([24c] / [25d], C10 / C17): the
 * chunk-table row carries two 6-bit page masks (E4M3, E2M1); the copies are six
 * unrolled pages with predicated per-format cp.async (the mask bits are the
 * predicates: no loop, no per-page branch, no select on an address) and the
 * expansion loops are format-outer over the masks, two pages per step, the fold
 * vote once per format per operand.
 *
 * K(t-1) and V(t) are issued as a pair ([25], C13 / C14): K(last) alone first
 * (finished before barrier_O, C7), then the peeled first pair (K(last-1),
 * V(last)) - the only two calls that compile the partial-tile copy arm, since
 * only tile kv_tile_idx can have valid < CTA_KV - then the loop over full pairs,
 * whose finish is unconditional in the static modules (every pair it finishes
 * is a (K, V) pair issued one iteration earlier), and the drain (V alone).  The
 * consumer (mma_f16) is unchanged.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_SPARSE_MIXED_MAINLOOP_CUH_
#define FLASHINFER_ATTENTION_HOPPER_SPARSE_MIXED_MAINLOOP_CUH_

#ifdef MIXED_FA3_TRACE
#include <cstdio>
#endif
#include <cutlass/cutlass.h>
#include <cutlass/pipeline/pipeline.hpp>

#include <stdexcept>
#include <string>
#include <type_traits>

#include "../../math.cuh"
#include "../page_transport.cuh"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "named_barrier.cuh"
#include "sparse_mainloop.cuh"
#include "utils.cuh"

namespace flashinfer {

using namespace cute;

// AdditionalParams carries the mixed transport when it has a page_format member.
template <typename T, typename = void>
struct has_mixed_page_format : std::false_type {};
template <typename T>
struct has_mixed_page_format<T, std::void_t<decltype(T::mixed_page_format)>> : std::true_type {};
template <typename T>
inline constexpr bool has_mixed_page_format_v = has_mixed_page_format<T>::value;

// A mainloop that owns extra shared-memory barriers exposes init_shared().
template <typename Mainloop, typename SharedStorage, typename = void>
struct mainloop_has_init_shared : std::false_type {};
template <typename Mainloop, typename SharedStorage>
struct mainloop_has_init_shared<
    Mainloop, SharedStorage,
    std::void_t<decltype(Mainloop::init_shared(std::declval<SharedStorage&>()))>>
    : std::true_type {};
template <typename Mainloop, typename SharedStorage>
inline constexpr bool mainloop_has_init_shared_v =
    mainloop_has_init_shared<Mainloop, SharedStorage>::value;

namespace mixed_detail {

// f32 product that keeps subnormal results (mul.rn.f32 without .ftz).  The
// module is compiled with flush-to-zero f32 arithmetic, but the scale factor
// bf16(f32(s) * g) must equal the reference's for |s g| < 2^-126 - a bf16
// subnormal (C9; e.g. s = 2^-9 with g = 1.1 x 2^-118 on the exact path).  One
// FMUL either way.
CUTLASS_DEVICE float mul_rn_f32_denorm(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

CUTLASS_DEVICE uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

// E2M1 x8 (nibbles 0..7 of src; low nibble = even coefficient) -> four A16
// pairs, out[k] = {value 2k+1 : value 2k}.
//
// bf16 ([25], C16; XQA csrc/xqa/mhaUtils.cuh e2m1x8ToBF16x2Pow2m126): bit
// placement, no LUT.  Data flow per output word k: w4 = src << 4 puts the even
// nibble 2k at bits [7:4] of byte k (its sign at the byte's msb); the odd
// nibble 2k+1 sits at bits [7:4] of src's byte k already.  prmt(w4, src, sel_k)
// with sign-replicate selectors builds [rep(sign odd), src.byte_k, rep(sign
// even), w4.byte_k]; << 2 moves each nibble's e e m to bits [8:6] of its half
// (the nibble's own sign bit lands at 9 and is masked) and the replicated sign
// to bit 15; the mask 0x81C0 per half keeps s | 000000 ee | m 000000.  Read as
// bf16 that half is the E2M1 value x 2^-126 exactly for every code: normal
// codes 1.m x 2^(ee-1) -> exponent field ee = 1.m x 2^(ee-127); code 001 (0.5)
// -> the bf16 subnormal 0.1 x 2^-126 = 2^-127 (mul.rn.bf16x2 takes subnormal
// inputs exactly, as the E4M3 placement below already relies on).  The 2^126 is
// folded into the block scale (OperandBases::gs4, kFp4Fold*: exact iff 2^-126
// <= |s g| < 3.9921875, voted per operand in expand_operand; the exact fallback
// multiplies the placed halves by 2^126 first).  13 instructions per 8 values
// (SHL + 4 x {PRMT, SHL, LOP3}) against the LUT's 20.  f16: CUTLASS's LUT (no
// fold: the f16 path multiplies by the plain scale).
template <typename A16>
CUTLASS_DEVICE void e2m1x8_to_a16(uint32_t src, uint32_t (&out)[4]) {
  if constexpr (std::is_same_v<A16, cutlass::half_t>) {
    cutlass::detail::_e2m1_to_half_x8(src, out[0], out[1], out[2], out[3]);
  } else {
    uint32_t const w4 = src << 4;
    // selector nibbles, low to high: w4.byte_k (k), rep(sign w4.byte_k) (8|k),
    // src.byte_k (4|k), rep(sign src.byte_k) (8|4|k).
    out[0] = (prmt(w4, src, 0xC480u) << 2) & 0x81C081C0u;
    out[1] = (prmt(w4, src, 0xD591u) << 2) & 0x81C081C0u;
    out[2] = (prmt(w4, src, 0xE6A2u) << 2) & 0x81C081C0u;
    out[3] = (prmt(w4, src, 0xF7B3u) << 2) & 0x81C081C0u;
  }
}

// E4M3 x4 (bytes of w) -> two A16 pairs {b1 : b0}, {b3 : b2}.  f16: cvt is exact.
//
// bf16 ([24a]; XQA [16], csrc/xqa/mhaUtils.cuh:633-662): bit placement, no cvt.
// Data flow per word: prmt with sign-replicate selectors spreads bytes b0, b1
// (b2, b3) to bytes 0, 2 with their sign bytes at 1, 3: [rep(s1), b1, rep(s0), b0];
// << 4 moves each byte to bits [11:4] of its half and the replicated sign to
// bit 15; the mask 0x87F0 per half keeps s | eeee | mmm and clears the sign copy
// at bit 11.  The half s|0000 eeee|mmm 0000 is a bf16 whose exponent field is
// the E4M3 exponent unbiased, i.e. exactly x * 2^-120 for every finite code:
// normal 1.mmm * 2^(E-7) -> 1.mmm * 2^(E-127); subnormal mmm * 2^-9 -> the bf16
// subnormal mmm * 2^-129 (unit 2^-133; mul.rn.bf16x2 handles subnormal inputs
// exactly, verified exhaustively on H200 for [16]).  NaN codes 0x7F / 0xFF
// become the finite 1.111 * 2^-112 (C9; the quantizer never emits NaN).
// The 2^120 is folded into the block scale (OperandBases::gs8, C9 bounds and
// the exact fallback in expand_block), and one HMUL2.BF16 rounds x * s once, as
// the reference does.  Per 4 values: 2 PRMT, 2 SHF, 2 LOP3 (from 2 F2FP, 2 SHF,
// 2 LOP3, 2 IMAD of the [23] cvt form).
template <typename A16>
CUTLASS_DEVICE void e4m3x4_to_a16(uint32_t w, uint32_t& p01, uint32_t& p23) {
  if constexpr (std::is_same_v<A16, cutlass::half_t>) {
    asm("{\n\t.reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, %2;\n\t"
        "cvt.rn.f16x2.e4m3x2 %0, lo;\n\t"
        "cvt.rn.f16x2.e4m3x2 %1, hi;\n\t}"
        : "=r"(p01), "=r"(p23)
        : "r"(w));
  } else {
    uint32_t const a = prmt(w, w, 0x9180u);  // [rep(sign b1), b1, rep(sign b0), b0]
    uint32_t const b = prmt(w, w, 0xB3A2u);  // [rep(sign b3), b3, rep(sign b2), b2]
    p01 = (a << 4) & 0x87F087F0u;
    p23 = (b << 4) & 0x87F087F0u;
  }
}

// bf16 FP8 fold constants (C9).  The placed value is x * 2^-120; the block
// scale carries 2^120.  bf16_rn(f32(s) * g * 2^120) is finite iff
// |s * g| < 255.5 (bf16 max is 255 * 2^120; [255.5, 256) rounds to +inf under
// round-to-nearest-even), and equals 2^120 * bf16_rn(f32(s) * g) iff the latter
// is a bf16 normal, i.e. |s * g| >= 2^-126, which for the smallest E4M3 scale
// 2^-9 is |g| >= 2^-117.  kFp8FoldMax is the per-block test bound (as f32:
// 1.99609375 * 2^127, representable); kFp8FoldSentinel is the per-operand
// "never fold" value: f32(s) * inf is inf (or NaN for s = 0), so |v| < bound is
// false for every block and the exact two-multiply path is taken.
template <typename A16>
inline constexpr bool kFp8FoldsPow2 = std::is_same_v<A16, cutlass::bfloat16_t>;
inline constexpr float kFp8Fold = 0x1p120f;
inline constexpr float kFp8FoldMax = 255.5f * 0x1p120f;
inline constexpr float kFp8FoldMinGlobal = 0x1p-117f;
inline constexpr uint32_t kFp8FoldSentinelBits = 0x7F800000u;  // +inf
inline constexpr uint32_t kTwoPow120Bf16x2 = 0x7B807B80u;      // bf16x2 {2^120, 2^120}

// bf16 FP4 fold constants ([25], C16).  The placed E2M1 value is x * 2^-126;
// the block scale carries 2^126.  bf16_rn(f32(s) * g * 2^126) is finite iff
// |s * g| < 3.9921875 (bf16 max below 4 is 3.984375; the midpoint to 4.0 rounds
// to even = 2^128 = inf), and equals 2^126 * bf16_rn(f32(s) * g) iff the latter
// is a bf16 normal: |s g| >= 2^-126, i.e. |g| >= 2^-117 for the smallest E4M3
// scale 2^-9 - the same lower-bound sentinel as fp8 (kFp8FoldMinGlobal).
// 3.9921875 * 2^126 = 1.99609375 * 2^127 = kFp8FoldMax: the same f32 constant.
inline constexpr float kFp4Fold = 0x1p126f;
inline constexpr float kFp4FoldMax = 3.9921875f * 0x1p126f;
inline constexpr uint32_t kTwoPow126Bf16x2 = 0x7E807E80u;  // bf16x2 {2^126, 2^126}

// Shared-window accesses with 32-bit addresses (the immediates are folded by ptxas).
CUTLASS_DEVICE uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(a));
  return v;
}
CUTLASS_DEVICE uint2 lds64(uint32_t a) {
  uint2 v;
  asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(a));
  return v;
}
CUTLASS_DEVICE uint32_t lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}
CUTLASS_DEVICE void sts128(uint32_t a, uint4 const& v) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w));
}
CUTLASS_DEVICE void sts32(uint32_t a, uint32_t v) {
  asm volatile("st.shared.b32 [%0], %1;" ::"r"(a), "r"(v));
}

template <typename A16>
CUTLASS_DEVICE uint32_t mul_a16x2(uint32_t x, uint32_t sf2) {
  uint32_t r;
  if constexpr (std::is_same_v<A16, cutlass::half_t>) {
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(x), "r"(sf2));
  } else {
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(x), "r"(sf2));
  }
  return r;
}

// 16 B / 8 B cp.async with a 32-bit shared address (the compiler folds the
// per-page constant offsets into the LDGSTS immediate).  Same cache policy as
// cutlass::arch::cp_async_zfill<16, Global> / <8, Always>.
CUTLASS_DEVICE void cp16(uint32_t smem, void const* gmem) {
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(smem), "l"(gmem));
}
CUTLASS_DEVICE void cp16_zfill(uint32_t smem, void const* gmem, bool pred) {
  int const n = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" ::"r"(smem), "l"(gmem),
               "r"(n));
}
CUTLASS_DEVICE void cp8(uint32_t smem, void const* gmem) {
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 8;\n" ::"r"(smem), "l"(gmem));
}
CUTLASS_DEVICE void cp8_zfill(uint32_t smem, void const* gmem, bool pred) {
  int const n = pred ? 8 : 0;
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 8, %2;\n" ::"r"(smem), "l"(gmem),
               "r"(n));
}
// [25d] Predicated forms for the dynamic module's per-format copies (C17): the
// PTX predicate keeps the copy one branch-free `@P LDGSTS` (a predicated-off
// lane issues nothing - no zero-fill of a destination another lane fills), and
// the src-size operand carries the D4 zero-fill of the partial arm.
CUTLASS_DEVICE void cp16_pred(uint32_t smem, void const* gmem, bool pred, int src_size) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %2, 0;\n\t"
      "@p cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %3;\n\t}" ::"r"(smem),
      "l"(gmem), "r"(int(pred)), "r"(src_size));
}
CUTLASS_DEVICE void cp8_pred(uint32_t smem, void const* gmem, bool pred, int src_size) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %2, 0;\n\t"
      "@p cp.async.ca.shared.global.L2::128B [%0], [%1], 8, %3;\n\t}" ::"r"(smem),
      "l"(gmem), "r"(int(pred)), "r"(src_size));
}
CUTLASS_DEVICE void cp4(uint32_t smem, void const* gmem) {
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 4;\n" ::"r"(smem), "l"(gmem));
}
CUTLASS_DEVICE void cp4_zfill(uint32_t smem, void const* gmem, bool pred) {
  int const n = pred ? 4 : 0;
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 4, %2;\n" ::"r"(smem), "l"(gmem),
               "r"(n));
}

#ifdef MIXED_FA3_TRACE
// Diagnosis-only timeline (ns).  Default builds do not compile this.
CUTLASS_DEVICE uint64_t globaltimer_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#endif

}  // namespace mixed_detail

template <typename AdditionalParams, typename Ktraits, bool CAUSAL>
struct SparseMixedCollectiveMainloop {
  using Base = SparseCollectiveMainloop<AdditionalParams, Ktraits, CAUSAL>;
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using IdType = typename Ktraits::IdType;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using TileShape_PDV = typename Ktraits::TileShape_PDV;
  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SharedStorage = typename Ktraits::SharedStorage;
  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using WarpScheduler = typename Base::WarpScheduler;
  using TMA_Q = typename Base::TMA_Q;
  using LayoutT = typename Base::LayoutT;
  using TileMeta = typename SharedStorage::TileMeta;

  static constexpr int CTA_Q = get<0>(TileShape_QKD{});
  static constexpr int CTA_KV = get<1>(TileShape_QKD{});
  static constexpr int HEAD_DIM = get<2>(TileShape_QKD{});
  static constexpr int NUM_STAGES = Ktraits::NUM_STAGES;
  // One producer warp group ([25], A9): every page of a tile is owned by the 128
  // producer threads; thread t serves all PAGES_PER_THREAD = 6 pages of a tile.
  static constexpr int NUM_COPY_THREADS = cutlass::NumThreadsPerWarpGroup;
  static constexpr uint32_t OWNER_THREADS = cutlass::NumThreadsPerWarpGroup;
  static constexpr uint32_t TOKENS_PER_PAGE = 16;
  static constexpr uint32_t PAGES_PER_TILE = CTA_KV / TOKENS_PER_PAGE;
  static constexpr uint32_t PAGES_PER_THREAD = PAGES_PER_TILE;
  static constexpr uint32_t D_BLOCK = 64;  // elements per SW128 atom row (128 B of A16)
  static constexpr uint32_t D_BLOCKS = HEAD_DIM / D_BLOCK;
  static constexpr uint32_t CHUNK_ELEMS = 16 / sizeof(DTypeKV);  // one 16 B cp.async
  static constexpr uint32_t CHUNKS_PER_ROW = HEAD_DIM / CHUNK_ELEMS;  // 16
  static constexpr uint32_t CHUNKS_PER_BLOCK = D_BLOCK / CHUNK_ELEMS;  // 8
  static constexpr uint32_t BLOCKS_PER_HEAD = HEAD_DIM / 16;  // scale groups per token
  static constexpr uint32_t FP8_ROW_CHUNKS = HEAD_DIM / 16;  // 8 x 16 B
  static constexpr uint32_t FP4_ROW_CHUNKS = HEAD_DIM / 32;  // 4 x 16 B
  static constexpr uint32_t STAGE_BYTES = CTA_KV * HEAD_DIM * sizeof(DTypeKV);
  static constexpr uint32_t ATOM_BYTES = 8 * 128;                 // one SW128 atom: 8 rows x 128 B
  static constexpr uint32_t PAGE_REGION_BYTES = 2 * ATOM_BYTES;  // 16 rows of one D-block
  // [24b] scale slots: one 8 B word pair per token row per page (the row's eight
  // block scales) in the first 128 B of the page's 512 B slot; the slot size is
  // [23]'s so that the shared-storage layout is unchanged (kernel_traits.cuh
  // kMixedScaleStageBytes).
  static constexpr uint32_t SCALE_ROW_BYTES = 8;
  static constexpr uint32_t SCALE_PAGE_BYTES = OWNER_THREADS * 4;  // 512 B page slot
  static constexpr uint32_t SCALE_STAGE_BYTES = PAGES_PER_TILE * SCALE_PAGE_BYTES;
  static_assert(TOKENS_PER_PAGE * SCALE_ROW_BYTES <= SCALE_PAGE_BYTES, "row slots fit the page slot");
  static_assert(SCALE_STAGE_BYTES == SharedStorage::kMixedScaleStageBytes, "scale stage size");

  // [22] compile-time page format: -1 dynamic (per-page tags), 0 A16, 1 E4M3, 2 E2M1,
  // read from the variant by the traits (kernel_traits.cuh mixed_variant_static_format;
  // the same constant selects the register split there: 72/216 a16, 136/184 otherwise).
  static constexpr int STATIC_FORMAT = Ktraits::kMixedStaticFormat;
  static_assert(STATIC_FORMAT >= -1 && STATIC_FORMAT <= 2, "static format is -1, 0, 1 or 2");
  static constexpr bool DYNAMIC = STATIC_FORMAT < 0;
  static constexpr bool STATIC_A16 = STATIC_FORMAT == 0;
  static constexpr bool STATIC_FP8 = STATIC_FORMAT == 1;
  static constexpr bool STATIC_FP4 = STATIC_FORMAT == 2;
  static constexpr bool HAS_A16 = DYNAMIC || STATIC_A16;
  static constexpr bool HAS_FP8 = DYNAMIC || STATIC_FP8;
  static constexpr bool HAS_FP4 = DYNAMIC || STATIC_FP4;
  static constexpr bool HAS_COMPRESSED = HAS_FP8 || HAS_FP4;

  // [21] chunk table geometry.
  static constexpr int CHUNK_TILES = SharedStorage::kMixedMetaChunkTiles;  // 16
  static constexpr int META_BUFFERS = SharedStorage::kMixedMetaBuffers;    // 2
  // The next chunk's rows are stored at this pair of the current chunk.  The
  // buffer they overwrite was last read at pair CHUNK_TILES-1 of the previous
  // chunk; a producer thread cannot lead another by more than NUM_STAGES pairs
  // (its acquire needs the consumers' release, which needs every producer
  // thread's commit), so the store is a WAR-safe CHUNK_STORE_PAIR+1 > NUM_STAGES
  // pairs behind that read.  Visibility to the group is the barrier right after.
  static constexpr int CHUNK_STORE_PAIR = 8;
  static_assert(CHUNK_STORE_PAIR > NUM_STAGES && CHUNK_STORE_PAIR < CHUNK_TILES - 1,
                "chunk store must trail the previous chunk's last read by > NUM_STAGES pairs");
  static_assert(CHUNK_TILES * 8 == OWNER_THREADS,
                "one thread of warp group 0 per (tile row, page slot)");
  static_assert(META_BUFFERS == 2, "double-buffered chunk table");

  static_assert(HEAD_DIM == 128 && sizeof(DTypeKV) == 2,
                "mixed pages are implemented for D=128 A16 (bf16/f16) math");
  static_assert(CTA_KV % TOKENS_PER_PAGE == 0, "KV tile must be whole pages");
  static_assert(CTA_KV <= OWNER_THREADS, "one page-owner thread per token");
  static_assert(PAGES_PER_TILE == 6, "the chunk-table row packs 6 tags + valid + flags in 8 B");
  static_assert(get<1>(TileShape_PDV{}) == HEAD_DIM && get<2>(TileShape_PDV{}) == CTA_KV);
  static_assert(Ktraits::NUM_PRODUCER_THREADS == NUM_COPY_THREADS,
                "the mixed mainloop uses the whole producer warp group");
  static_assert(BLOCKS_PER_HEAD == 8, "a token's 8 scale bytes are two 4 B words");
  static_assert(NUM_STAGES >= 3, "pending pair + current pair + consumer need three stages");
  // [25] C14: the pending record is one 32-bit word with the stage in bits 30-31.
  static_assert(NUM_STAGES <= 4, "the 32-bit pending word holds the stage index in bits 30..31");
  static_assert(STAGE_BYTES == (CTA_KV / 8) * D_BLOCKS * 1024, "stage size (D1)");
  // The copy destinations are thread constants plus immediate offsets; these pin
  // the layout facts they rely on.  The stage tensor is made from a smem_ptr, so
  // its SW128 swizzle (Swizzle<3,4,3>) acts on *byte* addresses (cute
  // make_tensor(smem_ptr, ComposedLayout<Swizzle, smem_ptr_flag_bits, L>)); the
  // element-domain view of that is as_position_independent_swizzle_layout, which
  // is what these assertions evaluate (the raw ComposedLayout would misapply
  // the byte swizzle to element offsets).  The swizzle permutes 16 B chunks
  // within an 8-row x 128 B atom only, so offsets by whole atoms are exact.
  static_assert(std::is_same_v<decltype(get_swizzle_portion(SmemLayoutK{})), Swizzle<3, 4, 3>> &&
                    std::is_same_v<decltype(get_swizzle_portion(SmemLayoutV{})), Swizzle<3, 4, 3>>,
                "K/V stages use the SW128 (Swizzle<3,4,3> on bytes) atom");
  using SmemLayoutKElems = decltype(as_position_independent_swizzle_layout(SmemLayoutK{}));
  using SmemLayoutVElems = decltype(as_position_independent_swizzle_layout(SmemLayoutV{}));
  static_assert(decltype(SmemLayoutKElems{}(_0{}, _0{}, _1{}))::value == CTA_KV * HEAD_DIM,
                "stage stride is one stage of elements");
  static_assert(decltype(SmemLayoutKElems{}(Int<16>{}, _0{}, _0{}))::value == PAGE_REGION_BYTES / 2,
                "page i of a D-block starts at i * PAGE_REGION_BYTES");
  static_assert(decltype(SmemLayoutKElems{}(Int<8>{}, _0{}, _0{}))::value == ATOM_BYTES / 2,
                "rows 8..15 of a page start ATOM_BYTES after rows 0..7");
  static_assert(decltype(SmemLayoutKElems{}(_0{}, Int<64>{}, _0{}))::value == CTA_KV * 64,
                "D-block 1 follows all rows of D-block 0");
  static_assert(decltype(SmemLayoutVElems{}(_0{}, _0{}, _1{}))::value == CTA_KV * HEAD_DIM &&
                    decltype(SmemLayoutVElems{}(Int<16>{}, _0{}, _0{}))::value == PAGE_REGION_BYTES / 2 &&
                    decltype(SmemLayoutVElems{}(Int<8>{}, _0{}, _0{}))::value == ATOM_BYTES / 2 &&
                    decltype(SmemLayoutVElems{}(_0{}, Int<64>{}, _0{}))::value == CTA_KV * 64,
                "V stage has the K stage's tiling");

  static constexpr bool UseSchedulerBarrier = Base::UseSchedulerBarrier;
  static constexpr bool USE_TMA_LOAD_KV = false;
  static constexpr uint32_t TmaTransactionBytesQ = Base::TmaTransactionBytesQ;
  static constexpr uint32_t TmaTransactionBytesK = 0;
  static constexpr uint32_t TmaTransactionBytesV = 0;

  // Tag values in the chunk table.  Out-of-range page slots of a tail tile are
  // tagged A16 in the dynamic module (their rows are zero-filled by cp.async)
  // and STATIC_FORMAT in a static one (rows >= valid are zero-filled by the
  // expansion, which never reads them).
  static constexpr uint8_t kTagA16 = static_cast<uint8_t>(KVPageFormat::kA16);
  static constexpr uint8_t kTagFP8 = static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8);
  static constexpr uint8_t kTagFP4 = static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4);
  static constexpr uint32_t kFlagCompressed = 1;
  static constexpr uint32_t kFlagFilled = 2;

  struct Arguments {
    DTypeQ const* Q_ptr;
    LayoutT layout_Q;
    DTypeKV const* K_ptr;
    LayoutT layout_K;
    DTypeKV const* V_ptr;
    LayoutT layout_V;
    IdType const* kv_indices;
    int window_left;
    int64_t k_page_stride;
    int64_t v_page_stride;
    uint32_t page_size;
    AdditionalParams additional_params;
  };

  struct Params {
    LayoutT layout_Q;
    TMA_Q tma_load_Q;
    DTypeKV* K_ptr;
    DTypeKV* V_ptr;
    uint32_t k_page_stride_b;  // bytes
    uint32_t v_page_stride_b;
    int64_t k_token_stride;  // elements
    int64_t v_token_stride;
    int64_t k_head_stride;
    int64_t v_head_stride;
    IdType* kv_indices;
    int window_left;
    uint32_t page_size;
    KVPageTransport<DTypeKV> transport;
    AdditionalParams additional_params;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.Q_ptr), args.layout_Q);
    TMA_Q tma_load_Q = make_tma_copy(typename Base::GmemTiledCopyQ{}, mQ, SmemLayoutQ{},
                                     select<0, 2>(TileShape_QKD{}), _1{});
    auto const& ap = args.additional_params;
    KVPageTransport<DTypeKV> transport{};
    transport.page_format = ap.mixed_page_format;
    auto& fp8 = transport.formats[static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8)];
    fp8.k_payload = ap.mixed_fp8_k_payload;
    fp8.v_payload = ap.mixed_fp8_v_payload;
    fp8.k_scales = ap.mixed_fp8_k_scales;
    fp8.v_scales = ap.mixed_fp8_v_scales;
    fp8.k_global_scale = ap.mixed_fp8_k_global_scale;
    fp8.v_global_scale = ap.mixed_fp8_v_global_scale;
    fp8.payload_stride = {static_cast<uint32_t>(ap.mixed_fp8_payload_stride_page),
                          static_cast<uint32_t>(ap.mixed_fp8_payload_stride_token),
                          static_cast<uint32_t>(ap.mixed_fp8_payload_stride_head)};
    fp8.scale_stride = {static_cast<uint32_t>(ap.mixed_fp8_scale_stride_page),
                        static_cast<uint32_t>(ap.mixed_fp8_scale_stride_token),
                        static_cast<uint32_t>(ap.mixed_fp8_scale_stride_head)};
    auto& fp4 = transport.formats[static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4)];
    fp4.k_payload = ap.mixed_fp4_k_payload;
    fp4.v_payload = ap.mixed_fp4_v_payload;
    fp4.k_scales = ap.mixed_fp4_k_scales;
    fp4.v_scales = ap.mixed_fp4_v_scales;
    fp4.k_global_scale = ap.mixed_fp4_k_global_scale;
    fp4.v_global_scale = ap.mixed_fp4_v_global_scale;
    fp4.payload_stride = {static_cast<uint32_t>(ap.mixed_fp4_payload_stride_page),
                          static_cast<uint32_t>(ap.mixed_fp4_payload_stride_token),
                          static_cast<uint32_t>(ap.mixed_fp4_payload_stride_head)};
    fp4.scale_stride = {static_cast<uint32_t>(ap.mixed_fp4_scale_stride_page),
                        static_cast<uint32_t>(ap.mixed_fp4_scale_stride_token),
                        static_cast<uint32_t>(ap.mixed_fp4_scale_stride_head)};

    if (args.page_size != TOKENS_PER_PAGE) {
      throw std::runtime_error("mixed KV pages require 16-token pages");
    }
    // The module is compiled for one page format ([22]); the run arguments must agree.
    if (static_cast<int>(ap.mixed_static_format) != STATIC_FORMAT) {
      throw std::runtime_error(
          "mixed KV pages: static_format of the run arguments does not match the module "
          "(build the wrapper with mixed_page_prefill_jit_args(..., static_format=...))");
    }
    if ((STATIC_FP8 && (ap.mixed_fp8_k_payload == nullptr || ap.mixed_fp8_v_payload == nullptr)) ||
        (STATIC_FP4 && (ap.mixed_fp4_k_payload == nullptr || ap.mixed_fp4_v_payload == nullptr))) {
      throw std::runtime_error("mixed KV pages: static format without its payload tensors");
    }
    // A16 rows are copied in 16 B chunks: every A16 row must be 16 B aligned.
    if ((reinterpret_cast<uintptr_t>(args.K_ptr) % 16) || (reinterpret_cast<uintptr_t>(args.V_ptr) % 16) ||
        (args.k_page_stride * int64_t(sizeof(DTypeKV))) % 16 || (args.v_page_stride * int64_t(sizeof(DTypeKV))) % 16 ||
        (stride<0>(args.layout_K) * int64_t(sizeof(DTypeKV))) % 16 ||
        (stride<0>(args.layout_V) * int64_t(sizeof(DTypeKV))) % 16 ||
        (stride<2>(args.layout_K) * int64_t(sizeof(DTypeKV))) % 16 ||
        (stride<2>(args.layout_V) * int64_t(sizeof(DTypeKV))) % 16) {
      throw std::runtime_error("mixed KV pages: A16 cache rows must be 16 B aligned");
    }
    // Compressed copies: FP8 blocks are 16 B cp.async, FP4 blocks 8 B, and a
    // row's eight scale bytes one 8 B cp.async ([24b]) - every source address must
    // keep that alignment across pages, tokens and heads.
    auto check_span = [](KVPageFormatSpan const& s, uint32_t block_align, char const* name) {
      constexpr uint32_t kScaleAlign = SCALE_ROW_BYTES;
      auto misaligned = [](void const* ptr, uint32_t a) {
        return ptr != nullptr && reinterpret_cast<uintptr_t>(ptr) % a != 0;
      };
      if (misaligned(s.k_payload, block_align) || misaligned(s.v_payload, block_align) ||
          s.payload_stride.page % block_align || s.payload_stride.token % block_align ||
          s.payload_stride.head % block_align || misaligned(s.k_scales, kScaleAlign) ||
          misaligned(s.v_scales, kScaleAlign) || s.scale_stride.page % kScaleAlign ||
          s.scale_stride.token % kScaleAlign || s.scale_stride.head % kScaleAlign) {
        throw std::runtime_error(std::string("mixed KV pages: ") + name +
                                 " payload rows must be block aligned and scale rows 8 B aligned");
      }
    };
    if (HAS_FP8) check_span(fp8, 16, "FP8");
    if (HAS_FP4) check_span(fp4, 8, "FP4");
    // Page terms are one IMAD.WIDE.U32 (page x 32-bit byte stride) per copy.
    int64_t const k_ps_b = args.k_page_stride * int64_t(sizeof(DTypeKV));
    int64_t const v_ps_b = args.v_page_stride * int64_t(sizeof(DTypeKV));
    if (k_ps_b < 0 || v_ps_b < 0 || k_ps_b > int64_t(UINT32_MAX) || v_ps_b > int64_t(UINT32_MAX)) {
      throw std::runtime_error("mixed KV pages: A16 page stride must fit in 32 bits of bytes");
    }
    return {args.layout_Q,
            tma_load_Q,
            const_cast<DTypeKV*>(args.K_ptr),
            const_cast<DTypeKV*>(args.V_ptr),
            static_cast<uint32_t>(k_ps_b),
            static_cast<uint32_t>(v_ps_b),
            static_cast<int64_t>(stride<0>(args.layout_K)),
            static_cast<int64_t>(stride<0>(args.layout_V)),
            static_cast<int64_t>(stride<2>(args.layout_K)),
            static_cast<int64_t>(stride<2>(args.layout_V)),
            const_cast<IdType*>(args.kv_indices),
            args.window_left,
            args.page_size,
            transport,
            args.additional_params};
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_Q.get_tma_descriptor());
  }

  CUTLASS_DEVICE
  int get_num_kv_tiles(Params const& mainloop_params, int q_tile_idx, const int qo_len,
                       const int kv_len) {
    // Local copies: cute::ceil_div takes by reference, which odr-uses the class
    // statics in device code (same pattern as SparseCollectiveMainloop).
    static constexpr int CTA_Q_ = CTA_Q;
    static constexpr int CTA_KV_ = CTA_KV;
    int num_kv_tiles = cute::ceil_div(kv_len, CTA_KV_);
    if constexpr (CAUSAL) {
      num_kv_tiles = std::min(num_kv_tiles,
                              cute::ceil_div((q_tile_idx + 1) * CTA_Q_ + kv_len - qo_len, CTA_KV_));
    }
    return num_kv_tiles;
  }

  CUTLASS_DEVICE static bool is_compressed(uint32_t f) { return f == kTagFP8 || f == kTagFP4; }

  // 16 B chunk c of element row `row` of stage `stage`: CuTe applies the SW128
  // swizzle, so the copies and the expansion agree on the layout by construction (D1).
  template <typename STensor>
  CUTLASS_DEVICE static void* chunk_ptr(STensor& sX, int stage, int row, uint32_t chunk) {
    return &sX(row, int(chunk * CHUNK_ELEMS), stage);
  }
  template <typename STensor>
  CUTLASS_DEVICE static uint32_t chunk_smem(STensor& sX, int stage, int row, uint32_t chunk) {
    return cute::cast_smem_ptr_to_uint(chunk_ptr(sX, stage, row, chunk));
  }

  // ------------------------------------------------------------------------
  // [21] Chunk table.  Thread t < 128 owns row t/8 (tile T - 16*chunk - t/8) and
  // page slot t%8 (slots 6, 7 idle).  A chunk's (index, tag) pairs are loaded
  // into registers early and stored later; see load().  Dynamic module ([24c]):
  // the row's tags are reduced to two page masks at the store.
  struct ChunkRegs {
    IdType page;
    uint32_t tag;
  };

  CUTLASS_DEVICE void chunk_load(Params const& p, IdType const* kv_indices_ptr, int top_tile,
                                 int chunk, int kv_len, int thread_idx, ChunkRegs& r) const {
    int const tile = top_tile - chunk * CHUNK_TILES - (thread_idx >> 3);
    int const slot = thread_idx & 7;
    r.page = 0;
    r.tag = DYNAMIC ? kTagA16 : uint32_t(STATIC_FORMAT);
    if (slot < int(PAGES_PER_TILE) && tile >= 0) {
      int const tok0 = tile * CTA_KV + slot * int(TOKENS_PER_PAGE);
      if (tok0 < kv_len) {
        r.page = kv_indices_ptr[tok0 / int(TOKENS_PER_PAGE)];
        if constexpr (DYNAMIC) r.tag = p.transport.page_format[r.page];
      }
    }
  }

  CUTLASS_DEVICE void chunk_store(TileMeta (*meta)[CHUNK_TILES], int top_tile, int chunk,
                                  int kv_len, int thread_idx, ChunkRegs const& r) const {
    int const row = thread_idx >> 3;
    int const slot = thread_idx & 7;
    int const tile = top_tile - chunk * CHUNK_TILES - row;
    TileMeta& e = meta[chunk & 1][row];
    if constexpr (DYNAMIC) {
      // [24c] Two 6-bit page masks by OR over the row's 8 lanes (idle slots 6, 7
      // and A16 pages contribute 0); the row's last word is written once by
      // lane 0: m8 | m4 << 8 | valid << 16 | flags << 24 (C10).
      uint32_t const octet = 0xFFu << (thread_idx & 24);
      uint32_t const m8 = __reduce_or_sync(octet, r.tag == kTagFP8 ? 1u << slot : 0u) & 0x3Fu;
      uint32_t const m4 = __reduce_or_sync(octet, r.tag == kTagFP4 ? 1u << slot : 0u) & 0x3Fu;
      if (slot < int(PAGES_PER_TILE)) e.pages[slot] = r.page;
      if (slot == 0) {
        int const cta_kv = CTA_KV;
        uint32_t const valid = uint32_t(max(0, min(cta_kv, kv_len - tile * cta_kv)));
        uint32_t const flags = kFlagFilled | ((m8 | m4) ? kFlagCompressed : 0u);
        mixed_detail::sts32(cute::cast_smem_ptr_to_uint(&e) + 28u,
                            m8 | (m4 << 8) | (valid << 16) | (flags << 24));
      }
    } else {
      // Static modules: the [23] text (the a16 module's SASS is the control).
      uint32_t const any = HAS_COMPRESSED ? 1u : 0u;
      if (slot < int(PAGES_PER_TILE)) {
        e.pages[slot] = r.page;
        e.tags[slot] = static_cast<uint8_t>(r.tag);
      }
      if (slot == 0) {
        int const cta_kv = CTA_KV;
        e.valid = static_cast<uint8_t>(max(0, min(cta_kv, kv_len - tile * cta_kv)));
        e.flags = static_cast<uint8_t>(kFlagFilled | (any ? kFlagCompressed : 0u));
      }
    }
  }

  // One tile's row of the chunk table in registers (two LDS.128, every module):
  // `pages` (6 indices), `w6` (static: tags 0..3; dynamic: unused) and `w7`
  // (static: tags 4, 5 | valid << 16 | flags << 24; dynamic ([24c], C10): m8 |
  // m4 << 8 | valid << 16 | flags << 24).
  struct TileRegs {
    uint32_t pages[PAGES_PER_TILE];
    uint32_t w6, w7;
    // Page index of tile page j; `j` is a constant after unrolling - never a
    // runtime index into pages[] (C2: local memory).
    CUTLASS_DEVICE uint32_t page(uint32_t j) const { return pages[j]; }
    CUTLASS_DEVICE uint32_t mask8() const { return w7 & 0x3Fu; }
    CUTLASS_DEVICE uint32_t mask4() const { return (w7 >> 8) & 0x3Fu; }
    CUTLASS_DEVICE uint32_t valid() const { return (w7 >> 16) & 0xFFu; }
    CUTLASS_DEVICE bool any_compressed() const { return (w7 >> 24) & kFlagCompressed; }
    // Pending record ([25] C14): one 32-bit word = w7's bits 0..25 (dynamic:
    // masks; static: tags 4, 5; both: valid at 16..23, flag bits 0-1 at 24, 25)
    // | stage << 30.  Nonzero whenever the row is filled (kFlagFilled = bit 25);
    // the flags byte's bits 6-7 are masked off so the stage cannot alias them.
    CUTLASS_DEVICE uint32_t pending_word(uint32_t stage) const {
      return (w7 & 0x03FFFFFFu) | (stage << 30);
    }
  };
  // Fields of a pending word.
  CUTLASS_DEVICE static uint32_t pending_stage(uint32_t tv) { return tv >> 30; }
  CUTLASS_DEVICE static uint32_t pending_mask8(uint32_t tv) { return tv & 0x3Fu; }
  CUTLASS_DEVICE static uint32_t pending_mask4(uint32_t tv) { return (tv >> 8) & 0x3Fu; }

  CUTLASS_DEVICE static TileRegs read_meta(TileMeta const (*meta)[CHUNK_TILES], int entry) {
    TileMeta const& e = meta[(entry / CHUNK_TILES) & 1][entry % CHUNK_TILES];
    uint4 const a = *reinterpret_cast<uint4 const*>(&e);
    uint4 const b = *(reinterpret_cast<uint4 const*>(&e) + 1);
    return TileRegs{{a.x, a.y, a.z, a.w, b.x, b.y}, b.z, b.w};
  }

  // ------------------------------------------------------------------------
  // Copies for one tile of one operand (D6: warp-contiguous copy ownership).
  // Within a page, consecutive threads own consecutive 16 B chunks of a row, so
  // one warp instruction covers whole contiguous rows in global memory and a
  // permutation of one row's slots in the swizzled stage.  Ownership:
  //  * A16 page   : thread t owns chunk t%16 of rows t/16 and t/16+8 (zero-filled past kv_len, D4)
  //  * compressed : thread t owns scale block b = t%8 of row r = t/8 ([23], D2/D3):
  //                 its packed bytes (FP8 16 B, FP4 8 B) land in the row's D-block 1 line (chunk b
  //                 for FP8; see land_row_line), and it copies the 4 B word of the page's
  //                 16 x 8 B block scales that holds block b of row r into its own slot
  //                 (scales[stage][page][t]; a warp's copy and read are 128 contiguous bytes),
  //                 so every thread reads only what its own cp.async wrote.  Rows past kv_len are copied with src-size 0
  //                 (payload and scale word zero-filled), so the expansion has no tail case.
  // A thread's row-within-page and chunk are therefore constants: every source
  // address is (thread constant) + page * (byte stride) - one IMAD.WIDE.U32 - and
  // every destination is (thread constant) + stage * STAGE_BYTES + i * PAGE_REGION_BYTES
  // (+ ATOM_BYTES for rows 8..15), because the stage is tiled from 8-row x 128 B
  // atoms in row-group-major order within each D-block (D1 static_asserts in the
  // shared storage).  The thread constants are computed once per work item.
  //
  // Thread t owns block t%8 of row t/8: eight consecutive lanes copy one row's
  // 128 B (FP8) / 64 B (FP4) global line.  Measured (ncu source counters): the
  // cp.async path coalesces a lane octet only when it is one contiguous global
  // line - a bijection that interleaved rows r, r+1 across each octet (chosen
  // for the quarter-warp bank rule of STS/LDS) ran the payload LDGSTS.128 at 32
  // wavefronts per warp instruction (ideal 4), while the A16 copies (same rule
  // as here) run at 4.  With one row per octet the two STS.128 of a block are
  // 2-way conflicted (a row's chunks 2b ^ (r&7) are 4 even/odd groups twice; D-
  // block 1 is 12288 B = 0 mod 128 B away): +2 wavefronts per STS.128, the
  // smaller cost by an order of magnitude.
  CUTLASS_DEVICE static uint32_t blk_row(uint32_t u) { return u >> 3; }
  CUTLASS_DEVICE static uint32_t blk_blk(uint32_t u) { return u & 7u; }
  // [24b] Output-chunk permutation (C11): the first store of block b of row r goes
  // to chunk 2b + swap, the second to 2b + 1 - swap, swap = ((b >> 2) ^ r) & 1.
  // Within one row (one STS.128 wavefront of 8 lanes) the first store then hits
  // positions {0,2,4,6} ^ x in the D-block-0 line (lanes b < 4) and {1,3,5,7} ^ x in
  // the D-block-1 line (b >= 4) - or the reverse on odd rows - eight distinct
  // bank groups: 4 wavefronts per STS.128 instead of the 2-way-conflicted 8 of
  // [23] (both lines at {0,2,4,6} ^ x).  The packed halves are read in the same
  // order (load_packed), so the data order is untouched.
  CUTLASS_DEVICE static uint32_t out_swap(uint32_t u) { return ((u >> 2) ^ (u >> 3)) & 1u; }

  struct OperandBases {
    uint8_t const* a16_src0;  // row u/16, chunk u%16 of page 0 (this thread)
    uint8_t const* a16_src1;  // row u/16 + 8
    uint32_t a16_ps;          // page stride, bytes
    uint32_t a16_dst;         // smem address of (row u/16, chunk u%16), stage 0, tile page h
    uint32_t out0;            // smem address of chunk 2b + swap of row r, stage 0, page h (first output)
    uint32_t out1;            // chunk 2b + 1 - swap (second output)
    uint32_t land8;           // FP8 landing (cp.async dst, 16 B aligned): logical chunk 8+b of row r (D-block 1 line, see land_row_line)
    uint32_t land4;           // FP4 landing (cp.async dst, 8 B aligned): 8 B in chunk 8 + b/2 + 4*(r&1), half b&1
    uint32_t sc_rd;           // smem address of this thread's 4 B scale word in the row's 8 B slot, stage 0
    float gs8, gs4;           // global scales of this operand (bf16: x 2^120 / x 2^126 or the +inf sentinel, C9 / C16)
    // [25] C13: per-item 64-bit source bases of the compressed formats (this
    // thread's block of page 0: + head * hs + row * ts + blk * BLOCK_BYTES; its
    // row's scale line: + head * shs + row * sts) and the 32-bit page strides
    // (bytes).  A page's source is base + page * stride: one IMAD.WIDE.U32.
    uint64_t p8, s8, p4, s4;
    uint32_t p8_ps, s8_ps, p4_ps, s4_ps;
  };
  // Landing (land_row_line): a row's eight packed blocks land in the row's
  // D-block 1 line, FP8 block b at physical chunk b ^ (r&7) (logical chunk 8+b),
  // FP4 block b as the 8 B half b&1 of chunk (b/2 + 4*(r&1)) ^ (r&7).  Measured
  // rule (ncu source counters): a cp.async lane octet coalesces only when its
  // eight destinations lie in one 128 B smem line - the A16 copies (one row line
  // per octet) run at the ideal 4 wavefronts per LDGSTS.128; landings spread
  // over the row's two D-block lines (the owner's own output chunks, in either
  // lane order) ran at 32, one wavefront per lane.  Within the line the eight
  // chunks are eight bank groups (LDS.128 conflict-free); FP4's odd rows use
  // chunks 4..7 so the 16-lane 64-bit LDS phases (rows r, r+1) are disjoint.
  // The landing chunk is another lane's output chunk (block 4 + b/2 of the same
  // row, same warp): every lane's LDS of a page precedes any lane's STS of it in
  // program order, made explicit by __syncwarp() before the stores (no CTA
  // barrier; the octet is one warp).
  // The PRMT selector that moves byte b%4 of the scale word to byte 0 (rest
  // zero) is derived per use (one LOP3).
  CUTLASS_DEVICE static uint32_t sc_sel(uint32_t u) { return 0x4440u | (u & 3u); }

  // Thread constants of one operand, once per work item.  Data flow: page j of
  // a tile is base + j * PAGE_REGION_BYTES (+ j * SCALE_PAGE_BYTES for the scale
  // slot) - an immediate - in every copy, load and store (C2 / C3: [R+imm], no
  // runtime page index); the [24b] swap is folded into out0 / out1 here and
  // into the landing half addresses in expand_bases.
  template <typename STensor>
  CUTLASS_DEVICE OperandBases make_bases(Params const& p, bool isK, int kv_head_idx, STensor& sX,
                                         uint8_t (*scales)[SCALE_STAGE_BYTES], int thread_idx) const {
    uint32_t const u = static_cast<uint32_t>(thread_idx);
    OperandBases b{};
    if constexpr (HAS_COMPRESSED) {
      uint32_t const r = blk_row(u), k = blk_blk(u), swap = out_swap(u);
      uint32_t const sc_base = cute::cast_smem_ptr_to_uint(&scales[0][0]);
      b.out0 = chunk_smem(sX, 0, int(r), 2 * k + swap);
      b.out1 = chunk_smem(sX, 0, int(r), 2 * k + 1 - swap);
      // Aligned copy destinations; the expansion's half order (swap) is applied
      // per pair in expand_bases, not folded here (the cp.async needs the aligned
      // address, and an item-invariant swapped copy was what ptxas spilled).
      b.land8 = chunk_smem(sX, 0, int(r), CHUNKS_PER_BLOCK + k);
      b.land4 = chunk_smem(sX, 0, int(r), CHUNKS_PER_BLOCK + k / 2 + 4 * (r & 1u)) + 8 * (k & 1u);
      // Row r's 8 B slot; this thread's word is the one holding block k (k / 4).
      // Lane k == 0 copies the slot (its sc_rd is the slot base), all eight read.
      b.sc_rd = sc_base + r * SCALE_ROW_BYTES + 4 * (k >> 2);
    }
    if constexpr (HAS_A16) {
      uint32_t const a_r = u / CHUNKS_PER_ROW, a_c = u % CHUNKS_PER_ROW;
      int64_t const ts = isK ? p.k_token_stride : p.v_token_stride;
      DTypeKV const* base = (isK ? p.K_ptr : p.V_ptr) +
                            int64_t(kv_head_idx) * (isK ? p.k_head_stride : p.v_head_stride);
      DTypeKV const* src0 = base + int64_t(a_r) * ts + a_c * CHUNK_ELEMS;
      b.a16_src0 = reinterpret_cast<uint8_t const*>(src0);
      b.a16_src1 = reinterpret_cast<uint8_t const*>(src0 + 8 * ts);
      b.a16_ps = isK ? p.k_page_stride_b : p.v_page_stride_b;
      b.a16_dst = chunk_smem(sX, 0, int(a_r), a_c);
    }
    if constexpr (HAS_FP8) {
      auto const& span = p.transport.formats[kTagFP8];
      b.p8 = compressed_base(isK ? span.k_payload : span.v_payload, span.payload_stride, kv_head_idx,
                             blk_row(u), blk_blk(u) * 16u);
      b.s8 = compressed_base(isK ? span.k_scales : span.v_scales, span.scale_stride, kv_head_idx,
                             blk_row(u), 0u);
      b.p8_ps = span.payload_stride.page;
      b.s8_ps = span.scale_stride.page;
      float const g = *(isK ? span.k_global_scale : span.v_global_scale);
      if constexpr (mixed_detail::kFp8FoldsPow2<DTypeKV>) {
        // [24a] C9: the placed decode is x * 2^-120; the fold 2^120 goes into the
        // block scale.  Lower bound per operand: |g| >= 2^-117 keeps bf16_rn(s * g)
        // normal for every nonzero E4M3 scale; below it gs8 is +inf so that every
        // block's fold test (expand_block) fails and the exact path is taken.  The
        // upper bound (|s * g| < 255.5) is per block and tested there.  g * 2^120
        // overflowing f32 (|g| >= 2^8) is +-inf and fails the same test.
        b.gs8 = fabsf(g) >= mixed_detail::kFp8FoldMinGlobal
                    ? g * mixed_detail::kFp8Fold
                    : __uint_as_float(mixed_detail::kFp8FoldSentinelBits);
      } else {
        b.gs8 = g;  // f16 decodes directly (cvt is exact); no fold, no test
      }
    }
    if constexpr (HAS_FP4) {
      auto const& span = p.transport.formats[kTagFP4];
      b.p4 = compressed_base(isK ? span.k_payload : span.v_payload, span.payload_stride, kv_head_idx,
                             blk_row(u), blk_blk(u) * 8u);
      b.s4 = compressed_base(isK ? span.k_scales : span.v_scales, span.scale_stride, kv_head_idx,
                             blk_row(u), 0u);
      b.p4_ps = span.payload_stride.page;
      b.s4_ps = span.scale_stride.page;
      float const g = *(isK ? span.k_global_scale : span.v_global_scale);
      if constexpr (mixed_detail::kFp8FoldsPow2<DTypeKV>) {
        // [25] C16: the placed E2M1 decode is x * 2^-126; the fold 2^126 goes
        // into the block scale under the same lower-bound sentinel as fp8 (|g|
        // >= 2^-117 keeps bf16_rn(s g) normal for every nonzero E4M3 scale); the
        // upper bound |s g| < 3.9921875 is tested per block and voted per
        // operand (expand_operand).  g * 2^126 is exact (power of two) or +-inf
        // (|g| >= 4), which fails the test.
        b.gs4 = fabsf(g) >= mixed_detail::kFp8FoldMinGlobal
                    ? g * mixed_detail::kFp4Fold
                    : __uint_as_float(mixed_detail::kFp8FoldSentinelBits);
      } else {
        b.gs4 = g;  // f16 decodes by LUT; no fold, no test
      }
    }
    return b;
  }

  // [25] C13: a compressed format's per-item 64-bit base for this thread: the
  // span pointer + head * hs + row * ts + blk_off (bytes), computed once per work
  // item in make_bases (F24 recomputed it per tile: the IADD3.X / VIADD chains
  // that topped the producer's stall samples).
  template <typename Ptr>
  CUTLASS_DEVICE static uint64_t compressed_base(Ptr ptr, KVPageByteStrides const& st,
                                                 int kv_head_idx, uint32_t row, uint32_t blk_off) {
    return reinterpret_cast<uint64_t>(ptr) + uint64_t(kv_head_idx) * uint64_t(st.head) +
           uint64_t(row) * uint64_t(st.token) + uint64_t(blk_off);
  }
  // Page `page` of a per-item base: one IMAD.WIDE.U32 (the host bounds the page
  // stride to 32 bits of bytes).
  CUTLASS_DEVICE static void const* page_src(uint64_t base, uint32_t page, uint32_t page_stride) {
    return reinterpret_cast<void const*>(base + uint64_t(page) * uint64_t(page_stride));
  }

  // Page placement of one copy / expansion: the byte offset of the page's region
  // from the operand base (copies and static expansion: j * PAGE_REGION_BYTES, an
  // immediate; dynamic expansion loops: i * PAGE_REGION_BYTES, one IMAD), the
  // same for the scale slot, and the page's first token for the D4 predicate.
  struct PagePos {
    uint32_t off;     // + PAGE_REGION_BYTES multiple
    uint32_t sc_off;  // + SCALE_PAGE_BYTES multiple
    uint32_t tok0;    // first token of the page in the tile
  };
  CUTLASS_DEVICE static PagePos static_page(uint32_t j) {
    return {j * PAGE_REGION_BYTES, j * SCALE_PAGE_BYTES, j * TOKENS_PER_PAGE};
  }
  CUTLASS_DEVICE static PagePos dynamic_page(uint32_t i) {
    return {i * PAGE_REGION_BYTES, i * SCALE_PAGE_BYTES, i * TOKENS_PER_PAGE};
  }

  // One page's copies.  FULL: the tile has CTA_KV valid tokens (no predicates).
  template <bool FULL>
  CUTLASS_DEVICE void copy_a16_page(OperandBases const& b, uint32_t page, PagePos const& pp,
                                    uint32_t dst_stage, uint32_t valid, int thread_idx) const {
    uint32_t const t = static_cast<uint32_t>(thread_idx);
    uint32_t const a_r = t / CHUNKS_PER_ROW;
    uint8_t const* s0 = b.a16_src0 + uint64_t(page) * uint64_t(b.a16_ps);
    uint8_t const* s1 = b.a16_src1 + uint64_t(page) * uint64_t(b.a16_ps);
    uint32_t const d = dst_stage + pp.off;
    if constexpr (FULL) {
      mixed_detail::cp16(d, s0);
      mixed_detail::cp16(d + ATOM_BYTES, s1);
    } else {
      bool const v0 = pp.tok0 + a_r < valid;
      bool const v1 = pp.tok0 + a_r + 8 < valid;
      mixed_detail::cp16_zfill(d, v0 ? s0 : b.a16_src0, v0);
      mixed_detail::cp16_zfill(d + ATOM_BYTES, v1 ? s1 : b.a16_src0, v1);
    }
  }

  // Compressed page: the block's packed bytes by every lane; the row's 8 B of
  // scales by the row's lane b == 0 only ([24b], C11: four active lanes per warp
  // copying four distinct 8 B sources into 32 contiguous bytes, instead of
  // eight 4 B copies per row at 7.95 wavefronts per warp instruction).  The
  // predicated-off lanes issue nothing (the instruction count is unchanged).
  // Data flow ([25] C13): src = base + page * stride (one IMAD.WIDE.U32 each for
  // the block and for the scale row) from the per-item OperandBases; the
  // destinations are thread constants + stage * STAGE_BYTES + pp.off (immediate).
  // FP8 blocks are 16 B copies (16 B aligned by the host check), FP4 blocks and
  // scale rows 8 B copies (8 B aligned only: odd blocks / odd heads are not 16 B
  // aligned, so no 16 B form with src-size 8 - design 3.4).
  // D4 (partial arm, !FULL): rows past `valid` copy with src-size 0, zero-filling
  // block and slot.  The source is passed unmodified: base + page * stride is an
  // in-bounds address of the transport tensor for every (page, row) (pages past
  // kv_len are page 0), and with src-size 0 no byte is read - the discipline of
  // CUTLASS's cp_async_zfill, which never touches the pointer.
  template <uint8_t FORMAT, bool FULL>
  CUTLASS_DEVICE void copy_compressed_page(OperandBases const& b, uint32_t page, PagePos const& pp,
                                           uint32_t stage, uint32_t valid, int thread_idx) const {
    uint32_t const u = static_cast<uint32_t>(thread_idx);
    bool const scale_leader = blk_blk(u) == 0;
    constexpr bool FP8 = FORMAT == kTagFP8;
    void const* src = page_src(FP8 ? b.p8 : b.p4, page, FP8 ? b.p8_ps : b.p4_ps);
    void const* ssrc = page_src(FP8 ? b.s8 : b.s4, page, FP8 ? b.s8_ps : b.s4_ps);
    uint32_t const land = (FP8 ? b.land8 : b.land4) + stage * STAGE_BYTES + pp.off;
    uint32_t const sdst = b.sc_rd + stage * SCALE_STAGE_BYTES + pp.sc_off;
    if constexpr (FULL) {
      if constexpr (FP8) {
        mixed_detail::cp16(land, src);
      } else {
        mixed_detail::cp8(land, src);
      }
      if (scale_leader) mixed_detail::cp8(sdst, ssrc);
    } else {
      bool const v = pp.tok0 + blk_row(u) < valid;
      if constexpr (FP8) {
        mixed_detail::cp16_zfill(land, src, v);
      } else {
        mixed_detail::cp8_zfill(land, src, v);
      }
      if (scale_leader) mixed_detail::cp8_zfill(sdst, ssrc, v);
    }
  }

  // [25d] One page of the dynamic module (C17).  Data flow: the page's format is
  // two mask bits (p8, p4; A16 = neither), warp-uniform data from the chunk
  // table.  Six source addresses by IMAD.WIDE.U32 from the six per-item bases
  // (A16 rows u/16 and u/16 + 8, FP8 block, FP8 scale row, FP4 block, FP4 scale
  // row) and six predicated cp.async of which exactly two execute:
  //   @pa  16 B  row u/16       -> a16_dst          @pa  16 B  row u/16 + 8 -> a16_dst + ATOM
  //   @p8  16 B  FP8 block      -> land8            @(p8 & leader)  8 B  FP8 scales -> sc_rd
  //   @p4   8 B  FP4 block      -> land4            @(p4 & leader)  8 B  FP4 scales -> sc_rd
  // A predicated-off copy issues nothing (no wavefront, no zero-fill), so the
  // seven non-leader lanes never touch the row's scale slot.  Destinations are
  // thread constants + stage * bytes + j * PAGE_REGION_BYTES (immediates).
  // Control flow: none (no loop, no per-page branch, no select on an address).
  // Partial arm (!FULL, the two per-item calls): the D4 predicate per copied row
  // (A16 rows a_r, a_r + 8; compressed row r) becomes the src-size operand
  // (16 / 8 or 0); the source is passed unmodified (in bounds; src-size 0 reads
  // nothing).
  template <bool FULL>
  CUTLASS_DEVICE void copy_dynamic_page(OperandBases const& b, uint32_t page, uint32_t j, bool p8,
                                        bool p4, uint32_t stage, uint32_t valid,
                                        int thread_idx) const {
    uint32_t const u = static_cast<uint32_t>(thread_idx);
    bool const pa = !(p8 || p4);
    bool const leader = blk_blk(u) == 0;
    PagePos const pp = static_page(j);
    void const* a0 = b.a16_src0 + uint64_t(page) * uint64_t(b.a16_ps);
    void const* a1 = b.a16_src1 + uint64_t(page) * uint64_t(b.a16_ps);
    void const* c8 = page_src(b.p8, page, b.p8_ps);
    void const* s8 = page_src(b.s8, page, b.s8_ps);
    void const* c4 = page_src(b.p4, page, b.p4_ps);
    void const* s4 = page_src(b.s4, page, b.s4_ps);
    uint32_t const dA = b.a16_dst + stage * STAGE_BYTES + pp.off;
    uint32_t const d8 = b.land8 + stage * STAGE_BYTES + pp.off;
    uint32_t const d4 = b.land4 + stage * STAGE_BYTES + pp.off;
    uint32_t const ds = b.sc_rd + stage * SCALE_STAGE_BYTES + pp.sc_off;
    int n0 = 16, n1 = 16, nc16 = 16, nc8 = 8;
    if constexpr (!FULL) {
      uint32_t const a_r = u / CHUNKS_PER_ROW;
      n0 = pp.tok0 + a_r < valid ? 16 : 0;
      n1 = pp.tok0 + a_r + 8 < valid ? 16 : 0;
      bool const vc = pp.tok0 + blk_row(u) < valid;
      nc16 = vc ? 16 : 0;
      nc8 = vc ? 8 : 0;
    } else {
      (void)valid;
    }
    mixed_detail::cp16_pred(dA, a0, pa, n0);
    mixed_detail::cp16_pred(dA + ATOM_BYTES, a1, pa, n1);
    mixed_detail::cp16_pred(d8, c8, p8, nc16);
    mixed_detail::cp8_pred(ds, s8, p8 && leader, nc8);
    mixed_detail::cp8_pred(d4, c4, p4, nc8);
    mixed_detail::cp8_pred(ds, s4, p4 && leader, nc8);
  }

  template <bool FULL>
  CUTLASS_DEVICE void issue_tile_copies(Params const& p, OperandBases const& b, TileRegs const& m,
                                        uint32_t stage, bool isK, int kv_head_idx,
                                        int thread_idx) const {
    (void)p; (void)isK; (void)kv_head_idx;
    uint32_t const valid = m.valid();
    if constexpr (!DYNAMIC) {
      // Static modules: unrolled over the six tile pages - the body is a handful
      // of instructions per page with immediate destinations; rolled loop
      // control cost as much as the copies.
      uint32_t const a16_dst_stage = b.a16_dst + stage * STAGE_BYTES;
#pragma unroll
      for (uint32_t j = 0; j < PAGES_PER_THREAD; ++j) {
        uint32_t const page = m.page(j);
        PagePos const pp = static_page(j);
        if constexpr (STATIC_A16) {
          copy_a16_page<FULL>(b, page, pp, a16_dst_stage, valid, thread_idx);
        } else if constexpr (STATIC_FP8) {
          copy_compressed_page<kTagFP8, FULL>(b, page, pp, stage, valid, thread_idx);
        } else {
          copy_compressed_page<kTagFP4, FULL>(b, page, pp, stage, valid, thread_idx);
        }
      }
    } else {
      // [25d] Dynamic module (C17): six unrolled pages, format by predication.
      uint32_t const m8 = m.mask8();
      uint32_t const m4 = m.mask4();
#pragma unroll
      for (uint32_t j = 0; j < PAGES_PER_TILE; ++j) {
        copy_dynamic_page<FULL>(b, m.page(j), j, ((m8 >> j) & 1u) != 0u, ((m4 >> j) & 1u) != 0u,
                                stage, valid, thread_idx);
      }
    }
  }

  // ------------------------------------------------------------------------
  // [23] Expansion: the copy owner decodes its block.  Thread (r, b) reads the
  // packed bytes it copied (chunk 2b of row r), its 4 B scale word, and stores
  // the block's 16 A16 values to chunks 2b, 2b+1 with STS.128 at immediate
  // offsets from per-stage 32-bit bases (D3: it reads only what it wrote and
  // writes only what it owns, so no thread waits on any other thread).
  //
  // Per FP8 block ([25], bf16): 2 LDS.64 + LDS.32 + scale chain 5 (PRMT,
  // F2FP.E4M3, HADD2, FMUL, FSETP) + F2FP.PACK + 8 x (PRMT, SHF, LOP3) + 8 HMUL2
  // + 2 STS.128 = 43 on the fold path; the fold vote (PLOP3 tree, VOTE.ALL,
  // BRA) is once per operand (expand_operand), not per block; the cold arm
  // adds 8 HMUL2 by 2^120 per block and one reload of the global scale per
  // operand.  Per FP4 block: 2 LDS.32 + LDS.32 + 5 + 26 (placement) + 8 HMUL2
  // + 2 STS = 44; cold arm + 8 HMUL2 by 2^126.

  // E4M3 scale byte (byte sel of the word) -> the reference's float(scale).
  // One PTX block with 32-bit operands only: e4m3 -> f16 (exact) -> f32 (exact),
  // F2FP.F16.E4M3.UNPACK_B + HADD2.F32 in SASS.  A 16-bit C++ operand (`"h"`
  // constraint / __half) here made ptxas materialise the half through the frame
  // (STL + LDL.LU.S16 per block) in the register-tight dynamic module.
  CUTLASS_DEVICE static float scale_byte_f32(uint32_t scale_word, uint32_t sel) {
    uint32_t const byte = __byte_perm(scale_word, 0u, sel);
    float f;
    asm("{\n"
        " .reg .b16 b0, b1, h0, h1;\n"
        " .reg .b32 h2;\n"
        " mov.b32 {b0, b1}, %1;\n"
        " cvt.rn.f16x2.e4m3x2 h2, b0;\n"
        " mov.b32 {h0, h1}, h2;\n"
        " cvt.f32.f16 %0, h0;\n"
        "}"
        : "=f"(f)
        : "r"(byte));
    return f;
  }
  // f32 -> A16 broadcast to both halves (one rounding: the reference's
  // static_cast<A16>(float(scale) * global)).
  CUTLASS_DEVICE static uint32_t a16x2_broadcast(float v) {
    if constexpr (std::is_same_v<DTypeKV, cutlass::half_t>) {
      __half2 const r = __floats2half2_rn(v, v);
      return reinterpret_cast<uint32_t const&>(r);
    } else {
      __nv_bfloat162 const r = __floats2bfloat162_rn(v, v);
      return reinterpret_cast<uint32_t const&>(r);
    }
  }
  // The operand's plain (unfolded) global scale, read from the grid-constant
  // parameters: used on the cold path only, so it is not held in a register.
  template <bool FP8>
  CUTLASS_DEVICE static float global_plain(Params const& p, bool isK) {
    auto const& span = p.transport.formats[FP8 ? kTagFP8 : kTagFP4];
    return *(isK ? span.k_global_scale : span.v_global_scale);
  }
  // bf16 modules fold a power of two into the block scale (C9 fp8, C16 fp4).
  static constexpr bool FOLDS = mixed_detail::kFp8FoldsPow2<DTypeKV>;
  // Per-block fold test: the folded scale product is finite and normal after the
  // fold (false for inf / NaN, i.e. for the +inf sentinel).  The same f32
  // constant serves both formats (255.5 * 2^120 == 3.9921875 * 2^126).
  CUTLASS_DEVICE static bool fold_ok(float v) { return fabsf(v) < mixed_detail::kFp8FoldMax; }
  // The operand's scale product for block scale byte `sel` of word `sw`:
  // f32(s) * gs (gs = g * 2^k or the +inf sentinel; f16: g), mul.rn without ftz.
  template <bool FP8>
  CUTLASS_DEVICE static float scale_product(OperandBases const& b, uint32_t sw, uint32_t t) {
    return mixed_detail::mul_rn_f32_denorm(scale_byte_f32(sw, sc_sel(t)), FP8 ? b.gs8 : b.gs4);
  }

  // Block layouts: FP8 pair 2j, 2j+1 = low, high halves of w[j] (the quantizer's
  // layout); FP4 nibbles 0..7 in v.x, 8..15 in v.y, low nibble = even coefficient.
  // The decode is inlined in expand_block.

  // Per-stage 32-bit bases of one operand's expansion (thread constants + stage).
  // [24b] l8a / l8b (l4a / l4b) are the two halves of the landing in the order the
  // outputs are stored: first half at + 8 swap (FP4: + 4 swap), second at the
  // other half - the landing chunk is 16 B (8 B) aligned, so it is the XOR.
  struct ExpandBases {
    uint32_t d0, d1;      // outputs (chunk 2b + swap, 2b + 1 - swap)
    uint32_t l8a, l8b;    // FP8 landing halves (8 B each)
    uint32_t l4a, l4b;    // FP4 landing halves (4 B each)
    uint32_t sc;          // scale word
  };
  // Per pair: four IADD (stage offsets), two LOP3 (first halves: the swap term
  // is OR-ed onto the stage-dependent address - land is 16 B / 8 B aligned and
  // STAGE_BYTES a multiple of 16, so | equals +, but ptxas cannot reassociate an
  // OR with the add and hoist `land + 8 swap` into an item-invariant register:
  // that hoisting, four registers per operand, was what it spilled in the
  // dynamic module) and two LOP3 (second halves, ^ 8 / ^ 4).
  CUTLASS_DEVICE static ExpandBases expand_bases(OperandBases const& b, uint32_t stage,
                                                 uint32_t t) {
    static_assert(STAGE_BYTES % 16 == 0, "stage offsets keep the half bits");
    uint32_t const so = stage * STAGE_BYTES;
    uint32_t const swap = out_swap(t);
    uint32_t const l8a = (b.land8 + so) | (8u * swap);
    uint32_t const l4a = (b.land4 + so) | (4u * swap);
    return {b.out0 + so, b.out1 + so, l8a, l8a ^ 8u, l4a, l4a ^ 4u,
            b.sc_rd + stage * SCALE_STAGE_BYTES};
  }

  // Packed input of this thread's block of tile page j.
  // Loaded one page ahead of the stores (the second output base d1 is not
  // provably disjoint from d0 / l4 for ptxas, so a load issued after a store
  // would not be hoisted above it).  Two half loads in store order ([24b]):
  // wavefront-neutral against the [23] LDS.128 / LDS.64 (16-lane LDS.64 phases:
  // rows r, r+1 read complementary halves of complementary chunk sets - 32
  // distinct banks; LDS.32: rows r, r+2 share a chunk set with complementary
  // (b&1, swap) bank pairs - 32 distinct banks), one more instruction per block.
  struct Packed {
    uint4 w;  // FP8: 16 B (first stored half in x, y); FP4: 8 B in x, y (first stored word in x)
  };
  // `off` is the page's byte offset from the stage base (PagePos::off).
  template <bool FP8>
  CUTLASS_DEVICE static Packed load_packed(ExpandBases const& e, uint32_t off) {
    Packed p;
    if constexpr (FP8) {
      uint2 const a = mixed_detail::lds64(e.l8a + off);
      uint2 const c = mixed_detail::lds64(e.l8b + off);
      p.w = uint4{a.x, a.y, c.x, c.y};
    } else {
      uint32_t const x = mixed_detail::lds32(e.l4a + off);
      uint32_t const y = mixed_detail::lds32(e.l4b + off);
      p.w = uint4{x, y, 0u, 0u};
    }
    return p;
  }
  // One block, given its scale factor.  Data flow (bf16): the placed decode lo,
  // hi = x * 2^-k (k = 120 fp8, 126 fp4) is independent of the scale; EXACT
  // (the cold arm) first multiplies the halves by 2^k (exact: the placed values
  // times 2^k are the E4M3 / E2M1 magnitudes, subnormal placements included) so
  // that sf2 = bf16x2(f32(s) * g) applies to the true values; the hot arm
  // applies sf2 = bf16x2(f32(s) * g * 2^k) to the placed values.  Either way one
  // rounding multiply per pair - the reference's arithmetic (C9 / C16).  f16
  // (no fold): EXACT is never instantiated.  Control flow: none.  No barrier
  // here: the caller (expand_operand) has issued every lane's landing loads of
  // the operand and met the operand's __syncwarp before any block is stored
  // (D3 / A7 / A9).  `off` is the page's byte offset from the stage base.
  template <bool FP8, bool EXACT>
  CUTLASS_DEVICE static void expand_block(ExpandBases const& e, Packed const& p, uint32_t sf2,
                                          uint32_t off) {
#if defined(MIXED_FA3_CONTROL_SKIP_EXPAND)
    // Timing control (design 2C): no decode, no stores; values are garbage.
    (void)e; (void)p; (void)sf2; (void)off;
    return;
#elif defined(MIXED_FA3_CONTROL_RAW_STS)
    // Timing control (design 2C): the STS wavefronts without the decode.
    (void)sf2;
    mixed_detail::sts128(e.d0 + off, p.w);
    mixed_detail::sts128(e.d1 + off, p.w);
    return;
#endif
    uint4 lo, hi;
    if constexpr (FP8) {
      mixed_detail::e4m3x4_to_a16<DTypeKV>(p.w.x, lo.x, lo.y);
      mixed_detail::e4m3x4_to_a16<DTypeKV>(p.w.y, lo.z, lo.w);
      mixed_detail::e4m3x4_to_a16<DTypeKV>(p.w.z, hi.x, hi.y);
      mixed_detail::e4m3x4_to_a16<DTypeKV>(p.w.w, hi.z, hi.w);
    } else {
      uint32_t a[4], c[4];
      mixed_detail::e2m1x8_to_a16<DTypeKV>(p.w.x, a);
      mixed_detail::e2m1x8_to_a16<DTypeKV>(p.w.y, c);
      lo = uint4{a[0], a[1], a[2], a[3]};
      hi = uint4{c[0], c[1], c[2], c[3]};
    }
    if constexpr (EXACT) {
      static_assert(FOLDS, "the exact arm exists only where a power of two is folded");
      constexpr uint32_t k2 = FP8 ? mixed_detail::kTwoPow120Bf16x2 : mixed_detail::kTwoPow126Bf16x2;
      lo.x = mixed_detail::mul_a16x2<DTypeKV>(lo.x, k2);
      lo.y = mixed_detail::mul_a16x2<DTypeKV>(lo.y, k2);
      lo.z = mixed_detail::mul_a16x2<DTypeKV>(lo.z, k2);
      lo.w = mixed_detail::mul_a16x2<DTypeKV>(lo.w, k2);
      hi.x = mixed_detail::mul_a16x2<DTypeKV>(hi.x, k2);
      hi.y = mixed_detail::mul_a16x2<DTypeKV>(hi.y, k2);
      hi.z = mixed_detail::mul_a16x2<DTypeKV>(hi.z, k2);
      hi.w = mixed_detail::mul_a16x2<DTypeKV>(hi.w, k2);
    }
    lo.x = mixed_detail::mul_a16x2<DTypeKV>(lo.x, sf2);
    lo.y = mixed_detail::mul_a16x2<DTypeKV>(lo.y, sf2);
    lo.z = mixed_detail::mul_a16x2<DTypeKV>(lo.z, sf2);
    lo.w = mixed_detail::mul_a16x2<DTypeKV>(lo.w, sf2);
    hi.x = mixed_detail::mul_a16x2<DTypeKV>(hi.x, sf2);
    hi.y = mixed_detail::mul_a16x2<DTypeKV>(hi.y, sf2);
    hi.z = mixed_detail::mul_a16x2<DTypeKV>(hi.z, sf2);
    hi.w = mixed_detail::mul_a16x2<DTypeKV>(hi.w, sf2);
    mixed_detail::sts128(e.d0 + off, lo);
    mixed_detail::sts128(e.d1 + off, hi);
  }

  // The six blocks of one operand (static modules), one arm.  EXACT selects the
  // cold (reference two-multiply) form; `pk` / `sw` are the landed words.
  template <bool FP8, bool EXACT>
  CUTLASS_DEVICE void expand_static_arm(Params const& prm, bool isK, ExpandBases const& e,
                                        OperandBases const& b, Packed const (&pk)[PAGES_PER_THREAD],
                                        uint32_t const (&sw)[PAGES_PER_THREAD],
                                        float const (&v)[PAGES_PER_THREAD], uint32_t t) const {
    (void)b;
    if constexpr (EXACT) {
      (void)v;
      // The cold arm's global scale: one LDC + LDG per operand, off the hot path.
      float const g = global_plain<FP8>(prm, isK);
#pragma unroll
      for (uint32_t j = 0; j < PAGES_PER_THREAD; ++j) {
        uint32_t const sf2 = a16x2_broadcast(
            mixed_detail::mul_rn_f32_denorm(scale_byte_f32(sw[j], sc_sel(t)), g));
        expand_block<FP8, true>(e, pk[j], sf2, static_page(j).off);
      }
    } else {
      (void)prm; (void)isK; (void)sw;
#pragma unroll
      for (uint32_t j = 0; j < PAGES_PER_THREAD; ++j) {
        expand_block<FP8, false>(e, pk[j], a16x2_broadcast(v[j]), static_page(j).off);
      }
    }
  }

  // One operand's pending tile.  `tv` is the tile's pending word.
  //
  // Static modules ([25], A9 / C12).  Data flow, in program order:
  //   (1) 6 x load_packed: this thread's own landings of the six pages (its own
  //       cp.async wrote them; its cp.async.wait_group made them visible to it);
  //   (2) __syncwarp: every lane of the warp is past its wait, so lane 0's
  //       landed 8 B scale slot is ordered before the other seven lanes' LDS.32
  //       of it (D3 as amended by A8), and every lane's landing loads of (1)
  //       are ordered before any lane's stores of (5) (the landing chunk is
  //       another lane's output chunk, A7);
  //   (3) 6 x LDS.32 scale words, 6 scale chains v_j = f32(s_j) * gs;
  //   (4) VOTE: one warp vote on AND_j fold_ok(v_j) (bf16 only), one uniform
  //       branch: hot arm (sf2_j = bf16x2(v_j)) or cold arm (2^k undone on the
  //       halves, sf2_j = bf16x2(f32(s_j) * g));
  //   (5) six block bodies of the chosen arm, each 8 HMUL2 + 2 STS.128.
  // Control flow: the only branch is the uniform one on the vote; the block
  // bodies are branch-free (C12: no VOTE / BRA / UMOV / register join inside a
  // body).  VOTE = false (the single-operand finish sites, F25c) compiles the
  // cold arm alone: reference-exact for every finite input, no vote.
  // f16 modules: no fold, the hot arm with the plain scale, no vote.
  //
  // Dynamic module ([25d]): see the arm below.
  template <bool VOTE>
  CUTLASS_DEVICE void expand_operand(Params const& prm, bool isK, OperandBases const& b,
                                     uint32_t tv, int stage, uint32_t t) const {
    ExpandBases const e = expand_bases(b, uint32_t(stage), t);
    if constexpr (!DYNAMIC) {
      (void)tv;  // the static modules' pending word carries only the stage
      constexpr bool FP8 = STATIC_FP8;
      constexpr uint32_t N = PAGES_PER_THREAD;
      Packed pk[N];
#pragma unroll
      for (uint32_t j = 0; j < N; ++j) pk[j] = load_packed<FP8>(e, static_page(j).off);
      __syncwarp();
      uint32_t sw[N];
#pragma unroll
      for (uint32_t j = 0; j < N; ++j) sw[j] = mixed_detail::lds32(e.sc + static_page(j).sc_off);
      float v[N];
#pragma unroll
      for (uint32_t j = 0; j < N; ++j) v[j] = scale_product<FP8>(b, sw[j], t);
      if constexpr (FOLDS && VOTE) {
        bool ok = fold_ok(v[0]);
#pragma unroll
        for (uint32_t j = 1; j < N; ++j) ok = ok && fold_ok(v[j]);
        if (__all_sync(0xFFFFFFFFu, ok)) {
          expand_static_arm<FP8, false>(prm, isK, e, b, pk, sw, v, t);
        } else {
          expand_static_arm<FP8, true>(prm, isK, e, b, pk, sw, v, t);
        }
      } else if constexpr (FOLDS) {
        expand_static_arm<FP8, true>(prm, isK, e, b, pk, sw, v, t);
      } else {
        expand_static_arm<FP8, false>(prm, isK, e, b, pk, sw, v, t);
      }
    } else {
      // [25d] Dynamic module (C10 / C17).  Data flow: (1) __syncwarp - every
      // lane is past its cp.async.wait_group, so lane 0's landed scale slots of
      // all six pages are readable by the row's lanes from here on (A9); (2) the
      // six scale words at immediate offsets and, per page, f32(s_j) once, its
      // two products with gs8 / gs4 and the two fold tests, each masked by the
      // page's format bit (A16 pages and the other format's pages do not vote;
      // their slots hold stale bytes, which is harmless); (3) one VOTE.ALL per
      // format and one uniform branch per format choosing the hot or the exact
      // loop body; (4) the format loops (expand_format_pages), two pages per
      // step.  Control flow: two uniform branches outside the loops; the loop
      // bodies carry only their back-edge.  f16 (no fold): the hot bodies.
      __syncwarp();
      uint32_t const m8 = pending_mask8(tv);
      uint32_t const m4 = pending_mask4(tv);
      if constexpr (FOLDS) {
        constexpr uint32_t N = PAGES_PER_TILE;
        uint32_t sw[N];
#pragma unroll
        for (uint32_t j = 0; j < N; ++j) sw[j] = mixed_detail::lds32(e.sc + static_page(j).sc_off);
        bool ok8 = true, ok4 = true;
#pragma unroll
        for (uint32_t j = 0; j < N; ++j) {
          float const f = scale_byte_f32(sw[j], sc_sel(t));
          bool const in8 = ((m8 >> j) & 1u) != 0u;
          bool const in4 = ((m4 >> j) & 1u) != 0u;
          ok8 = ok8 && (!in8 || fold_ok(mixed_detail::mul_rn_f32_denorm(f, b.gs8)));
          ok4 = ok4 && (!in4 || fold_ok(mixed_detail::mul_rn_f32_denorm(f, b.gs4)));
        }
        bool const hot8 = __all_sync(0xFFFFFFFFu, ok8);
        bool const hot4 = __all_sync(0xFFFFFFFFu, ok4);
        if (hot8) {
          expand_format_pages<true, false>(prm, isK, e, b, m8, t);
        } else {
          expand_format_pages<true, true>(prm, isK, e, b, m8, t);
        }
        if (hot4) {
          expand_format_pages<false, false>(prm, isK, e, b, m4, t);
        } else {
          expand_format_pages<false, true>(prm, isK, e, b, m4, t);
        }
      } else {
        expand_format_pages<true, false>(prm, isK, e, b, m8, t);
        expand_format_pages<false, false>(prm, isK, e, b, m4, t);
      }
    }
  }

  // [25d] One format's pages of a pending tile (dynamic module), one arm (EXACT
  // = the cold two-multiply form).  `m` is the page mask (warp-uniform).  Data
  // flow per step: the two current pages' scale words (LDS.32 at i * 512; the
  // slots are readable: the operand's __syncwarp precedes this loop) and scale
  // factors, the next step's landing loads (this thread's own copies; issued
  // before this step's stores so their latency hides under the decode), then
  // __syncwarp, then the two block bodies.  Hazards: the landing chunk a lane
  // reads is another lane's output chunk of the same page; every lane's loads
  // of a page are issued at or before the step preceding its stores and the
  // step's __syncwarp orders them before any lane's stores (A7).  Different
  // pages are disjoint.  An odd page count makes the last step's second page
  // equal its first: the duplicate decode stores the same values to the same
  // chunks (idempotent) - no branch on the page count; when no page follows,
  // the "next" loads re-read the current pages (harmless: before their stores).
  // Control flow: one rolled loop, back-edge only; the FP8 and FP4 loops touch
  // disjoint pages.
  template <bool FP8, bool EXACT>
  CUTLASS_DEVICE void expand_format_pages(Params const& prm, bool isK, ExpandBases const& e,
                                          OperandBases const& b, uint32_t m, uint32_t t) const {
    if (m == 0) return;  // warp-uniform
    float g = 0.f;
    if constexpr (EXACT) g = global_plain<FP8>(prm, isK);
    (void)prm; (void)isK;
    uint32_t i0 = __ffs(m) - 1;
    m &= m - 1;
    uint32_t i1 = m ? __ffs(m) - 1 : i0;
    m &= m - 1;
    Packed c0 = load_packed<FP8>(e, dynamic_page(i0).off);
    Packed c1 = load_packed<FP8>(e, dynamic_page(i1).off);
#pragma unroll 1
    for (;;) {
      bool const more = m != 0;
      uint32_t const n0 = more ? __ffs(m) - 1 : i0;
      uint32_t mm = m & (m - 1);
      uint32_t const n1 = mm ? __ffs(mm) - 1 : n0;
      mm &= mm - 1;
      uint32_t const sw0 = mixed_detail::lds32(e.sc + dynamic_page(i0).sc_off);
      uint32_t const sw1 = mixed_detail::lds32(e.sc + dynamic_page(i1).sc_off);
      Packed const x0 = load_packed<FP8>(e, dynamic_page(n0).off);
      Packed const x1 = load_packed<FP8>(e, dynamic_page(n1).off);
      uint32_t sf0, sf1;
      if constexpr (EXACT) {
        sf0 = a16x2_broadcast(mixed_detail::mul_rn_f32_denorm(scale_byte_f32(sw0, sc_sel(t)), g));
        sf1 = a16x2_broadcast(mixed_detail::mul_rn_f32_denorm(scale_byte_f32(sw1, sc_sel(t)), g));
      } else {
        sf0 = a16x2_broadcast(scale_product<FP8>(b, sw0, t));
        sf1 = a16x2_broadcast(scale_product<FP8>(b, sw1, t));
      }
      __syncwarp();  // every lane's loads of pages i0, i1 before any lane's stores of them
      expand_block<FP8, EXACT>(e, c0, sf0, dynamic_page(i0).off);
      expand_block<FP8, EXACT>(e, c1, sf1, dynamic_page(i1).off);
      if (!more) break;
      i0 = n0;
      i1 = n1;
      c0 = x0;
      c1 = x1;
      m = mm;
    }
  }

  // ------------------------------------------------------------------------
  // One operand of a tile.  Two completion modes (C4):
  //  * no compressed page: every thread commits with cp.async.mbarrier.arrive;
  //    the stage completes when the copies land.  Nobody waits.
  //  * otherwise: the operand becomes *pending*; one pair later each thread waits
  //    for its own copies of that commit group, expands its blocks, fences (D5)
  //    and arrives.  No group barrier: a thread reads only bytes it copied.
  // A pending record is one 32-bit word: the tile's masks / tags, valid, flag
  // bits and the stage index (TileRegs::pending_word); 0 means inactive (a
  // filled row always has kFlagFilled set).
  struct Operand {
    MainloopPipeline* pipeline;
    PipelineState* state;
    uint8_t (*scales)[SCALE_STAGE_BYTES];
    bool isK;
    OperandBases bases;
    uint32_t pending;  // finished (waited, expanded, committed) this iteration
    uint32_t staged;   // issued this iteration; becomes `pending` after the finish
  };

  // [25] C13: PARTIAL is a compile-time tag.  Only tile kv_tile_idx can have
  // valid < CTA_KV (valid = min(CTA_KV, kv_len - tile * CTA_KV) and kv_tile_idx *
  // CTA_KV < kv_len since num_kv_tiles <= ceil(kv_len / CTA_KV)); its K is issued
  // by the K(last)-alone call and its V by the peeled first pair, which are the
  // two PARTIAL call sites; the loop compiles the FULL arm only.  The a16 module
  // keeps the [23] runtime test (its SASS is the transport control).
  template <bool PARTIAL>
  CUTLASS_DEVICE void issue_operand(Params const& p, Operand& op, TileRegs const& m,
                                    int kv_head_idx, int thread_idx) const {
    uint32_t const stage = op.state->index();
    if constexpr (STATIC_A16) {
      if (m.valid() == uint32_t(CTA_KV)) {
        issue_tile_copies<true>(p, op.bases, m, stage, op.isK, kv_head_idx, thread_idx);
      } else {
        issue_tile_copies<false>(p, op.bases, m, stage, op.isK, kv_head_idx, thread_idx);
      }
    } else if constexpr (PARTIAL) {
      issue_tile_copies<false>(p, op.bases, m, stage, op.isK, kv_head_idx, thread_idx);
    } else {
      issue_tile_copies<true>(p, op.bases, m, stage, op.isK, kv_head_idx, thread_idx);
    }
    bool const compressed = HAS_COMPRESSED && (!DYNAMIC || m.any_compressed());
    if (compressed) {
      op.staged = m.pending_word(stage);
    } else {
      op.staged = 0;
      op.pipeline->producer_commit(*op.state, cutlass::arch::cpasync_barrier_arrive);
    }
    ++(*op.state);
  }

  // The tile issued this iteration is finished next iteration (one pair behind).
  CUTLASS_DEVICE static void rotate_pending(Operand& op) {
    op.pending = op.staged;
    op.staged = 0;
  }

#ifdef MIXED_FA3_TRACE
  // Diagnosis-only sub-stamps of finish_pending_pair (thread 0, %globaltimer ns):
  // t[0] after cp.async.wait_group, t[1] = t[0] (no barrier B since [23]),
  // t[2] after expand K, t[3] = t[2], t[4] after expand V, t[5] after the single
  // fence + both commits (reported as fcV).  An inactive operand leaves its two
  // stamps equal to the previous one.
  struct FinTrace {
    bool on;
    uint64_t t[6];
  };
#endif

  // Caller guarantees this thread's copies of the operand landed (cp.async wait).
  // `op` is always the K or the V local (explicit call sites), so op.isK is a
  // constant after inlining; it selects the cold path's global-scale reload only.
  // CHECK ([25] C14): the static compressed modules never call this with an
  // inactive operand (every issued tile is pending; the call sites are chosen by
  // the peel), so the test is compiled only for the dynamic module, whose
  // A16-only tiles are not pending; the word is warp-uniform data from smem.
  template <bool VOTE, bool CHECK>
  CUTLASS_DEVICE void expand_pending(Params const& prm, Operand const& op, int thread_idx) const {
    if constexpr (HAS_COMPRESSED) {
      if constexpr (CHECK) {
        if (op.pending == 0) return;
      }
      expand_operand<VOTE>(prm, op.isK, op.bases, op.pending, int(pending_stage(op.pending)),
                           uint32_t(thread_idx));
    }
  }
  // After the fence: arrive on the pending stage's full barrier.
  template <bool CHECK>
  CUTLASS_DEVICE void commit_pending(Operand& op) const {
    if constexpr (HAS_COMPRESSED) {
      if constexpr (CHECK) {
        if (op.pending == 0) return;
      }
      // PipelineAsync::producer_commit arrives on full_barrier[state.index()]; only the
      // index of the pending stage matters.
      op.pipeline->producer_commit(PipelineState(int(pending_stage(op.pending)), 0, 0));
      op.pending = 0;
    }
  }

  template <bool LEFT_SLIDING_WINDOW, typename Scheduler, typename BlockCoord>
  CUTLASS_DEVICE void load(Params const& mainloop_params, MainloopPipeline pipeline_k,
                           MainloopPipeline pipeline_v, PipelineState& smem_pipe_write_k,
                           PipelineState& smem_pipe_write_v, SharedStorage& shared_storage,
                           Scheduler& scheduler, typename Scheduler::Params const& scheduler_params,
                           typename Scheduler::WorkTileInfo& work_tile_info,
                           BlockCoord const& block_coord, int work_idx,
                           const int num_kv_tiles_outside_items_window = 0,
                           const int num_kv_tiles_prefix = 0) {
    int const thread_idx = threadIdx.x;
    int const warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (thread_idx / 32) % 4, 0);
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});
    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.layout_Q.shape());

    auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
        block_coord;

    Tensor gQ = get_local_tile_tensor(mQ, select<0, 2>(TileShape_QKD{}), qo_head_idx, qo_indptr,
                                      qo_len)(_, _, q_tile_idx);
    Tensor sQ_x = make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
    Tensor gQ_x = make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));
    auto [tQgQ, tQsQ] = tma_partition(mainloop_params.tma_load_Q, _0{}, Layout<_1>{},
                                      group_modes<0, 2>(sQ_x), group_modes<0, 2>(gQ_x));

    int const num_kv_tiles = get_num_kv_tiles(mainloop_params, q_tile_idx, qo_len, kv_len);
    int const kv_tile_idx = num_kv_tiles - 1;
    int swa_begin_kv_tile_idx = 0;
    if constexpr (LEFT_SLIDING_WINDOW) {
      swa_begin_kv_tile_idx = get_swa_begin_kv_tile_idx<CTA_Q, CTA_KV>(mainloop_params.window_left,
                                                                       q_tile_idx, qo_len, kv_len);
    }
    IdType const* kv_indices_ptr = mainloop_params.kv_indices + kv_indptr;

    // [21] chunk table: entry e holds tile kv_tile_idx - e.
    TileMeta(*meta)[CHUNK_TILES] = shared_storage.mixed_meta;
    int const num_chunks = (kv_tile_idx - swa_begin_kv_tile_idx) / CHUNK_TILES + 1;
    auto group_barrier = [&]() {
      cutlass::arch::NamedBarrier::sync(NUM_COPY_THREADS,
                                        static_cast<int>(NamedBarriers::kProducerWG));
    };
    // The operand bases are filled in after the chunk-0 gather below: holding
    // both operands' thread constants across that gather's two dependent global
    // loads spilled four words in the dynamic module (C3).
    Operand K{&pipeline_k, &smem_pipe_write_k, shared_storage.mixed_scales_k, true, {}, 0, 0};
    Operand V{&pipeline_v, &smem_pipe_write_v, shared_storage.mixed_scales_v, false, {}, 0, 0};
#ifdef MIXED_FA3_TRACE
    FinTrace ft{false, {0, 0, 0, 0, 0, 0}};
#endif

    // [25] C14 finish sites.  Data flow: cp.async.wait_group (this thread's own
    // copies), the operands' expansions (each with its own warp barrier between
    // its landing loads and its scale-slot loads, A9), one fence.proxy.async
    // (D5), the full-barrier arrivals.  Control flow: three sites per work item -
    //  * finish_pending_pair, in the loop: the pair issued one iteration earlier.
    //    By the peel below that is always a (K, V) pair, so in the static
    //    compressed modules both operands are pending and nothing is tested
    //    (CHECK = false: no BSSY/BSYNC); the dynamic module tests each operand
    //    (A16-only tiles are not pending).  wait_group 1: this pair's group may
    //    stay in flight.
    //  * finish_one(K) after K(last) alone (C7: before barrier_O.wait) and
    //    finish_one(V) at the drain: the other operand is never pending there
    //    (V has not been issued yet / the last pair always has tK = -1), so only
    //    the named operand is finished, with wait_group 0 and the exact body
    //    (VOTE = false: no vote, half the code of a hot + cold site).
    // Explicit K/V call sites: selecting an Operand by a runtime reference would
    // force both structs into local memory (C2).
    auto finish_pending_pair = [&]() {
      if constexpr (HAS_COMPRESSED) {
        cutlass::arch::cp_async_wait<1>();
#ifdef MIXED_FA3_TRACE
        if (ft.on) ft.t[0] = ft.t[1] = mixed_detail::globaltimer_ns();
#endif
        expand_pending<true, DYNAMIC>(mainloop_params, K, thread_idx);
#ifdef MIXED_FA3_TRACE
        if (ft.on) ft.t[2] = ft.t[3] = mixed_detail::globaltimer_ns();
#endif
        expand_pending<true, DYNAMIC>(mainloop_params, V, thread_idx);
#ifdef MIXED_FA3_TRACE
        if (ft.on) ft.t[4] = mixed_detail::globaltimer_ns();
#endif
        cutlass::arch::fence_view_async_shared();  // D5, once per pair
        commit_pending<DYNAMIC>(K);
        commit_pending<DYNAMIC>(V);
#ifdef MIXED_FA3_TRACE
        if (ft.on) ft.t[5] = mixed_detail::globaltimer_ns();
#endif
      }
    };
    auto finish_one = [&](Operand& op) {
      if constexpr (HAS_COMPRESSED) {
        cutlass::arch::cp_async_wait<0>();
        expand_pending<false, DYNAMIC>(mainloop_params, op, thread_idx);
        cutlass::arch::fence_view_async_shared();  // D5
        commit_pending<DYNAMIC>(op);
      }
    };

#ifdef MIXED_FA3_TRACE
    uint64_t const tr_enter = mixed_detail::globaltimer_ns();
    uint64_t tr_after_q = 0, tr_first_t0 = 0, tr_prev_end = 0;
    // bar = chunk store + group barrier (once per 16 pairs); gat = next-chunk
    // register loads (issue only).
    uint64_t tr_acq = 0, tr_bar = 0, tr_gat = 0, tr_iss = 0, tr_fin = 0, tr_gap = 0;
    uint64_t tr_acqK = 0, tr_acqV = 0;
    uint64_t tr_wait = 0, tr_barB = 0, tr_expK = 0, tr_fcK = 0, tr_expV = 0, tr_fcV = 0,
             tr_oth = 0, tr_chunk = 0;
    int tr_n = 0;
    bool const trace = (blockIdx.x == 0) && (work_idx <= 1) && (thread_idx == 0);
    ft.on = trace;
#endif

    // Chunk 0 (tiles kv_tile_idx .. kv_tile_idx-15): the one exposed metadata
    // round trip of the work item.  The previous item's trailing group barrier
    // ordered every reader of the table before this store.
    ChunkRegs cr;
    chunk_load(mainloop_params, kv_indices_ptr, kv_tile_idx, 0, kv_len, thread_idx, cr);
    chunk_store(meta, kv_tile_idx, 0, kv_len, thread_idx, cr);
    group_barrier();
    K.bases = make_bases(mainloop_params, true, kv_head_idx, sK, shared_storage.mixed_scales_k,
                         thread_idx);
    V.bases = make_bases(mainloop_params, false, kv_head_idx, sV, shared_storage.mixed_scales_v,
                         thread_idx);

    // --- pairs.  tK or tV may be -1 (absent).  The partial-tile copy arm is a
    // compile-time tag per operand (C13): PartialTag for the operand that can be
    // tile kv_tile_idx - K in the K(last)-alone call, V in the peeled first pair
    // - FullTag everywhere else.  Those two calls have nothing pending (K(last)
    // is finished by its own finish_one before Q), so they compile no finish;
    // the loop's pairs finish the previous (K, V) pair unconditionally (C14).
    using FullTag = std::integral_constant<bool, false>;
    using PartialTag = std::integral_constant<bool, true>;
    auto produce_pair = [&](auto kpart, auto vpart, int tK, int tV) {
      constexpr bool PARTIAL_K = decltype(kpart)::value;
      constexpr bool PARTIAL_V = decltype(vpart)::value;
      constexpr bool FINISH = !(PARTIAL_K || PARTIAL_V);
#ifdef MIXED_FA3_TRACE
      uint64_t tr0 = 0, tr0b = 0, tr1 = 0, tr3 = 0, tr4 = 0, tr5 = 0;
      if (trace) tr0 = mixed_detail::globaltimer_ns();
#endif
      if (tK >= 0) pipeline_k.producer_acquire(smem_pipe_write_k);
#ifdef MIXED_FA3_TRACE
      if (trace) tr0b = mixed_detail::globaltimer_ns();
#endif
      if (tV >= 0) pipeline_v.producer_acquire(smem_pipe_write_v);
#ifdef MIXED_FA3_TRACE
      if (trace) tr1 = tr3 = mixed_detail::globaltimer_ns();
#endif
      if (tK >= 0) {
        TileRegs const mK = read_meta(meta, kv_tile_idx - tK);
        issue_operand<PARTIAL_K>(mainloop_params, K, mK, kv_head_idx, thread_idx);
      }
      if (tV >= 0) {
        TileRegs const mV = read_meta(meta, kv_tile_idx - tV);
        issue_operand<PARTIAL_V>(mainloop_params, V, mV, kv_head_idx, thread_idx);
      }
      cutlass::arch::cp_async_fence();  // this pair's copies form one commit group
#ifdef MIXED_FA3_TRACE
      if (trace) {
        tr4 = mixed_detail::globaltimer_ns();
        // No pending operand (A16 pair): every sub-stamp reads as tr4 -> fin = oth.
        ft.t[0] = ft.t[1] = ft.t[2] = ft.t[3] = ft.t[4] = ft.t[5] = tr4;
      }
#endif
      // Previous pair's compressed tiles: their group is complete once at most
      // one group (this pair's) is outstanding.  Then this pair becomes pending.
      if constexpr (FINISH) finish_pending_pair();
      rotate_pending(K);
      rotate_pending(V);
#ifdef MIXED_FA3_TRACE
      if (trace) {
        tr5 = mixed_detail::globaltimer_ns();
        if (tr_n == 0) tr_first_t0 = tr0; else tr_gap += tr0 - tr_prev_end - tr_chunk;
        tr_chunk = 0;
        tr_acq += tr1 - tr0; tr_iss += tr4 - tr3;
        tr_fin += tr5 - tr4; tr_prev_end = tr5; ++tr_n;
        tr_acqK += tr0b - tr0; tr_acqV += tr1 - tr0b;
        tr_wait += ft.t[0] - tr4; tr_barB += ft.t[1] - ft.t[0]; tr_expK += ft.t[2] - ft.t[1];
        tr_fcK += ft.t[3] - ft.t[2]; tr_expV += ft.t[4] - ft.t[3]; tr_fcV += ft.t[5] - ft.t[4];
        tr_oth += tr5 - ft.t[5];
      }
#endif
    };

    // --- K(last) alone (the consumer starts on it before the first V) ---
    produce_pair(PartialTag{}, FullTag{}, kv_tile_idx, -1);
    // C7: K(last) must be committed *before* barrier_O.wait below.  The consumer
    // arrives on barrier_O (work_idx > 0) only after it has received K(last) and
    // run the first QK GEMM (mainloop_mma.cuh), so a K(last) left pending across
    // that wait deadlocks the second work item of a CTA.  Finishing it here
    // exposes one copy latency per work item, once.
    finish_one(K);

    // --- Q (unchanged from the gather producer) ---
    cutlass::arch::NamedBarrier::sync(Ktraits::NUM_MMA_THREADS + Ktraits::NUM_PRODUCER_THREADS,
                                      static_cast<int>(NamedBarriers::kQueryEmpty));
    if (warp_idx_in_warpgroup == 0) {
      int lane_predicate = cute::elect_one_sync();
      if (lane_predicate) {
        shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
        copy(mainloop_params.tma_load_Q.with(
                 reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                     shared_storage.barrier_Q),
                 /*mcast_mask=*/0),
             tQgQ, tQsQ);
      }
    }
    shared_storage.barrier_O.wait((work_idx + 1) % 2);
#ifdef MIXED_FA3_TRACE
    tr_after_q = mixed_detail::globaltimer_ns();
#endif

    // --- pairs (K(t-1), V(t)) in the consumer's order; the last one is V(swa_begin) alone ---
    // Pair j of chunk c (j = (kv_tile_idx - t) % 16) loads chunk c+1's rows at
    // j == 0 and stores them + synchronises at j == CHUNK_STORE_PAIR (see the
    // constant); pair 15 of chunk c is the first to read chunk c+1 (its K tile).
    // [25] The compressed modules peel the first pair (t = kv_tile_idx: V(last)
    // is the partial tile, K(last-1) is full) so that the loop compiles the FULL
    // copy arm for both operands and finishes the previous pair unconditionally;
    // the a16 module runs the [23] loop text unchanged (byte-identical).
    auto pair_step = [&](int t, auto kpart, auto vpart) {
      int const entry = kv_tile_idx - t;
      int const j = entry % CHUNK_TILES;
      int const chunk = entry / CHUNK_TILES;
      bool const next_chunk = chunk + 1 < num_chunks;
#ifdef MIXED_FA3_TRACE
      uint64_t trc0 = 0;
      if (trace) trc0 = mixed_detail::globaltimer_ns();
#endif
      if (j == 0 && next_chunk) {
        chunk_load(mainloop_params, kv_indices_ptr, kv_tile_idx, chunk + 1, kv_len, thread_idx, cr);
      }
#ifdef MIXED_FA3_TRACE
      if (trace) {
        uint64_t const trc1 = mixed_detail::globaltimer_ns();
        tr_gat += trc1 - trc0;
        tr_chunk += trc1 - trc0;
        trc0 = trc1;
      }
#endif
      if (j == CHUNK_STORE_PAIR && next_chunk) {
        chunk_store(meta, kv_tile_idx, chunk + 1, kv_len, thread_idx, cr);
        group_barrier();
      }
#ifdef MIXED_FA3_TRACE
      if (trace) {
        uint64_t const trc1 = mixed_detail::globaltimer_ns();
        tr_bar += trc1 - trc0;
        tr_chunk += trc1 - trc0;
      }
#endif
      if (t == swa_begin_kv_tile_idx) scheduler.prefetch_next_work(scheduler_params, work_tile_info);
      produce_pair(kpart, vpart, t - 1 >= swa_begin_kv_tile_idx ? t - 1 : -1, t);
    };
    if constexpr (HAS_COMPRESSED) {
      pair_step(kv_tile_idx, FullTag{}, PartialTag{});
    }
#pragma unroll 1
    for (int t = HAS_COMPRESSED ? kv_tile_idx - 1 : kv_tile_idx; t >= swa_begin_kv_tile_idx; --t) {
      pair_step(t, FullTag{}, FullTag{});
    }
    // Drain: the last pair's V (nothing newer to overlap with; K is never pending
    // here: the last pair always has tK = -1).
    finish_one(V);
    // The trailing group barrier orders the next work item's chunk-0 store after
    // every reader of this item's table.
    group_barrier();
#ifdef MIXED_FA3_TRACE
    if (blockIdx.x == 0 && thread_idx == 0 && work_idx <= 1) {
      uint64_t const tr_exit = mixed_detail::globaltimer_ns();
      printf("[w%d] item %6llu ns: enter->afterQ %6llu | afterQ->pair0 %6llu | pairs(%d) %6llu "
             "= acq %5llu bar %4llu gat %5llu iss %6llu fin %6llu gap %6llu | lastpair->exit %5llu"
             " | acq = K %5llu V %5llu | fin = wait %6llu barB %5llu expK %6llu fcK %5llu"
             " expV %6llu fcV %5llu oth %5llu\n",
             work_idx, (unsigned long long)(tr_exit - tr_enter),
             (unsigned long long)(tr_after_q - tr_enter),
             (unsigned long long)(tr_first_t0 - tr_after_q), tr_n,
             (unsigned long long)(tr_prev_end - tr_first_t0), (unsigned long long)tr_acq,
             (unsigned long long)tr_bar, (unsigned long long)tr_gat, (unsigned long long)tr_iss,
             (unsigned long long)tr_fin, (unsigned long long)tr_gap,
             (unsigned long long)(tr_exit - tr_prev_end), (unsigned long long)tr_acqK,
             (unsigned long long)tr_acqV, (unsigned long long)tr_wait,
             (unsigned long long)tr_barB, (unsigned long long)tr_expK,
             (unsigned long long)tr_fcK, (unsigned long long)tr_expV,
             (unsigned long long)tr_fcV, (unsigned long long)tr_oth);
    }
#endif
    scheduler.broadcast_next_work(work_tile_info);
  }

  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline_k, MainloopPipeline pipeline_v,
                                PipelineState& smem_pipe_write_k,
                                PipelineState& smem_pipe_write_v) {
    pipeline_k.producer_tail(smem_pipe_write_k);
    pipeline_v.producer_tail(smem_pipe_write_v);
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_SPARSE_MIXED_MAINLOOP_CUH_
