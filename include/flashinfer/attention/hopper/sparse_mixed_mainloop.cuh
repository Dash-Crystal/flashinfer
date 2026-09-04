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
 * Producer threads ([24b]): the a16 module runs one producer warp group (128
 * threads, u = t); the compressed and dynamic modules run two (256 threads):
 * thread t = 128 h + u applies every ownership rule below to u on the tile
 * pages i = h + 2 j (j = 0..2), so each thread copies and expands three pages
 * per operand per tile.  A row's eight lanes stay in one warp.
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
 * copy arm.  N = -1 is the dynamic module ([24c], C10): the chunk-table row
 * carries two 6-bit page masks (E4M3, E2M1) and the copy and expansion loops
 * are format-outer and page-rolled over the masks - one body per format, the
 * page index read from the table by LDS.32, no per-page format branch.
 *
 * K(t-1) and V(t) are issued as a pair.  The consumer (mma_f16) is unchanged.
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

CUTLASS_DEVICE uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

// E2M1 x8 (nibbles 0..7 of src; low nibble = even coefficient) -> four A16
// pairs, out[k] = {value 2k+1 : value 2k}.  bf16: 20 integer instructions per 8
// values by two byte LUTs (low/high byte of bf16(|m|), indexed by the 3
// magnitude bits through prmt), a prmt interleave, and the sign taken by prmt's
// sign-replicate mode from the byte whose msb is the nibble's sign bit (odd
// nibbles: src's byte msbs; even nibbles: (src << 4)'s), masked in.  Exact: every
// E2M1 value is a bf16 value.  f16: CUTLASS's LUT.
template <typename A16>
CUTLASS_DEVICE void e2m1x8_to_a16(uint32_t src, uint32_t (&out)[4]) {
  if constexpr (std::is_same_v<A16, cutlass::half_t>) {
    cutlass::detail::_e2m1_to_half_x8(src, out[0], out[1], out[2], out[3]);
  } else {
    // bf16(m), m = 0..7: 0000 3F00 3F80 3FC0 4000 4040 4080 40C0.
    constexpr uint32_t kLo03 = 0xC0800000u, kLo47 = 0xC0804000u;  // low bytes
    constexpr uint32_t kHi03 = 0x3F3F3F00u, kHi47 = 0x40404040u;  // high bytes (unsigned)
    uint32_t const s4 = src << 4;                    // byte msbs = signs of nibbles 0, 2, 4, 6
    uint32_t const m03 = src & 0x7777u;              // magnitudes of nibbles 0..3 as selectors
    uint32_t const m47 = (src >> 16) & 0x7777u;      // nibbles 4..7
    uint32_t const lo03 = prmt(kLo03, kLo47, m03), hi03 = prmt(kHi03, kHi47, m03);
    uint32_t const lo47 = prmt(kLo03, kLo47, m47), hi47 = prmt(kHi03, kHi47, m47);
    // {hi(n1) lo(n1) hi(n0) lo(n0)}: byte selectors 5,1,4,0; {n3 n2}: 7,3,6,2.
    uint32_t const u01 = prmt(lo03, hi03, 0x5140u), u23 = prmt(lo03, hi03, 0x7362u);
    uint32_t const u45 = prmt(lo47, hi47, 0x5140u), u67 = prmt(lo47, hi47, 0x7362u);
    // byte1 <- sign(even nibble) replicated from s4's byte k (selector 8|k);
    // byte3 <- sign(odd nibble) from src's byte k (selector 8|4|k); bytes 0, 2 masked off.
    uint32_t const g01 = prmt(s4, src, 0xC080u), g23 = prmt(s4, src, 0xD090u);
    uint32_t const g45 = prmt(s4, src, 0xE0A0u), g67 = prmt(s4, src, 0xF0B0u);
    out[0] = u01 | (g01 & 0x80008000u);
    out[1] = u23 | (g23 & 0x80008000u);
    out[2] = u45 | (g45 & 0x80008000u);
    out[3] = u67 | (g67 & 0x80008000u);
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
  // [24b] Producer warp groups (kernel_traits.cuh MixedAttentionKernelTraits): 2
  // for the compressed / dynamic modules, 1 for the a16 module.  A page is owned
  // by the OWNER_THREADS = 128 threads u = t & 127 of warp group h = t >> 7, which
  // serves tile pages i = h + NUM_PRODUCER_WGS * j, j < PAGES_PER_THREAD; every
  // [23] ownership formula is applied to u.  With one warp group u = t, h = 0 and
  // every expression below folds to the [23] text (a16 module byte-identical).
  static constexpr int NUM_PRODUCER_WGS = Ktraits::NUM_PRODUCER_WGS;
  static constexpr int NUM_COPY_THREADS = NUM_PRODUCER_WGS * cutlass::NumThreadsPerWarpGroup;
  static constexpr uint32_t OWNER_THREADS = cutlass::NumThreadsPerWarpGroup;
  static constexpr uint32_t TOKENS_PER_PAGE = 16;
  static constexpr uint32_t PAGES_PER_TILE = CTA_KV / TOKENS_PER_PAGE;
  static constexpr uint32_t PAGES_PER_THREAD = PAGES_PER_TILE / NUM_PRODUCER_WGS;
  static_assert(PAGES_PER_TILE % NUM_PRODUCER_WGS == 0, "pages split evenly by parity");
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
  // Byte distance between a thread's consecutive pages (i -> i + NUM_PRODUCER_WGS).
  static constexpr uint32_t PAGE_STEP_BYTES = NUM_PRODUCER_WGS * PAGE_REGION_BYTES;
  // [24b] scale slots: one 8 B word pair per token row per page (the row's eight
  // block scales) in the first 128 B of the page's 512 B slot; the slot size is
  // [23]'s so that the shared-storage layout is unchanged (kernel_traits.cuh
  // kMixedScaleStageBytes).
  static constexpr uint32_t SCALE_ROW_BYTES = 8;
  static constexpr uint32_t SCALE_PAGE_BYTES = OWNER_THREADS * 4;  // 512 B page slot
  static constexpr uint32_t SCALE_PAGE_STEP_BYTES = NUM_PRODUCER_WGS * SCALE_PAGE_BYTES;
  static constexpr uint32_t SCALE_STAGE_BYTES = PAGES_PER_TILE * SCALE_PAGE_BYTES;
  static_assert(TOKENS_PER_PAGE * SCALE_ROW_BYTES <= SCALE_PAGE_BYTES, "row slots fit the page slot");
  static_assert(SCALE_STAGE_BYTES == SharedStorage::kMixedScaleStageBytes, "scale stage size");

  // [22] compile-time page format: -1 dynamic (per-page tags), 0 A16, 1 E4M3, 2 E2M1,
  // read from the variant by the traits (kernel_traits.cuh mixed_variant_static_format;
  // the same constant sizes the producer warp-group count there).
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

  // One tile's row of the chunk table in registers.  One struct, two shapes
  // (the unused fields of either module are constant zero and dead after DCE):
  //  * static modules: two LDS.128 fill `pages` (6 indices), `w6` (tags 0..3)
  //    and `w7` (tags 4, 5 | valid << 16 | flags << 24); `row_addr` is unused;
  //  * dynamic module ([24c], C10): one LDS.32 fills `w7` = m8 | m4 << 8 |
  //    valid << 16 | flags << 24 and `row_addr` is the row's smem address;
  //    `pages` and `w6` are 0 (the pending word is then w7 << 32 | stage << 60);
  //    page indices are read from the row by LDS.32 (page_at), never from a
  //    runtime-indexed register array (C2).
  struct TileRegs {
    uint32_t pages[PAGES_PER_TILE];  // static modules only
    uint32_t w6, w7;                 // w6: static modules only
    uint32_t row_addr;               // dynamic module only
    // [24b] Page index of this thread's j-th page (tile page h + NUM_PRODUCER_WGS
    // * j, h = warp-group parity), static modules.  `j` is a constant after
    // unrolling; `h` is runtime, so the select is one SEL between two loaded
    // words - never a runtime index into pages[] (C2: local memory).
    CUTLASS_DEVICE uint32_t page(uint32_t h, uint32_t j) const {
      if constexpr (NUM_PRODUCER_WGS == 1) {
        return pages[j];
      } else {
        return h ? pages[2 * j + 1] : pages[2 * j];
      }
    }
    // Dynamic module: page index of tile page i (runtime) from the table.
    CUTLASS_DEVICE uint32_t page_at(uint32_t i) const {
      return mixed_detail::lds32(row_addr + 4 * i);
    }
    CUTLASS_DEVICE uint32_t mask8() const { return w7 & 0x3Fu; }
    CUTLASS_DEVICE uint32_t mask4() const { return (w7 >> 8) & 0x3Fu; }
    CUTLASS_DEVICE uint32_t valid() const { return (w7 >> 16) & 0xFFu; }
    CUTLASS_DEVICE bool any_compressed() const { return (w7 >> 24) & kFlagCompressed; }
    // Pending record: w7 (masks / tags, valid, flags), w6 and the stage index in
    // one word; nonzero whenever the row is filled.
    CUTLASS_DEVICE uint64_t pending_word(uint32_t stage) const {
      return (uint64_t(w7) << 32 | uint64_t(w6)) | (uint64_t(stage) << 60);
    }
  };
  static_assert(NUM_STAGES <= 16, "stage index lives in bits 60..63 of the pending word");
  // Masks of a pending word (dynamic module).
  CUTLASS_DEVICE static uint32_t pending_mask8(uint64_t tv) { return uint32_t(tv >> 32) & 0x3Fu; }
  CUTLASS_DEVICE static uint32_t pending_mask4(uint64_t tv) { return uint32_t(tv >> 40) & 0x3Fu; }
  // Pages of this warp group's parity (all six for one producer warp group).
  CUTLASS_DEVICE static uint32_t parity_mask(uint32_t h) {
    return NUM_PRODUCER_WGS == 1 ? 0x3Fu : (0x15u << h);
  }

  CUTLASS_DEVICE static TileRegs read_meta(TileMeta const (*meta)[CHUNK_TILES], int entry) {
    TileMeta const& e = meta[(entry / CHUNK_TILES) & 1][entry % CHUNK_TILES];
    if constexpr (DYNAMIC) {
      uint32_t const row_addr = cute::cast_smem_ptr_to_uint(&e);
      return TileRegs{{0u, 0u, 0u, 0u, 0u, 0u}, 0u, mixed_detail::lds32(row_addr + 28u), row_addr};
    } else {
      uint4 const a = *reinterpret_cast<uint4 const*>(&e);
      uint4 const b = *(reinterpret_cast<uint4 const*>(&e) + 1);
      return TileRegs{{a.x, a.y, a.z, a.w, b.x, b.y}, b.z, b.w, 0u};
    }
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
  // [24b] ownership indices: u = within-warp-group index (the [23] "t"), h = the
  // warp group's page parity.  Both fold to (t, 0) for one producer warp group.
  CUTLASS_DEVICE static uint32_t own_u(uint32_t t) {
    return NUM_PRODUCER_WGS == 1 ? t : (t & (OWNER_THREADS - 1u));
  }
  CUTLASS_DEVICE static uint32_t own_h(uint32_t t) {
    return NUM_PRODUCER_WGS == 1 ? 0u : (t >> 7);
  }
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
    uint32_t land8;           // FP8 landing: logical chunk 8+b of row r (D-block 1 line, see land_row_line)
    uint32_t land4;           // FP4 landing: 8 B in chunk 8 + b/2 + 4*(r&1), half b&1
    uint32_t sc_rd;           // smem address of this thread's 4 B scale word in the row's 8 B slot, page h, stage 0
    float gs8, gs4;           // global scales of this operand (FP8 bf16: x 2^120 or the +inf sentinel, C9)
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

  // Thread constants of one operand, once per work item.  Data flow: every smem
  // base carries the warp group's page offset (h * PAGE_REGION_BYTES, h *
  // SCALE_PAGE_BYTES) so that page j of this thread is base + j * PAGE_STEP_BYTES -
  // an immediate - in every copy, load and store (C2 / C3: [R+imm], no runtime
  // page index); the [24b] swap is folded into out0 / out1 here and into the
  // landing half addresses in expand_bases.
  template <typename STensor>
  CUTLASS_DEVICE OperandBases make_bases(Params const& p, bool isK, int kv_head_idx, STensor& sX,
                                         uint8_t (*scales)[SCALE_STAGE_BYTES], int thread_idx) const {
    uint32_t const t = static_cast<uint32_t>(thread_idx);
    uint32_t const u = own_u(t), h = own_h(t);
    // Static modules address page j of this thread at base + j * PAGE_STEP_BYTES
    // (parity folded here); the dynamic module ([24c]) addresses tile page i at
    // base + i * PAGE_REGION_BYTES with i from the page mask, so no parity offset.
    uint32_t const page_off = DYNAMIC ? 0u : h * PAGE_REGION_BYTES;
    uint32_t const sc_page_off = DYNAMIC ? 0u : h * SCALE_PAGE_BYTES;
    OperandBases b{};
    if constexpr (HAS_COMPRESSED) {
      uint32_t const r = blk_row(u), k = blk_blk(u), swap = out_swap(u);
      uint32_t const sc_base = cute::cast_smem_ptr_to_uint(&scales[0][0]);
      b.out0 = chunk_smem(sX, 0, int(r), 2 * k + swap) + page_off;
      b.out1 = chunk_smem(sX, 0, int(r), 2 * k + 1 - swap) + page_off;
      b.land8 = chunk_smem(sX, 0, int(r), CHUNKS_PER_BLOCK + k) + page_off;
      b.land4 = chunk_smem(sX, 0, int(r), CHUNKS_PER_BLOCK + k / 2 + 4 * (r & 1u)) + 8 * (k & 1u) +
                page_off;
      // Row r's 8 B slot; this thread's word is the one holding block k (k / 4).
      // Lane k == 0 copies the slot (its sc_rd is the slot base), all eight read.
      b.sc_rd = sc_base + sc_page_off + r * SCALE_ROW_BYTES + 4 * (k >> 2);
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
      b.a16_dst = chunk_smem(sX, 0, int(a_r), a_c) + page_off;
    }
    if constexpr (HAS_FP8) {
      auto const& span = p.transport.formats[kTagFP8];
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
      b.gs4 = *(isK ? span.k_global_scale : span.v_global_scale);
    }
    return b;
  }

  // Per-tile source bases of a compressed format (recomputed per tile: a few
  // IMAD.WIDE per operand, cheaper than holding them across the pair loop).
  struct CompressedSrc {
    uint8_t const* payload;  // this thread's packed block of page 0
    uint8_t const* scales;   // this thread's row's 8 scale bytes, page 0 (copied by lane b == 0)
    uint32_t payload_ps, scale_ps;
  };
  template <uint8_t FORMAT>
  CUTLASS_DEVICE CompressedSrc compressed_src(Params const& p, bool isK, int kv_head_idx,
                                              int thread_idx) const {
    uint32_t const u = own_u(static_cast<uint32_t>(thread_idx));
    KVPageFormatSpan const& span = p.transport.formats[FORMAT];
    uint32_t const row = blk_row(u), blk = blk_blk(u);
    constexpr uint32_t BLOCK_BYTES = FORMAT == kTagFP8 ? 16 : 8;
    CompressedSrc s;
    s.payload = static_cast<uint8_t const*>(isK ? span.k_payload : span.v_payload) +
                uint64_t(kv_head_idx) * span.payload_stride.head +
                uint64_t(row) * span.payload_stride.token + blk * BLOCK_BYTES;
    s.scales = (isK ? span.k_scales : span.v_scales) +
               uint64_t(kv_head_idx) * span.scale_stride.head +
               uint64_t(row) * span.scale_stride.token;
    s.payload_ps = span.payload_stride.page;
    s.scale_ps = span.scale_stride.page;
    return s;
  }

  // First token of this thread's j-th page within the tile: (h + NUM_PRODUCER_WGS
  // * j) * 16 - the runtime part h * 16 is one thread constant, the rest immediate.
  CUTLASS_DEVICE static uint32_t page_tok0(uint32_t h, uint32_t j) {
    return (h + NUM_PRODUCER_WGS * j) * TOKENS_PER_PAGE;
  }

  // Page placement of one copy / expansion: the byte offset of the page's region
  // from the operand base (static modules: j * PAGE_STEP_BYTES, an immediate;
  // dynamic module: i * PAGE_REGION_BYTES, one IMAD), the same for the scale
  // slot, and the page's first token for the D4 predicate.
  struct PagePos {
    uint32_t off;     // + PAGE_REGION_BYTES multiple
    uint32_t sc_off;  // + SCALE_PAGE_BYTES multiple
    uint32_t tok0;    // first token of the page in the tile
  };
  CUTLASS_DEVICE static PagePos static_page(uint32_t h, uint32_t j) {
    return {j * PAGE_STEP_BYTES, j * SCALE_PAGE_STEP_BYTES, page_tok0(h, j)};
  }
  CUTLASS_DEVICE static PagePos dynamic_page(uint32_t i) {
    return {i * PAGE_REGION_BYTES, i * SCALE_PAGE_BYTES, i * TOKENS_PER_PAGE};
  }

  // One page's copies.  FULL: the tile has CTA_KV valid tokens (no predicates).
  template <bool FULL>
  CUTLASS_DEVICE void copy_a16_page(OperandBases const& b, uint32_t page, PagePos const& pp,
                                    uint32_t dst_stage, uint32_t valid, int thread_idx) const {
    uint32_t const t = static_cast<uint32_t>(thread_idx);
    uint32_t const a_r = own_u(t) / CHUNKS_PER_ROW;
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
  // D4: rows past `valid` copy with src-size 0, zero-filling block and slot.
  template <uint8_t FORMAT, bool FULL>
  CUTLASS_DEVICE void copy_compressed_page(OperandBases const& b, CompressedSrc const& s,
                                           uint32_t page, PagePos const& pp, uint32_t stage,
                                           uint32_t valid, int thread_idx) const {
    uint32_t const t = static_cast<uint32_t>(thread_idx);
    uint32_t const u = own_u(t);
    bool const scale_leader = blk_blk(u) == 0;
    uint8_t const* src = s.payload + uint64_t(page) * uint64_t(s.payload_ps);
    uint32_t const land = (FORMAT == kTagFP8 ? b.land8 : b.land4) + stage * STAGE_BYTES + pp.off;
    uint32_t const sdst = b.sc_rd + stage * SCALE_STAGE_BYTES + pp.sc_off;
    uint8_t const* ssrc = s.scales + uint64_t(page) * uint64_t(s.scale_ps);
    if constexpr (FULL) {
      if constexpr (FORMAT == kTagFP8) {
        mixed_detail::cp16(land, src);
      } else {
        mixed_detail::cp8(land, src);
      }
      if (scale_leader) mixed_detail::cp8(sdst, ssrc);
    } else {
      bool const v = pp.tok0 + blk_row(u) < valid;
      if constexpr (FORMAT == kTagFP8) {
        mixed_detail::cp16_zfill(land, v ? src : s.payload, v);
      } else {
        mixed_detail::cp8_zfill(land, v ? src : s.payload, v);
      }
      if (scale_leader) mixed_detail::cp8_zfill(sdst, v ? ssrc : s.scales, v);
    }
  }

  template <bool FULL>
  CUTLASS_DEVICE void issue_tile_copies(Params const& p, OperandBases const& b, TileRegs const& m,
                                        uint32_t stage, bool isK, int kv_head_idx,
                                        int thread_idx) const {
    uint32_t const valid = m.valid();
    uint32_t const a16_dst_stage = b.a16_dst + stage * STAGE_BYTES;
    uint32_t const h = own_h(static_cast<uint32_t>(thread_idx));
    if constexpr (!DYNAMIC) {
      // Static modules: unrolled over this thread's pages (j: tile page h +
      // NUM_PRODUCER_WGS * j) - the body is a handful of instructions per page
      // with immediate destinations; rolled loop control cost as much as the
      // copies.
      CompressedSrc s8{}, s4{};
      if constexpr (STATIC_FP8) s8 = compressed_src<kTagFP8>(p, isK, kv_head_idx, thread_idx);
      if constexpr (STATIC_FP4) s4 = compressed_src<kTagFP4>(p, isK, kv_head_idx, thread_idx);
#pragma unroll
      for (uint32_t j = 0; j < PAGES_PER_THREAD; ++j) {
        uint32_t const page = m.page(h, j);
        PagePos const pp = static_page(h, j);
        if constexpr (STATIC_A16) {
          copy_a16_page<FULL>(b, page, pp, a16_dst_stage, valid, thread_idx);
        } else if constexpr (STATIC_FP8) {
          copy_compressed_page<kTagFP8, FULL>(b, s8, page, pp, stage, valid, thread_idx);
        } else {
          copy_compressed_page<kTagFP4, FULL>(b, s4, page, pp, stage, valid, thread_idx);
        }
      }
    } else {
      // [24c] Dynamic module, [40]'s pattern (C10): format-outer, page-rolled.
      // Data flow: the tile's two 6-bit masks (chunk table, w7) restricted to this
      // warp group's parity; the parity's PAGES_PER_THREAD page indices (tile
      // pages i = h + NUM_PRODUCER_WGS * j) are read from the table row up front
      // with independent LDS.32 into scalars - one smem latency per tile, off the
      // copy issue path - and selected per page by a constant-unrolled
      // compare/select chain on j = i / NUM_PRODUCER_WGS (C2: no runtime index
      // into a register array, no LDS on the LDGSTS address chain); destinations
      // base + i * PAGE_REGION_BYTES (one IMAD).  Control flow: per format one
      // rolled loop over the set bits, three warp-uniform loops (FP8, FP4, A16)
      // with one copy body each and no per-page format branch; each format's
      // source set-up (compressed_src) runs only if its mask is nonzero.  Every
      // quantity here is warp-uniform data from smem, so the loops never diverge.
      uint32_t const par = parity_mask(h);
      uint32_t const m8 = m.mask8() & par;
      uint32_t const m4 = m.mask4() & par;
      uint32_t const ma = par & ~(m8 | m4);
      uint32_t pg[PAGES_PER_THREAD];
#pragma unroll
      for (uint32_t j = 0; j < PAGES_PER_THREAD; ++j) {
        pg[j] = m.page_at(h + NUM_PRODUCER_WGS * j);
      }
      auto const page_of = [&pg](uint32_t i) {
        uint32_t const j = i / NUM_PRODUCER_WGS;
        uint32_t r = pg[0];
#pragma unroll
        for (uint32_t k = 1; k < PAGES_PER_THREAD; ++k) r = (j == k) ? pg[k] : r;
        return r;
      };
      if (m8) {
        CompressedSrc const s8 = compressed_src<kTagFP8>(p, isK, kv_head_idx, thread_idx);
#pragma unroll 1
        for (uint32_t mm = m8; mm; mm &= mm - 1) {
          uint32_t const i = __ffs(mm) - 1;
          copy_compressed_page<kTagFP8, FULL>(b, s8, page_of(i), dynamic_page(i), stage, valid,
                                              thread_idx);
        }
      }
      if (m4) {
        CompressedSrc const s4 = compressed_src<kTagFP4>(p, isK, kv_head_idx, thread_idx);
#pragma unroll 1
        for (uint32_t mm = m4; mm; mm &= mm - 1) {
          uint32_t const i = __ffs(mm) - 1;
          copy_compressed_page<kTagFP4, FULL>(b, s4, page_of(i), dynamic_page(i), stage, valid,
                                              thread_idx);
        }
      }
#pragma unroll 1
      for (uint32_t mm = ma; mm; mm &= mm - 1) {
        uint32_t const i = __ffs(mm) - 1;
        copy_a16_page<FULL>(b, page_of(i), dynamic_page(i), a16_dst_stage, valid, thread_idx);
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
  // Per FP8 block ([24a], bf16): LDS.128 + LDS.32 + 8 (PRMT, F2FP.E4M3, HADD2,
  // FMUL, FSETP, VOTE.ALL, BRA, F2FP.PACK) + 8 x (PRMT, SHF, LOP3) + 8 HMUL2 +
  // 2 STS.128 = 44 instructions on the fold path (the cold path adds 8 HMUL2 by
  // 2^120 and a reload of the global scale).
  // Per FP4 block: LDS.64 + LDS.32 + scale 5 + 2 x 20 (LUT) + 8 HMUL2 + 2 STS.

  // E4M3 scale byte (byte sel of the word) -> the reference's float(scale).
  CUTLASS_DEVICE static float scale_byte_f32(uint32_t scale_word, uint32_t sel) {
    uint32_t const byte = __byte_perm(scale_word, 0u, sel);
    uint32_t h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(static_cast<uint16_t>(byte)));
    return __half2float(reinterpret_cast<__half const&>(h2));
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
  // The operand's plain (unfolded) FP8 global scale, read from the grid-constant
  // parameters: used on the cold path only, so it is not held in a register.
  CUTLASS_DEVICE static float fp8_global_plain(Params const& p, bool isK) {
    auto const& span = p.transport.formats[kTagFP8];
    return *(isK ? span.k_global_scale : span.v_global_scale);
  }

  // Block layouts: FP8 pair 2j, 2j+1 = low, high halves of w[j] (the quantizer's
  // layout); FP4 nibbles 0..7 in v.x, 8..15 in v.y, low nibble = even coefficient.
  // The decode is inlined in expand_block (the fold vote sits between decode and
  // multiply).

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
  CUTLASS_DEVICE static ExpandBases expand_bases(OperandBases const& b, uint32_t stage,
                                                 uint32_t t) {
    uint32_t const so = stage * STAGE_BYTES;
    uint32_t const swap = out_swap(own_u(t));
    uint32_t const l8a = b.land8 + so + 8 * swap;
    uint32_t const l4a = b.land4 + so + 4 * swap;
    return {b.out0 + so, b.out1 + so, l8a, l8a ^ 8u, l4a, l4a ^ 4u,
            b.sc_rd + stage * SCALE_STAGE_BYTES};
  }

  // Packed input of this thread's page j (tile page h + NUM_PRODUCER_WGS * j).
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
  // One block.  Data flow (FP8, bf16): v = f32(s) * gs8 (gs8 = g * 2^120 or +inf);
  // the placed decode lo, hi = x * 2^-120 is independent of the scale.  Control
  // flow: ok = |v| < 255.5 * 2^120 per lane, voted over the warp (one page, four
  // rows: the granularity XQA votes at); the vote passes -> sf2 = bf16x2(v), one
  // rounding multiply per pair (fold path); it fails -> lo, hi *= 2^120 (exact:
  // x * 2^-120 * 2^120 = x, subnormal placements included), sf2 = bf16x2(f32(s)
  // * g) with g reloaded from the parameters, then the same rounding multiply -
  // the reference's arithmetic for every finite s and g (C9).  The vote result is
  // warp-uniform, so both arms reach the __syncwarp before the stores converged.
  // FP4 and f16 have no fold: sf2 = a16x2(f32(s) * g) directly.
  // `off` is the page's byte offset from the stage base (PagePos::off).
  template <bool FP8>
  CUTLASS_DEVICE void expand_block(Params const& prm, bool isK, ExpandBases const& e,
                                   OperandBases const& b, Packed const& p, uint32_t sw,
                                   uint32_t off, uint32_t t) const {
#if defined(MIXED_FA3_CONTROL_SKIP_EXPAND)
    // Timing control (design 2C): no decode, no stores; values are garbage.
    (void)prm; (void)isK; (void)e; (void)b; (void)p; (void)sw; (void)off; (void)t;
    return;
#elif defined(MIXED_FA3_CONTROL_RAW_STS)
    // Timing control (design 2C): the STS wavefronts without the decode.
    (void)prm; (void)isK; (void)b; (void)sw; (void)t;
    __syncwarp();
    mixed_detail::sts128(e.d0 + off, p.w);
    mixed_detail::sts128(e.d1 + off, p.w);
    return;
#endif
    float const v = scale_byte_f32(sw, sc_sel(own_u(t))) * (FP8 ? b.gs8 : b.gs4);
    uint4 lo, hi;
    uint32_t sf2;
    if constexpr (FP8) {
      mixed_detail::e4m3x4_to_a16<DTypeKV>(p.w.x, lo.x, lo.y);
      mixed_detail::e4m3x4_to_a16<DTypeKV>(p.w.y, lo.z, lo.w);
      mixed_detail::e4m3x4_to_a16<DTypeKV>(p.w.z, hi.x, hi.y);
      mixed_detail::e4m3x4_to_a16<DTypeKV>(p.w.w, hi.z, hi.w);
      if constexpr (mixed_detail::kFp8FoldsPow2<DTypeKV>) {
        bool const fold_ok = fabsf(v) < mixed_detail::kFp8FoldMax;  // false for inf / NaN
        if (__all_sync(0xFFFFFFFFu, fold_ok)) {
          sf2 = a16x2_broadcast(v);
        } else {
          // Cold: exact two-multiply form (XQA mha_sm90.cu:2791-2806).
          lo.x = mixed_detail::mul_a16x2<DTypeKV>(lo.x, mixed_detail::kTwoPow120Bf16x2);
          lo.y = mixed_detail::mul_a16x2<DTypeKV>(lo.y, mixed_detail::kTwoPow120Bf16x2);
          lo.z = mixed_detail::mul_a16x2<DTypeKV>(lo.z, mixed_detail::kTwoPow120Bf16x2);
          lo.w = mixed_detail::mul_a16x2<DTypeKV>(lo.w, mixed_detail::kTwoPow120Bf16x2);
          hi.x = mixed_detail::mul_a16x2<DTypeKV>(hi.x, mixed_detail::kTwoPow120Bf16x2);
          hi.y = mixed_detail::mul_a16x2<DTypeKV>(hi.y, mixed_detail::kTwoPow120Bf16x2);
          hi.z = mixed_detail::mul_a16x2<DTypeKV>(hi.z, mixed_detail::kTwoPow120Bf16x2);
          hi.w = mixed_detail::mul_a16x2<DTypeKV>(hi.w, mixed_detail::kTwoPow120Bf16x2);
          sf2 = a16x2_broadcast(scale_byte_f32(sw, sc_sel(own_u(t))) * fp8_global_plain(prm, isK));
        }
      } else {
        sf2 = a16x2_broadcast(v);
      }
    } else {
      uint32_t a[4], c[4];
      mixed_detail::e2m1x8_to_a16<DTypeKV>(p.w.x, a);
      mixed_detail::e2m1x8_to_a16<DTypeKV>(p.w.y, c);
      lo = uint4{a[0], a[1], a[2], a[3]};
      hi = uint4{c[0], c[1], c[2], c[3]};
      sf2 = a16x2_broadcast(v);
    }
    lo.x = mixed_detail::mul_a16x2<DTypeKV>(lo.x, sf2);
    lo.y = mixed_detail::mul_a16x2<DTypeKV>(lo.y, sf2);
    lo.z = mixed_detail::mul_a16x2<DTypeKV>(lo.z, sf2);
    lo.w = mixed_detail::mul_a16x2<DTypeKV>(lo.w, sf2);
    hi.x = mixed_detail::mul_a16x2<DTypeKV>(hi.x, sf2);
    hi.y = mixed_detail::mul_a16x2<DTypeKV>(hi.y, sf2);
    hi.z = mixed_detail::mul_a16x2<DTypeKV>(hi.z, sf2);
    hi.w = mixed_detail::mul_a16x2<DTypeKV>(hi.w, sf2);
    __syncwarp();  // the row's octet has read its landings (other lanes' output chunks)
    mixed_detail::sts128(e.d0 + off, lo);
    mixed_detail::sts128(e.d1 + off, hi);
  }

  // One operand's pending tile.  `tv` is the tile's pending word.  Static
  // modules: one body, loads pipelined one page ahead.  Dynamic module ([24c]):
  // the page masks are warp-uniform data from the pending word.
  CUTLASS_DEVICE void expand_operand(Params const& prm, bool isK, OperandBases const& b,
                                     uint64_t tv, int stage, uint32_t t) const {
    ExpandBases const e = expand_bases(b, uint32_t(stage), t);
    uint32_t const h = own_h(t);
    if constexpr (!DYNAMIC) {
      // This thread's pages j = 0 .. PAGES_PER_THREAD-1 (tile pages h +
      // NUM_PRODUCER_WGS * j) at immediate offsets.  Two pages per step (16
      // independent decode chains for the scheduler) with the next two pages'
      // loads issued before this step's stores.  An odd page count (3 per thread
      // with two producer warp groups) ends with one page; every bound below is
      // a compile-time constant.
      constexpr bool FP8 = STATIC_FP8;
      constexpr uint32_t N = PAGES_PER_THREAD;
      uint32_t sw[N];
#pragma unroll
      for (uint32_t j = 0; j < N; ++j) {
        sw[j] = mixed_detail::lds32(e.sc + static_page(h, j).sc_off);
      }
      Packed cur0 = load_packed<FP8>(e, static_page(h, 0).off);
      Packed cur1 = N > 1 ? load_packed<FP8>(e, static_page(h, 1).off) : Packed{};
#pragma unroll
      for (uint32_t j = 0; j < N; j += 2) {
        Packed next0{}, next1{};
        if (j + 2 < N) next0 = load_packed<FP8>(e, static_page(h, j + 2).off);
        if (j + 3 < N) next1 = load_packed<FP8>(e, static_page(h, j + 3).off);
        expand_block<FP8>(prm, isK, e, b, cur0, sw[j], static_page(h, j).off, t);
        if (j + 1 < N) {
          expand_block<FP8>(prm, isK, e, b, cur1, sw[j + 1], static_page(h, j + 1).off, t);
        }
        cur0 = next0;
        cur1 = next1;
      }
    } else {
      // [24c] Dynamic module (C10): format-outer, page-rolled over the pending
      // word's masks restricted to this warp group's parity.  Data flow: the
      // FP8 loop's first page and the FP4 loop's first page are loaded up front
      // (packed halves + scale word), so the FP4 loop's first load latency hides
      // under the FP8 expansion; inside a loop the next page's loads are issued
      // before this page's stores (the static path's one-ahead pipelining).
      // Control flow: two warp-uniform rolled loops with one expansion body
      // each; page offsets i * PAGE_REGION_BYTES / i * SCALE_PAGE_BYTES by IMAD;
      // no per-page format branch and no runtime-indexed register array.
      uint32_t const par = parity_mask(h);
      uint32_t m8 = pending_mask8(tv) & par;
      uint32_t m4 = pending_mask4(tv) & par;
      uint32_t i8 = 0, i4 = 0;
      Packed c8{}, c4{};
      uint32_t sw8 = 0, sw4 = 0;
      if (m8) {
        i8 = __ffs(m8) - 1;
        c8 = load_packed<true>(e, dynamic_page(i8).off);
        sw8 = mixed_detail::lds32(e.sc + dynamic_page(i8).sc_off);
      }
      if (m4) {
        i4 = __ffs(m4) - 1;
        c4 = load_packed<false>(e, dynamic_page(i4).off);
        sw4 = mixed_detail::lds32(e.sc + dynamic_page(i4).sc_off);
      }
      expand_format_pages<true>(prm, isK, e, b, m8, i8, c8, sw8, t);
      expand_format_pages<false>(prm, isK, e, b, m4, i4, c4, sw4, t);
    }
  }

  // [24c] One format's pages of a pending tile (dynamic module).  `m` is the
  // page mask; page `i` (its lowest set bit) is already loaded into `cur`, `sw`.
  // Each step issues the next page's loads, expands the current page, advances.
  template <bool FP8>
  CUTLASS_DEVICE void expand_format_pages(Params const& prm, bool isK, ExpandBases const& e,
                                          OperandBases const& b, uint32_t m, uint32_t i,
                                          Packed cur, uint32_t sw, uint32_t t) const {
    if (m == 0) return;  // warp-uniform
#pragma unroll 1
    for (;;) {
      m &= m - 1;
      uint32_t const n = m ? __ffs(m) - 1 : 0u;
      Packed next{};
      uint32_t nsw = 0;
      if (m) {
        next = load_packed<FP8>(e, dynamic_page(n).off);
        nsw = mixed_detail::lds32(e.sc + dynamic_page(n).sc_off);
      }
      expand_block<FP8>(prm, isK, e, b, cur, sw, dynamic_page(i).off, t);
      if (m == 0) break;
      i = n;
      cur = next;
      sw = nsw;
    }
  }

  // ------------------------------------------------------------------------
  // One operand of a tile.  Two completion modes (C4):
  //  * no compressed page: every thread commits with cp.async.mbarrier.arrive;
  //    the stage completes when the copies land.  Nobody waits.
  //  * otherwise: the operand becomes *pending*; one pair later each thread waits
  //    for its own copies of that commit group, expands its blocks, fences (D5)
  //    and arrives.  No group barrier: a thread reads only bytes it copied.
  // A pending record is one word: the tile's tags / valid / flags and the stage
  // index (TileRegs::pending_word); 0 means inactive (a filled row always has
  // kFlagFilled set).
  struct Operand {
    MainloopPipeline* pipeline;
    PipelineState* state;
    uint8_t (*scales)[SCALE_STAGE_BYTES];
    bool isK;
    OperandBases bases;
    uint64_t pending;  // finished (waited, expanded, committed) this iteration
    uint64_t staged;   // issued this iteration; becomes `pending` after the finish
  };

  CUTLASS_DEVICE void issue_operand(Params const& p, Operand& op, TileRegs const& m,
                                    int kv_head_idx, int thread_idx) const {
    uint32_t const stage = op.state->index();
    if (m.valid() == uint32_t(CTA_KV)) {
      issue_tile_copies<true>(p, op.bases, m, stage, op.isK, kv_head_idx, thread_idx);
    } else {
      issue_tile_copies<false>(p, op.bases, m, stage, op.isK, kv_head_idx, thread_idx);
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
  CUTLASS_DEVICE void expand_pending(Params const& prm, Operand const& op, int thread_idx) const {
    if constexpr (HAS_COMPRESSED) {
      if (op.pending == 0) return;
      expand_operand(prm, op.isK, op.bases, op.pending, int(op.pending >> 60),
                     uint32_t(thread_idx));
    }
  }
  // After the fence: arrive on the pending stage's full barrier.
  CUTLASS_DEVICE void commit_pending(Operand& op) const {
    if constexpr (HAS_COMPRESSED) {
      if (op.pending == 0) return;
      // PipelineAsync::producer_commit arrives on full_barrier[state.index()]; only the
      // index of the pending stage matters.
      op.pipeline->producer_commit(PipelineState(int(op.pending >> 60), 0, 0));
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

    // Wait for this thread's copies of the previous pair (all groups but the
    // newest), expand its blocks of both operands, fence once (D5) and commit.
    // No group barrier: the expansion reads only bytes this thread copied (D3).
    // Explicit K/V call sites: selecting an Operand by a runtime reference would
    // force both structs into local memory (C2).
    auto finish_pending_pair = [&](bool allow_newest_group) {
      if constexpr (HAS_COMPRESSED) {
      if (K.pending != 0 || V.pending != 0) {
        if (allow_newest_group) {
          cutlass::arch::cp_async_wait<1>();
        } else {
          cutlass::arch::cp_async_wait<0>();
        }
        // [24b] A8 / D3: the row's 8 B scale slot was copied by the row's lane 0;
        // after every lane's own wait the warp barrier orders that lane's
        // completed copy before the other seven lanes' LDS.32 of it (the same
        // ordering the landing chunks already rely on).  One per pair.
        __syncwarp();
#ifdef MIXED_FA3_TRACE
        if (ft.on) ft.t[0] = ft.t[1] = mixed_detail::globaltimer_ns();
#endif
        expand_pending(mainloop_params, K, thread_idx);
#ifdef MIXED_FA3_TRACE
        if (ft.on) ft.t[2] = ft.t[3] = mixed_detail::globaltimer_ns();
#endif
        expand_pending(mainloop_params, V, thread_idx);
#ifdef MIXED_FA3_TRACE
        if (ft.on) ft.t[4] = mixed_detail::globaltimer_ns();
#endif
        cutlass::arch::fence_view_async_shared();  // D5, once per pair
        commit_pending(K);
        commit_pending(V);
#ifdef MIXED_FA3_TRACE
        if (ft.on) ft.t[5] = mixed_detail::globaltimer_ns();
#endif
      }
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

    // [24b] Chunk-table gather by warp group 0 only (row t/8, slot t%8 of 128
    // threads; a second warp group's rows 16..31 would alias the other buffer
    // and run past it).  Warp-uniform (whole warp group), so chunk_store's
    // __reduce_or_sync stays well-formed; every producer thread still meets the
    // group barrier.  Folds to unconditional for one producer warp group.
    bool const gather_thread = NUM_PRODUCER_WGS == 1 || thread_idx < int(OWNER_THREADS);
    // Chunk 0 (tiles kv_tile_idx .. kv_tile_idx-15): the one exposed metadata
    // round trip of the work item.  The previous item's trailing group barrier
    // ordered every reader of the table before this store.
    ChunkRegs cr;
    if (gather_thread) {
      chunk_load(mainloop_params, kv_indices_ptr, kv_tile_idx, 0, kv_len, thread_idx, cr);
      chunk_store(meta, kv_tile_idx, 0, kv_len, thread_idx, cr);
    }
    group_barrier();
    K.bases = make_bases(mainloop_params, true, kv_head_idx, sK, shared_storage.mixed_scales_k,
                         thread_idx);
    V.bases = make_bases(mainloop_params, false, kv_head_idx, sV, shared_storage.mixed_scales_v,
                         thread_idx);

    // --- pairs.  tK or tV may be -1 (absent).  One inlined copy serves K(last)
    // alone, the steady-state pairs (K(t-1), V(t)) and the final V(swa_begin).
    auto produce_pair = [&](int tK, int tV) {
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
        issue_operand(mainloop_params, K, mK, kv_head_idx, thread_idx);
      }
      if (tV >= 0) {
        TileRegs const mV = read_meta(meta, kv_tile_idx - tV);
        issue_operand(mainloop_params, V, mV, kv_head_idx, thread_idx);
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
      finish_pending_pair(/*allow_newest_group=*/true);
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
    produce_pair(kv_tile_idx, -1);
    // C7: K(last) must be committed *before* barrier_O.wait below.  The consumer
    // arrives on barrier_O (work_idx > 0) only after it has received K(last) and
    // run the first QK GEMM (mainloop_mma.cuh), so a K(last) left pending across
    // that wait deadlocks the second work item of a CTA.  Finishing it here
    // exposes one copy latency per work item, once.
    finish_pending_pair(/*allow_newest_group=*/false);

    // --- Q (unchanged from the gather producer) ---
    cutlass::arch::NamedBarrier::sync(Ktraits::NUM_MMA_THREADS + Ktraits::NUM_PRODUCER_THREADS,
                                      static_cast<int>(NamedBarriers::kQueryEmpty));
    // [24b] The Q box is issued once per CTA: by warp 0 of the CTA.  With two
    // producer warp groups `warp_idx_in_warpgroup == 0` holds in both, and a
    // second issue against barrier_Q's transaction count of one would corrupt
    // its phase; with one it is the [23] text.
    bool const q_issuer = NUM_PRODUCER_WGS == 1 ? warp_idx_in_warpgroup == 0
                                                : thread_idx < int(cutlass::NumThreadsPerWarp);
    if (q_issuer) {
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
#pragma unroll 1
    for (int t = kv_tile_idx; t >= swa_begin_kv_tile_idx; --t) {
      int const entry = kv_tile_idx - t;
      int const j = entry % CHUNK_TILES;
      int const chunk = entry / CHUNK_TILES;
      bool const next_chunk = chunk + 1 < num_chunks;
#ifdef MIXED_FA3_TRACE
      uint64_t trc0 = 0;
      if (trace) trc0 = mixed_detail::globaltimer_ns();
#endif
      if (j == 0 && next_chunk && gather_thread) {
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
        if (gather_thread) chunk_store(meta, kv_tile_idx, chunk + 1, kv_len, thread_idx, cr);
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
      produce_pair(t - 1 >= swa_begin_kv_tile_idx ? t - 1 : -1, t);
    }
    // Drain: the last pair's compressed tiles (nothing newer to overlap with).
    finish_pending_pair(/*allow_newest_group=*/false);
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
