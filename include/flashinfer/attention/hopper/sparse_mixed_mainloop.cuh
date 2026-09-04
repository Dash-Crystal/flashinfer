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
 * FA3 paged producer moves it: 128 producer threads issue 16 B cp.async copies
 * straight into the consumer's SW128 K-major smem stage (a 16-token page is
 * too small a TMA box on sm90 - each TMA operation costs ~100-200 ns of issue
 * regardless of size, which made per-page TMA the critical path).
 *
 *  * A16 pages: thread t copies chunk t%16 of rows t/16 and t/16+8 of every
 *    page (D6) and the stage is committed with cp.async.mbarrier.arrive - the
 *    producer never waits.
 *  * Compressed pages ([23]): thread t owns scale block t%8 of row t/8 of every
 *    page (eight consecutive lanes copy one row's global line).  It copies the
 *    block's packed bytes (FP8 16 B, FP4 8 B) into the row's D-block 1 line (chunk
 *    b ^ (r&7): one 128 B smem line per lane octet, which is what the cp.async
 *    path needs to coalesce) and the 4 B scale word holding its block's
 *    scale (the four owners of a word copy the same bytes to the same address).
 *    The pair's copies form one cp.async commit group.  One pair later each
 *    thread waits for its own copies (cp.async.wait_group; no group barrier:
 *    a thread reads only bytes its own copies wrote, and the row's eight lanes
 *    are one warp, ordered by __syncwarp before their stores), decodes its blocks (FP8: cvt + shift with 2^112 folded into the
 *    scale; FP4: prmt LUT) and stores them to chunks 2b, 2b+1 with STS.128 at
 *    immediate offsets from per-stage 32-bit bases; one fence.proxy.async per
 *    pair, then the commits.  No byte is read by a thread that did not copy it.
 *
 * Page metadata ([21]): a two-buffer chunk table of 16 tiles (kernel_traits.cuh
 * MixedTileMeta).  Threads 0..127 each load one (kv_index, tag) of the next
 * chunk at the first pair of the current chunk, store them at its ninth pair
 * and synchronise the group once; every pair then reads its two tiles' rows
 * with four LDS.128 and the pending record carries the tags it needs, so the
 * pair body has no group barrier and no exposed global load.
 *
 * Static format ([22]): when the module's AttentionVariant carries
 * kMixedStaticFormat (MixedPageAttention<N>, module URI ..._static_format_N),
 * every page has that format at compile time: the a16 module compiles no tag
 * loads, no format branches and no expansion; the fp8/fp4 modules compile one
 * copy arm.  N = -1 is the dynamic module with the per-page switch.
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

// The AttentionVariant may carry the module's static page format (variants.cuh
// MixedPageAttention<N>); without it the producer reads per-page tags.
template <typename Variant, typename = void>
struct mixed_static_format_of : std::integral_constant<int, -1> {};
template <typename Variant>
struct mixed_static_format_of<Variant, std::void_t<decltype(Variant::kMixedStaticFormat)>>
    : std::integral_constant<int, Variant::kMixedStaticFormat> {};

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
// bf16: the f16 bit pattern s|eeeee|mmmmmmmmmm shifted right by 3 is
// 000s|eeeee|mmmmmmm, a bf16 whose exponent field is the f16's, i.e. the value
// x 2^-112 (exact: an E4M3 value has at most 3 significant mantissa bits, so
// the 3 bits shifted out - and the neighbouring half's bits that shift in - are
// zero; the f16 exponent field is 6..30 for nonzero E4M3, never 0 or 31).  The
// sign moves from bit 12 to bit 15 with one IMAD: a + 7 * (a & 0x1000).  The
// 2^112 is folded into the block scale (OperandBases::gs8), and one
// HMUL2.BF16 then rounds the exact product v * s once, as the reference does.
// Per pair: F2FP, SHF, LOP3, IMAD (versus F2FP, 2 x HADD2.F32, F2FP.PACK).
template <typename A16>
CUTLASS_DEVICE void e4m3x4_to_a16(uint32_t w, uint32_t& p01, uint32_t& p23) {
  uint32_t h01, h23;
  asm("{\n\t.reg .b16 lo, hi;\n\t"
      "mov.b32 {lo, hi}, %2;\n\t"
      "cvt.rn.f16x2.e4m3x2 %0, lo;\n\t"
      "cvt.rn.f16x2.e4m3x2 %1, hi;\n\t}"
      : "=r"(h01), "=r"(h23)
      : "r"(w));
  if constexpr (std::is_same_v<A16, cutlass::half_t>) {
    p01 = h01;
    p23 = h23;
  } else {
    uint32_t const a01 = h01 >> 3, a23 = h23 >> 3;
    p01 = a01 + 7u * (a01 & 0x10001000u);
    p23 = a23 + 7u * (a23 & 0x10001000u);
  }
}

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
  static constexpr int NUM_COPY_THREADS = cutlass::NumThreadsPerWarpGroup;
  static constexpr uint32_t TOKENS_PER_PAGE = 16;
  static constexpr uint32_t PAGES_PER_TILE = CTA_KV / TOKENS_PER_PAGE;
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
  static constexpr uint32_t SCALE_PAGE_BYTES = NUM_COPY_THREADS * 4;  // one 4 B word per thread
  static constexpr uint32_t SCALE_STAGE_BYTES = PAGES_PER_TILE * SCALE_PAGE_BYTES;
  static_assert(SCALE_STAGE_BYTES == SharedStorage::kMixedScaleStageBytes, "scale stage size");

  // [22] compile-time page format: -1 dynamic (per-page tags), 0 A16, 1 E4M3, 2 E2M1.
  static constexpr int STATIC_FORMAT =
      mixed_static_format_of<typename Ktraits::AttentionVariant>::value;
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
  static_assert(CHUNK_TILES * 8 == NUM_COPY_THREADS, "one thread per (tile row, page slot)");
  static_assert(META_BUFFERS == 2, "double-buffered chunk table");

  static_assert(HEAD_DIM == 128 && sizeof(DTypeKV) == 2,
                "mixed pages are implemented for D=128 A16 (bf16/f16) math");
  static_assert(CTA_KV % TOKENS_PER_PAGE == 0, "KV tile must be whole pages");
  static_assert(CTA_KV <= NUM_COPY_THREADS, "one producer thread per token");
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
    // Compressed copies: FP8 blocks are 16 B cp.async, FP4 blocks 8 B, scale
    // words 4 B - every source address must keep that alignment across pages,
    // tokens and heads.
    auto check_span = [](KVPageFormatSpan const& s, uint32_t block_align, char const* name) {
      auto misaligned = [](void const* ptr, uint32_t a) {
        return ptr != nullptr && reinterpret_cast<uintptr_t>(ptr) % a != 0;
      };
      if (misaligned(s.k_payload, block_align) || misaligned(s.v_payload, block_align) ||
          s.payload_stride.page % block_align || s.payload_stride.token % block_align ||
          s.payload_stride.head % block_align || misaligned(s.k_scales, 4) ||
          misaligned(s.v_scales, 4) || s.scale_stride.page % 4 || s.scale_stride.token % 4 ||
          s.scale_stride.head % 4) {
        throw std::runtime_error(std::string("mixed KV pages: ") + name +
                                 " payload rows must be block aligned and scale rows 4 B aligned");
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
  // [21] Chunk table.  Thread t owns row t/8 (tile T - 16*chunk - t/8) and page
  // slot t%8 (slots 6, 7 idle).  A chunk's (index, tag) pairs are loaded into
  // registers early and stored later; see load().
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
    uint32_t any;
    if constexpr (DYNAMIC) {
      // OR over the 8 lanes of this row (idle slots contribute 0).
      any = __reduce_or_sync(0xFFu << (thread_idx & 24), is_compressed(r.tag) ? 1u : 0u);
    } else {
      any = HAS_COMPRESSED ? 1u : 0u;
    }
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

  // One tile's row of the chunk table in registers (two LDS.128).  w6 = tags
  // 0..3, w7 = tags 4, 5 | valid << 16 | flags << 24.
  struct TileRegs {
    uint32_t pages[PAGES_PER_TILE];
    uint32_t w6, w7;
    CUTLASS_DEVICE uint32_t tag(uint32_t i) const {
      return i < 4 ? (w6 >> (8 * i)) & 0xFFu : (w7 >> (8 * (i - 4))) & 0xFFu;
    }
    CUTLASS_DEVICE uint32_t valid() const { return (w7 >> 16) & 0xFFu; }
    CUTLASS_DEVICE bool any_compressed() const { return (w7 >> 24) & kFlagCompressed; }
    // Pending record: tags, valid, flags and the stage index in one word.
    CUTLASS_DEVICE uint64_t pending_word(uint32_t stage) const {
      return (uint64_t(w7) << 32 | uint64_t(w6)) | (uint64_t(stage) << 60);
    }
  };
  static_assert(NUM_STAGES <= 16, "stage index lives in bits 60..63 of the pending word");

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
  CUTLASS_DEVICE static uint32_t blk_row(uint32_t t) { return t >> 3; }
  CUTLASS_DEVICE static uint32_t blk_blk(uint32_t t) { return t & 7u; }

  struct OperandBases {
    uint8_t const* a16_src0;  // row t/16, chunk t%16 of page 0 (this thread)
    uint8_t const* a16_src1;  // row t/16 + 8
    uint32_t a16_ps;          // page stride, bytes
    uint32_t a16_dst;         // smem address of (row t/16, chunk t%16), stage 0
    uint32_t out0;            // smem address of chunk 2b of row r, stage 0 (first output)
    uint32_t out1;            // chunk 2b+1 of row r (second output)
    uint32_t land8;           // FP8 landing: logical chunk 8+b of row r (D-block 1 line, see land_row_line)
    uint32_t land4;           // FP4 landing: 8 B in chunk 8 + b/2 + 4*(r&1), half b&1
    uint32_t sc_rd;           // smem address of this thread's scale slot, page 0, stage 0 (copy + read)
    float gs8, gs4;           // global scales of this operand (FP8: x 2^112, see e4m3x4_to_a16)
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
  CUTLASS_DEVICE static uint32_t sc_sel(uint32_t t) { return 0x4440u | (t & 3u); }

  template <typename STensor>
  CUTLASS_DEVICE OperandBases make_bases(Params const& p, bool isK, int kv_head_idx, STensor& sX,
                                         uint8_t (*scales)[SCALE_STAGE_BYTES], int thread_idx) const {
    uint32_t const t = static_cast<uint32_t>(thread_idx);
    OperandBases b{};
    if constexpr (HAS_COMPRESSED) {
      uint32_t const r = blk_row(t), k = blk_blk(t);
      uint32_t const sc_base = cute::cast_smem_ptr_to_uint(&scales[0][0]);
      b.out0 = chunk_smem(sX, 0, int(r), 2 * k);
      b.out1 = chunk_smem(sX, 0, int(r), 2 * k + 1);
      b.land8 = chunk_smem(sX, 0, int(r), CHUNKS_PER_BLOCK + k);
      b.land4 = chunk_smem(sX, 0, int(r), CHUNKS_PER_BLOCK + k / 2 + 4 * (r & 1u)) + 8 * (k & 1u);
      b.sc_rd = sc_base + 4 * t;
    }
    if constexpr (HAS_A16) {
      uint32_t const a_r = t / CHUNKS_PER_ROW, a_c = t % CHUNKS_PER_ROW;
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
      // bf16: the decode yields value x 2^-112 (exact), folded back here (exact
      // while block_scale x global < 2^16; the quantizer caps block scales at
      // 128 and the host checks the global).  f16 decodes directly.
      float const fold = std::is_same_v<DTypeKV, cutlass::bfloat16_t> ? 0x1p112f : 1.0f;
      b.gs8 = *(isK ? span.k_global_scale : span.v_global_scale) * fold;
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
    uint8_t const* scales;   // the scale word holding this thread's block, page 0
    uint32_t payload_ps, scale_ps;
  };
  template <uint8_t FORMAT>
  CUTLASS_DEVICE CompressedSrc compressed_src(Params const& p, bool isK, int kv_head_idx,
                                              int thread_idx) const {
    uint32_t const t = static_cast<uint32_t>(thread_idx);
    KVPageFormatSpan const& span = p.transport.formats[FORMAT];
    uint32_t const row = blk_row(t), blk = blk_blk(t);
    constexpr uint32_t BLOCK_BYTES = FORMAT == kTagFP8 ? 16 : 8;
    CompressedSrc s;
    s.payload = static_cast<uint8_t const*>(isK ? span.k_payload : span.v_payload) +
                uint64_t(kv_head_idx) * span.payload_stride.head +
                uint64_t(row) * span.payload_stride.token + blk * BLOCK_BYTES;
    s.scales = (isK ? span.k_scales : span.v_scales) +
               uint64_t(kv_head_idx) * span.scale_stride.head +
               uint64_t(row) * span.scale_stride.token + 4 * (blk / 4);
    s.payload_ps = span.payload_stride.page;
    s.scale_ps = span.scale_stride.page;
    return s;
  }

  // One page's copies.  FULL: the tile has CTA_KV valid tokens (no predicates).
  template <bool FULL>
  CUTLASS_DEVICE void copy_a16_page(OperandBases const& b, uint32_t page, uint32_t i,
                                    uint32_t dst_stage, uint32_t valid, int thread_idx) const {
    uint32_t const a_r = static_cast<uint32_t>(thread_idx) / CHUNKS_PER_ROW;
    uint8_t const* s0 = b.a16_src0 + uint64_t(page) * uint64_t(b.a16_ps);
    uint8_t const* s1 = b.a16_src1 + uint64_t(page) * uint64_t(b.a16_ps);
    uint32_t const d = dst_stage + i * PAGE_REGION_BYTES;
    if constexpr (FULL) {
      mixed_detail::cp16(d, s0);
      mixed_detail::cp16(d + ATOM_BYTES, s1);
    } else {
      bool const v0 = i * TOKENS_PER_PAGE + a_r < valid;
      bool const v1 = i * TOKENS_PER_PAGE + a_r + 8 < valid;
      mixed_detail::cp16_zfill(d, v0 ? s0 : b.a16_src0, v0);
      mixed_detail::cp16_zfill(d + ATOM_BYTES, v1 ? s1 : b.a16_src0, v1);
    }
  }

  template <uint8_t FORMAT, bool FULL>
  CUTLASS_DEVICE void copy_compressed_page(OperandBases const& b, CompressedSrc const& s,
                                           uint32_t page, uint32_t i, uint32_t stage,
                                           uint32_t valid, int thread_idx) const {
    uint32_t const t = static_cast<uint32_t>(thread_idx);
    uint32_t const tok0 = i * TOKENS_PER_PAGE;
    uint8_t const* src = s.payload + uint64_t(page) * uint64_t(s.payload_ps);
    uint32_t const land = (FORMAT == kTagFP8 ? b.land8 : b.land4) + stage * STAGE_BYTES +
                          i * PAGE_REGION_BYTES;
    uint32_t const sdst = b.sc_rd + stage * SCALE_STAGE_BYTES + i * SCALE_PAGE_BYTES;
    uint8_t const* ssrc = s.scales + uint64_t(page) * uint64_t(s.scale_ps);
    if constexpr (FULL) {
      if constexpr (FORMAT == kTagFP8) {
        mixed_detail::cp16(land, src);
      } else {
        mixed_detail::cp8(land, src);
      }
      mixed_detail::cp4(sdst, ssrc);
    } else {
      bool const v = tok0 + blk_row(t) < valid;  // D4: src-size 0 zero-fills block and scale
      if constexpr (FORMAT == kTagFP8) {
        mixed_detail::cp16_zfill(land, v ? src : s.payload, v);
      } else {
        mixed_detail::cp8_zfill(land, v ? src : s.payload, v);
      }
      mixed_detail::cp4_zfill(sdst, v ? ssrc : s.scales, v);
    }
  }

  template <bool FULL>
  CUTLASS_DEVICE void issue_tile_copies(Params const& p, OperandBases const& b, TileRegs const& m,
                                        uint32_t stage, bool isK, int kv_head_idx,
                                        int thread_idx) const {
    uint32_t const valid = m.valid();
    uint32_t const a16_dst_stage = b.a16_dst + stage * STAGE_BYTES;
    CompressedSrc s8{}, s4{};
    if constexpr (HAS_FP8) {
      if (!DYNAMIC || m.any_compressed()) s8 = compressed_src<kTagFP8>(p, isK, kv_head_idx, thread_idx);
    }
    if constexpr (HAS_FP4) {
      if (!DYNAMIC || m.any_compressed()) s4 = compressed_src<kTagFP4>(p, isK, kv_head_idx, thread_idx);
    }
    // Unrolled: the body is a handful of instructions per page; rolled loop
    // control cost as much as the copies.  The per-page format branch (dynamic
    // module) is warp-uniform.
#pragma unroll
    for (uint32_t i = 0; i < PAGES_PER_TILE; ++i) {
      uint32_t const page = m.pages[i];
      if constexpr (STATIC_A16) {
        copy_a16_page<FULL>(b, page, i, a16_dst_stage, valid, thread_idx);
      } else if constexpr (STATIC_FP8) {
        copy_compressed_page<kTagFP8, FULL>(b, s8, page, i, stage, valid, thread_idx);
      } else if constexpr (STATIC_FP4) {
        copy_compressed_page<kTagFP4, FULL>(b, s4, page, i, stage, valid, thread_idx);
      } else {
        uint32_t const f = m.tag(i);
        if (f == kTagFP8) {
          copy_compressed_page<kTagFP8, FULL>(b, s8, page, i, stage, valid, thread_idx);
        } else if (f == kTagFP4) {
          copy_compressed_page<kTagFP4, FULL>(b, s4, page, i, stage, valid, thread_idx);
        } else {
          copy_a16_page<FULL>(b, page, i, a16_dst_stage, valid, thread_idx);
        }
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
  // Per FP8 block: LDS.128 + LDS.32 + scale (PRMT, F2FP, HADD2, FMUL, F2FP.PACK)
  // + 8 x (F2FP, SHF, LOP3, IMAD) + 8 HMUL2 + 2 STS.128 = ~49 instructions.
  // Per FP4 block: LDS.64 + LDS.32 + scale 5 + 2 x 20 (LUT) + 8 HMUL2 + 2 STS.

  // E4M3 block scale x global scale -> A16 broadcast to both halves (bit-identical
  // to static_cast<A16>(float(scale) * global); `global` carries the FP8 fold).
  CUTLASS_DEVICE static uint32_t block_scale_a16x2(uint32_t scale_word, uint32_t sel,
                                                   float global) {
    uint32_t const byte = __byte_perm(scale_word, 0u, sel);
    uint32_t h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(static_cast<uint16_t>(byte)));
    float const v = __half2float(reinterpret_cast<__half const&>(h2)) * global;
    if constexpr (std::is_same_v<DTypeKV, cutlass::half_t>) {
      __half2 const r = __floats2half2_rn(v, v);
      return reinterpret_cast<uint32_t const&>(r);
    } else {
      __nv_bfloat162 const r = __floats2bfloat162_rn(v, v);
      return reinterpret_cast<uint32_t const&>(r);
    }
  }

  // 16 E4M3 (w) -> 8 A16 pairs, scaled by sf2.  Pair 2j, 2j+1 = low, high
  // halves of w[j] (the quantizer's layout).
  CUTLASS_DEVICE static void decode_fp8_block(uint4 const& w, uint32_t sf2, uint4& lo, uint4& hi) {
    mixed_detail::e4m3x4_to_a16<DTypeKV>(w.x, lo.x, lo.y);
    mixed_detail::e4m3x4_to_a16<DTypeKV>(w.y, lo.z, lo.w);
    mixed_detail::e4m3x4_to_a16<DTypeKV>(w.z, hi.x, hi.y);
    mixed_detail::e4m3x4_to_a16<DTypeKV>(w.w, hi.z, hi.w);
    lo.x = mixed_detail::mul_a16x2<DTypeKV>(lo.x, sf2);
    lo.y = mixed_detail::mul_a16x2<DTypeKV>(lo.y, sf2);
    lo.z = mixed_detail::mul_a16x2<DTypeKV>(lo.z, sf2);
    lo.w = mixed_detail::mul_a16x2<DTypeKV>(lo.w, sf2);
    hi.x = mixed_detail::mul_a16x2<DTypeKV>(hi.x, sf2);
    hi.y = mixed_detail::mul_a16x2<DTypeKV>(hi.y, sf2);
    hi.z = mixed_detail::mul_a16x2<DTypeKV>(hi.z, sf2);
    hi.w = mixed_detail::mul_a16x2<DTypeKV>(hi.w, sf2);
  }

  // 16 E2M1 (v: nibbles 0..7 in v.x, 8..15 in v.y; low nibble = even coefficient).
  CUTLASS_DEVICE static void decode_fp4_block(uint2 const& v, uint32_t sf2, uint4& lo, uint4& hi) {
    uint32_t a[4], b[4];
    mixed_detail::e2m1x8_to_a16<DTypeKV>(v.x, a);
    mixed_detail::e2m1x8_to_a16<DTypeKV>(v.y, b);
    lo.x = mixed_detail::mul_a16x2<DTypeKV>(a[0], sf2);
    lo.y = mixed_detail::mul_a16x2<DTypeKV>(a[1], sf2);
    lo.z = mixed_detail::mul_a16x2<DTypeKV>(a[2], sf2);
    lo.w = mixed_detail::mul_a16x2<DTypeKV>(a[3], sf2);
    hi.x = mixed_detail::mul_a16x2<DTypeKV>(b[0], sf2);
    hi.y = mixed_detail::mul_a16x2<DTypeKV>(b[1], sf2);
    hi.z = mixed_detail::mul_a16x2<DTypeKV>(b[2], sf2);
    hi.w = mixed_detail::mul_a16x2<DTypeKV>(b[3], sf2);
  }

  // Per-stage 32-bit bases of one operand's expansion (thread constants + stage).
  struct ExpandBases {
    uint32_t d0, d1, l8, l4, sc;  // outputs, FP8 landing, FP4 landing, scale slot
  };
  CUTLASS_DEVICE static ExpandBases expand_bases(OperandBases const& b, uint32_t stage,
                                                 uint32_t t) {
    uint32_t const so = stage * STAGE_BYTES;
    return {b.out0 + so, b.out1 + so, b.land8 + so, b.land4 + so, b.sc_rd + stage * SCALE_STAGE_BYTES};
  }

  // Packed input of page i.  Loaded one page ahead of the stores (the second
  // output base d1 is not provably disjoint from d0/l4 for ptxas, so a load
  // issued after a store would not be hoisted above it).
  struct Packed {
    uint4 w;  // FP8: 16 B; FP4: 8 B in x, y
  };
  template <bool FP8>
  CUTLASS_DEVICE static Packed load_packed(ExpandBases const& e, uint32_t i) {
    Packed p;
    if constexpr (FP8) {
      p.w = mixed_detail::lds128(e.l8 + i * PAGE_REGION_BYTES);
    } else {
      uint2 const v = mixed_detail::lds64(e.l4 + i * PAGE_REGION_BYTES);
      p.w = uint4{v.x, v.y, 0u, 0u};
    }
    return p;
  }
  template <bool FP8>
  CUTLASS_DEVICE void expand_block(ExpandBases const& e, OperandBases const& b, Packed const& p,
                                   uint32_t sw, uint32_t i, uint32_t t) const {
    uint32_t const sf2 = block_scale_a16x2(sw, sc_sel(t), FP8 ? b.gs8 : b.gs4);
    uint4 lo, hi;
    if constexpr (FP8) {
      decode_fp8_block(p.w, sf2, lo, hi);
    } else {
      decode_fp4_block(uint2{p.w.x, p.w.y}, sf2, lo, hi);
    }
    __syncwarp();  // the row's octet has read its landings (other lanes' output chunks)
    mixed_detail::sts128(e.d0 + i * PAGE_REGION_BYTES, lo);
    mixed_detail::sts128(e.d1 + i * PAGE_REGION_BYTES, hi);
  }

  // One operand's pending tile.  `tv` is the tile's pending word (tags).  Static
  // modules: one body, loads pipelined one page ahead.  Dynamic module: the
  // per-page format is warp-uniform data from the pending word.
  CUTLASS_DEVICE void expand_operand(OperandBases const& b, uint64_t tv, int stage,
                                     uint32_t t) const {
    ExpandBases const e = expand_bases(b, uint32_t(stage), t);
    uint32_t sw[PAGES_PER_TILE];
#pragma unroll
    for (uint32_t i = 0; i < PAGES_PER_TILE; ++i) {
      sw[i] = mixed_detail::lds32(e.sc + i * SCALE_PAGE_BYTES);
    }
    if constexpr (!DYNAMIC) {
      // Two pages per step (16 independent decode chains for the scheduler) with
      // the next two pages' loads issued before this step's stores.
      constexpr bool FP8 = STATIC_FP8;
      static_assert(PAGES_PER_TILE % 2 == 0, "pages are decoded two at a time");
      Packed cur0 = load_packed<FP8>(e, 0), cur1 = load_packed<FP8>(e, 1);
#pragma unroll
      for (uint32_t i = 0; i < PAGES_PER_TILE; i += 2) {
        Packed next0{}, next1{};
        if (i + 2 < PAGES_PER_TILE) {
          next0 = load_packed<FP8>(e, i + 2);
          next1 = load_packed<FP8>(e, i + 3);
        }
        expand_block<FP8>(e, b, cur0, sw[i], i, t);
        expand_block<FP8>(e, b, cur1, sw[i + 1], i + 1, t);
        cur0 = next0;
        cur1 = next1;
      }
    } else {
      auto tag = [&](uint32_t i) { return uint32_t(tv >> (8 * i)) & 0xFFu; };
      auto load = [&](uint32_t i) -> Packed {
        uint32_t const f = tag(i);
        if (f == kTagFP8) return load_packed<true>(e, i);
        if (f == kTagFP4) return load_packed<false>(e, i);
        return Packed{};
      };
      Packed cur = load(0);
#pragma unroll
      for (uint32_t i = 0; i < PAGES_PER_TILE; ++i) {
        Packed next{};
        if (i + 1 < PAGES_PER_TILE) next = load(i + 1);
        uint32_t const f = tag(i);
        if (f == kTagFP8) {
          expand_block<true>(e, b, cur, sw[i], i, t);
        } else if (f == kTagFP4) {
          expand_block<false>(e, b, cur, sw[i], i, t);
        }
        cur = next;
      }
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
  CUTLASS_DEVICE void expand_pending(Operand const& op, int thread_idx) const {
    if constexpr (HAS_COMPRESSED) {
      if (op.pending == 0) return;
      expand_operand(op.bases, op.pending, int(op.pending >> 60), uint32_t(thread_idx));
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
#ifdef MIXED_FA3_TRACE
        if (ft.on) ft.t[0] = ft.t[1] = mixed_detail::globaltimer_ns();
#endif
        expand_pending(K, thread_idx);
#ifdef MIXED_FA3_TRACE
        if (ft.on) ft.t[2] = ft.t[3] = mixed_detail::globaltimer_ns();
#endif
        expand_pending(V, thread_idx);
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
