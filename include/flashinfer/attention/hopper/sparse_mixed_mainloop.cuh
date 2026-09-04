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
 *  * Compressed pages: thread t copies one packed 16 B chunk (FP8: chunk t%8 of
 *    row t/8; FP4: threads < 64, chunk t%4 of row t/4) into the page's last
 *    64-element D-block of the same stage, threads < 16 copy row t's 8 B of
 *    scales.  The pair's copies form one cp.async commit group.  One pair later
 *    the producer waits for that group (its bytes have had a full pair time to
 *    land), expands each token in place - one thread per token, so no
 *    cross-thread hazard exists - fences, and commits.
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

// E2M1 x8 -> eight A16 values as four packed pairs, via CUTLASS's prmt LUT.
template <typename A16>
CUTLASS_DEVICE void e2m1x8_to_a16(uint32_t src, uint32_t (&out)[4]) {
  if constexpr (std::is_same_v<A16, cutlass::half_t>) {
    cutlass::detail::_e2m1_to_half_x8(src, out[0], out[1], out[2], out[3]);
  } else {
    cutlass::detail::_e2m1_to_bf16_x8(src, out[0], out[1], out[2], out[3]);
  }
}

template <typename A16>
CUTLASS_DEVICE uint32_t e4m3x2_to_a16(uint16_t fp8x2) {
  uint32_t fp16x2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(fp16x2) : "h"(fp8x2));
  if constexpr (std::is_same_v<A16, cutlass::half_t>) {
    return fp16x2;
  } else {
    __half2 const h = reinterpret_cast<__half2 const&>(fp16x2);
    __nv_bfloat162 const b = __float22bfloat162_rn(__half22float2(h));
    return reinterpret_cast<uint32_t const&>(b);
  }
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
  static constexpr uint32_t SCALE_ROW_BYTES = 8;
  static constexpr uint32_t SCALE_STAGE_BYTES = CTA_KV * SCALE_ROW_BYTES;

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
  static_assert(BLOCKS_PER_HEAD == 8, "scales are copied as one 8 B word per token");
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
  //  * FP8 page   : chunk t%8 of row t/8, into D-block 1 (D2)
  //  * FP4 page   : threads < 64, chunk t%4 of row t/4, into D-block 1
  //  * compressed : threads < 16 also copy row t's 8 B of scales.
  // A thread's row-within-page and chunk are therefore constants: every source
  // address is (thread constant) + page * (byte stride) - one IMAD.WIDE.U32 - and
  // every destination is (thread constant) + stage * STAGE_BYTES + i * PAGE_REGION_BYTES
  // (+ ATOM_BYTES for rows 8..15), because the stage is tiled from 8-row x 128 B
  // atoms in row-group-major order within each D-block (D1 static_asserts in the
  // shared storage).  The thread constants are computed once per work item.
  struct OperandBases {
    uint8_t const* a16_src0;  // row t/16, chunk t%16 of page 0 (this thread)
    uint8_t const* a16_src1;  // row t/16 + 8
    uint32_t a16_ps;          // page stride, bytes
    uint32_t a16_dst;         // smem address of (row t/16, chunk t%16), stage 0
    uint32_t f8_dst;          // smem address of the packed FP8 chunk, stage 0
    uint32_t f4_dst;          // ... FP4
    uint32_t sc_dst;          // smem address of scales[stage 0][t]
    float gs8, gs4;           // global scales of this operand
  };

  template <typename STensor>
  CUTLASS_DEVICE OperandBases make_bases(Params const& p, bool isK, int kv_head_idx, STensor& sX,
                                         uint8_t (*scales)[CTA_KV][BLOCKS_PER_HEAD],
                                         int thread_idx) const {
    uint32_t const t = static_cast<uint32_t>(thread_idx);
    OperandBases b{};
    if constexpr (HAS_COMPRESSED) {
      b.sc_dst = cute::cast_smem_ptr_to_uint(&scales[0][t % CTA_KV][0]);
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
      b.f8_dst = chunk_smem(sX, 0, int(t / FP8_ROW_CHUNKS),
                            (D_BLOCKS - 1) * CHUNKS_PER_BLOCK + t % FP8_ROW_CHUNKS);
      auto const& span = p.transport.formats[kTagFP8];
      b.gs8 = *(isK ? span.k_global_scale : span.v_global_scale);
    }
    if constexpr (HAS_FP4) {
      b.f4_dst = chunk_smem(sX, 0, int((t / FP4_ROW_CHUNKS) % TOKENS_PER_PAGE),
                            (D_BLOCKS - 1) * CHUNKS_PER_BLOCK + t % FP4_ROW_CHUNKS);
      auto const& span = p.transport.formats[kTagFP4];
      b.gs4 = *(isK ? span.k_global_scale : span.v_global_scale);
    }
    return b;
  }

  // Per-tile source bases of a compressed format (recomputed per tile: a few
  // IMAD.WIDE per operand, cheaper than holding them across the pair loop).
  struct CompressedSrc {
    uint8_t const* payload;  // this thread's packed chunk of page 0
    uint8_t const* scales;   // this thread's scale row of page 0
    uint32_t payload_ps, scale_ps;
  };
  template <uint8_t FORMAT>
  CUTLASS_DEVICE CompressedSrc compressed_src(Params const& p, bool isK, int kv_head_idx,
                                              int thread_idx) const {
    uint32_t const t = static_cast<uint32_t>(thread_idx);
    KVPageFormatSpan const& span = p.transport.formats[FORMAT];
    uint32_t const row = FORMAT == kTagFP8 ? t / FP8_ROW_CHUNKS : (t / FP4_ROW_CHUNKS) % TOKENS_PER_PAGE;
    uint32_t const chunk = FORMAT == kTagFP8 ? t % FP8_ROW_CHUNKS : t % FP4_ROW_CHUNKS;
    CompressedSrc s;
    s.payload = static_cast<uint8_t const*>(isK ? span.k_payload : span.v_payload) +
                uint64_t(kv_head_idx) * span.payload_stride.head +
                uint64_t(row) * span.payload_stride.token + chunk * 16;
    s.scales = (isK ? span.k_scales : span.v_scales) +
               uint64_t(kv_head_idx) * span.scale_stride.head + uint64_t(t) * span.scale_stride.token;
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
    if constexpr (FORMAT == kTagFP8) {
      uint32_t const row = t / FP8_ROW_CHUNKS;
      if (FULL || tok0 + row < valid) {
        mixed_detail::cp16(b.f8_dst + stage * STAGE_BYTES + i * PAGE_REGION_BYTES, src);
      }
    } else {
      uint32_t const row = (t / FP4_ROW_CHUNKS) % TOKENS_PER_PAGE;
      if (t < TOKENS_PER_PAGE * FP4_ROW_CHUNKS && (FULL || tok0 + row < valid)) {
        mixed_detail::cp16(b.f4_dst + stage * STAGE_BYTES + i * PAGE_REGION_BYTES, src);
      }
    }
    if (t < TOKENS_PER_PAGE && (FULL || tok0 + t < valid)) {
      mixed_detail::cp8(b.sc_dst + stage * SCALE_STAGE_BYTES + tok0 * SCALE_ROW_BYTES,
                        s.scales + uint64_t(page) * uint64_t(s.scale_ps));
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
  // The ragged part of the design, as data: a compressed format is a row of
  // `row_chunks` 16 B chunks holding 8 scale blocks of 16 coefficients, and one
  // decode primitive from a block's bits to eight A16 pairs.  Everything else
  // (scales, multiply, stores, addressing) is one body shared by all formats
  // and both operands.

  // One scale block of coefficients -> 8 A16 pairs.  FP8: 16 bytes in w[0..3]
  // (pair 2j, 2j+1 = low, high halves of w[j]).  FP4: 8 bytes in w[0..1]
  // (pairs 0..3 from w[0], 4..7 from w[1]).  Both match the quantizer's layout.
  CUTLASS_DEVICE static void decode_block(bool isFP8, uint4 const& w, uint4& lo, uint4& hi) {
    if (isFP8) {
      lo.x = mixed_detail::e4m3x2_to_a16<DTypeKV>(static_cast<uint16_t>(w.x));
      lo.y = mixed_detail::e4m3x2_to_a16<DTypeKV>(static_cast<uint16_t>(w.x >> 16));
      lo.z = mixed_detail::e4m3x2_to_a16<DTypeKV>(static_cast<uint16_t>(w.y));
      lo.w = mixed_detail::e4m3x2_to_a16<DTypeKV>(static_cast<uint16_t>(w.y >> 16));
      hi.x = mixed_detail::e4m3x2_to_a16<DTypeKV>(static_cast<uint16_t>(w.z));
      hi.y = mixed_detail::e4m3x2_to_a16<DTypeKV>(static_cast<uint16_t>(w.z >> 16));
      hi.z = mixed_detail::e4m3x2_to_a16<DTypeKV>(static_cast<uint16_t>(w.w));
      hi.w = mixed_detail::e4m3x2_to_a16<DTypeKV>(static_cast<uint16_t>(w.w >> 16));
    } else {
      uint32_t a[4], b[4];
      mixed_detail::e2m1x8_to_a16<DTypeKV>(w.x, a);
      mixed_detail::e2m1x8_to_a16<DTypeKV>(w.y, b);
      lo = uint4{a[0], a[1], a[2], a[3]};
      hi = uint4{b[0], b[1], b[2], b[3]};
    }
  }

  // E4M3 block scale x global scale -> A16 broadcast to both halves (bit-identical
  // to static_cast<A16>(float(scale) * global)).
  CUTLASS_DEVICE static uint32_t block_scale_a16x2(uint8_t scale, float global) {
    uint32_t h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(static_cast<uint16_t>(scale)));
    float const v = __half2float(reinterpret_cast<__half const&>(h2)) * global;
    uint16_t bits;
    if constexpr (std::is_same_v<DTypeKV, cutlass::half_t>) {
      __half const r = __float2half_rn(v);
      bits = reinterpret_cast<uint16_t const&>(r);
    } else {
      __nv_bfloat16 const r = __float2bfloat16_rn(v);
      bits = reinterpret_cast<uint16_t const&>(r);
    }
    return uint32_t(bits) | (uint32_t(bits) << 16);
  }

  // D3/D4: thread t expands token t in place (or zero-fills it).  Blocks are
  // processed in an order under which no output write lands on packed bits not
  // yet read (D2): the packed row occupies the first `row_chunks` slots of
  // D-block 1; block b's output goes to slots 2b, 2b+1.  FP8 (one block per
  // packed chunk): 0..7.  FP4 (two blocks per packed chunk): 0,1,2,3,4,6,7,5.
  // `tv` is the tile's pending word (tags, valid); the global scales come from
  // the operand's per-item bases (no global load in the pair body).
  template <typename STensor>
  CUTLASS_DEVICE void expand_token(uint64_t tv, STensor& sX, int stage, float gs8, float gs4,
                                   uint8_t const (*scales)[BLOCKS_PER_HEAD], int thread_idx) const {
    if (thread_idx >= CTA_KV) return;
    uint32_t const tok = thread_idx;
    uint32_t const f = uint32_t(tv >> (8 * (tok / TOKENS_PER_PAGE))) & 0xFFu;
    uint32_t const valid = uint32_t(tv >> 48) & 0xFFu;
    if (!is_compressed(f)) return;  // A16 rows were copied or zero-filled by cp.async
    int const row = int(tok);
    auto out = [&](uint32_t k) -> uint4* {
      return reinterpret_cast<uint4*>(chunk_ptr(sX, stage, row, k));
    };
    if (tok >= valid) {
      uint4 const zero{0u, 0u, 0u, 0u};
#pragma unroll 1
      for (uint32_t k = 0; k < CHUNKS_PER_ROW; ++k) *out(k) = zero;
      return;
    }
    bool const isFP8 = HAS_FP8 && (!HAS_FP4 || f == kTagFP8);
    float const global_scale = isFP8 ? gs8 : gs4;
    // Not unrolled: one body for both formats (unrolling lets the compiler
    // unswitch it into two).
#pragma unroll 1
    for (uint32_t step = 0; step < BLOCKS_PER_HEAD; ++step) {
      uint32_t const b = (!isFP8 && step >= 5) ? (step == 7 ? 5 : step + 1) : step;
      uint4 w;
      if (isFP8) {
        // Slot b of D-block 1 holds block b (addressed through the layout).
        w = *reinterpret_cast<uint4 const*>(
            chunk_ptr(sX, stage, row, (D_BLOCKS - 1) * CHUNKS_PER_BLOCK + b));
      } else {
        uint2 const v = *reinterpret_cast<uint2 const*>(
            static_cast<uint8_t const*>(
                chunk_ptr(sX, stage, row, (D_BLOCKS - 1) * CHUNKS_PER_BLOCK + b / 2)) +
            (b % 2) * 8);
        w = uint4{v.x, v.y, 0u, 0u};
      }
      uint32_t const sf2 = block_scale_a16x2(scales[tok][b], global_scale);
      uint4 lo, hi;
      decode_block(isFP8, w, lo, hi);
      lo.x = mixed_detail::mul_a16x2<DTypeKV>(lo.x, sf2);
      lo.y = mixed_detail::mul_a16x2<DTypeKV>(lo.y, sf2);
      lo.z = mixed_detail::mul_a16x2<DTypeKV>(lo.z, sf2);
      lo.w = mixed_detail::mul_a16x2<DTypeKV>(lo.w, sf2);
      hi.x = mixed_detail::mul_a16x2<DTypeKV>(hi.x, sf2);
      hi.y = mixed_detail::mul_a16x2<DTypeKV>(hi.y, sf2);
      hi.z = mixed_detail::mul_a16x2<DTypeKV>(hi.z, sf2);
      hi.w = mixed_detail::mul_a16x2<DTypeKV>(hi.w, sf2);
      *out(2 * b) = lo;
      *out(2 * b + 1) = hi;
    }
  }

  // ------------------------------------------------------------------------
  // One operand of a tile.  Two completion modes (C4):
  //  * no compressed page: every thread commits with cp.async.mbarrier.arrive;
  //    the stage completes when the copies land.  Nobody waits.
  //  * otherwise: the operand becomes *pending*; one pair later the group waits
  //    for its commit group, expands, fences (D5) and arrives.
  // A pending record is one word: the tile's tags / valid / flags and the stage
  // index (TileRegs::pending_word); 0 means inactive (a filled row always has
  // kFlagFilled set).
  struct Operand {
    MainloopPipeline* pipeline;
    PipelineState* state;
    uint8_t (*scales)[CTA_KV][BLOCKS_PER_HEAD];
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
  // t[0] after cp.async.wait_group, t[1] after barrier B, t[2] after expand K,
  // t[3] after K fence+commit, t[4] after expand V, t[5] after V fence+commit.
  // An inactive operand leaves its two stamps equal to the previous one.
  struct FinTrace {
    bool on;
    uint64_t t[6];
  };
#endif

  // Caller guarantees the operand's copies landed and are visible (cp.async
  // wait + group barrier).
  template <typename STensor>
  CUTLASS_DEVICE void finish_pending(Operand& op, STensor& sX, int thread_idx
#ifdef MIXED_FA3_TRACE
                                     ,
                                     FinTrace& ft, int ft_base
#endif
                                     ) const {
    if constexpr (HAS_COMPRESSED) {
      if (op.pending == 0) return;
      int const stage = int(op.pending >> 60);
      expand_token(op.pending, sX, stage, op.bases.gs8, op.bases.gs4, op.scales[stage],
                   thread_idx);
#ifdef MIXED_FA3_TRACE
      if (ft.on) ft.t[ft_base] = mixed_detail::globaltimer_ns();
#endif
      cutlass::arch::fence_view_async_shared();  // D5
      // PipelineAsync::producer_commit arrives on full_barrier[state.index()]; only the
      // index of the pending stage matters.
      op.pipeline->producer_commit(PipelineState(stage, 0, 0));
#ifdef MIXED_FA3_TRACE
      if (ft.on) ft.t[ft_base + 1] = mixed_detail::globaltimer_ns();
#endif
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
    Operand K{&pipeline_k, &smem_pipe_write_k, shared_storage.mixed_scales_k, true,
              make_bases(mainloop_params, true, kv_head_idx, sK, shared_storage.mixed_scales_k,
                         thread_idx),
              0, 0};
    Operand V{&pipeline_v, &smem_pipe_write_v, shared_storage.mixed_scales_v, false,
              make_bases(mainloop_params, false, kv_head_idx, sV, shared_storage.mixed_scales_v,
                         thread_idx),
              0, 0};
#ifdef MIXED_FA3_TRACE
    FinTrace ft{false, {0, 0, 0, 0, 0, 0}};
#endif

    // Wait for the previous pair's compressed copies (all groups but the newest),
    // make them visible to the group, expand and commit them.
    auto finish_pending_pair = [&](bool allow_newest_group) {
      if constexpr (HAS_COMPRESSED) {
      if (K.pending != 0 || V.pending != 0) {
        if (allow_newest_group) {
          cutlass::arch::cp_async_wait<1>();
        } else {
          cutlass::arch::cp_async_wait<0>();
        }
#ifdef MIXED_FA3_TRACE
        if (ft.on) ft.t[0] = mixed_detail::globaltimer_ns();
#endif
        group_barrier();  // barrier B: every thread's copies of the pending pair landed
#ifdef MIXED_FA3_TRACE
        if (ft.on) ft.t[1] = ft.t[2] = ft.t[3] = mixed_detail::globaltimer_ns();
#endif
        // Explicit call sites: selecting an Operand by a runtime reference would
        // force both structs into local memory (C2).
        finish_pending(K, sK, thread_idx
#ifdef MIXED_FA3_TRACE
                       ,
                       ft, 2
#endif
        );
#ifdef MIXED_FA3_TRACE
        if (ft.on) ft.t[4] = ft.t[5] = ft.t[3];
#endif
        finish_pending(V, sV, thread_idx
#ifdef MIXED_FA3_TRACE
                       ,
                       ft, 4
#endif
        );
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
