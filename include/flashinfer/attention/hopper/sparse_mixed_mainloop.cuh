/*
 * Copyright (c) 2026 by FlashInfer contributors.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Paged-KV producer for FA3 (Hopper) with ragged page formats.  Specification:
 * docs/mixed_kv_page_transport_dataflow.md (invariants D1-D5, C1-C5) and
 * docs/mixed_kv_page_transport_flow_as_written.md.
 *
 * Pages carry a one-byte tag: A16, block-scaled E4M3 or block-scaled E2M1
 * (one E4M3 scale per 16 coefficients).  Every KV byte moves the way the stock
 * FA3 paged producer moves it: 128 producer threads issue 16 B cp.async copies
 * straight into the consumer's SW128 K-major smem stage (a 16-token page is
 * too small a TMA box on sm90 - each TMA operation costs ~100-200 ns of issue
 * regardless of size, which made per-page TMA the critical path).
 *
 *  * A16 pages: token t's row is copied by producer thread t and the stage is
 *    committed with cp.async.mbarrier.arrive - the producer never waits.
 *  * Compressed pages: thread t copies token t's packed row (128 B / 64 B) and
 *    its 8 B of scales; the packed row lands in the page's last 64-element
 *    D-block of the same stage.  The pair's copies form one cp.async commit
 *    group.  One pair later the producer waits for that group (its bytes have
 *    had a full pair time to land), expands each token in place - one thread
 *    per token, so no cross-thread hazard exists - fences, and commits.
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
  // Page metadata ring: the pending pair (two tiles), the current pair (two
  // tiles, one shared with the pending pair) and the tile gathered one pair
  // ahead: tiles t+1, t, t-1, t-2 -> four slots.
  static constexpr uint32_t META_RING = 4;

  static_assert(HEAD_DIM == 128 && sizeof(DTypeKV) == 2,
                "mixed pages are implemented for D=128 A16 (bf16/f16) math");
  static_assert(CTA_KV % TOKENS_PER_PAGE == 0, "KV tile must be whole pages");
  static_assert(CTA_KV <= NUM_COPY_THREADS, "one producer thread per token");
  static_assert(PAGES_PER_TILE <= 32, "one lane per page");
  static_assert(get<1>(TileShape_PDV{}) == HEAD_DIM && get<2>(TileShape_PDV{}) == CTA_KV);
  static_assert(Ktraits::NUM_PRODUCER_THREADS == NUM_COPY_THREADS,
                "the mixed mainloop uses the whole producer warp group");
  static_assert(BLOCKS_PER_HEAD == 8, "scales are copied as one 8 B word per token");
  static_assert(NUM_STAGES >= 3, "pending pair + current pair + consumer need three stages");
  static_assert(Ktraits::SharedStorage::kMixedMetaRing == META_RING, "metadata ring size");

  static constexpr bool UseSchedulerBarrier = Base::UseSchedulerBarrier;
  static constexpr bool USE_TMA_LOAD_KV = false;
  static constexpr uint32_t TmaTransactionBytesQ = Base::TmaTransactionBytesQ;
  static constexpr uint32_t TmaTransactionBytesK = 0;
  static constexpr uint32_t TmaTransactionBytesV = 0;

  // C3: register budgets.  Producer live set ~60 (one D-block of packed words,
  // four scale words, addresses); consumers ~150.  104*128 + 200*256 = 64512.
  static constexpr int kProducerRegs = 104;
  static constexpr int kConsumerRegs = 200;

  static constexpr uint8_t kMixedBad = 0xFF;

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
    int64_t k_page_stride;  // elements
    int64_t v_page_stride;
    int64_t k_token_stride;
    int64_t v_token_stride;
    int64_t k_head_stride;
    int64_t v_head_stride;
    IdType* kv_indices;
    int window_left;
    uint32_t page_size;
    KVPageTransport<DTypeKV> transport;
    int static_format;  // -1: per-page tags; 0/1/2: every page has this format
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
    if ((ap.mixed_static_format == 1 && (ap.mixed_fp8_k_payload == nullptr ||
                                         ap.mixed_fp8_v_payload == nullptr)) ||
        (ap.mixed_static_format == 2 && (ap.mixed_fp4_k_payload == nullptr ||
                                         ap.mixed_fp4_v_payload == nullptr))) {
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
    return {args.layout_Q,
            tma_load_Q,
            const_cast<DTypeKV*>(args.K_ptr),
            const_cast<DTypeKV*>(args.V_ptr),
            args.k_page_stride,
            args.v_page_stride,
            static_cast<int64_t>(stride<0>(args.layout_K)),
            static_cast<int64_t>(stride<0>(args.layout_V)),
            static_cast<int64_t>(stride<2>(args.layout_K)),
            static_cast<int64_t>(stride<2>(args.layout_V)),
            const_cast<IdType*>(args.kv_indices),
            args.window_left,
            args.page_size,
            transport,
            static_cast<int>(ap.mixed_static_format),
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

  // ------------------------------------------------------------------------
  // Tile metadata (shared by K and V of the same tile): smem ring, C1/C2.
  // Lanes 0..PAGES_PER_TILE-1 resolve one page each; lane 0 records the valid
  // token count.  Visibility to the group comes from the next group barrier.
  CUTLASS_DEVICE void gather_tile_meta(Params const& p, IdType const* kv_indices_ptr,
                                       int kv_tile_idx, int kv_len, IdType* pages,
                                       uint8_t* formats, uint32_t* valid_tokens,
                                       int thread_idx) const {
    int const tile_tok0 = kv_tile_idx * CTA_KV;
    if (thread_idx == 0) {
      int const cta_kv = CTA_KV;
      *valid_tokens = static_cast<uint32_t>(max(0, min(cta_kv, kv_len - tile_tok0)));
    }
    if (thread_idx < int(PAGES_PER_TILE)) {
      int const tok0 = tile_tok0 + thread_idx * int(TOKENS_PER_PAGE);
      if (tok0 < kv_len) {
        IdType const page = kv_indices_ptr[tok0 / int(TOKENS_PER_PAGE)];
        pages[thread_idx] = page;
        formats[thread_idx] = p.static_format >= 0 ? static_cast<uint8_t>(p.static_format)
                                                   : p.transport.page_format[page];
      } else {
        pages[thread_idx] = 0;
        formats[thread_idx] = kMixedBad;
      }
    }
  }

  CUTLASS_DEVICE static bool is_compressed(uint8_t f) {
    return f == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8) ||
           f == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4);
  }

  CUTLASS_DEVICE static bool tile_has_compressed(uint8_t const* formats) {
    bool any = false;
#pragma unroll
    for (uint32_t i = 0; i < PAGES_PER_TILE; ++i) any |= is_compressed(formats[i]);
    return any;
  }

  // 16 B chunk c of element row `row` of stage `stage`: CuTe applies the SW128
  // swizzle, so the copies and the expansion agree on the layout by construction (D1).
  template <typename STensor>
  CUTLASS_DEVICE static void* chunk_ptr(STensor& sX, int stage, int row, uint32_t chunk) {
    return &sX(row, int(chunk * CHUNK_ELEMS), stage);
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
  // address is (thread constant) + (page term), and every destination is
  // (thread constant) + i * PAGE_REGION_BYTES (+ ATOM_BYTES for rows 8..15),
  // because the stage is tiled from 8-row x 128 B atoms in row-group-major
  // order within each D-block (D1 static_asserts in the shared storage).
  static constexpr uint32_t ATOM_BYTES = 8 * 128;
  static constexpr uint32_t PAGE_REGION_BYTES = 2 * ATOM_BYTES;  // 16 rows of one D-block

  template <typename STensor>
  CUTLASS_DEVICE void issue_tile_copies(Params const& p, IdType const* pages,
                                        uint8_t const* formats, uint32_t valid_tokens,
                                        STensor& sX, int stage, bool isK, int kv_head_idx,
                                        uint8_t (*scales)[BLOCKS_PER_HEAD], int thread_idx) const {
    uint32_t const t = static_cast<uint32_t>(thread_idx);
    // A16 ownership constants.
    uint32_t const a_r = t / CHUNKS_PER_ROW;  // 0..7
    uint32_t const a_c = t % CHUNKS_PER_ROW;
    int64_t const a16_ps = isK ? p.k_page_stride : p.v_page_stride;
    int64_t const a16_ts = isK ? p.k_token_stride : p.v_token_stride;
    DTypeKV const* a16_base =
        (isK ? p.K_ptr : p.V_ptr) + int64_t(kv_head_idx) * (isK ? p.k_head_stride : p.v_head_stride);
    DTypeKV const* a16_src0 = a16_base + int64_t(a_r) * a16_ts + a_c * CHUNK_ELEMS;
    uint8_t* const a16_dst0 = reinterpret_cast<uint8_t*>(chunk_ptr(sX, stage, int(a_r), a_c));
    // Compressed ownership constants (FP8: 16 x 8 chunks; FP4: 16 x 4 chunks).
    uint32_t const f8_r = t / FP8_ROW_CHUNKS, f8_c = t % FP8_ROW_CHUNKS;  // t < 128
    uint32_t const f4_r = (t / FP4_ROW_CHUNKS) % TOKENS_PER_PAGE, f4_c = t % FP4_ROW_CHUNKS;  // t < 64
    uint8_t* const f8_dst0 = reinterpret_cast<uint8_t*>(
        chunk_ptr(sX, stage, int(f8_r), (D_BLOCKS - 1) * CHUNKS_PER_BLOCK + f8_c));
    uint8_t* const f4_dst0 = reinterpret_cast<uint8_t*>(
        chunk_ptr(sX, stage, int(f4_r), (D_BLOCKS - 1) * CHUNKS_PER_BLOCK + f4_c));

    // Unrolled: the body is a dozen instructions per page, and rolled loop
    // control cost as much as the copies (PC sampling: BRA/LEA/VIADD-dominated
    // producer).  The per-page format branch is warp-uniform.
#pragma unroll
    for (uint32_t i = 0; i < PAGES_PER_TILE; ++i) {
      uint8_t const f = formats[i];
      uint32_t const page = static_cast<uint32_t>(pages[i]);
      uint32_t const tok0 = i * TOKENS_PER_PAGE;
      uint8_t* const dst_page = a16_dst0 + i * PAGE_REGION_BYTES;
      if (!is_compressed(f)) {
        // A16, or beyond the end (f == kMixedBad: every row is >= valid_tokens).
        DTypeKV const* src = a16_src0 + int64_t(page) * a16_ps;
        bool const v0 = tok0 + a_r < valid_tokens;
        bool const v1 = tok0 + a_r + 8 < valid_tokens;
        cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(
            dst_page, v0 ? src : a16_base, v0);
        cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(
            dst_page + ATOM_BYTES, v1 ? src + 8 * a16_ts : a16_base, v1);
        continue;
      }
      KVPageFormatSpan const& span = p.transport.formats[f];
      bool const isFP8 = f == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8);
      uint8_t const* const payload_page =
          static_cast<uint8_t const*>(isK ? span.k_payload : span.v_payload) +
          uint64_t(page) * span.payload_stride.page +
          uint64_t(kv_head_idx) * span.payload_stride.head;
      if (isFP8) {
        if (tok0 + f8_r < valid_tokens) {
          cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(
              f8_dst0 + i * PAGE_REGION_BYTES,
              payload_page + uint64_t(f8_r) * span.payload_stride.token + f8_c * 16, true);
        }
      } else if (t < TOKENS_PER_PAGE * FP4_ROW_CHUNKS) {
        if (tok0 + f4_r < valid_tokens) {
          cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Global>(
              f4_dst0 + i * PAGE_REGION_BYTES,
              payload_page + uint64_t(f4_r) * span.payload_stride.token + f4_c * 16, true);
        }
      }
      if (t < TOKENS_PER_PAGE && tok0 + t < valid_tokens) {
        uint8_t const* srow = (isK ? span.k_scales : span.v_scales) +
                              uint64_t(page) * span.scale_stride.page +
                              uint64_t(t) * span.scale_stride.token +
                              uint64_t(kv_head_idx) * span.scale_stride.head;
        cutlass::arch::cp_async_zfill<8, cutlass::arch::CacheOperation::Always>(scales[tok0 + t],
                                                                                srow, true);
      }
    }
  }

  // ------------------------------------------------------------------------
  // The ragged part of the design, as data: a compressed format is a row of
  // `row_chunks` 16 B chunks holding 8 scale blocks of 16 coefficients, and one
  // decode primitive from a block's bits to eight A16 pairs.  Everything else
  // (scales, multiply, stores, addressing) is one body shared by all formats
  // and both operands.
  struct FormatDesc {
    bool isFP8;
    uint32_t row_chunks;  // 8 (E4M3) or 4 (E2M1)
    float global_scale;
  };
  CUTLASS_DEVICE static FormatDesc format_desc(Params const& p, uint8_t f, bool isK) {
    bool const isFP8 = f == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8);
    KVPageFormatSpan const& span = p.transport.formats[isFP8 ? 1 : 2];
    return FormatDesc{isFP8, isFP8 ? FP8_ROW_CHUNKS : FP4_ROW_CHUNKS,
                      *(isK ? span.k_global_scale : span.v_global_scale)};
  }

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
  template <typename STensor>
  CUTLASS_DEVICE void expand_token(Params const& p, uint8_t const* formats, uint32_t valid_tokens,
                                   STensor& sX, int stage, bool isK,
                                   uint8_t const (*scales)[BLOCKS_PER_HEAD], int thread_idx) const {
    if (thread_idx >= CTA_KV) return;
    uint32_t const tok = thread_idx;
    uint8_t const f = formats[tok / TOKENS_PER_PAGE];
    if (!is_compressed(f)) return;  // A16 rows were copied or zero-filled by cp.async
    int const row = int(tok);
    auto out = [&](uint32_t k) -> uint4* {
      return reinterpret_cast<uint4*>(chunk_ptr(sX, stage, row, k));
    };
    if (tok >= valid_tokens) {
      uint4 const zero{0u, 0u, 0u, 0u};
#pragma unroll 1
      for (uint32_t k = 0; k < CHUNKS_PER_ROW; ++k) *out(k) = zero;
      return;
    }
    FormatDesc const fd = format_desc(p, f, isK);
    // Not unrolled: one body for both formats (unrolling lets the compiler
    // unswitch it into two).
#pragma unroll 1
    for (uint32_t step = 0; step < BLOCKS_PER_HEAD; ++step) {
      uint32_t const b = (!fd.isFP8 && step >= 5) ? (step == 7 ? 5 : step + 1) : step;
      uint4 w;
      if (fd.isFP8) {
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
      uint32_t const sf2 = block_scale_a16x2(scales[tok][b], fd.global_scale);
      uint4 lo, hi;
      decode_block(fd.isFP8, w, lo, hi);
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
  // An issued-but-uncommitted compressed tile of one operand.
  struct PendingTile {
    bool active;
    PipelineState state;
    int slot;
  };
  struct Operand {
    MainloopPipeline* pipeline;
    PipelineState* state;
    uint8_t (*scales)[CTA_KV][BLOCKS_PER_HEAD];
    bool isK;
    PendingTile pending;  // finished (waited, expanded, committed) this iteration
    PendingTile staged;   // issued this iteration; becomes `pending` after the finish
  };

  template <typename STensor>
  CUTLASS_DEVICE void issue_operand(Params const& p, Operand& op, STensor& sX, IdType const* pages,
                                    uint8_t const* formats, uint32_t valid_tokens, int slot,
                                    int kv_head_idx, int thread_idx) const {
    int const stage = op.state->index();
    issue_tile_copies(p, pages, formats, valid_tokens, sX, stage, op.isK, kv_head_idx,
                      op.scales[stage], thread_idx);
    if (tile_has_compressed(formats)) {
      op.staged = PendingTile{true, *op.state, slot};
    } else {
      op.staged.active = false;
      op.pipeline->producer_commit(*op.state, cutlass::arch::cpasync_barrier_arrive);
    }
    ++(*op.state);
  }

  // The tile issued this iteration is finished next iteration (one pair behind).
  CUTLASS_DEVICE static void rotate_pending(Operand& op) {
    op.pending = op.staged;
    op.staged.active = false;
  }

  // Caller guarantees the operand's copies landed and are visible (cp.async
  // wait + group barrier).
  template <typename STensor>
  CUTLASS_DEVICE void finish_pending(Params const& p, Operand& op, STensor& sX,
                                     uint8_t (*meta_formats)[PAGES_PER_TILE],
                                     uint32_t* meta_valid, int thread_idx) const {
    if (!op.pending.active) return;
    int const stage = op.pending.state.index();
    expand_token(p, meta_formats[op.pending.slot], meta_valid[op.pending.slot], sX, stage, op.isK,
                 op.scales[stage], thread_idx);
    cutlass::arch::fence_view_async_shared();  // D5
    op.pipeline->producer_commit(op.pending.state);
    op.pending.active = false;
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
    int kv_tile_idx = num_kv_tiles - 1;
    int swa_begin_kv_tile_idx = 0;
    if constexpr (LEFT_SLIDING_WINDOW) {
      swa_begin_kv_tile_idx = get_swa_begin_kv_tile_idx<CTA_Q, CTA_KV>(mainloop_params.window_left,
                                                                       q_tile_idx, qo_len, kv_len);
    }
    IdType const* kv_indices_ptr = mainloop_params.kv_indices + kv_indptr;

    auto* meta_pages = shared_storage.mixed_meta_pages;      // [META_RING][PAGES]
    auto* meta_formats = shared_storage.mixed_meta_formats;  // [META_RING][PAGES]
    auto* meta_valid = shared_storage.mixed_meta_valid;      // [META_RING]
    auto slot = [](int tile) { return tile % int(META_RING); };
    auto gather = [&](int tile) {
      if (tile >= 0) {
        gather_tile_meta(mainloop_params, kv_indices_ptr, tile, kv_len, meta_pages[slot(tile)],
                         meta_formats[slot(tile)], &meta_valid[slot(tile)], thread_idx);
      }
    };
    auto group_barrier = [&]() {
      cutlass::arch::NamedBarrier::sync(NUM_COPY_THREADS,
                                        static_cast<int>(NamedBarriers::kProducerWG));
    };
    PendingTile const none{false, PipelineState{}, 0};
    Operand K{&pipeline_k, &smem_pipe_write_k, shared_storage.mixed_scales_k, true, none, none};
    Operand V{&pipeline_v, &smem_pipe_write_v, shared_storage.mixed_scales_v, false, none, none};

    // Wait for the previous pair's compressed copies (all groups but the newest),
    // make them visible to the group, expand and commit them.
    auto finish_pending_pair = [&](bool allow_newest_group) {
      if (K.pending.active || V.pending.active) {
        if (allow_newest_group) {
          cutlass::arch::cp_async_wait<1>();
        } else {
          cutlass::arch::cp_async_wait<0>();
        }
        group_barrier();  // barrier B: every thread's copies of the pending pair landed
        // Explicit call sites: selecting an Operand by a runtime reference would
        // force both structs into local memory (C2).  The body is small now.
        finish_pending(mainloop_params, K, sK, meta_formats, meta_valid, thread_idx);
        finish_pending(mainloop_params, V, sV, meta_formats, meta_valid, thread_idx);
      }
    };

#ifdef MIXED_FA3_TRACE
    uint64_t const tr_enter = mixed_detail::globaltimer_ns();
    uint64_t tr_after_q = 0, tr_first_t0 = 0, tr_prev_end = 0;
    uint64_t tr_acq = 0, tr_bar = 0, tr_gat = 0, tr_iss = 0, tr_fin = 0, tr_gap = 0;
    int tr_n = 0;
#endif
    // Metadata for the first two tiles processed (t and t-1), C1.
    gather(kv_tile_idx);
    gather(kv_tile_idx - 1);

    // --- pairs.  tK or tV may be -1 (absent).  One inlined copy serves K(last)
    // alone, the steady-state pairs (K(t-1), V(t)) and the final V(swa_begin).
    auto produce_pair = [&](int tK, int tV) {
#ifdef MIXED_FA3_TRACE
      bool const trace = (blockIdx.x == 0) && (work_idx <= 1) && (thread_idx == 0);
      uint64_t tr0 = 0, tr1 = 0, tr2 = 0, tr3 = 0, tr4 = 0, tr5 = 0;
      if (trace) tr0 = mixed_detail::globaltimer_ns();
#endif
      if (tK >= 0) pipeline_k.producer_acquire(smem_pipe_write_k);
      if (tV >= 0) pipeline_v.producer_acquire(smem_pipe_write_v);
#ifdef MIXED_FA3_TRACE
      if (trace) tr1 = mixed_detail::globaltimer_ns();
#endif
      group_barrier();  // barrier A: metadata of tK, tV visible; slot (tK-1) free
#ifdef MIXED_FA3_TRACE
      if (trace) tr2 = mixed_detail::globaltimer_ns();
#endif
      // One pair ahead (C1): tiles tK+2 (pending), tK+1, tK, tK-1 -> four slots.
      if (tV >= 0) gather(tK - 1);
#ifdef MIXED_FA3_TRACE
      if (trace) tr3 = mixed_detail::globaltimer_ns();
#endif
      if (tK >= 0) {
        issue_operand(mainloop_params, K, sK, meta_pages[slot(tK)], meta_formats[slot(tK)],
                      meta_valid[slot(tK)], slot(tK), kv_head_idx, thread_idx);
      }
      if (tV >= 0) {
        issue_operand(mainloop_params, V, sV, meta_pages[slot(tV)], meta_formats[slot(tV)],
                      meta_valid[slot(tV)], slot(tV), kv_head_idx, thread_idx);
      }
      cutlass::arch::cp_async_fence();  // this pair's copies form one commit group
#ifdef MIXED_FA3_TRACE
      if (trace) tr4 = mixed_detail::globaltimer_ns();
#endif
      // Previous pair's compressed tiles: their group is complete once at most
      // one group (this pair's) is outstanding.  Then this pair becomes pending.
      finish_pending_pair(/*allow_newest_group=*/true);
      rotate_pending(K);
      rotate_pending(V);
#ifdef MIXED_FA3_TRACE
      if (trace) {
        tr5 = mixed_detail::globaltimer_ns();
        if (tr_n == 0) tr_first_t0 = tr0; else tr_gap += tr0 - tr_prev_end;
        tr_acq += tr1 - tr0; tr_bar += tr2 - tr1; tr_gat += tr3 - tr2; tr_iss += tr4 - tr3;
        tr_fin += tr5 - tr4; tr_prev_end = tr5; ++tr_n;
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
#pragma unroll 1
    for (int t = kv_tile_idx; t >= swa_begin_kv_tile_idx; --t) {
      if (t == swa_begin_kv_tile_idx) scheduler.prefetch_next_work(scheduler_params, work_tile_info);
      produce_pair(t - 1 >= swa_begin_kv_tile_idx ? t - 1 : -1, t);
    }
    // Drain: the last pair's compressed tiles (nothing newer to overlap with).
    finish_pending_pair(/*allow_newest_group=*/false);
    // The trailing group barrier orders the next work item's initial metadata
    // gathers after every reader of this item's ring slots.
    group_barrier();
#ifdef MIXED_FA3_TRACE
    if (blockIdx.x == 0 && thread_idx == 0 && work_idx <= 1) {
      uint64_t const tr_exit = mixed_detail::globaltimer_ns();
      printf("[w%d] item %6llu ns: enter->afterQ %6llu | afterQ->pair0 %6llu | pairs(%d) %6llu "
             "= acq %5llu bar %4llu gat %5llu iss %6llu fin %6llu gap %6llu | lastpair->exit %5llu\n",
             work_idx, (unsigned long long)(tr_exit - tr_enter),
             (unsigned long long)(tr_after_q - tr_enter),
             (unsigned long long)(tr_first_t0 - tr_after_q), tr_n,
             (unsigned long long)(tr_prev_end - tr_first_t0), (unsigned long long)tr_acq,
             (unsigned long long)tr_bar, (unsigned long long)tr_gat, (unsigned long long)tr_iss,
             (unsigned long long)tr_fin, (unsigned long long)tr_gap,
             (unsigned long long)(tr_exit - tr_prev_end));
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
