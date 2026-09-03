/*
 * Copyright (c) 2026 by FlashInfer contributors.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Paged-KV producer for FA3 (Hopper) with ragged page formats.  Specification:
 * docs/mixed_kv_page_transport_dataflow.md (invariants D1-D5, C1-C5).
 *
 * Pages carry a one-byte tag: A16, block-scaled E4M3 or block-scaled E2M1
 * (one E4M3 scale per 16 coefficients).  The producer warp group streams a
 * KV tile as pages.  A16 pages go through TMA straight into the consumer's
 * SW128 K-major smem stage and the TMA transaction completes the pipeline's
 * full barrier directly (the producer never waits for them, exactly like a
 * PipelineTmaAsync producer).  Compressed pages go through byte-typed TMA maps
 * (whole-head rows, TMA-swizzled) into the page's last 64-element D-block of
 * the same stage; the producer waits for that tile's transaction barrier,
 * expands the rows in place to A16 - one thread per token, so no cross-thread
 * hazard exists - and then commits the stage.  K(t-1) and V(t) are issued as
 * a pair and waited for once.  The consumer (mma_f16) is unchanged.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_SPARSE_MIXED_MAINLOOP_CUH_
#define FLASHINFER_ATTENTION_HOPPER_SPARSE_MIXED_MAINLOOP_CUH_

#include <cutlass/cutlass.h>
#include <cutlass/pipeline/pipeline.hpp>

#include <cuda.h>

#include <type_traits>

#include "../../math.cuh"
#include "../page_transport.cuh"
#include "../page_transport_tma.cuh"
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

// Four E4M3 block scales -> four A16 scales broadcast to both halves, bit-identical
// to static_cast<A16>(float(scale) * globalScale) per element.
template <typename A16>
CUTLASS_DEVICE void e4m3x4_scales_to_a16x2(uint32_t scaleWord, float globalScale,
                                          uint32_t (&sf2)[4]) {
  uint32_t lo, hi;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(lo) : "h"(static_cast<uint16_t>(scaleWord)));
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(hi) : "h"(static_cast<uint16_t>(scaleWord >> 16)));
  float2 const l = __half22float2(reinterpret_cast<__half2 const&>(lo));
  float2 const h = __half22float2(reinterpret_cast<__half2 const&>(hi));
  float const v[4] = {l.x * globalScale, l.y * globalScale, h.x * globalScale, h.y * globalScale};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint16_t bits;
    if constexpr (std::is_same_v<A16, cutlass::half_t>) {
      __half const s = __float2half_rn(v[i]);
      bits = reinterpret_cast<uint16_t const&>(s);
    } else {
      __nv_bfloat16 const s = __float2bfloat16_rn(v[i]);
      bits = reinterpret_cast<uint16_t const&>(s);
    }
    sf2[i] = uint32_t(bits) | (uint32_t(bits) << 16);
  }
}

CUTLASS_DEVICE void tma_load_page(CUtensorMap const* map, uint64_t* mbar, void* smem_dst,
                                  uint32_t crd0, uint32_t head, uint32_t page) {
  SM90_TMA_LOAD_4D::copy(map, mbar, static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_NORMAL),
                         smem_dst, static_cast<int32_t>(crd0), static_cast<int32_t>(head), 0,
                         static_cast<int32_t>(page));
}

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
  static constexpr uint32_t BLOCKS_PER_HEAD = HEAD_DIM / 16;
  static constexpr uint32_t FP8_ROW_BYTES = HEAD_DIM;
  static constexpr uint32_t FP4_ROW_BYTES = HEAD_DIM / 2;
  static constexpr uint32_t A16_PAGE_PART_BYTES = TOKENS_PER_PAGE * D_BLOCK * sizeof(DTypeKV);
  // Page metadata ring: pairs (K(t-1), V(t)) need tiles t-1 and t while tile
  // t-2 is being gathered for the next pair.
  static constexpr uint32_t META_RING = 3;

  static_assert(HEAD_DIM == 128 && sizeof(DTypeKV) == 2,
                "mixed pages are implemented for D=128 A16 (bf16/f16) math");
  static_assert(CTA_KV % TOKENS_PER_PAGE == 0, "KV tile must be whole pages");
  static_assert(CTA_KV <= NUM_COPY_THREADS, "one producer thread per token");
  static_assert(PAGES_PER_TILE <= 32, "one lane per page");
  static_assert(get<1>(TileShape_PDV{}) == HEAD_DIM && get<2>(TileShape_PDV{}) == CTA_KV);
  static_assert(Ktraits::NUM_PRODUCER_THREADS == NUM_COPY_THREADS,
                "the mixed mainloop uses the whole producer warp group");
  static_assert(BLOCKS_PER_HEAD == 8, "scales are copied as one 8 B word per token");

  static constexpr bool UseSchedulerBarrier = Base::UseSchedulerBarrier;
  static constexpr bool USE_TMA_LOAD_KV = false;  // the producer drives its own TMA
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
    IdType* kv_indices;
    int window_left;
    uint32_t page_size;
    // Six TMA descriptors: A16 K/V (element-typed, SW128), E4M3 K/V and E2M1
    // K/V (byte-typed whole-head rows, 128B / 64B swizzle).
    CUtensorMap mapA16K, mapA16V, mapFP8K, mapFP8V, mapFP4K, mapFP4V;
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
    auto& a16 = transport.formats[static_cast<uint8_t>(KVPageFormat::kA16)];
    a16.k_payload = args.K_ptr;
    a16.v_payload = args.V_ptr;
    a16.payload_stride = {static_cast<uint32_t>(args.k_page_stride * sizeof(DTypeKV)),
                          static_cast<uint32_t>(stride<0>(args.layout_K) * sizeof(DTypeKV)),
                          static_cast<uint32_t>(stride<2>(args.layout_K) * sizeof(DTypeKV))};
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
    if (args.v_page_stride != args.k_page_stride ||
        stride<0>(args.layout_V) != stride<0>(args.layout_K) ||
        stride<2>(args.layout_V) != stride<2>(args.layout_K)) {
      throw std::runtime_error("mixed KV pages: K and V caches must share strides");
    }
    // #7: a static format promises pages of that format exist; its payload must.
    if ((ap.mixed_static_format == 1 && (ap.mixed_fp8_k_payload == nullptr ||
                                         ap.mixed_fp8_v_payload == nullptr)) ||
        (ap.mixed_static_format == 2 && (ap.mixed_fp4_k_payload == nullptr ||
                                         ap.mixed_fp4_v_payload == nullptr))) {
      throw std::runtime_error("mixed KV pages: static format without its payload tensors");
    }
    constexpr CUtensorMapDataType_enum a16Type = std::is_same_v<DTypeKV, cutlass::half_t>
                                                     ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16
                                                     : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    uint32_t const nbKHeads = static_cast<uint32_t>(size<2>(args.layout_K.shape()));  // (N, D, H)
    MixedPageTensorMaps const maps = makeMixedPageTensorMaps<DTypeKV>(
        transport, a16Type, HEAD_DIM, nbKHeads, TOKENS_PER_PAGE, D_BLOCK, TOKENS_PER_PAGE);
    return {args.layout_Q,
            tma_load_Q,
            const_cast<IdType*>(args.kv_indices),
            args.window_left,
            args.page_size,
            maps.a16K,
            maps.a16V,
            maps.fp8K,
            maps.fp8V,
            maps.fp4K,
            maps.fp4V,
            transport,
            static_cast<int>(ap.mixed_static_format),
            args.additional_params};
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(&mainloop_params.mapA16K);
    cute::prefetch_tma_descriptor(&mainloop_params.mapA16V);
    cute::prefetch_tma_descriptor(&mainloop_params.mapFP8K);
    cute::prefetch_tma_descriptor(&mainloop_params.mapFP8V);
    cute::prefetch_tma_descriptor(&mainloop_params.mapFP4K);
    cute::prefetch_tma_descriptor(&mainloop_params.mapFP4V);
  }

  // Called by one thread before the kernel-wide __syncthreads.
  CUTLASS_DEVICE
  static void init_shared(SharedStorage& shared_storage) {
#pragma unroll
    for (int s = 0; s < NUM_STAGES; ++s) {
      shared_storage.mixed_tma_bar_k[s].init(1);
      shared_storage.mixed_tma_bar_v[s].init(1);
    }
    shared_storage.mixed_phase_k = 0;
    shared_storage.mixed_phase_v = 0;
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
  // Tile metadata (shared by K and V of the same tile): smem ring, C2.
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

  CUTLASS_DEVICE static bool tile_has_compressed(uint8_t const* formats) {
    bool any = false;
#pragma unroll
    for (uint32_t i = 0; i < PAGES_PER_TILE; ++i) {
      uint8_t const f = formats[i];
      any |= f == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8) ||
             f == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4);
    }
    return any;
  }

  // Bytes the TMA batch for one tile puts on its barrier (C4).
  CUTLASS_DEVICE static uint32_t tile_tx_bytes(uint8_t const* formats) {
    uint32_t bytes = 0;
#pragma unroll
    for (uint32_t i = 0; i < PAGES_PER_TILE; ++i) {
      uint8_t const f = formats[i];
      if (f == static_cast<uint8_t>(KVPageFormat::kA16)) {
        bytes += D_BLOCKS * A16_PAGE_PART_BYTES;
      } else if (f == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8)) {
        bytes += TOKENS_PER_PAGE * FP8_ROW_BYTES;
      } else if (f == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4)) {
        bytes += TOKENS_PER_PAGE * FP4_ROW_BYTES;
      }
    }
    return bytes;
  }

  // Address of element (row, col) of the K-major SW128 stage tensor sX(_, _, stage).
  template <typename STensor>
  CUTLASS_DEVICE static uint8_t* stage_elem_ptr(STensor& sX, int stage, int row, int col) {
    return reinterpret_cast<uint8_t*>(&sX(row, col, stage));
  }

  // One thread issues a tile's TMA batch onto `mbar`.  A16 pages: one box per
  // D-block into their final rows (D1).  Compressed pages: whole-head rows into
  // the page's last D-block region (D2).
  template <typename STensor>
  CUTLASS_DEVICE void issue_tile_tma(Params const& p, IdType const* pages, uint8_t const* formats,
                                     STensor& sX, int stage, bool isK, int kv_head_idx,
                                     uint64_t* mbar) const {
    CUtensorMap const* a16 = isK ? &p.mapA16K : &p.mapA16V;
    CUtensorMap const* fp8 = isK ? &p.mapFP8K : &p.mapFP8V;
    CUtensorMap const* fp4 = isK ? &p.mapFP4K : &p.mapFP4V;
#pragma unroll
    for (uint32_t i = 0; i < PAGES_PER_TILE; ++i) {
      uint8_t const f = formats[i];
      uint32_t const page = static_cast<uint32_t>(pages[i]);
      int const row0 = int(i * TOKENS_PER_PAGE);
      if (f == static_cast<uint8_t>(KVPageFormat::kA16)) {
#pragma unroll
        for (uint32_t d = 0; d < D_BLOCKS; ++d) {
          mixed_detail::tma_load_page(a16, mbar, stage_elem_ptr(sX, stage, row0, int(d * D_BLOCK)),
                                      d * D_BLOCK, kv_head_idx, page);
        }
      } else if (f == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8)) {
        mixed_detail::tma_load_page(fp8, mbar,
                                    stage_elem_ptr(sX, stage, row0, int((D_BLOCKS - 1) * D_BLOCK)),
                                    0, kv_head_idx, page);
      } else if (f == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4)) {
        mixed_detail::tma_load_page(fp4, mbar,
                                    stage_elem_ptr(sX, stage, row0, int((D_BLOCKS - 1) * D_BLOCK)),
                                    0, kv_head_idx, page);
      }
    }
  }

  // Thread t copies token t's whole-head scales (8 B) for a compressed page.
  CUTLASS_DEVICE void issue_tile_scales(Params const& p, IdType const* pages,
                                        uint8_t const* formats, bool isK, int kv_head_idx,
                                        uint8_t (*dst)[BLOCKS_PER_HEAD], int thread_idx) const {
    if (thread_idx >= CTA_KV) return;
    uint32_t const i = thread_idx / TOKENS_PER_PAGE;
    uint32_t const r = thread_idx % TOKENS_PER_PAGE;
    uint8_t const f = formats[i];
    if (f != static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8) &&
        f != static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4)) {
      return;
    }
    KVPageFormatSpan const& span = p.transport.formats[f];
    uint8_t const* src = (isK ? span.k_scales : span.v_scales) +
                         uint64_t(static_cast<uint32_t>(pages[i])) * span.scale_stride.page +
                         uint64_t(r) * span.scale_stride.token +
                         uint64_t(kv_head_idx) * span.scale_stride.head;
    cutlass::arch::cp_async_zfill<8, cutlass::arch::CacheOperation::Always>(dst[thread_idx], src,
                                                                            true);
  }

  // D3/D4: thread t expands token t in place (or zero-fills it).  The packed
  // row is read completely before the D-block-1 rows, which overwrite it, are
  // written; one D-block of output at a time bounds live registers (C3).
  template <typename STensor>
  CUTLASS_DEVICE void expand_token(Params const& p, uint8_t const* formats, uint32_t valid_tokens,
                                   STensor& sX, int stage, bool isK,
                                   uint8_t const (*scales)[BLOCKS_PER_HEAD], int thread_idx) const {
    if (thread_idx >= CTA_KV) return;
    uint32_t const tok = thread_idx;
    uint32_t const i = tok / TOKENS_PER_PAGE;
    uint32_t const r = tok % TOKENS_PER_PAGE;
    uint8_t const f = formats[i];
    bool const beyond = tok >= valid_tokens;
    if (f == static_cast<uint8_t>(KVPageFormat::kA16) && !beyond) return;

    auto chunk_ptr = [&](uint32_t d, uint32_t c) -> uint4* {
      return reinterpret_cast<uint4*>(
          stage_elem_ptr(sX, stage, int(i * TOKENS_PER_PAGE + r), int(d * D_BLOCK + c * 8)));
    };
    if (beyond || f == kMixedBad) {
      uint4 const zero{0u, 0u, 0u, 0u};
#pragma unroll
      for (uint32_t d = 0; d < D_BLOCKS; ++d) {
#pragma unroll
        for (uint32_t c = 0; c < 8; ++c) *chunk_ptr(d, c) = zero;
      }
      return;
    }
    bool const isFP8 = f == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8);
    KVPageFormatSpan const& span = p.transport.formats[isFP8 ? 1 : 2];
    float const globalScale = *(isK ? span.k_global_scale : span.v_global_scale);
    uint8_t const* packed_page =
        stage_elem_ptr(sX, stage, int(i * TOKENS_PER_PAGE), int((D_BLOCKS - 1) * D_BLOCK));
    uint32_t const scaleWord0 = *reinterpret_cast<uint32_t const*>(&scales[tok][0]);
    uint32_t const scaleWord1 = *reinterpret_cast<uint32_t const*>(&scales[tok][4]);

    if (isFP8) {
      // 128 B rows, 128B swizzle: chunk ^= r % 8.
      // The D-block-0 output rows do not overlap the packed row (it lives in the
      // D-block-1 region), so each D-block's four chunks are read just before
      // that D-block is written; the packed row is fully consumed before the
      // D-block-1 writes that overwrite it.
      uint8_t const* row = packed_page + r * FP8_ROW_BYTES;
#pragma unroll
      for (uint32_t d = 0; d < D_BLOCKS; ++d) {
        uint4 words[4];
#pragma unroll
        for (uint32_t cc = 0; cc < 4; ++cc) {
          uint32_t const c = d * 4 + cc;
          words[cc] = *reinterpret_cast<uint4 const*>(row + ((c ^ (r % 8)) * 16));
        }
        uint32_t sf2[4];
        mixed_detail::e4m3x4_scales_to_a16x2<DTypeKV>(d == 0 ? scaleWord0 : scaleWord1,
                                                       globalScale, sf2);
#pragma unroll
        for (uint32_t bb = 0; bb < 4; ++bb) {
          uint16_t const* pairs = reinterpret_cast<uint16_t const*>(&words[bb]);
          uint4 lo, hi;
          lo.x = mixed_detail::mul_a16x2<DTypeKV>(mixed_detail::e4m3x2_to_a16<DTypeKV>(pairs[0]), sf2[bb]);
          lo.y = mixed_detail::mul_a16x2<DTypeKV>(mixed_detail::e4m3x2_to_a16<DTypeKV>(pairs[1]), sf2[bb]);
          lo.z = mixed_detail::mul_a16x2<DTypeKV>(mixed_detail::e4m3x2_to_a16<DTypeKV>(pairs[2]), sf2[bb]);
          lo.w = mixed_detail::mul_a16x2<DTypeKV>(mixed_detail::e4m3x2_to_a16<DTypeKV>(pairs[3]), sf2[bb]);
          hi.x = mixed_detail::mul_a16x2<DTypeKV>(mixed_detail::e4m3x2_to_a16<DTypeKV>(pairs[4]), sf2[bb]);
          hi.y = mixed_detail::mul_a16x2<DTypeKV>(mixed_detail::e4m3x2_to_a16<DTypeKV>(pairs[5]), sf2[bb]);
          hi.z = mixed_detail::mul_a16x2<DTypeKV>(mixed_detail::e4m3x2_to_a16<DTypeKV>(pairs[6]), sf2[bb]);
          hi.w = mixed_detail::mul_a16x2<DTypeKV>(mixed_detail::e4m3x2_to_a16<DTypeKV>(pairs[7]), sf2[bb]);
          *chunk_ptr(d, 2 * bb) = lo;
          *chunk_ptr(d, 2 * bb + 1) = hi;
        }
      }
    } else {
      // 64 B rows, 64B swizzle: chunk ^= (r / 2) % 4.  Chunk c holds blocks 2c, 2c+1.
      uint8_t const* row = packed_page + r * FP4_ROW_BYTES;
      uint4 words[4];
#pragma unroll
      for (uint32_t c = 0; c < 4; ++c) {
        words[c] = *reinterpret_cast<uint4 const*>(row + ((c ^ ((r / 2) % 4)) * 16));
      }
#pragma unroll
      for (uint32_t d = 0; d < D_BLOCKS; ++d) {
        uint32_t sf2[4];
        mixed_detail::e4m3x4_scales_to_a16x2<DTypeKV>(d == 0 ? scaleWord0 : scaleWord1,
                                                       globalScale, sf2);
#pragma unroll
        for (uint32_t bb = 0; bb < 4; ++bb) {
          uint32_t const b = d * 4 + bb;
          uint32_t const* w = reinterpret_cast<uint32_t const*>(&words[b / 2]) + (b % 2) * 2;
          uint32_t lo4[4], hi4[4];
          mixed_detail::e2m1x8_to_a16<DTypeKV>(w[0], lo4);
          mixed_detail::e2m1x8_to_a16<DTypeKV>(w[1], hi4);
          uint4 lo, hi;
          lo.x = mixed_detail::mul_a16x2<DTypeKV>(lo4[0], sf2[bb]);
          lo.y = mixed_detail::mul_a16x2<DTypeKV>(lo4[1], sf2[bb]);
          lo.z = mixed_detail::mul_a16x2<DTypeKV>(lo4[2], sf2[bb]);
          lo.w = mixed_detail::mul_a16x2<DTypeKV>(lo4[3], sf2[bb]);
          hi.x = mixed_detail::mul_a16x2<DTypeKV>(hi4[0], sf2[bb]);
          hi.y = mixed_detail::mul_a16x2<DTypeKV>(hi4[1], sf2[bb]);
          hi.z = mixed_detail::mul_a16x2<DTypeKV>(hi4[2], sf2[bb]);
          hi.w = mixed_detail::mul_a16x2<DTypeKV>(hi4[3], sf2[bb]);
          *chunk_ptr(d, 2 * bb) = lo;
          *chunk_ptr(d, 2 * bb + 1) = hi;
        }
      }
    }
  }

  // ------------------------------------------------------------------------
  // One operand of one tile within a pair.  Two completion modes (C4):
  //  * tile without compressed pages and without a tail: thread 0 posts the
  //    TMA bytes as a transaction on the *pipeline's* full barrier; the stage
  //    completes when the bytes land and the 128 producer arrivals are in.
  //    The producer never waits for these loads.
  //  * otherwise: the bytes are posted on the private per-stage barrier, the
  //    group waits for it (and for the scale copies), expands, fences, and
  //    then arrives on the pipeline barrier.
  struct Operand {
    MainloopPipeline* pipeline;
    PipelineState* state;
    cutlass::arch::ClusterTransactionBarrier* tma_bars;
    uint8_t (*scales)[CTA_KV][BLOCKS_PER_HEAD];
    uint32_t* phase_bits;  // bit s = parity to wait for on tma_bars[s] (C2: no register array)
    bool isK;
    int stage;
    bool needs_expand;
  };

  template <typename STensor>
  CUTLASS_DEVICE void issue_operand(Params const& p, Operand& op, STensor& sX, IdType const* pages,
                                    uint8_t const* formats, uint32_t valid_tokens, int kv_head_idx,
                                    int thread_idx) const {
    op.stage = op.state->index();
    op.needs_expand = tile_has_compressed(formats) || valid_tokens < uint32_t(CTA_KV);
    if (thread_idx == 0) {
      uint32_t const tx = tile_tx_bytes(formats);
      if (op.needs_expand) {
        op.tma_bars[op.stage].arrive_and_expect_tx(tx);
        issue_tile_tma(p, pages, formats, sX, op.stage, op.isK, kv_head_idx,
                       reinterpret_cast<uint64_t*>(&op.tma_bars[op.stage]));
      } else {
        auto* full = reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType*>(
            op.pipeline->producer_get_barrier(*op.state));
        cutlass::arch::ClusterTransactionBarrier::expect_transaction(full, tx);
        issue_tile_tma(p, pages, formats, sX, op.stage, op.isK, kv_head_idx,
                       reinterpret_cast<uint64_t*>(full));
      }
    }
    if (op.needs_expand) {
      issue_tile_scales(p, pages, formats, op.isK, kv_head_idx, op.scales[op.stage], thread_idx);
    }
  }

  template <typename STensor>
  CUTLASS_DEVICE void finish_operand(Params const& p, Operand& op, STensor& sX,
                                     uint8_t const* formats, uint32_t valid_tokens,
                                     int thread_idx) const {
    if (op.needs_expand) {
      op.tma_bars[op.stage].wait((*op.phase_bits >> op.stage) & 1u);
      *op.phase_bits ^= 1u << op.stage;
      expand_token(p, formats, valid_tokens, sX, op.stage, op.isK, op.scales[op.stage],
                   thread_idx);
      cutlass::arch::fence_view_async_shared();  // D5
    }
    op.pipeline->producer_commit(*op.state);
    ++(*op.state);
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

    static_assert(NUM_STAGES <= 32);
    uint32_t phase_k = shared_storage.mixed_phase_k;
    uint32_t phase_v = shared_storage.mixed_phase_v;
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
    Operand K{&pipeline_k, &smem_pipe_write_k, shared_storage.mixed_tma_bar_k,
              shared_storage.mixed_scales_k, &phase_k, true, 0, false};
    Operand V{&pipeline_v, &smem_pipe_write_v, shared_storage.mixed_tma_bar_v,
              shared_storage.mixed_scales_v, &phase_v, false, 0, false};

    // Metadata for the first two tiles processed (t and t-1), C1.
    gather(kv_tile_idx);
    gather(kv_tile_idx - 1);

    // --- K(last) alone (the consumer starts on it before the first V) ---
    {
      pipeline_k.producer_acquire(smem_pipe_write_k);
      group_barrier();  // metadata visible; previous conversions of this stage done
      int const t = kv_tile_idx;
      issue_operand(mainloop_params, K, sK, meta_pages[slot(t)], meta_formats[slot(t)],
                    meta_valid[slot(t)], kv_head_idx, thread_idx);
      if (K.needs_expand) {
        cutlass::arch::cp_async_fence();
        cutlass::arch::cp_async_wait<0>();
        group_barrier();  // every thread's scale copies landed
      }
      finish_operand(mainloop_params, K, sK, meta_formats[slot(t)], meta_valid[slot(t)],
                     thread_idx);
    }

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

    // --- pairs (K(t-1), V(t)) in the consumer's order, then V(0) ---
    auto produce_pair = [&](int tK, int tV) {
      // tK may be -1 (no K); tV always valid.
      if (tK >= 0) pipeline_k.producer_acquire(smem_pipe_write_k);
      pipeline_v.producer_acquire(smem_pipe_write_v);
      group_barrier();
      // One pair ahead (C1).  Slot (tK-1)%3 is not read by this pair (slots tK,
      // tV are distinct from it) and its previous readers - the last pair's V
      // consumers - are all past the barrier above.
      gather(tK - 1);
      if (tK >= 0) {
        issue_operand(mainloop_params, K, sK, meta_pages[slot(tK)], meta_formats[slot(tK)],
                      meta_valid[slot(tK)], kv_head_idx, thread_idx);
      }
      issue_operand(mainloop_params, V, sV, meta_pages[slot(tV)], meta_formats[slot(tV)],
                    meta_valid[slot(tV)], kv_head_idx, thread_idx);
      bool const any_expand = (tK >= 0 && K.needs_expand) || V.needs_expand;
      if (any_expand) {
        cutlass::arch::cp_async_fence();
        cutlass::arch::cp_async_wait<0>();
        group_barrier();
      }
      if (tK >= 0) {
        finish_operand(mainloop_params, K, sK, meta_formats[slot(tK)], meta_valid[slot(tK)],
                       thread_idx);
      }
      finish_operand(mainloop_params, V, sV, meta_formats[slot(tV)], meta_valid[slot(tV)],
                     thread_idx);
    };

    if (kv_tile_idx == swa_begin_kv_tile_idx) {
      produce_pair(-1, kv_tile_idx);
    } else {
#pragma unroll 1
      for (; kv_tile_idx > swa_begin_kv_tile_idx; --kv_tile_idx) {
        produce_pair(kv_tile_idx - 1, kv_tile_idx);
      }
      scheduler.prefetch_next_work(scheduler_params, work_tile_info);
      produce_pair(-1, swa_begin_kv_tile_idx);
    }
    // All threads hold identical phase words (derived from smem-resident data).
    // The trailing group barrier orders this write-back and the next work
    // item's initial metadata gathers after every reader of this item.
    group_barrier();
    if (thread_idx == 0) {
      shared_storage.mixed_phase_k = phase_k;
      shared_storage.mixed_phase_v = phase_v;
    }
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
