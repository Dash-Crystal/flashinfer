/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_KERNEL_TRAITS_CUH_
#define FLASHINFER_ATTENTION_HOPPER_KERNEL_TRAITS_CUH_

#include <cstddef>
#include <type_traits>

#include "../../cutlass_utils.cuh"
#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

namespace flashinfer {

using namespace cute;

template <typename MainloopPipeline, class DTypeQ, class DTypeKV, class DTypeOut, class IdType,
          int CTA_KV, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorageQKVO {
  cute::array_aligned<DTypeQ, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<DTypeKV, cute::cosize_v<SmemLayoutK>> smem_k;
  union {
    cute::array_aligned<DTypeKV, cute::cosize_v<SmemLayoutV>> smem_v;
    cute::array_aligned<DTypeOut, cute::cosize_v<SmemLayoutO>> smem_o;
  };
  struct {
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    cutlass::arch::ClusterBarrier barrier_O;
    typename MainloopPipeline::SharedStorage pipeline_k;
    typename MainloopPipeline::SharedStorage pipeline_v;
  };
};

// Shared storage for the mixed-page producer: one 8 B whole-head block-scale
// word per token per stage, and the page-metadata chunk table.
//
// Chunk table: metadata (page index, format tag, valid-token count) for 16
// consecutive KV tiles per buffer, two buffers.  The producer fills the next
// chunk's buffer while it issues the current chunk (one dependent global round
// trip per 16 tiles instead of one per pair), and a pair reads its two tiles'
// rows with four LDS.128.  One row is 32 B: 6 page indices, 6 tags, the valid
// count and a flags byte (bit 0: any compressed page; bit 1: row filled).
template <typename IdType, int CTA_KV>
struct alignas(16) MixedTileMeta {
  static constexpr int kPages = CTA_KV / 16;
  IdType pages[kPages];
  uint8_t tags[kPages];
  uint8_t valid;
  uint8_t flags;
};

template <typename MainloopPipeline, typename DTypeQ, typename DTypeKV, typename DTypeOut,
          typename IdType, int CTA_KV, int NUM_STAGES, typename SmemLayoutQ,
          typename SmemLayoutK, typename SmemLayoutV, typename SmemLayoutO>
struct SharedStorageQKVOMixed {
  cute::array_aligned<DTypeQ, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<DTypeKV, cute::cosize_v<SmemLayoutK>> smem_k;
  union {
    cute::array_aligned<DTypeKV, cute::cosize_v<SmemLayoutV>> smem_v;
    cute::array_aligned<DTypeOut, cute::cosize_v<SmemLayoutO>> smem_o;
  };
  struct {
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    cutlass::arch::ClusterBarrier barrier_O;
    typename MainloopPipeline::SharedStorage pipeline_k;
    typename MainloopPipeline::SharedStorage pipeline_v;
  };
  // Block scales: per stage, per page, a 512 B slot.  [24b] uses its first 128 B
  // as one 8 B word pair per token row (the row's eight E4M3 block scales,
  // copied by the row's lane 0 with one cp.async and read by the row's eight
  // lanes after that lane's cp.async.wait_group and one __syncwarp per pair - D3
  // as amended by A8); [23] used one 4 B word per producer thread (7.95
  // wavefronts per warp copy instruction, A7).  The slot size is kept at [23]'s
  // so that every module's shared-storage layout - and the a16 module's SASS,
  // the transport control - is unchanged; 384 B per page slot are unused.
  static constexpr uint32_t kMixedScaleStageBytes = (CTA_KV / 16) * 128 * 4;
  alignas(16) uint8_t mixed_scales_k[NUM_STAGES][kMixedScaleStageBytes];
  alignas(16) uint8_t mixed_scales_v[NUM_STAGES][kMixedScaleStageBytes];
  static constexpr uint32_t kMixedMetaChunkTiles = 16;
  static constexpr uint32_t kMixedMetaBuffers = 2;
  using TileMeta = MixedTileMeta<IdType, CTA_KV>;
  static_assert(sizeof(TileMeta) == 32 && sizeof(IdType) == 4 && CTA_KV == 96,
                "the chunk-table row is two LDS.128 for 6 pages of int32 indices");
  TileMeta mixed_meta[kMixedMetaBuffers][kMixedMetaChunkTiles];

  // D1: the stage is tiled from 8-row x 128 B SW128 atoms (a page's 16 rows of
  // one 64-element D-block are one contiguous 2 KB region) and 1024 B aligned,
  // which is what the copies, the in-place expansion and wgmma all assume.
  // One stage = (CTA_KV/8 row groups) x (D/64 D-blocks) atoms of 8 rows x 128 B.
  static_assert(cute::cosize_v<SmemLayoutK> * sizeof(DTypeKV) / NUM_STAGES ==
                    (CTA_KV / 8) * (cute::size<1>(SmemLayoutK{}.shape()) / 64) * 1024,
                "K stage is not tiled from 8-row x 128 B SW128 atoms");
  static_assert(cute::cosize_v<SmemLayoutV> * sizeof(DTypeKV) / NUM_STAGES ==
                    (CTA_KV / 8) * (cute::size<1>(SmemLayoutV{}.shape()) / 64) * 1024,
                "V stage is not tiled from 8-row x 128 B SW128 atoms");
  static_assert(CTA_KV % 16 == 0, "KV tile must be whole 16-token pages");
};

template <bool USE_TMA_LOAD_KV, int HEAD_DIM_QK_, int HEAD_DIM_VO_, int CTA_Q_, int CTA_KV_,
          int NUM_STAGES_, typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_,
          typename AttentionVariant_>
struct AttentionKernelTraits {
  using AttentionVariant = AttentionVariant_;

  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;
  using DTypeQKAccum = float;

  static constexpr bool kMixedTraits = false;
  static constexpr int CTA_Q = CTA_Q_;
  static_assert(CTA_Q % 64 == 0);
  static constexpr int CTA_KV = CTA_KV_;
  static constexpr int HEAD_DIM_QK = HEAD_DIM_QK_;
  static constexpr int HEAD_DIM_VO = HEAD_DIM_VO_;
  static_assert(HEAD_DIM_QK % 32 == 0);
  static_assert(HEAD_DIM_VO % 32 == 0);

  static constexpr int NUM_WARPS = ((CTA_Q / 64) + 1) * 4;
  static constexpr int NUM_THREADS = NUM_WARPS * cutlass::NumThreadsPerWarp;
  // NOTE(Zihao): the following constant should only be used when TMA is enabled,
  // where only one warp inside a warp group is used for TMA.
  static constexpr int NUM_PRODUCER_THREADS =
      USE_TMA_LOAD_KV ? cutlass::NumThreadsPerWarp : 4 * cutlass::NumThreadsPerWarp;

  using TileShape_QKD = Shape<Int<CTA_Q>, Int<CTA_KV>, Int<HEAD_DIM_QK>>;
  using TileShape_PDV = Shape<Int<CTA_Q>, Int<HEAD_DIM_VO>, Int<CTA_KV>>;

  static constexpr int NUM_STAGES = NUM_STAGES_;

  using AtomLayoutQKD = Layout<Shape<Int<CTA_Q / 64>, _1, _1>>;
  using TiledMmaQK = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<DTypeQ, DTypeKV, DTypeQKAccum, TileShape_QKD>(), AtomLayoutQKD{}));
  using TiledMmaPV = decltype(cute::make_tiled_mma(
      cute::GMMA::rs_op_selector<DTypeKV, DTypeKV, /*ElementAccum=*/float, TileShape_PDV,
                                 GMMA::Major::K, GMMA::Major::MN>(),
      AtomLayoutQKD{}));

  static constexpr int NUM_MMA_THREADS = size(TiledMmaQK{});

  using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeQ, decltype(cute::get<0>(TileShape_QKD{})),
                                   decltype(cute::get<2>(TileShape_QKD{}))>());
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_QKD{})));

  using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeKV, decltype(cute::get<1>(TileShape_QKD{})),
                                   decltype(cute::get<2>(TileShape_QKD{}))>());
  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtomK{},
      make_shape(shape<1>(TileShape_QKD{}), shape<2>(TileShape_QKD{}), Int<NUM_STAGES>{})));

  using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeKV, decltype(cute::get<2>(TileShape_PDV{})),
                                   decltype(cute::get<1>(TileShape_PDV{}))>());
  using SmemLayoutV = decltype(tile_to_shape(
      SmemLayoutAtomV{},
      make_shape(get<2>(TileShape_PDV{}), get<1>(TileShape_PDV{}), Int<NUM_STAGES>{})));

  // Note this is the transpose in terms of the view, not in terms of memory.
  using SmemLayoutVt = decltype(composition(
      SmemLayoutV{}, make_ordered_layout(make_shape(get<1>(TileShape_PDV{}),
                                                    get<2>(TileShape_PDV{}), Int<NUM_STAGES>{}),
                                         Step<_2, _1, _3>{})));

  using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeO, decltype(cute::get<0>(TileShape_PDV{})),
                                   decltype(cute::get<1>(TileShape_PDV{}))>());
  using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 1>(TileShape_PDV{})));
  using MainloopPipeline =
      std::conditional_t<USE_TMA_LOAD_KV, typename cutlass::PipelineTmaAsync<NUM_STAGES>,
                         typename cutlass::PipelineAsync<NUM_STAGES>>;
  using PipelineState = typename cutlass::PipelineState<NUM_STAGES>;

  using SharedStorage = SharedStorageQKVO<MainloopPipeline, DTypeQ, DTypeKV, DTypeO, IdType, CTA_KV,
                                          SmemLayoutQ, SmemLayoutK, SmemLayoutV, SmemLayoutO>;
};

// The module's compile-time page format carried by the attention variant
// (variants.cuh MixedPageAttention<N>): -1 dynamic, 0 A16, 1 E4M3, 2 E2M1.
template <typename Variant, typename = void>
struct mixed_variant_static_format : std::integral_constant<int, -1> {};
template <typename Variant>
struct mixed_variant_static_format<Variant, std::void_t<decltype(Variant::kMixedStaticFormat)>>
    : std::integral_constant<int, Variant::kMixedStaticFormat> {};

// Same traits with the mixed-page shared storage.  The producer issues cp.async
// copies itself, so USE_TMA_LOAD_KV is false (PipelineAsync).
//
// [24b] Producer warp groups.  The compressed (E4M3, E2M1) and dynamic modules
// run TWO producer warp groups (16 warps, 512 threads): thread t = 128 h + u
// applies the [23] ownership formulas to u on the tile pages i = h + 2 j.  The
// static-A16 module has no expansion and keeps ONE producer warp group (12
// warps): its SASS is the transport control and stays byte-identical.
// Register pool (C3): __launch_bounds__(NUM_THREADS, 1) gives 65536 / 512 = 128
// registers per thread at launch; setmaxnreg then moves them so that
// NUM_PRODUCER_THREADS x PRODUCER_REGS + NUM_MMA_THREADS x CONSUMER_REGS <= 65536:
// 256 x 72 + 256 x 184 = 65536 exactly (12 warps: 128 x 72 + 256 x 216 = 64512).
// The consumer's fit at 184 is proven by ptxas -v (no C7507, STACK 0), not here.
template <int HEAD_DIM_QK_, int HEAD_DIM_VO_, int CTA_Q_, int CTA_KV_, int NUM_STAGES_,
          typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_,
          typename AttentionVariant_>
struct MixedAttentionKernelTraits
    : AttentionKernelTraits</*USE_TMA_LOAD_KV=*/false, HEAD_DIM_QK_, HEAD_DIM_VO_, CTA_Q_, CTA_KV_,
                            NUM_STAGES_, DTypeQ_, DTypeKV_, DTypeO_, IdType_, AttentionVariant_> {
  static constexpr bool kMixedTraits = true;
  using BaseTraits =
      AttentionKernelTraits<false, HEAD_DIM_QK_, HEAD_DIM_VO_, CTA_Q_, CTA_KV_, NUM_STAGES_,
                            DTypeQ_, DTypeKV_, DTypeO_, IdType_, AttentionVariant_>;
  static constexpr int kMixedStaticFormat = mixed_variant_static_format<AttentionVariant_>::value;
  static constexpr int NUM_PRODUCER_WGS = kMixedStaticFormat == 0 ? 1 : 2;
  static constexpr int NUM_WARPS = ((CTA_Q_ / 64) + NUM_PRODUCER_WGS) * 4;
  static constexpr int NUM_THREADS = NUM_WARPS * cutlass::NumThreadsPerWarp;
  static constexpr int NUM_PRODUCER_THREADS = NUM_PRODUCER_WGS * cutlass::NumThreadsPerWarpGroup;
  static constexpr int PRODUCER_REGS = 72;
  static constexpr int CONSUMER_REGS = NUM_PRODUCER_WGS == 2 ? 184 : 216;
  static_assert(NUM_PRODUCER_WGS == 1 || NUM_PRODUCER_WGS == 2, "one or two producer warp groups");
  static_assert(PRODUCER_REGS % 8 == 0 && CONSUMER_REGS % 8 == 0, "setmaxnreg takes multiples of 8");
  static_assert(NUM_PRODUCER_THREADS * PRODUCER_REGS + BaseTraits::NUM_MMA_THREADS * CONSUMER_REGS <=
                    65536,
                "register pool: producer + consumer allocations must fit the SM (C3)");
  using SharedStorage =
      SharedStorageQKVOMixed<typename BaseTraits::MainloopPipeline, DTypeQ_, DTypeKV_, DTypeO_,
                             IdType_, CTA_KV_, NUM_STAGES_, typename BaseTraits::SmemLayoutQ,
                             typename BaseTraits::SmemLayoutK, typename BaseTraits::SmemLayoutV,
                             typename BaseTraits::SmemLayoutO>;
  static_assert(offsetof(SharedStorage, smem_k) % 1024 == 0 &&
                    offsetof(SharedStorage, smem_v) % 1024 == 0,
                "K/V stages must be 1024 B aligned for the SW128 swizzle (D1)");
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_KERNEL_TRAITS_CUH_
