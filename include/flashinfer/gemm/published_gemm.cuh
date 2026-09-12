/*
 * Copyright (c) 2026 by FlashInfer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <utility>

#include "masked_gemm.cuh"

namespace flashinfer::published_gemm {

using namespace masked_gemm;

struct WorkMapping : ReadySwizzle {
  using Base = ReadySwizzle;
  static constexpr auto kReductionStrategy = Base::kAtomic;

  template <class... Args>
  CUTLASS_HOST_DEVICE WorkMapping(Args&&... args) : Base(std::forward<Args>(args)...) {
    // The finishing K peer owns the complete tile and its publication.
    reduction_blocks = 0;
  }
};

template <typename Base, int Rows, int Threads>
struct MaskedIterator : Base {
  using Element = typename Base::Element;
  using TensorCoord = typename Base::TensorCoord;
  struct Params : Base::Params {
    using Base::Params::Params;
    uint8_t const* padding = nullptr;
    int row_offset = 0;
  };
  bool live;

  CUTLASS_DEVICE MaskedIterator(Params const& params, Element* pointer, TensorCoord extent,
                                int thread, TensorCoord offset)
      : Base(params, pointer, extent, thread, offset) {
    bool visible = false;
    for (int row = offset.row() + thread; row < min(offset.row() + Rows, extent.row());
         row += Threads)
      visible |= !params.padding[params.row_offset + row];
    live = __syncthreads_or(visible);
  }
};

template <typename Base>
struct MaskedMma : Base {
  using Base::Base;
  using IteratorA =
      MaskedIterator<typename Base::IteratorA, Base::Shape::kM, Base::WarpCount::kCount * 32>;

  CUTLASS_DEVICE void operator()(int iterations, typename Base::FragmentC& accum, IteratorA a,
                                 typename Base::IteratorB b,
                                 typename Base::FragmentC const& source) {
    if (a.live) Base::operator()(iterations, accum, a, b, source);
  }
};

template <typename Base>
struct OutputIterator : Base {
  using Element = typename Base::Element;
  using TensorCoord = typename Base::TensorCoord;
  struct Params : Base::Params {
    using Base::Params::Params;
    int* publication = nullptr;
    int* peer_publication = nullptr;
    int groups = 0;
  };
  int* publication;
  int* peer_publication;
  int groups;
  int rows;
  int tiles_n;
  int tile_m;

  CUTLASS_DEVICE OutputIterator(Params const& params, Element* pointer, TensorCoord extent,
                                int thread, TensorCoord offset)
      : Base(params, pointer, extent, thread, offset),
        publication(params.publication),
        peer_publication(params.peer_publication),
        groups(params.groups),
        rows(extent.row()),
        tiles_n((extent.column() + Tile::kN - 1) / Tile::kN),
        tile_m(offset.row() / Tile::kM) {}
};

template <typename Element>
using Store =
    OutputIterator<PeerStoreIterator<typename LocalGemm<Element>::Epilogue::OutputTileIterator>>;

template <typename Element>
using StoreEpilogue = RebindEpilogue<typename LocalGemm<Element>::Epilogue, Store<Element>>;

template <typename Element>
struct PublishedEpilogue : StoreEpilogue<Element> {
  using Base = StoreEpilogue<Element>;
  using Base::Base;

  template <typename Accumulator>
  CUTLASS_DEVICE void operator()(typename Base::OutputOp const& operation,
                                 Store<Element> destination, Accumulator const& accum,
                                 Store<Element> source) {
    Base::operator()(operation, destination, accum, source);
    PublishRows<Tile::kM>(destination.publication, destination.peer_publication, destination.groups,
                          destination.tile_m, destination.tiles_n, destination.rows);
  }
};

template <typename Element>
using Kernel =
    cutlass::gemm::kernel::GemmUniversalStreamk<MaskedMma<typename LocalGemm<Element>::Mma>,
                                                PublishedEpilogue<Element>, WorkMapping>;

template <typename Element>
constexpr size_t SharedBytes() {
  return PersistentSharedBytes<Kernel<Element>>();
}

template <typename Element>
constexpr size_t PartialBytes(int slots) {
  return PartialWorkspaceBytes<Kernel<Element>>(slots);
}

template <typename Element>
constexpr size_t WorkspaceBytes(int slots) {
  return PartialBytes<Element>(slots) + (slots * sizeof(int) + 127) / 128 * 128;
}

template <typename Element>
__global__ __launch_bounds__(Kernel<Element>::kThreadCount) void PublishedGemm(
    typename Kernel<Element>::Params params, int logical_blocks) {
  extern __shared__ char storage[];
  RunResidentWork<Kernel<Element>>(params, storage, logical_blocks);
}

template <typename Element>
cudaError_t Prepare(int* resources, int* occupancy) {
  auto status = PrepareKernel(PublishedGemm<Element>, SharedBytes<Element>());
  if (status != cudaSuccess) return status;
  cudaFuncAttributes attributes;
  status = cudaFuncGetAttributes(&attributes, PublishedGemm<Element>);
  if (status != cudaSuccess) return status;
  resources[0] = attributes.numRegs;
  resources[1] = attributes.sharedSizeBytes + SharedBytes<Element>();
  resources[2] = Kernel<Element>::kThreadCount;
  resources[3] = Tile::kM;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      occupancy, PublishedGemm<Element>, Kernel<Element>::kThreadCount, SharedBytes<Element>());
}

template <typename Element>
cudaError_t Run(Element* a, Element* b, Element* out, int m, int n, int k, int lda, int ldb,
                int ldd, uint8_t const* padding, int row_offset, Element* peer, int* publication,
                int* peer_publication, int groups, void* workspace, int sms, int occupancy,
                cudaStream_t stream) {
  using Gemm = Kernel<Element>;
  typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k}, 1, {1.0f, 0.0f},
                                a, b, out, out, 0, 0, 0, 0, int64_t(lda), int64_t(ldb),
                                int64_t(ldd), int64_t(ldd), sms);
  typename Gemm::Params params(args, sms, occupancy);
  params.params_A.padding = padding;
  params.params_A.row_offset = row_offset;
  params.params_D.peer = peer;
  params.params_D.publication = publication;
  params.params_D.peer_publication = peer_publication;
  params.params_D.groups = groups;
  params.partials_workspace = workspace;
  params.barrier_workspace = static_cast<char*>(workspace) + PartialBytes<Element>(sms * occupancy);
  int const logical_blocks = params.block_mapping.get_num_blocks();
  int const blocks = min(sms * occupancy, logical_blocks);
  PublishedGemm<Element>
      <<<blocks, Gemm::kThreadCount, SharedBytes<Element>(), stream>>>(params, logical_blocks);
  return cudaGetLastError();
}

}  // namespace flashinfer::published_gemm
