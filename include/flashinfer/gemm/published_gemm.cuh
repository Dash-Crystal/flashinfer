/*
 * Copyright (c) 2026 by FlashInfer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "masked_gemm.cuh"

namespace flashinfer::published_gemm {

using namespace masked_gemm;

struct OutputTileMapping : cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle {
  CUTLASS_DEVICE static cutlass::gemm::GemmCoord get_tile_offset(int) {
    return GemmHorizontalThreadblockSwizzle::get_tile_offset({});
  }
};

struct WorkMapping : ReadySwizzle {
  using Base = ReadySwizzle;
  static constexpr auto kReductionStrategy = Base::kAtomic;

  WorkMapping() = default;

  WorkMapping(cutlass::gemm::GemmUniversalMode mode, cutlass::gemm::GemmCoord problem,
              cutlass::gemm::GemmCoord tile, int split, int occupancy, int sms, int available_sms,
              size_t a_bytes, size_t b_bytes, size_t c_bytes, int fragments)
      : Base(mode, problem, tile, split, occupancy, sms, available_sms, a_bytes, b_bytes, c_bytes,
             fragments) {
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

template <typename Element, bool StreamK = true>
using Kernel = std::conditional_t<
    StreamK,
    cutlass::gemm::kernel::GemmUniversalStreamk<MaskedMma<typename LocalGemm<Element>::Mma>,
                                                PublishedEpilogue<Element>, WorkMapping>,
    cutlass::gemm::kernel::Gemm<MaskedMma<typename LocalGemm<Element>::Mma>,
                                PublishedEpilogue<Element>, OutputTileMapping, false>>;

template <typename Element, bool StreamK = true>
constexpr size_t SharedBytes() {
  return sizeof(typename Kernel<Element, StreamK>::SharedStorage) + (StreamK ? 128 : 0);
}

template <typename Element>
constexpr size_t PartialBytes(int slots) {
  return PartialWorkspaceBytes<Kernel<Element>>(slots);
}

template <typename Element>
constexpr size_t WorkspaceBytes(int slots) {
  return PartialBytes<Element>(slots) + (slots * sizeof(int) + 127) / 128 * 128;
}

template <typename Element, bool StreamK>
__global__ __launch_bounds__(Kernel<Element, StreamK>::kThreadCount) void PublishedGemm(
    typename Kernel<Element, StreamK>::Params params, int logical_blocks) {
  extern __shared__ char storage[];
  if constexpr (StreamK) {
    RunResidentWork<Kernel<Element, StreamK>>(params, storage, logical_blocks);
  } else {
    Kernel<Element, StreamK>{}(
        params, *reinterpret_cast<typename Kernel<Element, StreamK>::SharedStorage*>(storage));
  }
}

template <typename Element, bool StreamK>
cudaError_t Prepare(int* resources, int* occupancy) {
  auto kernel = PublishedGemm<Element, StreamK>;
  constexpr size_t shared = SharedBytes<Element, StreamK>();
  auto status = PrepareKernel(kernel, shared);
  if (status != cudaSuccess) return status;
  cudaFuncAttributes attributes;
  status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) return status;
  resources[0] = attributes.numRegs;
  resources[1] = attributes.sharedSizeBytes + shared;
  resources[2] = Kernel<Element, StreamK>::kThreadCount;
  resources[3] = Tile::kM;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      occupancy, kernel, Kernel<Element, StreamK>::kThreadCount, shared);
}

template <typename Element, bool StreamK>
cudaError_t Run(Element* a, Element* b, Element* out, int m, int n, int k, int lda, int ldb,
                int ldd, uint8_t const* padding, int row_offset, Element* peer, int* publication,
                int* peer_publication, int groups, void* workspace, int sms, int occupancy,
                cudaStream_t stream) {
  using Gemm = Kernel<Element, StreamK>;
  auto params = [&] {
    if constexpr (StreamK) {
      typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k}, 1,
                                    {1.0f, 0.0f}, a, b, out, out, 0, 0, 0, 0, int64_t(lda),
                                    int64_t(ldb), int64_t(ldd), int64_t(ldd), sms);
      return typename Gemm::Params(args, sms, occupancy);
    } else {
      cutlass::gemm::GemmCoord problem(m, n, k);
      auto grid = OutputTileMapping::get_tiled_shape(problem, {Tile::kM, Tile::kN, Tile::kK}, 1);
      return typename Gemm::Params(problem, grid, {a, RowMajor(lda)}, {b, ColumnMajor(ldb)},
                                   {out, RowMajor(ldd)}, {out, RowMajor(ldd)}, {1.0f, 0.0f});
    }
  }();
  params.params_A.padding = padding;
  params.params_A.row_offset = row_offset;
  params.params_D.peer = peer;
  params.params_D.publication = publication;
  params.params_D.peer_publication = peer_publication;
  params.params_D.groups = groups;
  int logical_blocks = 0;
  dim3 grid;
  if constexpr (StreamK) {
    params.partials_workspace = workspace;
    params.barrier_workspace =
        static_cast<char*>(workspace) + PartialBytes<Element>(sms * occupancy);
    logical_blocks = params.block_mapping.get_num_blocks();
    grid = dim3(min(sms * occupancy, logical_blocks));
  } else {
    grid = OutputTileMapping::get_grid_shape(params.grid_tiled_shape);
  }
  PublishedGemm<Element, StreamK>
      <<<grid, Gemm::kThreadCount, SharedBytes<Element, StreamK>(), stream>>>(params,
                                                                              logical_blocks);
  return cudaGetLastError();
}

}  // namespace flashinfer::published_gemm
