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
  bool row_major = false;

  WorkMapping() = default;

  WorkMapping(cutlass::gemm::GemmUniversalMode mode, cutlass::gemm::GemmCoord problem,
              cutlass::gemm::GemmCoord tile, int split, int occupancy, int sms, int available_sms,
              size_t a_bytes, size_t b_bytes, size_t c_bytes, int fragments)
      : Base(mode, problem, tile, split, occupancy, sms, available_sms, a_bytes, b_bytes, c_bytes,
             fragments),
        row_major(available_sms == 1) {
    // The finishing K peer owns the complete tile and its publication.
    reduction_blocks = 0;
    if (row_major) {
      // Explicit persistent DP publishes complete row stripes early. Override
      // CUTLASS's tall/wide/cohort heuristics without changing compute tiles.
      cohort_raster = false;
      dp_blocks =
          ((problem.m() + tile.m() - 1) / tile.m()) * ((problem.n() + tile.n() - 1) / tile.n());
    }
  }

  CUTLASS_DEVICE int get_block_idx() const {
    if (row_major) {
      extern __shared__ char storage[];
      return int(blockIdx.x) + *reinterpret_cast<int*>(storage);
    }
    return Base::get_block_idx();
  }

  CUTLASS_DEVICE cutlass::gemm::GemmCoord get_tile_offset(int tile_idx) const {
    return row_major ? Base::get_tile_offset_row_major(tile_idx) : Base::get_tile_offset(tile_idx);
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
                                int thread, TensorCoord offset, int const* indices = nullptr)
      : Base(params, pointer, extent, thread, offset, indices) {
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
    tp_region::DeviceRegion const* native_region = nullptr;
    PushPublication const* publication_epoch = nullptr;
  };
  int* publication;
  int* peer_publication;
  int groups;
  int rows;
  int columns;
  tp_region::DeviceRegion const* native_region;
  PushPublication const* publication_epoch;
  int tiles_n;
  int tile_m;

  CUTLASS_DEVICE OutputIterator(Params const& params, Element* pointer, TensorCoord extent,
                                int thread, TensorCoord offset, int const* indices = nullptr)
      : Base(params, pointer, extent, thread, offset, indices),
        publication(params.publication),
        peer_publication(params.peer_publication),
        groups(params.groups),
        rows(extent.row()),
        columns(extent.column()),
        native_region(params.native_region),
        publication_epoch(params.publication_epoch),
        tiles_n((extent.column() + Tile::kN - 1) / Tile::kN),
        tile_m(offset.row() / Tile::kM) {}
};

template <typename Element, bool LocalOnly = false>
using Store = OutputIterator<std::conditional_t<
    LocalOnly, typename LocalGemm<Element>::Epilogue::OutputTileIterator,
    PeerStoreIterator<typename LocalGemm<Element>::Epilogue::OutputTileIterator>>>;

template <typename Element, bool LocalOnly = false>
using StoreEpilogue =
    RebindEpilogue<typename LocalGemm<Element>::Epilogue, Store<Element, LocalOnly>>;

template <typename Element, bool LocalOnly = false>
struct PublishedEpilogue : StoreEpilogue<Element, LocalOnly> {
  using Base = StoreEpilogue<Element, LocalOnly>;
  using Base::Base;

  template <typename Accumulator>
  CUTLASS_DEVICE void operator()(typename Base::OutputOp const& operation,
                                 Store<Element, LocalOnly> destination, Accumulator const& accum,
                                 Store<Element, LocalOnly> source) {
    Base::operator()(operation, destination, accum, source);
    PublishRows<Tile::kM, LocalOnly>(destination.publication, destination.peer_publication,
                                     destination.groups, destination.tile_m, destination.tiles_n,
                                     destination.rows, destination.native_region,
                                     destination.columns, destination.publication_epoch);
  }
};

template <typename Element, bool StreamK = true, bool LocalOnly = false>
using Kernel = std::conditional_t<
    StreamK,
    cutlass::gemm::kernel::GemmUniversalStreamk<MaskedMma<typename LocalGemm<Element>::Mma>,
                                                PublishedEpilogue<Element, LocalOnly>, WorkMapping>,
    cutlass::gemm::kernel::Gemm<MaskedMma<typename LocalGemm<Element>::Mma>,
                                PublishedEpilogue<Element, LocalOnly>, OutputTileMapping, false>>;

template <typename Element, bool StreamK = true, bool LocalOnly = false>
constexpr size_t SharedBytes() {
  return sizeof(typename Kernel<Element, StreamK, LocalOnly>::SharedStorage) + (StreamK ? 128 : 0);
}

template <typename Element, bool LocalOnly = false>
constexpr size_t PartialBytes(int slots) {
  return PartialWorkspaceBytes<Kernel<Element, true, LocalOnly>>(slots);
}

template <typename Element, bool LocalOnly = false>
constexpr size_t WorkspaceBytes(int slots) {
  return PartialBytes<Element, LocalOnly>(slots) + (slots * sizeof(int) + 127) / 128 * 128;
}

struct ResidentTasks {
  int* state = nullptr;  // next full tile, finished CTA count, retirement latch
  int* readiness = nullptr;
  int group_rows = 128;
  int rows = 0;
  int retire_rows = 0;
  int retire_workers = 0;
};

CUTLASS_DEVICE bool RetireForRows(ResidentTasks const& tasks) {
  if (int(blockIdx.x) >= tasks.retire_workers) return false;
  cuda::atomic_ref<int, cuda::thread_scope_device> latch(tasks.state[2]);
  if (latch.load(cuda::memory_order_acquire)) return true;
  for (int g = 0; g < (tasks.retire_rows + tasks.group_rows - 1) / tasks.group_rows; ++g) {
    cuda::atomic_ref<int, cuda::thread_scope_device> ready(tasks.readiness[g]);
    if (ready.load(cuda::memory_order_acquire) <
        min(tasks.group_rows, tasks.rows - g * tasks.group_rows))
      return false;
  }
  latch.store(1, cuda::memory_order_release);
  return true;
}

template <typename Gemm>
CUTLASS_DEVICE void RunClaimedWork(typename Gemm::Params const& params, char* storage,
                                   int logical_blocks, ResidentTasks const& tasks) {
  auto& shared = *reinterpret_cast<typename Gemm::SharedStorage*>(storage + 128);
  auto* control = reinterpret_cast<int*>(storage);
  for (;;) {
    if (threadIdx.x == 0) {
      int tile = logical_blocks;
      if (!RetireForRows(tasks)) {
        cuda::atomic_ref<int, cuda::thread_scope_device> next(tasks.state[0]);
        tile = next.fetch_add(1, cuda::memory_order_relaxed);
      }
      control[0] = tile - int(blockIdx.x);
      control[1] = tile < logical_blocks;
    }
    __syncthreads();
    if (!control[1]) break;
    Gemm::invoke(params, shared);
    // Finish all epilogue stores/publication before surrendering this worker.
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    cuda::atomic_ref<int, cuda::thread_scope_device> finished(tasks.state[1]);
    if (finished.fetch_add(1, cuda::memory_order_acq_rel) == int(gridDim.x) - 1) {
      tasks.state[0] = 0;
      tasks.state[2] = 0;
      finished.store(0, cuda::memory_order_relaxed);
    }
  }
}

template <typename Element, bool StreamK, bool LocalOnly = false>
__global__ __launch_bounds__(Kernel<Element, StreamK, LocalOnly>::kThreadCount) void PublishedGemm(
    typename Kernel<Element, StreamK, LocalOnly>::Params params, int logical_blocks,
    ResidentTasks tasks) {
  extern __shared__ char storage[];
  if constexpr (StreamK) {
    if (tasks.state) {
      RunClaimedWork<Kernel<Element, StreamK, LocalOnly>>(params, storage, logical_blocks, tasks);
    } else {
      RunResidentWork<Kernel<Element, StreamK, LocalOnly>>(params, storage, logical_blocks);
    }
  } else {
    Kernel<Element, StreamK, LocalOnly>{}(
        params,
        *reinterpret_cast<typename Kernel<Element, StreamK, LocalOnly>::SharedStorage*>(storage));
  }
}

template <typename Element, bool StreamK, bool LocalOnly = false>
cudaError_t Prepare(int* resources, int* occupancy) {
  auto kernel = PublishedGemm<Element, StreamK, LocalOnly>;
  constexpr size_t shared = SharedBytes<Element, StreamK, LocalOnly>();
  auto status = PrepareKernel(kernel, shared);
  if (status != cudaSuccess) return status;
  cudaFuncAttributes attributes;
  status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) return status;
  resources[0] = attributes.numRegs;
  resources[1] = attributes.sharedSizeBytes + shared;
  resources[2] = Kernel<Element, StreamK, LocalOnly>::kThreadCount;
  resources[3] = Tile::kM;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      occupancy, kernel, Kernel<Element, StreamK, LocalOnly>::kThreadCount, shared);
}

template <typename Element, bool StreamK, bool LocalOnly = false>
cudaError_t Run(Element* a, Element* b, Element* out, int m, int n, int k, int lda, int ldb,
                int ldd, uint8_t const* padding, int row_offset, Element* peer, int* publication,
                int* peer_publication, int groups, void* workspace, int sms, int occupancy,
                cudaStream_t stream, bool persistent_dp = false,
                tp_region::DeviceRegion const* native_region = nullptr, ResidentTasks tasks = {},
                PushPublication const* publication_epoch = nullptr) {
  if (tasks.state && (!persistent_dp || !StreamK)) return cudaErrorInvalidValue;
  tasks.rows = m;
  if (tasks.state && tasks.retire_workers &&
      (!tasks.readiness || tasks.group_rows <= 0 || tasks.retire_rows <= 0 ||
       tasks.retire_rows > m))
    return cudaErrorInvalidValue;
  using Gemm = Kernel<Element, StreamK, LocalOnly>;
  auto params = [&] {
    if constexpr (StreamK) {
      typename Gemm::Arguments args(
          cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k}, 1, {1.0f, 0.0f}, a, b, out, out, 0, 0,
          0, 0, int64_t(lda), int64_t(ldb), int64_t(ldd), int64_t(ldd), persistent_dp ? 1 : sms);
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
  if constexpr (!LocalOnly) params.params_D.peer = peer;
  params.params_D.publication = publication;
  params.params_D.peer_publication = peer_publication;
  params.params_D.groups = groups;
  params.params_D.native_region = native_region;
  params.params_D.publication_epoch = publication_epoch;
  int logical_blocks = 0;
  dim3 grid;
  if constexpr (StreamK) {
    params.partials_workspace = workspace;
    params.barrier_workspace =
        static_cast<char*>(workspace) + PartialBytes<Element, LocalOnly>(sms * occupancy);
    logical_blocks = params.block_mapping.get_num_blocks();
    grid = dim3(min(sms * occupancy, logical_blocks));
  } else {
    grid = OutputTileMapping::get_grid_shape(params.grid_tiled_shape);
  }
  if (tasks.state && (tasks.retire_workers < 0 || tasks.retire_workers >= int(grid.x)))
    return cudaErrorInvalidValue;
  PublishedGemm<Element, StreamK, LocalOnly>
      <<<grid, Gemm::kThreadCount, SharedBytes<Element, StreamK, LocalOnly>(), stream>>>(
          params, logical_blocks, tasks);
  return cudaGetLastError();
}

}  // namespace flashinfer::published_gemm
