/*
 * Copyright (c) 2026 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_GEMM_MASKED_GEMM_CUH_
#define FLASHINFER_GEMM_MASKED_GEMM_CUH_

#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/gemm/kernel/gemm_universal_streamk.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle_streamk.h>
#include <cutlass/numeric_types.h>

#include <cuda/atomic>

namespace flashinfer::masked_gemm {

using Tile = cutlass::gemm::GemmShape<128, 64, 64>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;

template <typename Element, typename Shape = Tile,
          typename WarpShape = cutlass::gemm::GemmShape<64, 32, 64>>
using LocalGemm = typename cutlass::gemm::kernel::DefaultGemm<
    Element, RowMajor, 8, Element, ColumnMajor, 8, Element, RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, Shape, WarpShape,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>, Swizzle, 3, false,
    cutlass::arch::OpMultiplyAdd>::GemmKernel;

template <typename Iterator>
struct PeerStoreIterator : Iterator {
  using Element = typename Iterator::Element;
  using TensorCoord = typename Iterator::TensorCoord;
  using Fragment = typename Iterator::Fragment;

  struct Params : Iterator::Params {
    Element* peer = nullptr;
    CUTLASS_HOST_DEVICE Params() = default;
    CUTLASS_HOST_DEVICE Params(RowMajor layout) : Iterator::Params(layout) {}
  };

  Iterator peer;

  CUTLASS_DEVICE PeerStoreIterator(Params const& params, Element* pointer, TensorCoord extent,
                                   int thread_idx, TensorCoord offset = TensorCoord(),
                                   int const* indices = nullptr)
      : Iterator(params, pointer, extent, thread_idx, offset, indices),
        peer(params, params.peer, extent, thread_idx, offset, indices) {}

  CUTLASS_DEVICE void store(Fragment const& fragment) const {
    Iterator::store(fragment);
    peer.store(fragment);
  }

  CUTLASS_DEVICE PeerStoreIterator& operator++() {
    Iterator::operator++();
    ++peer;
    return *this;
  }
};

template <typename Epilogue>
using PeerEpilogue = cutlass::epilogue::threadblock::Epilogue<
    typename Epilogue::Shape, typename Epilogue::WarpMmaOperator, Epilogue::kPartitionsK,
    PeerStoreIterator<typename Epilogue::OutputTileIterator>,
    typename Epilogue::AccumulatorFragmentIterator, typename Epilogue::WarpTileIterator,
    typename Epilogue::SharedLoadIterator, typename Epilogue::OutputOp, typename Epilogue::Padding>;

template <typename Element, bool Publish>
using Gemm =
    std::conditional_t<Publish,
                       cutlass::gemm::kernel::Gemm<
                           typename LocalGemm<Element>::Mma,
                           PeerEpilogue<typename LocalGemm<Element>::Epilogue>, Swizzle, false>,
                       LocalGemm<Element>>;

template <typename Element, bool Publish>
__global__ __launch_bounds__(Gemm<Element, Publish>::kThreadCount) void MaskedGemm(
    typename Gemm<Element, Publish>::Params params, const uint8_t* is_padding, int row_offset) {
  const auto tile = Swizzle::get_tile_offset(params.swizzle_log_tile);
  bool live = false;
  for (int i = threadIdx.x; i < Tile::kM; i += blockDim.x) {
    const int row = tile.m() * Tile::kM + i;
    live |= row < params.problem_size.m() && !is_padding[row_offset + row];
  }
  if (!__syncthreads_or(live)) return;
  extern __shared__ char storage[];
  Gemm<Element, Publish>()(
      params, *reinterpret_cast<typename Gemm<Element, Publish>::SharedStorage*>(storage));
  // Kernel completion orders every producer thread before dependent work.
  // The caller's system release/acquire handoff then publishes peer stores.
}

template <typename Kernel>
cudaError_t PrepareKernel(Kernel kernel, size_t shared_bytes) {
  auto status =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
  if (status != cudaSuccess) return status;
  return cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                              cudaSharedmemCarveoutMaxShared);
}

template <typename Element, bool Publish>
cudaError_t Prepare() {
  return PrepareKernel(MaskedGemm<Element, Publish>,
                       sizeof(typename Gemm<Element, Publish>::SharedStorage));
}

template <typename Element, bool Publish>
cudaError_t Run(Element* a, Element* b, Element* out, int m, int n, int k, int lda, int ldb,
                int ldd, const uint8_t* is_padding, int row_offset, Element* peer,
                cudaStream_t stream) {
  const cutlass::gemm::GemmCoord problem(m, n, k);
  const auto grid = Swizzle::get_tiled_shape(problem, {Tile::kM, Tile::kN, Tile::kK}, 1);
  typename Gemm<Element, Publish>::Params params(problem, grid, {a, RowMajor(lda)},
                                                 {b, ColumnMajor(ldb)}, {out, RowMajor(ldd)},
                                                 {out, RowMajor(ldd)}, {1.0f, 0.0f});
  if constexpr (Publish) params.params_D.peer = peer;
  MaskedGemm<Element, Publish>
      <<<Swizzle::get_grid_shape(grid), Gemm<Element, Publish>::kThreadCount,
         sizeof(typename Gemm<Element, Publish>::SharedStorage), stream>>>(params, is_padding,
                                                                           row_offset);
  return cudaGetLastError();
}

using ReadyTile = cutlass::gemm::GemmShape<64, 64, 64>;

template <typename Iterator, int TileRows, int Threads>
struct ReadyIterator : Iterator {
  using Element = typename Iterator::Element;
  using TensorCoord = typename Iterator::TensorCoord;
  struct Params : Iterator::Params {
    using Iterator::Params::Params;
    int* readiness = nullptr;
    int group_rows = 0;
  };

  CUTLASS_DEVICE ReadyIterator(Params const& params, Element* pointer, TensorCoord extent,
                               int thread, TensorCoord offset)
      : Iterator(params, pointer, extent, thread, offset) {
    const int end = min(offset.row() + TileRows, extent.row());
    for (int group = offset.row() / params.group_rows + thread;
         group <= (end - 1) / params.group_rows; group += Threads) {
      const int expected = min(params.group_rows, extent.row() - group * params.group_rows);
      cuda::atomic_ref<int, cuda::thread_scope_device> ready(params.readiness[group]);
      while (ready.load(cuda::memory_order_acquire) < expected) __nanosleep(64);
    }
    __syncthreads();
  }
};

template <typename Base>
struct ReadyMma : Base {
  using Base::Base;
  using IteratorA =
      ReadyIterator<typename Base::IteratorA, Base::Shape::kM, Base::WarpCount::kCount * 32>;
};

struct ReadySwizzle : cutlass::gemm::threadblock::ThreadblockSwizzleStreamK {
  using Base = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
  using Base::Base;

  CUTLASS_DEVICE int get_block_idx() const {
    extern __shared__ char storage[];
    const int offset = *reinterpret_cast<int*>(storage);
    // The first physical wave holds every Stream-K peer at the prepared
    // occupancy. Preserve CUTLASS's region ordering for that wave; later
    // virtual waves contain data-parallel work and reduction epilogues.
    return offset == 0 ? Base::get_block_idx() : int(blockIdx.x) + offset;
  }
};

template <typename Element>
using ReadyBase = LocalGemm<Element, ReadyTile, cutlass::gemm::GemmShape<32, 32, 64>>;

template <typename Element>
using ReadyGemm =
    cutlass::gemm::kernel::GemmUniversalStreamk<ReadyMma<typename ReadyBase<Element>::Mma>,
                                                typename ReadyBase<Element>::Epilogue,
                                                ReadySwizzle>;

template <typename Element>
constexpr size_t ReadySharedBytes() {
  static_assert(alignof(typename ReadyGemm<Element>::SharedStorage) <= 128);
  return 128 + sizeof(typename ReadyGemm<Element>::SharedStorage);
}

template <typename Element>
constexpr size_t ReadyPartialBytes(int slots) {
  return (slots * ReadyGemm<Element>::kWorkspaceBytesPerBlock + 127) / 128 * 128;
}

template <typename Element>
constexpr size_t ReadyWorkspaceBytes(int slots) {
  // Native separate reductions use one flag per accumulator fragment; their
  // tile count cannot exceed the number of resident Stream-K peers.
  constexpr int flags_per_slot = ReadyGemm<Element>::Epilogue::kAccumulatorFragments;
  return ReadyPartialBytes<Element>(slots) +
         (slots * flags_per_slot * sizeof(int) + 127) / 128 * 128;
}

template <typename Element>
__global__ __launch_bounds__(ReadyGemm<Element>::kThreadCount) void ReadyRowsGemm(
    typename ReadyGemm<Element>::Params params, int* readiness, int groups, int done_index,
    int logical_blocks) {
  extern __shared__ char storage[];
  auto& shared = *reinterpret_cast<typename ReadyGemm<Element>::SharedStorage*>(storage + 128);
  for (int offset = 0; int(blockIdx.x) + offset < logical_blocks; offset += gridDim.x) {
    if (threadIdx.x == 0) *reinterpret_cast<int*>(storage) = offset;
    __syncthreads();
    ReadyGemm<Element>::invoke(params, shared);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    cuda::atomic_ref<int, cuda::thread_scope_device> done(readiness[done_index]);
    if (done.fetch_add(1, cuda::memory_order_acq_rel) == gridDim.x - 1) {
      for (int group = 0; group < groups; ++group) readiness[group] = 0;
      done.store(0, cuda::memory_order_relaxed);
    }
  }
}

template <typename Element>
cudaError_t PrepareReady(int* occupancy) {
  auto status = PrepareKernel(ReadyRowsGemm<Element>, ReadySharedBytes<Element>());
  if (status != cudaSuccess) return status;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(occupancy, ReadyRowsGemm<Element>,
                                                       ReadyGemm<Element>::kThreadCount,
                                                       ReadySharedBytes<Element>());
}

template <typename Element>
cudaError_t RunReady(Element* a, Element* b, Element* out, int m, int n, int k, int lda, int ldb,
                     int ldd, int* readiness, int group_rows, int done_index, void* workspace,
                     int sms, int occupancy, int available_sms, cudaStream_t stream) {
  using Kernel = ReadyGemm<Element>;
  typename Kernel::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k}, 1,
                                  {1.0f, 0.0f}, a, b, out, out, 0, 0, 0, 0, int64_t(lda),
                                  int64_t(ldb), int64_t(ldd), int64_t(ldd), available_sms);
  typename Kernel::Params params(args, sms, occupancy);
  params.params_A.readiness = readiness;
  params.params_A.group_rows = group_rows;
  // Fixed offsets across shapes keep previous FP32 partials out of the barrier
  // allocation. CUTLASS resets its flags after the consuming peer's reduction.
  params.partials_workspace = workspace;
  params.barrier_workspace =
      static_cast<char*>(workspace) + ReadyPartialBytes<Element>(sms * occupancy);
  const int logical_blocks = params.block_mapping.get_num_blocks();
  const int blocks = min(available_sms * occupancy, logical_blocks);
  ReadyRowsGemm<Element><<<blocks, Kernel::kThreadCount, ReadySharedBytes<Element>(), stream>>>(
      params, readiness, (m + group_rows - 1) / group_rows, done_index, logical_blocks);
  return cudaGetLastError();
}

}  // namespace flashinfer::masked_gemm
#endif
