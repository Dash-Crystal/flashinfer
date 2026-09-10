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
#include <cutlass/gemm/kernel/gemm_universal.h>
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

struct ReadySwizzle : Swizzle {
  cutlass::gemm::GemmCoord tile;
  CUTLASS_DEVICE cutlass::gemm::GemmCoord get_tile_offset(int) const { return tile; }
};

template <typename Element>
using ReadyBase = LocalGemm<Element, ReadyTile, cutlass::gemm::GemmShape<32, 32, 64>>;

template <typename Element>
using ReadyGemm =
    cutlass::gemm::kernel::GemmUniversal<typename ReadyBase<Element>::Mma,
                                         typename ReadyBase<Element>::Epilogue, ReadySwizzle>;

// The caller reserves SMs for the producer, publishes complete row groups with
// device-release increments, and joins both kernels before reusing this workspace.
template <typename Element>
__global__ __launch_bounds__(ReadyGemm<Element>::kThreadCount) void ReadyRowsGemm(
    typename ReadyGemm<Element>::Params params, int* readiness, int group_rows, int done_index) {
  extern __shared__ char storage[];
  auto& shared = *reinterpret_cast<typename ReadyGemm<Element>::SharedStorage*>(storage);
  ReadySwizzle schedule;
  for (int n = blockIdx.x; n < params.grid_tiled_shape.n(); n += gridDim.x) {
    // Finish the M tiles while this CTA's K x N weight tile can remain in L2.
    for (int m = 0; m < params.grid_tiled_shape.m(); ++m) {
      const int begin = m * ReadyTile::kM;
      const int end = min(begin + ReadyTile::kM, params.problem_size.m());
      if (threadIdx.x == 0) {
        for (int group = begin / group_rows; group <= (end - 1) / group_rows; ++group) {
          const int expected = min(group_rows, params.problem_size.m() - group * group_rows);
          cuda::atomic_ref<int, cuda::thread_scope_device> ready(readiness[group]);
          while (ready.load(cuda::memory_order_acquire) < expected) __nanosleep(64);
        }
      }
      __syncthreads();
      schedule.tile = {m, n, 0};
      ReadyGemm<Element>().run_with_swizzle(params, shared, schedule);
      __syncthreads();
    }
  }
  if (threadIdx.x == 0) {
    cuda::atomic_ref<int, cuda::thread_scope_device> done(readiness[done_index]);
    if (done.fetch_add(1, cuda::memory_order_acq_rel) == gridDim.x - 1) {
      for (int group = 0; group < done_index; ++group) readiness[group] = 0;
      done.store(0, cuda::memory_order_relaxed);
    }
  }
}

template <typename Element>
cudaError_t PrepareReady() {
  return PrepareKernel(ReadyRowsGemm<Element>, sizeof(typename ReadyGemm<Element>::SharedStorage));
}

template <typename Element>
cudaError_t RunReady(Element* a, Element* b, Element* out, int m, int n, int k, int lda, int ldb,
                     int ldd, int* readiness, int group_rows, int done_index, int blocks,
                     cudaStream_t stream) {
  using Kernel = ReadyGemm<Element>;
  typename Kernel::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k}, 1,
                                  {1.0f, 0.0f}, a, b, out, out, 0, 0, 0, 0, int64_t(lda),
                                  int64_t(ldb), int64_t(ldd), int64_t(ldd));
  typename Kernel::Params params(args, blocks, 1);
  ReadyRowsGemm<Element>
      <<<blocks, Kernel::kThreadCount, sizeof(typename Kernel::SharedStorage), stream>>>(
          params, readiness, group_rows, done_index);
  return cudaGetLastError();
}

}  // namespace flashinfer::masked_gemm
#endif
