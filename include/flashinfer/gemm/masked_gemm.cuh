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
#include <cutlass/numeric_types.h>

namespace flashinfer::masked_gemm {

using Tile = cutlass::gemm::GemmShape<128, 64, 64>;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;

template <typename Element>
using LocalGemm = typename cutlass::gemm::kernel::DefaultGemm<
    Element, RowMajor, 8, Element, ColumnMajor, 8, Element, RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, Tile, cutlass::gemm::GemmShape<64, 32, 64>,
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

template <typename Element, bool Publish>
cudaError_t Prepare() {
  auto status = cudaFuncSetAttribute(MaskedGemm<Element, Publish>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     sizeof(typename Gemm<Element, Publish>::SharedStorage));
  if (status != cudaSuccess) return status;
  return cudaFuncSetAttribute(MaskedGemm<Element, Publish>,
                              cudaFuncAttributePreferredSharedMemoryCarveout,
                              cudaSharedmemCarveoutMaxShared);
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

}  // namespace flashinfer::masked_gemm
#endif
