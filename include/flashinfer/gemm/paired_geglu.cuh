/*
 * Copyright (c) 2026 by FlashInfer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "masked_gemm.cuh"

namespace flashinfer::paired_geglu {

using namespace masked_gemm;
using Element = cutlass::bfloat16_t;

// The existing linear epilogue converts FP32 accumulators to BF16 first.
// Weight columns are [gate0, up0, gate1, up1, ...].
template <typename Base>
struct CompactStore : Base {
  using ThreadMap = typename Base::ThreadMap;
  using Fragment = typename Base::Fragment;
  using TensorCoord = typename Base::TensorCoord;
  using Params = typename Base::Params;
  using Access = cutlass::AlignedArray<Element, Base::kElementsPerAccess / 2>;
  static_assert(Base::kElementsPerAccess % 2 == 0);

  Element* output;
  int64_t stride;

  CUTLASS_DEVICE CompactStore(Params const& params, Element* pointer, TensorCoord extent,
                              int thread, TensorCoord offset = TensorCoord(),
                              int const* indices = nullptr)
      : Base(params, pointer, extent, thread, offset, indices),
        output(pointer),
        stride(params.stride / sizeof(Element) / 2) {}

  CUTLASS_DEVICE void store(Fragment const& fragment) const {
    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int row_index = this->thread_start_row() + row * ThreadMap::Delta::kRow +
                          group * ThreadMap::Delta::kGroup + cluster * ThreadMap::Delta::kCluster;
          int fragment_row =
              (cluster * ThreadMap::Iterations::kGroup + group) * ThreadMap::Iterations::kRow + row;
          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            int column_index = this->thread_start_column() + column * ThreadMap::Delta::kColumn;
            int fragment_start =
                (fragment_row * ThreadMap::Iterations::kColumn + column) * Base::kElementsPerAccess;
            Access activated;
            CUTLASS_PRAGMA_UNROLL
            for (int pair = 0; pair < Base::kElementsPerAccess / 2; ++pair) {
              float gate = float(fragment[fragment_start + pair * 2]);
              float up = float(fragment[fragment_start + pair * 2 + 1]);
              constexpr float beta = M_SQRT2 * M_2_SQRTPI * 0.5f;
              float cube = gate * gate * gate;
              float inner = beta * (gate + 0.044715f * cube);
              Element gelu(0.5f * gate * (1.0f + ::tanhf(inner)));
              activated[pair] = Element(float(gelu) * up);
            }
            bool valid =
                output && row_index < this->extent_row() && column_index < this->extent_column();
            cutlass::arch::global_store<Access, sizeof(Access)>(
                activated, output + row_index * stride + column_index / 2, valid);
          }
        }
      }
    }
  }
};

using Store = CompactStore<typename LocalGemm<Element>::Epilogue::OutputTileIterator>;
using Epilogue = RebindEpilogue<typename LocalGemm<Element>::Epilogue, Store>;
using Kernel =
    cutlass::gemm::kernel::Gemm<typename LocalGemm<Element>::Mma, Epilogue, Swizzle, false>;

__global__ __launch_bounds__(Kernel::kThreadCount) void PairedGeGLU(Kernel::Params params,
                                                                    uint8_t const* padding) {
  extern __shared__ char storage[];
  auto tile = Swizzle::get_tile_offset(params.swizzle_log_tile);
  bool live = false;
  for (int offset = threadIdx.x; offset < Tile::kM; offset += blockDim.x) {
    int row = tile.m() * Tile::kM + offset;
    live |= row < params.problem_size.m() && (!padding || !padding[row]);
  }
  if (!__syncthreads_or(live)) return;
  Kernel{}(params, *reinterpret_cast<Kernel::SharedStorage*>(storage));
}

inline cudaError_t Prepare(int* registers, int* shared, int* threads) {
  auto status = PrepareKernel(PairedGeGLU, sizeof(Kernel::SharedStorage));
  if (status != cudaSuccess) return status;
  cudaFuncAttributes attributes;
  status = cudaFuncGetAttributes(&attributes, PairedGeGLU);
  if (status != cudaSuccess) return status;
  *registers = attributes.numRegs;
  *shared = attributes.sharedSizeBytes + sizeof(Kernel::SharedStorage);
  *threads = Kernel::kThreadCount;
  return cudaSuccess;
}

inline cudaError_t Run(Element* x, Element* packed_weight, Element* out, int m, int n, int k,
                       int lda, int ldb, int ldd, uint8_t const* padding, cudaStream_t stream) {
  cutlass::gemm::GemmCoord problem(m, 2 * n, k);
  auto tiled = Swizzle::get_tiled_shape(problem, {Tile::kM, Tile::kN, Tile::kK}, 1);
  // The store iterator maps this virtual interleaved layout onto compact output.
  Kernel::Params params(problem, tiled, {x, RowMajor(lda)}, {packed_weight, ColumnMajor(ldb)},
                        {out, RowMajor(2 * ldd)}, {out, RowMajor(2 * ldd)}, {1.0f, 0.0f});
  PairedGeGLU<<<Swizzle::get_grid_shape(tiled), Kernel::kThreadCount, sizeof(Kernel::SharedStorage),
                stream>>>(params, padding);
  return cudaGetLastError();
}

}  // namespace flashinfer::paired_geglu
