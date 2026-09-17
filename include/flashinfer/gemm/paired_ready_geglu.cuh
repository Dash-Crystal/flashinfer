/*
 * Copyright (c) 2026 by FlashInfer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "ready_tma_gemm.cuh"

namespace flashinfer::paired_ready_geglu {

using namespace cute;
using Element = cutlass::bfloat16_t;
namespace fusion = cutlass::epilogue::fusion;

// CUTLASS EVT: compact auxiliary store, with the ordinary D store disabled.
struct PairedStore {
  using ElementAux = Element;
  struct SharedStorage {};
  struct Arguments {
    Element* output = nullptr;
    int64_t stride = 0;
  };
  using Params = Arguments;

  template <class Shape>
  static Params to_underlying_arguments(Shape const&, Arguments const& args, void*) {
    return args;
  }
  template <class Shape>
  static bool can_implement(Shape const&, Arguments const&) {
    return true;
  }
  template <class Shape>
  static size_t get_workspace_size(Shape const&, Arguments const&) {
    return 0;
  }
  template <class Shape>
  static cutlass::Status initialize_workspace(Shape const&, Arguments const&, void*, cudaStream_t,
                                              cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  Params const* params;
  CUTLASS_HOST_DEVICE PairedStore() = default;
  CUTLASS_HOST_DEVICE PairedStore(Params const& params_, SharedStorage const&) : params(&params_) {}
  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }
  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(fusion::ProducerLoadArgs<Args...> const&) {
    return fusion::EmptyProducerLoadCallbacks{};
  }

  template <class Coordinates>
  struct Consumer : fusion::EmptyConsumerStoreCallbacks {
    Coordinates coordinates;
    Params const* params;
    int rows, columns;

    CUTLASS_DEVICE Consumer(Coordinates coordinates_, Params const* params_, int rows_,
                            int columns_)
        : coordinates(coordinates_), params(params_), rows(rows_), columns(columns_) {}

    template <typename Acc, typename Input, int Count>
    CUTLASS_DEVICE auto visit(cutlass::Array<Acc, Count> const&, int epi_v, int epi_m, int epi_n,
                              cutlass::Array<Input, Count> const& input) {
      static_assert(Count == 4, "Paired READY uses the SM120 four-value MMA fragment");
      auto coords = coalesce(coordinates(_, _, _, epi_m, epi_n));
      // SM80 CLayout and the SM120 R2S source mapping put each gate/up pair
      // in adjacent registers. No other thread's accumulator is needed.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Count; i += 2) {
        auto coord = coords(epi_v * Count + i);
        int row = get<0>(coord), column = get<1>(coord);
        if (row < rows && column < columns) {
          float gate = float(Element(input[i]));
          float up = float(Element(input[i + 1]));
          constexpr float beta = M_SQRT2 * M_2_SQRTPI * 0.5f;
          float cube = gate * gate * gate;
          float inner = beta * (gate + 0.044715f * cube);
          Element gelu(0.5f * gate * (1.0f + ::tanhf(inner)));
          params->output[int64_t(row) * params->stride + column / 2] = Element(float(gelu) * up);
        }
      }
      return input;
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(fusion::ConsumerStoreArgs<Args...> const& args) {
    auto [m, n, k, l] = args.problem_shape_mnkl;
    auto identity = make_identity_tensor(make_shape(m, n, l));
    auto coords = fusion::sm90_partition_for_epilogue<ReferenceSrc>(
        identity, args.tile_shape_mnk, args.tile_coord_mnkl, args.epi_tile, args.tiled_copy,
        args.thread_idx);
    return Consumer<decltype(coords)>(coords, params, m, n);
  }
};

using Ordinary = ready_tma_gemm::ReadyGemm<Element>;
using Callbacks = fusion::Sm90EVT<PairedStore, fusion::Sm90AccFetch>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, typename Ordinary::TileShape,
    ready_tma_gemm::ClusterShape, Shape<_64, _32>, float, float, void, cutlass::layout::RowMajor, 8,
    void, cutlass::layout::RowMajor, 8, cutlass::epilogue::TmaWarpSpecialized,
    Callbacks>::CollectiveOp;
using BaseKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, ready_tma_gemm::ReadyMainloop<typename Ordinary::Mainloop, 128>,
    Epilogue, cutlass::gemm::StaticPersistentScheduler>;

struct Kernel : BaseKernel {
#if FLASHINFER_READY_MATH_REGISTERS
  static constexpr int LoadThreads =
      BaseKernel::NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
  static constexpr int MathThreads = BaseKernel::MaxThreadsPerBlock - LoadThreads;
  static constexpr int EntryRegisterRequirement =
      ((BaseKernel::LoadRegisterRequirement * LoadThreads +
        BaseKernel::MmaRegisterRequirement * MathThreads + BaseKernel::MaxThreadsPerBlock * 8 - 1) /
       (BaseKernel::MaxThreadsPerBlock * 8)) *
      8;
#endif
  struct Params : BaseKernel::Params {
    int* readiness;
    int groups;
    int done_index;
  };
  CUTLASS_DEVICE void operator()(Params const& params, char* storage) {
    BaseKernel{}(params, storage);
    __syncthreads();
    masked_gemm::FinishReady(params.readiness, params.groups, params.done_index);
  }
};

inline cudaError_t Prepare(int* registers, int* shared, int* threads, int* occupancy) {
  auto kernel = cutlass::device_kernel<Kernel>;
  auto status = masked_gemm::PrepareKernel(kernel, Kernel::SharedStorageSize);
  if (status != cudaSuccess) return status;
  cudaFuncAttributes attributes;
  status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) return status;
  *registers = attributes.numRegs;
  *shared = attributes.sharedSizeBytes + Kernel::SharedStorageSize;
  *threads = Kernel::MaxThreadsPerBlock;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      occupancy, kernel, Kernel::MaxThreadsPerBlock, Kernel::SharedStorageSize);
}

inline cudaError_t Run(Element* x, Element* weight, Element* out, int m, int n, int k, int lda,
                       int ldb, int ldd, int* readiness, int group_rows, int done_index,
                       int available_sms, cudaStream_t stream) {
  BaseKernel::Arguments args;
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = {m, 2 * n, k, 1};
  args.mainloop = {{x, {lda, _1{}, 0}, weight, {ldb, _1{}, 0}}, readiness, group_rows, 0, m};
  args.epilogue.thread = {{}, {out, ldd}};
  args.hw_info.sm_count = available_sms;
  args.scheduler.raster_order = BaseKernel::TileScheduler::RasterOrderOptions::AlongN;
  args.scheduler.max_swizzle_size = 1;
  Kernel::Params params{BaseKernel::to_underlying_arguments(args, nullptr), readiness,
                        (m + group_rows - 1) / group_rows, done_index};
  cutlass::device_kernel<Kernel><<<Kernel::get_grid_shape(params), Kernel::MaxThreadsPerBlock,
                                   Kernel::SharedStorageSize, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace flashinfer::paired_ready_geglu
