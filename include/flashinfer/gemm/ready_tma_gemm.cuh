/*
 * Copyright (c) 2026 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_GEMM_READY_TMA_GEMM_CUH_
#define FLASHINFER_GEMM_READY_TMA_GEMM_CUH_

#include <cuda_occupancy.h>
#include <cutlass/device_kernel.h>

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

#include "masked_gemm.cuh"

namespace flashinfer::ready_tma_gemm {

using namespace cute;
using ReadyTile = cutlass::gemm::GemmShape<128, 64, 64>;
using TileShape = Shape<Int<ReadyTile::kM>, Int<ReadyTile::kN>, Int<ReadyTile::kK>>;
using ClusterShape = Shape<_1, _1, _1>;
using InputStride = Stride<int64_t, _1, int64_t>;
using DefaultRegisterAllocation =
    cutlass::gemm::kernel::detail::WarpSpecializedRegisterAllocationFor<
        cutlass::gemm::KernelTmaWarpSpecializedPingpongSm120<2>, false>::type;

struct ReadySchedule : cutlass::gemm::KernelTmaWarpSpecializedPingpongSm120<2> {
#if FLASHINFER_READY_MATH_REGISTERS
  using RegisterAllocation =
      cutlass::gemm::WarpSpecializedRegisterAllocation<DefaultRegisterAllocation::LoadRegisters,
                                                       FLASHINFER_READY_MATH_REGISTERS>;
#endif
};

template <typename Base>
struct ReadyMainloop : Base {
  struct Arguments : Base::Arguments {
    int* readiness;
    int group_rows;
  };
  struct Params : Base::Params {
    int* readiness;
    int group_rows;
    int rows;
  };

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& shape, Arguments const& args,
                                        void* workspace) {
    return {Base::to_underlying_arguments(shape, args, workspace), args.readiness, args.group_rows,
            get<0>(shape)};
  }

  template <class LoadInputs, class BlockCoord, class KTileIterator>
  CUTLASS_DEVICE void load(Params const& params, typename Base::MainloopPipeline pipeline,
                           typename Base::PipelineState state, LoadInputs const& inputs,
                           BlockCoord const& block, KTileIterator k_tile, int k_tiles, int lane,
                           uint32_t cluster_rank, typename Base::TensorStorage& tensors) {
    const int first = int(get<0>(block)) * ReadyTile::kM;
    const int end = min(first + ReadyTile::kM, params.rows);
    for (int group = first / params.group_rows + lane; group <= (end - 1) / params.group_rows;
         group += 32) {
      const int expected = min(params.group_rows, params.rows - group * params.group_rows);
      cuda::atomic_ref<int, cuda::thread_scope_device> ready(params.readiness[group]);
      while (ready.load(cuda::memory_order_acquire) < expected) __nanosleep(64);
    }
    __syncwarp();
    // Publication orders the producer's generic stores; the elected TMA
    // issuer must also acquire those bytes into the async proxy.
    asm volatile("fence.proxy.async.global;" ::: "memory");
    Base::load(params, pipeline, state, inputs, block, k_tile, k_tiles, lane, cluster_rank,
               tensors);
  }
};

template <typename Element>
struct ReadyGemm {
  using MmaAtom = std::conditional_t<std::is_same_v<Element, cutlass::bfloat16_t>,
                                     SM80_16x8x16_F32BF16BF16F32_TN, SM80_16x8x16_F32F16F16F32_TN>;
  using TiledMma =
      decltype(make_tiled_mma(MmaAtom{}, Layout<Shape<_2, _2, _1>>{}, Tile<_128, _32, _16>{}));
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, void,
      cutlass::layout::RowMajor, 8, Element, cutlass::layout::RowMajor, 8,
      cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
  // Compose the existing typed mainloop directly: the convenience builder's
  // F8/F6/F4 restriction is not a restriction of its TMA pipeline or epilogue.
  using Mainloop = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm120TmaWarpSpecialized<3, 2, ClusterShape, ReadySchedule>, TileShape,
      Element, InputStride, Element, InputStride, TiledMma, SM90_TMA_LOAD,
      UMMA::Layout_K_SW128_Atom<Element>, Copy_Atom<SM75_U32x4_LDSM_N, Element>, identity,
      SM90_TMA_LOAD, UMMA::Layout_K_SW128_Atom<Element>, Copy_Atom<SM75_U32x4_LDSM_N, Element>,
      identity>;
  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, ReadyMainloop<Mainloop>,
                                           Epilogue, cutlass::gemm::StaticPersistentScheduler>;
};

template <typename Element>
struct ReadyKernel : ReadyGemm<Element>::Kernel {
  using Base = typename ReadyGemm<Element>::Kernel;
#if FLASHINFER_READY_MATH_REGISTERS
  static constexpr int LoadThreads = Base::NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
  static constexpr int MathThreads = Base::MaxThreadsPerBlock - LoadThreads;
  // The CTA pool is fixed at entry; setmaxnreg only redistributes that pool.
  static constexpr int EntryRegisterRequirement =
      ((Base::LoadRegisterRequirement * LoadThreads + Base::MmaRegisterRequirement * MathThreads +
        Base::MaxThreadsPerBlock * 8 - 1) /
       (Base::MaxThreadsPerBlock * 8)) *
      8;
#endif
  struct Params : Base::Params {
    int* readiness;
    int groups;
    int done_index;
  };

  CUTLASS_DEVICE void operator()(Params const& params, char* storage) {
    Base{}(params, storage);
    __syncthreads();
    masked_gemm::FinishReady(params.readiness, params.groups, params.done_index);
  }
};

template <typename Element>
cudaError_t PrepareReady(int* occupancy) {
  using Kernel = ReadyKernel<Element>;
  auto status =
      masked_gemm::PrepareKernel(cutlass::device_kernel<Kernel>, Kernel::SharedStorageSize);
  if (status != cudaSuccess) return status;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(occupancy, cutlass::device_kernel<Kernel>,
                                                       Kernel::MaxThreadsPerBlock,
                                                       Kernel::SharedStorageSize);
}

template <typename Element>
cudaError_t PlanReady(cudaDeviceProp const& device, int producer_registers, size_t producer_shared,
                      int producer_threads, int* math_registers, bool* concurrent) {
  using Kernel = ReadyKernel<Element>;
  cudaFuncAttributes consumer;
  auto status = cudaFuncGetAttributes(&consumer, cutlass::device_kernel<Kernel>);
  if (status != cudaSuccess) return status;
  cudaOccDeviceProp hardware(device);
  cudaOccDeviceState state;
  state.carveoutConfig = SHAREDMEM_CARVEOUT_MAX_SHARED;
  cudaOccFuncAttributes consumer_attributes(consumer);
  cudaOccFuncAttributes producer_attributes;
  producer_attributes.maxThreadsPerBlock = device.maxThreadsPerBlock;
  producer_attributes.numRegs = producer_registers;
  producer_attributes.shmemLimitConfig = FUNC_SHMEM_LIMIT_OPTIN;
  producer_attributes.maxDynamicSharedSizeBytes = producer_shared;
  producer_attributes.numBlockBarriers = 1;
  cudaOccResult consumer_occupancy, producer_occupancy;
  int granularity, partitions;
  if (cudaOccRegAllocationGranularity(&granularity, &hardware) != CUDA_OCC_SUCCESS ||
      cudaOccSubPartitionsPerMultiprocessor(&partitions, &hardware) != CUDA_OCC_SUCCESS ||
      cudaOccMaxActiveBlocksPerMultiprocessor(&consumer_occupancy, &hardware, &consumer_attributes,
                                              &state, Kernel::MaxThreadsPerBlock,
                                              Kernel::SharedStorageSize) != CUDA_OCC_SUCCESS ||
      cudaOccMaxActiveBlocksPerMultiprocessor(&producer_occupancy, &hardware, &producer_attributes,
                                              &state, producer_threads,
                                              producer_shared) != CUDA_OCC_SUCCESS) {
    return cudaErrorInvalidValue;
  }
  const int producer_warps = __occDivideRoundUp(producer_threads, device.warpSize);
  const int consumer_warps = Kernel::MaxThreadsPerBlock / device.warpSize;
  const int producer_regs = producer_occupancy.allocatedRegistersPerBlock / producer_warps *
                            __occRoundUp(producer_warps, partitions);
  const int consumer_regs = consumer_occupancy.allocatedRegistersPerBlock;
  const int uniform_registers = (device.regsPerMultiprocessor - producer_regs) / consumer_warps /
                                granularity * granularity / device.warpSize;
  constexpr int consumer_threads = Kernel::MaxThreadsPerBlock;
  constexpr int load_threads = Kernel::NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
  constexpr int math_threads = consumer_threads - load_threads;
  const int budget =
      (uniform_registers * consumer_threads - int(Kernel::LoadRegisterRequirement) * load_threads) /
      math_threads;
  // setmaxnreg encodes registers in multiples of eight. This is a resource
  // bound from the paired producer, independent of GEMM timing or tile search.
  *math_registers =
      budget >= 24 ? std::min(DefaultRegisterAllocation::MathRegisters, budget / 8 * 8) : 0;
  *concurrent = consumer_occupancy.activeBlocksPerMultiprocessor > 0 &&
                producer_occupancy.activeBlocksPerMultiprocessor > 0 &&
                consumer_regs + producer_regs <= device.regsPerMultiprocessor &&
                consumer_occupancy.allocatedSharedMemPerBlock +
                        producer_occupancy.allocatedSharedMemPerBlock <=
                    device.sharedMemPerMultiprocessor &&
                (consumer_warps + __occRoundUp(producer_warps, partitions)) * device.warpSize <=
                    device.maxThreadsPerMultiProcessor &&
                consumer_occupancy.blockLimitBlocks >= 2;
  return cudaSuccess;
}

template <typename Element>
constexpr size_t ReadyWorkspaceBytes(int) {
  return 0;
}

template <typename Element>
cudaError_t RunReady(Element* a, Element* b, Element* out, int m, int n, int k, int lda, int ldb,
                     int ldd, int* readiness, int group_rows, int done_index, void* workspace,
                     int sms, int occupancy, int available_sms, cudaStream_t stream) {
  using Kernel = ReadyKernel<Element>;
  typename Kernel::Arguments args;
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = {m, n, k, 1};
  args.mainloop = {{a, {lda, _1{}, 0}, b, {ldb, _1{}, 0}}, readiness, group_rows};
  args.epilogue.thread = {1.0f, 0.0f};
  args.epilogue.ptr_D = out;
  args.epilogue.dD = {ldd, _1{}, 0};
  args.hw_info.sm_count = available_sms;
  typename Kernel::Params params{Kernel::to_underlying_arguments(args, workspace), readiness,
                                 (m + group_rows - 1) / group_rows, done_index};
  auto grid = Kernel::get_grid_shape(params);
  cutlass::device_kernel<Kernel>
      <<<grid, Kernel::MaxThreadsPerBlock, Kernel::SharedStorageSize, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace flashinfer::ready_tma_gemm
#endif
