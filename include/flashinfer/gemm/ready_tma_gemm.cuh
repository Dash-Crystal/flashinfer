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

#include "eligible_ready_scheduler.cuh"
#include "masked_gemm.cuh"

namespace flashinfer::ready_tma_gemm {

using namespace cute;
using ReadyTile = cutlass::gemm::GemmShape<FLASHINFER_READY_ROW_TILE, 64, 64>;
using TileShape = Shape<Int<ReadyTile::kM>, Int<ReadyTile::kN>, Int<ReadyTile::kK>>;
using ClusterShape = Shape<_1, _1, _1>;
using InputStride = Stride<int64_t, _1, int64_t>;
using DefaultRegisterAllocation =
    cutlass::gemm::kernel::detail::WarpSpecializedRegisterAllocationFor<
        cutlass::gemm::KernelTmaWarpSpecializedPingpongSm120<2>, false>::type;

struct ReadySchedule : cutlass::gemm::KernelTmaWarpSpecializedPingpongSm120<2> {
#if FLASHINFER_READY_STARTUP_RAMP
  static constexpr int OperandStorageRows = 128;
#endif
#if FLASHINFER_READY_MATH_REGISTERS
  using RegisterAllocation =
      cutlass::gemm::WarpSpecializedRegisterAllocation<DefaultRegisterAllocation::LoadRegisters,
                                                       FLASHINFER_READY_MATH_REGISTERS>;
#endif
};

template <typename Base, int Rows>
struct ReadyMainloop : Base {
  struct Arguments : Base::Arguments {
    int* readiness;
    int group_rows;
    int row_offset = 0;
    int total_rows = 0;
  };
  struct Params : Base::Params {
    int* readiness;
    int group_rows;
    int rows;
    int row_offset;
    int total_rows;
  };

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& shape, Arguments const& args,
                                        void* workspace) {
    return {Base::to_underlying_arguments(shape, args, workspace),
            args.readiness,
            args.group_rows,
            get<0>(shape),
            args.row_offset,
            args.total_rows ? args.total_rows : int(get<0>(shape))};
  }

  template <class LoadInputs, class BlockCoord, class KTileIterator>
  CUTLASS_DEVICE void load(Params const& params, typename Base::MainloopPipeline pipeline,
                           typename Base::PipelineState state, LoadInputs const& inputs,
                           BlockCoord const& block, KTileIterator k_tile, int k_tiles, int lane,
                           uint32_t cluster_rank, typename Base::TensorStorage& tensors) {
    auto ready_a = [&] {
      const int first = params.row_offset + int(get<0>(block)) * Rows;
      const int end = min(first + Rows, params.row_offset + params.rows);
      for (int group = first / params.group_rows + lane; group <= (end - 1) / params.group_rows;
           group += 32) {
        const int expected = min(params.group_rows, params.total_rows - group * params.group_rows);
        cuda::atomic_ref<int, cuda::thread_scope_device> ready(params.readiness[group]);
        while (ready.load(cuda::memory_order_acquire) < expected) __nanosleep(64);
      }
      __syncwarp();
      // Acquire published generic stores before issuing A's async loads.
      asm volatile("fence.proxy.async.global;" ::: "memory");
    };
    Base::load(params, pipeline, state, inputs, block, k_tile, k_tiles, lane, cluster_rank, tensors,
               ready_a, Int<Base::DispatchPolicy::Stages>{});
  }
};

template <typename Element, int Rows = ReadyTile::kM>
struct ReadyGemm {
  using TileShape = Shape<Int<Rows>, _64, _64>;
  using MmaAtom = std::conditional_t<std::is_same_v<Element, cutlass::bfloat16_t>,
                                     SM80_16x8x16_F32BF16BF16F32_TN, SM80_16x8x16_F32F16F16F32_TN>;
  using TiledMma =
      decltype(make_tiled_mma(MmaAtom{}, Layout<Shape<_2, _2, _1>>{}, Tile<Int<Rows>, _32, _16>{}));
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      Shape<Int<(Rows % 64 == 0 ? 64 : 32)>, _32>, float, float, void, cutlass::layout::RowMajor, 8,
      Element, cutlass::layout::RowMajor, 8, cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
  // Compose the existing typed mainloop directly: the convenience builder's
  // F8/F6/F4 restriction is not a restriction of its TMA pipeline or epilogue.
  using Mainloop = cutlass::gemm::collective::CollectiveMma<
      cutlass::gemm::MainloopSm120TmaWarpSpecialized<3, 2, ClusterShape, ReadySchedule>, TileShape,
      Element, InputStride, Element, InputStride, TiledMma, SM90_TMA_LOAD,
      UMMA::Layout_K_SW128_Atom<Element>, Copy_Atom<SM75_U32x4_LDSM_N, Element>, identity,
      SM90_TMA_LOAD, UMMA::Layout_K_SW128_Atom<Element>, Copy_Atom<SM75_U32x4_LDSM_N, Element>,
      identity>;
  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, ReadyMainloop<Mainloop, Rows>,
                                           Epilogue, EligibleReadySchedulerTag>;
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
    masked_gemm::FinishReady(params.readiness, params.groups, params.done_index,
                             params.scheduler.task_state,
                             params.scheduler.tiles_m * (params.scheduler.tiles_n + 1));
  }
};

template <typename Element>
struct GeometricReadyKernel : ReadyKernel<Element> {
  using Small = typename ReadyGemm<Element, 32>::Kernel;
  using Medium = typename ReadyGemm<Element, 96>::Kernel;
  using Large = typename ReadyGemm<Element, 128>::Kernel;
  using Mainloop = typename Large::CollectiveMainloop;
  using Pipeline = typename Mainloop::MainloopPipeline;
  using State = typename Mainloop::PipelineState;
  using Order = typename Large::MathWarpGroupOrderBarrier;
  static_assert(std::is_same_v<typename Small::CollectiveMainloop::TensorStorage,
                               typename Mainloop::TensorStorage>);
  static_assert(std::is_same_v<typename Medium::CollectiveMainloop::TensorStorage,
                               typename Mainloop::TensorStorage>);
  static constexpr int EpilogueBytes =
      std::max({sizeof(typename Small::CollectiveEpilogue::TensorStorage),
                sizeof(typename Medium::CollectiveEpilogue::TensorStorage),
                sizeof(typename Large::CollectiveEpilogue::TensorStorage)});
  struct SharedStorage {
    typename Mainloop::TensorStorage operands;
    alignas(1024) char epilogue[EpilogueBytes];
    typename Pipeline::SharedStorage pipeline;
    typename Order::SharedStorage order;
    typename Small::CollectiveEpilogue::PipelineStorage small_epilogue;
    typename Medium::CollectiveEpilogue::PipelineStorage medium_epilogue;
    typename Large::CollectiveEpilogue::PipelineStorage large_epilogue;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  struct Params {
    typename Small::Params small;
    typename Medium::Params medium;
    typename Large::Params large;
    int rows;
    int* readiness;
    int groups;
    int done_index;
  };

  template <class Phase>
  CUTLASS_DEVICE static auto epilogue_pipeline(
      typename Phase::Params const& params,
      typename Phase::CollectiveEpilogue::PipelineStorage& storage, int group) {
    using Epi = typename Phase::CollectiveEpilogue;
    using Load = typename Epi::LoadPipeline;
    typename Load::Params config;
    config.role = group ? Load::ThreadCategory::Consumer : Load::ThreadCategory::NonParticipant;
    config.dst_blockid = 0;
    config.producer_arv_count = 32;
    config.consumer_arv_count = 128;
    config.transaction_bytes = params.epilogue.tma_transaction_bytes;
    return Load(storage, config);
  }

  struct WorkItem {
    int phase;
    int row;
    int column;
    bool valid;
  };

  struct WorkStream {
    typename Large::TileScheduler scheduler;
    typename Large::TileScheduler::WorkTileInfo tile;
    int rows;
    bool first = true;
    bool medium = false;

    CUTLASS_DEVICE explicit WorkStream(Params const& params)
        : scheduler(params.large.scheduler),
          tile(scheduler.get_current_work()),
          rows(params.rows) {}

    CUTLASS_DEVICE WorkItem pop() {
      if (!tile.is_valid()) return {0, 0, 0, false};
      const bool split = tile.M_idx == 0 && (first || rows <= 128);
      WorkItem item{split ? (medium ? 1 : 0) : 2, tile.M_idx, tile.N_idx, true};
      if (split && !medium && rows > 32) {
        medium = true;
      } else {
        medium = false;
        first = false;
        scheduler.advance_to_next_work();
        tile = scheduler.get_current_work();
      }
      return item;
    }
  };

  template <class Phase>
  CUTLASS_DEVICE static void produce(typename Phase::Params const& params, WorkItem item,
                                     SharedStorage& storage, typename Pipeline::Params config,
                                     State& state, int k_tiles) {
    using Load = typename Phase::CollectiveMainloop;
    config.transaction_bytes = params.mainloop.tma_transaction_bytes;
    Pipeline pipeline(storage.pipeline, config, ClusterShape{}, cute::false_type{});
    auto inputs = Load{}.load_init(params.problem_shape, params.mainloop);
    auto block = make_coord(item.phase == 2 ? item.row : 0, item.column, _, 0);
    auto tile = cute::make_coord_iterator(shape<3>(get<0>(inputs)));
    Load{}.load(params.mainloop, pipeline, state, inputs, block, tile, k_tiles,
                cutlass::canonical_lane_idx(), 0, storage.operands);
    state.advance(k_tiles);
  }

  template <class Phase, class EpiloguePipeline>
  CUTLASS_DEVICE static void consume(typename Phase::Params const& params, WorkItem item,
                                     SharedStorage& storage, Pipeline pipeline, State& state,
                                     Order& order, EpiloguePipeline epi_load, int k_tiles) {
    using Load = typename Phase::CollectiveMainloop;
    using Epi = typename Phase::CollectiveEpilogue;
    typename Phase::TiledMma mma;
    auto block = make_coord(item.phase == 2 ? item.row : 0, item.column, _, 0);
    auto tile_shape = typename Phase::TileShape{};
    auto accum = partition_fragment_C(mma, take<0, 2>(tile_shape));
    const int thread = int(threadIdx.x) % 128;
    order.wait();
    Load{}.mma(pipeline, state, accum, k_tiles, thread, storage.operands, params.mainloop, block);
    order.arrive();
    state.advance(k_tiles * 2);

    order.wait();
    auto& tensors = *reinterpret_cast<typename Epi::TensorStorage*>(storage.epilogue);
    Epi epilogue(params.epilogue, tensors);
    using Store = typename Epi::StorePipeline;
    typename Store::Params store_config;
    store_config.always_wait = true;
    Store store(store_config);
    auto store_state = cutlass::make_producer_start_state<Store>();
    typename Epi::LoadPipelineState load_state;
    auto next = epilogue.store(epi_load, load_state, store, store_state, params.problem_shape,
                               tile_shape, block, accum, mma, thread, tensors);
    epilogue.store_tail(epi_load, get<0>(next), store, get<1>(next));
    order.arrive();
  }

  CUTLASS_DEVICE void operator()(Params const& params, char* buffer) {
    auto& storage = *reinterpret_cast<SharedStorage*>(buffer);
    const int group = int(threadIdx.x) / 128;
    const int warp = cutlass::canonical_warp_idx_sync();
    typename Pipeline::Params config;
    config.role = group ? Pipeline::ThreadCategory::Consumer : Pipeline::ThreadCategory::Producer;
    config.is_leader = int(threadIdx.x) == 0;
    config.num_consumers = 128;
    config.num_producers = 1;
    config.transaction_bytes = params.large.mainloop.tma_transaction_bytes;
    Pipeline pipeline(storage.pipeline, config, ClusterShape{});
    typename Order::Params order_config;
    order_config.group_id = group - 1;
    order_config.group_size = 128;
    Order order(storage.order, order_config);
    auto small_epi = epilogue_pipeline<Small>(params.small, storage.small_epilogue, group);
    auto medium_epi = epilogue_pipeline<Medium>(params.medium, storage.medium_epilogue, group);
    auto large_epi = epilogue_pipeline<Large>(params.large, storage.large_epilogue, group);
    if (warp == 0 && cute::elect_one_sync()) {
      Small::CollectiveMainloop::prefetch_tma_descriptors(params.small.mainloop);
      Medium::CollectiveMainloop::prefetch_tma_descriptors(params.medium.mainloop);
      Large::CollectiveMainloop::prefetch_tma_descriptors(params.large.mainloop);
      Small::CollectiveEpilogue::prefetch_tma_descriptors(params.small.epilogue);
      Medium::CollectiveEpilogue::prefetch_tma_descriptors(params.medium.epilogue);
      Large::CollectiveEpilogue::prefetch_tma_descriptors(params.large.epilogue);
    }
    __syncthreads();
    const int k_tiles = (get<2>(params.large.problem_shape) + 63) / 64;
    if (group == 0) {
      cutlass::arch::warpgroup_reg_dealloc<Large::LoadRegisterRequirement>();
      if (warp == 0) {
        auto state = cutlass::make_producer_start_state<Pipeline>();
        WorkStream work(params);
        // Pair startup pieces on one CTA, then consume full tiles without
        // draining the ring. Stage ownership survives every geometry change.
        for (auto item = work.pop(); item.valid; item = work.pop()) {
          if (item.phase == 0)
            produce<Small>(params.small, item, storage, config, state, k_tiles);
          else if (item.phase == 1)
            produce<Medium>(params.medium, item, storage, config, state, k_tiles);
          else
            produce<Large>(params.large, item, storage, config, state, k_tiles);
        }
        if (cute::elect_one_sync()) pipeline.producer_tail(state);
      }
    } else {
      cutlass::arch::warpgroup_reg_alloc<Large::MmaRegisterRequirement>();
      State state;
      state.advance(k_tiles * (group - 1));
      WorkStream work(params);
      if (group == 2) work.pop();
      for (auto item = work.pop(); item.valid; item = work.pop()) {
        if (item.phase == 0)
          consume<Small>(params.small, item, storage, pipeline, state, order, small_epi, k_tiles);
        else if (item.phase == 1)
          consume<Medium>(params.medium, item, storage, pipeline, state, order, medium_epi,
                          k_tiles);
        else
          consume<Large>(params.large, item, storage, pipeline, state, order, large_epi, k_tiles);
        work.pop();
      }
    }
    __syncthreads();
    masked_gemm::FinishReady(params.readiness, params.groups, params.done_index);
  }
};

template <typename Element>
using SelectedReadyKernel = std::conditional_t<FLASHINFER_READY_STARTUP_RAMP,
                                               GeometricReadyKernel<Element>, ReadyKernel<Element>>;

template <typename Element>
cudaError_t PrepareReady(int* occupancy) {
  using Kernel = SelectedReadyKernel<Element>;
  auto status =
      masked_gemm::PrepareKernel(cutlass::device_kernel<Kernel>, Kernel::SharedStorageSize);
  if (status != cudaSuccess) return status;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(occupancy, cutlass::device_kernel<Kernel>,
                                                       Kernel::MaxThreadsPerBlock,
                                                       Kernel::SharedStorageSize);
}

template <typename Element, typename Producers, typename Kernel = SelectedReadyKernel<Element>>
cudaError_t PlanReady(cudaDeviceProp const& device, Producers const& producers, int* math_registers,
                      bool* concurrent) {
  cudaFuncAttributes consumer;
  auto status = cudaFuncGetAttributes(&consumer, cutlass::device_kernel<Kernel>);
  if (status != cudaSuccess) return status;
  cudaOccDeviceProp hardware(device);
  cudaOccDeviceState state;
  state.carveoutConfig = SHAREDMEM_CARVEOUT_MAX_SHARED;
  cudaOccFuncAttributes consumer_attributes(consumer);
  cudaOccResult consumer_occupancy;
  int granularity, partitions;
  if (cudaOccRegAllocationGranularity(&granularity, &hardware) != CUDA_OCC_SUCCESS ||
      cudaOccSubPartitionsPerMultiprocessor(&partitions, &hardware) != CUDA_OCC_SUCCESS ||
      cudaOccMaxActiveBlocksPerMultiprocessor(&consumer_occupancy, &hardware, &consumer_attributes,
                                              &state, Kernel::MaxThreadsPerBlock,
                                              Kernel::SharedStorageSize) != CUDA_OCC_SUCCESS) {
    return cudaErrorInvalidValue;
  }
  int producer_regs = 0, producer_warps = 0, producer_shared = 0;
  *concurrent = consumer_occupancy.activeBlocksPerMultiprocessor > 0;
  for (auto const& producer : producers) {
    cudaOccFuncAttributes attributes;
    attributes.maxThreadsPerBlock = device.maxThreadsPerBlock;
    attributes.numRegs = producer[0];
    attributes.shmemLimitConfig = FUNC_SHMEM_LIMIT_OPTIN;
    attributes.maxDynamicSharedSizeBytes = producer[1];
    attributes.numBlockBarriers = 1;
    cudaOccResult occupancy;
    int const threads = producer[2];
    if (threads <= 0 ||
        cudaOccMaxActiveBlocksPerMultiprocessor(&occupancy, &hardware, &attributes, &state, threads,
                                                producer[1]) != CUDA_OCC_SUCCESS) {
      return cudaErrorInvalidValue;
    }
    int const warps = __occDivideRoundUp(threads, device.warpSize);
    int const allocated_warps = __occRoundUp(warps, partitions);
    producer_regs += occupancy.allocatedRegistersPerBlock / warps * allocated_warps;
    producer_warps += allocated_warps;
    producer_shared += occupancy.allocatedSharedMemPerBlock;
    *concurrent &= occupancy.activeBlocksPerMultiprocessor > 0;
  }
  const int consumer_warps = Kernel::MaxThreadsPerBlock / device.warpSize;
  const bool shared_capacity =
      consumer_occupancy.allocatedSharedMemPerBlock + producer_shared <=
          device.sharedMemPerMultiprocessor &&
      (consumer_warps + producer_warps) * device.warpSize <= device.maxThreadsPerMultiProcessor &&
      consumer_occupancy.blockLimitBlocks >= int(producers.size()) + 1;
  if (!shared_capacity) {
    // Register redistribution cannot repair a shared-memory or thread limit.
    *math_registers = 0;
    *concurrent = false;
    return cudaSuccess;
  }
  const int uniform_registers = (device.regsPerMultiprocessor - producer_regs) / consumer_warps /
                                granularity * granularity / device.warpSize;
  constexpr int consumer_threads = Kernel::MaxThreadsPerBlock;
  constexpr int load_threads = Kernel::NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
  constexpr int math_threads = consumer_threads - load_threads;
  const int budget =
      (uniform_registers * consumer_threads - int(Kernel::LoadRegisterRequirement) * load_threads) /
      math_threads;
  // Allocate registers for every producer that must progress during readiness waits.
  *math_registers =
      budget >= 24 ? std::min(DefaultRegisterAllocation::MathRegisters, budget / 8 * 8) : 0;
  *concurrent &=
      consumer_occupancy.allocatedRegistersPerBlock + producer_regs <= device.regsPerMultiprocessor;
  return cudaSuccess;
}

template <typename Element>
constexpr size_t ReadyWorkspaceBytes(int) {
  return 0;
}

template <typename Element>
cudaError_t RunReady(Element* a, Element* b, Element* out, int m, int n, int k, int lda, int ldb,
                     int ldd, int* readiness, int group_rows, int done_index, void* workspace,
                     int sms, int occupancy, int available_sms, cudaStream_t stream,
                     int tile_swizzle = 1, bool raster_m = false, int* task_state = nullptr) {
  if (task_state && (tile_swizzle != 1 || raster_m || FLASHINFER_READY_STARTUP_RAMP))
    return cudaErrorInvalidValue;
  using Kernel = SelectedReadyKernel<Element>;
  auto prepare = [&](auto kernel, int offset, int rows) {
    using Phase = decltype(kernel);
    typename Phase::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape = {std::max(rows, 1), n, k, 1};
    args.mainloop = {{a + int64_t(offset) * lda, {lda, _1{}, 0}, b, {ldb, _1{}, 0}},
                     readiness,
                     group_rows,
                     offset,
                     m};
    args.epilogue.thread = {1.0f, 0.0f};
    args.epilogue.ptr_D = out + int64_t(offset) * ldd;
    args.epilogue.dD = {ldd, _1{}, 0};
    args.hw_info.sm_count = available_sms;
    using RasterOrderOptions = typename Phase::TileScheduler::RasterOrderOptions;
    args.scheduler.raster_order =
        raster_m ? RasterOrderOptions::AlongM : RasterOrderOptions::AlongN;
    args.scheduler.max_swizzle_size = tile_swizzle;
    args.scheduler.task_state = task_state;
    args.scheduler.readiness = readiness;
    args.scheduler.group_rows = group_rows;
    return Phase::to_underlying_arguments(args, workspace);
  };
#if FLASHINFER_READY_STARTUP_RAMP
  static_assert(ReadyTile::kM == 128);
  const int small_rows = std::min(m, 32);
  const int medium_rows = std::min(m - small_rows, 96);
  typename Kernel::Params params{prepare(typename Kernel::Small{}, 0, small_rows),
                                 prepare(typename Kernel::Medium{}, small_rows, medium_rows),
                                 prepare(typename Kernel::Large{}, 0, m),
                                 m,
                                 readiness,
                                 (m + group_rows - 1) / group_rows,
                                 done_index};
  const int tiles_n = (n + 63) / 64;
  const int tiles_m = (m + 127) / 128;
  // The resident grid owns steady tiles; startup only subdivides its first tile.
  dim3 grid(1, std::min(available_sms, tiles_m * tiles_n), 1);
#else
  typename Kernel::Params params{prepare(typename ReadyGemm<Element>::Kernel{}, 0, m), readiness,
                                 (m + group_rows - 1) / group_rows, done_index};
  auto grid = Kernel::get_grid_shape(params);
#endif
  cutlass::device_kernel<Kernel>
      <<<grid, Kernel::MaxThreadsPerBlock, Kernel::SharedStorageSize, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace flashinfer::ready_tma_gemm
#endif
