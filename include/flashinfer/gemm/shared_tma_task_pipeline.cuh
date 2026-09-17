/* Copyright (c) 2026 by FlashInfer contributors. SPDX-License-Identifier:
 * Apache-2.0 */
#pragma once

#include "ready_tma_gemm.cuh"

namespace flashinfer::ready_tma_gemm {

// Common TMA compute substrate; operator epilogues provide dependency
// publication.
template <typename Element, int Operators>
struct SharedTmaTaskPipeline {
  using Reused = GeometricReadyKernel<Element>;
  using Phase = typename Reused::Large;
  using Mainloop = typename Reused::Mainloop;
  using Pipeline = typename Reused::Pipeline;
  using State = typename Reused::State;
  using Order = typename Reused::Order;
  using Scheduler = EligibleReadyScheduler;
  static_assert(Operators > 0);
  static constexpr int Threads = 288;
  static constexpr int LoaderThread = 256;

  struct Params {
    typename Phase::Params operations[Operators];
    int* completed_tiles[Operators];
    int* output_readiness[Operators];
    Element* compact_output[Operators];
    int64_t compact_stride[Operators];
    Element* local_output[Operators];
    Element* peer_output[Operators];
    int64_t output_stride[Operators];
    masked_gemm::PushPublication const* publication[Operators];
    masked_gemm::PushPublication const* tile_publication[Operators];
  };

  struct Instruction {
    int operation;
    int row;
    int column;
    int k_tiles;
    int k_start;
  };

  struct SharedStorage {
    typename Reused::SharedStorage gemm;
    Instruction instructions[2];
    int published[2];
    int released[2];
  };

  template <class Accumulator>
  CUTLASS_DEVICE static void store_paired(Accumulator const& accum, int thread, Instruction item,
                                          Params const& params) {
    typename Phase::TiledMma mma;
    auto coordinates = mma.get_slice(thread).partition_C(
        make_identity_tensor(take<0, 2>(typename Phase::TileShape{})));
    static_assert(decltype(size(accum))::value % 2 == 0);
    int op = item.operation;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum); i += 2) {
      int row = item.row * 128 + get<0>(coordinates(i));
      int column = item.column * 64 + get<1>(coordinates(i));
      Element gate_value(accum(i)), up_value(accum(i + 1));
      float gate = float(gate_value), up = float(up_value);
      constexpr float beta = M_SQRT2 * M_2_SQRTPI * 0.5f;
      float cube = gate * gate * gate;
      Element gelu(0.5f * gate * (1.0f + ::tanhf(beta * (gate + 0.044715f * cube))));
      Element activated(float(gelu) * up);
      bool valid = row < get<0>(params.operations[op].problem_shape) &&
                   column < get<1>(params.operations[op].problem_shape);
      cutlass::arch::global_store<Element, sizeof(Element)>(
          activated, params.compact_output[op] + row * params.compact_stride[op] + column / 2,
          valid);
    }
  }

  template <class Accumulator>
  CUTLASS_DEVICE static void store_output(Accumulator const& accum, int thread, Instruction item,
                                          Params const& params) {
    typename Phase::TiledMma mma;
    auto coordinates = mma.get_slice(thread).partition_C(
        make_identity_tensor(take<0, 2>(typename Phase::TileShape{})));
    int op = item.operation;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum); i += 2) {
      int row = item.row * 128 + get<0>(coordinates(i));
      int column = item.column * 64 + get<1>(coordinates(i));
      cutlass::Array<Element, 2> value;
      value[0] = Element(accum(i));
      value[1] = Element(accum(i + 1));
      bool valid = row < get<0>(params.operations[op].problem_shape) &&
                   column < get<1>(params.operations[op].problem_shape);
      int64_t offset = row * params.output_stride[op] + column;
      cutlass::arch::global_store<decltype(value), sizeof(value)>(
          value, params.local_output[op] + offset, valid);
      if (params.peer_output[op])
        cutlass::arch::global_store<decltype(value), sizeof(value)>(
            value, params.peer_output[op] + offset, valid);
    }
    if (params.peer_output[op]) __threadfence_system();
  }

  CUTLASS_DEVICE static void consume_direct(Params const& params, Instruction item,
                                            SharedStorage& storage, Pipeline pipeline, State state,
                                            Order& order, int thread) {
    typename Phase::TiledMma mma;
    auto accum = partition_fragment_C(mma, take<0, 2>(typename Phase::TileShape{}));
    auto block = make_coord(item.row, item.column, _, 0);
    order.wait();
    Mainloop{}.mma(pipeline, state, accum, item.k_tiles, thread, storage.gemm.operands,
                   params.operations[item.operation].mainloop, block);
    order.arrive();
    order.wait();
    if (params.local_output[item.operation])
      store_output(accum, thread, item, params);
    else
      store_paired(accum, thread, item, params);
    order.arrive();
  }

  CUTLASS_DEVICE static Instruction select(Params const& params, int& preferred, int k_start) {
    for (;;) {
      bool finished = true;
      for (int offset = 0; offset < Operators; ++offset) {
        int op = (preferred + offset) % Operators;
        auto claim = Scheduler::try_claim_ready_tile(params.operations[op].scheduler);
        if (claim.status == Scheduler::ClaimStatus::Ready) {
          preferred = (op + 1) % Operators;
          int k_tiles = (get<2>(params.operations[op].problem_shape) + 63) / 64;
          return {op, claim.tile.M_idx, claim.tile.N_idx, k_tiles, k_start};
        }
        finished &= claim.status == Scheduler::ClaimStatus::AllIssued;
      }
      if (finished) return {-1, 0, 0, 0, k_start};
      __nanosleep(64);
    }
  }

  CUTLASS_DEVICE static void publish(SharedStorage& storage, int sequence,
                                     Instruction instruction) {
    int slot = sequence % 2;
    int generation = sequence / 2;
    cuda::atomic_ref<int, cuda::thread_scope_block> released(storage.released[slot]);
    while (released.load(cuda::memory_order_acquire) != generation) __nanosleep(64);
    storage.instructions[slot] = instruction;
    cuda::atomic_ref<int, cuda::thread_scope_block> ready(storage.published[slot]);
    ready.store(generation + 1, cuda::memory_order_release);
  }

  CUTLASS_DEVICE static Instruction receive(SharedStorage& storage, int sequence, int group) {
    int slot = sequence % 2;
    int generation = sequence / 2 + 1;
    cuda::atomic_ref<int, cuda::thread_scope_block> ready(storage.published[slot]);
    while (ready.load(cuda::memory_order_acquire) != generation) __nanosleep(64);
    Instruction instruction = storage.instructions[slot];
    // All 128 math threads retain the descriptor before its slot is recycled.
    if (group == 1)
      asm volatile("bar.sync 14, 128;" ::: "memory");
    else
      asm volatile("bar.sync 15, 128;" ::: "memory");
    if (threadIdx.x % 128 == 0) {
      cuda::atomic_ref<int, cuda::thread_scope_block> released(storage.released[slot]);
      released.store(generation, cuda::memory_order_release);
    }
    return instruction;
  }

  CUTLASS_DEVICE void operator()(Params const& params, char* buffer) {
    auto& storage = *reinterpret_cast<SharedStorage*>(buffer);
    int group = threadIdx.x < LoaderThread ? int(threadIdx.x) / 128 + 1 : 0;
    int warp = cutlass::canonical_warp_idx_sync();
    if (threadIdx.x < 2) {
      storage.published[threadIdx.x] = 0;
      storage.released[threadIdx.x] = 0;
    }
    typename Pipeline::Params config;
    config.role = group ? Pipeline::ThreadCategory::Consumer : Pipeline::ThreadCategory::Producer;
    config.is_leader = threadIdx.x == LoaderThread;
    config.num_consumers = 128;
    config.num_producers = 1;
    config.transaction_bytes = params.operations[0].mainloop.tma_transaction_bytes;
    Pipeline pipeline(storage.gemm.pipeline, config, ClusterShape{});
    typename Order::Params order_config;
    order_config.group_id = group - 1;
    order_config.group_size = 128;
    Order order(storage.gemm.order, order_config);
    auto epilogue = Reused::template epilogue_pipeline<Phase>(params.operations[0],
                                                              storage.gemm.large_epilogue, group);
    if (warp == LoaderThread / 32 && cute::elect_one_sync()) {
      for (int op = 0; op < Operators; ++op) {
        Phase::CollectiveMainloop::prefetch_tma_descriptors(params.operations[op].mainloop);
        Phase::CollectiveEpilogue::prefetch_tma_descriptors(params.operations[op].epilogue);
      }
    }
    __syncthreads();
    if (group == 0) {
      if (warp == LoaderThread / 32) {
        auto state = cutlass::make_producer_start_state<Pipeline>();
        int preferred = int(blockIdx.x) % Operators;
        int k_start = 0;
        for (int sequence = 0;; ++sequence) {
          if (threadIdx.x == LoaderThread)
            publish(storage, sequence, select(params, preferred, k_start));
          __syncwarp();
          auto instruction = storage.instructions[sequence % 2];
          __syncwarp();
          if (instruction.operation < 0) {
            if (threadIdx.x == LoaderThread) publish(storage, sequence + 1, instruction);
            break;
          }
          typename Reused::WorkItem item{2, instruction.row, instruction.column, true};
          Reused::template produce<Phase>(params.operations[instruction.operation], item,
                                          storage.gemm, config, state, instruction.k_tiles);
          k_start += instruction.k_tiles;
        }
        if (cute::elect_one_sync()) pipeline.producer_tail(state);
      }
    } else {
      for (int sequence = group - 1;; sequence += 2) {
        auto instruction = receive(storage, sequence, group);
        if (instruction.operation < 0) break;
        State state;
        state.advance(instruction.k_start);
        typename Reused::WorkItem item{2, instruction.row, instruction.column, true};
        if (params.compact_output[instruction.operation] ||
            params.local_output[instruction.operation])
          consume_direct(params, instruction, storage, pipeline, state, order,
                         int(threadIdx.x) % 128);
        else
          Reused::template consume<Phase>(params.operations[instruction.operation], item,
                                          storage.gemm, pipeline, state, order, epilogue,
                                          instruction.k_tiles);
        int op = instruction.operation;
        if (params.output_readiness[op] || params.publication[op]) {
          if (group == 1)
            asm volatile("bar.sync 14, 128;" ::: "memory");
          else
            asm volatile("bar.sync 15, 128;" ::: "memory");
          if (threadIdx.x % 128 == 0) {
            auto const& scheduler = params.operations[op].scheduler;
            if (params.tile_publication[op])
              params.tile_publication[op]->publish(instruction.row * scheduler.tiles_n +
                                                   instruction.column);
            cuda::atomic_ref<int, cuda::thread_scope_device> completed(
                params.completed_tiles[op][instruction.row]);
            if (completed.fetch_add(1, cuda::memory_order_acq_rel) == scheduler.tiles_n - 1) {
              if (params.publication[op]) {
                int first = instruction.row * 4;
                int end = min(first + 4, (scheduler.rows + 31) / 32);
                for (int row_group = first; row_group < end; ++row_group)
                  params.publication[op]->publish(row_group);
              }
              if (params.output_readiness[op]) {
                cuda::atomic_ref<int, cuda::thread_scope_device> ready(
                    params.output_readiness[op][instruction.row]);
                ready.store(min(128, scheduler.rows - instruction.row * 128),
                            cuda::memory_order_release);
              }
            }
          }
        }
      }
    }
    __syncthreads();
  }
};

}  // namespace flashinfer::ready_tma_gemm
