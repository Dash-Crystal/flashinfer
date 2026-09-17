/*
 * Copyright (c) 2026 by FlashInfer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "paired_geglu.cuh"

namespace flashinfer::ffn_task_pipeline {

using namespace masked_gemm;
using Element = cutlass::bfloat16_t;

struct Task {
  int kind;  // 0: no eligible work, 1: gate/up, 2: down, 3: all work issued
  int m;
  int n;
};

struct TaskSwizzle : Swizzle {
  CUTLASS_DEVICE static cutlass::gemm::GemmCoord get_tile_offset(int) {
    extern __shared__ char ffn_storage[];
    auto const& task = *reinterpret_cast<Task const*>(ffn_storage);
    return {task.m, task.n, 0};
  }
};

using Up = cutlass::gemm::kernel::Gemm<typename LocalGemm<Element>::Mma, paired_geglu::Epilogue,
                                       TaskSwizzle, false>;
template <bool Push>
using Down = cutlass::gemm::kernel::Gemm<
    typename LocalGemm<Element>::Mma,
    std::conditional_t<Push, PeerEpilogue<typename LocalGemm<Element>::Epilogue>,
                       typename LocalGemm<Element>::Epilogue>,
    TaskSwizzle, false>;
static_assert(Up::kThreadCount == Down<false>::kThreadCount);
static_assert(Up::kThreadCount == Down<true>::kThreadCount);
static_assert(sizeof(Down<true>::SharedStorage) == sizeof(Down<false>::SharedStorage));
constexpr int kPrefixBytes = 128;
constexpr int kGemmSharedBytes = sizeof(Up::SharedStorage) > sizeof(Down<false>::SharedStorage)
                                     ? sizeof(Up::SharedStorage)
                                     : sizeof(Down<false>::SharedStorage);
constexpr int kSharedBytes = kPrefixBytes + kGemmSharedBytes;

template <bool Push>
struct Params {
  Up::Params up;
  typename Down<Push>::Params down;
  int* state;  // up claims, up completion counts, down claims, final-CTA count
  int* input_readiness;
  int input_group_rows;
  int rows;
  int tiles_m;
  int up_tiles_n;
  int down_tiles_n;
  int* publication;
  int publication_groups;
  int* retire_readiness;
  int retire_group_rows;
  int retire_rows;
  int retire_workers;
  tp_region::DeviceRegion const* native_region;
  int* admission_state;
  int total_workers;
  int* peer_publication;
  PushPublication const* publication_epoch;
};

template <bool Push>
CUTLASS_DEVICE bool RetireForConsumer(Params<Push> const& params, int worker_ticket) {
  if (params.admission_state) {
    if (worker_ticket == 0 || worker_ticket > params.retire_workers) return false;
  } else if (int(blockIdx.x) >= params.retire_workers) {
    return false;
  }
  cuda::atomic_ref<int, cuda::thread_scope_device> latched(params.state[3 * params.tiles_m + 1]);
  if (latched.load(cuda::memory_order_acquire)) return true;
  int groups = (params.retire_rows + params.retire_group_rows - 1) / params.retire_group_rows;
  for (int group = 0; group < groups; ++group) {
    int expected = min(params.retire_group_rows, params.rows - group * params.retire_group_rows);
    cuda::atomic_ref<int, cuda::thread_scope_device> ready(params.retire_readiness[group]);
    if (ready.load(cuda::memory_order_acquire) < expected) return false;
  }
  latched.store(1, cuda::memory_order_release);
  return true;
}

CUTLASS_DEVICE int Claim(int* address, int limit) {
  cuda::atomic_ref<int, cuda::thread_scope_device> next(*address);
  int value = next.load(cuda::memory_order_relaxed);
  while (value < limit) {
    if (next.compare_exchange_weak(value, value + 1, cuda::memory_order_relaxed)) return value;
  }
  return -1;
}

template <bool Push>
CUTLASS_DEVICE bool InputReady(Params<Push> const& params, int tile_m) {
  if (!params.input_readiness) return true;
  int first = tile_m * Tile::kM;
  int end = min(first + Tile::kM, params.rows);
  for (int group = first / params.input_group_rows; group <= (end - 1) / params.input_group_rows;
       ++group) {
    int expected = min(params.input_group_rows, params.rows - group * params.input_group_rows);
    cuda::atomic_ref<int, cuda::thread_scope_device> ready(params.input_readiness[group]);
    if (ready.load(cuda::memory_order_acquire) < expected) return false;
  }
  return true;
}

template <bool Push>
CUTLASS_DEVICE Task ClaimReadyUp(Params<Push> const& params) {
  for (int m = 0; m < params.tiles_m; ++m) {
    cuda::atomic_ref<int, cuda::thread_scope_device> next(params.state[m]);
    if (next.load(cuda::memory_order_relaxed) >= params.up_tiles_n) continue;
    if (InputReady(params, m)) {
      int n = Claim(params.state + m, params.up_tiles_n);
      if (n >= 0) return {1, m, n};
    }
  }
  return {0, 0, 0};
}

template <bool Push>
CUTLASS_DEVICE Task NextTask(Params<Push> const& params, bool prefer_up) {
  if constexpr (Push) {
    if (prefer_up) {
      Task task = ClaimReadyUp(params);
      if (task.kind) return task;
    }
  }
  bool all_issued = true;
  for (int m = 0; m < params.tiles_m; ++m) {
    cuda::atomic_ref<int, cuda::thread_scope_device> next(params.state[2 * params.tiles_m + m]);
    if (next.load(cuda::memory_order_relaxed) >= params.down_tiles_n) continue;
    all_issued = false;
    cuda::atomic_ref<int, cuda::thread_scope_device> done(params.state[params.tiles_m + m]);
    if (done.load(cuda::memory_order_acquire) == params.up_tiles_n) {
      int n = Claim(params.state + 2 * params.tiles_m + m, params.down_tiles_n);
      if (n >= 0) return {2, m, n};
    }
  }
  if (all_issued) return {3, 0, 0};
  return ClaimReadyUp(params);
}

template <bool Push>
__global__ __launch_bounds__(Up::kThreadCount) void FfnTasks(Params<Push> params) {
  extern __shared__ char ffn_storage[];
  auto& task = *reinterpret_cast<Task*>(ffn_storage);
  char* gemm_storage = ffn_storage + kPrefixBytes;
  auto& worker_ticket = *reinterpret_cast<int*>(ffn_storage + sizeof(Task));
  if (threadIdx.x == 0) {
    worker_ticket = int(blockIdx.x);
    if (params.admission_state) {
      cuda::atomic_ref<int, cuda::thread_scope_device> next(params.admission_state[0]);
      worker_ticket = next.fetch_add(1, cuda::memory_order_relaxed);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && params.admission_state && worker_ticket == 0) {
    cuda::atomic_ref<int, cuda::thread_scope_system> started(params.admission_state[1]);
    started.store(1, cuda::memory_order_release);
  }
  bool prefer_up = false;
  while (true) {
    if (threadIdx.x == 0)
      task = RetireForConsumer(params, worker_ticket) ? Task{3, 0, 0} : NextTask(params, prefer_up);
    __syncthreads();
    if (task.kind == 3) break;
    // Cover peer-store draining with a ready local GEMM after each down tile.
    if constexpr (Push) {
      if (threadIdx.x == 0 && task.kind) prefer_up = task.kind == 2;
    }
    if (task.kind == 0) {
      __nanosleep(64);
    } else if (task.kind == 1) {
      Up{}(params.up, *reinterpret_cast<Up::SharedStorage*>(gemm_storage));
      __syncthreads();
      if (threadIdx.x == 0) {
        cuda::atomic_ref<int, cuda::thread_scope_device> done(
            params.state[params.tiles_m + task.m]);
        done.fetch_add(1, cuda::memory_order_acq_rel);
      }
    } else {
      Down<Push>{}(params.down,
                   *reinterpret_cast<typename Down<Push>::SharedStorage*>(gemm_storage));
      if (params.publication) {
        PublishRows<Tile::kM, !Push>(params.publication, params.peer_publication,
                                     params.publication_groups, task.m, params.down_tiles_n,
                                     params.rows, params.native_region,
                                     params.down.problem_size.n(), params.publication_epoch);
      }
    }
    // A tile's shared storage is reused only after all of its threads finish.
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    cuda::atomic_ref<int, cuda::thread_scope_device> finished(params.state[3 * params.tiles_m]);
    if (finished.fetch_add(1, cuda::memory_order_acq_rel) == params.total_workers - 1) {
      for (int i = 0; i < 3 * params.tiles_m; ++i) params.state[i] = 0;
      if (params.input_readiness) {
        int groups = (params.rows + params.input_group_rows - 1) / params.input_group_rows;
        for (int i = 0; i < groups; ++i) params.input_readiness[i] = 0;
      }
      params.state[3 * params.tiles_m + 1] = 0;
      if (params.admission_state) params.admission_state[0] = 0;
      // started stays set until the next externally ordered invocation.
      finished.store(0, cuda::memory_order_relaxed);
    }
  }
}

template <bool Push = false>
inline cudaError_t Prepare(int* registers, int* shared, int* threads, int* occupancy) {
  auto kernel = FfnTasks<Push>;
  auto status = PrepareKernel(kernel, kSharedBytes);
  if (status != cudaSuccess) return status;
  cudaFuncAttributes attrs;
  status = cudaFuncGetAttributes(&attrs, kernel);
  if (status != cudaSuccess) return status;
  *registers = attrs.numRegs;
  *shared = attrs.sharedSizeBytes + kSharedBytes;
  *threads = Up::kThreadCount;
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(occupancy, kernel, Up::kThreadCount,
                                                       kSharedBytes);
}

template <bool Push = false>
inline cudaError_t Run(Element* x, Element* up_weight, Element* down_weight, Element* activation,
                       Element* out, int m, int hidden, int intermediate, int* state,
                       int* readiness, int group_rows, int* publication, int publication_groups,
                       int workers, int* retire_readiness, int retire_group_rows, int retire_rows,
                       int retire_workers, cudaStream_t stream,
                       tp_region::DeviceRegion const* native_region = nullptr,
                       int* admission_state = nullptr, Element* peer_out = nullptr,
                       int* peer_publication = nullptr,
                       PushPublication const* publication_epoch = nullptr) {
  if constexpr (Push) {
    if (!peer_out || !publication || !peer_publication || native_region)
      return cudaErrorInvalidValue;
  }
  cutlass::gemm::GemmCoord up_problem(m, 2 * intermediate, hidden);
  cutlass::gemm::GemmCoord down_problem(m, hidden, intermediate);
  auto tiled = [](cutlass::gemm::GemmCoord problem) {
    return TaskSwizzle::get_tiled_shape(problem, {Tile::kM, Tile::kN, Tile::kK}, 1);
  };
  Up::Params up(up_problem, tiled(up_problem), {x, RowMajor(hidden)},
                {up_weight, ColumnMajor(hidden)}, {activation, RowMajor(2 * intermediate)},
                {activation, RowMajor(2 * intermediate)}, {1.0f, 0.0f});
  typename Down<Push>::Params down(down_problem, tiled(down_problem),
                                   {activation, RowMajor(intermediate)},
                                   {down_weight, ColumnMajor(intermediate)},
                                   {out, RowMajor(hidden)}, {out, RowMajor(hidden)}, {1.0f, 0.0f});
  if constexpr (Push) down.params_D.peer = peer_out;
  Params<Push> params{up,
                      down,
                      state,
                      readiness,
                      group_rows,
                      m,
                      (m + Tile::kM - 1) / Tile::kM,
                      (2 * intermediate + Tile::kN - 1) / Tile::kN,
                      (hidden + Tile::kN - 1) / Tile::kN,
                      publication,
                      publication_groups,
                      retire_readiness,
                      retire_group_rows,
                      retire_rows,
                      retire_workers,
                      native_region,
                      admission_state,
                      workers,
                      peer_publication,
                      publication_epoch};
  FfnTasks<Push><<<workers, Up::kThreadCount, kSharedBytes, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace flashinfer::ffn_task_pipeline
