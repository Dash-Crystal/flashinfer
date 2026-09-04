/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_NAMED_BARRIERS_CUH_
#define FLASHINFER_ATTENTION_HOPPER_NAMED_BARRIERS_CUH_

#include <cuda_runtime.h>

#include <type_traits>

#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"

namespace flashinfer {

// Enumerates the reserved named barriers to avoid potential conflicts

enum class NamedBarriers {
  kQueryEmpty = 0,
  kValueEmpty = 1,
  kWarpSchedulerWG1 = 2,
  kWarpSchedulerWG2 = 3,
  kWarpSchedulerWG3 = 4,
  kPrefetchIndices = 5,
  kProducerWG = 6
};

// Number of producer warp groups of a kernel-traits type: 1 unless the traits
// define NUM_PRODUCER_WGS (the mixed-page traits use 2 for the compressed and
// dynamic modules, [24b]).  The consumer warp groups are the ones from this
// index on; every thread-count and barrier-index derivation below reads it, so
// stock traits keep their constants textually.
template <typename Ktraits, typename = void>
struct producer_warp_groups : std::integral_constant<int, 1> {};
template <typename Ktraits>
struct producer_warp_groups<Ktraits, std::void_t<decltype(Ktraits::NUM_PRODUCER_WGS)>>
    : std::integral_constant<int, Ktraits::NUM_PRODUCER_WGS> {};
template <typename Ktraits>
inline constexpr int producer_warp_groups_v = producer_warp_groups<Ktraits>::value;

// Ping-pong barrier of consumer warp group `warp_group_idx`: hardware ids
// kWarpSchedulerWG1.. indexed by the warp group's position among the consumers
// (kFirstConsumerWG = number of producer warp groups), so the ids stay 2, 3 (, 4)
// whatever the producer layout.
template <int kFirstConsumerWG = 1>
__device__ __forceinline__ int get_warp_group_barrier_idx(int warp_group_idx) {
  return static_cast<int>(NamedBarriers::kWarpSchedulerWG1) + warp_group_idx - kFirstConsumerWG;
}

template <int num_consumer_warp_groups, int kFirstConsumerWG = 1>
__device__ __forceinline__ int get_next_consumer_warp_group_idx() {
  static_assert(num_consumer_warp_groups == 2 || num_consumer_warp_groups == 3);
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  if constexpr (num_consumer_warp_groups == 2) {
    // F -> F+1, F+1 -> F  (F = 1: 1 -> 2, 2 -> 1)
    return 2 * kFirstConsumerWG + 1 - warp_group_idx;
  } else {
    // num_consumer_warp_groups == 3
    // F -> F+1, F+1 -> F+2, F+2 -> F
    return ((warp_group_idx - kFirstConsumerWG + 1) % 3) + kFirstConsumerWG;
  }
}

template <int num_consumer_warp_groups, int kFirstConsumerWG = 1>
__device__ __forceinline__ int get_prev_consumer_warp_group_idx() {
  static_assert(num_consumer_warp_groups == 2 || num_consumer_warp_groups == 3);
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  if constexpr (num_consumer_warp_groups == 2) {
    // F -> F+1, F+1 -> F
    return 2 * kFirstConsumerWG + 1 - warp_group_idx;
  } else {
    // num_consumer_warp_groups == 3
    // F -> F+2, F+1 -> F, F+2 -> F+1
    return ((warp_group_idx - kFirstConsumerWG + 2) % 3) + kFirstConsumerWG;
  }
}

template <typename Ktraits, bool UseSchedulerBarrier>
struct WarpScheduler {
  constexpr static int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  constexpr static int kFirstConsumerWG = producer_warp_groups_v<Ktraits>;
  static CUTLASS_DEVICE void barrier_sync() {
    if constexpr (UseSchedulerBarrier) {
      cutlass::arch::NamedBarrier::sync(
          NUM_MMA_THREADS,
          get_warp_group_barrier_idx<kFirstConsumerWG>(cutlass::canonical_warp_group_idx()));
    }
  }

  static CUTLASS_DEVICE void barrier_arrive() {
    if constexpr (!UseSchedulerBarrier) {
      return;
    }
    static_assert(NUM_MMA_THREADS == 2 * cutlass::NumThreadsPerWarpGroup ||
                  NUM_MMA_THREADS == 3 * cutlass::NumThreadsPerWarpGroup);
    if constexpr (NUM_MMA_THREADS == 2 * cutlass::NumThreadsPerWarpGroup) {
      cutlass::arch::NamedBarrier::arrive(
          NUM_MMA_THREADS, get_warp_group_barrier_idx<kFirstConsumerWG>(
                               get_next_consumer_warp_group_idx<2, kFirstConsumerWG>()));
    } else {
      cutlass::arch::NamedBarrier::arrive(
          NUM_MMA_THREADS, get_warp_group_barrier_idx<kFirstConsumerWG>(
                               get_next_consumer_warp_group_idx<3, kFirstConsumerWG>()));
      cutlass::arch::NamedBarrier::arrive(
          NUM_MMA_THREADS, get_warp_group_barrier_idx<kFirstConsumerWG>(
                               get_prev_consumer_warp_group_idx<3, kFirstConsumerWG>()));
    }
  }

  static CUTLASS_DEVICE void mma_init() {
    // Tell producer (warp 0) that smem_q is ready
    cutlass::arch::NamedBarrier::arrive(NUM_MMA_THREADS + Ktraits::NUM_PRODUCER_THREADS,
                                        /*id=*/static_cast<int>(NamedBarriers::kQueryEmpty));
    if constexpr (!UseSchedulerBarrier) {
      return;
    }
    static_assert(NUM_MMA_THREADS == 2 * cutlass::NumThreadsPerWarpGroup ||
                  NUM_MMA_THREADS == 3 * cutlass::NumThreadsPerWarpGroup);
    // Every consumer warp group but the first pre-arrives on the first one's
    // barrier (the first issues its first gemm without waiting).
    if (cutlass::canonical_warp_group_idx() > kFirstConsumerWG) {
      cutlass::arch::NamedBarrier::arrive(
          NUM_MMA_THREADS, /*id=*/static_cast<int>(NamedBarriers::kWarpSchedulerWG1));
    }
    if constexpr (NUM_MMA_THREADS == 3 * cutlass::NumThreadsPerWarpGroup) {
      if (cutlass::canonical_warp_group_idx() > kFirstConsumerWG + 1) {
        cutlass::arch::NamedBarrier::arrive(
            NUM_MMA_THREADS, /*id=*/static_cast<int>(NamedBarriers::kWarpSchedulerWG2));
      }
    }
  }

};  // struct WarpScheduler

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_NAMED_BARRIERS_CUH_
