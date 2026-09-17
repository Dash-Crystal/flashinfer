/* Copyright (c) 2026 by FlashInfer contributors. SPDX-License-Identifier:
 * Apache-2.0 */
#pragma once

#include <cuda/atomic>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>

namespace flashinfer::ready_tma_gemm {
struct EligibleReadySchedulerTag {};

class EligibleReadyScheduler : public cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90 {
  using Base = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90;

 public:
  struct Arguments : Base::Arguments {
    int* task_state = nullptr;
    int* readiness = nullptr;
    int group_rows = 128;
  };
  struct Params : Base::Params {
    int* task_state = nullptr;
    int* readiness = nullptr;
    int group_rows = 128;
    int rows = 0;
    int tile_rows = 128;
    int tiles_m = 0;
    int tiles_n = 0;
  };

  template <class Problem, class Tile, class Cluster>
  static Params to_underlying_arguments(Problem shape, Tile tile, Cluster cluster,
                                        cutlass::KernelHardwareInfo const& hardware,
                                        Arguments const& args, void* workspace = nullptr,
                                        uint32_t epilogue_subtile = 1, uint32_t k_alignment = 1) {
    Params result;
    static_cast<Base::Params&>(result) = Base::to_underlying_arguments(
        shape, tile, cluster, hardware, args, workspace, epilogue_subtile, k_alignment);
    result.task_state = args.task_state;
    result.readiness = args.readiness;
    result.group_rows = args.group_rows;
    result.rows = int(cute::get<0>(shape));
    result.tile_rows = int(cute::get<0>(tile));
    result.tiles_m = (result.rows + result.tile_rows - 1) / result.tile_rows;
    result.tiles_n =
        (int(cute::get<1>(shape)) + int(cute::get<1>(tile)) - 1) / int(cute::get<1>(tile));
    return result;
  }

  CUTLASS_DEVICE explicit EligibleReadyScheduler(Params const& params)
      : Base(params), eligible_(params) {}

  template <class Cluster>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(Cluster) {
    return get_current_work();
  }

  CUTLASS_DEVICE WorkTileInfo get_current_work() const {
    auto original = Base::get_current_work();
    if (!eligible_.task_state || !original.is_valid()) return original;
    // Restrict this specialization to an unswizzled, batch-one exact tile grid.
    const int ticket = original.M_idx * eligible_.tiles_n + original.N_idx;
    cuda::atomic_ref<int, cuda::thread_scope_device> mapping(eligible_.task_state[ticket]);
    int mapped = mapping.load(cuda::memory_order_acquire);
    if (mapped == 0 && mapping.compare_exchange_strong(mapped, -1, cuda::memory_order_acq_rel,
                                                       cuda::memory_order_acquire)) {
      // One elected owner per ticket; all loader/math replicas share its
      // result.
      mapped = claim_ready_tile();
      mapping.store(mapped, cuda::memory_order_release);
    }
    while ((mapped = mapping.load(cuda::memory_order_acquire)) <= 0) __nanosleep(64);
    --mapped;
    return {mapped / eligible_.tiles_n, mapped % eligible_.tiles_n, 0, true};
  }

  // AllIssued does not imply completion of tiles held by other workers.
  enum class ClaimStatus { Ready, TemporarilyEmpty, AllIssued };

  struct ClaimResult {
    ClaimStatus status;
    WorkTileInfo tile;
  };

  CUTLASS_DEVICE static ClaimResult try_claim_ready_tile(Params const& params) {
    int* claims = params.task_state + params.tiles_m * params.tiles_n;
    bool all_issued = true;
    for (int m = 0; m < params.tiles_m; ++m) {
      cuda::atomic_ref<int, cuda::thread_scope_device> next_n(claims[m]);
      int n = next_n.load(cuda::memory_order_relaxed);
      if (n >= params.tiles_n) continue;
      all_issued = false;
      int first = m * params.tile_rows;
      int end = min(first + params.tile_rows, params.rows);
      bool ready = true;
      for (int g = first / params.group_rows; g <= (end - 1) / params.group_rows; ++g) {
        cuda::atomic_ref<int, cuda::thread_scope_device> count(params.readiness[g]);
        ready &= count.load(cuda::memory_order_acquire) >=
                 min(params.group_rows, params.rows - g * params.group_rows);
      }
      if (!ready) continue;
      while (n < params.tiles_n) {
        if (next_n.compare_exchange_strong(n, n + 1, cuda::memory_order_relaxed))
          return {ClaimStatus::Ready, {m, n, 0, true}};
      }
    }
    return {all_issued ? ClaimStatus::AllIssued : ClaimStatus::TemporarilyEmpty, {0, 0, 0, false}};
  }

 private:
  Params eligible_;

  CUTLASS_DEVICE int claim_ready_tile() const {
    for (;;) {
      auto result = try_claim_ready_tile(eligible_);
      if (result.status == ClaimStatus::Ready)
        return 1 + result.tile.M_idx * eligible_.tiles_n + result.tile.N_idx;
      // Legacy callers own a ticket and require producer progress while
      // waiting.
      __nanosleep(64);
    }
  }
};
}  // namespace flashinfer::ready_tma_gemm

namespace cutlass::gemm::kernel::detail {
template <class Arch, class Tile, class Cluster, uint32_t Stages>
struct TileSchedulerSelector<flashinfer::ready_tma_gemm::EligibleReadySchedulerTag, Arch, Tile,
                             Cluster, Stages> {
  using Scheduler = flashinfer::ready_tma_gemm::EligibleReadyScheduler;
};
}  // namespace cutlass::gemm::kernel::detail
