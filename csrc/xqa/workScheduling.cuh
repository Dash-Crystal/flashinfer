// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace xqa_work {

struct SplitCost {
  uint64_t numerator;
  uint32_t splits;

  __host__ __device__ bool precedes(SplitCost other) const {
    uint64_t const a = numerator * other.splits;
    uint64_t const b = other.numerator * splits;
    return a < b || (a == b && splits < other.splits);
  }
};

// One tile of setup/merge work per CTA, plus its share of the sequence.
__host__ __device__ inline SplitCost cost(uint32_t slots, uint32_t sequences, uint32_t tiles,
                                          uint32_t splits) {
  uint64_t const waves = (uint64_t(sequences) * splits + slots - 1) / slots;
  return {waves * (uint64_t(tiles) + splits), splits};
}

__host__ __device__ inline uint32_t select(SplitCost best, SplitCost single) {
  return best.numerator * 100 < single.numerator * 95 * best.splits ? best.splits : 1;
}

__host__ __device__ inline uint32_t chooseSplits(uint32_t slots, uint32_t sequences,
                                                 uint32_t tiles) {
  SplitCost const single = cost(slots, sequences, tiles, 1);
  SplitCost best = single;
  for (uint32_t n = 2; n <= tiles; ++n) {
    SplitCost const candidate = cost(slots, sequences, tiles, n);
    if (candidate.precedes(best)) best = candidate;
  }
  return select(best, single);
}

__device__ inline uint32_t chooseSplitsWarp(uint32_t slots, uint32_t sequences, uint32_t tiles) {
  SplitCost const single = cost(slots, sequences, tiles, 1);
  SplitCost best = single;
  for (uint32_t n = 2 + (threadIdx.x % 32); n <= tiles; n += 32) {
    SplitCost const candidate = cost(slots, sequences, tiles, n);
    if (candidate.precedes(best)) best = candidate;
  }
  for (uint32_t delta = 16; delta != 0; delta /= 2) {
    SplitCost const other{__shfl_xor_sync(~0U, best.numerator, delta),
                          __shfl_xor_sync(~0U, best.splits, delta)};
    if (other.precedes(best)) best = other;
  }
  return select(best, single);
}

// Increasing the tile count can only select an equal or larger split count:
// within a wave the largest n wins; successive winning waves have lower slopes.
// Reserve the largest live grid, rather than requests * storage-capacity tiles.
inline uint32_t gridCapacity(uint32_t slots, uint32_t requests, uint32_t heads, uint32_t maxTiles) {
  uint32_t capacity = 0;
  for (uint32_t r = 1; r <= requests; ++r) {
    uint32_t const jobs = r * heads * chooseSplits(slots, r * heads, maxTiles);
    if (jobs > capacity) capacity = jobs;
  }
  return capacity;
}

}  // namespace xqa_work
