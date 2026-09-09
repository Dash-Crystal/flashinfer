// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace xqa_work {

__host__ __device__ inline uint32_t nextSplitPage(uint32_t page, uint32_t step,
                                                  uint32_t pagesPerTile, uint32_t splits) {
  return page + step + (page % pagesPerTile + step) / pagesPerTile * pagesPerTile * (splits - 1);
}

// One guard tile per request makes independently rounded query segments disjoint.
__host__ __device__ inline uint32_t raggedTileStart(uint32_t queryOffset, uint32_t request,
                                                    uint32_t heads, uint32_t rows) {
  return uint64_t(queryOffset) * heads / rows + request;
}

__device__ inline uint32_t raggedRequest(uint32_t const* queryOffsets, uint32_t requests,
                                         uint32_t tile, uint32_t heads, uint32_t rows) {
  uint32_t begin = 0, end = requests;
  while (begin + 1 < end) {
    uint32_t const middle = begin + (end - begin) / 2;
    if (raggedTileStart(queryOffsets[middle], middle, heads, rows) <= tile)
      begin = middle;
    else
      end = middle;
  }
  return begin;
}

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

// One warp produces [live jobs, splits, optional compact request indices].
__device__ inline void prepareWarp(uint32_t const* lengths, uint32_t requests, uint32_t heads,
                                   uint32_t slots, uint32_t tile, uint32_t window, uint32_t* output,
                                   uint32_t const* queryOffsets = nullptr, uint32_t queryHeads = 0,
                                   uint32_t queryRows = 0) {
  uint32_t const lane = threadIdx.x % 32;
  uint32_t live = 0;
  uint32_t maxTiles = 0;
  uint32_t queryJobs = 0;
  for (uint32_t first = 0; first < requests; first += 32) {
    uint32_t const r = first + lane;
    uint32_t const len = r < requests ? lengths[r] : 0;
    uint32_t const queries =
        queryOffsets != nullptr && r < requests ? queryOffsets[r + 1] - queryOffsets[r] : 1;
    bool const active = len != 0 && queries != 0;
    uint64_t const span = uint64_t(window) + queries - 1;
    uint32_t const begin = window != 0 && len > span ? len - span : 0;
    uint32_t const tiles = active ? (len + tile - 1) / tile - begin / tile : 0;
    maxTiles = max(maxTiles, tiles);
    if (queryOffsets != nullptr) {
      if (active)
        queryJobs =
            max(queryJobs, raggedTileStart(queryOffsets[r + 1], r + 1, queryHeads, queryRows));
    } else {
      uint32_t const mask = __ballot_sync(~0U, active);
      if (active) output[2 + live + __popc(mask & ((1U << lane) - 1))] = r;
      live += __popc(mask);
    }
  }
  if (queryOffsets != nullptr) live = __reduce_max_sync(~0U, queryJobs);
  maxTiles = __reduce_max_sync(~0U, maxTiles);
  uint32_t const splits = chooseSplitsWarp(slots, live * heads, maxTiles);
  if (lane == 0) {
    output[0] = live;
    output[1] = splits;
  }
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
