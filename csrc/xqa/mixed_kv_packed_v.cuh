/*
 * Copyright (c) 2026 by FlashInfer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace mixed_kv_fragments {

__device__ inline void copyPackedVAsync(SharedMem::StoredVSmemBuffer& tile, uint8_t* scales,
                                        flashinfer::KVPageStorage const& storage,
                                        MixedPageReferences<nbPagesPerVTile> const& pages,
                                        uint32_t head, uint32_t tokenBase, uint32_t headColumn,
                                        uint32_t warp) {
  constexpr uint32_t columns = grpLoadV ? headElems : warpVHeadElems;
  constexpr uint32_t steps = cacheVTileSeqLen / 16;
  constexpr uint32_t threads = (grpLoadV ? gemm1WarpsPerGrp : 1) * warp_size;
  auto* destination = reinterpret_cast<LdGrain*>(&tile);
#pragma unroll
  for (uint32_t item = warp * warp_size + laneId(); item < columns * steps; item += threads) {
    uint32_t const coefficient = item / steps;
    uint32_t const step = item % steps;
    uint32_t const absolute = tokenBase + step * 16;
    flashinfer::KVPageAddress const address{pages.values[absolute / tokensPerPage]};
    if (!address.allocated() || address.format() == flashinfer::KVPageFormat::kA16) continue;
    uint32_t const token = absolute % tokensPerPage;
    uint64_t const index = uint64_t(headColumn + coefficient) * tokensPerPage + token;
    auto const* payload = storage.payload(address, 0, head, true);
    if (address.format() == flashinfer::KVPageFormat::kBlockScaledFP4)
      ldgsts::copyAsync<8>(&destination[item], payload + index / 2, 8U);
    else
      ldgsts::copyAsyncCa16(&destination[item], payload + index, 16U);
    scales[item] = storage.scales(
        address, 0, head, true)[(headColumn + coefficient) * (tokensPerPage / 16) + token / 16];
  }
}

__device__ inline InstInMat<2, 1> fetchA16V(flashinfer::KVPageFormatSpan const& page, uint32_t head,
                                            uint32_t column, uint32_t tokenBase,
                                            uint32_t skipTokens, uint32_t cacheSeqLen) {
  InstInMat<2, 1> result;
#pragma unroll
  for (uint32_t half = 0; half < 2; ++half) {
    uint32_t word = 0;
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      uint32_t const token = tokenBase + (laneId() % 4) * 2 + half * 8 + j;
      uint32_t const coefficient = column + laneId() / 4;
      uint16_t bits = 0;
      if (page.allocated && token >= skipTokens && token < cacheSeqLen &&
          coefficient < validElemsPerHead) {
        auto const* address = static_cast<uint8_t const*>(page.v_payload) +
                              uint64_t(token % tokensPerPage) * page.payload_stride.token +
                              uint64_t(head) * page.payload_stride.head + coefficient * 2;
        bits = *reinterpret_cast<uint16_t const*>(address);
      }
      word |= uint32_t(bits) << (j * 16);
    }
    result.data[half][0] = word;
  }
  return result;
}

// m16n8k16 B owns one column and two token pairs per lane. The packed
// shared tile keeps one transposed 16-token codec block in each LdGrain.
template <flashinfer::KVPageFormat format>
__device__ inline InstInMat<2, 1> fetchPackedV(SharedMem::StoredVSmemBuffer const& tile,
                                               uint8_t const* scales, uint32_t headColumn,
                                               uint32_t token, float globalScale) {
  static_assert(format == flashinfer::KVPageFormat::kBlockScaledFP4 ||
                format == flashinfer::KVPageFormat::kBlockScaledFP8);
  uint32_t const coefficient = headColumn + laneId() / 4;
  uint32_t const index = coefficient * (cacheVTileSeqLen / 16) + token / 16;
  auto const* payload = reinterpret_cast<LdGrain const*>(&tile) + index;
  uint32_t const sf = broadcastA16Scale<InputElem>(
      convertE4M3ScaleToA16Bits<InputElem>(scales[index], globalScale));
  InstInMat<2, 1> result;
#pragma unroll
  for (uint32_t half = 0; half < 2; ++half) {
    uint32_t const pair = laneId() % 4 + half * 4;
    uint32_t converted;
    if constexpr (format == flashinfer::KVPageFormat::kBlockScaledFP4) {
      converted = convertE2M1x2ToA16<InputElem>(reinterpret_cast<uint8_t const*>(payload)[pair]);
    } else {
      converted = convertE4M3x2ToA16<InputElem>(reinterpret_cast<uint16_t const*>(payload)[pair]);
    }
    result.data[half][0] = mulA16x2<InputElem>(converted, sf);
  }
  return result;
}

}  // namespace mixed_kv_fragments
