/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Included after XQA's matrix atoms and shared tile types.
namespace mixed_kv_fragments {

using flashinfer::KVPageFormat;

template <typename Function>
__device__ inline void visit(uint8_t format, Function const& function) {
#if MIXED_PAGE_STATIC_FORMAT >= 0
  function(MixedFormatTag<MIXED_PAGE_STATIC_FORMAT>{});
#else
  if (format == static_cast<uint8_t>(KVPageFormat::kA16)) {
    function(MixedFormatTag<static_cast<uint8_t>(KVPageFormat::kA16)>{});
  } else if (format == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8)) {
    function(MixedFormatTag<static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8)>{});
  } else {
    function(MixedFormatTag<static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4)>{});
  }
#endif
}

template <KVPageFormat format>
__device__ inline InstInMat<2, 2> loadK(SharedMem::KSmemBuffer const& tile, uint32_t row,
                                        uint32_t block, uint32_t part, uint8_t const* scales,
                                        float globalScale) {
  if constexpr (format == KVPageFormat::kA16) {
    return loadInstInMat<2, 2, true, false, false>(this_warp(), tile, row, block * 2);
  } else {
    constexpr uint32_t blocksPerPart = kHeadPartBytes / 32;
    constexpr uint32_t scaleStride = mha::max(4U, blocksPerPart);
    uint32_t const scaleColumn = (part * blocksPerPart + block) % scaleStride;
    uint32_t const pair = (laneId() & 3U) * 2;
    InstInMat<2, 2> result;
#pragma unroll
    for (uint32_t n = 0; n < 2; ++n) {
      uint32_t const token = row + n * 8 + laneId() / 4;
      auto const* packed = reinterpret_cast<uint8_t const*>(&tile.template at<true>(token, block));
      uint8_t const scale = scales[token * scaleStride + scaleColumn];
      uint32_t low, high;
      if constexpr (format == KVPageFormat::kBlockScaledFP8) {
        low = *reinterpret_cast<uint16_t const*>(packed + pair);
        high = *reinterpret_cast<uint16_t const*>(packed + pair + 8);
      } else {
        low = packed[pair / 2];
        high = packed[pair / 2 + 4];
      }
      uint32_t const sf =
          broadcastA16Scale<InputElem>(convertE4M3ScaleToA16Bits<InputElem>(scale, globalScale));
      if constexpr (format == KVPageFormat::kBlockScaledFP8) {
        result.data[n][0] = mulA16x2<InputElem>(convertE4M3x2ToA16<InputElem>(low), sf);
        result.data[n][1] = mulA16x2<InputElem>(convertE4M3x2ToA16<InputElem>(high), sf);
      } else {
        result.data[n][0] = mulA16x2<InputElem>(convertE2M1x2ToA16<InputElem>(low), sf);
        result.data[n][1] = mulA16x2<InputElem>(convertE2M1x2ToA16<InputElem>(high), sf);
      }
    }
    return result;
  }
}

__device__ inline uint32_t vScaleRow(uint32_t token) {
  if constexpr (grpLoadV) {
    // Each copying warp owns a token interval followed by a dump row.
    return token + token / (cacheVTileSeqLen / gemm1WarpsPerGrp);
  } else {
    return token;
  }
}

template <KVPageFormat format>
__device__ inline InstInMat<2, 2> loadV(SharedMem::VSmemBuffer const& tile, uint32_t row,
                                        uint32_t block, uint8_t const* scales, float globalScale) {
  if constexpr (format == KVPageFormat::kA16) {
    return loadInstInMat<2, 2, false, true, false>(
        this_warp(), tile, row, block * 2, [](uint32_t token) { return mixedVFragmentRow(token); });
  } else {
    constexpr uint32_t scaleStride = mha::max(4U, SharedMem::VSmemBuffer::rowBytes / 32);
    auto const* address = &tile.template at<true>(row + laneId() % 16, block);
    auto const packed = [&]() {
      if constexpr (format == KVPageFormat::kBlockScaledFP8) {
        return ldmatrix_16x16_trans<1>(address);
      } else {
        return ldmatrix_16x16_trans_unpack_4b<1>(address);
      }
    }();
    uint32_t sf[2];
#pragma unroll
    for (uint32_t half = 0; half < 2; ++half) {
      uint32_t const token = row + half * 8 + (laneId() & 3U) * 2;
      uint16_t const low = convertE4M3ScaleToA16Bits<InputElem>(
          scales[vScaleRow(token) * scaleStride + block], globalScale);
      uint16_t const high = convertE4M3ScaleToA16Bits<InputElem>(
          scales[vScaleRow(token + 1) * scaleStride + block], globalScale);
      sf[half] = uint32_t(low) | (uint32_t(high) << 16);
    }
    InstInMat<2, 2> result;
#pragma unroll
    for (uint32_t n = 0; n < 2; ++n) {
      uint32_t const word = packed[n];
#pragma unroll
      for (uint32_t half = 0; half < 2; ++half) {
        uint32_t converted;
        if constexpr (format == KVPageFormat::kBlockScaledFP8) {
          converted = convertE4M3x2ToA16<InputElem>(word >> (half * 16));
        } else {
          converted = convertE2M1x2ToA16<InputElem>((word | (word >> 4)) >> (half * 16));
        }
        result.data[n][half] = mulA16x2<InputElem>(converted, sf[half]);
      }
    }
    return result;
  }
}

}  // namespace mixed_kv_fragments

__device__ inline void smemQKPartGemmMixed(Warp const& warp, WarpAcc& acc,
                                           SharedMem::QSmemBuffer const& q, uint32_t qColBeg,
                                           SharedMem::KSmemBuffer const& k,
                                           MixedPageFormats<nbPagesPerWarpTile> const& formats,
                                           uint8_t const* scales, uint32_t part,
                                           float fp8GlobalScale, float fp4GlobalScale) {
  constexpr uint32_t rows = warpTile.y / 16;
  // Static accumulator indices, with a single format dispatch outside the rolled
  // reduction loop. The old block/page unrolling replicated every converter.
#pragma unroll
  for (uint32_t tile = 0; tile < warpTile.x / 16; ++tile) {
    mixed_kv_fragments::visit(formats.values[tile * 16 / tokensPerPage], [&](auto tag) {
      constexpr auto format = static_cast<flashinfer::KVPageFormat>(decltype(tag)::value);
      float const scale =
          format == flashinfer::KVPageFormat::kBlockScaledFP8 ? fp8GlobalScale : fp4GlobalScale;
#pragma unroll 1
      for (uint32_t block = 0; block < kHeadPartBytes / 32; ++block) {
        auto const b = mixed_kv_fragments::loadK<format>(k, tile * 16, block, part, scales, scale);
        auto const a = loadQueryMatrix<2, 2, rows, 1>(warp, q, qColBeg + block * 2);
#pragma unroll
        for (uint32_t i = 0; i < rows; ++i) {
#pragma unroll
          for (uint32_t n = 0; n < 2; ++n) {
            uint32_t const operand[2][1] = {b.data[n][0], b.data[n][1]};
            mma<InputElem>(acc(i, tile * 2 + n).data, a(i, 0).data, operand);
          }
        }
      }
    });
  }
}

__device__ inline void smemXVPartGemmMixed(Warp const& warp, WarpAcc& acc, bool skipRescale,
                                           UniformRescaleMask, ThrdRegRowMax rowScales,
                                           SharedMem::XSmemBuffer const& x, uint32_t vTile,
                                           SharedMem::VSmemBuffer const& v,
                                           MixedPageFormats<nbPagesPerVTile> const& formats,
                                           uint8_t const* scales, uint32_t headSlice,
                                           float fp8GlobalScale, float fp4GlobalScale) {
  constexpr uint32_t rows = warpTile.y / 16;
  Vec<InputElem2, QuadRegRowMax::size> scalesQuad;
#if INPUT_FP16
  auto const converted = __float2half2_rn(rowScales);
#else
  auto const converted = __float2bfloat162_rn(rowScales);
#endif
  reinterpret_cast<QuadRegRowMax&>(scalesQuad) =
      replicateForQuad(warp, reinterpret_cast<ThrdRegRowMax const&>(converted));
#pragma unroll
  for (uint32_t tile = 0; tile < cacheVTileSeqLen / 16; ++tile) {
    uint32_t const column = SharedMem::XSmemBuffer::cols / nbCacheVTilesPerXTile * vTile + tile * 2;
    auto a = loadMatrix<2, 2, rows, 1, false, false, false, false>(warp, x, 0, column);
#pragma unroll
    for (uint32_t i = 0; i < rows; ++i) {
#pragma unroll
      for (uint32_t n = 0; n < 2; ++n) {
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          auto& value = reinterpret_cast<InputElem2&>(a(i, 0).data[j][n]);
          value = skipRescale ? value : value * scalesQuad[i * 2 + n];
        }
      }
    }
    mixed_kv_fragments::visit(formats.values[tile * 16 / tokensPerPage], [&](auto tag) {
      constexpr auto format = static_cast<flashinfer::KVPageFormat>(decltype(tag)::value);
      float const scale =
          format == flashinfer::KVPageFormat::kBlockScaledFP8 ? fp8GlobalScale : fp4GlobalScale;
#pragma unroll
      for (uint32_t block = 0; block < warpTile.x / 16; ++block) {
        auto const b = mixed_kv_fragments::loadV<format>(
            v, tile * 16, headSlice * (warpTile.x / 16) + block, scales, scale);
#pragma unroll
        for (uint32_t i = 0; i < rows; ++i) {
#pragma unroll
          for (uint32_t n = 0; n < 2; ++n) {
            uint32_t const operand[2][1] = {b.data[n][0], b.data[n][1]};
            mma<InputElem>(acc(i, block * 2 + n).data, a(i, 0).data, operand);
          }
        }
      }
    });
  }
}
