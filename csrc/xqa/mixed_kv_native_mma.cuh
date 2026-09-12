/*
 * Copyright (c) 2026 by FlashInfer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace mixed_kv_fragments {

struct NativeOperandFP8 {
  uint32_t values[2];
  float scales[2];
};

template <typename Load>
__device__ inline NativeOperandFP8 quantizeOperandFP8(Load const& load) {
  NativeOperandFP8 result;
  uint32_t const quad = laneId() / 4;
  uint32_t const first = (laneId() % 4) * 4;
#pragma unroll
  for (uint32_t m = 0; m < 2; ++m) {
    float values[4];
    float maximum = 0;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      values[j] = load(quad + m * 8, first + j);
      maximum = fmaxf(maximum, fabsf(values[j]));
    }
    maximum = fmaxf(maximum, __shfl_xor_sync(~0U, maximum, 1));
    maximum = fmaxf(maximum, __shfl_xor_sync(~0U, maximum, 2));
    result.scales[m] = maximum / 448.0f;
    float const inverse = maximum == 0 ? 0 : 448.0f / maximum;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) values[j] *= inverse;
    result.values[m] = flashinfer::math::fp32_vec_to_e4m3(values);
  }
  return result;
}

struct NativeValueFP8 {
  uint32_t values;
  float scales[2];
};

__device__ inline NativeValueFP8 prepareValueFP8(uint32_t values, uint8_t scaleBits,
                                                 float globalScale) {
  __nv_fp8_e4m3 scale;
  scale.__x = scaleBits;
  float const decoded = float(scale) * globalScale;
  NativeValueFP8 result{values, {}};
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j)
    result.scales[j] = __shfl_sync(~0U, decoded, ((laneId() % 4) * 2 + j) * 4);
  return result;
}

__device__ inline void nativeFP8Mma(InstAcc& acc, NativeOperandFP8 const& a,
                                    NativeValueFP8 const& b) {
  float partial[2][2] = {};
  mmaF8_k16(partial, a.values, b.values);
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
#pragma unroll
    for (uint32_t m = 0; m < 2; ++m) acc(m, j) += partial[m][j] * (a.scales[m] * b.scales[j]);
  }
}

__device__ inline uint32_t expandFP4ToFP8(uint16_t values) {
  constexpr uint64_t magnitudes = 0x4c4844403c383000ULL;
  uint32_t result = 0;
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t const nibble = (values >> (i * 4)) & 15U;
    uint32_t const bits = uint8_t(magnitudes >> ((nibble & 7U) * 8)) | ((nibble & 8U) << 4);
    result |= bits << (i * 8);
  }
  return result;
}

__device__ inline void copyNativeVAsync(SharedMem::StoredVSmemBuffer& tile, uint8_t* scales,
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

__device__ inline InstInMat<2, 1> fetchNativeA16V(flashinfer::KVPageFormatSpan const& page,
                                                  uint32_t head, uint32_t column,
                                                  uint32_t tokenBase, uint32_t skipTokens,
                                                  uint32_t cacheSeqLen) {
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

using NativePVOperands = Vec<NativeOperandFP8, warpTile.y / 16>;

__device__ inline void prepareNativeProbabilities(SharedMem::NativeProbabilities& dst,
                                                  SharedMem::XSmemBuffer const& x) {
  // The softmax producer converts once; every V warp and head slice reuses it.
  // The original A16 tile remains available to A16 pages.
#pragma unroll
  for (uint32_t block = 0; block < warpTile.x / 16; ++block) {
#pragma unroll
    for (uint32_t i = 0; i < divUp(SharedMem::qRows, 16U); ++i) {
      auto const a = quantizeOperandFP8([&](uint32_t row, uint32_t k) {
        if (i * 16 + row >= SharedMem::qRows) return 0.0f;
        auto const& grain = x.template at<true>(i * 16 + row, block * 2 + k / 8);
        return float(reinterpret_cast<InputElem const*>(&grain)[k % 8]);
      });
#pragma unroll
      for (uint32_t m = 0; m < 2; ++m) {
        uint32_t const row = i * 16 + laneId() / 4 + m * 8;
        if (row < SharedMem::qRows) {
          dst.values[block][row][laneId() % 4] = a.values[m];
          if (laneId() % 4 == 0) dst.scales[block][row] = a.scales[m];
        }
      }
    }
  }
}

__device__ inline NativePVOperands prepareNativePV(SharedMem::NativeProbabilities const& p,
                                                   uint32_t block, QuadRegRowMax const& rowScales) {
  NativePVOperands a;
#pragma unroll
  for (uint32_t i = 0; i < warpTile.y / 16; ++i) {
#pragma unroll
    for (uint32_t m = 0; m < 2; ++m) {
      uint32_t const row = i * 16 + laneId() / 4 + m * 8;
      a[i].values[m] = row < SharedMem::qRows ? p.values[block][row][laneId() % 4] : 0;
      a[i].scales[m] = row < SharedMem::qRows ? p.scales[block][row] * rowScales[i * 2 + m] : 0;
    }
  }
  return a;
}

template <flashinfer::KVPageFormat format>
__device__ inline void nativePV(WarpAcc& acc, NativePVOperands const& a,
                                SharedMem::StoredVSmemBuffer const& tile, uint8_t const* scales,
                                uint32_t headColumn, uint32_t token, float globalScale) {
  constexpr bool fp4 = format == flashinfer::KVPageFormat::kBlockScaledFP4;
#pragma unroll
  for (uint32_t n = 0; n < warpTile.x / 8; ++n) {
    uint32_t const coefficient = headColumn + n * 8 + laneId() / 4;
    uint32_t const index = coefficient * (cacheVTileSeqLen / 16) + token / 16;
    auto const* payload = reinterpret_cast<LdGrain const*>(&tile) + index;
    uint32_t const b =
        fp4 ? expandFP4ToFP8(reinterpret_cast<uint16_t const*>(payload)[laneId() % 4])
            : reinterpret_cast<uint32_t const*>(payload)[laneId() % 4];
    auto const operand = prepareValueFP8(b, scales[index], globalScale);
#pragma unroll
    for (uint32_t i = 0; i < warpTile.y / 16; ++i) nativeFP8Mma(acc(i, n), a[i], operand);
  }
}

}  // namespace mixed_kv_fragments
