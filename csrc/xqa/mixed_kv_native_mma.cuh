/*
 * Copyright (c) 2026 by FlashInfer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace mixed_kv_fragments {

__device__ inline float queryCoefficient(SharedMem::QSmemBuffer const& q, uint32_t row,
                                         uint32_t coefficient) {
  // Invert XQA's existing A16 query permutation within each 16-coefficient group.
  uint32_t const local = coefficient % 16;
  uint32_t const permuted = (local % 4) / 2 * 8 + (local / 4) * 2 + local % 2;
  uint32_t const column = coefficient / 16 * 2 + permuted / 8;
  auto const& grain = q.template at<qkSwizzle>(row % SharedMem::qRows, column);
  return float(reinterpret_cast<InputElem const*>(&grain)[permuted % 8]);
}

struct NativeQueryFP4 {
  uint32_t values[4];
  uint32_t scales;
};

__device__ inline NativeQueryFP4 quantizeQueryFP4(SharedMem::QSmemBuffer const& q, uint32_t row,
                                                  uint32_t column) {
  NativeQueryFP4 result;
  uint32_t scales[2][2];
  uint32_t const quad = laneId() / 4;
  uint32_t const lane = laneId() % 4;
#pragma unroll
  for (uint32_t k = 0; k < 2; ++k) {
#pragma unroll
    for (uint32_t m = 0; m < 2; ++m) {
      float values[8];
      float maximum = 0;
#pragma unroll
      for (uint32_t j = 0; j < 8; ++j) {
        values[j] = queryCoefficient(q, row + quad + m * 8, (column + k * 4 + lane) * 8 + j);
        maximum = fmaxf(maximum, fabsf(values[j]));
      }
      maximum = fmaxf(maximum, __shfl_xor_sync(~0U, maximum, 1));
      __nv_fp8_e4m3 const scale(fmaxf(maximum / 6.0f, 0x1p-9f));
      scales[m][k] = scale.__x;
      float const inverse = 1.0f / float(scale);
#pragma unroll
      for (uint32_t j = 0; j < 8; ++j) values[j] *= inverse;
      result.values[k * 2 + m] = flashinfer::math::fp32_vec_to_e2m1(values);
    }
  }
  // CUTLASS SM120 SFALayout: the low lane bit chooses the row's upper/lower half.
  uint32_t const low = lane & 1 ? scales[1][0] : scales[0][0];
  uint32_t const high = lane & 1 ? scales[1][1] : scales[0][1];
  uint32_t const scaleLane = quad * 4 + (lane & 1);
  result.scales = __shfl_sync(~0U, low, scaleLane) | (__shfl_sync(~0U, low, scaleLane + 2) << 8) |
                  (__shfl_sync(~0U, high, scaleLane) << 16) |
                  (__shfl_sync(~0U, high, scaleLane + 2) << 24);
  return result;
}

template <typename Acc, typename Query>
__device__ inline void nativeFP4QK(Acc& acc, Query const& queries,
                                   flashinfer::KVPageFormatSpan const& page, uint32_t head,
                                   uint32_t tokenBase, uint32_t skipTokens, uint32_t cacheSeqLen,
                                   uint32_t part, uint32_t tile, float globalScale) {
  using Mma =
      cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<cutlass::float_e2m1_t, cutlass::float_e2m1_t,
                                                    float, cutlass::float_ue4m3_t, 16>;
  static_assert(kHeadPartBytes == 128);
#pragma unroll
  for (uint32_t i = 0; i < warpTile.y / 16; ++i) {
    auto const& a = queries[i];
#pragma unroll
    for (uint32_t n = 0; n < 2; ++n) {
      uint32_t const relative = tile * 16 + n * 8 + laneId() / 4;
      uint32_t const absolute = tokenBase + relative;
      bool const valid = page.allocated && absolute >= skipTokens && absolute < cacheSeqLen;
      uint32_t b[2] = {};
      uint32_t sf = 0;
      if (valid) {
        auto const* payload = static_cast<uint8_t const*>(page.k_payload) +
                              uint64_t(absolute % tokensPerPage) * page.payload_stride.token +
                              uint64_t(head) * page.payload_stride.head + part * 32;
#pragma unroll
        for (uint32_t k = 0; k < 2; ++k)
          b[k] = reinterpret_cast<uint32_t const*>(payload)[k * 4 + laneId() % 4];
        auto const* scales = page.k_scales +
                             uint64_t(absolute % tokensPerPage) * page.scale_stride.token +
                             uint64_t(head) * page.scale_stride.head + part * 4;
        sf = *reinterpret_cast<uint32_t const*>(scales);
      }
      float partial[4] = {};
      Mma::fma(partial[0], partial[1], partial[2], partial[3], a.values[0], a.values[1],
               a.values[2], a.values[3], b[0], b[1], partial[0], partial[1], partial[2], partial[3],
               a.scales, sf);
#pragma unroll
      for (uint32_t m = 0; m < 2; ++m)
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j)
          acc(i, tile * 2 + n)(m, j) += partial[m * 2 + j] * globalScale;
    }
  }
}

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

__device__ inline void nativeFP8Mma(InstAcc& acc, NativeOperandFP8 const& a, uint32_t b,
                                    uint8_t scaleBits, float globalScale) {
  __nv_fp8_e4m3 scale;
  scale.__x = scaleBits;
  float const decoded = float(scale) * globalScale;
  float partial[2][2] = {};
  mmaF8_k16(partial, a.values, b);
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    float const columnScale = __shfl_sync(~0U, decoded, ((laneId() % 4) * 2 + j) * 4);
#pragma unroll
    for (uint32_t m = 0; m < 2; ++m) acc(m, j) += partial[m][j] * (a.scales[m] * columnScale);
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

template <flashinfer::KVPageFormat format>
__device__ inline void nativePV(WarpAcc& acc, SharedMem::XSmemBuffer const& x, uint32_t xColumn,
                                QuadRegRowMax const& rowScales,
                                flashinfer::KVPageStorage const& storage,
                                flashinfer::KVPageAddress address, uint32_t head,
                                uint32_t headColumn, uint32_t token, float globalScale) {
  constexpr bool fp4 = format == flashinfer::KVPageFormat::kBlockScaledFP4;
  Vec<NativeOperandFP8, warpTile.y / 16> a;
#pragma unroll
  for (uint32_t i = 0; i < warpTile.y / 16; ++i) {
    a[i] = quantizeOperandFP8([&](uint32_t row, uint32_t k) {
      auto const& grain = x.template at<true>(i * 16 + row, xColumn + k / 8);
      return float(reinterpret_cast<InputElem const*>(&grain)[k % 8]) * rowScales[i * 2 + row / 8];
    });
  }
#pragma unroll
  for (uint32_t n = 0; n < warpTile.x / 8; ++n) {
    uint32_t const coefficient = headColumn + n * 8 + laneId() / 4;
    uint32_t b = 0;
    uint8_t sf = 0;
    if (address.allocated()) {
      uint64_t const index = uint64_t(coefficient) * tokensPerPage + token + (laneId() % 4) * 4;
      auto const* payload = storage.payload(address, 0, head, true);
      if constexpr (fp4) {
        b = expandFP4ToFP8(*reinterpret_cast<uint16_t const*>(payload + index / 2));
      } else {
        b = *reinterpret_cast<uint32_t const*>(payload + index);
      }
      sf = storage.scales(address, 0, head, true)[coefficient * (tokensPerPage / 16) + token / 16];
    }
#pragma unroll
    for (uint32_t i = 0; i < warpTile.y / 16; ++i)
      nativeFP8Mma(acc(i, n), a[i], b, sf, globalScale);
  }
}

}  // namespace mixed_kv_fragments
