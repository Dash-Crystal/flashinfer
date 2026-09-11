/*
 * Copyright (c) 2026 by FlashInfer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace mixed_kv_fragments {

struct QueryRows {
  InputElem const* values;
  uint32_t rowBegin;
  uint32_t rows;
  uint32_t heads;

  __device__ InputElem const* row(uint32_t local) const {
    uint32_t const packed = rowBegin + local;
    return local < rows ? values + (packed / headGrpSize * heads + packed % headGrpSize) * headElems
                        : nullptr;
  }

  __device__ float operator()(uint32_t local, uint32_t coefficient) const {
    auto const* source = row(local);
    return source != nullptr && coefficient < validElemsPerHead ? float(source[coefficient]) : 0;
  }

  template <uint32_t Rows, typename Columns>
  __device__ auto matrix(uint32_t coefficient, Columns columns) const {
    Array2D<InstInMat<2, 2>, Rows, 1> result;
#pragma unroll
    for (uint32_t i = 0; i < Rows; ++i) {
#pragma unroll
      for (uint32_t m = 0; m < 2; ++m) {
        auto const* source = row(i * 16 + laneId() / 4 + m * 8);
#pragma unroll
        for (uint32_t k = 0; k < 2; ++k) {
          uint32_t const column = coefficient + columns(laneId() % 4, k);
          result(i, 0).data[k][m] = source != nullptr && column + 2 <= validElemsPerHead
                                        ? *reinterpret_cast<uint32_t const*>(source + column)
                                        : 0;
        }
      }
    }
    return result;
  }
};

struct NativeQueryFP4 {
  uint32_t values[4];
  uint32_t scales;
};

__device__ inline void prepareNativeQuery(SharedMem::NativeQuery& dst, QueryRows const& q,
                                          uint32_t warp) {
  // One conversion per query, shared by every KV tile and every QK warp.
  constexpr uint32_t groups = ctaShapeInWarps.x * warp_size / 4;
  for (uint32_t group = warp * 8 + laneId() / 4; group < SharedMem::qRows * headElems / 16;
       group += groups) {
    uint32_t const row = group % SharedMem::qRows;
    uint32_t const block = group / SharedMem::qRows;
    uint32_t const lane = laneId() % 4;
    float values[4];
    float maximum = 0;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      values[i] = q(row, block * 16 + lane * 4 + i);
      maximum = fmaxf(maximum, fabsf(values[i]));
    }
    maximum = fmaxf(maximum, __shfl_xor_sync(~0U, maximum, 1));
    maximum = fmaxf(maximum, __shfl_xor_sync(~0U, maximum, 2));
    float const inverse8 = maximum == 0 ? 0 : 448.0f / maximum;
    __nv_fp8_e4m3 const scale4(fmaxf(maximum / 6.0f, 0x1p-9f));
    float const inverse4 = 1.0f / float(scale4);
    float values8[4];
    float values4[8];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      values8[i] = values[i] * inverse8;
      values4[i] = values[i] * inverse4;
      values4[i + 4] = __shfl_xor_sync(~0U, values4[i], 1);
    }
    dst.fp8[block][row][lane] = flashinfer::math::fp32_vec_to_e4m3(values8);
    if ((lane & 1U) == 0)
      dst.fp4[block][row][lane / 2] = flashinfer::math::fp32_vec_to_e2m1(values4);
    if (lane == 0) {
      dst.fp8Scales[block][row] = maximum / 448.0f;
      dst.fp4Scales[block][row] = scale4.__x;
    }
  }
}

__device__ inline NativeQueryFP4 loadQueryFP4(SharedMem::NativeQuery const& q, uint32_t row,
                                              uint32_t part) {
  NativeQueryFP4 result;
  uint32_t const quad = laneId() / 4;
  uint32_t const lane = laneId() % 4;
#pragma unroll
  for (uint32_t k = 0; k < 2; ++k) {
#pragma unroll
    for (uint32_t m = 0; m < 2; ++m) {
      result.values[k * 2 + m] =
          q.fp4[part * 4 + k * 2 + lane / 2][(row + quad + m * 8) % SharedMem::qRows][lane % 2];
    }
  }
  // CUTLASS SM120 SFALayout: the low lane bit chooses the row's upper/lower half.
  result.scales = 0;
#pragma unroll
  for (uint32_t k = 0; k < 4; ++k)
    result.scales |=
        uint32_t(q.fp4Scales[part * 4 + k][(row + quad + (lane & 1U) * 8) % SharedMem::qRows])
        << (k * 8);
  return result;
}

template <typename Acc, typename Query>
__device__ inline void nativeFP4QK(Acc& acc, Query const& queries, SharedMem::KSmemBuffer const& k,
                                   uint8_t const* scales, uint32_t tile, float globalScale) {
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
      uint32_t b[2];
#pragma unroll
      for (uint32_t block = 0; block < 2; ++block) {
        auto const& grain = k.template at<true>(relative, block * 2 + (laneId() % 4) / 2);
        b[block] = reinterpret_cast<uint32_t const*>(&grain)[laneId() % 2];
      }
      uint32_t const sf = reinterpret_cast<uint32_t const*>(scales)[relative];
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

__device__ inline NativeOperandFP8 loadQueryFP8(SharedMem::NativeQuery const& q, uint32_t row,
                                                uint32_t block) {
  NativeOperandFP8 result;
#pragma unroll
  for (uint32_t m = 0; m < 2; ++m) {
    uint32_t const r = (row + laneId() / 4 + m * 8) % SharedMem::qRows;
    result.values[m] = q.fp8[block][r][laneId() % 4];
    result.scales[m] = q.fp8Scales[block][r];
  }
  return result;
}

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

template <flashinfer::KVPageFormat format>
__device__ inline void nativePV(WarpAcc& acc, SharedMem::XSmemBuffer const& x, uint32_t xColumn,
                                QuadRegRowMax const& rowScales,
                                SharedMem::StoredVSmemBuffer const& tile, uint8_t const* scales,
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
    uint32_t const index = coefficient * (cacheVTileSeqLen / 16) + token / 16;
    auto const* payload = reinterpret_cast<LdGrain const*>(&tile) + index;
    uint32_t const b =
        fp4 ? expandFP4ToFP8(reinterpret_cast<uint16_t const*>(payload)[laneId() % 4])
            : reinterpret_cast<uint32_t const*>(payload)[laneId() % 4];
    uint8_t const sf = scales[index];
#pragma unroll
    for (uint32_t i = 0; i < warpTile.y / 16; ++i)
      nativeFP8Mma(acc(i, n), a[i], b, sf, globalScale);
  }
}

}  // namespace mixed_kv_fragments
