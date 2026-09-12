/*
 * Copyright (c) 2026 by FlashInfer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "page_storage.cuh"

namespace flashinfer::page_codec {

constexpr int kBlockSize = 16;
// The consumer expands payload * block_scale into FP16 before global scaling.
constexpr float kFP8A16ScaleMax = 128.0f;

__device__ __forceinline__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

__device__ __forceinline__ float decode_e2m1_nibble(uint8_t code) {
  constexpr float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float value = magnitude[code & 7];
  return code & 8 ? -value : value;
}

// Preserve the mixed-page codec's ties toward smaller magnitude. Native NVFP4
// uses ties-to-even and cannot replace this conversion without changing bytes.
__device__ __forceinline__ uint8_t encode_e2m1_nibble(float value) {
  const float absolute = fabsf(value);
  int nearest = 0;
  float best = absolute;
#pragma unroll
  for (int code = 1; code < 8; ++code) {
    const float distance = fabsf(absolute - decode_e2m1_nibble(code));
    if (distance < best) {
      best = distance;
      nearest = code;
    }
  }
  return uint8_t(nearest | (signbit(value) ? 8 : 0));
}

// Participating contiguous lanes own exactly one block; warp_lane is explicit.
template <int VALUES>
__device__ __forceinline__ float block_amax(float const (&values)[VALUES], int warp_lane) {
  static_assert(VALUES > 0 && VALUES <= kBlockSize && (VALUES & (VALUES - 1)) == 0);
  constexpr int lanes = kBlockSize / VALUES;
  const uint32_t mask = ((1U << lanes) - 1) << (warp_lane / lanes * lanes);
  float maximum = 0;
#pragma unroll
  for (int i = 0; i < VALUES; ++i) maximum = fmaxf(maximum, fabsf(values[i]));
#pragma unroll
  for (int offset = lanes / 2; offset > 0; offset /= 2) {
    maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, offset, lanes));
  }
  return maximum;
}

__device__ __forceinline__ float encoding_inverse(__nv_fp8_e4m3 scale, float global_scale) {
  const float value = static_cast<float>(scale);
  return value == 0.0f ? 0.0f : reciprocal_approximate_ftz(global_scale * value);
}

template <KVPageFormat Format>
struct BlockCodec {
  static_assert(Format == KVPageFormat::kBlockScaledFP8 || Format == KVPageFormat::kBlockScaledFP4);
  static constexpr int kBits = Format == KVPageFormat::kBlockScaledFP4 ? 4 : 8;

  __device__ static __forceinline__ __nv_fp8_e4m3 scale(float amax, float global_scale) {
    constexpr float format_max = kBits == 4 ? 6.0f : 448.0f;
    constexpr float scale_max = kBits == 4 ? 448.0f : kFP8A16ScaleMax;
    float value = 0.0f;
    if (amax != 0.0f) {
      const float required =
          amax * reciprocal_approximate_ftz(global_scale) * reciprocal_approximate_ftz(format_max);
      value = fminf(fmaxf(required, 0x1p-9f), scale_max);
    }
    return __nv_fp8_e4m3(value);
  }

  // Increasing coefficient indices occupy increasing bits in the stored word.
  template <int VALUES>
  __device__ static __forceinline__ uint32_t encode(float const (&values)[VALUES], float inverse) {
    static_assert(VALUES > 0 && VALUES * kBits <= 32);
    uint32_t packed = 0;
    if constexpr (kBits == 8 && VALUES > 1) {
      static_assert(VALUES % 2 == 0);
#pragma unroll
      for (int i = 0; i < VALUES; i += 2) {
        const float2 pair = {values[i] * inverse, values[i + 1] * inverse};
        packed |= uint32_t(__nv_cvt_float2_to_fp8x2(pair, __NV_SATFINITE, __NV_E4M3)) << (i * 8);
      }
    } else {
#pragma unroll
      for (int i = 0; i < VALUES; ++i) {
        const float value = values[i] * inverse;
        const uint8_t code = kBits == 4 ? encode_e2m1_nibble(value) : __nv_fp8_e4m3(value).__x;
        packed |= uint32_t(code) << (i * kBits);
      }
    }
    return packed;
  }
};

}  // namespace flashinfer::page_codec
