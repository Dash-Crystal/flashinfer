/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// FP4 KV cache quantization kernels with linear (non-swizzled) block scale layout.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cuda/atomic>

#include "flashinfer/attention/page_storage.cuh"
#include "tvm_ffi_utils.h"

// Number of elements per block scale group
constexpr int NVFP4_BLOCK_SIZE = 16;
constexpr int BSFP8_BLOCK_SIZE = 16;
constexpr int MIXED_KV_SIGNATURE_BLOCK_SIZE = 32;
// Keep payload * block_scale finite when the attention tile algebra expands
// E4M3 directly into FP16 registers before applying the global scale.
constexpr float BSFP8_A16_SCALE_MAX = 128.0f;

// Software E2M1 is the SM90 producer fallback and the semantic reference for
// SM100/SM120 specializations.  Nibble order matches page_transport.cuh:
// low nibble is the even coefficient, high nibble is the odd coefficient.
__device__ __forceinline__ uint8_t encode_e2m1_nibble(float value) {
  constexpr float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float absolute = fabsf(value);
  int nearest = 0;
  float best = absolute;
#pragma unroll
  for (int code = 1; code < 8; ++code) {
    const float distance = fabsf(absolute - magnitude[code]);
    if (distance < best) {
      best = distance;
      nearest = code;
    }
  }
  return uint8_t(nearest | (signbit(value) ? 8 : 0));
}

__device__ __forceinline__ float decode_e2m1_nibble(uint8_t code) {
  constexpr float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float value = magnitude[code & 7];
  return code & 8 ? -value : value;
}

// Helper functions
__device__ __forceinline__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

__device__ __forceinline__ __nv_bfloat162 cuda_abs(__nv_bfloat162 a) {
  __nv_bfloat162 result;
  float fx = fabsf(__bfloat162float(a.x));
  float fy = fabsf(__bfloat162float(a.y));
  result.x = __float2bfloat16(fx);
  result.y = __float2bfloat16(fy);
  return result;
}

__device__ __forceinline__ half2 cuda_abs(half2 a) { return __habs2(a); }

__device__ __forceinline__ __nv_bfloat162 cuda_max(__nv_bfloat162 a, __nv_bfloat162 b) {
  __nv_bfloat162 result;
  result.x = __bfloat162float(a.x) > __bfloat162float(b.x) ? a.x : b.x;
  result.y = __bfloat162float(a.y) > __bfloat162float(b.y) ? a.y : b.y;
  return result;
}

__device__ __forceinline__ half2 cuda_max(half2 a, half2 b) { return __hmax2(a, b); }

// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y), "f"(array[2].x),
        "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
#else
  // E2M1 conversion requires SM100+; abort at runtime if this code path is ever reached.
  // Note: static_assert cannot be used here because NVCC evaluates both preprocessor branches.
  __trap();
  return 0;
#endif
}

// Quantize 8 FP16/BF16 values to E2M1 with FP8 E4M3 block scaling
template <typename InType>
__device__ uint32_t quantize_fp16_to_e2m1_with_scaling(InType (&vec)[4], float global_scale,
                                                       uint8_t* block_scale_out) {
  constexpr int SF_VEC_SIZE = 16;
  constexpr int CVT_ELTS_PER_THREAD = 8;
  constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_ELTS_PER_THREAD;

  auto localMax = cuda_abs(vec[0]);

#pragma unroll
  for (int i = 1; i < CVT_ELTS_PER_THREAD / 2; i++) {
    localMax = cuda_max(localMax, cuda_abs(vec[i]));
  }

  localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  if constexpr (CVT_NUM_THREADS_PER_SF == 4) {
    localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
  }

  float vecMax;
  if constexpr (std::is_same_v<InType, __nv_bfloat162>) {
    auto max_single =
        __bfloat162float(localMax.x) > __bfloat162float(localMax.y) ? localMax.x : localMax.y;
    vecMax = __bfloat162float(max_single);
  } else {
    vecMax = fmaxf(__half2float(localMax.x), __half2float(localMax.y));
  }

  uint8_t fp8_scale_val = 0;
  float output_scale = 0.0f;

  auto sf_value =
      reciprocal_approximate_ftz(global_scale) * (vecMax * reciprocal_approximate_ftz(6.0f));

  __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(sf_value);
  fp8_scale_val = tmp.__x;
  sf_value = static_cast<float>(tmp);

  output_scale = vecMax != 0 ? reciprocal_approximate_ftz(sf_value * global_scale) : 0.0f;

  if (block_scale_out) {
    *block_scale_out = fp8_scale_val;
  }

  float2 fp2_vals[CVT_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<InType, __nv_bfloat162>) {
      fp2_vals[i] = __bfloat1622float2(vec[i]);
    } else {
      fp2_vals[i] = __half22float2(vec[i]);
    }
    fp2_vals[i].x *= output_scale;
    fp2_vals[i].y *= output_scale;
  }

  uint32_t e2m1_vec = fp32_vec_to_e2m1(fp2_vals);

  return e2m1_vec;
}

// Type traits for FP16/BF16 packed types and zero-initialization
template <typename T>
struct fp16_traits;

template <>
struct fp16_traits<__nv_bfloat16> {
  using packed_type = __nv_bfloat162;
  static __device__ __forceinline__ __nv_bfloat16 zero() { return __float2bfloat16(0.0f); }
  static __device__ __forceinline__ __nv_bfloat162 zero2() { return __float2bfloat162_rn(0.0f); }
};

template <>
struct fp16_traits<half> {
  using packed_type = half2;
  static __device__ __forceinline__ half zero() { return __float2half(0.0f); }
  static __device__ __forceinline__ half2 zero2() { return __float2half2_rn(0.0f); }
};

// Unified quantization kernel for FP16/BF16 to NVFP4
template <typename InType, int BLOCK_SIZE = 128, int ELTS_PER_THREAD = 16>
__global__ void nvfp4_quant_kernel(const InType* __restrict__ input,
                                   const float* __restrict__ global_scale_ptr,
                                   uint8_t* __restrict__ fp4_output,
                                   uint8_t* __restrict__ block_scales, const int M, const int K) {
  using traits = fp16_traits<InType>;
  using PackedType = typename traits::packed_type;

  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (row >= M) return;

  __shared__ float global_scale;

  if (tid == 0) {
    global_scale = *global_scale_ptr;
  }
  __syncthreads();

  constexpr int CVT_ELTS_PER_THREAD = 8;
  constexpr int PACKED_PER_THREAD = CVT_ELTS_PER_THREAD / 2;
  const int elts_per_block = BLOCK_SIZE * CVT_ELTS_PER_THREAD;

  const InType* row_input = input + row * K;
  uint8_t* row_fp4 = fp4_output + row * (K / 2);
  uint8_t* row_scales = block_scales + row * (K / NVFP4_BLOCK_SIZE);

  for (int base_col = 0; base_col < K; base_col += elts_per_block) {
    const int col_start = base_col + tid * CVT_ELTS_PER_THREAD;

    if (col_start >= K) break;

    PackedType vec[4];

#pragma unroll
    for (int i = 0; i < PACKED_PER_THREAD; ++i) {
      const int col = col_start + i * 2;
      if (col + 1 < K) {
        vec[i] = *reinterpret_cast<const PackedType*>(&row_input[col]);
      } else if (col < K) {
        vec[i].x = row_input[col];
        vec[i].y = traits::zero();
      } else {
        vec[i] = traits::zero2();
      }
    }

    const int block_idx = col_start / NVFP4_BLOCK_SIZE;
    uint8_t* scale_out = (tid % 2 == 0) ? &row_scales[block_idx] : nullptr;

    uint32_t e2m1_vals = quantize_fp16_to_e2m1_with_scaling(vec, global_scale, scale_out);

    const int packed_idx = col_start / 2;
    if (packed_idx + 3 < K / 2) {
      *reinterpret_cast<uint32_t*>(&row_fp4[packed_idx]) = e2m1_vals;
    } else {
      uint8_t* bytes = reinterpret_cast<uint8_t*>(&e2m1_vals);
      for (int i = 0; i < 4 && packed_idx + i < K / 2; ++i) {
        row_fp4[packed_idx + i] = bytes[i];
      }
    }
  }
}

void nvfp4_kv_quant(TensorView input, TensorView global_scale, TensorView fp4_output,
                    TensorView block_scales) {
  CHECK_INPUT(input);
  CHECK_CUDA(global_scale);
  CHECK_INPUT(fp4_output);
  CHECK_INPUT(block_scales);

  const int M = input.size(0);
  const int K = input.size(1);

  TVM_FFI_ICHECK(input.ndim() == 2) << "input must be 2D";
  TVM_FFI_ICHECK(K % NVFP4_BLOCK_SIZE == 0)
      << "K dimension must be divisible by " << NVFP4_BLOCK_SIZE;
  TVM_FFI_ICHECK(fp4_output.ndim() == 2) << "fp4_output must be 2D";
  TVM_FFI_ICHECK(fp4_output.size(0) == M) << "fp4_output row count mismatch";
  TVM_FFI_ICHECK(fp4_output.size(1) == K / 2) << "fp4_output column count mismatch";
  TVM_FFI_ICHECK(block_scales.ndim() == 2) << "block_scales must be 2D";
  TVM_FFI_ICHECK(block_scales.size(0) == M) << "block_scales row count mismatch";
  TVM_FFI_ICHECK(block_scales.size(1) == K / NVFP4_BLOCK_SIZE)
      << "block_scales column count mismatch";
  TVM_FFI_ICHECK(global_scale.device().device_id == input.device().device_id)
      << "global_scale must be on the same device as input";
  TVM_FFI_ICHECK(fp4_output.device().device_id == input.device().device_id)
      << "fp4_output must be on the same device as input";
  TVM_FFI_ICHECK(block_scales.device().device_id == input.device().device_id)
      << "block_scales must be on the same device as input";

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  cudaStream_t stream = get_stream(input.device());

  const float* scale_ptr = static_cast<const float*>(global_scale.data_ptr());

  constexpr int BLOCK_SIZE = 128;
  dim3 grid(M);
  dim3 block(BLOCK_SIZE);

  constexpr int ELTS_PER_THREAD = 16;

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
    nvfp4_quant_kernel<c_type, BLOCK_SIZE, ELTS_PER_THREAD>
        <<<grid, block, 0, stream>>>(static_cast<const c_type*>(input.data_ptr()), scale_ptr,
                                     static_cast<uint8_t*>(fp4_output.data_ptr()),
                                     static_cast<uint8_t*>(block_scales.data_ptr()), M, K);
    return true;
  });
}

// One 16-lane subgroup seals one block. Eight subgroups per CTA process eight
// adjacent head-dimension blocks with coalesced reads and writes.
template <typename InType, int THREADS = 128>
__global__ void bsfp8_quant_kernel(const InType* __restrict__ input,
                                   const float* __restrict__ global_scale_ptr,
                                   uint8_t* __restrict__ fp8_output,
                                   uint8_t* __restrict__ block_scales, const int M, const int K) {
  const int row = blockIdx.x;
  if (row >= M) return;

  __shared__ float global_scale;
  if (threadIdx.x == 0) global_scale = *global_scale_ptr;
  __syncthreads();

  constexpr int GROUPS_PER_CTA = THREADS / BSFP8_BLOCK_SIZE;
  const int subgroup = threadIdx.x / BSFP8_BLOCK_SIZE;
  const int lane = threadIdx.x % BSFP8_BLOCK_SIZE;
  const int blocks_per_row = K / BSFP8_BLOCK_SIZE;
  for (int block = subgroup; block < blocks_per_row; block += GROUPS_PER_CTA) {
    const int col = block * BSFP8_BLOCK_SIZE + lane;
    float value;
    if constexpr (std::is_same_v<InType, __nv_bfloat16>) {
      value = __bfloat162float(input[row * K + col]);
    } else {
      value = __half2float(input[row * K + col]);
    }
    float block_max = fabsf(value);
#pragma unroll
    for (int offset = BSFP8_BLOCK_SIZE / 2; offset > 0; offset /= 2) {
      block_max =
          fmaxf(block_max, __shfl_xor_sync(uint32_t(-1), block_max, offset, BSFP8_BLOCK_SIZE));
    }

    float sf_value = block_max == 0.0f
                         ? 0.0f
                         : fminf(block_max * reciprocal_approximate_ftz(global_scale) *
                                     reciprocal_approximate_ftz(448.0f),
                                 BSFP8_A16_SCALE_MAX);
    __nv_fp8_e4m3 sf_fp8 = __nv_fp8_e4m3(sf_value);
    sf_value = static_cast<float>(sf_fp8);
    if (lane == 0) {
      block_scales[row * blocks_per_row + block] = sf_fp8.__x;
    }
    float const encode_scale =
        sf_value == 0.0f ? 0.0f : reciprocal_approximate_ftz(global_scale * sf_value);
    __nv_fp8_e4m3 encoded = __nv_fp8_e4m3(value * encode_scale);
    fp8_output[row * K + col] = encoded.__x;
  }
}

void bsfp8_kv_quant(TensorView input, TensorView global_scale, TensorView fp8_output,
                    TensorView block_scales) {
  CHECK_INPUT(input);
  CHECK_CUDA(global_scale);
  CHECK_INPUT(fp8_output);
  CHECK_INPUT(block_scales);

  const int M = input.size(0);
  const int K = input.size(1);
  TVM_FFI_ICHECK(input.ndim() == 2) << "input must be 2D";
  TVM_FFI_ICHECK(K % BSFP8_BLOCK_SIZE == 0)
      << "K dimension must be divisible by " << BSFP8_BLOCK_SIZE;
  TVM_FFI_ICHECK(fp8_output.ndim() == 2 && fp8_output.size(0) == M && fp8_output.size(1) == K)
      << "fp8_output shape mismatch";
  TVM_FFI_ICHECK(block_scales.ndim() == 2 && block_scales.size(0) == M &&
                 block_scales.size(1) == K / BSFP8_BLOCK_SIZE)
      << "block_scales shape mismatch";
  TVM_FFI_ICHECK(global_scale.device().device_id == input.device().device_id)
      << "global_scale must be on the same device as input";

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  cudaStream_t stream = get_stream(input.device());
  constexpr int THREADS = 128;
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input.dtype(), c_type, [&] {
    bsfp8_quant_kernel<c_type, THREADS>
        <<<M, THREADS, 0, stream>>>(static_cast<const c_type*>(input.data_ptr()),
                                    static_cast<const float*>(global_scale.data_ptr()),
                                    static_cast<uint8_t*>(fp8_output.data_ptr()),
                                    static_cast<uint8_t*>(block_scales.data_ptr()), M, K);
    return true;
  });
}

__global__ void mixed_kv_reset_reused_pages_kernel(const int32_t* __restrict__ reused_pages,
                                                   const int32_t* __restrict__ reused_count,
                                                   uint8_t* __restrict__ page_format,
                                                   float* __restrict__ page_router_stats,
                                                   const int capacity) {
  const int event = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = *reused_count;
  if (event < capacity && event < count) {
    const int32_t page = reused_pages[event];
    page_format[page] = 0;
    const float unmeasured = __int_as_float(0x7f800000);
    page_router_stats[page * 2] = unmeasured;
    page_router_stats[page * 2 + 1] = unmeasured;
  }
}

template <int THREADS, typename T>
__device__ __forceinline__ T mixed_kv_block_sum(T value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(uint32_t(-1), value, offset);
  }
  __shared__ T warp_values[THREADS / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_values[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < THREADS / 32 ? warp_values[lane] : T{0};
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      value += __shfl_down_sync(uint32_t(-1), value, offset);
    }
  }
  __syncthreads();
  return value;
}

template <int THREADS>
__device__ __forceinline__ float mixed_kv_block_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(uint32_t(-1), value, offset));
  }
  __shared__ float warp_values[THREADS / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_values[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < THREADS / 32 ? warp_values[lane] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      value = fmaxf(value, __shfl_down_sync(uint32_t(-1), value, offset));
    }
  }
  __syncthreads();
  return value;
}

template <typename InType>
__device__ __forceinline__ float mixed_kv_to_float(InType value) {
  if constexpr (std::is_same_v<InType, __nv_bfloat16>) {
    return __bfloat162float(value);
  } else {
    return __half2float(value);
  }
}

template <int THREADS, typename Neighbor, typename Signature>
__device__ __forceinline__ void mixed_kv_route_moments(int64_t pair_values,
                                                       int64_t signature_blocks, Neighbor neighbor,
                                                       Signature signature, float* partials) {
  float neighbor_dot = 0.0f;
  float neighbor_left_sq = 0.0f;
  float neighbor_right_sq = 0.0f;
  for (int64_t linear = threadIdx.x; linear < pair_values; linear += THREADS) {
    const auto pair = neighbor(linear);
    neighbor_dot = fmaf(pair.x, pair.y, neighbor_dot);
    neighbor_left_sq = fmaf(pair.x, pair.x, neighbor_left_sq);
    neighbor_right_sq = fmaf(pair.y, pair.y, neighbor_right_sq);
  }
  neighbor_dot = mixed_kv_block_sum<THREADS>(neighbor_dot);
  neighbor_left_sq = mixed_kv_block_sum<THREADS>(neighbor_left_sq);
  neighbor_right_sq = mixed_kv_block_sum<THREADS>(neighbor_right_sq);

  static_assert(THREADS % 32 == 0);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float peak_rms = 0.0f;
  for (int64_t block = warp; block < signature_blocks; block += THREADS / 32) {
    const float value = signature(block, lane);
    float sum_sq = value * value;
    float peak = fabsf(value);
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {
      sum_sq += __shfl_down_sync(uint32_t(-1), sum_sq, delta);
      peak = fmaxf(peak, __shfl_down_sync(uint32_t(-1), peak, delta));
    }
    if (lane == 0) {
      const float rms = sqrtf(sum_sq / float(MIXED_KV_SIGNATURE_BLOCK_SIZE));
      peak_rms = fmaxf(peak_rms, rms == 0.0f ? 0.0f : peak / rms);
    }
  }
  peak_rms = mixed_kv_block_max<THREADS>(peak_rms);

  if (threadIdx.x == 0) {
    partials[0] = neighbor_dot;
    partials[1] = neighbor_left_sq;
    partials[2] = neighbor_right_sq;
    partials[3] = peak_rms;
  }
}

template <int THREADS, typename Load>
__device__ __forceinline__ void mixed_kv_route_row(int token, int page_size, int head_dim,
                                                   Load load, float* partials) {
  const int blocks_per_kv = head_dim / MIXED_KV_SIGNATURE_BLOCK_SIZE;
  mixed_kv_route_moments<THREADS>(
      token + 1 < page_size ? 2 * head_dim : 0, 2 * blocks_per_kv,
      [&](int64_t linear) {
        const bool is_v = linear >= head_dim;
        const int dim = is_v ? linear - head_dim : linear;
        return make_float2(load(is_v, token, dim), load(is_v, token + 1, dim));
      },
      [&](int64_t block, int lane) {
        const bool is_v = block >= blocks_per_kv;
        const int dim_block = is_v ? block - blocks_per_kv : block;
        return load(is_v, token, dim_block * MIXED_KV_SIGNATURE_BLOCK_SIZE + lane);
      },
      partials);
}

__device__ __forceinline__ float2 mixed_kv_route_stats(float dot, float left_sq, float right_sq,
                                                       float peak_rms) {
  const float denom = sqrtf(left_sq * right_sq);
  return make_float2(denom == 0.0f ? 0.0f : dot / denom, peak_rms);
}

template <typename InType, int THREADS = 128>
__global__ void mixed_kv_route_rows_kernel(
    const InType* __restrict__ k_input, const InType* __restrict__ v_input,
    const int32_t* __restrict__ completed_pages, const int32_t* __restrict__ completed_count,
    float* __restrict__ page_router_partials, const int completed_capacity, const int page_size,
    const int num_heads, const int head_dim, const int64_t in_stride_page,
    const int64_t in_stride_token, const int64_t in_stride_head, const int64_t in_stride_dim) {
  const int event_token = blockIdx.x;
  const int event = event_token / page_size;
  if (event >= completed_capacity || event >= *completed_count) return;
  const int token = event_token - event * page_size;
  const int head = blockIdx.y;
  const int32_t page = completed_pages[event];

  const auto load = [&](bool is_v, int row_token, int dim) {
    const InType* input = is_v ? v_input : k_input;
    const int64_t offset = page * in_stride_page + row_token * in_stride_token +
                           head * in_stride_head + dim * in_stride_dim;
    return mixed_kv_to_float(input[offset]);
  };
  const int64_t partial = (static_cast<int64_t>(event_token) * num_heads + head) * 4;
  mixed_kv_route_row<THREADS>(token, page_size, head_dim, load, page_router_partials + partial);
}

template <int THREADS = 128>
__global__ void mixed_kv_finalize_route_kernel(const int32_t* __restrict__ completed_pages,
                                               const int32_t* __restrict__ completed_count,
                                               const float* __restrict__ page_router_partials,
                                               float* __restrict__ page_router_stats,
                                               const int completed_capacity, const int page_size,
                                               const int num_heads) {
  const int event = blockIdx.x;
  if (event >= completed_capacity || event >= *completed_count) return;
  float dot = 0.0f;
  float left_sq = 0.0f;
  float right_sq = 0.0f;
  float peak_rms = 0.0f;
  const int rows = page_size * num_heads;
  for (int row = threadIdx.x; row < rows; row += THREADS) {
    const int64_t partial = (static_cast<int64_t>(event) * rows + row) * 4;
    dot += page_router_partials[partial + 0];
    left_sq += page_router_partials[partial + 1];
    right_sq += page_router_partials[partial + 2];
    peak_rms = fmaxf(peak_rms, page_router_partials[partial + 3]);
  }
  dot = mixed_kv_block_sum<THREADS>(dot);
  left_sq = mixed_kv_block_sum<THREADS>(left_sq);
  right_sq = mixed_kv_block_sum<THREADS>(right_sq);
  peak_rms = mixed_kv_block_max<THREADS>(peak_rms);
  if (threadIdx.x == 0) {
    const int32_t page = completed_pages[event];
    const auto stats = mixed_kv_route_stats(dot, left_sq, right_sq, peak_rms);
    page_router_stats[page * 2] = stats.x;
    page_router_stats[page * 2 + 1] = stats.y;
  }
}

__device__ __forceinline__ uint8_t mixed_kv_select_format(
    float neighbor_cos, float peak_rms, const float* __restrict__ routing_thresholds) {
  const bool fp4 = neighbor_cos <= routing_thresholds[0] && peak_rms <= routing_thresholds[1];
  const bool fp8 = neighbor_cos <= routing_thresholds[2] && peak_rms <= routing_thresholds[3];
  return fp4 ? 2 : (fp8 ? 1 : 0);
}

__device__ __forceinline__ uint8_t
mixed_kv_select_format(const float* __restrict__ page_router_stats, const int32_t page,
                       const float* __restrict__ routing_thresholds) {
  return mixed_kv_select_format(page_router_stats[page * 2], page_router_stats[page * 2 + 1],
                                routing_thresholds);
}

struct MixedKVQuantizedBlock {
  uint8_t scale;
  uint8_t payload;
};

// Both rectangular and packed page writers use the original 16-lane codec.
__device__ __forceinline__ MixedKVQuantizedBlock mixed_kv_quantize_block(float value,
                                                                         float global_scale,
                                                                         uint8_t selected_format) {
  const int lane = threadIdx.x % BSFP8_BLOCK_SIZE;
  const uint32_t subgroup_mask = 0xffffU << (threadIdx.x & 16);
  float block_max = fabsf(value);
#pragma unroll
  for (int offset = BSFP8_BLOCK_SIZE / 2; offset > 0; offset /= 2) {
    block_max =
        fmaxf(block_max, __shfl_xor_sync(subgroup_mask, block_max, offset, BSFP8_BLOCK_SIZE));
  }

  const float format_max = selected_format == 2 ? 6.0f : 448.0f;
  const float scale_max = selected_format == 2 ? 448.0f : BSFP8_A16_SCALE_MAX;
  float sf_value = 0.0f;
  if (block_max != 0.0f) {
    const float required_sf = block_max * reciprocal_approximate_ftz(global_scale) *
                              reciprocal_approximate_ftz(format_max);
    constexpr float factors[9] = {
        0.75f, 0.8125f, 0.875f, 0.9375f, 1.0f, 1.0625f, 1.125f, 1.25f, 1.5f,
    };
    float best_objective = __int_as_float(0x7f800000);
#pragma unroll
    for (int candidate = 0; candidate < 9; ++candidate) {
      float candidate_sf = fminf(fmaxf(required_sf * factors[candidate], 0x1p-9f), scale_max);
      __nv_fp8_e4m3 candidate_sf_fp8 = __nv_fp8_e4m3(candidate_sf);
      candidate_sf = static_cast<float>(candidate_sf_fp8);
      const float encode_scale = reciprocal_approximate_ftz(global_scale * candidate_sf);
      float reconstructed;
      if (selected_format == 2) {
        reconstructed = decode_e2m1_nibble(encode_e2m1_nibble(value * encode_scale)) *
                        candidate_sf * global_scale;
      } else {
        __nv_fp8_e4m3 encoded = __nv_fp8_e4m3(value * encode_scale);
        reconstructed = static_cast<float>(encoded) * candidate_sf * global_scale;
      }
      const float residual = fabsf(reconstructed - value);
      float sum_squared = residual * residual;
      float max_residual = residual;
#pragma unroll
      for (int offset = BSFP8_BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        sum_squared += __shfl_xor_sync(subgroup_mask, sum_squared, offset, BSFP8_BLOCK_SIZE);
        max_residual = fmaxf(
            max_residual, __shfl_xor_sync(subgroup_mask, max_residual, offset, BSFP8_BLOCK_SIZE));
      }
      const float objective =
          sum_squared / float(BSFP8_BLOCK_SIZE) + 0.05f * max_residual * max_residual;
      if (objective < best_objective) {
        best_objective = objective;
        sf_value = candidate_sf;
      }
    }
  }
  __nv_fp8_e4m3 sf_fp8 = __nv_fp8_e4m3(sf_value);
  const float encode_scale =
      sf_value == 0.0f ? 0.0f : reciprocal_approximate_ftz(global_scale * sf_value);
  uint8_t payload;
  if (selected_format == 1) {
    payload = __nv_fp8_e4m3(value * encode_scale).__x;
  } else {
    const int pair_lane = lane & (BSFP8_BLOCK_SIZE / 2 - 1);
    const float low = __shfl_sync(subgroup_mask, value, pair_lane * 2, BSFP8_BLOCK_SIZE);
    const float high = __shfl_sync(subgroup_mask, value, pair_lane * 2 + 1, BSFP8_BLOCK_SIZE);
    payload =
        encode_e2m1_nibble(low * encode_scale) | (encode_e2m1_nibble(high * encode_scale) << 4);
  }
  return {sf_fp8.__x, payload};
}

// Match the existing bsfp8_quant_kernel / NVIDIA NVFP4 producer geometry:
// one 16-lane subgroup owns one scale block.  A CTA covers eight adjacent
// blocks, so scale selection and payload stores stay coalesced and no lane
// serializes an entire coefficient block.
template <typename InType, int THREADS = 128>
__global__ void mixed_kv_quant_rows_kernel(
    const InType* k_input, const InType* v_input, const int32_t* __restrict__ completed_pages,
    const int32_t* __restrict__ completed_count, const float* __restrict__ fp8_k_global_scale_ptr,
    const float* __restrict__ fp8_v_global_scale_ptr,
    const float* __restrict__ fp4_k_global_scale_ptr,
    const float* __restrict__ fp4_v_global_scale_ptr, uint8_t* fp8_k_output, uint8_t* fp8_v_output,
    uint8_t* __restrict__ fp8_k_block_scales, uint8_t* __restrict__ fp8_v_block_scales,
    uint8_t* fp4_k_output, uint8_t* fp4_v_output, uint8_t* __restrict__ fp4_k_block_scales,
    uint8_t* __restrict__ fp4_v_block_scales, const float* __restrict__ page_router_stats,
    const float* __restrict__ routing_thresholds, const int completed_capacity, const int page_size,
    const int num_heads, const int head_dim, const int64_t in_stride_page,
    const int64_t in_stride_token, const int64_t in_stride_head, const int64_t in_stride_dim,
    const int64_t fp8_stride_page, const int64_t fp8_stride_token, const int64_t fp8_stride_head,
    const int64_t fp8_stride_dim, const int64_t fp4_stride_page, const int64_t fp4_stride_token,
    const int64_t fp4_stride_head, const int64_t fp4_stride_dim, const int64_t sf_stride_page,
    const int64_t sf_stride_token, const int64_t sf_stride_head, const int64_t sf_stride_dim) {
  const int event_token = blockIdx.x;
  const int event = event_token / page_size;
  if (event >= completed_capacity || event >= *completed_count) return;
  const int token = event_token - event * page_size;
  const int head = blockIdx.y;
  const int32_t page = completed_pages[event];
  const uint8_t selected_format =
      mixed_kv_select_format(page_router_stats, page, routing_thresholds);
  if (selected_format == 0) {
    return;
  }

  static_assert(THREADS % BSFP8_BLOCK_SIZE == 0);
  constexpr int GROUPS_PER_CTA = THREADS / BSFP8_BLOCK_SIZE;
  const int subgroup = threadIdx.x / BSFP8_BLOCK_SIZE;
  const int lane = threadIdx.x % BSFP8_BLOCK_SIZE;
  const int dim_blocks = head_dim / BSFP8_BLOCK_SIZE;

  __shared__ float global_scales[4];
  if (threadIdx.x < 4) {
    const float* scale_ptrs[4] = {fp8_k_global_scale_ptr, fp8_v_global_scale_ptr,
                                  fp4_k_global_scale_ptr, fp4_v_global_scale_ptr};
    global_scales[threadIdx.x] = *scale_ptrs[threadIdx.x];
  }
  __syncthreads();

#pragma unroll
  for (int kv = 0; kv < 2; ++kv) {
    const bool is_v = kv != 0;
    const InType* input = is_v ? v_input : k_input;
    uint8_t* fp8_output = is_v ? fp8_v_output : fp8_k_output;
    uint8_t* fp8_scales = is_v ? fp8_v_block_scales : fp8_k_block_scales;
    uint8_t* fp4_output = is_v ? fp4_v_output : fp4_k_output;
    uint8_t* fp4_scales = is_v ? fp4_v_block_scales : fp4_k_block_scales;
    const float global_scale = global_scales[(selected_format == 2 ? 2 : 0) + kv];

    for (int base = 0; base < dim_blocks; base += GROUPS_PER_CTA) {
      const int dim_block = base + subgroup;
      const bool valid = dim_block < dim_blocks;
      const int dim = dim_block * BSFP8_BLOCK_SIZE + lane;
      const int64_t input_offset = page * in_stride_page + token * in_stride_token +
                                   head * in_stride_head + dim * in_stride_dim;
      const float value = valid ? mixed_kv_to_float(input[input_offset]) : 0.0f;
      const auto encoded = mixed_kv_quantize_block(value, global_scale, selected_format);
      // All source lanes have read before an in-place compressed store.
      __syncthreads();
      const int64_t sf_offset = page * sf_stride_page + token * sf_stride_token +
                                head * sf_stride_head + dim_block * sf_stride_dim;
      if (valid && lane == 0) {
        if (selected_format == 2) {
          fp4_scales[sf_offset] = encoded.scale;
        } else {
          fp8_scales[sf_offset] = encoded.scale;
        }
      }
      if (selected_format == 1) {
        if (valid)
          fp8_output[page * fp8_stride_page + token * fp8_stride_token + head * fp8_stride_head +
                     dim * fp8_stride_dim] = encoded.payload;
      } else {
        if (valid && lane < BSFP8_BLOCK_SIZE / 2) {
          const int packed_dim = dim_block * (BSFP8_BLOCK_SIZE / 2) + lane;
          const int64_t output_offset = page * fp4_stride_page + token * fp4_stride_token +
                                        head * fp4_stride_head + packed_dim * fp4_stride_dim;
          fp4_output[output_offset] = encoded.payload;
        }
      }
    }
  }
}

__global__ void mixed_kv_publish_pages_kernel(const int32_t* __restrict__ completed_pages,
                                              const int32_t* __restrict__ completed_count,
                                              uint8_t* __restrict__ page_format,
                                              const float* __restrict__ page_router_stats,
                                              const float* __restrict__ routing_thresholds,
                                              const int completed_capacity) {
  const int event = blockIdx.x * blockDim.x + threadIdx.x;
  if (event >= completed_capacity || event >= *completed_count) return;
  const int32_t page = completed_pages[event];
  page_format[page] = mixed_kv_select_format(page_router_stats, page, routing_thresholds);
}

void mixed_kv_quant_pages(
    TensorView k_input, TensorView v_input, TensorView reused_pages, TensorView reused_count,
    TensorView completed_pages, TensorView completed_count, TensorView fp8_k_global_scale,
    TensorView fp8_v_global_scale, TensorView fp4_k_global_scale, TensorView fp4_v_global_scale,
    TensorView fp8_k_output, TensorView fp8_v_output, TensorView fp8_k_block_scales,
    TensorView fp8_v_block_scales, TensorView fp4_k_output, TensorView fp4_v_output,
    TensorView fp4_k_block_scales, TensorView fp4_v_block_scales, TensorView page_router_partials,
    TensorView page_format, TensorView page_router_stats, TensorView routing_thresholds) {
  CHECK_CUDA(k_input);
  CHECK_CUDA(v_input);
  CHECK_CUDA(reused_pages);
  CHECK_CUDA(reused_count);
  CHECK_CUDA(completed_pages);
  CHECK_CUDA(completed_count);
  CHECK_CUDA(fp8_k_global_scale);
  CHECK_CUDA(fp8_v_global_scale);
  CHECK_CUDA(fp4_k_global_scale);
  CHECK_CUDA(fp4_v_global_scale);
  CHECK_CUDA(fp8_k_output);
  CHECK_CUDA(fp8_v_output);
  CHECK_CUDA(fp8_k_block_scales);
  CHECK_CUDA(fp8_v_block_scales);
  CHECK_CUDA(fp4_k_output);
  CHECK_CUDA(fp4_v_output);
  CHECK_CUDA(fp4_k_block_scales);
  CHECK_CUDA(fp4_v_block_scales);
  CHECK_CUDA(page_router_partials);
  CHECK_CUDA(page_format);
  CHECK_CUDA(page_router_stats);
  CHECK_CUDA(routing_thresholds);
  TVM_FFI_ICHECK(k_input.ndim() == 4 && v_input.ndim() == 4)
      << "K and V inputs must be [page, token, head, dim]";
  TVM_FFI_ICHECK(k_input.size(0) == v_input.size(0) && k_input.size(1) == v_input.size(1) &&
                 k_input.size(2) == v_input.size(2) && k_input.size(3) == v_input.size(3))
      << "K and V input shapes must match";
  TVM_FFI_ICHECK(k_input.stride(0) == v_input.stride(0) && k_input.stride(1) == v_input.stride(1) &&
                 k_input.stride(2) == v_input.stride(2) && k_input.stride(3) == v_input.stride(3))
      << "K and V input strides must match";
  const int num_pages = k_input.size(0);
  const int page_size = k_input.size(1);
  const int num_heads = k_input.size(2);
  const int head_dim = k_input.size(3);
  TVM_FFI_ICHECK(head_dim % MIXED_KV_SIGNATURE_BLOCK_SIZE == 0)
      << "head dimension must be divisible by " << MIXED_KV_SIGNATURE_BLOCK_SIZE;
  TVM_FFI_ICHECK(fp8_k_output.ndim() == 4 && fp8_v_output.ndim() == 4 &&
                 fp8_k_output.size(0) == num_pages && fp8_k_output.size(1) == page_size &&
                 fp8_k_output.size(2) == num_heads && fp8_k_output.size(3) == head_dim &&
                 fp8_v_output.size(0) == num_pages && fp8_v_output.size(1) == page_size &&
                 fp8_v_output.size(2) == num_heads && fp8_v_output.size(3) == head_dim)
      << "FP8 payload outputs must match the input shape";
  TVM_FFI_ICHECK(fp4_k_output.ndim() == 4 && fp4_v_output.ndim() == 4 &&
                 fp4_k_output.size(0) == num_pages && fp4_k_output.size(1) == page_size &&
                 fp4_k_output.size(2) == num_heads && fp4_k_output.size(3) == head_dim / 2 &&
                 fp4_v_output.size(0) == num_pages && fp4_v_output.size(1) == page_size &&
                 fp4_v_output.size(2) == num_heads && fp4_v_output.size(3) == head_dim / 2)
      << "FP4 payload outputs must match the packed input shape";
  TVM_FFI_ICHECK(
      fp8_k_block_scales.ndim() == 4 && fp8_v_block_scales.ndim() == 4 &&
      fp4_k_block_scales.ndim() == 4 && fp4_v_block_scales.ndim() == 4 &&
      fp8_k_block_scales.size(0) == num_pages && fp8_k_block_scales.size(1) == page_size &&
      fp8_k_block_scales.size(2) == num_heads &&
      fp8_k_block_scales.size(3) == head_dim / BSFP8_BLOCK_SIZE &&
      fp8_v_block_scales.size(0) == num_pages && fp8_v_block_scales.size(1) == page_size &&
      fp8_v_block_scales.size(2) == num_heads &&
      fp8_v_block_scales.size(3) == head_dim / BSFP8_BLOCK_SIZE &&
      fp4_k_block_scales.size(0) == num_pages && fp4_k_block_scales.size(1) == page_size &&
      fp4_k_block_scales.size(2) == num_heads &&
      fp4_k_block_scales.size(3) == head_dim / BSFP8_BLOCK_SIZE &&
      fp4_v_block_scales.size(0) == num_pages && fp4_v_block_scales.size(1) == page_size &&
      fp4_v_block_scales.size(2) == num_heads &&
      fp4_v_block_scales.size(3) == head_dim / BSFP8_BLOCK_SIZE)
      << "block-scale outputs have the wrong shape";
  TVM_FFI_ICHECK(reused_pages.ndim() == 1 && reused_pages.dtype() == dl_int32)
      << "reused_pages must be 1D int32";
  TVM_FFI_ICHECK(completed_pages.ndim() == 1 && completed_pages.dtype() == dl_int32)
      << "completed_pages must be 1D int32";
  TVM_FFI_ICHECK(reused_count.numel() == 1 && reused_count.dtype() == dl_int32)
      << "reused_count must be a scalar int32 tensor";
  TVM_FFI_ICHECK(completed_count.numel() == 1 && completed_count.dtype() == dl_int32)
      << "completed_count must be a scalar int32 tensor";
  TVM_FFI_ICHECK(page_format.ndim() == 1 && page_format.size(0) == num_pages)
      << "page_format must have one byte per physical page";
  TVM_FFI_ICHECK(page_router_stats.ndim() == 2 && page_router_stats.size(0) == num_pages &&
                 page_router_stats.size(1) == 2 && page_router_stats.dtype() == dl_float32)
      << "page_router_stats must be float32 [num_pages, 2]";
  const int completed_capacity = completed_pages.size(0);
  TVM_FFI_ICHECK(
      page_router_partials.ndim() == 4 && page_router_partials.size(0) == completed_capacity &&
      page_router_partials.size(1) == page_size && page_router_partials.size(2) == num_heads &&
      page_router_partials.size(3) == 4 && page_router_partials.dtype() == dl_float32)
      << "page_router_partials must be float32 [event_capacity, page_size, num_heads, 4]";
  TVM_FFI_ICHECK(routing_thresholds.ndim() == 1 && routing_thresholds.size(0) == 4 &&
                 routing_thresholds.dtype() == dl_float32)
      << "routing_thresholds must be float32 [4]";
  TVM_FFI_ICHECK(fp8_k_global_scale.numel() == 1 && fp8_v_global_scale.numel() == 1 &&
                 fp4_k_global_scale.numel() == 1 && fp4_v_global_scale.numel() == 1)
      << "all format global scales must be scalar tensors";

  ffi::CUDADeviceGuard device_guard(k_input.device().device_id);
  cudaStream_t stream = get_stream(k_input.device());
  const int reused_capacity = reused_pages.size(0);
  constexpr int ROUTE_THREADS = 128;
  constexpr int QUANT_THREADS = 128;
  constexpr int RESET_THREADS = 256;
  const int reset_blocks = (reused_capacity + RESET_THREADS - 1) / RESET_THREADS;
  mixed_kv_reset_reused_pages_kernel<<<reset_blocks, RESET_THREADS, 0, stream>>>(
      static_cast<const int32_t*>(reused_pages.data_ptr()),
      static_cast<const int32_t*>(reused_count.data_ptr()),
      static_cast<uint8_t*>(page_format.data_ptr()),
      static_cast<float*>(page_router_stats.data_ptr()), reused_capacity);
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(k_input.dtype(), c_type, [&] {
    const dim3 row_grid(completed_capacity * page_size, num_heads);
    mixed_kv_route_rows_kernel<c_type, ROUTE_THREADS><<<row_grid, ROUTE_THREADS, 0, stream>>>(
        static_cast<const c_type*>(k_input.data_ptr()),
        static_cast<const c_type*>(v_input.data_ptr()),
        static_cast<const int32_t*>(completed_pages.data_ptr()),
        static_cast<const int32_t*>(completed_count.data_ptr()),
        static_cast<float*>(page_router_partials.data_ptr()), completed_capacity, page_size,
        num_heads, head_dim, k_input.stride(0), k_input.stride(1), k_input.stride(2),
        k_input.stride(3));
    mixed_kv_finalize_route_kernel<ROUTE_THREADS><<<completed_capacity, ROUTE_THREADS, 0, stream>>>(
        static_cast<const int32_t*>(completed_pages.data_ptr()),
        static_cast<const int32_t*>(completed_count.data_ptr()),
        static_cast<const float*>(page_router_partials.data_ptr()),
        static_cast<float*>(page_router_stats.data_ptr()), completed_capacity, page_size,
        num_heads);
    mixed_kv_quant_rows_kernel<c_type, QUANT_THREADS><<<row_grid, QUANT_THREADS, 0, stream>>>(
        static_cast<const c_type*>(k_input.data_ptr()),
        static_cast<const c_type*>(v_input.data_ptr()),
        static_cast<const int32_t*>(completed_pages.data_ptr()),
        static_cast<const int32_t*>(completed_count.data_ptr()),
        static_cast<const float*>(fp8_k_global_scale.data_ptr()),
        static_cast<const float*>(fp8_v_global_scale.data_ptr()),
        static_cast<const float*>(fp4_k_global_scale.data_ptr()),
        static_cast<const float*>(fp4_v_global_scale.data_ptr()),
        static_cast<uint8_t*>(fp8_k_output.data_ptr()),
        static_cast<uint8_t*>(fp8_v_output.data_ptr()),
        static_cast<uint8_t*>(fp8_k_block_scales.data_ptr()),
        static_cast<uint8_t*>(fp8_v_block_scales.data_ptr()),
        static_cast<uint8_t*>(fp4_k_output.data_ptr()),
        static_cast<uint8_t*>(fp4_v_output.data_ptr()),
        static_cast<uint8_t*>(fp4_k_block_scales.data_ptr()),
        static_cast<uint8_t*>(fp4_v_block_scales.data_ptr()),
        static_cast<const float*>(page_router_stats.data_ptr()),
        static_cast<const float*>(routing_thresholds.data_ptr()), completed_capacity, page_size,
        num_heads, head_dim, k_input.stride(0), k_input.stride(1), k_input.stride(2),
        k_input.stride(3), fp8_k_output.stride(0), fp8_k_output.stride(1), fp8_k_output.stride(2),
        fp8_k_output.stride(3), fp4_k_output.stride(0), fp4_k_output.stride(1),
        fp4_k_output.stride(2), fp4_k_output.stride(3), fp8_k_block_scales.stride(0),
        fp8_k_block_scales.stride(1), fp8_k_block_scales.stride(2), fp8_k_block_scales.stride(3));
    const int publish_blocks = (completed_capacity + RESET_THREADS - 1) / RESET_THREADS;
    mixed_kv_publish_pages_kernel<<<publish_blocks, RESET_THREADS, 0, stream>>>(
        static_cast<const int32_t*>(completed_pages.data_ptr()),
        static_cast<const int32_t*>(completed_count.data_ptr()),
        static_cast<uint8_t*>(page_format.data_ptr()),
        static_cast<const float*>(page_router_stats.data_ptr()),
        static_cast<const float*>(routing_thresholds.data_ptr()), completed_capacity);
    return true;
  });
}

template <typename InType, int THREADS = 128>
__global__ void mixed_kv_arena_prepare_kernel(flashinfer::KVPageStorage storage,
                                              flashinfer::KVPageArena arena, const int32_t* reused,
                                              const int32_t* reused_count, const int32_t* classes,
                                              const float* global_scales, int capacity) {
  const int event = blockIdx.x;
  if (event >= capacity || event >= *reused_count) return;
  const int page = reused[event];
  const auto source = storage.address(page);
  if (source.allocated() && source.format() == flashinfer::KVPageFormat::kA16) return;
  __shared__ uint64_t pending;
  if (threadIdx.x == 0) {
    arena.reserve_block(page / storage.pages_per_block, true);
    const auto format = flashinfer::KVPageFormat::kA16;
    pending = arena.allocate(storage.geometry.extent_bytes(format), classes[0], format).value;
    if (pending == flashinfer::kUnallocatedKVPage) atomicAdd(arena.mandatory_failures, 1ULL);
  }
  __syncthreads();
  const flashinfer::KVPageAddress destination{pending};
  if (!destination.allocated()) return;
  if (source.allocated()) {
    const auto format = source.format();
    const auto* payload = storage.data + source.offset();
    const auto* scales = payload + storage.geometry.payload_bytes(format);
    auto* output = reinterpret_cast<InType*>(storage.data + destination.offset());
    for (uint64_t i = threadIdx.x; i < storage.geometry.values(); i += THREADS) {
      __nv_fp8_e4m3 scale;
      scale.__x = scales[i / BSFP8_BLOCK_SIZE];
      float value;
      if (format == flashinfer::KVPageFormat::kBlockScaledFP8) {
        __nv_fp8_e4m3 encoded;
        encoded.__x = payload[i];
        value = float(encoded);
      } else {
        value = decode_e2m1_nibble((payload[i / 2] >> ((i % 2) * 4)) & 15);
      }
      const int kv = (i / storage.geometry.head_dim) % 2;
      const int scale_index = (static_cast<int>(format) - 1) * 2 + kv;
      const InType block_scale = InType(float(scale) * global_scales[scale_index]);
      output[i] = InType(value * mixed_kv_to_float(block_scale));
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    storage.entry(page) = destination.value;
    arena.release(source);
  }
}

template <typename InType, int THREADS = 128>
__global__ void mixed_kv_arena_write_kernel(flashinfer::KVPageStorage storage, const InType* k,
                                            const InType* v, const int64_t* slots, int tokens,
                                            int64_t k_token_stride, int64_t k_head_stride,
                                            int64_t v_token_stride, int64_t v_head_stride) {
  const int input_token = blockIdx.x;
  if (input_token >= tokens) return;
  const int64_t slot = slots[input_token];
  if (slot < 0) return;
  const int page = slot / storage.geometry.tokens;
  const int token = slot % storage.geometry.tokens;
  const int head = blockIdx.y;
  const auto address = storage.address(page);
  if (!address.allocated() || address.format() != flashinfer::KVPageFormat::kA16) return;
  auto* dst_k = reinterpret_cast<InType*>(storage.payload(address, token, head, false));
  auto* dst_v = reinterpret_cast<InType*>(storage.payload(address, token, head, true));
  for (int dim = threadIdx.x; dim < storage.geometry.head_dim; dim += THREADS) {
    dst_k[dim] = k[int64_t(input_token) * k_token_stride + head * k_head_stride + dim];
    dst_v[dim] = v[int64_t(input_token) * v_token_stride + head * v_head_stride + dim];
  }
}

template <typename Row>
__device__ __forceinline__ void mixed_kv_completed_rows(int page_size,
                                                        const int32_t* completed_count,
                                                        int capacity, Row row) {
  const int count = min(capacity, *completed_count);
  // The graph reserves token capacity. Reuse those CTAs for completed page rows
  // instead of launching page_size times as many mostly empty blocks.
  for (int64_t event_token = blockIdx.x; event_token < int64_t(count) * page_size;
       event_token += gridDim.x) {
    row(event_token / page_size, event_token % page_size);
  }
}

template <typename InType, bool PACKED_SIGNATURE, int THREADS>
__global__ void mixed_kv_arena_route_kernel(flashinfer::KVPageStorage storage,
                                            flashinfer::KVPageArena arena, const int32_t* completed,
                                            const int32_t* completed_count, const int32_t* classes,
                                            float* stats, const float* thresholds,
                                            uint64_t* pending, int32_t* finished_rows,
                                            int capacity) {
  const int event = blockIdx.x;
  if (event >= capacity || event >= *completed_count) return;
  const int page = completed[event];
  const auto source = storage.address(page);
  if (threadIdx.x == 0) {
    finished_rows[event] = 0;
    pending[event] = source.value;
  }
  if (!source.allocated() || source.format() != flashinfer::KVPageFormat::kA16) return;

  const auto* input = reinterpret_cast<const InType*>(storage.data + source.offset());
  const int64_t token_values = int64_t(storage.geometry.heads) * 2 * storage.geometry.head_dim;
  const int blocks_per_row = storage.geometry.head_dim / MIXED_KV_SIGNATURE_BLOCK_SIZE;
  const int64_t signature_blocks =
      int64_t(storage.geometry.tokens) * storage.geometry.heads * 2 * blocks_per_row;
  __shared__ float moments[4];
  mixed_kv_route_moments<THREADS>(
      (storage.geometry.tokens - 1) * token_values, signature_blocks,
      [&](int64_t linear) {
        return make_float2(mixed_kv_to_float(input[linear]),
                           mixed_kv_to_float(input[linear + token_values]));
      },
      [&](int64_t block, int lane) {
        if constexpr (PACKED_SIGNATURE) {
          return mixed_kv_to_float(input[block * MIXED_KV_SIGNATURE_BLOCK_SIZE + lane]);
        } else {
          const int64_t row = block / blocks_per_row;
          const int dim = (block % blocks_per_row) * MIXED_KV_SIGNATURE_BLOCK_SIZE + lane;
          return mixed_kv_to_float(input[row * storage.geometry.head_dim + dim]);
        }
      },
      moments);
  if (threadIdx.x != 0) return;
  const auto route = mixed_kv_route_stats(moments[0], moments[1], moments[2], moments[3]);
  stats[event * 2] = route.x;
  stats[event * 2 + 1] = route.y;
  const uint8_t format = mixed_kv_select_format(route.x, route.y, thresholds);
  if (format == 0) {
    return;
  }
  const auto encoded_format = static_cast<flashinfer::KVPageFormat>(format);
  pending[event] =
      arena.allocate(storage.geometry.extent_bytes(encoded_format), classes[format], encoded_format)
          .value;
}

template <typename InType, int THREADS = 128>
__device__ __forceinline__ void mixed_kv_arena_quant_row(flashinfer::KVPageStorage storage,
                                                         flashinfer::KVPageAddress source,
                                                         flashinfer::KVPageAddress destination,
                                                         const float* global_scales, int token) {
  const int head = blockIdx.y;
  const auto format = static_cast<uint8_t>(destination.format());
  const int subgroup = threadIdx.x / BSFP8_BLOCK_SIZE;
  const int lane = threadIdx.x % BSFP8_BLOCK_SIZE;
  const int dim_blocks = storage.geometry.head_dim / BSFP8_BLOCK_SIZE;
#pragma unroll
  for (int kv = 0; kv < 2; ++kv) {
    const auto* input = reinterpret_cast<const InType*>(storage.payload(source, token, head, kv));
    auto* payload = storage.payload(destination, token, head, kv);
    auto* scales = storage.scales(destination, token, head, kv);
    const float global_scale = global_scales[(format - 1) * 2 + kv];
    for (int base = 0; base < dim_blocks; base += THREADS / BSFP8_BLOCK_SIZE) {
      const int block = base + subgroup;
      const bool valid = block < dim_blocks;
      const int dim = block * BSFP8_BLOCK_SIZE + lane;
      const float value = valid ? mixed_kv_to_float(input[dim]) : 0.0f;
      const auto encoded = mixed_kv_quantize_block(value, global_scale, format);
      if (valid && lane == 0) scales[block] = encoded.scale;
      if (format == 1) {
        if (valid) payload[dim] = encoded.payload;
      } else {
        if (valid && lane < BSFP8_BLOCK_SIZE / 2)
          payload[block * (BSFP8_BLOCK_SIZE / 2) + lane] = encoded.payload;
      }
    }
  }
}

template <typename InType, int THREADS = 128>
__global__ void mixed_kv_arena_quant_kernel(flashinfer::KVPageStorage storage,
                                            flashinfer::KVPageArena arena, const int32_t* completed,
                                            const int32_t* completed_count, const uint64_t* pending,
                                            int32_t* finished_rows, const float* global_scales,
                                            int capacity) {
  mixed_kv_completed_rows(
      storage.geometry.tokens, completed_count, capacity, [&](int event, int token) {
        const int page = completed[event];
        const auto source = storage.address(page);
        const flashinfer::KVPageAddress destination{pending[event]};
        if (!destination.allocated() || destination.value == source.value) return;
        mixed_kv_arena_quant_row<InType, THREADS>(storage, source, destination, global_scales,
                                                  token);
        __syncthreads();
        if (threadIdx.x == 0) {
          cuda::atomic_ref<int32_t, cuda::thread_scope_device> finished(finished_rows[event]);
          const int rows = storage.geometry.tokens * storage.geometry.heads;
          // Acquire the preceding tiles' releases before publishing and
          // freeing the source. No tile waits for another CTA to run.
          if (finished.fetch_add(1, cuda::memory_order_acq_rel) == rows - 1) {
            storage.entry(page) = destination.value;
            arena.release(source);
          }
        }
      });
}

flashinfer::KVPageArena make_mixed_kv_arena(TensorView data, TensorView pages, TensorView slabs,
                                            TensorView occupied, TensorView available,
                                            TensorView hints, TensorView counters,
                                            TensorView reservations, TensorView a16_classes,
                                            int64_t slab_bytes) {
  CHECK_CUDA(data);
  TVM_FFI_ICHECK(data.ndim() == 1 && data.dtype() == dl_uint8 && data.stride(0) == 1);
  TVM_FFI_ICHECK(pages.ndim() == 2 && pages.dtype() == dl_int64 && pages.stride(0) > 0 &&
                 pages.stride(1) == 1 && pages.size(1) > 0);
  TVM_FFI_ICHECK(slabs.ndim() == 2 && slabs.size(1) == 4 && slabs.dtype() == dl_int32);
  TVM_FFI_ICHECK(slabs.stride(1) == 1 && slabs.stride(0) == 4);
  TVM_FFI_ICHECK(slab_bytes > 0 && slab_bytes % flashinfer::kKVPageAlignment == 0);
  TVM_FFI_ICHECK(data.numel() / slab_bytes == slabs.size(0) && slabs.size(0) > 0);
  TVM_FFI_ICHECK(occupied.ndim() == 2 && occupied.dtype() == dl_int64 &&
                 occupied.size(0) == slabs.size(0));
  TVM_FFI_ICHECK(occupied.stride(1) == 1 && occupied.stride(0) == occupied.size(1));
  TVM_FFI_ICHECK(available.ndim() == 2 && available.dtype() == dl_int64 && available.size(0) >= 2);
  TVM_FFI_ICHECK(available.size(1) == (slabs.size(0) + 63) / 64);
  TVM_FFI_ICHECK(available.stride(1) == 1 && available.stride(0) == available.size(1));
  TVM_FFI_ICHECK(hints.ndim() == 1 && hints.dtype() == dl_int32 &&
                 hints.numel() == available.size(0) && hints.stride(0) == 1);
  TVM_FFI_ICHECK(counters.ndim() == 1 && counters.dtype() == dl_int64 && counters.numel() > 4 &&
                 counters.stride(0) == 1);
  TVM_FFI_ICHECK(reservations.ndim() == 1 && reservations.dtype() == dl_int32 &&
                 reservations.numel() == pages.size(0) && reservations.stride(0) == 1);
  TVM_FFI_ICHECK(a16_classes.ndim() == 1 && a16_classes.dtype() == dl_int32 &&
                 a16_classes.numel() == available.size(0) - 1 && a16_classes.stride(0) == 1);
  TVM_FFI_ICHECK(slab_bytes <= UINT32_MAX && slabs.size(0) <= UINT32_MAX &&
                 pages.stride(0) <= UINT32_MAX);
  for (auto tensor :
       {pages, slabs, occupied, available, hints, counters, reservations, a16_classes}) {
    TVM_FFI_ICHECK(tensor.device().device_type == data.device().device_type &&
                   tensor.device().device_id == data.device().device_id);
  }
  flashinfer::KVPageArena arena{static_cast<flashinfer::KVPageSlab*>(slabs.data_ptr()),
                                static_cast<uint64_t*>(occupied.data_ptr()),
                                static_cast<unsigned long long*>(available.data_ptr()),
                                static_cast<uint32_t*>(hints.data_ptr()),
                                static_cast<unsigned long long*>(counters.data_ptr()),
                                static_cast<unsigned long long*>(counters.data_ptr()) + 1,
                                static_cast<unsigned long long*>(counters.data_ptr()) + 2,
                                static_cast<unsigned long long*>(counters.data_ptr()) + 3,
                                static_cast<unsigned long long*>(counters.data_ptr()) + 4,
                                static_cast<uint32_t*>(reservations.data_ptr()),
                                static_cast<const uint32_t*>(a16_classes.data_ptr()),
                                static_cast<uint32_t>(slabs.size(0)),
                                static_cast<uint32_t>(slab_bytes),
                                static_cast<uint32_t>(occupied.size(1)),
                                static_cast<uint32_t>(available.size(0) - 1)};
  return arena;
}

// Copy a committed encoding verbatim. A later write promotes only that private
// page to A16; compressed prefixes never need a second persistent encoding.
template <bool COPY, int THREADS = 128>
__global__ void mixed_kv_arena_blocks_kernel(uint8_t* data, uint64_t* pages, uint32_t page_stride,
                                             flashinfer::KVPageArena arena,
                                             const int64_t* destinations,
                                             int64_t destination_stride, const int64_t* sources,
                                             int64_t source_stride, bool reserve) {
  const int event = blockIdx.x;
  const int column = blockIdx.y;
  const int64_t destination_block = destinations[int64_t(event) * destination_stride];
  if constexpr (COPY) {
    if (destination_block == sources[int64_t(event) * source_stride]) return;
  }
  auto* destination_entry = pages + destination_block * page_stride + column;
  __shared__ uint64_t destination;
  __shared__ uint64_t source;
  __shared__ uint32_t bytes;
  if (threadIdx.x == 0) {
    if (column == 0) arena.reserve_block(destination_block, reserve);
    auto const previous = atomicExch(reinterpret_cast<unsigned long long*>(destination_entry),
                                     flashinfer::kUnallocatedKVPage);
    arena.release({previous});
    if constexpr (COPY) {
      const int64_t source_block = sources[int64_t(event) * source_stride];
      const flashinfer::KVPageAddress address{pages[source_block * page_stride + column]};
      source = address.value;
      destination = flashinfer::kUnallocatedKVPage;
      if (address.allocated()) {
        auto const& slab = arena.slabs[address.offset() / arena.slab_bytes];
        bytes = slab.slot_bytes;
        destination = arena.allocate(bytes, slab.size_class, address.format()).value;
        if (destination == flashinfer::kUnallocatedKVPage)
          atomicAdd(arena.mandatory_failures, 1ULL);
      }
    }
  }
  if constexpr (COPY) {
    __syncthreads();
    const flashinfer::KVPageAddress src{source}, dst{destination};
    if (!dst.allocated()) return;
    auto const* input = reinterpret_cast<uint4 const*>(data + src.offset());
    auto* output = reinterpret_cast<uint4*>(data + dst.offset());
    for (uint32_t i = threadIdx.x; i < bytes / sizeof(uint4); i += THREADS) output[i] = input[i];
    __syncthreads();
    if (threadIdx.x == 0) *destination_entry = dst.value;
  }
}

void mixed_kv_arena_blocks(TensorView data, TensorView pages, TensorView slabs, TensorView occupied,
                           TensorView available, TensorView hints, TensorView counters,
                           TensorView reservations, TensorView a16_classes, int64_t slab_bytes,
                           TensorView destinations, ffi::Optional<TensorView> sources,
                           bool reserve) {
  auto arena = make_mixed_kv_arena(data, pages, slabs, occupied, available, hints, counters,
                                   reservations, a16_classes, slab_bytes);
  TVM_FFI_ICHECK(destinations.ndim() == 1 && destinations.dtype() == dl_int64 &&
                 destinations.stride(0) > 0);
  CHECK_CUDA(destinations);
  TVM_FFI_ICHECK(destinations.device().device_id == data.device().device_id);
  ffi::CUDADeviceGuard device_guard(data.device().device_id);
  auto stream = get_stream(data.device());
  if (destinations.numel() == 0) return;
  auto const grid = dim3(destinations.numel(), pages.size(1));
  if (sources.has_value()) {
    auto const src = sources.value();
    TVM_FFI_ICHECK(src.ndim() == 1 && src.dtype() == dl_int64 && src.stride(0) > 0 &&
                   src.numel() == destinations.numel());
    CHECK_CUDA(src);
    TVM_FFI_ICHECK(src.device().device_id == data.device().device_id);
    mixed_kv_arena_blocks_kernel<true><<<grid, 128, 0, stream>>>(
        static_cast<uint8_t*>(data.data_ptr()), static_cast<uint64_t*>(pages.data_ptr()),
        pages.stride(0), arena, static_cast<const int64_t*>(destinations.data_ptr()),
        destinations.stride(0), static_cast<const int64_t*>(src.data_ptr()), src.stride(0),
        reserve);
  } else {
    mixed_kv_arena_blocks_kernel<false><<<grid, 32, 0, stream>>>(
        static_cast<uint8_t*>(data.data_ptr()), static_cast<uint64_t*>(pages.data_ptr()),
        pages.stride(0), arena, static_cast<const int64_t*>(destinations.data_ptr()),
        destinations.stride(0), nullptr, 0, reserve);
  }
}

void mixed_kv_arena_update(TensorView k, TensorView v, TensorView slots, TensorView reused,
                           TensorView reused_count, TensorView completed,
                           TensorView completed_count, TensorView data, TensorView pages,
                           TensorView slabs, TensorView occupied, TensorView available,
                           TensorView hints, TensorView counters, TensorView reservations,
                           TensorView a16_classes, TensorView classes, TensorView finished_rows,
                           TensorView stats, TensorView pending, TensorView global_scales,
                           TensorView thresholds, int64_t page_size, int64_t slab_bytes) {
  CHECK_CUDA(k);
  CHECK_CUDA(v);
  CHECK_DIM(3, k);
  CHECK_DIM(3, v);
  TVM_FFI_ICHECK(k.size(0) == v.size(0) && k.size(1) == v.size(1) && k.size(2) == v.size(2));
  TVM_FFI_ICHECK(k.stride(2) == 1 && v.stride(2) == 1 && k.dtype() == v.dtype());
  TVM_FFI_ICHECK(page_size > 0 && k.size(1) > 0 && k.size(2) > 0 && k.size(2) % 16 == 0);
  TVM_FFI_ICHECK(slots.ndim() == 1 && slots.dtype() == dl_int64 && slots.stride(0) == 1);
  TVM_FFI_ICHECK(slots.size(0) <= k.size(0));
  TVM_FFI_ICHECK(classes.ndim() == 1 && classes.dtype() == dl_int32 && classes.numel() == 3 &&
                 classes.stride(0) == 1);
  for (auto list : {reused, completed}) {
    TVM_FFI_ICHECK(list.ndim() == 1 && list.dtype() == dl_int32 && list.stride(0) == 1);
  }
  for (auto count : {reused_count, completed_count}) {
    TVM_FFI_ICHECK(count.numel() == 1 && count.dtype() == dl_int32);
  }
  TVM_FFI_ICHECK(finished_rows.ndim() == 1 && finished_rows.dtype() == dl_int32 &&
                 finished_rows.numel() == completed.numel() && finished_rows.stride(0) == 1);
  TVM_FFI_ICHECK(stats.ndim() == 2 && stats.dtype() == dl_float32 &&
                 stats.size(0) == completed.numel() && stats.size(1) == 2);
  TVM_FFI_ICHECK(stats.stride(1) == 1 && stats.stride(0) == 2);
  TVM_FFI_ICHECK(pending.ndim() == 1 && pending.dtype() == dl_int64 &&
                 pending.numel() == completed.numel() && pending.stride(0) == 1);
  TVM_FFI_ICHECK(global_scales.numel() == 4 && global_scales.ndim() == 1 &&
                 global_scales.dtype() == dl_float32 && global_scales.stride(0) == 1);
  TVM_FFI_ICHECK(thresholds.numel() == 4 && thresholds.ndim() == 1 &&
                 thresholds.dtype() == dl_float32 && thresholds.stride(0) == 1);
  for (auto tensor : {v, slots, reused, reused_count, completed, completed_count, data, pages,
                      slabs, occupied, available, hints, counters, classes, finished_rows, stats,
                      pending, global_scales, thresholds}) {
    TVM_FFI_ICHECK(tensor.device().device_type == k.device().device_type &&
                   tensor.device().device_id == k.device().device_id);
  }
  flashinfer::KVPageStorage storage{
      static_cast<uint8_t*>(data.data_ptr()),
      static_cast<uint64_t*>(pages.data_ptr()),
      static_cast<uint32_t>(pages.stride(0)),
      static_cast<uint32_t>(pages.size(1)),
      {static_cast<uint32_t>(page_size), static_cast<uint32_t>(k.size(1)),
       static_cast<uint32_t>(k.size(2))}};
  TVM_FFI_ICHECK(storage.geometry.extent_bytes(flashinfer::KVPageFormat::kA16) <=
                 uint64_t(slab_bytes));
  TVM_FFI_ICHECK(slab_bytes <= UINT32_MAX && slabs.size(0) <= UINT32_MAX &&
                 pages.stride(0) <= UINT32_MAX);
  auto arena = make_mixed_kv_arena(data, pages, slabs, occupied, available, hints, counters,
                                   reservations, a16_classes, slab_bytes);
  ffi::CUDADeviceGuard device_guard(k.device().device_id);
  cudaStream_t stream = get_stream(k.device());
  const auto* reset = static_cast<const int32_t*>(reused.data_ptr());
  const auto* reset_count = static_cast<const int32_t*>(reused_count.data_ptr());
  const auto* sealed = static_cast<const int32_t*>(completed.data_ptr());
  const auto* sealed_count = static_cast<const int32_t*>(completed_count.data_ptr());
  const auto* size_classes = static_cast<const int32_t*>(classes.data_ptr());
  auto* route_stats = static_cast<float*>(stats.data_ptr());
  auto* row_completions = static_cast<int32_t*>(finished_rows.data_ptr());
  auto* destinations = static_cast<uint64_t*>(pending.data_ptr());
  const int capacity = completed.numel();
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(k.dtype(), c_type, [&] {
    if (reused.numel()) {
      mixed_kv_arena_prepare_kernel<c_type><<<reused.numel(), 128, 0, stream>>>(
          storage, arena, reset, reset_count, size_classes,
          static_cast<const float*>(global_scales.data_ptr()), reused.numel());
    }
    if (slots.numel()) {
      mixed_kv_arena_write_kernel<c_type><<<dim3(slots.numel(), k.size(1)), 128, 0, stream>>>(
          storage, static_cast<const c_type*>(k.data_ptr()),
          static_cast<const c_type*>(v.data_ptr()), static_cast<const int64_t*>(slots.data_ptr()),
          slots.numel(), k.stride(0), k.stride(1), v.stride(0), v.stride(1));
    }
    if (capacity) {
      const dim3 rows(capacity, k.size(1));
      DISPATCH_BOOL(k.size(2) % MIXED_KV_SIGNATURE_BLOCK_SIZE == 0, PACKED_SIGNATURE, [&] {
        constexpr int route_threads = 1024;
        mixed_kv_arena_route_kernel<c_type, PACKED_SIGNATURE, route_threads>
            <<<capacity, route_threads, 0, stream>>>(
                storage, arena, sealed, sealed_count, size_classes, route_stats,
                static_cast<const float*>(thresholds.data_ptr()), destinations, row_completions,
                capacity);
        return true;
      });
      mixed_kv_arena_quant_kernel<c_type><<<rows, 128, 0, stream>>>(
          storage, arena, sealed, sealed_count, destinations, row_completions,
          static_cast<const float*>(global_scales.data_ptr()), capacity);
    }
    return true;
  });
}

__global__ void mixed_kv_arena_capacity_kernel(flashinfer::KVPageArena arena,
                                               const uint32_t* a16_bytes, int kinds,
                                               int64_t* output) {
  const int kind = blockIdx.x;
  unsigned long long free = 0;
  for (uint32_t i = threadIdx.x; i < arena.slab_count; i += blockDim.x) {
    const auto slab = arena.slabs[i];
    if (kind == kinds) {
      free += slab.used == 0;
    } else if (slab.used && arena.a16_classes[slab.size_class] == uint32_t(kind) &&
               slab.slot_bytes == a16_bytes[kind]) {
      free += arena.slab_bytes / slab.slot_bytes - slab.used;
    }
  }
  free = mixed_kv_block_sum<256>(free);
  if (threadIdx.x == 0) {
    output[kind == kinds ? 4 : 5 + kind] = free;
    if (kind != kinds) output[5 + kinds + kind] = arena.page_counts[kind];
    if (kind == 0) {
      output[0] = *arena.allocated_bytes;
      output[1] = *arena.allocation_failures;
      output[2] = *arena.mandatory_failures;
      output[3] = *arena.reserved_blocks;
    }
  }
}

void mixed_kv_arena_capacity(TensorView data, TensorView pages, TensorView slabs,
                             TensorView occupied, TensorView available, TensorView hints,
                             TensorView counters, TensorView reservations, TensorView a16_classes,
                             int64_t slab_bytes, TensorView a16_bytes, TensorView output) {
  auto arena = make_mixed_kv_arena(data, pages, slabs, occupied, available, hints, counters,
                                   reservations, a16_classes, slab_bytes);
  const int kinds = counters.numel() - 4;
  TVM_FFI_ICHECK(a16_bytes.ndim() == 1 && a16_bytes.dtype() == dl_int32 &&
                 a16_bytes.numel() == kinds && a16_bytes.stride(0) == 1);
  TVM_FFI_ICHECK(output.ndim() == 1 && output.dtype() == dl_int64 &&
                 output.numel() == 5 + 2 * kinds && output.stride(0) == 1);
  for (auto tensor : {a16_bytes, output}) {
    TVM_FFI_ICHECK(tensor.device().device_type == data.device().device_type &&
                   tensor.device().device_id == data.device().device_id);
  }
  ffi::CUDADeviceGuard guard(data.device().device_id);
  mixed_kv_arena_capacity_kernel<<<kinds + 1, 256, 0, get_stream(data.device())>>>(
      arena, static_cast<const uint32_t*>(a16_bytes.data_ptr()), kinds,
      static_cast<int64_t*>(output.data_ptr()));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_kv_quant, nvfp4_kv_quant);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bsfp8_kv_quant, bsfp8_kv_quant);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mixed_kv_quant_pages, mixed_kv_quant_pages);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mixed_kv_arena_update, mixed_kv_arena_update);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mixed_kv_arena_blocks, mixed_kv_arena_blocks);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mixed_kv_arena_capacity, mixed_kv_arena_capacity);
