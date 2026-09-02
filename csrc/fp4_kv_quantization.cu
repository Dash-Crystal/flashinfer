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
      block_max = fmaxf(
          block_max,
          __shfl_xor_sync(uint32_t(-1), block_max, offset, BSFP8_BLOCK_SIZE));
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
    bsfp8_quant_kernel<c_type, THREADS><<<M, THREADS, 0, stream>>>(
        static_cast<const c_type*>(input.data_ptr()),
        static_cast<const float*>(global_scale.data_ptr()),
        static_cast<uint8_t*>(fp8_output.data_ptr()),
        static_cast<uint8_t*>(block_scales.data_ptr()), M, K);
    return true;
  });
}

__global__ void mixed_kv_reset_reused_pages_kernel(
    const int32_t* __restrict__ reused_pages, const int32_t* __restrict__ reused_count,
    uint8_t* __restrict__ page_format, float* __restrict__ page_router_stats,
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

template <int THREADS>
__device__ __forceinline__ float mixed_kv_block_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(uint32_t(-1), value, offset);
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
      value += __shfl_down_sync(uint32_t(-1), value, offset);
    }
  }
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

template <typename InType, int THREADS = 128>
__global__ void mixed_kv_route_rows_kernel(
    const InType* __restrict__ k_input, const InType* __restrict__ v_input,
    const int32_t* __restrict__ completed_pages,
    const int32_t* __restrict__ completed_count,
    float* __restrict__ page_router_partials,
    const int completed_capacity, const int page_size, const int num_heads, const int head_dim,
    const int64_t in_stride_page, const int64_t in_stride_token,
    const int64_t in_stride_head, const int64_t in_stride_dim) {
  const int event_token = blockIdx.x;
  const int event = event_token / page_size;
  if (event >= completed_capacity || event >= *completed_count) return;
  const int token = event_token - event * page_size;
  const int head = blockIdx.y;
  const int32_t page = completed_pages[event];

  float neighbor_dot = 0.0f;
  float neighbor_left_sq = 0.0f;
  float neighbor_right_sq = 0.0f;
  if (token + 1 < page_size) {
    for (int linear = threadIdx.x; linear < 2 * head_dim; linear += THREADS) {
      const bool is_v = linear >= head_dim;
      const int dim = is_v ? linear - head_dim : linear;
      const InType* input = is_v ? v_input : k_input;
      const int64_t offset = page * in_stride_page + token * in_stride_token +
                             head * in_stride_head + dim * in_stride_dim;
      const float left = mixed_kv_to_float(input[offset]);
      const float right = mixed_kv_to_float(input[offset + in_stride_token]);
      neighbor_dot = fmaf(left, right, neighbor_dot);
      neighbor_left_sq = fmaf(left, left, neighbor_left_sq);
      neighbor_right_sq = fmaf(right, right, neighbor_right_sq);
    }
  }
  neighbor_dot = mixed_kv_block_sum<THREADS>(neighbor_dot);
  neighbor_left_sq = mixed_kv_block_sum<THREADS>(neighbor_left_sq);
  neighbor_right_sq = mixed_kv_block_sum<THREADS>(neighbor_right_sq);

  static_assert(THREADS % 32 == 0);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float peak_rms = 0.0f;
  const int blocks_per_kv = head_dim / MIXED_KV_SIGNATURE_BLOCK_SIZE;
  for (int block = warp; block < 2 * blocks_per_kv; block += THREADS / 32) {
    const bool is_v = block >= blocks_per_kv;
    const int dim_block = is_v ? block - blocks_per_kv : block;
    const InType* input = is_v ? v_input : k_input;
    const int dim = dim_block * MIXED_KV_SIGNATURE_BLOCK_SIZE + lane;
    const int64_t offset = page * in_stride_page + token * in_stride_token +
                           head * in_stride_head + dim * in_stride_dim;
    const float value = mixed_kv_to_float(input[offset]);
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
    const int64_t partial =
        (static_cast<int64_t>(event_token) * num_heads + head) * 4;
    page_router_partials[partial + 0] = neighbor_dot;
    page_router_partials[partial + 1] = neighbor_left_sq;
    page_router_partials[partial + 2] = neighbor_right_sq;
    page_router_partials[partial + 3] = peak_rms;
  }
}

template <int THREADS = 128>
__global__ void mixed_kv_finalize_route_kernel(
    const int32_t* __restrict__ completed_pages,
    const int32_t* __restrict__ completed_count,
    const float* __restrict__ page_router_partials,
    float* __restrict__ page_router_stats,
    const int completed_capacity, const int page_size, const int num_heads) {
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
    const float denom = sqrtf(left_sq * right_sq);
    page_router_stats[page * 2] = denom == 0.0f ? 0.0f : dot / denom;
    page_router_stats[page * 2 + 1] = peak_rms;
  }
}

__device__ __forceinline__ uint8_t mixed_kv_select_format(
    const float* __restrict__ page_router_stats, const int32_t page,
    const float* __restrict__ routing_thresholds) {
  const float neighbor_cos = page_router_stats[page * 2];
  const float peak_rms = page_router_stats[page * 2 + 1];
  const bool fp4 = neighbor_cos <= routing_thresholds[0] &&
                   peak_rms <= routing_thresholds[1];
  const bool fp8 = neighbor_cos <= routing_thresholds[2] &&
                   peak_rms <= routing_thresholds[3];
  return fp4 ? 2 : (fp8 ? 1 : 0);
}

// Match the existing bsfp8_quant_kernel / NVIDIA NVFP4 producer geometry:
// one 16-lane subgroup owns one scale block.  A CTA covers eight adjacent
// blocks, so scale selection and payload stores stay coalesced and no lane
// serializes an entire coefficient block.
template <typename InType, int THREADS = 128>
__global__ void mixed_kv_quant_rows_kernel(
    const InType* __restrict__ k_input, const InType* __restrict__ v_input,
    const int32_t* __restrict__ completed_pages,
    const int32_t* __restrict__ completed_count,
    const float* __restrict__ fp8_k_global_scale_ptr,
    const float* __restrict__ fp8_v_global_scale_ptr,
    const float* __restrict__ fp4_k_global_scale_ptr,
    const float* __restrict__ fp4_v_global_scale_ptr,
    uint8_t* __restrict__ fp8_k_output, uint8_t* __restrict__ fp8_v_output,
    uint8_t* __restrict__ fp8_k_block_scales, uint8_t* __restrict__ fp8_v_block_scales,
    uint8_t* __restrict__ fp4_k_output, uint8_t* __restrict__ fp4_v_output,
    uint8_t* __restrict__ fp4_k_block_scales, uint8_t* __restrict__ fp4_v_block_scales,
    const float* __restrict__ page_router_stats,
    const float* __restrict__ routing_thresholds,
    const int completed_capacity, const int page_size, const int num_heads, const int head_dim,
    const int64_t in_stride_page, const int64_t in_stride_token,
    const int64_t in_stride_head, const int64_t in_stride_dim,
    const int64_t fp8_stride_page, const int64_t fp8_stride_token,
    const int64_t fp8_stride_head, const int64_t fp8_stride_dim,
    const int64_t fp4_stride_page, const int64_t fp4_stride_token,
    const int64_t fp4_stride_head, const int64_t fp4_stride_dim,
    const int64_t sf_stride_page, const int64_t sf_stride_token,
    const int64_t sf_stride_head, const int64_t sf_stride_dim) {
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
    const float* scale_ptrs[4] = {
        fp8_k_global_scale_ptr, fp8_v_global_scale_ptr,
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
    const float global_scale =
        global_scales[(selected_format == 2 ? 2 : 0) + kv];

    for (int dim_block = subgroup; dim_block < dim_blocks;
         dim_block += GROUPS_PER_CTA) {
      const int dim = dim_block * BSFP8_BLOCK_SIZE + lane;
      const int64_t input_offset = page * in_stride_page + token * in_stride_token +
                                   head * in_stride_head + dim * in_stride_dim;
      const float value = mixed_kv_to_float(input[input_offset]);
      float block_max = fabsf(value);
#pragma unroll
      for (int offset = BSFP8_BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        block_max = fmaxf(
            block_max,
            __shfl_xor_sync(uint32_t(-1), block_max, offset,
                            BSFP8_BLOCK_SIZE));
      }

      const float format_max = selected_format == 2 ? 6.0f : 448.0f;
      const float scale_max = selected_format == 2 ? 448.0f : BSFP8_A16_SCALE_MAX;
      float sf_value = 0.0f;
      if (block_max != 0.0f) {
        const float required_sf =
            block_max * reciprocal_approximate_ftz(global_scale) *
            reciprocal_approximate_ftz(format_max);
        constexpr float factors[9] = {
            0.75f, 0.8125f, 0.875f, 0.9375f, 1.0f,
            1.0625f, 1.125f, 1.25f, 1.5f,
        };
        float best_objective = __int_as_float(0x7f800000);
#pragma unroll
        for (int candidate = 0; candidate < 9; ++candidate) {
          float candidate_sf =
              fminf(fmaxf(required_sf * factors[candidate], 0x1p-9f), scale_max);
          __nv_fp8_e4m3 candidate_sf_fp8 = __nv_fp8_e4m3(candidate_sf);
          candidate_sf = static_cast<float>(candidate_sf_fp8);
          const float encode_scale =
              reciprocal_approximate_ftz(global_scale * candidate_sf);
          float reconstructed;
          if (selected_format == 2) {
            reconstructed =
                decode_e2m1_nibble(encode_e2m1_nibble(value * encode_scale)) *
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
            sum_squared += __shfl_xor_sync(uint32_t(-1), sum_squared, offset,
                                           BSFP8_BLOCK_SIZE);
            max_residual = fmaxf(
                max_residual,
                __shfl_xor_sync(uint32_t(-1), max_residual, offset,
                                BSFP8_BLOCK_SIZE));
          }
          const float objective = sum_squared / float(BSFP8_BLOCK_SIZE) +
                                  0.05f * max_residual * max_residual;
          if (objective < best_objective) {
            best_objective = objective;
            sf_value = candidate_sf;
          }
        }
      }
      __nv_fp8_e4m3 sf_fp8 = __nv_fp8_e4m3(sf_value);
      const int64_t sf_offset = page * sf_stride_page + token * sf_stride_token +
                                head * sf_stride_head + dim_block * sf_stride_dim;
      if (lane == 0) {
        if (selected_format == 2) {
          fp4_scales[sf_offset] = sf_fp8.__x;
        } else {
          fp8_scales[sf_offset] = sf_fp8.__x;
        }
      }
      const float encode_scale =
          sf_value == 0.0f ? 0.0f : reciprocal_approximate_ftz(global_scale * sf_value);
      if (selected_format == 1) {
        __nv_fp8_e4m3 encoded = __nv_fp8_e4m3(value * encode_scale);
        fp8_output[page * fp8_stride_page + token * fp8_stride_token +
                   head * fp8_stride_head + dim * fp8_stride_dim] = encoded.__x;
      } else {
        const int pair_lane = lane & (BSFP8_BLOCK_SIZE / 2 - 1);
        const float low = __shfl_sync(uint32_t(-1), value, pair_lane * 2,
                                      BSFP8_BLOCK_SIZE);
        const float high = __shfl_sync(uint32_t(-1), value, pair_lane * 2 + 1,
                                       BSFP8_BLOCK_SIZE);
        if (lane < BSFP8_BLOCK_SIZE / 2) {
          const uint8_t packed = encode_e2m1_nibble(low * encode_scale) |
                                 (encode_e2m1_nibble(high * encode_scale) << 4);
          const int packed_dim = dim_block * (BSFP8_BLOCK_SIZE / 2) + lane;
          const int64_t output_offset = page * fp4_stride_page + token * fp4_stride_token +
                                        head * fp4_stride_head + packed_dim * fp4_stride_dim;
          fp4_output[output_offset] = packed;
        }
      }
    }
  }
}

__global__ void mixed_kv_publish_pages_kernel(
    const int32_t* __restrict__ completed_pages,
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
    TensorView fp8_v_global_scale, TensorView fp4_k_global_scale,
    TensorView fp4_v_global_scale, TensorView fp8_k_output, TensorView fp8_v_output,
    TensorView fp8_k_block_scales, TensorView fp8_v_block_scales, TensorView fp4_k_output,
    TensorView fp4_v_output, TensorView fp4_k_block_scales, TensorView fp4_v_block_scales,
    TensorView page_router_partials, TensorView page_format, TensorView page_router_stats,
    TensorView routing_thresholds) {
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
  TVM_FFI_ICHECK(k_input.size(0) == v_input.size(0) &&
                 k_input.size(1) == v_input.size(1) &&
                 k_input.size(2) == v_input.size(2) &&
                 k_input.size(3) == v_input.size(3))
      << "K and V input shapes must match";
  TVM_FFI_ICHECK(k_input.stride(0) == v_input.stride(0) &&
                 k_input.stride(1) == v_input.stride(1) &&
                 k_input.stride(2) == v_input.stride(2) &&
                 k_input.stride(3) == v_input.stride(3))
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
  TVM_FFI_ICHECK(fp8_k_block_scales.ndim() == 4 && fp8_v_block_scales.ndim() == 4 &&
                 fp4_k_block_scales.ndim() == 4 && fp4_v_block_scales.ndim() == 4 &&
                 fp8_k_block_scales.size(0) == num_pages &&
                 fp8_k_block_scales.size(1) == page_size &&
                 fp8_k_block_scales.size(2) == num_heads &&
                 fp8_k_block_scales.size(3) == head_dim / BSFP8_BLOCK_SIZE &&
                 fp8_v_block_scales.size(0) == num_pages &&
                 fp8_v_block_scales.size(1) == page_size &&
                 fp8_v_block_scales.size(2) == num_heads &&
                 fp8_v_block_scales.size(3) == head_dim / BSFP8_BLOCK_SIZE &&
                 fp4_k_block_scales.size(0) == num_pages &&
                 fp4_k_block_scales.size(1) == page_size &&
                 fp4_k_block_scales.size(2) == num_heads &&
                 fp4_k_block_scales.size(3) == head_dim / BSFP8_BLOCK_SIZE &&
                 fp4_v_block_scales.size(0) == num_pages &&
                 fp4_v_block_scales.size(1) == page_size &&
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
  TVM_FFI_ICHECK(page_router_partials.ndim() == 4 &&
                 page_router_partials.size(0) == completed_capacity &&
                 page_router_partials.size(1) == page_size &&
                 page_router_partials.size(2) == num_heads &&
                 page_router_partials.size(3) == 4 &&
                 page_router_partials.dtype() == dl_float32)
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
    mixed_kv_route_rows_kernel<c_type, ROUTE_THREADS>
        <<<row_grid, ROUTE_THREADS, 0, stream>>>(
        static_cast<const c_type*>(k_input.data_ptr()),
        static_cast<const c_type*>(v_input.data_ptr()),
        static_cast<const int32_t*>(completed_pages.data_ptr()),
        static_cast<const int32_t*>(completed_count.data_ptr()),
        static_cast<float*>(page_router_partials.data_ptr()), completed_capacity, page_size,
        num_heads, head_dim, k_input.stride(0), k_input.stride(1), k_input.stride(2),
        k_input.stride(3));
    mixed_kv_finalize_route_kernel<ROUTE_THREADS>
        <<<completed_capacity, ROUTE_THREADS, 0, stream>>>(
        static_cast<const int32_t*>(completed_pages.data_ptr()),
        static_cast<const int32_t*>(completed_count.data_ptr()),
        static_cast<const float*>(page_router_partials.data_ptr()),
        static_cast<float*>(page_router_stats.data_ptr()), completed_capacity, page_size,
        num_heads);
    mixed_kv_quant_rows_kernel<c_type, QUANT_THREADS>
        <<<row_grid, QUANT_THREADS, 0, stream>>>(
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
        num_heads, head_dim,
        k_input.stride(0), k_input.stride(1), k_input.stride(2), k_input.stride(3),
        fp8_k_output.stride(0), fp8_k_output.stride(1), fp8_k_output.stride(2),
        fp8_k_output.stride(3), fp4_k_output.stride(0), fp4_k_output.stride(1),
        fp4_k_output.stride(2), fp4_k_output.stride(3), fp8_k_block_scales.stride(0),
        fp8_k_block_scales.stride(1), fp8_k_block_scales.stride(2),
        fp8_k_block_scales.stride(3));
    const int publish_blocks =
        (completed_capacity + RESET_THREADS - 1) / RESET_THREADS;
    mixed_kv_publish_pages_kernel<<<publish_blocks, RESET_THREADS, 0, stream>>>(
        static_cast<const int32_t*>(completed_pages.data_ptr()),
        static_cast<const int32_t*>(completed_count.data_ptr()),
        static_cast<uint8_t*>(page_format.data_ptr()),
        static_cast<const float*>(page_router_stats.data_ptr()),
        static_cast<const float*>(routing_thresholds.data_ptr()), completed_capacity);
    return true;
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_kv_quant, nvfp4_kv_quant);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bsfp8_kv_quant, bsfp8_kv_quant);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mixed_kv_quant_pages, mixed_kv_quant_pages);
