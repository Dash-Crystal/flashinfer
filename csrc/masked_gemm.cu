/*
 * Copyright (c) 2026 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <algorithm>
#include <flashinfer/gemm/masked_gemm.cuh>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

void PrepareMaskedGemm(int64_t device, bool bf16, bool publish) {
  ffi::CUDADeviceGuard guard(device);
  cudaError_t status;
  if (publish) {
    status = bf16 ? flashinfer::masked_gemm::Prepare<cutlass::bfloat16_t, true>()
                  : flashinfer::masked_gemm::Prepare<cutlass::half_t, true>();
  } else {
    status = bf16 ? flashinfer::masked_gemm::Prepare<cutlass::bfloat16_t, false>()
                  : flashinfer::masked_gemm::Prepare<cutlass::half_t, false>();
  }
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

void RunMaskedGemm(TensorView x, TensorView weight, TensorView out, TensorView is_padding,
                   int64_t row_offset, Optional<TensorView> peer_output) {
  ffi::CUDADeviceGuard guard(x.device().device_id);
  const auto stream = get_stream(x.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(x.dtype(), c_type, [&] {
    using Element = std::conditional_t<std::is_same_v<c_type, nv_bfloat16>, cutlass::bfloat16_t,
                                       cutlass::half_t>;
    auto run = peer_output.has_value() ? flashinfer::masked_gemm::Run<Element, true>
                                       : flashinfer::masked_gemm::Run<Element, false>;
    auto status = run(
        static_cast<Element*>(x.data_ptr()), static_cast<Element*>(weight.data_ptr()),
        static_cast<Element*>(out.data_ptr()), x.size(0), weight.size(1), x.size(1), x.stride(0),
        weight.stride(1), out.stride(0), static_cast<const uint8_t*>(is_padding.data_ptr()),
        row_offset,
        peer_output.has_value() ? static_cast<Element*>(peer_output.value().data_ptr()) : nullptr,
        stream);
    TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
    return true;
  });
}

int64_t PrepareReadyGemm(int64_t device, bool bf16, int64_t k, int64_t n, int64_t reserved_blocks) {
  ffi::CUDADeviceGuard guard(device);
  auto status = bf16 ? flashinfer::masked_gemm::PrepareReady<cutlass::bfloat16_t>()
                     : flashinfer::masked_gemm::PrepareReady<cutlass::half_t>();
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  int sms, l2_bytes;
  status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  status = cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  TVM_FFI_ICHECK(k > 0 && n > 0 && reserved_blocks > 0 && reserved_blocks < sms);
  const int64_t columns = flashinfer::masked_gemm::ReadyTile::kN;
  // Leave a quarter of L2 for activations and the concurrent producer.
  const int64_t cache_blocks =
      std::max<int64_t>(1, (int64_t(l2_bytes) * 3 / 4) / (k * columns * 2));
  return std::min({sms - reserved_blocks, (n + columns - 1) / columns, cache_blocks});
}

void RunReadyGemm(TensorView x, TensorView weight, TensorView out, TensorView readiness,
                  int64_t group_rows, int64_t blocks) {
  ffi::CUDADeviceGuard guard(x.device().device_id);
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(x.dtype(), c_type, [&] {
    using Element = std::conditional_t<std::is_same_v<c_type, nv_bfloat16>, cutlass::bfloat16_t,
                                       cutlass::half_t>;
    auto status = flashinfer::masked_gemm::RunReady(
        static_cast<Element*>(x.data_ptr()), static_cast<Element*>(weight.data_ptr()),
        static_cast<Element*>(out.data_ptr()), x.size(0), weight.size(1), x.size(1), x.stride(0),
        weight.stride(1), out.stride(0), static_cast<int*>(readiness.data_ptr()), group_rows,
        readiness.size(0) - 1, blocks, get_stream(x.device()));
    TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
    return true;
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare, PrepareMaskedGemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, RunMaskedGemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare_ready, PrepareReadyGemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_ready, RunReadyGemm);
