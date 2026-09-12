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
#include <tvm/ffi/container/array.h>

#include <algorithm>
#include <flashinfer/gemm/masked_gemm.cuh>

#if FLASHINFER_READY_TMA_SM120
#include <flashinfer/gemm/ready_tma_gemm.cuh>
namespace ready_gemm = flashinfer::ready_tma_gemm;
#else
namespace ready_gemm = flashinfer::masked_gemm;
#endif

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

void PrepareMaskedGemm(int64_t device, bool bf16, bool publish, bool tile_publication) {
  ffi::CUDADeviceGuard guard(device);
  cudaError_t status;
  if (tile_publication) {
    status = bf16 ? flashinfer::masked_gemm::Prepare<cutlass::bfloat16_t, true, true>()
                  : flashinfer::masked_gemm::Prepare<cutlass::half_t, true, true>();
  } else if (publish) {
    status = bf16 ? flashinfer::masked_gemm::Prepare<cutlass::bfloat16_t, true>()
                  : flashinfer::masked_gemm::Prepare<cutlass::half_t, true>();
  } else {
    status = bf16 ? flashinfer::masked_gemm::Prepare<cutlass::bfloat16_t, false>()
                  : flashinfer::masked_gemm::Prepare<cutlass::half_t, false>();
  }
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

void RunMaskedGemm(TensorView x, TensorView weight, TensorView out, TensorView is_padding,
                   int64_t row_offset, Optional<TensorView> peer_output,
                   Optional<TensorView> publication, Optional<TensorView> peer_publication) {
  ffi::CUDADeviceGuard guard(x.device().device_id);
  const auto stream = get_stream(x.device());
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(x.dtype(), c_type, [&] {
    using Element = std::conditional_t<std::is_same_v<c_type, nv_bfloat16>, cutlass::bfloat16_t,
                                       cutlass::half_t>;
    auto run = publication.has_value()   ? flashinfer::masked_gemm::Run<Element, true, true>
               : peer_output.has_value() ? flashinfer::masked_gemm::Run<Element, true>
                                         : flashinfer::masked_gemm::Run<Element, false>;
    auto status = run(
        static_cast<Element*>(x.data_ptr()), static_cast<Element*>(weight.data_ptr()),
        static_cast<Element*>(out.data_ptr()), x.size(0), weight.size(1), x.size(1), x.stride(0),
        weight.stride(1), out.stride(0), static_cast<const uint8_t*>(is_padding.data_ptr()),
        row_offset,
        peer_output.has_value() ? static_cast<Element*>(peer_output.value().data_ptr()) : nullptr,
        publication.has_value() ? static_cast<int*>(publication.value().data_ptr()) : nullptr,
        peer_publication.has_value() ? static_cast<int*>(peer_publication.value().data_ptr())
                                     : nullptr,
        publication.has_value() ? publication.value().size(1) : 0, stream);
    TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
    return true;
  });
}

tvm::ffi::Array<int64_t> PrepareReadyGemm(int64_t device, bool bf16, int64_t producer_registers,
                                          int64_t producer_shared, int64_t producer_threads) {
  ffi::CUDADeviceGuard guard(device);
  int occupancy;
  auto status = bf16 ? ready_gemm::PrepareReady<cutlass::bfloat16_t>(&occupancy)
                     : ready_gemm::PrepareReady<cutlass::half_t>(&occupancy);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  cudaDeviceProp properties;
  status = cudaGetDeviceProperties(&properties, device);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  const int sms = properties.multiProcessorCount;
  const int slots = sms * occupancy;
  const size_t bytes = bf16 ? ready_gemm::ReadyWorkspaceBytes<cutlass::bfloat16_t>(slots)
                            : ready_gemm::ReadyWorkspaceBytes<cutlass::half_t>(slots);
  int math_registers = 0;
  bool concurrent = false;
#if FLASHINFER_READY_TMA_SM120
  if (producer_threads > 0) {
    status = bf16 ? ready_gemm::PlanReady<cutlass::bfloat16_t>(properties, producer_registers,
                                                               producer_shared, producer_threads,
                                                               &math_registers, &concurrent)
                  : ready_gemm::PlanReady<cutlass::half_t>(properties, producer_registers,
                                                           producer_shared, producer_threads,
                                                           &math_registers, &concurrent);
    TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  }
#endif
  return {sms,
          occupancy,
          ready_gemm::ReadyTile::kM,
          int64_t(bytes),
          math_registers,
          int64_t(concurrent)};
}

void RunReadyGemm(TensorView x, TensorView weight, TensorView out, TensorView readiness,
                  TensorView workspace, int64_t group_rows, int64_t sms, int64_t occupancy,
                  int64_t reserved_blocks) {
  ffi::CUDADeviceGuard guard(x.device().device_id);
  TVM_FFI_ICHECK(reserved_blocks >= 0 && reserved_blocks < sms);
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(x.dtype(), c_type, [&] {
    using Element = std::conditional_t<std::is_same_v<c_type, nv_bfloat16>, cutlass::bfloat16_t,
                                       cutlass::half_t>;
    TVM_FFI_ICHECK(workspace.size(0) >= ready_gemm::ReadyWorkspaceBytes<Element>(sms * occupancy));
    auto status = ready_gemm::RunReady(
        static_cast<Element*>(x.data_ptr()), static_cast<Element*>(weight.data_ptr()),
        static_cast<Element*>(out.data_ptr()), x.size(0), weight.size(1), x.size(1), x.stride(0),
        weight.stride(1), out.stride(0), static_cast<int*>(readiness.data_ptr()), group_rows,
        readiness.size(0) - 1, workspace.data_ptr(), sms, occupancy, sms - reserved_blocks,
        get_stream(x.device()));
    TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
    return true;
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare, PrepareMaskedGemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, RunMaskedGemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare_ready, PrepareReadyGemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_ready, RunReadyGemm);
