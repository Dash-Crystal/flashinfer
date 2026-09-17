/*
 * Copyright (c) 2026 by FlashInfer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <tvm/ffi/container/array.h>

#include <flashinfer/gemm/paired_geglu.cuh>
#if FLASHINFER_PAIRED_READY
#include <flashinfer/gemm/paired_ready_geglu.cuh>
#endif
#include "tvm_ffi_utils.h"

namespace paired = flashinfer::paired_geglu;

tvm::ffi::Array<int64_t> PreparePairedGeGLU(int64_t device) {
  ffi::CUDADeviceGuard guard(device);
  int registers, shared, threads;
  auto status = paired::Prepare(&registers, &shared, &threads);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  return {registers, shared, threads};
}

void RunPairedGeGLU(TensorView x, TensorView weight, TensorView out,
                    tvm::ffi::Optional<TensorView> padding) {
  ffi::CUDADeviceGuard guard(x.device().device_id);
  auto status = paired::Run(
      static_cast<paired::Element*>(x.data_ptr()), static_cast<paired::Element*>(weight.data_ptr()),
      static_cast<paired::Element*>(out.data_ptr()), x.size(0), out.size(1), x.size(1), x.stride(0),
      weight.stride(1), out.stride(0),
      padding.has_value() ? static_cast<uint8_t const*>(padding.value().data_ptr()) : nullptr,
      get_stream(x.device()));
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare, PreparePairedGeGLU);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, RunPairedGeGLU);

#if FLASHINFER_PAIRED_READY
namespace ready = flashinfer::paired_ready_geglu;
tvm::ffi::Array<int64_t> PrepareReadyPairedGeGLU(
    int64_t device, tvm::ffi::Array<tvm::ffi::Array<int64_t>> producers) {
  ffi::CUDADeviceGuard guard(device);
  int registers, shared, threads, occupancy;
  auto status = ready::Prepare(&registers, &shared, &threads, &occupancy);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  cudaDeviceProp properties;
  status = cudaGetDeviceProperties(&properties, device);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  int math_registers;
  bool concurrent;
  for (auto const& producer : producers) TVM_FFI_ICHECK(producer.size() == 3);
  status =
      flashinfer::ready_tma_gemm::PlanReady<ready::Element, decltype(producers), ready::Kernel>(
          properties, producers, &math_registers, &concurrent);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  return {registers,  shared,        threads, occupancy, properties.multiProcessorCount,
          concurrent, math_registers};
}

void RunReadyPairedGeGLU(TensorView x, TensorView weight, TensorView out, TensorView readiness,
                         int64_t group_rows, int64_t done_index, int64_t available_sms) {
  ffi::CUDADeviceGuard guard(x.device().device_id);
  auto status = ready::Run(
      static_cast<ready::Element*>(x.data_ptr()), static_cast<ready::Element*>(weight.data_ptr()),
      static_cast<ready::Element*>(out.data_ptr()), x.size(0), out.size(1), x.size(1), x.stride(0),
      weight.stride(1), out.stride(0), static_cast<int*>(readiness.data_ptr()), group_rows,
      done_index, available_sms, get_stream(x.device()));
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare_ready, PrepareReadyPairedGeGLU);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_ready, RunReadyPairedGeGLU);
#endif
