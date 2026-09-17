/*
 * Copyright (c) 2026 by FlashInfer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <tvm/ffi/container/array.h>

#include <flashinfer/gemm/ffn_task_pipeline.cuh>

#include "tvm_ffi_utils.h"

namespace ffn = flashinfer::ffn_task_pipeline;

template <bool Push>
tvm::ffi::Array<int64_t> PrepareFfnTasks(int64_t device) {
  ffi::CUDADeviceGuard guard(device);
  int registers, shared, threads, occupancy;
  auto status = ffn::Prepare<Push>(&registers, &shared, &threads, &occupancy);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  cudaDeviceProp properties;
  status = cudaGetDeviceProperties(&properties, device);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  return {registers, shared, threads, occupancy, properties.multiProcessorCount};
}

void RunFfnTasks(TensorView x, TensorView up_weight, TensorView down_weight, TensorView activation,
                 TensorView out, TensorView state, tvm::ffi::Optional<TensorView> readiness,
                 int64_t group_rows, tvm::ffi::Optional<TensorView> publication, int64_t workers,
                 tvm::ffi::Optional<TensorView> retire_readiness, int64_t retire_group_rows,
                 int64_t retire_rows, int64_t retire_workers, int64_t native_region,
                 tvm::ffi::Optional<TensorView> admission_state) {
  ffi::CUDADeviceGuard guard(x.device().device_id);
  TVM_FFI_ICHECK(workers > 0 && group_rows > 0);
  auto status = ffn::Run(
      static_cast<ffn::Element*>(x.data_ptr()), static_cast<ffn::Element*>(up_weight.data_ptr()),
      static_cast<ffn::Element*>(down_weight.data_ptr()),
      static_cast<ffn::Element*>(activation.data_ptr()), static_cast<ffn::Element*>(out.data_ptr()),
      x.size(0), x.size(1), activation.size(1), static_cast<int*>(state.data_ptr()),
      readiness.has_value() ? static_cast<int*>(readiness.value().data_ptr()) : nullptr, group_rows,
      publication.has_value() ? static_cast<int*>(publication.value().data_ptr()) : nullptr,
      publication.has_value() ? publication.value().size(1) : 0, workers,
      retire_readiness.has_value() ? static_cast<int*>(retire_readiness.value().data_ptr())
                                   : nullptr,
      retire_group_rows, retire_rows, retire_workers, get_stream(x.device()),
      reinterpret_cast<tp_region::DeviceRegion const*>(native_region),
      admission_state.has_value() ? static_cast<int*>(admission_state.value().data_ptr())
                                  : nullptr);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

void RunFfnPush(TensorView x, TensorView up_weight, TensorView down_weight, TensorView activation,
                TensorView out, TensorView state, tvm::ffi::Optional<TensorView> readiness,
                int64_t group_rows, TensorView publication, int64_t workers,
                tvm::ffi::Optional<TensorView> retire_readiness, int64_t retire_group_rows,
                int64_t retire_rows, int64_t retire_workers, int64_t native_region,
                tvm::ffi::Optional<TensorView> admission_state, TensorView peer_out,
                TensorView peer_publication, int64_t publication_epoch) {
  ffi::CUDADeviceGuard guard(x.device().device_id);
  TVM_FFI_ICHECK(workers > 0 && group_rows > 0 && native_region == 0);
  auto status = ffn::Run<true>(
      static_cast<ffn::Element*>(x.data_ptr()), static_cast<ffn::Element*>(up_weight.data_ptr()),
      static_cast<ffn::Element*>(down_weight.data_ptr()),
      static_cast<ffn::Element*>(activation.data_ptr()), static_cast<ffn::Element*>(out.data_ptr()),
      x.size(0), x.size(1), activation.size(1), static_cast<int*>(state.data_ptr()),
      readiness.has_value() ? static_cast<int*>(readiness.value().data_ptr()) : nullptr, group_rows,
      static_cast<int*>(publication.data_ptr()), publication.size(1), workers,
      retire_readiness.has_value() ? static_cast<int*>(retire_readiness.value().data_ptr())
                                   : nullptr,
      retire_group_rows, retire_rows, retire_workers, get_stream(x.device()), nullptr,
      admission_state.has_value() ? static_cast<int*>(admission_state.value().data_ptr()) : nullptr,
      static_cast<ffn::Element*>(peer_out.data_ptr()),
      static_cast<int*>(peer_publication.data_ptr()),
      reinterpret_cast<flashinfer::masked_gemm::PushPublication const*>(publication_epoch));
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare, PrepareFfnTasks<false>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare_push, PrepareFfnTasks<true>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, RunFfnTasks);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_push, RunFfnPush);
bool SupportsSplitAdmission() { return true; }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(supports_split_admission, SupportsSplitAdmission);
bool SupportsEpochPublication() { return true; }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(supports_epoch_publication, SupportsEpochPublication);
