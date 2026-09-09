// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

#include <cub/block/block_scan.cuh>
#include <cuda/cmath>
#include <cuda/functional>

#include "tvm_ffi_utils.h"
#include "xqa/workScheduling.cuh"

using tvm::ffi::Optional;

__global__ void mixed_kv_prepare_metadata_kernel(
    int64_t const* slots, uint32_t numSlots, uint32_t pageSize,
    cuda::fast_mod_div<uint64_t> pageDivisor, uint32_t* writable, uint32_t* writableCount,
    uint32_t* completed, uint32_t* completedCount, uint32_t const* lengths, uint32_t requests,
    uint32_t* work, uint32_t heads, uint32_t residentSlots, uint32_t sequenceTile,
    uint32_t window) {
  using IndexScan = cub::BlockScan<int32_t, 256>;
  using CountScan = cub::BlockScan<uint32_t, 256>;
  __shared__ typename IndexScan::TempStorage indexStorage;
  __shared__ typename CountScan::TempStorage countStorage;
  if (work != nullptr && threadIdx.x < 32) {
    xqa_work::prepareWarp(lengths, requests, heads, residentSlots, sequenceTile, window, work);
  }
  uint32_t written = 0;
  uint32_t sealed = 0;
  int32_t previousTile = -1;
  for (uint32_t first = 0; first < numSlots; first += 256) {
    uint32_t const index = first + threadIdx.x;
    int64_t const slot = index < numSlots ? slots[index] : -1;
    bool const valid = slot >= 0;
    int32_t previous, last;
    IndexScan(indexStorage)
        .ExclusiveScan(valid ? int32_t(index) : -1, previous, previousTile, cuda::maximum<>{},
                       last);
    int64_t const predecessor = previous >= 0 ? slots[previous] : -1;
    uint32_t const page = valid ? uint64_t(slot) / pageDivisor : 0;
    bool const writablePage =
        valid && (predecessor < 0 || uint64_t(predecessor) / pageDivisor != page);
    bool const complete = valid && (uint64_t(slot) % pageDivisor == pageSize - 1);
    uint32_t prefix, count;
    CountScan(countStorage)
        .ExclusiveSum(uint32_t(writablePage) + (uint32_t(complete) << 16), prefix, count);
    if (writablePage) writable[written + (prefix & 0xffff)] = page;
    if (complete) completed[sealed + (prefix >> 16)] = page;
    written += count & 0xffff;
    sealed += count >> 16;
    previousTile = max(previousTile, last);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *writableCount = written;
    *completedCount = sealed;
  }
}

void mixed_kv_prepare_metadata(TensorView slots, TensorView writable, TensorView writableCount,
                               TensorView completed, TensorView completedCount, int64_t pageSize,
                               Optional<TensorView> lengths, Optional<TensorView> work,
                               int64_t heads, int64_t residentSlots, int64_t sequenceTile,
                               int64_t window) {
  TVM_FFI_ICHECK(slots.ndim() == 1 && slots.dtype() == dl_int64 && slots.stride(0) == 1);
  CHECK_CUDA(slots);
  TVM_FFI_ICHECK(pageSize > 0 && pageSize <= UINT32_MAX && slots.numel() <= INT32_MAX);
  for (auto tensor : {writable, writableCount, completed, completedCount}) {
    TVM_FFI_ICHECK(tensor.dtype() == dl_int32 && tensor.IsContiguous());
    CHECK_DEVICE(tensor, slots);
  }
  TVM_FFI_ICHECK(writableCount.numel() == 1 && completedCount.numel() == 1);
  if (work.has_value()) {
    TVM_FFI_ICHECK(lengths.has_value() && lengths.value().dtype() == dl_int32 &&
                   lengths.value().IsContiguous());
    CHECK_DEVICE(lengths.value(), slots);
    TVM_FFI_ICHECK(work.value().dtype() == dl_int32 && work.value().IsContiguous() &&
                   work.value().numel() >= lengths.value().numel() + 2);
    CHECK_DEVICE(work.value(), slots);
    TVM_FFI_ICHECK(heads > 0 && residentSlots > 0 && sequenceTile > 0 && window >= 0);
  }
  ffi::CUDADeviceGuard guard(slots.device().device_id);
  mixed_kv_prepare_metadata_kernel<<<1, 256, 0, get_stream(slots.device())>>>(
      static_cast<int64_t const*>(slots.data_ptr()), slots.numel(), pageSize,
      cuda::fast_mod_div<uint64_t>(pageSize), static_cast<uint32_t*>(writable.data_ptr()),
      static_cast<uint32_t*>(writableCount.data_ptr()),
      static_cast<uint32_t*>(completed.data_ptr()),
      static_cast<uint32_t*>(completedCount.data_ptr()),
      lengths.has_value() ? static_cast<uint32_t const*>(lengths.value().data_ptr()) : nullptr,
      lengths.has_value() ? lengths.value().numel() : 0,
      work.has_value() ? static_cast<uint32_t*>(work.value().data_ptr()) : nullptr, heads,
      residentSlots, sequenceTile, window);
  TVM_FFI_ICHECK(cudaPeekAtLastError() == cudaSuccess);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mixed_kv_prepare_metadata, mixed_kv_prepare_metadata);
