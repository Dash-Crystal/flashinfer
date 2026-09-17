/* Copyright (c) 2026 by FlashInfer contributors. SPDX-License-Identifier:
 * Apache-2.0 */
#include <tvm/ffi/container/array.h>

#include <flashinfer/gemm/shared_tma_task_pipeline.cuh>

#include "tvm_ffi_utils.h"

using Executor = flashinfer::ready_tma_gemm::SharedTmaTaskPipeline<cutlass::bfloat16_t, 3>;
using Phase = Executor::Phase;

__global__ __launch_bounds__(Executor::Threads, 1) void SharedTma(
    CUTLASS_GRID_CONSTANT const Executor::Params params) {
  extern __shared__ char storage[];
  Executor{}(params, storage);
}

tvm::ffi::Array<int64_t> PrepareSharedTma(int64_t device) {
  ffi::CUDADeviceGuard guard(device);
  auto status = flashinfer::masked_gemm::PrepareKernel(SharedTma, sizeof(Executor::SharedStorage));
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  cudaFuncAttributes attributes;
  status = cudaFuncGetAttributes(&attributes, SharedTma);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  return {attributes.numRegs, sizeof(Executor::SharedStorage), Executor::Threads};
}

template <bool Linked, bool Paired = false>
void RunSharedTmaImpl(tvm::ffi::Array<tvm::ffi::Tensor> inputs,
                      tvm::ffi::Array<tvm::ffi::Tensor> weights,
                      tvm::ffi::Array<tvm::ffi::Tensor> outputs,
                      tvm::ffi::Array<tvm::ffi::Tensor> readiness,
                      tvm::ffi::Array<tvm::ffi::Tensor> claims, int64_t workers,
                      cutlass::bfloat16_t* peer = nullptr,
                      flashinfer::masked_gemm::PushPublication const* publication = nullptr,
                      flashinfer::masked_gemm::PushPublication const* tile_publication = nullptr) {
  TVM_FFI_ICHECK(inputs.size() == 3 && weights.size() == 3 && outputs.size() == 3 &&
                 readiness.size() == 3 && claims.size() == 3 && workers > 0);
  ffi::CUDADeviceGuard guard(inputs[0].device().device_id);
  Executor::Params params{};
  for (int op = 0; op < 3; ++op) {
    auto x = inputs[op], w = weights[op], out = outputs[op];
    int m = x.size(0), k = x.size(1), n = w.size(0);
    bool compact = Paired && op == 0;
    int output_columns = compact ? n / 2 : n;
    TVM_FFI_ICHECK(w.size(1) == k && out.size(0) == m && out.size(1) == output_columns);
    TVM_FFI_ICHECK(m > 0 && n % 8 == 0 && k % 8 == 0);
    TVM_FFI_ICHECK(x.stride(1) == 1 && w.stride(1) == 1 && out.stride(1) == 1);
    int tiles_m = (m + 127) / 128, tiles_n = (n + 63) / 64;
    TVM_FFI_ICHECK(readiness[op].numel() >= tiles_m &&
                   claims[op].numel() >= tiles_m * (tiles_n + 1));
    typename Phase::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape = {m, n, k, 1};
    args.mainloop = {{static_cast<cutlass::bfloat16_t*>(x.data_ptr()),
                      {x.stride(0), cute::_1{}, 0},
                      static_cast<cutlass::bfloat16_t*>(w.data_ptr()),
                      {w.stride(0), cute::_1{}, 0}},
                     static_cast<int*>(readiness[op].data_ptr()),
                     128,
                     0,
                     m};
    args.epilogue.thread = {1.0f, 0.0f};
    args.epilogue.ptr_D = static_cast<cutlass::bfloat16_t*>(out.data_ptr());
    args.epilogue.dD = {(compact ? 2 : 1) * out.stride(0), cute::_1{}, 0};
    args.hw_info.sm_count = workers;
    args.scheduler.max_swizzle_size = 1;
    args.scheduler.task_state = static_cast<int*>(claims[op].data_ptr());
    args.scheduler.readiness = static_cast<int*>(readiness[op].data_ptr());
    args.scheduler.group_rows = 128;
    params.operations[op] = Phase::to_underlying_arguments(args, nullptr);
    if (compact) {
      params.compact_output[op] = static_cast<cutlass::bfloat16_t*>(out.data_ptr());
      params.compact_stride[op] = out.stride(0);
    }
    if (peer && op == 2) {
      params.local_output[op] = static_cast<cutlass::bfloat16_t*>(out.data_ptr());
      params.output_stride[op] = out.stride(0);
    }
    if (peer && op == 1) {
      params.local_output[op] = static_cast<cutlass::bfloat16_t*>(out.data_ptr());
      params.peer_output[op] = peer;
      params.output_stride[op] = out.stride(0);
      params.publication[op] = publication;
      params.tile_publication[op] = tile_publication;
      params.completed_tiles[op] = static_cast<int*>(claims[op].data_ptr());
      continue;
    }
    if constexpr (Linked) {
      if (op < 2) {
        auto next = inputs[op + 1];
        TVM_FFI_ICHECK(next.data_ptr() == out.data_ptr() && next.size(0) == m &&
                       next.size(1) == output_columns && next.stride(0) == out.stride(0));
        params.completed_tiles[op] = static_cast<int*>(claims[op].data_ptr());
        params.output_readiness[op] = static_cast<int*>(readiness[op + 1].data_ptr());
      }
    }
  }
  SharedTma<<<workers, Executor::Threads, sizeof(Executor::SharedStorage),
              get_stream(inputs[0].device())>>>(params);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess);
}

template <bool Linked, bool Paired = false>
void RunSharedTma(tvm::ffi::Array<tvm::ffi::Tensor> inputs,
                  tvm::ffi::Array<tvm::ffi::Tensor> weights,
                  tvm::ffi::Array<tvm::ffi::Tensor> outputs,
                  tvm::ffi::Array<tvm::ffi::Tensor> readiness,
                  tvm::ffi::Array<tvm::ffi::Tensor> claims, int64_t workers) {
  RunSharedTmaImpl<Linked, Paired>(inputs, weights, outputs, readiness, claims, workers);
}

template <bool PublishTiles = false>
void RunSharedTmaPush(tvm::ffi::Array<tvm::ffi::Tensor> inputs,
                      tvm::ffi::Array<tvm::ffi::Tensor> weights,
                      tvm::ffi::Array<tvm::ffi::Tensor> outputs,
                      tvm::ffi::Array<tvm::ffi::Tensor> readiness,
                      tvm::ffi::Array<tvm::ffi::Tensor> claims, int64_t workers,
                      tvm::ffi::Tensor peer_output, tvm::ffi::Tensor publication) {
  TVM_FFI_ICHECK(outputs.size() == 3);
  auto out = outputs[1];
  TVM_FFI_ICHECK(peer_output.size(0) == out.size(0) && peer_output.size(1) == out.size(1) &&
                 peer_output.stride(0) == out.stride(0) && peer_output.stride(1) == 1 &&
                 out.stride(0) % 2 == 0);
  TVM_FFI_ICHECK(publication.numel() * publication.dtype().bits / 8 >=
                 sizeof(flashinfer::masked_gemm::PushPublication) * (PublishTiles ? 2 : 1));
  auto descriptor =
      static_cast<flashinfer::masked_gemm::PushPublication const*>(publication.data_ptr());
  RunSharedTmaImpl<true, true>(inputs, weights, outputs, readiness, claims, workers,
                               static_cast<cutlass::bfloat16_t*>(peer_output.data_ptr()),
                               descriptor, PublishTiles ? descriptor + 1 : nullptr);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(prepare, PrepareSharedTma);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, RunSharedTma<false>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_linked, RunSharedTma<true>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_paired, (RunSharedTma<true, true>));
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_push, RunSharedTmaPush<false>);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_push_tiles, RunSharedTmaPush<true>);
