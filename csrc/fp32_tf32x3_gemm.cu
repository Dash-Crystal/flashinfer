/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */

#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_batched.h>

#include <limits>
#include <type_traits>

#include "tvm_ffi_utils.h"

namespace {

template <typename LayoutA, typename LayoutB>
void run(TensorView a, TensorView b, TensorView c, TensorView out, float alpha, float beta) {
  // CUTLASS example 27's stock three-product FP32 MMA, with its batched device adapter.
  using Gemm = cutlass::gemm::device::GemmBatched<
      float, LayoutA, float, LayoutB, float, cutlass::layout::RowMajor, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 64, 16>,
      cutlass::gemm::GemmShape<64, 32, 16>, cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 3, 4, 4,
      cutlass::arch::OpMultiplyAddFastF32>;
  int64_t lda = std::is_same_v<LayoutA, cutlass::layout::RowMajor> ? a.stride(1) : a.stride(2);
  int64_t ldb = std::is_same_v<LayoutB, cutlass::layout::RowMajor> ? b.stride(1) : b.stride(2);
  typename Gemm::Arguments args({int(a.size(1)), int(b.size(2)), int(a.size(2))},
                                {static_cast<float const*>(a.data_ptr()), int(lda)}, a.stride(0),
                                {static_cast<float const*>(b.data_ptr()), int(ldb)}, b.stride(0),
                                {static_cast<float const*>(c.data_ptr()), int(c.stride(1))},
                                c.stride(0),
                                {static_cast<float*>(out.data_ptr()), int(out.stride(1))},
                                out.stride(0), {alpha, beta}, int(a.size(0)));
  TVM_FFI_ICHECK(Gemm::can_implement(args) == cutlass::Status::kSuccess)
      << "Unsupported TF32x3 GEMM alignment";
  Gemm gemm;
  TVM_FFI_ICHECK(Gemm::get_workspace_size(args) == 0) << "Unexpected batched GEMM workspace";
  auto status = gemm(args, nullptr, get_stream(a.device()));
  TVM_FFI_ICHECK(status == cutlass::Status::kSuccess)
      << "TF32x3 GEMM failed with CUTLASS status " << int(status);
}

}  // namespace

void fp32_tf32x3_bmm(TensorView a, TensorView b, TensorView c, TensorView out, double alpha,
                     double beta) {
  for (auto value : {a, b, c, out}) {
    CHECK_CUDA(value);
    CHECK_INPUT_TYPE(value, dl_float32);
    CHECK_DIM(3, value);
    TVM_FFI_ICHECK_EQ(value.device().device_id, a.device().device_id);
    TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(value.data_ptr()) % 16 == 0);
    for (int dimension = 0; dimension < 3; ++dimension) {
      TVM_FFI_ICHECK(value.size(dimension) > 0 &&
                     value.size(dimension) <= std::numeric_limits<int>::max());
      TVM_FFI_ICHECK(value.stride(dimension) >= 0);
    }
  }
  TVM_FFI_ICHECK_EQ(a.size(0), b.size(0));
  TVM_FFI_ICHECK_EQ(a.size(0), c.size(0));
  TVM_FFI_ICHECK_EQ(a.size(0), out.size(0));
  TVM_FFI_ICHECK_EQ(a.size(2), b.size(1));
  TVM_FFI_ICHECK_EQ(a.size(1), c.size(1));
  TVM_FFI_ICHECK_EQ(a.size(1), out.size(1));
  TVM_FFI_ICHECK_EQ(b.size(2), c.size(2));
  TVM_FFI_ICHECK_EQ(b.size(2), out.size(2));
  TVM_FFI_ICHECK_EQ(c.stride(2), 1);
  TVM_FFI_ICHECK_EQ(out.stride(2), 1);
  for (auto value : {c, out}) {
    TVM_FFI_ICHECK(value.stride(1) >= value.size(2));
    TVM_FFI_ICHECK(value.stride(1) % 4 == 0);
    TVM_FFI_ICHECK(value.stride(1) <= std::numeric_limits<int>::max());
  }
  for (auto value : {a, b}) {
    bool row_major = value.stride(2) == 1 && value.stride(1) >= value.size(2);
    bool column_major = value.stride(1) == 1 && value.stride(2) >= value.size(1);
    TVM_FFI_ICHECK(row_major || column_major) << "Expected row or column major operands";
    int64_t leading = row_major ? value.stride(1) : value.stride(2);
    TVM_FFI_ICHECK(leading % 4 == 0 && leading <= std::numeric_limits<int>::max());
  }
  ffi::CUDADeviceGuard guard(a.device().device_id);
  using Row = cutlass::layout::RowMajor;
  using Col = cutlass::layout::ColumnMajor;
  if (a.stride(2) == 1) {
    if (b.stride(2) == 1)
      run<Row, Row>(a, b, c, out, alpha, beta);
    else
      run<Row, Col>(a, b, c, out, alpha, beta);
  } else {
    if (b.stride(2) == 1)
      run<Col, Row>(a, b, c, out, alpha, beta);
    else
      run<Col, Col>(a, b, c, out, alpha, beta);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(bmm, fp32_tf32x3_bmm);
