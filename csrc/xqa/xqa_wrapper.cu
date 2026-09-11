/*
 * Copyright (c) 2024 by FlashInfer team.
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

#include <tvm/ffi/container/array.h>

#include "../tvm_ffi_utils.h"
#include "mha.h"

using tvm::ffi::Optional;

#if MLA_WRAPPER
void xqa_wrapper_mla(int64_t multiProcessorCount, double qScale, Optional<TensorView> qScaleTensor,
                     TensorView output, TensorView q, TensorView kCacheVLLM, TensorView vCacheVLLM,
                     TensorView kvCachePageList, int64_t maxSeqLen, TensorView seqLen,
                     int64_t batchSize, double kvCacheScale, Optional<TensorView> kvScaleTensor,
                     TensorView semaphores, TensorView scratch, bool enable_pdl) {
  auto stream = get_stream(output.device());
  float const* qScalePtr = qScaleTensor.has_value()
                               ? reinterpret_cast<float const*>(qScaleTensor.value().data_ptr())
                               : nullptr;
  float const* kvScalePtr = kvScaleTensor.has_value()
                                ? reinterpret_cast<float const*>(kvScaleTensor.value().data_ptr())
                                : nullptr;
  // Extract strides from TensorView (in elements, not bytes)
  uint64_t kv_stride_page = kCacheVLLM.stride(0);
  uint64_t kv_stride_token = kCacheVLLM.stride(-2);
  uint64_t kv_stride_head = kCacheVLLM.stride(-3);

  launchMLAFlashInfer(multiProcessorCount, 1, qScale, qScalePtr,
                      reinterpret_cast<OutputHead*>(output.data_ptr()),
                      reinterpret_cast<InputHead const*>(q.data_ptr()),
                      reinterpret_cast<GMemCacheHead*>(kCacheVLLM.data_ptr()),
                      reinterpret_cast<GMemCacheHead*>(vCacheVLLM.data_ptr()),
                      reinterpret_cast<KVCachePageIndex const*>(kvCachePageList.data_ptr()),
                      maxSeqLen, reinterpret_cast<uint32_t const*>(seqLen.data_ptr()), batchSize,
                      kvCacheScale, kvScalePtr, reinterpret_cast<uint32_t*>(semaphores.data_ptr()),
                      reinterpret_cast<void*>(scratch.data_ptr()), enable_pdl, kv_stride_page,
                      kv_stride_token, kv_stride_head, stream);
}
#else

void xqa_wrapper(bool run_sm90_fp8_mha, int64_t multiProcessorCount, int64_t nbKHeads,
                 int64_t slidingWinSize, double qScale, Optional<TensorView> qScaleTensor,
                 TensorView output, double rcpOutScale, TensorView q,
                 Optional<TensorView> attentionSinks, TensorView kCacheVLLM, TensorView vCacheVLLM,
                 Optional<TensorView> kSfCacheVLLM, Optional<TensorView> vSfCacheVLLM,
                 Optional<TensorView> fp8KPayload, Optional<TensorView> fp8VPayload,
                 Optional<TensorView> fp8KScales, Optional<TensorView> fp8VScales,
                 Optional<TensorView> fp4KPayload, Optional<TensorView> fp4VPayload,
                 Optional<TensorView> fp4KScales, Optional<TensorView> fp4VScales,
                 Optional<TensorView> pageFormat, Optional<TensorView> pageStorage,
                 Optional<TensorView> pageAddresses, Optional<TensorView> fp8KGlobalScale,
                 Optional<TensorView> fp8VGlobalScale, Optional<TensorView> fp4KGlobalScale,
                 Optional<TensorView> fp4VGlobalScale, TensorView kvCachePageList,
                 int64_t maxSeqLen, TensorView seqLen, int64_t batchSize, double kvCacheScale,
                 Optional<TensorView> kvScaleTensor, int64_t qSeqLen,
                 Optional<TensorView> qCuSeqLens, Optional<TensorView> mask, TensorView semaphores,
                 TensorView scratch, bool enable_pdl, Optional<TensorView> attentionWork,
                 int64_t plannedGridCapacity) {
  TVM_FFI_ICHECK(plannedGridCapacity >= 0 && plannedGridCapacity <= UINT32_MAX);
  auto stream = get_stream(output.device());
  if (attentionWork.has_value()) {
    TVM_FFI_ICHECK(attentionWork.value().dtype() == dl_int32 &&
                   attentionWork.value().IsContiguous() &&
                   attentionWork.value().numel() >= batchSize + 2);
    CHECK_DEVICE(attentionWork.value(), output);
  }
  float const* attentionSinksPtr =
      attentionSinks.has_value() ? reinterpret_cast<float const*>(attentionSinks.value().data_ptr())
                                 : nullptr;
  float const* qScalePtr = qScaleTensor.has_value()
                               ? reinterpret_cast<float const*>(qScaleTensor.value().data_ptr())
                               : nullptr;
  float const* kvScalePtr = kvScaleTensor.has_value()
                                ? reinterpret_cast<float const*>(kvScaleTensor.value().data_ptr())
                                : nullptr;
  // Extract strides from TensorView (in elements, not bytes)
  uint64_t kv_stride_page = pageStorage.has_value() ? 0 : kCacheVLLM.stride(0);
  uint64_t kv_stride_token = pageStorage.has_value() ? 0 : kCacheVLLM.stride(-3);
  uint64_t kv_stride_head = pageStorage.has_value() ? 0 : kCacheVLLM.stride(-2);
#if ENABLE_4BIT_KV_CACHE
  uint64_t sf_stride_page = kv_stride_page;
  uint64_t sf_stride_token = kv_stride_token;
  uint64_t sf_stride_head = kv_stride_head;
  if (kSfCacheVLLM.has_value()) {
    sf_stride_page = kSfCacheVLLM.value().stride(0);
    sf_stride_token = kSfCacheVLLM.value().stride(-3);
    sf_stride_head = kSfCacheVLLM.value().stride(-2);
  }
#endif

#if ENABLE_MIXED_KV_CACHE
  PageTransport pageTransport{};
  TVM_FFI_ICHECK(fp8KGlobalScale.has_value() && fp8VGlobalScale.has_value() &&
                 fp4KGlobalScale.has_value() && fp4VGlobalScale.has_value());
  if (pageStorage.has_value()) {
    TVM_FFI_ICHECK(pageAddresses.has_value());
    auto const data = pageStorage.value();
    auto const pages = pageAddresses.value();
    TVM_FFI_ICHECK(data.ndim() == 1 && data.dtype() == dl_uint8 && data.stride(0) == 1);
    TVM_FFI_ICHECK(pages.ndim() == 2 && pages.dtype() == dl_int64 && pages.stride(0) > 0 &&
                   pages.stride(0) <= UINT32_MAX && pages.stride(1) == 1 && pages.size(1) > 0);
    CHECK_DEVICE(data, q);
    CHECK_DEVICE(pages, q);
    pageTransport.storage = {static_cast<uint8_t*>(data.data_ptr()),
                             static_cast<uint64_t*>(pages.data_ptr()),
                             static_cast<uint32_t>(pages.stride(0)),
                             static_cast<uint32_t>(pages.size(1)),
                             {tokensPerPage, static_cast<uint32_t>(nbKHeads), validElemsPerHead,
                              bool(XQA_MIXED_NATIVE_MMA)}};
  } else {
    TVM_FFI_ICHECK(fp8KPayload.has_value() && fp8VPayload.has_value() && fp8KScales.has_value() &&
                   fp8VScales.has_value() && fp4KPayload.has_value() && fp4VPayload.has_value() &&
                   fp4KScales.has_value() && fp4VScales.has_value() && pageFormat.has_value() &&
                   fp8KGlobalScale.has_value() && fp8VGlobalScale.has_value() &&
                   fp4KGlobalScale.has_value() && fp4VGlobalScale.has_value())
        << "mixed-page XQA requires all fixed-shape transport operands";
    auto const byte_stride = [](int64_t elements, uint64_t element_bytes) -> uint32_t {
      TVM_FFI_ICHECK_GE(elements, 0) << "mixed-page XQA requires nonnegative strides";
      uint64_t const bytes = uint64_t(elements) * element_bytes;
      TVM_FFI_ICHECK_LE(bytes, uint64_t{UINT32_MAX})
          << "mixed-page XQA byte stride exceeds the compact transport descriptor";
      return static_cast<uint32_t>(bytes);
    };
    pageTransport.page_format = reinterpret_cast<uint8_t const*>(pageFormat.value().data_ptr());
    auto& a16 = pageTransport.formats[static_cast<uint8_t>(flashinfer::KVPageFormat::kA16)];
    a16.k_payload = kCacheVLLM.data_ptr();
    a16.v_payload = vCacheVLLM.data_ptr();
    a16.payload_stride = {byte_stride(kCacheVLLM.stride(0), sizeof(InputElem)),
                          byte_stride(kCacheVLLM.stride(-3), sizeof(InputElem)),
                          byte_stride(kCacheVLLM.stride(-2), sizeof(InputElem))};
    auto& fp8 =
        pageTransport.formats[static_cast<uint8_t>(flashinfer::KVPageFormat::kBlockScaledFP8)];
    fp8.k_payload = fp8KPayload.value().data_ptr();
    fp8.v_payload = fp8VPayload.value().data_ptr();
    fp8.k_scales = reinterpret_cast<uint8_t const*>(fp8KScales.value().data_ptr());
    fp8.v_scales = reinterpret_cast<uint8_t const*>(fp8VScales.value().data_ptr());
    fp8.payload_stride = {byte_stride(fp8KPayload.value().stride(0), 1),
                          byte_stride(fp8KPayload.value().stride(-3), 1),
                          byte_stride(fp8KPayload.value().stride(-2), 1)};
    fp8.scale_stride = {byte_stride(fp8KScales.value().stride(0), 1),
                        byte_stride(fp8KScales.value().stride(-3), 1),
                        byte_stride(fp8KScales.value().stride(-2), 1)};
    auto& fp4 =
        pageTransport.formats[static_cast<uint8_t>(flashinfer::KVPageFormat::kBlockScaledFP4)];
    fp4.k_payload = fp4KPayload.value().data_ptr();
    fp4.v_payload = fp4VPayload.value().data_ptr();
    fp4.k_scales = reinterpret_cast<uint8_t const*>(fp4KScales.value().data_ptr());
    fp4.v_scales = reinterpret_cast<uint8_t const*>(fp4VScales.value().data_ptr());
    fp4.payload_stride = {byte_stride(fp4KPayload.value().stride(0), 1),
                          byte_stride(fp4KPayload.value().stride(-3), 1),
                          byte_stride(fp4KPayload.value().stride(-2), 1)};
    fp4.scale_stride = {byte_stride(fp4KScales.value().stride(0), 1),
                        byte_stride(fp4KScales.value().stride(-3), 1),
                        byte_stride(fp4KScales.value().stride(-2), 1)};
  }
  pageTransport.formats[1].k_global_scale =
      static_cast<const float*>(fp8KGlobalScale.value().data_ptr());
  pageTransport.formats[1].v_global_scale =
      static_cast<const float*>(fp8VGlobalScale.value().data_ptr());
  pageTransport.formats[2].k_global_scale =
      static_cast<const float*>(fp4KGlobalScale.value().data_ptr());
  pageTransport.formats[2].v_global_scale =
      static_cast<const float*>(fp4VGlobalScale.value().data_ptr());
#endif

#if SPEC_DEC
  MaskType const* maskPtr =
      mask.has_value() ? reinterpret_cast<MaskType const*>(mask.value().data_ptr()) : nullptr;
  // Optional ragged Q: cumulative draft lengths [batchSize + 1]; when set,
  // qSeqLen is the max draft length and q/mask/output are packed by qCuSeqLens.
  SeqLenDataType const* qCuSeqLensPtr =
      qCuSeqLens.has_value()
          ? reinterpret_cast<SeqLenDataType const*>(qCuSeqLens.value().data_ptr())
          : nullptr;
#endif

  void* kSfCachePtr = kSfCacheVLLM.has_value() ? kSfCacheVLLM.value().data_ptr() : nullptr;
  void* vSfCachePtr = vSfCacheVLLM.has_value() ? vSfCacheVLLM.value().data_ptr() : nullptr;

#if USE_SM90_MHA
  if (run_sm90_fp8_mha) {
#if SPEC_DEC
    // mha_sm90.cu's qCuSeqLens path is unvalidated; fail loudly.
    TVM_FFI_ICHECK(qCuSeqLensPtr == nullptr)
        << "ragged Q (q_cu_seq_lens) is not supported on the SM90 fp8 MHA path";
#endif
    launchHopperF8MHAFlashInfer(
        multiProcessorCount, nbKHeads, slidingWinSize, qScale, qScalePtr,
        reinterpret_cast<OutputHead*>(output.data_ptr()),
#if LOW_PREC_OUTPUT
        rcpOutScale,
#endif
        reinterpret_cast<InputHead const*>(q.data_ptr()), attentionSinksPtr,
        reinterpret_cast<GMemCacheHead*>(kCacheVLLM.data_ptr()),
        reinterpret_cast<GMemCacheHead*>(vCacheVLLM.data_ptr()),
#if ENABLE_MIXED_KV_CACHE
        pageTransport,
#endif
        reinterpret_cast<KVCachePageIndex const*>(kvCachePageList.data_ptr()), maxSeqLen,
        reinterpret_cast<uint32_t const*>(seqLen.data_ptr()), batchSize, kvCacheScale, kvScalePtr,
#if SPEC_DEC
        qSeqLen, qCuSeqLensPtr, maskPtr,
#endif
        reinterpret_cast<uint32_t*>(semaphores.data_ptr()),
        reinterpret_cast<void*>(scratch.data_ptr()), enable_pdl, kv_stride_page, kv_stride_token,
        kv_stride_head, stream);
    return;
  }
#endif

  launchMHAFlashInfer(
      multiProcessorCount, nbKHeads, slidingWinSize, qScale, qScalePtr,
      reinterpret_cast<OutputHead*>(output.data_ptr()),
#if LOW_PREC_OUTPUT
      rcpOutScale,
#endif
      reinterpret_cast<InputHead const*>(q.data_ptr()), attentionSinksPtr,
      reinterpret_cast<GMemCacheHead*>(kCacheVLLM.data_ptr()),
      reinterpret_cast<GMemCacheHead*>(vCacheVLLM.data_ptr()),
#if ENABLE_4BIT_KV_CACHE
      reinterpret_cast<GMemCacheHeadSf*>(kSfCachePtr),
      reinterpret_cast<GMemCacheHeadSf*>(vSfCachePtr),
#endif
#if ENABLE_MIXED_KV_CACHE
      pageTransport,
#endif
      reinterpret_cast<KVCachePageIndex const*>(kvCachePageList.data_ptr()), maxSeqLen,
      reinterpret_cast<uint32_t const*>(seqLen.data_ptr()), batchSize, kvCacheScale, kvScalePtr,
#if SPEC_DEC
      qSeqLen, qCuSeqLensPtr, maskPtr, q.size(0),
#endif
      reinterpret_cast<uint32_t*>(semaphores.data_ptr()),
      reinterpret_cast<void*>(scratch.data_ptr()), enable_pdl, kv_stride_page, kv_stride_token,
      kv_stride_head,
#if ENABLE_4BIT_KV_CACHE
      sf_stride_page, sf_stride_token, sf_stride_head,
#endif
      scratch.numel() * scratch.dtype().bits / 8,
      attentionWork.has_value() ? static_cast<uint32_t const*>(attentionWork.value().data_ptr())
                                : nullptr,
      plannedGridCapacity, stream);
}
#endif

#if MLA_WRAPPER
TVM_FFI_DLL_EXPORT_TYPED_FUNC(xqa_wrapper_mla, xqa_wrapper_mla);
#else
TVM_FFI_DLL_EXPORT_TYPED_FUNC(xqa_wrapper, xqa_wrapper);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(xqa_sequence_tile, xqaSequenceTile);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(xqa_grid_capacity, xqaGridCapacity);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(xqa_resident_slots, xqaResidentSlots);
tvm::ffi::Array<int64_t> xqa_split_kv_geometry() {
  auto const g = xqaSplitKVGeometry();
  return tvm::ffi::Array<int64_t>{g.scalarBytes, g.rows, g.columns, g.slices};
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(xqa_split_kv_geometry, xqa_split_kv_geometry);
#endif
