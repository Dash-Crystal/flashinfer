/*
 * Copyright (c) 2026 by FlashInfer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda.h>

#include <stdexcept>
#include <string>

#include "page_transport.cuh"

namespace flashinfer {

// Host-side TMA tensor maps over paged KV storage, shared by the sm90 XQA and
// FA3 mixed-page prologues.  Both describe a page as a 4-D tensor
// (row elements, kv heads, tokens per page, pages) and load one page of one
// head per box.

inline uint32_t tmaElemBytes(CUtensorMapDataType_enum dataType) {
  switch (dataType) {
    case CU_TENSOR_MAP_DATA_TYPE_UINT8:
      return 1;
    case CU_TENSOR_MAP_DATA_TYPE_UINT16:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
    case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
      return 2;
    case CU_TENSOR_MAP_DATA_TYPE_UINT32:
    case CU_TENSOR_MAP_DATA_TYPE_INT32:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ:
      return 4;
    case CU_TENSOR_MAP_DATA_TYPE_UINT64:
    case CU_TENSOR_MAP_DATA_TYPE_INT64:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT64:
      return 8;
    default:
      throw std::runtime_error("unsupported tensor map data type");
  }
}

inline CUtensorMapSwizzle tmaSwizzleForRowBytes(uint32_t boxRowBytes) {
  switch (boxRowBytes) {
    case 128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
    case 64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case 32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    default:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
  }
}

inline void checkCuTensorMap(CUresult r, char const* what) {
  if (r != CUDA_SUCCESS) {
    char const* name = nullptr;
    cuGetErrorName(r, &name);
    throw std::runtime_error(std::string(what) + ": " + (name ? name : "CUDA error"));
  }
}

// A16 (or any element-typed) paged KV cache.  Box = partElems elements of one
// head for min(tokensPerPage, nbTokensPerTile) tokens, swizzled by the part's
// byte width (128 B -> 128B swizzle), which is what both the XQA K/V stage
// buffers and CuTe's SW128 K-major smem atoms expect.  Strides are in elements.
inline CUtensorMap makePagedKVTensorMap(void const* addr, CUtensorMapDataType_enum dataType,
                                        uint32_t headElems, uint32_t nbKHeads,
                                        uint32_t tokensPerPage, uint32_t partElems,
                                        uint32_t nbTokensPerTile, uint64_t stride_page,
                                        uint64_t stride_token, uint64_t stride_head) {
  CUtensorMap tensorMap{};
  uint32_t const elemBytes = tmaElemBytes(dataType);
  uint64_t const globalDims[] = {headElems, nbKHeads, tokensPerPage, 1ULL << 31};
  uint64_t const globalStrides[] = {stride_head * elemBytes, stride_token * elemBytes,
                                    stride_page * elemBytes};
  uint32_t const partBytes = partElems * elemBytes;
  uint32_t const boxDims[] = {partElems, 1, tokensPerPage < nbTokensPerTile ? tokensPerPage : nbTokensPerTile, 1};
  uint32_t const elemStrides[] = {1, 1, 1, 1};
  CUtensorMapSwizzle const swizzle = tmaSwizzleForRowBytes(partBytes);
  if (swizzle == CU_TENSOR_MAP_SWIZZLE_NONE) {
    throw std::runtime_error("unsupported KV part width: " + std::to_string(partBytes) + " bytes");
  }
  checkCuTensorMap(cuTensorMapEncodeTiled(&tensorMap, dataType, 4, const_cast<void*>(addr),
                                          globalDims, globalStrides, boxDims, elemStrides,
                                          CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                                          CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
                   "cuTensorMapEncodeTiled(paged KV)");
  return tensorMap;
}

// Byte-typed map over a paged tensor of compressed rows
// ([page][token][head][rowBytes], byte strides).  A box is one page of one
// head: tokensPerPage rows of partBytes each, swizzled by the row width so a
// consumer reading one 16 B chunk per token across 8 tokens is conflict free.
inline CUtensorMap makePackedKVPagesTensorMap(void const* addr, uint32_t rowBytes,
                                              uint32_t nbKHeads, uint32_t tokensPerPage,
                                              uint32_t partBytes, uint64_t stride_page_bytes,
                                              uint64_t stride_token_bytes,
                                              uint64_t stride_head_bytes) {
  if (partBytes % 16 != 0 || rowBytes % partBytes != 0) {
    throw std::runtime_error("packed KV page box must be a 16-byte multiple dividing the row: " +
                             std::to_string(partBytes) + " of " + std::to_string(rowBytes));
  }
  if (stride_head_bytes % 16 != 0 || stride_token_bytes % 16 != 0 ||
      stride_page_bytes % 16 != 0) {
    throw std::runtime_error("packed KV page strides must be 16-byte multiples");
  }
  CUtensorMap tensorMap{};
  uint64_t const globalDims[] = {rowBytes, nbKHeads, tokensPerPage, 1ULL << 31};
  uint64_t const globalStrides[] = {stride_head_bytes, stride_token_bytes, stride_page_bytes};
  uint32_t const boxDims[] = {partBytes, 1, tokensPerPage, 1};
  uint32_t const elemStrides[] = {1, 1, 1, 1};
  checkCuTensorMap(
      cuTensorMapEncodeTiled(&tensorMap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
                             const_cast<void*>(addr), globalDims, globalStrides, boxDims,
                             elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                             tmaSwizzleForRowBytes(partBytes), CU_TENSOR_MAP_L2_PROMOTION_NONE,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled(packed KV pages)");
  return tensorMap;
}

// The six maps a mixed-page prologue needs: A16 K/V plus E4M3 and E2M1 K/V.
// A format whose payload is absent gets the A16 map as a placeholder; no page
// carries that tag, so it is never dereferenced.
struct MixedPageTensorMaps {
  CUtensorMap a16K, a16V, fp8K, fp8V, fp4K, fp4V;
};

template <typename A16>
inline MixedPageTensorMaps makeMixedPageTensorMaps(KVPageTransport<A16> const& transport,
                                                   CUtensorMapDataType_enum a16Type,
                                                   uint32_t headElems, uint32_t nbKHeads,
                                                   uint32_t tokensPerPage, uint32_t partElems,
                                                   uint32_t nbTokensPerTile) {
  auto const& a16 = transport.formats[static_cast<uint8_t>(KVPageFormat::kA16)];
  uint32_t const a16Bytes = tmaElemBytes(a16Type);
  auto a16Map = [&](void const* addr) {
    return makePagedKVTensorMap(addr, a16Type, headElems, nbKHeads, tokensPerPage, partElems,
                                nbTokensPerTile, a16.payload_stride.page / a16Bytes,
                                a16.payload_stride.token / a16Bytes,
                                a16.payload_stride.head / a16Bytes);
  };
  MixedPageTensorMaps maps;
  maps.a16K = a16Map(a16.k_payload);
  maps.a16V = a16Map(a16.v_payload);
  auto packedMap = [&](KVPageFormatSpan const& span, bool isK, uint32_t rowBytes,
                       CUtensorMap const& fallback) {
    void const* addr = isK ? span.k_payload : span.v_payload;
    if (addr == nullptr) {
      return fallback;
    }
    return makePackedKVPagesTensorMap(addr, rowBytes, nbKHeads, tokensPerPage, rowBytes,
                                      span.payload_stride.page, span.payload_stride.token,
                                      span.payload_stride.head);
  };
  auto const& fp8 = transport.formats[static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8)];
  auto const& fp4 = transport.formats[static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4)];
  maps.fp8K = packedMap(fp8, true, headElems, maps.a16K);
  maps.fp8V = packedMap(fp8, false, headElems, maps.a16V);
  maps.fp4K = packedMap(fp4, true, headElems / 2, maps.a16K);
  maps.fp4V = packedMap(fp4, false, headElems / 2, maps.a16V);
  return maps;
}

}  // namespace flashinfer
