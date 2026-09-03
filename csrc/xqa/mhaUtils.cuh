/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "cutlass/numeric_conversion.h"

#include "ldgsts.cuh"
#include "mha.h"
#include "utils.cuh"

// for beam search
template <typename Head, uint32_t tokensPerPage, uint32_t nbPages>
struct IndexedHeadPtrImpl {
  static_assert(tokensPerPage != 0 && nbPages != 0);
  uint32_t const* indices;  // values are in range [0, beamWidth)
  Head* pool;
  Vec<KVCachePageIndex, nbPages> const* pageIndices;
  uint32_t tokenOffset;   // token offset within the first page
  uint32_t headIdx;       // head index
  uint32_t stride_page;   // stride for each page (in units of Head)
  uint32_t stride_token;  // stride for each token (in units of Head)
  uint32_t stride_head;   // stride for each head (in units of Head)

  __device__ inline Head& operator[](uint32_t i) const { return *(*this + i); }

  __device__ inline Head* operator+(uint32_t i) const {
    uint32_t const beamIdx = indices[i];
    assert(beamIdx < beamWidth);
    uint32_t const absoluteTokenIdx = tokenOffset + i;
    auto const pageIdx = pageIndices[beamIdx][nbPages == 1 ? 0U : absoluteTokenIdx / tokensPerPage];
    return pool + pageIdx * stride_page + (absoluteTokenIdx % tokensPerPage) * stride_token +
           headIdx * stride_head;
  }
};

template <typename Head>
struct IndexedHeadPtrImpl<Head, 0, 0> {
  uint32_t const* indices;  // values are in range [0, beamWidth)
  Head* pointer;
  uint32_t offset;
  uint32_t beamStride;

  __device__ inline Head& operator[](uint32_t i) const { return *(*this + i); }

  __device__ inline Head* operator+(uint32_t i) const {
    assert(indices[i] < beamWidth);
    return pointer + (beamStride * indices[i] + offset + i);
  }
};

template <typename Head, uint32_t tokensPerPage, uint32_t nbPages = 0>
using IndexedHeadPtr = IndexedHeadPtrImpl<Head, tokensPerPage, nbPages>;

// for beamWidth = 1
template <typename Head, uint32_t tokensPerPage, uint32_t nbPages>
struct HeadPtr {
  static_assert(tokensPerPage != 0 && nbPages != 0);
  Head* pool;
  Vec<KVCachePageIndex, nbPages> pageIndices;
  uint32_t tokenOffset;   // token offset within the first page
  uint32_t headIdx;       // head index
  uint32_t stride_page;   // stride for each page (in units of Head)
  uint32_t stride_token;  // stride for each token (in units of Head)
  uint32_t stride_head;   // stride for each head (in units of Head)

  __device__ inline Head& operator[](uint32_t i) const { return *(*this + i); }

  __device__ inline Head* operator+(uint32_t i) const {
    uint32_t const absoluteTokenIdx = tokenOffset + i;
    auto const pageIdx = pageIndices[nbPages == 1 ? 0U : absoluteTokenIdx / tokensPerPage];
    return (pageIdx & (1U << 31))
               ? nullptr
               : pool + pageIdx * stride_page + (absoluteTokenIdx % tokensPerPage) * stride_token +
                     headIdx * stride_head;
  }
};

template <typename Head>
struct HeadPtr<Head, 0, 0> : TinyPtr<Head> {};

// template <typename Head>
// #if BEAM_WIDTH == 1
// using SrcHeadPtr = TinyPtr<Head const>;
// #else
// using SrcHeadPtr = IndexedHeadPtr<Head>;
// #endif

// @fixme: give evict first hint for last part.
template <typename Head, uint32_t maxNbCopiedHeads, uint32_t nbPartsPerHead,
          uint32_t grainBytesSmem, uint32_t grainBytesGmem, bool swizzle, bool isFull,
          uint32_t dstNbHeads, typename SrcHeadPtr, typename _LdGrain,
          typename LocalHeadIdxMap = uint32_t (*)(uint32_t)>
__device__ inline void copyPartialHeadsAsync(
    Warp const& warp,
    Array2D<_LdGrain, dstNbHeads, exactDiv(exactDiv(sizeof(Head), nbPartsPerHead), grainBytesSmem)>&
        dst,
    uint32_t dstHeadOffset, SrcHeadPtr const& src, uint32_t idxPart,
    uint32_t nbAvailHeads = maxNbCopiedHeads,
    LocalHeadIdxMap&& localHeadIdxMap = [](uint32_t x) { return x; }) {
  static_assert(maxNbCopiedHeads <= dstNbHeads);
  assert(idxPart < nbPartsPerHead);
  assert(dstHeadOffset + maxNbCopiedHeads <= dstNbHeads);
  assert(sizeof(Head) * (src.offset + maxNbCopiedHeads) <= (1UL << 32));
  assert(!isFull || nbAvailHeads >= maxNbCopiedHeads);
  constexpr uint32_t headBytes = sizeof(Head);
  constexpr uint32_t partBytes = exactDiv(headBytes, nbPartsPerHead);
  constexpr uint32_t warpLdBytes = partBytes * maxNbCopiedHeads;
  constexpr uint32_t thrdLdBytes = exactDiv(warpLdBytes, warp_size);
  assertIsPowerOf2<thrdLdBytes>();
  static_assert(thrdLdBytes >= grainBytesSmem);
  // a segment is responsible for loading one partial head collaboratively
  constexpr uint32_t thrdsPerSeg = exactDiv(partBytes, grainBytesSmem);
  static_assert(thrdsPerSeg > 0 && thrdsPerSeg <= warp_size);
  assertIsPowerOf2<thrdsPerSeg>();
  assert(__shfl_sync(0xFU << (laneId() / 4 * 4), src.offset, 0, 4) == src.offset);
  auto const warpLane = laneId();
  uint32_t const segIdx = warpLane / thrdsPerSeg;
  uint32_t const segLane = warpLane % thrdsPerSeg;
  constexpr uint32_t partsPerWarpInst = exactDiv(grainBytesSmem * warp_size, partBytes);
#pragma unroll
  for (uint32_t i = 0; i < thrdLdBytes / grainBytesSmem; i++) {
    uint32_t const idxHeadLocal = partsPerWarpInst * i + segIdx;
    assert(idxHeadLocal < maxNbCopiedHeads);
    bool const isHeadInBound = isFull || (idxHeadLocal < nbAvailHeads);
    constexpr uint32_t grainsPerPart = exactDiv(partBytes, grainBytesSmem);
    using SrcHead = mha::decay_t<decltype(src[0])>;
    constexpr uint32_t nbValidGrains = exactDiv(sizeof(SrcHead), grainBytesGmem);
    uint32_t const idxGrainInsideHead = grainsPerPart * idxPart + segLane;
    bool const isGrainInBound = (!isHeadPadded || idxGrainInsideHead < nbValidGrains);
    SrcHead const* const pSrcHead = src + localHeadIdxMap(idxHeadLocal);
    bool const isValidPage = (pSrcHead != nullptr);
    Vec<uint8_t, grainBytesGmem> const* const pSrc =
        reinterpret_cast<Vec<uint8_t, grainBytesGmem> const*>(pSrcHead) + idxGrainInsideHead;
    Vec<uint8_t, grainBytesSmem>* const pDst = reinterpret_cast<Vec<uint8_t, grainBytesSmem>*>(
        &dst.template at<swizzle>(dstHeadOffset + idxHeadLocal, segLane));
#if !ENABLE_4BIT_KV_CACHE
    // 4-bit KV cache is not bank-conflict free now.
    assert(!hasBankConflict(pDst));
#endif
    ldgsts::copyAsync<grainBytesGmem>(
        pDst, pSrc, isValidPage && isHeadInBound && isGrainInBound ? grainBytesGmem : 0u);
  }
}

#if ENABLE_MIXED_KV_CACHE
template <uint32_t nbPages>
struct MixedPageFormats {
  Vec<uint8_t, nbPages> values;
};

// Gather every page tag in the warp tile exactly once.  The page list is
// warp-uniform; elected lanes issue the byte loads and shuffles distribute the
// fixed-size metadata vector used by all block owners.
template <uint32_t nbPages>
__device__ inline MixedPageFormats<nbPages> gatherMixedPageFormats(
    PageTransport const& transport, Vec<KVCachePageIndex, nbPages> const& pages) {
  MixedPageFormats<nbPages> ret;
  uint32_t const lane = laneId();
  uint32_t value = 0;
  if (lane < nbPages && pages[lane] != kBAD_PAGE_INDEX) {
    value = transport.page_format[pages[lane]];
  }
#pragma unroll
  for (uint32_t i = 0; i < nbPages; ++i) {
    ret.values[i] = static_cast<uint8_t>(__shfl_sync(0xffffffffU, value, i));
  }
  return ret;
}

template <uint32_t nbPages>
__device__ inline bool needsMixedPageExpansion(
    MixedPageFormats<nbPages> const& formats) {
#if MIXED_PAGE_STATIC_FORMAT == 0
  unused(formats);
  return false;
#elif MIXED_PAGE_STATIC_FORMAT > 0
  unused(formats);
  return true;
#else
  uint8_t constexpr a16 = static_cast<uint8_t>(flashinfer::KVPageFormat::kA16);
  bool needsExpansion = false;
#pragma unroll
  for (uint32_t i = 0; i < nbPages; ++i) {
    needsExpansion |= formats.values[i] != a16;
  }
  return needsExpansion;
#endif
}

// Preserve copyPartialHeadsAsync's warp ownership and circular-buffer
// schedule. Each lane owns one 16-value block. Compressed payload occupies
// the first A16 grain; its single scale byte is staged in the second grain.
template <uint32_t maxNbCopiedHeads, uint32_t nbPartsPerHead, bool swizzle, bool isFull,
          bool compactPages = false, uint32_t nbWarps = 1, uint32_t dstNbHeads,
          uint32_t nbPages, typename _LdGrain>
__device__ inline void copyMixedPartialHeadsAsync(
    Array2D<_LdGrain, dstNbHeads,
            exactDiv(exactDiv(sizeof(PaddedCacheHead), nbPartsPerHead), grainBytes)>& dst,
    uint8_t* dstScales, uint32_t dstHeadOffset, PageTransport const& transport,
    Vec<KVCachePageIndex, nbPages> const& pages,
    MixedPageFormats<nbPages> const& formats, uint32_t tokenOffset, uint32_t sourceHeadOffset,
    uint32_t headIdx, bool isK, uint32_t idxPart, uint32_t nbAvailHeads = maxNbCopiedHeads,
    uint32_t idxWarp = 0) {
  static_assert(sizeof(PaddedCacheHead) % 32 == 0);
  constexpr uint32_t partBytes = exactDiv(sizeof(PaddedCacheHead), nbPartsPerHead);
  constexpr uint32_t grainsPerPart = exactDiv(partBytes, grainBytes);
  constexpr uint32_t blocksPerPart = exactDiv(grainsPerPart, 2);
  constexpr uint32_t blocksPerWarpInst = mha::max(1U, warp_size / maxNbCopiedHeads);
  constexpr uint32_t headsPerWarpInst = warp_size / blocksPerWarpInst;
  static_assert(warp_size % maxNbCopiedHeads == 0 || maxNbCopiedHeads % warp_size == 0);
  constexpr uint32_t totalBlocks = maxNbCopiedHeads * blocksPerPart;
  constexpr uint32_t nbThreads = nbWarps * warp_size;
  constexpr uint32_t iterations = divUp(totalBlocks, nbThreads);
  static_assert(!compactPages || tokensPerPage == 16,
                "compact mixed-page fragments require the vLLM 16-token page unit");
  assert(idxWarp < nbWarps);
  (void)blocksPerWarpInst;
  (void)headsPerWarpInst;

#pragma unroll
  for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
    uint32_t const linearBlock =
        iteration * nbThreads + idxWarp * warp_size + laneId();
    if (linearBlock >= totalBlocks) continue;
    uint32_t const localHead = linearBlock / blocksPerPart;
    uint32_t const blockInPart = linearBlock % blocksPerPart;
    bool const validHead = isFull || localHead < nbAvailHeads;
    uint32_t const absoluteToken = tokenOffset + sourceHeadOffset + localHead;
    uint32_t const localPage = absoluteToken / tokensPerPage;
    KVCachePageIndex const page = localPage < nbPages ? pages[localPage] : kBAD_PAGE_INDEX;
    uint32_t const token = absoluteToken % tokensPerPage;
    uint32_t const elem = (idxPart * blocksPerPart + blockInPart) * 16;
    bool const validElem = elem + 16 <= validElemsPerHead;
    bool const valid = validHead && validElem && page != kBAD_PAGE_INDEX;
    uint8_t const format =
#if MIXED_PAGE_STATIC_FORMAT >= 0
        valid ? MIXED_PAGE_STATIC_FORMAT : 0;
#else
        valid ? formats.values[localPage] : 0;
#endif
    auto const& span = transport.formats[format];
    auto const* payload = static_cast<uint8_t const*>(isK ? span.k_payload : span.v_payload);
    uint64_t const payloadBase =
        valid ? uint64_t(page) * span.payload_stride.page +
                    uint64_t(token) * span.payload_stride.token +
                    uint64_t(headIdx) * span.payload_stride.head
              : 0;
    using flashinfer::KVPageFormat;
    uint8_t constexpr a16Format = static_cast<uint8_t>(KVPageFormat::kA16);
    uint8_t constexpr fp8Format = static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8);
    bool const isA16 = format == a16Format;
    bool const isFP8 = format == fp8Format;
    uint64_t const payloadElemOffset = isA16 ? uint64_t(elem) * sizeof(InputElem)
                                             : (isFP8 ? elem : elem / 2);
    auto const* firstSource = payload + payloadBase + payloadElemOffset;
    bool const isFP4 = format == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4);

    if constexpr (compactPages) {
      if (isA16) {
        auto* first =
            &dst.template at<swizzle>(dstHeadOffset + localHead, blockInPart * 2);
        auto* second =
            &dst.template at<swizzle>(dstHeadOffset + localHead, blockInPart * 2 + 1);
        ldgsts::copyAsync<grainBytes>(first, firstSource, valid ? grainBytes : 0U);
        ldgsts::copyAsync<grainBytes>(second, firstSource + grainBytes,
                                     valid ? grainBytes : 0U);
      } else {
        // Retain the native tile row stride and place the compressed block in
        // the low half of that row.  This preserves ldmatrix-compatible row
        // addressing while keeping each page in a fixed-size slot.
        auto* packed =
            &dst.template at<swizzle>(dstHeadOffset + localHead, blockInPart);
        if (isFP4) {
          ldgsts::copyAsync<8>(packed, firstSource, valid ? 8U : 0U);
          ldgsts::copyAsync<8>(reinterpret_cast<uint8_t*>(packed) + 8,
                               firstSource + 8, 0U);
        } else {
          ldgsts::copyAsync<grainBytes>(packed, firstSource,
                                       valid ? grainBytes : 0U);
        }
      }
    } else {
      auto* first =
          &dst.template at<swizzle>(dstHeadOffset + localHead, blockInPart * 2);
      auto* second =
          &dst.template at<swizzle>(dstHeadOffset + localHead, blockInPart * 2 + 1);
      ldgsts::copyAsync<8>(first, firstSource, valid ? 8U : 0U);
      ldgsts::copyAsync<8>(reinterpret_cast<uint8_t*>(first) + 8,
                           firstSource + 8, valid && !isFP4 ? 8U : 0U);
      ldgsts::copyAsync<grainBytes>(second, firstSource + grainBytes,
                                   valid && isA16 ? grainBytes : 0U);
    }
  }

  static_assert(validElemsPerHead % 64 == 0);
  constexpr uint32_t scaleLoadBytes = mha::max(4U, blocksPerPart);
  static_assert(scaleLoadBytes == 4 || scaleLoadBytes == 8 || scaleLoadBytes == 16);
  uint32_t const scaleBlock = idxPart * blocksPerPart;
  uint32_t const scaleGroup = scaleBlock & ~(scaleLoadBytes - 1);
  constexpr uint32_t headIterations = divUp(maxNbCopiedHeads, nbThreads);
#pragma unroll
  for (uint32_t iteration = 0; iteration < headIterations; ++iteration) {
    uint32_t const localHead =
        iteration * nbThreads + idxWarp * warp_size + laneId();
    bool const validHead = (isFull || localHead < nbAvailHeads) &&
                           localHead < maxNbCopiedHeads;
    uint32_t const absoluteToken = tokenOffset + sourceHeadOffset + localHead;
    uint32_t const localPage = absoluteToken / tokensPerPage;
    KVCachePageIndex const page = localPage < nbPages ? pages[localPage]
                                                      : kBAD_PAGE_INDEX;
    uint32_t const token = absoluteToken % tokensPerPage;
    bool const valid = validHead && page != kBAD_PAGE_INDEX;
    uint8_t const format = valid ? formats.values[localPage] : 0;
    auto const& span = transport.formats[format];
    bool const compressed =
        format != static_cast<uint8_t>(flashinfer::KVPageFormat::kA16);
    auto const* scales = isK ? span.k_scales : span.v_scales;
    uint64_t const scaleOffset =
        valid && compressed
            ? uint64_t(page) * span.scale_stride.page +
                  uint64_t(token) * span.scale_stride.token +
                  uint64_t(headIdx) * span.scale_stride.head + scaleGroup
            : 0;
    auto const* scaleSource = compressed
                                  ? reinterpret_cast<uint8_t const*>(
                                        reinterpret_cast<uint64_t>(scales) + scaleOffset)
                                  : static_cast<uint8_t const*>(
                                        transport.formats[0].k_payload);
    uint32_t const destinationHead =
        localHead < maxNbCopiedHeads ? localHead : maxNbCopiedHeads;
    auto* scaleDestination = dstScales + destinationHead * scaleLoadBytes;
    ldgsts::copyAsync<scaleLoadBytes>(
        scaleDestination, scaleSource, valid && compressed ? scaleLoadBytes : 0U);
  }
}

template <typename A16>
__device__ inline uint32_t convertE4M3x2ToA16(uint16_t fp8x2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  if constexpr (mha::is_same_v<A16, __nv_bfloat16>) {
    uint32_t bf16x2;
    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(bf16x2) : "h"(fp8x2));
    return bf16x2;
  }
#endif
  uint32_t fp16x2;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(fp16x2) : "h"(fp8x2));
  if constexpr (mha::is_same_v<A16, half>) {
    return fp16x2;
  } else {
    __half2 const value = reinterpret_cast<__half2 const&>(fp16x2);
    __nv_bfloat162 const converted = __float22bfloat162_rn(__half22float2(value));
    return reinterpret_cast<uint32_t const&>(converted);
  }
}

template <typename A16>
__device__ inline uint32_t convertE2M1x2ToA16(uint8_t fp4x2) {
  uint32_t fp16x2;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  if constexpr (mha::is_same_v<A16, __nv_bfloat16>) {
    uint32_t bf16x2;
    asm volatile(
        "{\n"
        ".reg .b8 fp4_byte;\n"
        "mov.b32 {fp4_byte, _, _, _}, %1;\n"
        "cvt.rn.bf16x2.e2m1x2 %0, fp4_byte;\n"
        "}"
        : "=r"(bf16x2)
        : "r"(static_cast<uint32_t>(fp4x2)));
    return bf16x2;
  }
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm volatile(
      "{\n"
      ".reg .b8 fp4_byte;\n"
      "mov.b32 {fp4_byte, _, _, _}, %1;\n"
      "cvt.rn.f16x2.e2m1x2 %0, fp4_byte;\n"
      "}"
      : "=r"(fp16x2)
      : "r"(static_cast<uint32_t>(fp4x2)));
#else
  auto const fp4ToFp16Bits = [](uint32_t nibble) {
    uint32_t const magnitude = nibble & 7U;
    uint32_t const exponent = magnitude >> 1;
    uint32_t const mantissa = magnitude & 1U;
    uint32_t const finite = exponent == 0 ? mantissa * 0x3800U
                                           : ((exponent + 14U) << 10) | (mantissa << 9);
    return finite | ((nibble & 8U) << 12);
  };
  fp16x2 = fp4ToFp16Bits(fp4x2 & 0xfU) | (fp4ToFp16Bits(fp4x2 >> 4) << 16);
#endif
  if constexpr (mha::is_same_v<A16, half>) {
    return fp16x2;
  } else {
    __half2 const value = reinterpret_cast<__half2 const&>(fp16x2);
    __nv_bfloat162 const converted = __float22bfloat162_rn(__half22float2(value));
    return reinterpret_cast<uint32_t const&>(converted);
  }
}

// CUTLASS's Hopper mixed-input path decodes a complete 32-bit E2M1
// register at once.  Keep that vector width here instead of issuing eight
// scalar nibble conversions for each 16-value block.
template <typename A16>
__device__ inline Vec<uint32_t, 4> convertE2M1x8ToA16(uint32_t fp4x8) {
  Vec<uint32_t, 4> converted;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
#pragma unroll
  for (uint32_t pair = 0; pair < 4; ++pair) {
    converted[pair] = convertE2M1x2ToA16<A16>(
        static_cast<uint8_t>(fp4x8 >> (pair * 8)));
  }
#else
  if constexpr (mha::is_same_v<A16, half>) {
    cutlass::detail::_e2m1_to_half_x8(fp4x8, converted[0], converted[1],
                                      converted[2], converted[3]);
  } else {
    static_assert(mha::is_same_v<A16, __nv_bfloat16>);
    cutlass::detail::_e2m1_to_bf16_x8(fp4x8, converted[0], converted[1],
                                      converted[2], converted[3]);
  }
#endif
  return converted;
}

template <typename A16>
__device__ inline uint16_t convertE4M3ScaleToA16Bits(uint8_t scaleBits,
                                                     float globalScale) {
  auto const scale = reinterpret_cast<__nv_fp8_e4m3 const&>(scaleBits);
  A16 const a16Scale = static_cast<A16>(float(scale) * globalScale);
  return reinterpret_cast<uint16_t const&>(a16Scale);
}

template <typename A16>
__device__ inline uint32_t scaleA16x2(uint32_t a16x2Bits, uint8_t scaleBits,
                                     float globalScale) {
  return applyF16ScalingFactor<A16>(
      a16x2Bits, convertE4M3ScaleToA16Bits<A16>(scaleBits, globalScale));
}

template <typename A16>
__device__ inline uint32_t scaleA16x2Pair(uint32_t a16x2Bits, uint8_t scaleBits0,
                                         uint8_t scaleBits1, float globalScale) {
  auto const scale0 = reinterpret_cast<__nv_fp8_e4m3 const&>(scaleBits0);
  auto const scale1 = reinterpret_cast<__nv_fp8_e4m3 const&>(scaleBits1);
  A16 const a16Scale0 = static_cast<A16>(float(scale0) * globalScale);
  A16 const a16Scale1 = static_cast<A16>(float(scale1) * globalScale);
  uint16_t const scaleBitsA16_0 =
      reinterpret_cast<uint16_t const&>(a16Scale0);
  uint16_t const scaleBitsA16_1 =
      reinterpret_cast<uint16_t const&>(a16Scale1);
  uint32_t const scalePair = uint32_t(scaleBitsA16_0) |
                             (uint32_t(scaleBitsA16_1) << 16);
  uint32_t result;
  if constexpr (mha::is_same_v<A16, half>) {
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(result) : "r"(a16x2Bits), "r"(scalePair));
  } else {
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(result) : "r"(a16x2Bits), "r"(scalePair));
  }
  return result;
}

template <flashinfer::KVPageFormat format, typename A16>
__device__ inline void expandCompressedBlock16WithScale(uint32_t sf2, LdGrain& first,
                                                        LdGrain& second);

// Broadcast an A16 scale to both halves once per block; the per-pair multiply
// is then a single mul.rn.{bf16,f16}x2.
template <typename A16>
__device__ inline uint32_t broadcastA16Scale(uint16_t a16ScaleBits) {
  return uint32_t(a16ScaleBits) | (uint32_t(a16ScaleBits) << 16);
}

template <typename A16>
__device__ inline uint32_t mulA16x2(uint32_t x, uint32_t sf2) {
  uint32_t ret;
  if constexpr (mha::is_same_v<A16, half>) {
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(ret) : "r"(x), "r"(sf2));
  } else {
    static_assert(mha::is_same_v<A16, __nv_bfloat16>);
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(ret) : "r"(x), "r"(sf2));
  }
  return ret;
}

// Four E4M3 block scales (one 32-bit word) -> four A16 scales, bit-identical to
// static_cast<A16>(float(scale) * globalScale) per element.
template <typename A16>
__device__ inline Vec<uint16_t, 4> convertE4M3x4ScalesToA16Bits(uint32_t scaleWord,
                                                               float globalScale) {
  uint32_t lo16x2, hi16x2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(lo16x2) : "h"(static_cast<uint16_t>(scaleWord)));
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(hi16x2) : "h"(static_cast<uint16_t>(scaleWord >> 16)));
  float2 const lo = __half22float2(reinterpret_cast<__half2 const&>(lo16x2));
  float2 const hi = __half22float2(reinterpret_cast<__half2 const&>(hi16x2));
  Vec<uint16_t, 4> out;
  auto const pack = [&](float a, float b, uint32_t idx) {
    A16 const sa = static_cast<A16>(a * globalScale);
    A16 const sb = static_cast<A16>(b * globalScale);
    out[idx] = reinterpret_cast<uint16_t const&>(sa);
    out[idx + 1] = reinterpret_cast<uint16_t const&>(sb);
  };
  pack(lo.x, lo.y, 0);
  pack(hi.x, hi.y, 2);
  return out;
}

template <flashinfer::KVPageFormat format, typename A16>
__device__ inline void expandCompressedBlock16InPlace(
    uint8_t scaleBits, float globalScale, LdGrain& first, LdGrain& second) {
  expandCompressedBlock16WithScale<format, A16>(
      broadcastA16Scale<A16>(convertE4M3ScaleToA16Bits<A16>(scaleBits, globalScale)), first,
      second);
}

// sf2: the block's A16 scale broadcast to both 16-bit halves.
template <flashinfer::KVPageFormat format, typename A16>
__device__ inline void expandCompressedBlock16WithScale(uint32_t sf2, LdGrain& first,
                                                        LdGrain& second) {
  using flashinfer::KVPageFormat;
  static_assert(format == KVPageFormat::kBlockScaledFP8 ||
                format == KVPageFormat::kBlockScaledFP4);
  if constexpr (format == KVPageFormat::kBlockScaledFP8) {
    LdGrain const packed = first;
#pragma unroll
    for (uint32_t pair = 0; pair < 8; ++pair) {
      uint16_t const fp8x2 = reinterpret_cast<uint16_t const*>(&packed)[pair];
      uint32_t const scaled = mulA16x2<A16>(convertE4M3x2ToA16<A16>(fp8x2), sf2);
      if (pair < 4) {
        first[pair] = scaled;
      } else {
        second[pair - 4] = scaled;
      }
    }
  } else {
    Vec<uint32_t, 4> const convertedLo = convertE2M1x8ToA16<A16>(first[0]);
    Vec<uint32_t, 4> const convertedHi = convertE2M1x8ToA16<A16>(first[1]);
#pragma unroll
    for (uint32_t pair = 0; pair < 4; ++pair) {
      first[pair] = mulA16x2<A16>(convertedLo[pair], sf2);
      second[pair] = mulA16x2<A16>(convertedHi[pair], sf2);
    }
  }
}

template <typename A16>
__device__ inline void expandMixedBlock16InPlace(
    uint8_t format, uint8_t scaleBits, float fp8GlobalScale,
    float fp4GlobalScale, LdGrain& first, LdGrain& second) {
  using flashinfer::KVPageFormat;
  if (format == static_cast<uint8_t>(KVPageFormat::kA16)) return;
  if (format == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8)) {
    expandCompressedBlock16InPlace<KVPageFormat::kBlockScaledFP8, A16>(
        scaleBits, fp8GlobalScale, first, second);
  } else {
    assert(format == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4));
    expandCompressedBlock16InPlace<KVPageFormat::kBlockScaledFP4, A16>(
        scaleBits, fp4GlobalScale, first, second);
  }
}

template <uint32_t maxNbCopiedHeads, uint32_t nbPartsPerHead, bool swizzle,
          uint32_t dstNbHeads, uint32_t dstNbGrains, uint32_t nbPages,
          typename _LdGrain, uint32_t nbWarps = 1>
__device__ inline void expandMixedPartialHeadsInPlace(
    Array2D<_LdGrain, dstNbHeads, dstNbGrains>& dst,
    uint8_t const* scales, uint32_t dstHeadOffset,
    MixedPageFormats<nbPages> const& formats, uint32_t tokenOffset,
    uint32_t sourceHeadOffset, uint32_t idxPart, float fp8GlobalScale,
    float fp4GlobalScale, uint32_t idxWarp = 0) {
  constexpr uint32_t partBytes = exactDiv(sizeof(PaddedCacheHead), nbPartsPerHead);
  constexpr uint32_t blocksPerPart = exactDiv(partBytes, 2 * grainBytes);
  constexpr uint32_t totalBlocks = maxNbCopiedHeads * blocksPerPart;
  constexpr uint32_t nbThreads = nbWarps * warp_size;
  constexpr uint32_t iterations = divUp(totalBlocks, nbThreads);
  assert(idxWarp < nbWarps);
#pragma unroll
  for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
    uint32_t const linearBlock =
        iteration * nbThreads + idxWarp * warp_size + laneId();
    if (linearBlock >= totalBlocks) continue;
    uint32_t const localHead = linearBlock / blocksPerPart;
    uint32_t const blockInPart = linearBlock % blocksPerPart;
    uint32_t const absoluteToken = tokenOffset + sourceHeadOffset + localHead;
    uint32_t const localPage = absoluteToken / tokensPerPage;
    uint8_t const format =
#if MIXED_PAGE_STATIC_FORMAT >= 0
        MIXED_PAGE_STATIC_FORMAT;
#else
        formats.values[localPage];
#endif
    auto& first = dst.template at<swizzle>(dstHeadOffset + localHead, blockInPart * 2);
    auto& second = dst.template at<swizzle>(dstHeadOffset + localHead, blockInPart * 2 + 1);
    static_assert(validElemsPerHead % 64 == 0);
    constexpr uint32_t scaleLoadBytes = mha::max(4U, blocksPerPart);
    uint32_t const scaleBlock = idxPart * blocksPerPart + blockInPart;
    uint32_t const scaleGroup = scaleBlock & ~(scaleLoadBytes - 1);
    uint8_t const scaleBits =
        scales[localHead * scaleLoadBytes + scaleBlock - scaleGroup];
    expandMixedBlock16InPlace<InputElem>(format, scaleBits, fp8GlobalScale,
                                         fp4GlobalScale, first, second);
  }
  __syncwarp();
}

#endif  // ENABLE_MIXED_KV_CACHE

template <typename Head, uint32_t maxNbCopiedHeads, uint32_t nbWarps, uint32_t grainBytesSmem,
          uint32_t grainBytesGmem, bool swizzle, bool isFull, uint32_t dstNbHeads,
          typename SrcHeadPtr, typename _LdGrain, typename LocalHeadIdxMap = uint32_t (*)(uint32_t)>
__device__ inline void copyHeadsAsync(
    uint32_t idxWarp, Array2D<_LdGrain, dstNbHeads, exactDiv(sizeof(Head), grainBytesSmem)>& dst,
    SrcHeadPtr const& src, uint32_t nbAvailHeads = maxNbCopiedHeads,
    LocalHeadIdxMap&& localHeadIdxMap = [](uint32_t x) { return x; }) {
  assert(idxWarp < nbWarps);
  Warp const& warp = this_warp();
  constexpr uint32_t maxNbHeadsPerWarp = exactDiv(maxNbCopiedHeads, nbWarps);
  uint32_t const dstHeadOffset = maxNbHeadsPerWarp * idxWarp;
  uint32_t const warpNbAvailHeads =
      (dstHeadOffset < nbAvailHeads ? nbAvailHeads - dstHeadOffset : 0);
  constexpr uint32_t idxPart = 0;
  copyPartialHeadsAsync<Head, maxNbHeadsPerWarp, 1, grainBytesSmem, grainBytesGmem, swizzle, isFull,
                        dstNbHeads>(warp, dst, dstHeadOffset, src, idxPart, warpNbAvailHeads,
                                    [&](uint32_t x) { return localHeadIdxMap(dstHeadOffset + x); });
}

template <bool isAsync, uint32_t maxTotalNbGrains, uint32_t nbWarps, bool isFull = true>
__device__ inline void copyGrains(uint32_t idxWarp, LdGrain* dst, LdGrain const* src,
                                  uint32_t totalNbGrains = maxTotalNbGrains) {
  assert((isFull && totalNbGrains == maxTotalNbGrains) ||
         (!isFull && totalNbGrains <= maxTotalNbGrains));
  constexpr uint32_t nbThrds = warp_size * nbWarps;
  uint32_t const tid = warp_size * idxWarp + laneId();
// copy output to scratch
#pragma unroll
  for (uint32_t i = 0; i < divUp(maxTotalNbGrains, nbThrds); i++) {
    uint32_t const idx = nbThrds * i + tid;
    if (!(isFull && maxTotalNbGrains % nbThrds == 0) && idx >= totalNbGrains) {
      break;
    }
    if constexpr (isAsync) {
      ldgsts::copyAsync<grainBytes>(&dst[idx], &src[idx], grainBytes);
    } else {
      dst[idx] = src[idx];
    }
  }
}

// with ldmatrix, what we load for fp8 cache is T0:{e0,e1,e2,e3}; T1:{e4, e5, e6, e7};
// T2:{e8,e9,e10,e11}; T3:{e12, e13, e14, e15}; When casted to fp16, it will be T0:{e0, e1}; T1{e4,
// e5};...  | T0:{e2, e3}; T1{e6, e7}; ... We need to reorder Q to match that order. isFwd=false to
// revert the reorder.
template <uint32_t nbWarps, bool swizzled, bool isFwd, uint32_t cols, uint32_t rows>
__device__ inline void reorder16bQHeadsToMatch8bKCache(uint32_t idxWarp,
                                                       Array2D<LdGrain, rows, cols>& qHeads) {
  assert(idxWarp < nbWarps);
  constexpr uint32_t nbWarpIters = exactDiv(exactDiv(cols, 2) * rows, warp_size);  // warps * iters
  constexpr uint32_t nbWorkingWarps = mha::min(nbWarps, nbWarpIters);
  if (idxWarp >= nbWorkingWarps) {
    return;
  }
  static_assert(cols % 2 == 0);
  uint32_t const tid = warp_size * idxWarp + laneId();
  constexpr uint32_t iterCols = exactDiv(warp_size * nbWorkingWarps, rows) * 2;
  static_assert(cols % iterCols == 0,
                "fix this by reducing nbWorkingWarps, or use divUp and add runtime check");
  constexpr uint32_t nbIters = exactDiv(cols, iterCols);
  static_assert(nbIters == exactDiv(nbWarpIters, nbWorkingWarps));
  uint32_t const r = tid % rows;
  uint32_t const cInit = tid / rows * 2;
#pragma unroll
  for (uint32_t n = 0; n < nbIters; n++) {
    uint32_t const c = cInit + iterCols * n;
    LdGrain const src[2] = {
        qHeads.template at<swizzled>(r, c),
        qHeads.template at<swizzled>(r, c + 1),
    };
    auto const& s = reinterpret_cast<Vec<uint32_t, LdGrain::size * 2> const&>(src);
    if constexpr (isFwd) {
      qHeads.template at<swizzled>(r, c) = LdGrain{s[0], s[2], s[4], s[6]};
      qHeads.template at<swizzled>(r, c + 1) = LdGrain{s[1], s[3], s[5], s[7]};
    } else {
      qHeads.template at<swizzled>(r, c) = LdGrain{s[0], s[4], s[1], s[5]};
      qHeads.template at<swizzled>(r, c + 1) = LdGrain{s[2], s[6], s[3], s[7]};
    }
  }
}

template <bool usePagedKVCache>
struct KVCacheList;

template <>
struct KVCacheList<true> {
  GMemCacheHead* kCacheVLLM;
  GMemCacheHead* vCacheVLLM;
#if ENABLE_4BIT_KV_CACHE
  GMemCacheHeadSf* kSfCacheVLLM;
  GMemCacheHeadSf* vSfCacheVLLM;
#endif
#if ENABLE_MIXED_KV_CACHE
  PageTransport transport;
#endif
  KVCachePageIndex const*
      kvCachePageList;  // shape: KVCachePageIndex[batchSize][beamWidth][2][maxNbPagesPerSeq].
  SeqLenDataType const* seqLenList;  // shape: [batchSize][beamWidth] (for compatibility)
  uint32_t maxNbPagesPerSeq;
};

template <>
struct KVCacheList<false> {
  GMemKVCacheHead* data;  // shape: KVCacheHead[batchSize][beamWidth][2][nbKHeads][capacity]
  SeqLenDataType const* seqLenList;  // shape: [batchSize][beamWidth] (for compatibility)
  uint32_t capacity;
};

__device__ inline uint32_t getSeqLen(uint32_t const* seqLenList, uint32_t idxReq) {
  uint64_t cachePolicy;
  asm("createpolicy.fractional.L2::evict_last.b64 %0;\n" : "=l"(cachePolicy));
  uint32_t len;
  asm("ld.global.nc.L1::evict_last.L2::cache_hint.L2::256B.b32 %0, [%1], %2;\n"
      : "=r"(len)
      : "l"(&seqLenList[idxReq * beamWidth]), "l"(cachePolicy));
  for (uint32_t i = 0; i < beamWidth; i++) {
    assert(len == seqLenList[idxReq * beamWidth + i]);
  }
  return len;
}

template <bool isPaged>
__device__ inline uint32_t getCacheSeqLen(KVCacheList<isPaged> const& cacheList, uint32_t idxReq) {
  return getSeqLen(cacheList.seqLenList, idxReq);
}

__device__ inline uint32_t getCtxCacheSeqLen(BeamSearchParams const& beamSearchParams,
                                             uint32_t idxReq) {
  return getSeqLen(beamSearchParams.ctxLenList, idxReq);
}

template <uint32_t nbLoadedPages>
__device__ inline Vec<KVCachePageIndex, nbLoadedPages> getPage(KVCacheList<true> const& cacheList,
                                                               bool isK, uint32_t idxReq,
                                                               uint32_t idxBeam,
                                                               uint32_t idxPageBeg,
                                                               uint32_t nbPages) {
  auto const maxNbPagesPerSeq = cacheList.maxNbPagesPerSeq;
  Vec<KVCachePageIndex, nbLoadedPages> ret;
#pragma unroll
  for (uint32_t i = 0; i < nbLoadedPages; i++) {
    uint32_t const idxPage = idxPageBeg + i;
    ret[i] = (idxPage < nbPages ? cacheList.kvCachePageList[maxNbPagesPerSeq * idxReq + idxPage]
                                : kBAD_PAGE_INDEX);
  }
  return ret;
}

template <uint32_t nbWarps, uint32_t nbLoadedPages>
__device__ inline void loadPagesForBeamSearchAsync(
    uint32_t idxWarp, Vec<Vec<KVCachePageIndex, nbLoadedPages>, beamWidth>& dst,
    KVCacheList<true> const& cacheList, bool isK, uint32_t idxReq, uint32_t idxPageBeg,
    uint32_t nbPages) {
  assert(idxWarp < nbWarps);
  auto const maxNbPagesPerSeq = cacheList.maxNbPagesPerSeq;
  static_assert(beamWidth < warp_size);
  auto const tid = warp_size * idxWarp + laneId();
  auto const idxBeam = tid / nbLoadedPages;
  auto const idxLoadedPage = tid % nbLoadedPages;
  static_assert(warp_size * nbWarps >= beamWidth * nbLoadedPages);
  if (idxBeam < beamWidth) {
    constexpr uint32_t nbBytes = sizeof(KVCachePageIndex);
    uint32_t const idxPage = idxPageBeg + idxLoadedPage;
    ldgsts::copyAsync<nbBytes>(
        &dst[idxBeam][idxLoadedPage],
        &cacheList.kvCachePageList[beamWidth * 2 * maxNbPagesPerSeq * idxReq +
                                   2 * maxNbPagesPerSeq * idxBeam + (isK ? 0U : maxNbPagesPerSeq) +
                                   idxPage],
        idxPage < nbPages ? nbBytes : 0U);
  }
}

template <uint32_t nbWarps, uint32_t length, bool isFullTile = false>
__device__ inline void loadIndicesForBeamSearchAsync(uint32_t idxWarp, Vec<uint32_t, length>& dst,
                                                     BeamSearchParams const& params,
                                                     uint32_t idxReq, uint32_t idxBeam,
                                                     uint32_t uniformSeqOffset, uint32_t seqLen) {
  constexpr uint32_t nbThreads = warp_size * nbWarps;
  // constexpr uint32_t indicesPerInst = mha::min(exactDiv(grainBytes, sizeof(uint32_t)),
  // divUp(length, nbThreads));
  // // @fixme: std::bit_ceil on length
  constexpr uint32_t indicesPerInst = 1U;  // to handle unaligned case.
  constexpr uint32_t bytesPerInst = sizeof(uint32_t) * indicesPerInst;
  assertIsPowerOf2<indicesPerInst>();
  uint32_t const capacity = params.capacity;
  uint32_t const srcOffset = (idxReq * beamWidth + idxBeam) * capacity + uniformSeqOffset;
  uint32_t const tid = warp_size * idxWarp + laneId();
  constexpr uint32_t indicesPerIter = indicesPerInst * nbThreads;
#pragma unroll
  for (uint32_t i = 0; i < length / indicesPerIter; i++) {
    uint32_t const idx = indicesPerIter * i + indicesPerInst * tid;
    ldgsts::copyAsync<bytesPerInst>(
        &dst[idx], &params.indices[srcOffset + idx],
        (isFullTile || uniformSeqOffset + idx < seqLen) ? bytesPerInst : 0);
  }
  if constexpr (length % indicesPerIter != 0) {
    uint32_t const idx = indicesPerIter * (length / indicesPerIter) + indicesPerInst * tid;
    if (idx < length) {
      ldgsts::copyAsync<bytesPerInst>(
          &dst[idx], &params.indices[srcOffset + idx],
          (isFullTile || uniformSeqOffset + idx < seqLen) ? bytesPerInst : 0);
    }
  }
}

__device__ inline InputElem2 float2ToInputElem2(float2 src) {
  InputElem2 dst;
  if constexpr (mha::is_same_v<InputElem2, half2>) {
    reinterpret_cast<half2&>(dst) = __float22half2_rn(src);
    return dst;
  } else if constexpr (mha::is_same_v<InputElem2, nv_bfloat162>) {
    reinterpret_cast<nv_bfloat162&>(dst) = __float22bfloat162_rn(src);
    return dst;
  } else if constexpr (mha::is_same_v<InputElem2, __nv_fp8x2_e4m3>) {
    reinterpret_cast<__nv_fp8x2_e4m3&>(dst) = __nv_fp8x2_e4m3{src};
    return dst;
  } else {
    trap();
  }
}

template <bool real>
using TokenOrNone = RealTypeOrNone<real, CtaBarrier::arrival_token>;

template <bool real>
__device__ inline TokenOrNone<real> arrive(CtaBarrier* pBarrier) {
  if constexpr (real) {
    return pBarrier->arrive();
  } else {
    assert(pBarrier == nullptr);
    return None{};
  }
}

template <bool real>
__device__ inline void wait(CtaBarrier* pBarrier, TokenOrNone<real>&& token) {
  if constexpr (real) {
    pBarrier->wait(mha::move(token));
  } else {
    assert(pBarrier == nullptr);
    __syncwarp();
  }
}

template <bool real>
__device__ inline bool test_wait(CtaBarrier* pBarrier, TokenOrNone<real>&& token) {
  if constexpr (real) {
    uint32_t complete;
    asm volatile(
        "{\n"
        ".reg .pred       complete;\n"
        "mbarrier.test_wait.acquire.cta.shared::cta.b64 complete, [%1], %2;\n"
        "selp.b32 %0, 1, 0, complete;\n}\n"
        : "=r"(complete)
        : "l"(__cvta_generic_to_shared(pBarrier)), "l"(token));
    return bool(complete);
  } else {
    return false;
  }
}

template <bool real>
using ParityOrNone = RealTypeOrNone<real, bool>;

template <bool real>
__device__ inline void wait_parity(CtaBarrier* pBarrier, ParityOrNone<real> parity) {
  assert(real == (pBarrier != nullptr));
  if constexpr (real) {
    pBarrier->wait_parity(parity);
  } else {
    __syncwarp();
  }
}

template <bool real>
__device__ inline bool test_wait_parity(CtaBarrier* pBarrier, ParityOrNone<real> parity) {
  assert(real == (pBarrier != nullptr));
  if constexpr (real) {
#if USE_CUSTOM_BARRIER
    return pBarrier->test_wait_parity(parity);
#else
    return pBarrier->try_wait_parity_for(parity, cuda::std::chrono::nanoseconds(0));
#endif
  } else {
    return false;
  }
}

template <bool real = true>
__device__ inline ParityOrNone<real>& flip(ParityOrNone<real>& flip) {
  if constexpr (real) {
    flip = !flip;
  }
  return flip;
}

template <bool real = true>
__device__ inline ParityOrNone<real> getAndFlip(ParityOrNone<real>& flag) {
  ParityOrNone<real> const ret = flag;
  if constexpr (real) {
    flag = !flag;
  }
  return ret;
}
