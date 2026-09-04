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

// C2: read a register-resident vector element by index without local memory.  A
// runtime index into a Vec makes the compiler spill the whole vector (STL) and read
// it back (LDL); this compare/select chain stays in registers.  With a compile-time
// index (the mixed block loops, once the tile origin is known to be page-aligned) it
// folds to the element itself.
template <typename T, uint32_t n>
__device__ inline T selectByIndex(Vec<T, n> const& v, uint32_t idx) {
  T ret = v[0];
#pragma unroll
  for (uint32_t i = 1; i < n; ++i) {
    ret = (idx == i) ? v[i] : ret;
  }
  return ret;
}

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
    // pageIndices lives in registers; a lane-dependent index must not spill it (C2).
    auto const pageIdx =
        selectByIndex(pageIndices, nbPages == 1 ? 0U : absoluteTokenIdx / tokensPerPage);
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
  KVCachePageIndex const page = lane < nbPages ? selectByIndex(pages, lane) : kBAD_PAGE_INDEX;
  if (page != kBAD_PAGE_INDEX) {
    value = transport.page_format[page];
  }
#pragma unroll
  for (uint32_t i = 0; i < nbPages; ++i) {
    ret.values[i] = static_cast<uint8_t>(__shfl_sync(0xffffffffU, value, i));
  }
  return ret;
}

// Two-step form of gatherMixedPageFormats for a loader that prefetches page
// indices two tiles ahead: issue this lane's tag load as soon as the indices
// have landed (no consumer -> no stall) ...
template <uint32_t nbPages>
__device__ inline uint32_t mixedPageTagLane(PageTransport const& transport,
                                            Vec<KVCachePageIndex, nbPages> const& pages) {
  uint32_t const lane = laneId();
  uint32_t value = 0;
  KVCachePageIndex const page = lane < nbPages ? selectByIndex(pages, lane) : kBAD_PAGE_INDEX;
  if (page != kBAD_PAGE_INDEX) {
    value = transport.page_format[page];
  }
  return value;
}

// ... and broadcast it a tile later, when it is needed.
template <uint32_t nbPages>
__device__ inline MixedPageFormats<nbPages> broadcastMixedPageTags(uint32_t laneValue) {
  MixedPageFormats<nbPages> ret;
#pragma unroll
  for (uint32_t i = 0; i < nbPages; ++i) {
    ret.values[i] = static_cast<uint8_t>(__shfl_sync(0xffffffffU, laneValue, i));
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

// MIXED_KV_PROBE_C (measurement build only, plan P0.8 probe (c)): FP8 K-part copies fetch
// 64 B per token with the same LDGSTS count and the same per-line request pattern.
//   1: the lane's own 16-B block (sector p of the token's 128-B FP8 row) plus the same
//      block of the neighbouring part (sector p^1) -> 2x distinct sectors per request.
//   2: control -- the own block twice (same sector) -> same sectors, 16-B lanes.
// The shadow copy lands in dead shared memory (SharedMem::probeScratch); K only.
#ifndef MIXED_KV_PROBE_C
#define MIXED_KV_PROBE_C 0
#endif

// FP8 block payload copy in the expansion form: 0 = two 8 B cp.async.ca, 1 = one 16 B
// cp.async.ca (default), 2 = one 16 B cp.async.cg (measurement only, see below).
#ifndef MIXED_FP8_COPY
#define MIXED_FP8_COPY 1
#endif

// Per-page format dispatch ([40], Track S step 3).  A warp instruction of the
// copy or of the expansion covers blocks of a single page (blocksPerSpan is a
// multiple of the warp size for every supported part width), so the page format
// is warp-uniform per page: the block loop is page-outer, one branch per page
// selects a format-specialised body, and the dynamic module
// (MIXED_PAGE_STATIC_FORMAT < 0) rolls the page loop so each call site carries
// one A16 + one FP8 + one FP4 body instead of nbPages x 3 predicated variants
// (the 17.9 K-instruction sm90 SPEC_DEC kernel that stalled on instruction
// fetch).  Static modules unroll the page loop: the format folds and the branch
// vanishes.
template <uint8_t f>
struct MixedFormatTag {
  static constexpr uint8_t value = f;
};

constexpr uint32_t mixedPageLoopUnroll(uint32_t nbPageSpans) {
  return MIXED_PAGE_STATIC_FORMAT < 0 ? 1U : nbPageSpans;
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
    MixedPageFormats<nbPages> const& formats, uint32_t sourceHeadOffset,
    uint32_t headIdx, bool isK, uint32_t idxPart, uint32_t nbAvailHeads = maxNbCopiedHeads,
    uint32_t idxWarp = 0, uint8_t* probeScratch = nullptr) {
  // The tile origin is page-aligned (callers static_assert it), so a span of
  // headsPerSpan heads lies in one page: pages[] / formats[] are read once per
  // span (a compare/select chain over the register vector, no local memory).
  static_assert(sizeof(PaddedCacheHead) % 32 == 0);
  constexpr uint32_t partBytes = exactDiv(sizeof(PaddedCacheHead), nbPartsPerHead);
  constexpr uint32_t grainsPerPart = exactDiv(partBytes, grainBytes);
  constexpr uint32_t blocksPerPart = exactDiv(grainsPerPart, 2);
  constexpr uint32_t nbThreads = nbWarps * warp_size;
  constexpr uint32_t headsPerSpan = mha::min(tokensPerPage, maxNbCopiedHeads);
  static_assert(maxNbCopiedHeads % headsPerSpan == 0 && tokensPerPage % headsPerSpan == 0);
  constexpr uint32_t nbSpans = exactDiv(maxNbCopiedHeads, headsPerSpan);
  constexpr uint32_t blocksPerSpan = headsPerSpan * blocksPerPart;
  constexpr uint32_t iterationsPerSpan = divUp(blocksPerSpan, nbThreads);
  constexpr uint32_t pageLoopUnroll = mixedPageLoopUnroll(nbSpans);
  static_assert(!compactPages || tokensPerPage == 16,
                "compact mixed-page fragments require the vLLM 16-token page unit");
  assert(idxWarp < nbWarps);
  using flashinfer::KVPageFormat;
  uint8_t constexpr a16Format = static_cast<uint8_t>(KVPageFormat::kA16);
  uint8_t constexpr fp8Format = static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8);
  uint8_t constexpr fp4Format = static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4);

  auto const copySpan = [&](uint32_t span, KVCachePageIndex page, auto formatTag) {
    constexpr uint8_t format = decltype(formatTag)::value;
    constexpr bool isA16 = format == a16Format;
    constexpr bool isFP8 = format == fp8Format;
    constexpr bool isFP4 = format == fp4Format;
    static_assert(isA16 || isFP8 || isFP4);
    auto const& fmt = transport.formats[format];
    auto const* payload = static_cast<uint8_t const*>(isK ? fmt.k_payload : fmt.v_payload);
    bool const pageValid = page != kBAD_PAGE_INDEX;
    uint32_t const spanHead0 = span * headsPerSpan;
    uint32_t const token0 = (sourceHeadOffset + spanHead0) % tokensPerPage;
    // The page's own format also for !valid blocks: their payload copies are zero-fills
    // and the expansion of a zero payload is zero, i.e. the same tile bytes the former
    // "treat as A16 and zero-fill 32 B" produced, without a lane-varying format.
    uint64_t const pageBase = pageValid ? uint64_t(page) * fmt.payload_stride.page +
                                              uint64_t(headIdx) * fmt.payload_stride.head
                                        : 0;
#pragma unroll
    for (uint32_t iteration = 0; iteration < iterationsPerSpan; ++iteration) {
      uint32_t const blockInSpan = iteration * nbThreads + idxWarp * warp_size + laneId();
      if (blockInSpan >= blocksPerSpan) continue;
      uint32_t const headInSpan = blockInSpan / blocksPerPart;
      uint32_t const blockInPart = blockInSpan % blocksPerPart;
      uint32_t const localHead = spanHead0 + headInSpan;
      uint32_t const token = token0 + headInSpan;
      bool const validHead = isFull || localHead < nbAvailHeads;
      uint32_t const elem = (idxPart * blocksPerPart + blockInPart) * 16;
      bool const validElem = elem + 16 <= validElemsPerHead;
      bool const valid = validHead && validElem && pageValid;
      constexpr uint32_t payloadElemScale = isA16 ? sizeof(InputElem) : (isFP8 ? 1U : 0U);
      uint64_t const payloadElemOffset =
          isFP4 ? uint64_t(elem / 2) : uint64_t(elem) * payloadElemScale;
      auto const* firstSource = payload + pageBase +
                                (pageValid ? uint64_t(token) * fmt.payload_stride.token : 0) +
                                payloadElemOffset;

      if constexpr (compactPages) {
        if constexpr (isA16) {
          auto* first = &dst.template at<swizzle>(dstHeadOffset + localHead, blockInPart * 2);
          auto* second =
              &dst.template at<swizzle>(dstHeadOffset + localHead, blockInPart * 2 + 1);
          ldgsts::copyAsync<grainBytes>(first, firstSource, valid ? grainBytes : 0U);
          ldgsts::copyAsync<grainBytes>(second, firstSource + grainBytes,
                                       valid ? grainBytes : 0U);
        } else {
          // Retain the native tile row stride and place the compressed block in
          // the low half of that row.  This preserves ldmatrix-compatible row
          // addressing while keeping each page in a fixed-size slot.
          auto* packed = &dst.template at<swizzle>(dstHeadOffset + localHead, blockInPart);
          if constexpr (isFP4) {
            ldgsts::copyAsync<8>(packed, firstSource, valid ? 8U : 0U);
            ldgsts::copyAsync<8>(reinterpret_cast<uint8_t*>(packed) + 8, firstSource + 8, 0U);
          } else {
            ldgsts::copyAsync<grainBytes>(packed, firstSource, valid ? grainBytes : 0U);
          }
        }
      } else {
        auto* first = &dst.template at<swizzle>(dstHeadOffset + localHead, blockInPart * 2);
        auto* second =
            &dst.template at<swizzle>(dstHeadOffset + localHead, blockInPart * 2 + 1);
        bool probeTaken = false;
#if MIXED_KV_PROBE_C
        // Only the K instantiation (4 parts of 64 A16 bytes -> FP8 part = 32 B = one sector
        // of the token's 128-B row) carries the probe; V instantiations compile it out.
        if constexpr (isFP8 && nbPartsPerHead == 4 && partBytes == 64) {
          if (probeScratch != nullptr) {
            probeTaken = true;
            constexpr uint32_t fp8PartBytes = partBytes / 2;  // 32 B
            uint8_t const* shadowSource =
                (MIXED_KV_PROBE_C == 1)
                    ? firstSource +
                          ((idxPart & 1) ? -ptrdiff_t(fp8PartBytes) : ptrdiff_t(fp8PartBytes))
                    : firstSource;
            ldgsts::copyAsyncCa16(first, firstSource, valid ? 16U : 0U);
            ldgsts::copyAsyncCa16(probeScratch + laneId() * 16, shadowSource, valid ? 16U : 0U);
            ldgsts::copyAsync<grainBytes>(second, firstSource + grainBytes, 0U);
          }
        }
#endif
        if (!probeTaken) {
          // Expansion form: expandMixedPartialHeadsInPlace rewrites `second` (and, for
          // FP4, the upper 8 B of `first`) from the packed payload before anything reads
          // them, so those grains are not zero-filled here: one LDGSTS per compressed
          // block - FP4 8 B, FP8 the whole 16 B packed block as one L1-allocating
          // cp.async.ca (Track W [26]; was two 8 B halves.  cp.async.cg 16 B measured
          // 122 -> 177 us on the sm120 fp8 q=4 build: the L1-bypassing path does not
          // merge the lanes' 16 B pieces of a sector).  A16 blocks copy their full 32 B
          // as two grains (the stock copyPartialHeadsAsync pattern; the dynamic module
          // has no other A16 path).
          if constexpr (isA16) {
            ldgsts::copyAsync<grainBytes>(first, firstSource, valid ? grainBytes : 0U);
            ldgsts::copyAsync<grainBytes>(second, firstSource + grainBytes,
                                         valid ? grainBytes : 0U);
          } else if constexpr (isFP4) {
            ldgsts::copyAsync<8>(first, firstSource, valid ? 8U : 0U);
          } else {
#if MIXED_FP8_COPY == 0
            ldgsts::copyAsync<8>(first, firstSource, valid ? 8U : 0U);
            ldgsts::copyAsync<8>(reinterpret_cast<uint8_t*>(first) + 8, firstSource + 8,
                                 valid ? 8U : 0U);
#elif MIXED_FP8_COPY == 1
            ldgsts::copyAsyncCa16(first, firstSource, valid ? 16U : 0U);
#else
            ldgsts::copyAsync<grainBytes>(first, firstSource, valid ? grainBytes : 0U);
#endif
          }
        }
      }
    }
  };

#pragma unroll(pageLoopUnroll)
  for (uint32_t span = 0; span < nbSpans; ++span) {
    uint32_t const localPage = (sourceHeadOffset + span * headsPerSpan) / tokensPerPage;
    KVCachePageIndex const page =
        localPage < nbPages ? selectByIndex(pages, localPage) : kBAD_PAGE_INDEX;
#if MIXED_PAGE_STATIC_FORMAT >= 0
    unused(formats);
    copySpan(span, page, MixedFormatTag<MIXED_PAGE_STATIC_FORMAT>{});
#else
    uint8_t const format =
        localPage < nbPages ? selectByIndex(formats.values, localPage) : a16Format;
    if (format == a16Format) {
      copySpan(span, page, MixedFormatTag<a16Format>{});
    } else if (format == fp8Format) {
      copySpan(span, page, MixedFormatTag<fp8Format>{});
    } else {
      assert(format == fp4Format);
      copySpan(span, page, MixedFormatTag<fp4Format>{});
    }
#endif
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
    uint32_t const absoluteToken = sourceHeadOffset + localHead;
    // One lane per head: the page is lane / tokensPerPage plus an iteration constant,
    // a compare/select chain over the register vector (no local memory).
    uint32_t const localPage = absoluteToken / tokensPerPage;
    KVCachePageIndex const page =
        localPage < nbPages ? selectByIndex(pages, localPage) : kBAD_PAGE_INDEX;
    uint32_t const token = absoluteToken % tokensPerPage;
    bool const valid = validHead && page != kBAD_PAGE_INDEX;
    uint8_t const format =
#if MIXED_PAGE_STATIC_FORMAT >= 0
        valid ? MIXED_PAGE_STATIC_FORMAT : 0;
#else
        valid ? selectByIndex(formats.values, localPage) : 0;
#endif
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
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(bf16x2) : "h"(fp8x2));
    return bf16x2;
  }
#endif
  uint32_t fp16x2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(fp16x2) : "h"(fp8x2));
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
    asm(
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
  asm(
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

// ---- sm90 BF16 decode by bit placement (no e4m3x2/e2m1x2 -> bf16x2 cvt before sm100) ----
//
// An E4M3 byte s eeee mmm placed into a BF16 lane as s at bit 15, eeee at bits [10:7] and
// mmm at bits [6:4] is exactly x * 2^-120 for every finite code (the BF16 exponent field
// holds the 4-bit exponent unbiased; E4M3 subnormals m * 2^-9 land as the BF16 subnormals
// m * 2^-129, and mul.rn.bf16x2 handles subnormal inputs exactly - verified exhaustively
// on H200).  Likewise an E2M1 nibble s ee m placed as s at bit 15, ee at bits [8:7] and m
// at bit 6 is mag(n) * 2^-126.  The power of two is folded into the block scale (E4M3) or
// undone by one extra packed multiply (E2M1, whose block scales reach 448 and would not fit
// a 2^126 fold).  Per four values: two PRMT (byte spread + sign replicate), two shifts, two
// masks - four SASS per pair with the multiply instead of the five of the f16 detour.
__device__ inline uint32_t prmtSelfB32(uint32_t w, uint32_t selector) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %1, %2;" : "=r"(d) : "r"(w), "r"(selector));
  return d;
}
__device__ inline uint32_t prmtB32(uint32_t a, uint32_t b, uint32_t selector) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(selector));
  return d;
}
// Four E4M3 bytes (values 0..3 of a 16 B block word) -> BF16x2 pairs {v1:v0}, {v3:v2}, each
// value scaled by 2^-120.
__device__ inline void e4m3x4ToBF16x2Pow2m120(uint32_t w, uint32_t& lo, uint32_t& hi) {
  // [rep(sign b1), b1, rep(sign b0), b0] and [rep(sign b3), b3, rep(sign b2), b2].
  uint32_t const a = prmtSelfB32(w, 0x9180u);
  uint32_t const b = prmtSelfB32(w, 0xB3A2u);
  // << 4: byte -> bits [11:4] (sign copy at 11 cleared by the mask), replicated sign -> 15.
  lo = (a << 4) & 0x87F087F0u;
  hi = (b << 4) & 0x87F087F0u;
}
// Eight E2M1 nibbles (one packed word, value i = nibble i) -> four BF16x2 pairs, each value
// scaled by 2^-126.  out[k] = {v(2k+1) : v(2k)} from byte k: the even nibble's sign is moved
// to the byte's bit 7 by w << 4, the odd nibble's is there already.
__device__ inline void e2m1x8ToBF16x2Pow2m126(uint32_t w, uint32_t (&out)[4]) {
  uint32_t const w4 = w << 4;
#pragma unroll
  for (uint32_t k = 0; k < 4; k++) {
    // [rep(sign w byte k), w byte k, rep(sign w4 byte k), w4 byte k]
    uint32_t const sel = ((0xCu + k) << 12) | ((4u + k) << 8) | ((8u + k) << 4) | k;
    uint32_t const a = prmtB32(w4, w, sel);
    // << 2: s ee m of each nibble -> bits [9:6]; keep the replicated sign at 15 and ee m.
    out[k] = (a << 2) & 0x81C081C0u;
  }
}
// Four E4M3 block scales -> float, times `mul`.
__device__ inline void e4m3x4ScalesToFloat(uint32_t scaleWord, float mul, float (&out)[4]) {
  uint32_t lo16x2, hi16x2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(lo16x2) : "h"(static_cast<uint16_t>(scaleWord)));
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(hi16x2) : "h"(static_cast<uint16_t>(scaleWord >> 16)));
  float2 const lo = __half22float2(reinterpret_cast<__half2 const&>(lo16x2));
  float2 const hi = __half22float2(reinterpret_cast<__half2 const&>(hi16x2));
  out[0] = lo.x * mul;
  out[1] = lo.y * mul;
  out[2] = hi.x * mul;
  out[3] = hi.y * mul;
}
__device__ inline uint32_t bf16x2BitsFromFloats(float a, float b) {
  __nv_bfloat162 const v = __float22bfloat162_rn(float2{a, b});
  return reinterpret_cast<uint32_t const&>(v);
}

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

// Shared-window accessors for the expansion (32-bit shared addresses; ptxas folds the
// per-iteration constants into the LDS/STS immediate).
__device__ inline uint32_t smemAddr(void const* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ inline LdGrain ldsGrain(uint32_t addr) {
  LdGrain v;
  asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
               : "r"(addr));
  return v;
}
__device__ inline uint32_t ldsU8(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.u8 %0, [%1];" : "=r"(v) : "r"(addr));
  return v;
}
__device__ inline void stsGrain(uint32_t addr, LdGrain const& v) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(v[0]), "r"(v[1]),
               "r"(v[2]), "r"(v[3])
               : "memory");
}

template <uint32_t maxNbCopiedHeads, uint32_t nbPartsPerHead, bool swizzle,
          uint32_t dstNbHeads, uint32_t dstNbGrains, uint32_t nbPages,
          typename _LdGrain, uint32_t nbWarps = 1>
__device__ inline void expandMixedPartialHeadsInPlace(
    Array2D<_LdGrain, dstNbHeads, dstNbGrains>& dst,
    uint8_t const* scales, uint32_t dstHeadOffset,
    MixedPageFormats<nbPages> const& formats,
    uint32_t sourceHeadOffset, uint32_t idxPart, float fp8GlobalScale,
    float fp4GlobalScale, uint32_t idxWarp = 0) {
  // Page-outer like copyMixedPartialHeadsAsync ([40]): one format branch per page
  // span, a format-specialised body for its blocks, the page loop rolled in the
  // dynamic module.  A16 spans are skipped.
  // Offloaded-KV discipline (docs/mixed_kv_page_transport_targets.md): one shared-window
  // base per array with per-iteration constant offsets, and independent block bodies -
  // each block is LDS.128 (+ LDS.U8 scale) -> registers -> convert -> 2 x STS.128; no
  // generic LD/ST, no local memory.  `second` is never read (the packed payload lives in
  // `first`).
  using Tile = Array2D<_LdGrain, dstNbHeads, dstNbGrains>;
  static_assert(sizeof(_LdGrain) == grainBytes);
  constexpr uint32_t partBytes = exactDiv(sizeof(PaddedCacheHead), nbPartsPerHead);
  constexpr uint32_t blocksPerPart = exactDiv(partBytes, 2 * grainBytes);
  constexpr uint32_t nbThreads = nbWarps * warp_size;
  constexpr uint32_t headsPerSpan = mha::min(tokensPerPage, maxNbCopiedHeads);
  static_assert(maxNbCopiedHeads % headsPerSpan == 0 && tokensPerPage % headsPerSpan == 0);
  constexpr uint32_t nbSpans = exactDiv(maxNbCopiedHeads, headsPerSpan);
  constexpr uint32_t blocksPerSpan = headsPerSpan * blocksPerPart;
  constexpr uint32_t iterationsPerSpan = divUp(blocksPerSpan, nbThreads);
  constexpr uint32_t pageLoopUnroll = mixedPageLoopUnroll(nbSpans);
  static_assert(validElemsPerHead % 64 == 0);
  constexpr uint32_t scaleLoadBytes = mha::max(4U, blocksPerPart);
  static_assert(warp_size % blocksPerPart == 0,
                "a lane's block column must be the same in every iteration");
  constexpr uint32_t rowsPerIter = exactDiv(nbThreads, blocksPerPart);
  assert(idxWarp < nbWarps);
  using flashinfer::KVPageFormat;
  uint8_t constexpr a16Format = static_cast<uint8_t>(KVPageFormat::kA16);
  uint8_t constexpr fp8Format = static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8);
  uint8_t constexpr fp4Format = static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4);

  // This lane's block column and first row; later iterations / spans add constants.
  uint32_t const lane = laneId();
  uint32_t const blockInPart = lane % blocksPerPart;
  uint32_t const headInSpan0 = (idxWarp * warp_size + lane) / blocksPerPart;
  uint32_t const scaleBlock = idxPart * blocksPerPart + blockInPart;
  uint32_t const tileBase = smemAddr(&dst);
  uint32_t const scaleBase =
      smemAddr(scales) + headInSpan0 * scaleLoadBytes + (scaleBlock & (scaleLoadBytes - 1));

  auto const expandSpan = [&](uint32_t span, auto formatTag) {
    constexpr uint8_t formatValue = decltype(formatTag)::value;
    static_assert(formatValue == fp8Format || formatValue == fp4Format);
    constexpr KVPageFormat format = static_cast<KVPageFormat>(formatValue);
    float const globalScale = formatValue == fp8Format ? fp8GlobalScale : fp4GlobalScale;
    uint32_t const spanHead0 = span * headsPerSpan;
#pragma unroll
    for (uint32_t iteration = 0; iteration < iterationsPerSpan; ++iteration) {
      uint32_t const headInSpan = headInSpan0 + iteration * rowsPerIter;
      if (headsPerSpan % rowsPerIter != 0 && headInSpan >= headsPerSpan) continue;
      uint32_t const localHead = spanHead0 + headInSpan;
      uint32_t const row = dstHeadOffset + localHead;
      uint32_t const firstAddr =
          tileBase + Tile::template byteOffset<swizzle>(row, blockInPart * 2);
      uint32_t const secondAddr =
          tileBase + Tile::template byteOffset<swizzle>(row, blockInPart * 2 + 1);
      uint8_t const scaleBits =
          static_cast<uint8_t>(ldsU8(scaleBase + localHead * scaleLoadBytes -
                                     headInSpan0 * scaleLoadBytes));
      LdGrain first = ldsGrain(firstAddr);
      LdGrain second{};
      expandCompressedBlock16InPlace<format, InputElem>(scaleBits, globalScale, first, second);
      stsGrain(firstAddr, first);
      stsGrain(secondAddr, second);
    }
  };

#if MIXED_PAGE_STATIC_FORMAT == 0
  unused(formats);
  unused(expandSpan);
  unused(sourceHeadOffset);
  unused(a16Format);
#else
#pragma unroll(pageLoopUnroll)
  for (uint32_t span = 0; span < nbSpans; ++span) {
#if MIXED_PAGE_STATIC_FORMAT > 0
    unused(formats);
    unused(sourceHeadOffset);
    unused(a16Format);
    expandSpan(span, MixedFormatTag<MIXED_PAGE_STATIC_FORMAT>{});
#else
    uint32_t const localPage = (sourceHeadOffset + span * headsPerSpan) / tokensPerPage;
    assert(localPage < nbPages);
    uint8_t const format = selectByIndex(formats.values, localPage);
    if (format == fp8Format) {
      expandSpan(span, MixedFormatTag<fp8Format>{});
    } else if (format == fp4Format) {
      expandSpan(span, MixedFormatTag<fp4Format>{});
    } else {
      assert(format == a16Format);
    }
#endif
  }
#endif
  __syncwarp();
}


// ---- [44] Track S step 6 (sm90 SPEC_DEC bf16 build): placement decode + folded scale ----
//
// Selected by mha.cu's MIXED_BF16_PLACEMENT_EXPANSION at the two expansion call sites; every
// other build keeps expandMixedPartialHeadsInPlace above byte-for-byte.  The block expanders
// mirror mha_sm90.cu's expandE4M3BlockBF16 / expandE2M1BlockBF16 (that file is owned by another
// track and is not included here).
//
// Shared-window loads the expansion needs besides ldsGrain / stsGrain.
__device__ inline uint32_t ldsU16(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=r"(v) : "r"(addr));
  return v;
}
__device__ inline void ldsB64(uint32_t addr, uint32_t& lo, uint32_t& hi) {
  asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];" : "=r"(lo), "=r"(hi) : "r"(addr));
}
// Two E4M3 block scales (bytes 0 and 1 of s01) -> fp32, exact (E4M3 -> f16 -> f32 are both
// exact embeddings; the same route as e4m3x4ScalesToFloat / convertE4M3ScaleToA16Bits).
__device__ inline void e4m3x2ScalesToFloat(uint32_t s01, float& f0, float& f1) {
  uint32_t f16x2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16x2) : "h"(static_cast<uint16_t>(s01)));
  float2 const f = __half22float2(reinterpret_cast<__half2 const&>(f16x2));
  f0 = f.x;
  f1 = f.y;
}
// bf16 scale broadcast to both halves: one F2FP.BF16.PACK_AB with both inputs equal.
__device__ inline uint32_t bf16x2Broadcast(float f) { return bf16x2BitsFromFloats(f, f); }
template <bool kFold>
struct MixedFoldTag {
  static constexpr bool value = kFold;
};
// The warp's fold decision for one page span: every lane's two scaled block scales
// (s * g * 2^k in fp32) stay finite in bf16 (bound = bf16 max + half an ulp: 255.5 * 2^120), and
// |g| >= 2^-117 so that every s * g is fp32-normal (the smallest nonzero E4M3 scale is 2^-9),
// which is what makes bf16_rn(s * g * 2^k) == bf16_rn(s * g) * 2^k.  foldOk is warp-uniform; it
// is inside the vote so the result is uniform by construction.
__device__ inline bool foldScalePairFinite(float f0, float f1, bool foldOk) {
  float const fmax = fmaxf(fabsf(f0), fabsf(f1));
  return __all_sync(0xFFFFFFFFu, foldOk && fmax < 255.5f * 0x1p120f);
}
// One 16-value E4M3 block (16 packed bytes) -> 32 bf16 bytes.  kFold: sf2 = bf16(s * g * 2^120)
// broadcast; else sf2 = bf16(s * g) and the placed value (x * 2^-120) is first multiplied by
// exactly 2^120 (0x7B80; E4M3 magnitudes have <= 4 significant bits, x < 449, so the product is
// exact), giving the reference's single rounding either way.
template <bool kFold>
__device__ inline void expandE4M3Block16BF16Placed(LdGrain const& packed, uint32_t sf2,
                                                   LdGrain (&out)[2]) {
  constexpr uint32_t kTwoPow120x2 = 0x7B807B80u;
#pragma unroll
  for (uint32_t w = 0; w < 4; w++) {
    uint32_t lo, hi;
    e4m3x4ToBF16x2Pow2m120(packed[w], lo, hi);
    if constexpr (!kFold) {
      lo = mulA16x2<__nv_bfloat16>(lo, kTwoPow120x2);
      hi = mulA16x2<__nv_bfloat16>(hi, kTwoPow120x2);
    }
    out[w / 2][(w % 2) * 2] = mulA16x2<__nv_bfloat16>(lo, sf2);
    out[w / 2][(w % 2) * 2 + 1] = mulA16x2<__nv_bfloat16>(hi, sf2);
  }
}
// One 16-value E2M1 block (8 packed bytes: packed0 = values 0-7, packed1 = 8-15) -> 32 bf16
// bytes; placement is mag * 2^-126, the fold constant 2^126 (0x7E80).
template <bool kFold>
__device__ inline void expandE2M1Block16BF16Placed(uint32_t packed0, uint32_t packed1,
                                                   uint32_t sf2, LdGrain (&out)[2]) {
  constexpr uint32_t kTwoPow126x2 = 0x7E807E80u;
#pragma unroll
  for (uint32_t h = 0; h < 2; h++) {
    uint32_t v[4];
    e2m1x8ToBF16x2Pow2m126(h == 0 ? packed0 : packed1, v);
#pragma unroll
    for (uint32_t k = 0; k < 4; k++) {
      if constexpr (!kFold) {
        v[k] = mulA16x2<__nv_bfloat16>(v[k], kTwoPow126x2);
      }
      out[h][k] = mulA16x2<__nv_bfloat16>(v[k], sf2);
    }
  }
}

// In-place expansion of one 128 B K part (maxNbCopiedHeads rows) or one 32-row V half-tile
// with the bit-placement decode and the block scale folded with 2^k, half-row owner cut: lane l
// owns token tok = l % 16 of the span (page) and blocks 2h, 2h+1 with h = l / 16 (32 values), so
// the scale pair is one LDS.U16 and the vote / scale prep amortise over 32 values.  The copy's
// ownership is different (blockInSpan = 32 * iteration + lane: row lane / 4 + 8 * iteration,
// block lane % 4), hence the __syncwarp on entry.  Geometry is fixed by static_assert (128 B
// parts, 16-token pages, 4 B scale words, one warp, 128 B swizzled rows); any other
// instantiation fails to compile.
//
// Data flow, one call:
//   tile rows (128 B, physical grain = logical ^ (row % 8)); compressed block b of a row sits in
//   logical grain 2b (FP8: 16 B; FP4: 8 B in the low half), staged by the copy; one 4 B scale
//   word per row (byte b = E4M3 block scale of block b), staged by the copy's scale loop.
//   Lane: row r = 16 * span + tok, x = tok % 8 = r % 8 for every span; logical grains 4h + j
//   (j = 0..3) are physical (4h ^ x) ^ j - a permutation of one 4-aligned group that depends on
//   x & 3, so the lane holds four u32 addresses a_j = tileBase + byteOffset(tok, 4h + j), each
//   used as [a_j + 2048 * span]; the scale pair is bytes 2h, 2h+1 of the row's word:
//   [scales + 4 * tok + 2h + 64 * span].
//   Per span: LDS.U16 -> fp32 s_0, s_1 (exact) -> f_i = s_i * g * 2^k -> warp vote ->
//   [fold] sf2_i = bf16x2(f_i) | [fallback] sf2_i = bf16x2(s_i * g);
//   block 2h:   LDS.128 / LDS.64 [a_0] -> placement decode -> (fallback: x 2^k) -> x sf2_0 ->
//               STS.128 [a_0], STS.128 [a_1]
//   block 2h+1: LDS.128 / LDS.64 [a_2] -> ... x sf2_1 -> STS.128 [a_2], STS.128 [a_3]
//   A lane writes only grains it alone reads (its sources a_0, a_2 are among its outputs), so no
//   barrier is needed between its loads and stores.  Bank behaviour: an 8-lane LDS/STS phase is
//   8 consecutive rows at one logical grain -> 8 distinct physical grains, conflict-free; the
//   LDS.U16 phase reads 16 consecutive words (lanes 16-31 the same words' upper halves).
// Control flow:
//   __syncwarp  (cp.async.wait_group completes the executing thread's copies only; the payload
//                grains of row tok were copied by lanes 4 * (tok % 8) + 2h, + 2h + 1 at
//                iteration tok / 8, the scale word by lane tok + 16 * (span % 2))
//   for span (unrolled in static modules, rolled in the dynamic one):
//     format (warp-uniform: per-page tag; the dynamic module branches, static ones fold)
//     scales -> vote (convergent: inside the warp-uniform branch, all 32 lanes)
//     fold ? body<true> : body<false>   (both straight-line over the lane's two blocks)
//   __syncwarp  (before the warp's ldmatrix reads)
template <uint32_t maxNbCopiedHeads, uint32_t nbPartsPerHead, bool swizzle, uint32_t dstNbHeads,
          uint32_t dstNbGrains, uint32_t nbPages, typename _LdGrain>
__device__ inline void expandMixedPartialHeadsInPlaceBF16Placement(
    Array2D<_LdGrain, dstNbHeads, dstNbGrains>& dst, uint8_t const* scales,
    MixedPageFormats<nbPages> const& formats, uint32_t sourceHeadOffset, float fp8GlobalScale,
    float fp4GlobalScale) {
  using Tile = Array2D<_LdGrain, dstNbHeads, dstNbGrains>;
  // Every static_assert below is dependent on a template parameter so that a mixed build that
  // never instantiates this helper (fp16 input, other page sizes, sm120) compiles unchanged.
  static_assert(sizeof(_LdGrain) == grainBytes && mha::is_same_v<InputElem, __nv_bfloat16>,
                "placement decode is bf16-only");
  static_assert(sizeof(_LdGrain) == grainBytes && grainBytes == 16);
  static_assert(swizzle, "the cut assumes the 128 B-row swizzle c ^ (r % 8)");
  static_assert(Tile::rowBytes == 128 && dstNbGrains == 8, "128 B tile rows");
  static_assert(dstNbGrains == 8 && warp_size == 32 && tokensPerPage == 16);
  constexpr uint32_t partBytes = exactDiv(sizeof(PaddedCacheHead), nbPartsPerHead);
  static_assert(partBytes == 128, "128 B parts: 4 blocks per row, 64 blocks per 16-token span");
  constexpr uint32_t blocksPerPart = exactDiv(partBytes, 2 * grainBytes);
  static_assert(blocksPerPart == 4);
  constexpr uint32_t headsPerSpan = mha::min(tokensPerPage, maxNbCopiedHeads);
  static_assert(headsPerSpan == 16 && maxNbCopiedHeads % headsPerSpan == 0);
  constexpr uint32_t nbSpans = exactDiv(maxNbCopiedHeads, headsPerSpan);
  static_assert(headsPerSpan * blocksPerPart == 2 * warp_size,
                "one row and two adjacent blocks per lane per span");
  static_assert(mha::max(4U, blocksPerPart) == 4, "4 B scale word per row (scaleLoadBytes)");
  constexpr uint32_t scaleRowBytes = 4;
  constexpr uint32_t blocksPerLane = exactDiv(blocksPerPart, exactDiv(warp_size, headsPerSpan));
  static_assert(blocksPerLane == 2);
  static_assert(headsPerSpan % 8 == 0, "row + 16 span keeps row % 8: lane-constant swizzle term");
  constexpr uint32_t spanTileBytes = headsPerSpan * Tile::rowBytes;       // 2048
  constexpr uint32_t spanScaleBytes = headsPerSpan * scaleRowBytes;       // 64
  constexpr uint32_t pageLoopUnroll = mixedPageLoopUnroll(nbSpans);
  static_assert(dstNbHeads >= maxNbCopiedHeads);
  using flashinfer::KVPageFormat;
  uint8_t constexpr a16Format = static_cast<uint8_t>(KVPageFormat::kA16);
  uint8_t constexpr fp8Format = static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8);
  uint8_t constexpr fp4Format = static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4);

  // Every lane has passed its own cp.async.wait_group; this makes all lanes' copies of the
  // group visible to all lanes (the compact path has the same barrier at both sites).
  __syncwarp();

  // Lane constants: token tok of every span (row % 8 = tok % 8 for all spans), block pair h.
  uint32_t const lane = laneId();
  uint32_t const tok = lane % headsPerSpan;
  uint32_t const h = lane / headsPerSpan;
  uint32_t const grain0 = blocksPerLane * 2 * h;  // logical grain of block 2h = 4h
  uint32_t const tileBase = smemAddr(&dst);
  uint32_t const addr0 = tileBase + Tile::template byteOffset<swizzle>(tok, grain0 + 0);
  uint32_t const addr1 = tileBase + Tile::template byteOffset<swizzle>(tok, grain0 + 1);
  uint32_t const addr2 = tileBase + Tile::template byteOffset<swizzle>(tok, grain0 + 2);
  uint32_t const addr3 = tileBase + Tile::template byteOffset<swizzle>(tok, grain0 + 3);
  uint32_t const scaleAddr = smemAddr(scales) + tok * scaleRowBytes + blocksPerLane * h;

  auto const expandSpan = [&](uint32_t span, auto formatTag) {
    constexpr uint8_t formatValue = decltype(formatTag)::value;
    static_assert(formatValue == fp8Format || formatValue == fp4Format);
    constexpr bool isFP8 = formatValue == fp8Format;
    constexpr float pow2k = isFP8 ? 0x1p120f : 0x1p126f;
    float const g = isFP8 ? fp8GlobalScale : fp4GlobalScale;
    float const gFold = g * pow2k;                 // exact for |g| < 2^8; inf beyond -> fallback
    bool const foldOk = fabsf(g) >= 0x1p-117f;     // every s * g fp32-normal
    uint32_t const tileOff = span * spanTileBytes;
    uint32_t const scaleOff = span * spanScaleBytes;
    uint32_t const s01 = ldsU16(scaleAddr + scaleOff);  // scales of blocks 2h (lo), 2h+1 (hi)
    float r0, r1;
    e4m3x2ScalesToFloat(s01, r0, r1);
    float const f0 = r0 * gFold;
    float const f1 = r1 * gFold;
    bool const fold = foldScalePairFinite(f0, f1, foldOk);

    auto const body = [&](auto foldTag) {
      constexpr bool kFold = decltype(foldTag)::value;
      uint32_t const sf2_0 = kFold ? bf16x2Broadcast(f0) : bf16x2Broadcast(r0 * g);
      uint32_t const sf2_1 = kFold ? bf16x2Broadcast(f1) : bf16x2Broadcast(r1 * g);
      // One block: grain a holds the packed block; outputs go to grains a, b (logical 2b, 2b+1).
      auto const block = [&](uint32_t a, uint32_t b, uint32_t sf2) {
        LdGrain out[2];
        if constexpr (isFP8) {
          LdGrain const packed = ldsGrain(a);
          expandE4M3Block16BF16Placed<kFold>(packed, sf2, out);
        } else {
          uint32_t lo, hi;
          ldsB64(a, lo, hi);
          expandE2M1Block16BF16Placed<kFold>(lo, hi, sf2, out);
        }
        stsGrain(a, out[0]);
        stsGrain(b, out[1]);
      };
      block(addr0 + tileOff, addr1 + tileOff, sf2_0);  // block 2h   (logical grains 4h, 4h+1)
      block(addr2 + tileOff, addr3 + tileOff, sf2_1);  // block 2h+1 (logical grains 4h+2, 4h+3)
    };
    if (fold) {
      body(MixedFoldTag<true>{});
    } else {
      body(MixedFoldTag<false>{});
    }
  };

#if MIXED_PAGE_STATIC_FORMAT == 0
  unused(formats);
  unused(expandSpan);
  unused(sourceHeadOffset);
  unused(a16Format);
#else
#pragma unroll(pageLoopUnroll)
  for (uint32_t span = 0; span < nbSpans; ++span) {
#if MIXED_PAGE_STATIC_FORMAT > 0
    unused(formats);
    unused(sourceHeadOffset);
    unused(a16Format);
    expandSpan(span, MixedFormatTag<MIXED_PAGE_STATIC_FORMAT>{});
#else
    uint32_t const localPage = (sourceHeadOffset + span * headsPerSpan) / tokensPerPage;
    assert(localPage < nbPages);
    uint8_t const format = selectByIndex(formats.values, localPage);
    if (format == fp8Format) {
      expandSpan(span, MixedFormatTag<fp8Format>{});
    } else if (format == fp4Format) {
      expandSpan(span, MixedFormatTag<fp4Format>{});
    } else {
      assert(format == a16Format);
    }
#endif
  }
#endif
  __syncwarp();
}

// ---- [44] item 3 (gated on artifact A0): copy with hoisted per-lane / per-page constants ----
//
// Same ownership, formats, cache policies and zero-fill rules as copyMixedPartialHeadsAsync
// (lane = block l % 4 of rows l / 4 + 8 * iteration: coalesced 16 B / 8 B lanes per token row),
// same page-outer dispatch; only where the address arithmetic is done changes.  Selected by
// mha.cu's MIXED_HOISTED_COPY at the two mixed copy sites of the sm90 SPEC_DEC compact build;
// geometry fixed by static_assert (128 B parts, 16-token spans, 4 B scale words, one warp, row 0
// origin, page-aligned source), any other instantiation fails to compile.
//
// Data flow, one call (K: one 128 B part of 64 rows = 4 spans; V: one 32-row half-tile = 2):
//   lane constants: b = l % 4, x = l / 4 (= row % 8 for every row the lane copies: rows advance
//   by 8 per iteration and 16 per span, origin row 0), elem = (idxPart * 4 + b) * 16,
//   dstFirst = tileBase + byteOffset<swz>(x, 2b), dstSecond = ... (x, 2b + 1)   (u32)
//   per span (page): pageValid; laneSrc = payload + [page * stride.page + headIdx * stride.head +
//   x * stride.token] (0 if !pageValid) + elemOff(format) (u64: 3 IMAD.WIDE), iterStride = 8 *
//   stride.token (u64)
//   per iteration i in {0, 1}: src = laneSrc + i * iterStride (one 64-bit add), valid = pageValid
//   && 16 span + x + 8 i < nbAvailHeads (ISETP + PLOP3), LDGSTS [dstFirst + 2048 span + 1024 i]
//   (immediate) with src-size valid ? n : 0 (SEL): A16 two cg 16 B (second grain from src + 16),
//   FP8 one ca 16 B (MIXED_FP8_COPY 1; 0 / 2 as in the stock helper), FP4 one ca 8 B.
//   scale loop: lane = row l + 32 i of the tile; page l / 16 + 2 i, token l % 16; for compressed
//   pages one ca 4 B LDGSTS of the row's scale group [scales + page/token/head strides + 4 idxPart]
//   to [scaleBase + 4 l + 128 i], src-size 4 if the row is valid else 0 (zero scale word ->
//   the expansion's f = 0: vote-neutral, 0 x payload 0); A16 / bad pages issue nothing (their
//   scale word is never read: A16 spans are skipped by the expansion) - no pointer select.
// Control flow: page loop (unrolled in static modules, rolled in the dynamic one) -> warp-uniform
//   format branch (dynamic) -> two unrolled iterations; then the scale loop (1-2 iterations).
//   No __syncwarp here: cp.async completion is the caller's wait_group + the expansion's barrier.
template <uint32_t size>
__device__ inline void cpAsyncCaShared(uint32_t dstAddr, void const* src, uint32_t srcSize) {
  static_assert(size == 4 || size == 8 || size == 16);
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(dstAddr), "l"(src),
               "n"(size), "r"(srcSize));
}
__device__ inline void cpAsyncCgShared16(uint32_t dstAddr, void const* src, uint32_t srcSize) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dstAddr), "l"(src),
               "r"(srcSize));
}

template <uint32_t maxNbCopiedHeads, uint32_t nbPartsPerHead, bool swizzle, bool isK,
          uint32_t dstNbHeads, uint32_t nbPages, typename _LdGrain>
__device__ inline void copyMixedPartialHeadsAsyncHoisted(
    Array2D<_LdGrain, dstNbHeads,
            exactDiv(exactDiv(sizeof(PaddedCacheHead), nbPartsPerHead), grainBytes)>& dst,
    uint8_t* dstScales, PageTransport const& transport,
    Vec<KVCachePageIndex, nbPages> const& pages, MixedPageFormats<nbPages> const& formats,
    uint32_t headIdx, uint32_t idxPart, uint32_t nbAvailHeads) {
  using Tile = Array2D<_LdGrain, dstNbHeads,
                       exactDiv(exactDiv(sizeof(PaddedCacheHead), nbPartsPerHead), grainBytes)>;
  // Dependent static_asserts only (see expandMixedPartialHeadsInPlaceBF16Placement).
  static_assert(sizeof(_LdGrain) == grainBytes && grainBytes == 16);
  static_assert(swizzle && Tile::rowBytes == 128 && Tile::cols == 8, "128 B swizzled tile rows");
  static_assert(Tile::cols == 8 && warp_size == 32 && tokensPerPage == 16);
  constexpr uint32_t partBytes = exactDiv(sizeof(PaddedCacheHead), nbPartsPerHead);
  static_assert(partBytes == 128, "128 B parts");
  constexpr uint32_t blocksPerPart = exactDiv(exactDiv(partBytes, grainBytes), 2);
  static_assert(blocksPerPart == 4);
  constexpr uint32_t headsPerSpan = mha::min(tokensPerPage, maxNbCopiedHeads);
  static_assert(headsPerSpan == 16 && maxNbCopiedHeads % headsPerSpan == 0);
  constexpr uint32_t nbSpans = exactDiv(maxNbCopiedHeads, headsPerSpan);
  static_assert(nbSpans <= nbPages, "one page index per span (page-aligned tile origin)");
  constexpr uint32_t blocksPerSpan = headsPerSpan * blocksPerPart;
  constexpr uint32_t iterationsPerSpan = exactDiv(blocksPerSpan, warp_size);  // 2
  constexpr uint32_t rowsPerIter = exactDiv(warp_size, blocksPerPart);         // 8
  static_assert(rowsPerIter == 8 && headsPerSpan % 8 == 0, "row % 8 is a lane constant");
  static_assert(partBytes == 128 &&
                    validElemsPerHead == exactDiv(sizeof(PaddedCacheHead), sizeof(CacheElem)),
                "every 16-element block of the head is valid: no per-block elem check");
  static_assert(dstNbHeads >= maxNbCopiedHeads);
  static_assert(maxNbCopiedHeads % warp_size == 0, "scale loop: no dump row needed");
  constexpr uint32_t scaleLoadBytes = mha::max(4U, blocksPerPart);
  static_assert(scaleLoadBytes == 4);
  constexpr uint32_t spanTileBytes = headsPerSpan * Tile::rowBytes;   // 2048
  constexpr uint32_t iterTileBytes = rowsPerIter * Tile::rowBytes;    // 1024
  constexpr uint32_t pageLoopUnroll = mixedPageLoopUnroll(nbSpans);
  using flashinfer::KVPageFormat;
  uint8_t constexpr a16Format = static_cast<uint8_t>(KVPageFormat::kA16);
  uint8_t constexpr fp8Format = static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8);
  uint8_t constexpr fp4Format = static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4);

  // Lane constants.
  uint32_t const lane = laneId();
  uint32_t const blockInPart = lane % blocksPerPart;
  uint32_t const headInSpan0 = lane / blocksPerPart;  // = row % 8 for every copied row
  uint32_t const elem = (idxPart * blocksPerPart + blockInPart) * 16;
  uint32_t const tileBase = smemAddr(&dst);
  uint32_t const dstFirst = tileBase + Tile::template byteOffset<swizzle>(headInSpan0, blockInPart * 2);
  uint32_t const dstSecond =
      tileBase + Tile::template byteOffset<swizzle>(headInSpan0, blockInPart * 2 + 1);

  auto const copySpan = [&](uint32_t span, KVCachePageIndex page, auto formatTag) {
    constexpr uint8_t format = decltype(formatTag)::value;
    constexpr bool isA16 = format == a16Format;
    constexpr bool isFP8 = format == fp8Format;
    constexpr bool isFP4 = format == fp4Format;
    static_assert(isA16 || isFP8 || isFP4);
    auto const& fmt = transport.formats[format];
    uint8_t const* payload;
    if constexpr (isK) {
      payload = static_cast<uint8_t const*>(fmt.k_payload);
    } else {
      payload = static_cast<uint8_t const*>(fmt.v_payload);
    }
    bool const pageValid = page != kBAD_PAGE_INDEX;
    uint32_t const spanHead0 = span * headsPerSpan;
    // Byte offset of this lane's block inside the token row, per format.
    uint32_t const elemOff = isA16 ? elem * uint32_t(sizeof(InputElem)) : (isFP8 ? elem : elem / 2);
    // Page + head + lane-row terms once per span; the two iterations add 8 * stride.token.
    uint64_t const laneOff =
        pageValid ? uint64_t(page) * fmt.payload_stride.page +
                        uint64_t(headIdx) * fmt.payload_stride.head +
                        uint64_t(headInSpan0) * fmt.payload_stride.token
                  : 0;
    uint8_t const* const laneSrc = payload + laneOff + elemOff;
    uint64_t const iterStride = uint64_t(rowsPerIter) * fmt.payload_stride.token;
    uint32_t const dstSpan = span * spanTileBytes;
    auto const iteration = [&](uint32_t i) {
      uint8_t const* const src = laneSrc + i * iterStride;
      uint32_t const localHead = spanHead0 + headInSpan0 + i * rowsPerIter;
      bool const valid = pageValid && localHead < nbAvailHeads;
      uint32_t const dstOff = dstSpan + i * iterTileBytes;
      if constexpr (isA16) {
        cpAsyncCgShared16(dstFirst + dstOff, src, valid ? grainBytes : 0U);
        cpAsyncCgShared16(dstSecond + dstOff, src + grainBytes, valid ? grainBytes : 0U);
      } else if constexpr (isFP4) {
        cpAsyncCaShared<8>(dstFirst + dstOff, src, valid ? 8U : 0U);
      } else {
#if MIXED_FP8_COPY == 0
        cpAsyncCaShared<8>(dstFirst + dstOff, src, valid ? 8U : 0U);
        cpAsyncCaShared<8>(dstFirst + dstOff + 8, src + 8, valid ? 8U : 0U);
#elif MIXED_FP8_COPY == 1
        cpAsyncCaShared<16>(dstFirst + dstOff, src, valid ? grainBytes : 0U);
#else
        cpAsyncCgShared16(dstFirst + dstOff, src, valid ? grainBytes : 0U);
#endif
      }
    };
    static_assert(iterationsPerSpan == 2);
    iteration(0);  // rows 16 span + x
    iteration(1);  // rows 16 span + x + 8
  };

#pragma unroll(pageLoopUnroll)
  for (uint32_t span = 0; span < nbSpans; ++span) {
    KVCachePageIndex const page = selectByIndex(pages, span);
#if MIXED_PAGE_STATIC_FORMAT >= 0
    unused(formats);
    copySpan(span, page, MixedFormatTag<MIXED_PAGE_STATIC_FORMAT>{});
#else
    uint8_t const format = selectByIndex(formats.values, span);
    if (format == a16Format) {
      copySpan(span, page, MixedFormatTag<a16Format>{});
    } else if (format == fp8Format) {
      copySpan(span, page, MixedFormatTag<fp8Format>{});
    } else {
      assert(format == fp4Format);
      copySpan(span, page, MixedFormatTag<fp4Format>{});
    }
#endif
  }

  // Scale words: one lane per row, rows lane + 32 i; the row's 4 B group of this part.
  uint32_t const scaleGroup = idxPart * blocksPerPart;
  uint32_t const scaleBase = smemAddr(dstScales) + lane * scaleLoadBytes;
  uint32_t const token = lane % tokensPerPage;
  constexpr uint32_t headIterations = exactDiv(maxNbCopiedHeads, warp_size);
#pragma unroll
  for (uint32_t i = 0; i < headIterations; ++i) {
    uint32_t const localHead = i * warp_size + lane;
    uint32_t const localPage = i * exactDiv(warp_size, tokensPerPage) + lane / tokensPerPage;
    static_assert(headIterations * exactDiv(warp_size, tokensPerPage) <= nbPages);
    KVCachePageIndex const page = selectByIndex(pages, localPage);
    bool const pageValid = page != kBAD_PAGE_INDEX;
    uint8_t const format =
#if MIXED_PAGE_STATIC_FORMAT >= 0
        pageValid ? MIXED_PAGE_STATIC_FORMAT : a16Format;
#else
        pageValid ? selectByIndex(formats.values, localPage) : a16Format;
#endif
    bool const compressed = format != a16Format;
    if (compressed) {
      auto const& sp = transport.formats[format];
      uint8_t const* scales;
      if constexpr (isK) {
        scales = sp.k_scales;
      } else {
        scales = sp.v_scales;
      }
      bool const valid = localHead < nbAvailHeads;
      uint64_t const scaleOffset = uint64_t(page) * sp.scale_stride.page +
                                   uint64_t(token) * sp.scale_stride.token +
                                   uint64_t(headIdx) * sp.scale_stride.head + scaleGroup;
      cpAsyncCaShared<scaleLoadBytes>(scaleBase + i * warp_size * scaleLoadBytes,
                                      scales + scaleOffset, valid ? scaleLoadBytes : 0U);
    }
  }
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
