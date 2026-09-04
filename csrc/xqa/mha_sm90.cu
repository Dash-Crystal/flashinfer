/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "cuda_hint.cuh"
#include "defines.h"
#if !(IS_MLA)
#include "barriers.cuh"
#include "utils.cuh"
#include "utils.h"

#if SPEC_DEC
#define Q_HEADS_PER_CTA 64
#include "specDec.h"
#endif

#ifndef GENERATE_CUBIN
#include <cuda_runtime.h>

#include "hostUtils.h"
#include "tensorMap.h"
#endif
#include "gmma.cuh"
#include "mha.h"
#include "mhaUtils.cuh"
#include "mha_stdheaders.cuh"
#include "tma.h"

#define DBG_PRINT 0

#ifdef SPEC_Q_SEQ_LEN
static_assert(SPEC_DEC, "SPEC_Q_SEQ_LEN is only supported for SPEC_DEC");
constexpr uint32_t specDecQLen = SPEC_Q_SEQ_LEN;
static_assert(specDecQLen * headGrpSize <= 32, "SPEC_Q_SEQ_LEN macro value is too large");
#define SWAP_AB 1
#else
#define SWAP_AB (!SPEC_DEC)
#endif

#define IS_SUPPORTED_F16_CASE \
  ((CACHE_ELEM_ENUM == 0 || CACHE_ELEM_ENUM == 5) && !SPEC_DEC && SWAP_AB && \
   !USE_INPUT_KV && !LOW_PREC_OUTPUT)

inline constexpr bool swapAB = SWAP_AB;

#pragma region Config

static_assert((inputElemSize == cacheElemSize && mha::is_same_v<InputElem, CacheElem>) ||
              inputElemSize > cacheElemSize);
using MathElem =
    mha::conditional_t<(inputElemSize > cacheElemSize && mha::is_same_v<CacheElem, int8_t>),
                       InputElem, CacheElem>;

constexpr uint32_t gmmaWarpsPerGrp = 4;
constexpr uint32_t gmmaWarpGrpSize = warp_size * gmmaWarpsPerGrp;
constexpr uint32_t gemm0NbGmmaGrps = 1;
constexpr uint32_t gemm0NbThrds = gmmaWarpGrpSize * gemm0NbGmmaGrps;
constexpr uint32_t gemm0NbWarps = gmmaWarpsPerGrp * gemm0NbGmmaGrps;
#if SPEC_DEC && !SWAP_AB
inline constexpr uint32_t ctaNbQHeads = Q_HEADS_PER_CTA;
inline constexpr uint32_t inputTokensPerCta = ctaNbQHeads / headGrpSize;
constexpr uint32_t ctaNbValidQHeads = ctaNbQHeads;
#elif SPEC_DEC && SWAP_AB
inline constexpr uint32_t inputTokensPerCta = specDecQLen;
inline constexpr uint32_t ctaNbValidQHeads = headGrpSize * inputTokensPerCta;
inline constexpr uint32_t ctaNbQHeads = []() {
  static_assert(ctaNbValidQHeads <= 32, "ctaNbValidQHeads cannot exceed 32");
  if constexpr (ctaNbValidQHeads <= 8) {
    return 8;
  }
  if constexpr (ctaNbValidQHeads <= 16) {
    return 16;
  }
  return 32;
}();
#else
inline constexpr uint32_t ctaNbValidQHeads = headGrpSize * beamWidth;
inline constexpr uint32_t ctaNbQHeads = roundUp(ctaNbValidQHeads, swapAB ? 8U : 64U);
inline constexpr uint32_t inputTokensPerCta = 1;
#endif
constexpr uint32_t gemm0WarpGrpTileNbTokens = 64;
inline constexpr uint32_t gemm0CtaTileNbTokens = gemm0WarpGrpTileNbTokens * gemm0NbGmmaGrps;
constexpr uint32_t gemm1NbGmmaGrps = 1;
constexpr uint32_t gemm1NbThrds = gmmaWarpGrpSize * gemm1NbGmmaGrps;
constexpr uint32_t gemm1NbWarps = gmmaWarpsPerGrp * gemm1NbGmmaGrps;
constexpr uint32_t gemm1CtaTileNbTokens = gemm0CtaTileNbTokens;
constexpr uint32_t mathHeadBytes = sizeof(Vec<MathElem, headElems>);
constexpr uint32_t nbIOWarps = 4;
constexpr uint32_t nbIOThrds = warp_size * nbIOWarps;
// Stage depths. Deeper K (3) measured identical to 2 on H200; deeper V
// pushes the CTA past two per SM, which halves throughput at grid 136.
// Per-role clock64 stamps for the first tiles of CTA 0, printed at kernel end.
#ifndef MIXED_KV_TRACE
#define MIXED_KV_TRACE 0
#endif
// Stage depths in whole 64-token tiles; two of each keeps two CTAs per SM.
#ifndef MIXED_KV_KDEPTH
#define MIXED_KV_KDEPTH 3
#endif
#ifndef MIXED_KV_VDEPTH
#define MIXED_KV_VDEPTH 3
#endif
// P ring depth (mixed build).  Shared memory must stay under ~112 KB for two
// CTAs per SM: three K/V stages (needed to cover ~2 us of copy landing under a
// saturated memory system) leave room for two P entries.
#ifndef MIXED_KV_XDEPTH
#define MIXED_KV_XDEPTH 2
#endif
constexpr uint32_t mixedLoadWarpsPerOperand = 1;
constexpr uint32_t convertWarpsPerOperand = gmmaWarpsPerGrp;
#if ENABLE_MIXED_KV_CACHE
// Page-format tag recorded in shared memory for tile slots past the end of the
// sequence: the loader issues nothing for them and the converters zero-fill.
constexpr uint8_t kMixedBadPageFormat = 0xFF;
// Temporary attribution experiments (0 = production).
// bit 1: converters skip all work.  bit 2: loader issues no compressed-page
// TMA.  bit 4: loader issues no scale copies.  Bits compose.
#ifndef MIXED_KV_EXPERIMENT
#define MIXED_KV_EXPERIMENT 0
#endif
#if MIXED_KV_TRACE
// Per-SM count of resident CTAs of this kernel (trace builds only): +1 at CTA
// start, -1 at CTA end; CTA 0 samples its SM's count mid-run and prints it.
__device__ uint32_t mixedKvTraceSmResident[1024];
__device__ inline uint32_t mixedKvTraceSmId() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}
#endif
#endif
constexpr uint32_t nbConvertWarps =
    ENABLE_MIXED_KV_CACHE ? 2 * convertWarpsPerOperand : 0;
constexpr uint32_t ctaWarpGroups = ENABLE_MIXED_KV_CACHE ? 5 : 3;
constexpr uint32_t multiBlockMinNbTilesPerCta = 1;  // 3; // @fixme: need tuning
constexpr uint32_t multiBlockMinNbTiles = multiBlockMinNbTilesPerCta * 2;
constexpr uint32_t nbWarps = gemm0NbWarps + gemm1NbWarps + nbIOWarps + nbConvertWarps;

// Hardware named barriers for the intra-warp-group syncs of the two GEMM warp
// groups.  bar.sync id,128 replaces the mbarrier arrive + try_wait round trip
// of the per-tile colMax / rescale exchanges; ids 1/2 are unused (the converter
// groups sync with __syncwarp) and 0 is __syncthreads.  bar.sync orders shared
// memory like membar.cta, so the exchanges it protects need no extra fence.
constexpr uint32_t gemm0NamedBarId = 3;
constexpr uint32_t gemm1NamedBarId = 4;
__device__ inline void gemm0WarpGrpSync() { namedBarSync(gemm0NamedBarId, gemm0NbThrds); }
__device__ inline void gemm1WarpGrpSync() { namedBarSync(gemm1NamedBarId, gemm1NbThrds); }

constexpr uint32_t cacheHeadPartBytes = mha::min(paddedCacheHeadBytes, 128U);
constexpr uint32_t cacheHeadNbParts =
    exactDiv(paddedCacheHeadBytes, cacheHeadPartBytes);  // @fixme: support divUp in the future
constexpr uint32_t cacheHeadPartElems = exactDiv(headElems, cacheHeadNbParts);
constexpr uint32_t swizzleBytes = cacheHeadPartBytes;
static_assert(swizzleBytes == 128 || swizzleBytes == 64 || swizzleBytes == 32);

constexpr bool needInputCvt =
    inputElemSize > cacheElemSize&& mha::is_same_v<CacheElem, __nv_fp8_e4m3>;
constexpr bool needCacheCvt = inputElemSize > cacheElemSize&& mha::is_same_v<CacheElem, int8_t>;
static_assert(needInputCvt || needCacheCvt || mha::is_same_v<InputElem, CacheElem>);

using ShmQWiseVec = Vec<float, ctaNbQHeads>;

constexpr uint32_t qPartBytes = mha::min(mathHeadBytes, 128U);
constexpr uint32_t nbQParts = exactDiv(mathHeadBytes, qPartBytes);
constexpr uint32_t grainsPerQPart = exactDiv(qPartBytes, grainBytes);

constexpr uint32_t xPartBytes = mha::min(cacheElemSize * gemm0CtaTileNbTokens, 128U);
constexpr uint32_t nbXParts = exactDiv(cacheElemSize * gemm0CtaTileNbTokens, xPartBytes);
constexpr uint32_t grainsPerXPart = exactDiv(xPartBytes, grainBytes);
constexpr uint32_t cacheElemsPerGrain = exactDiv(grainBytes, cacheElemSize);

constexpr uint32_t grainsPerIOHead = exactDiv(ioHeadBytes, grainBytes);
constexpr uint32_t grainsPerPaddedInputHead = exactDiv(paddedInputHeadBytes, grainBytes);

#if USE_BEAM_SEARCH
constexpr uint32_t beamSearchGemm0CtaTileNbTokens = exactDiv(gemm0CtaTileNbTokens, beamWidth);
#endif

using PaddedOutHead = PaddedInputHead;

#pragma endregion Config

struct alignas(128) SharedMem {
  using KBuffer = Array2D<LdGrain, gemm0CtaTileNbTokens, exactDiv(cacheHeadPartBytes, grainBytes)>;
  // Mixed build: a K stage is a whole tile (every head part), released once
  // per tile.  nbKBuf counts stages; k[] holds nbKBuf * cacheHeadNbParts part
  // buffers, stage s part p at k[s * cacheHeadNbParts + p].
  static constexpr uint32_t nbKBuf = (ENABLE_MIXED_KV_CACHE && !SPEC_DEC) ? MIXED_KV_KDEPTH : 2;
  static constexpr uint32_t nbKPartBufs = ENABLE_MIXED_KV_CACHE ? nbKBuf * cacheHeadNbParts : nbKBuf;
  alignas(1024) KBuffer k[nbKPartBufs];  // as is loaded from global mem.
  using XBuffer = Vec<Array2D<LdGrain, ctaNbQHeads, grainsPerXPart>, nbXParts>;
  // The mixed build gives P a four-deep ring so gemm0 never waits on gemm1's
  // release of a tile two back; with two entries the two consumer groups run
  // in lock step and the cadence becomes their sum.
  // Mixed build: three entries (two make the consumer groups run in lock step;
  // the fourth was traded for a third K/V stage, which the copy-landing latency
  // under a saturated memory system needs).
  static constexpr uint32_t nbXBuf =
      ENABLE_MIXED_KV_CACHE
          ? MIXED_KV_XDEPTH
          : 2 * (gemm0CtaTileNbTokens >= gemm1CtaTileNbTokens
                     ? 1
                     : exactDiv(gemm1CtaTileNbTokens, gemm0CtaTileNbTokens));
  static constexpr bool vBufAlignedForSwizzle =
      ENABLE_MIXED_KV_CACHE || sizeof(XBuffer) % (cacheHeadPartBytes * 8) == 0;
  using VBuffer =
      Vec<Array2D<LdGrain, gemm1CtaTileNbTokens, exactDiv(cacheHeadPartBytes, grainBytes),
                  vBufAlignedForSwizzle>,
          cacheHeadNbParts>;
#if !SWAP_AB
  using VTBuffer =
      Array2D<LdGrain, headElems, exactDiv(gemm1CtaTileNbTokens, cacheElemsPerGrain), true>;
#endif
  static constexpr uint32_t nbVBuf = (ENABLE_MIXED_KV_CACHE && !SPEC_DEC) ? MIXED_KV_VDEPTH : 2;
#if CACHE_ELEM_ENUM == 0 || CACHE_ELEM_ENUM == 5
  using OutSwizzleBuf = Array2D<LdGrain, ctaNbQHeads, grainsPerPaddedInputHead>;
#elif CACHE_ELEM_ENUM == 2
  using OutSwizzleBuf = Array2D<Vec<Vec<InputElem, 4>, 4>, ctaNbQHeads, exactDiv(headElems, 4 * 4)>;
#endif
#if ENABLE_MIXED_KV_CACHE
  union ReusedXVOutSwizzleBuf {
    struct XV {
      XBuffer x;
    } xv;

    OutSwizzleBuf outSwizzle;
  } reusedXVOutSwizzleBuf[nbXBuf];
  alignas(1024) VBuffer vBufs[nbVBuf];
#if !SWAP_AB
  alignas(1024) VTBuffer vtBufs[nbVBuf];
#endif

  __device__ inline XBuffer& xBuf(uint32_t i) { return reusedXVOutSwizzleBuf[i].xv.x; }

  __device__ inline VBuffer& vBuf(uint32_t i) { return vBufs[i]; }
#if !SWAP_AB
  __device__ inline VTBuffer& vtBuf(uint32_t i) { return vtBufs[i]; }
#endif
#else
  static_assert(nbXBuf == nbVBuf);

  union ReusedXVOutSwizzleBuf {
    struct XV {
      XBuffer x;
      VBuffer v;
#if !SWAP_AB
      VTBuffer vt;
#endif
      // @fixme: also put xColMax and xColSum here
    } xv;

    OutSwizzleBuf outSwizzle;
  } reusedXVOutSwizzleBuf[nbXBuf];

  static_assert(sizeof(OutSwizzleBuf) <= sizeof(SharedMem::ReusedXVOutSwizzleBuf::XV),
                "need to use split output to avoid excessive shared memory usage");

  __device__ inline XBuffer& xBuf(uint32_t i) { return reusedXVOutSwizzleBuf[i].xv.x; }

  __device__ inline VBuffer& vBuf(uint32_t i) { return reusedXVOutSwizzleBuf[i].xv.v; }
#if !SWAP_AB
  __device__ inline VTBuffer& vtBuf(uint32_t i) { return reusedXVOutSwizzleBuf[i].xv.vt; }
#endif
#endif
  __device__ inline OutSwizzleBuf& outSwizzleBuf(uint32_t i) {
    return reusedXVOutSwizzleBuf[i].outSwizzle;
  }

  using QBuffer = Vec<Array2D<LdGrain, ctaNbQHeads, grainsPerQPart>, nbQParts>;
  QBuffer q;  // For gmma math. Conversion done if needed.

  // @fixme: move these into reusedXVOutSwizzleBuf
#if SWAP_AB
  ShmQWiseVec xColMax[nbXBuf];
  ShmQWiseVec xColSum[nbXBuf][gemm0NbWarps];
#else
  ShmQWiseVec xRowMax[nbXBuf];
  ShmQWiseVec xRowSum[nbXBuf];
#endif

  ShmQWiseVec gemm0CurrentSeqMax;  // !SWAP_AB running row max (SWAP_AB keeps it in registers)
  // col sum and max for the current gemm1 acc. Use shared memory to save some registers. register
  // storage will be 8x duplicate for swapAB and 4x duplicate for non-swapAB.
  ShmQWiseVec gemm1AccColMax;
  ShmQWiseVec gemm1AccColSum;
#if SWAP_AB
  // gemm0 colMax exchange: each warp publishes its tile-local column max into
  // its own slot of the tile's parity; after one bar.sync every warp reads the
  // four slots and folds them into a register-resident running max.  Slot
  // parity p is rewritten at tile t+2, after the sync of tile t+1 that every
  // warp passes only once it has read slot p at tile t: no WAR hazard, so no
  // second sync and no shared-memory atomics.
  ShmQWiseVec gemm0WarpColMax[2][gemm0NbWarps];
#endif

  static constexpr uint32_t nbPagesPerTile =
      gemm0CtaTileNbTokens >= tokensPerPage ? exactDiv(gemm0CtaTileNbTokens, tokensPerPage) : 1;
  Vec<KVCachePageIndex, nbPagesPerTile> pages[2];  // one for K and one for V
#if ENABLE_MIXED_KV_CACHE
  MixedPageFormats<nbPagesPerTile> pageFormats[2];
  MixedPageFormats<nbPagesPerTile> kFormats[nbKBuf];
  MixedPageFormats<nbPagesPerTile> vFormats[nbVBuf];
  // Compressed pages are delivered by TMA into dense per-page slots (one
  // 16-token slot per page, sized for the E4M3 part row; E2M1 uses the first
  // half of each row).  Converter warps expand them into k[]/vBuf(); A16 pages
  // are delivered by TMA straight into k[]/vBuf() and never touch these slots.
  // A compressed page is fetched as whole-head rows once per tile (64 B E2M1 /
  // 128 B E4M3 per token for D=128): one TMA box per page, full DRAM bursts.
  static constexpr uint32_t packedRowBytesFP8 = headElems;
  static constexpr uint32_t packedRowBytesFP4 = headElems / 2;
  static constexpr uint32_t packedSlotBytes = tokensPerPage * packedRowBytesFP8;
  using PackedTile = uint8_t[nbPagesPerTile][packedSlotBytes];
  // Compressed rows are delivered by TMA *into the stage's last head-part
  // buffer* (a whole-head packed tile is exactly one part buffer: 64 rows x
  // 128 B) with the 128B (E4M3) / 64B (E2M1) swizzle, and expanded in place:
  // the converter group reads every packed word it needs, barriers, then
  // writes the A16 rows of every part.  No separate packed buffers.
  static_assert(sizeof(PackedTile) == sizeof(KBuffer));
  // Page indices and format tags for chunks of this CTA's tiles, filled by the
  // load warp once per chunk so no per-tile dependent global load sits on its
  // critical path.  Two chunks per operand because scales are issued two
  // tiles ahead and may cross into the next chunk while this one is live.
  static constexpr uint32_t metaChunkTiles = 16;
  static constexpr uint32_t nbMetaChunks = 2;
  KVCachePageIndex metaPages[2][nbMetaChunks][metaChunkTiles][nbPagesPerTile];
  uint8_t metaFormats[2][nbMetaChunks][metaChunkTiles][nbPagesPerTile];
  // One E4M3 block scale per 16 values; the whole head for every tile token.
  // Scales for tile t+2 are fetched during tile t.  With V three stages deep
  // the loader has only waited for tile t-3's release, so tiles t-2..t+2 may
  // all be live: five ring entries.
  static constexpr uint32_t scaleBytesPerToken = exactDiv(headElems, 16);
  using TileScales = uint8_t[gemm0CtaTileNbTokens][scaleBytesPerToken];
  // Tiles t..t+2 have live scales with copies issued two tiles ahead; four entries.
  static constexpr uint32_t nbScaleTiles = 4;
  alignas(16) TileScales kScales[nbScaleTiles];
  alignas(16) TileScales vScales[nbScaleTiles];
#if MIXED_KV_TRACE
  static constexpr uint32_t nbTraceTiles = 8;
  long long trace[nbTraceTiles][16];
  // SM residency probe: this CTA's %smid and the number of CTAs of this kernel
  // resident on that SM when tile 4's slot 0 stamp is taken (includes self).
  uint32_t traceSmId;
  uint32_t traceResidentMid;
#endif
#endif

  // mem barriers

  CtaBarrierPair qBar;
  CtaBarrierPair kBar[nbKBuf];
  CtaBarrierPair vBar[nbVBuf];
#if ENABLE_MIXED_KV_CACHE
  CtaBarrier kLoadReady[nbKBuf];
  CtaBarrier vLoadReady[nbVBuf];
  CtaBarrier kPagesReady;
  CtaBarrier vPagesReady;
#endif
#if !SWAP_AB
  CtaBarrierPair vtBar[nbVBuf];
#endif
  CtaBarrierPair xBar[nbXBuf];

  // used internally in the gemm0 warp group
  // @fixme: use separate arrive and wait for all usage
  CtaBarrier gemm0WarpGrpBar;

  // used internally in the gemm1 warp group
  // @fixme: use separate arrive and wait for all usage
  CtaBarrier gemm1WarpGrpBar;

  bool isLastCta;
};

CUBIN_EXPORT __device__ constexpr uint32_t smemSize = sizeof(SharedMem);
#ifdef __CUDA_ARCH__
static_assert(smemSize < kMAX_SMEM_SIZE);
#endif

constexpr uint32_t nbQLdWarps = needInputCvt ? nbIOWarps - 2 : 1;
constexpr uint32_t nbQLdThrds = warp_size * nbQLdWarps;

#if CACHE_ELEM_ENUM == 0 || CACHE_ELEM_ENUM == 2 || CACHE_ELEM_ENUM == 5
template <uint32_t nbThrds = 64, uint32_t beamWidth = 1>
struct F16QToF8Converter {
  static_assert(inputElemSize == 2);
  using F16Vec = Vec<InputElem, exactDiv(grainBytes, inputElemSize)>;
#if CACHE_ELEM_ENUM == 0 || CACHE_ELEM_ENUM == 5
  using ShmVec = F16Vec;
#elif CACHE_ELEM_ENUM == 2
  using F8Vec = Vec<CacheElem, exactDiv(grainBytes, inputElemSize)>;
  using ShmVec = F8Vec;
#endif

  static constexpr uint32_t grainsPerPaddedInputHead = exactDiv(paddedInputHeadBytes, grainBytes);
  static constexpr uint32_t grainsPerPaddedInputQHeadGrp = grainsPerPaddedInputHead * headGrpSize;
#if !(SPEC_DEC)
  static constexpr uint32_t totalGrains = grainsPerPaddedInputQHeadGrp * beamWidth;
#else
  static_assert(beamWidth == 1);
  static constexpr uint32_t totalGrains = grainsPerPaddedInputQHeadGrp * inputTokensPerCta;
#endif
  static constexpr uint32_t nbIters = divUp(totalGrains, nbThrds);

  using RegData = Vec<F16Vec, nbIters>;

  static __device__ RegData load(uint32_t tid, TinyPtr<IOHead const> const& src,
                                 uint32_t const nbKHeads /*for beam search and spec dec*/,
                                 uint32_t nbTokens);
  static __device__ void store(uint32_t tid, SharedMem::QBuffer& dst, RegData const& data);
};
#endif  // CACHE_ELEM_ENUM

#if ENABLE_MIXED_KV_CACHE
template <typename PartBuf>
__device__ __forceinline__ void expandPackedStage(
    PartBuf* parts, SharedMem::TileScales const& scales,
    MixedPageFormats<SharedMem::nbPagesPerTile> const& formats, float fp8GlobalScale,
    float fp4GlobalScale, uint32_t idxWarp, uint32_t namedBarrierId);
#endif

struct KVTilePartLoader {
  static constexpr uint32_t nbParts = cacheHeadNbParts;
  static constexpr uint32_t partElems = exactDiv(headElems, nbParts);

  static_assert(gemm0CtaTileNbTokens % tokensPerPage == 0 ||
                tokensPerPage % gemm0CtaTileNbTokens == 0);
  static constexpr uint32_t nbPagesPerTile = SharedMem::nbPagesPerTile;

  uint32_t const nbKHeads;
  bool const isK;
  KVCacheList<usePagedKVCache> const& cacheList;
  uint32_t const idxReq;
  uint32_t const idxHeadGrp;

  CUtensorMap const& tensorMap;
#if ENABLE_MIXED_KV_CACHE
  MixedPageFormats<nbPagesPerTile>& formats;
#endif
  uint32_t const nbPages;  // for bound check
  Vec<KVCachePageIndex, nbPagesPerTile>& pages;
  uint32_t idxTileRef;  // idxTile used to load the pages
  uint32_t const baseOffset;

  __device__ KVTilePartLoader(bool isK, uint32_t nbKHeads,
                              KVCacheList<usePagedKVCache> const& cacheList, uint32_t idxReq,
                              uint32_t idxHeadGrp, CUtensorMap const& tensorMap, uint32_t nbPages,
                              Vec<KVCachePageIndex, nbPagesPerTile>& pageBuf
#if ENABLE_MIXED_KV_CACHE
                              , MixedPageFormats<nbPagesPerTile>& formatBuf
#endif
                              );
  // tensorMap is for one whole page ([nbKHeads*tokensPerPage][headElems]) or whole cache
  template <uint32_t nbTokens, bool alignedForSwizzle>
  __device__ void loadData(
      Array2D<LdGrain, nbTokens, exactDiv(cacheHeadPartBytes, grainBytes), alignedForSwizzle>& dst,
      uint32_t idxTile, uint32_t idxPart, CtaBarrier& bar);

#if ENABLE_MIXED_KV_CACHE
  // Transaction bytes one head part of this tile puts on the barrier: the A16
  // part rows, plus (for idxPart == 0 only) the whole-head compressed rows.
  __device__ uint32_t packedPartTxBytes(uint32_t idxPart) const;
  // Elected-lane TMA issue for one head part: A16 pages through the A16 tensor
  // map into dstA16; on idxPart == 0, compressed pages as whole-head rows
  // through the byte-typed maps into dstPacked.  Slots past the sequence end
  // issue nothing.
  template <bool alignedForSwizzle>
  __device__ void issuePackedPartLoad(
      Array2D<LdGrain, gemm0CtaTileNbTokens, exactDiv(cacheHeadPartBytes, grainBytes),
              alignedForSwizzle>& dstA16,
      SharedMem::PackedTile& dstPacked, uint32_t idxPart, CUtensorMap const& tensorMapFP8,
      CUtensorMap const& tensorMapFP4, CtaBarrier& bar) const;
  // Whole-head block scales for every compressed token of the tile; every lane
  // of the nbLoadWarps load warps owns one token.
  template <uint32_t nbLoadWarps>
  __device__ void loadPackedScales(SharedMem::TileScales& dstScales, uint32_t idxLoadWarp) const;
  // Page indices and format tags of one tile, as read from the shared-memory
  // metadata chunk.
  struct PagePrefetch {
    Vec<KVCachePageIndex, nbPagesPerTile> pages;
    Vec<uint8_t, nbPagesPerTile> formats;
  };
  // Fill this operand's metadata chunk for CTA tile iterations [idxIterBeg,
  // idxIterBeg + metaChunkTiles).  Warp-wide; one dependent global load pair
  // per chunk.
  __device__ void fillTileMeta(SharedMem& smem, uint32_t operand, uint32_t idxIterBeg,
                               uint32_t nbIters, uint32_t idxTileInit, uint32_t nbSubSeq) const;
  __device__ PagePrefetch readTileMeta(SharedMem const& smem, uint32_t operand,
                                       uint32_t idxIter) const;
  // Publish a tile's metadata to the loader's smem pages/formats (elected lane).
  __device__ void publishPages(PagePrefetch const& prefetched, uint32_t idxTile);
  // Block scales for a tile described by prefetched metadata.
  template <uint32_t nbLoadWarps>
  __device__ void loadPackedScalesFrom(PagePrefetch const& prefetched,
                                       SharedMem::TileScales& dstScales,
                                       uint32_t idxLoadWarp) const;
#endif

  __device__ void loadPages(uint32_t idxTile, bool publish = true);
  __device__ GMemKVCacheHead& getHead(uint32_t pos);
};

#if ENABLE_MIXED_KV_CACHE
__device__ __forceinline__ KVTilePartLoader::PagePrefetch readMixedTileMeta(
    SharedMem const& smem, uint32_t operand, uint32_t idxIter);
__device__ __forceinline__ void issueCompressedPageCopies(
    KVCacheList<usePagedKVCache> const& cacheList, bool isK, uint32_t idxHeadGrp,
    SharedMem const& smem, uint32_t operand, uint32_t idxIter, SharedMem::PackedTile& dstPacked,
    SharedMem::TileScales& dstScales, uint32_t idxWarp);
#endif

using GmmaAccCoreMat = Array2D<float, 2, 2>;
template <uint32_t nbRows, uint32_t nbCols>
using GmmaAcc =
    Array2D<GmmaAccCoreMat, exactDiv(nbRows, gmma::instM), exactDiv(nbCols, gmma::instNBase)>;

inline constexpr uint32_t gemm0M = (swapAB ? gemm0CtaTileNbTokens : ctaNbQHeads);
inline constexpr uint32_t gemm0N = (swapAB ? ctaNbQHeads : gemm0CtaTileNbTokens);

using Gemm0Acc = GmmaAcc<gemm0M, gemm0N>;

#if SWAP_AB
using RegColWiseVec = Vec<Vec<float, GmmaAccCoreMat::cols>, Gemm0Acc::cols>;
using UniformNeedRescaleMask = Vec<uint32_t, divUp(ctaNbQHeads, warp_size)>;
using RegSeqWiseVec = RegColWiseVec;
#else
using RegRowWiseVec = Vec<Vec<float, GmmaAccCoreMat::rows>, Gemm0Acc::rows>;
using UniformNeedRescaleMask =
    Vec<uint32_t, divUp(exactDiv(ShmQWiseVec::size, gmma::instM) * (gmma::instM / 4), warp_size)>;
using RegSeqWiseVec = RegRowWiseVec;
#endif

#if SPEC_DEC

__device__ inline uint32_t getInputSeqLen(SpecDecParams const& params, uint32_t idxReq) {
  return (params.qCuSeqLens == nullptr) ? params.qSeqLen
                                        : params.qCuSeqLens[idxReq + 1] - params.qCuSeqLens[idxReq];
}

__device__ inline uint32_t getInputTokOffset(SpecDecParams const& params, uint32_t idxReq) {
  return (params.qCuSeqLens == nullptr) ? params.qSeqLen * idxReq : params.qCuSeqLens[idxReq];
}

struct SpecDec {
  static inline constexpr uint32_t tileSize = gemm0CtaTileNbTokens;
  static inline constexpr uint32_t ctaMaxQSeqLen = (ctaNbQHeads / headGrpSize);
  using TileMaskRow = Vec<uint32_t, exactDiv(tileSize, 32)>;

  __device__ inline SpecDec(SpecDecParams const& params, uint32_t idxReq, uint32_t idxInputSubSeq,
                            uint32_t seqLen)
      : params(params), idxInputSubSeq(idxInputSubSeq), seqLen(seqLen) {
    inputSeqLen = getInputSeqLen(params, idxReq);
    baseOffset = divUp(params.qSeqLen, 32U) *
                 (getInputTokOffset(params, idxReq) + ctaMaxQSeqLen * idxInputSubSeq);
  }

  __device__ inline uint32_t unmaskedSeqLen() const { return seqLen - inputSeqLen; }

  __device__ inline bool needMask(uint32_t idxTile, uint32_t idxQTokInCta) const {
    return tileSize * (idxTile + 1) > unmaskedSeqLen() &&
           ctaMaxQSeqLen * idxInputSubSeq + idxQTokInCta < inputSeqLen && params.mask != nullptr;
  }

  __device__ inline int32_t maskColBeg(uint32_t idxTile) const {
    int32_t const convergedSeqLen = int32_t(unmaskedSeqLen());
    return static_cast<int32_t>(exactDiv(tileSize, 32) * idxTile) -
           static_cast<int32_t>(divUp(convergedSeqLen, 32));
  }

  __device__ inline TileMaskRow loadTileMaskRow(uint32_t idxTile, uint32_t idxQTokInCta) const {
    assert(needMask(idxTile, idxQTokInCta));
    constexpr uint32_t nbOrigElems = TileMaskRow::size + 1;
    Vec<uint32_t, nbOrigElems> orig;

    int32_t const cols = divUp<int32_t>(params.qSeqLen, 32);
    uint32_t const rowOffset = baseOffset + idxQTokInCta * cols;
    int32_t const colBeg = maskColBeg(idxTile);
#pragma unroll
    for (int32_t i = 0; i < int32_t(nbOrigElems); i++) {
      int32_t const idx = colBeg + i;
      orig[i] = inRange(idx, 0, cols) ? params.mask[rowOffset + idx] : (idx < 0 ? ~0U : 0U);
    }
    TileMaskRow mask;
    uint32_t const shift = (32 - unmaskedSeqLen() % 32) % 32;
#pragma unroll
    for (uint32_t i = 0; i < TileMaskRow::size; i++) {
      asm("shf.r.clamp.b32 %0, %1, %2, %3;\n"
          : "=r"(mask[i])
          : "r"(orig[i]), "r"(orig[i + 1]), "r"(shift));
    }
    return mask;
  }

  SpecDecParams const& params;
  uint32_t const idxInputSubSeq;
  uint32_t const seqLen;
  uint32_t inputSeqLen;
  uint32_t baseOffset;
};

__device__ void warpGrpApplyMask(Gemm0Acc& acc, SpecDec const& specDec,
#if SLIDING_WINDOW && !IS_SPEC_DEC_TREE
                                 int32_t tok0WinBeg,
#endif
                                 uint32_t cacheSeqLen, uint32_t idxTile, uint32_t warpRank);
#endif

#if SWAP_AB
__device__ RegColWiseVec computeWarpGrpColMax_sync(uint32_t warpRank,
                                                   ShmQWiseVec (&warpColMaxSlots)[gemm0NbWarps],
                                                   RegColWiseVec& runningColMax,
                                                   Gemm0Acc const& src);
__device__ void warpGrpApplyMask(uint32_t warpRank, Gemm0Acc& acc, uint32_t validRowBeg,
                                 uint32_t validRowEnd);
__device__ void warpGrpOnlineSoftmax(Gemm0Acc& acc, RegColWiseVec const& colMax);
__device__ RegColWiseVec computeWarpColSum(Gemm0Acc& src);
__device__ void storeGemm0AccToShm(uint32_t warpRank, uint32_t lane, SharedMem::XBuffer& smemX,
                                   CtaBarrier& barConsumed, Gemm0Acc const& acc);
__device__ RegColWiseVec loadShmColWiseVecWithDup(ShmQWiseVec const& smemVec);
__device__ RegColWiseVec loadGmemColWiseVecWithDup(ShmQWiseVec const& gmemVec, uint32_t bound);
#else
__device__ RegRowWiseVec computeWarpGrpRowMax_sync(uint32_t warpRank, ShmQWiseVec& smemColMax,
                                                   Gemm0Acc const& src);
__device__ void warpGrpApplyMask(Gemm0Acc& acc, uint32_t validColBeg, uint32_t validColEnd);
__device__ void warpGrpOnlineSoftmax(Gemm0Acc& acc, RegRowWiseVec const& colMax);
__device__ RegRowWiseVec computeWarpRowSum(Gemm0Acc& src);
__device__ void storeGemm0AccToShm(uint32_t warpRank, uint32_t lane, SharedMem::XBuffer& smemX,
                                   CtaBarrier& barConsumed, Gemm0Acc const& acc);
__device__ RegRowWiseVec loadShmRowWiseVecWithDup(uint32_t warpRank, ShmQWiseVec const& smemVec);
__device__ void storeShmRowWiseVec(uint32_t warpRank, ShmQWiseVec& smemVec,
                                   RegRowWiseVec const& regVec);
#endif

using RegMatAFrag = Array2D<Array2D<uint32_t, 2, 1>, 1, 2>;
constexpr uint32_t gemm1NbGmmaInstK = exactDiv(gemm1CtaTileNbTokens, gmma::instK<MathElem>);

#if ENABLE_MIXED_KV_CACHE
template <flashinfer::KVPageFormat format>
__device__ inline RegMatAFrag loadMixedKTileFragment(
    SharedMem::KBuffer const& tile, uint32_t pageInTile,
    uint32_t blockInPart, uint32_t idxPart, uint8_t const* scales,
    float fp8GlobalScale, float fp4GlobalScale) {
  using flashinfer::KVPageFormat;
  static_assert(tokensPerPage == 16);
  static_assert(format == KVPageFormat::kA16 ||
                format == KVPageFormat::kBlockScaledFP8 ||
                format == KVPageFormat::kBlockScaledFP4);
  constexpr uint32_t blocksPerPart = exactDiv(cacheHeadPartBytes, 2 * grainBytes);
  constexpr uint32_t scaleLoadBytes = mha::max(4U, blocksPerPart);
  uint32_t const laneInQuad = laneId() & 3U;
  uint32_t const tokenInHalf = laneId() >> 2;
  uint32_t const elemInBlock = laneInQuad * 2;
  uint32_t const scaleBlock = idxPart * blocksPerPart + blockInPart;
  uint32_t const scaleGroup = scaleBlock & ~(scaleLoadBytes - 1);
  uint32_t const scaleOffset = scaleBlock - scaleGroup;
  uint32_t const pageToken = pageInTile * tokensPerPage;

  RegMatAFrag fragment;
  if constexpr (format == KVPageFormat::kA16) {
    uint32_t const matrix = laneId() / 8;
    uint32_t const sourceK = matrix % 2;
    uint32_t const sourceM = matrix / 2;
    auto const* source = &tile.template at<true>(
        pageToken + 8 * sourceM + laneId() % 8,
        blockInPart * 2 + sourceK);
    auto const data = ldmatrix<false, 4>(source);
    return reinterpret_cast<RegMatAFrag const&>(data);
  }
  if constexpr (format == KVPageFormat::kBlockScaledFP8) {
    auto const packed = ldmatrix<false, 2>(&tile.template at<true>(
        pageToken + laneId() % tokensPerPage, blockInPart));
#pragma unroll
    for (uint32_t n = 0; n < 2; ++n) {
      uint32_t const token = n * 8 + tokenInHalf;
      uint8_t const scaleBits =
          scales[(pageToken + token) * scaleLoadBytes + scaleOffset];
      uint32_t const quadBase = laneId() & ~3U;
      uint32_t const pairLane = laneInQuad >> 1;
      uint32_t const halfShift = (laneInQuad & 1U) * 16U;
      uint32_t const lowWord =
          __shfl_sync(0xffffffffU, packed[n], quadBase + pairLane);
      uint32_t const highWord =
          __shfl_sync(0xffffffffU, packed[n], quadBase + pairLane + 2);
      uint16_t const low = static_cast<uint16_t>(lowWord >> halfShift);
      uint16_t const high = static_cast<uint16_t>(highWord >> halfShift);
      fragment(0, n)(0, 0) = scaleA16x2<InputElem>(
          convertE4M3x2ToA16<InputElem>(low), scaleBits, fp8GlobalScale);
      fragment(0, n)(1, 0) = scaleA16x2<InputElem>(
          convertE4M3x2ToA16<InputElem>(high), scaleBits, fp8GlobalScale);
    }
    return fragment;
  }
#pragma unroll
  for (uint32_t n = 0; n < 2; ++n) {
    uint32_t const token = n * 8 + tokenInHalf;
    auto const* packed = reinterpret_cast<uint8_t const*>(
        &tile.template at<true>(pageToken + token, blockInPart));
    uint8_t const scaleBits =
        scales[(pageToken + token) * scaleLoadBytes + scaleOffset];
    uint8_t const low = packed[elemInBlock / 2];
    uint8_t const high = packed[4 + elemInBlock / 2];
    fragment(0, n)(0, 0) = scaleA16x2<InputElem>(
        convertE2M1x2ToA16<InputElem>(low), scaleBits, fp4GlobalScale);
    fragment(0, n)(1, 0) = scaleA16x2<InputElem>(
        convertE2M1x2ToA16<InputElem>(high), scaleBits, fp4GlobalScale);
  }
  return fragment;
}

__device__ inline RegMatAFrag loadMixedKTileFragment(
    SharedMem::KBuffer const& tile, uint32_t pageInTile,
    uint32_t blockInPart, uint32_t idxPart,
    MixedPageFormats<SharedMem::nbPagesPerTile> const& formats,
    uint8_t const* scales, float fp8GlobalScale, float fp4GlobalScale) {
  using flashinfer::KVPageFormat;
#if MIXED_PAGE_STATIC_FORMAT == 0
  unused(formats);
  return loadMixedKTileFragment<KVPageFormat::kA16>(
      tile, pageInTile, blockInPart, idxPart, scales, fp8GlobalScale,
      fp4GlobalScale);
#elif MIXED_PAGE_STATIC_FORMAT == 1
  unused(formats);
  return loadMixedKTileFragment<KVPageFormat::kBlockScaledFP8>(
      tile, pageInTile, blockInPart, idxPart, scales, fp8GlobalScale,
      fp4GlobalScale);
#elif MIXED_PAGE_STATIC_FORMAT == 2
  unused(formats);
  return loadMixedKTileFragment<KVPageFormat::kBlockScaledFP4>(
      tile, pageInTile, blockInPart, idxPart, scales, fp8GlobalScale,
      fp4GlobalScale);
#else
  uint8_t const format = formats.values[pageInTile];
  if (format == static_cast<uint8_t>(KVPageFormat::kA16)) {
    return loadMixedKTileFragment<KVPageFormat::kA16>(
        tile, pageInTile, blockInPart, idxPart, scales, fp8GlobalScale,
        fp4GlobalScale);
  }
  if (format == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8)) {
    return loadMixedKTileFragment<KVPageFormat::kBlockScaledFP8>(
        tile, pageInTile, blockInPart, idxPart, scales, fp8GlobalScale,
        fp4GlobalScale);
  }
  assert(format == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4));
  return loadMixedKTileFragment<KVPageFormat::kBlockScaledFP4>(
      tile, pageInTile, blockInPart, idxPart, scales, fp8GlobalScale,
      fp4GlobalScale);
#endif
}

template <flashinfer::KVPageFormat format>
__device__ inline RegMatAFrag loadMixedVTileFragment(
    SharedMem::VBuffer::Elem const& tile, uint32_t pageInTile,
    uint32_t blockInPart, uint32_t idxPart, uint8_t const* scales,
    float fp8GlobalScale, float fp4GlobalScale) {
  using flashinfer::KVPageFormat;
  static_assert(tokensPerPage == 16);
  static_assert(format == KVPageFormat::kA16 ||
                format == KVPageFormat::kBlockScaledFP8 ||
                format == KVPageFormat::kBlockScaledFP4);
  constexpr uint32_t blocksPerPart =
      exactDiv(SharedMem::VBuffer::Elem::rowBytes, 2 * grainBytes);
  constexpr uint32_t scaleLoadBytes = mha::max(4U, blocksPerPart);
  uint32_t const laneInQuad = laneId() & 3U;
  uint32_t const headInHalf = laneId() >> 2;
  uint32_t const token0 = laneInQuad * 2;
  uint32_t const token1 = token0 + 1;
  uint32_t const token8 = token0 + 8;
  uint32_t const token9 = token1 + 8;
  uint32_t const scaleBlock = idxPart * blocksPerPart + blockInPart;
  uint32_t const scaleGroup = scaleBlock & ~(scaleLoadBytes - 1);
  uint32_t const scaleOffset = scaleBlock - scaleGroup;
  uint32_t const pageToken = pageInTile * tokensPerPage;

  auto const loadPair = [&](uint32_t head, uint32_t firstToken,
                            uint32_t secondToken) -> uint32_t {
    if constexpr (format == KVPageFormat::kA16) {
      uint32_t const grain = head / (grainBytes / sizeof(InputElem));
      uint32_t const byte =
          (head % (grainBytes / sizeof(InputElem))) * sizeof(InputElem);
      auto const* first = reinterpret_cast<uint8_t const*>(
                              &tile.template at<true>(pageToken + firstToken, grain)) +
                          byte;
      auto const* second = reinterpret_cast<uint8_t const*>(
                               &tile.template at<true>(pageToken + secondToken, grain)) +
                           byte;
      uint16_t const low = *reinterpret_cast<uint16_t const*>(first);
      uint16_t const high = *reinterpret_cast<uint16_t const*>(second);
      return uint32_t(low) | (uint32_t(high) << 16);
    }

    uint32_t const grain = head / 16;
    uint32_t const elem = head % 16;
    auto const* first = reinterpret_cast<uint8_t const*>(
        &tile.template at<true>(pageToken + firstToken, grain));
    auto const* second = reinterpret_cast<uint8_t const*>(
        &tile.template at<true>(pageToken + secondToken, grain));
    uint32_t packedA16;
    if constexpr (format == KVPageFormat::kBlockScaledFP8) {
      uint16_t const fp8x2 =
          uint16_t(first[elem]) | (uint16_t(second[elem]) << 8);
      packedA16 = convertE4M3x2ToA16<InputElem>(fp8x2);
    } else {
      uint8_t const shift = static_cast<uint8_t>((elem & 1U) * 4U);
      uint8_t const fp4x2 = static_cast<uint8_t>(
          ((first[elem / 2] >> shift) & 0xfU) |
          (((second[elem / 2] >> shift) & 0xfU) << 4));
      packedA16 = convertE2M1x2ToA16<InputElem>(fp4x2);
    }
    uint8_t const scale0 =
        scales[(pageToken + firstToken) * scaleLoadBytes + scaleOffset];
    uint8_t const scale1 =
        scales[(pageToken + secondToken) * scaleLoadBytes + scaleOffset];
    float const globalScale = format == KVPageFormat::kBlockScaledFP8
                                  ? fp8GlobalScale
                                  : fp4GlobalScale;
    return scaleA16x2Pair<InputElem>(packedA16, scale0, scale1, globalScale);
  };

  RegMatAFrag fragment;
#pragma unroll
  for (uint32_t n = 0; n < 2; ++n) {
    uint32_t const head = blockInPart * 16 + n * 8 + headInHalf;
    fragment(0, n)(0, 0) = loadPair(head, token0, token1);
    fragment(0, n)(1, 0) = loadPair(head, token8, token9);
  }
  return fragment;
}

__device__ inline RegMatAFrag loadMixedVTileFragment(
    SharedMem::VBuffer::Elem const& tile, uint32_t pageInTile,
    uint32_t blockInPart, uint32_t idxPart,
    MixedPageFormats<SharedMem::nbPagesPerTile> const& formats,
    uint8_t const* scales, float fp8GlobalScale, float fp4GlobalScale) {
  using flashinfer::KVPageFormat;
#if MIXED_PAGE_STATIC_FORMAT == 0
  unused(formats);
  return loadMixedVTileFragment<KVPageFormat::kA16>(
      tile, pageInTile, blockInPart, idxPart, scales, fp8GlobalScale,
      fp4GlobalScale);
#elif MIXED_PAGE_STATIC_FORMAT == 1
  unused(formats);
  return loadMixedVTileFragment<KVPageFormat::kBlockScaledFP8>(
      tile, pageInTile, blockInPart, idxPart, scales, fp8GlobalScale,
      fp4GlobalScale);
#elif MIXED_PAGE_STATIC_FORMAT == 2
  unused(formats);
  return loadMixedVTileFragment<KVPageFormat::kBlockScaledFP4>(
      tile, pageInTile, blockInPart, idxPart, scales, fp8GlobalScale,
      fp4GlobalScale);
#else
  uint8_t const format = formats.values[pageInTile];
  if (format == static_cast<uint8_t>(KVPageFormat::kA16)) {
    return loadMixedVTileFragment<KVPageFormat::kA16>(
        tile, pageInTile, blockInPart, idxPart, scales, fp8GlobalScale,
        fp4GlobalScale);
  }
  if (format == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8)) {
    return loadMixedVTileFragment<KVPageFormat::kBlockScaledFP8>(
        tile, pageInTile, blockInPart, idxPart, scales, fp8GlobalScale,
        fp4GlobalScale);
  }
  assert(format == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4));
  return loadMixedVTileFragment<KVPageFormat::kBlockScaledFP4>(
      tile, pageInTile, blockInPart, idxPart, scales, fp8GlobalScale,
      fp4GlobalScale);
#endif
}
#endif

#if SWAP_AB
constexpr uint32_t gemm1NbGmmaInstM = exactDiv(headElems, gmma::instM);
__device__ Vec<RegMatAFrag, gemm1NbGmmaInstM> loadVTileTransposed(uint32_t warpRank, uint32_t lane,
                                                                  SharedMem::VBuffer const& smemV,
                                                                  uint32_t idxGmmaInstK);
using Gemm1Acc = GmmaAcc<headElems, ctaNbQHeads>;
__device__ void rescaleGemm1AccForNewColMax_sync(uint32_t warpRank, ShmQWiseVec const& shmXColMax,
                                                 ShmQWiseVec const (&shmXColSum)[gemm0NbWarps],
                                                 ShmQWiseVec& shmAccColMax, Gemm1Acc& acc,
                                                 ShmQWiseVec& shmAccColSum);
template <bool dstIsStrided = false, typename DstHead>
__device__ void finalizeAndWriteOut_sync(
    uint32_t threadRank, uint32_t warpRank, DstHead* dst, SharedMem::OutSwizzleBuf& swizzleBuf,
    Gemm1Acc& acc, float xvoScale, CtaBarrier& warpGrpBar, ShmQWiseVec const& accColSum,
    ShmQWiseVec const& accColMax, ShmQWiseVec const* attentionSinksVec,
    uint32_t nbKHeads = 0 /* only for final result in spec dec. */);
#else
__device__ void transposeVTile(uint32_t warpRank, uint32_t lane, SharedMem::VTBuffer& dst,
                               SharedMem::VBuffer const& src);
using Gemm1Acc = GmmaAcc<ctaNbQHeads, headElems>;
__device__ void rescaleGemm1AccForNewRowMax_sync(uint32_t warpRank, ShmQWiseVec const& shmXRowMax,
                                                 ShmQWiseVec const(&shmXRowSum),
                                                 ShmQWiseVec& shmAccRowMax, Gemm1Acc& acc,
                                                 ShmQWiseVec& shmAccRowSum);
template <typename DstHead>
__device__ void finalizeAndWriteOut_sync(
    uint32_t warpRank, DstHead* dst, SharedMem::OutSwizzleBuf& swizzleBuf, Gemm1Acc& acc,
    float xvoScale, ShmQWiseVec const& accColSum,
    uint32_t nbKHeads /* only for final result in spec dec. set to 1 for workspace*/,
    uint32_t ctaNbValidTokens);
#endif

inline constexpr uint32_t ropeNbPairsPerThrdImpl(uint32_t nbThrds) {
  auto const val = divUp(exactDiv(validElemsPerHead, 2), nbThrds);
  assert(val <= 32);
  return val <= 2 ? val : (val <= 4 ? 4 : (val <= 8 ? 8 : (val <= 16 ? 16 : 32)));
}

template <uint32_t nbThrds>
inline constexpr uint32_t ropeNbPairsPerThrd = ropeNbPairsPerThrdImpl(nbThrds);

template <typename SrcElem, bool forNeox, uint32_t nbThrds, typename DstElem = float>
__device__ Vec<Vec<DstElem, 2>, ropeNbPairsPerThrd<nbThrds>> loadHead(
    Vec<SrcElem, validElemsPerHead> const& head, uint32_t tid);
template <bool forNeox, uint32_t nbPairsPerThrd>
__device__ mha::conditional_t<forNeox, Vec<Vec<CacheElem, nbPairsPerThrd>, 2>,
                              Vec<Vec<CacheElem, 2>, nbPairsPerThrd>>
applyRoPE(Vec<Vec<float, 2>, nbPairsPerThrd> const& data,
          Vec<Vec<float, 2>, nbPairsPerThrd> const& ropeCosSin);
template <bool forNeox, uint32_t nbThrds>
__device__ void storeRotatedPairsForKV(
    GMemCacheHead& dst,
    mha::conditional_t<forNeox, Vec<Vec<CacheElem, ropeNbPairsPerThrd<nbThrds>>, 2>,
                       Vec<Vec<CacheElem, 2>, ropeNbPairsPerThrd<nbThrds>>> const& src,
    uint32_t tid);
template <bool forNeox, uint32_t nbThrds>
__device__ void storeRotatedPairsForQ(
    SharedMem::QBuffer& dst,
    mha::conditional_t<forNeox, Vec<Vec<CacheElem, ropeNbPairsPerThrd<nbThrds>>, 2>,
                       Vec<Vec<CacheElem, 2>, ropeNbPairsPerThrd<nbThrds>>> const& src,
    uint32_t row, uint32_t tid);

class ScratchMem {
 public:
  struct alignas(8) SumMax {
    float sum;
    float max;
  };

  using ColWiseVec = Vec<SumMax, ctaNbValidQHeads>;

  HOST_DEVICE_FUNC ScratchMem(void* scratch, uint32_t maxTotalNbSubSeq, uint32_t nbInputSeqSplit)
      : mScratch{static_cast<mha::byte*>(scratch)} {
    uint32_t const nbChunks = maxTotalNbSubSeq * nbInputSeqSplit;
    Segmenter segmenter;
    constexpr uint32_t alignment = sizeof(Vec<IOHead, ctaNbValidQHeads>);
    mRowSumMax = segmenter.template newSeg<ColWiseVec>(nbChunks, alignment);
    mTokens = segmenter.template newSeg<Vec<IOHead, ctaNbValidQHeads>>(nbChunks, alignment);
  }

  HOST_DEVICE_FUNC TinyPtr<ColWiseVec> rowSumMax() const { return makePtr<ColWiseVec>(mRowSumMax); }

  HOST_DEVICE_FUNC TinyPtr<Vec<IOHead, ctaNbValidQHeads>> tokens() const {
    return makePtr<Vec<IOHead, ctaNbValidQHeads>>(mTokens);
  }

 private:
  template <typename T>
  HOST_DEVICE_FUNC TinyPtr<T> makePtr(uint32_t offset) const {
    return TinyPtr<mha::byte>{mScratch, offset}.template cast<T>();
  }

 private:
  mha::byte* mScratch;
  // offsets
  uint32_t mRowSumMax;
  uint32_t mTokens;
};

struct MultiBlockSMem {
  using ColWiseVec = ScratchMem::ColWiseVec;
  static constexpr uint32_t nbBuf = useSpecDec ? 2 : 4;
  static constexpr uint32_t nbIOWarps = nbBuf;
  using Elem = InputElem;
  using Head = Vec<Elem, headElems>;
  Vec<Vec<Head, ctaNbValidQHeads>, nbBuf> tokens;
  Vec<ColWiseVec, nbBuf> rowSumMax;
  Vec<CtaBarrierPair, nbBuf> barriers;
};

#ifndef NDEBUG
namespace dbg {
template <uint32_t nbGmmaInstM, uint32_t nbGmmaInstNBase>
__device__ void printAcc(CtaBarrier& warpGrpBar, uint32_t warpRank,
                         Array2D<GmmaAccCoreMat, nbGmmaInstM, nbGmmaInstNBase> const& acc) {
  for (int m = 0; m < nbGmmaInstM; m++) {
    for (int w = 0; w < 4; w++) {
      if (warpRank == w) {
        for (int a = 0; a < 2; a++) {
          for (int b = 0; b < 8; b++) {
            for (int n = 0; n < nbGmmaInstNBase; n++) {
              for (uint32_t i = 0; i < 4; i++) {
                if (laneId() == b * 4 + i) {
                  printf("%f, %f, ", acc(m, n)(a, 0), acc(m, n)(a, 1));
                }
                __syncwarp();
              }
            }
            if (laneId() == 0) {
              printf("\n");
            }
            __syncwarp();
          }
          if (laneId() == 0) {
            printf("\n");
          }
          __syncwarp();
        }
      }
      warpGrpBar.arrive_and_wait();
    }
  }
}

__device__ void printShmColWiseVec(ShmQWiseVec const& vec) {
  for (uint32_t i = 0; i < vec.size; i++) {
    printf("%f, ", vec[i]);
  }
  printf("\n");
}

template <typename Elem, bool swizzle, typename T, uint32_t rows, uint32_t cols,
          bool alignedForSwizzle>
__device__ void printArray2D(Array2D<T, rows, cols, alignedForSwizzle> const& src) {
  for (uint32_t i = 0; i < rows; i++) {
    for (uint32_t j = 0; j < cols; j++) {
      T const val = src.template at<swizzle>(i, j);
      for (uint32_t k = 0; k < exactDiv(sizeof(T), sizeof(Elem)); k++) {
        printf("%f, ", float(reinterpret_cast<Elem const*>(&val)[k]));
      }
    }
    printf("\n");
  }
}
}  // namespace dbg
#endif

CUBIN_EXPORT __device__ constexpr XQAKernelType kernelType =
    XQAKernelType::kHOPPER_WARP_SPECIALIZED;

CUBIN_EXPORT __global__
#ifdef NDEBUG
#if ENABLE_MIXED_KV_CACHE
// The mixed build is designed for two CTAs per SM (five warp groups, ~111 KB of
// shared memory); the register budget follows from that (65536 / 1280 -> 48),
// independent of OPTIMIZE_FOR_LATENCY.
__launch_bounds__(128 * ctaWarpGroups, 2)
#elif !OPTIMIZE_FOR_LATENCY
__launch_bounds__(128 * ctaWarpGroups, (headElems* ctaNbQHeads <= 128 * 16 ? 3 : 2))
#else
__launch_bounds__(128 * ctaWarpGroups)
#endif
#else
    __launch_bounds__(128 * ctaWarpGroups, 1)
#endif
    void kernel_mha(
        uint32_t const nbKHeads,
#if SLIDING_WINDOW
        uint32_t const slidingWinSize,
#endif
        float const qScale, float const* qScalePtr,
        OutputHead* __restrict__ const output,  // [nbReq][beamWidth][nbQHeads]
#if LOW_PREC_OUTPUT
        float rcpOutScale,
#endif
#if USE_INPUT_KV
        IOHead const* __restrict__ const qkv,  // [nbReq][beamWidth][nbQHeads+nbKHeads+nbVHeads],
#if ROPE_STYLE != 0
        Vec<float, validElemsPerHead> const* __restrict__ const ropeCosSin,  // [maxNbPosEmb]
#endif
#else
            IOHead const* __restrict__ const q, // [nbReq][beamWidth][nbQHeads],
#endif
        float const* attentionSinks,  // [headGrpSize]
        KVCacheList<usePagedKVCache> const cacheList,
#if USE_BEAM_SEARCH
        BeamSearchParams const beamSearchParams,
#endif
        uint32_t const batchSize, float kvCacheScale,
        float const* kvScalePtr,  // Same scale for K and V cache. Used only for int8/fp8 KV cache.
        __grid_constant__ CUtensorMap const tensorMapVLLMK,
        __grid_constant__ CUtensorMap const tensorMapVLLMV,
#if ENABLE_MIXED_KV_CACHE
        __grid_constant__ CUtensorMap const tensorMapFP8K,
        __grid_constant__ CUtensorMap const tensorMapFP8V,
        __grid_constant__ CUtensorMap const tensorMapFP4K,
        __grid_constant__ CUtensorMap const tensorMapFP4V,
#endif
#if SPEC_DEC
        SpecDecParams const specDecParams,
#endif
        uint32_t* __restrict__ const semaphores =
            nullptr,  // [nbReq][nbKHeads][divUp(specDecParams.qSeqLen, inputTokensPerCta)]
        void* __restrict__ const scratch = nullptr) {
  float const qScaleValue = qScalePtr != nullptr ? qScalePtr[0] : qScale;
  float const kvCacheScaleValue = kvScalePtr != nullptr ? kvScalePtr[0] : kvCacheScale;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900 && defined(__CUDA_ARCH_FEAT_SM90_ALL) && \
    (IS_SUPPORTED_F16_CASE || CACHE_ELEM_ENUM == 2) && BEAM_WIDTH == 1
  uint32_t const idxReq = blockIdx.z / nbKHeads;
#if SPEC_DEC
  uint32_t const reqInputTokBeg = getInputTokOffset(specDecParams, idxReq);
  uint32_t const reqInputTokEnd = getInputTokOffset(specDecParams, idxReq + 1);
  uint32_t const nbInputSeqSplit = gridDim.x;
  assert(nbInputSeqSplit == divUp(specDecParams.qSeqLen, inputTokensPerCta));
#else
  uint32_t const reqInputTokBeg = idxReq;
  uint32_t const reqInputTokEnd = idxReq + 1;
  constexpr uint32_t nbInputSeqSplit = 1;
  assert(gridDim.x == nbInputSeqSplit);
#endif
  uint32_t const idxHeadGrp = blockIdx.z % nbKHeads;  // inside one request
  assert(gridDim.z == nbKHeads * batchSize);
  uint32_t const cacheSeqLen = getCacheSeqLen<usePagedKVCache>(cacheList, idxReq);
  static_assert(gemm0CtaTileNbTokens == gemm1CtaTileNbTokens);
  constexpr uint32_t tileSize = gemm0CtaTileNbTokens;
#if SPEC_DEC
  uint32_t const idxInputSubSeq = blockIdx.x;
  uint32_t const inputSeqLen = reqInputTokEnd - reqInputTokBeg;
  uint32_t const ctaTokOffset = inputTokensPerCta * idxInputSubSeq;
  uint32_t const ctaNbValidTokens =
      mha::min(uint32_t{inputTokensPerCta}, inputSeqLen - ctaTokOffset);

  if (ctaTokOffset >= inputSeqLen) {
    return;
  }
#else
  uint32_t const idxInputSubSeq = 0;
  uint32_t const inputSeqLen = 1;
  uint32_t const ctaTokOffset = 0;
  uint32_t const ctaNbValidTokens = 1;
#endif
#if SLIDING_WINDOW && SPEC_DEC && !IS_SPEC_DEC_TREE
  // get the actual start position depending on ctaTokOffset, which is the draft token position per
  // CTA
  uint32_t const tok0SeqLen = cacheSeqLen - inputSeqLen + 1 + ctaTokOffset;
  int32_t const tok0WinBeg = int32_t(tok0SeqLen) - int32_t(slidingWinSize);
  uint32_t const nbTotalSkipTokens = mha::max(0, tok0WinBeg);
#elif SLIDING_WINDOW
  bool const rtIsReallySliding = (cacheSeqLen > slidingWinSize);
  // if SPEC_DEC && SLIDING_WINDOW && IS_SPEC_DEC_TREE, it should not do sliding
  assert(!SPEC_DEC || !rtIsReallySliding);
  uint32_t const nbTotalSkipTokens = rtIsReallySliding ? cacheSeqLen - slidingWinSize : 0;
#else
  constexpr bool rtIsReallySliding = false;
  constexpr uint32_t nbTotalSkipTokens = 0;
#endif
  uint32_t const nbSkipLeadingTiles = nbTotalSkipTokens / tileSize;
  uint32_t const tile0NbSkipTokens = nbTotalSkipTokens % tileSize;

#if USE_BEAM_SEARCH
  uint32_t const ctxCacheSeqLen = getCtxCacheSeqLen(beamSearchParams, idxReq);
  uint32_t const nbCtxKTiles = useKVCache ? ctxCacheSeqLen / gemm0CtaTileNbTokens : 0;
  uint32_t const nbDivergentKTiles =
      useKVCache
          ? divUp(cacheSeqLen - gemm0CtaTileNbTokens * nbCtxKTiles, beamSearchGemm0CtaTileNbTokens)
          : 0;
  uint32_t const nbKTiles = nbCtxKTiles + nbDivergentKTiles;
  uint32_t const nbVTiles = nbKTiles;
#else
  uint32_t const nbTiles = useKVCache ? divUp(cacheSeqLen, tileSize) : 0;
  // uint32_t const nbKTiles = nbTiles;
  // uint32_t const nbVTiles = nbTiles;
  uint32_t const nbTilesInUse = nbTiles - nbSkipLeadingTiles;
#endif
  uint32_t const maxNbSubSeq = gridDim.y;
  uint32_t const idxSubSeq = blockIdx.y;
  bool const isMultiBlockMode = (maxNbSubSeq > 1 && nbTilesInUse >= multiBlockMinNbTiles);
  uint32_t const idxKTileInit = nbSkipLeadingTiles + idxSubSeq;
  uint32_t const idxVTileInit = idxKTileInit;
  uint32_t const nbSubSeq =
      isMultiBlockMode ? mha::min(nbTilesInUse / multiBlockMinNbTilesPerCta, maxNbSubSeq) : 1;
  static_assert(multiBlockMinNbTiles >= multiBlockMinNbTilesPerCta * 2);
  assert(isMultiBlockMode == (nbSubSeq > 1));
  if (idxSubSeq >= nbSubSeq) {
    return;
  }
  uint32_t const ctaInputTokBeg = reqInputTokBeg + ctaTokOffset;
  auto const warpIdx = getWarpIdx(uint3{128, 1, ctaWarpGroups});
  auto const wid = warpIdx.z * 4 + warpIdx.x;
  if (wid == 0 && warpElectSync()) {
    tma::prefetchTensorMap(tensorMapVLLMK);
    tma::prefetchTensorMap(tensorMapVLLMV);
#if ENABLE_MIXED_KV_CACHE
    tma::prefetchTensorMap(tensorMapFP8K);
    tma::prefetchTensorMap(tensorMapFP8V);
    tma::prefetchTensorMap(tensorMapFP4K);
    tma::prefetchTensorMap(tensorMapFP4V);
#endif
  }
  extern __shared__ char smemByteBuf[];
  assert(dynamicSmemSize() >= sizeof(SharedMem));
  SharedMem& smem = *reinterpret_cast<SharedMem*>(&smemByteBuf[0]);
#if MIXED_KV_TRACE
  auto const traceStamp = [&](uint32_t slot, uint32_t idxIter, bool roleWarp0) {
    if (roleWarp0 && laneId() == 0 && idxIter < SharedMem::nbTraceTiles) {
      smem.trace[idxIter][slot] = clock64();
      if (slot == 0 && idxIter == 4) {
        smem.traceResidentMid = atomicAdd(&mixedKvTraceSmResident[smem.traceSmId], 0U);
      }
    }
  };
#define TRACE_STAMP(slot, it, w0) traceStamp((slot), (it), (w0))
#else
#define TRACE_STAMP(slot, it, w0) \
  do {                            \
  } while (0)
#endif

  constexpr uint32_t nbKBarWarps = SharedMem::nbKBuf;
  constexpr uint32_t nbVBarWarps = SharedMem::nbVBuf;
  constexpr uint32_t nbXBarWarps = SharedMem::nbXBuf;
  constexpr uint32_t nbBuffers = nbKBarWarps + nbVBarWarps + nbXBarWarps;
  static_assert(nbBuffers < nbWarps);
  if (wid < nbKBarWarps) {
    if (warpElectSync()) {
      smem.kBar[wid].initialize(
          gemm0NbThrds + (ENABLE_MIXED_KV_CACHE ? convertWarpsPerOperand * warp_size : 0),
          gemm0NbThrds +
              (ENABLE_MIXED_KV_CACHE ? mixedLoadWarpsPerOperand * warp_size
                                     : warp_size));
#if ENABLE_MIXED_KV_CACHE
      init(&smem.kLoadReady[wid],
           (convertWarpsPerOperand + mixedLoadWarpsPerOperand) * warp_size + 1);
#endif
    }
  } else if (wid < nbKBarWarps + nbVBarWarps) {
    uint32_t const i = wid - nbKBarWarps;
    if (warpElectSync()) {
      smem.vBar[i].initialize(
          gemm1NbThrds + (ENABLE_MIXED_KV_CACHE ? convertWarpsPerOperand * warp_size : 0),
          gemm1NbThrds +
              (ENABLE_MIXED_KV_CACHE ? mixedLoadWarpsPerOperand * warp_size
                                     : warp_size));
#if ENABLE_MIXED_KV_CACHE
      init(&smem.vLoadReady[i],
           (convertWarpsPerOperand + mixedLoadWarpsPerOperand) * warp_size + 1);
#endif
#if !SWAP_AB
      smem.vtBar[i].initialize(gemm1NbThrds * 2, gemm1NbThrds * 2);
#endif
    }
  } else if (wid < nbBuffers) {
    uint32_t const i = wid - nbKBarWarps - nbVBarWarps;
    if (warpElectSync()) {
      smem.xBar[i].initialize(gemm0NbThrds + gemm1NbThrds, gemm0NbThrds + gemm1NbThrds);
    }
  } else if (wid == nbBuffers) {
    if (warpElectSync()) {
      smem.qBar.initialize(gemm0NbThrds + nbQLdThrds, gemm0NbThrds + nbQLdThrds);
      init(&smem.gemm0WarpGrpBar, gemm0NbThrds);
      init(&smem.gemm1WarpGrpBar, gemm1NbThrds);
#if ENABLE_MIXED_KV_CACHE
      init(&smem.kPagesReady, mixedLoadWarpsPerOperand * warp_size);
      init(&smem.vPagesReady, mixedLoadWarpsPerOperand * warp_size);
#endif
    }
  }
#if MIXED_KV_TRACE
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    smem.traceSmId = mixedKvTraceSmId();
    atomicAdd(&mixedKvTraceSmResident[smem.traceSmId], 1U);
  }
#endif
  __syncthreads();

  uint32_t const nbPages = divUp(cacheSeqLen, tokensPerPage);

  constexpr bool isKVCacheQuantized = (cacheElemSize < 2);
  assert(idxKTileInit < nbTiles);
  uint32_t const nbIters = divUp(nbTiles - idxKTileInit, nbSubSeq);
  assert(nbIters >= 1);

  constexpr uint32_t gmmaInstK = gmma::instK<MathElem>;
  constexpr uint32_t grainsPerInstK = exactDiv(sizeof(MathElem) * gmmaInstK, grainBytes);

#if ENABLE_MIXED_KV_CACHE
  // Per-role register budget.  setmaxnreg.inc blocks until the CTA's pool has
  // the registers, and the pool is the launch allocation (640 x 48 = 30720),
  // so the split must balance within it: the IO group releases 24 x 128 =
  // 3072, the two converter groups (expansion + compressed-page copies, which
  // spill at 48) take 8 x 256 = 2048, the GEMM groups keep 48.
  // 128*24 + 2*128*56 + 2*128*48 = 29696 <= 30720.
  static_assert(ctaWarpGroups == 5);
  if (warpIdx.z == 2) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 24;\n" ::: "memory");
  } else if (warpIdx.z >= 3) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 56;\n" ::: "memory");
  }
#endif
  if (warpIdx.z == 0) {
#if SPEC_DEC
    SpecDec const specDec{specDecParams, idxReq, idxInputSubSeq, cacheSeqLen};
#endif

    // QK gemm
    constexpr uint32_t nbGmmaInstM = exactDiv(gemm0CtaTileNbTokens, gmma::instM);
    using Acc = GmmaAcc<gemm0CtaTileNbTokens, ctaNbQHeads>;

    unused(smem.qBar.consumed.arrive());
    for (auto& b : smem.kBar) {
      unused(b.consumed.arrive());
    }

    float const qkScale =
        qScaleValue * (isKVCacheQuantized ? kvCacheScaleValue : 1.f) *
        rsqrtf(validElemsPerHead);  // qkScale is applied onto Q*K.T before softmax.
    uint32_t const warpRank = warpIdx.x;

#if SWAP_AB
    // Running column max across the tiles of this CTA; every warp holds an
    // identical copy (all fold the same four per-warp slots in the same order).
    auto runningColMax = RegColWiseVec::filled(Vec<float, 2>::filled(safeInitRowMax));
#else
    // init once per sequence. It also works as global colMax across iterations.
    if (threadIdx.x < ctaNbQHeads) {
      smem.gemm0CurrentSeqMax[threadIdx.x] = safeInitRowMax;
    }
    smem.gemm0WarpGrpBar.arrive_and_wait();
#endif

    smem.qBar.produced.arrive_and_wait();
#if DBG_PRINT
    if (threadIdx.x == 0) {
      printf("q:\n");
      dbg::printArray2D<__nv_fp8_e4m3, true>(smem.q[0]);
    }
#endif

    auto const matDescQBase =
        gmma::makeMatDesc(nullptr, 0, SharedMem::QBuffer::Elem::rowBytes * 8,
                          gmma::getSwizzleMode<true>(SharedMem::QBuffer::Elem{}))
            .raw();
    for (uint32_t idxIter = 0; idxIter < nbIters; idxIter++) {
      uint32_t const idxKTile = idxKTileInit + idxIter * nbSubSeq;
      assert(idxKTile < nbTiles);
      Acc acc;  // no need to initialize. GMMA allows us to ignore acc initial values.
      gmma::fence();
      static_assert(cacheHeadNbParts == nbQParts);
#pragma unroll
#if ENABLE_MIXED_KV_CACHE
      uint32_t const idxKStage = idxIter % SharedMem::nbKBuf;
      smem.kBar[idxKStage].produced.arrive_and_wait();
      TRACE_STAMP(0, idxIter, warpRank == 0);
#endif
      for (uint32_t idxPart = 0; idxPart < cacheHeadNbParts; idxPart++) {
#if ENABLE_MIXED_KV_CACHE
        auto const idxKBuf = idxKStage * cacheHeadNbParts + idxPart;
        auto& kBar = smem.kBar[idxKStage];
#else
        auto const idxKBuf = (idxIter * cacheHeadNbParts + idxPart) % SharedMem::nbKBuf;
        auto& kBar = smem.kBar[idxKBuf];
#endif
        auto& kBuf = smem.k[idxKBuf];
        static_assert(SharedMem::KBuffer::rows % 8 == 0);
        auto const matDescKBase =
            gmma::makeMatDesc(nullptr, 0, SharedMem::KBuffer::rowBytes * 8, &smem.k[0],
                              gmma::getSwizzleMode<true>(SharedMem::KBuffer{}))
                .raw();
        assert(matDescKBase == gmma::makeMatDesc(nullptr, 0, SharedMem::KBuffer::rowBytes * 8,
                                                 gmma::getSwizzleMode<true>(SharedMem::KBuffer{}))
                                   .raw());
#if !ENABLE_MIXED_KV_CACHE
        arrive_tx_and_wait(kBar.produced,
                           exactDiv(sizeof(SharedMem::KBuffer), gemm0NbThrds));
#endif
        // if (threadIdx.x == 0) {
        //     printf("************* part %u *******\n", idxPart);
        //     printf("q:\n");
        //     dbg::printArray2D<__nv_fp8_e4m3, true>(smem.q[idxPart]);
        //     printf("k:\n");
        //     dbg::printArray2D<__nv_fp8_e4m3, true>(kBuf);
        // }
        constexpr uint32_t nbGmmaInstK = exactDiv(cacheHeadPartElems, gmmaInstK);
#pragma unroll
        for (uint32_t k = 0; k < nbGmmaInstK; k++) {
          bool const accHasVal = (idxPart != 0 || k != 0);
          auto const matDescQ = addAddr(matDescQBase, &smem.q[idxPart](0, grainsPerInstK * k));
#pragma unroll
          for (uint32_t m = 0; m < nbGmmaInstM; m++) {
            auto const matDescK = addAddr(matDescKBase, &kBuf(64 * m, grainsPerInstK * k));
#if SWAP_AB
            gmma::mma_async_shmA<MathElem, ctaNbQHeads>(
                reinterpret_cast<float(&)[exactDiv(ctaNbQHeads, gmma::instNBase)][2][2]>(acc(m, 0)),
                matDescK, matDescQ, accHasVal);
#else
            gmma::mma_async_shmA<MathElem, ctaNbQHeads>(
                reinterpret_cast<float(&)[exactDiv(ctaNbQHeads, gmma::instNBase)][2][2]>(acc(m, 0)),
                matDescQ, matDescK, accHasVal);
#endif
          }
        }
        gmma::commit_group();
#if !ENABLE_MIXED_KV_CACHE
        gmma::wait_group<0>();
        unused(kBar.consumed.arrive());
#endif
      }
#if ENABLE_MIXED_KV_CACHE
      gmma::wait_group<0>();
      unused(smem.kBar[idxKStage].consumed.arrive());
      TRACE_STAMP(1, idxIter, warpRank == 0);
#endif
#if !defined(NDEBUG) && DBG_PRINT
      dbg::printAcc(smem.gemm0WarpGrpBar, warpRank, acc);
#endif
      // apply qkScale
      acc = acc * qkScale;
      // apply mask
#if SPEC_DEC
      warpGrpApplyMask(acc, specDec,
#if SLIDING_WINDOW && !IS_SPEC_DEC_TREE
                       tok0WinBeg,
#endif
                       cacheSeqLen, idxKTile, warpRank);
#else
      bool const isFirstTile = (idxKTile == nbSkipLeadingTiles);
      bool const needMaskLeading = (rtIsReallySliding && isFirstTile && tile0NbSkipTokens > 0);
      bool const isLastTile = (idxKTile + 1 == nbTiles);
      bool const needMaskTrailing = isLastTile && cacheSeqLen % tileSize != 0;
      if (needMaskLeading || needMaskTrailing) {
        uint32_t const validTokenBeg = needMaskLeading ? tile0NbSkipTokens : 0;
        uint32_t const validTokenEnd = (needMaskTrailing ? cacheSeqLen % tileSize : tileSize);
        if (validTokenBeg > 0 || validTokenEnd < tileSize) {
#if SWAP_AB
          warpGrpApplyMask(warpRank, acc, validTokenBeg, validTokenEnd);
#else
          warpGrpApplyMask(acc, validTokenBeg, validTokenEnd);
#endif
        }
      }
#endif
      // update colMax in shared mem and get a register copy
#if SWAP_AB
      RegColWiseVec const colMax = computeWarpGrpColMax_sync(
          warpRank, smem.gemm0WarpColMax[idxIter % 2], runningColMax, acc);
      warpGrpOnlineSoftmax(acc, colMax);
      TRACE_STAMP(2, idxIter, warpRank == 0);
#else
      RegRowWiseVec const rowMax =
          computeWarpGrpRowMax_sync(warpRank, smem.gemm0CurrentSeqMax, acc);
      warpGrpOnlineSoftmax(acc, rowMax);
#endif

      // @fixme: may need fp32->fp8->fp32 before doing sum.
#if SWAP_AB
      RegColWiseVec const warpColSum = computeWarpColSum(acc);
#else
      RegRowWiseVec const rowSum = computeWarpRowSum(acc);
#endif

      // map 1 to fp8_max before conversion to fp8
      acc = acc * kE4M3_MAX;

      uint32_t const idxXBuf = idxIter % SharedMem::nbXBuf;
      auto& xBar = smem.xBar[idxXBuf];
      // @fixme: for fp16/bf16, try not to transpose acc here, and leave it to the next GEMM.
#if SWAP_AB
      storeGemm0AccToShm(warpRank, laneId(), smem.xBuf(idxXBuf), xBar.consumed, acc);
      // store colMax and warpColSum
      auto const lane = laneId();
      if (lane < 4) {
        auto& xColMax = smem.xColMax[idxXBuf];
        auto& xColSum = smem.xColSum[idxXBuf][warpRank];
#pragma unroll
        for (uint32_t n = 0; n < colMax.size; n++) {
#pragma unroll
          for (uint32_t j = 0; j < 2; j++) {
            if (warpRank == 0) {
              xColMax[8 * n + 2 * lane + j] = colMax[n][j];
            }
            xColSum[8 * n + 2 * lane + j] = warpColSum[n][j];
          }
        }
      }
#else
      storeGemm0AccToShm(warpRank, laneId(), smem.xBuf(idxXBuf), xBar.consumed, acc);
      storeShmRowWiseVec(warpRank, smem.xRowMax[idxXBuf], rowMax);
      storeShmRowWiseVec(warpRank, smem.xRowSum[idxXBuf], rowSum);
#endif

      __syncwarp();
      // the release semantics of arrive does not work for async consumers like gmma. additional
      // fence is needed.
      asm volatile("fence.proxy.async.shared::cta;\n");
      unused(xBar.produced.arrive());
      TRACE_STAMP(3, idxIter, warpRank == 0);
    }
    unused(smem.qBar.consumed.arrive());
  } else if (warpIdx.z == 1) {
    // XV GEMM
    for (auto& b : smem.vBar) {
      unused(b.consumed.arrive());
    }
#if !SWAP_AB
    for (auto& b : smem.vtBar) {
      unused(b.consumed.arrive());
    }
#endif
    for (auto& b : smem.xBar) {
      unused(b.consumed.arrive());
    }

    if (threadIdx.x < smem.gemm1AccColMax.size) {
      auto const idx = threadIdx.x;
      smem.gemm1AccColMax[idx] = safeInitRowMax;
      smem.gemm1AccColSum[idx] = 0;
    }
    smem.gemm1WarpGrpBar.arrive_and_wait();

    uint32_t const warpRank = warpIdx.x;

    constexpr float xScale = 1.f / kE4M3_MAX;
#if LOW_PREC_OUTPUT
    float const oScale = rcpOutScale;
#else
    constexpr float oScale = 1.F;
#endif
    float const xvoScale = xScale * (isKVCacheQuantized ? kvCacheScaleValue : 1.f) * oScale;

    Gemm1Acc acc{};  // init to zeros to avoid runtime checking for first gmma instruction.
    gmma::fence();

    static_assert(gemm0CtaTileNbTokens == gemm1CtaTileNbTokens, "not implemented");
    for (uint32_t idxIter = 0; idxIter < nbIters; idxIter++) {
      uint32_t idxVTile = idxVTileInit + idxIter * nbSubSeq;
      auto const idxVBuf = idxIter % SharedMem::nbVBuf;
      auto const idxXBuf = idxIter % SharedMem::nbXBuf;
      auto& vBar = smem.vBar[idxVBuf];
#if ENABLE_MIXED_KV_CACHE
      vBar.produced.arrive_and_wait();
#else
      arrive_tx_and_wait(vBar.produced,
                         exactDiv(sizeof(SharedMem::VBuffer), gemm1NbThrds));
#endif
      TRACE_STAMP(4, idxIter, warpRank == 0);
      auto const& vBuf = smem.vBuf(idxVBuf);
#if !SWAP_AB
      CtaBarrierPair& vtBar = smem.vtBar[idxVBuf];
      auto& vtBuf = smem.vtBuf(idxVBuf);
      vtBar.consumed.arrive_and_wait();
      transposeVTile(warpRank, laneId(), vtBuf, vBuf);
      vBar.consumed.arrive();
      vtBar.produced.arrive();
#endif
      auto& xBar = smem.xBar[idxXBuf];
      xBar.produced.arrive_and_wait();
      TRACE_STAMP(5, idxIter, warpRank == 0);
#if !defined(NDEBUG) && DBG_PRINT
#if SWAP_AB
      if (threadIdx.x == 0) {
        printf("colMax:\n");
        for (int i = 0; i < ctaNbQHeads; i++) {
          printf("%f, ", smem.xColMax[idxXBuf][i]);
        }
        printf("\n");
        printf("colSum:\n");
        for (int n = 0; n < 4; n++) {
          for (int i = 0; i < ctaNbQHeads; i++) {
            printf("%f, ", smem.xColSum[idxXBuf][n][i]);
          }
          printf("\n");
        }
        printf("\n");
        printf("X:\n");
        for (int i = 0; i < ctaNbQHeads; i++) {
          for (int j = 0; j < gemm0CtaTileNbTokens; j++) {
            auto const& elemsPerXPart = (cacheElemsPerGrain * grainsPerXPart);
            auto const e = reinterpret_cast<Vec<__nv_fp8_e4m3, 16>&>(
                smem.xBuf(idxXBuf)[j / elemsPerXPart].template at<true>(
                    i, j % elemsPerXPart / cacheElemsPerGrain))[j % cacheElemsPerGrain];
            printf("%.2f, ", float(e));
            if (j % 16 == 15) {
              printf("| ");
            }
          }
          printf("\n\n");
        }
      }
      smem.gemm1WarpGrpBar.arrive_and_wait();
#else
      if (blockIdx.y == 1 && threadIdx.x == 0) {
        printf("rowMax:\n");
        for (int i = 0; i < ctaNbQHeads; i++) {
          printf("%f, ", smem.xRowMax[idxXBuf][i]);
        }
        printf("\n");
        printf("rowSum:\n");
        for (int i = 0; i < ctaNbQHeads; i++) {
          printf("%f, ", smem.xRowSum[idxXBuf][i]);
        }
        printf("\n");
      }
      smem.gemm1WarpGrpBar.arrive_and_wait();
#endif
#endif

#if SWAP_AB
      // @fixme: if first tile, no need to rescale acc. For persistent CTA, just re-initialize acc
      // instead.
      rescaleGemm1AccForNewColMax_sync(warpRank, smem.xColMax[idxXBuf], smem.xColSum[idxXBuf],
                                       smem.gemm1AccColMax, acc, smem.gemm1AccColSum);
      TRACE_STAMP(6, idxIter, warpRank == 0);
#else
      rescaleGemm1AccForNewRowMax_sync(warpRank, smem.xRowMax[idxXBuf], smem.xRowSum[idxXBuf],
                                       smem.gemm1AccColMax, acc, smem.gemm1AccColSum);
#endif
      auto& xBuf = smem.xBuf(idxXBuf);

      auto const descXBase =
          gmma::makeMatDesc(nullptr, 0, SharedMem::XBuffer::Elem::rowBytes * 8,
                            gmma::getSwizzleMode<true>(SharedMem::XBuffer::Elem{}))
              .raw();
#if CACHE_ELEM_ENUM == 0 || CACHE_ELEM_ENUM == 5
      auto const descVBase =
          gmma::makeMatDesc(nullptr, 0, SharedMem::VBuffer::Elem::rowBytes * 8,
                            gmma::getSwizzleMode<true>(SharedMem::VBuffer::Elem{}))
              .raw();
#endif
#if SWAP_AB
//@fixme: to reduce code size, we can disable unroll and use double-buffer for LDSM in
// loadVTileTransposed.
#if CACHE_ELEM_ENUM == 0 || CACHE_ELEM_ENUM == 5
      // The rescale above wrote the accumulator registers with non-wgmma
      // instructions: fence them once before this tile's first PV wgmma.
      gmma::fence();
#endif
#pragma unroll
      for (uint32_t idxInstK = 0; idxInstK < gemm1NbGmmaInstK; idxInstK++) {
#if CACHE_ELEM_ENUM == 2
        Vec<RegMatAFrag, gemm1NbGmmaInstM> const fragA =
            loadVTileTransposed(warpRank, laneId(), vBuf, idxInstK);
#if !defined(NDEBUG) && DBG_PRINT
        if (threadIdx.x == 0) {
          printf("fragA:\nidxInstK == %u\n", idxInstK);
        }
        smem.gemm1WarpGrpBar.arrive_and_wait();
        for (int m = 0; m < 2; m++) {
          for (int w = 0; w < 4; w++) {
            if (warpRank == w) {
              if (laneId() == 0) {
                printf("    warpRank = %u\n", warpRank);
              }
              __syncwarp();
              for (int a = 0; a < 2; a++) {
                for (int b = 0; b < 8; b++) {
                  for (int c = 0; c < 2; c++) {
                    for (int d = 0; d < 4; d++) {
                      if (laneId() == b * 4 + d) {
                        for (int e = 0; e < 4; e++) {
                          auto const& elem4 =
                              reinterpret_cast<__nv_fp8_e4m3 const(&)[4]>(fragA[m](0, c)(a, 0));
                          printf("%.2f, ", float(elem4[e]));
                        }
                      }
                      __syncwarp();
                    }
                  }
                  if (laneId() == 0) {
                    printf("\n");
                  }
                  __syncwarp();
                }
                if (laneId() == 0 && a == 0) {
                  printf("----------------------\n");
                }
                __syncwarp();
              }
            }
            smem.gemm1WarpGrpBar.arrive_and_wait();
          }
        }
#endif
#endif
        BoundedVal<grainsPerInstK * gemm1NbGmmaInstK> const kOffsetInGrains{grainsPerInstK *
                                                                            idxInstK};
        auto const descX =
            addAddr(descXBase,
                    &xBuf[kOffsetInGrains.template divBy<SharedMem::XBuffer::Elem::cols>().get()](
                        0, kOffsetInGrains.template mod<SharedMem::XBuffer::Elem::cols>().get()));
#if CACHE_ELEM_ENUM == 2
        gmma::fence();
#endif
#pragma unroll
        for (uint32_t idxInstM = 0; idxInstM < gemm1NbGmmaInstM; idxInstM++) {
#if CACHE_ELEM_ENUM == 0 || CACHE_ELEM_ENUM == 5
          auto const descV =
              addAddr(descVBase, &vBuf[idxInstM](kOffsetInGrains.get() * cacheElemsPerGrain, 0));
          gmma::mma_async_shmA<MathElem, ctaNbQHeads, true, false>(
              reinterpret_cast<float(&)[exactDiv(ctaNbQHeads, gmma::instNBase)][2][2]>(
                  acc(idxInstM, 0)),
              descV, descX, true);
#elif CACHE_ELEM_ENUM == 2
          gmma::mma_async_regA<MathElem, ctaNbQHeads>(
              reinterpret_cast<float(&)[exactDiv(ctaNbQHeads, gmma::instNBase)][2][2]>(
                  acc(idxInstM, 0)),
              reinterpret_cast<uint32_t const(&)[2][2][1]>(fragA[idxInstM]), descX, true);
#endif
        }
#if CACHE_ELEM_ENUM == 2
        // Register-A fragments must stay live until the wgmma completes: commit
        // and drain per k-step.
        gmma::commit_group();
        gmma::wait_group<0>();
#endif
      }
#if CACHE_ELEM_ENUM == 0 || CACHE_ELEM_ENUM == 5
      // All gemm1NbGmmaInstK x gemm1NbGmmaInstM PV wgmmas read shared memory
      // only, so they stay in flight together: one commit and one drain per
      // tile instead of one per k-step.
      gmma::commit_group();
      gmma::wait_group<0>();
#endif
#else
      auto const descVTBase = gmma::makeMatDesc(nullptr, 0, SharedMem::VTBuffer::rowBytes * 8,
                                                gmma::getSwizzleMode<true>(SharedMem::VTBuffer{}))
                                  .raw();
      vtBar.produced.arrive_and_wait();
// if (idxIter == 1 && threadIdx.x == 0) {
//     printf("vtBuf:\n");
//     dbg::printArray2D<__nv_fp8_e4m3, true>(vtBuf);
// }
#pragma unroll
      for (uint32_t m = 0; m < Gemm1Acc::rows; m++) {
#pragma unroll
        for (uint32_t k = 0; k < gemm1NbGmmaInstK; k++) {
          BoundedVal<grainsPerInstK * gemm1NbGmmaInstK> const kOffsetInGrains{grainsPerInstK * k};
          auto const descX =
              addAddr(descXBase,
                      &xBuf[kOffsetInGrains.template divBy<SharedMem::XBuffer::Elem::cols>().get()](
                          gmma::instM * m,
                          kOffsetInGrains.template mod<SharedMem::XBuffer::Elem::cols>().get()));
          auto const descVT =
              addAddr(descVTBase,
                      &vtBuf(0, kOffsetInGrains.template mod<SharedMem::VTBuffer::cols>().get()));
          gmma::mma_async_shmA<MathElem, headElems>(
              reinterpret_cast<float(&)[exactDiv(headElems, gmma::instNBase)][2][2]>(acc(m, 0)),
              descX, descVT, true);
        }
      }
      gmma::commit_group();
      //@fixme: delay wait and consumption to next tile. Note that fragA must also persist until
      // finish of gmma.
      gmma::wait_group<0>();
#endif
      if (idxIter == nbIters - 1) {
        // gmma::wait_group should have already synchronized threads, so this may be unnecessary.
        smem.gemm1WarpGrpBar.arrive_and_wait();
        if (isMultiBlockMode) {
          ScratchMem const scratchMem{scratch, maxNbSubSeq * nbKHeads * batchSize, nbInputSeqSplit};
          uint32_t const idxSeq = nbKHeads * idxReq + idxHeadGrp;
          uint32_t const idxAllSubSeq = maxNbSubSeq * idxSeq + idxSubSeq;
          uint32_t const idxChunk = idxAllSubSeq * nbInputSeqSplit + idxInputSubSeq;
          // save row max/sum
          static_assert(ctaNbValidQHeads <= gmmaWarpsPerGrp * warp_size);
          if (threadIdx.x < ctaNbValidQHeads) {
            float const colMax = smem.gemm1AccColMax[threadIdx.x];
            float const colSum = smem.gemm1AccColSum[threadIdx.x];
            ScratchMem::SumMax sumMax;
            sumMax.sum = colSum;
            sumMax.max = colMax;
            (scratchMem.rowSumMax() + idxChunk).template cast<ScratchMem::SumMax>()[threadIdx.x] =
                sumMax;
          }
          // compute scratch ptr for output writing
          IOHead* const dst = (scratchMem.tokens() + idxChunk).template cast<IOHead>();
#if SWAP_AB
          finalizeAndWriteOut_sync(threadIdx.x, warpRank, dst, smem.outSwizzleBuf(idxXBuf), acc,
                                   xvoScale, smem.gemm1WarpGrpBar, smem.gemm1AccColSum,
                                   smem.gemm1AccColMax, nullptr);
#else
          finalizeAndWriteOut_sync(warpRank, dst, smem.outSwizzleBuf(idxXBuf), acc, xvoScale,
                                   smem.gemm1AccColSum, 1, ctaNbValidTokens);
#endif
        } else {
          uint32_t const outOffset =
              headGrpSize * (nbKHeads * (beamWidth * ctaInputTokBeg) + idxHeadGrp);
          OutputHead* const dst = &output[outOffset];
          ShmQWiseVec const* attentionSinksVec = nullptr;
          if (attentionSinks != nullptr) {
            attentionSinksVec =
                reinterpret_cast<ShmQWiseVec const*>(attentionSinks + headGrpSize * idxHeadGrp);
          }
#if SWAP_AB
          finalizeAndWriteOut_sync<SPEC_DEC>(threadIdx.x, warpRank, dst,
                                             smem.outSwizzleBuf(idxXBuf), acc, xvoScale,
                                             smem.gemm1WarpGrpBar, smem.gemm1AccColSum,
                                             smem.gemm1AccColMax, attentionSinksVec, nbKHeads);
#else
          finalizeAndWriteOut_sync(warpRank, dst, smem.outSwizzleBuf(idxXBuf), acc, xvoScale,
                                   smem.gemm1AccColSum, nbKHeads, ctaNbValidTokens);
#endif
        }
      }
      TRACE_STAMP(7, idxIter, warpRank == 0);
      unused(xBar.consumed.arrive());
#if SWAP_AB
      unused(vBar.consumed.arrive());
#else
      unused(vtBar.consumed.arrive());
#endif
    }
  } else if (warpIdx.z == 2) {
    // IO warps
    static_assert(beamWidth == 1);
#if ENABLE_PDL
    preExit();
#endif
#if ENABLE_PDL == 1
    acqBulk();
#endif
    uint32_t const newTokenPos = cacheSeqLen - 1;
    if (warpIdx.x < nbQLdWarps) {
      // load Q. Use register to load fp16 data and store fp8 to shared mem.
      // @fixme: If register pressure is high and shared mem pressure is low, switch to TMA instead.
      using QCvt = F16QToF8Converter<nbQLdThrds, beamWidth>;
      static_assert(beamWidth == 1);
#if USE_INPUT_KV
      TinyPtr<IOHead const> const qData{
          qkv, headGrpSize * idxHeadGrp + (headGrpSize + 2) * nbKHeads * idxReq};
      constexpr bool isNeox = (ROPE_STYLE == 1);
      constexpr uint32_t thrdsPerHead = mha::min(warp_size, divUp(headElems, 4U));
      uint32_t const lane = laneId();
      uint32_t const idxThrd = warpIdx.x * warp_size + lane;
      uint32_t const idxThrdGrp =
          (thrdsPerHead % 32 == 0 ? makeWarpUniform(this_warp(), idxThrd / thrdsPerHead)
                                  : idxThrd / thrdsPerHead);
      constexpr uint32_t nbThrdGrps = exactDiv(warp_size * nbQLdWarps, thrdsPerHead);
      uint32_t const tid = idxThrd % thrdsPerHead;
      smem.qBar.consumed.arrive_and_wait();
#if ROPE_STYLE != 0
      auto const& ropeCosSinHead =
          reinterpret_cast<Vec<float, validElemsPerHead> const&>(ropeCosSin[cacheSeqLen - 1]);
      auto const cosSinPairs = loadHead<float, false, thrdsPerHead>(ropeCosSinHead, tid);
#endif
#if ENABLE_PDL == 2
      acqBulk();
#endif
#pragma unroll
      for (uint32_t iter = 0; iter < divUp(headGrpSize, nbThrdGrps); iter++) {
        uint32_t const idxHead = nbThrdGrps * iter + idxThrdGrp;
        if (idxHead >= headGrpSize) {
          break;
        }
#if ROPE_STYLE == 0
        auto const rotatedPairs =
            loadHead<InputElem, isNeox, thrdsPerHead, MathElem>(qData[idxHead], tid);
#else
        auto const pairs = loadHead<InputElem, isNeox, thrdsPerHead>(qData[idxHead], tid);
        auto const rotatedPairs = applyRoPE<isNeox>(pairs, cosSinPairs);
#endif
        storeRotatedPairsForQ<isNeox, thrdsPerHead>(smem.q, rotatedPairs, idxHead, tid);
      }
#else
      TinyPtr<IOHead const> const qData{
          q, headGrpSize * (nbKHeads * (beamWidth * ctaInputTokBeg) + idxHeadGrp)};
#if ENABLE_PDL == 2
      acqBulk();
#endif
      auto const f16QData = QCvt::load(threadIdx.x, qData, nbKHeads, ctaNbValidTokens);

      smem.qBar.consumed.arrive_and_wait();
      QCvt::store(threadIdx.x, smem.q, f16QData);
#endif
      // the release semantics of arrive does not work for async consumers like gmma. additional
      // fence is needed.
      asm volatile("fence.proxy.async.shared::cta;\n");
      unused(smem.qBar.produced.arrive());
    }
    constexpr uint32_t kLoadWarpBeg = ENABLE_MIXED_KV_CACHE ? 0 : nbQLdWarps;
    constexpr uint32_t kLoadWarpCount =
        ENABLE_MIXED_KV_CACHE ? mixedLoadWarpsPerOperand : 1;
    constexpr uint32_t vLoadWarpBeg =
        ENABLE_MIXED_KV_CACHE ? mixedLoadWarpsPerOperand : nbQLdWarps + 1;
    constexpr uint32_t vLoadWarpCount =
        ENABLE_MIXED_KV_CACHE ? mixedLoadWarpsPerOperand : 1;
    if (warpIdx.x >= kLoadWarpBeg &&
        warpIdx.x < kLoadWarpBeg + kLoadWarpCount) {  // load k
      uint32_t const idxLoadWarp = warpIdx.x - kLoadWarpBeg;
      KVTilePartLoader kTilePartLoader{true,       nbKHeads,       cacheList, idxReq,
                                       idxHeadGrp, tensorMapVLLMK, nbPages,   smem.pages[0]
#if ENABLE_MIXED_KV_CACHE
                                       , smem.pageFormats[0]
#endif
      };
#if ENABLE_MIXED_KV_CACHE && ENABLE_PDL == 2
      acqBulk();
#endif
#if ENABLE_MIXED_KV_CACHE
      static_assert(!USE_INPUT_KV, "mixed pages do not append input KV in-kernel");
      static_assert(mixedLoadWarpsPerOperand == 1);
      // One warp streams K.  Per tile: wait for the stage (released by gemm0
      // two tiles back), issue every part's TMA, issue the *next* tile's block
      // scales, wait for this tile's scales (issued a tile ago), release the
      // stage.  Page metadata comes from a shared-memory chunk refilled once
      // per metaChunkTiles tiles, so nothing here waits on DRAM per tile.
      constexpr uint32_t kMeta = 0;
      kTilePartLoader.fillTileMeta(smem, kMeta, 0, nbIters, idxKTileInit, nbSubSeq);
      if (nbIters > SharedMem::metaChunkTiles) {
        kTilePartLoader.fillTileMeta(smem, kMeta, SharedMem::metaChunkTiles, nbIters,
                                     idxKTileInit, nbSubSeq);
      }
      // The first two metadata chunks are complete: the converter warps may read
      // page lists from here on (they copy the compressed pages themselves).
      unused(smem.kPagesReady.arrive());
      for (uint32_t idxIter = 0; idxIter < nbIters; idxIter++) {
        uint32_t const idxKTile = idxKTileInit + idxIter * nbSubSeq;
        uint32_t const idxKStage = idxIter % SharedMem::nbKBuf;
        kTilePartLoader.publishPages(kTilePartLoader.readTileMeta(smem, kMeta, idxIter), idxKTile);
        smem.kBar[idxKStage].consumed.arrive_and_wait();
#if MIXED_KV_TRACE < 3
        TRACE_STAMP(8, idxIter, true);
#endif
        if (warpElectSync()) {
          smem.kFormats[idxKStage] = kTilePartLoader.formats;
          uint32_t txBytes = 0;
#pragma unroll
          for (uint32_t idxPart = 0; idxPart < cacheHeadNbParts; idxPart++) {
            txBytes += kTilePartLoader.packedPartTxBytes(idxPart);
          }
          arrive_tx(smem.kLoadReady[idxKStage], txBytes);
#pragma unroll
          for (uint32_t idxPart = 0; idxPart < cacheHeadNbParts; idxPart++) {
            kTilePartLoader.issuePackedPartLoad(
                smem.k[idxKStage * cacheHeadNbParts + idxPart],
                reinterpret_cast<SharedMem::PackedTile&>(
                    smem.k[idxKStage * cacheHeadNbParts + cacheHeadNbParts - 1]),
                idxPart, tensorMapFP8K, tensorMapFP4K, smem.kLoadReady[idxKStage]);
          }
        }
        __syncwarp();
        {
          // Metadata chunk for tiles [ahead, ahead + 16) is filled two tiles
          // before its first use; the converters read tile t+1's entry after
          // this iteration's kLoadReady arrive, which orders the fill.
          uint32_t const ahead = idxIter + 2;
          if (ahead < nbIters && ahead % SharedMem::metaChunkTiles == 0 &&
              ahead >= 2 * SharedMem::metaChunkTiles) {
            kTilePartLoader.fillTileMeta(smem, kMeta, ahead, nbIters, idxKTileInit, nbSubSeq);
          }
        }
        unused(smem.kLoadReady[idxKStage].arrive());
#if MIXED_KV_TRACE < 3
        TRACE_STAMP(9, idxIter, true);
#endif
      }
#else
      for (uint32_t idxIter = 0; idxIter < nbIters; idxIter++) {
        uint32_t const idxKTile = idxKTileInit + idxIter * nbSubSeq;
        kTilePartLoader.loadPages(idxKTile, idxLoadWarp == 0);
#if USE_INPUT_KV || ENABLE_PDL == 2
#if SPEC_DEC
        bool const anyNewTokens =
            (gemm0CtaTileNbTokens * (idxKTile + 1) > cacheSeqLen - inputSeqLen);
#else
        bool const anyNewTokens = (gemm0CtaTileNbTokens * (idxKTile + 1) >= cacheSeqLen);
#endif
        if (anyNewTokens && idxLoadWarp == 0) {
#if ENABLE_PDL == 2
          acqBulk();
#endif
#if USE_INPUT_KV
          static_assert(beamWidth == 1);
          uint32_t const inputKHeadOffset =
              headGrpSize * nbKHeads + idxHeadGrp + (headGrpSize + 2) * nbKHeads * idxReq;
          IOHead const& inKHead = qkv[inputKHeadOffset];
          uint32_t const lane = laneId();
          float const rcpKScale = 1.F / kvCacheScaleValue;
#if ROPE_STYLE == 0
          constexpr bool isNeox = false;
          auto const pairs =
              loadHead<InputElem, isNeox, warp_size, float>(inKHead, lane) * rcpKScale;
          Vec<Vec<CacheElem, decltype(pairs)::Elem::size>, decltype(pairs)::size> convertedPairs;
          constexpr uint32_t nbElems = decltype(pairs)::Elem::size * decltype(pairs)::size;
          reinterpret_cast<Vec<CacheElem, nbElems>&>(convertedPairs) =
              convert<CacheElem>(reinterpret_cast<Vec<float, nbElems> const&>(pairs));
          storeRotatedPairsForKV<isNeox, warp_size>(kTilePartLoader.getHead(newTokenPos),
                                                    convertedPairs, lane);
#else
          constexpr bool isNeox = (ROPE_STYLE == 1);
          auto const pairs = loadHead<InputElem, isNeox, warp_size>(inKHead, lane) * rcpKScale;
          auto const& ropeCosSinHead =
              reinterpret_cast<Vec<float, validElemsPerHead> const&>(ropeCosSin[cacheSeqLen - 1]);
          auto const cosSinPairs = loadHead<float, false, warp_size>(ropeCosSinHead, lane);
          auto const rotatedPairs = applyRoPE<isNeox>(pairs, cosSinPairs);
          storeRotatedPairsForKV<isNeox, warp_size>(kTilePartLoader.getHead(newTokenPos),
                                                    rotatedPairs, lane);
#endif
          static_assert(inputSeqLen == 1);
          __syncwarp();
#endif
        }
#endif
        for (uint32_t idxPart = 0; idxPart < cacheHeadNbParts; idxPart++) {
          auto const idxKBuf = (idxIter * cacheHeadNbParts + idxPart) % SharedMem::nbKBuf;
          auto& kBar = smem.kBar[idxKBuf];
          kBar.consumed.arrive_and_wait();
          if (idxPart == 0) TRACE_STAMP(8, idxIter, idxLoadWarp == 0);
          if (warpElectSync()) {
            kTilePartLoader.loadData(smem.k[idxKBuf], idxKTile, idxPart, kBar.produced);
          }
          __syncwarp();
        }
      }
#endif
    } else if (warpIdx.x >= vLoadWarpBeg &&
               warpIdx.x < vLoadWarpBeg + vLoadWarpCount) {  // load v
      uint32_t const idxLoadWarp = warpIdx.x - vLoadWarpBeg;
      constexpr bool vAlignedForSwizzle = SharedMem::vBufAlignedForSwizzle;
      KVTilePartLoader vTileLoader{false,      nbKHeads,       cacheList, idxReq,
                                   idxHeadGrp, tensorMapVLLMV, nbPages,   smem.pages[1]
#if ENABLE_MIXED_KV_CACHE
                                   , smem.pageFormats[1]
#endif
      };
#if ENABLE_MIXED_KV_CACHE && ENABLE_PDL == 2
      acqBulk();
#endif
#if ENABLE_MIXED_KV_CACHE
      constexpr uint32_t vMeta = 1;
      vTileLoader.fillTileMeta(smem, vMeta, 0, nbIters, idxVTileInit, nbSubSeq);
      if (nbIters > SharedMem::metaChunkTiles) {
        vTileLoader.fillTileMeta(smem, vMeta, SharedMem::metaChunkTiles, nbIters, idxVTileInit,
                                 nbSubSeq);
      }
      unused(smem.vPagesReady.arrive());
      for (uint32_t idxIter = 0; idxIter < nbIters; idxIter++) {
        uint32_t const idxVTile = idxVTileInit + idxIter * nbSubSeq;
        uint32_t const idxVBuf = idxIter % SharedMem::nbVBuf;
        vTileLoader.publishPages(vTileLoader.readTileMeta(smem, vMeta, idxIter), idxVTile);
        smem.vBar[idxVBuf].consumed.arrive_and_wait();
#if MIXED_KV_TRACE < 2
        TRACE_STAMP(10, idxIter, true);
#endif
        if (warpElectSync()) {
          smem.vFormats[idxVBuf] = vTileLoader.formats;
          uint32_t txBytes = 0;
#pragma unroll
          for (uint32_t idxPart = 0; idxPart < cacheHeadNbParts; idxPart++) {
            txBytes += vTileLoader.packedPartTxBytes(idxPart);
          }
          arrive_tx(smem.vLoadReady[idxVBuf], txBytes);
#pragma unroll
          for (uint32_t idxPart = 0; idxPart < cacheHeadNbParts; idxPart++) {
            vTileLoader.issuePackedPartLoad(
                smem.vBuf(idxVBuf)[idxPart],
                reinterpret_cast<SharedMem::PackedTile&>(smem.vBuf(idxVBuf)[cacheHeadNbParts - 1]),
                idxPart, tensorMapFP8V, tensorMapFP4V, smem.vLoadReady[idxVBuf]);
          }
        }
        __syncwarp();
        {
          uint32_t const ahead = idxIter + 2;
          if (ahead < nbIters && ahead % SharedMem::metaChunkTiles == 0 &&
              ahead >= 2 * SharedMem::metaChunkTiles) {
            vTileLoader.fillTileMeta(smem, vMeta, ahead, nbIters, idxVTileInit, nbSubSeq);
          }
        }
        unused(smem.vLoadReady[idxVBuf].arrive());
#if MIXED_KV_TRACE < 2
        TRACE_STAMP(11, idxIter, true);
#endif
      }
#else
      for (uint32_t idxIter = 0; idxIter < nbIters; idxIter++) {
        uint32_t const idxVTile = idxVTileInit + idxIter * nbSubSeq;
        vTileLoader.loadPages(idxVTile, idxLoadWarp == 0);
#if USE_INPUT_KV || ENABLE_PDL == 2
#if SPEC_DEC
        bool const anyNewTokens =
            (gemm0CtaTileNbTokens * (idxVTile + 1) > cacheSeqLen - inputSeqLen);
#else
        bool const anyNewTokens = (gemm0CtaTileNbTokens * (idxVTile + 1) >= cacheSeqLen);
#endif
        if (anyNewTokens && idxLoadWarp == 0) {
#if ENABLE_PDL == 2
          acqBulk();
#endif
#if USE_INPUT_KV
          static_assert(beamWidth == 1);
          uint32_t const inputVHeadOffset =
              (headGrpSize + 1) * nbKHeads + idxHeadGrp + (headGrpSize + 2) * nbKHeads * idxReq;
          IOHead const& inVHead = qkv[inputVHeadOffset];
          uint32_t const lane = laneId();
          float const rcpVScale = 1.F / kvCacheScaleValue;
          constexpr bool isNeox = false;
          auto const pairs =
              loadHead<InputElem, isNeox, warp_size, float>(inVHead, lane) * rcpVScale;
          Vec<Vec<CacheElem, decltype(pairs)::Elem::size>, decltype(pairs)::size> convertedPairs;
          constexpr uint32_t nbElems = decltype(pairs)::Elem::size * decltype(pairs)::size;
          reinterpret_cast<Vec<CacheElem, nbElems>&>(convertedPairs) =
              convert<CacheElem>(reinterpret_cast<Vec<float, nbElems> const&>(pairs));
          static_assert(SPEC_DEC == 0);
          storeRotatedPairsForKV<isNeox, warp_size>(vTileLoader.getHead(newTokenPos),
                                                    convertedPairs, lane);
          __syncwarp();
#endif
        }
#endif

        uint32_t const idxVBuf = idxIter % SharedMem::nbVBuf;
        auto& vBar = smem.vBar[idxVBuf];
        vBar.consumed.arrive_and_wait();
        TRACE_STAMP(10, idxIter, idxLoadWarp == 0);
        if (warpElectSync()) {
#pragma unroll
          for (uint32_t idxPart = 0; idxPart < cacheHeadNbParts; idxPart++) {
            vTileLoader.loadData(smem.vBuf(idxVBuf)[idxPart], idxVTile, idxPart, vBar.produced);
          }
        }
        __syncwarp();
      }
#endif
    }
#if ENABLE_MIXED_KV_CACHE
  } else if (warpIdx.z == 3) {
    float const fp8KGlobalScale =
        *cacheList.transport.formats[static_cast<uint8_t>(
             flashinfer::KVPageFormat::kBlockScaledFP8)]
             .k_global_scale;
    float const fp4KGlobalScale =
        *cacheList.transport.formats[static_cast<uint8_t>(
             flashinfer::KVPageFormat::kBlockScaledFP4)]
             .k_global_scale;
    // Converter warps own the compressed pages end to end: they wait for the
    // stage to be free, copy the packed rows and scales (cp.async, one group per
    // tile), and expand them in place once landed.  Tile t+1's copies are issued
    // right after tile t is expanded, so their landing overlaps gemm0(t).
    constexpr uint32_t kMeta = 0;
    // Copies run kAhead tiles ahead of expansion (one per free stage beyond the
    // one being expanded); landing latency under a saturated memory system is
    // ~2 us, more than one consumer tile, so two ahead with three stages.
    constexpr uint32_t kAhead = SharedMem::nbKBuf - 1;
    auto issueKCopies = [&](uint32_t t) {
      uint32_t const stage = t % SharedMem::nbKBuf;
      // Wait (do not arrive) for gemm0's release of this stage's previous tile:
      // the (t / nbKBuf)-th completion of consumed[stage].
      smem.kBar[stage].consumed.wait_parity(toParity<SharedMem::nbKBuf>(t));
      issueCompressedPageCopies(
          cacheList, /*isK=*/true, idxHeadGrp, smem, kMeta, t,
          reinterpret_cast<SharedMem::PackedTile&>(smem.k[stage * cacheHeadNbParts + cacheHeadNbParts - 1]),
          smem.kScales[t % SharedMem::nbScaleTiles], warpIdx.x);
    };
    smem.kPagesReady.wait_parity(false);  // metadata chunks published by the loader
#pragma unroll
    for (uint32_t t = 0; t < kAhead; ++t) {
      if (t < nbIters) issueKCopies(t);
      ldgsts::commitGroup();
    }
    for (uint32_t idxIter = 0; idxIter < nbIters; ++idxIter) {
      uint32_t const idxKStage = idxIter % SharedMem::nbKBuf;
      ldgsts::waitGroup<kAhead - 1>();  // this lane's copies of tile idxIter landed
      __syncwarp();                     // ... and the other lanes' (same page, same warp)
#if MIXED_KV_TRACE >= 2
      TRACE_STAMP(11, idxIter, warpIdx.x == 0);  // K converter: tile idxIter landed
#endif
      smem.kLoadReady[idxKStage].arrive_and_wait();  // A16 rows landed, kFormats published
      TRACE_STAMP(12, idxIter, warpIdx.x == 0);
      expandPackedStage(&smem.k[idxKStage * cacheHeadNbParts],
                        smem.kScales[idxIter % SharedMem::nbScaleTiles], smem.kFormats[idxKStage],
                        fp8KGlobalScale, fp4KGlobalScale, warpIdx.x, /*namedBarrierId=*/1);
      TRACE_STAMP(13, idxIter, warpIdx.x == 0);
      asm volatile("fence.proxy.async.shared::cta;\n");
      unused(smem.kBar[idxKStage].produced.arrive());
#if MIXED_KV_TRACE >= 3
      TRACE_STAMP(8, idxIter, warpIdx.x == 0);  // K converter: fenced + produced
#endif
      if (idxIter + kAhead < nbIters) issueKCopies(idxIter + kAhead);
#if MIXED_KV_TRACE >= 3
      TRACE_STAMP(9, idxIter, warpIdx.x == 0);  // K converter: copies for t+2 issued (before commit)
#endif
      ldgsts::commitGroup();
#if MIXED_KV_TRACE >= 2
      TRACE_STAMP(10, idxIter, warpIdx.x == 0);  // K converter: copies for t+2 issued
#endif
    }
  } else {
    assert(warpIdx.z == 4);
    float const fp8VGlobalScale =
        *cacheList.transport.formats[static_cast<uint8_t>(
             flashinfer::KVPageFormat::kBlockScaledFP8)]
             .v_global_scale;
    float const fp4VGlobalScale =
        *cacheList.transport.formats[static_cast<uint8_t>(
             flashinfer::KVPageFormat::kBlockScaledFP4)]
             .v_global_scale;
    constexpr uint32_t vMeta = 1;
    constexpr uint32_t vAhead = SharedMem::nbVBuf - 1;
    auto issueVCopies = [&](uint32_t t) {
      uint32_t const buf = t % SharedMem::nbVBuf;
      smem.vBar[buf].consumed.wait_parity(toParity<SharedMem::nbVBuf>(t));
      issueCompressedPageCopies(
          cacheList, /*isK=*/false, idxHeadGrp, smem, vMeta, t,
          reinterpret_cast<SharedMem::PackedTile&>(smem.vBuf(buf)[cacheHeadNbParts - 1]),
          smem.vScales[t % SharedMem::nbScaleTiles], warpIdx.x);
    };
    smem.vPagesReady.wait_parity(false);
#pragma unroll
    for (uint32_t t = 0; t < vAhead; ++t) {
      if (t < nbIters) issueVCopies(t);
      ldgsts::commitGroup();
    }
    for (uint32_t idxIter = 0; idxIter < nbIters; ++idxIter) {
      uint32_t const idxVBuf = idxIter % SharedMem::nbVBuf;
      ldgsts::waitGroup<vAhead - 1>();
      __syncwarp();
      smem.vLoadReady[idxVBuf].arrive_and_wait();
      TRACE_STAMP(14, idxIter, warpIdx.x == 0);
      expandPackedStage(&smem.vBuf(idxVBuf)[0], smem.vScales[idxIter % SharedMem::nbScaleTiles],
                        smem.vFormats[idxVBuf], fp8VGlobalScale, fp4VGlobalScale, warpIdx.x,
                        /*namedBarrierId=*/2);
      asm volatile("fence.proxy.async.shared::cta;\n");
      unused(smem.vBar[idxVBuf].produced.arrive());
      TRACE_STAMP(15, idxIter, warpIdx.x == 0);
      if (idxIter + vAhead < nbIters) issueVCopies(idxIter + vAhead);
      ldgsts::commitGroup();
    }
#endif
  }
#if MIXED_KV_TRACE
  __syncthreads();
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 &&
      threadIdx.z == 0) {
    long long const t0 = smem.trace[0][8];
    for (uint32_t t = 0; t < SharedMem::nbTraceTiles && t < nbIters; t++) {
      printf("TRACE tile %u g0:kwait %lld mma %lld smax %lld xarr %lld | g1:vwait %lld xwait %lld rs %lld mma %lld | kl:start %lld iss %lld | vl:start %lld iss %lld | kc:ready %lld done %lld | vc:ready %lld done %lld\n",
             t, smem.trace[t][0] - t0, smem.trace[t][1] - t0, smem.trace[t][2] - t0,
             smem.trace[t][3] - t0, smem.trace[t][4] - t0, smem.trace[t][5] - t0,
             smem.trace[t][6] - t0, smem.trace[t][7] - t0, smem.trace[t][8] - t0,
             smem.trace[t][9] - t0, smem.trace[t][10] - t0, smem.trace[t][11] - t0,
             smem.trace[t][12] - t0, smem.trace[t][13] - t0, smem.trace[t][14] - t0,
             smem.trace[t][15] - t0);
    }
    printf("TRACE cta0 smid %u residentCtasAtTile4 %u nbIters %u nbSubSeq %u grid %u x %u x %u\n",
           smem.traceSmId, smem.traceResidentMid, nbIters, nbSubSeq, gridDim.x, gridDim.y,
           gridDim.z);
  }
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    atomicSub(&mixedKvTraceSmResident[smem.traceSmId], 1U);
  }
#endif
  __syncthreads();
  uint32_t const nbBarriers = &smem.gemm1WarpGrpBar - &smem.qBar.produced + 1;
  uint32_t const tid =
      threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
  assert(nbBarriers <= blockDim.x * blockDim.y * blockDim.z);
  if (tid < nbBarriers) {
    (&smem.qBar.produced)[tid].~CtaBarrier();
  }
  if (!isMultiBlockMode) {
    return;
  }
  bool& smemIsLastCta = smem.isLastCta;
  if (threadIdx.x == gemm1NbThrds - 1U && threadIdx.z == 0) {
    uint32_t const lastOld = nbSubSeq - 1;
    ScratchMem const scratchMem{scratch, maxNbSubSeq * nbKHeads * batchSize, nbInputSeqSplit};
    uint32_t const idxSeq = nbKHeads * idxReq + idxHeadGrp;
    uint32_t old;
    uint32_t const idxSemaphore = idxSeq * nbInputSeqSplit + idxInputSubSeq;
    auto const pSemaphore = &semaphores[idxSemaphore];
    asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;\n"
                 : "=r"(old)
                 : "l"(pSemaphore), "r"(lastOld));
    smemIsLastCta = (old == lastOld);
  }
  {
    assert(dynamicSmemSize() >= sizeof(MultiBlockSMem));
#ifndef __CUDACC_RTC__
    assert(sizeof(MultiBlockSMem) < offsetof(SharedMem, isLastCta));
#endif
    auto& smem = *reinterpret_cast<MultiBlockSMem*>(&smemByteBuf[0]);
    assert(blockDim.x >= MultiBlockSMem::nbBuf);
    constexpr uint32_t nbMathWarps = gemm0NbWarps + gemm1NbWarps;

    static_assert(nbWarps >= MultiBlockSMem::nbBuf);
    if (wid < MultiBlockSMem::nbBuf) {
      if (warpElectSync()) {
        smem.barriers[wid].initialize(isHeadPadded ? warp_size : 1U, nbMathWarps * warp_size);
        smem.barriers[wid].consumed.arrive(nbMathWarps * warp_size);
      }
    }
    __syncthreads();

    if (!smemIsLastCta) {
      return;
    }
    if (wid < nbMathWarps) {
      constexpr uint32_t headsPerWarp = divUp(ctaNbValidQHeads, nbMathWarps);
      using Acc = Vec<float, exactDiv(headElems, warp_size)>;

      struct HeadState {
        Acc acc;
        float sum;
        float max;
      };

      Vec<HeadState, headsPerWarp> states{};
      for (auto& s : states.data) {
        s.max = safeInitRowMax;
      }
      uint32_t const lane = laneId();
      for (uint32_t idxBlock = 0; idxBlock < nbSubSeq; idxBlock++) {
        uint32_t const idxBuf = idxBlock % MultiBlockSMem::nbBuf;
        auto& bar = smem.barriers[idxBuf];
        bar.produced.wait_parity(idxBlock / MultiBlockSMem::nbBuf % 2 != 0);
        for (uint32_t i = 0; i < headsPerWarp; i++) {
          uint32_t const idxHead = wid + nbMathWarps * i;
          if ((ctaNbValidQHeads % nbMathWarps != 0) && (idxHead >= ctaNbValidQHeads)) {
            break;
          }
          HeadState& state = states[i];
          auto const sumMax = smem.rowSumMax[idxBuf][idxHead];
          auto const data = convert<float>(reinterpret_cast<Vec<InputElem, Acc::size>&>(
              smem.tokens[idxBuf][idxHead][Acc::size * lane]));
          if (sumMax.max > state.max) {
            float const scale = expf(state.max - sumMax.max);
            state.max = sumMax.max;
            state.sum = state.sum * scale + sumMax.sum;
            state.acc = state.acc * scale + data * sumMax.sum;
          } else {
            float const scale = expf(sumMax.max - state.max);
            state.sum = state.sum + sumMax.sum * scale;
            state.acc = state.acc + data * (sumMax.sum * scale);
          }
        }
        unused(bar.consumed.arrive());
      }
      // Add the attention sinks.
      if (attentionSinks != nullptr) {
        for (uint32_t i = 0; i < headsPerWarp; i++) {
          uint32_t const idxHead = wid + nbMathWarps * i;
          float sink =
              expf(attentionSinks[mha::min(idxHead, headGrpSize - 1) + idxHeadGrp * headGrpSize] -
                   states[i].max);
          states[i].sum += sink;
        }
      }
      __syncthreads();
      uint32_t const outOffset =
          headGrpSize * (nbKHeads * (beamWidth * ctaInputTokBeg) + idxHeadGrp);
      auto const dst = &output[outOffset];
      for (uint32_t i = 0; i < headsPerWarp; i++) {
        uint32_t const idxHead = wid + nbMathWarps * i;
        if ((ctaNbValidQHeads % nbMathWarps != 0) && (idxHead >= ctaNbValidQHeads)) {
          break;
        }
#if SPEC_DEC
        uint32_t const idxToken = idxHead / headGrpSize;
        if (idxToken >= ctaNbValidTokens) {
          break;
        }
        uint32_t const tokenPad = headGrpSize * (nbKHeads - 1);
        uint32_t const idxDstHead = idxHead + idxToken * tokenPad;
#else
        uint32_t const idxDstHead = idxHead;
#endif
        auto const& s = states[i];
        auto const outData = convert<OutputElem>(s.acc * (1.f / s.sum));
        if (Acc::size * lane < validElemsPerHead) {
          reinterpret_cast<Vec<OutputElem, Acc::size>&>(dst[idxDstHead][Acc::size * lane]) =
              outData;
        }
      }
    } else if (wid < nbMathWarps + MultiBlockSMem::nbIOWarps) {
      static_assert(MultiBlockSMem::nbIOWarps <= MultiBlockSMem::nbBuf);
      ScratchMem const scratchMem{scratch, maxNbSubSeq * nbKHeads * batchSize, nbInputSeqSplit};
      uint32_t const idxSeq = nbKHeads * idxReq + idxHeadGrp;
      uint32_t const initIdxBlock = wid - nbMathWarps;
      // each warp loads data for a block
      for (uint32_t idxBlock = initIdxBlock; idxBlock < nbSubSeq;
           idxBlock += MultiBlockSMem::nbIOWarps) {
        uint32_t const idxAllSubSeq = maxNbSubSeq * idxSeq + idxBlock;
        uint32_t const idxChunk = idxAllSubSeq * nbInputSeqSplit + idxInputSubSeq;
        uint32_t const idxBuf = idxBlock % MultiBlockSMem::nbBuf;
        auto& bar = smem.barriers[idxBuf];
        bar.consumed.wait_parity(idxBlock / MultiBlockSMem::nbBuf % 2 != 0);
        auto const lane = laneId();
#pragma unroll
        for (uint32_t iter = 0; iter < divUp(ctaNbValidQHeads, warp_size); iter++) {
          uint32_t const i = iter * warp_size + lane;
          if (ctaNbValidQHeads % warp_size != 0 && i >= ctaNbValidQHeads) {
            break;
          }
          ldgsts::copyAsync<sizeof(smem.rowSumMax[idxBuf][i])>(
              &smem.rowSumMax[idxBuf][i], &scratchMem.rowSumMax()[idxChunk][i]);
        }
        ldgsts::barArrive(bar.produced, false);
        if constexpr (isHeadPadded) {
          static_assert(grainsPerPaddedInputHead <= warp_size);
          constexpr uint32_t headsPerIter = exactDiv(warp_size, grainsPerPaddedInputHead);
          constexpr uint32_t nbIters = divUp(ctaNbValidQHeads, headsPerIter);
          constexpr uint32_t nbWholeIters = ctaNbValidQHeads / headsPerIter;
#pragma unroll
          for (uint32_t i = 0; i < nbIters; i++) {
            uint32_t const idxHead =
                headsPerIter * i +
                BoundedVal<warp_size>{lane}.template divBy<grainsPerPaddedInputHead>().get();
            uint32_t const idxGrain =
                BoundedVal<warp_size>{lane}.template mod<grainsPerPaddedInputHead>().get();
            if (i < nbWholeIters || idxHead < ctaNbValidQHeads) {
              constexpr uint32_t nbElemsPerGrain =
                  exactDiv(grainBytes, sizeof(MultiBlockSMem::Elem));
              auto const dst = &smem.tokens[idxBuf][idxHead][nbElemsPerGrain * idxGrain];
              auto const src =
                  idxGrain < grainsPerIOHead
                      ? &scratchMem.tokens()[idxChunk][idxHead][nbElemsPerGrain * idxGrain]
                      : nullptr;
              ldgsts::copyAsync<grainBytes>(dst, src, idxGrain < grainsPerIOHead ? grainBytes : 0U);
            }
          }
          ldgsts::barArrive(bar.produced, true);
        } else {
          if (warpElectSync()) {
            tma::loadLinearAsync(&smem.tokens[idxBuf], &scratchMem.tokens()[idxChunk],
                                 sizeof(smem.tokens[idxBuf]), bar.produced);
            arrive_tx(bar.produced, sizeof(smem.tokens[idxBuf]), 1);
          }
        }
      }
      __syncthreads();
      uint32_t const idxBar = tid - warp_size * nbMathWarps;
      if (idxBar < MultiBlockSMem::nbBuf * 2) {
        reinterpret_cast<CtaBarrier*>(&smem.barriers[0])[idxBar].~CtaBarrier();
      }
    }
  }
#else
#if GENERATE_CUBIN
  static_assert("This kernel is for Hopper only");
#else
  asm volatile("trap;\n");
#endif
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && BEAM_WIDTH == 1
}

#if CACHE_ELEM_ENUM == 0 || CACHE_ELEM_ENUM == 2 || CACHE_ELEM_ENUM == 5
template <uint32_t nbThrds, uint32_t beamWidth>
__device__ inline typename F16QToF8Converter<nbThrds, beamWidth>::RegData
F16QToF8Converter<nbThrds, beamWidth>::load(uint32_t tid, TinyPtr<IOHead const> const& src,
                                            uint32_t const nbKHeads /*for beam search only*/,
                                            uint32_t nbTokens) {
#if !(SPEC_DEC)
  assert(nbTokens == 1);
  nbTokens = 1;
#endif
  typename F16QToF8Converter<nbThrds, beamWidth>::RegData dst;
#pragma unroll
  for (uint32_t iter = 0; iter < nbIters; iter++) {
    uint32_t const idxGrain = nbThrds * iter + tid;
    if (idxGrain >= totalGrains) {
      break;
    }
#if SPEC_DEC
    uint32_t const idxToken = idxGrain / grainsPerPaddedInputQHeadGrp;
    uint32_t const tokenPad = grainsPerPaddedInputQHeadGrp * (nbKHeads - 1);
    uint32_t offsetInGrains = idxGrain + tokenPad * idxToken;
    static_assert(beamWidth == 1);
#else
    uint32_t const idxBeam = beamWidth == 1 ? 0 : idxGrain / grainsPerPaddedInputQHeadGrp;
    uint32_t const beamPad = grainsPerPaddedInputQHeadGrp * (nbKHeads - 1);
    uint32_t offsetInGrains = idxGrain + beamPad * idxBeam;
#endif
    bool isGrainInBound = true;
    if constexpr (isHeadPadded) {
      uint32_t const idxGrainInsideHead = offsetInGrains % grainsPerPaddedInputHead;
      offsetInGrains =
          offsetInGrains / grainsPerPaddedInputHead * grainsPerIOHead + idxGrainInsideHead;
      isGrainInBound = (idxGrainInsideHead < grainsPerIOHead);
    }
#if SPEC_DEC
    isGrainInBound = isGrainInBound && (idxToken < nbTokens);
#endif
    LdGrain const srcGrain =
        isGrainInBound ? src.template cast<LdGrain const>()[offsetInGrains] : LdGrain{};
    static_assert(inputElemSize == 2);
    auto const& fp16Data =
        reinterpret_cast<Vec<InputElem, exactDiv(grainBytes, inputElemSize)> const&>(srcGrain);
    dst[iter] = idxGrain % grainsPerPaddedInputHead < grainsPerIOHead
                    ? fp16Data
                    : mha::decay_t<decltype(fp16Data)>{};
  }
  return dst;
}

template <uint32_t nbThrds, uint32_t beamWidth>
__device__ inline void F16QToF8Converter<nbThrds, beamWidth>::store(
    uint32_t tid, SharedMem::QBuffer& dst,
    F16QToF8Converter<nbThrds, beamWidth>::RegData const& data) {
#pragma unroll
  for (uint32_t iter = 0; iter < nbIters; iter++) {
    uint32_t const idxGrain = nbThrds * iter + tid;
    if (idxGrain >= totalGrains) {
      break;
    }
#if CACHE_ELEM_ENUM == 0 || CACHE_ELEM_ENUM == 5
    static_assert(inputElemSize == cacheElemSize);
    ShmVec const& shmData = data[iter];
    uint32_t const r = idxGrain / grainsPerPaddedInputHead;
    BoundedVal<grainsPerPaddedInputHead> const c = {idxGrain % grainsPerPaddedInputHead};

    dst[c.template divBy<grainsPerQPart>().get()].template at<true>(
        r, c.template mod<grainsPerQPart>().get()) = reinterpret_cast<LdGrain const&>(shmData);
#else
    auto const& fp16Data = data[iter];
    ShmVec shmData;
#pragma unroll
    for (uint32_t i = 0; i < fp16Data.size; i++) {
      shmData[i] = CacheElem{fp16Data[i]};
    }
    uint32_t const dstIdxGrain = idxGrain / 2;
    uint32_t const dstIdxHalfGrain = idxGrain % 2;
    constexpr uint32_t grainsPerCacheHead = exactDiv(paddedCacheHeadBytes, grainBytes);
    uint32_t const r = dstIdxGrain / grainsPerCacheHead;
    BoundedVal<grainsPerCacheHead> const c = {dstIdxGrain % grainsPerCacheHead};
    reinterpret_cast<Vec<ShmVec, 2>&>(
        dst[c.template divBy<grainsPerQPart>().get()].template at<true>(
            r, c.template mod<grainsPerQPart>().get()))[dstIdxHalfGrain] = shmData;
#endif
  }
}
#endif

__device__ inline KVTilePartLoader::KVTilePartLoader(bool isK, uint32_t nbKHeads,
                                                     KVCacheList<usePagedKVCache> const& cacheList,
                                                     uint32_t idxReq, uint32_t idxHeadGrp,
                                                     CUtensorMap const& tensorMap, uint32_t nbPages,
                                                     Vec<KVCachePageIndex, nbPagesPerTile>& pageBuf
#if ENABLE_MIXED_KV_CACHE
                                                     , MixedPageFormats<nbPagesPerTile>& formatBuf
#endif
                                                     )
    : nbKHeads{nbKHeads},
      isK{isK},
      cacheList{cacheList},
      idxReq{idxReq},
      idxHeadGrp{idxHeadGrp},
      tensorMap{tensorMap},
#if ENABLE_MIXED_KV_CACHE
      formats{formatBuf},
#endif
      nbPages{nbPages},
      pages{pageBuf},
      baseOffset{idxReq * cacheList.maxNbPagesPerSeq} {}

// tensorMap is for one whole page ([nbKHeads*tokensPerPage][headElems]) or whole cache
template <uint32_t nbTokens, bool alignedForSwizzle>
__device__ inline void KVTilePartLoader::loadData(
    Array2D<LdGrain, nbTokens, exactDiv(cacheHeadPartBytes, grainBytes), alignedForSwizzle>& dst,
    uint32_t idxTile, uint32_t idxPart, CtaBarrier& bar) {
  static_assert(nbTokens == gemm0CtaTileNbTokens);
  assert(idxTile == idxTileRef);
  if constexpr (nbTokens < tokensPerPage) {
    assert(nbPagesPerTile == 1);
    uint32_t const offset = nbTokens * (idxTile % exactDiv(tokensPerPage, nbTokens));
    tma::loadAsync(&dst, tensorMap,
                   DimsLE<4>{partElems * idxPart, idxHeadGrp, offset, (uint32_t)pages[0]}, bar);
  } else {
#pragma unroll
    for (uint32_t i = 0; i < nbPagesPerTile; i++) {
      tma::loadAsync(&dst(tokensPerPage * i, 0), tensorMap,
                     DimsLE<4>{partElems * idxPart, idxHeadGrp, 0, (uint32_t)pages[i]}, bar);
    }
  }
}

#if ENABLE_MIXED_KV_CACHE
// Format tag the loader acts on for tile page slot i.  Slots past the end of
// the sequence are tagged kMixedBadPageFormat in shared memory by loadPages so
// the converters can zero-fill them; under a static format every real page is
// that format and page_format is not consulted.
// Converter-side read of a tile's page list and tags from the loader-filled
// metadata chunks (same layout as KVTilePartLoader::readTileMeta).
__device__ __forceinline__ KVTilePartLoader::PagePrefetch readMixedTileMeta(
    SharedMem const& smem, uint32_t operand, uint32_t idxIter) {
  uint32_t const chunk = (idxIter / SharedMem::metaChunkTiles) % SharedMem::nbMetaChunks;
  uint32_t const i = idxIter % SharedMem::metaChunkTiles;
  KVTilePartLoader::PagePrefetch p;
#pragma unroll
  for (uint32_t j = 0; j < SharedMem::nbPagesPerTile; j++) {
    p.pages[j] = smem.metaPages[operand][chunk][i][j];
    p.formats[j] = smem.metaFormats[operand][chunk][i][j];
  }
  return p;
}

// Converter warp `idxWarp` copies page idxWarp of a tile if it is compressed:
// the packed rows into the page's slot of the stage's last head-part buffer
// (dense rows, TMA-swizzled chunk positions so the expansion is unchanged:
// 128 B rows chunk ^= row % 8, 64 B rows chunk ^= (row / 2) % 4) and the
// token's 8 B of block scales.  Warp-contiguous ownership: consecutive lanes
// own consecutive 16 B chunks of a row (D6).  Per-lane row/chunk are constants;
// per tile the only variable is the page (one multiply-add).  One cp.async
// group per tile is committed by the caller.  Issue budget (C6): ~30
// instructions per lane per tile.
__device__ __forceinline__ void issueCompressedPageCopies(
    KVCacheList<usePagedKVCache> const& cacheList, bool isK, uint32_t idxHeadGrp,
    SharedMem const& smem, uint32_t operand, uint32_t idxIter, SharedMem::PackedTile& dstPacked,
    SharedMem::TileScales& dstScales, uint32_t idxWarp) {
  using flashinfer::KVPageFormat;
  static_assert(SharedMem::nbPagesPerTile == convertWarpsPerOperand, "one converter warp per page");
  static_assert(tokensPerPage == 16 && SharedMem::packedRowBytesFP8 == 128 &&
                SharedMem::packedRowBytesFP4 == 64 && SharedMem::scaleBytesPerToken == 8);
  if constexpr (MIXED_KV_EXPERIMENT & 2) {
    return;
  }
  // This warp's page and tag only.
  uint32_t const chunk = (idxIter / SharedMem::metaChunkTiles) % SharedMem::nbMetaChunks;
  uint32_t const entry = idxIter % SharedMem::metaChunkTiles;
  uint8_t const tagged = smem.metaFormats[operand][chunk][entry][idxWarp];
#if MIXED_PAGE_STATIC_FORMAT >= 0
  uint8_t const format =
      tagged == kMixedBadPageFormat ? tagged : static_cast<uint8_t>(MIXED_PAGE_STATIC_FORMAT);
#else
  uint8_t const format = tagged;
#endif
  if (format == kMixedBadPageFormat || format == static_cast<uint8_t>(KVPageFormat::kA16)) {
    return;
  }
  uint32_t const page = static_cast<uint32_t>(smem.metaPages[operand][chunk][entry][idxWarp]);
  bool const isFP8 = format == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8);
  auto const& span = cacheList.transport.formats[format];
  uint32_t const lane = laneId();
  uint8_t const* const payloadPage =
      static_cast<uint8_t const*>(isK ? span.k_payload : span.v_payload) +
      uint64_t(page) * span.payload_stride.page + uint64_t(idxHeadGrp) * span.payload_stride.head;
  uint8_t* const slot = &dstPacked[idxWarp][0];
  if (isFP8) {
    // 16 rows x 8 chunks: lane owns chunk lane % 8 of rows lane / 8 + 4k.
    uint32_t const c = lane % 8;
    uint32_t const r0 = lane / 8;
    uint8_t const* src = payloadPage + uint64_t(r0) * span.payload_stride.token + c * 16;
    uint64_t const rowStep = 4ull * span.payload_stride.token;
#pragma unroll
    for (uint32_t k = 0; k < 4; k++) {
      uint32_t const r = r0 + 4 * k;  // r % 8 == r0 % 8 + 4 * (k % 2): compile-time per k given r0
      ldgsts::copyAsync<16>(slot + r * 128 + ((c ^ (r % 8)) * 16), src, 16);
      src += rowStep;
    }
  } else {
    // 16 rows x 4 chunks: lane owns chunk lane % 4 of rows lane / 4 + 8k.
    uint32_t const c = lane % 4;
    uint32_t const r0 = lane / 4;
    uint8_t const* src = payloadPage + uint64_t(r0) * span.payload_stride.token + c * 16;
    uint64_t const rowStep = 8ull * span.payload_stride.token;
#pragma unroll
    for (uint32_t k = 0; k < 2; k++) {
      uint32_t const r = r0 + 8 * k;  // (r / 2) % 4 == (r0 / 2) % 4
      ldgsts::copyAsync<16>(slot + r * 64 + ((c ^ ((r / 2) % 4)) * 16), src, 16);
      src += rowStep;
    }
  }
  if constexpr (!(MIXED_KV_EXPERIMENT & 4)) {
    if (lane < tokensPerPage) {
      uint8_t const* const src = (isK ? span.k_scales : span.v_scales) +
                                 uint64_t(page) * span.scale_stride.page +
                                 uint64_t(lane) * span.scale_stride.token +
                                 uint64_t(idxHeadGrp) * span.scale_stride.head;
      ldgsts::copyAsync<8>(&dstScales[idxWarp * tokensPerPage + lane][0], src, 8);
    }
  }
}

__device__ __forceinline__ uint8_t mixedTilePageFormat(
    MixedPageFormats<SharedMem::nbPagesPerTile> const& formats, uint32_t idxPage) {
  uint8_t const tagged = formats.values[idxPage];
#if MIXED_PAGE_STATIC_FORMAT >= 0
  return tagged == kMixedBadPageFormat ? tagged : static_cast<uint8_t>(MIXED_PAGE_STATIC_FORMAT);
#else
  return tagged;
#endif
}

__device__ __forceinline__ uint32_t KVTilePartLoader::packedPartTxBytes(uint32_t idxPart) const {
  using flashinfer::KVPageFormat;
  uint32_t bytes = 0;
#pragma unroll
  for (uint32_t i = 0; i < nbPagesPerTile; i++) {
    uint8_t const format = mixedTilePageFormat(formats, i);
    if (format == kMixedBadPageFormat) {
      continue;
    }
    if (format == static_cast<uint8_t>(KVPageFormat::kA16)) {
      bytes += tokensPerPage * cacheHeadPartBytes;
    }
  }
  (void)idxPart;
  return bytes;
}

template <bool alignedForSwizzle>
__device__ __forceinline__ void KVTilePartLoader::issuePackedPartLoad(
    Array2D<LdGrain, gemm0CtaTileNbTokens, exactDiv(cacheHeadPartBytes, grainBytes),
            alignedForSwizzle>& dstA16,
    SharedMem::PackedTile& dstPacked, uint32_t idxPart, CUtensorMap const& tensorMapFP8,
    CUtensorMap const& tensorMapFP4, CtaBarrier& bar) const {
  using flashinfer::KVPageFormat;
  static_assert(gemm0CtaTileNbTokens == nbPagesPerTile * tokensPerPage);
#pragma unroll
  for (uint32_t i = 0; i < nbPagesPerTile; i++) {
    uint8_t const format = mixedTilePageFormat(formats, i);
    if (format == kMixedBadPageFormat) {
      continue;
    }
    uint32_t const page = static_cast<uint32_t>(pages[i]);
    if (format == static_cast<uint8_t>(KVPageFormat::kA16)) {
      tma::loadAsync(&dstA16(tokensPerPage * i, 0), tensorMap,
                     DimsLE<4>{partElems * idxPart, idxHeadGrp, 0, page}, bar);
    }
    // Compressed pages: copied by the converter warps with cp.async (a 16-token
    // page is too small a TMA box - ~100-200 ns of issue each, measured - and
    // the converters own the bytes anyway).
  }
  (void)dstPacked;
  (void)tensorMapFP8;
  (void)tensorMapFP4;
}

template <uint32_t nbLoadWarps>
__device__ __forceinline__ void KVTilePartLoader::loadPackedScales(SharedMem::TileScales& dstScales,
                                                                    uint32_t idxLoadWarp) const {
  PagePrefetch current;
#pragma unroll
  for (uint32_t i = 0; i < nbPagesPerTile; i++) {
    current.pages[i] = pages[i];
    current.formats[i] = formats.values[i];
  }
  loadPackedScalesFrom<nbLoadWarps>(current, dstScales, idxLoadWarp);
}

template <uint32_t nbLoadWarps>
__device__ __forceinline__ void KVTilePartLoader::loadPackedScalesFrom(
    PagePrefetch const& prefetched, SharedMem::TileScales& dstScales, uint32_t idxLoadWarp) const {
  using flashinfer::KVPageFormat;
  constexpr uint32_t nbThreads = nbLoadWarps * warp_size;
  constexpr uint32_t scaleBytes = SharedMem::scaleBytesPerToken;
  static_assert(scaleBytes % 4 == 0, "whole-head block scales are loaded as 32-bit words");
  constexpr uint32_t nbWords = scaleBytes / 4;
  (void)nbWords;
  uint32_t const tid = idxLoadWarp * warp_size + laneId();
  if constexpr (MIXED_KV_EXPERIMENT & 4) {
    return;
  }
#pragma unroll
  for (uint32_t token = tid; token < gemm0CtaTileNbTokens; token += nbThreads) {
    uint32_t const idxPage = token / tokensPerPage;
    uint8_t const tagged = prefetched.formats[idxPage];
#if MIXED_PAGE_STATIC_FORMAT >= 0
    uint8_t const format =
        tagged == kMixedBadPageFormat ? tagged : static_cast<uint8_t>(MIXED_PAGE_STATIC_FORMAT);
#else
    uint8_t const format = tagged;
#endif
    if (format == kMixedBadPageFormat || format == static_cast<uint8_t>(KVPageFormat::kA16)) {
      continue;
    }
    auto const& span = cacheList.transport.formats[format];
    uint8_t const* const src =
        (isK ? span.k_scales : span.v_scales) +
        uint64_t(static_cast<uint32_t>(prefetched.pages[idxPage])) * span.scale_stride.page +
        uint64_t(token % tokensPerPage) * span.scale_stride.token +
        uint64_t(idxHeadGrp) * span.scale_stride.head;
    if constexpr (scaleBytes % 8 == 0) {
#pragma unroll
      for (uint32_t b = 0; b < scaleBytes; b += 8) {
        ldgsts::copyAsync<8>(&dstScales[token][b], src + b, 8);
      }
    } else {
#pragma unroll
      for (uint32_t w = 0; w < nbWords; w++) {
        ldgsts::copyAsync<4>(&dstScales[token][4 * w], src + 4 * w, 4);
      }
    }
  }
}

// Expand every compressed page of a whole K/V stage in place.  The packed
// rows live in the stage's last head-part buffer; A16 pages of the same tile
// were placed by TMA in their final rows and are skipped; slots past the
// sequence end are zero-filled so a masked P=0 never multiplies stale memory.
//
// The converter group is instruction-issue-bound (two CTAs share the SM's
// four schedulers with the GMMA groups), so the cut minimizes instructions per
// value: lane l owns one (token, head part) = 64 values = 4 blocks, paying for
// its page tag, its 4-byte scale word and its row address once.  Lanes of a
// warp cover 16 tokens x 2 parts of one page, so the format branch is
// warp-uniform; with the TMA swizzle, 32 lanes reading 16 B each touch 8
// distinct bank groups four times (optimal), and the 16 B swizzled stores
// likewise.  Because the last part's A16 rows overwrite the packed rows,
// every read precedes a named barrier and every write follows it.
template <typename PartBuf>
__device__ __forceinline__ void expandPackedStage(
    PartBuf* parts, SharedMem::TileScales const& scales,
    MixedPageFormats<SharedMem::nbPagesPerTile> const& formats, float fp8GlobalScale,
    float fp4GlobalScale, uint32_t idxWarp, uint32_t namedBarrierId) {
  using flashinfer::KVPageFormat;
  constexpr uint32_t blocksPerPart = exactDiv(cacheHeadPartElems, 16);
  constexpr uint32_t nbThreads = convertWarpsPerOperand * warp_size;
  static_assert(gemm0CtaTileNbTokens * cacheHeadNbParts == nbThreads,
                "one converter lane per (token, head part)");
  static_assert(blocksPerPart == 4 && SharedMem::packedRowBytesFP8 == 128 &&
                    SharedMem::packedRowBytesFP4 == 64,
                "swizzle decode assumes D=128 whole-head rows and 64-element parts");
  static_assert((tokensPerPage * cacheHeadNbParts) % warp_size == 0,
                "a warp must not straddle two pages");
  auto const& packed = *reinterpret_cast<SharedMem::PackedTile const*>(&parts[cacheHeadNbParts - 1]);
  uint32_t const tid = idxWarp * warp_size + laneId();
  uint32_t const token = tid / cacheHeadNbParts;
  uint32_t const p = tid % cacheHeadNbParts;
  uint32_t const idxPage = token / tokensPerPage;
  uint32_t const tokenInPage = token % tokensPerPage;
  uint8_t const format = mixedTilePageFormat(formats, idxPage);
  bool const isFP8 = format == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8);
  bool const isFP4 = format == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4);

  uint32_t scaleWord = 0;
  LdGrain words[blocksPerPart];
  if constexpr (!(MIXED_KV_EXPERIMENT & 1)) {
    if (isFP8 || isFP4) {
      scaleWord = *reinterpret_cast<uint32_t const*>(&scales[token][p * blocksPerPart]);
    }
    if (isFP8) {
      // 128B swizzle: 16 B chunk index XOR address bits [7,10) = row % 8.
      uint8_t const* row = &packed[idxPage][tokenInPage * SharedMem::packedRowBytesFP8];
#pragma unroll
      for (uint32_t b = 0; b < blocksPerPart; b++) {
        uint32_t const chunk = (p * blocksPerPart + b) ^ (tokenInPage % 8);
        words[b] = *reinterpret_cast<LdGrain const*>(row + chunk * 16);
      }
    } else if (isFP4) {
      // 64B swizzle: 16 B chunk index XOR address bits [7,9) = (row / 2) % 4.
      // Chunk c of a 64 B row holds blocks 2c and 2c+1 (8 B each).
      uint8_t const* row = &packed[idxPage][tokenInPage * SharedMem::packedRowBytesFP4];
#pragma unroll
      for (uint32_t c = 0; c < blocksPerPart / 2; c++) {
        uint32_t const chunk = (p * (blocksPerPart / 2) + c) ^ ((tokenInPage / 2) % 4);
        words[c] = *reinterpret_cast<LdGrain const*>(row + chunk * 16);
      }
    }
  }
  // Every packed read precedes every in-place write.  A warp covers exactly
  // one page's (token, part) lanes (static_assert above), so the hazard is
  // intra-warp and a warp sync suffices.
  (void)namedBarrierId;
  (void)nbThreads;
  __syncwarp();
  if constexpr (MIXED_KV_EXPERIMENT & 1) {
    return;
  }
  if (format == static_cast<uint8_t>(KVPageFormat::kA16)) {
    return;
  }
  // All four block scales of this (token, part) in one conversion.
  Vec<uint16_t, 4> const a16Scales = convertE4M3x4ScalesToA16Bits<InputElem>(
      scaleWord, isFP8 ? fp8GlobalScale : fp4GlobalScale);
#pragma unroll
  for (uint32_t b = 0; b < blocksPerPart; b++) {
    uint32_t const sf2 = broadcastA16Scale<InputElem>(a16Scales[b]);
    LdGrain first{};
    LdGrain second{};
    if (isFP8) {
      first = words[b];
      expandCompressedBlock16WithScale<KVPageFormat::kBlockScaledFP8, InputElem>(sf2, first,
                                                                                 second);
    } else if (isFP4) {
      first[0] = words[b / 2][(b % 2) * 2];
      first[1] = words[b / 2][(b % 2) * 2 + 1];
      expandCompressedBlock16WithScale<KVPageFormat::kBlockScaledFP4, InputElem>(sf2, first,
                                                                                 second);
    } else {
      assert(format == kMixedBadPageFormat);
    }
    parts[p].template at<true>(token, b * 2) = first;
    parts[p].template at<true>(token, b * 2 + 1) = second;
  }
  __syncwarp();
}
#endif

__device__ inline void KVTilePartLoader::loadPages(uint32_t idxTile,
                                                   bool publish) {
  uint32_t const idxPageBeg = gemm0CtaTileNbTokens >= tokensPerPage
                                  ? nbPagesPerTile * idxTile
                                  : idxTile / exactDiv(tokensPerPage, gemm0CtaTileNbTokens);
#pragma unroll
  for (uint32_t i = 0; i < nbPagesPerTile; i++) {
    uint32_t const idxPage = idxPageBeg + i;
    auto const page =
        idxPage < nbPages ? cacheList.kvCachePageList[baseOffset + idxPage] : kBAD_PAGE_INDEX;
    if (publish && warpElectSync()) {
      pages[i] = page;
#if ENABLE_MIXED_KV_CACHE
      formats.values[i] = page == kBAD_PAGE_INDEX ? kMixedBadPageFormat
                                                  : cacheList.transport.page_format[page];
#endif
    }
  }
  idxTileRef = idxTile;
  __syncwarp();
}

#if ENABLE_MIXED_KV_CACHE
__device__ __forceinline__ void KVTilePartLoader::fillTileMeta(SharedMem& smem, uint32_t operand,
                                                               uint32_t idxIterBeg,
                                                               uint32_t nbIters,
                                                               uint32_t idxTileInit,
                                                               uint32_t nbSubSeq) const {
  constexpr uint32_t nbEntries = SharedMem::metaChunkTiles * nbPagesPerTile;
  static_assert(nbEntries % warp_size == 0);
  uint32_t const lane = laneId();
#pragma unroll
  for (uint32_t e = lane; e < nbEntries; e += warp_size) {
    uint32_t const i = e / nbPagesPerTile;
    uint32_t const j = e % nbPagesPerTile;
    uint32_t const idxIter = idxIterBeg + i;
    KVCachePageIndex page = kBAD_PAGE_INDEX;
    if (idxIter < nbIters) {
      uint32_t const idxTile = idxTileInit + idxIter * nbSubSeq;
      uint32_t const idxPageBeg = gemm0CtaTileNbTokens >= tokensPerPage
                                      ? nbPagesPerTile * idxTile
                                      : idxTile / exactDiv(tokensPerPage, gemm0CtaTileNbTokens);
      uint32_t const idxPage = idxPageBeg + j;
      page = idxPage < nbPages ? cacheList.kvCachePageList[baseOffset + idxPage] : kBAD_PAGE_INDEX;
    }
    uint32_t const chunk = (idxIterBeg / SharedMem::metaChunkTiles) % SharedMem::nbMetaChunks;
    smem.metaPages[operand][chunk][i][j] = page;
    smem.metaFormats[operand][chunk][i][j] = page == kBAD_PAGE_INDEX
                                                 ? kMixedBadPageFormat
                                                 : cacheList.transport.page_format[page];
  }
  __syncwarp();
}

__device__ __forceinline__ KVTilePartLoader::PagePrefetch KVTilePartLoader::readTileMeta(
    SharedMem const& smem, uint32_t operand, uint32_t idxIter) const {
  uint32_t const chunk = (idxIter / SharedMem::metaChunkTiles) % SharedMem::nbMetaChunks;
  uint32_t const i = idxIter % SharedMem::metaChunkTiles;
  PagePrefetch p;
#pragma unroll
  for (uint32_t j = 0; j < nbPagesPerTile; j++) {
    p.pages[j] = smem.metaPages[operand][chunk][i][j];
    p.formats[j] = smem.metaFormats[operand][chunk][i][j];
  }
  return p;
}

__device__ __forceinline__ void KVTilePartLoader::publishPages(PagePrefetch const& prefetched,
                                                               uint32_t idxTile) {
  if (warpElectSync()) {
#pragma unroll
    for (uint32_t i = 0; i < nbPagesPerTile; i++) {
      pages[i] = prefetched.pages[i];
      formats.values[i] = prefetched.formats[i];
    }
  }
  idxTileRef = idxTile;
  __syncwarp();
}
#endif

__device__ inline GMemKVCacheHead& KVTilePartLoader::getHead(uint32_t pos) {
  constexpr uint32_t nbTokens = gemm0CtaTileNbTokens;
  // Raise a runtime error indicating not implemented
  assert(false && "KVTilePartLoader::getHead is not implemented");
  __trap();
}

#if SWAP_AB
#if SPEC_DEC
__device__ inline void warpGrpApplyMask(Gemm0Acc& acc, SpecDec const& specDec,
#if SLIDING_WINDOW && !IS_SPEC_DEC_TREE
                                        int32_t tok0WinBeg,
#endif
                                        uint32_t cacheSeqLen, uint32_t idxTile, uint32_t warpRank) {
  constexpr uint32_t tileSize = gemm0CtaTileNbTokens;
  static_assert(SPEC_Q_SEQ_LEN <= sizeof(MaskType) * 8, "not implemented");

  assert(cacheSeqLen >= SPEC_Q_SEQ_LEN);
  uint32_t const maskStartRow = cacheSeqLen - SPEC_Q_SEQ_LEN;
  uint32_t const tileStartRow = tileSize * idxTile;
  if (tileStartRow + tileSize < maskStartRow) {
    return;
  }

  uint32_t const idxInQuad = laneId() % 4;
  uint32_t const idxQuad = laneId() / 4;

#pragma unroll
  for (uint32_t n = 0; n < acc.cols; n++) {
#pragma unroll
    for (uint32_t j = 0; j < GmmaAccCoreMat::cols; j++) {
      uint32_t const col = GmmaAccCoreMat::cols * (4 * n + idxInQuad) + j;
      uint32_t const maskCol = col / headGrpSize;
      MaskType const bit_mask = (1ULL << (maskCol + 1)) - 1;

#pragma unroll
      for (uint32_t m = 0; m < acc.rows; m++) {
#pragma unroll
        for (uint32_t i = 0; i < GmmaAccCoreMat::rows; i++) {
          uint32_t const row = gmma::instM * m + gmma::instM / 4 * warpRank + 8 * i + idxQuad;
          uint32_t const globalRow = tileStartRow + row;
          if (globalRow >= cacheSeqLen) {
            acc(m, n)(i, j) = safeInitRowMax;
            continue;
          }
          if (globalRow >= maskStartRow) {
            uint32_t const maskRow = globalRow - maskStartRow;
            if ((bit_mask >> maskRow) == 0) {
              acc(m, n)(i, j) = safeInitRowMax;
            }
          }
        }
      }
    }
  }
}
#endif  // SPEC_DEC

// Tile-local column max reduced across the warp group and folded into the
// register-resident running max (see SharedMem::gemm0WarpColMax).  Returns the
// running max including this tile; bit-identical to the former shared-memory
// atomicMax chain (fmax over the same set of values in any order).
__device__ inline RegColWiseVec computeWarpGrpColMax_sync(
    uint32_t warpRank, ShmQWiseVec (&warpColMaxSlots)[gemm0NbWarps],
    RegColWiseVec& runningColMax, Gemm0Acc const& src) {
  auto colMax = RegColWiseVec::filled(Vec<float, 2>::filled(safeInitRowMax));
#pragma unroll
  for (uint32_t n = 0; n < src.cols; n++) {
    for (uint32_t j = 0; j < GmmaAccCoreMat::cols; j++) {
#pragma unroll
      for (uint32_t m = 0; m < src.rows; m++) {
#pragma unroll
        for (uint32_t i = 0; i < GmmaAccCoreMat::rows; i++) {
          colMax[n][j] = (m == 0 && i == 0) ? src(m, n)(i, j) : fmax(colMax[n][j], src(m, n)(i, j));
        }
      }
    }
  }

#pragma unroll
  for (uint32_t xorMask = 16; xorMask > 2; xorMask /= 2) {
#pragma unroll
    for (uint32_t n = 0; n < src.cols; n++) {
#pragma unroll
      for (uint32_t j = 0; j < 2; j++) {
        auto& x = colMax[n][j];
        x = fmax(x, __shfl_xor_sync(~0U, x, xorMask));
      }
    }
  }

  uint32_t const lane = laneId();
  if (lane < 4) {
#pragma unroll
    for (uint32_t n = 0; n < src.cols; n++) {
#pragma unroll
      for (uint32_t j = 0; j < 2; j++) {
        warpColMaxSlots[warpRank][8 * n + 2 * lane + j] = colMax[n][j];
      }
    }
  }
  gemm0WarpGrpSync();
  uint32_t const idxInQuad = lane % 4;

#pragma unroll
  for (uint32_t n = 0; n < src.cols; n++) {
#pragma unroll
    for (uint32_t j = 0; j < GmmaAccCoreMat::cols; j++) {
      float m = runningColMax[n][j];
#pragma unroll
      for (uint32_t w = 0; w < gemm0NbWarps; w++) {
        m = fmax(m, warpColMaxSlots[w][8 * n + 2 * idxInQuad + j]);
      }
      assert(colMax[n][j] <= m);
      runningColMax[n][j] = m;
    }
  }
  return runningColMax;
}

__device__ inline RegColWiseVec loadShmColWiseVecWithDup(ShmQWiseVec const& smemVec) {
  RegColWiseVec ret;
  constexpr uint32_t nbThrdsPerInstNBase = exactDiv(gmma::instNBase, GmmaAccCoreMat::cols);
  auto const idx = laneId() % nbThrdsPerInstNBase;
#pragma unroll
  for (uint32_t i = 0; i < exactDiv(ShmQWiseVec::size, gmma::instNBase); i++) {
    static_assert(nbThrdsPerInstNBase * RegColWiseVec::size ==
                  exactDiv(ShmQWiseVec::size, GmmaAccCoreMat::cols));
    ret[i] = reinterpret_cast<Vec<Vec<float, GmmaAccCoreMat::cols>,
                                  exactDiv(ShmQWiseVec::size, GmmaAccCoreMat::cols)> const&>(
        smemVec)[i * nbThrdsPerInstNBase + idx];
  }
  return ret;
}

__device__ inline RegColWiseVec loadGmemColWiseVecWithDup(ShmQWiseVec const& gmemVec,
                                                          uint32_t bound) {
  RegColWiseVec ret;
  constexpr uint32_t nbThrdsPerInstNBase = exactDiv(gmma::instNBase, GmmaAccCoreMat::cols);
  auto const idx = laneId() % nbThrdsPerInstNBase;
#pragma unroll
  for (uint32_t i = 0; i < exactDiv(ShmQWiseVec::size, gmma::instNBase); i++) {
    static_assert(nbThrdsPerInstNBase * RegColWiseVec::size ==
                  exactDiv(ShmQWiseVec::size, GmmaAccCoreMat::cols));
    uint32_t const clampedIdx = mha::min(i * nbThrdsPerInstNBase + idx, bound);
    uint32_t const baseOffset = clampedIdx * GmmaAccCoreMat::cols;
#pragma unroll
    for (uint32_t j = 0; j < GmmaAccCoreMat::cols; j++) {
      ret[i][j] = gmemVec[baseOffset + j];
    }
  }
  return ret;
}

__device__ inline void warpGrpApplyMask(uint32_t warpRank, Gemm0Acc& acc, uint32_t validRowBeg,
                                        uint32_t validRowEnd) {
  uint32_t const idxInQuad = laneId() % 4;
  uint32_t const idxQuad = laneId() / 4;
#pragma unroll
  for (uint32_t m = 0; m < acc.rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < GmmaAccCoreMat::rows; i++) {
      uint32_t const row = 64 * m + 16 * warpRank + 8 * i + idxQuad;
      if (row >= validRowBeg && row < validRowEnd) {
        continue;
      }
#pragma unroll
      for (uint32_t n = 0; n < acc.cols; n++) {
#pragma unroll
        for (uint32_t j = 0; j < GmmaAccCoreMat::cols; j++) {
          acc(m, n)(i, j) = safeInitRowMax;
        }
      }
    }
  }
}

__device__ inline void warpGrpOnlineSoftmax(Gemm0Acc& acc, RegColWiseVec const& colMax) {
#pragma unroll
  for (uint32_t n = 0; n < acc.cols; n++) {
#pragma unroll
    for (uint32_t j = 0; j < GmmaAccCoreMat::cols; j++) {
      float const maxVal = colMax[n][j];
      float const bias = maxVal * log2e;
#pragma unroll
      for (uint32_t m = 0; m < acc.rows; m++) {
#pragma unroll
        for (uint32_t i = 0; i < GmmaAccCoreMat::rows; i++) {
          float& elem = acc(m, n)(i, j);
          assert(maxVal >= elem);
          elem = exp2f(elem * log2e - bias);
        }
      }
    }
  }
}

__device__ inline RegColWiseVec computeWarpColSum(Gemm0Acc& src) {
  auto colSum = RegColWiseVec::filled(Vec<float, GmmaAccCoreMat::cols>::filled(0));
#pragma unroll
  for (uint32_t n = 0; n < src.cols; n++) {
#pragma unroll
    for (uint32_t j = 0; j < GmmaAccCoreMat::cols; j++) {
#pragma unroll
      for (uint32_t m = 0; m < src.rows; m++) {
#pragma unroll
        for (uint32_t i = 0; i < GmmaAccCoreMat::rows; i++) {
          colSum[n][j] = (m == 0 && i == 0) ? src(m, n)(i, j) : colSum[n][j] + src(m, n)(i, j);
        }
      }
    }
  }

#pragma unroll
  for (uint32_t xorMask = 16; xorMask > 2; xorMask /= 2) {
#pragma unroll
    for (uint32_t n = 0; n < src.cols; n++) {
#pragma unroll
      for (uint32_t j = 0; j < GmmaAccCoreMat::cols; j++) {
        auto& x = colSum[n][j];
        x += __shfl_xor_sync(~0U, x, xorMask);
      }
    }
  }
  return colSum;
}

__device__ inline void storeGemm0AccToShm(uint32_t warpRank, uint32_t lane,
                                          SharedMem::XBuffer& smemX, CtaBarrier& barConsumed,
                                          Gemm0Acc const& acc) {
#if CACHE_ELEM_ENUM == 0 || CACHE_ELEM_ENUM == 5
  using F16Acc = Array2D<Vec<uint32_t, 2>, Gemm0Acc::rows, Gemm0Acc::cols>;
  F16Acc f16Acc;
  reinterpret_cast<Vec<CacheElem, sizeof(f16Acc) / sizeof(CacheElem)>&>(f16Acc) =
      convert<CacheElem>(reinterpret_cast<Vec<float, sizeof(acc) / sizeof(float)> const&>(acc));
  static_assert(Gemm0Acc::size == 1 || Gemm0Acc::size % 2 == 0);
  uint32_t const idxHalf = lane / 16;
  uint32_t const idxInHalf = lane % 16;
  uint32_t const idxOctInsideHalf = idxInHalf / 8;
  uint32_t const idxRowInsideOct = lane % 8;
  uint32_t const warpBaseC = 16 * warpRank;
  auto const toAccCoords = [](uint32_t const idxAccCoreMat) -> std::pair<uint32_t, uint32_t> {
    uint32_t const accR = idxAccCoreMat / Gemm0Acc::cols;
    uint32_t const accC = idxAccCoreMat % Gemm0Acc::cols;
    return {accR, accC};
  };
  auto const getDstAddr = [&](uint32_t idxAccCoreMat) -> LdGrain* {
    auto const [accR, accC] = toAccCoords(idxAccCoreMat);
    static_assert(sizeof(MathElem) * gemm0CtaTileNbTokens == xPartBytes);
    uint32_t const idxPart = 0;
    uint32_t const dstR = accC * 8 + idxRowInsideOct;
    uint32_t const dstC =
        exactDiv(gmma::instM * accR + warpBaseC + 8 * idxOctInsideHalf, cacheElemsPerGrain);
    assert(dstC / exactDiv(xPartBytes, grainBytes) == idxPart);
    return &smemX[idxPart].template at<true>(dstR, dstC);
  };
  auto const getAccData = [&](uint32_t idxAccCoreMat) {
    auto const [accR, accC] = toAccCoords(idxAccCoreMat);
    return f16Acc(accR, accC);
  };

  barConsumed.arrive_and_wait();
#pragma unroll
  for (uint32_t iter = 0; iter < Gemm0Acc::size / 2; iter++) {
    auto const dstAddr = getDstAddr(iter * 2 + idxHalf);
    Vec<uint32_t, 2> const data[2] = {getAccData(iter * 2), getAccData(iter * 2 + 1)};
    stmatrix<true, 4>(dstAddr, reinterpret_cast<LdGrain const&>(data));
  }
  if constexpr (Gemm0Acc::size % 2 != 0) {
    auto const dstAddr = lane < 16 ? getDstAddr(Gemm0Acc::size - 1) : nullptr;
    stmatrix<true, 2>(dstAddr, getAccData(Gemm0Acc::size - 1));
  }
#elif CACHE_ELEM_ENUM == 2
  using F8Acc = Array2D<uint32_t, Gemm0Acc::rows, Gemm0Acc::cols>;
  F8Acc f8Acc;
#pragma unroll
  for (uint32_t i = 0; i < acc.rows; i++) {
#pragma unroll
    for (uint32_t j = 0; j < acc.cols; j++) {
      auto const& core = acc(i, j);
      static_assert(mha::is_same_v<MathElem, __nv_fp8_e4m3>);
      Vec<uint16_t, 2> const f8Data = {
          __nv_cvt_float2_to_fp8x2(float2{core(0, 0), core(1, 0)}, __NV_SATFINITE, __NV_E4M3),
          __nv_cvt_float2_to_fp8x2(float2{core(0, 1), core(1, 1)}, __NV_SATFINITE, __NV_E4M3)};
      f8Acc(i, j) = reinterpret_cast<uint32_t const&>(f8Data);
    }
  }

  if constexpr (F8Acc::size == 4 || F8Acc::size == 2 || F8Acc::size == 1) {
    LdGrain* dst = nullptr;
    if (F8Acc::size == 4 || lane < 8 * F8Acc::size) {
      uint32_t const idxCore = lane / 8;
      uint32_t const srcRow = idxCore / F8Acc::cols;
      uint32_t const srcCol = idxCore % F8Acc::cols;
      uint32_t const dstCoreRow = lane % 8;
      uint32_t const dstRow = srcCol * 8 + dstCoreRow;
      BoundedVal<SharedMem::XBuffer::size * SharedMem::XBuffer::Elem::cols> const dstCol{
          srcRow * 4 + warpRank};
      dst = &smemX[dstCol.template divBy<grainsPerXPart>().get()].template at<true>(
          dstRow, dstCol.template mod<grainsPerXPart>().get());
    }
    barConsumed.arrive_and_wait();
    stmatrix<true, F8Acc::size>(dst, reinterpret_cast<Vec<uint32_t, F8Acc::size> const&>(f8Acc));
  } else {
    // we need to use loops
    assert(false);
    trap();
  }
#endif
}

#else

__device__ inline RegRowWiseVec warpRowWiseReduce(RegRowWiseVec const& init, Gemm0Acc const& src,
                                                  float (*op)(float, float)) {
  RegRowWiseVec vec = init;
#pragma unroll
  for (uint32_t m = 0; m < src.rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < GmmaAccCoreMat::rows; i++) {
#pragma unroll
      for (uint32_t n = 0; n < src.cols; n++) {
#pragma unroll
        for (uint32_t j = 0; j < GmmaAccCoreMat::cols; j++) {
          // @fixme: check if compiler is reordering these op to hide latency.
          vec[m][i] = op(vec[m][i], src(m, n)(i, j));
        }
      }
    }
  }

#pragma unroll
  for (uint32_t xorMask = 2; xorMask != 0; xorMask /= 2) {
#pragma unroll
    for (uint32_t m = 0; m < src.rows; m++) {
#pragma unroll
      for (uint32_t i = 0; i < GmmaAccCoreMat::rows; i++) {
        auto& x = vec[m][i];
        x = op(x, __shfl_xor_sync(~0U, x, xorMask));
      }
    }
  }
  return vec;
}

__device__ inline RegRowWiseVec computeWarpGrpRowMax_sync(uint32_t warpRank,
                                                          ShmQWiseVec& smemRowMax,
                                                          Gemm0Acc const& src) {
  assert(warpRank < 4);
  RegRowWiseVec const init = loadShmRowWiseVecWithDup(warpRank, smemRowMax);
  RegRowWiseVec rowMax = warpRowWiseReduce(init, src, fmax);

  storeShmRowWiseVec(warpRank, smemRowMax, rowMax);
  __syncwarp();
  return rowMax;
}

#if SPEC_DEC
__device__ inline void warpGrpApplyMask(Gemm0Acc& acc, SpecDec const& specDec,
#if SLIDING_WINDOW && !IS_SPEC_DEC_TREE
                                        int32_t tok0WinBeg,
#endif
                                        uint32_t cacheSeqLen, uint32_t idxTile, uint32_t warpRank) {
  constexpr uint32_t tileSize = gemm0CtaTileNbTokens;
  auto const inputSeqLen = specDec.inputSeqLen;
  auto const idxInputSubSeq = specDec.idxInputSubSeq;
  constexpr uint64_t fullMask = ~uint64_t{0};
  static_assert(tileSize == sizeof(fullMask) * 8);
#if SLIDING_WINDOW && !IS_SPEC_DEC_TREE
  uint32_t const ctaTokOffset = inputTokensPerCta * idxInputSubSeq;
  Range const tileRange = {tileSize * idxTile, tileSize * idxTile + tileSize};
  Range const maxMaskOutRange = {0, mha::max(0, tok0WinBeg) + (inputTokensPerCta - 1)};
  bool const ctaNeedBegMask = tileRange.beg < maxMaskOutRange.end;
  assert(ctaNeedBegMask == overlap(tileRange, maxMaskOutRange));
  int32_t const tok0NbMaskOut = int32_t(tok0WinBeg) - int32_t(tileSize * idxTile);
#else
  constexpr bool ctaNeedBegMask = false;
  uint64_t const begMask = fullMask;
  int32_t const tok0NbMaskOut = -2147483648;
#endif
  uint32_t const offset = tileSize * idxTile;
  uint32_t const nbValidCols = mha::min(offset < cacheSeqLen ? cacheSeqLen - offset : 0U, tileSize);
  bool const ctaNeedEndMask = (nbValidCols < tileSize);
  bool const ctaNeedSpecDecMask = specDec.needMask(idxTile, 0);
  bool const needMask = ctaNeedBegMask || ctaNeedEndMask || ctaNeedSpecDecMask;
  if (!needMask) {
    return;
  }
  static_assert(tileSize == 64, "not implemented");
  auto const endMask = fullMask >> (tileSize - nbValidCols);

  uint32_t const idxInQuad = laneId() % 4;
  uint32_t const idxQuad = laneId() / 4;
#pragma unroll
  for (uint32_t m = 0; m < acc.rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < GmmaAccCoreMat::rows; i++) {
      uint32_t const row = gmma::instM * m + gmma::instM / 4 * warpRank + 8 * i + idxQuad;
      uint32_t const idxQTokInCta = row / headGrpSize;
      bool const isQTokValid =
          (headGrpSize * inputTokensPerCta == ctaNbQHeads) || (idxQTokInCta < inputTokensPerCta);
      auto const specDecMask = (isQTokValid && specDec.needMask(idxTile, idxQTokInCta))
                                   ? specDec.loadTileMaskRow(idxTile, idxQTokInCta)
                                   : SpecDec::TileMaskRow{~0U, ~0U};
#if SLIDING_WINDOW && !IS_SPEC_DEC_TREE
      int32_t const begNbMaskOut = tok0NbMaskOut + int32_t(idxQTokInCta);
      uint64_t const begMask = (begNbMaskOut > 0 ? fullMask << begNbMaskOut : fullMask);
#else
      uint64_t const begMask = fullMask;
#endif
      auto const mask = begMask & endMask & reinterpret_cast<uint64_t const&>(specDecMask);
      if (mask == ~uint64_t{0}) {
        continue;
      }
#if DBG_PRINT
      if (idxInQuad == 0) {
        printf("mask at row %d: %lx\n", row, mask);
      }
#endif
#pragma unroll
      for (uint32_t n = 0; n < acc.cols; n++) {
#pragma unroll
        for (uint32_t j = 0; j < GmmaAccCoreMat::cols; j++) {
          uint32_t const col = GmmaAccCoreMat::cols * (4 * n + idxInQuad) + j;
          assert((col < nbValidCols) == bool(endMask & (1ULL << col)));
          if ((mask & (1ULL << col)) == 0) {
            acc(m, n)(i, j) = safeInitRowMax;
          }
        }
      }
    }
  }
}
#else
__device__ inline void warpGrpApplyMask(Gemm0Acc& acc, uint32_t validColBeg, uint32_t validColEnd) {
  uint32_t const idxInQuad = laneId() % 4;
#pragma unroll
  for (uint32_t n = 0; n < acc.cols; n++) {
#pragma unroll
    for (uint32_t j = 0; j < GmmaAccCoreMat::cols; j++) {
      uint32_t const col = GmmaAccCoreMat::cols * (4 * n + idxInQuad) + j;
      if (col >= validColBeg && col < validColEnd) {
        continue;
      }
#pragma unroll
      for (uint32_t m = 0; m < acc.rows; m++) {
#pragma unroll
        for (uint32_t i = 0; i < GmmaAccCoreMat::rows; i++) {
          acc(m, n)(i, j) = safeInitRowMax;
        }
      }
    }
  }
}
#endif

__device__ inline void warpGrpOnlineSoftmax(Gemm0Acc& acc, RegRowWiseVec const& rowMax) {
#pragma unroll
  for (uint32_t m = 0; m < acc.rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < GmmaAccCoreMat::rows; i++) {
      float const maxVal = rowMax[m][i];
      float const bias = maxVal * log2e;
#pragma unroll
      for (uint32_t n = 0; n < acc.cols; n++) {
#pragma unroll
        for (uint32_t j = 0; j < GmmaAccCoreMat::cols; j++) {
          float& elem = acc(m, n)(i, j);
          assert(maxVal >= elem);
          elem = exp2f(elem * log2e - bias);
        }
      }
    }
  }
}

__device__ inline RegRowWiseVec computeWarpRowSum(Gemm0Acc& src) {
  return warpRowWiseReduce(RegRowWiseVec{}, src, [](float a, float b) { return a + b; });
}

__device__ inline RegRowWiseVec loadShmRowWiseVecWithDup(uint32_t warpRank,
                                                         ShmQWiseVec const& smemVec) {
  RegRowWiseVec vec;
  uint32_t const idxQuad = laneId() / 4;
#pragma unroll
  for (uint32_t m = 0; m < RegRowWiseVec::size; m++) {
#pragma unroll
    for (uint32_t i = 0; i < RegRowWiseVec::Elem::size; i++) {
      vec[m][i] = smemVec[gmma::instM * m + gmma::instM / 4 * warpRank + 8 * i + idxQuad];
    }
  }
  return vec;
}

__device__ void storeShmRowWiseVec(uint32_t warpRank, ShmQWiseVec& smemVec,
                                   RegRowWiseVec const& regVec) {
  uint32_t const lane = laneId();
  uint32_t const idxQuad = lane / 4;
  uint32_t const idxInQuad = lane % 4;
  bool const enable = (idxInQuad == 0);
#pragma unroll
  for (uint32_t m = 0; m < RegRowWiseVec::size; m++) {
#pragma unroll
    for (uint32_t i = 0; i < RegRowWiseVec::Elem::size; i++) {
      assert(__shfl_sync(~0U, regVec[m][i], idxQuad * 4) == regVec[m][i]);
      if (enable) {
        smemVec[gmma::instM * m + gmma::instM / 4 * warpRank + 8 * i + idxQuad] = regVec[m][i];
      }
    }
  }
}

// for X
// order: 0,8,1,9, 2,10,3,11, 4,12,5,13, 6,14,7,15, ...
__device__ inline void storeGemm0AccToShm(uint32_t warpRank, uint32_t lane,
                                          SharedMem::XBuffer& smemX, CtaBarrier& barConsumed,
                                          Gemm0Acc const& acc) {
  uint32_t const idxMat = lane / 8;
  uint32_t const idxRow = lane % 8;
  barConsumed.arrive_and_wait();
#pragma unroll
  for (uint32_t m = 0; m < Gemm0Acc::rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < GmmaAccCoreMat::rows; i++) {
      Vec<uint32_t, exactDiv(Gemm0Acc::cols, 2)> fp8Data;
#pragma unroll
      for (uint32_t n = 0; n < exactDiv(Gemm0Acc::cols, 2); n++) {
        reinterpret_cast<Vec<__nv_fp8x2_e4m3, 2>&>(fp8Data[n]) = {
            __nv_fp8x2_e4m3(float2{acc(m, n * 2)(i, 0), acc(m, n * 2 + 1)(i, 0)}),
            __nv_fp8x2_e4m3(float2{acc(m, n * 2)(i, 1), acc(m, n * 2 + 1)(i, 1)})};
      }
      static_assert(decltype(fp8Data)::size == 4);
      stmatrix_4x<false>(this_warp(),
                         &smemX[m].template at<true>(16 * warpRank + 8 * i + idxRow, idxMat),
                         fp8Data);
    }
  }
}
#endif

#if SWAP_AB
__device__ inline Vec<RegMatAFrag, gemm1NbGmmaInstM> loadVTileTransposed(
    uint32_t warpRank, uint32_t lane, SharedMem::VBuffer const& smemV, uint32_t idxGmmaInstK) {
  Vec<RegMatAFrag, gemm1NbGmmaInstM> fragA;
  constexpr uint32_t instK = gmma::instK<MathElem>;
#pragma unroll
  for (uint32_t i = 0; i < gemm1NbGmmaInstM; i++) {
    static_assert(exactDiv(gmma::instM, gmmaWarpsPerGrp) == grainBytes);
    constexpr uint32_t grainsPerPart = exactDiv(cacheHeadPartBytes, grainBytes);
#if CACHE_ELEM_ENUM == 0 || CACHE_ELEM_ENUM == 5
    uint32_t idxRow = lane % 8;
    uint32_t idxMat = lane / 8;
    uint32_t c = idxMat % 2;
    uint32_t r = idxMat / 2;
    auto const col = BoundedVal<2 * gmmaWarpsPerGrp * gemm1NbGmmaInstM>{
        2 * (gmmaWarpsPerGrp * i + warpRank) + c};
    auto const src = &smemV[col.template divBy<grainsPerPart>().get()].template at<true>(
        instK * idxGmmaInstK + 8 * r + idxRow, col.template mod<grainsPerPart>().get());
    auto const data = ldmatrix<true, 4>(src);
    fragA[i] = reinterpret_cast<RegMatAFrag const&>(data);
#elif CACHE_ELEM_ENUM == 2
    auto const col = BoundedVal<gmmaWarpsPerGrp * gemm1NbGmmaInstM>{gmmaWarpsPerGrp * i + warpRank};
    LdGrain const* src = &smemV[col.template divBy<grainsPerPart>().get()].template at<true>(
        instK * idxGmmaInstK + lane, col.template mod<grainsPerPart>().get());
    auto const data = ldmatrix<true, 4>(src);
    fragA[i](0, 0)(0, 0) = prmt(data[0], data[1], {0, 4, 2, 6});
    fragA[i](0, 0)(1, 0) = prmt(data[0], data[1], {1, 5, 3, 7});
    fragA[i](0, 1)(0, 0) = prmt(data[2], data[3], {0, 4, 2, 6});
    fragA[i](0, 1)(1, 0) = prmt(data[2], data[3], {1, 5, 3, 7});
#endif
  }
  return fragA;
}
#else
__device__ inline void transposeVTile(uint32_t warpRank, uint32_t lane, SharedMem::VTBuffer& dst,
                                      SharedMem::VBuffer const& src) {
  uint32_t const idxMat = lane / 8;
  uint32_t const idxRow = lane % 8;
#pragma unroll
  for (uint32_t m = 0; m < exactDiv(SharedMem::VTBuffer::rows, gmma::instM); m++) {
    static_assert(cacheHeadPartElems >= gmma::instM);
    uint32_t const idxPart = gmma::instM * m / cacheHeadPartElems;
    constexpr uint32_t grainsPerCacheHeadPart = exactDiv(cacheHeadPartElems, cacheElemsPerGrain);
#pragma unroll
    for (uint32_t n = 0; n < exactDiv(SharedMem::VTBuffer::cols, 2); n++) {
      LdGrain const a = ldmatrix_4x<true>(
          this_warp(), &src[idxPart].template at<true>(
                           32 * n + lane, exactDiv(gmma::instM, cacheElemsPerGrain) * m -
                                              grainsPerCacheHeadPart * idxPart + warpRank));
      LdGrain const b = {prmt(a[0], a[1], {0, 4, 2, 6}), prmt(a[0], a[1], {1, 5, 3, 7}),
                         prmt(a[2], a[3], {0, 4, 2, 6}), prmt(a[2], a[3], {1, 5, 3, 7})};
      uint32_t const i = idxMat % 2;
      uint32_t const j = idxMat / 2;
      stmatrix_4x<false>(
          this_warp(),
          &dst.template at<true>(gmma::instM * m + 16 * warpRank + 8 * i + idxRow, 2 * n + j), b);
    }
  }
}
#endif

#if SWAP_AB
__device__ inline Vec<float, divUp(ShmQWiseVec::size, warp_size)> loadShmColWiseVecNoDup(
    ShmQWiseVec const& shmVec) {
  Vec<float, divUp(ShmQWiseVec::size, warp_size)> ret;
#pragma unroll
  for (uint32_t i = 0; i < divUp(ShmQWiseVec::size, warp_size); i++) {
    uint32_t const idx = i * warp_size + laneId();
    bool const inBound = ((ShmQWiseVec::size % warp_size == 0) || (idx < ShmQWiseVec::size));
    ret[i] = (inBound ? shmVec[idx] : 0);
  }
  return ret;
}

__device__ inline void storeShmColWiseVecNoDup(
    ShmQWiseVec& shmVec, Vec<float, divUp(ShmQWiseVec::size, warp_size)> const& src) {
#pragma unroll
  for (uint32_t i = 0; i < divUp(ShmQWiseVec::size, warp_size); i++) {
    uint32_t const idx = i * warp_size + laneId();
    bool const inBound = ((ShmQWiseVec::size % warp_size == 0) || (idx < ShmQWiseVec::size));
    if (inBound) {
      shmVec[idx] = src[i];
    }
  }
}
#else
__device__ inline Vec<float, divUp(exactDiv(ShmQWiseVec::size, gmma::instM) * (gmma::instM / 4),
                                   warp_size)>
loadShmRowWiseVecNoDup(uint32_t warpRank, ShmQWiseVec const& shmVec) {
  constexpr uint32_t const nbElems = exactDiv(ShmQWiseVec::size, gmma::instM) * (gmma::instM / 4);
  Vec<float, divUp(nbElems, warp_size)> ret;
  uint32_t const lane = laneId();
  uint32_t const idxHalf = lane / (gmma::instM / 4);
  uint32_t const idxInHalf = lane % (gmma::instM / 4);
#pragma unroll
  for (uint32_t i = 0; i < divUp(nbElems, warp_size); i++) {
    uint32_t const idx =
        gmma::instM * 2 * i + gmma::instM * idxHalf + (gmma::instM / 4) * warpRank + idxInHalf;
    bool const inBound = ((nbElems % warp_size == 0) || (i + 1 < divUp(nbElems, warp_size)) ||
                          (idx < ShmQWiseVec::size));
    ret[i] = (inBound ? shmVec[idx] : 0);
  }
  return ret;
}

__device__ inline void storeShmRowWiseVecNoDup(
    uint32_t warpRank, ShmQWiseVec& shmVec,
    Vec<float, divUp(exactDiv(ShmQWiseVec::size, gmma::instM) * (gmma::instM / 4),
                     warp_size)> const& src) {
  constexpr uint32_t const nbElems = exactDiv(ShmQWiseVec::size, gmma::instM) * (gmma::instM / 4);
  Vec<float, divUp(nbElems, warp_size)> ret;
  uint32_t const lane = laneId();
  uint32_t const idxHalf = lane / (gmma::instM / 4);
  uint32_t const idxInHalf = lane % (gmma::instM / 4);
#pragma unroll
  for (uint32_t i = 0; i < divUp(nbElems, warp_size); i++) {
    uint32_t const idx =
        gmma::instM * 2 * i + gmma::instM * idxHalf + (gmma::instM / 4) * warpRank + idxInHalf;
    bool const inBound = ((nbElems % warp_size == 0) || (i + 1 < divUp(nbElems, warp_size)) ||
                          (idx < ShmQWiseVec::size));
    if (inBound) {
      shmVec[idx] = src[i];
    }
  }
}
#endif

#if SWAP_AB
__device__ inline void rescaleGemm1AccForNewColMax_sync(
    uint32_t warpRank, ShmQWiseVec const& shmXColMax, ShmQWiseVec const (&shmXColSum)[gemm0NbWarps],
    ShmQWiseVec& shmAccColMax, Gemm1Acc& acc, ShmQWiseVec& shmAccColSum) {
  auto accColSum = loadShmColWiseVecNoDup(shmAccColSum);

  auto const xColMax = loadShmColWiseVecNoDup(shmXColMax);
  auto const accColMax = loadShmColWiseVecNoDup(shmAccColMax);
  auto const needRescaleVec = (accColMax < xColMax);
  UniformNeedRescaleMask rescaleMask;
  bool anyNeedRescale = false;
#pragma unroll
  for (uint32_t i = 0; i < rescaleMask.size; i++) {
    assert(accColMax[i] <= xColMax[i]);
    rescaleMask[i] = __ballot_sync(~0U, needRescaleVec[i]);
    anyNeedRescale = anyNeedRescale || (rescaleMask[i] != 0);
  }
  if (anyNeedRescale) {
    auto const scaleVec = expf(accColMax - xColMax);
    auto const lane = laneId();
#pragma unroll
    for (uint32_t n = 0; n < Gemm1Acc::cols; n++) {
      uint32_t const vecIdx = gmma::instNBase * n / warp_size;
      uint32_t const offset = gmma::instNBase * n % warp_size;
      constexpr uint32_t nbThrdsPerInstNBase = exactDiv(gmma::instNBase, GmmaAccCoreMat::cols);
#pragma unroll
      for (uint32_t j = 0; j < GmmaAccCoreMat::cols; j++) {
        auto const mask = ((rescaleMask[vecIdx] >> (offset + j)) & 0b01010101U);
        auto getScale = [&] {
          return __shfl_sync(~0U, scaleVec[vecIdx],
                             offset + lane % nbThrdsPerInstNBase * GmmaAccCoreMat::cols + j);
        };
        assert((getScale() != 1) ==
               ((mask >> lane % nbThrdsPerInstNBase * GmmaAccCoreMat::cols) & 0x1U));
        bool const needRescale = (mask != 0);
        if (!needRescale) {  // this branch is warp-uniform
          continue;
        }
        float const scale = getScale();
#pragma unroll
        for (uint32_t m = 0; m < Gemm1Acc::rows; m++) {
#pragma unroll
          for (uint32_t i = 0; i < GmmaAccCoreMat::rows; i++) {
            acc(m, n)(i, j) *= scale;
          }
        }
      }
    }
    accColSum = accColSum * scaleVec;
  }
  // Every warp has loaded the previous accColMax/accColSum and rescaled its
  // accumulator; the group syncs once here (a split arrive -> work -> wait
  // cannot be expressed with bar.sync) before warp 3 overwrites the smem copy.
  gemm1WarpGrpSync();

  // @fixme: with atomic, we can let the first warp reaching here to do the update, instead of
  // always warp 3.
  uint32_t const warpRankForUpdate = gmmaWarpsPerGrp - 1;
  if (warpRank == warpRankForUpdate) {
    if (anyNeedRescale) {
      storeShmColWiseVecNoDup(shmAccColMax, xColMax);
    }
#pragma unroll
    for (uint32_t i = 0; i < gemm0NbWarps; i++) {
      accColSum = accColSum + loadShmColWiseVecNoDup(shmXColSum[i]);
    }
    storeShmColWiseVecNoDup(shmAccColSum, accColSum);
  }
  gemm1WarpGrpSync();
}
#else
__device__ inline void rescaleGemm1AccForNewRowMax_sync(uint32_t warpRank,
                                                        ShmQWiseVec const& shmXRowMax,
                                                        ShmQWiseVec const& shmXRowSum,
                                                        ShmQWiseVec& shmAccRowMax, Gemm1Acc& acc,
                                                        ShmQWiseVec& shmAccRowSum) {
  auto accRowSum = loadShmRowWiseVecNoDup(warpRank, shmAccRowSum);
  auto const xRowMax = loadShmRowWiseVecNoDup(warpRank, shmXRowMax);
  auto const accRowMax = loadShmRowWiseVecNoDup(warpRank, shmAccRowMax);
  assert(all(xRowMax >= accRowMax));
  auto const needRescaleVec = (accRowMax < xRowMax);
  UniformNeedRescaleMask rescaleMask;
  bool anyNeedRescale = false;
#pragma unroll
  for (uint32_t i = 0; i < rescaleMask.size; i++) {
    assert(accRowMax[i] <= xRowMax[i]);
    rescaleMask[i] = __ballot_sync(~0U, needRescaleVec[i]);
    anyNeedRescale = anyNeedRescale || (rescaleMask[i] != 0);
  }

  if (anyNeedRescale) {
    auto const scaleVec = expf(accRowMax - xRowMax);
    auto const lane = laneId();
#pragma unroll
    for (uint32_t m = 0; m < Gemm1Acc::rows; m++) {
#pragma unroll
      for (uint32_t i = 0; i < GmmaAccCoreMat::rows; i++) {
        uint8_t const mask = reinterpret_cast<uint8_t const(&)[2][2]>(rescaleMask[m / 2])[m % 2][i];
        bool const needRescale = (mask != 0);
        if (needRescale) {  // this branch is warp-uniform
          float const scale = __shfl_sync(~0U, scaleVec[m / 2], 16 * (m % 2) + 8 * i + lane / 4);
#pragma unroll
          for (uint32_t n = 0; n < Gemm1Acc::cols; n++) {
#pragma unroll
            for (uint32_t j = 0; j < GmmaAccCoreMat::cols; j++) {
              acc(m, n)(i, j) *= scale;
            }
          }
        }
      }
    }
    accRowSum = accRowSum * scaleVec;
  }
  __syncwarp();
  auto const xRowSum = loadShmRowWiseVecNoDup(warpRank, shmXRowSum);
  storeShmRowWiseVecNoDup(warpRank, shmAccRowSum, accRowSum + xRowSum);
  storeShmRowWiseVecNoDup(warpRank, shmAccRowMax, xRowMax);
  __syncwarp();
}
#endif

#if SWAP_AB
__device__ inline void rescaleAcc(Gemm1Acc& acc, RegColWiseVec const& scale) {
#pragma unroll
  for (uint32_t n = 0; n < Gemm1Acc::cols; n++) {
#pragma unroll
    for (uint32_t j = 0; j < GmmaAccCoreMat::cols; j++) {
#pragma unroll
      for (uint32_t m = 0; m < Gemm1Acc::rows; m++) {
#pragma unroll
        for (uint32_t i = 0; i < GmmaAccCoreMat::rows; i++) {
          acc(m, n)(i, j) *= scale[n][j];
        }
      }
    }
  }
}
#else
__device__ inline void rescaleAcc(Gemm1Acc& acc, RegRowWiseVec const& scale) {
#pragma unroll
  for (uint32_t m = 0; m < Gemm1Acc::rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < GmmaAccCoreMat::rows; i++) {
#pragma unroll
      for (uint32_t n = 0; n < Gemm1Acc::cols; n++) {
#pragma unroll
        for (uint32_t j = 0; j < GmmaAccCoreMat::cols; j++) {
          acc(m, n)(i, j) *= scale[m][i];
        }
      }
    }
  }
}
#endif

#if SWAP_AB
// @fixme: consider make this noinline
template <bool dstIsStrided = false, typename DstHead>
__device__ inline void saveTransposedOutput(uint32_t threadRank, uint32_t warpRank, DstHead* dst,
                                            SharedMem::OutSwizzleBuf& swizzleBuf,
                                            Gemm1Acc const& acc, CtaBarrier& warpGrpBar,
                                            uint32_t nbKHeads) {
  uint32_t const lane = laneId();
#if CACHE_ELEM_ENUM == 0 || CACHE_ELEM_ENUM == 5
  uint32_t const idxMat = lane / 8;
  uint32_t const idxRow = lane % 8;
#elif CACHE_ELEM_ENUM == 2
  uint32_t const idxQuad = lane / 4;
  uint32_t const idxInQuad = lane % 4;
#endif
#pragma unroll
  for (uint32_t m = 0; m < Gemm1Acc::rows; m++) {
#pragma unroll
    for (uint32_t n = 0; n < Gemm1Acc::cols; n++) {
      auto const& core = acc(m, n);
#if CACHE_ELEM_ENUM == 0 || CACHE_ELEM_ENUM == 5
      Vec<uint32_t, 2> f16Core;
      reinterpret_cast<Vec<InputElem, 4>&>(f16Core) =
          convert<InputElem>(reinterpret_cast<Vec<float, 4> const&>(core));
      auto const dst = idxMat < 2
                           ? &swizzleBuf.template at<true>(
                                 8 * n + idxRow, 2 * (gmmaWarpsPerGrp * m + warpRank) + idxMat)
                           : nullptr;
      stmatrix<true, 2>(dst, f16Core);
#elif CACHE_ELEM_ENUM == 2
      // each row is part of a b16 8x8 matrix and is transposed
      Array2D<InputElem, GmmaAccCoreMat::rows, GmmaAccCoreMat::cols> coreTrans;
      for (uint32_t i = 0; i < GmmaAccCoreMat::rows; i++) {
        static_assert(GmmaAccCoreMat::cols == 2 && sizeof(InputElem) == 2);
        InputElem2 const coreRow = float2ToInputElem2({core(i, 0), core(i, 1)});
        auto const coreRowTrans = movmatrix(reinterpret_cast<uint32_t const&>(coreRow));
        reinterpret_cast<uint32_t&>(coreTrans(i, 0)) = coreRowTrans;
      }
      // expect compiler to generate two PRMT instructions
      Vec<InputElem, 4> const data = {coreTrans(0, 0), coreTrans(1, 0), coreTrans(0, 1),
                                      coreTrans(1, 1)};
      swizzleBuf.template at<true>(
          gmma::instNBase * n + idxQuad,
          (gmma::instM * m + exactDiv(gmma::instM, gmmaWarpsPerGrp) * warpRank) / 16)[idxInQuad] =
          data;
#endif
    }
  }
  warpGrpBar.arrive_and_wait();

  constexpr uint32_t headsPerIter = exactDiv(grainBytes * gemm1NbThrds, paddedInputHeadBytes);
  constexpr uint32_t nbIters = divUp(ctaNbValidQHeads, headsPerIter);
  constexpr uint32_t nbWholeIters = ctaNbValidQHeads / headsPerIter;
  constexpr uint32_t nbGrainsPerHead = exactDiv(paddedInputHeadBytes, grainBytes);
  uint32_t const idxHeadBase = threadRank / nbGrainsPerHead;
  uint32_t const idxGrain = threadRank % nbGrainsPerHead;
#pragma unroll
  for (uint32_t iter = 0; iter < nbIters; iter++) {
    uint32_t const idxHead = idxHeadBase + iter * headsPerIter;
    if ((iter < nbWholeIters || idxHead < ctaNbValidQHeads) &&
        (!isHeadPadded || idxGrain < grainsPerIOHead)) {
#if CACHE_ELEM_ENUM == 0 || CACHE_ELEM_ENUM == 5
      auto const data = swizzleBuf.template at<true>(idxHead, idxGrain);
#elif CACHE_ELEM_ENUM == 2
      auto const data = reinterpret_cast<Vec<LdGrain, 2>&>(
          swizzleBuf.template at<true>(idxHead, idxGrain / 2))[idxGrain % 2];
#endif
      constexpr uint32_t inputElemsPerGrain = exactDiv(grainBytes, inputElemSize);
      auto const outVec = convert<typename DstHead::Elem>(
          reinterpret_cast<Vec<InputElem, inputElemsPerGrain> const&>(data));
      uint32_t dstHeadIdx = idxHead;
#ifdef SPEC_Q_SEQ_LEN
      if constexpr (dstIsStrided) {
        uint32_t const idxToken = idxHead / headGrpSize;
        if (idxToken < SPEC_Q_SEQ_LEN) {
          uint32_t const strideBetweenTokens = nbKHeads * headGrpSize;
          dstHeadIdx = idxToken * strideBetweenTokens + (idxHead % headGrpSize);
        }
      }
#endif
      reinterpret_cast<Vec<mha::decay_t<decltype(outVec)>, nbGrainsPerHead>&>(
          dst[dstHeadIdx])[idxGrain] = outVec;
    }
  }
}

template <bool dstIsStrided, typename DstHead>
__device__ inline void finalizeAndWriteOut_sync(
    uint32_t threadRank, uint32_t warpRank, DstHead* dst, SharedMem::OutSwizzleBuf& swizzleBuf,
    Gemm1Acc& acc, float xvoScale, CtaBarrier& warpGrpBar, ShmQWiseVec const& accColSum,
    ShmQWiseVec const& accColMax, ShmQWiseVec const* attentionSinksVec, uint32_t nbKHeads) {
  // @fixme: if ctaNbQHeads is large, use loadShmColWiseVecNoDup + rcp + shfl to avoid 8x waste of
  // mufu.rcp static_assert(ctaNbQHeads <= 8, "Warning: consider using loadShmColWiseVecNoDup + rcp
  // + shfl to avoid 8x waste of mufu.rcp");
  auto regColSum = loadShmColWiseVecWithDup(accColSum);
  if (attentionSinksVec != nullptr) {
    auto const regAccColMax = loadShmColWiseVecWithDup(accColMax);
    auto const regAttentionSinks = loadGmemColWiseVecWithDup(attentionSinksVec[0], headGrpSize - 1);
    auto regColSinks = expf(regAttentionSinks - regAccColMax);
    regColSum = regColSum + regColSinks;
  }
  auto const regOutScale = __frcp_rn(regColSum) * xvoScale;
  rescaleAcc(acc, regOutScale);

  saveTransposedOutput<dstIsStrided, DstHead>(threadRank, warpRank, dst, swizzleBuf, acc,
                                              warpGrpBar, nbKHeads);
  warpGrpBar.arrive_and_wait();
}
#else
template <typename DstHead>
__device__ inline void finalizeAndWriteOut_sync(
    uint32_t warpRank, DstHead* dst, SharedMem::OutSwizzleBuf& swizzleBuf, Gemm1Acc& acc,
    float xvoScale, ShmQWiseVec const& accRowSum,
    uint32_t nbKHeads /* for spec dec. set to 1 for workspace*/, uint32_t ctaNbValidTokens) {
  auto const regRowSum = loadShmRowWiseVecWithDup(warpRank, accRowSum);
  auto const regOutScale = __frcp_rn(regRowSum) * xvoScale;
  rescaleAcc(acc, regOutScale);

  using DstElem = typename DstHead::Elem;
  auto const lane = laneId();
  uint32_t const idxQuad = lane / 4;
  uint32_t const idxInQuad = lane % 4;
  using Atom = Vec<Vec<DstElem, 4>, 4>;
  using SwizzleBuf = Array2D<Vec<Vec<DstElem, 4>, 4>, ctaNbQHeads, exactDiv(headElems, 4 * 4)>;
  static_assert(sizeof(SwizzleBuf) <= sizeof(swizzleBuf));
  auto& buf = reinterpret_cast<SwizzleBuf&>(swizzleBuf);
#pragma unroll
  for (uint32_t m = 0; m < Gemm1Acc::rows; m++) {
#pragma unroll
    for (uint32_t i = 0; i < GmmaAccCoreMat::rows; i++) {
      uint32_t const r = gmma::instM * m + 16 * warpRank + 8 * i + idxQuad;
      static_assert(SwizzleBuf::cols == exactDiv(Gemm1Acc::cols, 2));
#pragma unroll
      for (uint32_t n = 0; n < exactDiv(Gemm1Acc::cols, 2); n++) {
        Vec<DstElem, 4> const v =
            convert<DstElem>(Vec<float, 4>{acc(m, n * 2)(i, 0), acc(m, n * 2 + 1)(i, 0),
                                           acc(m, n * 2)(i, 1), acc(m, n * 2 + 1)(i, 1)});
        //@fixme: without reinterpret_cast to V, the compiler generates wrong code, and require a
        //__syncwarp()
        // after rescaleAcc() to work around. Likely a bug of the compiler.
        //@todo: report a compiler bug.
        using V = Vec<uint32_t, exactDiv(sizeof(v), sizeof(uint32_t))>;
        reinterpret_cast<V&>(buf.template at<true>(r, n)[idxInQuad]) =
            reinterpret_cast<V const&>(v);
        // buf.template at<true>(r, n)[idxInQuad] = v;
      }
    }
  }
  __syncwarp();

#pragma unroll
  for (uint32_t m = 0; m < Gemm1Acc::rows; m++) {
    constexpr uint32_t srcHeadBytes = sizeof(DstElem) * headElems;
    constexpr uint32_t grpSize = exactDiv(srcHeadBytes, grainBytes);
    constexpr uint32_t nbGrps = exactDiv(warp_size, grpSize);
    uint32_t const idxGrp = lane / grpSize;
    constexpr uint32_t grainsPerAtom = exactDiv(sizeof(Atom), grainBytes);
    uint32_t const rowBase = gmma::instM * m + 16 * warpRank;
    constexpr uint32_t totalNbGrains = grainsPerAtom * SwizzleBuf::cols * 16;
    uint32_t const nbIters = divUp(totalNbGrains, nbGrps);
    constexpr bool wholeIters = (totalNbGrains % nbGrps == 0);
    constexpr bool wholeHeads = (validElemsPerHead == headElems);
#pragma unroll
    for (uint32_t iter = 0; iter < nbIters; iter++) {
      uint32_t const idxGrain = nbGrps * iter + idxGrp;
      constexpr uint32_t grainsPerSrcHead = exactDiv(srcHeadBytes, grainBytes);
      uint32_t const r = idxGrain / grainsPerSrcHead;
      if (!wholeIters && r >= 16) {
        break;
      }
      uint32_t const cGrain = idxGrain % grainsPerSrcHead;
      uint32_t const cAtom = cGrain / grainsPerAtom;
      constexpr uint32_t grainsPerDstHead = exactDiv(sizeof(DstHead), grainBytes);
      uint32_t const glbRow = gmma::instM * m + 16 * warpRank + r;
      if (ctaNbValidQHeads != ctaNbQHeads && glbRow >= ctaNbValidQHeads) {
        break;
      }
      if (wholeHeads || cGrain < grainsPerDstHead) {
        uint32_t const srcRow = rowBase + r;
        auto const data = reinterpret_cast<LdGrain(&)[grainsPerAtom]>(
            buf.template at<true>(srcRow, cAtom))[cGrain % grainsPerAtom];
#if SPEC_DEC
        static_assert(beamWidth == 1);
        uint32_t const idxToken = srcRow / headGrpSize;  // inside CTA
        if (idxToken >= ctaNbValidTokens) {
          break;
        }
        uint32_t const tokenPad = headGrpSize * (nbKHeads - 1);
        uint32_t const dstRow = srcRow + idxToken * tokenPad;
#else
        uint32_t const dstRow = srcRow;
#endif
        reinterpret_cast<LdGrain(&)[grainsPerDstHead]>(dst[dstRow])[cGrain] = data;
      }
    }
  }
}
#endif

template <typename SrcElem, bool forNeox, uint32_t nbThrds, typename DstElem>
__device__ inline Vec<Vec<DstElem, 2>, ropeNbPairsPerThrd<nbThrds>> loadHead(
    Vec<SrcElem, validElemsPerHead> const& head, uint32_t tid) {
  constexpr uint32_t nbPairs = exactDiv(validElemsPerHead, 2);
  constexpr uint32_t nbPairsPerThrd = ropeNbPairsPerThrd<nbThrds>;
  constexpr uint32_t nbWorkingThrds = exactDiv(nbPairs, nbPairsPerThrd);
  bool const isWorkingThrd = (nbWorkingThrds == nbThrds || tid < nbWorkingThrds);
  static_assert(nbPairs % nbPairsPerThrd == 0);
  Vec<Vec<DstElem, 2>, nbPairsPerThrd> ret;
  if constexpr (forNeox) {
    auto const& pairs =
        reinterpret_cast<Vec<Vec<Vec<SrcElem, nbPairsPerThrd>, nbWorkingThrds>, 2> const&>(head);
    auto const data = isWorkingThrd
                          ? Vec<Vec<SrcElem, nbPairsPerThrd>, 2>{pairs[0][tid], pairs[1][tid]}
                          : Vec<Vec<SrcElem, nbPairsPerThrd>, 2>{};
    Vec<Vec<DstElem, nbPairsPerThrd>, 2> const tmp = {convert<DstElem>(data[0]),
                                                      convert<DstElem>(data[1])};
#pragma unroll
    for (uint32_t i = 0; i < nbPairsPerThrd; i++) {
      ret[i][0] = tmp[0][i];
      ret[i][1] = tmp[1][i];
    }
  } else {
    auto const data =
        isWorkingThrd ? reinterpret_cast<Vec<Vec<SrcElem, 2>, nbPairsPerThrd> const*>(&head)[tid]
                      : Vec<Vec<SrcElem, 2>, nbPairsPerThrd>{};
#pragma unroll
    for (uint32_t i = 0; i < nbPairsPerThrd; i++) {
      ret[i] = convert<DstElem>(data[i]);
    }
  }
  return ret;
}

template <bool forNeox, uint32_t nbPairsPerThrd>
__device__ inline mha::conditional_t<forNeox, Vec<Vec<CacheElem, nbPairsPerThrd>, 2>,
                                     Vec<Vec<CacheElem, 2>, nbPairsPerThrd>>
applyRoPE(Vec<Vec<float, 2>, nbPairsPerThrd> const& data,
          Vec<Vec<float, 2>, nbPairsPerThrd> const& ropeCosSin) {
  Vec<Vec<float, 2>, nbPairsPerThrd> r;
#pragma unroll
  for (uint32_t i = 0; i < nbPairsPerThrd; i++) {
    float const x = data[i][0];
    float const y = data[i][1];
    float const c = ropeCosSin[i][0];
    float const s = ropeCosSin[i][1];
    r[i] = Vec<float, 2>{c * x - s * y, s * x + c * y};
  }
  if constexpr (forNeox) {
    Vec<Vec<float, nbPairsPerThrd>, 2> tmp;
#pragma unroll
    for (uint32_t i = 0; i < nbPairsPerThrd; i++) {
      tmp[0][i] = r[i][0];
      tmp[1][i] = r[i][1];
    }
    return Vec<Vec<CacheElem, nbPairsPerThrd>, 2>{convert<CacheElem>(tmp[0]),
                                                  convert<CacheElem>(tmp[1])};
  } else {
    Vec<Vec<CacheElem, 2>, nbPairsPerThrd> ret;
#pragma unroll
    for (uint32_t i = 0; i < nbPairsPerThrd; i++) {
      ret[i] = convert<CacheElem>(r[i]);
    }
    return ret;
  }
}

template <bool forNeox, uint32_t nbThrds>
__device__ inline void storeRotatedPairsForKV(
    GMemCacheHead& dst,
    mha::conditional_t<forNeox, Vec<Vec<CacheElem, ropeNbPairsPerThrd<nbThrds>>, 2>,
                       Vec<Vec<CacheElem, 2>, ropeNbPairsPerThrd<nbThrds>>> const& src,
    uint32_t tid) {
  constexpr uint32_t nbPairs = exactDiv(validElemsPerHead, 2);
  constexpr uint32_t nbPairsPerThrd = ropeNbPairsPerThrd<nbThrds>;
  constexpr uint32_t nbWorkingThrds = exactDiv(nbPairs, nbPairsPerThrd);
  bool const isWorkingThrd = (nbWorkingThrds == nbThrds || tid < nbWorkingThrds);
  static_assert(nbPairs % nbPairsPerThrd == 0);
  if (!isWorkingThrd) {
    return;
  }
  if constexpr (forNeox) {
    auto& pairs =
        reinterpret_cast<Vec<Vec<Vec<CacheElem, nbPairsPerThrd>, nbWorkingThrds>, 2>&>(dst);
    pairs[0][tid] = src[0];
    pairs[1][tid] = src[1];
  } else {
    reinterpret_cast<Vec<Vec<CacheElem, 2>, nbPairsPerThrd>*>(&dst)[tid] = src;
  }
}

template <bool forNeox, uint32_t nbThrds>
__device__ inline void storeRotatedPairsForQ(
    SharedMem::QBuffer& dst,
    mha::conditional_t<forNeox, Vec<Vec<CacheElem, ropeNbPairsPerThrd<nbThrds>>, 2>,
                       Vec<Vec<CacheElem, 2>, ropeNbPairsPerThrd<nbThrds>>> const& src,
    uint32_t row, uint32_t tid) {
  constexpr uint32_t nbPairs = exactDiv(validElemsPerHead, 2);
  constexpr uint32_t nbPairsPerThrd = ropeNbPairsPerThrd<nbThrds>;
  constexpr uint32_t nbWorkingThrds = exactDiv(nbPairs, nbPairsPerThrd);
  bool const isWorkingThrd = (nbWorkingThrds == nbThrds || tid < nbWorkingThrds);
  static_assert(nbPairs % nbPairsPerThrd == 0);
  if (isWorkingThrd) {
    if constexpr (forNeox) {
#pragma unroll
      for (uint32_t i = 0; i < 2; i++) {
        auto const byteOffset =
            BoundedVal<mathHeadBytes>{cacheElemSize * nbPairsPerThrd * (nbWorkingThrds * i + tid)};
        uint32_t const idxPart = byteOffset.template divBy<qPartBytes>().get();
        auto const byteOffsetInsidePart = byteOffset.template mod<qPartBytes>();
        uint32_t const idxGrain = byteOffsetInsidePart.template divBy<grainBytes>().get();
        LdGrain& grain = dst[idxPart].template at<true>(row, idxGrain);
        uint32_t const byteOffsetInsideGrain =
            byteOffsetInsidePart.template mod<grainBytes>().get();
        static_assert(cacheElemSize * nbPairsPerThrd <= grainBytes &&
                      grainBytes % (cacheElemSize * nbPairsPerThrd) == 0);
        reinterpret_cast<Vec<CacheElem, nbPairsPerThrd>&>(
            reinterpret_cast<mha::byte*>(&grain)[byteOffsetInsideGrain]) = src[i];
      }
    } else {
      auto const byteOffset = BoundedVal<mathHeadBytes>{cacheElemSize * 2 * nbPairsPerThrd * tid};
      uint32_t const idxPart = byteOffset.template divBy<qPartBytes>().get();
      auto const byteOffsetInsidePart = byteOffset.template mod<qPartBytes>();
      uint32_t const idxGrain = byteOffsetInsidePart.template divBy<grainBytes>().get();
      LdGrain& grain = dst[idxPart].template at<true>(row, idxGrain);
      uint32_t const byteOffsetInsideGrain = byteOffsetInsidePart.template mod<grainBytes>().get();
      static_assert(cacheElemSize * 2 * nbPairsPerThrd <= grainBytes &&
                    grainBytes % (cacheElemSize * 2 * nbPairsPerThrd) == 0);
      reinterpret_cast<Vec<Vec<CacheElem, 2>, nbPairsPerThrd>&>(
          reinterpret_cast<mha::byte*>(&grain)[byteOffsetInsideGrain]) = src;
    }
  }
  static_assert(validElemsPerHead % 16 == 0);
  __syncwarp();
  if constexpr (validElemsPerHead < headElems) {
    static_assert(validElemsPerHead >= headElems - exactDiv(headElems, nbQParts));
    constexpr uint32_t nbPadGrainsPerHead =
        exactDiv(headElems - validElemsPerHead, cacheElemsPerGrain);
    constexpr uint32_t nbPadGrains = nbPadGrainsPerHead * ctaNbQHeads;
    uint32_t const nbIters = divUp(nbPadGrains, nbThrds);
#pragma unroll
    for (uint32_t iter = 0; iter < nbIters; iter++) {
      uint32_t idx = tid + nbThrds * iter;
      if (idx >= nbPadGrains) {
        break;
      }
      uint32_t const r = idx / nbPadGrainsPerHead;
      uint32_t const c = grainsPerQPart - nbPadGrainsPerHead + idx % nbPadGrainsPerHead;
      dst[dst.size - 1].template at<true>(r, c) = LdGrain{};
    }
  }
}

#ifndef GENERATE_CUBIN
static uint32_t chooseNbSubSeq(uint32_t multiProcessorCount,
                               uint32_t batchSize, uint32_t nbKHeads,
                               uint32_t maxSeqLen, uint32_t ctasPerSm = 1) {
  uint32_t const maxNbSubSeq = divUp(maxSeqLen, gemm0CtaTileNbTokens);
  auto const env = std::getenv("XQA_NB_SUB_SEQ");
  if (env != nullptr) {
    int32_t const val = std::stoi(env);
    if (val > 0) {
      return mha::min<uint32_t>(val, maxNbSubSeq);
    }
  }
#if ENABLE_MIXED_KV_CACHE
  // The mixed CTA is small enough for several to be resident per SM.  The
  // hardware then packs the grid onto a fraction of the SMs unless there are
  // at least slots = SMs * ctasPerSm CTAs, and beyond that wave quantization
  // (a straggler wave of a few CTAs doubles the tail) dominates.  Pick the
  // split that minimizes waves * tiles-per-CTA with every slot filled; each
  // extra split also pays a partial-output merge, charged as one tile.
  {
    uint32_t const nbTiles = maxNbSubSeq;
    uint32_t const slots = multiProcessorCount * mha::max(1U, ctasPerSm);
    uint32_t best = 1;
    uint64_t bestCost = ~uint64_t{0};
    for (uint32_t n = 1; n <= maxNbSubSeq; n++) {
      if (nbTiles / n < multiBlockMinNbTilesPerCta) {
        break;
      }
      uint64_t const ctas = uint64_t(batchSize) * nbKHeads * n;
      bool const fillsSlots = ctas >= slots;
      uint64_t const waves = divUp(ctas, uint64_t(slots));
      uint64_t const cost = waves * divUp(nbTiles, n) + (n > 1 ? n : 0);
      // Prefer any split that fills the machine over one that does not.
      uint64_t const rankedCost = fillsSlots ? cost : cost + (uint64_t{1} << 40);
      if (rankedCost < bestCost) {
        bestCost = rankedCost;
        best = n;
      }
    }
    return best;
  }
#endif
  float const factor = 0.25f;
  return mha::min<uint32_t>(
      mha::max<uint32_t>(
          1U, static_cast<uint32_t>(
                  round(multiProcessorCount * 3 /
                        (batchSize * nbKHeads) * factor))),
      maxNbSubSeq);
}

#if ENABLE_MIXED_KV_CACHE
struct MixedPageTensorMaps {
  CUtensorMap fp8K, fp8V, fp4K, fp4V;
};

// Byte-typed, unswizzled tensor maps over the compressed payload tensors.  A
// format whose payload is absent gets the A16 map as a placeholder; the loader
// never selects it because no page carries that tag.
static MixedPageTensorMaps makeMixedPageTensorMaps(PageTransport const& transport,
                                                   uint32_t nbKHeads,
                                                   CUtensorMap const& fallbackK,
                                                   CUtensorMap const& fallbackV) {
  auto const make = [&](flashinfer::KVPageFormatSpan const& span, bool isK,
                        uint32_t rowBytes, uint32_t partBytes, CUtensorMap const& fallback) {
    void const* const addr = isK ? span.k_payload : span.v_payload;
    if (addr == nullptr) {
      return fallback;
    }
    return makeTensorMapForPackedKVPages(addr, rowBytes, nbKHeads, tokensPerPage, partBytes,
                                         span.payload_stride.page, span.payload_stride.token,
                                         span.payload_stride.head);
  };
  auto const& fp8 = transport.formats[static_cast<uint8_t>(flashinfer::KVPageFormat::kBlockScaledFP8)];
  auto const& fp4 = transport.formats[static_cast<uint8_t>(flashinfer::KVPageFormat::kBlockScaledFP4)];
  return MixedPageTensorMaps{
      make(fp8, true, SharedMem::packedRowBytesFP8, SharedMem::packedRowBytesFP8, fallbackK),
      make(fp8, false, SharedMem::packedRowBytesFP8, SharedMem::packedRowBytesFP8, fallbackV),
      make(fp4, true, SharedMem::packedRowBytesFP4, SharedMem::packedRowBytesFP4, fallbackK),
      make(fp4, false, SharedMem::packedRowBytesFP4, SharedMem::packedRowBytesFP4, fallbackV)};
}
#endif

void launchHopperF8MHA(
    cudaDeviceProp const& prop, uint32_t nbKHeads,
#if SLIDING_WINDOW
    uint32_t slidingWinSize,
#endif
    float qScale, float const* qScalePtr, OutputHead* output,
#if LOW_PREC_OUTPUT
    float rcpOutScale,
#endif
#if USE_INPUT_KV
    InputHead const* qkv,
#if ROPE_STYLE != 0
    Vec<float, validElemsPerHead> const* ropeCosSin,
#endif
#else
    InputHead const* q,
#endif
    float const* attentionSinks,  // [headGrpSize]
    GMemCacheHead* kCacheVLLM, GMemCacheHead* vCacheVLLM,
#if ENABLE_MIXED_KV_CACHE
    PageTransport const& pageTransport,
#endif
    KVCachePageIndex const*
        kvCachePageList,  // device pointer. shape:
                          // KVCachePage[batchSize][beamWidth][2][maxNbPagesPerSeq]
    uint32_t maxSeqLen, uint32_t const* seqLen,
#if USE_BEAM_SEARCH
    BeamSearchParams const& beamSearchParams,
#endif
    uint32_t batchSize, float kvCacheScale,
    float const* kvScalePtr,  // Same scale for K and V cache. Used only for int8/fp8 KV cache.
#if SPEC_DEC
    SpecDecParams const& specDecParams,
#endif
    uint32_t* semaphores, void* scratch, bool enable_pdl, uint64_t kv_stride_page,
    uint64_t kv_stride_token, uint64_t kv_stride_head, cudaStream_t stream) {
  if (beamWidth != 1) {
    throw std::runtime_error("not implemented");
  }
  static uint32_t const hostSmemSize = [&]() {
    uint32_t size;
    checkCuda(cudaMemcpyFromSymbol(&size, smemSize, sizeof(smemSize)));
    checkCuda(cudaFuncSetAttribute(kernel_mha, cudaFuncAttributeMaxDynamicSharedMemorySize, size));
#if ENABLE_MIXED_KV_CACHE
    // Two CTAs per SM need the full shared-memory carveout; left to the driver
    // it may pick a one-block configuration.
    checkCuda(cudaFuncSetAttribute(kernel_mha, cudaFuncAttributePreferredSharedMemoryCarveout,
                                   cudaSharedmemCarveoutMaxShared));
#endif
    return size;
  }();
  // printf("smemSize = %u\n", hostSmemSize);
  uint32_t const nbVHeads = nbKHeads;
  uint32_t const nbQHeads = nbKHeads * headGrpSize;
  uint32_t const nbQKVHeads = nbQHeads + nbKHeads + nbVHeads;
  static int const ctasPerSm = [&]() {
    int n = 0;
    checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &n, kernel_mha, warp_size * gmmaWarpsPerGrp * ctaWarpGroups, hostSmemSize));
    return n;
  }();
  uint32_t const nbSubSeqPerSeq = chooseNbSubSeq(
      prop.multiProcessorCount, batchSize, nbKHeads, maxSeqLen, static_cast<uint32_t>(ctasPerSm));
#if SPEC_DEC
  uint32_t const qSeqLen = specDecParams.qSeqLen;
#else
  uint32_t const qSeqLen = 1;
#endif
  // gridDim.z == nbKHeads * batchSize && gridDim.y == nbSubSeqPerSeq && gridDim.x ==
  // nbInputSeqSplit
  dim3 const dimGrid{divUp(qSeqLen, inputTokensPerCta), nbSubSeqPerSeq, nbKHeads * batchSize};
  dim3 const dimCta{warp_size * gmmaWarpsPerGrp, 1, ctaWarpGroups};
  auto const launchCfg = makeLaunchConfig(
      dimGrid, dimCta, hostSmemSize, stream,
      enable_pdl && !ENABLE_MIXED_KV_CACHE);
  uint32_t const maxNbPagesPerSeq = exactDiv(maxSeqLen, tokensPerPage);
  auto const dtype = [] {
    if (std::is_same_v<CacheElem, half>) {
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    } else if (std::is_same_v<CacheElem, __nv_bfloat16>) {
      return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    } else if (std::is_same_v<CacheElem, __nv_fp8_e4m3>) {
      return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    }
    throw std::runtime_error("unsupported cache element type");
  }();

  KVCacheList<true> const cacheList{kCacheVLLM, vCacheVLLM,
#if ENABLE_MIXED_KV_CACHE
                                    pageTransport,
#endif
                                    kvCachePageList, seqLen, maxNbPagesPerSeq};

  auto const tensorMapVLLMK = makeTensorMapForPagedKVCache(
      kCacheVLLM, dtype, validElemsPerHead, nbKHeads, tokensPerPage, cacheHeadPartElems,
      gemm0CtaTileNbTokens, kv_stride_page, kv_stride_token, kv_stride_head);
  auto const tensorMapVLLMV = makeTensorMapForPagedKVCache(
      vCacheVLLM, dtype, validElemsPerHead, nbKHeads, tokensPerPage, cacheHeadPartElems,
      gemm0CtaTileNbTokens, kv_stride_page, kv_stride_token, kv_stride_head);
#if ENABLE_MIXED_KV_CACHE
  auto const mixedMaps =
      makeMixedPageTensorMaps(pageTransport, nbKHeads, tensorMapVLLMK, tensorMapVLLMV);
#endif

  cudaError_t const err =
      cudaLaunchKernelEx(&launchCfg, &kernel_mha, nbKHeads,
#if SLIDING_WINDOW
                         slidingWinSize,
#endif
                         qScale, qScalePtr, output,
#if LOW_PREC_OUTPUT
                         rcpOutScale,
#endif
#if USE_INPUT_KV
                         qkv,
#if ROPE_STYLE != 0
                         ropeCosSin,
#endif
#else
                         q,
#endif
                         attentionSinks, cacheList,
#if USE_BEAM_SEARCH
                         beamSearchParams,
#endif
                         batchSize, kvCacheScale, kvScalePtr, tensorMapVLLMK, tensorMapVLLMV,
#if ENABLE_MIXED_KV_CACHE
                         mixedMaps.fp8K, mixedMaps.fp8V, mixedMaps.fp4K, mixedMaps.fp4V,
#endif
#if SPEC_DEC
                         specDecParams,
#endif
                         semaphores, scratch);
  checkCuda(err);
}
#endif

static uint32_t configureKernel() {
  uint32_t size;
  cudaMemcpyFromSymbol(&size, smemSize, sizeof(smemSize));
  cudaFuncSetAttribute(kernel_mha, cudaFuncAttributeMaxDynamicSharedMemorySize, size);
#if ENABLE_MIXED_KV_CACHE
  cudaFuncSetAttribute(kernel_mha, cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
#endif
  return size;
}

static uint32_t const hostSmemSize = configureKernel();

void launchHopperF8MHAFlashInfer(
    uint32_t multiProcessorCount, uint32_t nbKHeads, uint32_t slidingWinSize, float qScale,
    float const* qScalePtr, OutputHead* output,
#if LOW_PREC_OUTPUT
    float rcpOutScale,
#endif
    InputHead const* q, float const* attentionSinks, GMemCacheHead* kCacheVLLM,
    GMemCacheHead* vCacheVLLM,
#if ENABLE_MIXED_KV_CACHE
    PageTransport const& pageTransport,
#endif
    KVCachePageIndex const* kvCachePageList, uint32_t maxSeqLen,
    uint32_t const* seqLen, uint32_t batchSize, float kvCacheScale, float const* kvScalePtr,
#if SPEC_DEC
    uint32_t qSeqLen, uint32_t const* qCuSeqLens, MaskType const* mask,
#endif
    uint32_t* semaphores, void* scratch, bool enable_pdl, uint64_t kv_stride_page,
    uint64_t kv_stride_token, uint64_t kv_stride_head, cudaStream_t stream) {
  static int const ctasPerSm = [&]() {
    int n = 0;
    checkCuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &n, kernel_mha, warp_size * gmmaWarpsPerGrp * ctaWarpGroups, hostSmemSize));
    return n;
  }();
  uint32_t const nbSubSeqPerSeq = chooseNbSubSeq(
      multiProcessorCount, batchSize, nbKHeads, maxSeqLen, static_cast<uint32_t>(ctasPerSm));
#if SPEC_DEC
  auto specDecParams = SpecDecParams{qSeqLen, qCuSeqLens, mask};
  uint32_t const qLen = qSeqLen;
#else
  uint32_t const qLen = 1;
#endif
  dim3 const dimGrid{divUp(qLen, inputTokensPerCta), nbSubSeqPerSeq, nbKHeads * batchSize};
  dim3 const dimCta{warp_size * gmmaWarpsPerGrp, 1, ctaWarpGroups};
  auto const launchCfg = makeLaunchConfig(
      dimGrid, dimCta, hostSmemSize, stream,
      enable_pdl && !ENABLE_MIXED_KV_CACHE);
  uint32_t const maxNbPagesPerSeq = exactDiv(maxSeqLen, tokensPerPage);
  auto const dtype = [] {
    if (std::is_same_v<CacheElem, half>) {
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    } else if (std::is_same_v<CacheElem, __nv_bfloat16>) {
      return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    } else if (std::is_same_v<CacheElem, __nv_fp8_e4m3>) {
      return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    }
    throw std::runtime_error("unsupported cache element type");
  }();

  KVCacheList<true> const cacheList{kCacheVLLM, vCacheVLLM,
#if ENABLE_MIXED_KV_CACHE
                                    pageTransport,
#endif
                                    kvCachePageList, seqLen, maxNbPagesPerSeq};

  auto const tensorMapVLLMK = makeTensorMapForPagedKVCache(
      kCacheVLLM, dtype, validElemsPerHead, nbKHeads, tokensPerPage, cacheHeadPartElems,
      gemm0CtaTileNbTokens, kv_stride_page, kv_stride_token, kv_stride_head);
  auto const tensorMapVLLMV = makeTensorMapForPagedKVCache(
      vCacheVLLM, dtype, validElemsPerHead, nbKHeads, tokensPerPage, cacheHeadPartElems,
      gemm0CtaTileNbTokens, kv_stride_page, kv_stride_token, kv_stride_head);
#if ENABLE_MIXED_KV_CACHE
  auto const mixedMaps =
      makeMixedPageTensorMaps(pageTransport, nbKHeads, tensorMapVLLMK, tensorMapVLLMV);
#endif

  cudaError_t const err = cudaLaunchKernelEx(&launchCfg, &kernel_mha, nbKHeads,
#if SLIDING_WINDOW
                                             slidingWinSize,
#endif
                                             qScale, qScalePtr, output,
#if LOW_PREC_OUTPUT
                                             rcpOutScale,
#endif
                                             q, attentionSinks, cacheList,
                                             batchSize, kvCacheScale,
                                             kvScalePtr, tensorMapVLLMK, tensorMapVLLMV,
#if ENABLE_MIXED_KV_CACHE
                         mixedMaps.fp8K, mixedMaps.fp8V, mixedMaps.fp4K, mixedMaps.fp4V,
#endif
#if SPEC_DEC
                                             specDecParams,
#endif
                                             semaphores, scratch);
  checkCuda(err);
}
#endif
