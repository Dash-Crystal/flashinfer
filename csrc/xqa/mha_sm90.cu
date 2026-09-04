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
#ifndef MIXED_KV_TRACE_TILES
#define MIXED_KV_TRACE_TILES 8
#endif
// Trace window: CTA whose per-tile stamps are printed, and the first CTA-local
// tile recorded (stamps cover tiles [TILE0, TILE0 + MIXED_KV_TRACE_TILES)).
#ifndef MIXED_KV_TRACE_CTA
#define MIXED_KV_TRACE_CTA 0
#endif
#ifndef MIXED_KV_TRACE_TILE0
#define MIXED_KV_TRACE_TILE0 0
#endif
// Persistent, balanced CTA scheduling (docs/mixed_kv_speed_round2_lever8.md):
// the q=1 mixed build launches gridDim.x = ctasPerSm * SMs CTAs, each owning a
// contiguous range of the linearized (request, head, tile) space.  SPEC_DEC and
// the non-mixed kernel keep the 1 x nbSubSeq x (B*H) grid.
#define MIXED_KV_PERSISTENT (ENABLE_MIXED_KV_CACHE && !SPEC_DEC)
// Metadata chunk fill lead in tiles (section 8.2 of the lever [8] design):
// chunk for tiles [16k, 16k+16) is filled at loader iteration 16k - LEAD.
// WAR-safe for LEAD <= 15; 4 gives ~2 tile periods for the page-table pair.
#ifndef MIXED_KV_META_LEAD
#define MIXED_KV_META_LEAD 4
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
// Under a static compressed format no page of the stream is A16 (slots past the
// sequence end are zero-filled by the converters), so the load warps issue no
// TMA and take no part in the stage's `produced` barrier.
constexpr bool mixedLoaderTma = !(MIXED_PAGE_STATIC_FORMAT == 1 || MIXED_PAGE_STATIC_FORMAT == 2);
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
// Per-CTA %globaltimer record of the persistent build: {start, first K ready,
// last tile done, end}, indexed by blockIdx.x (P <= 4096 assumed for the trace).
constexpr uint32_t mixedKvCtaTraceSlots = 4096;
__device__ unsigned long long mixedKvCtaTrace[mixedKvCtaTraceSlots][4];
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
  // For gmma math. Conversion done if needed.  The persistent build holds two
  // Q buffers (item j in q[j & 1]) so the Q warp stores the next item's Q while
  // the current item runs and gemm0 never waits on the Q warp at an item
  // boundary (lever [8] design, section 8.5).
  static constexpr uint32_t nbQBuf = MIXED_KV_PERSISTENT ? 2 : 1;
  QBuffer q[nbQBuf];

  // @fixme: move these into reusedXVOutSwizzleBuf
#if SWAP_AB
  ShmQWiseVec xColMax[nbXBuf];
  ShmQWiseVec xColSum[nbXBuf][gemm0NbWarps];
#else
  ShmQWiseVec xRowMax[nbXBuf];
  ShmQWiseVec xRowSum[nbXBuf];
#endif

  ShmQWiseVec gemm0CurrentSeqMax;  // !SWAP_AB running row max (SWAP_AB keeps it in registers)
  // col sum and max for the current gemm1 acc.  !SWAP_AB: updated in shared
  // memory every tile.  SWAP_AB: the running values live in registers (one
  // float per lane, identical in all gemm1 warps) and are written here once,
  // before finalize / the multi-block save.
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
#if !MIXED_KV_PERSISTENT
  Vec<KVCachePageIndex, nbPagesPerTile> pages[2];  // one for K and one for V
#endif
#if ENABLE_MIXED_KV_CACHE
  // Compressed pages are delivered by TMA into dense per-page slots (one
  // 16-token slot per page, sized for the E4M3 part row; E2M1 uses the first
  // half of each row).  Converter warps expand them into k[]/vBuf(); A16 pages
  // are delivered by TMA straight into k[]/vBuf() and never touch these slots.
  // A compressed page is fetched as whole-head rows once per tile (64 B E2M1 /
  // 128 B E4M3 per token for D=128): one TMA box per page, full DRAM bursts.
  static constexpr uint32_t packedRowBytesFP8 = headElems;
  static constexpr uint32_t packedRowBytesFP4 = headElems / 2;
  // E2M1 rows are stored with a 5-chunk stride (bank group 5 * token + chunk
  // walks all 8 groups over 8 consecutive tokens, so the converter's reads and
  // the copies are conflict-free without a swizzle); 16 x 80 B fit the slot.
  static constexpr uint32_t packedRowStrideFP4 = packedRowBytesFP4 + 16;
  static constexpr uint32_t packedSlotBytes = tokensPerPage * packedRowBytesFP8;
  using PackedTile = uint8_t[nbPagesPerTile][packedSlotBytes];
  // Compressed rows are delivered by TMA *into the stage's last head-part
  // buffer* (a whole-head packed tile is exactly one part buffer: 64 rows x
  // 128 B) with the 128B (E4M3) / 64B (E2M1) swizzle, and expanded in place:
  // the converter group reads every packed word it needs, barriers, then
  // writes the A16 rows of every part.  No separate packed buffers.
  static_assert(sizeof(PackedTile) == sizeof(KBuffer));
  // Per-tile metadata records for chunks of this CTA's tiles, filled by the
  // load warp once per chunk so no per-tile dependent global load sits on its
  // critical path.  Two chunks per operand because copies are issued two
  // tiles ahead and may cross into the next chunk while this one is live.
  // The record carries everything a tile needs from its work item (lever [8],
  // section 2.2): the stage rings, converters and barriers see only the
  // CTA-local tile counter g; items exist only in the four walker warps.
  static constexpr uint32_t metaChunkTiles = 16;
  static constexpr uint32_t nbMetaChunks = 2;
  struct alignas(16) TileRecord {
    KVCachePageIndex pages[nbPagesPerTile];  // +0   kBAD_PAGE_INDEX past the sequence end
    uint32_t formats;                        // +16  byte j: tag of page j (kMixedBadPageFormat past the end)
    uint32_t tile;                           // +20  bits 0-7 validBeg, 8-15 validEnd, 16 first, 17 last,
                                             //      18 itemIsPartial, 19 itemIsCtaLast
    uint32_t idxReq;                         // +24
    uint32_t idxHeadGrp;                     // +28
  };
  static_assert(sizeof(TileRecord) == 32 && nbPagesPerTile == 4);
  static constexpr uint32_t tileFirstBit = 1u << 16;
  static constexpr uint32_t tileLastBit = 1u << 17;
  static constexpr uint32_t tilePartialBit = 1u << 18;
  static constexpr uint32_t tileCtaLastBit = 1u << 19;
  TileRecord meta[2][nbMetaChunks][metaChunkTiles];
#if MIXED_KV_PERSISTENT
  // Prologue scan result (IO warp 3, before the first __syncthreads): this
  // CTA's linear tile range and the position of its first tile.
  struct PersistentSched {
    uint32_t x0, x1;        // [x0, x1) in the linearized (request, head, tile) space
    uint32_t nbTotalTiles;  // T (0 when every sequence is empty)
    uint32_t req0, head0, tile0;  // sequence and in-use tile index holding x0
    uint32_t Lseq0;         // linear start of sequence (req0, head0)
    uint32_t seqLen0;       // cacheSeqLen(req0)
    uint32_t seqLen1;       // cacheSeqLen(req0 + 1), 0 if req0 + 1 == batchSize
  };
  PersistentSched sched;
  // Number of this CTA's items whose gemm1 finalize has completed (monotone;
  // st.release.cta by gemm1 thread 0, ld.acquire.cta by the merge warp).
  uint32_t finalizedItems;
#endif
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
  // Number of tiles [MIXED_KV_TRACE_TILE0, +n) of CTA MIXED_KV_TRACE_CTA whose
  // per-role stamps are recorded (-DMIXED_KV_TRACE_TILES=n; +128 B of shared
  // memory per tile).
  static constexpr uint32_t nbTraceTiles = MIXED_KV_TRACE_TILES;
  long long trace[nbTraceTiles][16];
  // SM residency probe: this CTA's %smid and the number of CTAs of this kernel
  // resident on that SM when tile 4's slot 0 stamp is taken (includes self).
  uint32_t traceSmId;
  uint32_t traceResidentMid;
#endif
#endif

  // mem barriers

  CtaBarrierPair qBar[nbQBuf];
  CtaBarrierPair kBar[nbKBuf];
  CtaBarrierPair vBar[nbVBuf];
#if ENABLE_MIXED_KV_CACHE
  // One per metadata chunk per operand (count = the load warp): the loader
  // arrives after each fill of the chunk, the converter warps wait once per
  // chunk (at the first tile whose metadata lives in it) before reading page
  // indices and tags from it.  Fill f of a chunk completes phase f.
  CtaBarrier kMetaReady[nbMetaChunks];
  CtaBarrier vMetaReady[nbMetaChunks];
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

#if !MIXED_KV_PERSISTENT
  bool isLastCta;
#endif
};

CUBIN_EXPORT __device__ constexpr uint32_t smemSize = sizeof(SharedMem);
#ifdef __CUDA_ARCH__
static_assert(smemSize < kMAX_SMEM_SIZE);
#if MIXED_KV_PERSISTENT
// Two CTAs per SM (H100/H200: 233472 B of shared memory per SM, 1 KB reserved
// per CTA) is the premise of the register split and of P = 2 x SMs.
static_assert(smemSize + 1024 <= 233472 / 2, "mixed persistent build must keep 2 CTAs/SM");
#endif
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
// Expand every compressed page of a whole K/V stage in place.  The packed
// rows live in the stage's last head-part buffer; A16 pages of the same tile
// were placed by TMA in their final rows and are skipped; slots past the
// sequence end are zero-filled so a masked P=0 never multiplies stale memory.
//
// The converter group is instruction-issue-bound (two CTAs share the SM's
// four schedulers with the GMMA groups), so the cut minimizes instructions per
// value: lane l owns one (token, head part) = 64 values = 4 blocks, paying for
// its 4-byte scale word and its row addresses once.  Lanes 0-15 of a warp are
// the 16 tokens of one page at head part 0, lanes 16-31 the same tokens at
// part 1, so the format branch is warp-uniform and every 8-lane phase of an
// LDS.128 / STS.128 covers 8 consecutive tokens of one part:
//  - A16 rows (128 B TMA swizzle, chunk ^= token % 8): 8 distinct bank groups
//    per phase.  The 8 store addresses of a lane are one row base whose bits
//    [6:4] hold token % 8, XORed with the immediate (2b + g) * 16 (one LOP3 each).
//  - E4M3 packed rows (same swizzle, issued by issueCompressedPageCopies):
//    block b of part p sits at chunk (4p + b) ^ (token % 8), i.e. the lane's
//    row base (bits [6:4] = 4p ^ token % 8) XOR b * 16: conflict-free reads.
//  - E2M1 packed rows have an 80 B stride (5 chunks, the 5th unused): the
//    lane's 4 blocks are 32 contiguous bytes at immediate offsets, and across 8
//    consecutive tokens the bank group 5 * token + c walks all 8 groups.
// All per-lane offsets are computed once per warp (ExpandLane) and kept in
// registers; per tile the only address work is one add of the stage base per
// stream plus the store/read XORs.  Because the last part's A16 rows overwrite
// the packed rows of the same page, every read precedes a warp sync and every
// write follows it.
//
// On sm90 with BF16 math the E4M3 and E2M1 decodes are bit placements (see
// mhaUtils.cuh): the value lands in a BF16 lane scaled by 2^-120 (E4M3) or
// 2^-126 (E2M1) and the power of two is folded into the block scale when every
// scale of the warp's tile stays finite after the fold (|s * global| < 255.5
// for E4M3, < 4 for E2M1; one warp vote per tile), else one extra packed
// multiply per pair undoes it.  Both forms are bit-exact against
// bf16(float(scale) * global) * bf16(value) as long as the product
// scale * global is a normal fp32 number, which the host-side global scale
// guarantees for |global| >= 2^-117 (checked once per warp; smaller values take
// the two-multiply form).
struct ExpandLane {
  uint32_t a16;    // this lane's A16 row in parts[p] (bits [6:4] = token % 8), bytes from parts[0]
  uint32_t fp8;    // chunk (4p) ^ (token % 8) of the lane's packed E4M3 row, bytes from parts[0]
  uint32_t fp4;    // the lane's 32 packed E2M1 bytes (80 B row stride), bytes from parts[0]
  uint32_t scale;  // the lane's 4 block scales, bytes from the tile's TileScales
};

struct ExpandScales {
  float fp8Global;      // per-format global scale
  float fp4Global;
  float fp8GlobalFold;  // global * 2^120 (E4M3 fold) / global * 2^126 (E2M1 fold)
  float fp4GlobalFold;
  bool fp8FoldOk;       // |global| >= 2^-117: every block scale * global is fp32-normal
  bool fp4FoldOk;
};

template <typename PartBuf>
__device__ __forceinline__ ExpandLane makeExpandLane(PartBuf const* /*stage*/, uint32_t idxWarp) {
  constexpr uint32_t nbThreads = convertWarpsPerOperand * warp_size;
  static_assert(gemm0CtaTileNbTokens * cacheHeadNbParts == nbThreads,
                "one converter lane per (token, head part)");
  static_assert(cacheHeadNbParts == 2 && tokensPerPage == 16 && warp_size == 32,
                "lanes 0-15 / 16-31 are one page's tokens at part 0 / 1");
  static_assert(SharedMem::nbPagesPerTile == convertWarpsPerOperand, "one warp per page");
  static_assert(sizeof(PartBuf) == gemm0CtaTileNbTokens * 128 && PartBuf::rowBytes == 128 &&
                    PartBuf::cols == 8 && sizeof(PartBuf) % 128 == 0,
                "128 B swizzled A16 rows");
  static_assert(SharedMem::packedRowBytesFP8 == 128 && SharedMem::packedSlotBytes == 2048 &&
                    SharedMem::scaleBytesPerToken == 8,
                "E4M3 rows are whole 128 B swizzled rows; 16 x 8 B scales per token");
  static_assert(tokensPerPage * SharedMem::packedRowStrideFP4 <= SharedMem::packedSlotBytes);
  uint32_t const lane = laneId();
  uint32_t const tokenInPage = lane % tokensPerPage;
  uint32_t const p = lane / tokensPerPage;
  uint32_t const token = idxWarp * tokensPerPage + tokenInPage;
  uint32_t const x = token % 8;
  constexpr uint32_t packedBase = (cacheHeadNbParts - 1) * sizeof(PartBuf);
  ExpandLane l;
  l.a16 = p * sizeof(PartBuf) + token * PartBuf::rowBytes + x * 16;
  l.fp8 = packedBase + idxWarp * SharedMem::packedSlotBytes +
          tokenInPage * SharedMem::packedRowBytesFP8 + ((4 * p) ^ x) * 16;
  l.fp4 = packedBase + idxWarp * SharedMem::packedSlotBytes +
          tokenInPage * SharedMem::packedRowStrideFP4 + p * 32;
  l.scale = token * SharedMem::scaleBytesPerToken + p * 4;
  return l;
}

__device__ __forceinline__ ExpandScales makeExpandScales(float fp8Global, float fp4Global) {
  ExpandScales s;
  s.fp8Global = fp8Global;
  s.fp4Global = fp4Global;
  s.fp8GlobalFold = fp8Global * 0x1p120f;
  s.fp4GlobalFold = fp4Global * 0x1p126f;
  s.fp8FoldOk = fabsf(fp8Global) >= 0x1p-117f;
  s.fp4FoldOk = fabsf(fp4Global) >= 0x1p-117f;
  return s;
}

template <typename PartBuf>
__device__ __forceinline__ void expandPackedStage(PartBuf* parts,
                                                  SharedMem::TileScales const& scales,
                                                  uint8_t tag, ExpandLane const& lane,
                                                  ExpandScales const& gs);
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
  uint32_t const nbPages;  // for bound check
  Vec<KVCachePageIndex, nbPagesPerTile>& pages;
  uint32_t idxTileRef;  // idxTile used to load the pages
  uint32_t const baseOffset;

  __device__ KVTilePartLoader(bool isK, uint32_t nbKHeads,
                              KVCacheList<usePagedKVCache> const& cacheList, uint32_t idxReq,
                              uint32_t idxHeadGrp, CUtensorMap const& tensorMap, uint32_t nbPages,
                              Vec<KVCachePageIndex, nbPagesPerTile>& pageBuf);
  // tensorMap is for one whole page ([nbKHeads*tokensPerPage][headElems]) or whole cache
  template <uint32_t nbTokens, bool alignedForSwizzle>
  __device__ void loadData(
      Array2D<LdGrain, nbTokens, exactDiv(cacheHeadPartBytes, grainBytes), alignedForSwizzle>& dst,
      uint32_t idxTile, uint32_t idxPart, CtaBarrier& bar);

  __device__ void loadPages(uint32_t idxTile, bool publish = true);
  __device__ GMemKVCacheHead& getHead(uint32_t pos);
};

#if MIXED_KV_PERSISTENT
// ---------------------------------------------------------------------------
// Persistent, balanced CTA scheduling (lever [8]).  Data flow and control flow
// are specified in docs/mixed_kv_speed_round2_lever8.md (sections 2, 8, 9);
// the summary that the code below implements:
//
//   linear tile space  x(r, h, t) = H * prefix(r) + h * tiles(r) + t,  T = H * prefix(B)
//   CTA c owns          [x_c, x_{c+1}),  x_c = ceil(c * T / P)          (64-bit products)
//   items               maximal runs of a CTA's range inside one (r, h); only the CTA's first
//                       and last item can be partial (scratch chunk 2c + isCtaLast)
//   walkers             IO warp 0 (K fill), 1 (V fill), 2 (Q), 3 (scan + merge) each hold one
//                       ItemCursor in registers; the same ItemCursor::next() produces the
//                       same item sequence in all four (clip limit = chunk end for the fills)
//   records             fill -> smem.meta[op][chunk][g % 16] (32 B) -> kMetaReady[chunk].arrive
//                       readers: converters at issue(g) after their parity wait; gemm0 after
//                       kBar[g%3].produced; gemm1 after vBar[g%3].produced; TMA loader at g
//   per-tile flags      first = (x == x0) || (t == 0); last = (x+1 == xEnd) || (t+1 == tiles)
//                       partial = !(Lseq >= x0 && Lseq + tiles <= xEnd); ctaLast = Lseq + tiles >= xEnd
//                       validBeg = (t == 0) ? tile0Skip : 0; validEnd = last-in-seq ? len % 64 : 64
//   chunk fills         chunk 0 in the prologue; chunk for tiles [16k, 16k+16) at loader
//                       iteration 16k - MIXED_KV_META_LEAD, exactly once per 32 tiles
//   merge               gemm1 finalize -> st.release.cta finalizedItems = j+1 ; merge warp
//                       ld.acquire.cta -> atom.acq_rel.gpu.inc semaphores[H*req+head] ->
//                       last arriver reads chunks {2c+1 | c0 <= c < c1} + {2c1 + (x_{c1+1} == Lend)}
// ---------------------------------------------------------------------------

// Sliding-window bookkeeping per sequence length (the only per-request inputs).
__device__ __forceinline__ uint32_t seqSkipTokens(uint32_t seqLen, uint32_t slidingWinSize) {
#if SLIDING_WINDOW
  return seqLen > slidingWinSize ? seqLen - slidingWinSize : 0;
#else
  (void)slidingWinSize;
  return 0;
#endif
}
// Tiles in use of a sequence: divUp(len, 64) minus wholly skipped leading tiles (0 for len 0).
__device__ __forceinline__ uint32_t seqTilesInUse(uint32_t seqLen, uint32_t slidingWinSize) {
  return divUp(seqLen, gemm0CtaTileNbTokens) -
         seqSkipTokens(seqLen, slidingWinSize) / gemm0CtaTileNbTokens;
}

// One piece of the item stream: the current item clipped to a limit.
struct ItemPiece {
  uint32_t req, head;
  uint32_t xBeg;       // linear index of the piece's first tile
  uint32_t tileInSeq;  // in-use tile index (0-based) of the piece's first tile
  uint32_t nb;         // tiles in the piece (>= 1)
  uint32_t tiles;      // tiles in use of the sequence
  uint32_t seqLen;
  bool partial;        // the item does not cover its whole sequence
  bool ctaLast;        // the item is the CTA's last item
};

// Register-resident walker of one CTA's items (warp-uniform; 9 registers).
struct ItemCursor {
  uint32_t x, xEnd, x0;
  uint32_t req, head;
  uint32_t tileInSeq;
  uint32_t seqLen;
  uint32_t nextSeqLen;
  uint32_t Lseq;

  __device__ __forceinline__ static ItemCursor init(SharedMem::PersistentSched const& s) {
    ItemCursor c;
    c.x = s.x0;
    c.xEnd = s.x1;
    c.x0 = s.x0;
    c.req = s.req0;
    c.head = s.head0;
    c.tileInSeq = s.tile0;
    c.seqLen = s.seqLen0;
    c.nextSeqLen = s.seqLen1;
    c.Lseq = s.Lseq0;
    return c;
  }
  __device__ __forceinline__ bool done() const { return x >= xEnd; }
  // The current item clipped to [x, min(xEnd, limit)); advances the cursor past it.
  __device__ __forceinline__ ItemPiece next(uint32_t limit,
                                            KVCacheList<usePagedKVCache> const& cacheList,
                                            uint32_t nbKHeads, uint32_t batchSize,
                                            uint32_t slidingWinSize) {
    assert(!done() && limit > x);
    uint32_t const tl = seqTilesInUse(seqLen, slidingWinSize);
    assert(tl > tileInSeq);
    ItemPiece p;
    p.req = req;
    p.head = head;
    p.xBeg = x;
    p.tileInSeq = tileInSeq;
    p.tiles = tl;
    p.seqLen = seqLen;
    uint32_t const end = mha::min(xEnd, limit);
    p.nb = mha::min(tl - tileInSeq, end - x);
    // Only the CTA's first item can start mid-sequence (Lseq < x0).
    p.partial = !(Lseq >= x0 && Lseq + tl <= xEnd);
    p.ctaLast = (Lseq + tl >= xEnd);
    x += p.nb;
    tileInSeq += p.nb;
    if (tileInSeq == tl) {
      tileInSeq = 0;
      Lseq += tl;
      head++;
      if (head == nbKHeads) {
        head = 0;
        req++;
        seqLen = nextSeqLen;
        nextSeqLen = (req + 1 < batchSize) ? getCacheSeqLen(cacheList, req + 1) : 0;
        // Requests without tiles in use are skipped (x < xEnd guarantees a
        // later request has tiles, so this stops before req == batchSize
        // whenever the cursor is not done).
        while (req < batchSize && seqTilesInUse(seqLen, slidingWinSize) == 0) {
          req++;
          seqLen = nextSeqLen;
          nextSeqLen = (req + 1 < batchSize) ? getCacheSeqLen(cacheList, req + 1) : 0;
        }
      }
    }
    return p;
  }
};

// Shared-window address of this operand's record for CTA tile g:
// meta[op][(g/16)%2][g%16] == meta[op][0][g % 32] since nbMetaChunks * metaChunkTiles == 32.
__device__ __forceinline__ uint32_t tileRecordAddr(SharedMem const& smem, uint32_t operand,
                                                   uint32_t g) {
  static_assert(SharedMem::nbMetaChunks * SharedMem::metaChunkTiles == 32);
  return static_cast<uint32_t>(__cvta_generic_to_shared(&smem.meta[operand][0][0])) +
         (g % 32) * uint32_t{sizeof(SharedMem::TileRecord)};
}
__device__ __forceinline__ uint32_t ldsU32(uint32_t addr) {
  return *reinterpret_cast<uint32_t const*>(__cvta_shared_to_generic(addr));
}
__device__ __forceinline__ uint2 ldsU64(uint32_t addr) {
  return *reinterpret_cast<uint2 const*>(__cvta_shared_to_generic(addr));
}
__device__ __forceinline__ uint4 ldsU128(uint32_t addr) {
  return *reinterpret_cast<uint4 const*>(__cvta_shared_to_generic(addr));
}
__device__ __forceinline__ void stsU32(uint32_t addr, uint32_t v) {
  *reinterpret_cast<uint32_t*>(__cvta_shared_to_generic(addr)) = v;
}
__device__ __forceinline__ void stsU128(uint32_t addr, uint4 v) {
  *reinterpret_cast<uint4*>(__cvta_shared_to_generic(addr)) = v;
}

// Prologue scan (one warp, before the first __syncthreads): T, this CTA's
// range and the position of its first tile, published to smem.sched.
__device__ inline void persistentPrologueScan(SharedMem& smem,
                                              KVCacheList<usePagedKVCache> const& cacheList,
                                              uint32_t nbKHeads, uint32_t batchSize,
                                              uint32_t slidingWinSize);
// Fill this operand's chunk for CTA tiles [gBeg, gBeg + 16) from the cursor
// (which must stand at linear tile x0 + gBeg); warp-wide.
__device__ inline void fillTileMeta(SharedMem& smem, uint32_t operand, uint32_t gBeg,
                                    ItemCursor& cur,
                                    KVCacheList<usePagedKVCache> const& cacheList,
                                    uint32_t nbKHeads, uint32_t batchSize,
                                    uint32_t slidingWinSize);
__device__ __forceinline__ uint32_t issueCompressedPageCopies(
    KVCacheList<usePagedKVCache> const& cacheList, bool isK, SharedMem const& smem,
    uint32_t operand, uint32_t idxIter, SharedMem::PackedTile& dstPacked,
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

// Register-A (RS) wgmma operand loaders for mixed pages were removed (plan
// lever [35], closing [18]): decoding E4M3/E2M1 K/V fragments into registers
// per k-step costs 35-50 SASS per step against a 10-20 cycle m64n8k16 wgmma,
// i.e. +1.2-1.4 us per group per tile at 4 cyc/instr and 2-3x that at the
// measured 10.8-13.3, on the binding gemm0 chain.  Expansion stays in the
// converter warp groups (expandPackedStage); see
// docs/mixed_kv_page_transport_backends.md, "RS-decode in the GEMM groups".

#if SWAP_AB
constexpr uint32_t gemm1NbGmmaInstM = exactDiv(headElems, gmma::instM);
__device__ Vec<RegMatAFrag, gemm1NbGmmaInstM> loadVTileTransposed(uint32_t warpRank, uint32_t lane,
                                                                  SharedMem::VBuffer const& smemV,
                                                                  uint32_t idxGmmaInstK);
using Gemm1Acc = GmmaAcc<headElems, ctaNbQHeads>;
// One column-wise float per lane without duplication: lane l <-> column l,
// lanes >= ctaNbQHeads hold 0 (the layout of loadShmColWiseVecNoDup).
using RegColWiseVecNoDup = Vec<float, divUp(ShmQWiseVec::size, warp_size)>;
__device__ RegColWiseVecNoDup loadShmColWiseVecNoDup(ShmQWiseVec const& shmVec);
__device__ void storeShmColWiseVecNoDup(ShmQWiseVec& shmVec, RegColWiseVecNoDup const& src);
__device__ inline RegColWiseVecNoDup initRegColWiseVecNoDup(float v) {
  RegColWiseVecNoDup ret;
#pragma unroll
  for (uint32_t i = 0; i < ret.size; i++) {
    uint32_t const idx = i * warp_size + laneId();
    bool const inBound = ((ShmQWiseVec::size % warp_size == 0) || (idx < ShmQWiseVec::size));
    ret[i] = inBound ? v : 0.f;
  }
  return ret;
}
__device__ void rescaleGemm1AccForNewColMax(ShmQWiseVec const& shmXColMax,
                                            ShmQWiseVec const (&shmXColSum)[gemm0NbWarps],
                                            RegColWiseVecNoDup& accColMax, Gemm1Acc& acc,
                                            RegColWiseVecNoDup& accColSum);
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
  static_assert(gemm0CtaTileNbTokens == gemm1CtaTileNbTokens);
  constexpr uint32_t tileSize = gemm0CtaTileNbTokens;
  // Inside this body the mixed build is always the persistent one (the mixed
  // SPEC_DEC combination is not a supported sm90 case and is not compiled here).
  static_assert(!ENABLE_MIXED_KV_CACHE || MIXED_KV_PERSISTENT);
#if MIXED_KV_PERSISTENT
  // Persistent grid {P, 1, 1}: no per-CTA request/head; the work decomposition
  // is computed by the prologue scan below and carried in per-tile records.
  static_assert(!USE_INPUT_KV && !USE_BEAM_SEARCH && beamWidth == 1);
  assert(gridDim.y == 1 && gridDim.z == 1);
#if SLIDING_WINDOW
  uint32_t const slidingWinSizeArg = slidingWinSize;
#else
  constexpr uint32_t slidingWinSizeArg = 0;
#endif
#else
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
#endif  // MIXED_KV_PERSISTENT
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
  // Per-tile stamps for CTA-local tiles [MIXED_KV_TRACE_TILE0, +nbTraceTiles);
  // every CTA records into its own shared memory, CTA MIXED_KV_TRACE_CTA prints.
  auto const traceStamp = [&](uint32_t slot, uint32_t idxIter, bool roleWarp0) {
    uint32_t const idx = idxIter - MIXED_KV_TRACE_TILE0;
    if (roleWarp0 && laneId() == 0 && idx < SharedMem::nbTraceTiles) {
      smem.trace[idx][slot] = clock64();
      if (slot == 0 && idx == 4) {
        smem.traceResidentMid = atomicAdd(&mixedKvTraceSmResident[smem.traceSmId], 0U);
      }
    }
  };
#define TRACE_STAMP(slot, it, w0) traceStamp((slot), (it), (w0))
#if MIXED_KV_PERSISTENT
  // Per-CTA %globaltimer record {start, first K ready, last tile done, end}.
  auto const ctaStamp = [&](uint32_t slot) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    mixedKvCtaTrace[blockIdx.x % mixedKvCtaTraceSlots][slot] = t;
  };
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    ctaStamp(0);
  }
#endif
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
  // Mixed build: a stage's `produced` phase completes when the GEMM group has
  // arrived (arrive_and_wait), every converter lane has arrived after its
  // in-place expansion, and - in builds where the load warp issues A16 TMA -
  // the load warp has arrived (expect_tx + 32) and the TMA bytes have landed.
  // One wait in the GEMM group therefore covers TMA landing and expansion.
  constexpr uint32_t mixedProducedExtra =
      ENABLE_MIXED_KV_CACHE
          ? convertWarpsPerOperand * warp_size +
                (mixedLoaderTma ? mixedLoadWarpsPerOperand * warp_size : 0)
          : 0;
  if (wid < nbKBarWarps) {
    if (warpElectSync()) {
      smem.kBar[wid].initialize(
          gemm0NbThrds + mixedProducedExtra,
          gemm0NbThrds +
              (ENABLE_MIXED_KV_CACHE ? mixedLoadWarpsPerOperand * warp_size
                                     : warp_size));
    }
  } else if (wid < nbKBarWarps + nbVBarWarps) {
    uint32_t const i = wid - nbKBarWarps;
    if (warpElectSync()) {
      smem.vBar[i].initialize(
          gemm1NbThrds + mixedProducedExtra,
          gemm1NbThrds +
              (ENABLE_MIXED_KV_CACHE ? mixedLoadWarpsPerOperand * warp_size
                                     : warp_size));
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
#pragma unroll
      for (uint32_t b = 0; b < SharedMem::nbQBuf; b++) {
        smem.qBar[b].initialize(gemm0NbThrds + nbQLdThrds, gemm0NbThrds + nbQLdThrds);
      }
      init(&smem.gemm0WarpGrpBar, gemm0NbThrds);
      init(&smem.gemm1WarpGrpBar, gemm1NbThrds);
#if ENABLE_MIXED_KV_CACHE
#pragma unroll
      for (uint32_t c = 0; c < SharedMem::nbMetaChunks; c++) {
        init(&smem.kMetaReady[c], mixedLoadWarpsPerOperand * warp_size);
        init(&smem.vMetaReady[c], mixedLoadWarpsPerOperand * warp_size);
      }
#endif
    }
  }
#if MIXED_KV_PERSISTENT
  // IO warp 3 (the merge warp) runs the prologue scan; thread 0 zeroes the
  // item counter.  Both precede the __syncthreads that every warp reaches.
  constexpr uint32_t persistentScanWid = 2 * 4 + 3;
  if (wid == persistentScanWid) {
    persistentPrologueScan(smem, cacheList, nbKHeads, batchSize, slidingWinSizeArg);
  }
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    smem.finalizedItems = 0;
  }
#endif
#if MIXED_KV_TRACE
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    smem.traceSmId = mixedKvTraceSmId();
    atomicAdd(&mixedKvTraceSmResident[smem.traceSmId], 1U);
  }
#endif
  __syncthreads();

  constexpr bool isKVCacheQuantized = (cacheElemSize < 2);
#if MIXED_KV_PERSISTENT
  // Every role's loop bound: this CTA's tiles G = x1 - x0 (warp-uniform LDS).
  uint32_t const nbCtaTiles = smem.sched.x1 - smem.sched.x0;
#else
  uint32_t const nbPages = divUp(cacheSeqLen, tokensPerPage);
  assert(idxKTileInit < nbTiles);
  uint32_t const nbIters = divUp(nbTiles - idxKTileInit, nbSubSeq);
  assert(nbIters >= 1);
#endif

  constexpr uint32_t gmmaInstK = gmma::instK<MathElem>;
  constexpr uint32_t grainsPerInstK = exactDiv(sizeof(MathElem) * gmmaInstK, grainBytes);

#if ENABLE_MIXED_KV_CACHE
  // Per-role register budget.  setmaxnreg.inc blocks until the CTA's pool has
  // the registers, and the pool is the launch allocation (640 x 48 = 30720),
  // so the split must balance within it.  ptxas also refuses a .dec below the
  // role's own register need and then drops EVERY setmaxnreg in the kernel
  // (C7507 "'setmaxnreg' ignored to maintain minimum register requirements"):
  // the former 24-register IO budget triggered exactly that, so the converters
  // ran at 48 and spilled.  IO fits in 40 (ptxas -v: no spill), so the GEMM
  // groups and the IO group each release 8 x 128 = 1024 (3072 total) and the two
  // converter groups take 8 x 256 = 2048:
  // 3*128*40 + 2*128*56 = 29696 <= 30720.  Verify with ptxas -v: no C7507, and
  // cuobjdump -sass shows two USETMAXREG (DEALLOC 0x28, TRY_ALLOC 0x38).
  static_assert(ctaWarpGroups == 5);
  if (warpIdx.z <= 2) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
  } else {
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

    // Phase-0 "free" arrivals: every Q buffer and every K stage starts free.
    for (auto& b : smem.qBar) {
      unused(b.consumed.arrive());
    }
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

#if MIXED_KV_PERSISTENT
    // gemm0 item state: j counts items (Q buffer j & 1); the Q wait sits on the
    // first tile of each item, after that tile's K wait (design section 8.1).
    uint32_t idxItem = 0;
#else
    smem.qBar[0].produced.arrive_and_wait();
#endif
#if DBG_PRINT
    if (threadIdx.x == 0) {
      printf("q:\n");
      dbg::printArray2D<__nv_fp8_e4m3, true>(smem.q[0][0]);
    }
#endif

    auto const matDescQBase =
        gmma::makeMatDesc(nullptr, 0, SharedMem::QBuffer::Elem::rowBytes * 8,
                          gmma::getSwizzleMode<true>(SharedMem::QBuffer::Elem{}))
            .raw();
#if MIXED_KV_PERSISTENT
    for (uint32_t idxIter = 0; idxIter < nbCtaTiles; idxIter++) {
#else
    for (uint32_t idxIter = 0; idxIter < nbIters; idxIter++) {
      uint32_t const idxKTile = idxKTileInit + idxIter * nbSubSeq;
      assert(idxKTile < nbTiles);
#endif
      Acc acc;  // no need to initialize. GMMA allows us to ignore acc initial values.
      gmma::fence();
      static_assert(cacheHeadNbParts == nbQParts);
#if ENABLE_MIXED_KV_CACHE
      uint32_t const idxKStage = idxIter % SharedMem::nbKBuf;
      smem.kBar[idxKStage].produced.arrive_and_wait();
      TRACE_STAMP(0, idxIter, warpRank == 0);
#endif
#if MIXED_KV_PERSISTENT
      // Tile record (C10: visible after the K wait through loader -> kMetaReady
      // -> converter -> kBar.produced).  Read before this tile's consumed.arrive
      // so the loader's chunk refill (WAR) is ordered after it.
      uint32_t const tileWord = ldsU32(tileRecordAddr(smem, /*operand=*/0, idxIter) + 20);
      bool const tileFirst = (tileWord & SharedMem::tileFirstBit) != 0;
      bool const tileLast = (tileWord & SharedMem::tileLastBit) != 0;
      uint32_t const idxQBuf = idxItem & 1U;
      if (tileFirst) {
        runningColMax = RegColWiseVec::filled(Vec<float, 2>::filled(safeInitRowMax));
        smem.qBar[idxQBuf].produced.arrive_and_wait();
      }
#if MIXED_KV_TRACE
      if (idxIter == 0 && warpRank == 0 && laneId() == 0) {
        ctaStamp(1);
      }
#endif
#else
      constexpr uint32_t idxQBuf = 0;
#endif
#pragma unroll
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
        //     dbg::printArray2D<__nv_fp8_e4m3, true>(smem.q[idxQBuf][idxPart]);
        //     printf("k:\n");
        //     dbg::printArray2D<__nv_fp8_e4m3, true>(kBuf);
        // }
        constexpr uint32_t nbGmmaInstK = exactDiv(cacheHeadPartElems, gmmaInstK);
#pragma unroll
        for (uint32_t k = 0; k < nbGmmaInstK; k++) {
          bool const accHasVal = (idxPart != 0 || k != 0);
          auto const matDescQ =
              addAddr(matDescQBase, &smem.q[idxQBuf][idxPart](0, grainsPerInstK * k));
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
#elif MIXED_KV_PERSISTENT
      {
        // Mask bounds from the record: validBeg > 0 only on a sliding-window
        // sequence's first tile, validEnd < 64 only on a sequence's last tile.
        uint32_t const validTokenBeg = tileWord & 0xFFU;
        uint32_t const validTokenEnd = (tileWord >> 8) & 0xFFU;
        if (validTokenBeg > 0 || validTokenEnd < tileSize) {
          warpGrpApplyMask(warpRank, acc, validTokenBeg, validTokenEnd);
        }
      }
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
#if MIXED_KV_PERSISTENT
      // End of item: release its Q buffer to the Q warp (qBar[b].consumed phase
      // m completes with this arrive for item 2m+b-2 and the Q warp's for 2m+b).
      if (tileLast) {
        unused(smem.qBar[idxQBuf].consumed.arrive());
        idxItem++;
      }
#endif
    }
#if !MIXED_KV_PERSISTENT
    unused(smem.qBar[0].consumed.arrive());
#endif
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

#if SWAP_AB
    // Running column max / sum of the accumulator, register-resident (lever
    // [1]): every warp holds the same copy and updates it from the same
    // xColMax / xColSum values in the same order, so the per-tile exchange
    // through smem.gemm1AccColMax/Sum and its two group syncs are gone; the
    // smem copies are written once before finalize.
    auto accColMax = initRegColWiseVecNoDup(safeInitRowMax);
    auto accColSum = initRegColWiseVecNoDup(0.f);
#else
    if (threadIdx.x < smem.gemm1AccColMax.size) {
      auto const idx = threadIdx.x;
      smem.gemm1AccColMax[idx] = safeInitRowMax;
      smem.gemm1AccColSum[idx] = 0;
    }
    smem.gemm1WarpGrpBar.arrive_and_wait();
#endif

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
#if MIXED_KV_PERSISTENT
    // gemm1 item state: j counts items (finalizedItems = j + 1 after item j).
    uint32_t idxItem = 0;
    for (uint32_t idxIter = 0; idxIter < nbCtaTiles; idxIter++) {
#else
    for (uint32_t idxIter = 0; idxIter < nbIters; idxIter++) {
      uint32_t idxVTile = idxVTileInit + idxIter * nbSubSeq;
#endif
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
#if MIXED_KV_PERSISTENT
      // Tile record (C10 through the V loader -> vMetaReady -> V converters ->
      // vBar.produced chain); read before this tile's xBar.consumed.arrive.
      uint32_t const recAddr = tileRecordAddr(smem, /*operand=*/1, idxIter);
      uint32_t const tileWord = ldsU32(recAddr + 20);
      bool const tileFirst = (tileWord & SharedMem::tileFirstBit) != 0;
      bool const tileLast = (tileWord & SharedMem::tileLastBit) != 0;
      if (tileFirst) {
        // New item: the accumulator and its running max / sum restart (the
        // gmma::fence before this tile's first PV wgmma covers these writes).
        acc = Gemm1Acc{};
        accColMax = initRegColWiseVecNoDup(safeInitRowMax);
        accColSum = initRegColWiseVecNoDup(0.f);
      }
#endif
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
      rescaleGemm1AccForNewColMax(smem.xColMax[idxXBuf], smem.xColSum[idxXBuf], accColMax, acc,
                                  accColSum);
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
#if MIXED_KV_PERSISTENT
      if (tileLast) {
        // Publish the register-resident running max / sum once for the
        // partial save and finalizeAndWriteOut_sync (every warp holds the
        // same values: one warp stores, the group syncs).
        if (warpRank == gmmaWarpsPerGrp - 1) {
          storeShmColWiseVecNoDup(smem.gemm1AccColMax, accColMax);
          storeShmColWiseVecNoDup(smem.gemm1AccColSum, accColSum);
        }
        gemm1WarpGrpSync();
        // Item routing from the record (warp-uniform LDS.64).
        uint2 const reqHead = ldsU64(recAddr + 24);
        uint32_t const idxReq = reqHead.x;
        uint32_t const idxHeadGrp = reqHead.y;
        bool const itemIsPartial = (tileWord & SharedMem::tilePartialBit) != 0;
        if (itemIsPartial) {
          // Scratch slot 2c + isCtaLast (C11); the merge warp of the last
          // arriving CTA of this sequence combines the slots.
          uint32_t const isCtaLast = (tileWord & SharedMem::tileCtaLastBit) != 0 ? 1U : 0U;
          ScratchMem const scratchMem{scratch, 2 * gridDim.x, 1};
          uint32_t const idxChunk = 2 * blockIdx.x + isCtaLast;
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
          IOHead* const dst = (scratchMem.tokens() + idxChunk).template cast<IOHead>();
          finalizeAndWriteOut_sync(threadIdx.x, warpRank, dst, smem.outSwizzleBuf(idxXBuf), acc,
                                   xvoScale, smem.gemm1WarpGrpBar, smem.gemm1AccColSum,
                                   smem.gemm1AccColMax, nullptr);
        } else {
          uint32_t const outOffset = headGrpSize * (nbKHeads * idxReq + idxHeadGrp);
          OutputHead* const dst = &output[outOffset];
          ShmQWiseVec const* attentionSinksVec = nullptr;
          if (attentionSinks != nullptr) {
            attentionSinksVec =
                reinterpret_cast<ShmQWiseVec const*>(attentionSinks + headGrpSize * idxHeadGrp);
          }
          finalizeAndWriteOut_sync<false>(threadIdx.x, warpRank, dst, smem.outSwizzleBuf(idxXBuf),
                                          acc, xvoScale, smem.gemm1WarpGrpBar,
                                          smem.gemm1AccColSum, smem.gemm1AccColMax,
                                          attentionSinksVec, nbKHeads);
        }
        // finalize ended with warpGrpBar.arrive_and_wait after every thread's
        // global stores; publish the item to the merge warp (C12).
        if (threadIdx.x == 0) {
          asm volatile("st.release.cta.shared::cta.u32 [%0], %1;\n" ::"r"(static_cast<uint32_t>(
                           __cvta_generic_to_shared(&smem.finalizedItems))),
                       "r"(idxItem + 1)
                       : "memory");
        }
        idxItem++;
      }
#else
      if (idxIter == nbIters - 1) {
#if SWAP_AB
        // Publish the register-resident running max / sum once for the
        // multi-block save and finalizeAndWriteOut_sync (every warp holds the
        // same values: one warp stores, the group syncs).
        if (warpRank == gmmaWarpsPerGrp - 1) {
          storeShmColWiseVecNoDup(smem.gemm1AccColMax, accColMax);
          storeShmColWiseVecNoDup(smem.gemm1AccColSum, accColSum);
        }
        gemm1WarpGrpSync();
#else
        // gmma::wait_group should have already synchronized threads, so this may be unnecessary.
        smem.gemm1WarpGrpBar.arrive_and_wait();
#endif
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
#endif  // MIXED_KV_PERSISTENT
      TRACE_STAMP(7, idxIter, warpRank == 0);
      unused(xBar.consumed.arrive());
#if SWAP_AB
      unused(vBar.consumed.arrive());
#else
      unused(vtBar.consumed.arrive());
#endif
#if MIXED_KV_PERSISTENT && MIXED_KV_TRACE
      if (idxIter + 1 == nbCtaTiles && warpRank == 0 && laneId() == 0) {
        ctaStamp(2);
      }
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
#if MIXED_KV_PERSISTENT
    // ------------------------------------------------------------------------
    // Persistent IO group (design sections 2.2, 2.6, 8.3-8.6):
    //   warp 0: K loader  - chunk fills of meta[K] + kMetaReady, per-tile kBar.consumed
    //                       arrive (and A16 TMA boxes in the mixed / a16 modules)
    //   warp 1: V loader  - mirror on meta[V] / vMetaReady / vBar
    //   warp 2: Q warp    - one Q load + store per item into q[j & 1] (qBar[j & 1])
    //   warp 3: merge     - per partial item: finalizedItems poll, semaphore, merge
    // Every warp walks the same item stream with its own register cursor.
    // Note: PDL is disabled for the mixed launch (makeLaunchConfig), so the scan
    // warp's seqLen loads before the __syncthreads need no griddepcontrol.wait.
    // ------------------------------------------------------------------------
    ItemCursor cursor = ItemCursor::init(smem.sched);
    if (warpIdx.x < 2) {
      uint32_t const operand = warpIdx.x;  // 0 = K, 1 = V
      bool const isK = (operand == 0);
      static_assert(SharedMem::nbKBuf == SharedMem::nbVBuf && mixedLoadWarpsPerOperand == 1);
      constexpr uint32_t nbStages = SharedMem::nbKBuf;
      // Shared addresses only are selected by operand (no struct references).
      CtaBarrierPair* const stageBar = isK ? &smem.kBar[0] : &smem.vBar[0];
      CtaBarrier* const metaReady = isK ? &smem.kMetaReady[0] : &smem.vMetaReady[0];
      // Chunk 0 now; chunk for tiles [16k, 16k+16) at iteration 16k - LEAD, once each.
      fillTileMeta(smem, operand, 0, cursor, cacheList, nbKHeads, batchSize, slidingWinSizeArg);
      unused(metaReady[0].arrive());
      static_assert(MIXED_KV_META_LEAD >= 1 && MIXED_KV_META_LEAD <= 15 &&
                    MIXED_KV_META_LEAD < SharedMem::metaChunkTiles);
      if constexpr (mixedLoaderTma) {
        // A16 pages go by TMA straight into their stage rows; the stage's
        // produced barrier carries the transaction bytes (expect_tx precedes
        // the boxes).  Stage s part p of K is smem.k[2s + p]; of V smem.vBufs[s][p]:
        // same byte layout (64 rows x 128 B per part, 16 KB per stage).
        static_assert(sizeof(SharedMem::KBuffer) * cacheHeadNbParts == sizeof(SharedMem::VBuffer) &&
                      sizeof(SharedMem::KBuffer) == gemm0CtaTileNbTokens * cacheHeadPartBytes);
        constexpr uint32_t stageBytes = sizeof(SharedMem::VBuffer);
        constexpr uint32_t partBytes = sizeof(SharedMem::KBuffer);
        constexpr uint32_t pageRowsBytes = tokensPerPage * cacheHeadPartBytes;
        constexpr uint32_t partElems = exactDiv(headElems, cacheHeadNbParts);
        CUtensorMap const& tensorMap = isK ? tensorMapVLLMK : tensorMapVLLMV;
        uint32_t const stageBase = static_cast<uint32_t>(
            __cvta_generic_to_shared(isK ? static_cast<void*>(&smem.k[0])
                                         : static_cast<void*>(&smem.vBufs[0])));
        for (uint32_t g = 0; g < nbCtaTiles; g++) {
          uint32_t const stage = g % nbStages;
          // This tile's pages / tags / head (own writes; warp-uniform LDS).
          uint32_t const rec = tileRecordAddr(smem, operand, g);
          uint4 const pagesVec = ldsU128(rec);
          uint32_t const formats = ldsU32(rec + 16);
          uint32_t const idxHeadGrp = ldsU32(rec + 28);
          // The stage is free (the GEMM group released tile g - nbStages).  This
          // wait also orders the chunk refill below after every reader of the
          // chunk's previous contents (section 8.1, WAR).
          stageBar[stage].consumed.arrive_and_wait();
          if (isK) {
#if MIXED_KV_TRACE < 3
            TRACE_STAMP(8, g, true);
#endif
          } else {
#if MIXED_KV_TRACE < 2
            TRACE_STAMP(10, g, true);
#endif
          }
          if (warpElectSync()) {
            uint32_t const pages[SharedMem::nbPagesPerTile] = {pagesVec.x, pagesVec.y, pagesVec.z,
                                                                pagesVec.w};
            bool isA16[SharedMem::nbPagesPerTile];
            uint32_t nbA16 = 0;
#pragma unroll
            for (uint32_t i = 0; i < SharedMem::nbPagesPerTile; i++) {
              uint8_t const tagged = static_cast<uint8_t>(formats >> (8 * i));
#if MIXED_PAGE_STATIC_FORMAT >= 0
              uint8_t const format = tagged == kMixedBadPageFormat
                                         ? tagged
                                         : static_cast<uint8_t>(MIXED_PAGE_STATIC_FORMAT);
#else
              uint8_t const format = tagged;
#endif
              isA16[i] = (format == static_cast<uint8_t>(flashinfer::KVPageFormat::kA16));
              nbA16 += isA16[i] ? 1U : 0U;
            }
            arrive_tx(stageBar[stage].produced, nbA16 * cacheHeadNbParts * pageRowsBytes,
                      mixedLoadWarpsPerOperand * warp_size);
#pragma unroll
            for (uint32_t idxPart = 0; idxPart < cacheHeadNbParts; idxPart++) {
#pragma unroll
              for (uint32_t i = 0; i < SharedMem::nbPagesPerTile; i++) {
                if (isA16[i]) {
                  uint32_t const dst =
                      stageBase + stage * stageBytes + idxPart * partBytes + i * pageRowsBytes;
                  tma::loadAsync(__cvta_shared_to_generic(dst), tensorMap,
                                 DimsLE<4>{partElems * idxPart, idxHeadGrp, 0, pages[i]},
                                 stageBar[stage].produced);
                }
              }
            }
          }
          __syncwarp();
          uint32_t const ahead = g + MIXED_KV_META_LEAD;
          if (ahead % SharedMem::metaChunkTiles == 0 && ahead < nbCtaTiles) {
            fillTileMeta(smem, operand, ahead, cursor, cacheList, nbKHeads, batchSize,
                         slidingWinSizeArg);
            unused(metaReady[(ahead / SharedMem::metaChunkTiles) % SharedMem::nbMetaChunks].arrive());
          }
          if (isK) {
#if MIXED_KV_TRACE < 3
            TRACE_STAMP(9, g, true);
#endif
          } else {
#if MIXED_KV_TRACE < 2
            TRACE_STAMP(11, g, true);
#endif
          }
        }
      } else {
        // Static compressed format: no A16 page exists, the loader's per-tile
        // work is the stage release arrive and the chunk refills.
        for (uint32_t g = 0; g < nbCtaTiles; g++) {
          uint32_t const stage = g % nbStages;
          stageBar[stage].consumed.arrive_and_wait();
          if (isK) {
#if MIXED_KV_TRACE < 3
            TRACE_STAMP(8, g, true);
#endif
          } else {
#if MIXED_KV_TRACE < 2
            TRACE_STAMP(10, g, true);
#endif
          }
          uint32_t const ahead = g + MIXED_KV_META_LEAD;
          if (ahead % SharedMem::metaChunkTiles == 0 && ahead < nbCtaTiles) {
            fillTileMeta(smem, operand, ahead, cursor, cacheList, nbKHeads, batchSize,
                         slidingWinSizeArg);
            unused(metaReady[(ahead / SharedMem::metaChunkTiles) % SharedMem::nbMetaChunks].arrive());
          }
          if (isK) {
#if MIXED_KV_TRACE < 3
            TRACE_STAMP(9, g, true);
#endif
          } else {
#if MIXED_KV_TRACE < 2
            TRACE_STAMP(11, g, true);
#endif
          }
        }
      }
    } else if (warpIdx.x == 2) {
      // Q warp: per item, load Q(req, head) into registers, wait for the item's
      // Q buffer (released by gemm0 two items back), store, fence, publish.
      using QCvt = F16QToF8Converter<nbQLdThrds, beamWidth>;
      static_assert(nbQLdWarps == 1 && nbQLdThrds == warp_size);
      uint32_t const lane = laneId();
      uint32_t idxItem = 0;
      while (!cursor.done()) {
        ItemPiece const item =
            cursor.next(cursor.xEnd, cacheList, nbKHeads, batchSize, slidingWinSizeArg);
        TinyPtr<IOHead const> const qData{q, headGrpSize * (nbKHeads * item.req + item.head)};
        auto const f16QData = QCvt::load(lane, qData, nbKHeads, 1);
        uint32_t const idxQBuf = idxItem & 1U;
        smem.qBar[idxQBuf].consumed.arrive_and_wait();
        QCvt::store(lane, smem.q[idxQBuf], f16QData);
        // the release semantics of arrive does not work for async consumers like gmma. additional
        // fence is needed.
        asm volatile("fence.proxy.async.shared::cta;\n");
        unused(smem.qBar[idxQBuf].produced.arrive());
        idxItem++;
      }
    } else {
      // Merge warp.  For each partial item j of this CTA: wait until gemm1 has
      // finalized item j (finalizedItems > j), take the sequence's semaphore,
      // and if this CTA is the last of the sequence's nbPartials arrivals,
      // combine the partial chunks in registers and write the output.
      static_assert(headElems == 128 && validElemsPerHead % 8 == 0,
                    "merge lane mapping: 8 lanes x 16 elements per head");
      uint32_t const lane = laneId();
      uint32_t const nbCtas = gridDim.x;
      uint64_t const nbTotalTiles = smem.sched.nbTotalTiles;
      uint32_t const finAddr =
          static_cast<uint32_t>(__cvta_generic_to_shared(&smem.finalizedItems));
      uint32_t idxItem = 0;
      while (!cursor.done()) {
        ItemPiece const item =
            cursor.next(cursor.xEnd, cacheList, nbKHeads, batchSize, slidingWinSizeArg);
        if (item.partial) {
          uint32_t const Lseq = item.xBeg - item.tileInSeq;  // linear start of the sequence
          uint32_t const Lend = Lseq + item.tiles;
          uint32_t const c0 = static_cast<uint32_t>(uint64_t(Lseq) * nbCtas / nbTotalTiles);
          uint32_t const c1 = static_cast<uint32_t>(uint64_t(Lend - 1) * nbCtas / nbTotalTiles);
          uint32_t const nbPartials = c1 - c0 + 1;
          assert(nbPartials >= 2 && c0 <= blockIdx.x && blockIdx.x <= c1);
          // 1. This CTA's chunk for item j is written (gemm1 -> st.release.cta).
          for (;;) {
            uint32_t fin;
            asm volatile("ld.acquire.cta.shared::cta.u32 %0, [%1];\n"
                         : "=r"(fin)
                         : "r"(finAddr)
                         : "memory");
            if (fin > idxItem) {
              break;
            }
            __nanosleep(1000);
          }
          // 2. Arrival count for the sequence (self-resetting inc, limit nbPartials - 1).
          uint32_t const idxSeq = nbKHeads * item.req + item.head;
          uint32_t old = 0;
          if (lane == 0) {
            asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;\n"
                         : "=r"(old)
                         : "l"(&semaphores[idxSeq]), "r"(nbPartials - 1)
                         : "memory");
          }
          old = __shfl_sync(~0U, old, 0);
          // Lane 0's acquire orders the whole warp's loads below (bar.warp.sync
          // is a memory-ordering point among the participating lanes).
          __syncwarp();
          if (old == nbPartials - 1) {
            // 3. Last arriver: combine chunks {2c+1 | c0 <= c < c1} and the c1 chunk
            //    (2c1+1 iff CTA c1's range ends with the sequence, else 2c1) (C11).
            ScratchMem const scratchMem{scratch, 2 * nbCtas, 1};
            uint32_t const xC1End = static_cast<uint32_t>(
                (uint64_t(c1 + 1) * nbTotalTiles + nbCtas - 1) / nbCtas);
            uint32_t const lastChunk = 2 * c1 + (xC1End == Lend ? 1U : 0U);
            uint32_t const hInGrp = lane / 8;
            uint32_t const elem0 = 16 * (lane % 8);
            OutputHead* const outHeads = &output[headGrpSize * (nbKHeads * item.req + item.head)];
            constexpr uint32_t headsPerPass = 4;
            constexpr uint32_t nbPasses = divUp(ctaNbValidQHeads, headsPerPass);
            using Acc8 = Vec<float, 8>;
#pragma unroll
            for (uint32_t pass = 0; pass < nbPasses; pass++) {
              uint32_t const idxHead = headsPerPass * pass + hInGrp;
              bool const headValid =
                  (ctaNbValidQHeads % headsPerPass == 0) || (idxHead < ctaNbValidQHeads);
#pragma unroll
              for (uint32_t half = 0; half < 2; half++) {
                uint32_t const elem = elem0 + 8 * half;
                bool const valid = headValid && (!isHeadPadded || elem < validElemsPerHead);
                Acc8 acc = Acc8::filled(0.f);
                float sum = 0.f;
                float mx = safeInitRowMax;
                for (uint32_t c = c0; c <= c1; c++) {
                  uint32_t const chunk = (c < c1) ? 2 * c + 1 : lastChunk;
                  if (valid) {
                    float2 const sm = __ldcg(reinterpret_cast<float2 const*>(
                        &scratchMem.rowSumMax()[chunk][idxHead]));
                    float const pSum = sm.x;
                    float const pMax = sm.y;
                    uint4 const raw = __ldcg(reinterpret_cast<uint4 const*>(
                        &scratchMem.tokens()[chunk][idxHead][elem]));
                    Acc8 const data = convert<float>(reinterpret_cast<Vec<InputElem, 8> const&>(raw));
                    if (pMax > mx) {
                      float const scale = expf(mx - pMax);
                      mx = pMax;
                      sum = sum * scale + pSum;
                      acc = acc * scale + data * pSum;
                    } else {
                      float const scale = expf(pMax - mx);
                      sum = sum + pSum * scale;
                      acc = acc + data * (pSum * scale);
                    }
                  }
                }
                if (valid) {
                  if (attentionSinks != nullptr) {
                    float const sink =
                        expf(attentionSinks[mha::min(idxHead, headGrpSize - 1) +
                                            item.head * headGrpSize] -
                             mx);
                    sum += sink;
                  }
                  auto const outData = convert<OutputElem>(acc * (1.f / sum));
                  reinterpret_cast<Vec<OutputElem, 8>&>(outHeads[idxHead][elem]) = outData;
                }
              }
            }
          }
        }
        idxItem++;
      }
#ifndef NDEBUG
      // Hang guard (section 8.3): gemm1 enumerates the same items.
      for (;;) {
        uint32_t fin;
        asm volatile("ld.acquire.cta.shared::cta.u32 %0, [%1];\n"
                     : "=r"(fin)
                     : "r"(finAddr)
                     : "memory");
        if (fin >= idxItem) {
          assert(fin == idxItem);
          break;
        }
        __nanosleep(1000);
      }
#endif
    }
#else
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
      smem.qBar[0].consumed.arrive_and_wait();
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
        storeRotatedPairsForQ<isNeox, thrdsPerHead>(smem.q[0], rotatedPairs, idxHead, tid);
      }
#else
      TinyPtr<IOHead const> const qData{
          q, headGrpSize * (nbKHeads * (beamWidth * ctaInputTokBeg) + idxHeadGrp)};
#if ENABLE_PDL == 2
      acqBulk();
#endif
      auto const f16QData = QCvt::load(threadIdx.x, qData, nbKHeads, ctaNbValidTokens);

      smem.qBar[0].consumed.arrive_and_wait();
      QCvt::store(threadIdx.x, smem.q[0], f16QData);
#endif
      // the release semantics of arrive does not work for async consumers like gmma. additional
      // fence is needed.
      asm volatile("fence.proxy.async.shared::cta;\n");
      unused(smem.qBar[0].produced.arrive());
    }
    constexpr uint32_t kLoadWarpBeg = nbQLdWarps;
    constexpr uint32_t kLoadWarpCount = 1;
    constexpr uint32_t vLoadWarpBeg = nbQLdWarps + 1;
    constexpr uint32_t vLoadWarpCount = 1;
    if (warpIdx.x >= kLoadWarpBeg &&
        warpIdx.x < kLoadWarpBeg + kLoadWarpCount) {  // load k
      uint32_t const idxLoadWarp = warpIdx.x - kLoadWarpBeg;
      KVTilePartLoader kTilePartLoader{true,       nbKHeads,       cacheList, idxReq,
                                       idxHeadGrp, tensorMapVLLMK, nbPages,   smem.pages[0]};
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
    } else if (warpIdx.x >= vLoadWarpBeg &&
               warpIdx.x < vLoadWarpBeg + vLoadWarpCount) {  // load v
      uint32_t const idxLoadWarp = warpIdx.x - vLoadWarpBeg;
      constexpr bool vAlignedForSwizzle = SharedMem::vBufAlignedForSwizzle;
      KVTilePartLoader vTileLoader{false,      nbKHeads,       cacheList, idxReq,
                                   idxHeadGrp, tensorMapVLLMV, nbPages,   smem.pages[1]};
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
    }
#endif  // MIXED_KV_PERSISTENT
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
    // This warp's page tag for tiles t .. t+kAhead-1, one byte each (byte 0 =
    // tile t): read from the metadata chunk when the tile's copies are issued,
    // consumed kAhead tiles later by the expansion.  No per-tile rendezvous
    // with the load warp: the A16 rows it fetches by TMA land in other pages'
    // rows and are tracked by the stage's produced barrier directly.
    uint32_t kTags = 0;
    // Lane constants of the expansion (offsets within a stage) and the scale
    // folds, computed once per warp and kept in registers across the tile loop.
    ExpandLane const kLane = makeExpandLane(&smem.k[0], warpIdx.x);
    ExpandScales const kGlobals = makeExpandScales(fp8KGlobalScale, fp4KGlobalScale);
    auto issueKCopies = [&](uint32_t t) -> uint32_t {
      uint32_t const stage = t % SharedMem::nbKBuf;
      if (t % SharedMem::metaChunkTiles == 0) {
        // First tile of a metadata chunk: wait for the loader's fill of it
        // (fill f of the chunk completes phase f).
        smem.kMetaReady[(t / SharedMem::metaChunkTiles) % SharedMem::nbMetaChunks].wait_parity(
            toParity<1>(t / (SharedMem::metaChunkTiles * SharedMem::nbMetaChunks)));
      }
      // Wait (do not arrive) for gemm0's release of this stage's previous tile:
      // the (t / nbKBuf)-th completion of consumed[stage].
      smem.kBar[stage].consumed.wait_parity(toParity<SharedMem::nbKBuf>(t));
      return issueCompressedPageCopies(
          cacheList, /*isK=*/true, smem, kMeta, t,
          reinterpret_cast<SharedMem::PackedTile&>(smem.k[stage * cacheHeadNbParts + cacheHeadNbParts - 1]),
          smem.kScales[t % SharedMem::nbScaleTiles], warpIdx.x);
    };
#pragma unroll
    for (uint32_t t = 0; t < kAhead; ++t) {
      if (t < nbCtaTiles) kTags |= issueKCopies(t) << (8 * t);
      ldgsts::commitGroup();
    }
    for (uint32_t idxIter = 0; idxIter < nbCtaTiles; ++idxIter) {
      uint32_t const idxKStage = idxIter % SharedMem::nbKBuf;
      ldgsts::waitGroup<kAhead - 1>();  // this lane's copies of tile idxIter landed
      __syncwarp();                     // ... and the other lanes' (same page, same warp)
#if MIXED_KV_TRACE >= 2
      TRACE_STAMP(11, idxIter, warpIdx.x == 0);  // K converter: tile idxIter landed
#endif
      TRACE_STAMP(12, idxIter, warpIdx.x == 0);  // K converter: ready (no rendezvous)
      expandPackedStage(&smem.k[idxKStage * cacheHeadNbParts],
                        smem.kScales[idxIter % SharedMem::nbScaleTiles],
                        static_cast<uint8_t>(kTags & 0xFFU), kLane, kGlobals);
      TRACE_STAMP(13, idxIter, warpIdx.x == 0);
      asm volatile("fence.proxy.async.shared::cta;\n");
      unused(smem.kBar[idxKStage].produced.arrive());
#if MIXED_KV_TRACE >= 3
      TRACE_STAMP(8, idxIter, warpIdx.x == 0);  // K converter: fenced + produced
#endif
      // Rotate: byte 0 <- tile idxIter+1; the tag of tile idxIter+kAhead, read
      // while issuing its copies, goes into byte kAhead-1.
      kTags >>= 8;
      if (idxIter + kAhead < nbCtaTiles) {
        kTags |= issueKCopies(idxIter + kAhead) << (8 * (kAhead - 1));
      }
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
    uint32_t vTags = 0;
    ExpandLane const vLane = makeExpandLane(&smem.vBuf(0)[0], warpIdx.x);
    ExpandScales const vGlobals = makeExpandScales(fp8VGlobalScale, fp4VGlobalScale);
    auto issueVCopies = [&](uint32_t t) -> uint32_t {
      uint32_t const buf = t % SharedMem::nbVBuf;
      if (t % SharedMem::metaChunkTiles == 0) {
        smem.vMetaReady[(t / SharedMem::metaChunkTiles) % SharedMem::nbMetaChunks].wait_parity(
            toParity<1>(t / (SharedMem::metaChunkTiles * SharedMem::nbMetaChunks)));
      }
      smem.vBar[buf].consumed.wait_parity(toParity<SharedMem::nbVBuf>(t));
      return issueCompressedPageCopies(
          cacheList, /*isK=*/false, smem, vMeta, t,
          reinterpret_cast<SharedMem::PackedTile&>(smem.vBuf(buf)[cacheHeadNbParts - 1]),
          smem.vScales[t % SharedMem::nbScaleTiles], warpIdx.x);
    };
#pragma unroll
    for (uint32_t t = 0; t < vAhead; ++t) {
      if (t < nbCtaTiles) vTags |= issueVCopies(t) << (8 * t);
      ldgsts::commitGroup();
    }
    for (uint32_t idxIter = 0; idxIter < nbCtaTiles; ++idxIter) {
      uint32_t const idxVBuf = idxIter % SharedMem::nbVBuf;
      ldgsts::waitGroup<vAhead - 1>();
      __syncwarp();
      TRACE_STAMP(14, idxIter, warpIdx.x == 0);
      expandPackedStage(&smem.vBuf(idxVBuf)[0], smem.vScales[idxIter % SharedMem::nbScaleTiles],
                        static_cast<uint8_t>(vTags & 0xFFU), vLane, vGlobals);
      asm volatile("fence.proxy.async.shared::cta;\n");
      unused(smem.vBar[idxVBuf].produced.arrive());
      TRACE_STAMP(15, idxIter, warpIdx.x == 0);
      vTags >>= 8;
      if (idxIter + vAhead < nbCtaTiles) {
        vTags |= issueVCopies(idxIter + vAhead) << (8 * (vAhead - 1));
      }
      ldgsts::commitGroup();
    }
#endif
  }
#if MIXED_KV_TRACE
  __syncthreads();
#if MIXED_KV_PERSISTENT
  uint32_t const traceNbTiles = nbCtaTiles;
  uint32_t const traceNbSubSeq = 1;
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    ctaStamp(3);
    auto const& r = mixedKvCtaTrace[blockIdx.x % mixedKvCtaTraceSlots];
    printf("TRACE ctarec %u start %llu firstk %llu last %llu end %llu tiles %u\n", blockIdx.x,
           r[0], r[1], r[2], r[3], nbCtaTiles);
  }
#else
  uint32_t const traceNbTiles = nbIters;
  uint32_t const traceNbSubSeq = nbSubSeq;
#endif
  if (blockIdx.x == MIXED_KV_TRACE_CTA && blockIdx.y == 0 && blockIdx.z == 0 &&
      threadIdx.x == 0 && threadIdx.z == 0) {
    long long const t0 = smem.trace[0][8];
    for (uint32_t t = 0; t < SharedMem::nbTraceTiles && MIXED_KV_TRACE_TILE0 + t < traceNbTiles;
         t++) {
      printf("TRACE tile %u g0:kwait %lld mma %lld smax %lld xarr %lld | g1:vwait %lld xwait %lld rs %lld mma %lld | kl:start %lld iss %lld | vl:start %lld iss %lld | kc:ready %lld done %lld | vc:ready %lld done %lld\n",
             MIXED_KV_TRACE_TILE0 + t, smem.trace[t][0] - t0, smem.trace[t][1] - t0,
             smem.trace[t][2] - t0,
             smem.trace[t][3] - t0, smem.trace[t][4] - t0, smem.trace[t][5] - t0,
             smem.trace[t][6] - t0, smem.trace[t][7] - t0, smem.trace[t][8] - t0,
             smem.trace[t][9] - t0, smem.trace[t][10] - t0, smem.trace[t][11] - t0,
             smem.trace[t][12] - t0, smem.trace[t][13] - t0, smem.trace[t][14] - t0,
             smem.trace[t][15] - t0);
    }
    printf("TRACE cta0 smid %u residentCtasAtTile4 %u nbIters %u nbSubSeq %u grid %u x %u x %u\n",
           smem.traceSmId, smem.traceResidentMid, traceNbTiles, traceNbSubSeq, gridDim.x,
           gridDim.y, gridDim.z);
  }
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    atomicSub(&mixedKvTraceSmResident[smem.traceSmId], 1U);
  }
#endif
  __syncthreads();
  uint32_t const nbBarriers = &smem.gemm1WarpGrpBar - &smem.qBar[0].produced + 1;
  uint32_t const tid =
      threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
  assert(nbBarriers <= blockDim.x * blockDim.y * blockDim.z);
  if (tid < nbBarriers) {
    (&smem.qBar[0].produced)[tid].~CtaBarrier();
  }
#if MIXED_KV_PERSISTENT
  // Partial outputs were merged by the merge warp inside the loop; nothing
  // else to do (MultiBlockSMem is not used by this build).
#else
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
#endif  // MIXED_KV_PERSISTENT
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
                                                     Vec<KVCachePageIndex, nbPagesPerTile>& pageBuf)
    : nbKHeads{nbKHeads},
      isK{isK},
      cacheList{cacheList},
      idxReq{idxReq},
      idxHeadGrp{idxHeadGrp},
      tensorMap{tensorMap},
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

#if MIXED_KV_PERSISTENT
// Converter warp `idxWarp` copies page idxWarp of a tile if it is compressed:
// the packed rows into the page's slot of the stage's last head-part buffer
// (E4M3: dense 128 B rows with the TMA swizzle chunk ^= row % 8; E2M1: 64 B
// rows at an 80 B stride, see SharedMem::packedRowStrideFP4) and the token's
// 8 B of block scales.  Warp-contiguous ownership: consecutive lanes
// own consecutive 16 B chunks of a row (D6).  Per-lane row/chunk are constants;
// per tile the variables are the page and the head coordinate, both from the
// tile's record (one LDS.U8 tag, one LDS.32 page, one LDS.32 head).  One
// cp.async group per tile is committed by the caller.  Issue budget (C6): ~35
// instructions per lane per tile.
// Returns this warp's page tag for the tile (kMixedBadPageFormat, kA16, or the
// compressed format), which the caller keeps for the tile's expansion.
__device__ __forceinline__ uint32_t issueCompressedPageCopies(
    KVCacheList<usePagedKVCache> const& cacheList, bool isK, SharedMem const& smem,
    uint32_t operand, uint32_t idxIter, SharedMem::PackedTile& dstPacked,
    SharedMem::TileScales& dstScales, uint32_t idxWarp) {
  using flashinfer::KVPageFormat;
  static_assert(SharedMem::nbPagesPerTile == convertWarpsPerOperand, "one converter warp per page");
  static_assert(tokensPerPage == 16 && SharedMem::packedRowBytesFP8 == 128 &&
                SharedMem::packedRowBytesFP4 == 64 && SharedMem::scaleBytesPerToken == 8);
  // This warp's page and tag only (record visibility: C10 / section 8.1).
  uint32_t const rec = tileRecordAddr(smem, operand, idxIter);
  uint8_t const tagged =
      *reinterpret_cast<uint8_t const*>(__cvta_shared_to_generic(rec + 16 + idxWarp));
#if MIXED_PAGE_STATIC_FORMAT >= 0
  uint8_t const format =
      tagged == kMixedBadPageFormat ? tagged : static_cast<uint8_t>(MIXED_PAGE_STATIC_FORMAT);
#else
  uint8_t const format = tagged;
#endif
  if constexpr (MIXED_KV_EXPERIMENT & 2) {
    return format;
  }
  if (format == kMixedBadPageFormat || format == static_cast<uint8_t>(KVPageFormat::kA16)) {
    return format;
  }
  uint32_t const page = ldsU32(rec + 4 * idxWarp);
  uint32_t const idxHeadGrp = ldsU32(rec + 28);
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
    // 16 rows x 4 chunks: lane owns chunk lane % 4 of rows lane / 4 + 8k (80 B stride).
    uint32_t const c = lane % 4;
    uint32_t const r0 = lane / 4;
    uint8_t const* src = payloadPage + uint64_t(r0) * span.payload_stride.token + c * 16;
    uint64_t const rowStep = 8ull * span.payload_stride.token;
#pragma unroll
    for (uint32_t k = 0; k < 2; k++) {
      uint32_t const r = r0 + 8 * k;
      ldgsts::copyAsync<16>(slot + r * SharedMem::packedRowStrideFP4 + c * 16, src, 16);
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
  return format;
}

// Warp-wide sum and inclusive scan (all 32 lanes participate).
__device__ __forceinline__ uint32_t warpSumU32(uint32_t v) {
#pragma unroll
  for (uint32_t m = 16; m > 0; m /= 2) {
    v += __shfl_xor_sync(~0U, v, m);
  }
  return v;
}
__device__ __forceinline__ uint32_t warpInclusiveScanU32(uint32_t v) {
  uint32_t const lane = laneId();
#pragma unroll
  for (uint32_t d = 1; d < warp_size; d *= 2) {
    uint32_t const o = __shfl_up_sync(~0U, v, d);
    v += (lane >= d) ? o : 0;
  }
  return v;
}

// Prologue scan.  Pass 1: T = H * sum_r tiles(r) (lane-strided seqLen loads,
// one warp reduction per 32 requests).  Range of this CTA: x_c = ceil(c*T/P).
// Pass 2 (L1-hot re-read): the request r with H*prefix(r) <= x0 < H*prefix(r+1)
// (requests without tiles have an empty interval and cannot hit), then head0 /
// tile0 / Lseq0 by division inside the request.  T == 0 (every sequence empty)
// gives x0 = x1 = 0 and no pass 2.  Lane 0 publishes to smem.sched; the caller's
// __syncthreads makes it visible to every walker.
__device__ inline void persistentPrologueScan(SharedMem& smem,
                                              KVCacheList<usePagedKVCache> const& cacheList,
                                              uint32_t nbKHeads, uint32_t batchSize,
                                              uint32_t slidingWinSize) {
  uint32_t const lane = laneId();
  uint32_t const nbCtas = gridDim.x;
  uint32_t const idxCta = blockIdx.x;
  uint32_t totalTiles = 0;
  for (uint32_t r0 = 0; r0 < batchSize; r0 += warp_size) {
    uint32_t const r = r0 + lane;
    uint32_t const t =
        r < batchSize ? seqTilesInUse(getCacheSeqLen(cacheList, r), slidingWinSize) : 0;
    totalTiles += warpSumU32(t);
  }
  uint64_t const T = uint64_t(nbKHeads) * totalTiles;
  uint32_t x0 = 0;
  uint32_t x1 = 0;
  if (T != 0) {
    x0 = static_cast<uint32_t>((uint64_t(idxCta) * T + nbCtas - 1) / nbCtas);
    x1 = static_cast<uint32_t>((uint64_t(idxCta + 1) * T + nbCtas - 1) / nbCtas);
  }
  uint32_t req0 = batchSize;
  uint32_t head0 = 0;
  uint32_t tile0 = 0;
  uint32_t Lseq0 = 0;
  uint32_t seqLen0 = 0;
  uint32_t seqLen1 = 0;
  if (x0 < x1) {
    uint32_t running = 0;  // tiles of requests before r0
    for (uint32_t r0 = 0; r0 < batchSize; r0 += warp_size) {
      uint32_t const r = r0 + lane;
      uint32_t const len = r < batchSize ? getCacheSeqLen(cacheList, r) : 0;
      uint32_t const t = seqTilesInUse(len, slidingWinSize);
      uint32_t const incl = warpInclusiveScanU32(t);
      uint32_t const lo = nbKHeads * (running + incl - t);
      uint32_t const hi = nbKHeads * (running + incl);
      bool const hit = (r < batchSize) && lo <= x0 && x0 < hi;
      uint32_t const mask = __ballot_sync(~0U, hit);
      if (mask != 0) {
        uint32_t const src = __ffs(mask) - 1;
        req0 = __shfl_sync(~0U, r, src);
        uint32_t const loHit = __shfl_sync(~0U, lo, src);
        uint32_t const tHit = __shfl_sync(~0U, t, src);
        seqLen0 = __shfl_sync(~0U, len, src);
        head0 = (x0 - loHit) / tHit;
        tile0 = (x0 - loHit) % tHit;
        Lseq0 = loHit + head0 * tHit;
        seqLen1 = (req0 + 1 < batchSize) ? getCacheSeqLen(cacheList, req0 + 1) : 0;
        break;
      }
      running += __shfl_sync(~0U, incl, warp_size - 1);
    }
    assert(req0 < batchSize);
  }
  if (lane == 0) {
    SharedMem::PersistentSched& s = smem.sched;
    s.x0 = x0;
    s.x1 = x1;
    s.nbTotalTiles = static_cast<uint32_t>(T);
    s.req0 = req0;
    s.head0 = head0;
    s.tile0 = tile0;
    s.Lseq0 = Lseq0;
    s.seqLen0 = seqLen0;
    s.seqLen1 = seqLen1;
  }
  __syncwarp();
}

// Chunk fill (section 8.4).  Lane owns entries (tile lane/4, page lane%4) and
// (tile lane/4 + 8, page lane%4) of the 16-tile chunk.
//  A. Walk the pieces overlapping [gBeg, gBeg+16) with the cursor (warp-uniform,
//     ALU only); a lane captures (seqTile, req, head, tile word, nbPages) for an
//     entry whose tile lies in the piece.
//  B. Both dependent page-table pairs of every lane are issued together.
//  C. The four page tags of a tile are gathered into its lane j == 0 (shfl);
//     lanes store their page (STS.32), lane j == 0 the second 16 B (STS.128).
// Entries past the CTA's range keep kBAD_PAGE_INDEX / kMixedBadPageFormat / 0.
__device__ inline void fillTileMeta(SharedMem& smem, uint32_t operand, uint32_t gBeg,
                                    ItemCursor& cur,
                                    KVCacheList<usePagedKVCache> const& cacheList,
                                    uint32_t nbKHeads, uint32_t batchSize,
                                    uint32_t slidingWinSize) {
  constexpr uint32_t tileSize = gemm0CtaTileNbTokens;
  static_assert(SharedMem::metaChunkTiles * SharedMem::nbPagesPerTile == 2 * warp_size &&
                SharedMem::nbPagesPerTile == 4);
  uint32_t const lane = laneId();
  uint32_t const j = lane % SharedMem::nbPagesPerTile;
  uint32_t const i0 = lane / SharedMem::nbPagesPerTile;  // entry 1 is tile i0 + 8
  uint32_t const chunkBeg = cur.x0 + gBeg;
  uint32_t const chunkEnd = chunkBeg + SharedMem::metaChunkTiles;
  assert(cur.done() || cur.x == chunkBeg);
  constexpr uint32_t kNoTile = ~0U;
  uint32_t seqTile0 = kNoTile, seqTile1 = kNoTile;
  uint32_t req0 = 0, req1 = 0, head0 = 0, head1 = 0, word0 = 0, word1 = 0;
  uint32_t nbPages0 = 0, nbPages1 = 0;
  // A.
  while (!cur.done() && cur.x < chunkEnd) {
    ItemPiece const p = cur.next(chunkEnd, cacheList, nbKHeads, batchSize, slidingWinSize);
    uint32_t const pBeg = p.xBeg - chunkBeg;
    uint32_t const skipTokens = seqSkipTokens(p.seqLen, slidingWinSize);
    uint32_t const skipTiles = skipTokens / tileSize;
    uint32_t const tile0Skip = skipTokens % tileSize;
    uint32_t const lastValid = (p.seqLen % tileSize == 0) ? tileSize : p.seqLen % tileSize;
    uint32_t const nbPagesReq = divUp(p.seqLen, tokensPerPage);
    uint32_t const itemBits = (p.partial ? SharedMem::tilePartialBit : 0U) |
                              (p.ctaLast ? SharedMem::tileCtaLastBit : 0U);
    auto const capture = [&](uint32_t i, uint32_t& seqTile, uint32_t& req, uint32_t& head,
                             uint32_t& word, uint32_t& nbPages) {
      if (i >= pBeg && i < pBeg + p.nb) {
        uint32_t const off = i - pBeg;
        uint32_t const t = p.tileInSeq + off;
        uint32_t const x = p.xBeg + off;
        bool const first = (x == cur.x0) || (t == 0);
        bool const lastInSeq = (t + 1 == p.tiles);
        bool const last = (x + 1 == cur.xEnd) || lastInSeq;
        uint32_t const validBeg = (t == 0) ? tile0Skip : 0U;
        uint32_t const validEnd = lastInSeq ? lastValid : tileSize;
        word = validBeg | (validEnd << 8) | (first ? SharedMem::tileFirstBit : 0U) |
               (last ? SharedMem::tileLastBit : 0U) | itemBits;
        seqTile = skipTiles + t;
        req = p.req;
        head = p.head;
        nbPages = nbPagesReq;
      }
    };
    capture(i0, seqTile0, req0, head0, word0, nbPages0);
    capture(i0 + 8, seqTile1, req1, head1, word1, nbPages1);
  }
  // B.
  auto const lookup = [&](uint32_t seqTile, uint32_t req, uint32_t nbPages,
                          KVCachePageIndex& page, uint32_t& fmt) {
    page = kBAD_PAGE_INDEX;
    if (seqTile != kNoTile) {
      uint32_t const idxPage = SharedMem::nbPagesPerTile * seqTile + j;
      if (idxPage < nbPages) {
        page = cacheList.kvCachePageList[req * cacheList.maxNbPagesPerSeq + idxPage];
      }
    }
    fmt = (page == kBAD_PAGE_INDEX) ? uint32_t{kMixedBadPageFormat}
                                    : uint32_t{cacheList.transport.page_format[page]};
  };
  KVCachePageIndex page0, page1;
  uint32_t fmt0, fmt1;
  lookup(seqTile0, req0, nbPages0, page0, fmt0);
  lookup(seqTile1, req1, nbPages1, page1, fmt1);
  // C.
  uint32_t const tileLane = lane & ~3U;
  auto const gather = [&](uint32_t fmt) {
    return __shfl_sync(~0U, fmt, tileLane) | (__shfl_sync(~0U, fmt, tileLane + 1) << 8) |
           (__shfl_sync(~0U, fmt, tileLane + 2) << 16) | (__shfl_sync(~0U, fmt, tileLane + 3) << 24);
  };
  uint32_t const formats0 = gather(fmt0);
  uint32_t const formats1 = gather(fmt1);
  uint32_t const rec0 = tileRecordAddr(smem, operand, gBeg + i0);
  uint32_t const rec1 = tileRecordAddr(smem, operand, gBeg + i0 + 8);
  stsU32(rec0 + 4 * j, static_cast<uint32_t>(page0));
  stsU32(rec1 + 4 * j, static_cast<uint32_t>(page1));
  if (j == 0) {
    stsU128(rec0 + 16, uint4{formats0, word0, req0, head0});
    stsU128(rec1 + 16, uint4{formats1, word1, req1, head1});
  }
  __syncwarp();
}
#endif

#if ENABLE_MIXED_KV_CACHE
// Expand one stage; see ExpandLane above for the lane cut and the address scheme.
namespace {
template <bool B>
struct FoldTag {
  static constexpr bool value = B;
};
__device__ __forceinline__ LdGrain* smemGrain(uint32_t shared_addr) {
  return reinterpret_cast<LdGrain*>(__cvta_shared_to_generic(shared_addr));
}
// The fold multiplier global * 2^k.  Static-format builds keep it in a register
// across the tile loop (ExpandScales); the mixed build carries both formats'
// lane offsets and would spill one (STACK 8), so there it is recomputed per
// tile from a laundered copy of the global (one FMUL, not hoistable).
__device__ __forceinline__ float foldMultiplier(float global, float precomputed, float pow2) {
#if MIXED_PAGE_STATIC_FORMAT < 0
  asm volatile("" : "+f"(global));
  (void)precomputed;
  return global * pow2;
#else
  (void)global;
  (void)pow2;
  return precomputed;
#endif
}
// One 16-value E4M3 block -> 32 A16 bytes, the folded (2^120 in the scale) or two-multiply form.
template <bool kFold>
__device__ __forceinline__ void expandE4M3BlockBF16(LdGrain const& packed, uint32_t sf2,
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
// One 16-value E2M1 block (8 packed bytes) -> 32 A16 bytes.
template <bool kFold>
__device__ __forceinline__ void expandE2M1BlockBF16(uint32_t packed0, uint32_t packed1,
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
// The warp's fold decision for one tile: every lane's four scaled block scales
// (scale * global * 2^k, fp32) stay finite in BF16.  The bound is bf16 max + half an ulp.
__device__ __forceinline__ bool foldScalesFinite(float const (&f)[4], bool foldOk) {
  float const fmax = fmaxf(fmaxf(fabsf(f[0]), fabsf(f[1])), fmaxf(fabsf(f[2]), fabsf(f[3])));
  return __all_sync(0xFFFFFFFFu, fmax < 255.5f * 0x1p120f) && foldOk;
}
}  // namespace

template <typename PartBuf>
__device__ __forceinline__ void expandPackedStage(PartBuf* parts,
                                                  SharedMem::TileScales const& scales,
                                                  uint8_t tag, ExpandLane const& lane,
                                                  ExpandScales const& gs) {
  using flashinfer::KVPageFormat;
  constexpr uint32_t blocksPerPart = exactDiv(cacheHeadPartElems, 16);
  static_assert(blocksPerPart == 4, "swizzle decode assumes D=128 whole-head rows");
  // `tag` is this warp's page tag (warp-uniform).  Under a static format the
  // only runtime information in it is "past the sequence end"; re-deriving the
  // format here keeps the FP8/FP4 branches compile-time for static builds.
#if MIXED_PAGE_STATIC_FORMAT >= 0
  uint8_t const format = tag == kMixedBadPageFormat
                             ? kMixedBadPageFormat
                             : static_cast<uint8_t>(MIXED_PAGE_STATIC_FORMAT);
#else
  uint8_t const format = tag;
#endif
  bool const isFP8 = format == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8);
  if constexpr (MIXED_KV_EXPERIMENT & 1) {
    __syncwarp();
    return;
  }
  if (format == static_cast<uint8_t>(KVPageFormat::kA16)) {
    __syncwarp();
    return;
  }
  // Shared-window addresses: the stage base is warp-uniform, the lane offsets
  // loop-invariant.  Store b, half g of this lane's row: a16Row ^ ((2b + g) * 16).
  uint32_t const stage = static_cast<uint32_t>(__cvta_generic_to_shared(parts));
  assert(stage % 128 == 0);
  uint32_t const a16Row = stage + lane.a16;
  auto const store = [&](uint32_t b, LdGrain const (&v)[2]) {
#pragma unroll
    for (uint32_t g = 0; g < 2; g++) {
      *smemGrain(a16Row ^ ((2 * b + g) * 16)) = v[g];
    }
  };
  if (format == kMixedBadPageFormat) {
    // Slot past the sequence end: zero the row (no packed data to read).
    __syncwarp();
    LdGrain const zero[2] = {LdGrain{}, LdGrain{}};
#pragma unroll
    for (uint32_t b = 0; b < blocksPerPart; b++) {
      store(b, zero);
    }
    __syncwarp();
    return;
  }
  uint32_t const scaleWord = *reinterpret_cast<uint32_t const*>(
      __cvta_shared_to_generic(static_cast<uint32_t>(__cvta_generic_to_shared(&scales)) +
                               lane.scale));
  constexpr bool kBF16Placement =
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
      mha::is_same_v<InputElem, __nv_bfloat16>;
#else
      false;
#endif
  if (isFP8) {
    uint32_t const row = stage + lane.fp8;
    LdGrain words[blocksPerPart];
#pragma unroll
    for (uint32_t b = 0; b < blocksPerPart; b++) {
      words[b] = *smemGrain(row ^ (b * 16));
    }
    __syncwarp();
    if constexpr (kBF16Placement) {
      float f[4];
      e4m3x4ScalesToFloat(scaleWord, foldMultiplier(gs.fp8Global, gs.fp8GlobalFold, 0x1p120f), f);
      bool const fold = foldScalesFinite(f, gs.fp8FoldOk);
      auto const run = [&](auto foldTag) {
        constexpr bool kFold = decltype(foldTag)::value;
        uint32_t sc01, sc23;
        if constexpr (kFold) {
          sc01 = bf16x2BitsFromFloats(f[0], f[1]);
          sc23 = bf16x2BitsFromFloats(f[2], f[3]);
        } else {
          float g[4];
          e4m3x4ScalesToFloat(scaleWord, gs.fp8Global, g);
          sc01 = bf16x2BitsFromFloats(g[0], g[1]);
          sc23 = bf16x2BitsFromFloats(g[2], g[3]);
        }
#pragma unroll
        for (uint32_t b = 0; b < blocksPerPart; b++) {
          uint32_t const sc = b < 2 ? sc01 : sc23;
          uint32_t const sf2 = (b % 2 == 0) ? prmtSelfB32(sc, 0x1010u) : prmtSelfB32(sc, 0x3232u);
          LdGrain out[2];
          expandE4M3BlockBF16<kFold>(words[b], sf2, out);
          store(b, out);
        }
      };
      if (fold) {
        run(FoldTag<true>{});
      } else {
        run(FoldTag<false>{});
      }
    } else {
      Vec<uint16_t, 4> const a16Scales =
          convertE4M3x4ScalesToA16Bits<InputElem>(scaleWord, gs.fp8Global);
#pragma unroll
      for (uint32_t b = 0; b < blocksPerPart; b++) {
        LdGrain out[2] = {words[b], LdGrain{}};
        expandCompressedBlock16WithScale<KVPageFormat::kBlockScaledFP8, InputElem>(
            broadcastA16Scale<InputElem>(a16Scales[b]), out[0], out[1]);
        store(b, out);
      }
    }
  } else {
    assert(format == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4));
    uint32_t const row = stage + lane.fp4;
    LdGrain words[blocksPerPart / 2];  // blocks 2c, 2c+1 in words[c]
#pragma unroll
    for (uint32_t c = 0; c < blocksPerPart / 2; c++) {
      words[c] = *smemGrain(row + c * 16);
    }
    __syncwarp();
    if constexpr (kBF16Placement) {
      float f[4];
      e4m3x4ScalesToFloat(scaleWord, foldMultiplier(gs.fp4Global, gs.fp4GlobalFold, 0x1p126f), f);
      bool const fold = foldScalesFinite(f, gs.fp4FoldOk);
      auto const run = [&](auto foldTag) {
        constexpr bool kFold = decltype(foldTag)::value;
        uint32_t sc01, sc23;
        if constexpr (kFold) {
          sc01 = bf16x2BitsFromFloats(f[0], f[1]);
          sc23 = bf16x2BitsFromFloats(f[2], f[3]);
        } else {
          float g[4];
          e4m3x4ScalesToFloat(scaleWord, gs.fp4Global, g);
          sc01 = bf16x2BitsFromFloats(g[0], g[1]);
          sc23 = bf16x2BitsFromFloats(g[2], g[3]);
        }
#pragma unroll
        for (uint32_t b = 0; b < blocksPerPart; b++) {
          uint32_t const sc = b < 2 ? sc01 : sc23;
          uint32_t const sf2 = (b % 2 == 0) ? prmtSelfB32(sc, 0x1010u) : prmtSelfB32(sc, 0x3232u);
          LdGrain out[2];
          expandE2M1BlockBF16<kFold>(words[b / 2][(b % 2) * 2], words[b / 2][(b % 2) * 2 + 1], sf2,
                                     out);
          store(b, out);
        }
      };
      if (fold) {
        run(FoldTag<true>{});
      } else {
        run(FoldTag<false>{});
      }
    } else {
      Vec<uint16_t, 4> const a16Scales =
          convertE4M3x4ScalesToA16Bits<InputElem>(scaleWord, gs.fp4Global);
#pragma unroll
      for (uint32_t b = 0; b < blocksPerPart; b++) {
        LdGrain out[2] = {LdGrain{}, LdGrain{}};
        out[0][0] = words[b / 2][(b % 2) * 2];
        out[0][1] = words[b / 2][(b % 2) * 2 + 1];
        expandCompressedBlock16WithScale<KVPageFormat::kBlockScaledFP4, InputElem>(
            broadcastA16Scale<InputElem>(a16Scales[b]), out[0], out[1]);
        store(b, out);
      }
    }
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
    }
  }
  idxTileRef = idxTile;
  __syncwarp();
}

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
__device__ inline RegColWiseVecNoDup loadShmColWiseVecNoDup(ShmQWiseVec const& shmVec) {
  RegColWiseVecNoDup ret;
#pragma unroll
  for (uint32_t i = 0; i < divUp(ShmQWiseVec::size, warp_size); i++) {
    uint32_t const idx = i * warp_size + laneId();
    bool const inBound = ((ShmQWiseVec::size % warp_size == 0) || (idx < ShmQWiseVec::size));
    ret[i] = (inBound ? shmVec[idx] : 0);
  }
  return ret;
}

__device__ inline void storeShmColWiseVecNoDup(ShmQWiseVec& shmVec,
                                               RegColWiseVecNoDup const& src) {
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
// Rescales acc for the running column max published by gemm0 in shmXColMax
// (read under xBar.produced acquire) and updates the register-resident running
// max / sum.  Every warp runs the identical update on the same inputs in the
// same order, so no group sync is needed; per column the arithmetic is that of
// the former shared-memory version (scale, then add the four warp sums).
__device__ inline void rescaleGemm1AccForNewColMax(
    ShmQWiseVec const& shmXColMax, ShmQWiseVec const (&shmXColSum)[gemm0NbWarps],
    RegColWiseVecNoDup& accColMax, Gemm1Acc& acc, RegColWiseVecNoDup& accColSum) {
  auto const xColMax = loadShmColWiseVecNoDup(shmXColMax);
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
  if (anyNeedRescale) {
    accColMax = xColMax;
  }
#pragma unroll
  for (uint32_t i = 0; i < gemm0NbWarps; i++) {
    accColSum = accColSum + loadShmColWiseVecNoDup(shmXColSum[i]);
  }
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
[[maybe_unused]] static uint32_t chooseNbSubSeq(uint32_t multiProcessorCount,
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

#if MIXED_KV_PERSISTENT
// Persistent grid size P (lever [8]): one CTA per resident slot.  Every CTA
// takes ceil/floor(T / P) tiles of the linearized work, so P above the slot
// count only shortens CTAs (and adds partials); P below it idles SMs.
// XQA_PERSISTENT_CTAS overrides for the conformance matrix (P = 1, 3, 5, ...).
static uint32_t choosePersistentGridSize(uint32_t multiProcessorCount, uint32_t ctasPerSm) {
  auto const env = std::getenv("XQA_PERSISTENT_CTAS");
  if (env != nullptr) {
    int32_t const val = std::stoi(env);
    if (val > 0) {
      return static_cast<uint32_t>(val);
    }
  }
  return multiProcessorCount * mha::max(1U, ctasPerSm);
}
#endif

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
#if MIXED_KV_PERSISTENT
  dim3 const dimGrid{
      choosePersistentGridSize(prop.multiProcessorCount, static_cast<uint32_t>(ctasPerSm)), 1, 1};
#else
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
#endif
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
#if SPEC_DEC
  auto specDecParams = SpecDecParams{qSeqLen, qCuSeqLens, mask};
#endif
#if MIXED_KV_PERSISTENT
  // Scratch: 2 chunks per CTA (one per possible partial item); semaphores one
  // per (request, head), zero at first use as before.
  dim3 const dimGrid{
      choosePersistentGridSize(multiProcessorCount, static_cast<uint32_t>(ctasPerSm)), 1, 1};
#else
  uint32_t const nbSubSeqPerSeq = chooseNbSubSeq(
      multiProcessorCount, batchSize, nbKHeads, maxSeqLen, static_cast<uint32_t>(ctasPerSm));
#if SPEC_DEC
  uint32_t const qLen = qSeqLen;
#else
  uint32_t const qLen = 1;
#endif
  dim3 const dimGrid{divUp(qLen, inputTokensPerCta), nbSubSeqPerSeq, nbKHeads * batchSize};
#endif
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
