/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Included after XQA's matrix atoms and shared tile types.
namespace mixed_kv_fragments {

using flashinfer::KVPageFormat;

// Match the four adjacent K coefficients owned by each lane in the mixed consumer.
struct A16KColumns {
  __device__ uint32_t operator()(uint32_t lane, uint32_t pair) const { return lane * 4 + pair * 2; }
};

template <typename Function>
__device__ inline void visit(uint8_t format, Function const& function) {
#if MIXED_PAGE_STATIC_FORMAT >= 0
  unused(format);
  function(MixedFormatTag<MIXED_PAGE_STATIC_FORMAT>{});
#else
  if (format == static_cast<uint8_t>(KVPageFormat::kA16)) {
    function(MixedFormatTag<static_cast<uint8_t>(KVPageFormat::kA16)>{});
  } else if (format == static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8)) {
    function(MixedFormatTag<static_cast<uint8_t>(KVPageFormat::kBlockScaledFP8)>{});
  } else {
    function(MixedFormatTag<static_cast<uint8_t>(KVPageFormat::kBlockScaledFP4)>{});
  }
#endif
}

template <KVPageFormat format>
using Fragment =
    mha::conditional_t<format == KVPageFormat::kA16, InstInMat<2, 2>, Vec<uint32_t, 2>>;

template <uint32_t count, bool unroll = false, typename Load, typename Consume>
__device__ inline void pipelineFragments(Load const& load, Consume const& consume) {
  static_assert(count > 0);
  constexpr uint32_t unrollCount = unroll ? count : 1;
  auto current = load(0);
#pragma unroll(unrollCount)
  for (uint32_t block = 0; block + 1 < count; ++block) {
    auto const next = load(block + 1);
    consume(current, block);
    current = next;
  }
  consume(current, count - 1);
}

template <KVPageFormat format>
__device__ inline Fragment<format> fetchK(SharedMem::KSmemBuffer const& tile, uint32_t row,
                                          uint32_t block, uint32_t part,
                                          flashinfer::KVPageFormatSpan const& page, uint32_t head,
                                          uint32_t tokenBase, uint32_t skipTokens,
                                          uint32_t cacheSeqLen) {
  if constexpr (format == KVPageFormat::kA16) {
    InstInMat<2, 2> result;
    uint32_t const quad = laneId() & 3U;
#pragma unroll
    for (uint32_t n = 0; n < 2; ++n) {
      uint32_t const token = row + n * 8 + laneId() / 4;
      uint32_t const absoluteToken = tokenBase + token;
      uint32_t const column =
          part * kHeadPartBytes + block * 32 + A16KColumns{}(quad, 0) * sizeof(InputElem);
      uint64_t bits = 0;
      if (page.allocated && absoluteToken >= skipTokens && absoluteToken < cacheSeqLen &&
          column + 8 <= validElemsPerHead * sizeof(InputElem)) {
        auto const* address = static_cast<uint8_t const*>(page.k_payload) +
                              uint64_t(absoluteToken % tokensPerPage) * page.payload_stride.token +
                              uint64_t(head) * page.payload_stride.head + column;
        bits = __ldg(reinterpret_cast<uint64_t const*>(address));
      }
      result.data[n][0] = uint32_t(bits);
      result.data[n][1] = uint32_t(bits >> 32);
    }
    return result;
  } else {
    auto const* address = &tile.template at<true>(row + laneId() % 16, block);
    if constexpr (format == KVPageFormat::kBlockScaledFP8) {
      return ldmatrix<false, 2>(address);
    } else {
      return ldmatrix_8x16_4x_unpack_4b<2>(address);
    }
  }
}

template <KVPageFormat format>
__device__ inline InstInMat<2, 2> convertK(Fragment<format> const& fragment,
                                           Vec<uint32_t, 2> const& scaleWords, uint32_t scaleColumn,
                                           float globalScale) {
  if constexpr (format == KVPageFormat::kA16) {
    return fragment;
  } else {
    InstInMat<2, 2> result;
    uint16_t const scaleBits = uint8_t(scaleWords[0] >> (scaleColumn * 8)) |
                               (uint16_t(uint8_t(scaleWords[1] >> (scaleColumn * 8))) << 8);
    uint32_t const scalePair = convertE4M3x2ScalesToA16<InputElem>(scaleBits, globalScale);
#pragma unroll
    for (uint32_t n = 0; n < 2; ++n) {
      uint32_t const sf = broadcastA16Scale<InputElem>(uint16_t(scalePair >> (n * 16)));
      uint32_t const word = fragment[n];
      if constexpr (format == KVPageFormat::kBlockScaledFP8) {
        result.data[n][0] = mulA16x2<InputElem>(convertE4M3x2ToA16<InputElem>(word), sf);
        result.data[n][1] = mulA16x2<InputElem>(convertE4M3x2ToA16<InputElem>(word >> 16), sf);
      } else {
        uint32_t const pairs = word | (word >> 4);
        result.data[n][0] = mulA16x2<InputElem>(convertE2M1x2ToA16<InputElem>(pairs), sf);
        result.data[n][1] = mulA16x2<InputElem>(convertE2M1x2ToA16<InputElem>(pairs >> 16), sf);
      }
    }
    return result;
  }
}

__device__ inline uint32_t vScaleRow(uint32_t token) {
  if constexpr (grpLoadV) {
    // Each copying warp owns a token interval followed by a dump row.
    return token + token / (cacheVTileSeqLen / gemm1WarpsPerGrp);
  } else {
    return token;
  }
}

template <KVPageFormat format>
__device__ inline Fragment<format> fetchV(SharedMem::VSmemBuffer const& tile, uint32_t row,
                                          uint32_t block) {
  if constexpr (format == KVPageFormat::kA16) {
    return loadInstInMat<2, 2, false, true, false>(
        this_warp(), tile, row, block * 2, [](uint32_t token) { return mixedVFragmentRow(token); });
  } else {
    auto const* address = &tile.template at<true>(row + laneId() % 16, block);
    if constexpr (format == KVPageFormat::kBlockScaledFP8) {
      return ldmatrix_16x16_trans<1>(address);
    } else {
      return ldmatrix_16x16_trans_unpack_4b<1>(address);
    }
  }
}

template <KVPageFormat format>
__device__ inline InstInMat<2, 2> convertV(Fragment<format> const& fragment,
                                           Vec<uint32_t, 4> const& scaleWords, uint32_t scaleColumn,
                                           float globalScale) {
  if constexpr (format == KVPageFormat::kA16) {
    return fragment;
  } else {
    uint32_t sf[2];
#pragma unroll
    for (uint32_t half = 0; half < 2; ++half) {
      uint16_t const bits = uint8_t(scaleWords[half * 2] >> (scaleColumn * 8)) |
                            (uint16_t(uint8_t(scaleWords[half * 2 + 1] >> (scaleColumn * 8))) << 8);
      sf[half] = convertE4M3x2ScalesToA16<InputElem>(bits, globalScale);
    }
    InstInMat<2, 2> result;
#pragma unroll
    for (uint32_t n = 0; n < 2; ++n) {
      uint32_t const word = fragment[n];
#pragma unroll
      for (uint32_t half = 0; half < 2; ++half) {
        uint32_t converted;
        if constexpr (format == KVPageFormat::kBlockScaledFP8) {
          converted = convertE4M3x2ToA16<InputElem>(word >> (half * 16));
        } else {
          converted = convertE2M1x2ToA16<InputElem>((word | (word >> 4)) >> (half * 16));
        }
        result.data[n][half] = mulA16x2<InputElem>(converted, sf[half]);
      }
    }
    return result;
  }
}

}  // namespace mixed_kv_fragments

__device__ inline void smemQKPartGemmMixed(
    Warp const& warp, WarpAcc& acc, SharedMem::QSmemBuffer const& q, uint32_t qColBeg,
    SharedMem::KSmemBuffer const& k, MixedPageReferences<nbPagesPerWarpTile> const& pages,
    PageTransport const& transport, uint32_t head, uint32_t tokenBase, uint32_t skipTokens,
    uint32_t cacheSeqLen, uint8_t const* scales, uint32_t part, float fp8GlobalScale,
    float fp4GlobalScale) {
  constexpr uint32_t rows = warpTile.y / 16;
  // Preserve A16 queries for both PV implementations; K expands only in registers.
  // Static accumulator indices, with a single format dispatch outside the rolled
  // reduction loop. The old block/page unrolling replicated every converter.
#pragma unroll
  for (uint32_t tile = 0; tile < warpTile.x / 16; ++tile) {
    flashinfer::KVPageAddress const address{pages.values[tile * 16 / tokensPerPage]};
    uint8_t const pageFormat = address.allocated() ? static_cast<uint8_t>(address.format()) : 0;
    mixed_kv_fragments::visit(pageFormat, [&](auto tag) {
      constexpr auto format = static_cast<flashinfer::KVPageFormat>(decltype(tag)::value);
      flashinfer::KVPageFormatSpan page;
      if constexpr (format == flashinfer::KVPageFormat::kA16) {
        page = transport.span(address, static_cast<uint8_t>(format));
      }
      float const scale =
          format == flashinfer::KVPageFormat::kBlockScaledFP8 ? fp8GlobalScale : fp4GlobalScale;
      static_assert(kHeadPartBytes <= 128);
      Vec<uint32_t, 2> scaleWords;
      if constexpr (format != flashinfer::KVPageFormat::kA16) {
#pragma unroll
        for (uint32_t n = 0; n < 2; ++n) {
          uint32_t const token = tile * 16 + n * 8 + laneId() / 4;
          scaleWords[n] = reinterpret_cast<uint32_t const*>(scales)[token];
        }
      }
      auto const fetch = [&](uint32_t block) {
        return mixed_kv_fragments::fetchK<format>(k, tile * 16, block, part, page, head, tokenBase,
                                                  skipTokens, cacheSeqLen);
      };
      auto const consume = [&](auto const& fragment, uint32_t block) {
        uint32_t const scaleColumn = (part * (kHeadPartBytes / 32) + block) % 4;
        auto const a = loadQueryMatrix<2, 2, rows, 1>(warp, q, qColBeg + block * 2);
        auto const b =
            mixed_kv_fragments::convertK<format>(fragment, scaleWords, scaleColumn, scale);
#pragma unroll
        for (uint32_t i = 0; i < rows; ++i) {
#pragma unroll
          for (uint32_t n = 0; n < 2; ++n) {
            uint32_t const operand[2][1] = {b.data[n][0], b.data[n][1]};
            mma<InputElem>(acc(i, tile * 2 + n).data, a(i, 0).data, operand);
          }
        }
      };
      // Fetch the next packed operand before converting and multiplying this one.
      mixed_kv_fragments::pipelineFragments<kHeadPartBytes / 32>(fetch, consume);
    });
  }
}

template <uint32_t HeadSplits>
__device__ inline void smemXVPartGemmMixed(
    Warp const& warp, Vec<WarpAcc, HeadSplits>& accs, bool skipRescale, UniformRescaleMask,
    ThrdRegRowMax rowScales, SharedMem::XSmemBuffer const& x, uint32_t vTile,
    SharedMem::StoredVSmemBuffer const& v, MixedPageFormats<nbPagesPerVTile> const& formats,
    uint8_t const* scales, uint32_t warpInGroup, float fp8GlobalScale, float fp4GlobalScale
#if XQA_MIXED_NATIVE_MMA
    ,
    MixedPageReferences<nbPagesPerVTile> const& pages, PageTransport const& transport,
    uint32_t head, uint32_t tokenBase, uint32_t headColumn, uint32_t skipTokens,
    uint32_t cacheSeqLen
#endif
) {
  constexpr uint32_t rows = warpTile.y / 16;
  Vec<InputElem2, QuadRegRowMax::size> scalesQuad;
  if (!skipRescale) {
#if INPUT_FP16
    auto const converted = __float2half2_rn(rowScales);
#else
    auto const converted = __float2bfloat162_rn(rowScales);
#endif
    reinterpret_cast<QuadRegRowMax&>(scalesQuad) =
        replicateForQuad(warp, reinterpret_cast<ThrdRegRowMax const&>(converted));
  }
#pragma unroll
  for (uint32_t tile = 0; tile < cacheVTileSeqLen / 16; ++tile) {
    uint32_t const column = SharedMem::XSmemBuffer::cols / nbCacheVTilesPerXTile * vTile + tile * 2;
    mixed_kv_fragments::visit(formats.values[tile * 16 / tokensPerPage], [&](auto tag) {
      constexpr auto format = static_cast<flashinfer::KVPageFormat>(decltype(tag)::value);
      float const scale =
          format == flashinfer::KVPageFormat::kBlockScaledFP8 ? fp8GlobalScale : fp4GlobalScale;
      auto a = loadMatrix<2, 2, rows, 1, false, false, false, false>(warp, x, 0, column);
      if (!skipRescale) {
#pragma unroll
        for (uint32_t i = 0; i < rows; ++i) {
#pragma unroll
          for (uint32_t n = 0; n < 2; ++n) {
#pragma unroll
            for (uint32_t j = 0; j < 2; ++j) {
              auto& value = reinterpret_cast<InputElem2&>(a(i, 0).data[j][n]);
              value = value * scalesQuad[i * 2 + n];
            }
          }
        }
      }
      static_assert(warpTile.x <= 64);
#pragma unroll
      for (uint32_t hs = 0; hs < HeadSplits; ++hs) {
        auto& acc = accs[hs];
        uint32_t const headSlice = grpLoadV ? vHeadSlice(warpInGroup, hs) : hs;
#if XQA_MIXED_NATIVE_MMA
        flashinfer::KVPageFormatSpan page;
        if constexpr (format == flashinfer::KVPageFormat::kA16) {
          page = transport.span(flashinfer::KVPageAddress{pages.values[tile * 16 / tokensPerPage]},
                                static_cast<uint8_t>(format));
        }
#pragma unroll
        for (uint32_t n = 0; n < warpTile.x / 8; ++n) {
          auto const b = [&] {
            if constexpr (format == flashinfer::KVPageFormat::kA16) {
              return mixed_kv_fragments::fetchA16V(page, head,
                                                   headColumn + headSlice * warpTile.x + n * 8,
                                                   tokenBase + tile * 16, skipTokens, cacheSeqLen);
            } else {
              return mixed_kv_fragments::fetchPackedV<format>(
                  v, scales, headSlice * warpTile.x + n * 8, tile * 16, scale);
            }
          }();
#pragma unroll
          for (uint32_t i = 0; i < rows; ++i) mma<InputElem>(acc(i, n).data, a(i, 0).data, b.data);
        }
#else
      constexpr uint32_t blocks = warpTile.x / 16;
      uint32_t const firstBlock = headSlice * blocks;
      Vec<uint32_t, 4> scaleWords;
      if constexpr (format != flashinfer::KVPageFormat::kA16) {
        constexpr uint32_t scaleStride = mha::max(4U, SharedMem::VSmemBuffer::rowBytes / 32);
#pragma unroll
        for (uint32_t half = 0; half < 2; ++half) {
          uint32_t const token = tile * 16 + half * 8 + (laneId() & 3U) * 2;
#pragma unroll
          for (uint32_t adjacent = 0; adjacent < 2; ++adjacent) {
            uint32_t const row = mixed_kv_fragments::vScaleRow(token + adjacent);
            scaleWords[half * 2 + adjacent] =
                *reinterpret_cast<uint32_t const*>(scales + row * scaleStride + (firstBlock & ~3U));
          }
        }
      }
      auto const fetch = [&](uint32_t block) {
        return mixed_kv_fragments::fetchV<format>(v, tile * 16, firstBlock + block);
      };
      auto const consume = [&](auto const& fragment, uint32_t block) {
        auto const b = mixed_kv_fragments::convertV<format>(fragment, scaleWords,
                                                            (firstBlock + block) % 4, scale);
#pragma unroll
        for (uint32_t i = 0; i < rows; ++i) {
#pragma unroll
          for (uint32_t n = 0; n < 2; ++n) {
            uint32_t const operand[2][1] = {b.data[n][0], b.data[n][1]};
            mma<InputElem>(acc(i, block * 2 + n).data, a(i, 0).data, operand);
          }
        }
      };
      // The output block must stay static so accumulators remain in registers.
      mixed_kv_fragments::pipelineFragments<blocks, true>(fetch, consume);
#endif
      }
    });
  }
}
