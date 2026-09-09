/*
 * Copyright (c) 2026 by FlashInfer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <stdint.h>

#include "page_storage.cuh"

namespace flashinfer {

// Storage description only.  All strides are bytes.  The attention loader
// gathers page_format, payload, and scale vectors; this type deliberately has
// no scalar coefficient accessor.
struct KVPageByteStrides {
  uint32_t page = 0;
  uint32_t token = 0;
  uint32_t head = 0;
};

struct KVPageFormatSpan {
  void const* k_payload = nullptr;
  void const* v_payload = nullptr;
  uint8_t const* k_scales = nullptr;
  uint8_t const* v_scales = nullptr;
  float const* k_global_scale = nullptr;
  float const* v_global_scale = nullptr;
  KVPageByteStrides payload_stride;
  KVPageByteStrides scale_stride;
  bool allocated = true;
};

template <typename A16, uint32_t PagesPerBlock = 0, uint32_t PageStride = 0>
struct KVPageTransport {
  KVPageFormatSpan formats[static_cast<uint8_t>(KVPageFormat::kNumFormats)];
  uint8_t const* page_format = nullptr;
  KVPageStorage storage{};

  template <int StaticFormat = -1>
  __device__ KVPageAddress resolve(uint32_t page, bool valid) const {
    if (!valid) return {};
    if (storage.pages != nullptr) return storage.template address<PagesPerBlock, PageStride>(page);
    const auto format = StaticFormat >= 0 ? StaticFormat : page_format[page];
    return KVPageAddress::make(uint64_t(page) << 2, static_cast<KVPageFormat>(format));
  }

  // Resolve the address once at prefetch; payload and scale loads share it.
  __device__ KVPageFormatSpan span(KVPageAddress address, uint8_t format) const {
    auto result = formats[format];
    result.allocated = address.allocated();
    if (storage.pages == nullptr) {
      const uint64_t page = address.allocated() ? address.offset() >> 2 : 0;
      result.k_payload =
          static_cast<const uint8_t*>(result.k_payload) + page * result.payload_stride.page;
      result.v_payload =
          static_cast<const uint8_t*>(result.v_payload) + page * result.payload_stride.page;
      if (format != static_cast<uint8_t>(KVPageFormat::kA16)) {
        result.k_scales += page * result.scale_stride.page;
        result.v_scales += page * result.scale_stride.page;
      }
      result.payload_stride.page = result.scale_stride.page = 0;
      return result;
    }
    const auto origin = address.allocated() ? address : KVPageAddress::make(0, KVPageFormat::kA16);
    result.k_payload = storage.payload(origin, 0, 0, false);
    result.v_payload = storage.payload(origin, 0, 0, true);
    result.k_scales = storage.scales(origin, 0, 0, false);
    result.v_scales = storage.scales(origin, 0, 0, true);
    const uint32_t row_bytes = storage.geometry.row_bytes(static_cast<KVPageFormat>(format));
    result.payload_stride = {0, storage.geometry.heads * 2 * row_bytes, 2 * row_bytes};
    result.scale_stride = {0, storage.geometry.heads * storage.geometry.head_dim / 8,
                           storage.geometry.head_dim / 8};
    return result;
  }
};

}  // namespace flashinfer
