/*
 * Copyright (c) 2026 by FlashInfer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <stdint.h>

namespace flashinfer {

enum class KVPageFormat : uint8_t {
  kA16 = 0,
  kBlockScaledFP8 = 1,
  kBlockScaledFP4 = 2,
  kNumFormats = 3,
};

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
};

template <typename A16>
struct KVPageTransport {
  KVPageFormatSpan formats[static_cast<uint8_t>(KVPageFormat::kNumFormats)];
  uint8_t const* page_format = nullptr;
};

}  // namespace flashinfer
