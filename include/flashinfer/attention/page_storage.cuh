/*
 * Copyright (c) 2026 by FlashInfer contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include <cuda/atomic>

namespace flashinfer {

template <typename T>
using KVPageAtomic = cuda::atomic_ref<T, cuda::thread_scope_device>;

enum class KVPageFormat : uint8_t {
  kA16 = 0,
  kBlockScaledFP8 = 1,
  kBlockScaledFP4 = 2,
  kNumFormats = 3,
};

constexpr uint64_t kUnallocatedKVPage = UINT64_MAX;
constexpr uint32_t kKVPageAlignment = 128;

// The low two address bits carry the format; all extents are 128-byte aligned.
// A table entry describes the one committed encoding of a logical layer page.
struct KVPageAddress {
  uint64_t value = kUnallocatedKVPage;

  __host__ __device__ bool allocated() const { return value != kUnallocatedKVPage; }
  __host__ __device__ uint64_t offset() const { return value & ~uint64_t{3}; }
  __host__ __device__ KVPageFormat format() const { return static_cast<KVPageFormat>(value & 3); }
  __host__ __device__ static KVPageAddress make(uint64_t offset, KVPageFormat format) {
    return {offset | static_cast<uint8_t>(format)};
  }
};

// Every page uses [token, head, K/V, coefficient] ordering. Scales follow the
// payload in [token, head, K/V, coefficient / 16] order. No A16 row gaps remain.
struct KVPageGeometry {
  uint32_t tokens;
  uint32_t heads;
  uint32_t head_dim;

  __host__ __device__ uint64_t values() const { return uint64_t(tokens) * heads * 2 * head_dim; }
  __host__ __device__ uint32_t row_bytes(KVPageFormat format) const {
    return format == KVPageFormat::kA16              ? head_dim * 2
           : format == KVPageFormat::kBlockScaledFP8 ? head_dim
                                                     : head_dim / 2;
  }
  __host__ __device__ uint64_t payload_bytes(KVPageFormat format) const {
    return uint64_t(tokens) * heads * 2 * row_bytes(format);
  }
  __host__ __device__ uint64_t encoded_bytes(KVPageFormat format) const {
    return payload_bytes(format) + (format == KVPageFormat::kA16 ? 0 : values() / 16);
  }
  __host__ __device__ uint64_t extent_bytes(KVPageFormat format) const {
    return (encoded_bytes(format) + kKVPageAlignment - 1) / kKVPageAlignment * kKVPageAlignment;
  }
  __host__ __device__ uint64_t row(uint32_t token, uint32_t head, bool is_v) const {
    return (uint64_t(token) * heads + head) * 2 + uint32_t(is_v);
  }
};

struct KVPageStorage {
  uint8_t* data;
  uint64_t* pages;
  uint32_t page_stride;
  uint32_t pages_per_block;
  KVPageGeometry geometry;

  __device__ uint64_t& entry(uint32_t page) const {
    return pages[uint64_t(page / pages_per_block) * page_stride + page % pages_per_block];
  }
  template <uint32_t PagesPerBlock = 0, uint32_t PageStride = 0>
  __device__ KVPageAddress address(uint32_t page) const {
    uint32_t const ratio = PagesPerBlock ? PagesPerBlock : pages_per_block;
    uint32_t const stride = PageStride ? PageStride : page_stride;
    return {pages[uint64_t(page / ratio) * stride + page % ratio]};
  }
  __device__ uint8_t* payload(KVPageAddress address, uint32_t token, uint32_t head,
                              bool is_v) const {
    return data + address.offset() +
           geometry.row(token, head, is_v) * geometry.row_bytes(address.format());
  }
  __device__ uint8_t* scales(KVPageAddress address, uint32_t token, uint32_t head,
                             bool is_v) const {
    return data + address.offset() + geometry.payload_bytes(address.format()) +
           geometry.row(token, head, is_v) * (geometry.head_dim / 16);
  }
};

struct KVPageSlab {
  uint32_t lock;
  uint32_t slot_bytes;
  uint32_t size_class;
  uint32_t used;

  __device__ bool try_lock() {
    uint32_t expected = 0;
    return KVPageAtomic<uint32_t>(lock).compare_exchange_strong(
        expected, 1U, cuda::memory_order_acquire, cuda::memory_order_relaxed);
  }

  __device__ void unlock() { KVPageAtomic<uint32_t>(lock).store(0U, cuda::memory_order_release); }
};

// Slabs are assigned on demand by extent size, not permanently partitioned by
// format. An empty slab can immediately serve any realized page geometry.
// Allocator entry points are called by one elected lane per CTA; other lanes
// must not spin on a lock held by a divergent lane in their own warp.
struct KVPageArena {
  KVPageSlab* slabs;
  uint64_t* occupied;
  unsigned long long* available;
  uint32_t* hints;
  unsigned long long* allocated_bytes;
  unsigned long long* allocation_failures;
  unsigned long long* mandatory_failures;
  unsigned long long* reserved_blocks;
  unsigned long long* page_counts;
  uint32_t* block_reservations;
  const uint32_t* a16_classes;
  uint32_t slab_count;
  uint32_t slab_bytes;
  uint32_t bitmap_words;
  uint32_t size_classes;

  __device__ void reserve_block(uint32_t block, bool reserve) const {
    uint32_t const old = atomicExch(block_reservations + block, uint32_t(reserve));
    if (old != uint32_t(reserve)) atomicAdd(reserved_blocks, reserve ? 1ULL : UINT64_MAX);
  }

  __device__ uint32_t availability_words() const { return (slab_count + 63) / 64; }
  __device__ void mark_available(uint32_t size_class, uint32_t slab, bool value) const {
    auto* word = available + uint64_t(size_class) * availability_words() + slab / 64;
    auto const bit = 1ULL << (slab % 64);
    if (value)
      atomicOr(word, bit);
    else
      atomicAnd(word, ~bit);
  }

  __device__ KVPageAddress allocate(uint32_t extent_bytes, uint32_t size_class,
                                    KVPageFormat format) const {
    uint32_t const slots = slab_bytes / extent_bytes;
    if (slots == 0 || slots > bitmap_words * 64 || size_class >= size_classes) {
      atomicAdd(allocation_failures, 1ULL);
      return {};
    }
    bool contended;
    do {
      contended = false;
      // The extra availability row represents empty, unassigned slabs. Existing
      // slabs of the requested size are filled before another slab is claimed.
      for (uint32_t pass = 0; pass < 2; ++pass) {
        uint32_t const list = pass == 0 ? size_class : size_classes;
        uint32_t const words = availability_words();
        uint32_t const start =
            KVPageAtomic<uint32_t>(hints[list]).load(cuda::memory_order_relaxed) % words;
        for (uint32_t i = 0; i < words; ++i) {
          uint32_t const word = (start + i) % words;
          auto* entry = available + uint64_t(list) * words + word;
          uint64_t candidates =
              KVPageAtomic<unsigned long long>(*entry).load(cuda::memory_order_relaxed);
          while (candidates != 0) {
            uint32_t const bit = __ffsll(static_cast<long long>(candidates)) - 1;
            candidates &= candidates - 1;
            uint32_t const index = word * 64 + bit;
            KVPageSlab* slab = slabs + index;
            if (!slab->try_lock()) {
              contended = true;
              continue;
            }
            uint32_t const used = slab->used;
            bool const eligible =
                pass == 0 ? used != 0 && slab->size_class == size_class : used == 0;
            if (eligible && used < slots) {
              uint64_t* bits = occupied + uint64_t(index) * bitmap_words;
              for (uint32_t j = 0; j < (slots + 63) / 64; ++j) {
                uint32_t const remaining = slots - j * 64;
                uint64_t const valid =
                    remaining >= 64 ? UINT64_MAX : (uint64_t{1} << remaining) - 1;
                uint64_t const free = ~bits[j] & valid;
                if (free == 0) continue;
                uint32_t const slot_bit = __ffsll(static_cast<long long>(free)) - 1;
                bits[j] |= uint64_t{1} << slot_bit;
                slab->slot_bytes = extent_bytes;
                slab->size_class = size_class;
                slab->used = used + 1;
                if (used == 0) mark_available(size_classes, index, false);
                mark_available(size_class, index, used + 1 < slots);
                slab->unlock();
                KVPageAtomic<uint32_t>(hints[list]).store(word, cuda::memory_order_relaxed);
                atomicAdd(allocated_bytes, static_cast<unsigned long long>(extent_bytes));
                atomicAdd(page_counts + a16_classes[size_class], 1ULL);
                return KVPageAddress::make(
                    uint64_t(index) * slab_bytes + uint64_t(j * 64 + slot_bit) * extent_bytes,
                    format);
              }
            }
            slab->unlock();
          }
        }
      }
    } while (contended);
    atomicAdd(allocation_failures, 1ULL);
    return {};
  }

  __device__ void release(KVPageAddress address) const {
    if (!address.allocated()) return;
    uint32_t const index = address.offset() / slab_bytes;
    KVPageSlab* slab = slabs + index;
    while (!slab->try_lock()) {
      __nanosleep(64);
    }
    uint32_t const extent_bytes = slab->slot_bytes;
    uint32_t const a16_class = a16_classes[slab->size_class];
    uint32_t const slot = (address.offset() % slab_bytes) / extent_bytes;
    occupied[uint64_t(index) * bitmap_words + slot / 64] &= ~(uint64_t{1} << (slot % 64));
    uint32_t const used = --slab->used;
    mark_available(slab->size_class, index, used != 0);
    if (used == 0) mark_available(size_classes, index, true);
    slab->unlock();
    atomicAdd(allocated_bytes, 0ULL - static_cast<unsigned long long>(extent_bytes));
    atomicAdd(page_counts + a16_class, UINT64_MAX);
  }
};

}  // namespace flashinfer
