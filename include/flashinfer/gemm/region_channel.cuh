// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mscclpp/port_channel_device.hpp>

namespace tp_region {
// Created once outside capture. Every group has its own completion identity.
struct DeviceRegion {
  mscclpp::PortChannelDeviceHandle* channels;
  std::uint64_t** inbound;
  std::uint64_t** expected;
  std::uint64_t group_bytes;
  std::uint64_t buffer_bytes;
  std::uint32_t groups;
};

#if defined(__CUDACC__)
// Exactly one producer thread calls this after all tiles contributing to the
// group have published their stores. Existing full-K GEMM/retirement is retained.
__device__ __forceinline__ void publish_completed_group(DeviceRegion region, std::uint32_t group,
                                                        std::uint64_t bytes) {
  assert(group < region.groups && bytes > 0 && bytes <= region.group_bytes);
  __threadfence_system();
  const std::uint64_t offset = group * region.group_bytes;
  assert(offset <= region.buffer_bytes && bytes <= region.buffer_bytes - offset);
  region.channels[group].putWithSignal(offset, offset, bytes);
}
#endif
}  // namespace tp_region
