#pragma once

#include "engine/MemoryGovernor.hpp"

#include <cstdint>
#include <optional>

namespace splash::engine::test {

// A host sampler whose answer the test sets and whose calls it counts.
struct FakeHost {
  std::optional<uint64_t> availableBytes;
  uint64_t inactiveBytes = 0;
  uint64_t wiredBytes = 0;
  uint64_t compressorBytes = 0;
  int samples = 0;

  // The governor's host provider over this fake, which must outlive it.
  [[nodiscard]] MemoryGovernor::HostAvailableMemoryProvider sampler() {
    return [this]() -> std::optional<HostMemorySample> {
      ++samples;
      if (!availableBytes)
        return std::nullopt;
      return HostMemorySample{*availableBytes, inactiveBytes, wiredBytes,
                              compressorBytes};
    };
  }
};

} // namespace splash::engine::test
