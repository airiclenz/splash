// CPU checks of the vision scratch arena sizing: the power-of-two patch
// capacity an image admits and how the arena's bytes scale with it. No GPU
// work runs; the binary links the Metal backend only because Vision does.

#include "ops/Vision.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

using namespace splash;

namespace {

// Largest relative deviation from an exact patch-count ratio that arena tile
// padding and 16 KiB view alignment may add.
constexpr double kRatioTolerance = 0.01;

// Throws with message when condition is false.
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Capacity is the smallest power of two covering the image, never below the
// minimum arena and never above the configured per-image limit.
void testCapacityRounding() {
  constexpr uint32_t maximum = ops::kMaximumImagePatches;
  constexpr std::array<std::pair<uint32_t, uint32_t>, 9> cases{{
      {0, 1024},
      {4, 1024},
      {1024, 1024},
      {1025, 2048},
      {4096, 4096},
      {4097, 8192},
      {16384, 16384},
      {16385, 16384},
      {std::numeric_limits<uint32_t>::max(), 16384},
  }};
  for (const auto &[patches, expected] : cases) {
    require(ops::imagePatchCapacity(patches, maximum) == expected,
            "image patch capacity is not the covering power of two");
  }
  static_assert(ops::imagePatchCapacity(64, maximum) ==
                ops::kMinimumImagePatchCapacity);
}

// A configured limit below the minimum arena caps the capacity at that limit.
void testCapacityCappedBySmallLimit() {
  require(ops::imagePatchCapacity(64, 512) == 512 &&
              ops::imagePatchCapacity(600, 516) == 516,
          "image patch capacity exceeded the configured limit");
}

// Returns whether larger / smaller is within kRatioTolerance of expected.
bool isNearRatio(uint64_t larger, uint64_t smaller, double expected) {
  const double ratio =
      static_cast<double>(larger) / static_cast<double>(smaller);
  return ratio >= expected * (1.0 - kRatioTolerance) &&
         ratio <= expected * (1.0 + kRatioTolerance);
}

// Arena bytes scale linearly with capacity: a 4096-patch arena is about a
// quarter of the 16384-patch arena every image used to admit.
void testScratchBytesScaleWithCapacity() {
  const ops::VisionLayout layout;
  const uint64_t full =
      ops::Vision::scratchBytes(layout, ops::kMaximumImagePatches);
  const uint64_t quarter = ops::Vision::scratchBytes(layout, 4096);
  const uint64_t minimum =
      ops::Vision::scratchBytes(layout, ops::kMinimumImagePatchCapacity);

  require(isNearRatio(full, quarter, 4.0),
          "4096-patch arena is not a quarter of the 16384-patch arena");
  require(isNearRatio(full, minimum, 16.0),
          "minimum arena is not a sixteenth of the 16384-patch arena");
  std::cout << "vision_scratch_bytes 1024=" << minimum << " 4096=" << quarter
            << " 16384=" << full << '\n';
}

// Arena sizes that do not cover a positive multiple of four patches throw.
void testScratchBytesRejectsInvalidCapacity() {
  const ops::VisionLayout layout;
  for (uint32_t patches : {0U, 1022U}) {
    bool isRejected = false;
    try {
      static_cast<void>(ops::Vision::scratchBytes(layout, patches));
    } catch (const std::invalid_argument &) {
      isRejected = true;
    }
    require(isRejected, "invalid vision scratch capacity was accepted");
  }
}

} // namespace

int main() {
  try {
    testCapacityRounding();
    testCapacityCappedBySmallLimit();
    testScratchBytesScaleWithCapacity();
    testScratchBytesRejectsInvalidCapacity();
    std::cout << "vision scratch tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "vision scratch tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
