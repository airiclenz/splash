// Checks of the vision scratch arena: the power-of-two patch capacity an
// image admits and how the arena's bytes scale with it (CPU), the GEMM tile
// family each GPU family runs (CPU), and the GPU zero fill the arena gets
// before its first encode. No model weights load.

#include "ops/Vision.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

using namespace splash;

namespace {

constexpr std::string_view kZeroPipeline = "vision_zero_words";
constexpr uint64_t kZeroWordBytes = 16;
constexpr uint8_t kFillByte = 0xFF;

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

// Apple7/8 run every encode GEMM on a vision_gemm_sgf_* register tile;
// Apple9 and later run the MPP tiles.
void testGemmTileFollowsGpuFamily() {
  constexpr std::string_view registerPrefix = "vision_gemm_sgf_";
  constexpr std::array<std::pair<uint32_t, bool>, 4> families{{
      {7, true},
      {8, true},
      {9, false},
      {10, false},
  }};
  for (const auto &[family, isRegister] : families) {
    const ops::VisionGemmTile tile = ops::Vision::gemmTile(family);
    const ops::VisionKernelShapes shapes =
        ops::Vision::kernelShapes(ops::VisionLayout{}, ops::ImageGrid{2, 2}, tile);
    require(tile == (isRegister ? ops::VisionGemmTile::Register
                                : ops::VisionGemmTile::Mpp),
            "GPU family selected the wrong vision GEMM tile");
    for (const ops::VisionGemmShape &shape : shapes.gemms) {
      require(std::string_view(shape.pipeline).starts_with(registerPrefix) ==
                  isRegister,
              "vision GEMM pipeline does not follow the GPU family");
    }
  }
}

// vision_zero_words zeroes exactly word_count 16-byte words of a 0xFF buffer
// through its grid-stride loop and leaves the words past the count untouched.
void testZeroWordsClearsBuffer(metal::MetalBackend &backend) {
  constexpr uint32_t groups = 2;
  constexpr uint32_t words =
      3 * groups * metal::CommandGraph::kDefaultThreads + 7;
  constexpr uint32_t guardWords = 5;
  constexpr uint64_t bytes = uint64_t{words + guardWords} * kZeroWordBytes;
  metal::MetalBuffer buffer = backend.allocateBuffer(
      bytes, metal::BufferStorage::Shared, "zero-words");
  std::memset(buffer.contents(), kFillByte, bytes);
  metal::CommandGraph graph;
  graph.add(kZeroPipeline, {buffer}, words, {groups, 1, 1});

  static_cast<void>(backend.submitCommand(graph.dispatches()));

  const auto *contents = static_cast<const uint8_t *>(buffer.contents());
  const uint64_t zeroed = uint64_t{words} * kZeroWordBytes;
  for (uint64_t index = 0; index < bytes; ++index) {
    require(contents[index] == (index < zeroed ? 0 : kFillByte),
            "vision_zero_words did not zero exactly its words");
  }
}

// Returns how many dispatches of graph run the arena zero fill.
size_t countZeroDispatches(const metal::CommandGraph &graph) {
  size_t count = 0;
  for (const metal::ComputeDispatch &dispatch : graph.dispatches())
    count += dispatch.pipelineName == kZeroPipeline;
  return count;
}

// A new encoder zeroes its whole arena at dispatch 0 of its first encode()
// graph and never again. Default weights suffice: graphs are only recorded.
void testFirstEncodeZeroesArenaOnce(metal::MetalBackend &backend) {
  const ops::VisionWeights weights;
  ops::Vision encoder(backend, weights, ops::kMinimumImagePatchCapacity);
  const ops::ImageGrid grid{2, 2};
  metal::MetalBuffer pixels = backend.allocateBuffer(
      grid.pixelBytes(), metal::BufferStorage::Private, "pixels");
  metal::MetalBuffer embeddings = backend.allocateBuffer(
      uint64_t{ops::Vision::embeddingRows(grid)} *
          weights.layout.outputHiddenSize * sizeof(uint16_t),
      metal::BufferStorage::Private, "embeddings");
  metal::CommandGraph first;
  metal::CommandGraph second;

  encoder.encode(first, grid, pixels, embeddings);
  encoder.encode(second, grid, pixels, embeddings);

  const metal::ComputeDispatch &zero = first.dispatches().front();
  require(zero.pipelineName == kZeroPipeline &&
              zero.buffers.front().buffer.sizeBytes() == encoder.arenaBytes(),
          "first encode does not open with the arena zero fill");
  require(countZeroDispatches(first) == 1 && countZeroDispatches(second) == 0,
          "the arena zero fill ran other than once");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: vision-scratch METALLIB\n";
    return 2;
  }
  try {
    testCapacityRounding();
    testCapacityCappedBySmallLimit();
    testScratchBytesScaleWithCapacity();
    testScratchBytesRejectsInvalidCapacity();
    testGemmTileFollowsGpuFamily();
    metal::MetalBackend backend(argv[1]);
    testZeroWordsClearsBuffer(backend);
    testFirstEncodeZeroesArenaOnce(backend);
    std::cout << "vision scratch tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "vision scratch tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
