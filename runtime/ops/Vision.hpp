#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <vector>

namespace splash::ops {

struct VisionLayout final {
  uint32_t depth = 27;
  uint32_t hiddenSize = 1152;
  uint32_t patchDimension = 1536;
  uint32_t intermediateSize = 4304;
  uint32_t paddedIntermediateSize = 4352;
  uint32_t mergedHiddenSize = 4608;
  uint32_t outputHiddenSize = 5120;
  uint32_t heads = 16;
  uint32_t headDimension = 72;
  uint32_t positionGridSide = 48;
  uint32_t patchSize = 16;
  uint32_t spatialMerge = 2;

  bool operator==(const VisionLayout &) const = default;
};

struct VisionAffine final {
  metal::MetalBuffer weight;
  metal::MetalBuffer bias;
};

struct VisionNorm final {
  metal::MetalBuffer weight;
  metal::MetalBuffer bias;
};

struct VisionBlock final {
  VisionNorm norm1;
  VisionAffine qkv;
  VisionAffine projection;
  VisionNorm norm2;
  VisionAffine upProjection;
  VisionAffine downProjection;
};

struct VisionWeights final {
  VisionLayout layout;
  VisionAffine patchEmbedding;
  metal::MetalBuffer positionTable;
  std::vector<VisionBlock> blocks;
  VisionNorm mergerNorm;
  VisionAffine mergerUpProjection;
  VisionAffine mergerDownProjection;
};

// Input geometry shared by the frontend and runtime: RGB patch size and the
// maximum number of patches per image.
inline constexpr uint32_t kImagePatchPixels = 16;
inline constexpr uint32_t kImageChannels = 3;
inline constexpr uint32_t kMaximumImagePatches = 16384;
// Smallest vision scratch arena, in patches. Arenas grow in powers of two from
// here, so common image sizes share one arena and regrowth stays rare.
inline constexpr uint32_t kMinimumImagePatchCapacity = 1024;
// Patch capacity of the scratch arena an image of `patches` needs: the
// smallest power of two at least max(kMinimumImagePatchCapacity, patches),
// capped at maximumPatches, the configured per-image limit. A capped result is
// a valid arena size because that limit is a positive multiple of four.
[[nodiscard]] constexpr uint32_t
imagePatchCapacity(uint32_t patches, uint32_t maximumPatches) noexcept {
  const uint64_t needed = std::max(kMinimumImagePatchCapacity, patches);
  return static_cast<uint32_t>(
      std::min<uint64_t>(std::bit_ceil(needed), maximumPatches));
}
[[nodiscard]] constexpr uint64_t imagePixelBytes(uint32_t gridHeight,
                                                 uint32_t gridWidth) noexcept {
  return uint64_t{gridHeight} * kImagePatchPixels * gridWidth *
         kImagePatchPixels * kImageChannels;
}
// Patch-grid sides are multiples of the 2x2 spatial merge. Each merged token
// covers four patches and follows row-major order over the merged grid.
struct ImageGrid final {
  uint32_t height = 0; // patches
  uint32_t width = 0;

  [[nodiscard]] bool valid() const noexcept {
    return height >= 2 && width >= 2 && height % 2 == 0 && width % 2 == 0;
  }
  [[nodiscard]] uint32_t patches() const noexcept { return height * width; }
  [[nodiscard]] uint32_t mergedTokens() const noexcept {
    return patches() / 4;
  }
  [[nodiscard]] uint64_t pixelBytes() const noexcept {
    return imagePixelBytes(height, width);
  }
};

// GPU vision encoder with one reusable scratch arena sized for maximumPatches;
// callers size it with imagePatchCapacity() for the largest image it encodes.
// encode() appends one image's dispatches, from resized uint8 RGB pixels to
// bf16 language embeddings; the first encode() after construction also zeroes
// the GPU-private arena at the head of its graph, so that graph must be
// submitted before any later one. Injection uses the encoded buffers
// independently, so the arena can be released after encoding while later
// chunks inject rows.
class Vision final {
public:
  // Scratch footprint before construction, so the caller can reserve it.
  [[nodiscard]] static uint64_t
  scratchBytes(const VisionLayout &layout, uint32_t maximumPatches);
  // Embedding buffers are written in whole merger tiles: the caller allocates
  // this many rows of layout.outputHiddenSize bf16 values per image.
  [[nodiscard]] static uint32_t embeddingRows(ImageGrid grid) noexcept;

  Vision(metal::MetalBackend &backend, const VisionWeights &model,
         uint32_t maximumPatches);
  Vision(const Vision &) = delete;
  Vision &operator=(const Vision &) = delete;

  [[nodiscard]] uint64_t arenaBytes() const noexcept {
    return arena_.sizeBytes();
  }
  // Largest image, in patches, encode() accepts.
  [[nodiscard]] uint32_t maximumPatches() const noexcept {
    return maximumPatches_;
  }

  void encode(metal::CommandGraph &graph, ImageGrid grid,
              const metal::MetalBuffer &pixels,
              const metal::MetalBuffer &embeddings);

  // Overwrites rows of a packed bf16 hidden buffer with embedding rows of
  // hiddenSize values each.
  static void inject(metal::CommandGraph &graph,
                     const metal::MetalBuffer &embeddings,
                     const metal::MetalBuffer &packedHidden,
                     uint32_t hiddenSize, uint32_t sourceRow,
                     uint32_t destinationRow, uint32_t rows);

private:
  enum class Scratch : uint32_t {
    Patches,
    Positions,
    RopeCos,
    RopeSin,
    Hidden,
    Normalized,
    Qkv,
    Intermediate,
    Queries,
    Keys,
    Values,
    Attention,
    Context,
    Count,
  };

  [[nodiscard]] const metal::MetalBuffer &scratch(Scratch tensor) const noexcept {
    return scratch_[static_cast<uint32_t>(tensor)];
  }
  void addGemm(metal::CommandGraph &graph, const char *pipeline,
               const metal::MetalBuffer &input,
               const VisionAffine &weights,
               const metal::MetalBuffer &output,
               const metal::MetalBuffer &residual, uint32_t outputSize,
               uint32_t inputSize, uint32_t rows, uint32_t tileRows,
               uint32_t tileColumns) const;
  void addNorm(metal::CommandGraph &graph, const metal::MetalBuffer &input,
               const VisionNorm &weights,
               const metal::MetalBuffer &output, uint32_t rows) const;
  void addArenaZero(metal::CommandGraph &graph) const;

  const VisionWeights &model_;
  uint32_t maximumPatches_ = 0;
  metal::MetalBuffer arena_;
  // Whether an encode() graph already carries the arena's zero fill.
  bool isArenaZeroQueued_ = false;
  metal::MetalBuffer scratch_[static_cast<uint32_t>(Scratch::Count)];
};

} // namespace splash::ops
