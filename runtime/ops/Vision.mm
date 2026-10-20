#include "Vision.hpp"

#include "metal/abi/Vision.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace splash::ops {
namespace {

using metal::CommandGraph;
using metal::MetalBuffer;

constexpr uint32_t kGemmRowTile = 64;
constexpr uint32_t kGemmColumnTile = 128;
constexpr uint32_t kMergerRowTile = 32;
constexpr uint32_t kMergerColumnTile = 256;
constexpr uint32_t kKeyTile = 128;
constexpr uint32_t kQueryTile = 64;
constexpr uint32_t kQkDimension = 80; // head dimension 72 padded to 16
constexpr uint64_t kArenaAlignment = 16 * 1024;
constexpr uint64_t kBf16Bytes = 2;
// The arena zero fill writes 16-byte words from at most this many threadgroups.
constexpr uint64_t kZeroWordBytes = 16;
constexpr uint64_t kZeroGroupLimit = 1024;

uint32_t roundUp(uint32_t value, uint32_t multiple) noexcept {
  return (value + multiple - 1) / multiple * multiple;
}

uint64_t alignArena(uint64_t bytes) noexcept {
  return (bytes + kArenaAlignment - 1) / kArenaAlignment * kArenaAlignment;
}

// Scratch tensor byte sizes for one encoder sized to maximumPatches. GEMM
// tiles read and write whole 64-row tiles, attention reads whole 128-key
// tiles, and the merger reads the normalized rows as (patches / 4, 4608).
std::array<uint64_t, 13> scratchLayout(const VisionLayout &layout,
                                       uint32_t maximumPatches) {
  const uint64_t rows = roundUp(maximumPatches, kGemmRowTile);
  const uint64_t padded = roundUp(maximumPatches, kKeyTile);
  const uint64_t mergedRows = roundUp(maximumPatches / 4, kMergerRowTile);
  const uint64_t hidden = layout.hiddenSize;
  const uint64_t headRows = uint64_t{layout.heads} * padded;
  return {
      rows * layout.patchDimension * kBf16Bytes,                 // Patches
      rows * hidden * kBf16Bytes,                                // Positions
      uint64_t{maximumPatches} * layout.headDimension * 4,       // RopeCos
      uint64_t{maximumPatches} * layout.headDimension * 4,       // RopeSin
      rows * hidden * kBf16Bytes,                                // Hidden
      std::max(rows * hidden, mergedRows * layout.mergedHiddenSize) *
          kBf16Bytes,                                            // Normalized
      std::max(rows * 3 * hidden, mergedRows * layout.mergedHiddenSize) *
          kBf16Bytes,                                            // Qkv
      rows * layout.paddedIntermediateSize * kBf16Bytes,         // Intermediate
      headRows * kQkDimension * kBf16Bytes,                      // Queries
      headRows * kQkDimension * kBf16Bytes,                      // Keys
      headRows * layout.headDimension * kBf16Bytes,              // Values
      headRows * layout.headDimension * kBf16Bytes,              // Attention
      rows * hidden * kBf16Bytes,                                // Context
  };
}

void requireLayout(const VisionLayout &layout) {
  // Packages share the same vision tower; only the language-space projection
  // width varies with the text model.
  VisionLayout tower = layout;
  tower.outputHiddenSize = VisionLayout{}.outputHiddenSize;
  if (tower != VisionLayout{} || !layout.outputHiddenSize ||
      layout.headDimension + 8 != kQkDimension ||
      layout.hiddenSize % kGemmColumnTile ||
      layout.paddedIntermediateSize % kGemmColumnTile ||
      layout.mergedHiddenSize % kMergerColumnTile ||
      layout.outputHiddenSize % kMergerColumnTile ||
      layout.patchDimension % kGemmColumnTile) {
    throw std::invalid_argument(
        "vision kernels are specialized for the Qwen3.5 27-block tower");
  }
}

} // namespace

uint64_t Vision::scratchBytes(const VisionLayout &layout,
                              uint32_t maximumPatches) {
  requireLayout(layout);
  if (!maximumPatches || maximumPatches % 4) {
    throw std::invalid_argument(
        "vision scratch must cover a positive multiple of four patches");
  }
  uint64_t total = 0;
  for (uint64_t bytes : scratchLayout(layout, maximumPatches)) {
    const uint64_t aligned = alignArena(bytes);
    if (aligned > std::numeric_limits<uint64_t>::max() - total) {
      throw std::overflow_error("vision scratch byte count overflows");
    }
    total += aligned;
  }
  return total;
}

uint32_t Vision::embeddingRows(ImageGrid grid) noexcept {
  return roundUp(grid.mergedTokens(), kMergerRowTile);
}

VisionKernelShapes Vision::kernelShapes(const VisionLayout &layout,
                                        ImageGrid grid) {
  requireLayout(layout);
  if (!grid.valid())
    throw std::invalid_argument("vision kernel shapes need a valid image grid");
  const uint32_t tokens = grid.patches();
  const uint32_t merged = grid.mergedTokens();
  const uint32_t hidden = layout.hiddenSize;
  auto blockGemm = [&](const char *pipeline, uint32_t outputSize,
                       uint32_t inputSize) {
    return VisionGemmShape{pipeline, outputSize,   inputSize,
                           tokens,   kGemmRowTile, kGemmColumnTile};
  };
  auto mergerGemm = [&](const char *pipeline, uint32_t outputSize) {
    return VisionGemmShape{pipeline, outputSize,     layout.mergedHiddenSize,
                           merged,   kMergerRowTile, kMergerColumnTile};
  };
  // In VisionGemm order.
  return {
      {
          blockGemm("vision_gemm_m64n128_residual", hidden,
                    layout.patchDimension),
          blockGemm("vision_gemm_m64n128", 3 * hidden, hidden),
          blockGemm("vision_gemm_m64n128_residual", hidden, hidden),
          blockGemm("vision_gemm_m64n128_gelu_tanh",
                    layout.paddedIntermediateSize, hidden),
          blockGemm("vision_gemm_m64n128_residual", hidden,
                    layout.paddedIntermediateSize),
          mergerGemm("vision_gemm_m32n256_gelu_erf", layout.mergedHiddenSize),
          mergerGemm("vision_gemm_m32n256", layout.outputHiddenSize),
      },
      {tokens, roundUp(tokens, kKeyTile), layout.heads, kQueryTile,
       kQkDimension, layout.headDimension,
       1.0F / std::sqrt(static_cast<float>(layout.headDimension))},
  };
}

Vision::Vision(metal::MetalBackend &backend, const VisionWeights &model,
               uint32_t maximumPatches)
    : model_(model), maximumPatches_(maximumPatches) {
  const uint64_t total = scratchBytes(model.layout, maximumPatches);
  // Only kernels touch the arena, so it lives in private storage; the first
  // encode() zeroes it on the GPU (addArenaZero).
  arena_ = backend.allocateBuffer(total, metal::BufferStorage::Private,
                                  "vision-scratch");
  uint64_t cursor = 0;
  const auto layout = scratchLayout(model.layout, maximumPatches);
  for (uint32_t index = 0; index < layout.size(); ++index) {
    scratch_[index] = backend.view(arena_, cursor, layout[index]);
    cursor += alignArena(layout[index]);
  }
  if (cursor != total)
    throw std::logic_error("vision scratch arena mismatch");
}

void Vision::addGemm(CommandGraph &graph, const VisionGemmShape &shape,
                     const MetalBuffer &input, const VisionAffine &weights,
                     const MetalBuffer &output,
                     const MetalBuffer &residual) const {
  graph.add(shape.pipeline,
            {input, weights.weight, weights.bias, output, residual},
            VisionGemmParams{shape.outputSize, shape.inputSize},
            {shape.rowTiles(), shape.outputSize / shape.tileColumns, 1});
}

void Vision::addNorm(CommandGraph &graph, const MetalBuffer &input,
                     const VisionNorm &weights, const MetalBuffer &output,
                     uint32_t rows) const {
  graph.add("vision_layer_norm", {input, weights.weight, weights.bias, output},
            VisionNormParams{model_.layout.hiddenSize}, {rows, 1, 1});
}

// Appends the one-time zero fill of the whole arena. Padding rows and tokens
// are read by whole tiles but never consumed as results; zeroed storage
// guarantees they are finite. One GPU pass replaces a CPU write of the arena.
void Vision::addArenaZero(CommandGraph &graph) const {
  const uint64_t words = arena_.sizeBytes() / kZeroWordBytes;
  if (words > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("vision scratch arena exceeds the zero fill");
  const uint64_t groups =
      (words + CommandGraph::kDefaultThreads - 1) / CommandGraph::kDefaultThreads;
  graph.add("vision_zero_words", {arena_}, static_cast<uint32_t>(words),
            {std::min(groups, kZeroGroupLimit), 1, 1});
}

void Vision::encode(CommandGraph &graph, ImageGrid grid,
                    const MetalBuffer &pixels,
                    const MetalBuffer &embeddings) {
  const VisionLayout &layout = model_.layout;
  const uint32_t tokens = grid.patches();
  if (!grid.valid() || tokens > maximumPatches_) {
    throw std::invalid_argument("image grid exceeds the vision encoder");
  }
  if (!pixels || pixels.sizeBytes() < grid.pixelBytes()) {
    throw std::invalid_argument("image pixels do not cover the grid");
  }
  const uint64_t embeddingBytes = uint64_t{embeddingRows(grid)} *
                                  layout.outputHiddenSize * kBf16Bytes;
  if (!embeddings || embeddings.sizeBytes() < embeddingBytes) {
    throw std::invalid_argument("embedding buffer is too small for the grid");
  }

  if (!isArenaZeroQueued_) {
    addArenaZero(graph);
    isArenaZeroQueued_ = true;
  }

  const VisionKernelShapes shapes = kernelShapes(layout, grid);
  const VisionAttentionShape &attention = shapes.attention;
  const VisionGridParams gridParams{grid.height, grid.width};
  const VisionQkvParams qkvParams{tokens, attention.paddedTokens};
  const VisionAttentionParams attentionParams{tokens, attention.paddedTokens,
                                              attention.scale};
  const MetalBuffer &hidden = scratch(Scratch::Hidden);
  const MetalBuffer &normalized = scratch(Scratch::Normalized);
  const MetalBuffer &qkv = scratch(Scratch::Qkv);
  const MetalBuffer &context = scratch(Scratch::Context);

  graph.add("vision_patchify", {pixels, scratch(Scratch::Patches)}, gridParams,
            {tokens, 1, 1});
  graph.add("vision_prepare_positions",
            {model_.positionTable, scratch(Scratch::Positions),
             scratch(Scratch::RopeCos), scratch(Scratch::RopeSin)},
            gridParams, {tokens, 1, 1});
  addGemm(graph, shapes.gemm(VisionGemm::PatchEmbedding),
          scratch(Scratch::Patches), model_.patchEmbedding, hidden,
          scratch(Scratch::Positions));

  for (const VisionBlock &block : model_.blocks) {
    addNorm(graph, hidden, block.norm1, normalized, tokens);
    addGemm(graph, shapes.gemm(VisionGemm::Qkv), normalized, block.qkv, qkv,
            hidden);
    // Padded key tokens are zeroed by the prepare pass so every 128-key tile
    // is finite under the softmax mask.
    graph.add("vision_qkv_prepare",
              {qkv, scratch(Scratch::RopeCos), scratch(Scratch::RopeSin),
               scratch(Scratch::Queries), scratch(Scratch::Keys),
               scratch(Scratch::Values)},
              qkvParams, {attention.paddedTokens, 1, 1});
    graph.add("vision_attention",
              {scratch(Scratch::Queries), scratch(Scratch::Keys),
               scratch(Scratch::Values), scratch(Scratch::Attention)},
              attentionParams, {attention.queryTiles(), attention.heads, 1});
    graph.add("vision_attention_pack", {scratch(Scratch::Attention), context},
              qkvParams, {tokens, 1, 1});
    addGemm(graph, shapes.gemm(VisionGemm::Projection), context,
            block.projection, hidden, hidden);
    addNorm(graph, hidden, block.norm2, normalized, tokens);
    addGemm(graph, shapes.gemm(VisionGemm::Up), normalized, block.upProjection,
            scratch(Scratch::Intermediate), hidden);
    addGemm(graph, shapes.gemm(VisionGemm::Down),
            scratch(Scratch::Intermediate), block.downProjection, hidden,
            hidden);
  }

  // The merger reads the normalized rows as (merged, 4608): four consecutive
  // block-major patches form one merged token.
  addNorm(graph, hidden, model_.mergerNorm, normalized, tokens);
  addGemm(graph, shapes.gemm(VisionGemm::MergerUp), normalized,
          model_.mergerUpProjection, qkv, hidden);
  addGemm(graph, shapes.gemm(VisionGemm::MergerDown), qkv,
          model_.mergerDownProjection, embeddings, hidden);
}

void Vision::inject(CommandGraph &graph, const MetalBuffer &embeddings,
                    const MetalBuffer &packedHidden, uint32_t hiddenSize,
                    uint32_t sourceRow, uint32_t destinationRow,
                    uint32_t rows) {
  if (!rows || !hiddenSize)
    throw std::invalid_argument("vision injection requires rows and a width");
  const VisionInjectParams params{sourceRow, destinationRow, rows, hiddenSize};
  graph.add("vision_inject_embeddings", {embeddings, packedHidden}, params,
            {std::min<uint64_t>((uint64_t{rows} * hiddenSize + 255) / 256,
                                1024),
             1, 1});
}

} // namespace splash::ops
