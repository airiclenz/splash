// Parity of the Apple7/8 register-tile vision kernels with the MPP kernels
// they replace: every GEMM of one image's encode (vision_gemm_sgf_*), at its
// real input and output widths, on random bf16 inputs with a ragged last row
// tile, and attention (vision_attention_sgf) over ragged query and key tiles.
// The kernels accumulate in different orders, and MPP attention rounds its
// probabilities to bf16, so outputs agree within a tolerance, not bitwise.
// No model weights load.

#include "metal/abi/Vision.h"
#include "ops/Vision.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace splash;

namespace {

// Largest |register - mpp| / max(1, |mpp|) over the valid output rows: a few
// bf16 rounding steps at unit scale.
constexpr float kTolerance = 0.02F;
// Valid rows past the first full row tile, so the last tile is ragged.
constexpr uint32_t kRaggedRows = 5;
constexpr float kBiasScale = 0.1F;
// The outputs must reach unit scale, or a zero output would pass.
constexpr float kMinimumOutputMagnitude = 0.5F;
constexpr uint32_t kSeed = 0x5EED;
// Query and key magnitude: scores spread several units apart, so each
// softmax concentrates on a few keys and outputs reach unit scale.
constexpr float kAttentionScoreScale = 4.0F;
// Attention grids: 308 tokens leave the last query and key tiles ragged; 60
// tokens fit one 64-key block, so half of the register kernel's simdgroups
// own no keys.
constexpr std::array<ops::ImageGrid, 2> kAttentionGrids{{{14, 22}, {6, 10}}};

// Throws with message when condition is false.
void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

// A shared buffer of count bf16 values, zeroed.
metal::MetalBuffer bf16Buffer(metal::MetalBackend &backend, uint64_t count,
                              const char *label) {
  metal::MetalBuffer buffer = backend.allocateBuffer(
      count * sizeof(__bf16), metal::BufferStorage::Shared, label);
  std::fill_n(static_cast<__bf16 *>(buffer.contents()), count, __bf16(0.0F));
  return buffer;
}

// Fills the first count values with uniform values in [-scale, scale].
void fill(const metal::MetalBuffer &buffer, uint64_t count, float scale,
          std::mt19937 &random) {
  std::uniform_real_distribution<float> values(-scale, scale);
  auto *data = static_cast<__bf16 *>(buffer.contents());
  for (uint64_t index = 0; index < count; ++index)
    data[index] = __bf16(values(random));
}

// The bindings of one GEMM dispatch; the register and MPP tiles share them.
struct GemmBuffers final {
  metal::MetalBuffer input;
  metal::MetalBuffer weights;
  metal::MetalBuffer bias;
  metal::MetalBuffer residual;
};

// Runs pipeline over shape into a new zeroed output buffer.
metal::MetalBuffer runGemm(metal::MetalBackend &backend,
                           const ops::VisionGemmShape &shape,
                           std::string_view pipeline,
                           const GemmBuffers &buffers) {
  metal::MetalBuffer output = bf16Buffer(
      backend, uint64_t{shape.paddedRows()} * shape.outputSize, "output");
  metal::CommandGraph graph;
  graph.add(pipeline,
            {buffers.input, buffers.weights, buffers.bias, output,
             buffers.residual},
            VisionGemmParams{shape.outputSize, shape.inputSize},
            {shape.rowTiles(), shape.outputSize / shape.tileColumns, 1});
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  return output;
}

// The register tile of one encode GEMM matches its MPP tile on random rows,
// weights, bias and residual within kTolerance over the valid rows.
void testRegisterTileMatchesMpp(metal::MetalBackend &backend,
                                const ops::VisionGemmShape &mpp,
                                const ops::VisionGemmShape &registerTile,
                                std::mt19937 &random) {
  require(registerTile.tileRows == mpp.tileRows &&
              registerTile.tileColumns == mpp.tileColumns,
          std::string(registerTile.pipeline) + " tiles differ from the MPP tile");
  ops::VisionGemmShape shape = mpp;
  shape.rows = shape.tileRows + kRaggedRows;
  const uint64_t padded = shape.paddedRows();
  const uint64_t validOutputs = uint64_t{shape.rows} * shape.outputSize;
  const uint64_t weightCount = uint64_t{shape.outputSize} * shape.inputSize;
  GemmBuffers buffers{
      bf16Buffer(backend, padded * shape.inputSize, "input"),
      bf16Buffer(backend, weightCount, "weights"),
      bf16Buffer(backend, shape.outputSize, "bias"),
      bf16Buffer(backend, padded * shape.outputSize, "residual"),
  };
  fill(buffers.input, uint64_t{shape.rows} * shape.inputSize, 1.0F, random);
  fill(buffers.weights, weightCount,
       1.0F / std::sqrt(static_cast<float>(shape.inputSize)), random);
  fill(buffers.bias, shape.outputSize, kBiasScale, random);
  fill(buffers.residual, validOutputs, 1.0F, random);

  const metal::MetalBuffer expected =
      runGemm(backend, shape, mpp.pipeline, buffers);
  const metal::MetalBuffer actual =
      runGemm(backend, shape, registerTile.pipeline, buffers);

  const auto *reference = static_cast<const __bf16 *>(expected.contents());
  const auto *candidate = static_cast<const __bf16 *>(actual.contents());
  float maximumError = 0.0F;
  float maximumMagnitude = 0.0F;
  for (uint64_t index = 0; index < validOutputs; ++index) {
    const float value = static_cast<float>(reference[index]);
    const float error =
        std::fabs(static_cast<float>(candidate[index]) - value) /
        std::max(1.0F, std::fabs(value));
    maximumError = std::max(maximumError, std::isnan(error) ? INFINITY : error);
    maximumMagnitude = std::max(maximumMagnitude, std::fabs(value));
  }
  std::cout << registerTile.pipeline << " " << shape.outputSize << "x"
            << shape.inputSize << " max_error=" << maximumError << '\n';
  require(maximumMagnitude >= kMinimumOutputMagnitude,
          std::string(mpp.pipeline) + " produced no unit-scale output");
  require(maximumError <= kTolerance,
          std::string(registerTile.pipeline) + " strays from " + mpp.pipeline);
}

// Fills dimensions [0, headDimension) of every valid token row of a
// (heads, paddedTokens, width) buffer; padded tokens and dimensions stay zero,
// as vision_qkv_prepare leaves them.
void fillTokens(const metal::MetalBuffer &buffer,
                const ops::VisionAttentionShape &shape, uint32_t width,
                float scale, std::mt19937 &random) {
  std::uniform_real_distribution<float> values(-scale, scale);
  auto *data = static_cast<__bf16 *>(buffer.contents());
  for (uint32_t head = 0; head < shape.heads; ++head)
    for (uint32_t token = 0; token < shape.tokens; ++token) {
      __bf16 *row =
          data + (uint64_t{head} * shape.paddedTokens + token) * width;
      for (uint32_t dimension = 0; dimension < shape.headDimension; ++dimension)
        row[dimension] = __bf16(values(random));
    }
}

// The register attention kernel matches MPP attention on random queries,
// keys and values within kTolerance over every valid token of every head.
void testRegisterAttentionMatchesMpp(metal::MetalBackend &backend,
                                     const ops::VisionAttentionShape &mpp,
                                     const ops::VisionAttentionShape &registerTile,
                                     std::mt19937 &random) {
  require(registerTile.queryTile == mpp.queryTile &&
              registerTile.paddedTokens == mpp.paddedTokens,
          std::string(registerTile.pipeline) + " grid differs from the MPP grid");
  const uint64_t headTokens = uint64_t{mpp.heads} * mpp.paddedTokens;
  const metal::MetalBuffer queries =
      bf16Buffer(backend, headTokens * mpp.qkDimension, "queries");
  const metal::MetalBuffer keys =
      bf16Buffer(backend, headTokens * mpp.qkDimension, "keys");
  const metal::MetalBuffer values =
      bf16Buffer(backend, headTokens * mpp.headDimension, "values");
  fillTokens(queries, mpp, mpp.qkDimension, kAttentionScoreScale, random);
  fillTokens(keys, mpp, mpp.qkDimension, kAttentionScoreScale, random);
  fillTokens(values, mpp, mpp.headDimension, 1.0F, random);

  auto run = [&](std::string_view pipeline) {
    metal::MetalBuffer output =
        bf16Buffer(backend, headTokens * mpp.headDimension, "output");
    metal::CommandGraph graph;
    graph.add(pipeline, {queries, keys, values, output},
              VisionAttentionParams{mpp.tokens, mpp.paddedTokens, mpp.scale},
              {mpp.queryTiles(), mpp.heads, 1});
    static_cast<void>(backend.submitCommand(graph.dispatches()));
    return output;
  };
  const metal::MetalBuffer expected = run(mpp.pipeline);
  const metal::MetalBuffer actual = run(registerTile.pipeline);

  const auto *reference = static_cast<const __bf16 *>(expected.contents());
  const auto *candidate = static_cast<const __bf16 *>(actual.contents());
  float maximumError = 0.0F;
  float maximumMagnitude = 0.0F;
  for (uint64_t row = 0; row < headTokens; ++row) {
    if (row % mpp.paddedTokens >= mpp.tokens)
      continue;
    for (uint64_t index = row * mpp.headDimension;
         index < (row + 1) * mpp.headDimension; ++index) {
      const float value = static_cast<float>(reference[index]);
      const float error =
          std::fabs(static_cast<float>(candidate[index]) - value) /
          std::max(1.0F, std::fabs(value));
      maximumError =
          std::max(maximumError, std::isnan(error) ? INFINITY : error);
      maximumMagnitude = std::max(maximumMagnitude, std::fabs(value));
    }
  }
  std::cout << registerTile.pipeline << " tokens=" << mpp.tokens
            << " max_error=" << maximumError << '\n';
  require(maximumMagnitude >= kMinimumOutputMagnitude,
          std::string(mpp.pipeline) + " produced no unit-scale output");
  require(maximumError <= kTolerance,
          std::string(registerTile.pipeline) + " strays from " + mpp.pipeline);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: vision-tile-metal METALLIB\n";
    return 2;
  }
  try {
    metal::MetalBackend backend(argv[1]);
    std::mt19937 random(kSeed);
    const ops::ImageGrid grid{2, 2};
    const ops::VisionKernelShapes mpp = ops::Vision::kernelShapes(
        ops::VisionLayout{}, grid, ops::VisionGemmTile::Mpp);
    const ops::VisionKernelShapes registerTiles = ops::Vision::kernelShapes(
        ops::VisionLayout{}, grid, ops::VisionGemmTile::Register);
    for (size_t index = 0; index < mpp.gemms.size(); ++index)
      testRegisterTileMatchesMpp(backend, mpp.gemms[index],
                                 registerTiles.gemms[index], random);
    for (const ops::ImageGrid attentionGrid : kAttentionGrids)
      testRegisterAttentionMatchesMpp(
          backend,
          ops::Vision::kernelShapes(ops::VisionLayout{}, attentionGrid,
                                    ops::VisionGemmTile::Mpp)
              .attention,
          ops::Vision::kernelShapes(ops::VisionLayout{}, attentionGrid,
                                    ops::VisionGemmTile::Register)
              .attention,
          random);
    std::cout << "vision tile tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "vision tile tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
