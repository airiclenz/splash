#include "../../runtime/metal/MetalBackend.hpp"
#include "metal/abi/Linear.h"
#include "metal/abi/MoE.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::ComputeDispatch;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;

constexpr uint32_t kQuantGroup = 64;
// The Apple7/8 register-matrix tiles: 32 rows x 64 or 128 columns on four
// simdgroups. Only mma64 ships; the 128-column form (each simdgroup 32 x 32)
// timed 2.1x slower on an M2 Pro and was removed, so its rows appear only for
// a candidate metallib that carries it.
constexpr uint32_t kMmaThreads = 128;
struct MmaTile {
  std::string pipeline;
  uint32_t columns;
};
const MmaTile kMma64{"prefill_linear_q4_mma64", 64};
const MmaTile kMma128{"prefill_linear_q4_mma128", 128};

// A metallib without a tile lacks its functions: each side of a
// BASE-vs-candidate comparison times only the tiles it has.
bool hasFunction(const std::string &metallibPath, const std::string &name) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithURL:
      [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallibPath.c_str()]]
                                               error:&error];
  if (!library) throw std::runtime_error("could not load " + metallibPath);
  return [library.functionNames
      containsObject:[NSString stringWithUTF8String:name.c_str()]];
}

MetalBuffer shared(MetalBackend &backend, uint64_t bytes,
                   const std::string &label) {
  return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

void profileShape(MetalBackend &backend, uint32_t rows, uint32_t inputSize,
                  uint32_t outputSize, uint32_t tileRows,
                  uint32_t tileColumns, const std::string &sumPipeline,
                  const std::string &projectionPipeline,
                  const std::string &label, bool residual = false,
                  uint32_t threads = 256) {
  const uint32_t allocatedRows = (rows + tileRows - 1) / tileRows * tileRows;
  const uint64_t inputElements = uint64_t{allocatedRows} * inputSize;
  const uint64_t outputElements = uint64_t{allocatedRows} * outputSize;
  const uint64_t weightElements = uint64_t{inputSize} * outputSize;
  const uint64_t parameterElements = weightElements / kQuantGroup;
  const uint64_t sumElements =
      uint64_t{allocatedRows} * (inputSize / kQuantGroup);

  MetalBuffer input =
      shared(backend, inputElements * sizeof(__bf16), label + " input");
  MetalBuffer weights = shared(backend, weightElements / 2, label + " weights");
  MetalBuffer scales =
      shared(backend, parameterElements * sizeof(__bf16), label + " scales");
  MetalBuffer biases =
      shared(backend, parameterElements * sizeof(__bf16), label + " biases");
  MetalBuffer output =
      shared(backend, outputElements * sizeof(__bf16), label + " output");
  MetalBuffer sums =
      shared(backend, sumElements * sizeof(float), label + " sums");
  // Benchmark resident, nonzero data rather than demand-zero buffer pages.
  std::memset(input.contents(), 0x3c, input.sizeBytes());
  std::memset(weights.contents(), 0x5a, weights.sizeBytes());
  std::memset(scales.contents(), 0x3c, scales.sizeBytes());
  std::memset(biases.contents(), 0x3c, biases.sizeBytes());
  Q4PrefillParams params{outputSize, inputSize};

  ComputeDispatch sum;
  sum.pipelineName = sumPipeline;
  sum.buffers = {{0, input}, {1, sums}};
  sum.bytes = {{2, &params, sizeof(params)}};
  sum.threadgroups = {(rows + tileRows - 1) / tileRows, 1, 1};
  sum.threadsPerThreadgroup = {256, 1, 1};

  ComputeDispatch projection;
  projection.pipelineName = projectionPipeline;
  if (residual) {
    MetalBuffer residualRows =
        shared(backend, outputElements * sizeof(__bf16), label + " residual");
    projection.buffers = {{0, input}, {1, weights}, {2, scales}, {3, biases},
                          {4, residualRows}, {5, output}, {6, sums}};
    projection.bytes = {{7, &params, sizeof(params)}};
  } else {
    projection.buffers = {{0, input}, {1, weights}, {2, scales}, {3, biases},
                          {4, output}, {5, sums}};
    projection.bytes = {{6, &params, sizeof(params)}};
  }
  // X advances through row tiles. Metal therefore keeps one weight tile hot
  // while it schedules many row tiles before advancing output tile Y.
  projection.threadgroups = {(rows + tileRows - 1) / tileRows,
                             outputSize / tileColumns, 1};
  projection.threadsPerThreadgroup = {threads, 1, 1};

  for (uint32_t warmup = 0; warmup < 2; ++warmup) {
    static_cast<void>(backend.submit(sum));
    static_cast<void>(backend.submit(projection));
  }
  std::vector<double> sumSamples;
  std::vector<double> projectionSamples;
  for (uint32_t repeat = 0; repeat < 7; ++repeat) {
    sumSamples.push_back(backend.submit(sum).gpuSeconds * 1000.0);
    projectionSamples.push_back(backend.submit(projection).gpuSeconds * 1000.0);
  }
  std::cout << label << " rows=" << rows << " sums_ms=" << median(sumSamples)
            << " projection_ms=" << median(projectionSamples) << '\n';
}

// The fused up projection (silu(gate)*up plus the next GEMM's input sums)
// carries an extra gate buffer and an output-sums sink.
void profileUpSilu(MetalBackend &backend, uint32_t rows, uint32_t inputSize,
                   uint32_t outputSize, const std::string &projectionPipeline,
                   const std::string &label, uint32_t tileColumns = 256,
                   uint32_t threads = 256) {
  const uint32_t allocatedRows = (rows + 31) / 32 * 32;
  const uint64_t inputElements = uint64_t{allocatedRows} * inputSize;
  const uint64_t outputElements = uint64_t{allocatedRows} * outputSize;
  const uint64_t weightElements = uint64_t{inputSize} * outputSize;
  const uint64_t parameterElements = weightElements / kQuantGroup;
  const uint64_t sumElements =
      uint64_t{allocatedRows} * (inputSize / kQuantGroup);
  const uint64_t outputSumElements =
      uint64_t{allocatedRows} * (outputSize / kQuantGroup);

  MetalBuffer input =
      shared(backend, inputElements * sizeof(__bf16), label + " input");
  MetalBuffer weights = shared(backend, weightElements / 2, label + " weights");
  MetalBuffer scales =
      shared(backend, parameterElements * sizeof(__bf16), label + " scales");
  MetalBuffer biases =
      shared(backend, parameterElements * sizeof(__bf16), label + " biases");
  MetalBuffer gate = shared(backend, outputElements * sizeof(__bf16), label + " gate");
  MetalBuffer output =
      shared(backend, outputElements * sizeof(__bf16), label + " output");
  MetalBuffer sums =
      shared(backend, sumElements * sizeof(float), label + " sums");
  MetalBuffer outputSums = shared(backend, outputSumElements * sizeof(float),
                                  label + " output sums");
  std::memset(input.contents(), 0x3c, input.sizeBytes());
  std::memset(weights.contents(), 0x5a, weights.sizeBytes());
  std::memset(scales.contents(), 0x3c, scales.sizeBytes());
  std::memset(biases.contents(), 0x3c, biases.sizeBytes());
  std::memset(gate.contents(), 0x3c, gate.sizeBytes());
  Q4PrefillParams params{outputSize, inputSize};

  ComputeDispatch sum;
  sum.pipelineName = "prefill_linear_q4_sums32";
  sum.buffers = {{0, input}, {1, sums}};
  sum.bytes = {{2, &params, sizeof(params)}};
  sum.threadgroups = {(rows + 31) / 32, 1, 1};
  sum.threadsPerThreadgroup = {256, 1, 1};

  ComputeDispatch projection;
  projection.pipelineName = projectionPipeline;
  projection.buffers = {{0, input}, {1, weights}, {2, scales}, {3, biases},
                        {4, gate}, {5, output}, {6, sums}, {7, outputSums}};
  projection.bytes = {{8, &params, sizeof(params)}};
  projection.threadgroups = {(rows + 31) / 32, outputSize / tileColumns, 1};
  projection.threadsPerThreadgroup = {threads, 1, 1};

  for (uint32_t warmup = 0; warmup < 2; ++warmup) {
    static_cast<void>(backend.submit(sum));
    static_cast<void>(backend.submit(projection));
  }
  std::vector<double> sumSamples;
  std::vector<double> projectionSamples;
  for (uint32_t repeat = 0; repeat < 7; ++repeat) {
    sumSamples.push_back(backend.submit(sum).gpuSeconds * 1000.0);
    projectionSamples.push_back(backend.submit(projection).gpuSeconds * 1000.0);
  }
  std::cout << label << " rows=" << rows << " sums_ms=" << median(sumSamples)
            << " projection_ms=" << median(projectionSamples) << '\n';
}

void profileRmsSums(MetalBackend &backend, uint32_t rows, uint32_t width,
                    uint32_t tileRows, const std::string &sumPipeline,
                    const std::string &fusedPipeline,
                    const std::string &label) {
  const uint32_t allocatedRows = (rows + tileRows - 1) / tileRows * tileRows;
  MetalBuffer input = shared(
      backend, uint64_t{allocatedRows} * width * sizeof(__bf16),
      label + " input");
  MetalBuffer weight =
      shared(backend, uint64_t{width} * sizeof(__bf16), label + " weight");
  MetalBuffer output = shared(
      backend, uint64_t{allocatedRows} * width * sizeof(__bf16),
      label + " output");
  MetalBuffer sums = shared(
      backend, uint64_t{allocatedRows} * (width / kQuantGroup) * sizeof(float),
      label + " sums");
  Q4PrefillParams params{width, width};
  ComputeDispatch rms{"norm_rms",
                      {{0, input}, {1, weight}, {2, output}},
                      {{3, &width, sizeof(width)}},
                      {rows, 1, 1},
                      {256, 1, 1}};
  ComputeDispatch sum{sumPipeline,
                      {{0, output}, {1, sums}},
                      {{2, &params, sizeof(params)}},
                      {(rows + tileRows - 1) / tileRows, 1, 1},
                      {256, 1, 1}};
  ComputeDispatch fused{fusedPipeline,
                        {{0, input}, {1, weight}, {2, output}, {3, sums}},
                        {{4, &width, sizeof(width)}},
                        {rows, 1, 1},
                        {256, 1, 1}};
  for (uint32_t warmup = 0; warmup < 2; ++warmup) {
    static_cast<void>(backend.submit(rms));
    static_cast<void>(backend.submit(sum));
    static_cast<void>(backend.submit(fused));
  }
  std::vector<double> separateSamples;
  std::vector<double> fusedSamples;
  for (uint32_t repeat = 0; repeat < 7; ++repeat) {
    separateSamples.push_back(
        (backend.submit(rms).gpuSeconds + backend.submit(sum).gpuSeconds) *
        1000.0);
    fusedSamples.push_back(backend.submit(fused).gpuSeconds * 1000.0);
  }
  const double separate = median(separateSamples);
  const double fusedMilliseconds = median(fusedSamples);
  std::cout << label << " rows=" << rows << " separate_ms=" << separate
            << " fused_ms=" << fusedMilliseconds
            << " speedup=" << separate / fusedMilliseconds << '\n';
}

// A router scores pipeline and the rows x experts and threads per
// threadgroup that MoE::add dispatches it with.
struct RouterTile final {
  const char *pipeline;
  uint32_t rows;
  uint32_t experts;
  uint32_t threads;
};

constexpr std::array<RouterTile, 2> kMppRouterTiles{{
    {"moe_route_scores_q8_m8", 8, 32, 256},
    {"moe_route_scores_q8_m32", 32, 128, 256},
}};
constexpr std::array<RouterTile, 2> kRegisterRouterTiles{{
    {"moe_route_scores_q8_m8_sgf", 8, 16, 128},
    {"moe_route_scores_q8_m32_sgf", 32, 16, 256},
}};

// The MoE router scores at the 35B-A3B shape (hidden 2048, 256 experts,
// top 8): every score tile over the same synthetic q8 router, whose weights,
// scales and biases sit per quant group as (group * 256 + expert) [* 64].
// The samples alternate the tiles so drift lands on all of them alike. The
// Apple7/8 register tiles (_sgf) are timed beside the MPP tiles whenever the
// metallib carries them.
void profileRouter(MetalBackend &backend, uint32_t rows,
                   bool registerTiles) {
  constexpr uint32_t kHidden = 2048;
  constexpr uint32_t kExperts = 256;
  constexpr uint32_t kTopK = 8;
  const uint64_t parameterElements =
      uint64_t{kHidden / kQuantGroup} * kExperts;
  MetalBuffer input = shared(backend, uint64_t{rows} * kHidden * sizeof(__bf16),
                             "router input");
  MetalBuffer weights =
      shared(backend, parameterElements * kQuantGroup, "router weights");
  MetalBuffer scales = shared(backend, parameterElements * sizeof(__bf16),
                              "router scales");
  MetalBuffer biases = shared(backend, parameterElements * sizeof(__bf16),
                              "router biases");
  MetalBuffer scores = shared(
      backend, uint64_t{rows} * kExperts * sizeof(__bf16), "router scores");
  std::memset(input.contents(), 0x3c, input.sizeBytes());
  std::memset(weights.contents(), 0x5a, weights.sizeBytes());
  std::memset(scales.contents(), 0x3c, scales.sizeBytes());
  std::memset(biases.contents(), 0x3c, biases.sizeBytes());
  const MoeRouteParams params{rows, kHidden, kExperts, kTopK};

  std::vector<RouterTile> tiles(kMppRouterTiles.begin(),
                                kMppRouterTiles.end());
  if (registerTiles)
    tiles.insert(tiles.end(), kRegisterRouterTiles.begin(),
                 kRegisterRouterTiles.end());
  std::vector<ComputeDispatch> dispatches;
  for (const RouterTile &tile : tiles)
    dispatches.push_back(
        {tile.pipeline,
         {{0, input}, {1, weights}, {2, scales}, {3, biases}, {4, scores}},
         {{5, &params, sizeof(params)}},
         {(rows + tile.rows - 1) / tile.rows, kExperts / tile.experts, 1},
         {tile.threads, 1, 1}});
  for (uint32_t warmup = 0; warmup < 2; ++warmup)
    for (const ComputeDispatch &dispatch : dispatches)
      static_cast<void>(backend.submit(dispatch));
  std::vector<std::vector<double>> samples(dispatches.size());
  for (uint32_t repeat = 0; repeat < 7; ++repeat)
    for (size_t index = 0; index < dispatches.size(); ++index)
      samples[index].push_back(backend.submit(dispatches[index]).gpuSeconds *
                               1000.0);
  for (size_t index = 0; index < tiles.size(); ++index)
    std::cout << tiles[index].pipeline << " rows=" << rows
              << " router_ms=" << median(samples[index]) << '\n';
}

// An Apple7/8 MoE gate/up register tile, its precomputed-sums variant and
// its column tile.
struct ExpertTile final {
  const char *pipeline;
  const char *sumsPipeline;
  uint32_t rows;
  uint32_t columns;
};

constexpr std::array<ExpertTile, 2> kExpertTiles{{
    {"moe_expert_gate_up_q4_mma_m8", "moe_expert_gate_up_q4_mma_m8_sums", 8, 64},
    {"moe_expert_gate_up_q4_mma_m32", "moe_expert_gate_up_q4_mma_m32_sums", 32,
     32},
}};
constexpr uint32_t kExpertPairs = 6;
constexpr uint32_t kExpertSubmits = 9;

uint32_t mixBits(uint64_t value) {
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdULL;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ULL;
  value ^= value >> 33;
  return static_cast<uint32_t>(value);
}

double medianSubmit(MetalBackend &backend, const ComputeDispatch &dispatch) {
  std::vector<double> samples;
  for (uint32_t submit = 0; submit < kExpertSubmits; ++submit)
    samples.push_back(backend.submit(dispatch).gpuSeconds * 1000.0);
  return median(samples);
}

// The 35B-A3B fused gate/up expert pass (hidden 2048, intermediate 512, 256
// experts, top 8) on the Apple7/8 register tiles, against the variant reading
// per-row, per-group input sums that moe_expert_input_sums_mma precomputes in
// the tile's order. Tokens route to hashed experts, so the tiles carry
// ragged row counts. The grouped input holds varied bf16 of mixed exponents,
// and both outputs start zeroed, so the variant's output must match the
// tile's bit for bit before it is timed. Each of kExpertPairs pairs times the
// tile (A) and the variant (B) in ABBA order, each sample a median of
// kExpertSubmits submits; the variant's sums kernel is reported apart.
void profileMoeExpert(MetalBackend &backend, uint32_t tokens,
                      const ExpertTile &tile, bool sumsVariant) {
  constexpr uint32_t kHidden = 2048;
  constexpr uint32_t kIntermediate = 512;
  constexpr uint32_t kExperts = 256;
  constexpr uint32_t kTopK = 8;
  constexpr uint32_t kGroups = kHidden / kQuantGroup;
  std::vector<uint32_t> routes(kExperts, 0);
  for (uint32_t token = 0; token < tokens; ++token)
    for (uint32_t k = 0; k < kTopK; ++k)
      ++routes[mixBits(uint64_t{token} * kTopK + k) % kExperts];
  std::vector<MoeTileDescriptor> descriptors;
  for (uint32_t expert = 0; expert < kExperts; ++expert)
    for (uint32_t row = 0; row < routes[expert]; row += tile.rows)
      descriptors.push_back(
          {expert, std::min(tile.rows, routes[expert] - row)});
  const auto tileCount = static_cast<uint32_t>(descriptors.size());
  const uint64_t groupedRows = uint64_t{tileCount} * tile.rows;

  const uint64_t elements = uint64_t{kIntermediate} * kHidden;
  const uint64_t weightBytes = elements / 2;
  const uint64_t parameterBytes = elements / 32;
  const uint64_t stride = weightBytes + 2 * parameterBytes;
  const auto packedExperts = [&](const std::string &name) {
    MetalBuffer packed = shared(backend, stride * kExperts, "moe " + name);
    auto *bytes = static_cast<uint8_t *>(packed.contents());
    std::memset(bytes, 0x5a, packed.sizeBytes());
    for (uint32_t expert = 0; expert < kExperts; ++expert)
      std::memset(bytes + expert * stride + weightBytes, 0x3c,
                  2 * parameterBytes);
    return packed;
  };
  MetalBuffer gate = packedExperts("gate");
  MetalBuffer up = packedExperts("up");
  MetalBuffer input = shared(backend, groupedRows * kHidden * sizeof(__bf16),
                             "moe grouped input");
  auto *inputBits = static_cast<uint16_t *>(input.contents());
  for (uint64_t index = 0; index < groupedRows * kHidden; ++index) {
    const uint32_t bits = mixBits(index);
    // Sign, an exponent in [2^-4, 2^3] and seven mantissa bits.
    inputBits[index] = static_cast<uint16_t>(
        ((bits & 1) << 15) | ((123 + (bits >> 1) % 8) << 7) | ((bits >> 4) & 0x7f));
  }
  MetalBuffer tiles = shared(backend, descriptors.size() * sizeof(MoeTileDescriptor),
                             "moe tiles");
  std::memcpy(tiles.contents(), descriptors.data(), tiles.sizeBytes());
  MetalBuffer count = shared(backend, sizeof(uint32_t), "moe tile count");
  std::memcpy(count.contents(), &tileCount, sizeof(tileCount));
  MetalBuffer sums = shared(backend, groupedRows * kGroups * sizeof(float),
                            "moe input sums");
  const uint64_t outputBytes = groupedRows * kIntermediate * sizeof(__bf16);
  MetalBuffer output = shared(backend, outputBytes, "moe output");
  MetalBuffer sumsOutput = shared(backend, outputBytes, "moe sums output");
  std::memset(output.contents(), 0, outputBytes);
  std::memset(sumsOutput.contents(), 0, outputBytes);
  const MoeExpertParams params{kHidden, kIntermediate, kExperts, 0, stride,
                               stride};
  const uint32_t inputSize = kHidden;

  // The shared-expert slabs are never read: no descriptor names expert 256.
  const ComputeDispatch inTile{
      tile.pipeline,
      {{0, input}, {1, tiles}, {2, count}, {3, gate}, {4, up}, {5, gate},
       {6, up}, {7, output}},
      {{8, &params, sizeof(params)}},
      {kIntermediate / tile.columns, tileCount, 1},
      {kMmaThreads, 1, 1}};
  const std::string label = std::string(tile.pipeline) + " tokens=" +
                            std::to_string(tokens) +
                            " tiles=" + std::to_string(tileCount);
  if (!sumsVariant) {
    static_cast<void>(backend.submit(inTile));
    std::cout << label << " tile_ms=" << medianSubmit(backend, inTile) << '\n';
    return;
  }
  const ComputeDispatch sum{
      "moe_expert_input_sums_mma",
      {{0, input}, {1, sums}},
      {{2, &inputSize, sizeof(inputSize)}},
      {kGroups / 4, static_cast<uint32_t>(groupedRows / 8), 1},
      {kMmaThreads, 1, 1}};
  const ComputeDispatch precomputed{
      tile.sumsPipeline,
      {{0, input}, {1, tiles}, {2, count}, {3, gate}, {4, up}, {5, gate},
       {6, up}, {7, sumsOutput}, {8, sums}},
      {{9, &params, sizeof(params)}},
      {kIntermediate / tile.columns, tileCount, 1},
      {kMmaThreads, 1, 1}};
  static_cast<void>(backend.submit(sum));
  static_cast<void>(backend.submit(inTile));
  static_cast<void>(backend.submit(precomputed));
  if (std::memcmp(output.contents(), sumsOutput.contents(), outputBytes) != 0)
    throw std::runtime_error(label + ": precomputed sums change the output");

  std::vector<double> tileSamples;
  std::vector<double> sumsSamples;
  uint32_t wins = 0;
  for (uint32_t pair = 0; pair < kExpertPairs; ++pair) {
    double a = 0;
    double b = 0;
    if (pair % 2 == 0) {
      a = medianSubmit(backend, inTile);
      b = medianSubmit(backend, precomputed);
    } else {
      b = medianSubmit(backend, precomputed);
      a = medianSubmit(backend, inTile);
    }
    tileSamples.push_back(a);
    sumsSamples.push_back(b);
    wins += b < a ? 1 : 0;
  }
  const double tileMs = median(tileSamples);
  const double sumsMs = median(sumsSamples);
  std::cout << label << " tile_ms=" << tileMs << " sums_variant_ms=" << sumsMs
            << " change_pct=" << (sumsMs / tileMs - 1.0) * 100.0
            << " variant_wins=" << wins << '/' << kExpertPairs
            << " sums_kernel_ms=" << medianSubmit(backend, sum)
            << " bitwise=equal\n";
}

void run(const std::string &metallibPath) {
  MetalBackend backend(metallibPath);
  constexpr std::array<uint32_t, 13> kProfileRows{
      17u, 24u, 32u, 40u, 48u, 56u, 64u, 128u, 256u, 384u, 1'024u,
      1'808u, 2'048u};
  const auto &capabilities = backend.capabilities();
  std::cout << "device=\"" << capabilities.deviceName
            << "\" apple_gpu_family=" << capabilities.appleGpuFamily
            << " recommended_working_set_bytes="
            << capabilities.recommendedMaxWorkingSetBytes
            << " threadgroup_memory="
            << capabilities.maxThreadgroupMemoryBytes
            << " threadgroup_width=" << capabilities.maxThreadgroupWidth
            << '\n';
  // Only Apple7/8 select the register-matrix tiles; their rows carry the
  // pipeline name.
  const bool registerMatrix = capabilities.appleGpuFamily < 9;
  std::vector<MmaTile> mmaTiles{kMma64};
  if (hasFunction(metallibPath, kMma128.pipeline)) mmaTiles.push_back(kMma128);
  const auto profileMma = [&](uint32_t rows, uint32_t inputSize,
                              uint32_t outputSize, const std::string &label) {
    if (!registerMatrix) return;
    for (const MmaTile &tile : mmaTiles)
      profileShape(backend, rows, inputSize, outputSize, 32, tile.columns,
                   "prefill_linear_q4_sums32", tile.pipeline,
                   label + "_" + tile.pipeline, false, kMmaThreads);
  };
  // Each plain projection with the three tiles the policy may choose: N128
  // and N256 on eight SIMD groups and N128 on four (the Apple10 default),
  // plus the register-matrix tiles on Apple7/8.
  const auto profileTiles = [&](uint32_t rows, uint32_t inputSize,
                                uint32_t outputSize, const std::string &label) {
    profileShape(backend, rows, inputSize, outputSize, 32, 128,
                 "prefill_linear_q4_sums32", "prefill_linear_q4_n128",
                 label + "_m32n128");
    profileShape(backend, rows, inputSize, outputSize, 32, 256,
                 "prefill_linear_q4_sums32", "prefill_linear_q4_n256",
                 label + "_m32n256");
    profileShape(backend, rows, inputSize, outputSize, 32, 128,
                 "prefill_linear_q4_sums32", "prefill_linear_q4_n128_sg4",
                 label + "_m32n128_sg4", false, 128);
    profileMma(rows, inputSize, outputSize, label);
  };
  for (uint32_t rows : kProfileRows) {
    profileTiles(rows, 5120, 17408, "ffn_up");
    profileShape(backend, rows, 17408, 5120, 32, 256, "prefill_linear_q4_sums32",
                 "prefill_linear_q4_n256", "ffn_down_m32n256");
    profileMma(rows, 17408, 5120, "ffn_down");
    profileUpSilu(backend, rows, 5120, 17408,
                  "prefill_linear_q4_n256_up_silu_sums", "ffn_up_silu_m32n256");
    profileUpSilu(backend, rows, 5120, 17408,
                  "prefill_linear_q4_n128_up_silu_sums_sg4",
                  "ffn_up_silu_m32n128_sg4", 128, 128);
    if (registerMatrix)
      for (const MmaTile &tile : mmaTiles)
        profileUpSilu(backend, rows, 5120, 17408, tile.pipeline + "_up_silu_sums",
                      "ffn_up_silu_" + tile.pipeline + "_up_silu_sums",
                      tile.columns, kMmaThreads);
    profileTiles(rows, 5120, 16640, "gdn_input");
    profileTiles(rows, 5120, 14336, "attention_input");
    profileTiles(rows, 6144, 5120, "mixer_output");
    profileTiles(rows, 25600, 5120, "draft_context");
    profileTiles(rows, 2048, 12544, "moe_gdn_input");
    profileTiles(rows, 2048, 9216, "moe_attention_input");
    profileTiles(rows, 4096, 2048, "moe_mixer_output");
  }
  for (uint32_t rows : kProfileRows) {
    profileShape(backend, rows, 5120, 6144, 32, 128, "prefill_linear_q4_sums32",
                 "prefill_linear_q4_n128", "draft_qkv_m32n128");
    profileMma(rows, 5120, 6144, "draft_qkv");
  }
  // The residual FFN down projection with both tile widths: the executor
  // picks 128 columns when the 256-column grid would leave cores idle.
  for (uint32_t rows : kProfileRows) {
    profileShape(backend, rows, 17408, 5120, 32, 128, "prefill_linear_q4_sums32",
                 "prefill_linear_q4_n128_residual",
                 "ffn_down_residual_m32n128", true);
    profileShape(backend, rows, 17408, 5120, 32, 256, "prefill_linear_q4_sums32",
                 "prefill_linear_q4_n256_residual",
                 "ffn_down_residual_m32n256", true);
    profileShape(backend, rows, 17408, 5120, 32, 128, "prefill_linear_q4_sums32",
                 "prefill_linear_q4_n128_residual_sg4",
                 "ffn_down_residual_m32n128_sg4", true, 128);
    if (registerMatrix)
      for (const MmaTile &tile : mmaTiles)
        profileShape(backend, rows, 17408, 5120, 32, tile.columns,
                     "prefill_linear_q4_sums32", tile.pipeline + "_residual",
                     "ffn_down_residual_" + tile.pipeline + "_residual", true,
                     kMmaThreads);
  }
  for (uint32_t rows : {128u, 512u, 1'808u, 2'048u}) {
    profileRmsSums(backend, rows, 5120, 32, "prefill_linear_q4_sums32",
                   "prefill_norm_rms_sums32", "rms_sums_m32");
  }
  // Batch 1 verifies eight rows; the prefill chunks use the profile rows.
  const bool registerRouter =
      hasFunction(metallibPath, kRegisterRouterTiles[0].pipeline);
  profileRouter(backend, 8, registerRouter);
  for (uint32_t rows : kProfileRows)
    profileRouter(backend, rows, registerRouter);
  // The register expert tiles run on any GPU; only Apple7/8 select them.
  const bool sumsVariant = hasFunction(metallibPath, kExpertTiles[0].sumsPipeline);
  for (uint32_t tokens : {8u, 32u, 128u, 512u, 2'048u})
    for (const ExpertTile &tile : kExpertTiles)
      profileMoeExpert(backend, tokens, tile, sumsVariant);
}

} // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 2) {
      std::cerr << "usage: q4_prefill_profile <metallib>\n";
      return 2;
    }
    try {
      run(argv[1]);
    } catch (const std::exception &error) {
      std::cerr << "FAIL: " << error.what() << '\n';
      return 1;
    }
  }
  return 0;
}
