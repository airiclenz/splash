// Times Splash's vision encoder kernels on random bf16 inputs, with no model
// loaded: each GEMM of one image's encode (vision_gemm_* at the patch
// embedding, block QKV, projection, up and down, and merger up and down
// shapes) and attention, at 1024, 4096 and 16384 patches by default.
// Shapes and pipelines come from ops::Vision::kernelShapes with the tile
// family of the device's GPU (Vision::gemmTile), so they match production
// dispatch. Each
// case reports the median GPU time of its one-dispatch command. These are
// kernel timings, not a correctness oracle (vision-encoder is).
//
// Variant 0 runs METALLIB's production pipeline. --compare-metallib adds the
// same pipeline from a second metallib and fails when its output bits differ.
// --alternate KERNEL=PIPELINE (repeatable) adds PIPELINE from METALLIB for
// every case of the production pipeline KERNEL, e.g. a register tile against
// the MPP tile, and fails when its output strays beyond --tolerance from
// variant 0: |a - b| / max(1, |a|) over the valid output rows. KERNEL may name
// either tile family's pipeline: on Apple7/8 production runs vision_gemm_sgf_*
// and vision_attention_sgf, so vision_gemm_sgf_m64n128=vision_gemm_m64n128 or
// vision_attention_sgf=vision_attention times the MPP kernel against it.
// --allow-diff turns either failure into a warning. Variants rotate order
// every round (ABBA for two) on identical inputs.
//
// usage: vision-profile METALLIB [--patches 1024,4096,16384] [--kernels LIST]
//                       [--repeat N] [--compare-metallib PATH]
//                       [--alternate KERNEL=PIPELINE]... [--tolerance X]
//                       [--allow-diff]
// --kernels takes case kernel names: patch-embedding, qkv, projection, up,
// down, merger-up, merger-down, attention.
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Vision.h"
#include "ops/Vision.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;

constexpr std::array<uint32_t, 3> kDefaultPatches{1024, 4096, 16384};
constexpr double kWarmupSeconds = 0.1;
constexpr float kDefaultTolerance = 0.02F;
constexpr uint64_t kSeed = 0x5EED5EEDULL;
// Bias magnitude relative to the unit-scale products.
constexpr float kBiasScale = 0.1F;
constexpr std::string_view kResidualSuffix = "_residual";

// A case kernel: one GEMM of the encode, or attention when gemm is empty.
struct Kernel final {
  std::string_view name;
  std::optional<VisionGemm> gemm;
};

constexpr std::array kKernels{
    Kernel{"patch-embedding", VisionGemm::PatchEmbedding},
    Kernel{"qkv", VisionGemm::Qkv},
    Kernel{"projection", VisionGemm::Projection},
    Kernel{"up", VisionGemm::Up},
    Kernel{"down", VisionGemm::Down},
    Kernel{"merger-up", VisionGemm::MergerUp},
    Kernel{"merger-down", VisionGemm::MergerDown},
    Kernel{"attention", std::nullopt},
};

uint16_t toBfloat(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7FFFU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(bits >> 16);
}

float fromBfloat(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
}

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  // Uniform in [-1, 1).
  float unit() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((state_ >> 40) & 0xFFFFFF) / 8388608.0F - 1.0F;
  }

private:
  uint64_t state_;
};

MetalBuffer shared(MetalBackend &backend, uint64_t elements, const char *label) {
  MetalBuffer buffer =
      backend.allocateBuffer(elements * sizeof(uint16_t), BufferStorage::Shared, label);
  std::memset(buffer.contents(), 0, buffer.sizeBytes());
  return buffer;
}

// Fills `count` bf16 values from `first` with uniform values of magnitude scale.
void fill(const MetalBuffer &buffer, Random &random, float scale, uint64_t first,
          uint64_t count) {
  auto *values = static_cast<uint16_t *>(buffer.contents()) + first;
  for (uint64_t index = 0; index < count; ++index)
    values[index] = toBfloat(random.unit() * scale);
}

// The output elements a variant must reproduce: `blocks` runs of
// validElements, each starting blockElements after the previous one.
struct OutputRegion final {
  uint32_t blocks = 1;
  uint64_t blockElements = 0;
  uint64_t validElements = 0;
};

// One timed single-dispatch graph on its own inputs. The graph's bindings keep
// every input alive; re-submitting it rewrites the same output.
struct Workload final {
  MetalBackend *backend = nullptr;
  CommandGraph graph;
  MetalBuffer output;
  OutputRegion region;
};

// Random activations, weights scaled by 1/sqrt(inputSize) so outputs stay near
// unit scale, and a separate residual so repeated submits stay identical.
Workload gemmWorkload(MetalBackend &backend, const VisionGemmShape &shape,
                      std::string_view pipeline) {
  Random random(kSeed);
  const uint64_t rows = shape.paddedRows();
  const MetalBuffer input = shared(backend, rows * shape.inputSize, "vision input");
  const MetalBuffer weights =
      shared(backend, uint64_t{shape.outputSize} * shape.inputSize, "vision weights");
  const MetalBuffer bias = shared(backend, shape.outputSize, "vision bias");
  const MetalBuffer output = shared(backend, rows * shape.outputSize, "vision output");
  fill(input, random, 1.0F, 0, uint64_t{shape.rows} * shape.inputSize);
  fill(weights, random, 1.0F / std::sqrt(static_cast<float>(shape.inputSize)), 0,
       uint64_t{shape.outputSize} * shape.inputSize);
  fill(bias, random, kBiasScale, 0, shape.outputSize);
  // Non-residual tiles never read the residual binding; production binds the
  // hidden rows there, this binds the output.
  MetalBuffer residual = output;
  if (std::string_view(shape.pipeline).ends_with(kResidualSuffix)) {
    residual = shared(backend, rows * shape.outputSize, "vision residual");
    fill(residual, random, 1.0F, 0, uint64_t{shape.rows} * shape.outputSize);
  }

  Workload workload{&backend, CommandGraph(1), output,
                    {1, 0, uint64_t{shape.rows} * shape.outputSize}};
  workload.graph.add(pipeline, {input, weights, bias, output, residual},
                     VisionGemmParams{shape.outputSize, shape.inputSize},
                     {shape.rowTiles(), shape.outputSize / shape.tileColumns, 1});
  return workload;
}

// Random queries, keys and values for the valid tokens of every head, zero in
// the padded tokens and the query/key dimensions past headDimension, as
// vision_qkv_prepare leaves them.
Workload attentionWorkload(MetalBackend &backend, const VisionAttentionShape &shape,
                           std::string_view pipeline) {
  Random random(kSeed);
  const uint64_t headTokens = uint64_t{shape.heads} * shape.paddedTokens;
  const MetalBuffer queries = shared(backend, headTokens * shape.qkDimension, "vision queries");
  const MetalBuffer keys = shared(backend, headTokens * shape.qkDimension, "vision keys");
  const MetalBuffer values = shared(backend, headTokens * shape.headDimension, "vision values");
  const MetalBuffer output = shared(backend, headTokens * shape.headDimension, "vision output");
  for (uint64_t row = 0; row < headTokens; ++row) {
    if (row % shape.paddedTokens >= shape.tokens) continue;
    fill(queries, random, 1.0F, row * shape.qkDimension, shape.headDimension);
    fill(keys, random, 1.0F, row * shape.qkDimension, shape.headDimension);
    fill(values, random, 1.0F, row * shape.headDimension, shape.headDimension);
  }

  Workload workload{&backend, CommandGraph(1), output,
                    {shape.heads, uint64_t{shape.paddedTokens} * shape.headDimension,
                     uint64_t{shape.tokens} * shape.headDimension}};
  workload.graph.add(pipeline, {queries, keys, values, output},
                     VisionAttentionParams{shape.tokens, shape.paddedTokens, shape.scale},
                     {shape.queryTiles(), shape.heads, 1});
  return workload;
}

// How a variant's valid output compares with variant 0's.
struct Comparison final {
  bool isIdentical = true;
  // Largest |a - b| / max(1, |a|), a from variant 0.
  float maxError = 0.0F;
};

Comparison compareOutputs(const Workload &reference, const Workload &candidate) {
  const auto *expected = static_cast<const uint16_t *>(reference.output.contents());
  const auto *actual = static_cast<const uint16_t *>(candidate.output.contents());
  const OutputRegion &region = reference.region;
  Comparison comparison;
  for (uint64_t block = 0; block < region.blocks; ++block) {
    const uint64_t first = block * region.blockElements;
    for (uint64_t index = first; index < first + region.validElements; ++index) {
      if (expected[index] == actual[index]) continue;
      comparison.isIdentical = false;
      const float wanted = fromBfloat(expected[index]);
      const float error = std::abs(wanted - fromBfloat(actual[index])) /
                          std::max(1.0F, std::abs(wanted));
      // A NaN on one side counts as an infinite error, which no tolerance passes.
      comparison.maxError =
          std::isnan(error) ? std::numeric_limits<float>::infinity()
                            : std::max(comparison.maxError, error);
    }
  }
  return comparison;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

// Warms every variant, then returns each one's median GPU milliseconds,
// rotating the order every round (ABBA for two). Each graph is one dispatch,
// so its command's GPU time is the kernel's.
std::vector<double> measure(std::vector<Workload> &workloads, uint32_t repeat) {
  const size_t count = workloads.size();
  auto submit = [&](size_t variant) {
    Workload &workload = workloads[variant];
    return workload.backend->submitCommand(workload.graph.dispatches()).gpuSeconds;
  };
  double warmup = 0.0;
  while (warmup < kWarmupSeconds)
    for (size_t i = 0; i < count; ++i)
      warmup += submit(i);

  std::vector<std::vector<double>> samples(count);
  for (uint32_t round = 0; round < repeat; ++round)
    for (size_t offset = 0; offset < count; ++offset) {
      const size_t i = (round + offset) % count;
      samples[i].push_back(submit(i) * 1000.0);
    }
  std::vector<double> milliseconds;
  milliseconds.reserve(count);
  for (std::vector<double> &variant : samples)
    milliseconds.push_back(median(std::move(variant)));
  return milliseconds;
}

uint32_t parseCount(std::string_view text, uint32_t minimum, uint32_t maximum,
                    std::string_view option) {
  uint32_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      value < minimum || value > maximum)
    throw std::invalid_argument(std::string(option) + " requires integers from " +
                                std::to_string(minimum) + " to " + std::to_string(maximum));
  return value;
}

std::vector<std::string> splitList(const std::string &text) {
  std::vector<std::string> items;
  size_t start = 0;
  while (start <= text.size()) {
    const size_t comma = text.find(',', start);
    items.push_back(text.substr(start, comma == std::string::npos ? std::string::npos
                                                                   : comma - start));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return items;
}

// The production pipelines --alternate may replace, of either tile family.
std::set<std::string, std::less<>> productionPipelines() {
  std::set<std::string, std::less<>> names;
  for (const VisionGemmTile tile : {VisionGemmTile::Mpp, VisionGemmTile::Register}) {
    const VisionKernelShapes shapes = Vision::kernelShapes(VisionLayout{}, ImageGrid{2, 2}, tile);
    for (const VisionGemmShape &shape : shapes.gemms)
      names.insert(shape.pipeline);
    names.insert(shapes.attention.pipeline);
  }
  return names;
}

struct Options final {
  std::string metallib;
  std::string comparisonLibrary;
  std::vector<uint32_t> patches{kDefaultPatches.begin(), kDefaultPatches.end()};
  std::vector<Kernel> kernels{kKernels.begin(), kKernels.end()};
  // Production pipeline to alternate pipeline. Graphs view these strings, so
  // the options outlive every graph.
  std::map<std::string, std::string, std::less<>> alternates;
  uint32_t repeat = 5;
  float tolerance = kDefaultTolerance;
  bool isDiffAllowed = false;
};

std::vector<Kernel> parseKernels(const std::string &text) {
  std::vector<Kernel> kernels;
  for (const std::string &name : splitList(text)) {
    const auto found = std::find_if(kKernels.begin(), kKernels.end(),
                                    [&](const Kernel &kernel) { return kernel.name == name; });
    if (found == kKernels.end())
      throw std::invalid_argument("--kernels does not know " + name);
    kernels.push_back(*found);
  }
  return kernels;
}

void parseAlternate(const std::string &text, Options &options) {
  const size_t equals = text.find('=');
  const std::string kernel = text.substr(0, equals);
  if (equals == std::string::npos || equals + 1 == text.size() ||
      !productionPipelines().contains(kernel))
    throw std::invalid_argument("--alternate takes KERNEL=PIPELINE with KERNEL a "
                                "vision_gemm_* or vision_attention* pipeline");
  options.alternates[kernel] = text.substr(equals + 1);
}

float parseTolerance(const std::string &text) {
  float value = 0.0F;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      !(value >= 0.0F))
    throw std::invalid_argument("--tolerance takes a non-negative number");
  return value;
}

Options parseOptions(int argc, const char *argv[]) {
  Options options;
  options.metallib = argv[1];
  for (int index = 2; index < argc; ++index) {
    const std::string option(argv[index]);
    if (option == "--allow-diff") {
      options.isDiffAllowed = true;
      continue;
    }
    if (index + 1 >= argc)
      throw std::invalid_argument(option + " requires a value");
    const std::string value(argv[++index]);
    if (option == "--patches") {
      options.patches.clear();
      for (const std::string &item : splitList(value)) {
        const uint32_t patches = parseCount(item, 4, kMaximumImagePatches, option);
        if (patches % 4)
          throw std::invalid_argument("--patches takes multiples of four");
        options.patches.push_back(patches);
      }
    } else if (option == "--kernels") options.kernels = parseKernels(value);
    else if (option == "--repeat")
      options.repeat = parseCount(value, 1, std::numeric_limits<uint32_t>::max(), option);
    else if (option == "--compare-metallib") options.comparisonLibrary = value;
    else if (option == "--alternate") parseAlternate(value, options);
    else if (option == "--tolerance") options.tolerance = parseTolerance(value);
    else throw std::invalid_argument("unknown option " + option);
  }
  return options;
}

// The squarest valid grid of `patches` patches (a multiple of four): GEMM and
// attention depend only on the patch count.
ImageGrid gridFor(uint32_t patches) {
  uint32_t height = static_cast<uint32_t>(std::sqrt(static_cast<double>(patches)));
  while (height > 2 && (height % 2 || patches % height || (patches / height) % 2))
    --height;
  return {height, patches / height};
}

// Fails on a mismatch, or warns with --allow-diff.
void expectSame(bool isSame, const std::string &message, bool isDiffAllowed) {
  if (isSame) return;
  if (!isDiffAllowed) throw std::runtime_error(message);
  std::cerr << "warning: " << message << '\n';
}

// One variant of a case: which metallib and which pipeline it runs.
struct Variant final {
  MetalBackend *backend = nullptr;
  size_t metallib = 0;
  std::string_view pipeline;
};

std::string json(const std::string &name, const Variant &variant, size_t index,
                 double milliseconds, const Comparison &comparison) {
  return "{\"variant\":" + std::to_string(index) + ",\"case\":\"" + name +
         "\",\"metallib\":" + std::to_string(variant.metallib) + ",\"pipeline\":\"" +
         std::string(variant.pipeline) + "\",\"gpu_ms\":" + std::to_string(milliseconds) +
         ",\"identical\":" + (comparison.isIdentical ? "true" : "false") +
         ",\"max_error\":" + std::to_string(comparison.maxError) + "}";
}

void run(const Options &options) {
  MetalBackend backend(options.metallib);
  std::unique_ptr<MetalBackend> comparison;
  if (!options.comparisonLibrary.empty())
    comparison = std::make_unique<MetalBackend>(options.comparisonLibrary);
  const VisionGemmTile gemmTile = Vision::gemmTile(backend.capabilities().appleGpuFamily);
  std::cerr << "device " << backend.capabilities().deviceName
            << ", one vision kernel per command, median GPU time of " << options.repeat
            << " commands (ms)\n";
  std::cout << "{\"device\":\"" << backend.capabilities().deviceName << "\",\"cases\":[";
  bool isFirst = true;
  for (const Kernel &kernel : options.kernels)
    for (uint32_t patches : options.patches) {
      const VisionKernelShapes shapes =
          Vision::kernelShapes(VisionLayout{}, gridFor(patches), gemmTile);
      const std::string_view production =
          kernel.gemm ? shapes.gemm(*kernel.gemm).pipeline : shapes.attention.pipeline;
      std::vector<Variant> variants{{&backend, 0, production}};
      if (comparison) variants.push_back({comparison.get(), 1, production});
      const auto alternate = options.alternates.find(production);
      if (alternate != options.alternates.end())
        variants.push_back({&backend, 0, alternate->second});

      std::vector<Workload> workloads;
      workloads.reserve(variants.size());
      for (const Variant &variant : variants)
        workloads.push_back(kernel.gemm ? gemmWorkload(*variant.backend,
                                                       shapes.gemm(*kernel.gemm),
                                                       variant.pipeline)
                                        : attentionWorkload(*variant.backend,
                                                            shapes.attention,
                                                            variant.pipeline));
      const std::vector<double> timings = measure(workloads, options.repeat);
      const std::string name =
          std::string(kernel.name) + "/patches=" + std::to_string(patches);
      for (size_t i = 0; i < variants.size(); ++i) {
        const Comparison result = compareOutputs(workloads[0], workloads[i]);
        std::cout << (isFirst ? "" : ",") << json(name, variants[i], i, timings[i], result);
        isFirst = false;
        std::cerr << name << " variant=" << i << " " << variants[i].pipeline
                  << " gpu=" << timings[i] << " ms max_error=" << result.maxError
                  << "\n";
        if (variants[i].pipeline == production)
          expectSame(result.isIdentical,
                     name + ": comparison metallib changed the output bits",
                     options.isDiffAllowed);
        else
          expectSame(result.maxError <= options.tolerance,
                     name + ": " + std::string(variants[i].pipeline) +
                         " exceeds the tolerance",
                     options.isDiffAllowed);
      }
    }
  std::cout << "]}\n";
}

} // namespace

int main(int argc, const char *argv[]) {
  if (argc < 2) {
    std::cerr << "usage: vision-profile METALLIB [--patches LIST] [--kernels LIST] "
                 "[--repeat N] [--compare-metallib PATH] "
                 "[--alternate KERNEL=PIPELINE]... [--tolerance X] [--allow-diff]\n";
    return 64;
  }
  try {
    run(parseOptions(argc, argv));
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "vision-profile: " << error.what() << '\n';
    return 70;
  }
}
