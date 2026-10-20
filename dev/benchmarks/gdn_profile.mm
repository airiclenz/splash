// Times Splash's GDN kernels on synthetic weights and state, with no model
// loaded: the verify decode (verify_gdn_fused and its Q4-preparing variant,
// eight rows over two layers), the commit (verify_gdn_commit) at retained
// counts 1, 4 and 8, and the prefill chunk (prefill_gdn_prepare, _scan and
// _gate), for the 48- and 32-value-head layouts. Each case reports the median
// GPU time of the whole graph and, with dispatch profiling, of each pipeline.
// These are kernel timings, not a correctness oracle (gdn-decode and gdn-metal
// are); the fixtures are copied from those tests.
//
// Every case hashes its output bytes. --compare-metallib alternates a second
// metallib ABBA on identical inputs and fails when its output bytes differ;
// --hash-out saves the per-case hashes and --hash-in checks them, so a saved
// pre-change binary can be compared with a candidate whose host dispatch
// differs. --allow-diff turns either mismatch into a warning.
//
// --first-row-state times, under each retained=1 commit case's name, the
// decode that stores row 0's state itself (GdnStoredRow::First) and needs no
// commit, hashing the same cells; its time compares with a run's decode plus
// retained=1 commit, and --hash-in checks it against that run's commit hash.
//
// usage: gdn-profile METALLIB [--lanes 1,4] [--repeat N]
//                    [--compare-metallib PATH] [--allow-diff]
//                    [--hash-out FILE] [--hash-in FILE] [--first-row-state]
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "ops/GDN.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;

constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kMaxLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kPrefillTokens = SPLASH_PREFILL_TOKEN_BUDGET;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kLayers = 2;
constexpr std::array kShapes{GdnShape{16, 48, 128, 10240, 16640},
                             GdnShape{16, 32, 128, 8192, 12544}};
constexpr std::array<uint32_t, 3> kRetainedCounts{1, 4, kRows};
constexpr double kWarmupSeconds = 0.1;
constexpr uint64_t kHashBasis = 0xCBF29CE484222325ULL;
constexpr uint64_t kHashPrime = 0x100000001B3ULL;

uint16_t toBfloat(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7FFFU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(bits >> 16);
}

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  float unit() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>((state_ >> 40) & 0xFFFFFF) / 8388608.0F - 1.0F;
  }
  float gauss() {
    float sum = 0.0F;
    for (int i = 0; i < 4; ++i)
      sum += unit();
    return sum * 0.8660254F;
  }

private:
  uint64_t state_;
};

uint64_t align16k(uint64_t bytes) { return (bytes + 16383) & ~uint64_t{16383}; }

MetalBuffer shared(MetalBackend &backend, uint64_t bytes, const char *label) {
  MetalBuffer buffer = backend.allocateBuffer(bytes, BufferStorage::Shared, label);
  std::memset(buffer.contents(), 0, buffer.sizeBytes());
  return buffer;
}

void fill(MetalBuffer &buffer, Random &random, float scale) {
  auto *values = static_cast<uint16_t *>(buffer.contents());
  for (uint64_t index = 0; index < buffer.sizeBytes() / 2; ++index)
    values[index] = toBfloat(random.unit() * scale);
}

// The production state cell layout: every layer's conv rows, then every
// layer's recurrent state, both padded to 16 KiB.
struct Cell final {
  uint64_t convLayerBytes = 0;
  uint64_t recurrentLayerBytes = 0;
  uint64_t convBytes = 0;
  uint64_t bytes = 0;

  explicit Cell(const GdnShape &shape)
      : convLayerBytes(align16k(uint64_t{3} * shape.convolutionDimension * 2)),
        recurrentLayerBytes(align16k(uint64_t{shape.valueHeads} * kHeadDim *
                                     kHeadDim * 4)),
        convBytes(kLayers * convLayerBytes),
        bytes(convBytes + kLayers * recurrentLayerBytes) {}

  GdnStateStrides strides() const {
    return {convLayerBytes, recurrentLayerBytes, convBytes};
  }
};

// The verify decode and commit inputs of gdn-decode's fixture: a two-layer
// state cell per lane slot and per-layer views of the decoded q/k/v and gates.
struct DecodeFixture final {
  const GdnShape &shape;
  Cell cell;
  MetalBuffer packed, convWeights, mixed, decayWeights, timeBias, decay, beta,
      recurrent, mixerNorm, hidden, arrived, generation, retained;
  std::array<MetalBuffer, kMaxLanes> current, next;
  std::vector<MetalBuffer> packedLayer, mixedLayer, decayLayer, betaLayer;

  DecodeFixture(MetalBackend &backend, const GdnShape &geometry, uint32_t lanes)
      : shape(geometry), cell(geometry) {
    Random random(0x6D4E1234ULL + lanes);
    const uint64_t packedStride = uint64_t{kRows} * shape.packedWidth;
    const uint64_t mixedStride = uint64_t{kRows} * shape.convolutionDimension;
    const uint64_t gateStride = uint64_t{kRows} * shape.valueHeads;
    packed = shared(backend, kLayers * kMaxLanes * packedStride * 2, "gdn packed");
    fill(packed, random, 1.0F);
    convWeights = shared(backend, uint64_t{shape.convolutionDimension} * 4 * 2,
                         "gdn conv weights");
    fill(convWeights, random, 0.5F);
    mixed = shared(backend, kLayers * kMaxLanes * mixedStride * 2, "gdn mixed");
    decayWeights = shared(backend, uint64_t{shape.valueHeads} * 4, "gdn decay weights");
    for (uint32_t head = 0; head < shape.valueHeads; ++head)
      static_cast<float *>(decayWeights.contents())[head] = -std::exp(random.unit() * 1.5F);
    timeBias = shared(backend, uint64_t{shape.valueHeads} * 2, "gdn time bias");
    fill(timeBias, random, 0.5F);
    decay = shared(backend, kLayers * kMaxLanes * gateStride * 4, "gdn decay");
    beta = shared(backend, kLayers * kMaxLanes * gateStride * 2, "gdn beta");
    const uint64_t rowBytes = uint64_t{kMaxLanes} * kRows * shape.valueHeads * kHeadDim * 2;
    recurrent = shared(backend, rowBytes, "gdn recurrent rows");
    hidden = shared(backend, rowBytes, "gdn hidden");
    mixerNorm = shared(backend, uint64_t{kHeadDim} * 2, "gdn mixer norm");
    fill(mixerNorm, random, 1.0F);
    arrived = shared(backend, kMaxLanes * 4, "gdn arrived");
    generation = shared(backend, kMaxLanes * 4, "gdn generation");
    retained = shared(backend, kMaxLanes * 4, "gdn retained");
    for (uint32_t lane = 0; lane < kMaxLanes; ++lane) {
      current[lane] = shared(backend, cell.bytes, "gdn current");
      next[lane] = shared(backend, cell.bytes, "gdn next");
      auto *bytes = static_cast<uint8_t *>(current[lane].contents());
      auto *conv = reinterpret_cast<uint16_t *>(bytes);
      for (uint64_t index = 0; index < cell.convBytes / 2; ++index)
        conv[index] = toBfloat(random.unit());
      auto *state = reinterpret_cast<float *>(bytes + cell.convBytes);
      for (uint64_t index = 0; index < (cell.bytes - cell.convBytes) / 4; ++index)
        state[index] = random.unit() * 0.5F;
    }
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      packedLayer.push_back(backend.view(packed, layer * kMaxLanes * packedStride * 2,
                                         kMaxLanes * packedStride * 2));
      mixedLayer.push_back(backend.view(mixed, layer * kMaxLanes * mixedStride * 2,
                                        kMaxLanes * mixedStride * 2));
      decayLayer.push_back(backend.view(decay, layer * kMaxLanes * gateStride * 4,
                                        kMaxLanes * gateStride * 4));
      betaLayer.push_back(backend.view(beta, layer * kMaxLanes * gateStride * 2,
                                       kMaxLanes * gateStride * 2));
    }
  }

  GdnDecodeBuffers decodeBuffers(uint32_t layer) const {
    return {packedLayer[layer], convWeights,       current,
            next,               mixedLayer[layer], decayWeights,
            timeBias,           decayLayer[layer], betaLayer[layer],
            recurrent,          mixerNorm,         hidden,
            arrived,            generation};
  }
  GdnCommitBuffers commitBuffers() const {
    return {packed, mixed, decay, beta, current, next, retained};
  }

  // Every layer's decode, as the model graph encodes it per verify step.
  CommandGraph decodeGraph(uint32_t lanes, const LinearScratch &scratch,
                           GdnStoredRow storedRow = GdnStoredRow::Last) const {
    CommandGraph graph;
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
      GdnDecodeBuffers buffers = decodeBuffers(layer);
      buffers.linearScratch = scratch;
      GDN::addDecode(graph, buffers, shape, lanes, layer, cell.strides(), storedRow);
    }
    return graph;
  }

  // The decode's deterministic outputs; the arrived/generation counters
  // advance with every submit, so they are left out.
  std::vector<MetalBuffer> decodeOutputs(uint32_t lanes) const {
    std::vector<MetalBuffer> outputs(next.begin(), next.begin() + lanes);
    outputs.insert(outputs.end(), {mixed, decay, beta, recurrent, hidden});
    return outputs;
  }
};

// The prefill chunk inputs of gdn-metal's randomPrefill: gaussian packed rows,
// carried convolution rows and state, and mostly gentle decays.
GdnPrefillBuffers randomPrefill(MetalBackend &backend, const GdnShape &shape,
                                uint32_t tokens) {
  const uint32_t valueHeads = shape.valueHeads;
  const uint32_t convDim = shape.convolutionDimension;
  const uint64_t keyWidth = uint64_t{shape.keyHeads} * kHeadDim;
  const uint64_t valueWidth = uint64_t{valueHeads} * kHeadDim;
  const uint64_t stateElements = valueWidth * kHeadDim;
  GdnPrefillBuffers buffers{
      shared(backend, uint64_t{tokens} * shape.packedWidth * 2, "packed"),
      shared(backend, uint64_t{convDim} * 4 * 2, "conv weights"),
      shared(backend, uint64_t{3} * convDim * 2, "conv in"),
      shared(backend, uint64_t{3} * convDim * 2, "conv out"),
      shared(backend, tokens * keyWidth * 2, "queries"),
      shared(backend, tokens * keyWidth * 2, "keys"),
      shared(backend, tokens * valueWidth * 2, "values"),
      shared(backend, valueHeads * 4, "a scale"),
      shared(backend, valueHeads * 2, "dt bias"),
      shared(backend, uint64_t{tokens} * valueHeads * 4, "decay"),
      shared(backend, uint64_t{tokens} * valueHeads * 2, "beta"),
      shared(backend, stateElements * 4, "state in"),
      shared(backend, stateElements * 4, "state out"),
      shared(backend, tokens * valueWidth * 2, "recurrent rows"),
      shared(backend, kHeadDim * 2, "mixer norm"),
      shared(backend, tokens * valueWidth * 2, "hidden")};

  Random random(0x9E3779B97F4A7C15ULL ^ (uint64_t{valueHeads} << 32) ^ tokens);
  auto gaussian = [&](MetalBuffer &buffer, double scale) {
    auto *values = static_cast<uint16_t *>(buffer.contents());
    for (uint64_t i = 0; i < buffer.sizeBytes() / 2; ++i)
      values[i] = toBfloat(static_cast<float>(scale * random.gauss()));
  };
  gaussian(buffers.packed, 1.0);
  gaussian(buffers.convolutionWeights, 0.3);
  gaussian(buffers.convolutionIn, 1.0);
  auto *aScale = static_cast<float *>(buffers.decayWeights.contents());
  auto *dtBias = static_cast<uint16_t *>(buffers.timeBias.contents());
  for (uint32_t head = 0; head < valueHeads; ++head) {
    // One head in eleven at the strongest a_scale, whose decay underflows.
    const float unit = 0.5F * (random.unit() + 1.0F);
    aScale[head] = head % 11 == 7 ? -105.0F : -(0.05F + 8.0F * unit * unit);
    dtBias[head] = toBfloat(random.gauss());
  }
  auto *stateIn = static_cast<float *>(buffers.recurrentIn.contents());
  for (uint64_t i = 0; i < stateElements; ++i)
    stateIn[i] = 0.05F * random.gauss();
  auto *mixerNorm = static_cast<uint16_t *>(buffers.mixerNorm.contents());
  for (uint32_t i = 0; i < kHeadDim; ++i)
    mixerNorm[i] = toBfloat(static_cast<float>(1.0 + 0.2 * random.gauss()));
  return buffers;
}

// One timed graph and the buffers whose bytes it must reproduce. The graph's
// bindings keep every input alive; re-submitting it rewrites the same outputs.
struct Workload final {
  CommandGraph graph;
  std::vector<MetalBuffer> outputs;
};
using WorkloadFactory = std::function<Workload(MetalBackend &)>;

Workload decodeWorkload(MetalBackend &backend, const GdnShape &shape, uint32_t lanes,
                        bool prepareQ4) {
  const DecodeFixture fixture(backend, shape, lanes);
  const uint64_t width = uint64_t{shape.valueHeads} * shape.headDimension;
  LinearScratch scratch;
  if (prepareQ4)
    scratch = {shared(backend, width * 16 * lanes, "gdn q4 table"),
               shared(backend, width / 2 * lanes, "gdn q4 sums"), {}, {}};
  Workload workload{fixture.decodeGraph(lanes, scratch), fixture.decodeOutputs(lanes)};
  if (prepareQ4)
    workload.outputs.insert(workload.outputs.end(), {scratch.input, scratch.sums});
  return workload;
}

// The commit replays `retained` rows from the incoming cell over one decode's
// q/k/v and gates; eight leave the decoded cell as it is.
Workload commitWorkload(MetalBackend &backend, const GdnShape &shape, uint32_t lanes,
                        uint32_t retained) {
  const DecodeFixture fixture(backend, shape, lanes);
  static_cast<void>(backend.submitCommand(fixture.decodeGraph(lanes, {}).dispatches()));
  std::fill_n(static_cast<uint32_t *>(fixture.retained.contents()), kMaxLanes, retained);
  Workload workload;
  GDN::addCommit(workload.graph, fixture.commitBuffers(), shape, kLayers, lanes,
                 fixture.cell.strides());
  workload.outputs.assign(fixture.next.begin(), fixture.next.begin() + lanes);
  return workload;
}

// The speculation-off decode, which stores the cell a retained=1 commit
// writes and so replaces the decode and that commit.
Workload firstRowWorkload(MetalBackend &backend, const GdnShape &shape, uint32_t lanes) {
  const DecodeFixture fixture(backend, shape, lanes);
  Workload workload{fixture.decodeGraph(lanes, {}, GdnStoredRow::First), {}};
  workload.outputs.assign(fixture.next.begin(), fixture.next.begin() + lanes);
  return workload;
}

Workload prefillWorkload(MetalBackend &backend, const GdnShape &shape) {
  const GdnPrefillBuffers buffers = randomPrefill(backend, shape, kPrefillTokens);
  Workload workload;
  GDN::addPrefill(workload.graph, buffers, shape, kPrefillTokens);
  workload.outputs = {buffers.convolutionOut, buffers.queries,      buffers.keys,
                      buffers.values,         buffers.decay,        buffers.beta,
                      buffers.recurrentOut,   buffers.recurrentRows, buffers.hidden};
  return workload;
}

// FNV-1a over 64-bit words, each buffer's size mixed in first.
uint64_t hashOutputs(const std::vector<MetalBuffer> &outputs) {
  uint64_t hash = kHashBasis;
  auto mix = [&](uint64_t word) { hash = (hash ^ word) * kHashPrime; };
  for (const MetalBuffer &buffer : outputs) {
    const auto *bytes = static_cast<const uint8_t *>(buffer.contents());
    const uint64_t size = buffer.sizeBytes();
    mix(size);
    uint64_t offset = 0;
    for (; offset + sizeof(uint64_t) <= size; offset += sizeof(uint64_t)) {
      uint64_t word;
      std::memcpy(&word, bytes + offset, sizeof(word));
      mix(word);
    }
    for (; offset < size; ++offset)
      mix(bytes[offset]);
  }
  return hash;
}

std::string hex(uint64_t value) {
  char text[17];
  std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(value));
  return text;
}

struct Result final {
  double fusedMilliseconds = 0.0;
  std::map<std::string, double> pipelineMilliseconds;
  uint64_t hash = 0;
};

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

// Warms every variant, hashes its outputs, then times the fused graph and
// each dispatch alone, rotating the order every round (ABBA for two).
std::vector<Result> measure(std::span<MetalBackend *const> backends,
                            const WorkloadFactory &factory, uint32_t repeat) {
  std::vector<Workload> workloads;
  workloads.reserve(backends.size());
  for (MetalBackend *backend : backends)
    workloads.push_back(factory(*backend));
  auto submit = [&](size_t variant) {
    return backends[variant]->submitCommand(workloads[variant].graph.dispatches()).gpuSeconds;
  };
  std::vector<Result> results(backends.size());
  double warmup = 0.0;
  while (warmup < kWarmupSeconds)
    for (size_t i = 0; i < backends.size(); ++i)
      warmup += submit(i);
  for (size_t i = 0; i < backends.size(); ++i)
    results[i].hash = hashOutputs(workloads[i].outputs);

  std::vector<std::vector<double>> fused(backends.size());
  for (uint32_t round = 0; round < repeat; ++round)
    for (size_t offset = 0; offset < backends.size(); ++offset) {
      const size_t i = (round + offset) % backends.size();
      fused[i].push_back(submit(i) * 1000.0);
    }

  std::vector<std::map<std::string, std::vector<double>>> perPipeline(backends.size());
  for (MetalBackend *backend : backends)
    backend->setDispatchProfiling(true);
  // Round 0 is a profiled warm-up and is not recorded.
  for (uint64_t round = 0; round <= repeat; ++round)
    for (size_t offset = 0; offset < backends.size(); ++offset) {
      const size_t i = (round + offset) % backends.size();
      static_cast<void>(submit(i));
      std::map<std::string, double> run;
      for (const auto &timing : backends[i]->takeDispatchProfile())
        run[timing.pipelineName] += timing.gpuSeconds * 1000.0;
      if (round)
        for (const auto &[name, milliseconds] : run)
          perPipeline[i][name].push_back(milliseconds);
    }
  for (size_t i = 0; i < backends.size(); ++i) {
    backends[i]->setDispatchProfiling(false);
    results[i].fusedMilliseconds = median(fused[i]);
    for (auto &[name, samples] : perPipeline[i])
      results[i].pipelineMilliseconds[name] = median(samples);
  }
  return results;
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

std::vector<uint32_t> parseList(const std::string &text, uint32_t minimum,
                                uint32_t maximum, std::string_view option) {
  std::vector<uint32_t> values;
  size_t start = 0;
  while (start <= text.size()) {
    const size_t comma = text.find(',', start);
    const size_t length = comma == std::string::npos ? std::string::npos : comma - start;
    values.push_back(parseCount(text.substr(start, length), minimum, maximum, option));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return values;
}

struct Options final {
  std::string metallib;
  std::string comparisonLibrary;
  std::string hashOut;
  std::string hashIn;
  std::vector<uint32_t> lanes{1, kMaxLanes};
  uint32_t repeat = 5;
  bool isDiffAllowed = false;
  bool isFirstRowState = false;
};

Options parseOptions(int argc, const char *argv[]) {
  Options options;
  options.metallib = argv[1];
  for (int index = 2; index < argc; ++index) {
    const std::string option(argv[index]);
    if (option == "--allow-diff") {
      options.isDiffAllowed = true;
      continue;
    }
    if (option == "--first-row-state") {
      options.isFirstRowState = true;
      continue;
    }
    if (index + 1 >= argc)
      throw std::invalid_argument(option + " requires a value");
    const std::string value(argv[++index]);
    if (option == "--lanes") options.lanes = parseList(value, 1, kMaxLanes, option);
    else if (option == "--repeat")
      options.repeat = parseCount(value, 1, std::numeric_limits<uint32_t>::max(), option);
    else if (option == "--compare-metallib") options.comparisonLibrary = value;
    else if (option == "--hash-out") options.hashOut = value;
    else if (option == "--hash-in") options.hashIn = value;
    else throw std::invalid_argument("unknown option " + option);
  }
  return options;
}

// One "case hash" pair per line, as --hash-out writes them.
std::map<std::string, uint64_t> readHashes(const std::string &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("cannot read --hash-in file " + path);
  std::map<std::string, uint64_t> hashes;
  std::string name, value;
  while (file >> name >> value)
    hashes[name] = std::stoull(value, nullptr, 16);
  return hashes;
}

void writeHashes(const std::string &path, const std::vector<std::pair<std::string, uint64_t>> &hashes) {
  std::ofstream file(path);
  for (const auto &[name, hash] : hashes)
    file << name << ' ' << hex(hash) << '\n';
  if (!file)
    throw std::runtime_error("cannot write --hash-out file " + path);
}

// Fails on a mismatch, or warns with --allow-diff.
void expectSame(bool isSame, const std::string &message, bool isDiffAllowed) {
  if (isSame) return;
  if (!isDiffAllowed) throw std::runtime_error(message);
  std::cerr << "warning: " << message << '\n';
}

std::string json(const std::string &name, const Result &result, size_t variant) {
  std::string out = "{\"variant\":" + std::to_string(variant) + ",\"case\":\"" + name +
                    "\",\"fused_ms\":" + std::to_string(result.fusedMilliseconds) +
                    ",\"hash\":\"" + hex(result.hash) + "\",\"pipelines\":{";
  bool isFirst = true;
  for (const auto &[pipeline, milliseconds] : result.pipelineMilliseconds) {
    out += (isFirst ? "\"" : ",\"") + pipeline + "\":" + std::to_string(milliseconds);
    isFirst = false;
  }
  return out + "}}";
}

struct Case final {
  std::string name;
  WorkloadFactory factory;
};

std::vector<Case> cases(const std::vector<uint32_t> &laneCounts, bool isFirstRowState) {
  std::vector<Case> result;
  for (const GdnShape &shape : kShapes) {
    const std::string heads = "vh" + std::to_string(shape.valueHeads);
    for (uint32_t lanes : laneCounts) {
      const std::string lane = "/lanes=" + std::to_string(lanes);
      for (bool prepareQ4 : {false, true})
        result.push_back({heads + (prepareQ4 ? "/decode-q4" : "/decode") + lane,
                          [&shape, lanes, prepareQ4](MetalBackend &backend) {
                            return decodeWorkload(backend, shape, lanes, prepareQ4);
                          }});
      for (uint32_t retained : kRetainedCounts)
        result.push_back({heads + "/commit" + lane + "/retained=" + std::to_string(retained),
                          [&shape, lanes, retained, isFirstRowState](MetalBackend &backend) {
                            return isFirstRowState && retained == 1
                                       ? firstRowWorkload(backend, shape, lanes)
                                       : commitWorkload(backend, shape, lanes, retained);
                          }});
    }
    result.push_back({heads + "/prefill/tokens=" + std::to_string(kPrefillTokens),
                      [&shape](MetalBackend &backend) { return prefillWorkload(backend, shape); }});
  }
  return result;
}

void run(const Options &options) {
  MetalBackend backend(options.metallib);
  std::unique_ptr<MetalBackend> comparison;
  std::vector<MetalBackend *> backends{&backend};
  if (!options.comparisonLibrary.empty()) {
    comparison = std::make_unique<MetalBackend>(options.comparisonLibrary);
    backends.push_back(comparison.get());
  }
  const std::map<std::string, uint64_t> expected =
      options.hashIn.empty() ? std::map<std::string, uint64_t>{} : readHashes(options.hashIn);
  std::cerr << "device " << backend.capabilities().deviceName << ", " << kLayers
            << " GDN layers, median of " << options.repeat << " graphs (ms)"
            << (options.isFirstRowState ? ", retained=1 commits as first-row decodes" : "")
            << "\n";
  std::cout << "{\"device\":\"" << backend.capabilities().deviceName << "\",\"cases\":[";
  std::vector<std::pair<std::string, uint64_t>> hashes;
  bool isFirst = true;
  for (const Case &item : cases(options.lanes, options.isFirstRowState)) {
    const std::vector<Result> results = measure(backends, item.factory, options.repeat);
    for (size_t i = 0; i < results.size(); ++i) {
      std::cout << (isFirst ? "" : ",") << json(item.name, results[i], i);
      isFirst = false;
      std::cerr << item.name << " variant=" << i << " fused=" << results[i].fusedMilliseconds
                << " ms hash=" << hex(results[i].hash) << "\n";
      expectSame(results[i].hash == results[0].hash,
                 item.name + ": comparison metallib changed the output bytes",
                 options.isDiffAllowed);
    }
    if (!options.hashIn.empty()) {
      const auto found = expected.find(item.name);
      expectSame(found != expected.end() && found->second == results[0].hash,
                 item.name + ": output hash differs from " + options.hashIn,
                 options.isDiffAllowed);
    }
    hashes.emplace_back(item.name, results[0].hash);
  }
  std::cout << "]}\n";
  if (!options.hashOut.empty())
    writeHashes(options.hashOut, hashes);
}

} // namespace

int main(int argc, const char *argv[]) {
  if (argc < 2) {
    std::cerr << "usage: gdn-profile METALLIB [--lanes LIST] [--repeat N] "
                 "[--compare-metallib PATH] [--allow-diff] [--hash-out FILE] "
                 "[--hash-in FILE] [--first-row-state]\n";
    return 64;
  }
  try {
    run(parseOptions(argc, argv));
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "gdn-profile: " << error.what() << '\n';
    return 70;
  }
}
