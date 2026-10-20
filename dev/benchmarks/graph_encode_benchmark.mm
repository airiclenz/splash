// Times the host side of one Qwen3.8-27B B1 decode cycle without model
// weights: building the cycle's CommandGraph, the decode arena views it takes,
// and submitCommandAsync up to commit. The graph has the production cycle's
// dispatch order, binding counts and parameter payloads (rope tables, DFlash
// draft and selection, target verify, policy, acceptance, GDN and draft
// context commits), but every dispatch runs one no-op kernel, so GPU time
// stays out of the measurement. Views come from a real DecodeArena sized for
// the 27B geometry; weights, KV pages, GDN states and draft rings are small
// stand-in allocations created once, as a loaded model's are.
//
// The timer stops when submitCommandAsync returns; the command's completion
// and the graph's destruction fall outside it. The median over --repeat
// graphs is reported after a few untimed warm-up cycles.
//
// usage: graph-encode-benchmark METALLIB [--repeat N] [--json]
// METALLIB is build/engine-tests/metal-backend.metallib, which holds the
// no-op kernel.
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/DraftAttention.h"
#include "metal/abi/Embedding.h"
#include "metal/abi/GDN.h"
#include "metal/abi/Linear.h"
#include "metal/abi/PagedAttention.h"
#include "metal/abi/RoPE.h"
#include "metal/abi/Sampling.h"
#include "model/Qwen3_8.hpp"
#include "model/RuntimeArenas.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace splash;
using metal::CommandGraph;
using metal::MetalBuffer;
using model::DecodeTensor;

constexpr std::string_view kPipeline = "graph_encode_benchmark_noop_dispatch";
constexpr uint32_t kLanes = model::kLaneCount;
// The benchmark times the single-request decode cycle.
constexpr uint32_t kWidth = 1;
constexpr uint32_t kWarmupCycles = 3;
constexpr uint32_t kDefaultRepeat = 25;
constexpr metal::DispatchSize kGroups{1, 1, 1};
// Every stand-in weight is one fixed-size view of its layer's allocation.
constexpr uint64_t kSectionBytes = 256;
constexpr uint32_t kSectionsPerFile = 32;

using LaneBuffers = std::array<MetalBuffer, kLanes>;

struct Projection final {
  MetalBuffer weights;
  MetalBuffer scales;
  MetalBuffer biases;
};

// Hands out views of one allocation, the way a loaded weight file hands out
// its sections.
class WeightFile final {
public:
  explicit WeightFile(metal::MetalBackend &backend)
      : backend_(backend),
        allocation_(backend.allocateBuffer(kSectionBytes * kSectionsPerFile,
                                           metal::BufferStorage::Private,
                                           "graph-encode-weights")) {}

  MetalBuffer section() {
    if (next_ == kSectionsPerFile)
      throw std::logic_error("stand-in weight file is full");
    return backend_.view(allocation_, kSectionBytes * next_++, kSectionBytes);
  }
  Projection projection() { return {section(), section(), section()}; }

private:
  metal::MetalBackend &backend_;
  MetalBuffer allocation_;
  uint32_t next_ = 0;
};

struct TargetLayer final {
  bool isAttention = false;
  MetalBuffer inputNorm;
  MetalBuffer postNorm;
  Projection mixerInput;
  Projection mixerOutput;
  Projection gate;
  Projection up;
  Projection down;
  // GDN mixers.
  MetalBuffer convolution;
  MetalBuffer decay;
  MetalBuffer timeBias;
  MetalBuffer mixerNorm;
  // Attention mixers.
  MetalBuffer queryNorm;
  MetalBuffer keyNorm;
};

struct DraftLayer final {
  MetalBuffer inputNorm;
  MetalBuffer postNorm;
  MetalBuffer attentionConvolution;
  MetalBuffer mlpConvolution;
  MetalBuffer queryNorm;
  MetalBuffer keyNorm;
  Projection attentionDynamic;
  Projection qkv;
  Projection output;
  Projection mlpDynamic;
  Projection gate;
  Projection up;
  Projection down;
};

// Everything a serving runtime holds across cycles: weights, KV pages, the
// slot's GDN state pair and draft rings, and the RoPE inverse frequencies.
struct Resources final {
  std::vector<TargetLayer> target;
  std::vector<DraftLayer> draft;
  Projection embedding;
  Projection head;
  Projection selector;
  Projection context;
  MetalBuffer finalNorm;
  MetalBuffer draftFinalNorm;
  MetalBuffer hiddenNorm;
  MetalBuffer predecessorCodebook;
  MetalBuffer successorCodebook;
  MetalBuffer targetFrequencies;
  MetalBuffer draftFrequencies;
  std::vector<kv::LayerStorage> kvLayers;
  MetalBuffer currentState;
  MetalBuffer nextState;
  std::vector<MetalBuffer> ringKeys;
  std::vector<MetalBuffer> ringValues;
};

model::RuntimeGeometry geometry27b() {
  model::RuntimeGeometry geometry;
  geometry.target = model::qwenTargetGeometry(model::Qwen3_8Weights{});
  geometry.target.kvLayout.format = kv::Format::Int8;
  geometry.draftState = geometry.draft.stateLayout();
  if (!geometry.target.valid() || !geometry.draftState.valid())
    throw std::logic_error("invalid synthetic 27B geometry");
  return geometry;
}

TargetLayer targetLayer(metal::MetalBackend &backend, bool isAttention) {
  WeightFile file(backend);
  TargetLayer layer;
  layer.isAttention = isAttention;
  layer.inputNorm = file.section();
  layer.postNorm = file.section();
  layer.mixerInput = file.projection();
  layer.mixerOutput = file.projection();
  layer.gate = file.projection();
  layer.up = file.projection();
  layer.down = file.projection();
  if (isAttention) {
    layer.queryNorm = file.section();
    layer.keyNorm = file.section();
    return layer;
  }
  layer.convolution = file.section();
  layer.decay = file.section();
  layer.timeBias = file.section();
  layer.mixerNorm = file.section();
  return layer;
}

DraftLayer draftLayer(metal::MetalBackend &backend) {
  WeightFile file(backend);
  DraftLayer layer;
  layer.inputNorm = file.section();
  layer.postNorm = file.section();
  layer.attentionConvolution = file.section();
  layer.mlpConvolution = file.section();
  layer.queryNorm = file.section();
  layer.keyNorm = file.section();
  layer.attentionDynamic = file.projection();
  layer.qkv = file.projection();
  layer.output = file.projection();
  layer.mlpDynamic = file.projection();
  layer.gate = file.projection();
  layer.up = file.projection();
  layer.down = file.projection();
  return layer;
}

void addModelWeights(metal::MetalBackend &backend,
                     const model::RuntimeGeometry &geometry,
                     Resources &resources) {
  const model::QwenTargetGeometry &target = geometry.target;
  for (uint32_t layer = 0; layer < target.layers; ++layer) {
    resources.target.push_back(
        targetLayer(backend, model::Qwen3_8Layout{}.isFullAttentionLayer(layer)));
  }
  for (uint32_t layer = 0; layer < geometry.draft.layers; ++layer)
    resources.draft.push_back(draftLayer(backend));
  WeightFile globals(backend);
  resources.embedding = globals.projection();
  resources.head = globals.projection();
  resources.selector = globals.projection();
  resources.context = globals.projection();
  resources.finalNorm = globals.section();
  resources.draftFinalNorm = globals.section();
  resources.hiddenNorm = globals.section();
  resources.predecessorCodebook = globals.section();
  resources.successorCodebook = globals.section();
  resources.targetFrequencies = globals.section();
  resources.draftFrequencies = globals.section();
}

void addSlotState(metal::MetalBackend &backend,
                  const model::RuntimeGeometry &geometry, Resources &resources) {
  const uint32_t attentionLayers = geometry.target.kvLayout.attentionLayers;
  const MetalBuffer pages = backend.allocateBuffer(
      kSectionBytes * 4 * attentionLayers, metal::BufferStorage::Private,
      "graph-encode-kv");
  for (uint32_t layer = 0; layer < attentionLayers; ++layer) {
    auto page = [&](uint32_t part) {
      return backend.view(pages, kSectionBytes * (4 * layer + part), kSectionBytes);
    };
    resources.kvLayers.push_back({page(0), page(1), page(2), page(3), kv::Format::Int8});
  }
  const MetalBuffer states = backend.allocateBuffer(
      kSectionBytes * 2, metal::BufferStorage::Private, "graph-encode-gdn-state");
  resources.currentState = backend.view(states, 0, kSectionBytes);
  resources.nextState = backend.view(states, kSectionBytes, kSectionBytes);
  const uint32_t draftLayers = geometry.draft.layers;
  const MetalBuffer rings = backend.allocateBuffer(
      kSectionBytes * 2 * draftLayers, metal::BufferStorage::Private,
      "graph-encode-draft-rings");
  for (uint32_t layer = 0; layer < draftLayers; ++layer) {
    resources.ringKeys.push_back(
        backend.view(rings, kSectionBytes * 2 * layer, kSectionBytes));
    resources.ringValues.push_back(
        backend.view(rings, kSectionBytes * (2 * layer + 1), kSectionBytes));
  }
}

Resources makeResources(metal::MetalBackend &backend,
                        const model::RuntimeGeometry &geometry) {
  Resources resources;
  addModelWeights(backend, geometry, resources);
  addSlotState(backend, geometry, resources);
  return resources;
}

LaneBuffers sameOnEveryLane(const MetalBuffer &buffer) {
  LaneBuffers lanes;
  lanes.fill(buffer);
  return lanes;
}

void appendLanes(metal::BufferList &bindings, const LaneBuffers &first,
                 const LaneBuffers &second) {
  bindings.insert(bindings.end(), first.begin(), first.end());
  bindings.insert(bindings.end(), second.begin(), second.end());
}

// Builds one decode cycle's graph the way Runtime::decodeAsync does for a
// single greedy, unconstrained request with speculation on.
class DecodeCycle final {
public:
  DecodeCycle(const model::RuntimeGeometry &geometry,
              const model::DecodeArena &arena, const Resources &resources)
      : geometry_(geometry), arena_(arena), resources_(resources),
        scratch_(arena.linearScratch()) {}

  // Encodes the whole cycle into graph; returns the arena views it looked up.
  uint32_t encode(CommandGraph &graph) {
    views_ = 0;
    addRopeTables(graph);
    addEmbedding(graph, DecodeTensor::DraftInputTokens, DecodeTensor::DraftHidden0);
    addDraft(graph);
    addSelection(graph);
    addVerifyInput(graph);
    addEmbedding(graph, DecodeTensor::InputTokens, DecodeTensor::Hidden0);
    addTarget(graph);
    addPolicy(graph);
    addAcceptance(graph);
    addGdnCommit(graph);
    addContextCommit(graph);
    return views_;
  }

private:
  MetalBuffer packed(DecodeTensor tensor) {
    ++views_;
    return arena_.packed(tensor, kWidth);
  }

  void addNorm(CommandGraph &graph, const MetalBuffer &input,
               const MetalBuffer &weight, const MetalBuffer &output) {
    graph.add(kPipeline,
              {input, weight, output, scratch_.input, scratch_.sums},
              geometry_.target.hiddenSize, kGroups);
  }

  // The simdgroup decode linear: an input-preparation dispatch unless a fused
  // producer already prepared the input, then one dispatch binding the
  // prepared input, projection, output, scratch and the epilogue's buffers.
  void addLinear(CommandGraph &graph, const MetalBuffer &input,
                 const Projection &projection, const MetalBuffer &output,
                 bool isInputPrepared,
                 std::initializer_list<MetalBuffer> epilogue = {}) {
    if (!isInputPrepared) {
      graph.add(kPipeline, {input, scratch_.input, scratch_.sums},
                geometry_.target.hiddenSize, kGroups);
    }
    metal::BufferList bindings{
        scratch_.input,  projection.weights, projection.scales,
        projection.biases, output,           scratch_.sums,
        scratch_.partials, scratch_.counters};
    bindings.insert(bindings.end(), epilogue);
    graph.add(kPipeline, std::move(bindings), Q4Params{}, kGroups);
  }

  void addRopeTables(CommandGraph &graph) {
    graph.add(kPipeline,
              {packed(DecodeTensor::Positions), packed(DecodeTensor::DraftPositions),
               resources_.targetFrequencies, resources_.draftFrequencies,
               packed(DecodeTensor::RopeCos), packed(DecodeTensor::RopeSin),
               packed(DecodeTensor::DraftRopeCos), packed(DecodeTensor::DraftRopeSin)},
              RopeTableParams{}, kGroups);
  }

  void addEmbedding(CommandGraph &graph, DecodeTensor tokens, DecodeTensor output) {
    const Projection &table = resources_.embedding;
    graph.add(kPipeline,
              {packed(tokens), table.weights, table.scales, table.biases,
               packed(output)},
              Q4EmbeddingParams{}, kGroups);
  }

  struct DraftBuffers final {
    std::array<MetalBuffer, 2> hidden;
    MetalBuffer normalized, dynamic, convolved, proposalQkv, attention;
    MetalBuffer projected, residual, intermediate, finalHidden, logits;
    MetalBuffer selectorHidden, queryKeys, queryValues, ropeCos, ropeSin;
  };

  DraftBuffers draftBuffers() {
    DraftBuffers buffers;
    buffers.hidden = {packed(DecodeTensor::DraftHidden0),
                      packed(DecodeTensor::DraftHidden1)};
    buffers.normalized = packed(DecodeTensor::DraftNormalized);
    buffers.dynamic = packed(DecodeTensor::DraftDynamic);
    buffers.convolved = packed(DecodeTensor::DraftConvolved);
    buffers.proposalQkv = packed(DecodeTensor::DraftProposalQkv);
    buffers.attention = packed(DecodeTensor::DraftAttention);
    buffers.projected = packed(DecodeTensor::DraftProjected);
    buffers.residual = packed(DecodeTensor::DraftResidual);
    buffers.intermediate = packed(DecodeTensor::DraftIntermediate);
    buffers.finalHidden = packed(DecodeTensor::DraftFinalHidden);
    buffers.logits = packed(DecodeTensor::Logits);
    buffers.selectorHidden = packed(DecodeTensor::SelectorHidden);
    buffers.queryKeys = packed(DecodeTensor::DraftQueryKeys);
    buffers.queryValues = packed(DecodeTensor::DraftQueryValues);
    buffers.ropeCos = packed(DecodeTensor::DraftRopeCos);
    buffers.ropeSin = packed(DecodeTensor::DraftRopeSin);
    return buffers;
  }

  void addConvolution(CommandGraph &graph, const MetalBuffer &input,
                      const DraftBuffers &buffers, const MetalBuffer &weights,
                      const MetalBuffer &residual, const MetalBuffer &output) {
    graph.add(kPipeline,
              {input, buffers.dynamic, weights, residual, output},
              DraftConvBatchParams{}, kGroups);
  }

  void addDraftAttention(CommandGraph &graph, const DraftBuffers &buffers,
                         const DraftLayer &layer, uint32_t index) {
    graph.add(kPipeline,
              {buffers.proposalQkv, buffers.attention, layer.queryNorm,
               layer.keyNorm, buffers.ropeCos, buffers.ropeSin,
               buffers.queryKeys, buffers.queryValues},
              DraftQkvBatchParams{}, kGroups);
    metal::BufferList bindings{buffers.attention};
    appendLanes(bindings, sameOnEveryLane(resources_.ringKeys[index]),
                sameOnEveryLane(resources_.ringValues[index]));
    bindings.push_back(buffers.queryKeys);
    bindings.push_back(buffers.queryValues);
    graph.add(kPipeline, std::move(bindings),
              DraftAttentionBatchParams{}, kGroups);
    graph.add(kPipeline, {buffers.attention},
              DraftAttentionBatchParams{}, kGroups);
    graph.add(kPipeline, {buffers.attention, buffers.proposalQkv},
              DraftQkvBatchParams{}, kGroups);
  }

  void addDraftLayer(CommandGraph &graph, const DraftBuffers &buffers,
                     uint32_t index) {
    const DraftLayer &layer = resources_.draft[index];
    const MetalBuffer &current = buffers.hidden[index & 1];
    const MetalBuffer &next = buffers.hidden[(index & 1) ^ 1];
    addNorm(graph, current, layer.inputNorm, buffers.normalized);
    addLinear(graph, buffers.normalized, layer.attentionDynamic, buffers.dynamic, true);
    addConvolution(graph, buffers.normalized, buffers, layer.attentionConvolution,
                   current, buffers.convolved);
    addLinear(graph, buffers.convolved, layer.qkv, buffers.proposalQkv, false);
    addDraftAttention(graph, buffers, layer, index);
    addLinear(graph, buffers.proposalQkv, layer.output, buffers.projected, false);
    addConvolution(graph, buffers.projected, buffers, layer.attentionConvolution,
                   current, buffers.residual);
    addNorm(graph, buffers.residual, layer.postNorm, buffers.normalized);
    addLinear(graph, buffers.normalized, layer.mlpDynamic, buffers.dynamic, true);
    addConvolution(graph, buffers.normalized, buffers, layer.mlpConvolution,
                   buffers.residual, buffers.convolved);
    addLinear(graph, buffers.convolved, layer.gate, buffers.intermediate, false,
              {layer.up.weights, layer.up.scales, layer.up.biases});
    addLinear(graph, buffers.intermediate, layer.down, buffers.projected, false);
    addConvolution(graph, buffers.projected, buffers, layer.mlpConvolution,
                   buffers.residual, next);
  }

  void addDraft(CommandGraph &graph) {
    const DraftBuffers buffers = draftBuffers();
    const uint32_t layers = static_cast<uint32_t>(resources_.draft.size());
    for (uint32_t layer = 0; layer < layers; ++layer)
      addDraftLayer(graph, buffers, layer);
    addNorm(graph, buffers.hidden[layers & 1], resources_.draftFinalNorm,
            buffers.finalHidden);
    addLinear(graph, buffers.finalHidden, resources_.head, buffers.logits, true);
    addLinear(graph, buffers.finalHidden, resources_.selector,
              buffers.selectorHidden, true);
  }

  void addSelection(CommandGraph &graph) {
    const MetalBuffer logits = packed(DecodeTensor::Logits);
    const MetalBuffer partialIds = packed(DecodeTensor::TopPartialIds);
    const MetalBuffer partialValues = packed(DecodeTensor::TopPartialValues);
    const MetalBuffer candidates = packed(DecodeTensor::Candidates);
    const MetalBuffer unary = packed(DecodeTensor::Unary);
    const MetalBuffer selectorHidden = packed(DecodeTensor::SelectorHidden);
    const MetalBuffer uniforms = packed(DecodeTensor::SamplingUniforms);
    const MetalBuffer proposed = packed(DecodeTensor::ProposedTokens);
    const MetalBuffer probabilities = packed(DecodeTensor::ProposalProbs);
    graph.add(kPipeline, {logits, partialIds, partialValues},
              geometry_.target.vocabularySize, kGroups);
    graph.add(kPipeline,
              {partialIds, partialValues, candidates, unary, selectorHidden,
               resources_.predecessorCodebook, resources_.successorCodebook},
              SelectorBatchParams{}, kGroups);
    graph.add(kPipeline,
              {candidates, unary, partialValues, uniforms, proposed, probabilities},
              SelectorBatchParams{}, kGroups);
  }

  void addVerifyInput(CommandGraph &graph) {
    graph.add(kPipeline,
              {packed(DecodeTensor::DraftInputTokens),
               packed(DecodeTensor::ProposedTokens), packed(DecodeTensor::InputTokens)},
              VerifyInputBatchParams{}, kGroups);
  }

  struct TargetBuffers final {
    std::array<MetalBuffer, 2> hidden;
    MetalBuffer normalized, recurrent, gdnHidden, gdnOutput, intermediate;
    MetalBuffer fullPacked, fullQueries, partials, statistics, fullAttention;
    MetalBuffer attentionHidden, attentionOutput, ropeCos, ropeSin, arrived;
    MetalBuffer generation, captured, finalHidden, logits;
    LaneBuffers pageTables, currentStates, nextStates;
    std::vector<MetalBuffer> gdnPacked, gdnMixed, gdnDecay, gdnBeta;
    std::vector<MetalBuffer> chunkKeys, chunkValues;
    std::span<const kv::LayerStorage> kvLayers;
  };

  void takeTargetActivations(TargetBuffers &buffers) {
    buffers.hidden = {packed(DecodeTensor::Hidden0), packed(DecodeTensor::Hidden1)};
    buffers.normalized = packed(DecodeTensor::Normalized);
    buffers.recurrent = packed(DecodeTensor::Recurrent);
    buffers.gdnHidden = packed(DecodeTensor::GdnHidden);
    buffers.gdnOutput = packed(DecodeTensor::GdnOutput);
    buffers.intermediate = packed(DecodeTensor::Intermediate);
    buffers.fullPacked = packed(DecodeTensor::FullPacked);
    buffers.fullQueries = packed(DecodeTensor::FullQueries);
    buffers.partials = packed(DecodeTensor::AttentionPartials);
    buffers.statistics = packed(DecodeTensor::AttentionStatistics);
    buffers.fullAttention = packed(DecodeTensor::FullAttention);
    buffers.attentionHidden = packed(DecodeTensor::AttentionHidden);
    buffers.attentionOutput = packed(DecodeTensor::AttentionOutput);
    buffers.ropeCos = packed(DecodeTensor::RopeCos);
    buffers.ropeSin = packed(DecodeTensor::RopeSin);
    buffers.arrived = packed(DecodeTensor::Arrived);
    buffers.generation = packed(DecodeTensor::Generation);
    buffers.captured = packed(DecodeTensor::CapturedTargetHidden);
    buffers.finalHidden = packed(DecodeTensor::FinalHidden);
    buffers.logits = packed(DecodeTensor::Logits);
    // A dense target takes its MoE tensors too; they are empty views.
    for (uint32_t tensor = static_cast<uint32_t>(DecodeTensor::MoeSelectedExperts);
         tensor <= static_cast<uint32_t>(DecodeTensor::MoeInputSums); ++tensor)
      static_cast<void>(packed(static_cast<DecodeTensor>(tensor)));
  }

  void takeTargetLayerSlices(TargetBuffers &buffers) {
    const uint32_t gdnLayers = geometry_.target.stateLayout.layers;
    const uint32_t attentionLayers = geometry_.target.kvLayout.attentionLayers;
    for (auto *slices : {&buffers.gdnPacked, &buffers.gdnMixed,
                         &buffers.gdnDecay, &buffers.gdnBeta})
      slices->resize(gdnLayers);
    for (uint32_t layer = 0; layer < gdnLayers; ++layer) {
      buffers.gdnPacked[layer] = slice(DecodeTensor::VerifyPackedBase, layer);
      buffers.gdnMixed[layer] = slice(DecodeTensor::VerifyMixedBase, layer);
      buffers.gdnDecay[layer] = slice(DecodeTensor::VerifyDecayBase, layer);
      buffers.gdnBeta[layer] = slice(DecodeTensor::VerifyBetaBase, layer);
    }
    buffers.chunkKeys.resize(attentionLayers);
    buffers.chunkValues.resize(attentionLayers);
    buffers.kvLayers = resources_.kvLayers;
    for (uint32_t layer = 0; layer < attentionLayers; ++layer) {
      views_ += 2;
      buffers.chunkKeys[layer] =
          arena_.attentionBatchSlice(DecodeTensor::ChunkKeysBase, layer, kWidth);
      buffers.chunkValues[layer] =
          arena_.attentionBatchSlice(DecodeTensor::ChunkValuesBase, layer, kWidth);
    }
  }

  MetalBuffer slice(DecodeTensor tensor, uint32_t layer) {
    ++views_;
    return arena_.gdnBatchSlice(tensor, layer, kWidth);
  }

  TargetBuffers targetBuffers() {
    TargetBuffers buffers;
    takeTargetActivations(buffers);
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      buffers.pageTables[lane] = arena_.get(0, DecodeTensor::PageTable);
      buffers.currentStates[lane] = resources_.currentState;
      buffers.nextStates[lane] = resources_.nextState;
    }
    takeTargetLayerSlices(buffers);
    return buffers;
  }

  void addGdnMixer(CommandGraph &graph, const TargetBuffers &buffers,
                   const TargetLayer &layer, const MetalBuffer &input,
                   uint32_t index) {
    addLinear(graph, buffers.normalized, layer.mixerInput,
              buffers.gdnPacked[index], true);
    metal::BufferList bindings{buffers.gdnPacked[index], layer.convolution};
    appendLanes(bindings, buffers.currentStates, buffers.nextStates);
    bindings.insert(bindings.end(),
                    {buffers.gdnMixed[index], layer.decay, layer.timeBias,
                     buffers.gdnDecay[index], buffers.gdnBeta[index],
                     buffers.recurrent, layer.mixerNorm, buffers.gdnHidden,
                     buffers.arrived, buffers.generation, scratch_.input,
                     scratch_.sums});
    graph.add(kPipeline, std::move(bindings),
              GDNDecodeBatchParams{}, kGroups);
    addLinear(graph, buffers.gdnHidden, layer.mixerOutput, buffers.gdnOutput,
              true, {input});
  }

  void addPagedVerify(CommandGraph &graph, const TargetBuffers &buffers,
                      uint32_t index) {
    const kv::LayerStorage &pages = buffers.kvLayers[index];
    const std::array<SplashChunkedPrefillParams, kLanes> stores{};
    const std::array<SplashQ8VerifyAttentionParams, kLanes> attention{};
    metal::BufferList store{buffers.chunkKeys[index], buffers.chunkValues[index],
                                   pages.keyData, pages.keyScales,
                                   pages.valueData, pages.valueScales};
    store.insert(store.end(), buffers.pageTables.begin(), buffers.pageTables.end());
    graph.add(kPipeline, std::move(store), stores, kGroups);
    metal::BufferList split{buffers.fullQueries, pages.keyData, pages.keyScales,
                                   pages.valueData, pages.valueScales,
                                   buffers.partials, buffers.statistics};
    split.insert(split.end(), buffers.pageTables.begin(), buffers.pageTables.end());
    graph.add(kPipeline, std::move(split), attention, kGroups);
    graph.add(kPipeline,
              {buffers.partials, buffers.statistics, buffers.fullAttention},
              attention, kGroups);
  }

  void addAttentionMixer(CommandGraph &graph, const TargetBuffers &buffers,
                         const TargetLayer &layer, const MetalBuffer &input,
                         uint32_t index) {
    addLinear(graph, buffers.normalized, layer.mixerInput, buffers.fullPacked, true);
    graph.add(kPipeline,
              {buffers.fullPacked, layer.queryNorm, layer.keyNorm, buffers.ropeCos,
               buffers.ropeSin, buffers.fullQueries, buffers.chunkKeys[index],
               buffers.chunkValues[index]},
              FullDecodeBatchParams{}, kGroups);
    addPagedVerify(graph, buffers, index);
    graph.add(kPipeline,
              {buffers.fullPacked, buffers.fullAttention, buffers.attentionHidden,
               scratch_.input, scratch_.sums},
              FullDecodeBatchParams{}, kGroups);
    addLinear(graph, buffers.attentionHidden, layer.mixerOutput,
              buffers.attentionOutput, true, {input});
  }

  void addTargetLayer(CommandGraph &graph, const TargetBuffers &buffers,
                      uint32_t index, uint32_t &gdnIndex, uint32_t &attentionIndex) {
    const TargetLayer &layer = resources_.target[index];
    const MetalBuffer &input = buffers.hidden[index & 1];
    const MetalBuffer &output = buffers.hidden[(index & 1) ^ 1];
    addNorm(graph, input, layer.inputNorm, buffers.normalized);
    if (layer.isAttention)
      addAttentionMixer(graph, buffers, layer, input, attentionIndex++);
    else
      addGdnMixer(graph, buffers, layer, input, gdnIndex++);
    const MetalBuffer &residual =
        layer.isAttention ? buffers.attentionOutput : buffers.gdnOutput;
    addNorm(graph, residual, layer.postNorm, buffers.normalized);
    addLinear(graph, buffers.normalized, layer.gate, buffers.intermediate, true,
              {layer.up.weights, layer.up.scales, layer.up.biases});
    addLinear(graph, buffers.intermediate, layer.down, output, false, {residual});
    const auto captures = geometry_.target.captureLayers();
    if (std::find(captures.begin(), captures.end(), index) != captures.end()) {
      graph.add(kPipeline, {output, buffers.captured},
                CaptureParams{}, kGroups);
    }
  }

  void addTarget(CommandGraph &graph) {
    const TargetBuffers buffers = targetBuffers();
    uint32_t gdnIndex = 0;
    uint32_t attentionIndex = 0;
    const uint32_t layers = geometry_.target.layers;
    for (uint32_t layer = 0; layer < layers; ++layer)
      addTargetLayer(graph, buffers, layer, gdnIndex, attentionIndex);
    addNorm(graph, buffers.hidden[layers & 1], resources_.finalNorm,
            buffers.finalHidden);
    addLinear(graph, buffers.finalHidden, resources_.head, buffers.logits, true);
  }

  // Greedy unconstrained lanes take the sharded argmax and its reduction.
  void addPolicy(CommandGraph &graph) {
    const MetalBuffer logits = packed(DecodeTensor::Logits);
    for (DecodeTensor unused :
         {DecodeTensor::TargetTopPartialIds, DecodeTensor::TargetTopPartialValues,
          DecodeTensor::TargetTopIds, DecodeTensor::TargetTopProbs,
          DecodeTensor::SamplingUniforms, DecodeTensor::ConstraintMasks})
      static_cast<void>(packed(unused));
    const MetalBuffer outputTokens = packed(DecodeTensor::OutputTokens);
    const MetalBuffer values = packed(DecodeTensor::ArgmaxValues);
    const MetalBuffer indices = packed(DecodeTensor::ArgmaxIndices);
    graph.add(kPipeline, {logits, values, indices},
              geometry_.target.vocabularySize, kGroups);
    graph.add(kPipeline, {values, indices, outputTokens}, kGroups);
  }

  void addAcceptance(CommandGraph &graph) {
    graph.add(kPipeline,
              {packed(DecodeTensor::ProposedTokens), packed(DecodeTensor::Candidates),
               packed(DecodeTensor::ProposalProbs), packed(DecodeTensor::TargetTopIds),
               packed(DecodeTensor::TargetTopProbs),
               packed(DecodeTensor::SamplingUniforms),
               packed(DecodeTensor::OutputTokens), packed(DecodeTensor::RetainedCount),
               packed(DecodeTensor::NextAnchor), packed(DecodeTensor::AcceptedCount)},
              AcceptBatchParams{}, kGroups);
  }

  void addGdnCommit(CommandGraph &graph) {
    views_ += 4;
    metal::BufferList bindings{
        arena_.gdnStorage(DecodeTensor::VerifyPackedBase),
        arena_.gdnStorage(DecodeTensor::VerifyMixedBase),
        arena_.gdnStorage(DecodeTensor::VerifyDecayBase),
        arena_.gdnStorage(DecodeTensor::VerifyBetaBase)};
    appendLanes(bindings, sameOnEveryLane(resources_.currentState),
                sameOnEveryLane(resources_.nextState));
    bindings.push_back(packed(DecodeTensor::RetainedCount));
    graph.add(kPipeline, std::move(bindings),
              GDNBatchCommitParams{}, kGroups);
  }

  void addContextCommit(CommandGraph &graph) {
    const MetalBuffer captured = packed(DecodeTensor::CapturedTargetHidden);
    const MetalBuffer projected = packed(DecodeTensor::ContextProjected);
    const MetalBuffer hidden = packed(DecodeTensor::ContextHidden);
    const MetalBuffer qkv = packed(DecodeTensor::ContextQkv);
    const MetalBuffer ropeCos = packed(DecodeTensor::DraftRopeCos);
    const MetalBuffer ropeSin = packed(DecodeTensor::DraftRopeSin);
    const MetalBuffer retained = packed(DecodeTensor::RetainedCount);
    addLinear(graph, captured, resources_.context, projected, false);
    addNorm(graph, projected, resources_.hiddenNorm, hidden);
    for (uint32_t layer = 0; layer < resources_.draft.size(); ++layer) {
      const DraftLayer &weights = resources_.draft[layer];
      addLinear(graph, hidden, weights.qkv, qkv, true);
      metal::BufferList bindings{qkv, weights.keyNorm, ropeCos, ropeSin};
      appendLanes(bindings, sameOnEveryLane(resources_.ringKeys[layer]),
                  sameOnEveryLane(resources_.ringValues[layer]));
      bindings.push_back(retained);
      graph.add(kPipeline, std::move(bindings),
                DraftContextBatchParams{}, kGroups);
    }
  }

  const model::RuntimeGeometry &geometry_;
  const model::DecodeArena &arena_;
  const Resources &resources_;
  const ops::LinearScratch scratch_;
  uint32_t views_ = 0;
};

struct Cycle final {
  double hostSeconds = 0.0;
  size_t dispatches = 0;
  uint32_t views = 0;
};

// The timer covers graph build, arena views, validation and encoding up to
// commit; the GPU completion wait and graph teardown fall outside it. The graph
// reserves expectedDispatches, as Runtime reserves the last decode graph's size.
Cycle runCycle(metal::MetalBackend &backend, DecodeCycle &cycle,
               size_t expectedDispatches) {
  const auto started = std::chrono::steady_clock::now();
  CommandGraph graph{expectedDispatches};
  const uint32_t views = cycle.encode(graph);
  metal::CommandTicket ticket = backend.submitCommandAsync(graph.dispatches());
  const auto committed = std::chrono::steady_clock::now();
  static_cast<void>(ticket.wait());
  return {std::chrono::duration<double>(committed - started).count(),
          graph.dispatches().size(), views};
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  return values.size() % 2 ? values[middle]
                           : (values[middle - 1] + values[middle]) / 2.0;
}

struct Options final {
  std::string metallib;
  uint32_t repeat = kDefaultRepeat;
  bool isJson = false;
};

uint32_t parseRepeat(std::string_view text) {
  uint32_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !value)
    throw std::invalid_argument("--repeat requires a positive integer");
  return value;
}

Options parseOptions(int argc, const char *argv[]) {
  Options options;
  options.metallib = argv[1];
  for (int index = 2; index < argc; ++index) {
    const std::string_view option(argv[index]);
    if (option == "--json") {
      options.isJson = true;
    } else if (option == "--repeat") {
      if (++index >= argc)
        throw std::invalid_argument("--repeat requires a value");
      options.repeat = parseRepeat(argv[index]);
    } else {
      throw std::invalid_argument("unknown option " + std::string(option));
    }
  }
  return options;
}

void report(const Options &options, const std::string &device,
            const std::vector<Cycle> &cycles) {
  std::vector<double> milliseconds;
  for (const Cycle &cycle : cycles)
    milliseconds.push_back(cycle.hostSeconds * 1000.0);
  const double middle = median(milliseconds);
  const auto [fastest, slowest] =
      std::minmax_element(milliseconds.begin(), milliseconds.end());
  const size_t dispatches = cycles.front().dispatches;
  const uint32_t views = cycles.front().views;
  std::cout << std::fixed << std::setprecision(4);
  if (!options.isJson) {
    std::cout << "device " << device << ", qwen3.8-27b B1 decode cycle, "
              << dispatches << " dispatches, " << views << " arena views\n"
              << "host encode median " << middle << " ms (min " << *fastest
              << ", max " << *slowest << ") over " << cycles.size()
              << " graphs, " << middle * 1000.0 / static_cast<double>(dispatches)
              << " us per dispatch\n";
    return;
  }
  std::cout << "{\"device\":\"" << device << "\",\"shape\":\"qwen3.8-27b\","
            << "\"lanes\":" << kWidth << ",\"dispatches\":" << dispatches
            << ",\"views\":" << views << ",\"repeat\":" << cycles.size()
            << ",\"median_ms\":" << middle << ",\"min_ms\":" << *fastest
            << ",\"max_ms\":" << *slowest << ",\"samples_ms\":[";
  for (size_t index = 0; index < milliseconds.size(); ++index)
    std::cout << (index ? "," : "") << milliseconds[index];
  std::cout << "]}\n";
}

} // namespace

int main(int argc, const char *argv[]) {
  try {
    if (argc < 2) {
      std::cerr << "usage: graph-encode-benchmark METALLIB [--repeat N] [--json]\n";
      return 64;
    }
    const Options options = parseOptions(argc, argv);
    metal::MetalBackend backend(options.metallib);
    const model::RuntimeGeometry geometry = geometry27b();
    const ops::ExecutionPlans operators(backend.capabilities());
    const model::DecodeArena arena(backend, geometry, operators);
    const Resources resources = makeResources(backend, geometry);
    DecodeCycle cycle(geometry, arena, resources);
    size_t dispatches = 0;
    for (uint32_t warmup = 0; warmup < kWarmupCycles; ++warmup)
      dispatches = runCycle(backend, cycle, dispatches).dispatches;
    std::vector<Cycle> cycles;
    cycles.reserve(options.repeat);
    for (uint32_t repeat = 0; repeat < options.repeat; ++repeat)
      cycles.push_back(runCycle(backend, cycle, dispatches));
    report(options, backend.capabilities().deviceName, cycles);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "graph-encode-benchmark: " << error.what() << '\n';
    return 1;
  }
}
