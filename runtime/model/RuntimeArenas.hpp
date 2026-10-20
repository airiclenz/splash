#pragma once

// Shared Qwen prefill/decode scratch layouts and allocation owners.

#include "model/DFlashDraft.hpp"
#include "model/ModelFactory.hpp"
#include "model/QwenTarget.hpp"

#include "metal/MetalBackend.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/PagedKv.hpp"

#include <algorithm>
#include <cmath>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace splash::model {

inline constexpr uint32_t kLaneCount = ExecutionLimits::maximumBatchWidth;
inline constexpr uint32_t kDecodeRows = ExecutionLimits::targetVerifyRows;
// Per-lane sampling uniforms handed to the sampler each cycle.
inline constexpr uint32_t kSamplingUniformCount = 16;
inline constexpr uint32_t kDraftProposalTokens = ExecutionLimits::draftProposalTokens;
inline constexpr uint32_t kPrefillRows = ExecutionLimits::prefillTokenBudget;
inline constexpr uint32_t kTileRows = kv::kPageTokens;
inline constexpr uint32_t kPackedAttentionRows =
    kPrefillRows + kLaneCount * (kTileRows - 1);
inline constexpr uint32_t kDraftCacheStride = ExecutionLimits::draftContextTokens;
inline constexpr uint64_t kArenaAlignment = 16 * 1024;
inline constexpr uint32_t kMaximumPageTableEntries =
    (kv::kMaximumPhysicalTokens + kv::kPageTokens - 1) / kv::kPageTokens;

struct RuntimeGeometry final {
  QwenTargetGeometry target;
  DFlashDraftLayout draft;
  DraftStateLayout draftState;

  [[nodiscard]] static RuntimeGeometry from(
      const ModelPackage &package, kv::Format format = kv::Format::Int8) {
    RuntimeGeometry result;
    result.target = std::visit(
        [](const auto &weights) { return qwenTargetGeometry(weights); },
        package.target);
    result.target.kvLayout = package.targetKvLayout(format);
    result.draft = package.draft.layout;
    result.draftState = result.draft.stateLayout();
    if (!result.target.valid() || !result.draftState.valid() ||
        result.target.hiddenSize != result.draft.hiddenSize ||
        result.target.vocabularySize != result.draft.vocabularySize ||
        result.target.capturedHiddenSize() != result.draft.targetHiddenSize) {
      throw std::invalid_argument("invalid model runtime geometry");
    }
    return result;
  }

  [[nodiscard]] uint32_t maskWords() const noexcept {
    return (target.vocabularySize + 31) / 32;
  }
  [[nodiscard]] uint32_t projectionSumsWidth() const noexcept {
    uint32_t maximumInput = std::max(
        {target.hiddenSize, target.attentionWidth,
         target.capturedHiddenSize(), target.ffnScratchWidth(),
         draft.targetHiddenSize, draft.hiddenSize,
         draft.intermediateSize});
    return (maximumInput + kQ4GroupElements - 1) / kQ4GroupElements;
  }
};

constexpr uint64_t alignArena(uint64_t bytes) noexcept {
  return (bytes + kArenaAlignment - 1) & ~(kArenaAlignment - 1);
}

inline uint64_t checkedAdd(uint64_t left, uint64_t right,
                    std::string_view description) {
  if (left > std::numeric_limits<uint64_t>::max() - right) {
    throw std::overflow_error(std::string(description) + " overflows");
  }
  return left + right;
}

inline uint64_t checkedMultiply(uint64_t left, uint64_t right,
                         std::string_view description) {
  if (left && right > std::numeric_limits<uint64_t>::max() / left) {
    throw std::overflow_error(std::string(description) + " overflows");
  }
  return left * right;
}

template <class T> constexpr uint64_t bytesFor(uint64_t elements) noexcept {
  return elements * sizeof(T);
}

enum class PrefillTensor : uint32_t {
  Hidden0,
  Hidden1,
  InputTokens,
  Normalized,
  Captured,
  GdnPacked,
  GdnQueries,
  GdnKeys,
  GdnValues,
  GdnDecay,
  GdnBeta,
  Recurrent,
  GdnHidden,
  GdnOutput,
  GateIntermediate,
  Intermediate,
  FullPacked,
  FullQueries,
  FullAttention,
  AttentionPartials,
  AttentionStatistics,
  AttentionHidden,
  AttentionOutput,
  ProjectionSums,
  DownProjectionSums,
  TargetPositions,
  DraftPositions,
  TargetInverseFrequencies,
  DraftInverseFrequencies,
  RopeCos,
  RopeSin,
  ContextProjected,
  ContextHidden,
  ContextQkv,
  DraftRopeCos,
  DraftRopeSin,
  ChunkKeys,
  ChunkValues,
  MoeSelectedExperts,
  MoeRoutingWeights,
  MoeTileDescriptors,
  MoeTileCount,
  MoeGroupedRoutes,
  MoeRouteRows,
  MoeGroupedInput,
  MoeExpertIntermediate,
  MoeExpertOutput,
  Count,
};

constexpr uint32_t prefillTensorCount =
    static_cast<uint32_t>(PrefillTensor::Count);

// Sizes depend on the geometry and the installed operator choices; the arena
// bounds include the operator defaults and every installed configuration.
[[nodiscard]] std::array<uint64_t, prefillTensorCount>
prefillTensorBytes(const RuntimeGeometry &geometry,
                   const ops::ExecutionPlans &operators);
[[nodiscard]] uint64_t plannedPrefillBytes(const RuntimeGeometry &geometry,
                                           const ops::ExecutionPlans &operators);

class PrefillArena final {
public:
  PrefillArena(metal::MetalBackend &backend, const RuntimeGeometry &geometry,
                const ops::ExecutionPlans &operators)
      : bytes_(plannedPrefillBytes(geometry, operators)) {
    const auto sizes = prefillTensorBytes(geometry, operators);
    base_ = backend.allocateBuffer(bytes_, metal::BufferStorage::Shared,
                                   "qwen-shared-prefill");
    uint64_t cursor = 0;
    for (uint32_t index = 0; index < sizes.size(); ++index) {
      if (sizes[index])
        tensors_[index] = backend.view(base_, cursor, sizes[index]);
      cursor += alignArena(sizes[index]);
    }
    if (cursor != bytes_)
      throw std::logic_error("prefill arena mismatch");
    auto *target = static_cast<float *>(
        get(PrefillTensor::TargetInverseFrequencies).contents());
    auto *draft = static_cast<float *>(
        get(PrefillTensor::DraftInverseFrequencies).contents());
    if (!target || !draft)
      throw std::logic_error("RoPE frequencies are not CPU-visible");
    for (uint32_t dim = 0; dim < geometry.target.rotaryPairs; ++dim) {
      target[dim] =
          std::pow(geometry.target.rotaryTheta,
                   -static_cast<float>(dim) / geometry.target.rotaryPairs);
    }
    const uint32_t draftRotaryPairs = geometry.draftState.headDimension / 2;
    for (uint32_t dim = 0; dim < draftRotaryPairs; ++dim) {
      draft[dim] = std::pow(geometry.draft.rotaryTheta,
                            -static_cast<float>(dim) / draftRotaryPairs);
    }
  }

  [[nodiscard]] metal::MetalBuffer get(PrefillTensor tensor) const {
    return tensors_[static_cast<uint32_t>(tensor)];
  }
  [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }

private:
  metal::MetalBuffer base_;
  std::array<metal::MetalBuffer, prefillTensorCount> tensors_{};
  uint64_t bytes_ = 0;
};

enum class DecodeTensor : uint32_t {
  Hidden0,
  Hidden1,
  InputTokens,
  Normalized,
  Recurrent,
  GdnHidden,
  GdnOutput,
  Intermediate,
  FullPacked,
  FullQueries,
  AttentionPartials,
  AttentionStatistics,
  FullAttention,
  AttentionHidden,
  AttentionOutput,
  Positions,
  DraftPositions,
  RopeCos,
  RopeSin,
  Arrived,
  Generation,
  ContextProjected,
  ContextHidden,
  ContextQkv,
  CapturedTargetHidden,
  DraftQueryKeys,
  DraftQueryValues,
  DraftRopeCos,
  DraftRopeSin,
  FinalHidden,
  Logits,
  ArgmaxValues,
  ArgmaxIndices,
  TargetTopPartialIds,
  TargetTopPartialValues,
  TargetTopIds,
  TargetTopProbs,
  SamplingUniforms,
  ConstraintMasks,
  OutputTokens,
  RetainedCount,
  NextAnchor,
  AcceptedCount,
  DraftInputTokens,
  DraftHidden0,
  DraftHidden1,
  DraftNormalized,
  DraftDynamic,
  DraftConvolved,
  DraftProposalQkv,
  DraftAttention,
  DraftProjected,
  DraftResidual,
  DraftIntermediate,
  DraftFinalHidden,
  SelectorHidden,
  Candidates,
  Unary,
  TopPartialIds,
  TopPartialValues,
  ProposalProbs,
  ProposedTokens,
  PageTable,
  VerifyPackedBase,
  VerifyMixedBase,
  VerifyDecayBase,
  VerifyBetaBase,
  ChunkKeysBase,
  ChunkValuesBase,
  MoeSelectedExperts,
  MoeRoutingWeights,
  MoeTileDescriptors,
  MoeTileCount,
  MoeGroupedRoutes,
  MoeRouteRows,
  MoeGroupedInput,
  MoeExpertIntermediate,
  MoeExpertOutput,
  Count,
};

constexpr uint32_t decodeTensorCount =
    static_cast<uint32_t>(DecodeTensor::Count);

constexpr bool isGdnLayerTensor(DecodeTensor tensor) noexcept {
  return tensor == DecodeTensor::VerifyPackedBase ||
         tensor == DecodeTensor::VerifyMixedBase ||
         tensor == DecodeTensor::VerifyDecayBase ||
         tensor == DecodeTensor::VerifyBetaBase;
}

constexpr bool isAttentionLayerTensor(DecodeTensor tensor) noexcept {
  return tensor == DecodeTensor::ChunkKeysBase ||
         tensor == DecodeTensor::ChunkValuesBase;
}

constexpr bool isLayerMajorTensor(DecodeTensor tensor) noexcept {
  return isGdnLayerTensor(tensor) || isAttentionLayerTensor(tensor);
}

// The layer-major tensors are consecutive enumerators, so each family indexes
// its precomputed views by distance from its first member.
static_assert(static_cast<uint32_t>(DecodeTensor::VerifyBetaBase) -
                      static_cast<uint32_t>(DecodeTensor::VerifyPackedBase) ==
                  3,
              "GDN layer tensors must be consecutive");
static_assert(static_cast<uint32_t>(DecodeTensor::ChunkValuesBase) -
                      static_cast<uint32_t>(DecodeTensor::ChunkKeysBase) ==
                  1,
              "attention layer tensors must be consecutive");

[[nodiscard]] uint64_t decodeChunkLayerBytes(const RuntimeGeometry &geometry) noexcept;
[[nodiscard]] std::array<uint64_t, decodeTensorCount>
decodeTensorBytes(const RuntimeGeometry &geometry,
                  const ops::ExecutionPlans &operators);
// A decode tensor is one packed M32 allocation.  B1/B2/B3/B4 are prefixes
// containing 8/16/24/32 rows. Lanes are never separated by arena-alignment
// holes; only whole tensor boundaries are aligned.
[[nodiscard]] uint64_t decodeArenaBaseBytes(const RuntimeGeometry &geometry,
                                            const ops::ExecutionPlans &operators);

class DecodeArena final {
public:
  DecodeArena(metal::MetalBackend &backend, RuntimeGeometry geometry,
               const ops::ExecutionPlans &operators)
      : backend_(backend), geometry_(std::move(geometry)) {
    const uint64_t baseBytes = decodeArenaBaseBytes(geometry_, operators);
    auto sizes = decodeTensorBytes(geometry_, operators);
    base_ = backend_.allocateBuffer(baseBytes, metal::BufferStorage::Shared,
                                    "qwen-shared-decode");
    uint64_t cursor = 0;
    for (uint32_t tensor = 0; tensor < sizes.size(); ++tensor) {
      uint64_t stride = sizes[tensor];
      offsets_[tensor] = cursor;
      sizes_[tensor] = sizes[tensor];
      const DecodeTensor kind = static_cast<DecodeTensor>(tensor);
      if (sizes[tensor] && !isLayerMajorTensor(kind)) {
        for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
          tensors_[lane][tensor] = backend_.view(
              base_, cursor + uint64_t{lane} * stride, sizes[tensor]);
        }
        for (uint32_t lanes = 1; lanes <= kLaneCount; ++lanes) {
          packed_[lanes - 1][tensor] =
              backend_.view(base_, cursor, uint64_t{lanes} * sizes[tensor]);
        }
      }
      cursor +=
          alignArena(checkedMultiply(stride, kLaneCount, "decode tensor"));
    }
    if (cursor != baseBytes)
      throw std::logic_error("decode arena mismatch");
    buildGdnViews();
    buildAttentionViews();

    const uint64_t denseScratchBytes = gateScratchBytes(geometry_, operators);
    if (denseScratchBytes) {
      gateScratch_ = backend_.allocateBuffer(
          denseScratchBytes, metal::BufferStorage::Private, "qwen-gate-scratch");
    }
    const auto linearSize = linearScratchSize(geometry_, operators);
    if (linearSize.bytes()) {
      linearScratch_.input = backend_.allocateBuffer(
          linearSize.input, metal::BufferStorage::Private, "q4-input");
      linearScratch_.sums = backend_.allocateBuffer(
          linearSize.sums, metal::BufferStorage::Private, "q4-sums");
      linearScratch_.partials = backend_.allocateBuffer(
          linearSize.partials, metal::BufferStorage::Private, "q4-partials");
      linearScratch_.counters = backend_.allocateBuffer(
          linearSize.counters, metal::BufferStorage::Shared, "q4-counters");
      std::memset(linearScratch_.counters.contents(), 0, linearSize.counters);
    }
    bytes_ = checkedAdd(checkedAdd(baseBytes, denseScratchBytes, "decode arena"),
                        linearSize.bytes(), "Q4 decode scratch");
  }

  [[nodiscard]] metal::MetalBuffer get(uint32_t lane, DecodeTensor tensor) const {
    if (lane >= kLaneCount)
      throw std::out_of_range("invalid decode lane");
    if (isLayerMajorTensor(tensor)) {
      throw std::logic_error(
          "layer-major decode scratch requires a layer view");
    }
    return tensors_[lane][static_cast<uint32_t>(tensor)];
  }

  [[nodiscard]] const metal::MetalBuffer &packed(DecodeTensor tensor,
                                                 uint32_t lanes) const {
    if (!lanes || lanes > kLaneCount)
      throw std::out_of_range("invalid packed decode width");
    if (isLayerMajorTensor(tensor)) {
      throw std::logic_error(
          "layer-major decode scratch requires a layer view");
    }
    // Model-specific scratch is represented by an empty buffer.  The
    // selected target graph consumes either dense-FFN or MoE tensors, never
    // both, so no dummy allocation is needed for the inactive operator.
    return packed_[lanes - 1][static_cast<uint32_t>(tensor)];
  }

  [[nodiscard]] ops::LinearScratch linearScratch() const { return linearScratch_; }
  static ops::LinearScratchSize linearScratchSize(const RuntimeGeometry &geometry,
                                                 const ops::ExecutionPlans &operators);

  [[nodiscard]] metal::MetalBuffer gateScratch() const { return gateScratch_; }

  static uint64_t gateScratchBytes(const RuntimeGeometry &geometry,
                                  const ops::ExecutionPlans &operators) {
    // One private gate buffer is reused serially by the target dense FFN (when
    // present) and the always-dense DFlash draft. Sparse target FFNs use their
    // own route-major arena tensors, but must not remove the draft's scratch.
    const uint64_t draft = operators.gateUpWorkspace(
        {geometry.draft.intermediateSize, geometry.draft.hiddenSize});
    const uint64_t target = geometry.target.denseIntermediateSize
        ? operators.gateUpWorkspace({geometry.target.denseIntermediateSize,
                                     geometry.target.hiddenSize})
        : 0;
    return std::max(target, draft);
  }

  [[nodiscard]] const metal::MetalBuffer &
  gdnBatchSlice(DecodeTensor base, uint32_t gdnLayer, uint32_t lanes) const {
    const uint32_t layers = geometry_.target.stateLayout.layers;
    if (!lanes || lanes > kLaneCount || gdnLayer >= layers ||
        !isGdnLayerTensor(base))
      throw std::out_of_range("invalid batched GDN layer");
    const metal::MetalBuffer &view = gdnSlices_[layerViewIndex(
        gdnFamilyIndex(base), gdnLayer, layers, lanes)];
    if (!view)
      throw std::logic_error("batched GDN layer is empty");
    return view;
  }

  [[nodiscard]] const metal::MetalBuffer &gdnStorage(DecodeTensor base) const {
    if (!isGdnLayerTensor(base))
      throw std::invalid_argument("tensor is not GDN replay scratch");
    const metal::MetalBuffer &view = gdnStorage_[gdnFamilyIndex(base)];
    if (!view)
      throw std::logic_error("GDN replay scratch is empty");
    return view;
  }

  [[nodiscard]] const metal::MetalBuffer &
  attentionBatchSlice(DecodeTensor base, uint32_t attentionLayer,
                      uint32_t lanes) const {
    const uint32_t layers = geometry_.target.kvLayout.attentionLayers;
    if (!lanes || lanes > kLaneCount || attentionLayer >= layers ||
        !isAttentionLayerTensor(base)) {
      throw std::out_of_range("invalid batched attention layer");
    }
    const metal::MetalBuffer &view = attentionSlices_[layerViewIndex(
        attentionFamilyIndex(base), attentionLayer, layers, lanes)];
    if (!view)
      throw std::logic_error("batched attention layer is empty");
    return view;
  }

  [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }

private:
  static constexpr uint32_t kGdnTensors = 4;
  static constexpr uint32_t kAttentionTensors = 2;

  static constexpr uint32_t gdnFamilyIndex(DecodeTensor tensor) noexcept {
    return static_cast<uint32_t>(tensor) -
           static_cast<uint32_t>(DecodeTensor::VerifyPackedBase);
  }
  static constexpr uint32_t attentionFamilyIndex(DecodeTensor tensor) noexcept {
    return static_cast<uint32_t>(tensor) -
           static_cast<uint32_t>(DecodeTensor::ChunkKeysBase);
  }
  // Layer views are stored family-major, then layer, then batch width 1-4.
  static size_t layerViewIndex(uint32_t family, uint32_t layer,
                               uint32_t layers, uint32_t lanes) noexcept {
    return (size_t{family} * layers + layer) * kLaneCount + (lanes - 1);
  }

  // A layer-major tensor holds kLaneCount strides per layer, laid out layer
  // by layer; a width-N slice is the first N strides of one layer.
  void buildLayerViews(std::vector<metal::MetalBuffer> &views, uint32_t index,
                       uint32_t family, uint32_t layers, uint64_t stride) {
    for (uint32_t layer = 0; layer < layers; ++layer) {
      const uint64_t relative = uint64_t{layer} * kLaneCount * stride;
      for (uint32_t lanes = 1; lanes <= kLaneCount; ++lanes) {
        const uint64_t bytes = uint64_t{lanes} * stride;
        if (relative + bytes > sizes_[index] * kLaneCount)
          throw std::logic_error("batched layer exceeds decode arena");
        views[layerViewIndex(family, layer, layers, lanes)] =
            backend_.view(base_, offsets_[index] + relative, bytes);
      }
    }
  }

  void buildGdnViews() {
    const uint32_t layers = geometry_.target.stateLayout.layers;
    gdnSlices_.resize(size_t{kGdnTensors} * layers * kLaneCount);
    for (uint32_t family = 0; family < kGdnTensors; ++family) {
      const uint32_t index =
          static_cast<uint32_t>(DecodeTensor::VerifyPackedBase) + family;
      if (!sizes_[index])
        continue;
      gdnStorage_[family] =
          backend_.view(base_, offsets_[index], sizes_[index] * kLaneCount);
      // The planner sized each GDN tensor as layers x stride, so the stride
      // is recovered here, not supplied.
      const uint64_t stride = sizes_[index] / layers;
      if (stride)
        buildLayerViews(gdnSlices_, index, family, layers, stride);
    }
  }

  void buildAttentionViews() {
    const uint32_t layers = geometry_.target.kvLayout.attentionLayers;
    const uint64_t stride = decodeChunkLayerBytes(geometry_);
    attentionSlices_.resize(size_t{kAttentionTensors} * layers * kLaneCount);
    for (uint32_t family = 0; family < kAttentionTensors; ++family) {
      const uint32_t index =
          static_cast<uint32_t>(DecodeTensor::ChunkKeysBase) + family;
      if (sizes_[index] && stride)
        buildLayerViews(attentionSlices_, index, family, layers, stride);
    }
  }

  metal::MetalBackend &backend_;
  RuntimeGeometry geometry_;
  metal::MetalBuffer base_;
  std::array<std::array<metal::MetalBuffer, decodeTensorCount>, kLaneCount> tensors_{};
  // Every batch view a decode step binds, taken once at construction so a
  // step only looks them up. packed_ is indexed [width - 1][tensor].
  std::array<std::array<metal::MetalBuffer, decodeTensorCount>, kLaneCount> packed_{};
  std::array<metal::MetalBuffer, kGdnTensors> gdnStorage_{};
  std::vector<metal::MetalBuffer> gdnSlices_;
  std::vector<metal::MetalBuffer> attentionSlices_;
  std::array<uint64_t, decodeTensorCount> offsets_{};
  std::array<uint64_t, decodeTensorCount> sizes_{};
  metal::MetalBuffer gateScratch_;
  ops::LinearScratch linearScratch_;
  uint64_t bytes_ = 0;
};

// A decode step binds these views without taking a new one.
static_assert(std::is_same_v<decltype(std::declval<const DecodeArena &>().packed(
                                 DecodeTensor::Hidden0, 1)),
                             const metal::MetalBuffer &>);
static_assert(std::is_same_v<decltype(std::declval<const DecodeArena &>().gdnBatchSlice(
                                 DecodeTensor::VerifyPackedBase, 0, 1)),
                             const metal::MetalBuffer &>);
static_assert(std::is_same_v<decltype(std::declval<const DecodeArena &>().gdnStorage(
                                 DecodeTensor::VerifyPackedBase)),
                             const metal::MetalBuffer &>);
static_assert(std::is_same_v<decltype(std::declval<const DecodeArena &>().attentionBatchSlice(
                                 DecodeTensor::ChunkKeysBase, 0, 1)),
                             const metal::MetalBuffer &>);

[[nodiscard]] uint64_t plannedDecodeBytes(const RuntimeGeometry &geometry,
                                          const ops::ExecutionPlans &operators);

} // namespace splash::model
