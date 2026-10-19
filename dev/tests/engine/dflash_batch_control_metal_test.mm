#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "model/Model.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using splash::metal::BufferStorage;
using splash::metal::ComputeDispatch;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;

constexpr uint32_t kRows = splash::model::ExecutionLimits::targetVerifyRows;
constexpr uint32_t kProposals =
    splash::model::ExecutionLimits::draftProposalTokens;
constexpr uint32_t kLanes =
    splash::model::ExecutionLimits::maximumBatchWidth;

MetalBuffer shared(MetalBackend &backend, uint64_t bytes, const char *label) {
  return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
}

template <class T> T *contents(const MetalBuffer &buffer) {
  return static_cast<T *>(buffer.contents());
}

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void runWidth(MetalBackend &backend, uint32_t width,
              const std::array<uint32_t, kLanes> &acceptedReference,
              const std::array<uint32_t, kLanes> &remainingReference,
              uint32_t stopLane = kLanes, uint32_t stopRow = 0) {
  require(width >= 1 && width <= kLanes, "invalid test width");
  MetalBuffer draft = shared(backend, kLanes * kProposals * sizeof(uint32_t),
                             "accept-draft");
  MetalBuffer draftIds =
      shared(backend, kLanes * kProposals * 16 * sizeof(uint32_t),
             "accept-draft-ids");
  MetalBuffer draftProbabilities =
      shared(backend, kLanes * kProposals * 16 * sizeof(float),
             "accept-draft-probabilities");
  MetalBuffer targetIds =
      shared(backend, kLanes * kRows * 32 * sizeof(uint32_t),
             "accept-target-ids");
  MetalBuffer targetProbabilities =
      shared(backend, kLanes * kRows * 32 * sizeof(float),
             "accept-target-probabilities");
  MetalBuffer uniforms =
      shared(backend, kLanes * 2 * kRows * sizeof(float), "accept-uniforms");
  MetalBuffer output = shared(backend, kLanes * kRows * sizeof(uint32_t),
                              "accept-output");
  MetalBuffer retained =
      shared(backend, kLanes * sizeof(uint32_t), "accept-retained");
  MetalBuffer next =
      shared(backend, kLanes * sizeof(uint32_t), "accept-next");
  MetalBuffer accepted =
      shared(backend, kLanes * sizeof(uint32_t), "accept-count");

  auto *draftTokens = contents<uint32_t>(draft);
  auto *targetTokens = contents<uint32_t>(output);
  std::memset(draftIds.contents(), 0, draftIds.sizeBytes());
  std::memset(draftProbabilities.contents(), 0,
              draftProbabilities.sizeBytes());
  std::memset(targetIds.contents(), 0, targetIds.sizeBytes());
  std::memset(targetProbabilities.contents(), 0,
              targetProbabilities.sizeBytes());
  std::memset(uniforms.contents(), 0, uniforms.sizeBytes());
  std::memset(retained.contents(), 0, retained.sizeBytes());
  std::memset(next.contents(), 0, next.sizeBytes());
  std::memset(accepted.contents(), 0, accepted.sizeBytes());
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    for (uint32_t token = 0; token < kProposals; ++token) {
      const uint32_t proposal = 1000 + lane * 100 + token;
      draftTokens[lane * kProposals + token] = proposal;
      targetTokens[lane * kRows + token] =
          token < acceptedReference[lane] ? proposal : proposal + 50;
    }
    targetTokens[lane * kRows + kRows - 1] = 9000 + lane;
  }

  constexpr uint32_t kStopToken = 248044;
  if (stopLane < width) {
    require(stopRow < kRows, "invalid stop row");
    targetTokens[stopLane * kRows + stopRow] = kStopToken;
    if (stopRow < acceptedReference[stopLane])
      draftTokens[stopLane * kProposals + stopRow] = kStopToken;
  }

  AcceptBatchParams params{};
  std::copy(remainingReference.begin(), remainingReference.end(),
            std::begin(params.remaining));
  params.stop_token_0 = kStopToken;
  params.stop_token_1 = 248046;
  params.lanes = width;
  ComputeDispatch dispatch;
  dispatch.pipelineName = "decode_accept_dflash";
  dispatch.buffers = {{0, draft},
                      {1, draftIds},
                      {2, draftProbabilities},
                      {3, targetIds},
                      {4, targetProbabilities},
                      {5, uniforms},
                      {6, output},
                      {7, retained},
                      {8, next},
                      {9, accepted}};
  dispatch.bytes = {{10, &params, sizeof(params)}};
  dispatch.threadgroups = {width, 1, 1};
  dispatch.threadsPerThreadgroup = {1, 1, 1};
  static_cast<void>(backend.submit(dispatch));

  const auto *retainedCounts = contents<uint32_t>(retained);
  const auto *nextTokens = contents<uint32_t>(next);
  const auto *acceptedCounts = contents<uint32_t>(accepted);
  for (uint32_t lane = 0; lane < width; ++lane) {
    uint32_t expectedRetained =
        std::min(acceptedReference[lane] + 1, remainingReference[lane]);
    if (lane == stopLane && stopRow < expectedRetained)
      expectedRetained = stopRow + 1;
    require(acceptedCounts[lane] == acceptedReference[lane],
            "accepted proposal count mismatch");
    require(retainedCounts[lane] == expectedRetained,
            "retained token count mismatch");
    require(nextTokens[lane] ==
                targetTokens[lane * kRows + expectedRetained - 1],
            "next anchor mismatch");
  }
}

// A step without proposals retains exactly the target's row-0 token in every
// lane. The proposal buffers hold garbage that the seven-proposal path would
// act on: greedy lanes propose the target's own tokens (all seven would be
// accepted), and sampled lanes carry a non-zero q that the residual draw
// would subtract, which moves its pick from token A to token B.
void runWithoutProposals(MetalBackend &backend, uint32_t width,
                         uint32_t samplingMask) {
  require(width >= 1 && width <= kLanes, "invalid test width");
  constexpr uint32_t kDraftCandidates = 16;
  constexpr uint32_t kTargetCandidates = 32;
  constexpr float kBonusUniform = 0.25F;
  constexpr float kGarbageUniform = 0.75F;
  constexpr float kGarbageDraftProbability = 0.9F;
  MetalBuffer draft = shared(backend, kLanes * kProposals * sizeof(uint32_t),
                             "no-proposal-draft");
  MetalBuffer draftIds = shared(
      backend, kLanes * kProposals * kDraftCandidates * sizeof(uint32_t),
      "no-proposal-draft-ids");
  MetalBuffer draftProbabilities =
      shared(backend, kLanes * kProposals * kDraftCandidates * sizeof(float),
             "no-proposal-draft-probabilities");
  MetalBuffer targetIds =
      shared(backend, kLanes * kRows * kTargetCandidates * sizeof(uint32_t),
             "no-proposal-target-ids");
  MetalBuffer targetProbabilities =
      shared(backend, kLanes * kRows * kTargetCandidates * sizeof(float),
             "no-proposal-target-probabilities");
  MetalBuffer uniforms = shared(backend, kLanes * 2 * kRows * sizeof(float),
                                "no-proposal-uniforms");
  MetalBuffer output = shared(backend, kLanes * kRows * sizeof(uint32_t),
                              "no-proposal-output");
  MetalBuffer retained =
      shared(backend, kLanes * sizeof(uint32_t), "no-proposal-retained");
  MetalBuffer next =
      shared(backend, kLanes * sizeof(uint32_t), "no-proposal-next");
  MetalBuffer accepted =
      shared(backend, kLanes * sizeof(uint32_t), "no-proposal-count");

  auto *draftTokens = contents<uint32_t>(draft);
  auto *draftCandidates = contents<uint32_t>(draftIds);
  auto *draftProbabilityValues = contents<float>(draftProbabilities);
  auto *targetCandidates = contents<uint32_t>(targetIds);
  auto *targetProbabilityValues = contents<float>(targetProbabilities);
  auto *uniformValues = contents<float>(uniforms);
  auto *targetTokens = contents<uint32_t>(output);
  std::memset(draftIds.contents(), 0, draftIds.sizeBytes());
  std::memset(draftProbabilities.contents(), 0,
              draftProbabilities.sizeBytes());
  std::memset(targetIds.contents(), 0, targetIds.sizeBytes());
  std::memset(targetProbabilities.contents(), 0,
              targetProbabilities.sizeBytes());
  std::memset(retained.contents(), 0, retained.sizeBytes());
  std::memset(next.contents(), 0, next.sizeBytes());
  std::memset(accepted.contents(), 0, accepted.sizeBytes());
  std::array<uint32_t, kLanes> expected{};
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    const uint32_t tokenA = 7000 + lane;
    const uint32_t tokenB = 7500 + lane;
    for (uint32_t row = 0; row < kRows; ++row)
      targetTokens[lane * kRows + row] = 3000 + lane * 100 + row;
    for (uint32_t token = 0; token < kProposals; ++token) {
      draftTokens[lane * kProposals + token] =
          targetTokens[lane * kRows + token];
      uint32_t *ids = draftCandidates + (lane * kProposals + token) *
                                            kDraftCandidates;
      float *probabilities = draftProbabilityValues +
                             (lane * kProposals + token) * kDraftCandidates;
      ids[0] = tokenA;
      probabilities[0] = kGarbageDraftProbability;
    }
    for (uint32_t row = 0; row < kRows; ++row) {
      uint32_t *ids = targetCandidates + (lane * kRows + row) *
                                             kTargetCandidates;
      float *probabilities = targetProbabilityValues +
                             (lane * kRows + row) * kTargetCandidates;
      ids[0] = tokenA;
      ids[1] = tokenB;
      probabilities[0] = 0.5F;
      probabilities[1] = 0.5F;
    }
    for (uint32_t index = 0; index < 2 * kRows; ++index)
      uniformValues[lane * 2 * kRows + index] = kGarbageUniform;
    uniformValues[lane * 2 * kRows + 2 * kRows - 1] = kBonusUniform;
    const bool isSampled = (samplingMask & (uint32_t{1} << lane)) != 0;
    expected[lane] = isSampled ? tokenA : targetTokens[lane * kRows];
  }

  AcceptBatchParams params{};
  std::fill(std::begin(params.remaining), std::end(params.remaining), kRows);
  params.stop_token_0 = 248044;
  params.stop_token_1 = 248046;
  params.lanes = width;
  params.sampling_mask = samplingMask;
  params.no_proposals = 1;
  ComputeDispatch dispatch;
  dispatch.pipelineName = "decode_accept_dflash";
  dispatch.buffers = {{0, draft},
                      {1, draftIds},
                      {2, draftProbabilities},
                      {3, targetIds},
                      {4, targetProbabilities},
                      {5, uniforms},
                      {6, output},
                      {7, retained},
                      {8, next},
                      {9, accepted}};
  dispatch.bytes = {{10, &params, sizeof(params)}};
  dispatch.threadgroups = {width, 1, 1};
  dispatch.threadsPerThreadgroup = {1, 1, 1};
  static_cast<void>(backend.submit(dispatch));

  const auto *retainedCounts = contents<uint32_t>(retained);
  const auto *nextTokens = contents<uint32_t>(next);
  const auto *acceptedCounts = contents<uint32_t>(accepted);
  for (uint32_t lane = 0; lane < width; ++lane) {
    require(acceptedCounts[lane] == 0,
            "no-proposal step accepted a proposal");
    require(retainedCounts[lane] == 1,
            "no-proposal step retained other than one token");
    require(targetTokens[lane * kRows] == expected[lane],
            "no-proposal step retained a token other than the row-0 sample");
    require(nextTokens[lane] == expected[lane],
            "no-proposal next anchor mismatch");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: dflash-batch-control METALLIB");
    MetalBackend backend(argv[1]);
    constexpr std::array<uint32_t, kLanes> lowerAccepted{0, 1, 2, 3};
    constexpr std::array<uint32_t, kLanes> upperAccepted{4, 5, 6, 7};
    constexpr std::array<uint32_t, kLanes> fullRemaining{8, 8, 8, 8};
    for (uint32_t width = 1; width <= kLanes; ++width) {
      runWidth(backend, width, lowerAccepted, fullRemaining);
      runWidth(backend, width, upperAccepted, fullRemaining);
    }

    // Output limits and stop tokens shorten the committed prefix without
    // changing the physical eight-row graph.  The accepted proposal count is
    // still seven in every lane; only retained rows and the next anchor move.
    constexpr std::array<uint32_t, kLanes> allAccepted{7, 7, 7, 7};
    constexpr std::array<uint32_t, kLanes> shortRemaining{1, 2, 3, 8};
    runWidth(backend, kLanes, allAccepted, shortRemaining, 3, 3);

    // Without proposals every lane keeps one token, greedy or sampled.
    constexpr uint32_t kGreedyMask = 0;
    constexpr uint32_t kMixedMask = 0b1010;
    constexpr uint32_t kSampledMask = (uint32_t{1} << kLanes) - 1;
    for (uint32_t width = 1; width <= kLanes; ++width) {
      runWithoutProposals(backend, width, kGreedyMask);
      runWithoutProposals(backend, width, kMixedMask);
      runWithoutProposals(backend, width, kSampledMask);
    }
    std::cout << "dflash_batch_control_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "dflash_batch_control_metal_test: FAIL: " << error.what()
              << '\n';
    return 1;
  }
}
