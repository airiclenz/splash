#pragma once

// Engine-test doubles shared by the engine test binaries: a KV backing whose
// growth a test can block or gate, a fake model that records what the engine
// asked of it, and an event sink that records what the engine reported.

#include "TestImmediateTicket.hpp"
#include "engine/Engine.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace splash::engine::test {

class Backing final : public KvBacking {
public:
  explicit Backing(uint32_t pages) : resident_(pages) {}
  uint32_t pageCount() const noexcept override { return resident_.size(); }
  uint64_t bytesPerPage() const noexcept override { return 4096; }
  bool isResident(uint32_t page) const override { return resident_.at(page); }
  splash::metal::AllocationResult ensureResident(uint32_t page) override {
    ++growthAttempts;
    if (growthBlocked || (growthAllowed && !growthAllowed()))
      return allocationFailure;
    uint32_t first = extentFirstPage(page);
    for (uint32_t i = first; i < first + extentPageCount(page); ++i) {
      resident_.at(i) = true;
    }
    return true;
  }
  bool releaseBackingForPage(uint32_t page) override {
    uint32_t first = extentFirstPage(page);
    for (uint32_t i = first; i < first + extentPageCount(page); ++i) {
      resident_.at(i) = false;
    }
    releasePending = deferRelease;
    return true;
  }
  bool releaseReady() const noexcept override {
    if (completeReleaseOnPoll)
      releasePending = false;
    return !releasePending;
  }
  uint32_t extentFirstPage(uint32_t page) const override {
    return page - page % 4;
  }
  uint32_t extentPageCount(uint32_t page) const override {
    return std::min<uint32_t>(4, resident_.size() - extentFirstPage(page));
  }
  bool deferRelease = false;
  mutable bool releasePending = false;
  bool completeReleaseOnPoll = false;
  bool growthBlocked = false;
  metal::AllocationFailure allocationFailure = metal::AllocationFailure::Capacity;
  uint32_t growthAttempts = 0;
  std::function<bool()> growthAllowed;

private:
  std::vector<bool> resident_;
};

// refillsReserve models a state whose drop refills its producing lane's
// checkpoint reserve, so it returns nothing to the model's pool.
class State final : public CompositeState {
public:
  explicit State(bool carriesDraft = true, bool refillsReserve = false)
      : carriesDraft_(carriesDraft), refillsReserve_(refillsReserve) {}
  uint64_t bytes() const noexcept override { return 64; }
  uint64_t reclaimableBytes() const noexcept override {
    return refillsReserve_ ? 0 : bytes();
  }
  bool carriesDraft() const noexcept override { return carriesDraft_; }

private:
  bool carriesDraft_;
  bool refillsReserve_;
};

// A live state moved out at suspension. Dropping it parks its bytes in the
// fake's idle pool, as the production state returns its cell to the model's
// pool, where only idle-state reclaim frees it.
class ParkingState final : public CompositeState {
public:
  ParkingState(bool carriesDraft, std::shared_ptr<uint64_t> idlePool)
      : carriesDraft_(carriesDraft), idlePool_(std::move(idlePool)) {}
  ~ParkingState() override { *idlePool_ += bytes(); }
  uint64_t bytes() const noexcept override { return 64; }
  bool carriesDraft() const noexcept override { return carriesDraft_; }

private:
  bool carriesDraft_;
  std::shared_ptr<uint64_t> idlePool_;
};

struct MaskOverlapState final {
  uint64_t requestId = 0;
  bool emitted = false;
  bool provided = false;
  bool abandoned = false;
};

class MaskOverlapTicket final : public ModelBatchTicket {
public:
  explicit MaskOverlapTicket(std::shared_ptr<MaskOverlapState> state)
      : state_(std::move(state)) {}

  std::vector<ModelMaskRequest> takeMaskRequests() override {
    if (state_->emitted)
      return {};
    state_->emitted = true;
    return {{state_->requestId, {10, 11, 12, 13, 14, 15, 16, 17}}};
  }
  bool ownsMaskWait(uint64_t id) const noexcept override {
    return state_->emitted && id == state_->requestId;
  }
  void abandonMask(uint64_t id) noexcept override {
    if (id == state_->requestId)
      state_->abandoned = true;
  }
  bool ready() const noexcept override {
    return state_->provided || state_->abandoned;
  }
  std::vector<ModelStepResult> wait() override {
    if (!ready())
      throw std::logic_error("mask-overlap ticket completed without input");
    return {{state_->requestId, 0, {42}, true, DecodeStage::Regular, 7, 0}};
  }
  double wallMilliseconds() const noexcept override { return 1.0; }

private:
  std::shared_ptr<MaskOverlapState> state_;
};

// A decode command that stays in flight until the test releases it.
class HeldTicket final : public ModelBatchTicket {
public:
  HeldTicket(std::vector<ModelStepResult> results,
             std::shared_ptr<bool> released)
      : results_(std::move(results)), released_(std::move(released)) {}
  bool ready() const noexcept override { return *released_; }
  std::vector<ModelStepResult> wait() override { return std::move(results_); }
  double wallMilliseconds() const noexcept override { return 1.0; }

private:
  std::vector<ModelStepResult> results_;
  std::shared_ptr<bool> released_;
};

class Executor final : public model::Model {
public:
  explicit Executor(
      uint32_t maximumCells = model::ExecutionLimits::maximumBatchWidth)
      : maximumCells(maximumCells) {}

  StateAdmission begin(const ModelRequest &request) override {
    ++beginAttempts;
    if (beginObserver) beginObserver();
    if (beginGrowthBlocked && beginGrowthBlocked())
      return {{}, StateFailure::MemoryPressure, beginAllocationFailure};
    if (deniedBegins) {
      --deniedBegins;
      return {{}, StateFailure::MemoryPressure};
    }
    for (uint32_t slot = 0; slot < maximumCells; ++slot) {
      const bool used = std::any_of(
          requests.begin(), requests.end(), [slot](const auto &entry) {
            return entry.second.resident && entry.second.slot == slot;
          });
      if (!used) {
        requests.emplace(
            request.id,
            Request{slot, 0, true, false, 0, request.maxNewTokens});
        return {slot, StateFailure::None};
      }
    }
    return {{}, StateFailure::ConcurrencyLimit};
  }
  void suspend(uint64_t id) override {
    Request &entry = requests.at(id);
    if (!entry.resident)
      throw std::logic_error("request is already suspended");
    suspendedPositions.push_back(entry.position);
    entry.resident = false;
    entry.position = 0;
    ++suspensions;
    if (physicalGrowthBlocked && unblockGrowthOnSuspend)
      *physicalGrowthBlocked = false;
  }
  // Opt-in: the production model moves the lane's live state out at its
  // committed boundary. Off, the fake keeps the default (suspend, nothing
  // handed over), so the existing suspension pins still replay.
  std::shared_ptr<const CompositeState>
  suspendWithState(uint64_t id, uint32_t boundaryTokens) override {
    if (!moveStateOnSuspend)
      return model::Model::suspendWithState(id, boundaryTokens);
    if (!boundaryTokens || boundaryTokens % KvCache::pageTokens ||
        requests.at(id).position != boundaryTokens)
      throw std::logic_error("suspension boundary is not the lane's length");
    suspend(id);
    ++movedStates;
    return std::make_shared<ParkingState>(
        targetOnlyAtBoundary != boundaryTokens, parkedStateBytes);
  }
  StateAdmission resume(const ModelRequest &request) override {
    ++resumeAttempts;
    const uint64_t id = request.id;
    Request &entry = requests.at(id);
    for (uint32_t slot = 0; slot < maximumCells; ++slot) {
      const bool used = std::any_of(
          requests.begin(), requests.end(), [&](const auto &candidate) {
            return candidate.first != id && candidate.second.resident &&
                   candidate.second.slot == slot;
          });
      if (!used) {
        entry.slot = slot;
        entry.resident = true;
        entry.position = 0;
        entry.replaying = true;
        resumedPrompts.emplace_back(request.prompt.begin(), request.prompt.end());
        ++resumptions;
        return {slot, StateFailure::None};
      }
    }
    return {{}, StateFailure::ConcurrencyLimit};
  }
  void restore(uint64_t id, uint32_t length,
               std::shared_ptr<const CompositeState> state,
               bool restoreDraftState) override {
    if (!state)
      throw std::runtime_error("empty restore state");
    requests.at(id).position = length;
    restored += length;
    restoredLengths[id] = length;
    restoredDraft = restoreDraftState;
  }
  void setDraftContextPlan(uint64_t id, DraftContextPlan plan) override {
    plans[id] = std::move(plan);
  }
  std::vector<ModelStepResult> prefill(const BatchPlan &,
                                       std::span<const ModelBatchItem> items) {
    prefillWidths.push_back(static_cast<uint32_t>(items.size()));
    std::vector<ModelStepResult> result;
    for (const auto &item : items) {
      requests.at(item.requestId).position += item.tokenCount;
      prefillRows += item.tokenCount;
      ModelStepResult step{item.requestId, item.tokenCount, {}, false,
                           requests.at(item.requestId).replaying
                               ? replayDecodeStage : DecodeStage::Regular,
                           0, 0};
      if (prefillAnchor && !requests.at(item.requestId).replaying) {
        // Prefill selected a stop token or the last budgeted token: emitted
        // now, without a KV row, and the request never decodes.
        step.outputTokens = {42};
        step.outputTokensWithoutKv = 1;
        step.finished = *prefillAnchor;
      }
      result.push_back(std::move(step));
    }
    return result;
  }
  std::vector<ModelStepResult> decode(const BatchPlan &plan,
                                      std::span<const ModelBatchItem> items) {
    std::vector<ModelStepResult> result;
    for (const auto &item : items) {
      // The production runtime stores eight verify rows from the lane's
      // position; the engine must have covered them with page-table entries.
      if (uint64_t{item.pageTable.size()} * KvCache::pageTokens <
          item.logicalPosition + model::ExecutionLimits::targetVerifyRows) {
        throw std::invalid_argument("page_table_too_short");
      }
      if (plan.cohort == BatchCohort::Constrained &&
          plan.decodeStage == DecodeStage::RequestInitialMask) {
        result.push_back({item.requestId,
                          0,
                          {},
                          false,
                          DecodeStage::ApplyInitialMask,
                          0,
                          0});
      } else if (decodeRowsPerStep) {
        result.push_back(decodeRows(item));
      } else {
        ModelStepResult step{item.requestId, 0, {42}, decodeFinishes,
                             DecodeStage::Regular, 0, 0};
        step.outputTokensWithoutKv = decodeTokensWithoutKv;
        result.push_back(std::move(step));
      }
    }
    return result;
  }
  // Opt-in multi-token decode: like the production runtime, a step retains
  // at most the remaining budget, decodeRowsPerStep rows and the engine's
  // retained cap, each row with a KV row, advancing the lane's position.
  ModelStepResult decodeRows(const ModelBatchItem &item) {
    Request &lane = requests.at(item.requestId);
    uint32_t rows =
        std::min(decodeRowsPerStep, lane.maxNewTokens - lane.generated);
    if (item.retainedTokenLimit) {
      retainedLimits.push_back(item.retainedTokenLimit);
      if (!ignoresRetainedLimit)
        rows = std::min(rows, item.retainedTokenLimit);
    }
    lane.generated += rows;
    lane.position += rows;
    return {item.requestId, 0, std::vector<uint32_t>(rows, 42), false,
            DecodeStage::Regular, 0, 0};
  }
  std::unique_ptr<ModelBatchTicket>
  submit(const BatchPlan &plan, std::span<const ModelBatchItem> items,
         std::function<void()> completion) override {
    if (plan.kind == WorkKind::Decode &&
        plan.cohort == BatchCohort::Constrained &&
        plan.decodeStage == DecodeStage::ApplyInitialMask) {
      overlap = std::make_shared<MaskOverlapState>();
      overlap->requestId = items.front().requestId;
      return std::make_unique<MaskOverlapTicket>(overlap);
    }
    if (plan.kind == WorkKind::Decode && holdDecodeUntil) {
      return std::make_unique<HeldTicket>(decode(plan, items), holdDecodeUntil);
    }
    if (plan.kind == WorkKind::Prefill && holdPrefillUntil) {
      return std::make_unique<HeldTicket>(prefill(plan, items), holdPrefillUntil);
    }
    return splash::test::immediateTicket(plan.kind == WorkKind::Prefill
                                             ? prefill(plan, items)
                                             : decode(plan, items),
                                         completion);
  }
  // The production model copies the lane's state into a cache slot at its
  // current page-aligned boundary and returns nullptr when no slot is free
  // and the governor denies a new one. The fake denies the next
  // `deniedSnapshots` calls, or every call made at `denySnapshotAtBoundary`.
  std::shared_ptr<const CompositeState> snapshot(uint64_t id) override {
    ++snapshotAttempts;
    if (snapshotObserver)
      snapshotObserver();
    if (deniedSnapshots) {
      --deniedSnapshots;
      return nullptr;
    }
    if (denySnapshotAtBoundary &&
        *denySnapshotAtBoundary == requests.at(id).position) {
      return nullptr;
    }
    ++snapshots;
    return std::make_shared<State>(
        targetOnlyAtBoundary != requests.at(id).position,
        reserveRefillingSnapshots);
  }
  uint64_t reclaimIdleState() noexcept override {
    if (idleStateReclaimable && !idleStateReclaimable())
      return 0;
    const uint64_t released = reclaimableIdleStateBytes + *parkedStateBytes;
    reclaimableIdleStateBytes = 0;
    *parkedStateBytes = 0;
    reclaimedIdleStateBytes += released;
    if (released && physicalGrowthBlocked)
      *physicalGrowthBlocked = false;
    return released;
  }
  void provideMask(uint64_t id, std::span<const uint32_t>) override {
    if (overlap && overlap->emitted && overlap->requestId == id)
      overlap->provided = true;
  }
  void end(uint64_t id) override { requests.erase(id); }

  struct Request {
    uint32_t slot = 0;
    uint32_t position = 0;
    bool resident = false;
    bool replaying = false;
    // Output the host continuation emitted; survives suspension.
    uint32_t generated = 0;
    uint32_t maxNewTokens = 0;
  };
  std::unordered_map<uint64_t, Request> requests;
  std::unordered_map<uint64_t, DraftContextPlan> plans;
  uint32_t prefillRows = 0;
  uint32_t restored = 0;
  uint32_t snapshots = 0;
  uint32_t snapshotAttempts = 0;
  uint32_t deniedSnapshots = 0;
  std::optional<uint32_t> denySnapshotAtBoundary;
  // A snapshot taken at this boundary is target-only.
  std::optional<uint32_t> targetOnlyAtBoundary;
  // Snapshots return states whose drop refills their lane's reserve.
  bool reserveRefillingSnapshots = false;
  std::unordered_map<uint64_t, uint32_t> restoredLengths;
  uint32_t beginAttempts = 0;
  uint32_t deniedBegins = 0;
  uint32_t suspensions = 0;
  // The lane position each suspension found, before it was reset.
  std::vector<uint32_t> suspendedPositions;
  bool moveStateOnSuspend = false;
  uint32_t movedStates = 0;
  // Bytes of dropped moved states, parked until idle-state reclaim.
  std::shared_ptr<uint64_t> parkedStateBytes = std::make_shared<uint64_t>(0);
  uint32_t resumptions = 0;
  uint32_t resumeAttempts = 0;
  uint32_t maximumCells = model::ExecutionLimits::maximumBatchWidth;
  std::function<void()> beginObserver;
  std::function<void()> snapshotObserver;
  std::function<bool()> beginGrowthBlocked;
  std::vector<std::vector<uint32_t>> resumedPrompts;
  std::vector<uint32_t> prefillWidths;
  bool restoredDraft = false;
  metal::AllocationFailure beginAllocationFailure = metal::AllocationFailure::None;
  uint64_t reclaimableIdleStateBytes = 0;
  // When set, idle state is reclaimable only while this holds.
  std::function<bool()> idleStateReclaimable;
  uint64_t reclaimedIdleStateBytes = 0;
  bool *physicalGrowthBlocked = nullptr;
  bool unblockGrowthOnSuspend = true;
  bool decodeFinishes = true;
  DecodeStage replayDecodeStage = DecodeStage::Regular;
  uint32_t decodeTokensWithoutKv = 0;
  // Nonzero selects decodeRows: rows each Regular decode step may retain.
  uint32_t decodeRowsPerStep = 0;
  // decodeRows commits past the engine's cap, as an uncapped runtime would.
  bool ignoresRetainedLimit = false;
  // Every nonzero retained cap decodeRows received.
  std::vector<uint32_t> retainedLimits;
  // Set when prefill itself ends the request: the value is `finished` (stop).
  std::optional<bool> prefillAnchor;
  std::shared_ptr<bool> holdDecodeUntil;
  std::shared_ptr<bool> holdPrefillUntil;
  std::shared_ptr<MaskOverlapState> overlap;
};

class Events final : public EngineEventSink {
public:
  void batchCompleted(WorkKind, uint32_t, uint32_t, uint32_t, uint32_t,
                      uint32_t, double) override {}
  void started(uint64_t requestId, EngineCacheStatus cache, uint32_t matched,
               uint32_t) override {
    startIds.push_back(requestId);
    starts.emplace_back(std::move(cache), matched);
  }
  void promptProgress(uint64_t id, uint32_t processed) override {
    progress[id].push_back(processed);
  }
  void tokens(uint64_t id, std::span<const uint32_t> values) override {
    emitted += values.size();
    auto &transcript = outputs[id];
    transcript.insert(transcript.end(), values.begin(), values.end());
  }
  void maskRequested(uint64_t requestId,
                     std::span<const uint32_t> simulation) override {
    maskRequests.emplace_back(
        requestId, std::vector<uint32_t>(simulation.begin(), simulation.end()));
  }
  void completed(uint64_t id, EngineFinishReason reason, uint32_t prompt,
                 uint32_t completion, std::span<const float>) override {
    ++completedCount;
    usage[id] = {prompt, completion};
    reasons[id] = reason;
  }
  void failed(uint64_t, std::string code, std::string message,
              bool retryable) override {
    ++failedCount;
    failures.push_back(std::move(code));
    failureDetails.emplace_back(std::move(message), retryable);
  }
  void capacityExhausted(uint64_t, uint32_t, uint32_t, uint64_t) override {
    ++capacityExhaustedCount;
  }

  std::unordered_map<uint64_t, std::vector<uint32_t>> progress;
  std::vector<std::pair<EngineCacheStatus, uint32_t>> starts;
  std::vector<uint64_t> startIds;
  std::unordered_map<uint64_t, std::vector<uint32_t>> outputs;
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> usage;
  std::unordered_map<uint64_t, EngineFinishReason> reasons;
  std::vector<std::string> failures;
  std::vector<std::pair<std::string, bool>> failureDetails;
  uint32_t emitted = 0;
  uint32_t completedCount = 0;
  uint32_t failedCount = 0;
  uint32_t capacityExhaustedCount = 0;
  std::vector<std::pair<uint64_t, std::vector<uint32_t>>> maskRequests;
};

} // namespace splash::engine::test
