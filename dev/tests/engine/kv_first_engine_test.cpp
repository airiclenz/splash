#include "TestImmediateTicket.hpp"
#include "engine/Engine.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

using namespace splash;
using namespace splash::engine;

namespace {

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

class State final : public CompositeState {
public:
  explicit State(bool carriesDraft = true) : carriesDraft_(carriesDraft) {}
  uint64_t bytes() const noexcept override { return 64; }
  bool carriesDraft() const noexcept override { return carriesDraft_; }

private:
  bool carriesDraft_;
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
    return test::immediateTicket(plan.kind == WorkKind::Prefill
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
    return std::make_shared<State>(targetOnlyAtBoundary !=
                                   requests.at(id).position);
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

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

EngineRequest request(uint64_t id, const std::vector<uint32_t> &prompt) {
  EngineRequest result;
  result.id = id;
  result.prompt = prompt;
  result.maxNewTokens = 1;
  result.deadlineMilliseconds = 10000;
  return result;
}

using FailureCauses = std::array<uint64_t, kPublicationFailureCauses>;

// The checkpoint failure causes of a run that counted `count` under `cause`
// and nothing under any other.
FailureCauses onlyCause(PublicationFailure cause, uint64_t count) {
  FailureCauses causes{};
  causes[static_cast<std::size_t>(cause)] = count;
  return causes;
}

void runUntilIdle(engine::Engine &engine) {
  for (uint32_t step = 0; step < 32 && !engine.idle(); ++step) {
    static_cast<void>(engine.tick(step + 1));
  }
  require(engine.idle(), "engine did not reach idle");
}

// Finishes maximumBatchWidth unrelated one-page conversations, ids and
// tokens from firstId, so every earlier conversation end leaves the engine's
// ring and loses its protection; the new ends hold the ring instead.
void pushOutConversationEnds(engine::Engine &engine, uint32_t firstId) {
  for (uint32_t index = 0; index < model::ExecutionLimits::maximumBatchWidth;
       ++index) {
    const uint32_t id = firstId + index;
    engine.submit(request(id, std::vector<uint32_t>(65, id)));
    runUntilIdle(engine);
  }
}

// Runs request id until its prompt-end state is cached, then cancels it: the
// state stays cached without the conversation-end protection that only a
// Stop or Length finish keeps.
void cancelAfterPromptEnd(engine::Engine &engine, engine::Cache &resources,
                          uint64_t id, const std::vector<uint32_t> &prompt) {
  const uint32_t promptEnd = static_cast<uint32_t>(
      (prompt.size() - 1) / KvCache::pageTokens * KvCache::pageTokens);
  engine.submit(request(id, prompt));
  for (uint32_t step = 0;
       step < 32 && resources.cachedTokens(prompt) < promptEnd; ++step)
    static_cast<void>(engine.tick(step + 1));
  require(resources.cachedTokens(prompt) == promptEnd && !engine.idle(),
          "request did not stop running at its published prompt end");
  engine.cancel(id);
  runUntilIdle(engine);
}

void testConcurrentColdPrefixesComputeOnce() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache cache(pool, CacheNamespace{});
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  for (uint32_t id = 1; id <= 4; ++id) {
    std::vector<uint32_t> prompt(193, 7);
    std::fill(prompt.begin() + 160, prompt.end(), id + 10);
    engine.submit(request(id, prompt));
  }
  static_cast<void>(engine.tick(0));
  require(model.requests.size() == 1,
          "shared cold prefix allocated redundant active state cells");
  runUntilIdle(engine);
  require(model.prefillRows == 160 + 4 * 33 && model.restored == 3 * 160,
          "concurrent cold requests recomputed their shared prefix");
  require(events.completedCount == 4 && events.failedCount == 0 &&
              events.emitted == 4,
          "shared prefill lost independent completions");
}

void testSharedPrefillRebuildsTheMissingJunctionOnce() {
  Backing backing(512);
  KvPool pool(backing);
  engine::Cache cache(pool, CacheNamespace{});
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  engine.submit(request(1, std::vector<uint32_t>(6530, 7)));
  runUntilIdle(engine);
  std::vector<uint32_t> branch(6575, 7);
  std::fill(branch.begin() + 6517, branch.end(), 8);
  // The first request kept its 4096 checkpoint; the branch is KV-only past it.
  const uint32_t kept = 4096;
  {
    auto hit = cache.lookup(branch);
    require(hit.kvBoundary == 6496 && hit.resumeBoundary() == kept,
            "fixture did not recreate a KV-only internal branch");
  }
  const uint32_t before = model.prefillRows;
  for (uint32_t id = 2; id <= 5; ++id) {
    std::fill(branch.begin() + 6517, branch.end(), id + 10);
    engine.submit(request(id, branch));
  }
  runUntilIdle(engine);
  require(model.prefillRows - before == 6496 - kept + 4 * (6575 - 6496) &&
              model.restored == kept + 3 * 6496 && events.completedCount == 5,
          "concurrent internal branches each rebuilt the missing GDN state");
}

void testSharedPrefillReleasesDifferentJunctionsIndependently() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache cache(pool, CacheNamespace{});
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  engine.submit(request(1, std::vector<uint32_t>(193, 7)));
  for (uint32_t id = 2; id <= 3; ++id) {
    std::vector<uint32_t> branch(193, 7);
    std::fill(branch.begin() + (id == 2 ? 96 : 160), branch.end(), id + 10);
    engine.submit(request(id, branch));
  }
  static_cast<void>(engine.tick(0));
  static_cast<void>(engine.tick(1));
  static_cast<void>(engine.tick(2));
  require(model.requests.size() == 2 && events.startIds.back() == 2,
          "a ready junction waited for the producer's longer shared prefix");
  runUntilIdle(engine);
  require(model.prefillRows == 193 + 97 + 33 && events.completedCount == 3,
          "different shared boundaries were not restored independently");
}

void testSharedPrefillEvictedPublicationFallsBack() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache cache(pool, CacheNamespace{});
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  engine.submit(request(1, std::vector<uint32_t>(193, 7)));
  engine.submit(request(2, std::vector<uint32_t>(193, 7)));
  static_cast<void>(engine.tick(0));
  static_cast<void>(engine.tick(1));
  require(cache.reclaimOneState(),
          "published shared prefix was pinned against pressure reclamation");
  runUntilIdle(engine);
  require(events.completedCount == 2 && model.prefillRows == 386 &&
              cache.snapshot().activeRequests == 0,
          "evicted shared publication stranded its waiter");
}

void testSharedPrefillProducerFailureReleasesWaiters() {
  for (bool cancelled : {false, true}) {
    Backing backing(64);
    KvPool pool(backing);
    engine::Cache cache(pool, CacheNamespace{});
    Executor model;
    Events events;
    engine::Engine engine({}, cache, model, events);
    engine.submit(request(1, std::vector<uint32_t>(193, 7)));
    engine.submit(request(2, std::vector<uint32_t>(193, 7)));
    static_cast<void>(engine.tick(0));
    require(engine.snapshot().scheduler.waitingPrefix == 1 &&
                engine.resourceWaitSnapshot(0).memory == 0 &&
                model.requests.size() == 1,
            "shared prefill waiter was admitted before its prefix was ready");
    if (cancelled)
      engine.cancel(1);
    else
      engine.failRequest(1, "test_failure", "producer failed");
    runUntilIdle(engine);
    require(events.outputs[2] == std::vector<uint32_t>{42} &&
                cache.snapshot().activeRequests == 0 && model.requests.empty(),
            "failed prefix producer stranded a waiter or leaked resources");
  }
}

void testSharedPrefillWaiterCancellationAndDeadline() {
  for (bool cancelled : {false, true}) {
    Backing backing(64);
    KvPool pool(backing);
    engine::Cache cache(pool, CacheNamespace{});
    Executor model;
    Events events;
    engine::Engine engine({}, cache, model, events);
    engine.submit(request(1, std::vector<uint32_t>(193, 7)));
    auto waiter = request(2, std::vector<uint32_t>(193, 7));
    waiter.deadlineMilliseconds = 1;
    engine.submit(std::move(waiter));
    static_cast<void>(engine.tick(0));
    if (cancelled)
      engine.cancel(2);
    runUntilIdle(engine);
    require(model.beginAttempts == 1 && events.outputs[2].empty() &&
                events.outputs[1] == std::vector<uint32_t>{42} &&
                engine.snapshot().scheduler.waitingPrefix == 0,
            "expired prefix waiter allocated a cell or interrupted its producer");
  }
}

void testSharedPrefillFailedPublicationFallsBack() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache cache(pool, CacheNamespace{});
  Executor model;
  model.deniedSnapshots = 100;
  Events events;
  engine::Engine engine({}, cache, model, events);
  for (uint32_t id = 1; id <= 4; ++id)
    engine.submit(request(id, std::vector<uint32_t>(193, 7)));
  runUntilIdle(engine);
  require(events.completedCount == 4 && events.failedCount == 0 &&
              model.prefillRows == 4 * 193 && cache.snapshot().activeRequests == 0,
          "a missing prefix snapshot stranded dependent requests");
}

void testSharedPrefillDoesNotBlockUnrelatedWork() {
  for (bool images : {false, true}) {
    Backing backing(64);
    KvPool pool(backing);
    engine::Cache cache(pool, CacheNamespace{});
    Executor model;
    Events events;
    engine::Engine engine({}, cache, model, events);
    for (uint32_t id = 1; id <= 4; ++id) {
      auto value = request(id, std::vector<uint32_t>(193, images ? 7 : id));
      if (images) {
        value.images = {{0, 16, 8, 8, id, id}};
        value.imagePixels.assign(value.images.front().pixelBytes(), 1);
      }
      engine.submit(std::move(value));
    }
    static_cast<void>(engine.tick(0));
    require(model.requests.size() == 4 &&
                engine.snapshot().scheduler.waitingPrefix == 0,
            "unrelated token or image prefixes were serialized");
    runUntilIdle(engine);
    require(model.prefillRows == 4 * 193 && events.completedCount == 4,
            "unrelated work incorrectly reused a shared state");
  }
}

void testSharedPrefillHonorsPriorityAndLateArrival() {
  for (bool foreground : {false, true}) {
    Backing backing(512);
    KvPool pool(backing);
    engine::Cache cache(pool, CacheNamespace{});
    Executor model;
    Events events;
    engine::Engine engine({}, cache, model, events);
    auto producer = request(1, std::vector<uint32_t>(5001, 7));
    producer.priority = RequestPriority::Background;
    engine.submit(std::move(producer));
    static_cast<void>(engine.tick(0));
    auto waiter = request(2, std::vector<uint32_t>(5001, 7));
    waiter.priority = foreground ? RequestPriority::Foreground
                                : RequestPriority::Background;
    engine.submit(std::move(waiter));
    static_cast<void>(engine.tick(1));
    static_cast<void>(engine.tick(2));
    require(model.requests.size() == (foreground ? 2u : 1u),
            "prefix admission ignored priority or a late arrival");
    runUntilIdle(engine);
    require(events.completedCount == 2 && events.failedCount == 0,
            "late prefix waiter did not finish");
    if (!foreground)
      require(model.prefillRows == 5010 && model.restored == 4992,
              "late arrival missed the producer's planned replay point");
  }
}

void testLateSharedPrefillExtendsTheProducerPlan() {
  for (bool denyCheckpoint : {false, true}) {
    Backing backing(512);
    KvPool pool(backing);
    engine::Cache cache(pool, CacheNamespace{});
    Executor model;
    if (denyCheckpoint)
      model.denySnapshotAtBoundary = 4096;
    Events events;
    engine::Engine engine({}, cache, model, events);
    engine.submit(request(1, std::vector<uint32_t>(6601, 7)));
    static_cast<void>(engine.tick(0));
    for (uint32_t id = 2; id <= 4; ++id) {
      std::vector<uint32_t> branch(6601, 7);
      std::fill(branch.begin() + 6500, branch.end(), id + 10);
      engine.submit(request(id, branch));
    }
    runUntilIdle(engine);
    require(events.completedCount == 4 && events.failedCount == 0 &&
                model.prefillRows == 6601 + 3 * (6601 - 6496) &&
                model.restored == 3 * 6496,
            "late siblings recomputed the prefix after the producer's checkpoint");
  }
}

void testSharedPrefillCapacityFailureDoesNotDeadlock() {
  Backing backing(4);
  KvPool pool(backing);
  engine::Cache cache(pool, CacheNamespace{});
  Executor model;
  Events events;
  engine::Engine engine({}, cache, model, events);
  for (uint32_t id = 1; id <= 4; ++id)
    engine.submit(request(id, std::vector<uint32_t>(193, 7)));
  for (uint32_t step = 0; step < 64 && !engine.idle(); ++step)
    static_cast<void>(engine.tick(step * 1000));
  require(engine.idle() && cache.snapshot().activeRequests == 0 &&
              model.requests.empty() && events.emitted == 0,
          "capacity failure stranded a shared prefix producer or waiter");
  engine.submit(request(5, std::vector<uint32_t>(33, 9)));
  runUntilIdle(engine);
  require(events.outputs[5] == std::vector<uint32_t>{42},
          "capacity failure prevented subsequent service");
}

void testColdPublishesReplayStateAndLazyJunctionCanRebuildIt() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  Events events;
  engine::Engine engine({.maxContext = 102400}, resources, executor, events);
  require(engine.snapshot().maximumContextTokens == 102400,
          "snapshot lost the configured context limit");
  std::vector<uint32_t> prompt(65);
  for (uint32_t i = 0; i < prompt.size(); ++i)
    prompt[i] = i + 1;

  engine.submit(request(1, prompt));
  runUntilIdle(engine);
  require(executor.prefillRows == 65 && executor.snapshots == 1 &&
              engine.snapshot().replayStatePublications == 1,
          "cold request did not retain its latest Page32 replay state");
  require(events.starts.size() == 1 &&
              events.starts[0].first == EngineCacheStatus::Miss,
          "cold request reported a cache hit");

  require(resources.reclaimOneState(),
          "test could not remove the latest replay state");
  engine.submit(request(2, prompt));
  runUntilIdle(engine);
  require(executor.restored == 0 && executor.prefillRows == 130 &&
              executor.snapshots == 2 &&
              engine.snapshot().junctionMaterializations == 1,
          "second request did not lazily materialize its proven KV junction");
  require(events.starts.size() == 2 &&
              events.starts[1].first == EngineCacheStatus::Miss,
          "KV-only junction replay was reported as a state hit");

  engine.submit(request(3, prompt));
  runUntilIdle(engine);
  require(executor.restored == 64 && executor.prefillRows == 131,
          "cache hit did not replay exactly one real input token");
  require(events.starts.size() == 3 &&
              events.starts[2] ==
                  std::pair<EngineCacheStatus, uint32_t>{
                      EngineCacheStatus::PrefixHit, 64},
          "state-backed hit accounting is wrong");
  require(events.completedCount == 3 && events.failedCount == 0 &&
              events.emitted == 3,
          "request lifecycle did not complete cleanly");
}

void testConcurrentDuplicateStateSkipsSnapshotCapture() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  std::vector<uint32_t> prompt(65);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = index + 1;

  engine.submit(request(101, prompt));
  engine.submit(request(102, prompt));
  runUntilIdle(engine);

  // The second request restores the first publication without reserving a
  // redundant state cell or capturing the same state again.
  const auto snapshot = engine.snapshot();
  require(executor.snapshotAttempts == 1 && executor.snapshots == 1,
          "duplicate state publication performed a second snapshot capture");
  require(snapshot.resources.stateCache.entries == 1 &&
              snapshot.resources.stateCache.publications == 1 &&
              snapshot.cacheHits == 1 && executor.prefillRows == 66 &&
              snapshot.replayStatePublications == 1 &&
              snapshot.replayStatePublicationFailures == 0,
          "duplicate state publication was not reused and accounted");
}

void testImageSpansKeyPrefixIdentity() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  // Image runs render as one placeholder id, so two prompts with different
  // images are token-identical; only the span digests differ.
  std::vector<uint32_t> prompt(65, 248056);
  auto withImage = [&](uint64_t id, uint64_t digest) {
    EngineRequest result = request(id, prompt);
    result.images = {{8, 16, 8, 8, digest, digest ^ 0xabcdULL}};
    result.imagePixels.assign(result.images[0].pixelBytes(), 1);
    return result;
  };

  engine.submit(withImage(1, 0x1111));
  runUntilIdle(engine);
  require(events.starts.size() == 1 &&
              events.starts[0].first == EngineCacheStatus::Miss,
          "image producer unexpectedly hit the cache");
  engine.submit(withImage(2, 0x2222));
  runUntilIdle(engine);
  require(events.starts.size() == 2 &&
              events.starts[1] ==
                  std::pair<EngineCacheStatus, uint32_t>{
                      EngineCacheStatus::Miss, 0},
          "a different image falsely matched the cached prefix");
  engine.submit(withImage(3, 0x1111));
  runUntilIdle(engine);
  require(events.starts.size() == 3 &&
              events.starts[2] ==
                  std::pair<EngineCacheStatus, uint32_t>{
                      EngineCacheStatus::PrefixHit, 64},
          "an identical image did not reuse the cached prefix");
  engine.submit(request(4, prompt));
  runUntilIdle(engine);
  require(events.starts.size() == 4 &&
              events.starts[3] ==
                  std::pair<EngineCacheStatus, uint32_t>{
                      EngineCacheStatus::Miss, 0},
          "a text-only prompt matched an image-keyed prefix");

  EngineRequest malformed = withImage(5, 0x1111);
  malformed.images[0].tokens = 15;
  bool rejected = false;
  try {
    engine.submit(std::move(malformed));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected,
          "image span with the wrong merged token count was admitted");
  require(events.completedCount == 4 && events.failedCount == 0,
          "image request lifecycle did not complete cleanly");
}

void testOneRequestPublishesJunctionAndLatestReplayState() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  std::vector<uint32_t> prompt(97);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = index + 1;
  engine.submit(request(5, prompt));
  runUntilIdle(engine);
  while (resources.snapshot().stateCache.entries != 0) {
    require(resources.reclaimOneState(),
            "test could not leave a KV-only shared prefix");
  }

  prompt.resize(161, 777);
  engine.submit(request(6, prompt));
  runUntilIdle(engine);

  const DraftContextPlan &plan = executor.plans.at(6);
  const auto snapshot = engine.snapshot();
  require(plan.boundaries.size() == 3 && plan.boundaries[0].boundary == 96 &&
              plan.boundaries[1].boundary == 160 &&
              plan.boundaries[2].boundary == 161,
          "junction, latest replay state, and active end were not ordered");
  require(executor.snapshots == 3 && snapshot.junctionMaterializations == 1 &&
              snapshot.replayStatePublications == 2 &&
              snapshot.resources.stateCache.entries == 2,
          "one request did not retain both sparse composite states");
}

// A denied snapshot at the latest replay boundary recycles the least recently
// used unprotected state, which is an older unrelated state pushed out of the
// conversation-ends ring, not the junction state this same request published
// one command earlier.
void testLatestReplayDenialRecyclesOlderStateNotTheJunction() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  std::vector<uint32_t> prompt(97);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = index + 1;
  engine.submit(request(7, prompt));
  runUntilIdle(engine);
  while (resources.snapshot().stateCache.entries != 0) {
    require(resources.reclaimOneState(),
            "test could not remove the old composite state");
  }
  engine.submit(request(70, std::vector<uint32_t>(65, 7000)));
  runUntilIdle(engine);
  pushOutConversationEnds(engine, 7100);
  const uint32_t ringEnds = model::ExecutionLimits::maximumBatchWidth;
  const auto older = resources.snapshot();
  require(older.stateCache.entries == 1 + ringEnds &&
              older.stateCache.evictions == 1,
          "older unrelated state was not left in the cache");

  prompt.resize(161, 888);
  engine.submit(request(8, prompt));
  require(engine.tick(1) && engine.tick(2),
          "request did not publish its junction state");
  require(engine.snapshot().junctionMaterializations == 1 &&
              resources.snapshot().stateCache.entries == 2 + ringEnds,
          "lazy junction was not materialized in the first command");
  executor.deniedSnapshots = 1;
  runUntilIdle(engine);

  const DraftContextPlan &plan = executor.plans.at(8);
  const auto snapshot = engine.snapshot();
  require(plan.boundaries.size() == 3 && plan.boundaries[0].boundary == 96 &&
              plan.boundaries[1].boundary == 160 &&
              plan.boundaries[2].boundary == 161,
          "boundaries were not armed without reservation");
  require(executor.deniedSnapshots == 0 &&
              snapshot.recycledStatePublications == 1 &&
              snapshot.replayStatePublications == 3 + ringEnds &&
              snapshot.replayStatePublicationFailures == 0 &&
              snapshot.resources.stateCache.evictions == 2 &&
              snapshot.resources.stateCache.entries == 2 + ringEnds,
          "latest-state denial did not recycle exactly one older state");

  // The junction state survived the recycle: the original prompt resumes
  // from it, while the recycled prompt is back to a KV-only junction.
  const uint32_t restored = executor.restored;
  prompt.resize(97);
  engine.submit(request(9, prompt));
  runUntilIdle(engine);
  require(executor.restored == restored + 96 &&
              events.starts.back() ==
                  std::pair<EngineCacheStatus, uint32_t>{
                      EngineCacheStatus::PrefixHit, 96},
          "latest-state denial discarded the independent junction state");
  engine.submit(request(71, std::vector<uint32_t>(65, 7000)));
  runUntilIdle(engine);
  require(events.starts.back().first == EngineCacheStatus::Miss &&
              engine.snapshot().junctionMaterializations == 2,
          "recycled state was not the older unrelated one");
}

void testCancellationAfterJunctionDiscardsLaterState() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  std::vector<uint32_t> prompt(97);
  for (uint32_t index = 0; index < prompt.size(); ++index)
    prompt[index] = index + 1;
  engine.submit(request(9, prompt));
  runUntilIdle(engine);
  while (resources.snapshot().stateCache.entries != 0) {
    require(resources.reclaimOneState(),
            "test could not remove the old composite state");
  }

  prompt.resize(161, 999);
  engine.submit(request(10, prompt));
  require(engine.tick(1) && engine.tick(2),
          "request did not publish its first sparse state");
  const auto published = engine.snapshot();
  require(published.resources.stateCache.entries == 1 &&
              published.junctionMaterializations == 1 &&
              executor.snapshotAttempts == 2,
          "junction publication did not land in the first command");
  engine.cancel(10);
  runUntilIdle(engine);
  // The armed 160 boundary is dropped with the lane: no snapshot, no
  // failure counted, and the junction state stays cached.
  const auto cancelled = engine.snapshot();
  require(executor.snapshotAttempts == 2 &&
              cancelled.resources.stateCache.entries == 1 &&
              cancelled.resources.stateCache.publications == 2 &&
              cancelled.replayStatePublications == 1 &&
              cancelled.replayStatePublicationFailures == 0 &&
              cancelled.cancelled == 1 && executor.requests.empty() &&
              cancelled.resources.activeRequests == 0,
          "cancellation leaked or removed the wrong sparse state");
}

// Nothing is reserved at admission, so a boundary the model cannot snapshot
// costs exactly one attempt: the failure is counted under its purpose, the
// request is served normally, and with an empty cache nothing is recycled.
void testDeniedSnapshotCostsOnlyThatAttempt() {
  Backing backing(16);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  executor.deniedSnapshots = 100;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  std::vector<uint32_t> prompt(65, 7);

  engine.submit(request(3, prompt));
  runUntilIdle(engine);
  const DraftContextPlan &plan = executor.plans.at(3);
  require(plan.boundaries.size() == 2 && plan.boundaries[0].boundary == 64 &&
              plan.boundaries[1].boundary == 65 &&
              plan.draftContextRows() == prompt.size(),
          "replay boundary was not armed without reservation");
  require(executor.snapshotAttempts == 1 && executor.snapshots == 0 &&
              engine.snapshot().replayStatePublicationFailures == 1 &&
              engine.snapshot().junctionMaterializationFailures == 0 &&
              engine.snapshot().recycledStatePublications == 0 &&
              resources.snapshot().stateCache.evictions == 0 &&
              events.completedCount == 1,
          "latest replay-state denial was counted incorrectly");

  // The first pass left a usable KV chain but no state. The second lookup is
  // therefore a real lazy junction; its denied snapshot is counted as such
  // while the request replays through the KV prefix normally.
  engine.submit(request(4, prompt));
  runUntilIdle(engine);
  require(executor.snapshotAttempts == 2 && executor.snapshots == 0 &&
              engine.snapshot().junctionMaterializationFailures == 1 &&
              engine.snapshot().replayStatePublicationFailures == 1 &&
              engine.snapshot().checkpointPublicationFailures == 0 &&
              engine.snapshot().checkpointPublicationFailureCauses ==
                  FailureCauses{} &&
              events.starts.back().first == EngineCacheStatus::Miss &&
              executor.prefillRows == 130 && events.completedCount == 2 &&
              events.failedCount == 0,
          "denied lazy junction failed the request, was not counted, or "
          "counted a checkpoint failure cause");

  // The denial is transient: the next lane over the same prefix lands it.
  executor.deniedSnapshots = 0;
  engine.submit(request(5, prompt));
  runUntilIdle(engine);
  require(executor.snapshots == 1 &&
              engine.snapshot().junctionMaterializations == 1 &&
              resources.snapshot().stateCache.entries == 1,
          "junction was not materialized once snapshots were possible");
}

// A snapshot the model denies once at a replay boundary lands by recycling
// the least recently used cached state: one eviction, the same number of
// entries, and the recycled slot now holds the new lane's state.
void testDeniedSnapshotRecyclesLruStateAndRetries() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> older(65, 1);
  const std::vector<uint32_t> newer(65, 2);
  engine.submit(request(1, older));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  require(cached.stateCache.entries == 1 && cached.stateCache.evictions == 0,
          "recycle fixture did not cache the older state");

  executor.deniedSnapshots = 1;
  engine.submit(request(2, newer));
  runUntilIdle(engine);
  const auto recycled = engine.snapshot();
  require(executor.snapshotAttempts == 3 && executor.snapshots == 2 &&
              recycled.recycledStatePublications == 1 &&
              recycled.replayStatePublications == 2 &&
              recycled.replayStatePublicationFailures == 0 &&
              events.completedCount == 2,
          "denied snapshot was not retried after recycling a state");
  require(recycled.resources.stateCache.evictions == 1 &&
              recycled.resources.stateCache.entries == 1 &&
              recycled.resources.stateCache.publications == 2 &&
              recycled.resources.stateCache.bytes == cached.stateCache.bytes,
          "recycling changed the cached state footprint");

  // The surviving state belongs to the new lane: re-sending its prompt is a
  // state hit, while the older prompt is back to a KV-only junction.
  engine.submit(request(3, newer));
  runUntilIdle(engine);
  require(executor.restored == 64 &&
              events.starts.back() ==
                  std::pair<EngineCacheStatus, uint32_t>{
                      EngineCacheStatus::PrefixHit, 64},
          "recycled slot does not hold the new lane's state");
  engine.submit(request(4, older));
  runUntilIdle(engine);
  require(executor.restored == 64 &&
              events.starts.back().first == EngineCacheStatus::Miss &&
              engine.snapshot().junctionMaterializations == 1,
          "recycled state was not the least recently used one");
}

// When the retry after recycling is denied as well, the boundary fails and
// the engine has paid exactly one cached state for it. Persistent denial
// never drains the rest of the cache.
void testPersistentSnapshotDenialRecyclesAtMostOneState() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  for (uint64_t id : {1, 2, 3}) {
    engine.submit(request(id, std::vector<uint32_t>(65, id)));
    runUntilIdle(engine);
  }
  require(resources.snapshot().stateCache.entries == 3,
          "persistent-denial fixture did not cache three states");

  executor.deniedSnapshots = 100;
  engine.submit(request(4, std::vector<uint32_t>(65, 4)));
  runUntilIdle(engine);
  const auto after = engine.snapshot();
  require(executor.snapshotAttempts == 5 && executor.snapshots == 3 &&
              executor.deniedSnapshots == 98 &&
              after.replayStatePublicationFailures == 1 &&
              after.recycledStatePublications == 0 &&
              events.completedCount == 4 && events.failedCount == 0,
          "persistently denied snapshot was retried more than once");
  require(after.resources.stateCache.evictions == 1 &&
              after.resources.stateCache.entries == 2,
          "persistent snapshot denial recycled more than one cached state");
}

void testLongSuffixSkipsDraftRestore() {
  Backing backing(256);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  std::vector<uint32_t> prefix(65);
  for (uint32_t index = 0; index < prefix.size(); ++index) {
    prefix[index] = index + 1;
  }
  engine.submit(request(10, prefix));
  runUntilIdle(engine);

  std::vector<uint32_t> extended = prefix;
  extended.resize(4097, 99);
  engine.submit(request(11, extended));
  runUntilIdle(engine);
  require(!executor.restoredDraft &&
              executor.plans.at(11).draftStateRestoreSkipped,
          "long suffix copied a draft ring that its final window overwrites");
}

// One 65-token prefix cached as a full state at 32 and a target-only state
// at 64, the deepest one a lookup of any extension of it finds.
struct TargetOnlyFixture final {
  TargetOnlyFixture()
      : pool(backing), resources(pool, CacheNamespace{}), executor(1),
        engine({}, resources, executor, events) {
    for (uint32_t index = 0; index < prefix.size(); ++index)
      prefix[index] = index + 1;
    engine.submit(request(1, extended(33)));
    runUntilIdle(engine);
    executor.targetOnlyAtBoundary = 64;
    engine.submit(request(2, prefix));
    runUntilIdle(engine);
    executor.targetOnlyAtBoundary.reset();
    const auto lookup = resources.lookup(extended(4097));
    require(lookup.resumeBoundary() == 64 &&
                !lookup.state->state()->carriesDraft(),
            "fixture did not cache a target-only state at 64");
  }

  // The prefix truncated or padded to length tokens.
  std::vector<uint32_t> extended(uint32_t length) const {
    std::vector<uint32_t> result(prefix.begin(),
                                 prefix.begin() +
                                     std::min<size_t>(length, prefix.size()));
    result.resize(length, 99);
    return result;
  }

  std::vector<uint32_t> prefix = std::vector<uint32_t>(65);
  Backing backing{512};
  KvPool pool;
  engine::Cache resources;
  Executor executor;
  Events events;
  engine::Engine engine;
};

// The first planned boundary lies a whole draft window past the target-only
// state: it restores without draft and plans with the target-only input.
void testTargetOnlyStateRestoresWithoutDraft() {
  TargetOnlyFixture fixture;

  fixture.engine.submit(request(3, fixture.extended(4097)));
  runUntilIdle(fixture.engine);

  const DraftContextPlan &plan = fixture.executor.plans.at(3);
  require(fixture.executor.restoredLengths.at(3) == 64 &&
              !fixture.executor.restoredDraft && plan.targetOnlyRestore &&
              plan.draftStateRestoreSkipped &&
              fixture.events.failedCount == 0,
          "usable target-only state was not restored without its draft");
}

// A shorter remainder would continue draft rows the state lacks: admission
// walks past it to the full state below before anything is restored.
void testShortRemainderWalksPastTargetOnlyState() {
  TargetOnlyFixture fixture;

  fixture.engine.submit(request(3, fixture.extended(200)));
  runUntilIdle(fixture.engine);

  const DraftContextPlan &plan = fixture.executor.plans.at(3);
  require(fixture.executor.restoredLengths.at(3) == 32 &&
              fixture.executor.restoredDraft && !plan.targetOnlyRestore &&
              fixture.events.failedCount == 0 &&
              fixture.resources.snapshot().stateCache.pinned == 0,
          "short remainder restored a target-only state");
}

// A queued peer's junction is a planned boundary too: one too close to the
// target-only state makes it unusable although the replay point is far.
void testQueuedPeerJunctionWalksPastTargetOnlyState() {
  TargetOnlyFixture fixture;
  const std::vector<uint32_t> prompt = fixture.extended(4097);
  std::vector<uint32_t> peer = prompt;
  std::fill(peer.begin() + 1024, peer.end(), 77);

  fixture.engine.submit(request(3, prompt));
  fixture.engine.submit(request(4, peer));
  runUntilIdle(fixture.engine);

  require(fixture.events.startIds.size() >= 4 &&
              fixture.events.startIds[2] == 3,
          "fixture did not admit the request ahead of its queued peer");
  const DraftContextPlan &plan = fixture.executor.plans.at(3);
  require(fixture.executor.restoredLengths.at(3) == 32 &&
              !plan.targetOnlyRestore && plan.boundaries.front().boundary == 64 &&
              fixture.events.failedCount == 0 &&
              fixture.events.completedCount == 4,
          "a queued peer's junction did not reject the target-only state");
}

// A lane cancelled while the command that ends at its armed boundary is in
// flight publishes nothing when that command drains: no snapshot is taken,
// no failure is counted, and the blocks committed earlier stay cached.
void testCancellationInFlightAtBoundaryPublishesNoState() {
  Backing backing(256);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  std::vector<uint32_t> prompt(4097, 11);

  engine.submit(request(4, prompt));
  require(engine.tick(1) && engine.tick(2),
          "long request did not complete its first prefill command");
  require(resources.snapshot().kvCache.blocks == 64 &&
              executor.snapshotAttempts == 0,
          "first prefill command changed the state cache");
  require(engine.tick(3) && engine.commandInFlight(),
          "second prefill command was not launched");
  engine.cancel(4);
  require(engine.commandInFlight() && executor.requests.size() == 1,
          "in-flight cancellation released resources owned by the command");
  runUntilIdle(engine);
  const auto snapshot = engine.snapshot();
  require(executor.snapshotAttempts == 0 &&
              snapshot.resources.stateCache.entries == 0 &&
              snapshot.resources.stateCache.publications == 0 &&
              snapshot.replayStatePublications == 0 &&
              snapshot.replayStatePublicationFailures == 0 &&
              snapshot.cancelled == 1,
          "cancelled command snapshotted or counted its boundary");
  require(snapshot.resources.kvCache.blocks == 64 &&
              snapshot.resources.pool.pagesActive == 0 &&
              snapshot.resources.activeRequests == 0 &&
              executor.requests.empty(),
          "cancellation leaked active resources or dropped committed KV");
}

void testActiveCellGrowthReclaimsCachedStateAndRetries() {
  for (bool hostPressure : {false, true}) {
    Backing backing(16);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    Events events;
    EngineConfig config;
    config.growthPaused = [] { return false; };
    engine::Engine engine(config, resources, executor, events);

    engine.submit(request(20, std::vector<uint32_t>(65, 7)));
    runUntilIdle(engine);
    const auto cached = resources.snapshot();
    require(cached.stateCache.entries == 1,
            "state-reclaim setup did not publish a composite state");

    if (hostPressure) {
      backing.growthBlocked = true;
      executor.physicalGrowthBlocked = &backing.growthBlocked;
      executor.beginGrowthBlocked = [&] { return backing.growthBlocked; };
      executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
      executor.reclaimableIdleStateBytes = 350'224'384;
    } else {
      executor.deniedBegins = 1;
    }
    const uint32_t attempts = executor.beginAttempts;
    engine.submit(request(21, {8}));
    runUntilIdle(engine);
    const auto after = resources.snapshot();
    require(executor.beginAttempts == attempts + 2 &&
                events.completedCount == 2,
            "active-cell growth did not reclaim memory and retry");
    if (hostPressure) {
      require(executor.reclaimedIdleStateBytes == 350'224'384 &&
                  !backing.growthBlocked &&
                  after.stateCache.entries == cached.stateCache.entries &&
                  after.stateCache.evictions == cached.stateCache.evictions,
              "host-pressure state admission evicted cache or ignored idle memory");
    } else {
      require(after.stateCache.entries == 0,
              "active-cell growth did not reclaim cached state");
    }
  }
}

void testKvGrowthReclaimsIdleStateBeforeCache() {
  for (auto failure : {metal::AllocationFailure::Capacity,
                       metal::AllocationFailure::HostPressure}) {
    Backing backing(12);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    Events events;
    EngineConfig config;
    config.growthPaused = [] { return false; };
    engine::Engine engine(config, resources, executor, events);

    engine.submit(request(25, std::vector<uint32_t>(65, 25)));
    runUntilIdle(engine);
    const auto cached = resources.snapshot();
    require(cached.stateCache.entries == 1,
            "idle-state reclaim setup did not retain cache state");

    backing.growthBlocked = true;
    backing.allocationFailure = failure;
    executor.physicalGrowthBlocked = &backing.growthBlocked;
    executor.unblockGrowthOnSuspend = false;
    executor.reclaimableIdleStateBytes = 350'224'384;
    engine.submit(request(26, std::vector<uint32_t>(161, 26)));
    runUntilIdle(engine);

    const auto after = resources.snapshot();
    require(executor.reclaimedIdleStateBytes == 350'224'384 &&
                !backing.growthBlocked && events.completedCount == 2 &&
                engine.snapshot().resourceSuspensions == 0,
            "KV growth did not reclaim model-owned idle state before suspension");
    // The growth denial neither evicts the cached state nor touches the growing
    // lane's own armed boundary: its replay state still lands.
    require(after.stateCache.evictions == cached.stateCache.evictions &&
                after.stateCache.entries == cached.stateCache.entries + 1 &&
                engine.snapshot().replayStatePublications == 2 &&
                engine.snapshot().recycledStatePublications == 0,
            "KV growth evicted useful composite state before idle state memory");
  }
}

// Two lanes prefill toward their replay boundaries in one command. The
// physical KV growth denial the first lane meets is answered by reclaiming
// idle model state, never by dropping any lane's armed boundary: both replay
// states land.
void testKvGrowthDenialKeepsEveryLaneReplayState() {
  Backing backing(16);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  backing.growthBlocked = true;
  executor.physicalGrowthBlocked = &backing.growthBlocked;
  executor.reclaimableIdleStateBytes = 1U << 20;

  engine.submit(request(300, std::vector<uint32_t>(161, 300)));
  engine.submit(request(301, std::vector<uint32_t>(161, 301)));
  runUntilIdle(engine);
  const auto after = engine.snapshot();
  require(executor.reclaimedIdleStateBytes == (1U << 20) &&
              !backing.growthBlocked && executor.suspensions == 0 &&
              events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "physical growth denial suspended or failed a lane");
  require(!executor.prefillWidths.empty() && executor.prefillWidths[0] == 2,
          "both lanes did not prefill to their boundaries in one command");
  require(executor.snapshots == 2 && after.replayStatePublications == 2 &&
              after.replayStatePublicationFailures == 0 &&
              after.recycledStatePublications == 0 &&
              after.resources.stateCache.entries == 2 &&
              after.resources.stateCache.evictions == 0,
          "KV growth denial cost a lane its replay state");
}

void testPressureReclaimRespectsStateLifetimes() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  // Cancelled past its prompt end, the request leaves its state cached but
  // outside the conversation-ends ring, so Warning reclaim may take it.
  cancelAfterPromptEnd(engine, resources, 27, std::vector<uint32_t>(65, 27));
  const auto cached = resources.snapshot();
  require(cached.stateCache.entries == 1 && cached.kvCache.blocks == 2,
          "pressure setup did not retain KV and composite state");

  // A shrink that nothing is waiting for stops at the resume point. Freeing
  // its cell gains the host a little; the next request pays a full replay.
  static_cast<void>(engine.reclaimMemory({.reclaimEmptyKvExtents = true,
                                          .targetBytes = 64,
                                          .keepResumePoint = true}));
  require(resources.snapshot().stateCache.entries == 1,
          "a speculative shrink discarded the only resume point");

  executor.reclaimableIdleStateBytes = 128;
  const uint64_t first = engine.reclaimMemory(
      {.reclaimEmptyKvExtents = true, .targetBytes = 64});
  require(first >= 128 && executor.reclaimedIdleStateBytes == 128,
          "pressure reclaim did not release idle active-state backing first");
  require(
      resources.snapshot().stateCache.entries == 1,
      "pressure reclaim evicted a cached state before satisfying its target");

  // Drain any initially resident but unused test extents, then prove cached
  // state is the next lifecycle selected while its parent KV remains usable.
  static_cast<void>(engine.reclaimMemory({.reclaimEmptyKvExtents = true}));
  require(engine.reclaimMemory({.reclaimEmptyKvExtents = true,
                                .targetBytes = 64}) >= 64,
          "pressure reclaim did not release immutable cached state");
  const auto stateEvicted = resources.snapshot();
  require(stateEvicted.stateCache.entries == 0 &&
              stateEvicted.kvCache.blocks == 2,
          "cached-state eviction incorrectly removed target KV");

  static_cast<void>(
      engine.reclaimMemory({.reclaimEmptyKvExtents = true,
                            .evictAllUnpinnedPrefixes = true,
                            .targetBytes = std::numeric_limits<uint64_t>::max()}));
  const auto critical = resources.snapshot();
  require(critical.stateCache.entries == 0 && critical.kvCache.blocks == 0 &&
              critical.pool.residentBackingBytes == 0,
          "critical pressure left evictable cached state or KV backing");
}

// Maps the first extents before the pool sees the backing, so they start out
// resident and empty.
Backing &mapLeadingPages(Backing &backing, uint32_t pages) {
  for (uint32_t page = 0; page < pages; page += 4)
    static_cast<void>(backing.ensureResident(page));
  return backing;
}

// Runs a long prompt's first prefill chunk, which fills sixteen extents and
// leaves the two further mapped extents empty, then applies one directive at
// the command-free point before the next chunk.
KvPoolSnapshot reclaimBesideRunningLane(
    std::initializer_list<MemoryReclaimDirective> directives,
    bool &deferredAfterFirst) {
  Backing backing(256);
  KvPool pool(mapLeadingPages(backing, 72));
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  engine.submit(request(40, std::vector<uint32_t>(4097, 40)));
  for (uint32_t step = 1;
       step <= 8 && (resources.snapshot().pool.pagesActive == 0 ||
                     engine.commandInFlight());
       ++step) {
    static_cast<void>(engine.tick(step));
  }
  const KvPoolSnapshot running = resources.snapshot().pool;
  require(running.pagesActive == 64 && running.pagesFreeResident == 8 &&
              running.reclaimableExtents == 2 && !engine.commandInFlight(),
          "runway setup did not leave a running lane beside two empty extents");

  // Pace the release so a kept runway is distinguishable from deferred work.
  backing.deferRelease = true;
  bool first = true;
  for (const MemoryReclaimDirective &directive : directives) {
    static_cast<void>(engine.reclaimMemory(directive));
    if (first)
      deferredAfterFirst = engine.reclaimDeferred();
    first = false;
    backing.releasePending = false;
  }
  const KvPoolSnapshot after = resources.snapshot().pool;
  engine.cancel(40);
  runUntilIdle(engine);
  return after;
}

// Queues a request for memory with no resident lane: a host-pressure begin
// denial while growth is paused, as in QueuedPrefixFixture, beside two empty
// mapped extents. Applies one directive and returns the free resident pages.
uint32_t reclaimBesideQueuedRequest(const MemoryReclaimDirective &directive) {
  bool blocked = true;
  Backing backing(16);
  KvPool pool(mapLeadingPages(backing, 8));
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return blocked; };
  Events events;
  EngineConfig config;
  config.growthPaused = [&] { return blocked; };
  engine::Engine engine(config, resources, executor, events);
  engine.submit(request(41, {41}));
  static_cast<void>(engine.tick(1));
  require(engine.resourceWaitSnapshot(1).memory == 1 &&
              resources.snapshot().pool.pagesActive == 0,
          "runway setup did not queue a request beside no resident lane");

  static_cast<void>(engine.reclaimMemory(directive));
  const uint32_t freeResident = resources.snapshot().pool.pagesFreeResident;
  engine.cancel(41);
  runUntilIdle(engine);
  return freeResident;
}

void testPressureReclaimKeepsRunwayOnlyForWarning() {
  const MemoryReclaimDirective warning{.reclaimEmptyKvExtents = true};
  const MemoryReclaimDirective critical{
      .reclaimEmptyKvExtents = true,
      .evictAllUnpinnedPrefixes = true,
      .targetBytes = std::numeric_limits<uint64_t>::max()};

  // The second pass runs once the first release completed: nothing paces it.
  bool deferred = true;
  const KvPoolSnapshot warned =
      reclaimBesideRunningLane({warning, warning}, deferred);
  require(warned.pagesFreeResident == 4 && warned.reclaimableExtents == 1,
          "warning reclaim unmapped the runway a running lane's chunk needs");
  require(!deferred, "the kept runway re-ran the pressure pass as deferred");

  const KvPoolSnapshot criticalAfterWarning =
      reclaimBesideRunningLane({warning, critical}, deferred);
  require(criticalAfterWarning.pagesFreeResident == 0 &&
              criticalAfterWarning.reclaimableExtents == 0,
          "critical reclaim left an empty extent mapped");

  // A queued request takes the runway next, even with no resident lane.
  require(reclaimBesideQueuedRequest(warning) > 0,
          "warning reclaim unmapped the runway a queued request needs");
  require(reclaimBesideQueuedRequest(critical) == 0,
          "critical reclaim kept a runway for a queued request");

  // With no running lane and no waiting request the runway serves nothing;
  // warning unmaps it too.
  Backing backing(16);
  KvPool pool(mapLeadingPages(backing, 8));
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  static_cast<void>(engine.reclaimMemory(warning));
  require(resources.snapshot().pool.pagesFreeResident == 0,
          "warning reclaim kept a runway no request needs");
}

void testConcurrencyLimitDoesNotEvictCache() {
  Backing backing(256);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  engine.submit(request(22, std::vector<uint32_t>(65, 22)));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  require(cached.stateCache.entries == 1,
          "concurrency-limit setup did not retain a cache state");

  engine.submit(request(23, std::vector<uint32_t>(4097, 23)));
  require(engine.tick(1) && engine.tick(2),
          "resident request did not start long prefill");
  engine.submit(request(24, {24}));
  require(engine.tick(3), "saturated engine made no resident progress");
  const auto waiting = engine.resourceWaitSnapshot(13.0);
  require(waiting.concurrency == 1 && waiting.memory == 0 &&
              waiting.suspended == 0 && waiting.oldestWaitMilliseconds == 10.0,
          "concurrency wait diagnostics were confused with memory pressure");

  const auto saturated = resources.snapshot();
  require(saturated.stateCache.entries == cached.stateCache.entries &&
              saturated.stateCache.evictions == cached.stateCache.evictions,
          "active-cell saturation was mistaken for memory pressure");

  engine.cancel(23);
  engine.cancel(24);
  runUntilIdle(engine);
}

void testHostPressureDoesNotDrainCacheOnStateAdmission() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  MemoryPressure pressure = MemoryPressure::Normal;
  EngineConfig config;
  config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
  engine::Engine engine(config, resources, executor, events);
  engine.submit(request(200, std::vector<uint32_t>(65, 200)));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  const uint64_t coldMisses = engine.snapshot().coldMisses;
  const uint32_t attempts = executor.beginAttempts;

  pressure = MemoryPressure::Warning;
  executor.deniedBegins = 100;
  engine.submit(request(201, {201}));
  static_cast<void>(engine.tick(20.0));
  static_cast<void>(engine.tick(50.0));
  require(executor.beginAttempts == attempts + 1 &&
              resources.snapshot().stateCache.entries ==
                  cached.stateCache.entries &&
              resources.snapshot().kvCache.blocks == cached.kvCache.blocks &&
              resources.snapshot().lookup.lookups == cached.lookup.lookups &&
              engine.snapshot().coldMisses == coldMisses &&
              engine.snapshot().scheduler.waitingResources == 1,
          "paused state admission drained the cache or retried without backoff");

  pressure = MemoryPressure::Normal;
  executor.deniedBegins = 0;
  for (double now = 120.0; now < 140.0 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));
  require(engine.idle() && events.completedCount == 2 &&
              resources.snapshot().lookup.lookups == cached.lookup.lookups + 1 &&
              engine.snapshot().coldMisses == coldMisses + 1 &&
              events.capacityExhaustedCount == 0 && events.failedCount == 0,
          "state admission did not recover after host pressure cleared");
}

// Host pressure pauses foreground cache eviction for KV growth, but landing a
// denied snapshot by recycling one cached state is memory-neutral and still
// happens: the state footprint does not grow, and KV is untouched.
void testHostPressureStillRecyclesLruStateForDeniedSnapshot() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  MemoryPressure pressure = MemoryPressure::Normal;
  EngineConfig config;
  config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
  engine::Engine engine(config, resources, executor, events);
  engine.submit(request(210, std::vector<uint32_t>(65, 210)));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  const uint32_t attempts = executor.snapshotAttempts;

  pressure = MemoryPressure::Warning;
  executor.deniedSnapshots = 1;
  engine.submit(request(211, std::vector<uint32_t>(65, 211)));
  runUntilIdle(engine);
  const auto after = engine.snapshot();
  require(engine.idle() && events.completedCount == 2 &&
              executor.snapshotAttempts == attempts + 2 &&
              after.recycledStatePublications == 1 &&
              after.replayStatePublications == 2 &&
              after.replayStatePublicationFailures == 0,
          "denied snapshot under host pressure was not landed by recycling");
  require(after.resources.stateCache.evictions ==
                  cached.stateCache.evictions + 1 &&
              after.resources.stateCache.entries == cached.stateCache.entries &&
              after.resources.stateCache.bytes == cached.stateCache.bytes &&
              after.resources.kvCache.blocks == cached.kvCache.blocks + 2,
          "recycling under host pressure grew state or drained KV cache");
}

// A lone request short of pages under host pressure takes idle cached pages
// that are already resident instead of being suspended and replaying its
// prefix later. Reuse only happens when it can cover the shortfall.
void testSingletonHostPressureReusesIdleCacheInsteadOfSuspending() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  MemoryPressure pressure = MemoryPressure::Normal;
  EngineConfig config;
  config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
  engine::Engine engine(config, resources, executor, events);
  engine.submit(request(230, std::vector<uint32_t>(65, 230)));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  require(cached.kvCache.blocks >= 1 && cached.pool.pagesPrefix >= 2 &&
              cached.pool.pagesFreeResident + cached.pool.pagesPrefix >= 3,
          "idle-cache reuse fixture geometry changed");

  pressure = MemoryPressure::Warning;
  backing.growthBlocked = true;
  engine.submit(request(231, std::vector<uint32_t>(65, 231)));
  runUntilIdle(engine);
  require(engine.idle() && events.completedCount == 2 &&
              events.failedCount == 0 && events.capacityExhaustedCount == 0,
          "lone request under host pressure did not complete on idle cache");
  require(executor.suspensions == 0 &&
              engine.snapshot().resourceSuspensions == 0 &&
              backing.growthAttempts > 0,
          "lone request was suspended although idle cached pages could serve it");
  require(resources.snapshot().pool.pagesResident == cached.pool.pagesResident,
          "idle-cache reuse grew the resident footprint");
}

void testSingletonHostPressureWaitRecoversOrTerminates() {
  for (uint32_t outcome = 0; outcome < 4; ++outcome) {
    Backing backing(32);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    Events events;
    MemoryPressure pressure = MemoryPressure::Normal;
    EngineConfig config;
    config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
    config.resourceWaitTimeoutMilliseconds = outcome == 3 ? 100.0 : 30000.0;
    engine::Engine engine(config, resources, executor, events);
    engine.submit(request(220, std::vector<uint32_t>(65, 220)));
    runUntilIdle(engine);
    const auto cached = resources.snapshot();

    pressure = MemoryPressure::Warning;
    backing.growthBlocked = true;
    auto value = request(221, std::vector<uint32_t>(161, 221));
    value.deadlineMilliseconds = 300.0;
    engine.submit(std::move(value));
    static_cast<void>(engine.tick(20.0));
    static_cast<void>(engine.tick(21.0));
    // Suspension drops the lane's armed boundary silently: it is neither a
    // snapshot attempt nor a publication failure.
    require(executor.suspensions == 1 && events.capacityExhaustedCount == 0 &&
                resources.snapshot().kvCache.blocks == cached.kvCache.blocks &&
                resources.snapshot().stateCache.evictions ==
                    cached.stateCache.evictions &&
                executor.snapshotAttempts == 1 &&
                engine.snapshot().replayStatePublicationFailures == 0,
            "singleton host pressure killed active work or drained its cache");
    const uint32_t attempts = executor.resumeAttempts;
    static_cast<void>(engine.tick(50.0));
    require(executor.resumeAttempts == attempts,
            "suspended host-pressure request ignored resource backoff");
    const auto waiting = engine.resourceWaitSnapshot(50.0);
    require(waiting.memory == 1 && waiting.concurrency == 0 &&
                waiting.suspended == 1 && waiting.oldestWaitMilliseconds >= 29.0,
            "resource diagnostics lost suspension reason or original wait time");

    if (outcome == 0) {
      pressure = MemoryPressure::Normal;
      backing.growthBlocked = false;
      for (double now = 121.0; now < 140.0 && !engine.idle(); ++now)
        static_cast<void>(engine.tick(now));
      // The resumed lane re-arms its replay boundary and publishes normally.
      require(events.completedCount == 2 && events.failedCount == 0 &&
                  engine.snapshot().replayStatePublications == 2 &&
                  engine.snapshot().replayStatePublicationFailures == 0,
              "singleton request did not recover after pressure cleared");
    } else if (outcome == 1) {
      engine.cancel(221);
      static_cast<void>(engine.tick(60.0));
      require(engine.snapshot().cancelled == 1,
              "host-pressure wait ignored cancellation");
    } else if (outcome == 3) {
      static_cast<void>(engine.tick(150.0));
      require(events.failures == std::vector<std::string>{"resource_timeout"} &&
                  events.failureDetails.back().second,
              "persistent memory pressure did not fail with a retryable timeout");
    } else {
      static_cast<void>(engine.tick(301.0));
      require(events.completedCount + events.failedCount == 2,
              "host-pressure wait ignored deadline");
    }
    require(engine.idle() && executor.requests.empty() &&
                resources.snapshot().activeRequests == 0 &&
                resources.snapshot().pool.pagesActive == 0 &&
                events.capacityExhaustedCount == 0 &&
                resources.snapshot().stateCache.entries ==
                    cached.stateCache.entries + (outcome == 0 ? 1U : 0U),
            "host-pressure wait leaked request state or active KV");
    const auto cleared = engine.resourceWaitSnapshot(400.0);
    require(cleared.memory == 0 && cleared.concurrency == 0 &&
                cleared.suspended == 0 && cleared.oldestWaitMilliseconds == 0.0,
            "terminal work retained resource wait diagnostics");
  }
}

// A sole resident decode lane on a fresh cache. Its 100-token prompt leaves
// the first extent's pages full once 21 tokens are decoded, so the next
// step's verify rows need an extent that blocked growth denies for host
// pressure.
struct SoleDecodeLane final {
  static constexpr uint64_t laneId = 270;
  static constexpr uint32_t tokensBeforeDenial = 21;

  Backing backing{32};
  KvPool pool{backing};
  engine::Cache resources{pool, CacheNamespace{}};
  Executor executor{2};
  Events events;
  engine::Engine engine;
  double now = 1;

  explicit SoleDecodeLane(double resourceWaitTimeoutMilliseconds)
      : engine(configured(resourceWaitTimeoutMilliseconds), resources,
               executor, events) {
    backing.allocationFailure = metal::AllocationFailure::HostPressure;
    executor.decodeFinishes = false;
    auto lane = request(laneId, std::vector<uint32_t>(100, 70));
    lane.maxNewTokens = 10'000;
    lane.deadlineMilliseconds = 100'000;
    engine.submit(std::move(lane));
    require(tickUntil([&] { return events.emitted > 0; }),
            "sole decode lane fixture did not start decoding");
  }

  EngineConfig configured(double resourceWaitTimeoutMilliseconds) {
    EngineConfig config;
    config.resourceWaitTimeoutMilliseconds = resourceWaitTimeoutMilliseconds;
    config.growthPaused = [this] { return backing.growthBlocked; };
    return config;
  }

  // Ticks once a millisecond until `done` holds, at most 1000 ticks; `now`
  // is left at the tick that satisfied it.
  bool tickUntil(const std::function<bool()> &done) {
    for (const double last = now + 1000; now < last; ++now) {
      static_cast<void>(engine.tick(now));
      if (done())
        return true;
    }
    return false;
  }

  // Blocks growth and decodes until the lane is first denied, returning that
  // denial's time.
  double decodeUntilBackedOff() {
    backing.growthBlocked = true;
    require(tickUntil([&] {
              return engine.resourceWaitSnapshot(now).memory == 1 ||
                     executor.suspensions != 0;
            }) && executor.suspensions == 0,
            "sole decode lane under host pressure suspended instead of "
            "backing off");
    return now;
  }

  [[nodiscard]] bool resident(uint64_t id) const {
    return executor.requests.at(id).resident;
  }
};

void testSoleDecodeLaneBacksOffUnderHostPressure() {
  SoleDecodeLane fixture(150.0);
  const double denied = fixture.decodeUntilBackedOff();
  const uint32_t emitted = fixture.events.emitted;
  require(emitted == SoleDecodeLane::tokensBeforeDenial,
          "fixture lane was denied at an unexpected position");

  const ResourceWaitSnapshot waiting =
      fixture.engine.resourceWaitSnapshot(denied + 40.0);
  require(waiting.memory == 1 && waiting.suspended == 0 &&
              waiting.oldestWaitMilliseconds == 40.0,
          "backing-off lane was not reported as waiting on memory");
  require(fixture.engine.nextWakeupMilliseconds() == denied + 100.0,
          "next wakeup lost the backing-off lane's retry time");
  for (double now = denied + 1.0; now < denied + 100.0; ++now)
    require(!fixture.engine.tick(now), "back-off skip reported progress");
  require(fixture.events.emitted == emitted &&
              fixture.executor.suspensions == 0 &&
              fixture.resident(SoleDecodeLane::laneId),
          "backing-off lane decoded or left residency before its retry");

  fixture.backing.growthBlocked = false;
  require(fixture.engine.tick(denied + 100.0),
          "backing-off lane did not retry at its retry time");
  fixture.now = denied + 101.0;
  require(fixture.tickUntil([&] { return fixture.events.emitted > emitted; }),
          "lane did not resume decoding when memory returned");
  require(fixture.engine.resourceWaitSnapshot(fixture.now).memory == 0,
          "granted lane still reported a memory wait");

  // Past the first denial's bound: only a reset back-off holds the lane.
  const double deniedAgain = fixture.decodeUntilBackedOff();
  require(deniedAgain > denied + 150.0 &&
              fixture.engine.resourceWaitSnapshot(deniedAgain).
                      oldestWaitMilliseconds == 0.0,
          "a granted lane's back-off was not reset");

  fixture.backing.growthBlocked = false;
  fixture.executor.decodeFinishes = true;
  fixture.now = deniedAgain + 100.0;
  require(fixture.tickUntil([&] { return fixture.engine.idle(); }),
          "backed-off lane did not finish");
  require(fixture.events.completedCount == 1 &&
              fixture.events.failedCount == 0 &&
              fixture.engine.snapshot().resourceSuspensions == 0,
          "backed-off lane failed or was suspended");
}

void testSoleDecodeLaneSuspendsAtTheResourceWaitBound() {
  SoleDecodeLane fixture(100.0);
  const double denied = fixture.decodeUntilBackedOff();

  static_cast<void>(fixture.engine.tick(denied + 50.0));
  require(fixture.executor.suspensions == 0,
          "lane suspended before the resource wait bound");
  static_cast<void>(fixture.engine.tick(denied + 100.0));
  require(fixture.executor.suspensions == 1 &&
              fixture.engine.snapshot().resourceDecodeSuspensions == 1 &&
              !fixture.resident(SoleDecodeLane::laneId),
          "lane was not suspended at the resource wait bound");
  const ResourceWaitSnapshot waiting =
      fixture.engine.resourceWaitSnapshot(denied + 100.0);
  require(waiting.memory == 1 && waiting.suspended == 1 &&
              waiting.oldestWaitMilliseconds == 100.0,
          "suspension lost the back-off's wait time");

  // The suspension takes a fresh wait deadline, not the first denial's.
  static_cast<void>(fixture.engine.tick(denied + 150.0));
  require(fixture.events.failedCount == 0,
          "bound suspension failed on the first denial's wait deadline");
  static_cast<void>(fixture.engine.tick(denied + 201.0));
  require(fixture.events.failures ==
              std::vector<std::string>{"resource_timeout"},
          "bound suspension did not take a fresh resource wait deadline");
}

// A sole resident prefill lane on a fresh cache. Its first dispatch fills
// the first extent's pages at 128 of its 129 prompt tokens, so the last
// prompt token needs an extent that blocked growth denies for host pressure.
struct SolePrefillLane final {
  static constexpr uint64_t laneId = 275;
  static constexpr uint32_t tokensBeforeDenial = 128;

  Backing backing{32};
  KvPool pool{backing};
  engine::Cache resources{pool, CacheNamespace{}};
  Executor executor{1};
  Events events;
  engine::Engine engine;
  double now = 1;
  double denied = 0;

  // Prefills until the first dispatch lands, then blocks growth and ticks
  // until the lane's next chunk is denied, recording that denial's time.
  explicit SolePrefillLane(double resourceWaitTimeoutMilliseconds)
      : engine(configured(resourceWaitTimeoutMilliseconds), resources,
               executor, events) {
    backing.allocationFailure = metal::AllocationFailure::HostPressure;
    engine.submit(request(laneId, std::vector<uint32_t>(129, 75)));
    require(tickUntil([&] { return !executor.prefillWidths.empty(); }) &&
                executor.requests.at(laneId).position == tokensBeforeDenial,
            "sole prefill lane fixture did not prefill its first extent");
    backing.growthBlocked = true;
    require(tickUntil([&] {
              return engine.resourceWaitSnapshot(now).memory == 1 ||
                     executor.suspensions != 0;
            }) && executor.suspensions == 0,
            "sole prefill lane under host pressure suspended instead of "
            "backing off");
    denied = now;
  }

  EngineConfig configured(double resourceWaitTimeoutMilliseconds) {
    EngineConfig config;
    config.resourceWaitTimeoutMilliseconds = resourceWaitTimeoutMilliseconds;
    config.growthPaused = [this] { return backing.growthBlocked; };
    return config;
  }

  // Ticks once a millisecond until `done` holds, at most 1000 ticks; `now`
  // is left at the tick that satisfied it.
  bool tickUntil(const std::function<bool()> &done) {
    for (const double last = now + 1000; now < last; ++now) {
      static_cast<void>(engine.tick(now));
      if (done())
        return true;
    }
    return false;
  }

  [[nodiscard]] bool resident() const {
    return executor.requests.at(laneId).resident;
  }
};

void testSolePrefillLaneBacksOffUnderHostPressure() {
  SolePrefillLane fixture(150.0);
  const double denied = fixture.denied;

  const ResourceWaitSnapshot waiting =
      fixture.engine.resourceWaitSnapshot(denied + 40.0);
  require(waiting.memory == 1 && waiting.suspended == 0 &&
              waiting.oldestWaitMilliseconds == 40.0,
          "backing-off prefill lane was not reported as waiting on memory");
  for (double now = denied + 1.0; now < denied + 100.0; ++now)
    require(!fixture.engine.tick(now),
            "prefill back-off skip reported progress");
  require(fixture.executor.prefillWidths.size() == 1 &&
              fixture.executor.suspensions == 0 && fixture.resident(),
          "backing-off prefill lane ran or left residency before its retry");

  fixture.backing.growthBlocked = false;
  require(fixture.engine.tick(denied + 100.0),
          "backing-off prefill lane did not retry at its retry time");
  fixture.now = denied + 101.0;
  require(fixture.tickUntil([&] { return fixture.engine.idle(); }),
          "backed-off prefill lane did not finish");
  require(fixture.events.completedCount == 1 &&
              fixture.events.failedCount == 0 &&
              fixture.engine.snapshot().resourceSuspensions == 0 &&
              fixture.engine.snapshot().resourceReplayTokens == 0,
          "backed-off prefill lane failed, suspended or replayed");
}

void testSolePrefillLaneSuspendsAtTheResourceWaitBound() {
  SolePrefillLane fixture(100.0);
  const double denied = fixture.denied;

  static_cast<void>(fixture.engine.tick(denied + 50.0));
  require(fixture.executor.suspensions == 0,
          "prefill lane suspended before the resource wait bound");
  static_cast<void>(fixture.engine.tick(denied + 100.0));
  require(fixture.executor.suspensions == 1 &&
              fixture.engine.snapshot().resourceSuspensions == 1 &&
              fixture.engine.snapshot().resourceDecodeSuspensions == 0 &&
              !fixture.resident(),
          "prefill lane was not suspended at the resource wait bound");
  const ResourceWaitSnapshot waiting =
      fixture.engine.resourceWaitSnapshot(denied + 100.0);
  require(waiting.memory == 1 && waiting.suspended == 1 &&
              waiting.oldestWaitMilliseconds == 100.0,
          "prefill suspension lost the back-off's wait time");
}

// A suspended peer drains the engine: the lane suspends at its denial, as
// before the back-off, whether the peer suspended before the lane's first
// denial or during its back-off. A waiting peer of more important priority
// suspends it too.
void testSoleDecodeLaneSuspendsBesideAWaitingPeer() {
  constexpr uint64_t peerId = 276;
  for (uint32_t scenario = 0; scenario < 3; ++scenario) {
    SoleDecodeLane fixture(30'000.0);
    auto peer = request(peerId, std::vector<uint32_t>(64, 76));
    peer.deadlineMilliseconds = 100'000;
    double backoffEnds = 100'000.0;
    if (scenario == 1) {
      backoffEnds = fixture.decodeUntilBackedOff() + 100.0;
      ++fixture.now;
    } else {
      fixture.backing.growthBlocked = true;
    }
    if (scenario == 2) {
      peer.priority = RequestPriority::Foreground;
      fixture.executor.beginGrowthBlocked = [] { return true; };
      fixture.executor.beginAllocationFailure =
          metal::AllocationFailure::HostPressure;
    }
    fixture.engine.submit(std::move(peer));

    const uint32_t lanesSuspended = scenario == 2 ? 1 : 2;
    require(fixture.tickUntil([&] {
              return fixture.executor.suspensions == lanesSuspended;
            }) && fixture.now < backoffEnds,
            "lane backed off beside a suspended or more important peer");
    require(!fixture.resident(SoleDecodeLane::laneId) &&
                fixture.events.emitted == SoleDecodeLane::tokensBeforeDenial,
            "lane did not suspend at its denial beside a waiting peer");
  }
}

void testKvPressureNarrowsTheRealBatch() {
  Backing backing(3);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  engine.submit(request(30, std::vector<uint32_t>(33, 30)));
  engine.submit(request(31, std::vector<uint32_t>(33, 31)));
  runUntilIdle(engine);

  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "KV pressure failed a lane that could run in a narrower batch");
  require(std::find(executor.prefillWidths.begin(),
                    executor.prefillWidths.end(),
                    1) != executor.prefillWidths.end(),
          "resource pressure did not degrade B2 prefill to real B1 commands");
}

// Cached state and KV share one physical budget. Required KV growth for a
// live lane takes the least recently used cached state through the unified
// reclaim order instead of failing or suspending, and the growing lane still
// publishes its own replay state afterwards.
void testKvGrowthReclaimsCachedStateWhenBudgetIsShared() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  backing.growthAllowed = [&] {
    return resources.snapshot().stateCache.entries == 0;
  };
  Events events;
  engine::Engine engine({}, resources, executor, events);
  engine.submit(request(230, std::vector<uint32_t>(65, 230)));
  runUntilIdle(engine);
  const auto cached = resources.snapshot();
  require(cached.stateCache.entries == 1 && cached.pool.pagesResident == 4,
          "shared-budget fixture did not cache one state in one extent");

  engine.submit(request(231, std::vector<uint32_t>(161, 231)));
  runUntilIdle(engine);
  const auto after = engine.snapshot();
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0 &&
              executor.suspensions == 0 && executor.prefillRows == 226,
          "required KV growth failed or suspended behind cached state");
  require(after.resources.stateCache.evictions == 1 &&
              after.resources.stateCache.entries == 1 &&
              after.replayStatePublications == 2 &&
              after.recycledStatePublications == 0 &&
              after.resources.pool.pagesResident == 8,
          "KV growth did not take exactly the older cached state");
  require(after.resources.pool.pagesActive == 0 &&
              after.resources.activeRequests == 0,
          "grown request leaked its active resources");
}

void testRequiredWorkDoesNotReserveAnExtraPage() {
  for (uint32_t promptTokens : {1U, 24U}) {
    Backing backing(1);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    Events events;
    engine::Engine engine({}, resources, executor, events);
    engine.submit(request(231, std::vector<uint32_t>(promptTokens, 231)));
    runUntilIdle(engine);
    require(events.completedCount == 1 && events.emitted == 1 &&
                events.capacityExhaustedCount == 0 &&
                engine.snapshot().scheduler.decodeBatchesByWidth[0] == 1,
            "unused KV runway rejected a request whose full verify fits");
  }
}

void testAdmissionPinsDesiredStateAndCountsOnlySuccess() {
  Backing backing(16);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> desired(65, 240);
  engine.submit(request(240, desired));
  runUntilIdle(engine);
  engine.submit(request(241, std::vector<uint32_t>(65, 241)));
  runUntilIdle(engine);
  const auto before = engine.snapshot();
  require(before.resources.stateCache.entries == 2,
          "pinning test did not publish both resident states");
  uint32_t pinnedAttempts = 0;
  executor.beginObserver = [&] {
    if (resources.snapshot().stateCache.pinned == 1) ++pinnedAttempts;
  };
  executor.deniedBegins = 2;
  engine.submit(request(242, desired));
  runUntilIdle(engine);
  const auto after = engine.snapshot();
  require(pinnedAttempts == 3 && events.completedCount == 3 &&
              after.resources.stateCache.entries == 1 &&
              after.resources.stateCache.pinned == 0 &&
              events.starts.back().first == EngineCacheStatus::PrefixHit &&
              events.starts.back().second == 64 &&
              after.cacheHits == before.cacheHits + 1 &&
              after.resources.lookup.lookups == before.resources.lookup.lookups + 1,
          "successful retry failed to restore/count the protected cache state");
}

void testAdmissionCanDropItsOwnCachePinToMakeProgress() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(65, 245);
  engine.submit(request(245, prompt));
  runUntilIdle(engine);
  require(resources.snapshot().stateCache.entries == 1,
          "self-pin test did not retain its initial state");
  // Under the new ceiling, one active request fits only after releasing the
  // sole cached state. Its lookup lease must not create a permanent deadlock.
  executor.beginGrowthBlocked = [&] {
    return resources.snapshot().stateCache.entries != 0;
  };
  engine.submit(request(246, prompt));
  runUntilIdle(engine);
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0 && executor.restored == 0 &&
              executor.prefillRows == 130 &&
              events.starts.back().first == EngineCacheStatus::Miss &&
              engine.snapshot().coldMisses == 2 && engine.snapshot().cacheHits == 0 &&
              resources.snapshot().lookup.lookups == 2,
          "admission waited on its own cache pin instead of recomputing cold");
}

void testSingletonCapacityFailureTerminatesCleanly() {
  Backing backing(1);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  engine.submit(request(40, std::vector<uint32_t>(33, 40)));
  runUntilIdle(engine);
  require(events.completedCount == 0 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 1 && engine.idle(),
          "B1 capacity failure did not emit exactly one terminal event");
  require(resources.snapshot().activeRequests == 0 && executor.requests.empty(),
          "B1 capacity failure leaked backend resources");
}

void testQueuedLongPrefillsLeaveRoomForShortWork() {
  Backing backing(2048);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  for (uint64_t id = 1; id <= 4; ++id)
    engine.submit(request(id, std::vector<uint32_t>(8193, id)));
  require(engine.tick(1) && engine.tick(2) &&
              executor.requests.size() == 1 && executor.prefillRows == 2048,
          "long prefills reserved cells without executable work");
  engine.submit(request(5, std::vector<uint32_t>(65, 5)));
  require(engine.tick(3) && executor.requests.contains(5) &&
              executor.requests.at(5).position > 0 && events.completedCount == 0,
          "short arrival waited for a long prefill to finish");
  for (uint32_t now = 4; now < 200 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));
  require(engine.idle() && events.completedCount == 5 &&
              events.failedCount == 0 && events.capacityExhaustedCount == 0 &&
              executor.prefillRows == 4 * 8193 + 65 && executor.suspensions == 0,
          "admission lost work, introduced replay, or stranded queued requests");
}

void testAdmissionUsesCachedRemainingWork() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> warm(4097, 47);
  engine.submit(request(1, warm));
  runUntilIdle(engine);
  engine.submit(request(2, std::vector<uint32_t>(8193, 48)));
  require(engine.tick(1) && engine.tick(2), "cold prefill did not start");
  engine.submit(request(3, warm));
  require(engine.tick(3) && executor.requests.contains(3) &&
              executor.restored == 4096 && executor.requests.at(3).position == 4097,
          "cached prompt was scheduled by total length instead of remaining work");
  runUntilIdle(engine);
  require(events.completedCount == 3 && events.failedCount == 0,
          "cache-aware admission failed to finish");
}

void testFailedAdmissionDoesNotBlockOtherWork() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return executor.beginAttempts == 1; };
  engine.submit(request(1, std::vector<uint32_t>(4097, 47)));
  engine.submit(request(2, std::vector<uint32_t>(8193, 48)));
  require(engine.tick(1) && executor.requests.contains(2),
          "failed prefill admission blocked a runnable peer");
  runUntilIdle(engine);
  require(events.completedCount == 2 && events.failedCount == 0,
          "failed admission did not recover");
}

void testSchedulingWaitDoesNotConsumeMemoryTimeout() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  Events events;
  engine::Engine engine({.resourceWaitTimeoutMilliseconds = 1000.0},
                        resources, executor, events);
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  executor.beginGrowthBlocked = [&] { return executor.beginAttempts == 1; };
  engine.submit(request(1, std::vector<uint32_t>(8193, 47)));
  static_cast<void>(engine.tick(1));
  auto urgent = request(2, std::vector<uint32_t>(4097, 48));
  urgent.priority = RequestPriority::Foreground;
  engine.submit(std::move(urgent));
  require(engine.tick(102) && engine.resourceWaitSnapshot(102).memory == 0,
          "scheduler delay retained a stale memory wait");
  for (uint32_t now = 1100; now < 1200 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));
  require(engine.idle() && events.completedCount == 2 && events.failedCount == 0,
          "scheduling delay triggered the memory wait timeout");
}

void testUnadmittedRequestsHonorCancellationAndDeadline() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  engine.submit(request(1, std::vector<uint32_t>(8193, 1)));
  engine.submit(request(2, std::vector<uint32_t>(8193, 2)));
  auto expiring = request(3, std::vector<uint32_t>(8193, 3));
  expiring.deadlineMilliseconds = 3;
  engine.submit(std::move(expiring));
  require(engine.tick(1) && engine.tick(2), "long prefill did not start");
  engine.cancel(2);
  static_cast<void>(engine.tick(3));
  runUntilIdle(engine);
  require(engine.snapshot().cancelled == 1 && events.failedCount == 1 &&
              events.completedCount == 2 && executor.beginAttempts == 1,
          "queued cancellation or deadline allocated resources or failed cleanup");
}

void testGrowthKeepsPrefillProgressWhenAnUnstartedPeerCanYield() {
  for (RequestPriority priority : {RequestPriority::Normal,
                                   RequestPriority::Background}) {
    Backing backing(512);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(2);
    Events events;
    engine::Engine engine({}, resources, executor, events);
    backing.growthAllowed = [&] {
      const auto started = executor.requests.find(48);
      return started == executor.requests.end() ||
             started->second.position < 2048 ||
             std::count_if(executor.requests.begin(), executor.requests.end(),
                           [](const auto &entry) { return entry.second.resident; }) < 2;
    };
    engine.submit(request(48, std::vector<uint32_t>(4096, 48)));
    auto peer = request(49, std::vector<uint32_t>(8192, 49));
    peer.priority = priority;
    engine.submit(std::move(peer));
    require(engine.tick(1) && engine.tick(2) &&
                executor.requests.at(48).position == 2048 &&
                !executor.requests.contains(49),
            "unstarted prefill reserved a cell before it had scheduled work");
    require(engine.tick(3) && executor.requests.at(48).resident,
            "KV growth discarded completed prefill");
    if (priority == RequestPriority::Normal) {
      // The final prefill slice admits a peer into its remaining row budget;
      // if that cell prevents KV growth, the unstarted peer must yield.
      require(executor.suspensions == 1 && !executor.requests.at(49).resident,
              "KV growth did not yield the unstarted peer");
    } else {
      require(executor.suspensions == 0 && !executor.requests.contains(49),
              "lower-priority work reserved a cell before its dispatch");
    }
    runUntilIdle(engine);
    require(events.completedCount == 2 && events.failedCount == 0 &&
                events.capacityExhaustedCount == 0 && executor.prefillRows == 12288,
            "work-aware admission lost prefill work or failed to complete");
  }
}

void testGrowthYieldsLowerPriorityResidentOutsideBatch() {
  Backing backing(512);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  backing.growthAllowed = [&] {
    return std::count_if(executor.requests.begin(), executor.requests.end(),
                         [](const auto &entry) { return entry.second.resident; }) < 2;
  };
  engine.submit(request(48, std::vector<uint32_t>(4096, 48)));
  require(engine.tick(1) && engine.tick(2) &&
              executor.requests.at(48).position == 2048,
          "priority fixture did not advance the initial request");
  auto foreground = request(49, std::vector<uint32_t>(8192, 49));
  foreground.priority = RequestPriority::Foreground;
  engine.submit(std::move(foreground));
  require(engine.tick(3) && executor.suspensions == 1 &&
              !executor.requests.at(48).resident &&
              executor.requests.at(49).resident,
          "resource recovery preempted foreground work for a lower-priority peer");
  runUntilIdle(engine);
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "priority preemption failed to finish both requests");
}

constexpr double kPeerSuspensionTicks = 4;
constexpr double kSuspensionPointTickLimit = 64;

// Suspends request 48 once it has prefilled `suspendAt` tokens, by admitting a
// foreground peer `peerId` that leaves no room for both lanes.
void suspendPrefillForForegroundPeer(engine::Engine &engine,
                                     engine::Cache &resources,
                                     Executor &executor, uint64_t peerId,
                                     uint32_t suspendAt, double &now) {
  for (; now < kSuspensionPointTickLimit && !(executor.requests.contains(48) &&
                       executor.requests.at(48).resident &&
                       executor.requests.at(48).position == suspendAt);
       ++now)
    static_cast<void>(engine.tick(now));
  require(executor.requests.at(48).position == suspendAt,
          "replay fixture did not reach the suspension point");
  // An earlier peer's idle cached pages would let this one run without
  // growth; evict them so its admission must displace request 48.
  static_cast<void>(resources.reclaimCache(
      std::numeric_limits<uint64_t>::max(), true));
  const uint32_t suspensions = executor.suspensions;
  auto foreground = request(peerId, std::vector<uint32_t>(8192, peerId));
  foreground.priority = RequestPriority::Foreground;
  engine.submit(std::move(foreground));
  // The peer is admitted first; its growth then displaces request 48.
  const double deadline = now + kPeerSuspensionTicks;
  for (; now < deadline && executor.suspensions == suspensions; ++now) {
    require(executor.requests.at(48).position == suspendAt,
            "prefilling lane advanced past the suspension point");
    static_cast<void>(engine.tick(now));
  }
  require(executor.suspensions == suspensions + 1 &&
              !executor.requests.at(48).resident,
          "foreground peer did not suspend the prefilling lane");
}

void testMidPrefillSuspensionCountsOnlyProcessedTokensAsReplay() {
  Backing backing(512);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  // Nothing is restored, so the resume replays from the first token.
  executor.deniedSnapshots = std::numeric_limits<uint32_t>::max();
  Events events;
  engine::Engine engine({}, resources, executor, events);
  backing.growthAllowed = [&] {
    return std::count_if(executor.requests.begin(), executor.requests.end(),
                         [](const auto &entry) { return entry.second.resident; }) < 2;
  };
  engine.submit(request(48, std::vector<uint32_t>(4096, 48)));
  double now = 1;

  suspendPrefillForForegroundPeer(engine, resources, executor, 49, 2048, now);
  for (; now < 128 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));

  require(engine.idle() && events.completedCount == 2 &&
              events.failedCount == 0 && executor.restored == 0 &&
              engine.snapshot().resourceResumptions == 1,
          "mid-prefill replay fixture did not finish both requests");
  // The 2048 prompt tokens never processed before the suspension are
  // ordinary prefill, not replay.
  require(engine.snapshot().resourceReplayTokens == 2048,
          "unprocessed prompt suffix was counted as cache replay");
  require(engine.snapshot().resourceDecodeSuspensions == 0 &&
              engine.snapshot().resourceDecodeReplayTokens == 0,
          "a prefill suspension was counted as a decode suspension");
}

void testRepeatedSuspensionKeepsTheFirstReplayEnd() {
  Backing backing(512);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  // Nothing is restored, so every resume replays from the first token.
  executor.deniedSnapshots = std::numeric_limits<uint32_t>::max();
  Events events;
  engine::Engine engine({}, resources, executor, events);
  backing.growthAllowed = [&] {
    return std::count_if(executor.requests.begin(), executor.requests.end(),
                         [](const auto &entry) { return entry.second.resident; }) < 2;
  };
  engine.submit(request(48, std::vector<uint32_t>(8192, 48)));
  double now = 1;

  suspendPrefillForForegroundPeer(engine, resources, executor, 49, 4096, now);
  // The second suspension lands inside the replay, below the first end.
  suspendPrefillForForegroundPeer(engine, resources, executor, 50, 2048, now);
  for (; now < 192 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));

  require(engine.idle() && events.completedCount == 3 &&
              events.failedCount == 0 && executor.restored == 0 &&
              engine.snapshot().resourceResumptions == 2,
          "repeated suspension fixture did not finish every request");
  // First partial replay: 2048 rows. Second replay: the 4096 rows processed
  // before the first suspension; the rest of the prompt is ordinary prefill.
  require(engine.snapshot().resourceReplayTokens == 2048 + 4096,
          "a second suspension lowered the replay end of the first");
}

void testSuspensionDuringDecodeReplayKeepsCountingDecodeReplay() {
  Backing backing(512);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  executor.decodeFinishes = false;
  // Nothing is restored, so every resume replays from the first token.
  executor.deniedSnapshots = std::numeric_limits<uint32_t>::max();
  Events events;
  engine::Engine engine({}, resources, executor, events);
  backing.growthAllowed = [&] {
    return std::count_if(executor.requests.begin(), executor.requests.end(),
                         [](const auto &entry) { return entry.second.resident; }) < 2;
  };
  auto decoding = request(48, std::vector<uint32_t>(4096, 48));
  decoding.maxNewTokens = 64;
  engine.submit(std::move(decoding));
  double now = 1;
  for (; now < kSuspensionPointTickLimit && !events.outputs.contains(48); ++now)
    static_cast<void>(engine.tick(now));
  auto foreground = request(49, std::vector<uint32_t>(8192, 49));
  foreground.priority = RequestPriority::Foreground;
  engine.submit(std::move(foreground));
  for (; now < kSuspensionPointTickLimit && executor.suspensions == 0; ++now)
    static_cast<void>(engine.tick(now));
  require(executor.suspensions == 1 &&
              engine.snapshot().resourceDecodeSuspensions == 1,
          "foreground peer did not suspend the decoding lane");
  const uint64_t history = 4096 + events.outputs.at(48).size();

  // The second suspension lands inside the replay of the decode history.
  suspendPrefillForForegroundPeer(engine, resources, executor, 50, 2048, now);
  for (; now < 256 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));

  require(engine.idle() && events.completedCount == 3 &&
              events.failedCount == 0 &&
              engine.snapshot().resourceSuspensions == 2,
          "decode replay suspension fixture did not finish every request");
  require(engine.snapshot().resourceDecodeSuspensions == 1,
          "a suspension during replay was counted as a decode suspension");
  // First partial replay: 2048 rows. Second replay: the whole decode history.
  require(engine.snapshot().resourceReplayTokens == 2048 + history &&
              engine.snapshot().resourceDecodeReplayTokens == 2048 + history,
          "a suspension during decode replay stopped counting decode replay");
}

// testMidPrefillSuspensionCountsOnlyProcessedTokensAsReplay's fixture, with
// the fake moving its live state out at suspension.
struct LiveStateSuspension {
  Backing backing{512};
  KvPool pool{backing};
  engine::Cache resources{pool, CacheNamespace{}};
  Executor executor{2};
  Events events;
  engine::Engine engine;
  double now = 1;

  explicit LiveStateSuspension(EngineConfig config = {})
      : engine(config, resources, executor, events) {
    executor.moveStateOnSuspend = true;
    backing.growthAllowed = [this] {
      return std::count_if(executor.requests.begin(), executor.requests.end(),
                           [](const auto &entry) {
                             return entry.second.resident;
                           }) < 2;
    };
  }

  void runToIdle() {
    for (; now < 256 && !engine.idle(); ++now)
      static_cast<void>(engine.tick(now));
    require(engine.idle() && events.failedCount == 0,
            "live-state suspension fixture did not finish its requests");
  }

  // The boundary of the deepest state cached on `tokens`.
  uint32_t cachedBoundary(std::span<const uint32_t> tokens) {
    return resources.lookup(tokens).resumeBoundary();
  }
};

void testPageAlignedSuspensionResumesFromItsLiveState() {
  LiveStateSuspension fixture;
  const std::vector<uint32_t> prompt(4096, 48);
  fixture.engine.submit(request(48, prompt));

  suspendPrefillForForegroundPeer(fixture.engine, fixture.resources,
                                  fixture.executor, 49, 2048, fixture.now);

  require(fixture.executor.movedStates == 1 &&
              fixture.engine.snapshot().checkpointPublications == 1 &&
              fixture.cachedBoundary(prompt) == 2048,
          "page-aligned suspension did not publish its live state");
  fixture.runToIdle();
  require(fixture.executor.restoredLengths.at(48) == 2048 &&
              fixture.executor.restoredDraft &&
              fixture.engine.snapshot().resourceReplayTokens == 0,
          "resume replayed rows its live state covered");
  // Adopted as the lane's checkpoint, the state is its deepest one at the
  // prompt end: kept beside the prompt-end state.
  require(fixture.cachedBoundary(std::span(prompt).first(2049)) == 2048,
          "the request did not keep its suspension state as its checkpoint");
}

void testUnalignedSuspensionPublishesNothing() {
  LiveStateSuspension fixture;
  fixture.backing.growthAllowed = nullptr;
  fixture.executor.decodeFinishes = false;
  fixture.engine.submit(request(48, std::vector<uint32_t>(4096, 48)));
  // A short peer leads the first command, leaving request 48 an unaligned
  // share of the budget; it then decodes while request 48 needs growth.
  auto decoding = request(47, std::vector<uint32_t>(10, 47));
  decoding.maxNewTokens = 32;
  fixture.engine.submit(std::move(decoding));
  require(fixture.engine.tick(1) &&
              fixture.executor.requests.at(48).position == 2038,
          "unaligned fixture did not stop request 48 mid-page");

  fixture.backing.growthBlocked = true;
  for (fixture.now = 2; fixture.now < 16 && !fixture.executor.suspensions;
       ++fixture.now)
    static_cast<void>(fixture.engine.tick(fixture.now));
  require(fixture.executor.suspendedPositions ==
                  std::vector<uint32_t>{2038} &&
              fixture.executor.movedStates == 0 &&
              fixture.engine.snapshot().checkpointPublications == 0,
          "unaligned suspension published a state");
  fixture.backing.growthBlocked = false;
  fixture.executor.decodeFinishes = true;
  fixture.runToIdle();

  // No state lay below it, so the resume replays from the first token.
  require(!fixture.executor.restoredLengths.contains(48) &&
              fixture.engine.snapshot().resourceReplayTokens == 2038,
          "unaligned suspension did not resume from the deepest state");
}

void testSuspensionAtAStateBlockKeepsThatState() {
  bool paused = false;
  EngineConfig config;
  config.prefillCheckpointTokens = 2048;
  // Paused growth suspends without the growth reclaim that would first
  // evict the lane's checkpoint.
  config.growthPaused = [&] { return paused; };
  LiveStateSuspension fixture(config);
  fixture.backing.growthAllowed = nullptr;
  const std::vector<uint32_t> prompt(4096, 48);
  fixture.engine.submit(request(48, prompt));
  for (; fixture.now < 8 &&
         fixture.engine.snapshot().checkpointPublications == 0;
       ++fixture.now)
    static_cast<void>(fixture.engine.tick(fixture.now));
  require(fixture.executor.requests.at(48).position == 2048 &&
              fixture.engine.snapshot().checkpointPublications == 1,
          "state-block fixture did not checkpoint at 2048");

  paused = fixture.backing.growthBlocked = true;
  static_cast<void>(fixture.engine.tick(fixture.now++));

  // The lane's checkpoint already sits at the block: nothing is moved.
  require(fixture.executor.suspendedPositions ==
                  std::vector<uint32_t>{2048} &&
              fixture.executor.movedStates == 0 &&
              fixture.engine.snapshot().checkpointPublications == 1 &&
              fixture.cachedBoundary(prompt) == 2048,
          "suspension at a state's block replaced or lost that state");
  paused = fixture.backing.growthBlocked = false;
  fixture.runToIdle();
  require(fixture.executor.restoredLengths.at(48) == 2048 &&
              fixture.engine.snapshot().resourceReplayTokens == 0,
          "suspension at a state's block did not resume from it");
}

void testSuspensionStateReplacesTheLaneCheckpoint() {
  bool paused = false;
  EngineConfig config;
  config.growthPaused = [&] { return paused; };
  LiveStateSuspension fixture(config);
  fixture.backing.growthAllowed = nullptr;
  const std::vector<uint32_t> prompt(12288, 48);
  fixture.engine.submit(request(48, prompt));
  // The lane checkpoints at 4096; 6144 is no planned boundary.
  const auto &lanes = fixture.executor.requests;
  for (; fixture.now < 16 &&
         !(lanes.contains(48) && lanes.at(48).position == 6144);
       ++fixture.now)
    static_cast<void>(fixture.engine.tick(fixture.now));
  require(fixture.cachedBoundary(prompt) == 4096,
          "replacement fixture did not checkpoint at 4096");

  paused = fixture.backing.growthBlocked = true;
  for (const double deadline = fixture.now + kPeerSuspensionTicks;
       fixture.now < deadline && !fixture.executor.suspensions; ++fixture.now)
    static_cast<void>(fixture.engine.tick(fixture.now));

  // The suspension state is the lane's recovery point now; the earlier
  // checkpoint is retired rather than left behind.
  require(fixture.executor.suspendedPositions ==
                  std::vector<uint32_t>{6144} &&
              fixture.executor.movedStates == 1 &&
              fixture.cachedBoundary(prompt) == 6144 &&
              fixture.cachedBoundary(std::span(prompt).first(6143)) == 0,
          "suspension state did not replace the lane's checkpoint");
  paused = fixture.backing.growthBlocked = false;
  fixture.runToIdle();
  // The suspension state rolled to the 8192 checkpoint, the deepest below
  // the prompt end, which the completed request keeps.
  require(fixture.executor.restoredLengths.at(48) == 6144 &&
              fixture.cachedBoundary(std::span(prompt).first(12256)) == 8192 &&
              fixture.cachedBoundary(std::span(prompt).first(8193)) == 8192 &&
              fixture.cachedBoundary(std::span(prompt).first(6145)) == 0,
          "a completed request kept other than its deepest checkpoint");
}

void testDecodeSuspensionPublishesNothing() {
  bool paused = false;
  EngineConfig config;
  // Paused growth suspends without the growth reclaim that would first
  // evict the decoding lane's replay state.
  config.growthPaused = [&] { return paused; };
  LiveStateSuspension fixture(config);
  fixture.backing.growthAllowed = nullptr;
  fixture.executor.decodeFinishes = false;
  const std::vector<uint32_t> prompt(4096, 48);
  auto decoding = request(48, prompt);
  decoding.maxNewTokens = 32;
  fixture.engine.submit(std::move(decoding));
  for (; fixture.now < 16 && fixture.events.outputs[48].empty(); ++fixture.now)
    static_cast<void>(fixture.engine.tick(fixture.now));
  require(!fixture.events.outputs[48].empty() &&
              fixture.cachedBoundary(prompt) == 4064,
          "decode fixture did not reach decode");

  paused = fixture.backing.growthBlocked = true;
  auto foreground = request(49, std::vector<uint32_t>(8192, 49));
  foreground.priority = RequestPriority::Foreground;
  fixture.engine.submit(std::move(foreground));
  for (const double deadline = fixture.now + kPeerSuspensionTicks;
       fixture.now < deadline && !fixture.executor.suspensions; ++fixture.now)
    static_cast<void>(fixture.engine.tick(fixture.now));
  require(fixture.executor.suspensions == 1 &&
              fixture.executor.suspendedPositions.front() == prompt.size() &&
              fixture.executor.movedStates == 0 &&
              fixture.engine.snapshot().checkpointPublications == 0,
          "decode suspension published a state");

  // Warning reclaim keeps the deepest usable state for the resume.
  static_cast<void>(fixture.resources.reclaimCache(
      std::numeric_limits<uint64_t>::max(), false));
  require(fixture.cachedBoundary(prompt) == 4064,
          "Warning reclaim took a decode suspension's resume state");
  paused = fixture.backing.growthBlocked = false;
  fixture.executor.decodeFinishes = true;
  fixture.runToIdle();
  require(fixture.executor.restoredLengths.at(48) == 4064,
          "decode suspension did not resume from its replay state");
}

void testSuspensionBeforeAnyPrefillPublishesNothing() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  executor.moveStateOnSuspend = true;
  executor.physicalGrowthBlocked = &backing.growthBlocked;
  Events events;
  engine::Engine engine({}, resources, executor, events);

  backing.growthBlocked = true;
  engine.submit(request(50, {50}));
  engine.submit(request(51, {51}));
  runUntilIdle(engine);

  require(executor.suspendedPositions == std::vector<uint32_t>{0} &&
              executor.movedStates == 0 && events.completedCount == 2 &&
              events.failedCount == 0,
          "a suspension at position 0 moved state or failed");
}

void testShortTargetOnlySuspensionStateIsDropped() {
  LiveStateSuspension fixture;
  // The first planned boundary, 4064, lies less than a draft window past it.
  fixture.executor.targetOnlyAtBoundary = 2048;
  fixture.engine.submit(request(48, std::vector<uint32_t>(4096, 48)));

  suspendPrefillForForegroundPeer(fixture.engine, fixture.resources,
                                  fixture.executor, 49, 2048, fixture.now);

  require(fixture.executor.movedStates == 1 &&
              fixture.engine.snapshot().checkpointPublications == 0 &&
              fixture.resources.snapshot().stateCache.entries == 0,
          "an unusable target-only suspension state was published");
  require(*fixture.executor.parkedStateBytes == 0 &&
              fixture.executor.reclaimedIdleStateBytes == 64,
          "the dropped moved state stayed parked in the idle pool");
  fixture.runToIdle();
  require(!fixture.executor.restoredLengths.contains(48) &&
              fixture.engine.snapshot().resourceReplayTokens == 2048,
          "dropped suspension state was restored");
}

// Publishes KV for `tokens` under a cache-only request, with an ordinary
// full state at `stateBoundary`.
void cachePrefix(engine::Cache &resources, uint64_t requestId,
                 std::span<const uint32_t> tokens, uint32_t stateBoundary) {
  resources.beginRequest(requestId);
  require(resources.ensureTokens(requestId, tokens.size()).granted(),
          "cache-only prefix pages were not acquired");
  static_cast<void>(resources.publishCommittedBlocks(
      requestId, tokens, static_cast<uint32_t>(tokens.size())));
  resources.publishCompositeState(resources.blockAt(requestId, stateBoundary),
                                  std::make_shared<State>());
  resources.endRequest(requestId);
}

void testRejectedTargetOnlySuspensionStateFallsBack() {
  LiveStateSuspension fixture;
  // The first planned boundary, the 4096 checkpoint, lies a whole draft
  // window past it: the suspension publishes the target-only state.
  fixture.executor.targetOnlyAtBoundary = 2048;
  const std::vector<uint32_t> prompt(8192, 48);
  fixture.engine.submit(request(48, prompt));
  suspendPrefillForForegroundPeer(fixture.engine, fixture.resources,
                                  fixture.executor, 49, 2048, fixture.now);
  require(fixture.engine.snapshot().checkpointPublications == 1 &&
              fixture.cachedBoundary(prompt) == 2048,
          "a usable target-only suspension state was not published");

  // Cached KV to 3072 adds a junction admit plans before that checkpoint.
  cachePrefix(fixture.resources, 999, std::span(prompt).first(3072), 1024);
  fixture.runToIdle();
  require(fixture.executor.restoredLengths.at(48) == 1024,
          "admit did not walk past a target-only state it rejects");
}

void testSuspensionStateIsProtectedUntilUnprotectedStatesGo() {
  LiveStateSuspension fixture;
  const std::vector<uint32_t> prompt(4096, 48);
  fixture.engine.submit(request(48, prompt));
  suspendPrefillForForegroundPeer(fixture.engine, fixture.resources,
                                  fixture.executor, 49, 2048, fixture.now);
  uint64_t published = 0;
  {
    const CacheLookup lookup = fixture.resources.lookup(prompt);
    require(lookup.resumeBoundary() == 2048,
            "protection fixture did not publish the suspension state");
    published = lookup.state->kvBlock();
  }

  static_cast<void>(fixture.resources.reclaimCache(
      std::numeric_limits<uint64_t>::max(), false));
  require(fixture.resources.carriesState(published),
          "Warning reclaim took the published suspension state");

  const std::vector<uint32_t> other(64, 998);
  cachePrefix(fixture.resources, 998, other, 32);
  uint64_t unprotected = 0;
  {
    const CacheLookup lookup = fixture.resources.lookup(other);
    unprotected = lookup.state->kvBlock();
  }
  for (uint32_t step = 0;
       step < 16 && fixture.resources.carriesState(unprotected); ++step) {
    require(fixture.resources.carriesState(published),
            "growth reclaim took the suspension state before an unprotected one");
    static_cast<void>(fixture.resources.reclaimOneForGrowth());
  }
  require(!fixture.resources.carriesState(unprotected) &&
              fixture.resources.carriesState(published),
          "growth reclaim did not take the unprotected state first");
  fixture.runToIdle();
}

void testPrefillGrowthPreservesAnActiveDecodePeer() {
  Backing backing(512);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  engine.submit(request(48, std::vector<uint32_t>(8192, 48)));
  require(engine.tick(1) && engine.tick(2), "prefill setup did not progress");
  auto decoding = request(49, {49});
  decoding.maxNewTokens = 32;
  engine.submit(std::move(decoding));
  require(engine.tick(3) && engine.tick(4) &&
              executor.requests.at(49).position == 1,
          "short peer did not enter decode");
  backing.growthBlocked = true;
  for (uint32_t step = 5; step < 15 && !executor.suspensions; ++step)
    static_cast<void>(engine.tick(step));
  require(executor.suspensions == 1 && !executor.requests.at(48).resident &&
              executor.requests.at(49).resident,
          "prefill growth interrupted an equal-priority decode stream");
  backing.growthBlocked = false;
  executor.decodeFinishes = true;
  runUntilIdle(engine);
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "mixed-phase pressure failed to finish both requests");
}

void testPhysicalKvPressureSuspendsInsteadOfKillingActiveWork() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  executor.physicalGrowthBlocked = &backing.growthBlocked;
  Events events;
  engine::Engine engine({}, resources, executor, events);

  backing.growthBlocked = true;
  engine.submit(request(50, {50}));
  engine.submit(request(51, {51}));
  runUntilIdle(engine);

  require(executor.suspensions == 1 && executor.resumptions == 1 &&
              engine.snapshot().resourceSuspensions == 1 &&
              engine.snapshot().resourceResumptions == 1,
          "physical KV pressure did not suspend and resume one lane");
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "recoverable physical pressure killed an active request");
}

void testPhysicalPressureRetryIsBackedOffWithoutProgress() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  executor.physicalGrowthBlocked = &backing.growthBlocked;
  executor.unblockGrowthOnSuspend = false;
  Events events;
  MemoryPressure pressure = MemoryPressure::Warning;
  EngineConfig config;
  config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
  engine::Engine engine(config, resources, executor, events);

  backing.growthBlocked = true;
  engine.submit(request(52, {52}));
  engine.submit(request(53, {53}));
  require(engine.tick(1.0) && engine.tick(2.0),
          "physical pressure did not suspend both blocked lanes");
  const uint32_t attemptsBeforeBackoff = executor.resumeAttempts;
  require(executor.suspensions == 2 && attemptsBeforeBackoff == 0,
          "suspended lanes were retried before a resource wake-up");
  require(!engine.tick(50.0) &&
              executor.resumeAttempts == attemptsBeforeBackoff,
          "resource wait retried on an unrelated scheduler tick");

  backing.growthBlocked = false;
  pressure = MemoryPressure::Normal;
  require(engine.tick(102.0), "resource retry timer did not wake the engine");
  for (double now = 103.0; now < 120.0 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));
  require(engine.idle() && executor.resumptions == 2 &&
              events.completedCount == 2 && events.failedCount == 0,
          "backed-off resource requests did not recover cleanly");
}

// While a suspended request recovers, admitQueued() retries only suspended
// requests; a stale non-suspended waiter must not pin the wakeup at 0 ms.
void testRecoveryWakeupIgnoresWaitersItDoesNotRetry() {
  for (bool waiterDeadline : {false, true}) {
    Backing backing(8);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    Events events;
    EngineConfig config;
    config.growthPaused = [] { return true; };
    engine::Engine engine(config, resources, executor, events);

    backing.growthBlocked = true;
    engine.submit(request(60, {60}));
    auto waiter = request(61, {61});
    if (waiterDeadline)
      waiter.deadlineMilliseconds = 50.0;
    engine.submit(std::move(waiter));
    require(engine.tick(1.0), "blocked lane did not suspend");
    const auto setup = engine.resourceWaitSnapshot(1.0);
    require(setup.suspended == 1 && setup.memory == 1 &&
                setup.concurrency == 1 && !setup.draining,
            "fixture lacks one suspended and one non-suspended waiter");

    executor.reclaimableIdleStateBytes = 128;
    require(engine.reclaimMemory({.reclaimEmptyKvExtents = true}) > 0,
            "reclaim did not bump the resource epoch");
    // The suspended request's own epoch is stale too: one tick retries it,
    // the retry fails, and its wait is re-armed at the current epoch.
    static_cast<void>(engine.tick(10.0));
    require(executor.resumeAttempts == 1 &&
                engine.resourceWaitSnapshot(10.0).suspended == 1,
            "recovery did not retry the suspended request once");
    if (waiterDeadline) {
      require(engine.nextWakeupMilliseconds() == 50.0,
              "recovery dropped a non-suspended waiter's deadline");
      continue;
    }
    require(engine.nextWakeupMilliseconds() == 110.0,
            "stale non-suspended waiter woke the recovering engine at once");

    // Sleep until each reported wakeup, as FdTransport::run does: the ticks
    // over the interval are bounded by its retry back-off, not a 0 ms spin.
    constexpr double interval = 1000.0;
    constexpr double backoff = 100.0;
    double now = 10.0;
    uint32_t ticks = 0;
    while (ticks <= 1000) {
      now = std::max(now, engine.nextWakeupMilliseconds().value_or(now));
      if (now >= 10.0 + interval)
        break;
      static_cast<void>(engine.tick(now));
      ++ticks;
    }
    require(ticks <= interval / backoff,
            "recovery busy-spun on a stale non-suspended waiter");
  }
}

void testRecoveryDrainDoesNotConsumeResourceWaitBudget() {
  for (bool expireRequest : {false, true}) {
    Backing backing(2);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(2);
    executor.decodeFinishes = false;
    Events events;
    engine::Engine engine({}, resources, executor, events);
    for (uint64_t id : {250, 251}) {
      auto value = request(id, std::vector<uint32_t>(24, id));
      value.maxNewTokens = 4;
      value.deadlineMilliseconds = expireRequest ? 30000 : 1000000;
      engine.submit(std::move(value));
    }
    for (double now = 1; now <= 5; ++now)
      require(engine.tick(now), "drain fixture made no progress");
    require(executor.suspensions == 1, "drain fixture did not preempt a lane");
    executor.holdDecodeUntil = std::make_shared<bool>(false);
    require(engine.tick(6) && engine.commandInFlight(),
            "resident peer did not start its command");
    static_cast<void>(engine.tick(30006));
    require(executor.resumeAttempts == 0,
            "recovery retried while a resident peer was still running");
    if (expireRequest) {
      require(!events.failures.empty() &&
                  std::all_of(events.failures.begin(), events.failures.end(),
                              [](const auto &code) { return code == "deadline_exceeded"; }),
              "draining suppressed the request deadline");
    } else {
      require(events.failedCount == 0 && engine.nextWakeupMilliseconds() > 30006,
              "deliberate drain consumed the resource wait limit");
      const auto wait = engine.resourceWaitSnapshot(30006);
      require(wait.draining && wait.suspended == 1 && wait.memory == 1 &&
                  wait.oldestWaitMilliseconds >= 30000,
              "draining hid the elapsed resource wait from diagnostics");
    }
    *executor.holdDecodeUntil = true;
    for (double now = 30007; now < 30100 && !engine.idle(); ++now)
      static_cast<void>(engine.tick(now));
    require(engine.idle() && resources.snapshot().activeRequests == 0,
            "drain fixture did not release its resources");
    if (!expireRequest)
      require(events.completedCount == 2 && events.failedCount == 0 &&
                  executor.resumptions == 1,
              "suspended work did not resume after its peer completed");
  }
}

void testDecodePreemptionReplaysCommittedHistoryWithoutRepeatingOutput() {
  Backing backing(2);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  executor.decodeFinishes = false;
  // A replay result may carry a previously prepared consumer stage. It is
  // metadata, not a request to ask the frontend for that initial mask again.
  executor.replayDecodeStage = DecodeStage::ApplyInitialMask;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  for (uint64_t id : {250, 251}) {
    auto value = request(id, std::vector<uint32_t>(24, id));
    value.maxNewTokens = 4;
    value.returnProgress = true;
    engine.submit(std::move(value));
  }
  for (double now = 1; now <= 4; ++now)
    require(engine.tick(now), "decode replay setup made no progress");
  require(events.emitted == 2 && engine.tick(5) && executor.suspensions == 1,
          "full active KV did not preempt a lane at the verify boundary");
  uint64_t preempted = 0;
  for (const auto &[id, state] : executor.requests) {
    if (!state.resident) preempted = id;
  }
  require(preempted && resources.snapshot().activeRequests == 1 &&
              resources.snapshot().pool.pagesActive == 1,
          "preempted decode retained its active KV ownership");

  // New admission and the elapsed retry timer must not disturb the resident
  // lane while it uses the released headroom to finish.
  executor.holdDecodeUntil = std::make_shared<bool>(false);
  engine.submit(request(252, {252}));
  require(engine.tick(6) && engine.commandInFlight(),
          "surviving lane could not launch after preemption");
  require(!engine.tick(150) && engine.nextWakeupMilliseconds() == 1150.0 &&
              events.startIds.size() == 2 && executor.resumeAttempts == 0,
          "draining recovery busy-woke or admitted a competing request");
  *executor.holdDecodeUntil = true;
  for (double now = 151; now < 210 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));
  require(engine.idle() && executor.resumptions == 2 &&
              events.completedCount == 3 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "decode preemption did not finish all original/new requests");
  require(events.progress.at(250) == std::vector<uint32_t>({0, 24}) &&
              events.progress.at(251) == std::vector<uint32_t>({0, 24}),
          "decode recovery reported generated history as prompt progress");
  std::vector<uint32_t> committedHistory(24, preempted);
  committedHistory.push_back(42);
  // The newly admitted request also yields once because the recovering
  // continuation has reserved both pages. Neither replays generated output.
  require(executor.resumedPrompts ==
              std::vector<std::vector<uint32_t>>{committedHistory, {252}} &&
              engine.snapshot().resourceReplayTokens == committedHistory.size(),
          "replay omitted or duplicated committed generated history");
  require(engine.snapshot().resourceDecodeSuspensions == 1 &&
              engine.snapshot().resourceDecodeReplayTokens ==
                  committedHistory.size(),
          "decode preemption did not count its suspension and replay");
  require(events.outputs.at(250) == std::vector<uint32_t>(4, 42) &&
              events.outputs.at(251) == std::vector<uint32_t>(4, 42) &&
              events.usage.at(250) == std::pair<uint32_t, uint32_t>{24, 4} &&
              events.usage.at(251) == std::pair<uint32_t, uint32_t>{24, 4} &&
              events.startIds.size() == 3 &&
              events.maskRequests.empty() &&
              engine.snapshot().resources.lookup.lookups == 3 &&
              engine.snapshot().coldMisses == 3,
          "resource replay emitted prior output or changed request usage/hits");
  require(executor.requests.empty() && executor.snapshotAttempts == 0 &&
              resources.snapshot().pool.pagesActive == 0 &&
              resources.snapshot().activeRequests == 0,
          "decode replay leaked active resources");
}

void testLongDecodePreemptionPlansTheCurrentReplayBoundary() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  executor.decodeFinishes = false;
  executor.deniedSnapshots = std::numeric_limits<uint32_t>::max();
  Events events;
  MemoryPressure pressure = MemoryPressure::Normal;
  EngineConfig config;
  // Keep the old 4096-token replay boundary distinct from checkpoints.
  config.prefillCheckpointTokens = 8192;
  config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
  engine::Engine engine(config, resources, executor, events);

  constexpr uint64_t id = 255;
  const std::vector<uint32_t> prompt(4097, 5);
  auto value = request(id, prompt);
  value.maxNewTokens = 10'000;
  value.deadlineMilliseconds = 100'000;
  engine.submit(std::move(value));
  double now = 1;
  for (; now < 20'000 && executor.suspensions == 0; ++now) {
    if (events.emitted >= 4200) {
      backing.growthBlocked = true;
      pressure = MemoryPressure::Warning;
    }
    static_cast<void>(engine.tick(now));
  }
  require(executor.suspensions == 1 && !engine.commandInFlight(),
          "long decode did not suspend at KV growth pressure");
  const std::vector<uint32_t> emitted = events.outputs.at(id);
  std::vector<uint32_t> history = prompt;
  history.insert(history.end(), emitted.begin(), emitted.end());
  require(resources.snapshot().stateCache.entries == 0,
          "snapshot denial left a composite state to restore");

  // Retain a KV junction one draft window past the original prompt boundary.
  // The old prompt boundary, this junction, and the generated history's end
  // would require three capture spans if replay still used the old boundary.
  constexpr uint32_t retainedKvTokens = 6144;
  while (resources.snapshot().kvCache.blocks >
         retainedKvTokens / KvCache::pageTokens) {
    require(resources.reclaimOne(CacheReclaimMode::ReuseBacking).madeProgress,
            "could not reclaim the suspended decode's KV tail");
  }
  {
    auto lookup = resources.lookup(history);
    require(lookup.kvBoundary == retainedKvTokens &&
                lookup.resumeBoundary() == 0,
            "long replay did not retain the intended stateless KV junction");
  }

  backing.growthBlocked = false;
  pressure = MemoryPressure::Normal;
  executor.deniedSnapshots = 0;
  executor.decodeFinishes = true;
  now += 101;
  require(engine.tick(now++), "long replay did not resume after pressure eased");
  const auto &plan = executor.plans.at(id);
  require(plan.replayEnd == history.size() && plan.captureSpans.size() == 2,
          "resumed draft plan did not cover generated history in two spans");
  const double finishBy = now + 100;
  for (; now < finishBy && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));
  require(engine.idle() && executor.resumptions == 1 &&
              events.completedCount == 1 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0,
          "long decode did not complete after partial KV reclamation");
  require(executor.resumedPrompts ==
              std::vector<std::vector<uint32_t>>{history} &&
              engine.snapshot().resourceReplayTokens == history.size(),
          "long replay omitted or duplicated committed history");
  auto expectedOutput = emitted;
  expectedOutput.push_back(42);
  require(events.outputs.at(id) == expectedOutput &&
              events.usage.at(id) == std::pair<uint32_t, uint32_t>{
                                         prompt.size(), expectedOutput.size()} &&
              events.startIds == std::vector<uint64_t>{id},
          "long replay repeated output or changed original request usage");
  require(executor.requests.empty() &&
              resources.snapshot().pool.pagesActive == 0 &&
              resources.snapshot().activeRequests == 0,
          "long replay leaked active resources");
}

void testPreemptedDecodeRestoresItsResidentCompositeState() {
  Backing backing(6);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  for (uint64_t id : {260, 261}) {
    auto value = request(id, std::vector<uint32_t>(65, id));
    value.maxNewTokens = 30;
    engine.submit(std::move(value));
  }
  for (double now = 1; now < 300 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));
  require(engine.idle() && executor.suspensions == 1 &&
              executor.resumptions == 1 && executor.restored == 64 &&
              executor.resumedPrompts.size() == 1 &&
              executor.resumedPrompts.front().size() == 89 &&
              engine.snapshot().resourceReplayTokens == 25,
          "preempted decode did not use its cached state plus exact suffix");
  require(events.completedCount == 2 && events.failedCount == 0 &&
              events.capacityExhaustedCount == 0 && events.startIds.size() == 2 &&
              events.outputs.at(260) == std::vector<uint32_t>(30, 42) &&
              events.outputs.at(261) == std::vector<uint32_t>(30, 42) &&
              events.usage.at(260) == std::pair<uint32_t, uint32_t>{65, 30} &&
              engine.snapshot().resources.lookup.lookups == 2 &&
              engine.snapshot().cacheHits == 0 &&
              engine.snapshot().coldMisses == 2,
          "internal cache restore changed output or request accounting");
}

void testRepeatedPreemptionRespectsBackoffAndCancellation() {
  for (bool cancel : {false, true}) {
    Backing backing(4);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    executor.physicalGrowthBlocked = &backing.growthBlocked;
    executor.unblockGrowthOnSuspend = false;
    Events events;
    EngineConfig config;
    config.growthPaused = [] { return true; };
    engine::Engine engine(config, resources, executor, events);
    backing.growthBlocked = true;
    auto value = request(270, {270});
    value.deadlineMilliseconds = 350;
    engine.submit(std::move(value));
    require(engine.tick(1) && executor.suspensions == 1,
            "singleton host pressure did not preempt safely");
    require(!engine.tick(50) && !engine.tick(101) &&
                executor.resumeAttempts == 1 &&
                !engine.tick(150) && !engine.tick(201) &&
                executor.resumeAttempts == 2 &&
                engine.snapshot().resourceSuspensions == 1 &&
                engine.snapshot().resourceResumptions == 0 &&
                executor.prefillRows == 0,
            "failed recovery replayed work or ignored retry backoff");
    require(engine.nextWakeupMilliseconds() == 301 &&
                events.startIds.size() == 1 &&
                engine.snapshot().resources.lookup.lookups == 1 &&
                resources.snapshot().pool.pagesActive == 0 &&
                resources.snapshot().activeRequests == 0,
            "repeated preemption retained KV or duplicated admission counters");
    if (cancel) {
      engine.cancel(270);
      static_cast<void>(engine.tick(202));
    } else {
      static_cast<void>(engine.tick(350));
    }
    require(engine.idle() && executor.requests.empty() &&
                events.completedCount == (cancel ? 1U : 0U) &&
                events.failedCount == (cancel ? 0U : 1U) &&
                events.capacityExhaustedCount == 0 && events.emitted == 0,
            "repeated preemption ignored its terminal deadline/cancel");
  }
}

void testStateAdmissionWaitsForKvRelease() {
  enum class Outcome { Recover, CompleteOnPoll, NoCapacity, Cancel, Timeout };
  for (uint32_t pageCount : {4U, 8U}) {
    for (auto outcome : {Outcome::Recover, Outcome::CompleteOnPoll,
                         Outcome::NoCapacity, Outcome::Cancel, Outcome::Timeout}) {
      if (pageCount != 4 && outcome == Outcome::CompleteOnPoll)
        continue;
      Backing backing(pageCount);
      KvPool pool(backing);
      engine::Cache cache(pool, CacheNamespace{});
      auto pages = pool.acquirePages(pageCount, false);
      require(pages.granted(), "could not seed resident KV backing");
      for (uint32_t page : pages.pages)
        pool.releasePage(page, false);
      backing.deferRelease = true;
      backing.completeReleaseOnPoll = outcome == Outcome::CompleteOnPoll;
      Executor executor(1);
      executor.beginGrowthBlocked = [&] {
        return pool.snapshot().pagesResident != 0 || backing.releasePending ||
               outcome == Outcome::NoCapacity;
      };
      executor.beginAllocationFailure = metal::AllocationFailure::EngineBudget;
      Events events;
      EngineConfig config;
      config.resourceWaitTimeoutMilliseconds = 500;
      engine::Engine engine(config, cache, executor, events);
      engine.submit(request(284, {284}));
      static_cast<void>(engine.tick(1));
      require(!engine.idle() && events.failedCount == 0 &&
                  executor.prefillRows == 0,
              "in-flight KV release became terminal state admission failure");
      const uint32_t attempts = executor.beginAttempts;
      static_cast<void>(engine.tick(50));
      require(executor.beginAttempts == attempts,
              "pending release bypassed resource retry backoff");
      if (outcome == Outcome::Recover || outcome == Outcome::CompleteOnPoll ||
          outcome == Outcome::NoCapacity) {
        backing.releasePending = false;
        backing.deferRelease = false;
        for (double now = 101; now < 400 && !engine.idle(); ++now)
          static_cast<void>(engine.tick(now));
        if (outcome == Outcome::NoCapacity)
          require(events.failures == std::vector<std::string>{"capacity_exhausted"},
                  "true capacity exhaustion kept waiting after reclamation");
        else
          require(events.completedCount == 1 && events.failedCount == 0,
                  "state admission did not resume after KV release");
      } else if (outcome == Outcome::Cancel) {
        engine.cancel(284);
        static_cast<void>(engine.tick(51));
      } else {
        static_cast<void>(engine.tick(502));
        require(events.failures == std::vector<std::string>{"resource_timeout"},
                "pending release bypassed resource wait deadline");
      }
      require(engine.idle() && cache.snapshot().activeRequests == 0 &&
                  executor.requests.empty(),
              "pending-release state admission leaked ownership");
    }
  }
}

void testAdmissionsWaitForBackgroundRelease() {
  enum class Outcome { Recover, CompleteOnPoll, NoCapacity, Cancel, Timeout };
  enum class Path { State, RunningKv, ResumeKv };
  for (auto path : {Path::State, Path::RunningKv, Path::ResumeKv}) {
    const bool stateAdmission = path == Path::State;
    const bool resuming = path == Path::ResumeKv;
    for (auto outcome : {Outcome::Recover, Outcome::CompleteOnPoll,
                         Outcome::NoCapacity, Outcome::Cancel, Outcome::Timeout}) {
      Backing backing(8);
      KvPool pool(backing);
      engine::Cache cache(pool, CacheNamespace{});
      Executor executor(1);
      Events events;
      EngineConfig config;
      config.resourceWaitTimeoutMilliseconds = 500;
      engine::Engine engine(config, cache, executor, events);
      engine.submit(request(285, {285}));
      // The sole lane backs off at its first denial and suspends at the
      // resource wait bound.
      const double suspended = 1.0 + config.resourceWaitTimeoutMilliseconds;
      if (resuming) {
        backing.growthBlocked = true;
        backing.allocationFailure = metal::AllocationFailure::HostPressure;
        static_cast<void>(engine.tick(1));
        static_cast<void>(engine.tick(suspended));
        require(executor.suspensions == 1, "fixture did not suspend request");
        backing.growthBlocked = false;
      }
      auto pages = pool.acquirePages(4, false);
      require(pages.granted(), "could not seed background KV release");
      for (uint32_t page : pages.pages)
        pool.releasePage(page, false);
      backing.deferRelease = true;
      require(pool.reclaimEmptyExtents(false) == 1,
              "fixture did not start background release");
      backing.completeReleaseOnPoll = outcome == Outcome::CompleteOnPoll;
      backing.allocationFailure = metal::AllocationFailure::EngineBudget;
      backing.growthAllowed = [&] {
        return !backing.releasePending && outcome != Outcome::NoCapacity;
      };
      if (stateAdmission) {
        executor.beginAllocationFailure = metal::AllocationFailure::EngineBudget;
        executor.beginGrowthBlocked = [&] { return !backing.growthAllowed(); };
      }
      const double start = resuming ? suspended + 100 : 1;
      static_cast<void>(engine.tick(start));
      require(events.failedCount == 0 && events.capacityExhaustedCount == 0,
              "pending/just-completed release became terminal KV failure");
      if (outcome != Outcome::CompleteOnPoll) {
        const auto attempts = backing.growthAttempts + executor.beginAttempts;
        static_cast<void>(engine.tick(start + 50));
        require(backing.growthAttempts + executor.beginAttempts == attempts,
                "KV release wait bypassed retry backoff");
      }
      if (outcome == Outcome::Cancel) {
        engine.cancel(285);
        static_cast<void>(engine.tick(start + 51));
        require(events.completedCount == 1, "KV release wait ignored cancel");
      } else if (outcome == Outcome::Timeout) {
        static_cast<void>(engine.tick(start + 501));
        require(events.failures == std::vector<std::string>{"resource_timeout"},
                "KV release wait ignored timeout");
      } else {
        backing.releasePending = false;
        for (double now = start + 100; now < start + 400 && !engine.idle(); ++now)
          static_cast<void>(engine.tick(now));
        if (outcome == Outcome::NoCapacity)
          require(events.failures == std::vector<std::string>{"capacity_exhausted"},
                  "true KV exhaustion kept waiting after release");
        else
          require(events.completedCount == 1 && events.failedCount == 0,
                  "KV allocation did not recover after release");
      }
      require(engine.idle() && executor.requests.empty() &&
                  cache.snapshot().activeRequests == 0 &&
                  cache.snapshot().pool.pagesActive == 0,
              "KV release wait leaked request ownership");
    }
  }
}

void testAllocationCausesRemainRetryableAndDistinct() {
  for (bool stateAllocation : {false, true}) {
    for (auto reason : {metal::AllocationFailure::HostPressure,
                        metal::AllocationFailure::EngineBudget,
                        metal::AllocationFailure::DriverRejected}) {
      Backing backing(8);
      KvPool pool(backing);
      engine::Cache cache(pool, CacheNamespace{});
      Executor executor(1);
      Events events;
      engine::Engine engine({}, cache, executor, events);
      backing.growthBlocked = !stateAllocation;
      backing.allocationFailure = reason;
      executor.beginGrowthBlocked = [stateAllocation] { return stateAllocation; };
      executor.beginAllocationFailure = reason;
      engine.submit(request(285, {285}));
      static_cast<void>(engine.tick(1));
      if (reason == metal::AllocationFailure::HostPressure) {
        require(!engine.idle() && events.failedCount == 0 &&
                    executor.prefillRows == 0,
                "temporary host admission failure became a terminal error");
        engine.cancel(285);
        static_cast<void>(engine.tick(2));
      } else {
        require(engine.idle() && events.failures ==
                    std::vector<std::string>{"capacity_exhausted"} &&
                    events.failureDetails.size() == 1 &&
                    events.failureDetails[0].second &&
                    events.failureDetails[0].first.find(
                        metal::allocationFailureName(reason)) != std::string::npos,
                "allocation failure lost its cause or became nonretryable");
      }
      require(engine.idle() && executor.requests.empty() &&
                  cache.snapshot().activeRequests == 0,
              "allocation denial leaked request ownership");
    }
  }
}

void testRecoveryAdmitsFailedKvTargetBeforeReplaying() {
  // The sole lane backs off at its first denial (tick 3) and suspends at this
  // bound; the suspension's fresh wait outlasts the retries below.
  constexpr double resourceWaitMilliseconds = 1000.0;
  constexpr double suspended = 3.0 + resourceWaitMilliseconds;
  for (bool sharePrefix : {false, true}) {
    Backing backing(64);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(2);
    executor.unblockGrowthOnSuspend = false;
    executor.physicalGrowthBlocked = &backing.growthBlocked;
    Events events;
    EngineConfig config;
    config.resourceWaitTimeoutMilliseconds = resourceWaitMilliseconds;
    // Models the request-sized headroom denial while global pressure is Normal.
    config.growthPaused = [] { return false; };
    engine::Engine engine(config, resources, executor, events);
    const std::vector<uint32_t> prompt(129, 280);
    if (sharePrefix) {
      engine.submit(request(279, std::vector<uint32_t>(65, 280)));
      runUntilIdle(engine);
    }
    backing.allocationFailure = metal::AllocationFailure::HostPressure;
    backing.growthAllowed = [&] { return !backing.growthBlocked; };
    auto input = request(280, prompt);
    input.returnProgress = true;
    engine.submit(std::move(input));
    // First dispatch reaches the page-aligned replay state (128 tokens).
    require(engine.tick(1) && engine.tick(2), "recovery fixture did not prefill");
    const auto cached = resources.snapshot();
    const uint32_t restored = executor.restored;
    {
      auto lookup = resources.lookup(prompt);
      require(lookup.resumeBoundary() == 128 &&
                  cached.pool.pagesResident == cached.pool.pagesActive,
              "recovery fixture needs a state backed only by active prefix pages");
    }
    backing.growthBlocked = true;
    static_cast<void>(engine.tick(3));
    require(engine.snapshot().resourceSuspensions == 0,
            "the sole prefill lane did not back off");
    require(engine.tick(suspended) &&
                engine.snapshot().resourceSuspensions == 1,
            "request-sized host denial was treated as permanent capacity");
    require(resources.snapshot().stateCache.entries == cached.stateCache.entries &&
                resources.snapshot().stateCache.evictions == cached.stateCache.evictions,
            "active prefix pages were mistaken for idle backing and lost their state");
    const uint64_t rows = executor.prefillRows;
    const uint64_t replay = engine.snapshot().resourceReplayTokens;
    for (double now :
         {suspended + 100.0, suspended + 200.0, suspended + 300.0}) {
      require(!engine.tick(now) && !engine.commandInFlight() &&
                  executor.prefillRows == rows && executor.restored == restored &&
                  engine.snapshot().resourceReplayTokens == replay &&
                  engine.snapshot().resourceResumptions == 0 &&
                  resources.snapshot().activeRequests == 0 &&
                  resources.snapshot().pool.pagesActive == 0,
              "insufficient KV replayed committed rows or retained active leases");
    }
    const auto reported = events.progress.at(280);
    require(reported.back() == 128, "prefill recovery lost completed progress");
    backing.growthBlocked = false;
    require(engine.tick(suspended + 400.0),
            "recovery did not resume when KV became available");
    for (double now = suspended + 401.0; now < suspended + 422.0 &&
                                         !engine.idle();
         ++now)
      static_cast<void>(engine.tick(now));
    auto expected = reported;
    expected.push_back(129);
    require(events.progress.at(280) == expected,
            "prefill recovery duplicated or regressed progress");
    require(engine.idle() && engine.snapshot().resourceResumptions == 1 &&
                executor.restored == restored + 128 &&
                engine.snapshot().resourceReplayTokens == 0 &&
                events.outputs.at(280) == std::vector<uint32_t>{42} &&
                resources.snapshot().activeRequests == 0 &&
                resources.snapshot().pool.pagesActive == 0 &&
                events.failedCount == 0 && events.capacityExhaustedCount == 0,
            "KV recovery failed to continue cleanly after headroom recovered");
  }
}

// A Warning pass as production runs it with host headroom left: no request
// is treated as idle (keepResumePoint false) and nothing is evicted wholesale.
MemoryReclaimDirective warningReclaim(uint64_t targetBytes) {
  return {.reclaimEmptyKvExtents = true,
          .evictAllUnpinnedPrefixes = false,
          .targetBytes = targetBytes,
          .keepResumePoint = false};
}

// Request 280 suspended by a request-sized host denial while global pressure
// stays Normal (growthPaused false), after its first dispatch left an ordinary
// replay state at 128 of its 129 prompt tokens. With peerState, request 279
// first leaves an unrelated state at 64 on the same prefix, cancelled past
// its prompt end so no conversation end protects it. The sole lane backs off
// at its first denial and suspends at the resource wait bound.
struct SuspendedResumeFixture final {
  explicit SuspendedResumeFixture(bool peerState = false)
      : pool(backing), resources(pool, CacheNamespace{}), executor(2),
        engine(normalPressureConfig(), resources, executor, events) {
    executor.unblockGrowthOnSuspend = false;
    executor.physicalGrowthBlocked = &backing.growthBlocked;
    if (peerState)
      cancelAfterPromptEnd(engine, resources, 279, peerPrompt);
    backing.allocationFailure = metal::AllocationFailure::HostPressure;
    backing.growthAllowed = [this] { return !backing.growthBlocked; };
    engine.submit(request(280, prompt));
    require(engine.tick(1) && engine.tick(2) &&
                resources.cachedTokens(prompt) == 128,
            "suspension fixture did not publish the 128-token replay state");
    backing.growthBlocked = true;
    static_cast<void>(engine.tick(3));
    require(engine.snapshot().resourceSuspensions == 0,
            "suspension fixture's sole lane did not back off");
    require(engine.tick(3 + kResourceWaitMilliseconds) &&
                engine.snapshot().resourceSuspensions == 1,
            "suspension fixture did not suspend the request");
    restoredBeforeResume = executor.restored;
  }

  static EngineConfig normalPressureConfig() {
    EngineConfig config;
    config.resourceWaitTimeoutMilliseconds = kResourceWaitMilliseconds;
    config.growthPaused = [] { return false; };
    return config;
  }

  // Lets the suspended request resume and run to completion.
  void resume() {
    backing.growthBlocked = false;
    for (double now = 403; now < 425 && !engine.idle(); ++now)
      static_cast<void>(engine.tick(now));
    require(engine.idle() && engine.snapshot().resourceResumptions == 1 &&
                events.completedCount >= 1 && events.failedCount == 0,
            "suspended request did not resume and complete");
  }

  // Long enough that the suspension's fresh wait outlasts resume().
  static constexpr double kResourceWaitMilliseconds = 300.0;
  const std::vector<uint32_t> prompt = std::vector<uint32_t>(129, 280);
  const std::vector<uint32_t> peerPrompt = std::vector<uint32_t>(65, 280);
  Backing backing{64};
  KvPool pool;
  engine::Cache resources;
  Executor executor;
  Events events;
  engine::Engine engine;
  uint32_t restoredBeforeResume = 0;
};

void testWarningReclaimKeepsSuspendedRequestsResumeState() {
  SuspendedResumeFixture fixture;

  static_cast<void>(fixture.engine.reclaimMemory(
      warningReclaim(std::numeric_limits<uint64_t>::max())));
  require(fixture.resources.cachedTokens(fixture.prompt) == 128,
          "warning reclaim evicted a suspended request's resume state");
  fixture.resume();

  // The resume restores the 128-token state and replays nothing: every row
  // below the suspension point comes back from the cache.
  require(fixture.executor.restored - fixture.restoredBeforeResume == 128 &&
              fixture.engine.snapshot().resourceReplayTokens == 0,
          "the resume replayed rows its spared state covers");
}

void testWarningReclaimWalksPastProtectedResumeState() {
  SuspendedResumeFixture fixture(true);
  // Return the empty backing first, so the next target is met by one state.
  static_cast<void>(fixture.engine.reclaimMemory(warningReclaim(0)));
  {
    // Releasing this restore pin makes the peer state the newest one,
    // leaving the protected state oldest in eviction order.
    const CacheLookup peer = fixture.resources.lookup(fixture.peerPrompt);
    require(peer.resumeBoundary() == 64, "peer state was not cached");
  }
  const uint32_t entries = fixture.resources.snapshot().stateCache.entries;

  static_cast<void>(
      fixture.engine.reclaimMemory(warningReclaim(State{}.bytes())));

  require(fixture.resources.snapshot().stateCache.entries == entries - 1 &&
              fixture.resources.cachedTokens(fixture.prompt) == 128 &&
              fixture.resources.cachedTokens(fixture.peerPrompt) == 0,
          "warning reclaim stalled on the protected oldest state");
}

void testCriticalReclaimEvictsSuspendedRequestsResumeState() {
  SuspendedResumeFixture fixture;

  static_cast<void>(fixture.engine.reclaimMemory(
      {.reclaimEmptyKvExtents = true,
       .evictAllUnpinnedPrefixes = true,
       .targetBytes = std::numeric_limits<uint64_t>::max()}));
  require(fixture.resources.snapshot().stateCache.entries == 0,
          "critical reclaim spared a suspended request's resume state");
  fixture.resume();

  require(fixture.executor.restored == fixture.restoredBeforeResume &&
              fixture.engine.snapshot().resourceReplayTokens == 128,
          "a resume without its state did not replay the processed prefix");
}

void testCancelledSuspendedRequestReleasesItsResumeExemption() {
  SuspendedResumeFixture fixture;

  fixture.engine.cancel(280);
  runUntilIdle(fixture.engine);
  static_cast<void>(fixture.engine.reclaimMemory(
      warningReclaim(std::numeric_limits<uint64_t>::max())));

  require(fixture.resources.cachedTokens(fixture.prompt) == 0,
          "a cancelled request's resume state stayed exempt from reclaim");
}

void testSuspendedLanesSharingAResumeStateKeepItUntilBothRelease() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  executor.unblockGrowthOnSuspend = false;
  executor.physicalGrowthBlocked = &backing.growthBlocked;
  Events events;
  engine::Engine engine(SuspendedResumeFixture::normalPressureConfig(),
                        resources, executor, events);
  const std::vector<uint32_t> shared(129, 290);
  // Cancelled past its prompt end: only the two lanes protect the state.
  cancelAfterPromptEnd(engine, resources, 289, shared);
  require(resources.cachedTokens(shared) == 128,
          "shared-state fixture did not cache the 128-token state");
  // Unmap the empty extent the finished request left, so neither lane can
  // prefill past the shared state before it is suspended.
  static_cast<void>(engine.reclaimMemory(warningReclaim(0)));
  require(resources.snapshot().pool.pagesFreeResident == 0,
          "shared-state fixture left free resident pages");
  backing.allocationFailure = metal::AllocationFailure::HostPressure;
  backing.growthAllowed = [&] { return !backing.growthBlocked; };
  backing.growthBlocked = true;
  // Both lanes restore the shared state, then need a page that cannot map.
  for (uint64_t id : {290, 291}) {
    std::vector<uint32_t> prompt = shared;
    prompt.insert(prompt.end(), 128, static_cast<uint32_t>(id));
    engine.submit(request(id, prompt));
  }
  for (double now = 1;
       now < 16 && engine.snapshot().resourceSuspensions < 2; ++now)
    static_cast<void>(engine.tick(now));
  require(engine.snapshot().resourceSuspensions == 2 &&
              executor.restored == 256,
          "shared-state fixture did not suspend two lanes on one state");
  const MemoryReclaimDirective warning =
      warningReclaim(std::numeric_limits<uint64_t>::max());

  static_cast<void>(engine.reclaimMemory(warning));
  require(resources.cachedTokens(shared) == 128 &&
              resources.snapshot().stateCache.entries == 1,
          "warning reclaim evicted a state two suspended lanes resume from");
  engine.cancel(290);
  static_cast<void>(engine.reclaimMemory(warning));
  require(resources.cachedTokens(shared) == 128,
          "one lane's release dropped the state its peer still resumes from");
  engine.cancel(291);
  static_cast<void>(engine.reclaimMemory(warning));
  require(resources.cachedTokens(shared) == 0,
          "the shared state stayed exempt after both lanes released it");
}

// Request 301 queued for memory by a host-pressure begin denial while growth
// is paused, after request 300 left a composite state at 64 of their shared
// 65-token prompt, cancelled past its prompt end so no conversation end
// protects it. A host denial skips admit's growth reclaim, so the state
// survives until the reclaim under test.
struct QueuedPrefixFixture final {
  QueuedPrefixFixture()
      : pool(backing), resources(pool, CacheNamespace{}), executor(1),
        engine(pausedWhileBlocked(), resources, executor, events) {
    cancelAfterPromptEnd(engine, resources, 300, prompt);
    blocked = true;
    executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
    executor.beginGrowthBlocked = [this] { return blocked; };
    engine.submit(request(301, prompt));
    static_cast<void>(engine.tick(1));
    require(engine.resourceWaitSnapshot(1).memory == 1,
            "queued-prefix fixture did not queue the request for memory");
    evictionsBefore = resources.snapshot().stateCache.evictions;
  }

  EngineConfig pausedWhileBlocked() {
    EngineConfig config;
    config.growthPaused = [this] { return blocked; };
    return config;
  }

  // Ticks past the retry backoff until the queued request is admitted.
  void admit() {
    for (double now = 2; now < 1000 && events.starts.size() < 2; ++now)
      static_cast<void>(engine.tick(now));
    require(events.starts.size() == 2, "the queued request was not admitted");
  }

  bool blocked = false;
  const std::vector<uint32_t> prompt = std::vector<uint32_t>(65, 300);
  Backing backing{16};
  KvPool pool;
  engine::Cache resources;
  Executor executor;
  Events events;
  engine::Engine engine;
  uint64_t evictionsBefore = 0;
};

void testWarningReclaimKeepsQueuedRequestsPrefix() {
  QueuedPrefixFixture fixture;

  static_cast<void>(fixture.engine.reclaimMemory(
      warningReclaim(std::numeric_limits<uint64_t>::max())));
  fixture.blocked = false;
  fixture.admit();

  require(fixture.events.starts.back().first == EngineCacheStatus::PrefixHit &&
              fixture.events.starts.back().second == 64 &&
              fixture.resources.snapshot().stateCache.evictions ==
                  fixture.evictionsBefore,
          "warning reclaim evicted a queued request's cached prefix");
}

void testRetriedQueuedRequestHoldsOneProtection() {
  QueuedPrefixFixture fixture;
  const uint32_t attempts = fixture.executor.beginAttempts;
  for (double now = 2;
       now < 1000 && fixture.executor.beginAttempts == attempts; ++now)
    static_cast<void>(fixture.engine.tick(now));
  require(fixture.executor.beginAttempts == attempts + 1 &&
              fixture.engine.resourceWaitSnapshot(1000).memory == 1,
          "the queued request was not denied a second time");

  fixture.engine.cancel(301);
  static_cast<void>(fixture.engine.reclaimMemory(
      warningReclaim(std::numeric_limits<uint64_t>::max())));

  require(fixture.resources.cachedTokens(fixture.prompt) == 0,
          "a twice-denied request left a protection behind on cancel");
}

void testQueuedRequestAdmittedColdDropsItsProtection() {
  QueuedPrefixFixture fixture;
  // The retry is denied while its own lookup pins the state, so admission
  // drops the pin; idle model state freed only then admits it cold with the
  // found state still cached. The prefill stays in flight meanwhile.
  const auto unpinned = [&] {
    return fixture.resources.snapshot().stateCache.pinned == 0;
  };
  fixture.executor.beginAllocationFailure = metal::AllocationFailure::None;
  fixture.executor.beginGrowthBlocked = [unpinned] { return !unpinned(); };
  fixture.executor.reclaimableIdleStateBytes = 1;
  fixture.executor.idleStateReclaimable = unpinned;
  fixture.executor.holdPrefillUntil = std::make_shared<bool>(false);
  fixture.blocked = false;
  fixture.admit();
  require(fixture.events.starts.back().first == EngineCacheStatus::Miss &&
              fixture.resources.cachedTokens(fixture.prompt) == 64,
          "the queued request did not start cold beside its found state");

  static_cast<void>(fixture.engine.reclaimMemory(
      warningReclaim(std::numeric_limits<uint64_t>::max())));

  require(fixture.resources.snapshot().stateCache.evictions ==
              fixture.evictionsBefore + 1,
          "a request admitted cold kept protecting the prefix it found");
  *fixture.executor.holdPrefillUntil = true;
  runUntilIdle(fixture.engine);
}

void testGrowthReclaimTakesAQueuedPrefixRatherThanStall() {
  QueuedPrefixFixture fixture;
  // Admission fits only once the queued request's own protected state is
  // gone: growth reclaim must fall back to it.
  fixture.executor.beginAllocationFailure = metal::AllocationFailure::None;
  fixture.executor.beginGrowthBlocked = [&] {
    return fixture.resources.snapshot().stateCache.entries != 0;
  };
  fixture.blocked = false;
  fixture.admit();

  require(fixture.events.starts.back().first == EngineCacheStatus::Miss &&
              fixture.resources.snapshot().stateCache.evictions ==
                  fixture.evictionsBefore + 1,
          "growth reclaim stalled on the queued request's protected state");
  runUntilIdle(fixture.engine);
  require(fixture.events.completedCount == 2 &&
              fixture.events.failedCount == 0,
          "the queued request did not complete after growth reclaim");
}

// A sole request is denied by the engine budget while its lookup pins the
// state it found at 64. Cached KV of another prompt, held by a cache-only
// request, turns evictable during the first reclaim after the lease drop,
// newer than the unpinned state; one eviction then lifts the denial. Growth
// reclaim must take that KV, not the found state, and admit cold.
void testGrowthReclaimSparesAFoundStateForCachedKv() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(65, 310);
  cachePrefix(resources, 999, std::span(prompt).first(64), 64);
  constexpr uint64_t kHolderId = 998;
  const std::vector<uint32_t> other(64, 998);
  resources.beginRequest(kHolderId);
  require(resources.ensureTokens(kHolderId, other.size()).granted(),
          "the KV holder's pages were not acquired");
  static_cast<void>(resources.publishCommittedBlocks(
      kHolderId, other, static_cast<uint32_t>(other.size())));
  // The model's idle-state reclaim opens every growth reclaim: the first one
  // after the lease drop ends the holder, leaving its KV the only other
  // candidate.
  bool holding = true;
  executor.reclaimableIdleStateBytes = 1;
  executor.idleStateReclaimable = [&] {
    if (holding && executor.beginAttempts &&
        resources.snapshot().stateCache.pinned == 0) {
      resources.endRequest(kHolderId);
      holding = false;
    }
    return false;
  };
  const CacheSnapshot before = resources.snapshot();
  executor.beginAllocationFailure = metal::AllocationFailure::EngineBudget;
  executor.beginGrowthBlocked = [&] {
    const CacheSnapshot now = resources.snapshot();
    return now.kvCache.evictions == before.kvCache.evictions &&
           now.stateCache.evictions == before.stateCache.evictions;
  };
  executor.holdPrefillUntil = std::make_shared<bool>(false);

  engine.submit(request(310, prompt));
  static_cast<void>(engine.tick(1));
  require(events.starts.size() == 1 &&
              events.starts.back().first == EngineCacheStatus::Miss &&
              resources.cachedTokens(prompt) == 64,
          "growth reclaim took the found state before another KV leaf");
  engine.cancel(310);
  *executor.holdPrefillUntil = true;
  runUntilIdle(engine);
  engine.submit(request(311, prompt));
  runUntilIdle(engine);

  require(events.starts.back().first == EngineCacheStatus::PrefixHit &&
              events.starts.back().second == 64,
          "a resend did not hit the state the denied request found");
}

// Request 301 queued for memory by a non-host begin denial while request 302
// stays resident, decoding, beside the composite state request 300 left at
// 64 of their shared 65-token prompt. Growth is never paused, so only the
// resident peer can keep admission from dropping the found state's lease.
struct ResidentPeerQueueFixture final {
  explicit ResidentPeerQueueFixture(double resourceWaitTimeoutMilliseconds)
      : pool(backing), resources(pool, CacheNamespace{}), executor(2),
        engine(configured(resourceWaitTimeoutMilliseconds), resources,
               executor, events) {
    engine.submit(request(300, prompt));
    runUntilIdle(engine);
    require(resources.cachedTokens(prompt) == 64,
            "resident-peer fixture did not cache the 64-token state");
    executor.decodeFinishes = false;
    auto peer = request(302, {302});
    peer.maxNewTokens = kPeerTokens;
    peer.deadlineMilliseconds = kFarDeadlineMilliseconds;
    engine.submit(std::move(peer));
    static_cast<void>(engine.tick(now));
    require(events.startIds.back() == 302, "the peer lane was not admitted");
  }

  static EngineConfig configured(double resourceWaitTimeoutMilliseconds) {
    EngineConfig config;
    config.resourceWaitTimeoutMilliseconds = resourceWaitTimeoutMilliseconds;
    config.growthPaused = [] { return false; };
    return config;
  }

  // Submits request 301 under a begin denial `blocked` decides, and ticks
  // through its first admission attempt.
  void queue(std::function<bool()> blocked) {
    executor.beginAllocationFailure = metal::AllocationFailure::None;
    executor.beginGrowthBlocked = std::move(blocked);
    auto queued = request(301, prompt);
    queued.deadlineMilliseconds = kFarDeadlineMilliseconds;
    engine.submit(std::move(queued));
    // The queued request reaches admission behind the peer's decode step.
    const uint32_t attemptsBefore = executor.beginAttempts;
    for (uint32_t step = 0;
         step < kAdmissionTicks && executor.beginAttempts == attemptsBefore;
         ++step)
      static_cast<void>(engine.tick(++now));
    require(engine.resourceWaitSnapshot(now).memory == 1,
            "resident-peer fixture did not queue the request for memory");
  }

  bool started(uint64_t id) const {
    return std::find(events.startIds.begin(), events.startIds.end(), id) !=
           events.startIds.end();
  }

  static constexpr uint32_t kPeerTokens = 100000;
  static constexpr double kFarDeadlineMilliseconds = 1.0e9;
  static constexpr uint32_t kAdmissionTicks = 10;
  const std::vector<uint32_t> prompt = std::vector<uint32_t>(65, 300);
  Backing backing{16};
  KvPool pool;
  engine::Cache resources;
  Executor executor;
  Events events;
  engine::Engine engine;
  double now = 100;
};

void testQueuedRequestKeepsItsFoundStateWhileAPeerIsResident() {
  // Each retry tick lands just past the 100 ms resource retry backoff.
  constexpr double kRetryStrideMilliseconds = 101.0;
  constexpr uint32_t kDeniedRetries = 5;
  ResidentPeerQueueFixture fixture(30000.0);
  bool blocked = true;
  fixture.queue([&] { return blocked; });
  require(fixture.resources.cachedTokens(fixture.prompt) == 64,
          "the first denial evicted the state a peer-blocked request found");

  for (uint32_t retry = 0; retry < kDeniedRetries; ++retry) {
    fixture.now += kRetryStrideMilliseconds;
    static_cast<void>(fixture.engine.tick(fixture.now));
    require(fixture.resources.cachedTokens(fixture.prompt) == 64,
            "a retried denial evicted the state a peer-blocked request found");
  }
  blocked = false;
  for (uint32_t step = 0;
       step < ResidentPeerQueueFixture::kAdmissionTicks &&
       !fixture.started(301);
       ++step)
    static_cast<void>(
        fixture.engine.tick(fixture.now += kRetryStrideMilliseconds));

  require(fixture.started(301) &&
              fixture.events.starts.back().first ==
                  EngineCacheStatus::PrefixHit &&
              fixture.events.starts.back().second == 64,
          "a peer-blocked request did not restore the state it found");
}

void testQueuedRequestOutlivedByAPeerAdmitsCold() {
  constexpr double kTimeoutMilliseconds = 1000.0;
  constexpr double kStrideMilliseconds = 150.0;
  ResidentPeerQueueFixture fixture(kTimeoutMilliseconds);
  const auto pinned = [&] {
    return fixture.resources.snapshot().stateCache.pinned != 0;
  };
  fixture.queue(pinned);

  // The peer stays resident past the queued request's wait deadline.
  const double deadline = fixture.now + kTimeoutMilliseconds;
  while (fixture.now < deadline + kStrideMilliseconds)
    static_cast<void>(fixture.engine.tick(fixture.now += kStrideMilliseconds));

  require(fixture.started(301) &&
              fixture.events.starts.back().first == EngineCacheStatus::Miss &&
              fixture.events.failures.empty() &&
              fixture.executor.requests.contains(302),
          "a request outlived by a resident peer did not admit cold");
}

void testAdmissionReopensAfterLastSuspendedRequestResumes() {
  Backing backing(32);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  executor.decodeFinishes = false;
  Events events;
  MemoryPressure pressure = MemoryPressure::Warning;
  EngineConfig config;
  config.growthPaused = [&] { return pressure != MemoryPressure::Normal; };
  engine::Engine engine(config, resources, executor, events);

  backing.growthBlocked = true;
  auto longRequest = request(271, {271});
  longRequest.maxNewTokens = 1000;
  engine.submit(std::move(longRequest));
  require(engine.tick(1) && executor.suspensions == 1,
          "recovery admission fixture did not suspend its only lane");
  pressure = MemoryPressure::Normal;
  backing.growthBlocked = false;
  require(engine.tick(101) && engine.tick(102) && executor.resumptions == 1,
          "recovery admission fixture did not resume its last waiting lane");

  engine.submit(request(272, {272}));
  for (double now = 103; now < 115 && events.startIds.size() < 2; ++now)
    static_cast<void>(engine.tick(now));
  require(events.startIds == std::vector<uint64_t>({271, 272}) &&
              executor.requests.contains(271) &&
              engine.snapshot().resourceSuspensions == 1,
          "finished resource recovery blocked new work until decode completed");
  engine.cancel(271);
  engine.cancel(272);
  for (double now = 115; now < 125 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));
  require(engine.idle() && resources.snapshot().activeRequests == 0,
          "recovery admission fixture did not release its lanes");
}

void testAdmissionRespectsPriorityBeforeHashOrder() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  EngineRequest background = request(100, {1});
  background.priority = RequestPriority::Background;
  EngineRequest foreground = request(1, {2});
  foreground.priority = RequestPriority::Foreground;
  engine.submit(std::move(background));
  engine.submit(std::move(foreground));
  require(engine.tick(2), "priority admission made no progress");
  require(!events.startIds.empty() && events.startIds.front() == 1,
          "unordered request storage admitted background work first");
  runUntilIdle(engine);
}

void advanceToOverlappedVerify(engine::Engine &engine, Executor &executor,
                               Events &events, uint64_t requestId) {
  require(engine.tick(1) && engine.tick(2) && engine.tick(3) && engine.tick(4),
          "constrained request did not reach its initial mask");
  require(events.maskRequests.size() == 1 &&
              events.maskRequests[0].first == requestId &&
              events.maskRequests[0].second.empty(),
          "initial constrained mask request is malformed");
  const std::array<uint32_t, 1> initialMask{1};
  engine.provideMask(requestId, initialMask);
  require(engine.tick(5) && engine.tick(6),
          "constrained request did not launch overlapped verification");
  require(engine.commandInFlight() && executor.overlap &&
              events.maskRequests.size() == 2 &&
              events.maskRequests[1].first == requestId &&
              events.maskRequests[1].second.size() == 8,
          "target forward did not retain its batch while requesting a mask");
}

EngineRequest constrainedRequest(uint64_t id, double deadline = 10'000.0) {
  EngineRequest value = request(id, {1});
  value.maxNewTokens = 2;
  value.cohort = BatchCohort::Constrained;
  value.constraint = ConstraintMode::TokenMask;
  value.deadlineMilliseconds = deadline;
  return value;
}

void testConstraintMaskOverlapsInsideOneSchedulerBatch() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  engine.submit(constrainedRequest(200));
  advanceToOverlappedVerify(engine, executor, events, 200);
  const std::array<uint32_t, 1> verifyMask{1};
  engine.provideMask(200, verifyMask);
  require(engine.tick(7) && engine.idle() && events.completedCount == 1 &&
              events.failedCount == 0 && events.emitted == 1,
          "overlapped verify mask did not complete the owning batch");
}

void testConstraintMaskWaitHonorsCancelAndDeadline() {
  {
    Backing backing(8);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    Events events;
    engine::Engine engine({}, resources, executor, events);
    engine.submit(constrainedRequest(201));
    advanceToOverlappedVerify(engine, executor, events, 201);
    engine.cancel(201);
    const std::array<uint32_t, 1> lateMask{1};
    engine.provideMask(201, lateMask);
    engine.failRequest(201, "invalid_mask_response", "late mask");
    require(executor.overlap->abandoned && engine.tick(7) && engine.idle() &&
                events.completedCount == 1 && events.failedCount == 0 &&
                events.emitted == 0,
            "cancelled mask wait left an active scheduler batch");
  }

  {
    Backing backing(8);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    Events events;
    engine::Engine engine({}, resources, executor, events);
    engine.submit(constrainedRequest(202, 100.0));
    advanceToOverlappedVerify(engine, executor, events, 202);
    require(engine.tick(100.0) && executor.overlap->abandoned &&
                engine.idle() && events.failedCount == 1,
            "deadline did not terminate an in-flight host mask wait");
  }
}

void testDecodeNearContextCeilingCoversVerifyRows() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  executor.decodeFinishes = false;
  Events events;
  EngineConfig config;
  config.maxContext = 64;
  engine::Engine engine(config, resources, executor, events);

  // prompt + max_tokens fill a context that is a whole number of pages; the
  // last decode cycles store verify rows past the logical ceiling.
  EngineRequest value = request(10, std::vector<uint32_t>(57, 10));
  value.maxNewTokens = 7;
  engine.submit(value);
  runUntilIdle(engine);
  require(events.completedCount == 1 && events.failedCount == 0 &&
              events.emitted == 7,
          "decode at the context ceiling lost its verify-row page coverage");
}

void testExpiredMaskWaitFinalizesWhileAnotherCommandRuns() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);

  engine.submit(constrainedRequest(300, 100.0));
  require(engine.tick(1) && engine.tick(2) && engine.tick(3) && engine.tick(4),
          "constrained request did not reach its initial mask wait");
  require(events.maskRequests.size() == 1, "initial mask was not requested");

  executor.holdDecodeUntil = std::make_shared<bool>(false);
  engine.submit(request(301, {2}));
  require(engine.tick(5) && engine.tick(6) && engine.tick(7) &&
              engine.commandInFlight(),
          "peer decode command was not held in flight");

  // The waiting request expires while the peer command is still running. It
  // must be finalized now; its late mask then finds no live request.
  require(engine.tick(150.0) && events.failedCount == 1 &&
              engine.commandInFlight(),
          "expired mask wait was not finalized behind an in-flight command");
  *executor.holdDecodeUntil = true;
  for (double now = 151.0; now < 160.0 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));
  require(engine.idle() && events.completedCount == 1 &&
              events.failedCount == 1,
          "peer request did not complete after the expired wait was removed");
}

void testOrdinaryInFlightDeadlineDrainsWithoutPublishingOrOutput() {
  for (WorkKind heldKind : {WorkKind::Prefill, WorkKind::Decode}) {
    Backing backing(8);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    Events events;
    engine::Engine engine({}, resources, executor, events);
    auto released = std::make_shared<bool>(false);
    if (heldKind == WorkKind::Prefill)
      executor.holdPrefillUntil = released;
    else
      executor.holdDecodeUntil = released;
    auto value = request(310, std::vector<uint32_t>(65, 310));
    value.deadlineMilliseconds = 100;
    engine.submit(std::move(value));
    const uint32_t launchTick = heldKind == WorkKind::Prefill ? 1 : 5;
    for (uint32_t tick = 1; tick <= launchTick; ++tick)
      require(engine.tick(tick), "request did not launch the held command");
    require(engine.commandInFlight(), "test command did not remain in flight");
    const auto before = resources.snapshot();
    require(engine.tick(100) && engine.commandInFlight() &&
                executor.requests.size() == 1 && events.failedCount == 0 &&
                resources.snapshot().pool.pagesActive == before.pool.pagesActive,
            "ordinary deadline released resources owned by in-flight Metal");
    require(engine.nextWakeupMilliseconds() == 1100.0 && !engine.tick(101),
            "expired in-flight request lost its bounded health wakeup");
    engine.cancel(310); // A later cancel must not replace the deadline failure.
    *released = true;
    // A held prefill expires at its armed boundary and never snapshots; a
    // held decode follows the prefill that already published at 64.
    const uint32_t expectedSnapshots = heldKind == WorkKind::Prefill ? 0 : 1;
    require(engine.tick(102) && engine.idle() && events.failedCount == 1 &&
                events.failures.front() == "deadline_exceeded" &&
                events.completedCount == 0 && events.emitted == 0 &&
                executor.snapshotAttempts == expectedSnapshots &&
                executor.requests.empty(),
            "expired ordinary command emitted output or lost its deadline");
    const auto after = resources.snapshot();
    require(after.stateCache.publications == before.stateCache.publications &&
                after.kvCache.blocks == before.kvCache.blocks &&
                after.pool.pagesActive == 0 && after.activeRequests == 0,
            "expired ordinary command published cache state or leaked resources");
  }
}

void testStalledSuspensionFailsWithCapacity() {
  Backing backing(8);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  executor.physicalGrowthBlocked = &backing.growthBlocked;
  executor.unblockGrowthOnSuspend = false;
  Events events;
  engine::Engine engine({}, resources, executor, events);

  backing.growthBlocked = true;
  engine.submit(request(60, {60}));
  engine.submit(request(61, {61}));
  require(engine.tick(1.0) && engine.tick(2.0) && executor.suspensions == 1 &&
              events.capacityExhaustedCount == 1,
          "preemption did not identify that the remaining singleton cannot fit");

  // Nothing completes between suspension and retry, so suspending again would
  // only rotate the same two lanes forever.
  for (double now = 102.0; now < 140.0 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));
  require(engine.idle() && events.capacityExhaustedCount == 2 &&
              engine.snapshot().resourceSuspensions == 1 &&
              engine.snapshot().resourceResumptions == 0 && executor.prefillRows == 0,
          "stalled suspension did not converge to a capacity failure");
}

void testTerminalAnchorWithoutKvIsNotCached() {
  for (uint32_t withoutKv : {1U, 0U}) {
    Backing backing(8);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    executor.decodeTokensWithoutKv = withoutKv;
    Events events;
    engine::Engine engine({}, resources, executor, events);

    // 31 prompt tokens plus the emitted token complete one Page32 block only
    // when that token has a stored KV row.
    engine.submit(request(70, std::vector<uint32_t>(31, 70)));
    runUntilIdle(engine);
    require(events.completedCount == 1 && events.emitted == 1,
            "terminal anchor request did not complete");
    require(resources.snapshot().kvCache.blocks == (withoutKv ? 0U : 1U),
            "token without a KV row changed the cached block count");
  }
}

void testPrefillCanCompleteTheRequest() {
  for (bool stop : {false, true}) {
    Backing backing(8);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    executor.prefillAnchor = stop;
    Events events;
    engine::Engine engine({}, resources, executor, events);

    EngineRequest value = request(90, std::vector<uint32_t>(40, 90));
    value.maxNewTokens = stop ? 8 : 1;
    engine.submit(std::move(value));
    runUntilIdle(engine);
    require(events.completedCount == 1 && events.emitted == 1 &&
                events.failedCount == 0,
            "request finished by prefill did not complete cleanly");
    require(engine.snapshot().scheduler.decodeBatchesByWidth[0] == 0,
            "request finished by prefill was decoded");
  }
}

const uint32_t defaultCheckpointTokens = EngineConfig{}.prefillCheckpointTokens;

void runUntilCheckpoint(engine::Engine &engine, uint64_t publications) {
  for (uint32_t step = 0; step < 128; ++step) {
    static_cast<void>(engine.tick(step + 1));
    if (engine.snapshot().checkpointPublications >= publications)
      return;
  }
  throw std::runtime_error("engine did not publish the expected checkpoint");
}

void testCancelledColdPrefillResumesItsLatestCheckpoint() {
  Backing backing(2048);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(30001, 19);
  engine.submit(request(400, prompt));
  runUntilCheckpoint(engine, 2);
  require(resources.snapshot().stateCache.entries == 1 &&
              resources.snapshot().stateCache.bytes == 64,
          "cold prefill accumulated superseded progress states");
  engine.cancel(400);
  runUntilIdle(engine);
  const uint32_t computed = executor.prefillRows;
  engine.submit(request(401, prompt));
  runUntilIdle(engine);
  const uint32_t restored = 2 * defaultCheckpointTokens;
  require(events.starts.back().second == restored &&
              executor.prefillRows - computed == prompt.size() - restored &&
              events.failedCount == 0,
          "cancelled cold prefill was recomputed before its completed checkpoint");
  require(resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "successful retry did not keep only its deepest checkpoint");
}

void testConcurrentProgressRetainsAtMostOnePointPerLane() {
  Backing backing(2048);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(25001, 20);
  engine.submit(request(410, prompt));
  engine.submit(request(411, prompt));
  uint32_t maximumEntries = 0;
  for (uint32_t step = 0; step < 128 && !engine.idle(); ++step) {
    static_cast<void>(engine.tick(step + 1));
    maximumEntries =
        std::max(maximumEntries, resources.snapshot().stateCache.entries);
  }
  const uint32_t checkpoints = prompt.size() / defaultCheckpointTokens;
  require(engine.idle() && events.completedCount == 2 &&
              engine.snapshot().checkpointPublications >= checkpoints &&
              engine.snapshot().checkpointPublications <= 2 * checkpoints &&
              engine.snapshot().cacheHits == 1 &&
              executor.prefillRows == prompt.size() * 2 - 24992 &&
              maximumEntries <= 2 &&
              resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "concurrent prompts accumulated progress states beyond their active lanes");
}

void testSharedCheckpointSurvivesPeerRollingReplacement() {
  Backing backing(2048);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  std::vector<uint32_t> original(25001, 20);
  auto background = request(412, original);
  background.priority = RequestPriority::Background;
  engine.submit(std::move(background));
  runUntilCheckpoint(engine, 1);
  std::vector<uint32_t> branch = original;
  std::fill(branch.begin() + defaultCheckpointTokens, branch.end(), 21);
  auto foreground = request(413, branch);
  foreground.priority = RequestPriority::Foreground;
  engine.submit(std::move(foreground));
  runUntilCheckpoint(engine, 2);
  require(resources.lookup(original).resumeBoundary() == defaultCheckpointTokens,
          "rolling a shared checkpoint retired the paused peer's recovery point");
  engine.cancel(412);
  engine.cancel(413);
  runUntilIdle(engine);
  require(events.failedCount == 0, "shared checkpoint cancellation failed");
}

void testRepeatedRetriesRollTheRestoredCheckpoint() {
  Backing backing(2048);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(40001, 31);
  for (uint32_t attempt = 0; attempt < 3; ++attempt) {
    engine.submit(request(500 + attempt, prompt));
    runUntilCheckpoint(engine, attempt + 1);
    require(events.starts.back().second == attempt * defaultCheckpointTokens &&
                resources.snapshot().stateCache.entries == 1 &&
                resources.snapshot().stateCache.checkpointEntries == 1,
            "retry promoted or accumulated intermediate checkpoints");
    engine.cancel(500 + attempt);
    runUntilIdle(engine);
  }
  engine.submit(request(503, prompt));
  runUntilIdle(engine);
  require(events.starts.back().second == 3 * defaultCheckpointTokens &&
              executor.prefillRows == prompt.size() &&
              resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "successful retry recomputed or retained superseded recovery states");
}

void testRetryCancelledBeforeNextCheckpointKeepsItsSource() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(25001, 32);
  engine.submit(request(510, prompt));
  runUntilCheckpoint(engine, 1);
  engine.cancel(510);
  runUntilIdle(engine);
  for (uint32_t id : {511U, 512U}) {
    engine.submit(request(id, prompt));
    require(engine.tick(1) && engine.commandInFlight() &&
                events.starts.back().second == defaultCheckpointTokens,
            "retry did not restore the previous progress point");
    engine.cancel(id);
    runUntilIdle(engine);
    require(resources.lookup(prompt).resumeBoundary() ==
                    defaultCheckpointTokens &&
                resources.snapshot().stateCache.checkpointEntries == 1,
            "cancelled retry lost or promoted its unchanged recovery point");
  }
}

void testRestoredCheckpointAtReplayEndBecomesOrdinary() {
  for (uint32_t suffix : {1U, 31U}) {
    Backing backing(1024);
    KvPool pool(backing);
    engine::Cache resources(pool, CacheNamespace{});
    Executor executor(1);
    Events events;
    engine::Engine engine({}, resources, executor, events);
    const std::vector<uint32_t> prompt(25001, 33);
    engine.submit(request(520, prompt));
    runUntilCheckpoint(engine, 1);
    engine.cancel(520);
    runUntilIdle(engine);
    const uint32_t snapshots = executor.snapshots;
    std::vector<uint32_t> shorter(
        prompt.begin(), prompt.begin() + defaultCheckpointTokens + suffix);
    engine.submit(request(521, shorter));
    runUntilIdle(engine);
    require(events.starts.back().second == defaultCheckpointTokens &&
                executor.snapshots == snapshots &&
                engine.snapshot().deduplicatedStatePublications == 1 &&
                resources.snapshot().stateCache.entries == 1 &&
                resources.snapshot().stateCache.checkpointEntries == 0 &&
                resources.lookup(shorter).resumeBoundary() ==
                    defaultCheckpointTokens,
            "restored replay endpoint was copied or retired as temporary");
  }
}

void testRetryRetiresCheckpointAtDeeperJunction() {
  Backing backing(2048);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({.prefillCheckpointTokens = 8192}, resources, executor,
                        events);
  const std::vector<uint32_t> prompt(30001, 34);
  engine.submit(request(530, prompt));
  runUntilCheckpoint(engine, 2);
  require(engine.tick(50) && engine.tick(51) && !engine.commandInFlight() &&
              executor.requests.at(530).position == 18432,
          "fixture did not commit past the last recovery point");
  engine.cancel(530);
  runUntilIdle(engine);
  engine.submit(request(531, prompt));
  require(engine.tick(1) && engine.tick(2) &&
              events.starts.back().second == 16384 &&
              engine.snapshot().junctionMaterializations == 1 &&
              resources.snapshot().stateCache.entries == 1 &&
              resources.snapshot().stateCache.checkpointEntries == 0,
          "retry kept an earlier recovery point after publishing its junction");
  runUntilIdle(engine);
  require(resources.snapshot().stateCache.entries == 3 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "retry did not retain its junction, replay state and deepest checkpoint");
}

void testPinnedCheckpointSkipsReplacementButNotOrdinaryState() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(18001, 35);
  engine.submit(request(540, prompt));
  runUntilCheckpoint(engine, 1);
  auto pinned = resources.lookup(prompt);
  require(pinned.resumeBoundary() == defaultCheckpointTokens,
          "fixture did not pin its checkpoint");
  runUntilIdle(engine);
  require(engine.snapshot().checkpointPublications == 1 &&
              engine.snapshot().checkpointPublicationFailures ==
                  prompt.size() / defaultCheckpointTokens - 1 &&
              engine.snapshot().checkpointPublicationFailureCauses ==
                  onlyCause(PublicationFailure::RetireBlocked,
                            prompt.size() / defaultCheckpointTokens - 1) &&
              executor.snapshotAttempts == 2 &&
              resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 1 &&
              resources.lookup(prompt).resumeBoundary() == 17984,
          "pinned recovery point was overwritten or blocked ordinary publication");
  pinned = {};
  require(resources.reclaimOneState() &&
              resources.snapshot().stateCache.checkpointEntries == 0 &&
              resources.lookup(prompt).resumeBoundary() == 17984,
          "released recovery pin did not rejoin the lower-priority queue");
}

// A lane's checkpoint at 8192 of a 10241-token prompt, published before
// the lane's next 2048-token chunk.
std::vector<uint32_t> checkpointLanePrompt(uint32_t token) {
  return std::vector<uint32_t>(defaultCheckpointTokens + 2049, token);
}

// Under paused growth the lane's next chunk takes idle cached pages: the
// idle prefix's state and KV go, and the lane's fresh checkpoint stays. The
// idle conversation's end is first pushed out of the ring, so its state is
// the only unprotected one.
void testPausedGrowthReclaimKeepsTheResidentLanesCheckpoint() {
  Backing backing(512);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  bool paused = false;
  EngineConfig config;
  config.growthPaused = [&] { return paused; };
  engine::Engine engine(config, resources, executor, events);
  const std::vector<uint32_t> idle(4097, 600);
  engine.submit(request(600, idle));
  runUntilIdle(engine);
  pushOutConversationEnds(engine, 6000);
  const uint64_t promptEnd = engine.snapshot().replayStatePublications + 1;
  engine.submit(request(601, checkpointLanePrompt(601)));
  runUntilCheckpoint(engine, 1);
  const uint64_t evictions = resources.snapshot().stateCache.evictions;

  paused = true;
  backing.growthBlocked = true;
  backing.allocationFailure = metal::AllocationFailure::HostPressure;
  for (uint32_t step = 0;
       step < 64 && engine.snapshot().replayStatePublications < promptEnd;
       ++step)
    static_cast<void>(engine.tick(step + 1));

  const StateCacheSnapshot states = resources.snapshot().stateCache;
  require(engine.snapshot().replayStatePublications == promptEnd &&
              executor.suspensions == 0 && resources.cachedTokens(idle) == 0 &&
              states.evictions == evictions + 1 &&
              states.checkpointEvictions == 0 &&
              states.checkpointRetirements == 0,
          "paused growth reclaim evicted the resident lane's checkpoint");

  // The decode step's growth fits in the pages the idle prefix left; the
  // lane finishes with its conversation end, the prompt-end state and the
  // checkpoint kept beside it, still cached.
  runUntilIdle(engine);
  const uint32_t ringEnds = model::ExecutionLimits::maximumBatchWidth;
  const StateCacheSnapshot finished = resources.snapshot().stateCache;
  require(events.completedCount == 2 + ringEnds && executor.suspensions == 0 &&
              finished.checkpointEvictions == 0 &&
              finished.checkpointEntries == 1 &&
              finished.entries == 2 + ringEnds,
          "paused growth reclaim took the lane's conversation end");
}

// With the lane's protected checkpoint the only state left, growth reclaim
// for a new lane takes it rather than stall the admission.
void testGrowthReclaimTakesAResidentCheckpointRatherThanStall() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  engine.submit(request(610, checkpointLanePrompt(610)));
  runUntilCheckpoint(engine, 1);
  require(resources.snapshot().stateCache.entries == 1 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "fixture did not leave the lane's checkpoint as the only state");
  executor.holdPrefillUntil = std::make_shared<bool>(false);
  executor.beginGrowthBlocked = [&] {
    return resources.snapshot().stateCache.entries != 0;
  };

  engine.submit(request(611, std::vector<uint32_t>(65, 611)));
  for (double now = 200; now < 1000 && events.starts.size() < 2; ++now)
    static_cast<void>(engine.tick(now));

  require(events.starts.size() == 2 &&
              resources.snapshot().stateCache.checkpointEvictions == 1,
          "growth reclaim stalled on the resident lane's protected checkpoint");
  *executor.holdPrefillUntil = true;
  runUntilIdle(engine);
  require(events.completedCount == 2 && events.failedCount == 0,
          "requests did not complete after taking the protected checkpoint");
}

// A lane suspended by host pressure, once its back-off reaches the resource
// wait bound, resumes from its latest checkpoint and, once finished, keeps
// only its conversation end, protected until pushed out of the ring.
void testHostPressureSuspensionResumesFromTheLanesCheckpoint() {
  Backing backing(512);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  bool paused = false;
  EngineConfig config;
  config.resourceWaitTimeoutMilliseconds = 1000.0;
  config.growthPaused = [&] { return paused; };
  engine::Engine engine(config, resources, executor, events);
  const std::vector<uint32_t> prompt = checkpointLanePrompt(620);
  engine.submit(request(620, prompt));
  runUntilCheckpoint(engine, 1);
  const uint32_t restored = executor.restored;
  paused = true;
  backing.growthBlocked = true;
  backing.allocationFailure = metal::AllocationFailure::HostPressure;
  static_cast<void>(engine.tick(200));
  require(executor.suspensions == 0, "the sole lane did not back off");
  static_cast<void>(engine.tick(1200));
  require(executor.suspensions == 1,
          "host pressure did not suspend the lane");

  static_cast<void>(engine.reclaimMemory(
      warningReclaim(std::numeric_limits<uint64_t>::max())));
  paused = false;
  backing.growthBlocked = false;
  for (double now = 1201; now < 3000 && !engine.idle(); ++now)
    static_cast<void>(engine.tick(now));

  require(engine.idle() && events.completedCount == 1 &&
              executor.restored - restored == defaultCheckpointTokens,
          "suspended lane did not resume from its checkpoint");
  const MemoryReclaimDirective warning =
      warningReclaim(std::numeric_limits<uint64_t>::max());
  static_cast<void>(engine.reclaimMemory(warning));
  require(resources.cachedTokens(prompt) != 0,
          "warning reclaim took a finished lane's conversation end");
  pushOutConversationEnds(engine, 6200);
  static_cast<void>(engine.reclaimMemory(warning));
  require(resources.cachedTokens(prompt) == 0 &&
              resources.snapshot().stateCache.checkpointEntries == 0,
          "a finished lane left a protection behind");
}

// A pinned checkpoint survives the lane's rolling replacement and is kept
// beside the prompt-end state: protected with the lane's conversation end
// while the lane stays resident and once it finishes, and released only when
// that end leaves the ring.
void testOverwrittenCheckpointLeavesNoProtectionBehind() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(18001, 630);
  engine.submit(request(630, prompt));
  runUntilCheckpoint(engine, 1);
  auto pinned = resources.lookup(prompt);
  require(pinned.resumeBoundary() == defaultCheckpointTokens,
          "fixture did not pin its checkpoint");
  for (uint32_t step = 0;
       step < 128 && !engine.snapshot().replayStatePublications; ++step)
    static_cast<void>(engine.tick(200 + step));
  require(engine.snapshot().replayStatePublications == 1 && !engine.idle() &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "fixture did not move the lane past its pinned checkpoint");

  pinned = {};
  const MemoryReclaimDirective warning =
      warningReclaim(std::numeric_limits<uint64_t>::max());
  static_cast<void>(engine.reclaimMemory(warning));
  require(resources.snapshot().stateCache.checkpointEntries == 1,
          "warning reclaim took a resident lane's kept checkpoint");
  runUntilIdle(engine);
  static_cast<void>(engine.reclaimMemory(warning));
  require(resources.snapshot().stateCache.checkpointEntries == 1,
          "warning reclaim took a finished lane's kept checkpoint");

  pushOutConversationEnds(engine, 6300);
  static_cast<void>(engine.reclaimMemory(warning));

  require(resources.snapshot().stateCache.checkpointEntries == 0,
          "an overwritten checkpoint kept the lane's protection");
}

void testFailedReplacementContinuesWithoutRecoveryPoint() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(25001, 36);
  engine.submit(request(550, prompt));
  runUntilCheckpoint(engine, 1);
  executor.denySnapshotAtBoundary = 2 * defaultCheckpointTokens;
  for (uint32_t step = 0; step < 32 &&
       !engine.snapshot().checkpointPublicationFailures; ++step)
    static_cast<void>(engine.tick(step + 1));
  require(engine.snapshot().checkpointPublicationFailures == 1 &&
              resources.snapshot().stateCache.entries == 0 &&
              events.failedCount == 0,
          "denied replacement kept the retired checkpoint or failed inference");
  require(engine.snapshot().checkpointPublicationFailureCauses ==
              onlyCause(PublicationFailure::SnapshotDenied, 1),
          "a denied checkpoint snapshot was not counted as snapshot_denied");
  runUntilIdle(engine);
  require(events.completedCount == 1 && executor.prefillRows == prompt.size() &&
              resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "inference did not recover from a skipped checkpoint publication");
}

void testRollingHandleCannotRetirePromotedState() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(25001, 37);
  engine.submit(request(560, prompt));
  runUntilCheckpoint(engine, 1);
  {
    auto existing = resources.lookup(prompt);
    require(resources.reuseCompositeState(existing.state->kvBlock()),
            "shared ordinary boundary could not reuse its checkpoint");
  }
  runUntilIdle(engine);
  const std::vector<uint32_t> prefix(
      prompt.begin(), prompt.begin() + defaultCheckpointTokens + 1);
  require(resources.snapshot().stateCache.entries == 3 &&
              resources.snapshot().stateCache.checkpointEntries == 1 &&
              resources.lookup(prefix).resumeBoundary() ==
                  defaultCheckpointTokens,
          "old rolling handle deleted a state promoted by another request");
}

void testCheckpointDenialPreservesUnrelatedHotState() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> hot(65, 21);
  engine.submit(request(420, hot));
  runUntilIdle(engine);
  executor.denySnapshotAtBoundary = defaultCheckpointTokens;
  engine.submit(request(421, std::vector<uint32_t>(18001, 22)));
  for (uint32_t step = 0; step < 32 &&
       !engine.snapshot().checkpointPublicationFailures; ++step)
    static_cast<void>(engine.tick(step + 1));
  require(engine.snapshot().checkpointPublicationFailures == 1 &&
              resources.snapshot().stateCache.evictions == 0 &&
              resources.lookup(hot).resumeBoundary() == 64,
          "optional progress allocation displaced an unrelated hot prefix");
  engine.cancel(421);
  runUntilIdle(engine);
}

void testCheckpointRecyclesItsBufferBeforeReplacement() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  engine.submit(request(430, std::vector<uint32_t>(25001, 23)));
  runUntilCheckpoint(engine, 1);
  executor.snapshotObserver = [&] {
    require(resources.snapshot().stateCache.entries == 0,
            "checkpoint replacement allocated before retiring its old state");
  };
  runUntilCheckpoint(engine, 2);
  require(resources.snapshot().stateCache.entries == 1 &&
              resources.snapshot().stateCache.evictions == 0 &&
              resources.snapshot().stateCache.checkpointRetirements == 1 &&
              engine.snapshot().checkpointPublicationFailures == 0,
          "progress could not replace its old state within the allocation budget");
  engine.cancel(430);
  runUntilIdle(engine);
}

void testCancelAtCheckpointDoesNotPublishDrainingCommand() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({.prefillCheckpointTokens = 8192}, resources, executor,
                        events);
  engine.submit(request(440, std::vector<uint32_t>(18001, 24)));
  for (uint32_t step = 0; step < 32; ++step) {
    static_cast<void>(engine.tick(step + 1));
    if (!engine.commandInFlight() && executor.requests.at(440).position == 6144)
      break;
  }
  executor.holdPrefillUntil = std::make_shared<bool>(false);
  static_cast<void>(engine.tick(40));
  require(engine.commandInFlight() && executor.requests.at(440).position == 8192,
          "fixture did not stop inside the checkpoint command");
  engine.cancel(440);
  *executor.holdPrefillUntil = true;
  runUntilIdle(engine);
  require(executor.snapshotAttempts == 0 &&
              engine.snapshot().checkpointPublications == 0 &&
              resources.snapshot().kvCache.blocks == 6144 / KvCache::pageTokens,
          "cancellation published a checkpoint from the draining command");
}

// The prompt-end state keeps the deepest checkpoint beside it, unprotected
// until the prompt-end publication lands: a governor-denied prompt-end
// capture recycles that checkpoint, not an unrelated hot state, and the
// prompt-end state wins.
void testFinalStateRecyclesItsCheckpointBeforeUnrelatedHotState() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> hot(65, 27);
  engine.submit(request(460, hot));
  runUntilIdle(engine);
  const std::vector<uint32_t> prompt(18001, 28);
  engine.submit(request(461, prompt));
  runUntilCheckpoint(engine, prompt.size() / defaultCheckpointTokens);
  const EngineSnapshot before = engine.snapshot();

  executor.deniedSnapshots = 1;
  runUntilIdle(engine);

  const EngineSnapshot after = engine.snapshot();
  require(after.recycledStatePublications ==
                  before.recycledStatePublications + 1 &&
              after.resources.stateCache.checkpointEvictions ==
                  before.resources.stateCache.checkpointEvictions + 1 &&
              after.resources.stateCache.entries == 2 &&
              after.resources.stateCache.checkpointEntries == 0 &&
              resources.lookup(hot).resumeBoundary() == 64 &&
              resources.lookup(prompt).resumeBoundary() == 17984,
          "denied prompt-end capture recycled hot state before its kept checkpoint");
}

void testFinalJunctionKeepsTheDeepestCheckpoint() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(20001, 29);
  resources.beginRequest(470);
  require(resources.ensureTokens(470, 20000).granted(),
          "junction fixture could not allocate its KV prefix");
  static_cast<void>(resources.publishCommittedBlocks(470, prompt, 20000));
  resources.endRequest(470);
  engine.submit(request(471, prompt));
  runUntilIdle(engine);
  const uint32_t deepest =
      prompt.size() / defaultCheckpointTokens * defaultCheckpointTokens;
  require(engine.snapshot().checkpointPublications ==
                  prompt.size() / defaultCheckpointTokens &&
              engine.snapshot().junctionMaterializations == 1 &&
              resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 1 &&
              resources.lookup(prompt).resumeBoundary() == 20000 &&
              resources.lookup(std::span(prompt).first(deepest + 1))
                      .resumeBoundary() == deepest,
          "prompt-end junction did not keep its deepest checkpoint beside it");
}

// Ticks until the engine is idle, long enough for a multi-row decode reply.
void runReplyUntilIdle(engine::Engine &engine) {
  for (uint32_t step = 0; step < 4096 && !engine.idle(); ++step)
    static_cast<void>(engine.tick(step + 1));
  require(engine.idle(), "engine did not finish the reply");
}

EngineRequest replyRequest(uint64_t id, const std::vector<uint32_t> &prompt,
                           uint32_t maxNewTokens) {
  EngineRequest result = request(id, prompt);
  result.maxNewTokens = maxNewTokens;
  result.deadlineMilliseconds = 100000;
  return result;
}

// A completed request keeps its deepest prefill checkpoint beside its
// prompt-end state, even after a reply that rolled a decode checkpoint; a
// prompt diverging between the two resumes from the checkpoint.
void testDivergentPromptResumesFromTheKeptCheckpoint() {
  Backing backing(2048);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  executor.decodeRowsPerStep = 7;
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const uint32_t deepest = 3 * defaultCheckpointTokens;
  const std::vector<uint32_t> prompt(deepest + 100, 70);
  // The reply crosses the next interval boundary: one decode checkpoint.
  engine.submit(replyRequest(700, prompt, defaultCheckpointTokens));
  runReplyUntilIdle(engine);
  require(engine.snapshot().checkpointPublications == 4 &&
              resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "completed request did not keep only its deepest prefill checkpoint");

  std::vector<uint32_t> divergent(prompt.begin(),
                                  prompt.begin() + deepest + 40);
  std::fill(divergent.begin() + deepest + 12, divergent.end(), 71);
  engine.submit(request(701, divergent));
  runUntilIdle(engine);

  require(events.starts.back().second == deepest && events.failedCount == 0,
          "divergent prompt did not resume from the kept checkpoint");
}

// A conversation holds one kept checkpoint: a turn that keeps none carries
// the earlier turn's handle forward, and the next turn that keeps its own
// retires it.
void testConversationKeepsOneCheckpoint() {
  Backing backing(2048);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const uint32_t first = 2 * defaultCheckpointTokens;
  std::vector<uint32_t> turn(first + 200, 72);
  engine.submit(request(710, turn));
  runUntilIdle(engine);
  const StateCacheSnapshot opened = resources.snapshot().stateCache;
  require(opened.entries == 2 && opened.checkpointEntries == 1,
          "first turn did not keep its deepest checkpoint");

  // A short turn crosses no interval boundary: it keeps no checkpoint.
  turn.insert(turn.end(), 40, 73);
  engine.submit(request(711, turn));
  runUntilIdle(engine);
  const StateCacheSnapshot carried = resources.snapshot().stateCache;
  require(events.starts.back().second > first &&
              carried.entries == 3 && carried.checkpointEntries == 1 &&
              carried.checkpointRetirements == opened.checkpointRetirements,
          "a turn without its own checkpoint retired the conversation's");

  const uint32_t second = 3 * defaultCheckpointTokens;
  turn.insert(turn.end(), second + 100 - turn.size(), 74);
  engine.submit(request(712, turn));
  runUntilIdle(engine);
  const StateCacheSnapshot closed = resources.snapshot().stateCache;
  require(closed.entries == 4 && closed.checkpointEntries == 1 &&
              closed.checkpointRetirements ==
                  carried.checkpointRetirements + 1 &&
              resources.lookup(std::span(turn).first(second + 1))
                      .resumeBoundary() == second &&
              resources.lookup(std::span(turn).first(first + 1))
                      .resumeBoundary() == 0,
          "the conversation kept more than its latest checkpoint");
}

// The kept checkpoint stays disposable: reclaim takes it before an older,
// unrelated ordinary state and before the prompt-end state beside it.
void testKeptCheckpointIsReclaimedBeforeOrdinaryStates() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> hot(65, 75);
  engine.submit(request(720, hot));
  runUntilIdle(engine);
  const std::vector<uint32_t> prompt(defaultCheckpointTokens + 100, 76);
  engine.submit(request(721, prompt));
  runUntilIdle(engine);
  require(resources.snapshot().stateCache.entries == 3 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "fixture did not keep its checkpoint beside two ordinary states");

  require(resources.reclaimOne().madeProgress,
          "reclaim found nothing to take");

  const StateCacheSnapshot states = resources.snapshot().stateCache;
  require(states.checkpointEvictions == 1 && states.checkpointEntries == 0 &&
              states.entries == 2 &&
              resources.lookup(hot).resumeBoundary() == 64 &&
              resources.lookup(prompt).resumeBoundary() ==
                  defaultCheckpointTokens + 96,
          "reclaim took an ordinary state before the kept checkpoint");
}

// Ticks until request id finishes while a peer lane keeps running.
void runUntilFinished(engine::Engine &engine, const Events &events,
                      uint64_t id) {
  for (uint32_t step = 0; step < 128 && !events.reasons.contains(id); ++step)
    static_cast<void>(engine.tick(step + 1));
  require(events.reasons.contains(id),
          "request did not finish beside its peer");
}

// Ticks until the engine has published one more replay state: a newly
// submitted request's prompt-end state.
void runUntilPromptEnd(engine::Engine &engine) {
  const uint64_t published = engine.snapshot().replayStatePublications;
  for (uint32_t step = 0;
       step < 128 && engine.snapshot().replayStatePublications == published;
       ++step)
    static_cast<void>(engine.tick(step + 1));
  require(engine.snapshot().replayStatePublications > published,
          "request did not publish its prompt-end state");
}

// Two conversations warmed and extended in turn, every state protected by a
// conversation end: while the first's extension stays resident, a
// governor-denied prompt-end capture of the second's recycles the checkpoint
// the second conversation inherited, never the first's older kept
// checkpoint.
void testDeniedCaptureRecyclesItsOwnConversationsCheckpoint() {
  Backing backing(2048);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const uint32_t deepest = 2 * defaultCheckpointTokens;
  std::vector<uint32_t> first(deepest + 200, 880);
  std::vector<uint32_t> second(deepest + 200, 881);
  engine.submit(request(880, first));
  runUntilIdle(engine);
  engine.submit(request(881, second));
  runUntilIdle(engine);
  first.insert(first.end(), 40, 882);
  engine.submit(replyRequest(882, first, 1000));
  runUntilPromptEnd(engine);
  require(!engine.idle() &&
              resources.snapshot().stateCache.checkpointEntries == 2,
          "fixture did not keep both conversations' checkpoints");
  const EngineSnapshot before = engine.snapshot();

  second.insert(second.end(), 40, 883);
  executor.deniedSnapshots = 1;
  engine.submit(request(883, second));
  runUntilFinished(engine, events, 883);

  const EngineSnapshot after = engine.snapshot();
  require(executor.deniedSnapshots == 0 &&
              after.recycledStatePublications ==
                  before.recycledStatePublications + 1 &&
              after.resources.stateCache.checkpointEvictions ==
                  before.resources.stateCache.checkpointEvictions + 1 &&
              after.replayStatePublicationFailures == 0 &&
              after.checkpointPublicationFailures == 0,
          "denied prompt-end capture did not land on a recycled checkpoint");
  require(resources.lookup(std::span(first).first(deepest + 1))
                      .resumeBoundary() == deepest &&
              resources.lookup(std::span(second).first(deepest + 1))
                      .resumeBoundary() == 0,
          "denied capture took the resident conversation's kept checkpoint");
  engine.cancel(882);
  runUntilIdle(engine);
}

// A checkpoint of the lane's conversation that a peer also holds is not the
// lane's to recycle: a peer resumed from the conversation's kept checkpoint
// keeps it beside its own prompt end, so a denied capture of the
// conversation's next turn recycles an idle state instead.
void testDeniedCaptureSkipsACheckpointAPeerHolds() {
  Backing backing(2048);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> idle(129, 890);
  cancelAfterPromptEnd(engine, resources, 890, idle);
  const uint32_t deepest = 2 * defaultCheckpointTokens;
  std::vector<uint32_t> prompt(deepest + 200, 891);
  engine.submit(request(891, prompt));
  runUntilIdle(engine);
  std::vector<uint32_t> divergent(prompt.begin(),
                                  prompt.begin() + deepest + 40);
  std::fill(divergent.begin() + deepest + 12, divergent.end(), 892);
  engine.submit(replyRequest(892, divergent, 1000));
  runUntilPromptEnd(engine);
  require(!engine.idle() && events.starts.back().second == deepest &&
              resources.lookup(idle).resumeBoundary() == 128,
          "peer did not resume from the kept checkpoint beside an idle state");
  const EngineSnapshot before = engine.snapshot();

  prompt.insert(prompt.end(), 40, 893);
  executor.deniedSnapshots = 1;
  engine.submit(request(893, prompt));
  runUntilFinished(engine, events, 893);

  const EngineSnapshot after = engine.snapshot();
  require(executor.deniedSnapshots == 0 &&
              after.recycledStatePublications ==
                  before.recycledStatePublications + 1 &&
              after.resources.stateCache.checkpointEvictions ==
                  before.resources.stateCache.checkpointEvictions &&
              after.replayStatePublicationFailures == 0,
          "denied prompt-end capture did not land on a recycled state");
  require(resources.lookup(std::span(prompt).first(deepest + 1))
                      .resumeBoundary() == deepest &&
              resources.lookup(idle).resumeBoundary() == 0,
          "denied capture recycled a checkpoint its peer holds");
  engine.cancel(892);
  runUntilIdle(engine);
}

// A one-lane engine over a 64-page backing, for conversation-end tests.
struct ConversationEndFixture final {
  // Runs request id over prompt to its Length finish.
  void finish(uint64_t id, const std::vector<uint32_t> &prompt) {
    engine.submit(request(id, prompt));
    runUntilIdle(engine);
  }

  // A Warning pass with an unbounded target.
  void warningPass() {
    static_cast<void>(engine.reclaimMemory(
        warningReclaim(std::numeric_limits<uint64_t>::max())));
  }

  Backing backing{64};
  KvPool pool{backing};
  engine::Cache resources{pool, CacheNamespace{}};
  Executor executor{1};
  Events events;
  engine::Engine engine{EngineConfig{}, resources, executor, events};
};

// Two conversations each finish with a prompt-end state; a Warning pass keeps
// both, and each re-sent, extended prompt resumes from its own.
void testWarningReclaimKeepsTheNewestConversationEnds() {
  ConversationEndFixture fixture;
  std::vector<uint32_t> first(129, 800);
  std::vector<uint32_t> second(129, 801);
  fixture.finish(800, first);
  fixture.finish(801, second);

  fixture.warningPass();

  const std::pair<EngineCacheStatus, uint32_t> promptEndHit{
      EngineCacheStatus::PrefixHit, 128};
  first.resize(200, 802);
  fixture.finish(802, first);
  require(fixture.events.starts.back() == promptEndHit,
          "warning reclaim took the older conversation's prompt-end state");
  second.resize(200, 803);
  fixture.finish(803, second);
  require(fixture.events.starts.back() == promptEndHit,
          "warning reclaim took the newer conversation's prompt-end state");
}

// Each follow-up turn replaces its conversation's end rather than pushing a
// new one: after four turns of one conversation, an idle conversation's end
// is still in the ring and a Warning pass keeps it.
void testFollowUpTurnsReplaceTheirConversationsEnd() {
  ConversationEndFixture fixture;
  const std::vector<uint32_t> idle(129, 810);
  fixture.finish(810, idle);
  std::vector<uint32_t> turn(65, 811);
  for (uint64_t id = 811; id < 815; ++id) {
    fixture.finish(id, turn);
    turn.resize(turn.size() + KvCache::pageTokens, static_cast<uint32_t>(id));
  }
  require(fixture.events.starts.back().first == EngineCacheStatus::PrefixHit,
          "follow-up turns did not resume from their conversation's end");

  fixture.warningPass();

  require(fixture.resources.cachedTokens(idle) == 128,
          "follow-up turns pushed an idle conversation's end out of the ring");
}

// The ring holds maximumBatchWidth conversation ends: one more finished
// conversation releases the oldest end's protections, and only that one.
void testAnotherConversationReleasesTheOldestEnd() {
  ConversationEndFixture fixture;
  const uint32_t ringEnds = model::ExecutionLimits::maximumBatchWidth;
  std::vector<std::vector<uint32_t>> prompts;
  for (uint32_t index = 0; index <= ringEnds; ++index)
    prompts.emplace_back(129, 820 + index);
  for (uint32_t index = 0; index < ringEnds; ++index)
    fixture.finish(820 + index, prompts[index]);
  fixture.warningPass();
  require(fixture.resources.cachedTokens(prompts.front()) == 128,
          "warning reclaim took an end the ring still holds");

  fixture.finish(820 + ringEnds, prompts.back());
  fixture.warningPass();

  require(
      fixture.resources.cachedTokens(prompts.front()) == 0,
      "the oldest conversation end stayed protected after it was pushed out");
  for (uint32_t index = 1; index <= ringEnds; ++index)
    require(fixture.resources.cachedTokens(prompts[index]) == 128,
            "pushing out the oldest end released a newer one");
}

// A Stop finish moves the request's end into the ring like a Length finish.
void testAStoppedRequestKeepsItsConversationEnd() {
  ConversationEndFixture fixture;
  const std::vector<uint32_t> prompt(129, 830);
  fixture.engine.submit(replyRequest(830, prompt, 1000));
  double now = 1;
  for (; now < 64 && fixture.events.outputs[830].empty(); ++now)
    static_cast<void>(fixture.engine.tick(now));
  while (fixture.engine.commandInFlight())
    static_cast<void>(fixture.engine.tick(now++));
  fixture.engine.stop(830);
  runUntilIdle(fixture.engine);
  require(fixture.events.reasons.at(830) == EngineFinishReason::Stop,
          "the request did not finish as Stop");

  fixture.warningPass();

  require(fixture.resources.cachedTokens(prompt) == 128,
          "warning reclaim took a stopped request's conversation end");
}

// The end protection taken at the prompt-end publication holds through a
// suspension and resume, and the finished request's end survives Warning.
void testResumedRequestKeepsItsConversationEnd() {
  SuspendedResumeFixture fixture;
  fixture.resume();

  static_cast<void>(fixture.engine.reclaimMemory(
      warningReclaim(std::numeric_limits<uint64_t>::max())));

  require(fixture.resources.cachedTokens(fixture.prompt) == 128,
          "warning reclaim took a resumed request's conversation end");
}

// With a conversation end the only state left, growth reclaim for a new
// request takes it rather than stall the admission.
void testGrowthReclaimTakesAConversationEndRatherThanStall() {
  ConversationEndFixture fixture;
  Executor &executor = fixture.executor;
  engine::Cache &resources = fixture.resources;
  fixture.finish(840, std::vector<uint32_t>(129, 840));
  require(resources.snapshot().stateCache.entries == 1,
          "fixture did not leave the conversation end as the only state");
  executor.beginGrowthBlocked = [&] {
    return resources.snapshot().stateCache.entries != 0;
  };

  fixture.engine.submit(request(841, std::vector<uint32_t>(65, 841)));
  for (double now = 200; now < 1000 && fixture.events.starts.size() < 2; ++now)
    static_cast<void>(fixture.engine.tick(now));

  require(fixture.events.starts.size() == 2 &&
              resources.snapshot().stateCache.evictions == 1,
          "growth reclaim stalled on a protected conversation end");
  runUntilIdle(fixture.engine);
  require(fixture.events.completedCount == 2 && fixture.events.failedCount == 0,
          "requests did not complete after taking the conversation end");
}

// A resident lane's prompt-end state and the checkpoint kept beside it are
// its conversation end from their publication: a Warning pass keeps both
// while the lane decodes.
void testWarningReclaimKeepsAResidentLanesConversationEnd() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(defaultCheckpointTokens + 100, 860);
  engine.submit(replyRequest(860, prompt, 1000));
  for (uint32_t step = 0;
       step < 128 && !engine.snapshot().replayStatePublications; ++step)
    static_cast<void>(engine.tick(step + 1));
  require(!engine.idle() && resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "fixture did not keep a checkpoint beside a resident prompt end");

  static_cast<void>(engine.reclaimMemory(
      warningReclaim(std::numeric_limits<uint64_t>::max())));

  require(resources.snapshot().stateCache.entries == 2 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "warning reclaim took a resident lane's conversation end");
  engine.cancel(860);
  runUntilIdle(engine);
}

// While macOS reports warn, every pressure pass sheds cache whatever the
// governor's own headroom: beside two decoding lanes it takes an idle
// conversation's state and keeps both lanes' prompt-end states and kept
// checkpoints. The directive is the policy's OS-warn pass with nothing
// waiting; an unbounded target shows what the pass may take, not how much.
void testOsWarningPassKeepsTwoDecodingLanesConversationEnds() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(2);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> idlePrompt(129, 870);
  cancelAfterPromptEnd(engine, resources, 870, idlePrompt);
  const uint64_t idlePublications =
      engine.snapshot().replayStatePublications;
  const std::vector<uint32_t> firstPrompt(defaultCheckpointTokens + 100, 871);
  const std::vector<uint32_t> secondPrompt(defaultCheckpointTokens + 100, 872);
  engine.submit(replyRequest(871, firstPrompt, 1000));
  engine.submit(replyRequest(872, secondPrompt, 1000));
  for (uint32_t step = 0;
       step < 256 &&
       engine.snapshot().replayStatePublications < idlePublications + 2;
       ++step)
    static_cast<void>(engine.tick(step + 1));
  require(!engine.idle() && resources.cachedTokens(idlePrompt) != 0 &&
              resources.snapshot().stateCache.entries == 5 &&
              resources.snapshot().stateCache.checkpointEntries == 2,
          "fixture did not keep two decoding lanes beside an idle state");

  static_cast<void>(engine.reclaimMemory(
      {.reclaimEmptyKvExtents = true,
       .evictAllUnpinnedPrefixes = false,
       .targetBytes = std::numeric_limits<uint64_t>::max(),
       .keepResumePoint = true}));

  require(resources.cachedTokens(idlePrompt) == 0,
          "an OS-warn pass kept an idle conversation's cache");
  require(resources.snapshot().stateCache.entries == 4 &&
              resources.snapshot().stateCache.checkpointEntries == 2,
          "an OS-warn pass took a decoding lane's conversation end");
  engine.cancel(871);
  engine.cancel(872);
  runUntilIdle(engine);
}

// A cancelled or failed request releases the end protection it took at its
// prompt-end publication: a Warning pass then takes the state.
void testCancelledOrFailedRequestReleasesItsConversationEnd() {
  for (const bool failed : {false, true}) {
    ConversationEndFixture fixture;
    const std::vector<uint32_t> prompt(129, failed ? 851 : 850);
    fixture.engine.submit(replyRequest(850, prompt, 1000));
    for (uint32_t step = 0;
         step < 64 && !fixture.engine.snapshot().replayStatePublications;
         ++step)
      static_cast<void>(fixture.engine.tick(step + 1));
    require(!fixture.engine.idle() &&
                fixture.resources.cachedTokens(prompt) == 128,
            "fixture did not publish the prompt end of a running request");

    if (failed)
      fixture.engine.failRequest(850, "test_failure",
                                 "failed after prompt end");
    else
      fixture.engine.cancel(850);
    runUntilIdle(fixture.engine);
    fixture.warningPass();

    require(fixture.resources.cachedTokens(prompt) == 0,
            "an unfinished request's conversation end stayed protected");
  }
}

void testShortSuffixContinuesCheckpointDraftState() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> prompt(18001, 30);
  engine.submit(request(480, prompt));
  runUntilCheckpoint(engine, 1);
  engine.cancel(480);
  runUntilIdle(engine);
  std::vector<uint32_t> shorter(
      prompt.begin(), prompt.begin() + defaultCheckpointTokens + 209);
  engine.submit(request(481, shorter));
  runUntilIdle(engine);
  require(events.starts.back().second == defaultCheckpointTokens &&
              executor.restoredDraft &&
              executor.plans.at(481).draftContextRows() == 209 &&
              !executor.plans.at(481).draftStateRestoreSkipped,
          "short checkpoint suffix discarded or rebuilt its restored draft window");
}

void testDefaultCheckpointRestoresLatestCommittedPrefix() {
  Backing backing(2048);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  const std::vector<uint32_t> donor(32769, 61);
  engine.submit(request(600, donor));
  for (uint32_t step = 0; step < 128; ++step) {
    static_cast<void>(engine.tick(step + 1));
    if (!engine.commandInFlight() &&
        executor.requests.at(600).position == 22528)
      break;
  }
  const auto committed = engine.snapshot();
  require(!engine.commandInFlight() && executor.prefillRows == 22528 &&
              executor.requests.at(600).position == 22528 &&
              committed.resources.kvCache.blocks == 22528 / KvCache::pageTokens,
          "fixture did not commit 22528 prompt tokens before cancellation");
  require(committed.checkpointPublications == 5 &&
              committed.checkpointPublicationFailures == 0 &&
              committed.resources.stateCache.entries == 1 &&
              committed.resources.stateCache.checkpointEntries == 1 &&
              committed.resources.stateCache.checkpointRetirements == 4,
          "default checkpoints accumulated or lost temporary states");
  {
    const auto lookup = resources.lookup(donor);
    require(lookup.kvBoundary == 22528 && lookup.resumeBoundary() == 20480,
            "completed KV tail did not retain its preceding 4K checkpoint");
  }
  engine.cancel(600);
  runUntilIdle(engine);
  require(engine.snapshot().cancelled == 1 &&
              resources.snapshot().stateCache.checkpointEntries == 1,
          "cancellation removed its completed recovery point");

  std::vector<uint32_t> branch(donor.begin(), donor.begin() + 22529);
  branch.back() = 62;
  engine.submit(request(601, branch));
  runUntilIdle(engine);
  const auto finished = engine.snapshot();
  require(events.startIds.back() == 601 &&
              events.starts.back().second == 20480 &&
              executor.prefillRows - 22528 == 2049 &&
              finished.completed == 1 && finished.cancelled == 1 &&
              events.failedCount == 0,
          "branch replayed before its latest checkpoint or failed completion");
  // The restored 20480 checkpoint is the branch's deepest one: kept.
  require(finished.checkpointPublications == 5 &&
              finished.junctionMaterializations == 1 &&
              finished.resources.stateCache.entries == 2 &&
              finished.resources.stateCache.checkpointEntries == 1 &&
              finished.resources.stateCache.checkpointRetirements == 4 &&
              resources.lookup(branch).resumeBoundary() == 22528,
          "branch completion did not keep its deepest checkpoint");
}

void testCheckpointIntervalValidationAndDisable() {
  Backing backing(1024);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  for (uint32_t invalid : {1U, 2047U, 2049U}) {
    bool rejected = false;
    try {
      engine::Engine engine({.prefillCheckpointTokens = invalid}, resources,
                            executor, events);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "invalid checkpoint interval was accepted");
  }
  engine::Engine engine({.prefillCheckpointTokens = 0}, resources, executor, events);
  engine.submit(request(450, std::vector<uint32_t>(18001, 25)));
  runUntilIdle(engine);
  require(engine.snapshot().checkpointPublications == 0 && executor.snapshots == 1 &&
              resources.snapshot().stateCache.entries == 1,
          "disabling checkpoints changed existing replay-state behavior");
}

// A lane decoding seven rows a step from a 100-token prompt, honouring the
// engine's retained cap. Growth pauses on demand, so a suspension waits for
// the test to resume it.
struct RollingDecode {
  static constexpr uint32_t promptTokens = 100;

  Backing backing{1024};
  KvPool pool{backing};
  engine::Cache resources{pool, CacheNamespace{}};
  Executor executor{2};
  Events events;
  bool paused = false;
  engine::Engine engine;
  double now = 1;

  explicit RollingDecode(uint32_t checkpointTokens = defaultCheckpointTokens)
      : engine(config(checkpointTokens, paused), resources, executor, events) {
    executor.decodeRowsPerStep = 7;
  }

  static EngineConfig config(uint32_t checkpointTokens, const bool &paused) {
    EngineConfig result;
    result.prefillCheckpointTokens = checkpointTokens;
    result.growthPaused = [&paused] { return paused; };
    return result;
  }

  void submit(uint64_t id, uint32_t maxNewTokens) {
    auto decoding = request(id, std::vector<uint32_t>(promptTokens, 51));
    decoding.maxNewTokens = maxNewTokens;
    decoding.deadlineMilliseconds = 100000;
    engine.submit(std::move(decoding));
  }

  // Tokens of the lane's exact history: prompt plus emitted output.
  uint32_t stored(uint64_t id) {
    return promptTokens + static_cast<uint32_t>(events.outputs[id].size());
  }

  void decodeUntil(uint64_t id, uint32_t tokens) {
    for (; now < 20000 && stored(id) < tokens; ++now)
      static_cast<void>(engine.tick(now));
    require(stored(id) >= tokens, "rolling decode did not reach its target");
  }

  void runToIdle() {
    for (const double deadline = now + 20000; now < deadline && !engine.idle();
         ++now)
      static_cast<void>(engine.tick(now));
    require(engine.idle() && events.failedCount == 0,
            "rolling decode fixture did not finish its requests");
  }
};

void testDecodePublishesRollingCheckpoints() {
  RollingDecode fixture;
  const uint32_t interval = defaultCheckpointTokens;
  const uint32_t maxNewTokens = 2 * interval + 600;
  fixture.submit(520, maxNewTokens);
  fixture.decodeUntil(520, 2 * interval + 40);
  require(fixture.engine.snapshot().checkpointPublications == 2 &&
              fixture.engine.snapshot().checkpointPublicationFailures == 0 &&
              !fixture.executor.retainedLimits.empty(),
          "a long decode did not land a checkpoint on each boundary");

  fixture.paused = fixture.backing.growthBlocked = true;
  for (const double deadline = fixture.now + 64;
       fixture.now < deadline && !fixture.executor.suspensions; ++fixture.now)
    static_cast<void>(fixture.engine.tick(fixture.now));
  require(fixture.executor.suspensions == 1 &&
              fixture.engine.snapshot().resourceDecodeSuspensions == 1,
          "paused growth did not suspend the decoding lane");
  const uint32_t history = fixture.executor.suspendedPositions.back();
  fixture.paused = fixture.backing.growthBlocked = false;
  fixture.runToIdle();

  require(fixture.executor.restoredLengths.contains(520) &&
              fixture.executor.restoredLengths.at(520) == 2 * interval &&
              fixture.engine.snapshot().resourceDecodeReplayTokens ==
                  history - 2 * interval &&
              history - 2 * interval <= interval,
          "a decode suspension replayed past its latest checkpoint");
  require(fixture.events.usage.at(520).second == maxNewTokens &&
              fixture.events.outputs.at(520).size() == maxNewTokens,
          "a capped decode step ended the reply early");
}

// A decode lane whose capture is denied recycles its own rolling checkpoint,
// never the older checkpoint another conversation keeps.
void testDeniedDecodeCaptureRecyclesTheLanesOwnCheckpoint() {
  RollingDecode fixture;
  const uint32_t interval = defaultCheckpointTokens;
  const std::vector<uint32_t> other(interval + 100, 545);
  fixture.engine.submit(request(545, other));
  fixture.runToIdle();
  fixture.submit(546, 2 * interval + 600);
  fixture.decodeUntil(546, interval + 40);
  const EngineSnapshot before = fixture.engine.snapshot();

  fixture.executor.deniedSnapshots = 1;
  fixture.decodeUntil(546, 2 * interval + 40);

  const EngineSnapshot after = fixture.engine.snapshot();
  require(fixture.executor.deniedSnapshots == 0 &&
              after.checkpointPublicationFailures == 0 &&
              after.checkpointPublications ==
                  before.checkpointPublications + 1 &&
              after.recycledStatePublications ==
                  before.recycledStatePublications + 1,
          "denied decode capture did not land on a recycled checkpoint");
  require(fixture.resources.lookup(std::span(other).first(interval + 1))
                  .resumeBoundary() == interval,
          "denied decode capture took another conversation's checkpoint");
  fixture.runToIdle();
}

// A client stop between batches finishes the lane as a normal stop: counted
// completed, not cancelled, with its rolling checkpoint retired.
void testStopBetweenBatchesFinishesAsCompleted() {
  RollingDecode fixture;
  fixture.submit(530, 2 * defaultCheckpointTokens);
  fixture.decodeUntil(530, RollingDecode::promptTokens +
                               defaultCheckpointTokens + 40);
  while (fixture.engine.commandInFlight())
    static_cast<void>(fixture.engine.tick(fixture.now++));
  require(fixture.engine.snapshot().checkpointPublications >= 1,
          "stop fixture did not publish a decode checkpoint");
  const uint64_t retiredBefore =
      fixture.resources.snapshot().stateCache.checkpointRetirements;

  fixture.engine.stop(530);

  const auto stopped = fixture.engine.snapshot();
  require(fixture.events.reasons.at(530) == EngineFinishReason::Stop &&
              stopped.completed == 1 && stopped.cancelled == 0 &&
              fixture.resources.snapshot().stateCache.checkpointRetirements >
                  retiredBefore,
          "a stop between batches did not finish as a completed request");
  fixture.runToIdle();
  require(fixture.resources.snapshot().activeRequests == 0 &&
              fixture.executor.requests.empty(),
          "a stopped request kept its lane");
}

// A stop of a lane inside the in-flight batch waits for the batch to drain,
// then finishes as Stop without emitting that batch's output.
void testStopInFlightFinishesAfterTheBatch() {
  RollingDecode fixture;
  fixture.submit(531, 1000);
  fixture.decodeUntil(531, RollingDecode::promptTokens + 20);
  if (!fixture.engine.commandInFlight())
    static_cast<void>(fixture.engine.tick(fixture.now++));
  require(fixture.engine.commandInFlight(),
          "stop fixture did not launch a decode command");
  const size_t emittedBefore = fixture.events.outputs.at(531).size();

  fixture.engine.stop(531);
  require(fixture.engine.commandInFlight() &&
              fixture.events.completedCount == 0,
          "an in-flight stop finished before its batch drained");
  fixture.runToIdle();

  const auto stopped = fixture.engine.snapshot();
  require(fixture.events.reasons.at(531) == EngineFinishReason::Stop &&
              stopped.completed == 1 && stopped.cancelled == 0 &&
              fixture.events.outputs.at(531).size() == emittedBefore,
          "an in-flight stop did not finish as Stop after its batch");
}

// A request stopped before it ever started finishes as Cancelled; an unknown
// or already finished id is ignored.
void testStopBeforeStartFinishesAsCancelled() {
  Backing backing(64);
  KvPool pool(backing);
  engine::Cache resources(pool, CacheNamespace{});
  Executor executor(1);
  Events events;
  engine::Engine engine({}, resources, executor, events);
  engine.submit(request(532, std::vector<uint32_t>(64, 5)));

  engine.stop(532);
  runUntilIdle(engine);
  engine.stop(532);
  engine.stop(999);

  const auto stopped = engine.snapshot();
  require(events.startIds.empty() && events.completedCount == 1 &&
              events.reasons.at(532) == EngineFinishReason::Cancelled &&
              stopped.cancelled == 1 && stopped.completed == 0,
          "an unstarted stop did not finish as a single cancellation");
}

// A lane suspended for resources has no cell but did start: a stop finishes
// it as Stop, counted completed, and releases its suspension.
void testStopOfASuspendedLaneFinishesAsCompleted() {
  SoleDecodeLane fixture(100.0);
  const double denied = fixture.decodeUntilBackedOff();
  static_cast<void>(fixture.engine.tick(denied + 100.0));
  require(fixture.executor.suspensions == 1 &&
              !fixture.resident(SoleDecodeLane::laneId) &&
              fixture.events.emitted > 0,
          "stop fixture did not suspend a decoding lane");

  fixture.engine.stop(SoleDecodeLane::laneId);
  fixture.now = denied + 101.0;
  require(fixture.tickUntil([&] { return fixture.engine.idle(); }),
          "a stopped suspended lane did not leave the engine");

  const auto stopped = fixture.engine.snapshot();
  require(fixture.events.reasons.at(SoleDecodeLane::laneId) ==
                  EngineFinishReason::Stop &&
              stopped.completed == 1 && stopped.cancelled == 0 &&
              fixture.events.failedCount == 0 &&
              fixture.resources.snapshot().activeRequests == 0,
          "a suspended lane's stop did not finish as a completed request");
}

void testDecodeSuspensionOnTheBoundaryResumesFromThePreviousCheckpoint() {
  const uint32_t interval = 2048;
  RollingDecode fixture(interval);
  fixture.submit(521, 3 * interval);
  fixture.decodeUntil(521, 2 * interval);
  require(fixture.stored(521) == 2 * interval &&
              fixture.engine.snapshot().checkpointPublications == 2,
          "the capped commit did not land on the second boundary");

  // The foreground peer cannot grow while growth is paused; the decoding
  // lane yields to it before decoding past the boundary.
  fixture.paused = fixture.backing.growthBlocked = true;
  auto foreground = request(522, std::vector<uint32_t>(4096, 52));
  foreground.priority = RequestPriority::Foreground;
  fixture.engine.submit(std::move(foreground));
  for (const double deadline = fixture.now + kPeerSuspensionTicks;
       fixture.now < deadline && !fixture.executor.suspensions; ++fixture.now)
    static_cast<void>(fixture.engine.tick(fixture.now));
  require(fixture.executor.suspensions == 1 &&
              fixture.executor.suspendedPositions.back() == 2 * interval,
          "the decoding lane did not suspend on its boundary");

  static_cast<void>(fixture.resources.reclaimCache(
      std::numeric_limits<uint64_t>::max(), false));
  fixture.paused = fixture.backing.growthBlocked = false;
  fixture.runToIdle();
  require(fixture.executor.restoredLengths.contains(521) &&
              fixture.executor.restoredLengths.at(521) == interval &&
              fixture.engine.snapshot().resourceDecodeReplayTokens == interval,
          "a suspension on the boundary did not resume from the previous "
          "checkpoint");
}

void testDecodeBoundaryStateIsReusedNotPublishedTwice() {
  const uint32_t interval = 2048;
  RollingDecode fixture(interval);
  fixture.submit(523, interval + 200);
  fixture.submit(524, interval + 200);
  fixture.decodeUntil(523, RollingDecode::promptTokens + 8);
  fixture.decodeUntil(524, RollingDecode::promptTokens + 8);
  const EngineSnapshot before = fixture.engine.snapshot();
  const uint32_t snapshots = fixture.executor.snapshots;

  fixture.decodeUntil(523, interval + 8);
  fixture.decodeUntil(524, interval + 8);
  const EngineSnapshot after = fixture.engine.snapshot();
  require(after.checkpointPublications - before.checkpointPublications == 1 &&
              after.deduplicatedStatePublications -
                      before.deduplicatedStatePublications ==
                  1 &&
              fixture.executor.snapshots - snapshots == 1 &&
              after.checkpointPublicationFailures == 0,
          "two lanes published the same decode boundary twice");
  fixture.runToIdle();
}

void testUncappedCommitPastTheBoundaryCountsOneFailure() {
  const uint32_t interval = 2048;
  RollingDecode fixture(interval);
  fixture.executor.ignoresRetainedLimit = true;
  // 100 + 7k never equals 2048: the uncapped lane passes the boundary.
  fixture.submit(525, interval);
  fixture.runToIdle();
  require(fixture.engine.snapshot().checkpointPublications == 0 &&
              fixture.engine.snapshot().checkpointPublicationFailures == 1 &&
              fixture.events.outputs.at(525).size() == interval,
          "an uncapped commit past the boundary was not one failure");
}

void testDisabledCheckpointsLeaveDecodeUncapped() {
  RollingDecode fixture(0);
  fixture.submit(527, 2100);
  fixture.runToIdle();
  require(fixture.executor.retainedLimits.empty() &&
              fixture.engine.snapshot().checkpointPublications == 0 &&
              fixture.engine.snapshot().checkpointPublicationFailures == 0,
          "disabled checkpoints capped or checkpointed a decode");
}

void testPositionalBatchItemDecodesUncapped() {
  Executor executor;
  const std::vector<uint32_t> prompt(RollingDecode::promptTokens, 53);
  ModelRequest modelRequest;
  modelRequest.id = 526;
  modelRequest.prompt = prompt;
  modelRequest.maxNewTokens = 64;
  require(executor.begin(modelRequest).granted(),
          "fake did not admit the lane");
  executor.decodeRowsPerStep = model::ExecutionLimits::targetVerifyRows;
  const std::vector<uint32_t> pages{0};
  const ModelBatchItem items[] = {{526, 0, 0, 0, 1, pages, 0, {}}};
  BatchPlan plan;
  plan.kind = WorkKind::Decode;

  const auto results = executor.decode(plan, items);

  require(items[0].retainedTokenLimit == 0 && executor.retainedLimits.empty() &&
              results.front().outputTokens.size() ==
                  model::ExecutionLimits::targetVerifyRows,
          "a positional batch item was capped");
}

} // namespace

int main() {
  try {
    testConcurrentColdPrefixesComputeOnce();
    testSharedPrefillRebuildsTheMissingJunctionOnce();
    testSharedPrefillReleasesDifferentJunctionsIndependently();
    testSharedPrefillEvictedPublicationFallsBack();
    testSharedPrefillProducerFailureReleasesWaiters();
    testSharedPrefillWaiterCancellationAndDeadline();
    testSharedPrefillFailedPublicationFallsBack();
    testSharedPrefillDoesNotBlockUnrelatedWork();
    testSharedPrefillHonorsPriorityAndLateArrival();
    testLateSharedPrefillExtendsTheProducerPlan();
    testSharedPrefillCapacityFailureDoesNotDeadlock();
    testCancelledColdPrefillResumesItsLatestCheckpoint();
    testConcurrentProgressRetainsAtMostOnePointPerLane();
    testSharedCheckpointSurvivesPeerRollingReplacement();
    testRepeatedRetriesRollTheRestoredCheckpoint();
    testRetryCancelledBeforeNextCheckpointKeepsItsSource();
    testRestoredCheckpointAtReplayEndBecomesOrdinary();
    testRetryRetiresCheckpointAtDeeperJunction();
    testPinnedCheckpointSkipsReplacementButNotOrdinaryState();
    testPausedGrowthReclaimKeepsTheResidentLanesCheckpoint();
    testGrowthReclaimTakesAResidentCheckpointRatherThanStall();
    testHostPressureSuspensionResumesFromTheLanesCheckpoint();
    testOverwrittenCheckpointLeavesNoProtectionBehind();
    testFailedReplacementContinuesWithoutRecoveryPoint();
    testRollingHandleCannotRetirePromotedState();
    testCheckpointDenialPreservesUnrelatedHotState();
    testCheckpointRecyclesItsBufferBeforeReplacement();
    testCancelAtCheckpointDoesNotPublishDrainingCommand();
    testFinalStateRecyclesItsCheckpointBeforeUnrelatedHotState();
    testFinalJunctionKeepsTheDeepestCheckpoint();
    testDivergentPromptResumesFromTheKeptCheckpoint();
    testConversationKeepsOneCheckpoint();
    testKeptCheckpointIsReclaimedBeforeOrdinaryStates();
    testDeniedCaptureRecyclesItsOwnConversationsCheckpoint();
    testDeniedCaptureSkipsACheckpointAPeerHolds();
    testWarningReclaimKeepsTheNewestConversationEnds();
    testFollowUpTurnsReplaceTheirConversationsEnd();
    testAnotherConversationReleasesTheOldestEnd();
    testAStoppedRequestKeepsItsConversationEnd();
    testResumedRequestKeepsItsConversationEnd();
    testGrowthReclaimTakesAConversationEndRatherThanStall();
    testWarningReclaimKeepsAResidentLanesConversationEnd();
    testOsWarningPassKeepsTwoDecodingLanesConversationEnds();
    testCancelledOrFailedRequestReleasesItsConversationEnd();
    testShortSuffixContinuesCheckpointDraftState();
    testDefaultCheckpointRestoresLatestCommittedPrefix();
    testCheckpointIntervalValidationAndDisable();
    testDecodePublishesRollingCheckpoints();
    testDeniedDecodeCaptureRecyclesTheLanesOwnCheckpoint();
    testStopBetweenBatchesFinishesAsCompleted();
    testStopInFlightFinishesAfterTheBatch();
    testStopBeforeStartFinishesAsCancelled();
    testStopOfASuspendedLaneFinishesAsCompleted();
    testDecodeSuspensionOnTheBoundaryResumesFromThePreviousCheckpoint();
    testDecodeBoundaryStateIsReusedNotPublishedTwice();
    testUncappedCommitPastTheBoundaryCountsOneFailure();
    testDisabledCheckpointsLeaveDecodeUncapped();
    testPositionalBatchItemDecodesUncapped();
    testColdPublishesReplayStateAndLazyJunctionCanRebuildIt();
    testConcurrentDuplicateStateSkipsSnapshotCapture();
    testImageSpansKeyPrefixIdentity();
    testOneRequestPublishesJunctionAndLatestReplayState();
    testLatestReplayDenialRecyclesOlderStateNotTheJunction();
    testCancellationAfterJunctionDiscardsLaterState();
    testDeniedSnapshotCostsOnlyThatAttempt();
    testDeniedSnapshotRecyclesLruStateAndRetries();
    testPersistentSnapshotDenialRecyclesAtMostOneState();
    testLongSuffixSkipsDraftRestore();
    testTargetOnlyStateRestoresWithoutDraft();
    testShortRemainderWalksPastTargetOnlyState();
    testQueuedPeerJunctionWalksPastTargetOnlyState();
    testCancellationInFlightAtBoundaryPublishesNoState();
    testActiveCellGrowthReclaimsCachedStateAndRetries();
    testKvGrowthReclaimsIdleStateBeforeCache();
    testKvGrowthDenialKeepsEveryLaneReplayState();
    testPressureReclaimRespectsStateLifetimes();
    testPressureReclaimKeepsRunwayOnlyForWarning();
    testConcurrencyLimitDoesNotEvictCache();
    testHostPressureDoesNotDrainCacheOnStateAdmission();
    testHostPressureStillRecyclesLruStateForDeniedSnapshot();
    testSingletonHostPressureReusesIdleCacheInsteadOfSuspending();
    testSingletonHostPressureWaitRecoversOrTerminates();
    testSoleDecodeLaneBacksOffUnderHostPressure();
    testSoleDecodeLaneSuspendsAtTheResourceWaitBound();
    testSolePrefillLaneBacksOffUnderHostPressure();
    testSolePrefillLaneSuspendsAtTheResourceWaitBound();
    testSoleDecodeLaneSuspendsBesideAWaitingPeer();
    testKvPressureNarrowsTheRealBatch();
    testKvGrowthReclaimsCachedStateWhenBudgetIsShared();
    testRequiredWorkDoesNotReserveAnExtraPage();
    testAdmissionPinsDesiredStateAndCountsOnlySuccess();
    testAdmissionCanDropItsOwnCachePinToMakeProgress();
    testSingletonCapacityFailureTerminatesCleanly();
    testQueuedLongPrefillsLeaveRoomForShortWork();
    testAdmissionUsesCachedRemainingWork();
    testFailedAdmissionDoesNotBlockOtherWork();
    testSchedulingWaitDoesNotConsumeMemoryTimeout();
    testUnadmittedRequestsHonorCancellationAndDeadline();
    testGrowthKeepsPrefillProgressWhenAnUnstartedPeerCanYield();
    testGrowthYieldsLowerPriorityResidentOutsideBatch();
    testMidPrefillSuspensionCountsOnlyProcessedTokensAsReplay();
    testPageAlignedSuspensionResumesFromItsLiveState();
    testUnalignedSuspensionPublishesNothing();
    testSuspensionAtAStateBlockKeepsThatState();
    testSuspensionStateReplacesTheLaneCheckpoint();
    testDecodeSuspensionPublishesNothing();
    testSuspensionBeforeAnyPrefillPublishesNothing();
    testShortTargetOnlySuspensionStateIsDropped();
    testRejectedTargetOnlySuspensionStateFallsBack();
    testSuspensionStateIsProtectedUntilUnprotectedStatesGo();
    testRepeatedSuspensionKeepsTheFirstReplayEnd();
    testSuspensionDuringDecodeReplayKeepsCountingDecodeReplay();
    testPrefillGrowthPreservesAnActiveDecodePeer();
    testPhysicalKvPressureSuspendsInsteadOfKillingActiveWork();
    testRecoveryDrainDoesNotConsumeResourceWaitBudget();
    testPhysicalPressureRetryIsBackedOffWithoutProgress();
    testRecoveryWakeupIgnoresWaitersItDoesNotRetry();
    testDecodePreemptionReplaysCommittedHistoryWithoutRepeatingOutput();
    testLongDecodePreemptionPlansTheCurrentReplayBoundary();
    testPreemptedDecodeRestoresItsResidentCompositeState();
    testRepeatedPreemptionRespectsBackoffAndCancellation();
    testAdmissionReopensAfterLastSuspendedRequestResumes();
    testRecoveryAdmitsFailedKvTargetBeforeReplaying();
    testWarningReclaimKeepsSuspendedRequestsResumeState();
    testWarningReclaimWalksPastProtectedResumeState();
    testCriticalReclaimEvictsSuspendedRequestsResumeState();
    testCancelledSuspendedRequestReleasesItsResumeExemption();
    testSuspendedLanesSharingAResumeStateKeepItUntilBothRelease();
    testWarningReclaimKeepsQueuedRequestsPrefix();
    testRetriedQueuedRequestHoldsOneProtection();
    testQueuedRequestAdmittedColdDropsItsProtection();
    testGrowthReclaimTakesAQueuedPrefixRatherThanStall();
    testGrowthReclaimSparesAFoundStateForCachedKv();
    testQueuedRequestKeepsItsFoundStateWhileAPeerIsResident();
    testQueuedRequestOutlivedByAPeerAdmitsCold();
    testStateAdmissionWaitsForKvRelease();
    testAdmissionsWaitForBackgroundRelease();
    testAllocationCausesRemainRetryableAndDistinct();
    testAdmissionRespectsPriorityBeforeHashOrder();
    testConstraintMaskOverlapsInsideOneSchedulerBatch();
    testConstraintMaskWaitHonorsCancelAndDeadline();
    testDecodeNearContextCeilingCoversVerifyRows();
    testExpiredMaskWaitFinalizesWhileAnotherCommandRuns();
    testOrdinaryInFlightDeadlineDrainsWithoutPublishingOrOutput();
    testStalledSuspensionFailsWithCapacity();
    testTerminalAnchorWithoutKvIsNotCached();
    testPrefillCanCompleteTheRequest();
    std::cout << "KV-first engine tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "KV-first engine tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
