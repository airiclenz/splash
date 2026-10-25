#include "engine/Engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace splash::engine {
namespace {

constexpr double kResourceRetryBackoffMilliseconds = 100.0;
// The share of the resource wait timeout, counted back from the deadline,
// during which a queued request no longer keeps its found state for a peer.
constexpr double kColdFallbackWaitFraction = 0.5;
constexpr double kHealthCheckIntervalMilliseconds = 1000.0;
// Failure codes an in-flight lane carries until its batch drains, then
// finishes as Cancelled or Stop rather than as a failure.
constexpr char kCancelledFailureCode[] = "cancelled";
constexpr char kStoppedFailureCode[] = "stopped";
constexpr double kMicrosecondsPerMillisecond = 1000.0;

// ============================================================================
// True while a resource wait has more than kColdFallbackWaitFraction of its
// timeout left, or has no deadline yet (deferResourceRetry sets it only after
// a first denial). The margin covers peer steps longer than the retry
// backoff, so the cold-admission fallback still runs before resource_timeout.
bool isResourceWaitYoung(double deadlineMilliseconds,
                         double timeoutMilliseconds,
                         double nowMilliseconds) noexcept {
  return deadlineMilliseconds <= 0.0 ||
         nowMilliseconds < deadlineMilliseconds -
                               timeoutMilliseconds * kColdFallbackWaitFraction;
}

// ============================================================================
// A batch wall in whole microseconds; a negative or NaN wall counts as 0.
uint64_t wholeMicroseconds(double milliseconds) noexcept {
  return milliseconds > 0.0 ? static_cast<uint64_t>(std::llround(
                                  milliseconds * kMicrosecondsPerMillisecond))
                            : 0;
}

uint32_t replayStateBoundary(uint32_t tokens) noexcept {
  return tokens > 1 ? (tokens - 1) / KvCache::pageTokens * KvCache::pageTokens
                    : 0;
}

// The plan of a model that does not draft: planDraftContext's replay range
// and restore boundary with no capture or boundary, so no draft row is
// replayed, and a restored state never brings its draft window.
DraftContextPlan undraftedPlan(uint32_t replayBegin, uint32_t replayEnd,
                               std::optional<uint32_t> restoredBoundary,
                               bool targetOnlyRestore) {
  DraftContextPlan result;
  result.replayBegin = replayBegin;
  result.replayEnd = replayEnd;
  result.restoredDraftBoundary = restoredBoundary;
  result.targetOnlyRestore = targetOnlyRestore;
  result.targetPrefillRows = replayEnd - replayBegin;
  result.draftContextRowsAvoided = result.targetPrefillRows;
  result.draftStateRestoreSkipped = restoredBoundary ? 1 : 0;
  return result;
}

} // namespace

Engine::Engine(EngineConfig config, Cache &cache, model::Model &model,
               EngineEventSink &events)
    : config_(config), cache_(cache), model_(model), events_(events),
      scheduler_(config_.contendedPrefillRows) {
  if (!config_.maxContext || !config_.vocabularySize) {
    throw std::invalid_argument("context and vocabulary sizes must be positive");
  }
  if (!std::isfinite(config_.resourceWaitTimeoutMilliseconds) ||
      config_.resourceWaitTimeoutMilliseconds <= 0.0)
    throw std::invalid_argument("resource wait timeout must be positive and finite");
  if (config_.prefillCheckpointTokens &&
      (config_.prefillCheckpointTokens <
           model::ExecutionLimits::draftContextTokens ||
       config_.prefillCheckpointTokens % KvCache::pageTokens)) {
    throw std::invalid_argument(
        "prefill checkpoint interval must span a draft window and whole KV pages");
  }
}

void Engine::submit(EngineRequest value) {
  const bool scoring = !value.scoreTokens.empty();
  if (!value.id || value.prompt.empty() ||
      (scoring ? value.maxNewTokens != 0 : !value.maxNewTokens) ||
      value.prompt.size() + value.maxNewTokens > config_.maxContext ||
      !std::isfinite(value.deadlineMilliseconds) ||
      value.deadlineMilliseconds <= 0.0) {
    throw std::invalid_argument("invalid backend request");
  }
  if (std::any_of(value.prompt.begin(), value.prompt.end(), [&](uint32_t token) {
        return token >= config_.vocabularySize;
      })) {
    throw std::invalid_argument("prompt token is out of vocabulary");
  }
  uint64_t previousImageEnd = 0;
  uint64_t pixelBytes = 0;
  for (const ImageSpan &image : value.images) {
    const uint64_t patches = uint64_t{image.gridHeight} * image.gridWidth;
    if (image.gridHeight < 2 || image.gridWidth < 2 || image.gridHeight % 2 ||
        image.gridWidth % 2 || patches > config_.maxImagePatches ||
        image.tokens != (image.gridHeight / 2) * (image.gridWidth / 2) ||
        image.offset < previousImageEnd ||
        uint64_t{image.offset} + image.tokens > value.prompt.size()) {
      throw std::invalid_argument("invalid backend request image span");
    }
    previousImageEnd = image.end();
    pixelBytes += image.pixelBytes();
  }
  if (value.imagePixels.size() != pixelBytes) {
    throw std::invalid_argument("invalid backend request image pixels");
  }
  const uint64_t id = value.id;
  if (scoring) {
    if (value.cohort != BatchCohort::Greedy ||
        value.constraint != ConstraintMode::None || !value.images.empty() ||
        value.sampling.temperature != 0.0f || value.sampling.topP != 1.0f ||
        value.sampling.topK != 0 ||
        value.scoreTokens.size() < model::ExecutionLimits::minimumScoreOptions ||
        value.scoreTokens.size() > model::ExecutionLimits::maximumScoreOptions) {
      throw std::invalid_argument("invalid score request");
    }
    std::vector<uint32_t> distinct(value.scoreTokens.begin(),
                                   value.scoreTokens.end());
    std::sort(distinct.begin(), distinct.end());
    if (std::adjacent_find(distinct.begin(), distinct.end()) !=
            distinct.end() ||
        std::any_of(distinct.begin(), distinct.end(), [&](uint32_t token) {
          return token >= config_.vocabularySize;
        })) {
      throw std::invalid_argument("score token is out of vocabulary");
    }
  }
  Request requestState;
  requestState.promptTokens = static_cast<uint32_t>(value.prompt.size());
  requestState.replayTokens = requestState.promptTokens;
  requestState.request = std::move(value);
  auto [entry, inserted] = requests_.emplace(id, std::move(requestState));
  if (!inserted)
    throw std::invalid_argument("duplicate backend request id");
  const Request &stored = entry->second;
  scheduler_.submit(
      {.id = id,
       .priority = stored.request.priority,
       .cohort = stored.request.cohort,
       .promptTokens = stored.promptTokens,
       .deadlineMilliseconds = stored.request.deadlineMilliseconds});
  ++counters_.submitted;
}

void Engine::cancel(uint64_t id) {
  auto found = requests_.find(id);
  if (found == requests_.end() || found->second.finalized)
    return;
  if (endAfterInFlightBatch(found->second,
                            {kCancelledFailureCode, "request cancelled"}))
    return;
  finish(found->second, EngineFinishReason::Cancelled, {});
}

void Engine::stop(uint64_t id) {
  auto found = requests_.find(id);
  if (found == requests_.end() || found->second.finalized)
    return;
  Request &active = found->second;
  // Never started: no lane and no output yet, so nothing was generated to
  // stop on. A resource-suspended lane has no cell but did start.
  if (!active.stateCell && !active.suspended) {
    finish(active, EngineFinishReason::Cancelled, {});
    return;
  }
  if (endAfterInFlightBatch(active, {kStoppedFailureCode, "request stopped"}))
    return;
  finish(active, EngineFinishReason::Stop, {});
}

// ============================================================================
// When the lane rides the in-flight batch, records `failure` for the batch's
// completion to act on and abandons its mask wait, returning true; an
// in-flight command cannot be revoked, so the lane ends once it drains.
bool Engine::endAfterInFlightBatch(Request &active, Failure failure) {
  if (!pending_)
    return false;
  const uint64_t id = active.request.id;
  for (const BatchItem &item : pending_->plan.items) {
    if (item.requestId != id)
      continue;
    // An earlier deadline failure of the same in-flight lane stands.
    if (!active.failure)
      active.failure = std::move(failure);
    pending_->ticket->abandonMask(id);
    return true;
  }
  return false;
}

void Engine::failRequest(uint64_t id, std::string code, std::string message) {
  Request &active = request(id);
  if (active.finalized || active.failure)
    return;
  Failure failure{std::move(code), std::move(message)};
  if (pending_) {
    for (const BatchItem &item : pending_->plan.items) {
      if (item.requestId == id) {
        active.failure = std::move(failure);
        pending_->ticket->abandonMask(id);
        return;
      }
    }
  }
  finishFailure(active, std::move(failure));
}

void Engine::provideMask(uint64_t id, std::span<const uint32_t> words) {
  Request &active = request(id);
  const bool ownedByActiveBatch =
      pending_ && pending_->ticket->ownsMaskWait(id);
  // A request that already left its mask wait (cancellation, deadline, or a
  // failure raced the frontend) treats the response as stale.
  if (active.finalized || active.failure ||
      (!ownedByActiveBatch && scheduler_.phase(id) != Phase::WaitingMask)) {
    return;
  }
  model_.provideMask(id, words);
  if (!ownedByActiveBatch)
    scheduler_.maskReady(id);
}

void Engine::setCompletionNotifier(std::function<void()> notifier) {
  completionNotifier_ = std::move(notifier);
}

bool Engine::tick(double now) {
  model_.checkHealth();
  nextHealthCheckMilliseconds_ = now + kHealthCheckIntervalMilliseconds;
  bool progressed = scheduler_.expireDeadlines(now);
  const bool draining = drainingForRecovery();
  for (auto &[_, active] : requests_) {
    // Admission is deliberately paused while resident peers finish. Start a
    // fresh resource wait only if admission still fails after that drain.
    if (draining) {
      active.resourceWait.deadlineMilliseconds = 0.0;
      continue;
    }
    if (!active.finalized && active.resourceWait.deadlineMilliseconds > 0.0 &&
        now >= active.resourceWait.deadlineMilliseconds) {
      finishFailure(active, {"resource_timeout", "memory did not become available within the resource wait limit", true});
      progressed = true;
    }
  }
  // Finalize expired requests now, even while a command is in flight, so a
  // late mask or cancel for them is a no-op rather than a scheduler error.
  if (progressed)
    sweepTerminal();
  if (pending_) {
    auto forwardMaskRequests = [&] {
      for (ModelMaskRequest &request : pending_->ticket->takeMaskRequests()) {
        events_.maskRequested(request.requestId, request.simulationTokens);
        progressed = true;
      }
    };
    forwardMaskRequests();
    for (const BatchItem &item : pending_->plan.items) {
      Request &active = request(item.requestId);
      if (!active.failure &&
          active.request.deadlineMilliseconds <= now) {
        active.failure =
            Failure{"deadline_exceeded", "request deadline exceeded"};
        pending_->ticket->abandonMask(item.requestId);
        progressed = true;
      }
    }
    // A mask response, cancellation, or deadline can make the commit tail
    // runnable without another Metal completion wake.
    forwardMaskRequests();
    if (!pending_->ticket->ready())
      return progressed;
    Pending command = std::move(*pending_);
    pending_.reset();
    std::vector<ModelStepResult> results = command.ticket->wait();
    apply(command.plan, results, command.ticket->wallMilliseconds(),
          command.ticket->prefillTimingIsRepresentative());
    sweepTerminal();
    return true;
  }

  progressed = admitQueued(now) || progressed;
  sweepTerminal();
  auto plan = scheduler_.next();
  if (!plan)
    return progressed;
  std::vector<ModelBatchItem> items;
  const PrepareOutcome prepared = prepare(*plan, items, now);
  if (prepared == PrepareOutcome::BackedOff)
    return progressed;
  if (prepared == PrepareOutcome::Yielded) {
    sweepTerminal();
    return true;
  }

  std::unique_ptr<ModelBatchTicket> ticket =
      model_.submit(*plan, items, completionNotifier_);
  if (!ticket) {
    throw std::logic_error("model returned an empty command ticket");
  }
  scheduler_.commit(*plan);
  pending_ = Pending{std::move(*plan), std::move(ticket)};
  return true;
}

bool Engine::idle() const noexcept { return requests_.empty() && !pending_; }

bool Engine::drainingForRecovery() const {
  return recoveringResources_ &&
         std::any_of(requests_.begin(), requests_.end(), [](const auto &entry) {
           return entry.second.stateCell.has_value();
         });
}

std::optional<double> Engine::nextWakeupMilliseconds() const {
  std::optional<double> result;
  if (pending_ || model_.needsHealthCheck())
    result = nextHealthCheckMilliseconds_;
  const bool draining = drainingForRecovery();
  // admitQueued() retries only suspended requests while one recovers, so a
  // non-resident, non-suspended waiter's retry time (0 once its epoch is
  // stale) must not wake the loop; its deadlines still do.
  const bool recovering =
      recoveringResources_ &&
      std::any_of(requests_.begin(), requests_.end(),
                  [](const auto &entry) { return entry.second.suspended; });
  for (const auto &[_, active] : requests_) {
    if (active.finalized || active.failure)
      continue;
    if (!result || active.request.deadlineMilliseconds < *result)
      result = active.request.deadlineMilliseconds;
    if (!draining && active.resourceWait.deadlineMilliseconds > 0.0 &&
        (!result || active.resourceWait.deadlineMilliseconds < *result))
      result = active.resourceWait.deadlineMilliseconds;
    if (draining || active.resourceWait.retryMilliseconds <= 0.0)
      continue;
    if (recovering && !active.suspended && !active.stateCell)
      continue;
    const double wakeup = active.resourceWait.epoch == resourceEpoch_
                              ? active.resourceWait.retryMilliseconds
                              : 0.0;
    if (!result || wakeup < *result)
      result = wakeup;
  }
  return result;
}

EngineSnapshot Engine::snapshot() const {
  EngineSnapshot result = counters_;
  result.maximumContextTokens = config_.maxContext;
  result.scheduler = scheduler_.snapshot();
  result.resources = cache_.snapshot();
  return result;
}

ResourceWaitSnapshot Engine::resourceWaitSnapshot(double now) const {
  ResourceWaitSnapshot result;
  result.draining = drainingForRecovery();
  for (const auto &[id, active] : requests_) {
    if (active.finalized)
      continue;
    // A resident lane backing off waits on memory without leaving its phase.
    if (active.backoffStartedMilliseconds) {
      ++result.memory;
      result.oldestWaitMilliseconds =
          std::max(result.oldestWaitMilliseconds,
                   now - *active.backoffStartedMilliseconds);
      continue;
    }
    const Phase phase = scheduler_.phase(id);
    if (phase == Phase::Queued && active.scheduleWaitStartedMilliseconds) {
      ++result.schedule;
      result.oldestWaitMilliseconds =
          std::max(result.oldestWaitMilliseconds,
                   now - *active.scheduleWaitStartedMilliseconds);
      continue;
    }
    if (phase != Phase::WaitingResources)
      continue;
    if (active.resourceWait.reason == StateFailure::ConcurrencyLimit)
      ++result.concurrency;
    else
      ++result.memory;
    if (active.suspended)
      ++result.suspended;
    if (active.resourceWait.startedMilliseconds)
      result.oldestWaitMilliseconds = std::max(
          result.oldestWaitMilliseconds, now - *active.resourceWait.startedMilliseconds);
  }
  return result;
}

bool Engine::admitQueued(double now) {
  const bool recovering = std::any_of(
      requests_.begin(), requests_.end(),
      [](const auto &entry) { return entry.second.suspended; });
  // Once pressure has preempted work, let resident lanes finish before
  // spending their released headroom on a retry or a new request. Recover
  // one lane at a time; this prevents repeated B4 admission/preemption churn.
  if (!recovering)
    recoveringResources_ = false;
  if (drainingForRecovery())
    return false;
  const std::vector<uint64_t> order = scheduler_.admissionOrder();
  if (recovering) {
    for (uint64_t id : order) {
      Request &active = request(id);
      if (active.suspended && resourceRetryReady(active, now) && admit(active, now))
        return true;
      // A new request waits behind recovery as it would for rows.
      if (!active.suspended && scheduler_.phase(id) == Phase::Queued &&
          !active.scheduleWaitStartedMilliseconds)
        active.scheduleWaitStartedMilliseconds = now;
    }
    return false;
  }

  std::vector<PrefillAdmission> candidates;
  for (uint64_t id : order) {
    Request &active = request(id);
    if (!resourceRetryReady(active, now))
      continue;
    active.admissionProbe =
        cache_.probe(active.request.prompt, active.request.images);
    const uint32_t cached = active.admissionProbe->cachedTokens();
    if (pendingSharedPrefill(active, cached)) {
      active.admissionProbe.reset();
      active.resourceWait = {};
      scheduler_.waitForPrefix(id);
      continue;
    }
    candidates.push_back({id, cached});
  }
  bool progressed = false;
  while (!candidates.empty()) {
    const auto selected = scheduler_.prefillAdmissionOrder(candidates);
    if (selected.empty())
      break;
    for (uint64_t id : selected) {
      progressed = admit(request(id), now) || progressed;
      std::erase_if(candidates, [id](const auto &value) {
        return value.requestId == id;
      });
    }
    // Failed admissions must not prevent other eligible work from running.
    if (progressed)
      break;
  }
  // Waiting for scheduling does not consume the memory-retry deadline.
  // Admitting only what the next prefill command serves is deliberate: under
  // fewest-remaining-first, equal cold prompts run one after another, which
  // minimizes mean time to first token (each one starts decoding as early as
  // possible) and reserves no cells, states or KV for a prompt that would
  // get no rows. A shorter arrival still overtakes; the wait is reported in
  // /status.admission.waiting_schedule and as the reply's queue time.
  for (const auto &candidate : candidates) {
    Request &active = request(candidate.requestId);
    active.admissionProbe.reset();
    if (!active.scheduleWaitStartedMilliseconds)
      active.scheduleWaitStartedMilliseconds =
          active.resourceWait.startedMilliseconds.value_or(now);
    active.resourceWait = {};
    scheduler_.deferAdmission(candidate.requestId);
  }
  return progressed;
}

uint32_t Engine::sharedPrefillBoundary(const Request &left,
                                       const Request &right) {
  const auto prompt = [](const Request &value) -> std::span<const uint32_t> {
    return value.exactTokens.empty()
               ? std::span<const uint32_t>(value.request.prompt)
               : std::span<const uint32_t>(value.exactTokens)
                     .first(value.promptTokens);
  };
  const auto a = prompt(left);
  const auto b = prompt(right);
  const auto end = std::mismatch(a.begin(), a.end(), b.begin(), b.end()).first;
  uint32_t boundary = std::min<uint32_t>(
      static_cast<uint32_t>(end - a.begin()),
      std::min(replayStateBoundary(left.promptTokens),
               replayStateBoundary(right.promptTokens)));
  boundary -= boundary % KvCache::pageTokens;
  if (!left.request.images.empty() || !right.request.images.empty()) {
    for (uint32_t offset = 0; offset < boundary; offset += KvCache::pageTokens) {
      if (blockImageIdentity(offset, KvCache::pageTokens, left.request.images) !=
          blockImageIdentity(offset, KvCache::pageTokens, right.request.images))
        return offset;
    }
  }
  return boundary;
}

bool Engine::pendingSharedPrefill(const Request &active,
                                  uint32_t resumeBoundary) const {
  for (const auto &[id, peer] : requests_) {
    if (!peer.stateCell || peer.finalized || peer.failure ||
        peer.request.priority > active.request.priority ||
        scheduler_.phase(id) != Phase::Prefill)
      continue;
    const uint32_t shared = sharedPrefillBoundary(active, peer);
    for (size_t i = peer.stateBoundaryCursor; i < peer.stateBoundaries.size(); ++i) {
      const uint32_t boundary = peer.stateBoundaries[i].tokens;
      if (boundary > resumeBoundary && boundary <= shared)
        return true;
    }
  }
  return false;
}

bool Engine::admit(Request &active, double now) {
  const bool resuming = active.suspended;
  ModelRequest modelRequest = active.request.modelView();
  if (resuming)
    modelRequest.prompt = active.exactTokens;
  CacheLookup lookup = cache_.lookup(
      modelRequest.prompt, active.request.images,
      active.admissionProbe ? &*active.admissionProbe : nullptr);
  active.admissionProbe.reset();
  // Decided before any restore pins the chain: a target-only state too close
  // to its first planned boundary gives way to the next shallower state.
  while (!restorable(active, lookup))
    lookup = cache_.shallowerState(lookup);
  // Only unstarted requests wait for a resident producer. Recheck planned
  // boundaries each step so producer loss leaves no stale dependency or lease.
  if (!resuming && pendingSharedPrefill(active, lookup.resumeBoundary())) {
    active.resourceWait = {};
    scheduler_.waitForPrefix(active.request.id);
    return false;
  }
  bool executorStarted = false;
  bool resourcesStarted = false;
  try {
    const auto activate = [&] {
      return resuming ? model_.resume(modelRequest) : model_.begin(modelRequest);
    };
    const uint64_t releaseGeneration = cache_.releaseGeneration();
    // The lease drop below forgets the state this lookup found; a denied
    // unstarted request protects it by block from its first denial.
    const uint64_t foundStateBlock =
        lookup.state ? lookup.state->kvBlock() : 0;
    // Nothing in the retry loop below changes residency, so one scan serves
    // both the lease drop and the capacity check after it.
    const bool anotherResident = std::any_of(
        requests_.begin(), requests_.end(), [&](const auto &entry) {
          return entry.first != active.request.id && entry.second.stateCell;
        });
    StateAdmission admission = activate();
    if (!admission.granted() && !resuming) {
      // Protect the found state before any reclaim below: once the lease
      // drop forgets it, growth reclaim takes it only after every other
      // candidate, and Warning reclaim spares it while the request waits.
      // A retried wait swaps its protection rather than stack it.
      countEvictedQueuedPrefix(active);
      cache_.unprotectState(std::exchange(active.queuedProtection, {}));
      if (foundStateBlock)
        active.queuedProtection = cache_.protectState(foundStateBlock);
    }
    bool reclaimedForAdmission = false;
    while (!admission.granted() &&
           admission.failure == StateFailure::MemoryPressure) {
      const bool hostPressure =
          admission.allocationFailure == metal::AllocationFailure::HostPressure;
      if (hostPressure ? reclaimIdleState() : reclaimForGrowth(active)) {
        reclaimedForAdmission = true;
        admission = activate();
        continue;
      }
      if (hostPressure)
        break;
      // A useful restore remains pinned throughout ordinary eviction. If
      // that pin is the last obstacle to admitting even one lane, prefer
      // cold recomputation over waiting forever for our own cache lease.
      // With a peer resident, its retirement may free the room instead, so
      // a young wait keeps the found state rather than evict it.
      if (!growthPaused() && lookup.state &&
          !(anotherResident &&
            isResourceWaitYoung(active.resourceWait.deadlineMilliseconds,
                                config_.resourceWaitTimeoutMilliseconds,
                                now))) {
        lookup = {};
        continue;
      }
      break;
    }
    if (!admission.granted()) {
      const bool terminalAllocation =
          admission.allocationFailure == metal::AllocationFailure::EngineBudget ||
          admission.allocationFailure == metal::AllocationFailure::DriverRejected;
      const bool releasingBudget = budgetMayRecover(
          admission.allocationFailure, releaseGeneration, reclaimedForAdmission);
      if (terminalAllocation && !growthPaused() && !anotherResident &&
          !releasingBudget) {
        finishFailure(active,
                      {"capacity_exhausted",
                       std::string("could not allocate request state: ") +
                           metal::allocationFailureName(
                               admission.allocationFailure),
                       true});
        return true;
      }
      scheduler_.waitForResources(active.request.id);
      deferResourceRetry(active, now, admission.failure);
      // Denied for host pressure beside a resident lane, the wait holds no
      // deadline, as a recovery drain does: that lane's progress or finish
      // returns the memory, so timing out would fail the request only for
      // waiting its turn. The first denial with no lane resident starts a
      // fresh deadline.
      if (anotherResident && admission.allocationFailure ==
                                 metal::AllocationFailure::HostPressure)
        active.resourceWait.deadlineMilliseconds = 0.0;
      return false;
    }
    executorStarted = true;
    cache_.beginRequest(active.request.id);
    resourcesStarted = true;
    active.stateCell = *admission.cell;
    const uint32_t resumeBoundary = lookup.resumeBoundary();
    if (lookup.state)
      cache_.restoreRequest(active.request.id, lookup);
    if (resuming) {
      const auto [kv, retryableBudget] =
          admitKv(active, active.resumeKvTargetTokens);
      if (!kv.granted()) {
        // The host continuation survives this failed admission. No recurrent
        // state restore or replay has run, and all temporary leases are freed.
        model_.suspend(active.request.id);
        cache_.endRequest(active.request.id);
        active.stateCell.reset();
        executorStarted = resourcesStarted = false;
        if (!growthPaused() && !retryableBudget &&
            kv.allocationFailure != metal::AllocationFailure::HostPressure) {
          finishCapacity(active, kv);
          return true;
        }
        deferResourceRetry(active, now);
        return false;
      }
      active.resumeKvTargetTokens = 0;
    }
    active.resourceWait = {};
    // Admitted: the wait, and the queued prefix's protection, end here.
    countEvictedQueuedPrefix(active);
    cache_.unprotectState(std::exchange(active.queuedProtection, {}));
    if (resuming)
      scheduler_.resumeFromResources(active.request.id, resumeBoundary,
                                     active.replayTokens);
    const bool targetOnlyRestore =
        lookup.state && !lookup.state->state()->carriesDraft();
    DraftContextPlan draft = configureDraftStatePlan(
        active, resumeBoundary, lookup.junctionBoundary(), targetOnlyRestore);
    if (targetOnlyRestore && !draft.draftStateRestoreSkipped) {
      throw std::logic_error("target-only state was planned to restore draft");
    }
    setLatestCheckpoint(active, {});
    retirePreviousCheckpoint(active);
    if (lookup.state) {
      model_.restore(active.request.id, resumeBoundary, lookup.state->state(),
                     !draft.draftStateRestoreSkipped);
      if (!resuming) {
        const KeptCheckpoints inherited =
            cache_.keptCheckpoint(lookup.state->kvBlock());
        active.inheritedCheckpoint = inherited.prefill;
        active.inheritedDecodeCheckpoint = inherited.decode;
        active.restoredStateBlock = lookup.state->kvBlock();
      }
      setLatestCheckpoint(active,
                          cache_.checkpointState(lookup.state->kvBlock()));
      // A restored endpoint already has the ordinary replay state we need.
      // Other restored progress points retain their rolling lifetime.
      if (active.latestCheckpoint &&
          resumeBoundary == replayStateBoundary(active.replayTokens)) {
        static_cast<void>(
            cache_.reuseCompositeState(active.latestCheckpoint.kvBlock));
        ++counters_.deduplicatedStatePublications;
        setLatestCheckpoint(active, {});
      }
    }
    model_.setDraftContextPlan(active.request.id, std::move(draft));
    active.admissionStartTokens = resumeBoundary;
    if (!resuming)
      active.timing.prefillStartTokens = resumeBoundary;
    if (resuming) {
      // A failed resume attempt returns above and keeps the exemption.
      countEvictedResumeState(active);
      cache_.unprotectState(std::exchange(active.resumeProtection, {}));
      active.suspended = false;
      active.replaying = true;
      ++counters_.resourceResumptions;
      counters_.resourceResumeRestoredTokens += resumeBoundary;
      return true;
    }
    active.exactTokens = std::move(active.request.prompt);
    scheduler_.resourcesReady(active.request.id, resumeBoundary);
    cache_.recordLookup(lookup);
    events_.started(active.request.id,
                    resumeBoundary ? EngineCacheStatus::PrefixHit
                                   : EngineCacheStatus::Miss,
                    resumeBoundary, *admission.cell);
    if (active.request.returnProgress) {
      active.reportedPromptTokens = resumeBoundary;
      events_.promptProgress(active.request.id, resumeBoundary);
    }
    if (resumeBoundary) {
      ++counters_.cacheHits;
      counters_.reusedTokens += resumeBoundary;
    } else {
      ++counters_.coldMisses;
    }
    return true;
  } catch (...) {
    discardPendingStateBoundaries(active);
    if (executorStarted)
      model_.end(active.request.id);
    if (resourcesStarted)
      cache_.endRequest(active.request.id);
    active.stateCell.reset();
    throw;
  }
}

bool Engine::resourceRetryReady(const Request &active,
                                double now) const noexcept {
  return active.resourceWait.retryMilliseconds <= 0.0 ||
         active.resourceWait.epoch != resourceEpoch_ ||
         now >= active.resourceWait.retryMilliseconds;
}

void Engine::deferResourceRetry(Request &active, double now,
                                StateFailure reason) noexcept {
  auto &wait = active.resourceWait;
  if (!wait.startedMilliseconds)
    wait.startedMilliseconds = now;
  wait.reason = reason;
  if (reason == StateFailure::ConcurrencyLimit)
    wait.deadlineMilliseconds = 0.0;
  else if (wait.deadlineMilliseconds <= 0.0)
    wait.deadlineMilliseconds = now + config_.resourceWaitTimeoutMilliseconds;
  wait.epoch = resourceEpoch_;
  wait.retryMilliseconds = now + kResourceRetryBackoffMilliseconds;
}

void Engine::signalResourceProgress() noexcept {
  if (resourceEpoch_ != std::numeric_limits<uint64_t>::max())
    ++resourceEpoch_;
}

DraftContextPlan Engine::configureDraftStatePlan(Request &active,
                                                 uint32_t stateBoundary,
                                                 uint32_t junctionBoundary,
                                                 bool targetOnlyRestore) {
  if (!active.stateBoundaries.empty() || active.stateBoundaryCursor != 0) {
    throw std::logic_error("request already has a composite-state plan");
  }

  const auto addCandidate = [&](uint32_t tokens,
                                Request::StateBoundary::Purpose purpose) {
    if (!tokens || tokens <= stateBoundary)
      return;
    for (size_t index = 0; index < active.stateBoundaries.size(); ++index) {
      if (active.stateBoundaries[index].tokens != tokens)
        continue;
      if (purpose > active.stateBoundaries[index].purpose)
        active.stateBoundaries[index].purpose = purpose;
      return;
    }
    active.stateBoundaries.push_back({tokens, purpose});
  };

  // Plan draft windows before prefill; arbitrary chunk ends do not carry a
  // complete draft state. Progress points remain disposable after restoration.
  const uint32_t latestReplayBoundary = replayStateBoundary(active.replayTokens);
  if (const uint32_t interval = config_.prefillCheckpointTokens) {
    for (uint64_t boundary = (uint64_t{stateBoundary} / interval + 1) * interval;
         boundary < latestReplayBoundary; boundary += interval) {
      addCandidate(static_cast<uint32_t>(boundary),
                   Request::StateBoundary::Purpose::Checkpoint);
    }
  }
  addCandidate(junctionBoundary, Request::StateBoundary::Purpose::Junction);
  addCandidate(latestReplayBoundary, Request::StateBoundary::Purpose::Replay);
  std::sort(active.stateBoundaries.begin(), active.stateBoundaries.end(),
            [](const Request::StateBoundary &left,
               const Request::StateBoundary &right) {
              return left.tokens < right.tokens;
            });

  static_cast<void>(addSharedPrefillBoundaries(active, stateBoundary));

  try {
    DraftContextPlan draft =
        pendingDraftStatePlan(active, stateBoundary, targetOnlyRestore);
    armNextStateBoundary(active);
    return draft;
  } catch (...) {
    discardPendingStateBoundaries(active);
    throw;
  }
}

uint32_t Engine::peerJunction(const Request &active, uint64_t peerId,
                              const Request &peer, uint32_t after) {
  if (active.suspended || active.replaying || peerId == active.request.id ||
      peer.stateCell || peer.suspended || peer.finalized || peer.failure ||
      peer.request.priority < active.request.priority)
    return 0;
  const uint32_t shared = sharedPrefillBoundary(active, peer);
  return shared > after && shared < replayStateBoundary(active.replayTokens)
             ? shared
             : 0;
}

uint32_t Engine::firstPlannedBoundary(const Request &active,
                                      uint32_t stateBoundary,
                                      uint32_t junctionBoundary) const {
  // Mirrors the candidates configureDraftStatePlan plans; the draft plan
  // ends at replayTokens when no state boundary lies before it.
  uint32_t first = active.replayTokens;
  const auto consider = [&](uint32_t tokens) {
    if (tokens > stateBoundary && tokens < first)
      first = tokens;
  };
  const uint32_t replay = replayStateBoundary(active.replayTokens);
  if (const uint32_t interval = config_.prefillCheckpointTokens) {
    const uint64_t checkpoint = (uint64_t{stateBoundary} / interval + 1) * interval;
    if (checkpoint < replay)
      consider(static_cast<uint32_t>(checkpoint));
  }
  consider(junctionBoundary);
  consider(replay);
  for (const auto &[id, peer] : requests_)
    consider(peerJunction(active, id, peer, stateBoundary));
  return first;
}

// Replay rebuilds a draft window only when the first planned boundary
// captures a whole one; a shorter span would continue draft rows a
// target-only state at stateBoundary does not hold.
bool Engine::rebuildsDraftWindow(const Request &active, uint32_t stateBoundary,
                                 uint32_t junctionBoundary) const {
  return firstPlannedBoundary(active, stateBoundary, junctionBoundary) -
             stateBoundary >=
         model::ExecutionLimits::draftContextTokens;
}

bool Engine::restorable(const Request &active,
                        const CacheLookup &lookup) const {
  // Without speculation no draft window is ever continued or rebuilt.
  if (!config_.isSpeculationEnabled || !lookup.state ||
      lookup.state->state()->carriesDraft())
    return true;
  return rebuildsDraftWindow(active, lookup.resumeBoundary(),
                             lookup.junctionBoundary());
}

bool Engine::addSharedPrefillBoundaries(Request &active, uint32_t after) {
  bool changed = false;
  for (const auto &[id, peer] : requests_) {
    const uint32_t shared = peerJunction(active, id, peer, after);
    if (!shared)
      continue;
    auto found = std::lower_bound(
        active.stateBoundaries.begin(), active.stateBoundaries.end(), shared,
        [](const auto &point, uint32_t tokens) { return point.tokens < tokens; });
    if (found == active.stateBoundaries.end() || found->tokens != shared) {
      active.stateBoundaries.insert(
          found, {shared, Request::StateBoundary::Purpose::Junction});
      changed = true;
    } else if (found->purpose == Request::StateBoundary::Purpose::Checkpoint) {
      found->purpose = Request::StateBoundary::Purpose::Junction;
    }
  }
  return changed;
}

DraftContextPlan Engine::pendingDraftStatePlan(const Request &active,
                                               uint32_t stateBoundary,
                                               bool targetOnlyRestore) const {
  const std::optional<uint32_t> restoredBoundary =
      stateBoundary ? std::optional<uint32_t>(stateBoundary) : std::nullopt;
  if (!config_.isSpeculationEnabled)
    return undraftedPlan(stateBoundary, active.replayTokens, restoredBoundary,
                         targetOnlyRestore);
  std::vector<uint32_t> boundaries;
  boundaries.reserve(active.stateBoundaries.size() - active.stateBoundaryCursor);
  for (size_t i = active.stateBoundaryCursor; i < active.stateBoundaries.size(); ++i)
    boundaries.push_back(active.stateBoundaries[i].tokens);
  return planDraftContext(stateBoundary, active.replayTokens, restoredBoundary,
                          boundaries, targetOnlyRestore);
}

void Engine::armNextStateBoundary(Request &active) {
  const std::optional<uint32_t> next =
      active.stateBoundaryCursor < active.stateBoundaries.size()
          ? std::optional<uint32_t>(
                active.stateBoundaries[active.stateBoundaryCursor].tokens)
          : std::nullopt;
  scheduler_.setPrefillBoundary(active.request.id, next);
}

void Engine::discardPendingStateBoundaries(Request &active) noexcept {
  active.stateBoundaries.clear();
  active.stateBoundaryCursor = 0;
}

// Whether another live lane holds checkpoint as its latest or kept previous
// recovery point. Walks the request table once.
bool Engine::peerHoldsCheckpoint(const Request &active,
                                 StateCheckpoint checkpoint) const {
  const auto same = [&](StateCheckpoint held) {
    return held.kvBlock == checkpoint.kvBlock &&
           held.publication == checkpoint.publication;
  };
  return std::any_of(requests_.begin(), requests_.end(), [&](const auto &entry) {
    const auto &peer = entry.second;
    return &peer != &active && !peer.finalized &&
           (same(peer.latestCheckpoint) || same(peer.previousCheckpoint));
  });
}

bool Engine::retireCheckpoint(Request &active) {
  // Shared progress points remain disposable under memory pressure, but a
  // lane's normal rolling replacement must not retire its peer's recovery point.
  const auto point = active.latestCheckpoint;
  if (point && peerHoldsCheckpoint(active, point)) {
    setLatestCheckpoint(active, {});
    return true;
  }
  if (!cache_.retireCheckpointState(active.latestCheckpoint))
    return false;
  setLatestCheckpoint(active, {});
  return true;
}

// Ends the lane's kept previous checkpoint and retires its state, unless a
// peer still holds it. A pinned state stays cached as a disposable checkpoint.
void Engine::retirePreviousCheckpoint(Request &active) {
  const StateCheckpoint point = std::exchange(active.previousCheckpoint, {});
  cache_.unprotectState(
      std::exchange(active.previousCheckpointProtection, {}));
  if (point && !peerHoldsCheckpoint(active, point))
    static_cast<void>(cache_.retireCheckpointState(point));
}

// Ends the lane's hold on its kept previous checkpoint and leaves the state
// cached: a suspension may resume from it under its own protection.
void Engine::dropPreviousCheckpoint(Request &active) noexcept {
  active.previousCheckpoint = {};
  cache_.unprotectState(
      std::exchange(active.previousCheckpointProtection, {}));
}

// Points the lane's recovery point at checkpoint and moves the lane's
// protection with it, so no write leaves a stale protection count behind.
// Costs one state-table lookup per side: O(1) per publication.
void Engine::setLatestCheckpoint(Request &active, StateCheckpoint checkpoint) {
  cache_.unprotectState(std::exchange(active.checkpointProtection, {}));
  active.latestCheckpoint = checkpoint;
  if (checkpoint)
    active.checkpointProtection = cache_.protectState(checkpoint.kvBlock);
}

// At the prompt-end publication, moves the lane's deepest prefill checkpoint
// out of latestCheckpoint, and out of its protection, into keptCheckpoint, so
// the publication neither retires it before its capture nor after. A
// checkpoint on the prompt-end block itself is no extra state: the
// publication promotes it. O(1).
void Engine::keepDeepestCheckpoint(Request &active, uint64_t promptEndBlock) {
  active.promptEndPublished = true;
  if (!active.latestCheckpoint ||
      active.latestCheckpoint.kvBlock == promptEndBlock)
    return;
  active.keptCheckpoint = active.latestCheckpoint;
  setLatestCheckpoint(active, {});
}

// Records the conversation's one kept checkpoint on the prompt-end state at
// promptEndBlock. A lane whose kept checkpoint is still cached records it and
// retires the one it inherited, unless a peer holds that as a recovery point;
// otherwise the inherited handle is carried forward. O(1), plus one walk of
// the request table when retiring.
void Engine::recordKeptCheckpoint(Request &active, uint64_t promptEndBlock) {
  const StateCheckpoint own = active.keptCheckpoint;
  const StateCheckpoint cached = cache_.checkpointState(own.kvBlock);
  const bool keepsOwn = own && cached.publication == own.publication;
  if (!keepsOwn) {
    cache_.recordKeptCheckpoint(promptEndBlock,
                                {.prefill = active.inheritedCheckpoint});
    return;
  }
  const StateCheckpoint inherited = active.inheritedCheckpoint;
  const bool sameAsOwn = inherited.kvBlock == own.kvBlock &&
                         inherited.publication == own.publication;
  if (inherited && !sameAsOwn && !peerHoldsCheckpoint(active, inherited))
    static_cast<void>(cache_.retireCheckpointState(inherited));
  cache_.recordKeptCheckpoint(promptEndBlock, {.prefill = own});
}

// ============================================================================
// Whether checkpoint is a cached decode checkpoint of the lane's reply: one
// held past the prompt-end publication that is neither the conversation's
// kept prefill checkpoint nor the one the lane inherited. O(1).
bool Engine::isDecodeCheckpoint(const Request &active,
                                StateCheckpoint checkpoint) const noexcept {
  return checkpoint && active.promptEndPublished &&
         checkpoint.kvBlock != active.keptCheckpoint.kvBlock &&
         checkpoint.kvBlock != active.inheritedCheckpoint.kvBlock &&
         cache_.checkpointState(checkpoint.kvBlock).publication ==
             checkpoint.publication;
}

// ============================================================================
// At a Stop or Length finish, keeps the lane's latest decode checkpoint, its
// deepest, as the conversation's kept decode checkpoint instead of retiring
// it, so a follow-up prompt extending the reply resumes from it rather than
// re-prefilling the reply. Keeping its own retires the decode checkpoint the
// lane inherited, unless a peer holds it as a recovery point; a lane without
// its own retires its latest checkpoint as before and carries the inherited
// one forward, so a conversation keeps at most one. Runs once per finish:
// O(1), plus one walk of the request table when retiring.
void Engine::keepDecodeCheckpoint(Request &active) {
  const StateCheckpoint own = active.latestCheckpoint;
  const StateCheckpoint inherited = active.inheritedDecodeCheckpoint;
  if (!isDecodeCheckpoint(active, own)) {
    static_cast<void>(retireCheckpoint(active));
    recordDecodeCheckpoint(active, inherited);
    return;
  }
  if (inherited && inherited.kvBlock != own.kvBlock &&
      !peerHoldsCheckpoint(active, inherited))
    static_cast<void>(cache_.retireCheckpointState(inherited));
  recordDecodeCheckpoint(active, own);
  // The conversation end now protects it; the lane's own hold ends here.
  setLatestCheckpoint(active, {});
}

// ============================================================================
// Records decode as the conversation's kept decode checkpoint beside its
// prompt-end state and protects it with the conversation end. Its own state
// records the conversation's kept prefill checkpoint (the lane's own while
// cached, else the inherited one), so a follow-up that resumes from decode
// inherits that one and retires it once it keeps its own. No-op for an empty
// or evicted handle. O(1).
void Engine::recordDecodeCheckpoint(Request &active, StateCheckpoint decode) {
  if (!decode ||
      cache_.checkpointState(decode.kvBlock).publication != decode.publication)
    return;
  const StateCheckpoint own = active.keptCheckpoint;
  const bool ownCached =
      own && cache_.checkpointState(own.kvBlock).publication == own.publication;
  const StateCheckpoint prefill = ownCached ? own : active.inheritedCheckpoint;
  cache_.recordKeptCheckpoint(active.conversationEnd.promptEnd.kvBlock,
                              {.prefill = prefill, .decode = decode});
  cache_.recordKeptCheckpoint(decode.kvBlock, {.prefill = prefill});
  cache_.unprotectState(active.conversationEnd.decodeCheckpoint);
  active.conversationEnd.decodeCheckpoint =
      cache_.protectState(decode.kvBlock);
}

// ============================================================================
// Protects the conversation's resume states once its prompt-end state at
// promptEndBlock is published: that state and the checkpoint recorded beside
// it, while it is still that checkpoint publication. O(1).
void Engine::protectConversationEnd(Request &active, uint64_t promptEndBlock) {
  releaseConversationEnd(active.conversationEnd);
  active.conversationEnd.promptEnd = cache_.protectState(promptEndBlock);
  const StateCheckpoint kept = cache_.keptCheckpoint(promptEndBlock).prefill;
  if (kept &&
      cache_.checkpointState(kept.kvBlock).publication == kept.publication)
    active.conversationEnd.keptCheckpoint = cache_.protectState(kept.kvBlock);
}

// ============================================================================
// Moves the finishing request's end into conversationEnds_ as the newest
// entry. The entry of the conversation it continues (holding the state its
// first admission restored, or its own prompt-end state) is replaced;
// otherwise a full ring pushes out its oldest entry. A replaced or pushed-out
// entry loses its protections. O(maximumBatchWidth) per finish.
void Engine::keepConversationEnd(Request &active) {
  ConversationEnd end = std::exchange(active.conversationEnd, {});
  if (!end)
    return;
  const auto first = conversationEnds_.begin();
  const auto last = first + static_cast<std::ptrdiff_t>(conversationEndCount_);
  auto dropped = std::find_if(first, last, [&](const ConversationEnd &entry) {
    return entry.holds(active.restoredStateBlock) ||
           entry.holds(end.promptEnd.kvBlock);
  });
  if (dropped == last && conversationEndCount_ == conversationEnds_.size())
    dropped = first;
  if (dropped != last) {
    releaseConversationEnd(*dropped);
    std::move(dropped + 1, last, dropped);
    --conversationEndCount_;
  }
  conversationEnds_[conversationEndCount_++] = end;
}

// ============================================================================
// Drops every protection of end and empties it. Stale handles are no-ops.
void Engine::releaseConversationEnd(ConversationEnd &end) noexcept {
  cache_.unprotectState(std::exchange(end.promptEnd, {}));
  cache_.unprotectState(std::exchange(end.keptCheckpoint, {}));
  cache_.unprotectState(std::exchange(end.decodeCheckpoint, {}));
}

// Counts one checkpoint publication failure in the total and under its cause,
// so the causes always sum to the total. Runs only on a failed publication.
void Engine::countCheckpointFailure(PublicationFailure cause) noexcept {
  ++counters_.checkpointPublicationFailures;
  ++counters_.checkpointPublicationFailureCauses[static_cast<std::size_t>(
      cause)];
}

// ============================================================================
// The protections active's own conversation holds on checkpoint's
// publication: the lane's latest, previous and kept handles, and the kept
// prefill and decode checkpoints of the conversation end it continues, the
// entry its finish will replace. A stale handle matches nothing.
// O(maximumBatchWidth).
uint32_t Engine::ownProtections(const Request &active,
                                StateCheckpoint checkpoint) const {
  const auto same = [&](StateProtection protection) {
    return protection.kvBlock == checkpoint.kvBlock &&
           protection.publication == checkpoint.publication;
  };
  uint32_t count = same(active.checkpointProtection) +
                   same(active.previousCheckpointProtection) +
                   same(active.conversationEnd.keptCheckpoint);
  const auto first = conversationEnds_.begin();
  const auto last = first + static_cast<std::ptrdiff_t>(conversationEndCount_);
  const auto continued = std::find_if(first, last, [&](const auto &entry) {
    return entry.holds(active.restoredStateBlock);
  });
  if (continued != last)
    count += same(continued->keptCheckpoint) +
             same(continued->decodeCheckpoint);
  return count;
}

// ============================================================================
// Recycles one checkpoint of the lane's own conversation for its denied
// capture: its latest checkpoint, which a decode publication replaces anyway,
// then its kept checkpoint, then the one it inherited. One that a peer lane,
// a suspended or queued request, or another conversation's end protects is
// skipped. True when one was evicted. Runs only on a denied capture: at most
// three O(maximumBatchWidth) counts.
bool Engine::recycleOwnCheckpoint(Request &active) {
  for (const StateCheckpoint candidate :
       {active.latestCheckpoint, active.keptCheckpoint,
        active.inheritedCheckpoint}) {
    if (!candidate ||
        !cache_.recycleCheckpointState(candidate,
                                       ownProtections(active, candidate)))
      continue;
    if (candidate.kvBlock == active.latestCheckpoint.kvBlock)
      setLatestCheckpoint(active, {});
    return true;
  }
  return false;
}

// ============================================================================
// Copies the lane's live state for a publication. When no slot is free it
// recycles one of the lane's own checkpoints and retries, then reclaims
// cached states, a checkpoint for a checkpoint, retrying after each; nullptr
// when that fails too. A reclaimed state of a live peer whose reserve is
// short refills that reserve rather than the pool, so reclaim goes on past
// it; it stops after the first state whose whole bytes reached the pool (a
// denial that outlasts a pooled slot is not a shortage), when nothing
// qualifies, or after model::ExecutionLimits::maximumBatchWidth states. Runs
// only on a denied capture; nothing per step.
std::shared_ptr<const CompositeState> Engine::captureState(Request &active,
                                                           bool checkpoint) {
  const uint64_t requestId = active.request.id;
  std::shared_ptr<const CompositeState> state = model_.snapshot(requestId);
  if (state)
    return state;
  if (recycleOwnCheckpoint(active))
    state = model_.snapshot(requestId);

  // Each refill makes a peer's reserve whole, so its later states reach the
  // pool: one round per lane bounds the walk.
  const bool takeResumeStates = !state && !anotherLaneResident(active);
  for (uint32_t round = 0;
       !state && round < model::ExecutionLimits::maximumBatchWidth; ++round) {
    const StateEviction eviction =
        cache_.reclaimOneState(checkpoint, takeResumeStates);
    if (!eviction)
      break;
    state = model_.snapshot(requestId);
    if (eviction.isWhollyReclaimed())
      break;
  }
  if (state)
    ++counters_.recycledStatePublications;
  return state;
}

void Engine::publishReachedStateBoundaries(Request &active,
                                           uint32_t promptProcessed) {
  bool materialized = false;
  while (active.stateBoundaryCursor < active.stateBoundaries.size() &&
         active.stateBoundaries[active.stateBoundaryCursor].tokens <=
             promptProcessed) {
    const Request::StateBoundary objective =
        active.stateBoundaries[active.stateBoundaryCursor++];
    const bool checkpoint =
        objective.purpose == Request::StateBoundary::Purpose::Checkpoint;
    const bool junction =
        objective.purpose == Request::StateBoundary::Purpose::Junction;
    uint64_t &failures = junction ? counters_.junctionMaterializationFailures
                                  : counters_.replayStatePublicationFailures;
    // A checkpoint failure also names its cause; Replay and Junction
    // failures count only their totals.
    const auto countFailure = [&](PublicationFailure cause) {
      if (checkpoint)
        countCheckpointFailure(cause);
      else
        ++failures;
    };
    uint64_t &publications = checkpoint ? counters_.checkpointPublications
                             : junction ? counters_.junctionMaterializations
                                        : counters_.replayStatePublications;
    // The scheduler ends a command exactly at an armed boundary; a boundary
    // passed inside a command has no materialized state to copy.
    if (objective.tokens != promptProcessed) {
      countFailure(PublicationFailure::MissedBoundary);
      continue;
    }
    materialized = true;
    try {
      const uint64_t block = cache_.blockAt(active.request.id, objective.tokens);
      // The prompt-end state, whatever purpose merged there, keeps the
      // deepest prefill checkpoint beside it rather than retiring it.
      const bool promptEnd =
          !checkpoint && !active.promptEndPublished &&
          objective.tokens == replayStateBoundary(active.promptTokens);
      if (promptEnd)
        keepDeepestCheckpoint(active, block);
      if (cache_.reuseCompositeState(block, checkpoint)) {
        ++counters_.deduplicatedStatePublications;
      } else {
        // Recycle the previous recovery point before allocating its
        // replacement; at the prompt end it was kept instead. A restore lease
        // can delay this optional publication.
        if (!retireCheckpoint(active) && checkpoint) {
          countFailure(PublicationFailure::RetireBlocked);
          continue;
        }
        std::shared_ptr<const CompositeState> state =
            captureState(active, checkpoint);
        if (!state) {
          countFailure(PublicationFailure::SnapshotDenied);
          continue;
        }
        cache_.publishCompositeState(block, std::move(state), checkpoint);
        ++publications;
      }
      if (promptEnd) {
        recordKeptCheckpoint(active, block);
        protectConversationEnd(active, block);
      }
      if (active.latestCheckpoint.kvBlock != block)
        static_cast<void>(retireCheckpoint(active));
      setLatestCheckpoint(active, checkpoint ? cache_.checkpointState(block)
                                             : StateCheckpoint{});
    } catch (const std::exception &) {
      countFailure(PublicationFailure::Error);
    }
  }
  // Late siblings can extend the remaining plan only where both target and
  // draft states are complete, never at an arbitrary in-flight chunk boundary.
  if (materialized && addSharedPrefillBoundaries(active, promptProcessed))
    model_.setDraftContextPlan(
        active.request.id, pendingDraftStatePlan(active, promptProcessed));
  if (active.stateBoundaryCursor == active.stateBoundaries.size()) {
    active.stateBoundaries.clear();
    active.stateBoundaryCursor = 0;
  }
  armNextStateBoundary(active);
}

// The verify rows a decode step from storedTokens may retain so that its
// commit lands on, never past, the next multiple of prefillCheckpointTokens;
// 0 (no cap) while that boundary is beyond one step or checkpoints are
// disabled. One division per decode lane per step.
uint32_t
Engine::decodeRetainedTokenLimit(uint64_t storedTokens) const noexcept {
  const uint32_t interval = config_.prefillCheckpointTokens;
  if (!interval)
    return 0;
  const uint64_t untilBoundary = interval - storedTokens % interval;
  return untilBoundary < model::ExecutionLimits::targetVerifyRows
             ? static_cast<uint32_t>(untilBoundary)
             : 0;
}

// Rolls a decoding lane's checkpoint after a commit from previousTokens to
// storedTokens. On the commit that lands on the next multiple B of
// prefillCheckpointTokens, the live state is published at B as the lane's
// checkpoint, or a state already cached at B is reused; the checkpoint it
// replaces is kept until a later commit passes B, so a suspension at exactly
// B, which cannot restore B's own state, resumes from it rather than from the
// prompt end. A commit that passes B without landing counts one publication
// failure. Never arms a scheduler boundary. Costs one division per decode
// step and one state copy per interval; a lane holds two checkpoints only
// between landing on B and its next commit.
void Engine::publishDecodeCheckpoint(Request &active, uint32_t previousTokens,
                                     uint32_t storedTokens) {
  const uint32_t interval = config_.prefillCheckpointTokens;
  if (!interval || storedTokens <= previousTokens)
    return;
  if (active.previousCheckpoint)
    retirePreviousCheckpoint(active);
  const uint64_t boundary =
      (uint64_t{previousTokens} / interval + 1) * interval;
  if (storedTokens < boundary)
    return;
  if (storedTokens > boundary) {
    countCheckpointFailure(PublicationFailure::DecodeOvershoot);
    return;
  }
  try {
    const uint64_t block = cache_.blockAt(active.request.id, storedTokens);
    if (cache_.reuseCompositeState(block, true)) {
      ++counters_.deduplicatedStatePublications;
    } else {
      std::shared_ptr<const CompositeState> state =
          captureState(active, true);
      if (!state) {
        countCheckpointFailure(PublicationFailure::SnapshotDenied);
        return;
      }
      cache_.publishCompositeState(block, std::move(state), true);
      ++counters_.checkpointPublications;
    }
    if (active.latestCheckpoint.kvBlock != block) {
      active.previousCheckpoint = std::exchange(active.latestCheckpoint, {});
      active.previousCheckpointProtection =
          std::exchange(active.checkpointProtection, {});
    }
    setLatestCheckpoint(active, cache_.checkpointState(block));
  } catch (const std::exception &) {
    countCheckpointFailure(PublicationFailure::Error);
  }
}

Engine::PrepareOutcome Engine::prepare(BatchPlan &plan,
                                       std::vector<ModelBatchItem> &items,
                                       double now) {
  items.reserve(plan.items.size());
  std::vector<BatchItem> admitted;
  admitted.reserve(plan.items.size());
  struct Denied final {
    uint64_t requestId = 0;
    TokenAdmission admission;
    uint64_t workEnd;
    bool retryableBudget;
  };
  std::vector<Denied> denied;
  denied.reserve(plan.items.size());
  bool backedOff = false;
  for (const BatchItem &scheduled : plan.items) {
    Request &active = request(scheduled.requestId);
    if (!active.stateCell)
      throw std::logic_error("scheduled request is not resident");
    // nextWakeupMilliseconds skips every retry wake while draining, so a
    // draining engine sends the lane to admitKv, suspending it if denied.
    if (active.backoffStartedMilliseconds && !resourceRetryReady(active, now) &&
        !drainingForRecovery()) {
      backedOff = true;
      continue;
    }
    const uint64_t position = plan.kind == WorkKind::Prefill
                                  ? scheduled.promptOffset
                                  : active.exactTokens.size();
    const uint64_t workEnd =
        plan.kind == WorkKind::Prefill
            ? position + scheduled.tokenCount
            : position + model::ExecutionLimits::targetVerifyRows;
    const auto [admission, retryableBudget] = admitKv(active, workEnd);
    if (!admission.granted()) {
      denied.push_back(
          Denied{active.request.id, admission, workEnd, retryableBudget});
      continue;
    }
    if (active.backoffStartedMilliseconds) {
      active.backoffStartedMilliseconds.reset();
      active.resourceWait = {};
    }
    admitted.push_back(scheduled);
    ModelBatchItem item;
    item.requestId = active.request.id;
    item.stateSlot = *active.stateCell;
    item.logicalPosition = position;
    item.promptOffset =
        plan.kind == WorkKind::Prefill ? scheduled.promptOffset : 0;
    item.tokenCount = scheduled.tokenCount;
    if (plan.kind == WorkKind::Decode)
      item.retainedTokenLimit = decodeRetainedTokenLimit(position);
    const PageTableView pageTable = cache_.pageTable(active.request.id);
    item.pageTable = pageTable.pages;
    item.pageTableRevision = pageTable.revision;
    if (plan.kind == WorkKind::Prefill) {
      item.inputTokens =
          std::span<const uint32_t>(active.exactTokens)
              .subspan(scheduled.promptOffset, scheduled.tokenCount);
    }
    items.push_back(std::move(item));
  }
  if (!admitted.empty()) {
    plan.items = std::move(admitted);
    return PrepareOutcome::Submit;
  }

  // Partial admissions execute at their actual width. If no lane fits, choose
  // among all runnable residents: an unstarted peer can release its state
  // cell before completed prefill is discarded.
  if (denied.empty()) {
    if (backedOff)
      return PrepareOutcome::BackedOff;
    throw std::logic_error("empty resource admission result");
  }
  const auto yieldsBefore = [&](const Request &a, const Request &b) {
    if (a.request.priority != b.request.priority)
      return a.request.priority > b.request.priority;
    const Phase aPhase = scheduler_.phase(a.request.id);
    const Phase bPhase = scheduler_.phase(b.request.id);
    // At equal priority, prefer uninterrupted streaming over less replay work.
    if (aPhase != bPhase)
      return aPhase == Phase::Prefill;
    return completedTokens(a) < completedTokens(b);
  };
  const Denied &victim = *std::min_element(
      denied.begin(), denied.end(),
      [&](const Denied &left, const Denied &right) {
        return yieldsBefore(request(left.requestId), request(right.requestId));
      });
  Request *selected = &request(victim.requestId);
  uint64_t resumeTarget = victim.workEnd;
  for (auto &[id, candidate] : requests_) {
    if (!candidate.stateCell)
      continue;
    // Requests enter the scheduler before they can acquire a resident cell.
    const Phase phase = scheduler_.phase(id);
    if ((phase == Phase::Prefill || phase == Phase::Decode) &&
        yieldsBefore(candidate, *selected)) {
      selected = &candidate;
      // This peer has not failed a growth attempt. Retain its current KV
      // capacity as the resume target, not the blocked lane's requirement.
      resumeTarget =
          uint64_t{cache_.pageTable(id).pages.size()} * KvCache::pageTokens;
    }
  }
  Request &active = *selected;
  if (backOffSoleLane(active, victim.admission, now))
    return PrepareOutcome::BackedOff;
  const bool anotherResident =
      std::any_of(requests_.begin(), requests_.end(), [&](const auto &entry) {
        return entry.first != active.request.id && entry.second.stateCell;
      });
  if (growthPaused() || anotherResident || victim.retryableBudget ||
      victim.admission.allocationFailure ==
          metal::AllocationFailure::HostPressure) {
    suspendForGrowth(active, resumeTarget, now);
    return PrepareOutcome::Yielded;
  }
  finishCapacity(request(victim.requestId), victim.admission);
  return PrepareOutcome::Yielded;
}

// ============================================================================
// Holds the sole resident lane, prefill or decode, denied for host pressure,
// instead of suspending it into a replay of its completed work: it retries
// after kResourceRetryBackoffMilliseconds, or sooner on resource progress,
// while no Queued or WaitingResources request has a more important priority
// and no request is suspended. Once resourceWaitTimeoutMilliseconds have
// passed since its first denial it suspends as usual, with a fresh resource
// wait. Runs on the denial path only: one scan of the requests, no allocation.
bool Engine::backOffSoleLane(Request &active, const TokenAdmission &admission,
                             double now) {
  if (admission.allocationFailure != metal::AllocationFailure::HostPressure ||
      drainingForRecovery())
    return false;
  const std::optional<double> started = active.backoffStartedMilliseconds;
  if (started && now >= *started + config_.resourceWaitTimeoutMilliseconds)
    return false;
  for (const auto &[id, peer] : requests_) {
    if (id == active.request.id || peer.finalized)
      continue;
    if (peer.stateCell)
      return false;
    const Phase phase = scheduler_.phase(id);
    if ((phase == Phase::Queued || phase == Phase::WaitingResources) &&
        peer.request.priority < active.request.priority)
      return false;
  }
  if (!started)
    active.backoffStartedMilliseconds = now;
  active.resourceWait.epoch = resourceEpoch_;
  active.resourceWait.retryMilliseconds = now + kResourceRetryBackoffMilliseconds;
  return true;
}

Engine::KvAdmission Engine::admitKv(Request &active, uint64_t workEnd) {
  const uint64_t releaseGeneration = cache_.releaseGeneration();
  bool reclaimed = false;
  TokenAdmission admission = cache_.ensureTokens(active.request.id, workEnd);
  while (!admission.granted() &&
         admission.failure == KvPageAcquireFailure::PhysicalCapacity) {
    const bool paused = growthPaused() ||
        admission.allocationFailure == metal::AllocationFailure::HostPressure;
    const bool progressed = paused
        ? reuseIdleBackingWhilePaused(active, admission)
        : reclaimForGrowth(active, CacheReclaimMode::ReuseBacking);
    if (!progressed)
      break;
    reclaimed = true;
    admission = cache_.ensureTokens(active.request.id, workEnd);
  }
  return {admission,
          !admission.granted() &&
              budgetMayRecover(admission.allocationFailure, releaseGeneration,
                               reclaimed)};
}

bool Engine::budgetMayRecover(metal::AllocationFailure failure,
                             uint64_t generation, bool reclaimed) const {
  if (failure != metal::AllocationFailure::EngineBudget)
    return false;
  const bool pending = cache_.releasePending();
  return pending || reclaimed || cache_.releaseGeneration() != generation;
}

bool Engine::growthPaused() const {
  return config_.growthPaused && config_.growthPaused();
}

// ============================================================================
// One reclaim step for the request's denied growth or admission. A suspended
// lane's resume state is taken only when no other lane is resident: the sole
// resident lane's fallback, or the admission a recovery resumes first.
bool Engine::reclaimForGrowth(const Request &active, CacheReclaimMode mode) {
  if (reclaimIdleState())
    return true;
  // The background pressure controller owns physical shrink. Retrying a
  // paused allocator here would drain the cache before macOS can acknowledge
  // any reclaimed bytes.
  if (growthPaused())
    return false;
  const CacheReclaimResult reclaimed =
      cache_.reclaimOneForGrowth(mode, !anotherLaneResident(active));
  if (reclaimed.madeProgress)
    signalResourceProgress();
  return reclaimed.madeProgress;
}

bool Engine::reclaimIdleState() noexcept {
  if (!model_.reclaimIdleState())
    return false;
  signalResourceProgress();
  return true;
}

// Host pressure pauses growth, and the pressure controller owns physical
// shrink. Backing that stays resident is outside that accounting: a request
// short of pages may take idle cached pages instead of being suspended and
// replaying its whole prefix once the pause lifts. Cache is only evicted when
// the resident idle pages can actually cover the shortfall; otherwise the
// request yields as before and the cache survives for later hits. A suspended
// lane's resume state is taken only when no other lane is resident.
bool Engine::reuseIdleBackingWhilePaused(const Request &active,
                                         const TokenAdmission &admission) {
  if (reclaimIdleState())
    return true;
  const KvPoolSnapshot pool = cache_.snapshot().pool;
  // Cached prefixes can also have active owners; those pages cannot be reused.
  const uint32_t reusable = pool.pagesResident - pool.pagesActive;
  if (reusable < admission.additionalPages)
    return false;
  const CacheReclaimResult reused = cache_.reclaimOneForGrowth(
      CacheReclaimMode::ReuseBacking, !anotherLaneResident(active));
  if (reused.madeProgress) {
    // An evicted state parks its buffers in the model's pool (a live lane's
    // own state refills that lane's short reserve instead); under
    // pressure that memory goes back to the host now rather than waiting for
    // the next background pass.
    while (model_.reclaimIdleState()) {
    }
    signalResourceProgress();
  }
  return reused.madeProgress;
}

// ============================================================================
// True while a lane other than the request holds a resident state cell.
// Runs on denial paths only: one scan of the requests, no allocation.
bool Engine::anotherLaneResident(const Request &active) const {
  return std::any_of(requests_.begin(), requests_.end(), [&](const auto &entry) {
    return entry.first != active.request.id && entry.second.stateCell;
  });
}

// Tokens a resident lane has processed: the prompt position during prefill,
// the whole exact history (prompt plus committed output) during decode.
uint32_t Engine::completedTokens(const Request &active) const {
  return scheduler_.phase(active.request.id) == Phase::Prefill
             ? scheduler_.promptProcessed(active.request.id)
             : static_cast<uint32_t>(active.exactTokens.size());
}

// Suspends the lane's model. Mid-prefill (0 < position < the exact history)
// at a page boundary whose block carries no state, the lane's live state is
// moved out and published there as its checkpoint, replacing the previous
// one, so the resume restores at position instead of replaying from an
// earlier state. A decode suspension publishes nothing: its position is the
// whole history. Expects the lane marked suspended with its replay length
// set, so the target-only test sees the plan its resume will make. Costs one
// state-table lookup and a pointer move: no copy, no allocation.
void Engine::suspendModel(Request &active, uint32_t position) {
  const uint64_t id = active.request.id;
  const bool midPrefill = position && position < active.exactTokens.size() &&
                          position % KvCache::pageTokens == 0;
  const uint64_t block = midPrefill ? cache_.blockAt(id, position) : 0;
  if (!block || cache_.carriesState(block)) {
    model_.suspend(id);
    return;
  }
  std::shared_ptr<const CompositeState> state =
      model_.suspendWithState(id, position);
  if (!state)
    return;
  // A hint: admit's restorable check is authoritative. A KV junction past
  // position, unknown here, only lowers the first boundary admit checks.
  const bool usable = !config_.isSpeculationEnabled || state->carriesDraft() ||
                      rebuildsDraftWindow(active, position, 0);
  const bool retiring = usable && active.latestCheckpoint;
  if (usable) {
    static_cast<void>(retireCheckpoint(active));
    cache_.publishCompositeState(block, std::move(state), true);
    ++counters_.checkpointPublications;
    setLatestCheckpoint(active, cache_.checkpointState(block));
  }
  state.reset();
  // The model drained its idle pool as it suspended; a dropped state or a
  // retired checkpoint parks its buffers there, which the suspension returns
  // to the host now.
  if (!usable || retiring) {
    while (model_.reclaimIdleState()) {
    }
  }
}

void Engine::suspendForGrowth(Request &active, uint64_t workEnd, double now) {
  if (!active.stateCell || active.suspended) {
    throw std::logic_error("request cannot be suspended for growth");
  }
  if (!active.stateBoundaries.empty()) {
    discardPendingStateBoundaries(active);
    scheduler_.setPrefillBoundary(active.request.id, std::nullopt);
  }
  const uint32_t position = completedTokens(active);
  const bool decodePhase =
      scheduler_.phase(active.request.id) != Phase::Prefill;
  active.suspended = true;
  active.replayTokens = static_cast<uint32_t>(active.exactTokens.size());
  suspendModel(active, position);
  // Warning reclaim must not turn the resume into a full replay, and neither
  // must another lane's growth or publication: while another lane is
  // resident they never take this state, the one published at suspension
  // included. Critical reclaim takes it outright, and so does the sole
  // resident lane's growth, or an admission with no lane resident, after
  // every unprotected one.
  // A decode lane suspended exactly on its checkpoint boundary protects the
  // kept previous checkpoint: its resume cannot restore the boundary's state.
  active.resumeProtection =
      cache_.protectDeepestState(active.request.id, active.replayTokens);
  // A suspended lane's rolling recovery points are protected only as the
  // resume state, when one is the deepest usable one. Its conversation end,
  // once published, stays protected until the request finishes or ends.
  cache_.unprotectState(std::exchange(active.checkpointProtection, {}));
  dropPreviousCheckpoint(active);
  cache_.endRequest(active.request.id);
  active.stateCell.reset();
  active.resumeKvTargetTokens = workEnd;
  recoveringResources_ = true;
  // A suspension during resumed replay must keep counting the rows processed
  // before the first suspension.
  active.replayCountEnd = std::max(active.replayCountEnd, position);
  scheduler_.suspendForResources(active.request.id);
  // A lane that backed off has waited on memory since its first denial; the
  // suspension still takes a fresh wait deadline.
  if (active.backoffStartedMilliseconds)
    active.resourceWait.startedMilliseconds =
        std::exchange(active.backoffStartedMilliseconds, std::nullopt);
  deferResourceRetry(active, now);
  ++counters_.resourceSuspensions;
  ++active.timing.suspensions;
  if (decodePhase) {
    // Never cleared here: a suspension during the replay is Prefill phase
    // and the replay it interrupts still recomputes decode history.
    active.hasDecodeSuspension = true;
    ++counters_.resourceDecodeSuspensions;
    ++active.timing.decodeSuspensions;
  }
}

// A running lane whose next chunk overflows the free pages of partly used
// extents takes an empty extent. Unmapping the last one would make that chunk
// wait on a mapping a host-constrained governor denies, suspending the lane.
// The prefill chunk is bounded by the full token budget, so this errs toward
// keeping the runway. Costs one scheduler and one page-table lookup per
// resident request, without allocation. A request still waiting for admission
// keeps the runway through waitingRequestNeedsKvRunway() instead.
bool Engine::residentLaneNeedsKvRunway() const {
  const uint64_t packedFreePages = cache_.partlyUsedFreePages();
  uint64_t neededPages = 0;
  for (const auto &[id, active] : requests_) {
    if (!active.stateCell || active.suspended)
      continue;
    const Phase phase = scheduler_.phase(id);
    uint64_t chunkEnd = 0;
    if (phase == Phase::Prefill) {
      chunkEnd = std::min<uint64_t>(
          active.exactTokens.size(),
          uint64_t{scheduler_.promptProcessed(id)} +
              model::ExecutionLimits::prefillTokenBudget);
    } else if (phase == Phase::Decode || phase == Phase::WaitingMask) {
      chunkEnd = active.exactTokens.size() +
                 model::ExecutionLimits::targetVerifyRows;
    } else {
      continue;
    }
    const uint64_t chunkPages =
        (chunkEnd + KvCache::pageTokens - 1) / KvCache::pageTokens;
    const uint64_t heldPages = cache_.pageTable(id).pages.size();
    if (chunkPages > heldPages)
      neededPages += chunkPages - heldPages;
  }
  return neededPages > packedFreePages;
}

// A request waiting for admission (queued, waiting on resources or suspended)
// takes its first pages next; with no resident lane the last empty extent is
// the only mapping it can take while a host-constrained governor denies new
// ones. Costs one scheduler lookup per request, without allocation, and stops
// at the first waiter.
bool Engine::waitingRequestNeedsKvRunway() const {
  return std::any_of(requests_.begin(), requests_.end(), [&](const auto &entry) {
    const auto &[id, active] = entry;
    if (active.finalized || active.failure)
      return false;
    if (active.suspended)
      return true;
    const Phase phase = scheduler_.phase(id);
    return phase == Phase::Queued || phase == Phase::WaitingResources;
  });
}

uint64_t Engine::reclaimMemory(const MemoryReclaimDirective &directive) {
  if (!directive.reclaimEmptyKvExtents)
    return 0;

  uint64_t released = 0;
  while (const uint64_t idle = model_.reclaimIdleState())
    released += idle;
  const uint64_t remaining =
      released >= directive.targetBytes ? 0 : directive.targetBytes - released;
  // Critical pressure releases every empty KV extent. A warning pass, even a
  // zero-byte one, releases them too, except one kept mapped while a running
  // lane's next chunk or a waiting request's admission needs it; each check
  // walks the request table once per pressure pass, never per token. A
  // warning pass also spares each protected state, a suspended request's
  // resume state, a queued request's prefix, a resident lane's latest
  // checkpoint or a newest conversation end's resume states; Critical evicts
  // it.
  const bool warning = !directive.evictAllUnpinnedPrefixes;
  reclaimKeptKvRunway_ = warning && (residentLaneNeedsKvRunway() ||
                                     waitingRequestNeedsKvRunway());
  released += cache_.reclaimCache(remaining, directive.evictAllUnpinnedPrefixes,
                                 directive.keepResumePoint,
                                 reclaimKeptKvRunway_);
  // Evicted states park their buffers in the model's pool (a live lane's own
  // state refills that lane's short reserve instead); a pressure pass
  // returns that memory to the host now rather than keeping it warm.
  while (model_.reclaimIdleState()) {
  }
  if (released)
    signalResourceProgress();
  return released;
}

void Engine::apply(const BatchPlan &plan,
                   std::span<const ModelStepResult> results,
                   double wallMilliseconds, bool representativePrefillTiming) {
  if (results.size() != plan.items.size()) {
    throw std::logic_error("model result count changed");
  }
  std::vector<StepResult> schedulerResults;
  schedulerResults.reserve(results.size());
  uint32_t inputTokens = 0;
  uint32_t outputTokens = 0;
  uint32_t draftedTokens = 0;
  uint32_t acceptedDraftTokens = 0;
  const uint64_t wallMicros = wholeMicroseconds(wallMilliseconds);
  // The last lane whose prefill rows count; read only when inputTokens > 0.
  size_t lastCountedIndex = 0;
  for (size_t index = 0; index < results.size(); ++index) {
    const ModelStepResult &result = results[index];
    const BatchItem &item = plan.items[index];
    if (result.requestId != item.requestId) {
      throw std::logic_error("model result order changed");
    }
    Request &active = request(result.requestId);
    if (!result.failure.empty() && !active.failure) {
      // The model rejected this lane's own numerical result. An earlier
      // cancellation or deadline failure of the same lane still stands.
      active.failure = Failure{"model_result_invalid", result.failure};
    }
    if (active.failure) {
      // An in-flight Metal command cannot be revoked safely. Its provisional
      // writes remain invisible, but a cancelled, deadline-expired or
      // model-rejected request must not publish cache state or emit output
      // when that command drains.
      schedulerResults.push_back({active.request.id,
                                  result.consumedPromptTokens, true,
                                  result.nextDecodeStage});
      continue;
    }
    if (result.outputTokensWithoutKv > result.outputTokens.size()) {
      throw std::logic_error("model reported more uncommitted tokens than output");
    }
    if (plan.kind == WorkKind::Prefill) {
      const uint32_t promptProcessed =
          item.promptOffset + result.consumedPromptTokens;
      inputTokens += result.consumedPromptTokens;
      active.timing.prefilledTokens += result.consumedPromptTokens;
      active.timing.prefillWallMicros += wallMicros;
      if (result.consumedPromptTokens)
        lastCountedIndex = index;
      // Rows past the suspension point were never processed: ordinary prefill.
      if (active.replaying && item.promptOffset < active.replayCountEnd) {
        const uint32_t replayed =
            std::min(promptProcessed, active.replayCountEnd) - item.promptOffset;
        counters_.resourceReplayTokens += replayed;
        active.timing.replayTokens += replayed;
        if (active.hasDecodeSuspension)
          counters_.resourceDecodeReplayTokens += replayed;
      }
      if (promptProcessed == active.replayTokens) {
        active.replaying = false;
        active.hasDecodeSuspension = false;
      }
      static_cast<void>(cache_.publishCommittedBlocks(
          active.request.id, active.exactTokens, promptProcessed,
          active.request.images));
      publishReachedStateBoundaries(active, promptProcessed);
      // Recovery may replay an already reported prefix, including generated
      // history.
      const uint32_t processed = std::min(promptProcessed, active.promptTokens);
      if (active.request.returnProgress &&
          processed > active.reportedPromptTokens) {
        active.reportedPromptTokens = processed;
        events_.promptProgress(active.request.id, processed);
      }
    }
    const uint32_t previousTokens =
        static_cast<uint32_t>(active.exactTokens.size());
    uint32_t storedTokens = previousTokens;
    if (!result.outputTokens.empty()) {
      active.exactTokens.insert(active.exactTokens.end(),
                                result.outputTokens.begin(),
                                result.outputTokens.end());
      outputTokens += static_cast<uint32_t>(result.outputTokens.size());
      events_.tokens(active.request.id, result.outputTokens);
    }
    if (plan.kind == WorkKind::Decode) {
      draftedTokens += result.draftedTokens;
      acceptedDraftTokens += result.acceptedDraftTokens;
      ++active.timing.decodeSteps;
      active.timing.decodeWallMicros += wallMicros;
      active.timing.draftedTokens += result.draftedTokens;
      active.timing.acceptedDraftTokens += result.acceptedDraftTokens;
      // A terminal anchor is emitted without a target KV row; it never enters
      // a cached block.
      storedTokens = static_cast<uint32_t>(active.exactTokens.size()) -
                     result.outputTokensWithoutKv;
      static_cast<void>(cache_.publishCommittedBlocks(
          active.request.id, active.exactTokens, storedTokens,
          active.request.images));
    }
    const uint64_t completionTokens =
        active.exactTokens.size() - active.promptTokens;
    const bool scoring = !active.request.scoreTokens.empty();
    if (scoring && !result.outputTokens.empty()) {
      throw std::logic_error("score request produced output tokens");
    }
    if (!result.scoreLogits.empty()) {
      if (!scoring ||
          result.scoreLogits.size() != active.request.scoreTokens.size()) {
        throw std::logic_error("model returned mismatched score logits");
      }
      active.scoreLogits = result.scoreLogits;
    }
    // Score requests carry maxNewTokens == 0; only the model's finished flag
    // on the final prompt chunk completes them.
    const bool complete =
        result.finished ||
        (!scoring && completionTokens >= active.request.maxNewTokens);
    if (scoring && complete &&
        item.promptOffset + result.consumedPromptTokens !=
            active.promptTokens) {
      throw std::logic_error("score request finished before the prompt ended");
    }
    if (result.outputTokensWithoutKv && !complete) {
      throw std::logic_error("model emitted an uncommitted token and continued");
    }
    // A finishing step publishes nothing: finish retires the lane's points.
    if (plan.kind == WorkKind::Decode && !complete)
      publishDecodeCheckpoint(active, previousTokens, storedTokens);
    schedulerResults.push_back({active.request.id, result.consumedPromptTokens,
                                complete, result.nextDecodeStage});
    if (plan.kind == WorkKind::Decode && waitsForMask(result.nextDecodeStage)) {
      events_.maskRequested(active.request.id, {});
    }
  }
  if (plan.kind == WorkKind::Prefill && inputTokens)
    chargePrefillClasses(results, wallMicros, inputTokens, lastCountedIndex);
  scheduler_.complete(plan, schedulerResults, wallMilliseconds,
                      representativePrefillTiming);
  events_.batchCompleted(plan.kind, plan.width(), inputTokens, outputTokens,
                         draftedTokens, acceptedDraftTokens, wallMilliseconds);

  for (size_t index = 0; index < results.size(); ++index) {
    const ModelStepResult &result = results[index];
    Request &active = request(result.requestId);
    if (active.failure) {
      Failure failure = std::move(*active.failure);
      active.failure.reset();
      if (failure.code == kCancelledFailureCode) {
        finish(active, EngineFinishReason::Cancelled, {});
      } else if (failure.code == kStoppedFailureCode) {
        finish(active, EngineFinishReason::Stop, {});
      } else {
        finishFailure(active, std::move(failure));
      }
    } else if (schedulerResults[index].finished) {
      finish(active, result.finished ? EngineFinishReason::Stop
                                     : EngineFinishReason::Length,
             active.scoreLogits);
    }
  }
}

// ============================================================================
// Splits a prefill batch's wall between the cold and cached-tail classes:
// each counted lane (not failed, rows > 0) takes wallMicros x its rows /
// countedRows, and the last counted lane also takes the integer remainder,
// so the two classes gain exactly wallMicros. O(lanes), no allocation.
void Engine::chargePrefillClasses(std::span<const ModelStepResult> results,
                                  uint64_t wallMicros, uint32_t countedRows,
                                  size_t lastCountedIndex) {
  uint64_t chargedMicros = 0;
  for (size_t index = 0; index <= lastCountedIndex; ++index) {
    const ModelStepResult &result = results[index];
    const Request &active = request(result.requestId);
    if (active.failure || !result.consumedPromptTokens)
      continue;
    const uint64_t share =
        index == lastCountedIndex
            ? wallMicros - chargedMicros
            : wallMicros * result.consumedPromptTokens / countedRows;
    chargedMicros += share;
    if (active.admissionStartTokens) {
      counters_.cachedPrefillTokens += result.consumedPromptTokens;
      counters_.cachedPrefillWallMicros += share;
    } else {
      counters_.coldPrefillTokens += result.consumedPromptTokens;
      counters_.coldPrefillWallMicros += share;
    }
  }
}

void Engine::finish(Request &active, EngineFinishReason reason,
                    std::span<const float> optionLogits) {
  if (active.finalized)
    return;
  // A batch completion has already ended a finishing lane's scheduler phase;
  // a client stop between batches has not.
  if (reason == EngineFinishReason::Cancelled) {
    scheduler_.cancel(active.request.id);
  } else if (reason == EngineFinishReason::Stop) {
    scheduler_.stop(active.request.id);
  }
  active.finalized = true;
  const uint32_t completionTokens =
      active.exactTokens.size() > active.promptTokens
          ? static_cast<uint32_t>(active.exactTokens.size() -
                                  active.promptTokens)
          : 0;
  events_.completed(active.request.id, reason, active.promptTokens,
                    completionTokens, optionLogits, active.timing);
  if (reason == EngineFinishReason::Cancelled) {
    ++counters_.cancelled;
  } else {
    keepDecodeCheckpoint(active);
    retirePreviousCheckpoint(active);
    keepConversationEnd(active);
    ++counters_.completed;
  }
  release(active);
}

void Engine::finishFailure(Request &active, Failure failure) {
  if (active.finalized)
    return;
  scheduler_.fail(active.request.id);
  active.finalized = true;
  events_.failed(active.request.id, std::move(failure.code),
                 std::move(failure.message), failure.retryable);
  ++counters_.failed;
  release(active);
}

void Engine::finishCapacity(Request &active, const TokenAdmission &admission) {
  if (admission.allocationFailure != metal::AllocationFailure::None &&
      admission.allocationFailure != metal::AllocationFailure::Capacity) {
    finishFailure(active,
                  {"capacity_exhausted",
                   std::string("could not allocate KV target: ") +
                       metal::allocationFailureName(admission.allocationFailure) +
                       " (additional_pages=" +
                       std::to_string(admission.additionalPages) +
                       ", logical_pages_free=" +
                       std::to_string(admission.availablePages) + ")",
                   true});
    return;
  }
  if (active.finalized)
    return;
  scheduler_.fail(active.request.id);
  active.finalized = true;
  events_.capacityExhausted(active.request.id, admission.additionalPages,
                            admission.availablePages, 0);
  ++counters_.failed;
  release(active);
}

void Engine::release(Request &active) {
  active.resourceWait = {};
  active.backoffStartedMilliseconds.reset();
  if (active.suspended)
    countEvictedResumeState(active);
  cache_.unprotectState(std::exchange(active.resumeProtection, {}));
  countEvictedQueuedPrefix(active);
  cache_.unprotectState(std::exchange(active.queuedProtection, {}));
  cache_.unprotectState(std::exchange(active.checkpointProtection, {}));
  dropPreviousCheckpoint(active);
  // Empty after a Stop or Length finish moved it into conversationEnds_.
  releaseConversationEnd(active.conversationEnd);
  if (active.stateCell || active.suspended) {
    discardPendingStateBoundaries(active);
    model_.end(active.request.id);
    cache_.endRequest(active.request.id);
    active.stateCell.reset();
    active.suspended = false;
    signalResourceProgress();
  }
}

// ============================================================================
// Counts the resume state of a suspended lane that is about to resume or end
// when eviction took it during the suspension. O(1); a lane that found no
// state to protect at suspension counts nothing.
void Engine::countEvictedResumeState(const Request &active) noexcept {
  if (active.resumeProtection &&
      !cache_.isProtectedStateResident(active.resumeProtection))
    ++counters_.resourceResumeStateEvictions;
}

// ============================================================================
// Counts a queued request whose protected found state eviction or retirement
// took while it waited, as that protection ends: at a retried wait's swap, at
// admission, or when the request ends. O(1); counted at most once per request,
// so a retried wait that re-protects a shallower state and loses it too adds
// nothing further.
void Engine::countEvictedQueuedPrefix(Request &active) noexcept {
  if (active.queuedPrefixLossCounted || !active.queuedProtection ||
      cache_.isProtectedStateResident(active.queuedProtection))
    return;
  active.queuedPrefixLossCounted = true;
  ++counters_.resourceQueuedPrefixEvictions;
}

void Engine::sweepTerminal() {
  for (auto iterator = requests_.begin(); iterator != requests_.end();) {
    const uint64_t id = iterator->first;
    Request &active = iterator->second;
    const Phase phase = scheduler_.phase(id);
    if (phase == Phase::Failed && !active.finalized) {
      finishFailure(active, {"deadline_exceeded", "request deadline elapsed"});
    }
    if (!active.finalized) {
      ++iterator;
      continue;
    }
    scheduler_.remove(id);
    iterator = requests_.erase(iterator);
  }
}

Engine::Request &Engine::request(uint64_t id) {
  auto found = requests_.find(id);
  if (found == requests_.end())
    throw std::out_of_range("unknown request");
  return found->second;
}

} // namespace splash::engine
