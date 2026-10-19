#pragma once

#include "ops/Vision.hpp"
#include "engine/Cache.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/Scheduler.hpp"
#include "engine/Types.hpp"
#include "ops/PagedKv.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace splash::engine {

struct EngineConfig final {
  uint32_t maxContext = kv::kMaximumLogicalTokens;
  uint32_t vocabularySize = std::numeric_limits<uint32_t>::max();
  // Two draft windows balance recovery granularity and capture work. Prefill
  // and decode both checkpoint at its multiples; a decode step's retained
  // rows are capped so its commit lands on one. Zero disables progress
  // checkpoints and the cap without changing reusable end states.
  uint32_t prefillCheckpointTokens =
      2 * model::ExecutionLimits::draftContextTokens;
  // Patches per image the model's vision scratch covers.
  uint32_t maxImagePatches = ops::kMaximumImagePatches;
  // Rows below which a contended prefill command never shrinks; a power of
  // two in [64, model::ExecutionLimits::prefillTokenBudget].
  uint32_t contendedPrefillRows = Scheduler::kDefaultContendedPrefillRows;
  double resourceWaitTimeoutMilliseconds = 30000.0;
  // Host growth admission, supplied by the runtime governor. Queried only on
  // failed allocation, never on the successful decode path.
  std::function<bool()> growthPaused;
};

struct ResourceWaitSnapshot final {
  uint32_t memory = 0;
  uint32_t concurrency = 0;
  uint32_t suspended = 0;
  double oldestWaitMilliseconds = 0.0;
  bool draining = false;
};

// Why a checkpoint publication failed: its prefill boundary passed inside a
// command (MissedBoundary), the previous checkpoint could not be retired before
// allocating (RetireBlocked), no state could be captured even after recycling
// one (SnapshotDenied), a decode commit passed its boundary without landing on
// it (DecodeOvershoot), or the publication threw (Error). Indexes
// EngineSnapshot::checkpointPublicationFailureCauses.
enum class PublicationFailure : uint8_t {
  MissedBoundary,
  RetireBlocked,
  SnapshotDenied,
  DecodeOvershoot,
  Error,
};
inline constexpr std::size_t kPublicationFailureCauses =
    static_cast<std::size_t>(PublicationFailure::Error) + 1;

struct EngineSnapshot final {
  SchedulerSnapshot scheduler;
  CacheSnapshot resources;
  uint32_t maximumContextTokens = 0;
  uint64_t submitted = 0;
  uint64_t completed = 0;
  uint64_t cancelled = 0;
  uint64_t failed = 0;
  uint64_t cacheHits = 0;
  uint64_t coldMisses = 0;
  uint64_t reusedTokens = 0;
  uint64_t replayStatePublications = 0;
  uint64_t deduplicatedStatePublications = 0;
  // Publications completed after reclaiming a cached state.
  uint64_t recycledStatePublications = 0;
  uint64_t replayStatePublicationFailures = 0;
  uint64_t junctionMaterializations = 0;
  uint64_t junctionMaterializationFailures = 0;
  // Prefill and decode progress checkpoints; a decode commit that passed its
  // boundary without landing on it counts as a failure.
  uint64_t checkpointPublications = 0;
  uint64_t checkpointPublicationFailures = 0;
  // checkpointPublicationFailures split by PublicationFailure; sums to it.
  std::array<uint64_t, kPublicationFailureCauses>
      checkpointPublicationFailureCauses{};
  uint64_t resourceSuspensions = 0;
  uint64_t resourceResumptions = 0;
  // Prefill rows after preemption that recompute tokens the request had
  // already processed before a suspension; an unfinished prompt suffix is
  // ordinary prefill and never counts.
  uint64_t resourceReplayTokens = 0;
  // The resourceSuspensions taken in the Decode or WaitingMask phase.
  uint64_t resourceDecodeSuspensions = 0;
  // The resourceReplayTokens of requests whose replay a decode-phase
  // suspension started, including rows replayed again after a further
  // suspension during that replay.
  uint64_t resourceDecodeReplayTokens = 0;
};

// KV blocks define prefix identity; composite recurrent state is attached
// at sparse progress points, replay boundaries, and shared KV junctions.
class Engine final {
public:
  Engine(EngineConfig config, Cache &cache, model::Model &model,
         EngineEventSink &events);

  void submit(EngineRequest request);
  void observePrefill(uint32_t rows, double wallMilliseconds) {
    scheduler_.observePrefill(rows, wallMilliseconds);
  }
  void cancel(uint64_t requestId);
  void failRequest(uint64_t requestId, std::string code, std::string message);
  void provideMask(uint64_t requestId, std::span<const uint32_t> words);
  void setCompletionNotifier(std::function<void()> notifier);

  [[nodiscard]] bool tick(double nowMilliseconds);
  [[nodiscard]] bool idle() const noexcept;
  [[nodiscard]] bool commandInFlight() const noexcept {
    return pending_.has_value();
  }
  [[nodiscard]] std::optional<double> nextWakeupMilliseconds() const;
  [[nodiscard]] EngineSnapshot snapshot() const;
  [[nodiscard]] ResourceWaitSnapshot resourceWaitSnapshot(double nowMilliseconds) const;

  // Runs only at a command-completion safe point. Reclaim order follows
  // ownership and preserves reusable prefixes for as long as possible: idle
  // model state, unused KV backing, disposable checkpoints, then ordinary
  // state/KV in LRU order. A resident lane's latest checkpoint is protected:
  // Warning spares it and Critical takes it.
  // Live command buffers are never eviction candidates. Physical KV release
  // is paced one extent at a time; reclaimDeferred() reports that the pass
  // stopped behind an in-flight release and should run again shortly.
  // A warning pass keeps one empty KV extent mapped while a running lane's
  // next chunk needs it or a request waits for admission (queued, waiting on
  // resources or suspended); a critical pass releases every empty extent.
  [[nodiscard]] uint64_t reclaimMemory(const MemoryReclaimDirective &directive);
  [[nodiscard]] bool reclaimDeferred() const noexcept {
    return cache_.releaseDeferred(reclaimKeptKvRunway_);
  }

private:
  struct Failure final {
    std::string code;
    std::string message;
    bool retryable = false;
  };

  struct ResourceWait final {
    StateFailure reason = StateFailure::None;
    std::optional<double> startedMilliseconds;
    double deadlineMilliseconds = 0.0;
    double retryMilliseconds = 0.0;
    uint64_t epoch = 0;
  };

  struct Request final {
    struct StateBoundary final {
      enum class Purpose : uint8_t { Checkpoint, Replay, Junction };
      uint32_t tokens = 0;
      Purpose purpose = Purpose::Replay;
    };

    EngineRequest request;
    std::optional<uint32_t> stateCell;
    bool suspended = false;
    uint32_t promptTokens = 0;
    uint32_t reportedPromptTokens = 0;
    uint32_t replayTokens = 0;
    // Deepest position processed before any suspension: replay rows below it
    // count as reprocessing, rows past it as ordinary prefill. Never lowered.
    uint32_t replayCountEnd = 0;
    // A failed dispatch must fit before replay can consume any model work.
    uint64_t resumeKvTargetTokens = 0;
    ResourceWait resourceWait;
    std::vector<uint32_t> exactTokens;
    std::optional<CacheProbe> admissionProbe;
    // First denial of a sole resident decode lane backing off under host
    // pressure instead of suspending; its retry time is resourceWait's, whose
    // deadline stays 0 while the lane is resident. Ends on a grant, at
    // suspension and at release.
    std::optional<double> backoffStartedMilliseconds;
    std::vector<StateBoundary> stateBoundaries;
    size_t stateBoundaryCursor = 0;
    // Written only through setLatestCheckpoint, which moves
    // checkpointProtection with it.
    StateCheckpoint latestCheckpoint;
    // A resident lane's latestCheckpoint, spared by Warning reclaim and taken
    // by growth reclaim and publication recycling only after every
    // unprotected state. Dropped when the lane suspends or ends.
    StateProtection checkpointProtection;
    // A decoding lane's checkpoint replaced at its last decode boundary B,
    // kept until a later commit passes B: a suspension at exactly B cannot
    // restore B's own state, its lookup leaving one token to recompute.
    // Protected like latestCheckpoint; ends on the first commit past B, on
    // admission, suspension and release, and is retired when the request
    // finishes.
    StateCheckpoint previousCheckpoint;
    StateProtection previousCheckpointProtection;
    // The deepest prefill checkpoint below the prompt end, moved out of
    // latestCheckpoint at the prompt-end publication, before its capture, so
    // it stays cached beside the prompt-end state: a later prompt diverging
    // between the two resumes from it. Unprotected, so reclaim takes it
    // before any ordinary state. Written at most once per request; decode
    // checkpoint rolling and finish never touch it.
    StateCheckpoint keptCheckpoint;
    // The kept checkpoint recorded on the state this request restored at its
    // first admission: retired once this request keeps its own, otherwise
    // carried forward onto its prompt-end state, so a conversation holds at
    // most one kept checkpoint.
    StateCheckpoint inheritedCheckpoint;
    // The prompt-end publication was attempted; a resumed replay passing the
    // prompt end again keeps nothing further.
    bool promptEndPublished = false;
    // The state a suspended request resumes from, spared by Warning reclaim
    // until the resume succeeds or the request ends.
    StateProtection resumeProtection;
    // The cached prefix an unstarted request found before waiting for
    // resources, spared by Warning reclaim until admission or the request
    // ends. Never held with resumeProtection.
    StateProtection queuedProtection;
    // The scheduler owns the terminal phase; this flag records that the
    // corresponding event was emitted and model/resource ownership ended.
    bool finalized = false;
    std::optional<Failure> failure;
    bool replaying = false;
    // Set by a Decode or WaitingMask suspension and kept until the replay of
    // the committed history ends, even across a further suspension during
    // that replay, so every replayed row counts as decode replay.
    bool hasDecodeSuspension = false;
    // Captured once the final prompt chunk completes; emitted with Done.
    std::vector<float> scoreLogits;
  };


  struct Pending final {
    BatchPlan plan;
    std::unique_ptr<ModelBatchTicket> ticket;
  };
  double nextHealthCheckMilliseconds_ = 0.0;

  [[nodiscard]] Request &request(uint64_t requestId);
  [[nodiscard]] bool admitQueued(double nowMilliseconds);
  [[nodiscard]] bool admit(Request &request, double nowMilliseconds);
  [[nodiscard]] static uint32_t sharedPrefillBoundary(const Request &left,
                                                      const Request &right);
  [[nodiscard]] bool pendingSharedPrefill(const Request &request,
                                          uint32_t resumeBoundary) const;
  [[nodiscard]] static uint32_t peerJunction(const Request &request,
                                             uint64_t peerId,
                                             const Request &peer,
                                             uint32_t after);
  [[nodiscard]] uint32_t firstPlannedBoundary(const Request &request,
                                              uint32_t stateBoundary,
                                              uint32_t junctionBoundary) const;
  [[nodiscard]] bool rebuildsDraftWindow(const Request &request,
                                         uint32_t stateBoundary,
                                         uint32_t junctionBoundary) const;
  [[nodiscard]] bool restorable(const Request &request,
                                const CacheLookup &lookup) const;
  [[nodiscard]] DraftContextPlan
  configureDraftStatePlan(Request &request, uint32_t stateBoundary,
                          uint32_t junctionBoundary, bool targetOnlyRestore);
  [[nodiscard]] bool addSharedPrefillBoundaries(Request &request, uint32_t after);
  [[nodiscard]] DraftContextPlan
  pendingDraftStatePlan(const Request &request, uint32_t stateBoundary,
                        bool targetOnlyRestore = false) const;
  void armNextStateBoundary(Request &request);
  void discardPendingStateBoundaries(Request &request) noexcept;
  [[nodiscard]] bool peerHoldsCheckpoint(const Request &request,
                                         StateCheckpoint checkpoint) const;
  [[nodiscard]] bool retireCheckpoint(Request &request);
  void retirePreviousCheckpoint(Request &request);
  void dropPreviousCheckpoint(Request &request) noexcept;
  void setLatestCheckpoint(Request &request, StateCheckpoint checkpoint);
  void keepDeepestCheckpoint(Request &request, uint64_t promptEndBlock);
  void recordKeptCheckpoint(Request &request, uint64_t promptEndBlock);
  [[nodiscard]] std::shared_ptr<const CompositeState>
  captureState(uint64_t requestId, bool checkpoint);
  void countCheckpointFailure(PublicationFailure cause) noexcept;
  void publishReachedStateBoundaries(Request &request,
                                     uint32_t promptProcessed);
  [[nodiscard]] uint32_t
  decodeRetainedTokenLimit(uint64_t storedTokens) const noexcept;
  void publishDecodeCheckpoint(Request &request, uint32_t previousTokens,
                               uint32_t storedTokens);
  // What prepare left for tick: a batch to submit, a lane suspended or
  // failed, or only back-off skips, which leave the loop to sleep until
  // nextWakeupMilliseconds.
  enum class PrepareOutcome : uint8_t { Submit, Yielded, BackedOff };
  [[nodiscard]] PrepareOutcome prepare(BatchPlan &plan,
                                       std::vector<ModelBatchItem> &items,
                                       double nowMilliseconds);
  [[nodiscard]] bool backOffSoleDecodeLane(Request &request,
                                           const TokenAdmission &admission,
                                           WorkKind kind,
                                           double nowMilliseconds);
  [[nodiscard]] bool reclaimForGrowth(
      CacheReclaimMode mode = CacheReclaimMode::ReleaseBacking);
  [[nodiscard]] bool reclaimIdleState() noexcept;
  [[nodiscard]] bool reuseIdleBackingWhilePaused(const TokenAdmission &admission);
  [[nodiscard]] bool growthPaused() const;
  [[nodiscard]] uint32_t completedTokens(const Request &request) const;
  [[nodiscard]] bool residentLaneNeedsKvRunway() const;
  [[nodiscard]] bool waitingRequestNeedsKvRunway() const;
  struct KvAdmission {
    TokenAdmission allocation;
    bool retryableBudget = false;
  };
  [[nodiscard]] KvAdmission admitKv(Request &request, uint64_t workEnd);
  [[nodiscard]] bool budgetMayRecover(metal::AllocationFailure failure,
                                      uint64_t generation, bool reclaimed) const;
  void suspendModel(Request &request, uint32_t position);
  void suspendForGrowth(Request &request, uint64_t workEnd,
                        double nowMilliseconds);
  [[nodiscard]] bool resourceRetryReady(const Request &request,
                                        double nowMilliseconds) const noexcept;
  void deferResourceRetry(Request &request, double nowMilliseconds,
                          StateFailure reason = StateFailure::MemoryPressure) noexcept;
  void signalResourceProgress() noexcept;
  void apply(const BatchPlan &plan, std::span<const ModelStepResult> results,
             double wallMilliseconds, bool representativePrefillTiming);
  void finish(Request &request, EngineFinishReason reason,
              std::span<const float> optionLogits);
  void finishFailure(Request &request, Failure failure);
  void finishCapacity(Request &request, const TokenAdmission &admission);
  void release(Request &request);
  void sweepTerminal();

  EngineConfig config_;
  Cache &cache_;
  model::Model &model_;
  EngineEventSink &events_;
  Scheduler scheduler_;
  std::unordered_map<uint64_t, Request> requests_;
  // Pressure preempted work and a resident lane still holds its state cell.
  [[nodiscard]] bool drainingForRecovery() const;
  std::function<void()> completionNotifier_;
  std::optional<Pending> pending_;
  uint64_t resourceEpoch_ = 1;
  bool recoveringResources_ = false;
  // The last pressure pass kept a KV runway extent mapped; that extent is not
  // deferred reclaim work.
  bool reclaimKeptKvRunway_ = false;
  EngineSnapshot counters_;
};

} // namespace splash::engine
