// An agent session served under a real MemoryGovernor: back-to-back turns of
// one conversation on a host whose spare memory runs inside the warning
// margin, the pattern Seqcom drives through a live server.

#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestMetalMemory.hpp"
#include "engine/Engine.hpp"
#include "engine/MemoryGovernor.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;
using splash::test::Executor;
using splash::test::require;
using splash::test::State;

namespace {

constexpr uint32_t kBackingPages = 1024;
constexpr uint32_t kPagesPerExtent = 4;
constexpr uint64_t kPageBytes = 1ULL << 20;
constexpr uint64_t kExtentBytes = kPagesPerExtent * kPageBytes;
constexpr uint64_t kEngineLimitBytes = 40ULL << 30;
constexpr uint64_t kHostReserveBytes = 2ULL << 30;

constexpr uint32_t kTurns = 8;
constexpr uint32_t kFirstPromptTokens = 4000;
constexpr uint32_t kReplyTokens = 200;
constexpr uint32_t kToolResultTokens = 300;
constexpr uint32_t kUnrelatedPromptTokens = 1000;
constexpr uint32_t kUnrelatedReplyTokens = 20;
constexpr uint64_t kUnrelatedId = 100;
constexpr uint32_t kConversationToken = 51;
constexpr uint32_t kToolToken = 61;
constexpr uint32_t kUnrelatedToken = 7;

constexpr uint64_t extentsFor(uint64_t tokens) {
  const uint64_t pages = (tokens + KvCache::pageTokens - 1) / KvCache::pageTokens;
  return (pages + kPagesPerExtent - 1) / kPagesPerExtent;
}
// The KV history of a turn, through its reply and verify rows.
constexpr uint64_t historyTokens(uint32_t turn) {
  return kFirstPromptTokens + (turn - 1) * uint64_t{kReplyTokens + kToolResultTokens} +
         kReplyTokens + model::ExecutionLimits::targetVerifyRows;
}
// Extents the host can give the engine's KV before its spare memory falls
// inside the warning margin: the final turn's history less the unrelated
// request's. The history of the turn before the unrelated request fits
// beside that request's KV; the session outgrows the room before it ends,
// so growth no request in service needs is refused and the paced reclaim
// trims the cache.
constexpr uint64_t kSessionExtents =
    extentsFor(historyTokens(kTurns)) - extentsFor(kUnrelatedPromptTokens);
static_assert(extentsFor(historyTokens(kTurns / 2)) + extentsFor(kUnrelatedPromptTokens) <=
                  kSessionExtents,
              "the unrelated request must fit beside the turn before it");
static_assert(kSessionExtents < kBackingPages / kPagesPerExtent,
              "the backing must outgrow the host");
// The host's spare memory with no KV resident: the spare reserve, the
// warning margin and the session's extents.
constexpr uint64_t kHostBytes =
    kHostReserveBytes + kHostWarningMarginBytes + kSessionExtents * kExtentBytes;
// What macOS reports to each pressure pass. Under its warning growth pauses
// and every pass returns up to the warning margin's worth of cache, keeping
// only the newest resume point: the unrelated request's KV would take the
// session's idle pages and its point would displace the session's, so the
// host's own margin constrains the session instead.
constexpr MemoryPressure kSystemPressure = MemoryPressure::Normal;

constexpr double kTickMilliseconds = 25.0;
// The native loop's control pass samples the host every 500 ms.
constexpr double kPressureTickMilliseconds = 500.0;
constexpr double kRequestDeadlineMilliseconds = 600'000.0;
constexpr uint32_t kMaximumTicksPerRequest = 20000;

// The engine events the session reads: each start's matched tokens, each
// request's output, and how many requests completed or failed.
class Events final : public EngineEventSink {
public:
  void batchCompleted(WorkKind, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, double,
                      double) override {}
  void started(uint64_t requestId, uint32_t matched, uint32_t) override {
    starts.emplace(requestId, matched);
  }
  void tokens(uint64_t id, std::span<const uint32_t> values) override {
    auto &transcript = outputs[id];
    transcript.insert(transcript.end(), values.begin(), values.end());
  }
  void maskRequested(uint64_t, std::span<const uint32_t>) override {}
  void completed(uint64_t, EngineFinishReason, bool, uint32_t, uint32_t, uint32_t,
                 std::span<const float>, const SpeculativeCounts &,
                 const EngineRequestTiming &) override {
    ++completedCount;
  }
  void failed(uint64_t, LaneOutcome, std::string, uint32_t) override { ++failedCount; }

  // The matched prompt tokens of each request's first start.
  std::unordered_map<uint64_t, uint32_t> starts;
  std::unordered_map<uint64_t, std::vector<uint32_t>> outputs;
  uint32_t completedCount = 0;
  uint32_t failedCount = 0;
};

// The page-aligned prompt end a finished request's state resumes from.
uint32_t promptEnd(const std::vector<uint32_t> &prompt) {
  return static_cast<uint32_t>((prompt.size() - 1) / KvCache::pageTokens * KvCache::pageTokens);
}

// The engine, its doubles and a real governor wired as the bootstrap and the
// native loop wire them: KV growth and lane begins are admitted through the
// governor, growthPaused and serving read it, and the pressure pass
// (MemoryControl::run's steps) runs between engine commands. The host's
// spare memory is the fixed host less the KV allocated at each sample.
struct PressureSession final {
  test::TestKvStorage storage{kBackingPages, kPageBytes, kPagesPerExtent};
  KvPool pool{storage, 0};
  engine::Cache resources{pool, nullptr, nullptr};
  Executor executor;
  Events events;
  metal::MetalBackend backend{"unused"};
  MemoryGovernor governor{backend,           kEngineLimitBytes, kHostReserveBytes,
                          kHostReserveBytes, [this] { return hostMemory(); }, 0};
  MemoryPressurePolicy pressurePolicy;
  engine::Engine engine;
  double now = 1.0;
  double nextPressurePass = 0.0;
  uint32_t reclaimPasses = 0;

  PressureSession() : engine(configured(), resources, executor, events) {
    storage.commandInFlight = [this] { return engine.commandInFlight(); };
    storage.allocationFailure = metal::AllocationFailure::HostPressure;
    storage.growthAllowed = [this](uint32_t) { return admit(kExtentBytes); };
    // A lane's cell is admitted through the governor as the production
    // model admits a new one; the fake's state bytes are too small to move
    // the host sample.
    executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
    executor.beginGrowthBlocked = [this] { return !admit(State().bytes()); };
    executor.decodeFinishes = false;
  }

  EngineConfig configured() {
    EngineConfig config;
    config.growthPaused = [this] { return !governor.snapshot().hostGrowthAllowed; };
    config.serving = [this](bool serving) { return governor.setServing(serving); };
    return test::engineConfig(std::move(config));
  }

  std::optional<HostMemory> hostMemory() const {
    const uint64_t spare = kHostBytes - storage.allocatedBytes();
    return HostMemory{spare, spare};
  }

  // One transactional growth, as the production allocators take it.
  bool admit(uint64_t bytes) {
    return static_cast<bool>(governor.allocationAdmission()(bytes, [] {}));
  }

  // MemoryControl::run's steps: true while a paced KV release is in flight.
  bool pressurePass() {
    governor.setPressure(kSystemPressure);
    const ResourceWaitSnapshot wait = engine.resourceWaitSnapshot(now);
    const bool requestWaiting = wait.memory || wait.suspended || wait.heldBehindRefusal;
    governor.setRequestWaiting(requestWaiting);
    const MemoryGovernorSnapshot memory = governor.evaluate();
    const std::optional<MemoryReclaimDirective> directive =
        pressurePolicy.update(memory, now, requestWaiting);
    if (!directive)
      return false;
    if (directive->targetBytes)
      ++reclaimPasses;
    const MemoryReclaimResult reclaim = engine.reclaimMemory(*directive);
    pressurePolicy.reclaimed(*directive, reclaim);
    governor.reclaimed(reclaim.outcome);
    return reclaim.outcome == ReclaimOutcome::Pending;
  }

  void serve(uint64_t id, const std::vector<uint32_t> &prompt, uint32_t replyTokens) {
    EngineRequest turn;
    turn.id = id;
    turn.prompt = prompt;
    turn.maxNewTokens = replyTokens;
    turn.deadlineMilliseconds = now + kRequestDeadlineMilliseconds;
    engine.submit(std::move(turn));
    for (uint32_t tick = 0; tick < kMaximumTicksPerRequest && !engine.idle();
         ++tick, now += kTickMilliseconds) {
      static_cast<void>(engine.tick(now));
      if (engine.commandInFlight() || now < nextPressurePass)
        continue;
      nextPressurePass = pressurePass() ? now : now + kPressureTickMilliseconds;
    }
    require(engine.idle(), "a session request did not finish");
  }

  // The tokens a request with this id matched when it started.
  uint32_t matchedTokens(uint64_t id) const {
    const auto found = events.starts.find(id);
    require(found != events.starts.end(), "a session request never started");
    return found->second;
  }
};

std::vector<uint32_t> followUpTurn(const std::vector<uint32_t> &prompt,
                                   const std::vector<uint32_t> &reply) {
  std::vector<uint32_t> turn;
  turn.reserve(prompt.size() + reply.size() + kToolResultTokens);
  turn.insert(turn.end(), prompt.begin(), prompt.end());
  turn.insert(turn.end(), reply.begin(), reply.end());
  turn.insert(turn.end(), kToolResultTokens, kToolToken);
  return turn;
}

// Eight back-to-back turns of one conversation, each the previous prompt plus
// its reply and a tool result, with one unrelated request between turns 4 and
// 5: every follow-up resumes at least from the previous prompt end, no lane
// is suspended for resources, and nothing is replayed.
void testAgentSessionResumesEveryTurnUnderAHostConstrainedGovernor() {
  test::metalStatistics() = {};
  PressureSession session;
  std::vector<uint32_t> prompt(kFirstPromptTokens, kConversationToken);
  std::vector<uint32_t> previousPromptEnds;

  for (uint64_t id = 1; id <= kTurns; ++id) {
    session.serve(id, prompt, kReplyTokens);
    previousPromptEnds.push_back(promptEnd(prompt));
    if (id == kTurns / 2) {
      session.serve(kUnrelatedId, std::vector<uint32_t>(kUnrelatedPromptTokens, kUnrelatedToken),
                    kUnrelatedReplyTokens);
    }
    prompt = followUpTurn(prompt, session.events.outputs.at(id));
  }

  for (uint64_t id = 2; id <= kTurns; ++id) {
    require(session.matchedTokens(id) >= previousPromptEnds.at(id - 2),
            "turn " + std::to_string(id) + " matched " +
                std::to_string(session.matchedTokens(id)) +
                " tokens, short of the previous prompt end " +
                std::to_string(previousPromptEnds.at(id - 2)));
  }
  const EngineSnapshot engine = session.engine.snapshot();
  require(session.events.completedCount == kTurns + 1 && session.events.failedCount == 0,
          "the session did not complete every request");
  require(engine.resourceSuspensions == 0 && session.executor.suspensions == 0,
          "a lane was suspended for resources");
  require(engine.resourceReplayTokens == 0,
          "the session replayed tokens it had already processed");
  require(session.governor.snapshot().deniedHostReservations != 0 && session.reclaimPasses != 0,
          "the host never constrained the session");
}

} // namespace

int main() {
  try {
    testAgentSessionResumesEveryTurnUnderAHostConstrainedGovernor();
    std::cout << "pressure session tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "pressure session tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
