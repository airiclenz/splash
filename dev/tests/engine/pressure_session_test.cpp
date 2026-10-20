// An agent session served under a real MemoryGovernor: back-to-back turns of
// one conversation on a host held near the serving floor while the OS reports
// memory warn, the pattern Seqcom drives through a live server.

#include "TestEngineDoubles.hpp"
#include "TestFakeHost.hpp"
#include "engine/Engine.hpp"
#include "engine/MemoryGovernor.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;
using namespace splash::engine::test;

namespace {

constexpr uint32_t kBackingPages = 512;
constexpr uint32_t kPagesPerExtent = 4;
constexpr uint64_t kPageBytes = 4096;
constexpr uint64_t kExtentBytes = kPagesPerExtent * kPageBytes;
constexpr HostCushion kCushion{64 * 1024, 4 * kExtentBytes,
                               8 * kExtentBytes};
constexpr uint64_t kEngineLimitBytes = 1ULL << 30;

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
  const uint64_t pages =
      (tokens + KvCache::pageTokens - 1) / KvCache::pageTokens;
  return (pages + kPagesPerExtent - 1) / kPagesPerExtent;
}
// The KV history of the final turn, through its reply and verify rows.
constexpr uint64_t kFinalHistoryTokens =
    kFirstPromptTokens +
    (kTurns - 1) * uint64_t{kReplyTokens + kToolResultTokens} + kReplyTokens +
    model::ExecutionLimits::targetVerifyRows;
// Extents the host can give the engine's KV before growth would leave less
// than the warning margin: the final turn's history with a runway extent and
// one spare fits, that history beside the unrelated request's does not.
constexpr uint64_t kSessionExtents = extentsFor(kFinalHistoryTokens) + 2;
static_assert(extentsFor(kUnrelatedPromptTokens) > 2,
              "the unrelated request must not fit beside the final turn");
static_assert(kSessionExtents < kBackingPages / kPagesPerExtent,
              "the backing must outgrow the host");
// Host memory available with no KV resident: the serving floor plus the
// session's extents.
constexpr uint64_t kHostBytes =
    kCushion.admissionFloorBytes() + kSessionExtents * kExtentBytes;

constexpr double kTickMilliseconds = 25.0;
// main.mm's control handler samples the host every 500 ms.
constexpr double kPressureTickMilliseconds = 500.0;
constexpr uint32_t kMaximumTicksPerRequest = 20000;

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

// The page-aligned prompt end a finished request's state resumes from.
uint32_t promptEnd(const std::vector<uint32_t> &prompt) {
  return static_cast<uint32_t>((prompt.size() - 1) / KvCache::pageTokens *
                               KvCache::pageTokens);
}

// The engine, its doubles and a real governor wired as Bootstrap.mm and
// main.mm wire them: growth admission and growthPaused read the governor, and
// the pressure tick (MemoryPressurePolicy -> Engine::reclaimMemory) runs
// between engine ticks with OS warn held. Host memory available is the fixed
// host less the KV backing resident at each sample.
struct PressureSession final {
  Backing backing{kBackingPages};
  KvPool pool{backing};
  engine::Cache resources{pool, CacheNamespace{}};
  Executor executor;
  Events events;
  FakeHost host{kHostBytes};
  MemoryGovernor governor{[](bool) { return metal::MetalMemoryStats{}; },
                          kEngineLimitBytes, kCushion, host.sampler()};
  MemoryPressurePolicy pressurePolicy{kExtentBytes};
  engine::Engine engine;
  double now = 1.0;
  double nextPressureTick = 0.0;
  uint32_t reclaimPasses = 0;

  PressureSession() : engine(configured(), resources, executor, events) {
    backing.allocationFailure = metal::AllocationFailure::HostPressure;
    backing.growthAllowed = [this] { return admit(kExtentBytes); };
    executor.decodeRowsPerStep = 7;
    // A lane's cell is admitted through the governor as the production
    // model admits a new one; the fake's state bytes are too small to move
    // the host sample. Captures stay ungated: the production model serves a
    // capture from the slot a recycled state frees, and the fake keeps no
    // slot pool to model that reuse.
    executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
    executor.beginGrowthBlocked = [this] {
      return !admit(State{}.bytes());
    };
  }

  EngineConfig configured() {
    EngineConfig config;
    config.growthPaused = [this] {
      return !governor.snapshot().hostGrowthAllowed;
    };
    return config;
  }

  uint64_t hostAvailableBytes() const {
    uint64_t resident = 0;
    for (uint32_t page = 0; page < backing.pageCount(); ++page)
      resident += backing.isResident(page) ? kPageBytes : 0;
    return kHostBytes - resident;
  }

  // One transactional growth, as the production allocators take it.
  bool admit(uint64_t bytes) {
    host.availableBytes = hostAvailableBytes();
    auto reservation = governor.tryReserve(bytes);
    if (!reservation)
      return false;
    reservation->commit();
    return true;
  }

  // main.mm's control handler: true while a paced KV release is in flight.
  bool pressureTick() {
    governor.setPressure(MemoryPressure::Warning);
    host.availableBytes = hostAvailableBytes();
    governor.observe();
    const MemoryGovernorSnapshot memory = governor.snapshot();
    const ResourceWaitSnapshot wait = engine.resourceWaitSnapshot(now);
    const MemoryReclaimDirective directive =
        pressurePolicy.update(memory, now, wait.memory || wait.suspended);
    if (!directive.reclaimEmptyKvExtents)
      return false;
    if (directive.targetBytes)
      ++reclaimPasses;
    static_cast<void>(engine.reclaimMemory(directive));
    return engine.reclaimDeferred();
  }

  void serve(uint64_t id, const std::vector<uint32_t> &prompt,
             uint32_t replyTokens) {
    EngineRequest turn;
    turn.id = id;
    turn.prompt = prompt;
    turn.maxNewTokens = replyTokens;
    turn.deadlineMilliseconds = now + 600'000.0;
    engine.submit(std::move(turn));
    for (uint32_t tick = 0; tick < kMaximumTicksPerRequest && !engine.idle();
         ++tick, now += kTickMilliseconds) {
      static_cast<void>(engine.tick(now));
      if (engine.commandInFlight() || now < nextPressureTick)
        continue;
      nextPressureTick =
          pressureTick() ? now : now + kPressureTickMilliseconds;
    }
    require(engine.idle(), "a session request did not finish");
  }

  // The tokens a request with this id matched when it started.
  uint32_t matchedTokens(uint64_t id) const {
    const auto found =
        std::find(events.startIds.begin(), events.startIds.end(), id);
    require(found != events.startIds.end(), "a session request never started");
    return events.starts.at(found - events.startIds.begin()).second;
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
// 5: every follow-up resumes at least from the previous prompt end, no sole
// prefill lane is suspended for resources, and nothing is replayed.
void testAgentSessionResumesEveryTurnUnderAHostConstrainedGovernor() {
  PressureSession session;
  std::vector<uint32_t> prompt(kFirstPromptTokens, kConversationToken);
  std::vector<uint32_t> previousPromptEnds;

  for (uint64_t id = 1; id <= kTurns; ++id) {
    session.serve(id, prompt, kReplyTokens);
    previousPromptEnds.push_back(promptEnd(prompt));
    if (id == kTurns / 2) {
      session.serve(kUnrelatedId,
                    std::vector<uint32_t>(kUnrelatedPromptTokens,
                                          kUnrelatedToken),
                    kUnrelatedReplyTokens);
    }
    prompt = followUpTurn(prompt, session.events.outputs.at(id));
  }

  for (uint64_t id = 2; id <= kTurns; ++id) {
    require(session.matchedTokens(id) >= previousPromptEnds.at(id - 2),
            "a follow-up turn did not reuse the previous prompt end");
  }
  const EngineSnapshot engine = session.engine.snapshot();
  require(session.events.completedCount == kTurns + 1 &&
              session.events.failedCount == 0,
          "the session did not complete every request");
  require(engine.resourceSuspensions == 0 &&
              session.executor.suspensions == 0,
          "a sole prefill lane was suspended for resources");
  require(engine.resourceReplayTokens == 0,
          "the session replayed tokens it had already processed");
  require(session.governor.snapshot().deniedHostReservations != 0 &&
              session.reclaimPasses != 0,
          "the host cushion never constrained the session");
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
