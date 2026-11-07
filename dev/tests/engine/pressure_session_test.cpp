// An agent session served under a real MemoryGovernor: back-to-back turns of
// one conversation on a host whose spare memory runs inside the margin, the
// pattern Seqcom drives through a live server.

#include "EngineHarness.hpp"
#include "FakeMemory.hpp"
#include "TestChecks.hpp"
#include "TestMetalMemory.hpp"
#include "engine/Engine.hpp"
#include "engine/MemoryGovernor.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::engine;
using splash::test::require;

namespace {

constexpr uint32_t kBackingPages = 1024;
constexpr uint32_t kPagesPerExtent = 4;
constexpr uint64_t kPageBytes = 1ULL << 20;
constexpr uint64_t kExtentBytes = kPagesPerExtent * kPageBytes;
constexpr uint64_t kOtherBytes = 8ULL << 30;

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
// inside the margin: the final turn's history less the unrelated request's.
// The history of the turn before the unrelated request fits beside that
// request's KV; the session outgrows the room before it ends, so the host
// refuses growth and the cache gives pages back to the session's requests.
constexpr uint64_t kSessionExtents =
    extentsFor(historyTokens(kTurns)) - extentsFor(kUnrelatedPromptTokens);
static_assert(extentsFor(historyTokens(kTurns / 2)) + extentsFor(kUnrelatedPromptTokens) <=
                  kSessionExtents,
              "the unrelated request must fit beside the turn before it");
static_assert(kSessionExtents < kBackingPages / kPagesPerExtent,
              "the backing must outgrow the host");
// The host's memory: what other applications hold, the spare reserve, the
// margin and the session's extents. macOS reports normal pressure
// throughout, so the host's own margin constrains the session.
constexpr uint64_t kHostBytes =
    kOtherBytes + test::GovernorLimits{}.hostSpareReserveBytes + kHostMarginBytes + kSessionExtents * kExtentBytes;

// The native loop's control pass samples the host every 500 ms.
constexpr double kPressurePassMilliseconds = 500.0;
constexpr double kRequestDeadlineMilliseconds = 600'000.0;
constexpr uint32_t kMaximumStepsPerRequest = 100'000;

// The page-aligned prompt end a finished request's state resumes from.
uint32_t promptEnd(const std::vector<uint32_t> &prompt) {
  return static_cast<uint32_t>((prompt.size() - 1) / KvCache::pageTokens * KvCache::pageTokens);
}

// The engine on the fake model, its KV and states allocated through
// FakeMemory under a real MemoryGovernor, as the bootstrap wires them, and
// the governor's control pass between engine commands as the native loop
// runs it (MemoryControl::run): every 500 ms, and again at the next
// command-free point while it waits for what lands.
struct PressureSession final {
  test::EngineHarness harness{{.pages = kBackingPages,
                               .pageBytes = kPageBytes,
                               .extentPages = kPagesPerExtent,
                               .memory = test::FakeMemory::Config{.hostBytes = kHostBytes,
                                                                  .otherBytes = kOtherBytes}}};
  double nextPressurePass = 0.0;

  PressureSession() { harness.model.decodeFinishes = false; }

  void serve(uint64_t id, const std::vector<uint32_t> &prompt, uint32_t replyTokens) {
    EngineRequest turn;
    turn.id = id;
    turn.prompt = prompt;
    turn.maxNewTokens = replyTokens;
    turn.deadlineMilliseconds = harness.now() + kRequestDeadlineMilliseconds;
    harness.engine.submit(std::move(turn));
    harness.runUntil(
        [&] {
          if (!harness.engine.commandInFlight() && harness.now() >= nextPressurePass)
            nextPressurePass = harness.governor->controlPass(harness.now())
                                   ? harness.now()
                                   : harness.now() + kPressurePassMilliseconds;
          return harness.engine.idle();
        },
        "a session request did not finish", kMaximumStepsPerRequest);
  }

  // The tokens a request with this id matched when it started.
  uint32_t matchedTokens(uint64_t id) const {
    const std::vector<uint64_t> &ids = harness.events.startIds;
    for (size_t index = 0; index < ids.size(); ++index)
      if (ids[index] == id)
        return harness.events.starts[index];
    throw std::logic_error("a session request never started");
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
    prompt = followUpTurn(prompt, session.harness.events.outputs.at(id));
  }

  for (uint64_t id = 2; id <= kTurns; ++id) {
    require(session.matchedTokens(id) >= previousPromptEnds.at(id - 2),
            "turn " + std::to_string(id) + " matched " +
                std::to_string(session.matchedTokens(id)) +
                " tokens, short of the previous prompt end " +
                std::to_string(previousPromptEnds.at(id - 2)));
  }
  const test::EngineHarness &harness = session.harness;
  const EngineSnapshot engine = harness.engine.snapshot();
  require(harness.events.completedCount == kTurns + 1 && harness.events.failedCount == 0,
          "the session did not complete every request");
  require(engine.resourceSuspensions == 0 && harness.model.suspensions == 0,
          "a lane was suspended for resources");
  require(engine.resourceReplayTokens == 0,
          "the session replayed tokens it had already processed");
  require(harness.governor->snapshot().deniedHostReservations != 0 && engine.resources.kvEvictions != 0,
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
