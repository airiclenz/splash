#include "engine/Status.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using namespace splash;
using namespace splash::engine;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

bool close(double left, double right) { return std::abs(left - right) < 1e-9; }

void testLatencyWindowAndThroughput() {
  RuntimeMetrics metrics(3);
  for (uint64_t id = 1; id <= 4; ++id) {
    double submitted = double(id) * 100.0;
    const double first = submitted + double(id) * 10.0;
    metrics.tokens(submitted, std::nullopt, 1, first);
    metrics.tokens(submitted, first, 2, first + 8.0);
  }

  metrics.batchCompleted(WorkKind::Prefill, 1, 1000, 0, 0, 0, 100.0);
  metrics.batchCompleted(WorkKind::Decode, 4, 0, 8, 28, 4, 40.0);
  metrics.capacityFailed();
  metrics.metalFailed();

  RuntimeMetricsSnapshot snapshot = metrics.snapshot();
  require(close(snapshot.ttftP50Milliseconds, 30.0) &&
              close(snapshot.ttftP95Milliseconds, 40.0) &&
              snapshot.ttftSamples == 3,
          "bounded TTFT percentile window is incorrect");
  require(close(snapshot.itlP50Milliseconds, 4.0) &&
              close(snapshot.itlP95Milliseconds, 4.0) &&
              snapshot.itlSamples == 3,
          "per-token ITL samples are incorrect");
  require(close(snapshot.prefillTokensPerSecond, 10000.0) &&
              close(snapshot.decodeTokensPerSecond, 200.0) &&
              snapshot.prefillInputTokens == 1000 &&
              close(snapshot.prefillWallMilliseconds, 100.0) &&
              snapshot.decodeOutputTokens == 8 &&
              close(snapshot.decodeWallMilliseconds, 40.0),
          "batch throughput metrics are incorrect");
  require(snapshot.draftedTokens == 28 && snapshot.acceptedDraftTokens == 4 &&
              close(snapshot.draftAcceptanceRate, 1.0 / 7.0) &&
              snapshot.capacityFailures == 1 && snapshot.metalFailures == 1,
          "lifetime runtime counters are incorrect");
  require(snapshot.currentPrefillBatch.valid &&
              snapshot.currentPrefillBatch.width == 1 &&
              snapshot.currentPrefillBatch.inputTokens == 1000 &&
              close(snapshot.currentPrefillBatch.tokensPerSecond, 10000.0) &&
              snapshot.currentDecodeBatch.valid &&
              snapshot.currentDecodeBatch.width == 4 &&
              snapshot.currentDecodeBatch.outputTokens == 8 &&
              snapshot.currentDecodeBatch.draftedTokens == 28 &&
              snapshot.currentDecodeBatch.acceptedDraftTokens == 4 &&
              close(snapshot.currentDecodeBatch.tokensPerSecond, 200.0),
          "current prefill/decode batch samples are incomplete");
}

constexpr uint32_t kSelectionWindow = 4096;

// The sort-based nearest-rank formula the selection must reproduce exactly.
double sortedPercentile(std::vector<double> samples, double fraction) {
  if (samples.empty())
    return 0.0;
  std::sort(samples.begin(), samples.end());
  const size_t index =
      static_cast<size_t>(std::ceil(fraction * samples.size())) - 1;
  return samples[std::min(index, samples.size() - 1)];
}

// Records one TTFT and one ITL sample with the given latencies.
void recordLatencies(RuntimeMetrics &metrics, double ttft, double itl) {
  metrics.tokens(0.0, std::nullopt, 1, ttft);
  metrics.tokens(0.0, 0.0, 1, itl);
}

// Quarter-millisecond steps over a narrow range, so windows carry ties.
double randomLatency(std::mt19937 &generator) {
  std::uniform_int_distribution<int> quarters(0, 400);
  return double(quarters(generator)) / 4.0;
}

void testSelectionMatchesSortedPercentiles() {
  std::mt19937 generator(20261019);
  for (const size_t size : {size_t(1), size_t(2), size_t(3),
                            size_t(kSelectionWindow)}) {
    RuntimeMetrics metrics(kSelectionWindow);
    std::vector<double> ttft;
    std::vector<double> itl;
    for (size_t sample = 0; sample < size; ++sample) {
      ttft.push_back(randomLatency(generator));
      itl.push_back(randomLatency(generator));
      recordLatencies(metrics, ttft.back(), itl.back());
    }

    const RuntimeMetricsSnapshot snapshot = metrics.snapshot();

    require(snapshot.ttftP50Milliseconds == sortedPercentile(ttft, 0.50) &&
                snapshot.ttftP95Milliseconds == sortedPercentile(ttft, 0.95) &&
                snapshot.itlP50Milliseconds == sortedPercentile(itl, 0.50) &&
                snapshot.itlP95Milliseconds == sortedPercentile(itl, 0.95),
            "selected percentiles differ from the sorted reference");
  }
}

void testPollsNeverGrowScratch() {
  std::mt19937 generator(7);
  RuntimeMetrics metrics(kSelectionWindow);
  const size_t reserved = metrics.percentileScratchCapacity();
  require(reserved >= kSelectionWindow,
          "percentile scratch is not reserved to the window");

  for (size_t sample = 0; sample < kSelectionWindow + 64; ++sample) {
    recordLatencies(metrics, randomLatency(generator),
                    randomLatency(generator));
    if (sample % 512 == 0 || sample >= kSelectionWindow - 1)
      static_cast<void>(metrics.snapshot());
    require(metrics.percentileScratchCapacity() == reserved,
            "a percentile poll reallocated the scratch");
  }
}

void testValidation() {
  bool threw = false;
  try {
    RuntimeMetrics invalid(0);
    static_cast<void>(invalid);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "zero metrics window was accepted");

  RuntimeMetrics metrics;
  threw = false;
  try {
    metrics.tokens(10.0, 20.0, 1, 19.0);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "backwards token metrics were accepted");
}

} // namespace

int main() {
  try {
    testLatencyWindowAndThroughput();
    testSelectionMatchesSortedPercentiles();
    testPollsNeverGrowScratch();
    testValidation();
    std::cout << "runtime metrics tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "runtime metrics test failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
