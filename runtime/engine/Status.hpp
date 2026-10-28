#pragma once

#include "engine/MemoryPlan.hpp"
#include "engine/Engine.hpp"
#include "metal/MetalBackend.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryAudit.hpp"
#include "engine/ThermalState.hpp"
#include "model/ExpertStoreStats.hpp"
#include "model/KvReadStats.hpp"
#include "model/SparseRowStore.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace splash::model {
struct ModelTelemetry;
} // namespace splash::model

namespace splash::engine {

struct RuntimeCacheIdentity;

enum class WarmupStepStatus { Pending, Complete, MemoryLimited };

struct WarmupReport {
  WarmupStepStatus maximumPrefill = WarmupStepStatus::Pending;
  std::array<WarmupStepStatus, model::ExecutionLimits::maximumBatchWidth>
      decodeBatches{};
  WarmupStepStatus compositeStateRestore = WarmupStepStatus::Pending;
  // Exact executor-selected kernel geometry for the fixed 2048-row path.
  std::string maximumPrefillDetail;
};

// Most recently completed real batch of one work kind. Status queries during
// GPU work retain this sample until the next same-kind batch completes.
struct RuntimeBatchMetricsSnapshot {
  bool valid = false;
  uint32_t width = 0;
  uint32_t inputTokens = 0;
  uint32_t outputTokens = 0;
  uint32_t draftedTokens = 0;
  uint32_t acceptedDraftTokens = 0;
  double wallMilliseconds = 0.0;
  double tokensPerSecond = 0.0;
};

struct RuntimeMetricsSnapshot {
  double ttftP50Milliseconds = 0.0;
  double ttftP95Milliseconds = 0.0;
  uint32_t ttftSamples = 0;
  double itlP50Milliseconds = 0.0;
  double itlP95Milliseconds = 0.0;
  uint32_t itlSamples = 0;
  double prefillTokensPerSecond = 0.0;
  double decodeTokensPerSecond = 0.0;
  // Lifetime native batch counters. Consumers take two status snapshots and
  // subtract them to measure one isolated workload without conflating TTFT
  // with transport latency.
  uint64_t prefillInputTokens = 0;
  double prefillWallMilliseconds = 0.0;
  uint64_t decodeOutputTokens = 0;
  double decodeWallMilliseconds = 0.0;
  // The engine's time for its decode commands, each from the previous
  // command's retirement (or its plan after idleness) to its own: the GPU
  // command plus the host work around it.
  double decodeCycleMilliseconds = 0.0;
  uint64_t draftedTokens = 0;
  uint64_t acceptedDraftTokens = 0;
  double draftAcceptanceRate = 0.0;
  uint64_t capacityFailures = 0;
  uint64_t metalFailures = 0;
  RuntimeBatchMetricsSnapshot currentPrefillBatch;
  RuntimeBatchMetricsSnapshot currentDecodeBatch;
};

// Single-owner metrics accumulator for the native event loop. It stores a
// bounded latency window and lifetime throughput/correctness counters and
// emits the snapshot consumed by runtimeStatusJson().
class RuntimeMetrics final {
public:
  // TTFT and ITL samples the percentiles cover, by default.
  static constexpr uint32_t kLatencyWindow = 4096;

  explicit RuntimeMetrics(uint32_t latencyWindow = kLatencyWindow);

  void tokens(double submittedMilliseconds,
              std::optional<double> previousTokenMilliseconds,
              uint32_t count, double nowMilliseconds);
  void batchCompleted(WorkKind kind, uint32_t width, uint32_t inputTokens,
                      uint32_t outputTokens, uint32_t draftedTokens,
                      uint32_t acceptedDraftTokens, double wallMilliseconds,
                      double cycleMilliseconds);
  void capacityFailed();
  void metalFailed();

  [[nodiscard]] RuntimeMetricsSnapshot snapshot() const;

  // Capacity of the selection scratch snapshot() copies each window into;
  // tests pin it to show a poll never reallocates.
  [[nodiscard]] size_t percentileScratchCapacity() const {
    return percentileScratch_.capacity();
  }

private:
  struct LatencyPercentiles {
    double p50 = 0.0;
    double p95 = 0.0;
  };

  void append(std::deque<double> &samples, double value);
  [[nodiscard]] LatencyPercentiles
  percentiles(const std::deque<double> &samples) const;
  static size_t percentileIndex(double fraction, size_t count);

  uint32_t latencyWindow_ = 0;
  std::deque<double> ttftMilliseconds_;
  std::deque<double> itlMilliseconds_;
  // Reused selection buffer, reserved to the window cap so polls never
  // allocate. Mutable because snapshot() is const; safe because the status
  // provider is its sole caller, on the engine thread.
  mutable std::vector<double> percentileScratch_;
  uint64_t prefillTokens_ = 0;
  uint64_t decodeTokens_ = 0;
  double prefillWallMilliseconds_ = 0.0;
  double decodeWallMilliseconds_ = 0.0;
  double decodeCycleMilliseconds_ = 0.0;
  uint64_t draftedTokens_ = 0;
  uint64_t acceptedDraftTokens_ = 0;
  uint64_t capacityFailures_ = 0;
  uint64_t metalFailures_ = 0;
  RuntimeBatchMetricsSnapshot currentPrefillBatch_;
  RuntimeBatchMetricsSnapshot currentDecodeBatch_;
};

inline RuntimeMetrics::RuntimeMetrics(uint32_t latencyWindow)
    : latencyWindow_(latencyWindow) {
  if (!latencyWindow_)
    throw std::invalid_argument("the latency window must hold a sample");
  percentileScratch_.reserve(latencyWindow_);
}

inline void RuntimeMetrics::tokens(
    double submittedMilliseconds,
    std::optional<double> previousTokenMilliseconds, uint32_t count,
    double nowMilliseconds) {
  if (!count || !std::isfinite(submittedMilliseconds) ||
      !std::isfinite(nowMilliseconds) ||
      nowMilliseconds < submittedMilliseconds) {
    throw std::invalid_argument("invalid metrics token event");
  }
  if (!previousTokenMilliseconds) {
    append(ttftMilliseconds_, nowMilliseconds - submittedMilliseconds);
  } else if (std::isfinite(*previousTokenMilliseconds) &&
             nowMilliseconds >= *previousTokenMilliseconds) {
    append(itlMilliseconds_,
           (nowMilliseconds - *previousTokenMilliseconds) / count);
  } else {
    throw std::invalid_argument("metrics token clock moved backwards");
  }
}

inline void RuntimeMetrics::batchCompleted(
    WorkKind kind, uint32_t width, uint32_t inputTokens,
    uint32_t outputTokens, uint32_t draftedTokens,
    uint32_t acceptedDraftTokens, double wallMilliseconds,
    double cycleMilliseconds) {
  if (!width || !std::isfinite(wallMilliseconds) || wallMilliseconds < 0.0 ||
      !std::isfinite(cycleMilliseconds) || cycleMilliseconds < 0.0 ||
      acceptedDraftTokens > draftedTokens) {
    throw std::invalid_argument("invalid completed batch metrics");
  }
  RuntimeBatchMetricsSnapshot current{
      true,          width,         inputTokens, outputTokens,
      draftedTokens, acceptedDraftTokens, wallMilliseconds, 0.0,
  };
  if (kind == WorkKind::Prefill) {
    prefillTokens_ += inputTokens;
    prefillWallMilliseconds_ += wallMilliseconds;
    if (wallMilliseconds > 0.0)
      current.tokensPerSecond = double(inputTokens) * 1000.0 / wallMilliseconds;
    currentPrefillBatch_ = current;
  } else {
    decodeTokens_ += outputTokens;
    decodeWallMilliseconds_ += wallMilliseconds;
    decodeCycleMilliseconds_ += cycleMilliseconds;
    draftedTokens_ += draftedTokens;
    acceptedDraftTokens_ += acceptedDraftTokens;
    if (wallMilliseconds > 0.0)
      current.tokensPerSecond = double(outputTokens) * 1000.0 / wallMilliseconds;
    currentDecodeBatch_ = current;
  }
}

inline void RuntimeMetrics::capacityFailed() {
  if (capacityFailures_ != std::numeric_limits<uint64_t>::max())
    ++capacityFailures_;
}

inline void RuntimeMetrics::metalFailed() {
  if (metalFailures_ != std::numeric_limits<uint64_t>::max())
    ++metalFailures_;
}

inline RuntimeMetricsSnapshot RuntimeMetrics::snapshot() const {
  RuntimeMetricsSnapshot result;
  const LatencyPercentiles ttft = percentiles(ttftMilliseconds_);
  result.ttftP50Milliseconds = ttft.p50;
  result.ttftP95Milliseconds = ttft.p95;
  result.ttftSamples = static_cast<uint32_t>(ttftMilliseconds_.size());
  const LatencyPercentiles itl = percentiles(itlMilliseconds_);
  result.itlP50Milliseconds = itl.p50;
  result.itlP95Milliseconds = itl.p95;
  result.itlSamples = static_cast<uint32_t>(itlMilliseconds_.size());
  if (prefillWallMilliseconds_ > 0.0) {
    result.prefillTokensPerSecond =
        double(prefillTokens_) * 1000.0 / prefillWallMilliseconds_;
  }
  if (decodeWallMilliseconds_ > 0.0) {
    result.decodeTokensPerSecond =
        double(decodeTokens_) * 1000.0 / decodeWallMilliseconds_;
  }
  result.prefillInputTokens = prefillTokens_;
  result.prefillWallMilliseconds = prefillWallMilliseconds_;
  result.decodeOutputTokens = decodeTokens_;
  result.decodeWallMilliseconds = decodeWallMilliseconds_;
  result.decodeCycleMilliseconds = decodeCycleMilliseconds_;
  result.draftedTokens = draftedTokens_;
  result.acceptedDraftTokens = acceptedDraftTokens_;
  if (draftedTokens_) {
    result.draftAcceptanceRate =
        double(acceptedDraftTokens_) / double(draftedTokens_);
  }
  result.capacityFailures = capacityFailures_;
  result.metalFailures = metalFailures_;
  result.currentPrefillBatch = currentPrefillBatch_;
  result.currentDecodeBatch = currentDecodeBatch_;
  return result;
}

inline void RuntimeMetrics::append(std::deque<double> &samples, double value) {
  if (!std::isfinite(value) || value < 0.0)
    return;
  if (samples.size() == latencyWindow_)
    samples.pop_front();
  samples.push_back(value);
}

// Nearest-rank index of a fraction in a sorted window of count samples.
inline size_t RuntimeMetrics::percentileIndex(double fraction, size_t count) {
  const size_t index = static_cast<size_t>(std::ceil(fraction * count)) - 1;
  return std::min(index, count - 1);
}

// One copy into the reserved scratch, then O(n) selection: p95 first, so p50
// (never the larger index) only needs selecting within the part below it.
inline RuntimeMetrics::LatencyPercentiles
RuntimeMetrics::percentiles(const std::deque<double> &samples) const {
  if (samples.empty())
    return {};
  percentileScratch_.assign(samples.begin(), samples.end());
  const auto first = percentileScratch_.begin();
  const size_t p95Index = percentileIndex(0.95, percentileScratch_.size());
  const size_t p50Index = percentileIndex(0.50, percentileScratch_.size());
  std::nth_element(first, first + p95Index, percentileScratch_.end());
  if (p50Index < p95Index)
    std::nth_element(first, first + p50Index, first + p95Index);
  return {percentileScratch_[p50Index], percentileScratch_[p95Index]};
}

// The native loop's own timing, which its transport measures.
struct NativeLoopTiming {
  double maxTickMilliseconds = 0.0;
};

// The model's weights as the native loop releases and restores them
// (NativeRuntime::releaseWeights).
struct WeightsSnapshot {
  // --idle-release: infinite with `off`, when only a release request frees them.
  double idleReleaseSeconds = 0.0;
  // From a release until the restore has written the last image back.
  bool released = false;
  // From the request that found them released until the restore ends.
  bool restoring = false;
  // The times they were written back, each for a request.
  uint64_t restores = 0;
  // The restores that gave up for want of memory: their requests failed
  // retryably and the weights were released again.
  uint64_t restoreFailures = 0;
};

// The prefill FFN's Neural Engine split (RuntimeResources::aneFfnSnapshot).
struct AneFfnSnapshot {
  // Off: the start left the GPU alone; Split: the split serves; Stopped: it
  // stopped while serving, and the GPU runs alone until the engine restarts.
  enum class State : uint8_t { Off, Split, Stopped };
  State state = State::Off;
  // The start's outcome (AneFfnOutcome), or why the split stopped.
  std::string reason;
  // The split's share and least chunk rows; 0 when off.
  double share = 0.0;
  uint32_t minimumRows = 0;
  // What it ran (ops::AneFfn::Served).
  uint64_t commands = 0, evaluations = 0;
  double aneMilliseconds = 0.0;
};

// A spill-capable target's KV on the SSD cache as /status reports it
// (kv_spill): what the cache holds and did (Cache::spillSnapshot), and its
// readers' memory and reads.
struct KvSpillStatus final {
  KvSpillSnapshot cache;
  std::optional<model::KvBlockCacheStats> blockCache;
  std::optional<model::KvHistoryStreamStats> historyStream;
};

// Single source for /status and native protocol status events.
[[nodiscard]] std::string runtimeStatusJson(
    const EngineMemoryPlan &plan, const EngineSnapshot &core,
    const metal::MetalMemoryStats &metalMemory, const WarmupReport &warmup,
    const MemoryAuditResult &memoryAudit, const RuntimeMetricsSnapshot &metrics,
    const model::ModelTelemetry &executorTelemetry,
    const RuntimeCacheIdentity &cacheIdentity,
    const MemoryGovernorSnapshot &memoryGovernor, bool metalHealthy,
    std::string metalFailureReason, const ResourceWaitSnapshot &resourceWait,
    const NativeLoopTiming &loop, const WeightsSnapshot &weights,
    const AneFfnSnapshot &aneFfn, ThermalState thermalState,
    const std::optional<model::SparseRowStats> &streamedTable = std::nullopt,
    const std::optional<model::ExpertStoreStats> &streamedExperts = std::nullopt,
    const std::optional<KvSpillStatus> &kvSpill = std::nullopt);

} // namespace splash::engine
