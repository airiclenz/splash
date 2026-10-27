#pragma once

#include "engine/MemoryPlan.hpp"
#include "metal/MetalBackend.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace splash::engine {

struct HostMemoryPages {
  uint64_t active = 0;
  uint64_t inactive = 0;
  uint64_t speculative = 0;
  uint64_t wired = 0;
  uint64_t compressor = 0;
  uint64_t fileBacked = 0;
  uint64_t purgeable = 0;
};

// The server's own hot weights: the bytes mapped and the part of them the
// host holds wired at the moment of the sample.
struct WeightResidency {
  uint64_t bytes = 0;
  uint64_t wiredBytes = 0;
};

// Physical memory minus used pages, crediting pageable file-backed and
// purgeable pages regardless of active/inactive status, then charging the
// unwired part of weights: those pages are file-backed and would be credited,
// yet decode must fault them back in. Wired weights already count as used, so
// the result does not depend on whether the GPU holds them wired at the
// sample. Saturates at 0. The governor also enforces the engine budget, host
// reserve and system pressure.
[[nodiscard]] uint64_t estimateHostAvailableMemory(
    const HostMemoryPages &pages, uint64_t pageSize,
    uint64_t physicalMemoryBytes, WeightResidency weights = {}) noexcept;

// One mapped weight file, read-only and owned elsewhere.
struct MappedWeightRange {
  const void *address = nullptr;
  uint64_t bytes = 0;
};

// The page queries one weight sample spends, at 0.5-1 us each; a range too
// small for a share still gets one.
inline constexpr std::size_t kWeightWiringSamplePages = 256;

// Measures how much of the mapped weights is wired by querying the disposition
// of evenly spaced pages, each standing for an equal share of its range.
// Ranges get samples in proportion to their bytes, at least one each and at
// most one per page. The page addresses are fixed at construction, so a
// sample allocates nothing and takes no lock.
//
// The kernel reports an unmapped address as an unwired page, not as an
// error, so the sampler must not outlive the ranges' mappings: a sample after
// an unmap would charge those bytes as unwired.
class WeightWiringSampler final {
public:
  // Throws std::invalid_argument for a range with bytes and no address.
  explicit WeightWiringSampler(std::span<const MappedWeightRange> ranges);

  // bytes is the sum of the ranges' bytes; wiredBytes the share of it whose
  // sampled pages are wired. A failed query counts its share as wired, so a
  // host that refuses the query charges nothing, as before this estimate.
  [[nodiscard]] WeightResidency sample() const noexcept;

private:
  struct SamplePage {
    uint64_t address = 0;
    uint64_t bytes = 0;
  };

  std::vector<SamplePage> pages_;
  uint64_t bytes_ = 0;
};

// One host page reading: the available estimate plus the page categories that
// explain a low one, in bytes. The breakdown is telemetry only; admission and
// the latch read availableBytes alone.
struct HostMemorySample {
  uint64_t availableBytes = 0;
  uint64_t inactiveBytes = 0;
  uint64_t wiredBytes = 0;
  uint64_t compressorBytes = 0;
  // The hot weights the estimate charged for, and their wired part.
  uint64_t weightBytes = 0;
  uint64_t weightWiredBytes = 0;
};

// One host_statistics64 call, plus one weights->sample() when weights is
// given (the serving governor, once the model is mapped); empty when the host
// cannot be measured.
[[nodiscard]] std::optional<HostMemorySample>
queryHostAvailableMemory(const WeightWiringSampler *weights = nullptr) noexcept;

enum class MemoryPressure : uint8_t {
  Normal,
  Warning,
  Critical,
};

// One spelling of the levels for status JSON and startup diagnostics.
[[nodiscard]] inline const char *
memoryPressureName(MemoryPressure pressure) noexcept {
  switch (pressure) {
  case MemoryPressure::Normal:
    return "normal";
  case MemoryPressure::Warning:
    return "warning";
  case MemoryPressure::Critical:
    return "critical";
  }
  return "critical";
}

[[nodiscard]] std::optional<MemoryPressure> querySystemMemoryPressure() noexcept;

struct MemoryGovernorSnapshot {
  uint64_t limitBytes = 0;
  uint64_t observedResidentBytes = 0;
  uint64_t reservedBytes = 0;
  uint64_t headroomBytes = 0;
  MemoryPressure pressure = MemoryPressure::Normal;
  // Failed reservation attempts, including retries of the same request.
  uint64_t deniedReservations = 0;
  // The same denials split by the bound that refused them: host pressure or
  // the engine budget. They sum to deniedReservations.
  uint64_t deniedHostReservations = 0;
  uint64_t deniedEngineReservations = 0;
  bool hostMeasurementValid = false;
  uint64_t hostAvailableBytes = 0;
  // The page breakdown of the same sample, zero while the measurement is
  // invalid.
  uint64_t hostInactiveBytes = 0;
  uint64_t hostWiredBytes = 0;
  // The hot weights charged against the estimate and their wired part.
  uint64_t hostWeightBytes = 0;
  uint64_t hostWeightWiredBytes = 0;
  uint64_t hostCompressorBytes = 0;
  uint64_t hostReserveBytes = 0;
  // The cushion's margins over the reserve: serving growth needs the warning
  // margin left after it; a Warning reclaim episode ends at the recovery margin.
  uint64_t hostWarningMarginBytes = 0;
  uint64_t hostRecoveryMarginBytes = 0;
  uint64_t hostHeadroomBytes = 0;
  MemoryPressure systemPressure = MemoryPressure::Normal;
  bool growthAllowed = true;
  bool hostGrowthAllowed = true;
};

struct MemoryReclaimDirective {
  bool reclaimEmptyKvExtents = false;
  bool evictAllUnpinnedPrefixes = false;
  uint64_t targetBytes = 0;
  // Keep the newest state publication, the point a follow-up request resumes
  // from. Only a shrink that nothing is waiting for can afford to.
  bool keepResumePoint = false;
  // The least the pass sheds while the OS itself reports warn, carried apart
  // from targetBytes (the governor-headroom part): a runway extent the pass
  // keeps mapped offsets targetBytes, never this floor.
  uint64_t osWarningFloorBytes = 0;
};

// Bounded shrink passes separated by a telemetry settling interval. New host
// pressure is never offset by bytes reclaimed earlier in the same episode.
class MemoryPressurePolicy final {
public:
  // osWarningPassBytes is the least a pass takes while the OS itself reports
  // warn: one KV extent, the unit cache backing returns to the host in.
  explicit MemoryPressurePolicy(uint64_t osWarningPassBytes) noexcept
      : osWarningPassBytes_(osWarningPassBytes ? osWarningPassBytes : 1) {}

  // requestWaiting reports whether a request cannot proceed for want of
  // memory. Without one the pass is speculative and keeps the resume point.
  [[nodiscard]] MemoryReclaimDirective
  update(const MemoryGovernorSnapshot &snapshot, double nowMilliseconds,
         bool requestWaiting) noexcept;

private:
  uint64_t osWarningPassBytes_;
  double nextReclaimMilliseconds_ = 0.0;
};

// The sole physical-memory admission ledger. It does not allocate, evict, or
// schedule work; it only gives a short-lived byte reservation to a caller that
// is about to commit a placement heap. That keeps policy out of MetalBackend
// and makes every growth operation transactional.
class MemoryGovernor final {
public:
  using HostAvailableMemoryProvider =
      std::function<std::optional<HostMemorySample>()>;
  // Reads the Metal footprint: the backend's refreshed device sample when
  // its argument is true, its cached accounting otherwise. Must not throw.
  using MetalMemoryStatsProvider =
      std::function<metal::MetalMemoryStats(bool refreshDevice)>;

  class Reservation final {
  public:
    Reservation() = default;
    ~Reservation();
    Reservation(const Reservation &) = delete;
    Reservation &operator=(const Reservation &) = delete;
    Reservation(Reservation &&) noexcept;
    Reservation &operator=(Reservation &&) noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    void commit();

  private:
    Reservation(MemoryGovernor *owner, uint64_t bytes);
    void release() noexcept;

    MemoryGovernor *owner_ = nullptr;
    uint64_t bytes_ = 0;

    friend class MemoryGovernor;
  };

  // Derives the margins from a bare reserve per EngineMemoryPolicy.
  MemoryGovernor(metal::MetalBackend &backend, uint64_t limitBytes,
                 uint64_t hostReserveBytes)
      : MemoryGovernor(backend, limitBytes,
                       EngineMemoryPolicy::hostCushionForReserve(
                           hostReserveBytes)) {}
  MemoryGovernor(metal::MetalBackend &backend, uint64_t limitBytes,
                 HostCushion hostCushion)
      : MemoryGovernor(backend, limitBytes, hostCushion,
                       [] { return queryHostAvailableMemory(); }) {}
  // The backend adapters live here so MemoryGovernor.cpp never calls into
  // MetalBackend and links without it.
  MemoryGovernor(metal::MetalBackend &backend, uint64_t limitBytes,
                 HostCushion hostCushion,
                 HostAvailableMemoryProvider hostAvailableMemory)
      : MemoryGovernor(
            [metalBackend = &backend](bool refreshDevice) {
              return refreshDevice ? metalBackend->refreshMemoryStats()
                                   : metalBackend->memoryStats();
            },
            limitBytes, hostCushion, std::move(hostAvailableMemory)) {}
  // Samples the host once and stores that sample, and the latch it sets, as
  // the first state snapshot() reports. Throws std::invalid_argument for a
  // zero reserve or warning margin, or a recovery margin below the warning
  // margin.
  MemoryGovernor(MetalMemoryStatsProvider metalMemoryStats,
                 uint64_t limitBytes, HostCushion hostCushion,
                 HostAvailableMemoryProvider hostAvailableMemory);

  // Grants while the engine budget fits, the host keeps at least the warning
  // margin above the reserve after the growth (while bootstrapping: stays
  // above the reserve), and pressure is not Critical. The Warning latch never
  // denies on its own.
  [[nodiscard]] std::optional<Reservation> tryReserve(
      uint64_t bytes, metal::AllocationFailure *failure = nullptr);
  // Low-level storage/model components receive only this transactional
  // callback, so physical allocation stays governed without introducing a
  // reverse dependency on engine policy.
  [[nodiscard]] metal::AllocationAdmission allocationAdmission() noexcept;
  void setPressure(MemoryPressure pressure) noexcept;
  // While set, host admission applies the load guard's floor (the reserve)
  // instead of the serving floor, so load and warmup allocate as far as the
  // guard lets them. A new governor applies the serving rule; startup sets
  // the flag and clears it where serving begins, after which growth stays
  // paused until headroom reaches the serving floor.
  void setBootstrapping(bool bootstrapping) noexcept;
  // Samples host memory, stores the sample and advances the Warning latch,
  // which drives reported pressure and paced reclaim, never admission. Only
  // this and tryReserve() move the latch, so reported pressure never depends
  // on how often anyone reads snapshot().
  void observe() noexcept;
  // Reads the stored host sample and latch, plus the backend's cached
  // accounting: no host sample, no latch write.
  [[nodiscard]] MemoryGovernorSnapshot snapshot() const noexcept;

private:
  [[nodiscard]] uint64_t
  observedResidentBytes(bool refreshDevice = false) const noexcept;
  [[nodiscard]] std::optional<HostMemorySample>
  sampleHostAvailable() const noexcept;
  [[nodiscard]] uint64_t
  hostHeadroomBytes(const std::optional<HostMemorySample> &hostSample,
                    uint64_t reservedBytes) const noexcept;
  // Stores hostSample as the current sample and advances the latch from
  // it. Callers hold mutex_ (or own the governor exclusively).
  // The host half of admission, shared by tryReserve() and snapshot() so
  // hostGrowthAllowed reports exactly what a reservation would be told.
  [[nodiscard]] bool hostAdmits(uint64_t hostHeadroom,
                                MemoryPressure pressure) const noexcept;
  [[nodiscard]] MemoryPressure updateEffectivePressure(
      const std::optional<HostMemorySample> &hostSample,
      uint64_t reservedBytes) noexcept;
  [[nodiscard]] MemoryPressure effectivePressure() const noexcept;
  void release(uint64_t bytes) noexcept;

  MetalMemoryStatsProvider metalMemoryStats_;
  uint64_t limitBytes_ = 0;
  HostCushion hostCushion_;
  HostAvailableMemoryProvider hostAvailableMemory_;
  mutable std::mutex mutex_;
  uint64_t reservedBytes_ = 0;
  uint64_t deniedReservations_ = 0;
  uint64_t deniedHostReservations_ = 0;
  uint64_t deniedEngineReservations_ = 0;
  MemoryPressure systemPressure_ = MemoryPressure::Normal;
  std::optional<HostMemorySample> sampledHost_;
  bool hostConstrained_ = false;
  bool isBootstrapping_ = false;
};

} // namespace splash::engine
