#include "TestFakeHost.hpp"
#include "engine/MemoryGovernor.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

using namespace splash;
using namespace splash::engine;

namespace {

constexpr HostCushion kCushion{64 * 1024, 1 * kGiB, 2 * kGiB};
constexpr uint64_t kLimitBytes = 8 * kGiB;
constexpr uint64_t kResidentBytes = 1 * kGiB;
constexpr uint64_t kAmpleHostBytes = kCushion.reserveBytes + 3 * kGiB;
constexpr uint64_t kLowHostBytes =
    kCushion.reserveBytes + kCushion.warningMarginBytes / 2;
// Above the warning margin, below the recovery margin: a held latch stays.
constexpr uint64_t kLatchedHostBytes =
    kCushion.reserveBytes +
    (kCushion.warningMarginBytes + kCushion.recoveryMarginBytes) / 2;
constexpr int kRepeatedSnapshots = 16;
constexpr uint64_t kInactiveHostBytes = 3 * kGiB;
constexpr uint64_t kWiredHostBytes = 5 * kGiB;
constexpr uint64_t kCompressorHostBytes = 1 * kGiB;

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

// The shared fake host, starting ample.
struct FakeHost final : engine::test::FakeHost {
  FakeHost()
      : engine::test::FakeHost{kAmpleHostBytes, kInactiveHostBytes,
                               kWiredHostBytes, kCompressorHostBytes} {}
};

MemoryGovernor makeGovernor(FakeHost &host,
                            HostCushion cushion = kCushion) {
  return MemoryGovernor(
      [](bool) {
        metal::MetalMemoryStats stats;
        stats.allocatedBytes = kResidentBytes;
        return stats;
      },
      kLimitBytes, cushion, host.sampler());
}

void testSnapshotAfterConstructionUsesStartupSample() {
  FakeHost host;
  MemoryGovernor governor = makeGovernor(host);
  require(host.samples == 1, "construction did not sample the host once");

  const MemoryGovernorSnapshot state = governor.snapshot();
  require(host.samples == 1, "the first snapshot sampled the host");
  require(state.hostMeasurementValid &&
              state.hostAvailableBytes == kAmpleHostBytes,
          "the startup host sample was not stored");
  require(state.pressure == MemoryPressure::Normal &&
              state.hostGrowthAllowed && state.growthAllowed,
          "a snapshot before any observe() reported growth paused");
  require(state.observedResidentBytes == kResidentBytes,
          "the injected Metal stats were not read");
}

void testSnapshotNeitherSamplesNorLatches() {
  FakeHost host;
  MemoryGovernor governor = makeGovernor(host);
  host.availableBytes = kLowHostBytes;
  for (int index = 0; index < kRepeatedSnapshots; ++index) {
    const MemoryGovernorSnapshot state = governor.snapshot();
    require(state.pressure == MemoryPressure::Normal &&
                state.hostAvailableBytes == kAmpleHostBytes,
            "a snapshot latched a host change it was never told about");
  }
  require(host.samples == 1, "repeated snapshots sampled the host");

  governor.observe();
  require(host.samples == 2, "observe() did not sample the host once");
  host.availableBytes = kAmpleHostBytes;
  for (int index = 0; index < kRepeatedSnapshots; ++index) {
    const MemoryGovernorSnapshot state = governor.snapshot();
    require(state.pressure == MemoryPressure::Warning &&
                !state.hostGrowthAllowed,
            "a snapshot cleared the Warning latch");
  }
  require(host.samples == 2, "repeated latched snapshots sampled the host");
}

void testObserveLatchesWarningAndClearsAtRecovery() {
  FakeHost host;
  MemoryGovernor governor = makeGovernor(host);

  host.availableBytes = kLowHostBytes;
  governor.observe();
  MemoryGovernorSnapshot state = governor.snapshot();
  require(state.pressure == MemoryPressure::Warning &&
              !state.hostGrowthAllowed && !state.growthAllowed,
          "observe() did not latch Warning below the warning margin");

  host.availableBytes = kLatchedHostBytes;
  governor.observe();
  require(governor.snapshot().pressure == MemoryPressure::Warning,
          "the latch cleared before the recovery margin");

  host.availableBytes = kAmpleHostBytes;
  governor.observe();
  state = governor.snapshot();
  require(state.pressure == MemoryPressure::Normal &&
              state.hostGrowthAllowed && state.growthAllowed,
          "observe() did not clear the latch at recovery");

  host.availableBytes.reset();
  governor.observe();
  state = governor.snapshot();
  require(!state.hostMeasurementValid &&
              state.pressure == MemoryPressure::Warning,
          "missing host telemetry did not fail closed");
}

void testTryReserveAdvancesLatch() {
  FakeHost host;
  MemoryGovernor governor = makeGovernor(host);

  host.availableBytes = kLowHostBytes;
  metal::AllocationFailure failure = metal::AllocationFailure::None;
  require(!governor.tryReserve(1, &failure).has_value() &&
              failure == metal::AllocationFailure::HostPressure,
          "low host headroom admitted growth");
  require(host.samples == 2, "tryReserve() did not sample the host once");
  require(governor.snapshot().pressure == MemoryPressure::Warning,
          "tryReserve() did not advance the latch");

  host.availableBytes = kAmpleHostBytes;
  require(governor.tryReserve(1).has_value() &&
              governor.snapshot().pressure == MemoryPressure::Normal,
          "tryReserve() did not clear the latch at recovery");
}

// Each denial counts once under the bound that refused it and once in the
// total; a grant counts nowhere.
void testDenialsAreCountedByCause() {
  FakeHost host;
  MemoryGovernor governor = makeGovernor(host);

  metal::AllocationFailure engineFailure = metal::AllocationFailure::None;
  const bool grantedOverBudget =
      governor.tryReserve(kLimitBytes, &engineFailure).has_value();
  const MemoryGovernorSnapshot afterEngine = governor.snapshot();
  host.availableBytes = kLowHostBytes;
  metal::AllocationFailure hostFailure = metal::AllocationFailure::None;
  const bool grantedUnderPressure =
      governor.tryReserve(1, &hostFailure).has_value();
  host.availableBytes = kAmpleHostBytes;
  const bool grantedAtRecovery = governor.tryReserve(1).has_value();
  const MemoryGovernorSnapshot counted = governor.snapshot();

  require(!grantedOverBudget &&
              engineFailure == metal::AllocationFailure::EngineBudget &&
              afterEngine.deniedReservations == 1 &&
              afterEngine.deniedEngineReservations == 1 &&
              afterEngine.deniedHostReservations == 0,
          "an engine-budget denial was not counted under its cause");
  require(!grantedUnderPressure &&
              hostFailure == metal::AllocationFailure::HostPressure,
          "low host headroom admitted growth");
  require(grantedAtRecovery && counted.deniedReservations == 2 &&
              counted.deniedHostReservations == 1 &&
              counted.deniedEngineReservations == 1,
          "the denial causes do not sum to the total");
}

void testReservationThatFitsIsGrantedWhileWarningLatched() {
  FakeHost host;
  MemoryGovernor governor = makeGovernor(host);
  host.availableBytes = kLowHostBytes;
  governor.observe();
  host.availableBytes = kLatchedHostBytes;
  governor.observe();

  const MemoryGovernorSnapshot latched = governor.snapshot();
  auto reservation = governor.tryReserve(1);

  require(latched.pressure == MemoryPressure::Warning &&
              latched.hostGrowthAllowed && latched.growthAllowed,
          "a held latch reported host growth paused");
  require(reservation.has_value() &&
              governor.snapshot().pressure == MemoryPressure::Warning,
          "the Warning latch denied a reservation that fits");
}

// The pre-load check refuses at or below admissionFloorBytes() and a one-byte
// serving reservation meets the same floor, the snapshot agreeing with it;
// the load guard passes inside the warning margin and refuses at the reserve.
void testStartupAndServingShareOneFloor() {
  FakeHost host;
  MemoryGovernor governor = makeGovernor(host);
  const uint64_t floorBytes = kCushion.admissionFloorBytes();

  const bool preLoadAtFloor = hostStandsAboveFloor(floorBytes, floorBytes);
  const bool preLoadAboveFloor =
      hostStandsAboveFloor(floorBytes + 1, floorBytes);
  const bool guardInBand =
      hostStandsAboveFloor(kLowHostBytes, kCushion.reserveBytes);
  const bool guardAtReserve =
      hostStandsAboveFloor(kCushion.reserveBytes, kCushion.reserveBytes);
  host.availableBytes = floorBytes;
  metal::AllocationFailure failure = metal::AllocationFailure::None;
  const bool grantedAtFloor = governor.tryReserve(1, &failure).has_value();
  const bool reportedAtFloor = governor.snapshot().hostGrowthAllowed;
  host.availableBytes = floorBytes + 1;
  const bool grantedAboveFloor = governor.tryReserve(1).has_value();
  host.availableBytes = floorBytes - 1;
  governor.observe();
  const bool reportedBelowFloor = governor.snapshot().hostGrowthAllowed;

  require(!preLoadAtFloor && preLoadAboveFloor,
          "the pre-load check did not stand on the admission floor");
  require(guardInBand && !guardAtReserve,
          "the load guard did not stand on the reserve");
  require(!grantedAtFloor &&
              failure == metal::AllocationFailure::HostPressure,
          "serving admitted growth at the admission floor");
  require(grantedAboveFloor,
          "serving refused growth above the admission floor");
  require(reportedAtFloor && !reportedBelowFloor,
          "the snapshot's host test differs from the reservation's");
}

// While bootstrapping, a governor inside the warning margin admits growth
// that keeps the host above the reserve and denies growth that reaches it;
// once serving begins the serving floor applies and growth reports paused.
void testBootstrapPhaseAdmitsAgainstReserve() {
  FakeHost host;
  MemoryGovernor governor = makeGovernor(host);
  governor.setBootstrapping(true);

  host.availableBytes = kLowHostBytes;
  const bool grantedInBand = governor.tryReserve(1).has_value();
  const bool reportedInBand = governor.snapshot().hostGrowthAllowed;
  host.availableBytes = kCushion.reserveBytes + 1;
  metal::AllocationFailure atReserve = metal::AllocationFailure::None;
  const bool grantedAtReserve = governor.tryReserve(1, &atReserve).has_value();
  governor.setBootstrapping(false);
  host.availableBytes = kLowHostBytes;
  metal::AllocationFailure serving = metal::AllocationFailure::None;
  const bool grantedServing = governor.tryReserve(1, &serving).has_value();
  const MemoryGovernorSnapshot servingSnapshot = governor.snapshot();

  require(grantedInBand && reportedInBand,
          "bootstrap denied growth that keeps the host above the reserve");
  require(!grantedAtReserve &&
              atReserve == metal::AllocationFailure::HostPressure,
          "bootstrap admitted growth that reaches the reserve");
  require(!grantedServing &&
              serving == metal::AllocationFailure::HostPressure &&
              !servingSnapshot.hostGrowthAllowed &&
              !servingSnapshot.growthAllowed,
          "serving inside the warning margin did not pause growth");
}

void testCushionScalesWithPhysicalMemory() {
  struct Case {
    uint64_t physicalBytes;
    uint64_t reserveBytes;
    uint64_t warningBytes;
    uint64_t recoveryBytes;
  };
  const uint64_t sixteenGiBReserve = 16 * kGiB / 10;
  const Case cases[] = {
      {8 * kGiB, 8 * kGiB / 10, 256 * kMiB, 512 * kMiB},
      {16 * kGiB, sixteenGiBReserve, sixteenGiBReserve / 4,
       sixteenGiBReserve / 2},
      {32 * kGiB, 2 * kGiB, 512 * kMiB, 1 * kGiB},
      {128 * kGiB, 2 * kGiB, 512 * kMiB, 1 * kGiB},
  };

  for (const Case &expected : cases) {
    const HostCushion cushion =
        EngineMemoryPolicy::hostCushion(expected.physicalBytes);
    require(cushion.reserveBytes == expected.reserveBytes &&
                cushion.warningMarginBytes == expected.warningBytes &&
                cushion.recoveryMarginBytes == expected.recoveryBytes,
            "the host cushion does not follow the ratified formulas");
  }
}

void testInvalidCushionRejected() {
  FakeHost host;
  const HostCushion invalid[] = {
      {0, kCushion.warningMarginBytes, kCushion.recoveryMarginBytes},
      {kCushion.reserveBytes, 0, kCushion.recoveryMarginBytes},
      {kCushion.reserveBytes, kCushion.recoveryMarginBytes,
       kCushion.warningMarginBytes},
  };

  for (const HostCushion &cushion : invalid) {
    bool rejected = false;
    try {
      static_cast<void>(makeGovernor(host, cushion));
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "an incoherent host cushion was accepted");
  }
}

void testObserveStoresPageBreakdown() {
  FakeHost host;
  MemoryGovernor governor = makeGovernor(host);
  host.wiredBytes = 2 * kWiredHostBytes;

  const MemoryGovernorSnapshot stored = governor.snapshot();
  require(stored.hostInactiveBytes == kInactiveHostBytes &&
              stored.hostWiredBytes == kWiredHostBytes &&
              stored.hostCompressorBytes == kCompressorHostBytes,
          "the startup page breakdown was not stored");

  governor.observe();
  const MemoryGovernorSnapshot observed = governor.snapshot();
  require(observed.hostWiredBytes == 2 * kWiredHostBytes &&
              observed.hostInactiveBytes == kInactiveHostBytes,
          "observe() did not store the new page breakdown");
  require(host.samples == 2, "the page breakdown cost an extra host sample");

  host.availableBytes.reset();
  governor.observe();
  const MemoryGovernorSnapshot invalid = governor.snapshot();
  require(!invalid.hostMeasurementValid && invalid.hostInactiveBytes == 0 &&
              invalid.hostWiredBytes == 0 && invalid.hostCompressorBytes == 0,
          "a failed sample kept a stale page breakdown");
}

void testSystemPressureReadsWithoutSampling() {
  FakeHost host;
  MemoryGovernor governor = makeGovernor(host);
  governor.setPressure(MemoryPressure::Critical);
  const MemoryGovernorSnapshot state = governor.snapshot();
  require(state.pressure == MemoryPressure::Critical &&
              state.systemPressure == MemoryPressure::Critical &&
              !state.growthAllowed,
          "a snapshot hid the system's critical verdict");
  require(host.samples == 1, "system pressure forced a host sample");
}


// Host page sets measured on a 32 GiB, 16 KiB-page host serving a 27B model
// whose hot (target + draft) weights are kHotWeightBytes. The same state is
// read with those weights pageable (idleUnwired) and GPU-wired (idleWired):
// wiring moves exactly kHotWeightBytes of pages from file-backed to wired.
constexpr uint64_t kHostPageBytes = 16384;
constexpr uint64_t kHostPhysicalBytes = 34'359'738'368;
constexpr uint64_t kHotWeightBytes = 16'425'680'896;
constexpr HostMemoryPages kIdleUnwiredPages{
    .active = 721'064, .inactive = 793'264, .speculative = 1'554,
    .wired = 180'029, .compressor = 271'014, .fileBacked = 1'264'808,
    .purgeable = 8'231};
constexpr HostMemoryPages kIdleWiredPages{
    .active = 121'064, .inactive = 390'720, .speculative = 1'554,
    .wired = 1'182'573, .compressor = 271'014, .fileBacked = 262'264,
    .purgeable = 8'231};
// The floor of the run that swapped: 18.4 GB of mlocked ballast, most of the
// weights evicted, the remaining file-backed pages counted as available.
constexpr HostMemoryPages kSwapFloorPages{
    .active = 300'000, .inactive = 153'857, .wired = 1'541'748,
    .compressor = 68'970, .fileBacked = 250'000};
// The raw estimate today credits the unwired weights as available.
constexpr uint64_t kIdleRawAvailableBytes = 22'991'110'144;
constexpr uint64_t kIdleHonestAvailableBytes = 6'565'429'248;
constexpr uint64_t kIdleHonestHeadroomBytes = 4'417'945'600;
constexpr uint64_t kSwapFloorRawAvailableBytes = 4'629'741'568;
constexpr uint64_t kSwapFloorRawHeadroomBytes = 2'482'257'920;

uint64_t measuredEstimate(const HostMemoryPages &pages,
                          WeightResidency weights = {}) {
  return estimateHostAvailableMemory(pages, kHostPageBytes,
                                     kHostPhysicalBytes, weights);
}

// A governor on the measured host's cushion whose provider returns the
// corrected sample for pages and weights, which the test may change.
struct MeasuredHost final {
  HostMemoryPages pages = kIdleUnwiredPages;
  WeightResidency weights{kHotWeightBytes, 0};

  [[nodiscard]] MemoryGovernor::HostAvailableMemoryProvider sampler() {
    return [this]() -> std::optional<HostMemorySample> {
      return HostMemorySample{.availableBytes = measuredEstimate(pages, weights),
                              .weightBytes = weights.bytes,
                              .weightWiredBytes = weights.wiredBytes};
    };
  }
};

MemoryGovernor makeMeasuredGovernor(MeasuredHost &host) {
  return MemoryGovernor(
      [](bool) {
        metal::MetalMemoryStats stats;
        stats.allocatedBytes = kResidentBytes;
        return stats;
      },
      kLimitBytes, EngineMemoryPolicy::hostCushion(kHostPhysicalBytes),
      host.sampler());
}

// The bug: at the floor of the run that swapped, the raw estimate read
// 2.48 GB of headroom (Normal) while the unwired hot weights it credited
// could not be resident. Charging them reads zero and latches Warning.
void testEstimate_swapFloorWithUnwiredWeights_readsNoHeadroom() {
  MeasuredHost host;
  MemoryGovernor governor = makeMeasuredGovernor(host);
  const uint64_t reserveBytes =
      EngineMemoryPolicy::hostCushion(kHostPhysicalBytes).reserveBytes;

  const uint64_t raw = measuredEstimate(kSwapFloorPages);
  const uint64_t charged =
      measuredEstimate(kSwapFloorPages, {kHotWeightBytes, 0});
  host.pages = kSwapFloorPages;
  governor.observe();
  const MemoryGovernorSnapshot state = governor.snapshot();

  require(raw == kSwapFloorRawAvailableBytes &&
              raw - reserveBytes == kSwapFloorRawHeadroomBytes,
          "the swap-floor page set no longer reproduces today's reading");
  require(charged == 0,
          "unwired hot weights were credited as available host memory");
  require(state.pressure == MemoryPressure::Warning &&
              !state.hostGrowthAllowed && state.hostHeadroomBytes == 0,
          "the governor read headroom at the floor where the host swapped");
}

// The measured idle state: today's estimate reads 22.99 GB, of which
// 16.4 GB is the server's own unwired hot weights.
void testEstimate_idleWithUnwiredWeights_chargesThem() {
  const uint64_t raw = measuredEstimate(kIdleUnwiredPages);

  const uint64_t charged =
      measuredEstimate(kIdleUnwiredPages, {kHotWeightBytes, 0});

  require(raw == kIdleRawAvailableBytes,
          "the idle page set no longer reproduces today's reading");
  require(charged == kIdleHonestAvailableBytes,
          "the idle estimate did not charge the unwired hot weights");
}

// Wired weights already count as used, so the corrected estimate is the
// same whether the sample lands while the GPU holds them wired or not.
void testEstimate_sameStateWiredOrUnwired_readsTheSame() {
  const uint64_t unwired =
      measuredEstimate(kIdleUnwiredPages, {kHotWeightBytes, 0});
  const uint64_t wired =
      measuredEstimate(kIdleWiredPages, {kHotWeightBytes, kHotWeightBytes});
  const uint64_t wiredWithoutWeights = measuredEstimate(kIdleWiredPages);

  require(unwired == kIdleHonestAvailableBytes &&
              wired == kIdleHonestAvailableBytes &&
              wiredWithoutWeights == kIdleHonestAvailableBytes,
          "the estimate depends on whether the weights happen to be wired");
}

// Only the unwired part of the weights is charged; wired bytes above the
// weight bytes never credit, and the charge saturates at zero.
void testEstimate_weightCharge_isUnwiredBytesSaturating() {
  constexpr uint64_t physicalPages = 100;
  constexpr HostMemoryPages pages{.active = 20, .wired = 10};
  const uint64_t raw = estimateHostAvailableMemory(pages, kHostPageBytes,
                                                   physicalPages * kHostPageBytes);
  auto withWeights = [&](uint64_t weightPages, uint64_t wiredPages) {
    return estimateHostAvailableMemory(
        pages, kHostPageBytes, physicalPages * kHostPageBytes,
        {weightPages * kHostPageBytes, wiredPages * kHostPageBytes});
  };

  const uint64_t defaulted = estimateHostAvailableMemory(
      pages, kHostPageBytes, physicalPages * kHostPageBytes, {});
  const uint64_t none = withWeights(0, 0);
  const uint64_t fullyWired = withWeights(40, 40);
  const uint64_t fullyUnwired = withWeights(40, 0);
  const uint64_t partlyWired = withWeights(40, 30);
  const uint64_t overWired = withWeights(40, 60);
  const uint64_t aboveAvailable = withWeights(71, 0);
  const uint64_t atAvailable = withWeights(70, 0);

  require(raw == 70 * kHostPageBytes && defaulted == raw && none == raw,
          "no weights changed the estimate");
  require(fullyWired == raw, "fully wired weights were charged twice");
  require(fullyUnwired == 30 * kHostPageBytes,
          "fully unwired weights were not charged in full");
  require(partlyWired == 60 * kHostPageBytes,
          "partly wired weights were not charged their unwired part");
  require(overWired == raw,
          "wired bytes above the weight bytes credited availability");
  require(aboveAvailable == 0 && atAvailable == 0,
          "the weight charge did not saturate at zero");
}

// The idle shape must not raise a false Warning or reclaim at normal
// pressure, whether the weights are pageable or wired.
void testGovernor_idleWithHotWeights_staysNormal() {
  struct Case {
    HostMemoryPages pages;
    WeightResidency weights;
  };
  const Case cases[] = {
      {kIdleUnwiredPages, {kHotWeightBytes, 0}},
      {kIdleWiredPages, {kHotWeightBytes, kHotWeightBytes}},
  };

  for (const Case &idle : cases) {
    MeasuredHost host{idle.pages, idle.weights};
    MemoryGovernor governor = makeMeasuredGovernor(host);
    MemoryPressurePolicy policy(64 * kMiB);
    bool reclaimed = false;
    for (int index = 0; index < kRepeatedSnapshots; ++index) {
      governor.observe();
      const MemoryReclaimDirective directive =
          policy.update(governor.snapshot(), 1000.0 * index, false);
      reclaimed = reclaimed || directive.reclaimEmptyKvExtents ||
                  directive.evictAllUnpinnedPrefixes || directive.targetBytes;
    }
    const MemoryGovernorSnapshot state = governor.snapshot();

    require(state.hostHeadroomBytes == kIdleHonestHeadroomBytes,
            "the idle headroom is not the honest 4.42 GB");
    require(state.pressure == MemoryPressure::Normal &&
                state.hostGrowthAllowed && state.growthAllowed,
            "the idle server latched a false Warning");
    require(!reclaimed, "the idle server reclaimed at normal pressure");
  }
}

void testObserve_storesWeightResidency() {
  MeasuredHost host;
  host.weights = {kHotWeightBytes, kHotWeightBytes / 4};
  MemoryGovernor governor = makeMeasuredGovernor(host);
  const MemoryGovernorSnapshot stored = governor.snapshot();

  host.weights = {kHotWeightBytes, kHotWeightBytes};
  host.pages = kIdleWiredPages;
  governor.observe();
  const MemoryGovernorSnapshot observed = governor.snapshot();

  require(stored.hostWeightBytes == kHotWeightBytes &&
              stored.hostWeightWiredBytes == kHotWeightBytes / 4,
          "the startup weight residency was not stored");
  require(observed.hostWeightBytes == kHotWeightBytes &&
              observed.hostWeightWiredBytes == kHotWeightBytes,
          "observe() did not store the new weight residency");
}

void testObserve_invalidSample_clearsWeightResidency() {
  bool measurable = true;
  MemoryGovernor governor(
      [](bool) {
        metal::MetalMemoryStats stats;
        stats.allocatedBytes = kResidentBytes;
        return stats;
      },
      kLimitBytes, kCushion,
      [&measurable]() -> std::optional<HostMemorySample> {
        if (!measurable)
          return std::nullopt;
        return HostMemorySample{.availableBytes = kAmpleHostBytes,
                                .weightBytes = kHotWeightBytes,
                                .weightWiredBytes = kHotWeightBytes / 2};
      });

  measurable = false;
  governor.observe();
  const MemoryGovernorSnapshot invalid = governor.snapshot();

  require(!invalid.hostMeasurementValid && invalid.hostWeightBytes == 0 &&
              invalid.hostWeightWiredBytes == 0,
          "a failed sample kept a stale weight residency");
}

// An anonymous page-aligned mapping standing in for a weight file, unlocked
// and unmapped on destruction.
class TestMapping final {
public:
  explicit TestMapping(std::size_t pages)
      : bytes_(pages * static_cast<std::size_t>(sysconf(_SC_PAGESIZE))) {
    void *address = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANON, -1, 0);
    require(address != MAP_FAILED, "unable to map test weight pages");
    address_ = address;
  }
  ~TestMapping() {
    munlock(address_, bytes_);
    munmap(address_, bytes_);
  }
  TestMapping(const TestMapping &) = delete;
  TestMapping &operator=(const TestMapping &) = delete;

  [[nodiscard]] MappedWeightRange range() const noexcept {
    return {address_, bytes_};
  }
  [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }
  void lock(std::size_t offset, std::size_t bytes) {
    require(mlock(static_cast<char *>(address_) + offset, bytes) == 0,
            "unable to mlock test weight pages");
  }
  void lock() { lock(0, bytes_); }
  void unlock() noexcept { munlock(address_, bytes_); }

private:
  std::size_t bytes_ = 0;
  void *address_ = nullptr;
};

constexpr std::size_t kFirstRangePages = 24;
constexpr std::size_t kSecondRangePages = 40;

void testWeightWiringSampler_nothingLocked_readsNoWiredBytes() {
  TestMapping first(kFirstRangePages);
  TestMapping second(kSecondRangePages);
  const std::vector<MappedWeightRange> ranges{first.range(), second.range()};
  const WeightWiringSampler sampler(ranges);

  const WeightResidency residency = sampler.sample();

  require(residency.bytes == first.bytes() + second.bytes() &&
              residency.wiredBytes == 0,
          "pageable weight pages were read as wired");
}

void testWeightWiringSampler_allLocked_readsAllWired() {
  TestMapping first(kFirstRangePages);
  TestMapping second(kSecondRangePages);
  const std::vector<MappedWeightRange> ranges{first.range(), second.range()};
  const WeightWiringSampler sampler(ranges);
  first.lock();
  second.lock();

  const WeightResidency residency = sampler.sample();

  require(residency.bytes == first.bytes() + second.bytes() &&
              residency.wiredBytes == residency.bytes,
          "wired weight pages were not read as wired");
}

void testWeightWiringSampler_oneRangeLocked_readsThatRangeExactly() {
  TestMapping first(kFirstRangePages);
  TestMapping second(kSecondRangePages);
  const std::vector<MappedWeightRange> ranges{first.range(), second.range()};
  const WeightWiringSampler sampler(ranges);
  second.lock();

  const WeightResidency residency = sampler.sample();
  second.unlock();
  const WeightResidency released = sampler.sample();

  require(residency.wiredBytes == second.bytes(),
          "a wired range was not read as exactly its own bytes");
  require(released.wiredBytes == 0, "an unwired range still read as wired");
}

// A range larger than the sample budget is sampled evenly: half of it wired
// reads half, within the weight of one sample on either side.
void testWeightWiringSampler_largeRangeHalfLocked_readsAboutHalf() {
  const std::size_t pages = 4 * kWeightWiringSamplePages;
  TestMapping mapping(pages);
  const std::vector<MappedWeightRange> ranges{mapping.range()};
  const WeightWiringSampler sampler(ranges);
  mapping.lock(0, static_cast<std::size_t>(mapping.bytes() / 2));

  const WeightResidency residency = sampler.sample();

  const uint64_t sampleBytes = mapping.bytes() / kWeightWiringSamplePages;
  const uint64_t half = mapping.bytes() / 2;
  require(residency.bytes == mapping.bytes(),
          "a large range did not report its bytes");
  require(residency.wiredBytes + sampleBytes >= half &&
              residency.wiredBytes <= half + sampleBytes,
          "a half-wired range was not sampled evenly across its pages");
}

void testWeightWiringSampler_noRanges_readsNothing() {
  const std::vector<MappedWeightRange> ranges;
  const WeightWiringSampler sampler(ranges);

  const WeightResidency residency = sampler.sample();

  require(residency.bytes == 0 && residency.wiredBytes == 0,
          "an empty sampler reported weight bytes");
}

void testQueryHostAvailableMemory_withSampler_reportsWeightResidency() {
  TestMapping mapping(kFirstRangePages);
  const std::vector<MappedWeightRange> ranges{mapping.range()};
  const WeightWiringSampler sampler(ranges);
  mapping.lock();

  const std::optional<HostMemorySample> withWeights =
      queryHostAvailableMemory(&sampler);
  const std::optional<HostMemorySample> withoutWeights =
      queryHostAvailableMemory();

  require(withWeights && withWeights->weightBytes == mapping.bytes() &&
              withWeights->weightWiredBytes == mapping.bytes(),
          "the host sample did not report the sampler's weight residency");
  require(withoutWeights && withoutWeights->weightBytes == 0 &&
              withoutWeights->weightWiredBytes == 0,
          "a host sample without a sampler reported weight bytes");
}

} // namespace

int main() {
  try {
    testSnapshotAfterConstructionUsesStartupSample();
    testSnapshotNeitherSamplesNorLatches();
    testObserveLatchesWarningAndClearsAtRecovery();
    testTryReserveAdvancesLatch();
    testDenialsAreCountedByCause();
    testReservationThatFitsIsGrantedWhileWarningLatched();
    testStartupAndServingShareOneFloor();
    testBootstrapPhaseAdmitsAgainstReserve();
    testCushionScalesWithPhysicalMemory();
    testInvalidCushionRejected();
    testSystemPressureReadsWithoutSampling();
    testObserveStoresPageBreakdown();
    testEstimate_swapFloorWithUnwiredWeights_readsNoHeadroom();
    testEstimate_idleWithUnwiredWeights_chargesThem();
    testEstimate_sameStateWiredOrUnwired_readsTheSame();
    testEstimate_weightCharge_isUnwiredBytesSaturating();
    testGovernor_idleWithHotWeights_staysNormal();
    testObserve_storesWeightResidency();
    testObserve_invalidSample_clearsWeightResidency();
    testWeightWiringSampler_nothingLocked_readsNoWiredBytes();
    testWeightWiringSampler_allLocked_readsAllWired();
    testWeightWiringSampler_oneRangeLocked_readsThatRangeExactly();
    testWeightWiringSampler_largeRangeHalfLocked_readsAboutHalf();
    testWeightWiringSampler_noRanges_readsNothing();
    testQueryHostAvailableMemory_withSampler_reportsWeightResidency();
    std::cout << "memory governor tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "memory governor tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
