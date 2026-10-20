#include "engine/MemoryGovernor.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>

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

// A host sampler whose answer the test sets and whose calls it counts.
struct FakeHost {
  std::optional<uint64_t> availableBytes = kAmpleHostBytes;
  uint64_t inactiveBytes = kInactiveHostBytes;
  uint64_t wiredBytes = kWiredHostBytes;
  uint64_t compressorBytes = kCompressorHostBytes;
  int samples = 0;
};

MemoryGovernor makeGovernor(FakeHost &host,
                            HostCushion cushion = kCushion) {
  return MemoryGovernor(
      [](bool) {
        metal::MetalMemoryStats stats;
        stats.allocatedBytes = kResidentBytes;
        return stats;
      },
      kLimitBytes, cushion,
      [&host]() -> std::optional<HostMemorySample> {
        ++host.samples;
        if (!host.availableBytes)
          return std::nullopt;
        return HostMemorySample{*host.availableBytes, host.inactiveBytes,
                                host.wiredBytes, host.compressorBytes};
      });
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
    std::cout << "memory governor tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "memory governor tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
