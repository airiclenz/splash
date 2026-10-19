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

// requireStartupHeadroom refuses at or below admissionFloorBytes(); a one-byte
// reservation must meet the same floor, and the snapshot must agree with it.
void testStartupAndServingShareOneFloor() {
  FakeHost host;
  MemoryGovernor governor = makeGovernor(host);

  host.availableBytes = kCushion.admissionFloorBytes();
  metal::AllocationFailure failure = metal::AllocationFailure::None;
  const bool grantedAtFloor = governor.tryReserve(1, &failure).has_value();
  const bool reportedAtFloor = governor.snapshot().hostGrowthAllowed;
  host.availableBytes = kCushion.admissionFloorBytes() + 1;
  const bool grantedAboveFloor = governor.tryReserve(1).has_value();
  host.availableBytes = kCushion.admissionFloorBytes() - 1;
  governor.observe();
  const bool reportedBelowFloor = governor.snapshot().hostGrowthAllowed;

  require(!grantedAtFloor &&
              failure == metal::AllocationFailure::HostPressure,
          "serving admitted growth at the startup floor");
  require(grantedAboveFloor, "serving refused growth above the startup floor");
  require(reportedAtFloor && !reportedBelowFloor,
          "the snapshot's host test differs from the reservation's");
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
    testReservationThatFitsIsGrantedWhileWarningLatched();
    testStartupAndServingShareOneFloor();
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
