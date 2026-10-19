#include "engine/MemoryGovernor.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>

using namespace splash;
using namespace splash::engine;

namespace {

constexpr uint64_t kGiB = 1ULL << 30;
constexpr uint64_t kHostReserveBytes = 64 * 1024;
constexpr uint64_t kLimitBytes = 8 * kGiB;
constexpr uint64_t kResidentBytes = 1 * kGiB;
constexpr uint64_t kAmpleHostBytes = kHostReserveBytes + 3 * kGiB;
constexpr uint64_t kLowHostBytes =
    kHostReserveBytes + kHostWarningMarginBytes / 2;
constexpr uint64_t kHysteresisHostBytes =
    kHostReserveBytes +
    (kHostWarningMarginBytes + kHostRecoveryMarginBytes) / 2;
constexpr int kRepeatedSnapshots = 16;

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

// A host sampler whose answer the test sets and whose calls it counts.
struct FakeHost {
  std::optional<uint64_t> availableBytes = kAmpleHostBytes;
  int samples = 0;
};

MemoryGovernor makeGovernor(FakeHost &host) {
  return MemoryGovernor(
      [](bool) {
        metal::MetalMemoryStats stats;
        stats.allocatedBytes = kResidentBytes;
        return stats;
      },
      kLimitBytes, kHostReserveBytes,
      [&host] {
        ++host.samples;
        return host.availableBytes;
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

  host.availableBytes = kHysteresisHostBytes;
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
    testSystemPressureReadsWithoutSampling();
    std::cout << "memory governor tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "memory governor tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
