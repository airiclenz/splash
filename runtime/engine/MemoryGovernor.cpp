#include "engine/MemoryGovernor.hpp"

#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace splash::engine {

uint64_t estimateHostAvailableMemory(const HostMemoryPages &statistics,
                                     uint64_t pageSize,
                                     uint64_t physicalMemoryBytes) noexcept {
  if (!pageSize || !physicalMemoryBytes) return 0;
  const uint64_t maximum = std::numeric_limits<uint64_t>::max();
  uint64_t usedPages = 0;
  for (uint64_t pages : {statistics.active, statistics.inactive,
                        statistics.speculative, statistics.wired,
                        statistics.compressor}) {
    if (pages > maximum - usedPages) return 0;
    usedPages += pages;
  }
  // Mach's external_page_count excludes wired pages. Unlike free_count,
  // these used-page categories do not already include speculative pages.
  if (statistics.fileBacked > usedPages) return 0;
  usedPages -= statistics.fileBacked;
  if (statistics.purgeable > usedPages) return 0;
  usedPages -= statistics.purgeable;
  if (usedPages > maximum / pageSize) return 0;
  const uint64_t usedBytes = usedPages * pageSize;
  return usedBytes < physicalMemoryBytes ? physicalMemoryBytes - usedBytes : 0;
}

std::optional<HostMemorySample> queryHostAvailableMemory() noexcept {
  // Physical capacity is immutable; don't add a sysctl to every allocation.
  static const uint64_t physicalMemoryBytes = [] {
    uint64_t bytes = 0;
    size_t size = sizeof(bytes);
    return sysctlbyname("hw.memsize", &bytes, &size, nullptr, 0) == 0 &&
                   size == sizeof(bytes) ? bytes : uint64_t{0};
  }();
  if (!physicalMemoryBytes) return std::nullopt;
  mach_port_t host = mach_host_self();
  vm_size_t pageSize = 0;
  vm_statistics64_data_t statistics{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  kern_return_t pageResult = host_page_size(host, &pageSize);
  kern_return_t statisticsResult = pageResult == KERN_SUCCESS
      ? host_statistics64(host, HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&statistics),
                          &count)
      : pageResult;
  mach_port_deallocate(mach_task_self(), host);
  if (statisticsResult != KERN_SUCCESS || !pageSize) return std::nullopt;

  const HostMemoryPages pages{
      .active = statistics.active_count,
      .inactive = statistics.inactive_count,
      .speculative = statistics.speculative_count,
      .wired = statistics.wire_count,
      .compressor = statistics.compressor_page_count,
      .fileBacked = statistics.external_page_count,
      .purgeable = statistics.purgeable_count};
  // Mach page counts are 32-bit, so their byte sizes cannot overflow.
  return HostMemorySample{
      .availableBytes =
          estimateHostAvailableMemory(pages, pageSize, physicalMemoryBytes),
      .inactiveBytes = pages.inactive * pageSize,
      .wiredBytes = pages.wired * pageSize,
      .compressorBytes = pages.compressor * pageSize};
}

std::optional<MemoryPressure> querySystemMemoryPressure() noexcept {
  uint32_t level = 0;
  size_t size = sizeof(level);
  if (sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &size,
                   nullptr, 0) != 0 || size != sizeof(level))
    return std::nullopt;
  switch (level) {
  case DISPATCH_MEMORYPRESSURE_NORMAL: return MemoryPressure::Normal;
  case DISPATCH_MEMORYPRESSURE_WARN: return MemoryPressure::Warning;
  case DISPATCH_MEMORYPRESSURE_CRITICAL: return MemoryPressure::Critical;
  default: return std::nullopt;
  }
}

MemoryGovernor::Reservation::Reservation(MemoryGovernor *owner, uint64_t bytes)
    : owner_(owner), bytes_(bytes) {}

MemoryGovernor::Reservation::~Reservation() { release(); }

MemoryGovernor::Reservation::Reservation(Reservation &&other) noexcept
    : owner_(other.owner_), bytes_(other.bytes_) {
  other.owner_ = nullptr;
  other.bytes_ = 0;
}

MemoryGovernor::Reservation &
MemoryGovernor::Reservation::operator=(Reservation &&other) noexcept {
  if (this == &other)
    return *this;
  release();
  owner_ = other.owner_;
  bytes_ = other.bytes_;
  other.owner_ = nullptr;
  other.bytes_ = 0;
  return *this;
}

MemoryGovernor::Reservation::operator bool() const noexcept {
  return owner_ && bytes_;
}

void MemoryGovernor::Reservation::commit() { release(); }

void MemoryGovernor::Reservation::release() noexcept {
  if (owner_ && bytes_)
    owner_->release(bytes_);
  owner_ = nullptr;
  bytes_ = 0;
}

MemoryGovernor::MemoryGovernor(
    MetalMemoryStatsProvider metalMemoryStats, uint64_t limitBytes,
    HostCushion hostCushion,
    HostAvailableMemoryProvider hostAvailableMemory)
    : metalMemoryStats_(std::move(metalMemoryStats)), limitBytes_(limitBytes),
      hostCushion_(hostCushion),
      hostAvailableMemory_(std::move(hostAvailableMemory)) {
  if (!metalMemoryStats_) {
    throw std::invalid_argument("Metal memory-stats provider must be present");
  }
  if (!limitBytes_) {
    throw std::invalid_argument("memory governor limit must be positive");
  }
  if (!hostCushion_.reserveBytes) {
    throw std::invalid_argument("host memory reserve must be positive");
  }
  // A zero warning margin would admit growth with no host telemetry at all.
  if (!hostCushion_.warningMarginBytes) {
    throw std::invalid_argument("host warning margin must be positive");
  }
  if (hostCushion_.recoveryMarginBytes < hostCushion_.warningMarginBytes) {
    throw std::invalid_argument(
        "host recovery margin must not be below the warning margin");
  }
  if (!hostAvailableMemory_) {
    throw std::invalid_argument(
        "host available-memory provider must be present");
  }
  if (observedResidentBytes(true) > limitBytes_) {
    throw metal::MetalAllocationError(
        "existing Metal allocations exceed memory governor limit",
        metal::AllocationFailure::EngineBudget);
  }
  std::optional<HostMemorySample> hostSample = hostAvailableMemory_();
  if (!hostSample || hostSample->availableBytes <= hostCushion_.reserveBytes) {
    throw metal::MetalAllocationError(
        "host available memory does not satisfy the system reserve",
        metal::AllocationFailure::HostPressure);
  }
  // Without a stored sample, a snapshot() before the first observe() would
  // report host growth paused.
  static_cast<void>(updateEffectivePressure(hostSample, reservedBytes_));
}

uint64_t
MemoryGovernor::observedResidentBytes(bool refreshDevice) const noexcept {
  metal::MetalMemoryStats memory = metalMemoryStats_(refreshDevice);
  uint64_t accounted = memory.allocatedBytes;
  if (memory.sparseResidentBytes <=
      std::numeric_limits<uint64_t>::max() - accounted) {
    accounted += memory.sparseResidentBytes;
  } else {
    accounted = std::numeric_limits<uint64_t>::max();
  }
  return std::max(accounted, memory.deviceCurrentAllocatedBytes);
}

std::optional<HostMemorySample>
MemoryGovernor::sampleHostAvailable() const noexcept {
  try {
    return hostAvailableMemory_();
  } catch (...) {
    return std::nullopt;
  }
}

uint64_t MemoryGovernor::hostHeadroomBytes(
    const std::optional<HostMemorySample> &hostSample,
    uint64_t reservedBytes) const noexcept {
  if (!hostSample || hostSample->availableBytes <= hostCushion_.reserveBytes)
    return 0;
  const uint64_t availableAfterReserve =
      hostSample->availableBytes - hostCushion_.reserveBytes;
  return reservedBytes < availableAfterReserve
      ? availableAfterReserve - reservedBytes
      : 0;
}

bool MemoryGovernor::hostAdmits(uint64_t hostHeadroom,
                                MemoryPressure pressure) const noexcept {
  if (pressure == MemoryPressure::Critical)
    return false;
  // The headroom is already net of the reserve: while bootstrapping any of
  // it left after the growth keeps the host above the load guard's floor.
  return isBootstrapping_ ? hostHeadroom > 0
                          : hostHeadroom >= hostCushion_.warningMarginBytes;
}

std::optional<MemoryGovernor::Reservation>
MemoryGovernor::tryReserve(uint64_t bytes, metal::AllocationFailure *failure) {
  if (failure)
    *failure = metal::AllocationFailure::None;
  if (!bytes) {
    throw std::invalid_argument("memory reservation must be positive");
  }
  std::lock_guard lock(mutex_);
  uint64_t observed = observedResidentBytes(true);
  bool overflows =
      reservedBytes_ > std::numeric_limits<uint64_t>::max() - bytes;
  uint64_t requested =
      overflows ? std::numeric_limits<uint64_t>::max() : reservedBytes_ + bytes;
  std::optional<HostMemorySample> hostSample = sampleHostAvailable();
  MemoryPressure pressure = updateEffectivePressure(
      hostSample, reservedBytes_);
  bool engineFits = !overflows && observed <= limitBytes_ &&
                    requested <= limitBytes_ - observed;
  if (!engineFits ||
      !hostAdmits(hostHeadroomBytes(hostSample, requested), pressure)) {
    if (failure)
      *failure = !engineFits ? metal::AllocationFailure::EngineBudget
                            : metal::AllocationFailure::HostPressure;
    if (deniedReservations_ != std::numeric_limits<uint64_t>::max()) {
      ++deniedReservations_;
    }
    return std::nullopt;
  }
  reservedBytes_ = requested;
  return Reservation(this, bytes);
}

metal::AllocationAdmission MemoryGovernor::allocationAdmission() noexcept {
  return [this](uint64_t bytes, const std::function<void()> &allocate)
             -> metal::AllocationResult {
    metal::AllocationFailure failure;
    auto reservation = tryReserve(bytes, &failure);
    if (!reservation)
      return failure;
    try {
      allocate();
    } catch (const metal::MetalAllocationError &error) {
      // Host headroom is an estimate; the driver can still deny placement.
      return error.failure();
    }
    reservation->commit();
    return true;
  };
}

void MemoryGovernor::setPressure(MemoryPressure pressure) noexcept {
  std::lock_guard lock(mutex_);
  systemPressure_ = pressure;
}

void MemoryGovernor::setBootstrapping(bool bootstrapping) noexcept {
  std::lock_guard lock(mutex_);
  isBootstrapping_ = bootstrapping;
}

void MemoryGovernor::observe() noexcept {
  std::lock_guard lock(mutex_);
  static_cast<void>(
      updateEffectivePressure(sampleHostAvailable(), reservedBytes_));
}

MemoryGovernorSnapshot MemoryGovernor::snapshot() const noexcept {
  std::lock_guard lock(mutex_);
  uint64_t observed = observedResidentBytes();
  uint64_t used = observed;
  if (reservedBytes_ <= std::numeric_limits<uint64_t>::max() - used) {
    used += reservedBytes_;
  } else {
    used = std::numeric_limits<uint64_t>::max();
  }
  uint64_t hostHeadroom = hostHeadroomBytes(sampledHost_, reservedBytes_);
  const HostMemorySample host = sampledHost_.value_or(HostMemorySample{});
  MemoryPressure pressure = effectivePressure();
  bool hostGrowthAllowed = hostAdmits(hostHeadroom, pressure);
  bool growthAllowed = hostGrowthAllowed && used < limitBytes_;
  return {
      limitBytes_,
      observed,
      reservedBytes_,
      used < limitBytes_ ? limitBytes_ - used : 0,
      pressure,
      deniedReservations_,
      sampledHost_.has_value(),
      host.availableBytes,
      host.inactiveBytes,
      host.wiredBytes,
      host.compressorBytes,
      hostCushion_.reserveBytes,
      hostCushion_.warningMarginBytes,
      hostCushion_.recoveryMarginBytes,
      hostHeadroom,
      systemPressure_,
      growthAllowed,
      hostGrowthAllowed,
  };
}

MemoryPressure MemoryGovernor::updateEffectivePressure(
    const std::optional<HostMemorySample> &hostSample,
    uint64_t reservedBytes) noexcept {
  sampledHost_ = hostSample;
  uint64_t hostHeadroom = hostHeadroomBytes(hostSample, reservedBytes);

  // The latch enters below the warning margin and holds until the recovery
  // margin, so paced reclaim runs a whole episode. It drives reported
  // pressure and reclaim only; admission reads the headroom itself. Only the
  // system's critical signal warrants dropping every evictable cache entry.
  hostConstrained_ = !hostSample ||
      hostHeadroom < hostCushion_.warningMarginBytes ||
      (hostConstrained_ && hostHeadroom < hostCushion_.recoveryMarginBytes);
  return effectivePressure();
}

MemoryPressure MemoryGovernor::effectivePressure() const noexcept {
  if (systemPressure_ == MemoryPressure::Critical)
    return MemoryPressure::Critical;
  if (hostConstrained_ || systemPressure_ == MemoryPressure::Warning)
    return MemoryPressure::Warning;
  return MemoryPressure::Normal;
}

void MemoryGovernor::release(uint64_t bytes) noexcept {
  std::lock_guard lock(mutex_);
  if (bytes > reservedBytes_) {
    reservedBytes_ = 0;
    return;
  }
  reservedBytes_ -= bytes;
}

MemoryReclaimDirective MemoryPressurePolicy::update(
    const MemoryGovernorSnapshot &snapshot, double nowMilliseconds,
    bool requestWaiting) noexcept {
  if (snapshot.pressure == MemoryPressure::Normal) {
    nextReclaimMilliseconds_ = 0.0;
    return {};
  }
  if (snapshot.pressure == MemoryPressure::Critical) {
    return {true, true, std::numeric_limits<uint64_t>::max()};
  }
  if (nowMilliseconds < nextReclaimMilliseconds_)
    return {true, false, 0};
  // The host samples every 500 ms. Allow counters to settle between batches,
  // but keep responding if another application continues consuming memory.
  nextReclaimMilliseconds_ = nowMilliseconds + 1000.0;

  // Missing telemetry pauses allocation, but is not evidence that live
  // cache must be discarded. Empty backing can still be returned.
  if (!snapshot.hostMeasurementValid &&
      snapshot.systemPressure == MemoryPressure::Normal)
    return {true, false, 0};

  uint64_t desired =
      snapshot.hostHeadroomBytes < snapshot.hostRecoveryMarginBytes
      ? snapshot.hostRecoveryMarginBytes - snapshot.hostHeadroomBytes
      : 0;
  // Recovering the last stretch to the recovery margin is worth far less
  // than the resume point it would otherwise discard, so a pass with nothing
  // waiting keeps that publication and takes the rest. A waiting request
  // outranks it.
  return {true, false, std::min(desired, snapshot.hostWarningMarginBytes),
          !requestWaiting};
}

} // namespace splash::engine
