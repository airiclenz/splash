#include "engine/MemoryGovernor.hpp"
#include "model/QwenState.hpp"
#include "ops/PageStorage.hpp"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

using namespace splash;
using namespace splash::engine;

namespace {

constexpr model::GdnStateLayout kTargetState{48, 3, 10'240, 48, 128, 128};
constexpr model::DraftStateLayout kDraftState{5, 8, 2'048, 128};
constexpr model::CompositeStateLayout kStateLayout{kTargetState, kDraftState};

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename Exception = std::exception>
void requireThrows(const std::function<void()> &operation,
                   const char *message) {
  try {
    operation();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}

uint32_t &word(const metal::MetalBuffer &buffer, uint64_t byteOffset = 0) {
  require(byteOffset + sizeof(uint32_t) <= buffer.sizeBytes(),
          "test marker is outside buffer");
  auto *bytes = static_cast<uint8_t *>(buffer.contents());
  require(bytes != nullptr, "test buffer is not CPU-visible");
  return *reinterpret_cast<uint32_t *>(bytes + byteOffset);
}

void testLayoutFormulas() {
  require(kTargetState.convolutionLayerBytes() == 65'536,
          "GDN convolution layer formula is wrong");
  require(kTargetState.convolutionBytes() == 3'145'728,
          "GDN convolution parity formula is wrong");
  require(kTargetState.recurrentLayerBytes() == 3'145'728,
          "GDN recurrent layer formula is wrong");
  require(kTargetState.recurrentBytes() == 150'994'944,
          "GDN recurrent parity formula is wrong");
  require(kDraftState.tensorBytes() == 4'194'304,
          "draft tensor formula is wrong");
  require(kDraftState.ringBytes() == 41'943'040,
          "draft state formula is wrong");
  require(kStateLayout.activeCellBytes() == 350'224'384,
          "per-slot byte formula is wrong");
  require(uint64_t{model::ExecutionLimits::maximumBatchWidth} *
                  kStateLayout.activeCellBytes() ==
              1'400'897'536,
          "four-slot byte formula is wrong");
  require(kStateLayout.cachedBytes() == 196'083'712,
          "prefix byte formula is wrong");
}

// A lane without a complete draft window yields a target-only state: its
// active GDN cell moves out uncopied and unadmitted, and restores into another
// lane with that lane's draft window reset. Snapshots stay strict.
void testTargetOnlyStates(metal::MetalBackend &backend,
                          const metal::AllocationAdmission &admitState,
                          bool &admitNewAllocations) {
  using model::QwenLogicalLengths;
  using model::QwenStateStorage;

  QwenStateStorage storage(backend, admitState, kStateLayout);
  require(storage.tryActivateSlot(0, 501) && storage.tryActivateSlot(1, 502) &&
              storage.tryActivateSlot(2, 503),
          "target-only fixture lanes were not activated");
  const auto &source = storage.buffers(0);
  const auto &destination = storage.buffers(1);
  const QwenLogicalLengths partial{4'096, 2'048, 1'024, 1'024};
  storage.updateLengths(0, partial);
  storage.swapParity(0);
  word(source.gdn[1].convolutionBase) = 0x61616161;
  word(source.gdn[1].recurrentBase) = 0x62626262;
  word(destination.draft[0].keys) = 0x63636363;
  requireThrows<std::invalid_argument>(
      [&] { static_cast<void>(storage.snapshot(0)); },
      "snapshot accepted an incomplete draft window");

  const uint64_t backendBefore = backend.memoryStats().allocatedBytes;
  const uint64_t storageBefore = storage.actualAllocatedBytes();
  const uint32_t idleBefore = storage.idleCells();
  admitNewAllocations = false;
  const auto targetOnly = storage.moveState(0);
  admitNewAllocations = true;
  require(targetOnly && !targetOnly->carriesDraft() &&
              targetOnly->bytes() == kTargetState.cellBytes(),
          "incomplete draft window did not yield a target-only state");
  require(backend.memoryStats().allocatedBytes == backendBefore &&
              storage.actualAllocatedBytes() == storageBefore &&
              storage.idleCells() == idleBefore,
          "moving a lane's state allocated or took a buffer");
  require(source.gdn[1].stateBase.contents() == nullptr &&
              source.gdn[0].stateBase.contents() != nullptr &&
              source.draft.size() == kDraftState.layers,
          "move did not take exactly the active GDN cell");

  requireThrows<std::invalid_argument>(
      [&] { storage.restore(1, *targetOnly, true); },
      "target-only state restored a draft it does not carry");
  storage.restore(1, *targetOnly, false);
  require(word(destination.gdn[0].convolutionBase) == 0x61616161 &&
              word(destination.gdn[0].recurrentBase) == 0x62626262 &&
              word(destination.draft[0].keys) == 0x63636363,
          "target-only restore did not deliver only the moved GDN state");
  require(storage.metadata(1).lengths ==
              QwenLogicalLengths{4'096, 4'096, 0, 0},
          "target-only restore kept draft metadata it does not carry");
  storage.releaseSlot(0, 501);

  for (const QwenLogicalLengths lengths :
       {QwenLogicalLengths{100, 0, 50, 50}, QwenLogicalLengths{0, 0, 0, 1}}) {
    storage.updateLengths(2, lengths);
    const auto unaligned = storage.moveState(2);
    require(!unaligned->carriesDraft(), "short lane state carried draft");
    requireThrows<std::invalid_argument>(
        [&] { storage.restore(1, *unaligned, false); },
        "target-only restore accepted zero or unaligned target tokens");
    storage.releaseSlot(2, 503);
    require(static_cast<bool>(storage.tryActivateSlot(2, 503)),
            "lane did not reactivate after a moved state");
  }

  // A complete window moves the ring too and restores as a full state.
  require(static_cast<bool>(storage.tryActivateSlot(0, 504)),
          "lane did not reactivate after a moved state");
  const QwenLogicalLengths complete{2'048, 0, 2'048, 0};
  storage.updateLengths(0, complete);
  const auto full = storage.moveState(0);
  require(full->carriesDraft() && full->bytes() == kStateLayout.cachedBytes() &&
              source.draft.empty(),
          "complete draft window did not move its ring");
  storage.restore(1, *full, true);
  require(storage.metadata(1).lengths == complete,
          "moved full state did not restore its draft window");
  storage.releaseSlot(0, 504);
}

// Runtime::suspendWithState's sequence: the live state moves out, then the
// slot and the idle pool are released. The moved cell survives both without
// a copy or an admitted allocation, restores at the lane's length, and parks
// in the pool only once the state is dropped.
void testSuspensionKeepsTheMovedCell(metal::MetalBackend &backend,
                                     const metal::AllocationAdmission &admitState,
                                     bool &admitNewAllocations) {
  using model::QwenLogicalLengths;
  using model::QwenStateStorage;

  QwenStateStorage storage(backend, admitState, kStateLayout);
  require(storage.tryActivateSlot(0, 601) && storage.tryActivateSlot(1, 602),
          "suspension fixture lanes were not activated");
  const QwenLogicalLengths boundary{2'048, 0, 2'048, 0};
  storage.updateLengths(0, boundary);
  const auto &source = storage.buffers(0);
  const uint32_t active = storage.metadata(0).activeParity;
  word(source.gdn[active].convolutionBase) = 0x71717171;
  word(source.gdn[active].recurrentBase) = 0x72727272;
  const uint64_t backendBefore = backend.memoryStats().allocatedBytes;

  admitNewAllocations = false;
  auto moved = storage.moveState(0);
  storage.releaseSlot(0, 601);
  static_cast<void>(storage.releaseIdle(0, 0));
  admitNewAllocations = true;

  require(moved && moved->carriesDraft() && storage.idleCells() == 0 &&
              backend.memoryStats().allocatedBytes <= backendBefore,
          "suspension copied or allocated the moved state");
  storage.restore(1, *moved, true);
  const auto &destination = storage.buffers(1);
  const uint32_t restored = storage.metadata(1).activeParity;
  require(word(destination.gdn[restored].convolutionBase) == 0x71717171 &&
              word(destination.gdn[restored].recurrentBase) == 0x72727272 &&
              storage.metadata(1).lengths == boundary,
          "moved state lost its cell or its length across the release");
  moved.reset();
  require(storage.idleCells() == 1,
          "a dropped moved state did not park its cell in the pool");
  storage.releaseSlot(1, 602);
}

void run(const std::string &metallib) {
  using model::QwenCompositeState;
  using model::QwenLogicalLengths;
  using model::QwenStateStorage;

  testLayoutFormulas();
  metal::MetalBackend backend(metallib);
  MemoryGovernor governor(
      backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1);
  // Switched off to prove that a reserved or pooled cache slot needs no new
  // admission.
  bool admitNewAllocations = true;
  auto admitState = [&governor, &admitNewAllocations](
                        uint64_t bytes,
                        const std::function<void()> &allocate) {
    if (!admitNewAllocations)
      return false;
    auto reservation = governor.tryReserve(bytes);
    if (!reservation)
      return false;
    allocate();
    reservation->commit();
    return true;
  };
  constexpr kv::Layout kvLayout{16, 4, 256};
  kv::PageStorage pageStorage(backend, governor.allocationAdmission(),
                                kvLayout,
                                kvLayout.sparseMappingBatchPages());
  uint64_t beforeStorage = backend.memoryStats().allocatedBytes;
  uint64_t observedStorageActual = 0;
  uint64_t observedSlotActual = 0;
  uint64_t observedPrefixActual = 0;

  {
    QwenStateStorage storage(backend, admitState, kStateLayout);
    observedStorageActual = storage.actualAllocatedBytes();
    observedSlotActual = storage.actualSlotBytes(0);
    require(storage.actualAllocatedBytes() == 0 &&
                backend.memoryStats().allocatedBytes == beforeStorage,
            "state cells were allocated eagerly");
    for (uint32_t slot = 0;
         slot < model::ExecutionLimits::maximumBatchWidth;
         ++slot) {
      require(!storage.metadata(slot).assigned, "slot was assigned eagerly");
      require(storage.actualSlotBytes(slot) == 0,
              "idle slot has physical backing before activation");
    }
    requireThrows<std::out_of_range>(
        [&] { static_cast<void>(storage.buffers(4)); },
        "storage exposed more than four slots");

    require(storage.tryActivateSlot(0, 101) && storage.tryActivateSlot(1, 202),
            "state cell activation failed");
    observedStorageActual = storage.actualAllocatedBytes();
    observedSlotActual = storage.actualSlotBytes(0);
    require(observedSlotActual >=
                kStateLayout.activeCellBytes() + kStateLayout.cachedBytes(),
            "activated slot allocation is below declared lane and reserve bytes");
    require(storage.releaseIdle(0, 0) == 0 &&
                storage.actualSlotBytes(0) == observedSlotActual,
            "idle reclaim took a live lane's reserve");
    require(storage.actualSlotBytes(1) == observedSlotActual &&
                storage.actualAllocatedBytes() == 2 * observedSlotActual,
            "activated slot accounting is not incremental");

    const auto &slot0 = storage.buffers(0);
    const auto &slot1 = storage.buffers(1);
    void *stableGdnBase = slot0.gdn[0].stateBase.contents();
    void *stableConvBase = slot0.gdn[0].convolutionBase.contents();
    void *stableDraftBase = slot0.draft[0].keys.contents();
    require(stableGdnBase && stableConvBase && stableDraftBase,
            "stable slot buffers are not CPU-visible");
    require(stableGdnBase != slot1.gdn[0].stateBase.contents(),
            "two slots alias one GDN allocation");
    require(slot0.gdn[0].convolutionLayers[1].contents() ==
                static_cast<uint8_t *>(stableConvBase) +
                    kTargetState.convolutionLayerBytes(),
            "GDN layer view offset is wrong");
    require(slot0.gdn[0].stateBase.sizeBytes() ==
                    kTargetState.cellBytes() &&
                slot0.draft.size() == kDraftState.layers &&
                slot0.draft[0].keys.sizeBytes() == kDraftState.tensorBytes(),
            "split state-buffer sizes are wrong");

    require(storage.metadata(0).assigned &&
                storage.metadata(0).requestId == 101 &&
                storage.metadata(0).activeParity == 0,
            "slot activation metadata is wrong");
    requireThrows<std::logic_error>(
        [&] { static_cast<void>(storage.tryActivateSlot(0, 303)); },
        "double slot activation was accepted");

    // Hot metadata updates must leave every buffer and marker untouched.
    word(slot0.gdn[0].convolutionBase) = 0x10101010;
    word(slot0.gdn[1].convolutionBase) = 0x21212121;
    word(slot0.gdn[1].recurrentBase) = 0x31313131;
    word(slot0.draft[0].keys) = 0x41414141;
    word(slot0.draft[4].values,
         kDraftState.tensorBytes() - sizeof(uint32_t)) = 0x51515151;
    QwenLogicalLengths lengths{2'048, 0, 2'048, 0};
    storage.updateLengths(0, lengths);
    require(storage.metadata(0).lengths.draftCommitCursor == 0,
            "draft ring cursor is wrong");
    require(storage.metadata(0).lengths.draftLength == 2'048,
            "draft resident length is wrong");
    require(word(slot0.gdn[1].convolutionBase) == 0x21212121 &&
                word(slot0.draft[0].keys) == 0x41414141,
            "length update copied or cleared hot state");
    storage.swapParity(0);
    require(storage.metadata(0).activeParity == 1,
            "parity swap did not select parity one");
    require(word(slot0.gdn[0].convolutionBase) == 0x10101010 &&
                word(slot0.gdn[1].convolutionBase) == 0x21212121,
            "parity swap copied hot state");

    // Publication copies the lane's active-parity GDN cell and its draft
    // ring into the cache slot the lane reserved at admission, so a denying
    // governor cannot refuse the first checkpoint. The lane keeps its own
    // cells: no address changes, no aliasing, no parity handoff.
    const uint64_t beforePrefix = backend.memoryStats().allocatedBytes;
    const uint64_t storageBeforePrefix = storage.actualAllocatedBytes();
    void *laneActiveGdnBase = slot0.gdn[1].stateBase.contents();
    admitNewAllocations = false;
    std::shared_ptr<const QwenCompositeState> prefix = storage.snapshot(0);
    admitNewAllocations = true;
    require(prefix != nullptr,
            "first snapshot under a denying governor did not use the reserve");
    observedPrefixActual = observedSlotActual - storage.actualSlotBytes(0);
    require(prefix->bytes() == kStateLayout.cachedBytes(),
            "cached state footprint is not the declared cached bytes");
    require(observedPrefixActual >= kStateLayout.cachedBytes() &&
                backend.memoryStats().allocatedBytes == beforePrefix &&
                storage.actualAllocatedBytes() == storageBeforePrefix,
            "reserved cache slot is below declared bytes or was reallocated");
    require(slot0.gdn[1].stateBase.contents() == laneActiveGdnBase &&
                slot0.gdn[0].stateBase.contents() == stableGdnBase &&
                slot0.draft[0].keys.contents() == stableDraftBase,
            "snapshot moved or aliased the lane's own cells");
    require(storage.metadata(0).activeParity == 1 &&
                storage.metadata(0).lengths == lengths,
            "snapshot changed the lane's metadata");

    // The cached copy is independent of the lane: writes to the lane's
    // active cell or draft ring after publication never reach a restore.
    word(slot0.gdn[1].convolutionBase) = 0xa1a1a1a1;
    word(slot0.gdn[1].recurrentBase) = 0xa2a2a2a2;
    word(slot0.draft[0].keys) = 0xa3a3a3a3;
    word(slot1.gdn[0].convolutionBase) = 0xb0b0b0b0;
    word(slot1.gdn[1].convolutionBase) = 0xb1b1b1b1;
    word(slot1.draft[0].keys) = 0xb2b2b2b2;
    void *destinationGdnBase = slot1.gdn[0].stateBase.contents();
    void *destinationDraftBase = slot1.draft[0].keys.contents();
    storage.restore(1, *prefix, true);

    require(storage.metadata(1).requestId == 202 &&
                storage.metadata(1).activeParity == 0 &&
                storage.metadata(1).lengths == lengths,
            "restore lost owner, changed parity, or lengths");
    require(slot1.gdn[0].stateBase.contents() == destinationGdnBase &&
                slot1.draft[0].keys.contents() == destinationDraftBase,
            "restore replaced the destination's buffers");
    require(word(slot1.gdn[0].convolutionBase) == 0x21212121 &&
                word(slot1.gdn[0].recurrentBase) == 0x31313131,
            "restore did not deliver the pre-mutation GDN snapshot");
    require(word(slot1.gdn[1].convolutionBase) == 0xb1b1b1b1,
            "restore overwrote inactive parity");
    require(
        word(slot1.draft[0].keys) == 0x41414141 &&
            word(slot1.draft[4].values, kDraftState.tensorBytes() -
                                            sizeof(uint32_t)) == 0x51515151,
        "restore did not deliver the pre-mutation draft ring snapshot");
    require(word(slot0.gdn[1].convolutionBase) == 0xa1a1a1a1 &&
                word(slot0.gdn[1].recurrentBase) == 0xa2a2a2a2 &&
                word(slot0.draft[0].keys) == 0xa3a3a3a3,
            "restore wrote back into the source lane");

    // A suffix that will rebuild a full 2048-token window restores only GDN.
    // A cached draft ring must not consume a 40 MiB copy merely to be
    // overwritten by the next prefill commands.
    storage.swapParity(1);
    word(slot1.draft[0].keys) = 0xd2d2d2d2;
    storage.restore(1, *prefix, false);
    require(word(slot1.gdn[storage.metadata(1).activeParity].convolutionBase) ==
                0x21212121,
            "GDN-only restore did not restore convolution state");
    require(word(slot1.draft[0].keys) == 0xd2d2d2d2,
            "GDN-only restore copied an obsolete draft ring");
    require(storage.metadata(1).lengths.targetTokens == 2048 &&
                storage.metadata(1).lengths.draftLength == 0 &&
                storage.metadata(1).lengths.draftBase == 2048,
            "GDN-only restore exposed stale draft metadata");
    // Cancellation releases ownership and returns the lane's buffers to the
    // pool. Reactivation takes them back in the same order and initializes
    // every state that can be read at logical length zero.
    void *reusableGdnBase = slot0.gdn[0].stateBase.contents();
    void *reusableDraftBase = slot0.draft[0].keys.contents();
    word(slot0.gdn[0].convolutionBase) = 0xc1c1c1c1;
    word(slot0.gdn[0].recurrentBase) = 0xc2c2c2c2;
    // The published prefix holds the spent reserve, so only the lane's own
    // cells and ring return.
    storage.releaseSlot(0, 101);
    require(!storage.metadata(0).assigned && storage.metadata(0).requestId == 0,
            "cancellation did not release metadata");
    require(storage.idleCells() == 2 && storage.idleRings() == 1 &&
                storage.actualSlotBytes(0) == 0,
            "released lane buffers did not return to the pool");
    requireThrows<std::logic_error>([&] { storage.swapParity(0); },
                                    "unassigned slot accepted a parity update");
    require(static_cast<bool>(storage.tryActivateSlot(0, 303)), "state cell reuse failed");
    require(storage.idleCells() == 0 && storage.idleRings() == 0 &&
                storage.actualSlotBytes(0) == observedSlotActual,
            "reactivation left pooled buffers behind or no fresh reserve");
    require(slot0.gdn[0].stateBase.contents() == reusableGdnBase &&
                slot0.draft[0].keys.contents() == reusableDraftBase,
            "slot reuse changed stable buffer addresses");
    require(storage.metadata(0).requestId == 303 &&
                storage.metadata(0).activeParity == 0 &&
                storage.metadata(0).lengths == QwenLogicalLengths{},
            "slot reuse did not reset logical state");
    require(word(slot0.gdn[0].convolutionBase) == 0 &&
                word(slot0.gdn[0].recurrentBase) == 0,
            "slot reuse did not initialize readable state");

    // Rejected publications fail before any cache slot is taken or admitted.
    const uint64_t beforeRejected = storage.actualAllocatedBytes();
    storage.updateLengths(0, {128, 0, 127, 127});
    requireThrows<std::invalid_argument>(
        [&] { static_cast<void>(storage.snapshot(0)); },
        "snapshot accepted divergent target/draft lengths");
    requireThrows<std::invalid_argument>(
        [&] {
          storage.updateLengths(0, {129, 0, 129, 129});
          static_cast<void>(storage.snapshot(0));
        },
        "unaligned prefix snapshot was accepted");
    require(storage.actualAllocatedBytes() == beforeRejected,
            "rejected snapshot allocated or consumed a cache slot");
    requireThrows<std::logic_error>([&] { storage.releaseSlot(0, 404); },
                                    "slot release accepted the wrong owner");

    // Slot release returns the unspent reserve with the lane's buffers.
    const uint64_t beforeSuspend = storage.actualAllocatedBytes();
    const uint64_t releasedSlotBytes = storage.actualSlotBytes(0);
    storage.releaseSlot(0, 303);
    require(releasedSlotBytes == observedSlotActual &&
                storage.idleCells() == 3 && storage.idleRings() == 2,
            "slot release kept its reserve out of the pool");
    require(storage.releaseIdle(0, 0) == releasedSlotBytes,
            "recomputation preemption retained active backing");
    require(!storage.metadata(0).assigned && storage.actualSlotBytes(0) == 0 &&
                storage.actualAllocatedBytes() == beforeSuspend - releasedSlotBytes,
            "preempted GDN or draft bytes remain outside the cache");
    require(storage.tryActivateSlot(0, 303) &&
                storage.metadata(0).assigned &&
                storage.metadata(0).lengths == QwenLogicalLengths{} &&
                word(slot0.gdn[0].convolutionBase) == 0 &&
                word(slot0.gdn[0].recurrentBase) == 0,
            "recomputation did not start from a fresh empty state");

    // Dropping a cached state returns its buffers to the storage's pool rather
    // than freeing them: accounting stays flat. A publication takes the lane's
    // reserve first, then the pooled buffers, both without a governor
    // admission; only then does the governor decide. Only releaseIdle returns
    // pooled bytes to macOS.
    const uint64_t beforeDrop = storage.actualAllocatedBytes();
    const uint64_t backendBeforeDrop = backend.memoryStats().allocatedBytes;
    prefix.reset();
    require(storage.actualAllocatedBytes() == beforeDrop &&
                backend.memoryStats().allocatedBytes == backendBeforeDrop &&
                storage.idleCells() == 1 && storage.idleRings() == 1,
            "dropped cached state freed its slot instead of pooling it");
    storage.updateLengths(0, lengths);
    word(slot0.gdn[0].convolutionBase) = 0xe1e1e1e1;
    word(slot0.gdn[0].recurrentBase) = 0xe2e2e2e2;
    word(slot0.draft[0].keys) = 0xe3e3e3e3;
    admitNewAllocations = false;
    std::shared_ptr<const QwenCompositeState> reserved = storage.snapshot(0);
    require(reserved != nullptr && storage.idleCells() == 1 &&
                storage.idleRings() == 1 &&
                storage.actualSlotBytes(0) ==
                    observedSlotActual - observedPrefixActual,
            "snapshot did not take the lane's reserve before the pool");
    std::shared_ptr<const QwenCompositeState> pooled = storage.snapshot(0);
    require(pooled != nullptr, "snapshot did not reuse the pooled cache slot");
    require(pooled->bytes() == kStateLayout.cachedBytes() &&
                storage.idleCells() == 0 && storage.idleRings() == 0 &&
                storage.actualAllocatedBytes() == beforeDrop &&
                backend.memoryStats().allocatedBytes == backendBeforeDrop,
            "pooled cache slot reuse allocated new buffers");
    require(storage.snapshot(0) == nullptr,
            "snapshot with a spent reserve and an empty pool bypassed the governor");
    require(storage.actualAllocatedBytes() == beforeDrop,
            "denied snapshot leaked cache slot bytes");
    admitNewAllocations = true;
    std::shared_ptr<const QwenCompositeState> fresh = storage.snapshot(0);
    require(fresh != nullptr &&
                backend.memoryStats().allocatedBytes ==
                    backendBeforeDrop + observedPrefixActual &&
                storage.actualAllocatedBytes() ==
                    beforeDrop + observedPrefixActual,
            "admitted cache slot allocation is unaccounted");
    fresh.reset();
    storage.restore(1, *reserved, true);
    require(storage.metadata(1).activeParity == 1 &&
                storage.metadata(1).lengths == lengths &&
                word(slot1.gdn[1].convolutionBase) == 0xe1e1e1e1 &&
                word(slot1.gdn[1].recurrentBase) == 0xe2e2e2e2 &&
                word(slot1.draft[0].keys) == 0xe3e3e3e3,
            "reserved cache slot served stale contents");
    require(word(slot1.gdn[0].convolutionBase) == 0x21212121,
            "restore from the reserved slot overwrote inactive parity");

    // An evicted state's buffers reach the pool even while its lane's reserve
    // is short: reclaim credits those bytes, so idle reclaim must free them.
    pooled.reset();
    require(storage.actualSlotBytes(0) ==
                    observedSlotActual - observedPrefixActual &&
                storage.idleCells() == 2 && storage.idleRings() == 2,
            "an evicted state refilled its lane's reserve");
    require(storage.releaseIdle(0, 0) == 2 * observedPrefixActual &&
                storage.actualAllocatedBytes() ==
                    beforeDrop - observedPrefixActual,
            "releaseIdle did not free the evicted cache slots");

    // The lane's own retired checkpoint refills its reserve instead, out of
    // idle reclaim's reach, so the next publication needs no admission.
    reserved->markRetired();
    reserved.reset();
    require(storage.actualSlotBytes(0) == observedSlotActual &&
                storage.idleCells() == 0 && storage.idleRings() == 0 &&
                storage.releaseIdle(0, 0) == 0 &&
                storage.actualAllocatedBytes() ==
                    beforeDrop - observedPrefixActual,
            "a retired checkpoint did not refill its lane's reserve");
    require(storage.metadata(0).assigned && storage.metadata(1).assigned &&
                storage.actualSlotBytes(1) == observedSlotActual,
            "pool reclaim touched active lane cells");
    admitNewAllocations = false;
    std::shared_ptr<const QwenCompositeState> retained = storage.snapshot(0);
    admitNewAllocations = true;
    require(retained != nullptr,
            "a later snapshot under a denying governor did not use the reserve");

    // A live cached state survives its lane's release and reclaim. Once its
    // lane is gone even a retired state returns to the pool, and only a later
    // reclaim frees it together with idle cells.
    require(storage.actualAllocatedBytes() == 2 * observedSlotActual,
            "retained snapshot accounting is wrong");
    storage.releaseSlot(0, 303);
    storage.releaseSlot(1, 202);
    require(storage.idleCells() == 5 && storage.idleRings() == 3,
            "released lane buffers are missing from the pool");
    retained->markRetired();
    retained.reset();
    require(storage.idleCells() == 6 && storage.idleRings() == 4,
            "dropped cached state did not return its buffers to the pool");
    // Releasing down to one admitted lane's worth keeps three cells and two
    // rings warm.
    require(storage.releaseIdle(3, 2) == observedSlotActual &&
                storage.idleCells() == 3 && storage.idleRings() == 2,
            "partial idle release did not keep the requested buffers");
    require(storage.releaseIdle(0, 0) == observedSlotActual,
            "idle lane buffers were not reclaimed");
    require(storage.idleCells() == 0 && storage.idleRings() == 0,
            "reclaimed buffers remain pooled");
    require(storage.actualAllocatedBytes() == 0,
            "reclaimed state cells remain accounted");
  }
  testTargetOnlyStates(backend, admitState, admitNewAllocations);
  testSuspensionKeepsTheMovedCell(backend, admitState, admitNewAllocations);
  require(backend.memoryStats().allocatedBytes == beforeStorage,
          "destroyed state slots remained in actual allocation count");

  std::cout << "qwen state storage tests passed: slot_declared="
            << kStateLayout.activeCellBytes() + kStateLayout.cachedBytes()
            << " slot_actual=" << observedSlotActual << " four_slots_declared="
            << uint64_t{model::ExecutionLimits::maximumBatchWidth} *
                   (kStateLayout.activeCellBytes() + kStateLayout.cachedBytes())
            << " four_slots_actual=" << observedStorageActual
            << " composite_declared=" << kStateLayout.cachedBytes()
            << " prefix_actual=" << observedPrefixActual << '\n';
}

} // namespace

int main(int argc, const char **argv) {
  if (argc != 2) {
    std::cerr << "usage: qwen_state_storage_test METALLIB\n";
    return EXIT_FAILURE;
  }
  @autoreleasepool {
    try {
      run(argv[1]);
      return EXIT_SUCCESS;
    } catch (const std::exception &error) {
      std::cerr << "qwen state storage test failed: " << error.what() << '\n';
      return EXIT_FAILURE;
    }
  }
}
