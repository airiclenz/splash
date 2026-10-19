#include "model/QwenState.hpp"

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace splash::model {
namespace {

using metal::MetalBuffer;

void *writableContents(const MetalBuffer &buffer, const char *name) {
  void *contents = buffer.contents();
  if (!contents) {
    throw std::logic_error(std::string(name) + " is not CPU-visible");
  }
  return contents;
}

void copyExact(const MetalBuffer &destination, const MetalBuffer &source,
               const char *name) {
  if (destination.sizeBytes() != source.sizeBytes()) {
    throw std::logic_error(std::string(name) + " shape mismatch");
  }
  std::memcpy(writableContents(destination, name),
              writableContents(source, name), destination.sizeBytes());
}

void clear(const MetalBuffer &buffer, const char *name) {
  std::memset(writableContents(buffer, name), 0, buffer.sizeBytes());
}

// An admitted slot holds the lane's two GDN parities plus the reserve's cell,
// and the lane's draft ring plus the reserve's ring.
constexpr uint32_t kSlotCells = 3;
constexpr uint32_t kSlotRings = 2;
constexpr uint32_t kReserveCell = 2;
constexpr uint32_t kReserveRing = 1;

// Takes up to taken.size() buffers from the back of the pool, in pop order.
template <class Buffer, std::size_t Count>
uint32_t takePooled(std::vector<std::shared_ptr<Buffer>> &pool,
                    std::array<std::shared_ptr<Buffer>, Count> &taken) {
  uint32_t count = 0;
  while (count < Count && !pool.empty()) {
    taken[count++] = std::move(pool.back());
    pool.pop_back();
  }
  return count;
}

// Undoes takePooled, so the pool regains its previous order.
template <class Buffer, std::size_t Count>
void returnPooled(std::vector<std::shared_ptr<Buffer>> &pool,
                  std::array<std::shared_ptr<Buffer>, Count> &taken,
                  uint32_t count) {
  while (count > 0)
    pool.push_back(std::move(taken[--count]));
}

// Returns a dropped buffer to the reserve when one is offered and short of
// it, else to the pool. A null buffer goes nowhere.
template <class Buffer>
void returnBuffer(std::shared_ptr<Buffer> &buffer,
                  std::type_identity_t<std::shared_ptr<Buffer>> *reserve,
                  std::vector<std::shared_ptr<Buffer>> &pool) {
  if (!buffer)
    return;
  if (reserve && !*reserve) {
    *reserve = std::move(buffer);
    return;
  }
  pool.push_back(std::move(buffer));
}

} // namespace

QwenGdnCell::QwenGdnCell(metal::MetalBackend &backend,
                         std::shared_ptr<StateAllocationTracker> tracker,
                         GdnStateLayout layout, std::string_view label)
    : tracker_(std::move(tracker)) {
  if (!tracker_)
    throw std::invalid_argument("Qwen state allocation tracker is empty");
  if (!layout.valid())
    throw std::invalid_argument("Qwen GDN state layout is invalid");
  const uint64_t before = backend.memoryStats().allocatedBytes;
  buffers_.stateBase = backend.allocateBuffer(
      layout.cellBytes(), metal::BufferStorage::Shared, label);
  buffers_.convolutionBase = backend.view(
      buffers_.stateBase, 0, layout.convolutionBytes());
  buffers_.convolutionLayers.resize(layout.layers);
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    buffers_.convolutionLayers[layer] =
        backend.view(buffers_.convolutionBase,
                     uint64_t{layer} * layout.convolutionLayerBytes(),
                     layout.convolutionLayerBytes());
  }
  buffers_.recurrentBase =
      backend.view(buffers_.stateBase, layout.convolutionBytes(),
                   layout.recurrentBytes());
  buffers_.recurrentLayers.resize(layout.layers);
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    buffers_.recurrentLayers[layer] = backend.view(
        buffers_.recurrentBase,
        uint64_t{layer} * layout.recurrentLayerBytes(),
        layout.recurrentLayerBytes());
  }
  actualAllocatedBytes_ =
      metal::allocationDelta(before, backend.memoryStats().allocatedBytes);
  if (actualAllocatedBytes_ < layout.cellBytes()) {
    throw std::logic_error("Qwen GDN allocation is below declared bytes");
  }
  tracker_->bytes.fetch_add(actualAllocatedBytes_, std::memory_order_relaxed);
}

QwenGdnCell::~QwenGdnCell() {
  tracker_->bytes.fetch_sub(actualAllocatedBytes_, std::memory_order_relaxed);
}

QwenCompositeState::QwenCompositeState(
    std::shared_ptr<QwenBufferPool> pool,
    std::shared_ptr<QwenSlotReserves> reserves, QwenStateOrigin origin,
    QwenCacheSlot slot, CompositeStateLayout layout, QwenLogicalLengths lengths)
    : pool_(std::move(pool)), reserves_(std::move(reserves)), origin_(origin),
      slot_(std::move(slot)), layout_(layout), lengths_(lengths) {
  if (!pool_ || !reserves_ || !slot_.gdn) {
    throw std::invalid_argument("composite state buffers are empty");
  }
  // Only a target-only state, whose lane had no complete draft window, may
  // come without a draft ring.
  if (!slot_.draft && lengths_.hasCompleteDraftWindow(layout_.draft.tokens)) {
    throw std::invalid_argument("composite state lost its draft ring");
  }
  if (origin_.slot >= reserves_->size() || !origin_.requestId)
    throw std::invalid_argument("composite state origin is not a live slot");
}

QwenCompositeState::~QwenCompositeState() {
  if (!pool_->open)
    return;
  // Only a retired checkpoint refills its lane's reserve. An evicted state's
  // bytes are credited to reclaim, so they must reach the pool.
  QwenCacheSlot *reserve = nullptr;
  if (retired_) {
    QwenSlotReserve &origin = (*reserves_)[origin_.slot];
    if (origin.requestId == origin_.requestId)
      reserve = &origin.buffers;
  }
  returnBuffer(slot_.gdn, reserve ? &reserve->gdn : nullptr, pool_->cells);
  returnBuffer(slot_.draft, reserve ? &reserve->draft : nullptr, pool_->rings);
}

QwenStateStorage::QwenStateStorage(metal::MetalBackend &backend,
                                   metal::AllocationAdmission admitAllocation,
                                   CompositeStateLayout layout)
    : backend_(backend), admitAllocation_(std::move(admitAllocation)),
      layout_(layout),
      allocations_(std::make_shared<StateAllocationTracker>()),
      pool_(std::make_shared<QwenBufferPool>()),
      reserves_(std::make_shared<QwenSlotReserves>()) {
  if (!admitAllocation_)
    throw std::invalid_argument("Qwen state allocation admission is required");
  if (!layout_.valid() ||
      layout_.draft.tokens != ExecutionLimits::draftContextTokens) {
    throw std::invalid_argument("Qwen composite state layout is invalid");
  }
}

QwenStateStorage::~QwenStateStorage() {
  pool_->open = false;
  pool_->cells.clear();
  pool_->rings.clear();
  reserves_->fill({});
}

const QwenSlotBuffers &QwenStateStorage::buffers(uint32_t index) const {
  return slot(index).buffers;
}

const QwenSlotMetadata &QwenStateStorage::metadata(uint32_t index) const {
  return slot(index).metadata;
}

metal::AllocationResult QwenStateStorage::tryActivateSlot(uint32_t index, uint64_t requestId) {
  if (!requestId)
    throw std::invalid_argument("request id must be non-zero");
  Slot &current = slot(index);
  if (current.metadata.assigned) {
    throw std::logic_error("Qwen state slot is already assigned");
  }
  if (auto admission = allocateSlot(index); !admission)
    return admission;

  // A fresh recurrent sequence reads parity zero immediately. Parity one is
  // fully overwritten by the first transition. Draft validity is controlled
  // by the zero logical lengths below.
  clear(current.buffers.gdn[0].convolutionBase, "slot convolution state");
  clear(current.buffers.gdn[0].recurrentBase, "slot recurrent state");
  current.metadata = {true, requestId, 0, {}};
  (*reserves_)[index].requestId = requestId;
  return true;
}

void QwenStateStorage::releaseSlot(uint32_t index, uint64_t requestId) {
  Slot &current = slot(index);
  requireAssigned(current);
  if (!requestId || current.metadata.requestId != requestId) {
    throw std::logic_error("Qwen state slot owner mismatch");
  }
  // Reserve first, then parity one, so the next activation pops parity zero
  // first and a reactivated lane gets its previous buffers back in the same
  // order. A reserve a snapshot spent returns nothing here.
  QwenSlotReserve &reserve = (*reserves_)[index];
  returnBuffer(reserve.buffers.gdn, nullptr, pool_->cells);
  returnBuffer(reserve.buffers.draft, nullptr, pool_->rings);
  reserve.requestId = 0;
  for (uint32_t parity = current.gdn.size(); parity > 0;) {
    --parity;
    returnBuffer(current.gdn[parity], nullptr, pool_->cells);
  }
  returnBuffer(current.draft, nullptr, pool_->rings);
  refreshViews(current);
  current.metadata = {};
}

uint64_t QwenStateStorage::releaseIdle(uint32_t keepCells,
                                       uint32_t keepRings) noexcept {
  const uint64_t before = backend_.memoryStats().allocatedBytes;
  while (pool_->cells.size() > keepCells)
    pool_->cells.pop_back();
  while (pool_->rings.size() > keepRings)
    pool_->rings.pop_back();
  const uint64_t after = backend_.memoryStats().allocatedBytes;
  return before >= after ? before - after : 0;
}

uint32_t QwenStateStorage::idleCells() const noexcept {
  return static_cast<uint32_t>(pool_->cells.size());
}

uint32_t QwenStateStorage::idleRings() const noexcept {
  return static_cast<uint32_t>(pool_->rings.size());
}

void QwenStateStorage::updateLengths(uint32_t index,
                                     QwenLogicalLengths lengths) {
  validateLengths(lengths, false);
  Slot &current = slot(index);
  requireAssigned(current);
  current.metadata.lengths = lengths;
}

void QwenStateStorage::swapParity(uint32_t index) {
  Slot &current = slot(index);
  requireAssigned(current);
  current.metadata.activeParity ^= 1;
}

std::shared_ptr<const QwenCompositeState>
QwenStateStorage::snapshot(uint32_t index) {
  Slot &source = slot(index);
  requireAssigned(source);
  validateLengths(source.metadata.lengths, true);
  // The lane's reserve first, then pooled buffers, then the governor. A
  // denied admission puts a taken cell back where it came from and drops a
  // fresh one, leaving no trace.
  QwenCacheSlot &reserve = (*reserves_)[index].buffers;
  QwenCacheSlot cacheSlot;
  const bool reservedCell = static_cast<bool>(reserve.gdn);
  const bool pooledCell = !reservedCell && !pool_->cells.empty();
  cacheSlot.gdn = reservedCell ? std::move(reserve.gdn)
                               : acquireCell("qwen-state-cache-gdn");
  if (!cacheSlot.gdn)
    return nullptr;
  cacheSlot.draft = reserve.draft ? std::move(reserve.draft)
                                  : acquireRing("qwen-state-cache-draft");
  if (!cacheSlot.draft) {
    if (reservedCell)
      reserve.gdn = std::move(cacheSlot.gdn);
    else if (pooledCell)
      pool_->cells.push_back(std::move(cacheSlot.gdn));
    return nullptr;
  }
  const uint32_t active = source.metadata.activeParity;
  copyExact(cacheSlot.gdn->buffers().stateBase,
            source.gdn[active]->buffers().stateBase, "cached GDN state");
  for (uint32_t layer = 0; layer < source.buffers.draft.size(); ++layer) {
    copyExact(cacheSlot.draft->layers()[layer].keys,
              source.buffers.draft[layer].keys, "cached draft keys");
    copyExact(cacheSlot.draft->layers()[layer].values,
              source.buffers.draft[layer].values, "cached draft values");
  }
  return std::shared_ptr<const QwenCompositeState>(new QwenCompositeState(
      pool_, reserves_, {index, source.metadata.requestId},
      std::move(cacheSlot), layout_, source.metadata.lengths));
}

std::shared_ptr<const QwenCompositeState>
QwenStateStorage::moveState(uint32_t index) {
  Slot &source = slot(index);
  requireAssigned(source);
  const uint32_t active = source.metadata.activeParity;
  const QwenLogicalLengths lengths = source.metadata.lengths;
  QwenCacheSlot moved;
  moved.gdn = std::move(source.gdn[active]);
  source.buffers.gdn[active] = {};
  if (lengths.hasCompleteDraftWindow(layout_.draft.tokens)) {
    moved.draft = std::move(source.draft);
    source.buffers.draft.clear();
  }
  return std::shared_ptr<const QwenCompositeState>(new QwenCompositeState(
      pool_, reserves_, {index, source.metadata.requestId}, std::move(moved),
      layout_, lengths));
}

std::shared_ptr<QwenGdnCell>
QwenStateStorage::acquireCell(std::string_view label) {
  if (pool_->cells.empty())
    return allocateGdnCell(label);
  std::shared_ptr<QwenGdnCell> cell = std::move(pool_->cells.back());
  pool_->cells.pop_back();
  return cell;
}

std::shared_ptr<DFlashDraftRing>
QwenStateStorage::acquireRing(std::string_view label) {
  if (pool_->rings.empty())
    return allocateDraftRing(label);
  std::shared_ptr<DFlashDraftRing> ring = std::move(pool_->rings.back());
  pool_->rings.pop_back();
  return ring;
}

void QwenStateStorage::restore(uint32_t index, const CompositeState &state,
                               bool restoreDraftState) {
  const auto *typed = dynamic_cast<const QwenCompositeState *>(&state);
  if (!typed) {
    throw std::invalid_argument("composite state is not Qwen state");
  }
  if (typed->layout_ != layout_) {
    throw std::invalid_argument("composite state layout does not match model");
  }
  const QwenLogicalLengths &stored = typed->lengths_;
  if (typed->carriesDraft()) {
    validateLengths(stored, true);
  } else if (restoreDraftState) {
    throw std::invalid_argument("target-only composite state has no draft");
  } else if (!stored.targetTokens || stored.targetTokens % kv::kPageTokens) {
    throw std::invalid_argument(
        "target-only composite state requires page-aligned target tokens");
  }
  Slot &destination = slot(index);
  requireAssigned(destination);
  if (!typed->slot_.gdn) {
    throw std::invalid_argument("incompatible Qwen composite state");
  }

  const uint32_t active = destination.metadata.activeParity;
  copyExact(destination.gdn[active]->buffers().stateBase,
            typed->slot_.gdn->buffers().stateBase, "restored GDN state");
  if (restoreDraftState) {
    for (uint32_t layer = 0; layer < destination.buffers.draft.size();
         ++layer) {
      copyExact(destination.buffers.draft[layer].keys,
                typed->slot_.draft->layers()[layer].keys,
                "restored draft keys");
      copyExact(destination.buffers.draft[layer].values,
                typed->slot_.draft->layers()[layer].values,
                "restored draft values");
    }
  }
  uint64_t requestId = destination.metadata.requestId;
  QwenLogicalLengths lengths = stored;
  if (!restoreDraftState) {
    lengths.draftBase = lengths.targetTokens;
    lengths.draftLength = 0;
    lengths.draftCommitCursor =
        lengths.targetTokens % layout_.draft.tokens;
  }
  destination.metadata = {true, requestId, active, lengths};
}

uint64_t QwenStateStorage::actualSlotBytes(uint32_t index) const {
  const Slot &current = slot(index);
  uint64_t result = 0;
  if (current.gdn[0])
    result += current.gdn[0]->actualAllocatedBytes();
  if (current.gdn[1])
    result += current.gdn[1]->actualAllocatedBytes();
  if (current.draft)
    result += current.draft->actualAllocatedBytes();
  const QwenCacheSlot &reserve = (*reserves_)[index].buffers;
  if (reserve.gdn)
    result += reserve.gdn->actualAllocatedBytes();
  if (reserve.draft)
    result += reserve.draft->actualAllocatedBytes();
  return result;
}

metal::AllocationResult QwenStateStorage::allocateSlot(uint32_t index) {
  Slot &destination = slot(index);
  QwenCacheSlot &reserve = (*reserves_)[index].buffers;
  if (destination.gdn[0] || destination.gdn[1] || destination.draft ||
      reserve.gdn || reserve.draft) {
    throw std::logic_error("idle Qwen state slot still owns buffers");
  }
  // Pooled buffers first, then the governor for what the pool lacks. The
  // lane and its reserve are admitted as a whole: a denied admission leaves
  // no trace, pooled buffers go back and fresh ones are dropped.
  std::array<std::shared_ptr<QwenGdnCell>, kSlotCells> cells;
  std::array<std::shared_ptr<DFlashDraftRing>, kSlotRings> rings;
  metal::AllocationFailure failure = metal::AllocationFailure::None;
  const uint32_t pooledCells = takePooled(pool_->cells, cells);
  const uint32_t pooledRings = takePooled(pool_->rings, rings);
  const auto giveBack = [&] {
    returnPooled(pool_->cells, cells, pooledCells);
    returnPooled(pool_->rings, rings, pooledRings);
  };
  const std::string prefix = "qwen-state-cell-" + std::to_string(index);
  for (uint32_t cell = pooledCells; cell < cells.size(); ++cell) {
    cells[cell] = allocateGdnCell(
        cell == kReserveCell ? prefix + "-reserve-gdn"
                             : prefix + "-gdn-" + std::to_string(cell),
        &failure);
    if (!cells[cell]) {
      giveBack();
      return failure;
    }
  }
  for (uint32_t ring = pooledRings; ring < rings.size(); ++ring) {
    rings[ring] = allocateDraftRing(
        ring == kReserveRing ? prefix + "-reserve-draft" : prefix + "-draft",
        &failure);
    if (!rings[ring]) {
      giveBack();
      return failure;
    }
  }
  destination.gdn = {std::move(cells[0]), std::move(cells[1])};
  destination.draft = std::move(rings[0]);
  reserve = {std::move(cells[kReserveCell]), std::move(rings[kReserveRing])};
  refreshViews(destination);
  return true;
}

std::shared_ptr<QwenGdnCell>
QwenStateStorage::allocateGdnCell(std::string_view label,
                                     metal::AllocationFailure *failure) {
  std::shared_ptr<QwenGdnCell> result;
  const auto admission = admitAllocation_(layout_.target.cellBytes(), [&] {
        result = std::shared_ptr<QwenGdnCell>(
            new QwenGdnCell(backend_, allocations_, layout_.target, label));
      });
  if (!admission) {
    if (failure)
      *failure = admission.failure;
    return {};
  }
  if (!result)
    throw std::logic_error("state admission skipped GDN allocation");
  return result;
}

std::shared_ptr<DFlashDraftRing>
QwenStateStorage::allocateDraftRing(std::string_view label,
                                     metal::AllocationFailure *failure) {
  std::shared_ptr<DFlashDraftRing> result;
  const auto admission = admitAllocation_(layout_.draft.ringBytes(), [&] {
        result = std::shared_ptr<DFlashDraftRing>(
            new DFlashDraftRing(backend_, allocations_, layout_.draft,
                                label));
      });
  if (!admission) {
    if (failure)
      *failure = admission.failure;
    return {};
  }
  if (!result)
    throw std::logic_error("state admission skipped draft allocation");
  return result;
}

void QwenStateStorage::refreshViews(Slot &current) {
  for (uint32_t parity = 0; parity < current.gdn.size(); ++parity) {
    current.buffers.gdn[parity] = current.gdn[parity]
                                      ? current.gdn[parity]->buffers()
                                      : GdnParityBuffers{};
  }
  current.buffers.draft =
      current.draft ? current.draft->layers()
                    : std::vector<DFlashDraftRingLayer>{};
}

QwenStateStorage::Slot &QwenStateStorage::slot(uint32_t index) {
  if (index >= slots_.size()) {
    throw std::out_of_range("invalid Qwen state slot");
  }
  return slots_[index];
}

const QwenStateStorage::Slot &QwenStateStorage::slot(uint32_t index) const {
  if (index >= slots_.size()) {
    throw std::out_of_range("invalid Qwen state slot");
  }
  return slots_[index];
}

void QwenStateStorage::validateLengths(const QwenLogicalLengths &lengths,
                                       bool cacheSnapshot) const {
  if (lengths.draftLength > layout_.draft.tokens ||
      lengths.draftCommitCursor >= layout_.draft.tokens ||
      lengths.draftEnd() > lengths.targetTokens) {
    throw std::invalid_argument("invalid draft ring metadata");
  }
  if (cacheSnapshot &&
      (!lengths.targetTokens ||
       !lengths.hasCompleteDraftWindow(layout_.draft.tokens) ||
       lengths.targetTokens % kv::kPageTokens)) {
    throw std::invalid_argument(
        "composite snapshot requires equal page-aligned committed lengths");
  }
}

void QwenStateStorage::requireAssigned(const Slot &current) {
  if (!current.metadata.assigned || !current.metadata.requestId) {
    throw std::logic_error("Qwen state slot is not assigned");
  }
}

} // namespace splash::model
