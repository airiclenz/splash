#pragma once

#include "engine/CacheRecency.hpp"
#include "engine/KvCache.hpp"
#include "model/Model.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>

namespace splash::engine {

class StateCache;

struct StateCheckpoint final {
  uint64_t kvBlock = 0;
  uint64_t publication = 0;
  [[nodiscard]] explicit operator bool() const noexcept { return kvBlock != 0; }
};

// The checkpoints a conversation keeps beside its prompt-end state: its
// deepest prefill checkpoint and its finished reply's deepest decode
// checkpoint. The kept decode checkpoint's own state records only the prefill
// handle. Either handle may be empty, or stale once its publication is gone.
struct KeptCheckpoints final {
  StateCheckpoint prefill;
  StateCheckpoint decode;
};

// Names one publication a request needs: a suspended request's resume state,
// a queued request's cached prefix, a resident lane's latest checkpoint, or a
// conversation end's prompt-end state or kept checkpoints.
// A handle outlives neither the publication's eviction nor its replacement:
// releasing a stale handle is a no-op.
struct StateProtection final {
  uint64_t kvBlock = 0;
  uint64_t publication = 0;
  [[nodiscard]] explicit operator bool() const noexcept { return kvBlock != 0; }
};

class CompositeStateLease final {
public:
  CompositeStateLease(const CompositeStateLease &) = delete;
  CompositeStateLease &operator=(const CompositeStateLease &) = delete;
  CompositeStateLease(CompositeStateLease &&other) noexcept;
  CompositeStateLease &operator=(CompositeStateLease &&other) noexcept;
  ~CompositeStateLease() noexcept;

  [[nodiscard]] explicit operator bool() const noexcept {
    return owner_ != nullptr;
  }
  [[nodiscard]] uint64_t kvBlock() const noexcept { return kvBlock_; }
  [[nodiscard]] uint32_t boundary() const noexcept { return boundary_; }
  [[nodiscard]] const std::shared_ptr<const CompositeState> &
  state() const noexcept {
    return state_;
  }
  void reset() noexcept;

private:
  friend class StateCache;
  CompositeStateLease(StateCache &owner, uint64_t kvBlock, uint32_t boundary,
                      std::shared_ptr<const CompositeState> state) noexcept;

  StateCache *owner_ = nullptr;
  uint64_t kvBlock_ = 0;
  uint32_t boundary_ = 0;
  std::shared_ptr<const CompositeState> state_;
};

struct StateCacheSnapshot {
  uint32_t entries = 0;
  uint32_t pinned = 0;
  uint64_t bytes = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t publications = 0;
  uint64_t deduplicatedPublications = 0;
  uint64_t evictions = 0;
  uint32_t checkpointEntries = 0;
  uint64_t checkpointBytes = 0;
  uint64_t checkpointRetirements = 0;
  // Pressure and logical eviction; rolling retirements are counted separately.
  uint64_t checkpointEvictions = 0;
};

struct StateEviction final {
  bool evicted = false;
  uint64_t reclaimedBytes = 0;
};

// Attaches one immutable composite state to a complete target-KV block: the
// target-recurrent state, plus the draft-context window unless the state is
// target-only. The pair is restored atomically.
class StateCache final {
public:
  StateCache(KvCache &kv, CacheRecency &recency) : kv_(kv), recency_(recency) {}
  StateCache(const StateCache &) = delete;
  StateCache &operator=(const StateCache &) = delete;

  [[nodiscard]] std::optional<CompositeStateLease>
  acquireDeepest(std::span<const uint64_t> kvChain);
  // Acquisition pins backing; accounting occurs only when admission succeeds.
  void recordLookup(bool hit) noexcept { hit ? ++hits_ : ++misses_; }

  // Reuses a published state without a restore pin or lookup accounting.
  // A normal boundary upgrades a checkpoint; a checkpoint cannot downgrade
  // an ordinary state. Restoration alone never changes the boundary's role.
  [[nodiscard]] bool touchIfResident(uint64_t kvBlock, bool checkpoint = false);

  void publish(uint64_t kvBlock, std::shared_ptr<const CompositeState> state,
               bool checkpoint = false);
  // Publication identity protects replacement states from stale handles.
  [[nodiscard]] StateCheckpoint checkpoint(uint64_t kvBlock) const noexcept;
  // Ensures this publication is no longer a disposable checkpoint. Returns
  // false only when the matching checkpoint is pinned; absent, replaced and
  // upgraded publications already satisfy the postcondition.
  bool retireCheckpoint(StateCheckpoint checkpoint) noexcept;
  // Refreshes recency within the state's class. No-op when absent or pinned.
  void touch(uint64_t kvBlock) noexcept;
  // Records on the state published at kvBlock the checkpoints kept beside
  // it by the conversation whose prompt ended there, or whose kept decode
  // checkpoint it is. Replaces any earlier record; no-op when no state is
  // published.
  void recordKeptCheckpoint(uint64_t kvBlock, KeptCheckpoints kept) noexcept;
  // The checkpoints recorded beside the state at kvBlock, or empty handles.
  // A handle may be stale: retiring a stale handle is a no-op.
  [[nodiscard]] KeptCheckpoints keptCheckpoint(uint64_t kvBlock) const noexcept;

  [[nodiscard]] bool contains(uint64_t kvBlock) const noexcept;
  // Marks the resident publication as one a request needs.
  // Protection takes no pin and no lease: the state stays evictable, and only
  // a candidate search that spares protected states walks past it. Each call
  // needs one matching unprotect; lanes sharing a state count separately.
  // Returns an empty handle when no state is published at kvBlock.
  [[nodiscard]] StateProtection protect(uint64_t kvBlock);
  // Drops one protection. No-op for an empty handle or a publication that has
  // since been evicted or replaced.
  void unprotect(StateProtection protection) noexcept;
  // Unpinned checkpoints precede ordinary states regardless of recency.
  // keepResumePoint withholds the publication a follow-up request would
  // resume from, leaving no candidate when it is the only one. Checkpoints
  // keep their disposable priority: one is withheld only when no ordinary
  // state exists. spareProtected walks each queue past protected states, at
  // a cost bounded by the protected states ahead of the first other one;
  // without it the search is O(1). Growth reclaim and publication recycling
  // spare them first and fall back to them; only Warning reclaim never takes
  // one.
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  evictionCandidate(bool keepResumePoint = false,
                    bool spareProtected = false) const noexcept;
  [[nodiscard]] StateEviction evict(uint64_t kvBlock) noexcept;
  // Evicts exactly this checkpoint publication, counted as a checkpoint
  // eviction, while no more than allowedProtections protections hold it.
  // False when it is absent, replaced, upgraded to an ordinary state, pinned
  // or held by more protections. O(1).
  [[nodiscard]] bool evictCheckpoint(StateCheckpoint checkpoint,
                                     uint32_t allowedProtections) noexcept;
  [[nodiscard]] StateCacheSnapshot snapshot() const noexcept;

private:
  friend class CompositeStateLease;

  struct Entry {
    std::shared_ptr<const CompositeState> state;
    uint32_t pins = 0;
    // Requests that need this publication: suspended, queued, or resident
    // lanes holding it as their latest checkpoint, and conversation ends
    // holding it as a resume state.
    uint32_t protections = 0;
    uint64_t lastUsed = 0;
    uint64_t previousEvictable = 0;
    uint64_t nextEvictable = 0;
    bool evictable = false;
    bool checkpoint = false;
    uint64_t publication = 0;
    // The checkpoints kept beside this prompt-end state or kept decode
    // checkpoint; see recordKeptCheckpoint.
    KeptCheckpoints kept;
  };

  struct EvictionQueue {
    uint64_t oldest = 0;
    uint64_t newest = 0;
  };

  // The unpinned publication a follow-up request would resume from: the
  // newest ordinary state, or the newest checkpoint when none exists.
  [[nodiscard]] uint64_t resumePoint() const noexcept;
  [[nodiscard]] StateEviction erase(uint64_t kvBlock, bool retirement) noexcept;
  void release(uint64_t kvBlock) noexcept;
  [[nodiscard]] std::optional<CompositeStateLease>
  acquireBlock(uint64_t kvBlock);
  void insertEvictable(uint64_t kvBlock) noexcept;
  void removeEvictable(uint64_t kvBlock) noexcept;

  KvCache &kv_;
  CacheRecency &recency_;
  std::unordered_map<uint64_t, Entry> entries_;
  uint64_t bytes_ = 0;
  EvictionQueue ordinaryEviction_;
  EvictionQueue checkpointEviction_;
  uint32_t pinnedEntries_ = 0;
  uint64_t hits_ = 0;
  uint64_t misses_ = 0;
  uint64_t publications_ = 0;
  uint64_t deduplicatedPublications_ = 0;
  uint64_t evictions_ = 0;
  uint64_t checkpointEntries_ = 0;
  uint64_t checkpointBytes_ = 0;
  uint64_t checkpointRetirements_ = 0;
  uint64_t checkpointEvictions_ = 0;
};

} // namespace splash::engine
