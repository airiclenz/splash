#pragma once

#include "engine/KvCache.hpp"
#include "engine/KvPool.hpp"
#include "engine/StateCache.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace splash::engine {

struct CacheLookup final {
  uint32_t kvBoundary = 0;
  std::optional<CompositeStateLease> state;

  [[nodiscard]] uint32_t resumeBoundary() const noexcept {
    return state ? state->boundary() : 0;
  }
  [[nodiscard]] uint32_t junctionBoundary() const noexcept {
    return kvBoundary > resumeBoundary() ? kvBoundary : 0;
  }
};

class Cache;

class CacheProbe final {
public:
  [[nodiscard]] uint32_t cachedTokens() const noexcept { return cachedTokens_; }

private:
  friend class Cache;

  std::vector<uint64_t> blocks_;
  // Only pages examined by matchedBlocks can affect this probe's result.
  std::vector<uint32_t> checkedTokens_;
  std::vector<ImageSpan> images_;
  const Cache *owner_ = nullptr;
  size_t promptSize_ = 0;
  uint64_t kvGeneration_ = 0;
  uint32_t cachedTokens_ = 0;
};

struct CacheLookupSnapshot final {
  uint64_t lookups = 0;
  uint64_t kvHitTokens = 0;
  uint64_t stateHitTokens = 0;
  uint64_t lazyJunctions = 0;
};

struct CacheSnapshot final {
  KvPoolSnapshot pool;
  KvCache::Snapshot kvCache;
  StateCacheSnapshot stateCache;
  CacheLookupSnapshot lookup;
  uint32_t activeRequests = 0;
};

struct TokenAdmission final {
  KvPageAcquireFailure failure = KvPageAcquireFailure::None;
  uint32_t additionalPages = 0;
  uint32_t availablePages = 0;
  metal::AllocationFailure allocationFailure = metal::AllocationFailure::None;

  [[nodiscard]] bool granted() const noexcept {
    return failure == KvPageAcquireFailure::None;
  }
};

struct PageTableView final {
  std::span<const uint32_t> pages;
  uint64_t revision = 0;
};

struct CacheReclaimResult final {
  bool madeProgress = false;
  uint64_t reclaimedBytes = 0;
};

enum class CacheReclaimMode { ReuseBacking, ReleaseBacking };

// Owns active Q8 page leases, the content-addressed KV graph and cached
// composite states. Physical recurrent-state cells remain model-owned.
class Cache final {
public:
  Cache(KvPool &pool, CacheNamespace cacheNamespace);
  Cache(const Cache &) = delete;
  Cache &operator=(const Cache &) = delete;

  void beginRequest(uint64_t requestId);
  void endRequest(uint64_t requestId);

  // Scheduling probe only: does not pin, touch recency, or count a hit.
  [[nodiscard]] uint32_t
  cachedTokens(std::span<const uint32_t> prompt,
               std::span<const ImageSpan> images = {}) const;
  [[nodiscard]] CacheProbe
  probe(std::span<const uint32_t> prompt,
        std::span<const ImageSpan> images = {}) const;

  // Pin the usable prefix before potentially evicting for active allocations.
  // Accounting is separate: failed admission retries are not extra samples.
  [[nodiscard]] CacheLookup lookup(
      std::span<const uint32_t> prompt,
      std::span<const ImageSpan> images = {},
      const CacheProbe *probe = nullptr);
  void recordLookup(const CacheLookup &lookup);
  // Returns the lookup with its state replaced by the deepest one strictly
  // below the rejected state's block, pinned, or none; kvBoundary is kept.
  // Assigning the result over the argument releases the rejected lease. The
  // engine calls it, before restoreRequest, for a state it cannot use.
  [[nodiscard]] CacheLookup shallowerState(const CacheLookup &lookup);
  void restoreRequest(uint64_t requestId, const CacheLookup &lookup);

  // Grows the request's page table to cover tokenCount tokens. Short of
  // logical pages it evicts the oldest cached KV leaves; one that carries a
  // suspended request's resume state is taken only while this request is the
  // sole resident one.
  [[nodiscard]] TokenAdmission ensureTokens(uint64_t requestId,
                                            uint64_t tokenCount);
  [[nodiscard]] PageTableView pageTable(uint64_t requestId) const;

  // Canonicalizes every newly complete Page32 block. Duplicate content swaps
  // the request to the existing immutable page after the writer command has
  // completed; no active command ever aliases a writable page.
  [[nodiscard]] uint64_t publishCommittedBlocks(
      uint64_t requestId, std::span<const uint32_t> exactTokens,
      uint32_t committedTokens, std::span<const ImageSpan> images = {});
  [[nodiscard]] uint64_t blockAt(uint64_t requestId, uint32_t boundary) const;
  [[nodiscard]] bool reuseCompositeState(uint64_t kvBlock, bool checkpoint = false);
  void publishCompositeState(uint64_t kvBlock,
                             std::shared_ptr<const CompositeState> state,
                             bool checkpoint = false);
  [[nodiscard]] StateCheckpoint checkpointState(uint64_t kvBlock) const noexcept;
  // True while a composite state is published at kvBlock. A side-effect-free
  // test: unlike reuseCompositeState it neither touches recency nor demotes
  // a checkpoint nor counts a deduplicated publication.
  [[nodiscard]] bool carriesState(uint64_t kvBlock) const noexcept;
  // False only while this exact disposable publication is pinned.
  bool retireCheckpointState(StateCheckpoint checkpoint) noexcept;
  // Records the checkpoints kept beside the prompt-end state, or kept decode
  // checkpoint, at kvBlock, so a later request that resumes from that state
  // can find and retire them.
  // No-op when no state is published at kvBlock.
  void recordKeptCheckpoint(uint64_t kvBlock, KeptCheckpoints kept) noexcept;
  // The checkpoints recorded beside the state at kvBlock, or empty handles;
  // possibly stale.
  [[nodiscard]] KeptCheckpoints keptCheckpoint(uint64_t kvBlock) const noexcept;

  // One cache reclaimer for memory growth and pressure warnings. After empty
  // backing, disposable checkpoints are reclaimed first, except protected
  // ones such as a resident lane's latest checkpoint or a conversation end's
  // kept checkpoints and junction state. Ordinary states and
  // state-free KV leaves retain their shared oldest-first access order.
  // Active requests and pinned restores are never selected. Physical release
  // is paced by the backing: while an earlier release is still being torn
  // down, this pass stops instead of evicting cache whose extents could not
  // be released yet; the caller retries once releaseDeferred() clears.
  // keepResumePoint stops short of the newest state publication. A shrink
  // that no request is waiting for gains the one cell that publication holds
  // and costs the next request a replay of its whole prompt, because a
  // hybrid model cannot resume from cached KV without the recurrent state.
  // Older publications and state-free KV are still reclaimed.
  // keepRunway leaves one empty extent mapped: the free resident runway a
  // running request's next chunk, or a waiting request's admission, takes
  // without asking for a new mapping. Its bytes count toward targetBytes, so
  // no cache is evicted to make up for them. Without it every empty extent is
  // released.
  // A pass without evictAll (Warning pressure) spares every protected state
  // (a suspended request's resume state, a queued request's prefix, a
  // resident lane's latest checkpoint, a conversation end's prompt-end state,
  // kept checkpoints and junction state) and evicts the next publication
  // instead; evictAll (Critical pressure) takes protected states too, resume
  // states included.
  [[nodiscard]] uint64_t reclaimCache(uint64_t targetBytes, bool evictAll,
                                      bool keepResumePoint = false,
                                      bool keepRunway = false);
  // One bounded reclaim step for an allocation retry. Progress is distinct
  // from physical bytes because evicting a KV reference can make a resident
  // page reusable without immediately emptying its extent.
  // sparing names the protected states it walks past and never takes; an
  // evicted state counts under cause.
  [[nodiscard]] CacheReclaimResult reclaimOne(
      CacheReclaimMode mode = CacheReclaimMode::ReleaseBacking,
      bool keepResumePoint = false, bool keepRunway = false,
      StateSparing sparing = StateSparing::None,
      StateEvictionCause cause = StateEvictionCause::Growth);
  // One reclaim step for an allocation a request is waiting on: spares
  // protected states while any other state or KV qualifies, then takes the
  // oldest protected state rather than stall, except a suspended request's
  // resume state: that one is taken only with takeResumeStates, which the
  // caller passes when no lane other than the growing one is resident. Costs
  // the protected states at the head of each eviction queue, bounded by
  // waiting and resident lanes. An evicted state counts under cause: Growth
  // for a denied allocation, PausedBacking for idle backing reused while
  // growth is paused.
  [[nodiscard]] CacheReclaimResult reclaimOneForGrowth(
      CacheReclaimMode mode = CacheReclaimMode::ReleaseBacking,
      bool takeResumeStates = false,
      StateEvictionCause cause = StateEvictionCause::Growth);
  // Recycles an unpinned state, preferring checkpoints, for a required state
  // publication. KV bytes only return once an extent unmaps. A protected
  // state is taken only when no other state qualifies, and a suspended
  // request's resume state only with takeResumeStates (no lane other than
  // the publishing one is resident); with checkpointsOnly only an
  // unprotected checkpoint qualifies, so a protected checkpoint is recycled
  // rather than none. The result is false when nothing qualified, and names
  // the evicted state's bytes and the part of them that reached the model's
  // pool: a live lane's short reserve takes the rest.
  [[nodiscard]] StateEviction reclaimOneState(bool checkpointsOnly = false,
                                              bool takeResumeStates = false);
  // Recycles exactly this checkpoint publication for a required state
  // publication, through the eviction path, while no more than
  // heldProtections protections hold it: the caller passes the ones its own
  // conversation holds, so a state another request needs is never taken.
  // False when it is gone, pinned or held by anyone else. O(1).
  [[nodiscard]] bool recycleCheckpointState(StateCheckpoint checkpoint,
                                            uint32_t heldProtections) noexcept;
  // Protects the deepest composite state on the request's published prefix
  // that a resume of resumeTokens tokens can restore, as a resume state: from
  // Warning reclaim, and from growth and publication reclaim while another
  // lane is resident. Like lookup, it leaves one token to recompute, so a
  // state on the block ending at resumeTokens is skipped. Call before
  // endRequest; returns an empty handle when no such state exists. Reads the
  // request's own block list: no lookup, hashing or pin.
  [[nodiscard]] StateProtection protectDeepestState(uint64_t requestId,
                                                    uint32_t resumeTokens);
  // Protects the state published at kvBlock: the prefix a queued request
  // found and will restore, a resident lane's latest checkpoint, or a
  // conversation end's prompt-end state, kept checkpoints or junction state.
  // Returns an empty handle when no state is published there.
  [[nodiscard]] StateProtection protectState(uint64_t kvBlock);
  // Ends one protection. No-op once its publication is gone.
  void unprotectState(StateProtection protection) noexcept;
  // True while the publication protection names is still published: false
  // once eviction or retirement took it, even if a newer state has since
  // been published at the same block. O(1).
  [[nodiscard]] bool
  isProtectedStateResident(StateProtection protection) const noexcept;
  // Empty resident backing exists but the previous release is still in
  // flight; more reclaim work becomes possible without evicting anything.
  // keepRunway does not count the one extent a runway-keeping pass leaves.
  [[nodiscard]] bool releaseDeferred(bool keepRunway = false) const noexcept;
  // Free resident pages outside empty extents: what a request can take while
  // every empty extent stays reclaimable.
  [[nodiscard]] uint32_t partlyUsedFreePages() const noexcept;
  // Includes the last unmap, even when no empty extent remains to reclaim.
  [[nodiscard]] bool releasePending() const noexcept;
  [[nodiscard]] uint64_t releaseGeneration() const noexcept;
  // Startup cleanup only: unmap unused backing without evicting cache data,
  // keeping one runway extent. Waits for each paced release.
  void releaseUnusedKvBacking();
  [[nodiscard]] CacheSnapshot snapshot() const;

private:
  [[nodiscard]] std::vector<uint64_t>
  matchedBlocks(std::span<const uint32_t> prompt,
                std::span<const ImageSpan> images) const;

  struct Request final {
    std::vector<uint32_t> pages;
    std::vector<uint64_t> cachedBlocks;
    uint64_t pageTableRevision = 0;
  };

  [[nodiscard]] Request &request(uint64_t requestId);
  [[nodiscard]] const Request &request(uint64_t requestId) const;
  [[nodiscard]] bool makeLogicalPages(uint32_t count);
  [[nodiscard]] bool evictOneKvBlock();
  [[nodiscard]] std::optional<CacheEvictionCandidate>
  oldestStateFreeKvBlock() const;
  [[nodiscard]] uint64_t reclaimEmptyExtents(bool keepRunway);

  KvPool &pool_;
  CacheRecency recency_;
  KvCache kv_;
  StateCache states_;
  std::unordered_map<uint64_t, Request> requests_;
  CacheLookupSnapshot lookup_;
};

} // namespace splash::engine
