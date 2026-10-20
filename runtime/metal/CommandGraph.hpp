#pragma once

#include "MetalBackend.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace splash::metal {

// The buffers one dispatch binds, in index order; inline up to the production
// maximum, so building one from a braced list allocates nothing.
using BufferList = InlineVector<MetalBuffer, kInlineBufferBindings>;

// An ordered dispatch list for one command buffer. Buffers bind at indices
// 0..n-1; an optional parameter struct binds at index n and is copied into
// graph-owned storage until submission.
class CommandGraph final {
public:
  static constexpr uint32_t kDefaultThreads = 256;

  CommandGraph() = default;
  // Reserves room for expectedDispatches so encoding never regrows the list.
  explicit CommandGraph(size_t expectedDispatches) { dispatches_.reserve(expectedDispatches); }
  // Dispatches point into payloadChunks_; a copy would keep pointing at the
  // source. A move keeps every chunk, and so every payload address.
  CommandGraph(const CommandGraph &) = delete;
  CommandGraph &operator=(const CommandGraph &) = delete;
  CommandGraph(CommandGraph &&) noexcept = default;
  CommandGraph &operator=(CommandGraph &&) noexcept = default;

  void add(std::string pipeline, BufferList buffers, DispatchSize groups,
           DispatchSize threads = {kDefaultThreads, 1, 1}) {
    push(std::move(pipeline), std::move(buffers), groups, threads);
  }

  template <class Params>
  void add(std::string pipeline, BufferList buffers, const Params &params,
           DispatchSize groups, DispatchSize threads = {kDefaultThreads, 1, 1}) {
    static_assert(std::is_trivially_copyable_v<Params>,
                  "dispatch parameters must be plain data");
    static_assert(alignof(Params) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__,
                  "dispatch parameters exceed the payload chunk alignment");
    void *payload = storePayload(sizeof(Params), alignof(Params));
    std::memcpy(payload, &params, sizeof(Params));
    ComputeDispatch &dispatch =
        push(std::move(pipeline), std::move(buffers), groups, threads);
    dispatch.bytes.push_back({static_cast<uint32_t>(dispatch.buffers.size()),
                              payload, sizeof(Params)});
  }

  [[nodiscard]] bool empty() const noexcept { return dispatches_.empty(); }
  [[nodiscard]] std::span<const ComputeDispatch> dispatches() const noexcept {
    return dispatches_;
  }

private:
  // One chunk holds the parameter payloads of a few hundred dispatches.
  static constexpr size_t kPayloadChunkBytes = 16 * 1024;

  ComputeDispatch &push(std::string pipeline, BufferList buffers,
                        DispatchSize groups, DispatchSize threads) {
    ComputeDispatch &dispatch = dispatches_.emplace_back();
    dispatch.pipelineName = std::move(pipeline);
    dispatch.threadgroups = groups;
    dispatch.threadsPerThreadgroup = threads;
    dispatch.buffers.reserve(buffers.size());
    for (uint32_t index = 0; index < buffers.size(); ++index) {
      dispatch.buffers.push_back({index, std::move(buffers[index])});
    }
    return dispatch;
  }

  // Returns size bytes aligned to alignment that stay at one address for the
  // graph's lifetime: chunks are never resized or freed before the graph.
  void *storePayload(size_t size, size_t alignment) {
    size_t offset = (payloadUsed_ + alignment - 1) & ~(alignment - 1);
    if (payloadChunks_.empty() || offset + size > payloadChunkBytes_) {
      payloadChunkBytes_ = std::max(kPayloadChunkBytes, size);
      payloadChunks_.push_back(std::make_unique_for_overwrite<std::byte[]>(payloadChunkBytes_));
      offset = 0;
    }
    payloadUsed_ = offset + size;
    return payloadChunks_.back().get() + offset;
  }

  std::vector<std::unique_ptr<std::byte[]>> payloadChunks_;
  size_t payloadChunkBytes_ = 0;
  size_t payloadUsed_ = 0;
  std::vector<ComputeDispatch> dispatches_;
};

} // namespace splash::metal
