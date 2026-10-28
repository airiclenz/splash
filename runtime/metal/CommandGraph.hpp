#pragma once

#include "MetalBackend.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace splash::metal {

// The buffers one dispatch binds, in index order; inline up to the production
// maximum, so building one from a braced list allocates nothing.
using BufferList = InlineVector<MetalBuffer, kInlineBufferBindings>;

// A dispatch's pipeline name, which the dispatch owns: a std::string the
// caller built moves in, and a literal or view is copied once.
struct PipelineName final {
  PipelineName(std::string name) : value(std::move(name)) {}
  PipelineName(std::string_view name) : value(name) {}
  PipelineName(const char *name) : value(name) {}

  std::string value;
};

// Bindings a caller collected in a std::vector rather than a BufferList.
template <class Buffers>
concept BufferVector = std::same_as<std::remove_cvref_t<Buffers>, std::vector<MetalBuffer>>;

// An ordered dispatch list for one command, with the event steps that order
// it against another agent or the CPU (EventStep); MetalBackend splits a
// command into Metal command buffers at its event signals and host steps.
// Buffers bind at indices 0..n-1; an optional parameter struct binds at
// index n and is copied into graph-owned storage until submission.
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

  void add(PipelineName pipeline, BufferList buffers, DispatchSize groups,
           DispatchSize threads = {kDefaultThreads, 1, 1}) {
    push(std::move(pipeline), std::move(buffers), groups, threads);
  }

  template <class Params>
  void add(PipelineName pipeline, BufferList buffers, const Params &params,
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

  // A dispatch of the grid in `grid`, which an earlier dispatch writes
  // within `groups` (ComputeDispatch::indirectThreadgroups).
  template <class Params>
  void add(PipelineName pipeline, BufferList buffers, const Params &params,
           DispatchSize groups, IndirectGrid grid, DispatchSize threads) {
    add(std::move(pipeline), std::move(buffers), params, groups, threads);
    dispatches_.back().indirectThreadgroups = std::move(grid);
  }

  // The same dispatches with their bindings in a std::vector, moved or copied
  // into the inline list; a braced list never deduces these.
  template <BufferVector Buffers>
  void add(PipelineName pipeline, Buffers &&buffers, DispatchSize groups,
           DispatchSize threads = {kDefaultThreads, 1, 1}) {
    add(std::move(pipeline), bufferList(std::forward<Buffers>(buffers)), groups, threads);
  }

  template <BufferVector Buffers, class Params>
  void add(PipelineName pipeline, Buffers &&buffers, const Params &params,
           DispatchSize groups, DispatchSize threads = {kDefaultThreads, 1, 1}) {
    add(std::move(pipeline), bufferList(std::forward<Buffers>(buffers)), params, groups,
        threads);
  }

  template <BufferVector Buffers, class Params>
  void add(PipelineName pipeline, Buffers &&buffers, const Params &params,
           DispatchSize groups, IndirectGrid grid, DispatchSize threads) {
    add(std::move(pipeline), bufferList(std::forward<Buffers>(buffers)), params, groups,
        std::move(grid), threads);
  }

  // Event steps after the dispatches added so far (EventStep): signal once
  // all earlier work has completed, or hold all later work until the event
  // reaches value.
  void signal(SharedEvent event, uint64_t value) {
    step(std::move(event), value, EventStep::Kind::Signal);
  }
  void wait(SharedEvent event, uint64_t value) {
    step(std::move(event), value, EventStep::Kind::Wait);
  }
  // Ends the Metal command buffer after the dispatches added so far; the
  // next one runs them after it on the same queue (EventStep::Kind::Break).
  void split() { step({}, 0, EventStep::Kind::Break); }
  // A host step after the dispatches added so far (EventStep::Kind::Host):
  // `work` runs on the CPU between the work before and the work after it,
  // `timed` gets its round trip once the command completed, `gpu` says
  // whether the GPU spins through it, and `then` runs once the work after it
  // may go.
  void host(std::function<void()> work, std::function<void(const HostStepTiming &)> timed = {},
            HostStepGpu gpu = HostStepGpu::Idle, std::function<void()> then = {}) {
    events_.push_back({dispatches_.size(), {}, 0, EventStep::Kind::Host, std::move(work), std::move(timed), gpu,
                       std::move(then)});
  }

  [[nodiscard]] bool empty() const noexcept { return dispatches_.empty(); }
  // The dispatches alone; command() has the event steps too.
  [[nodiscard]] std::span<const ComputeDispatch> dispatches() const noexcept {
    return dispatches_;
  }
  [[nodiscard]] Command command() const noexcept {
    return {dispatches_, events_};
  }

private:
  // One chunk holds the parameter payloads of a few hundred dispatches.
  static constexpr size_t kPayloadChunkBytes = 16 * 1024;

  template <BufferVector Buffers>
  static BufferList bufferList(Buffers &&buffers) {
    BufferList list;
    list.reserve(buffers.size());
    for (auto &buffer : buffers) {
      if constexpr (std::is_lvalue_reference_v<Buffers>)
        list.push_back(buffer);
      else
        list.push_back(std::move(buffer));
    }
    return list;
  }

  ComputeDispatch &push(PipelineName pipeline, BufferList buffers,
                        DispatchSize groups, DispatchSize threads) {
    ComputeDispatch &dispatch = dispatches_.emplace_back();
    dispatch.pipelineName = std::move(pipeline.value);
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

  void step(SharedEvent event, uint64_t value, EventStep::Kind kind) {
    events_.push_back({dispatches_.size(), std::move(event), value, kind});
  }

  std::vector<std::unique_ptr<std::byte[]>> payloadChunks_;
  size_t payloadChunkBytes_ = 0;
  size_t payloadUsed_ = 0;
  std::vector<ComputeDispatch> dispatches_;
  std::vector<EventStep> events_;
};

} // namespace splash::metal
