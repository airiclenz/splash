#pragma once

#include "model/Model.hpp"
#include "model/ModelFactory.hpp"

#include <cstdint>
#include <memory>

namespace splash::model {

// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// The draft-model work a prefill or decode graph encodes. Without speculation
// no graph encodes any of it, so states stay target-only: no draft ring is
// written and no draft window is kept.
struct DraftEncoders final {
  // Target hidden rows captured as draft context: prefill spans, decode verify.
  bool isTargetHiddenCaptured = false;
  // The draft forward that proposes tokens (decode only).
  bool isDraftProposed = false;
  // The draft context written into the slot's draft ring: the packed context
  // prefill, or the decode's context commit.
  bool isDraftContextCommitted = false;

  bool operator==(const DraftEncoders &) const = default;
};

// ============================================================================
// The draft encoders a graph of the given kind encodes, given speculation.
[[nodiscard]] constexpr DraftEncoders
draftEncoders(WorkKind kind, bool isSpeculationEnabled) noexcept {
  if (!isSpeculationEnabled)
    return {};
  return {true, kind == WorkKind::Decode, true};
}

class Runtime final : public RuntimeModel {
public:
  explicit Runtime(RuntimeContext context);
  ~Runtime() override;
  void checkHealth() override;
  [[nodiscard]] bool needsHealthCheck() const noexcept override;

  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;

  // Direct native-oracle entry point. Production admission uses
  // begin() and installs its cache-aware plan explicitly.
  void beginColdRequest(const ModelRequest &request, uint32_t stateSlot);
  [[nodiscard]] StateAdmission
  begin(const ModelRequest &request) override;
  void suspend(uint64_t requestId) override;
  [[nodiscard]] std::shared_ptr<const CompositeState>
  suspendWithState(uint64_t requestId, uint32_t boundaryTokens) override;
  [[nodiscard]] StateAdmission
  resume(const ModelRequest &request) override;
  void restore(uint64_t requestId, uint32_t restoredPrefixLength,
                     std::shared_ptr<const CompositeState> restoredState,
                     bool restoreDraftState) override;
  void setDraftContextPlan(uint64_t requestId, DraftContextPlan plan) override;
  [[nodiscard]] std::vector<ModelStepResult>
  prefill(const BatchPlan &plan, std::span<const ModelBatchItem> items);
  [[nodiscard]] std::unique_ptr<ModelBatchTicket>
  submit(const BatchPlan &plan, std::span<const ModelBatchItem> items,
              std::function<void()> completion) override;
  [[nodiscard]] std::vector<ModelStepResult>
  decode(const BatchPlan &plan, std::span<const ModelBatchItem> items);
  [[nodiscard]] std::shared_ptr<const CompositeState>
  snapshot(uint64_t requestId) override;
  [[nodiscard]] uint64_t reclaimIdleState() noexcept override;
  void provideMask(uint64_t requestId,
                   std::span<const uint32_t> words) override;
  void end(uint64_t requestId) override;

  [[nodiscard]] WarmupStepResult warmupPrefill(uint32_t rows) override;
  [[nodiscard]] WarmupStepResult
  warmupDecodeBatch(uint32_t width) override;
  [[nodiscard]] WarmupStepResult warmupDraftVerifyCommit() override;
  [[nodiscard]] WarmupStepResult
  warmupCompositeStateRestore() override;
  [[nodiscard]] ModelMemoryActual
  actualRuntimeMemory() const override;

  [[nodiscard]] ModelTelemetry
  telemetry() const noexcept override;

private:
  void prepareWarmupDecode(uint64_t requestId, uint32_t anchor);
  [[nodiscard]] metal::AllocationResult beginAt(const ModelRequest &request,
                                       uint32_t stateSlot);
  [[nodiscard]] std::unique_ptr<ModelBatchTicket>
  prefillAsync(const BatchPlan &plan, std::span<const ModelBatchItem> items,
               std::function<void()> completion);
  [[nodiscard]] std::unique_ptr<ModelBatchTicket>
  decodeAsync(const BatchPlan &plan, std::span<const ModelBatchItem> items,
              std::function<void()> completion);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model
