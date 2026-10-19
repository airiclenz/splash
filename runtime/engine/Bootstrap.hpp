#pragma once

#include "engine/NativeRuntime.hpp"
#include "engine/RuntimeResources.hpp"
#include "engine/Scheduler.hpp"
#include "engine/Status.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::engine {

enum class RuntimeBootstrapStage {
    ResourceAssembly,
    ModelCreation,
    MaximumPrefill,
    DecodeWarmup,
    DraftVerifyCommit,
    CompositeStateRestore,
    MemoryAudit,
    AnnounceReady,
    Ready,
};

[[nodiscard]] std::string_view runtimeBootstrapStageName(
    RuntimeBootstrapStage stage);

struct RuntimeBootstrapReport {
    bool ready = false;
    RuntimeBootstrapStage stage = RuntimeBootstrapStage::ResourceAssembly;
    RuntimeResourceFailure resourceFailure = RuntimeResourceFailure::Other;
    std::string message;
    WarmupReport warmup;
    MemoryAuditResult memoryAudit;
    // Always present once an immutable plan exists. On planning failure this
    // is the full BudgetValidationStatus JSON instead.
    std::string memoryPlanJson;
    std::string budgetDescription;

    [[nodiscard]] std::string describe() const;
};

class RuntimeBootstrapError final : public std::runtime_error {
public:
    explicit RuntimeBootstrapError(RuntimeBootstrapReport report);
    explicit RuntimeBootstrapError(const RuntimeResourcesError &error);

    [[nodiscard]] const RuntimeBootstrapReport &report() const noexcept {
        return report_;
    }

private:
    RuntimeBootstrapReport report_;
};

struct RuntimeBootstrapConfig {
    RuntimeResourcesConfig resources;
    NativeLoopConfig nativeLoop;
    protocol::ProtocolLimits protocolLimits;
};

// A serve-native command line the operator must correct; the message names
// the offending option.
class NativeUsageError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The serve-native options that follow the four positionals.
struct NativeOptions {
    kv::Format kvFormat = kv::Format::Int8;
    // Off keeps the target verify graph but skips the draft proposal.
    bool isSpeculationEnabled = true;
    // The contended prefill floor; see Scheduler::kDefaultContendedPrefillRows.
    uint32_t contendedPrefillRows = Scheduler::kDefaultContendedPrefillRows;
};

// Parses `[--kv-format int8|bf16] [--no-speculation]
// [--contended-prefill-rows N]` in any order. Throws NativeUsageError on an
// unknown, repeated, incomplete or invalid option.
[[nodiscard]] NativeOptions parseNativeOptions(
    std::span<const std::string_view> arguments);

// The resolved serve-native command line.
struct NativeArguments {
    std::filesystem::path modelRoot;
    model::ModelDescriptor model;
    // Zero means auto.
    uint32_t maxContext = 0;
    // Zero means auto.
    uint64_t maxMemoryBytes = 0;
    NativeOptions options;
};

// Facts about this process the bootstrap configuration binds.
struct NativeProcessIdentity {
    std::filesystem::path metallibPath;
    std::string buildId;
    uint64_t engineInstanceId = 1;
};

// Builds the bootstrap configuration for one serve-native process. The
// caller adds the live pressure, cancellation and metrics hooks.
[[nodiscard]] RuntimeBootstrapConfig nativeBootstrapConfig(
    const NativeArguments &arguments,
    const NativeProcessIdentity &process);

using ActualMemoryReporter =
    std::function<ActualMemoryReport(uint64_t estimatedWarmupPeakBytes)>;

// Complete owner returned only after the real loop has emitted its binary
// ReadyEvent. No partially warmed instance escapes start().
class RuntimeBootstrap final {
public:
    [[nodiscard]] static std::unique_ptr<RuntimeBootstrap> start(
        RuntimeBootstrapConfig config,
        NativeRuntime::ByteSink output,
        NativeRuntime::StatusProvider statusProvider);

    // Completes warmup and validation before announcing readiness.
    [[nodiscard]] static RuntimeBootstrapReport requireWarmupAndAnnounce(
        const EngineMemoryPlan &memoryPlan,
        model::RuntimeModel &modelRuntime,
        ActualMemoryReporter memoryReporter,
        NativeRuntime &nativeLoop);

    RuntimeBootstrap(const RuntimeBootstrap &) = delete;
    RuntimeBootstrap &operator=(const RuntimeBootstrap &) = delete;
    ~RuntimeBootstrap();

    [[nodiscard]] RuntimeResources &resources() noexcept {
        return *resources_;
    }
    [[nodiscard]] model::RuntimeModel &modelRuntime() noexcept { return *model_; }
    [[nodiscard]] NativeRuntime &nativeLoop() noexcept {
        return *nativeLoop_;
    }
    [[nodiscard]] const RuntimeBootstrapReport &report() const noexcept {
        return report_;
    }

private:
    RuntimeBootstrap(std::unique_ptr<RuntimeResources> resources,
                     std::unique_ptr<model::RuntimeModel> modelRuntime,
                     std::unique_ptr<NativeRuntime> nativeLoop,
                     RuntimeBootstrapReport report);

    // Reverse destruction order is loop -> modelRuntime -> resources.
    std::unique_ptr<RuntimeResources> resources_;
    std::unique_ptr<model::RuntimeModel> model_;
    std::unique_ptr<NativeRuntime> nativeLoop_;
    RuntimeBootstrapReport report_;
};

}  // namespace splash::engine
