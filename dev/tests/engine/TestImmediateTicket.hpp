#pragma once

#include "engine/Types.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace splash::test {

class ImmediateTicket final : public ModelBatchTicket {
public:
  explicit ImmediateTicket(std::vector<ModelStepResult> results,
                           double wallMilliseconds = 0.0)
      : results_(std::move(results)), wallMilliseconds_(wallMilliseconds) {}

  [[nodiscard]] bool ready() const noexcept override { return true; }
  [[nodiscard]] std::vector<ModelStepResult> wait() override {
    return std::move(results_);
  }
  [[nodiscard]] double wallMilliseconds() const noexcept override {
    return wallMilliseconds_;
  }

private:
  std::vector<ModelStepResult> results_;
  double wallMilliseconds_ = 0.0;
};

inline std::unique_ptr<ModelBatchTicket>
immediateTicket(std::vector<ModelStepResult> results,
                const std::function<void()> &completion,
                double wallMilliseconds = 0.0) {
  if (completion)
    completion();
  return std::make_unique<ImmediateTicket>(std::move(results),
                                           wallMilliseconds);
}

} // namespace splash::test
