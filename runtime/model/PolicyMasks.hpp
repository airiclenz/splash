#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace splash::model {

// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// A mask word with every vocabulary bit set: the row admits every token.
inline constexpr uint32_t kAllowAllMaskWord =
    std::numeric_limits<uint32_t>::max();

// ============================================================================
// Writes one lane's constraint-mask rows for a decode step.
//
// `laneRegion` is the lane's whole ConstraintMasks region, a whole number of
// `maskWords`-word rows. An unconstrained lane writes nothing: the sampling
// kernels read the masks only for lanes whose policy is constrained, so the
// region's stale contents are never observed. A constrained lane copies
// `masks` (whole rows) to the front of the region and sets every remaining row
// to all-ones, so no row of an earlier step survives; empty `masks` (an
// abandoned mask wait) leaves every row all-ones.
//
// Throws std::invalid_argument when `maskWords` is zero, the region or `masks`
// is not a whole number of rows, `masks` exceeds the region, or an
// unconstrained lane is handed masks.
inline void writeConstraintMasks(std::span<uint32_t> laneRegion,
                                 bool isConstrained,
                                 std::span<const uint32_t> masks,
                                 uint32_t maskWords) {
  if (!maskWords || laneRegion.size() % maskWords ||
      masks.size() % maskWords) {
    throw std::invalid_argument("constraint mask rows are malformed");
  }
  if (!isConstrained) {
    if (!masks.empty())
      throw std::invalid_argument("unconstrained lane was given masks");
    return;
  }
  if (masks.size() > laneRegion.size())
    throw std::invalid_argument("constraint mask exceeds decode arena");

  const auto tail = std::copy(masks.begin(), masks.end(), laneRegion.begin());
  std::fill(tail, laneRegion.end(), kAllowAllMaskWord);
}

} // namespace splash::model
