#include "model/PolicyMasks.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using namespace splash::model;

// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// Rows per lane region: the anchor plus eight verify rows.
constexpr uint32_t kStepRows = 9;

// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// Words per mask row in these tests.
constexpr uint32_t kMaskWords = 4;

// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// A word no write produces, marking contents left from an earlier step.
constexpr uint32_t kStaleWord = 0xDEADBEEFU;

// ============================================================================
// Throws with `message` when `condition` is false.
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

// ============================================================================
// A lane region holding only stale words.
std::vector<uint32_t> staleRegion() {
  return std::vector<uint32_t>(uint64_t{kStepRows} * kMaskWords, kStaleWord);
}

// ============================================================================
// `rows` mask rows whose words are distinct from both stale and all-ones.
std::vector<uint32_t> maskRows(uint32_t rows) {
  std::vector<uint32_t> masks(uint64_t{rows} * kMaskWords);
  for (uint32_t index = 0; index < masks.size(); ++index)
    masks[index] = index + 1;
  return masks;
}

// ============================================================================
// True when every word from `begin` on admits every token.
bool allOnesFrom(const std::vector<uint32_t> &region, uint64_t begin) {
  for (uint64_t index = begin; index < region.size(); ++index) {
    if (region[index] != kAllowAllMaskWord)
      return false;
  }
  return true;
}

// ============================================================================
// True when `call` throws std::invalid_argument.
template <class Call> bool rejects(Call call) {
  try {
    call();
  } catch (const std::invalid_argument &) {
    return true;
  }
  return false;
}

// ============================================================================
void writeConstraintMasks_Unconstrained_WritesNothing() {
  std::vector<uint32_t> region = staleRegion();

  writeConstraintMasks(region, false, {}, kMaskWords);

  require(region == staleRegion(), "unconstrained lane wrote mask words");
}

// ============================================================================
void writeConstraintMasks_SuppliedRows_CopiesThenFillsOnes() {
  std::vector<uint32_t> region = staleRegion();
  const std::vector<uint32_t> masks = maskRows(3);

  writeConstraintMasks(region, true, masks, kMaskWords);

  require(std::equal(masks.begin(), masks.end(), region.begin()),
          "supplied mask rows were not copied to the front");
  require(allOnesFrom(region, masks.size()),
          "rows after the supplied masks are not all-ones");
}

// ============================================================================
void writeConstraintMasks_Abandoned_WritesAllOnesRows() {
  std::vector<uint32_t> region = staleRegion();

  writeConstraintMasks(region, true, {}, kMaskWords);

  require(allOnesFrom(region, 0), "abandoned lane rows are not all-ones");
}

// ============================================================================
void writeConstraintMasks_ShorterStep_LeavesNoEarlierRow() {
  std::vector<uint32_t> region = staleRegion();
  const std::vector<uint32_t> full = maskRows(kStepRows);
  const std::vector<uint32_t> single = maskRows(1);

  writeConstraintMasks(region, true, full, kMaskWords);
  writeConstraintMasks(region, true, single, kMaskWords);

  require(std::equal(single.begin(), single.end(), region.begin()) &&
              allOnesFrom(region, single.size()),
          "an earlier step's mask rows leaked into a shorter step");
}

// ============================================================================
void writeConstraintMasks_InvalidRows_Rejected() {
  std::vector<uint32_t> region = staleRegion();
  const std::vector<uint32_t> tooMany = maskRows(kStepRows + 1);
  const std::vector<uint32_t> partial(kMaskWords + 1, 1);
  const std::vector<uint32_t> one = maskRows(1);

  require(rejects([&] {
            writeConstraintMasks(region, true, tooMany, kMaskWords);
          }),
          "masks larger than the region were accepted");
  require(rejects([&] {
            writeConstraintMasks(region, true, partial, kMaskWords);
          }),
          "a partial mask row was accepted");
  require(rejects([&] { writeConstraintMasks(region, true, one, 0); }),
          "zero mask words were accepted");
  require(rejects([&] {
            writeConstraintMasks(region, false, one, kMaskWords);
          }),
          "an unconstrained lane accepted masks");
  require(region == staleRegion(), "a rejected write changed the region");
}

} // namespace

int main() {
  try {
    writeConstraintMasks_Unconstrained_WritesNothing();
    writeConstraintMasks_SuppliedRows_CopiesThenFillsOnes();
    writeConstraintMasks_Abandoned_WritesAllOnesRows();
    writeConstraintMasks_ShorterStep_LeavesNoEarlierRow();
    writeConstraintMasks_InvalidRows_Rejected();
    std::cout << "policy masks tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "policy masks test failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
