#include "metal/kernels/common/paged_attention_sgf.h"
#include "metal/kernels/common/paged_attention_tile.h"

// Prefill runs the shared device-operand page loop
// (splash_paged_attention_tile) over eight query rows of one KV head.

// Prefill split: KV heads vary first, then query tiles, then history splits.
template <uint KVHeads, uint QueryHeadsPerKVHead, typename CacheElement>
inline void splash_prefill_attention_split_phase(
    device bfloat *queries, device float *partials,
    device float *statistics, device const SplashKvPage *page_table,
    constant SplashPrefillAttentionParams &params,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale, uint3 group,
    uint thread_index) {
  const uint split = group.z;
  const SplashAttentionTile tile =
      splash_prefill_attention_tile<KVHeads, QueryHeadsPerKVHead>(params, group.x, group.y);
  if (!tile.active || split >= tile.splits)
    return;
  splash_paged_attention_tile<KVHeads, QueryHeadsPerKVHead,
                              SPLASH_PREFILL_ATTENTION_TILE_ROWS, CacheElement>(
      queries + tile.offset, page_table, params.kv, group.x, tile.first_position,
      tile.active_rows, tile.splits, split, partials, statistics,
      tile.first_slot + split, scores, probabilities, row_max, row_sum,
      previous_scale, rescale, thread_index);
}

template <uint KVHeads, uint QueryHeadsPerKVHead>
inline void splash_prefill_attention_reduce_phase(
    device const float *partials, device const float *statistics,
    device bfloat *output,
    constant SplashPrefillAttentionParams &params, uint3 group,
    uint thread_index, threadgroup float *weights,
    threadgroup float *group_values) {
  constexpr ushort M = SPLASH_PREFILL_ATTENTION_TILE_ROWS * QueryHeadsPerKVHead;
  constexpr ushort D = SplashKvHeadDimension;
  const uint fused_row = group.y;
  const SplashAttentionTile tile =
      splash_prefill_attention_tile<KVHeads, QueryHeadsPerKVHead>(params, group.x, group.z);
  if (!tile.active || fused_row >= M || thread_index >= D)
    return;
  splash_attention_reduce_row<QueryHeadsPerKVHead,
                              SPLASH_PREFILL_ATTENTION_TILE_ROWS>(
      partials, statistics, output + tile.offset, tile.first_position,
      tile.active_rows, tile.splits, tile.first_slot, fused_row, thread_index,
      weights, group_values);
}

#define PAGED_PREFILL_SPLIT(Name, Heads, Group, CacheElement)                  \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const SplashKvPage *page_table [[buffer(3)]],                     \
      constant SplashPrefillAttentionParams &params [[buffer(4)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint M = Group * SPLASH_PREFILL_ATTENTION_TILE_ROWS;             \
    constexpr uint N = SplashKvPageTokens;                                     \
    alignas(16) threadgroup float scores[M * N];                               \
    alignas(16) threadgroup bfloat probabilities[M * N];                       \
    threadgroup float row_max[M];                                              \
    threadgroup float row_sum[M];                                              \
    threadgroup float previous_scale[M];                                       \
    threadgroup atomic_uint rescale;                                           \
    splash_prefill_attention_split_phase<Heads, Group, CacheElement>(          \
        queries, partials, statistics, page_table, params, scores,             \
        probabilities, row_max, row_sum, previous_scale, &rescale, group,      \
        thread_index);                                                         \
  }

PAGED_PREFILL_SPLIT(prefill_attention_q8_split, 4, 6, int8_t)
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv2_g8, 2, 8, int8_t)
// BF16 shares the page loop and reduction, without quantization scales.
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split, 4, 6, bfloat)
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv2_g8, 2, 8, bfloat)
#undef PAGED_PREFILL_SPLIT

// Apple7/8 register tile (paged_attention_sgf.h): the bindings and grid of
// the split above with 32 x G threads, one simdgroup per eight fused rows.
// Slots, partials and statistics are the MPP tile's, so the reduce below
// combines them unchanged.
#define SGF_PREFILL_SPLIT(Name, Heads, Group, CacheElement)                    \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const SplashKvPage *page_table [[buffer(3)]],                     \
      constant SplashPrefillAttentionParams &params [[buffer(4)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint lane [[thread_index_in_simdgroup]]) {                               \
    threadgroup uint query_stage[Group * 32 * 32];                             \
    constexpr uint D = SplashKvHeadDimension;                                  \
    const uint kv_head = group.x, tile = group.y, split = group.z;             \
    const uint tile_start = tile * SplashPrefillTileRows;                      \
    if (!splash_prefill_attention_contract_valid(params) ||                    \
        kv_head >= Heads || split >= params.split_count ||                     \
        tile_start >= params.rows || sg >= Group)                              \
      return;                                                                  \
    const uint active_rows =                                                   \
        min(SplashPrefillTileRows, params.rows - tile_start);                  \
    const ulong tile_offset =                                                  \
        (ulong(kv_head) * params.chunk_stride + tile_start) * Group * D;       \
    const ulong slot =                                                         \
        (ulong(tile) * Heads + kv_head) * params.split_count + split;          \
    q8sgf::tile<Heads, Group, CacheElement>(                                   \
        queries + tile_offset, page_table, params.kv, kv_head,                 \
        params.committed_tokens + tile_start, active_rows, params.split_count, \
        split, partials, statistics, slot, sg, lane, query_stage);                          \
  }

SGF_PREFILL_SPLIT(prefill_attention_q8_split_sgf, 4, 6, int8_t)
SGF_PREFILL_SPLIT(prefill_attention_q8_split_sgf_kv2_g8, 2, 8, int8_t)
SGF_PREFILL_SPLIT(prefill_attention_bf16_split_sgf, 4, 6, bfloat)
SGF_PREFILL_SPLIT(prefill_attention_bf16_split_sgf_kv2_g8, 2, 8, bfloat)
#undef SGF_PREFILL_SPLIT

#define PAGED_PREFILL_REDUCE(Name, Heads, Group)                               \
  kernel void Name(                                                            \
      device const float *partials [[buffer(0)]],                              \
      device const float *statistics [[buffer(1)]],                            \
      device bfloat *output [[buffer(2)]],                                     \
      constant SplashPrefillAttentionParams &params [[buffer(3)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    threadgroup float weights[SplashPrefillMaximumSplits];                     \
    threadgroup float group_values[8];                                         \
    splash_prefill_attention_reduce_phase<Heads, Group>(                       \
        partials, statistics, output, params, group, thread_index, weights,    \
        group_values);                                                         \
  }
PAGED_PREFILL_REDUCE(prefill_attention_reduce, 4, 6)
PAGED_PREFILL_REDUCE(prefill_attention_reduce_kv2_g8, 2, 8)
#undef PAGED_PREFILL_REDUCE
