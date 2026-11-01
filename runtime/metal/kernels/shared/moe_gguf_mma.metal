// GGUF MoE experts for Apple7/8 (M1/M2) on the simdgroup MMA tile (kernels/common/gguf_mma_tile.h): the bindings,
// grid and outputs of moe_expert_gguf_m* (kernels/shared/moe_gguf.metal), whose MPP matmul2d runs far below the MMA
// rate there. A tile runs the 8-, 16- or 32-row MMAs that hold its live rows.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_mma_tile.h"
#include "metal/kernels/common/gguf_column_tile.h"
#include "metal/kernels/common/gguf_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"

using namespace gguf_mma;

template <ushort Rows, GgufEpilogue Ep, uint Shared>
kernel void moe_expert_gguf_mma(device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                                device const uint *tile_count [[buffer(2)]], device uchar *w0 [[buffer(3)]],
                                device uchar *w1 [[buffer(4)]], device uchar *meta [[buffer(5)]], device uchar *sw0 [[buffer(6)]],
                                device uchar *sw1 [[buffer(7)]], device uchar *smeta [[buffer(8)]],
                                device bfloat *output [[buffer(9)]], device bfloat *aux [[buffer(10)]],
                                device const uint *slots [[buffer(11)]], constant MoeGgufExpertParams &p [[buffer(12)]],
                                uint2 group [[threadgroup_position_in_grid]],
                                uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[GGUF_TILE_COLUMNS * kStride];
  threadgroup half2 tl[kQuantPairTableEntries];
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  const MoeGgufSegment s = moe_gguf_segment<Shared>(tile.expert, p, slots, w0, w1, meta, sw0, sw1, smeta);
  device bfloat *x = input + ulong(group.y) * Rows * p.input_size;
  const ulong out = ulong(group.y) * Rows * p.output_size;
  const uint origin = group.x * GGUF_TILE_COLUMNS + simd_group * kColumns, thread_index = simd_group * 32 + simd_lane;
  // A tile of at most MOE_GGUF_COLUMN_ROWS live rows runs the column tile (kernels/common/gguf_column_tile.h), as
  // moe_expert_gguf's does: a GEMV in registers without a stage or barrier, where the staged MMA tile's padded 8-row
  // steps and their per-step stage barrier bound a one- or two-row tile, not DRAM. A decode tile of a sparse route
  // (Qwen3.8-Flash-Next: ten of 512 experts a layer) is almost always such a tile. Q2_0 only: measured on M1 Max,
  // the column tile halves these tiles' time on Q2_0 but runs ~11% slower than the staged MMA on IQ3_XXS, whose
  // heavier weights and dequant outweigh the stage it skips; formats not measured stay on the staged MMA.
  if (moe_column_tile(tiles, group.y) && s.format == FmtQ20::Id) {
    float sums[MOE_GGUF_COLUMN_ROWS];
    column_dot_any<MOE_GGUF_COLUMN_ROWS>(s.format, x, tile.rows, p.input_size, s.w0, s.w1, s.meta, origin + simd_lane,
                                         0, p.input_size / 32, tl, thread_index, GGUF_STAGED_THREADS, sums);
    for (uint row = 0; row < tile.rows; ++row) {
      const ulong o = out + ulong(row) * p.output_size + origin + simd_lane;
      output[o] = gguf_epilogue<Ep>(sums[row], aux, o);
    }
    return;
  }
  const auto run = [&](auto fragments) {
    constexpr uint FM = decltype(fragments)::value;
    float2 acc[FN][FM];
    zero(acc);
    accumulate_any<FM>(s.format, x, s.w0, s.w1, s.meta, p.input_size, origin, stage + simd_group * kSimdgroupStage, tl,
                       thread_index, GGUF_STAGED_THREADS, simd_lane, 0, p.input_size / kStep, acc);
    elements(acc, sgmatrix::lane_map(simd_lane), [&](uint row, uint column, float v) {
      const ulong o = out + ulong(row) * p.output_size + origin + column;
      if constexpr (Ep == EpUpWithGate) v = float(bfloat(v)) * splash_silu(float(aux[o]));
      output[o] = bfloat(v);
    });
  };
  if constexpr (Rows == 8) run(integral_constant<uint, 1>{});
  else if (tile.rows <= 8) run(integral_constant<uint, 1>{});
  else if (tile.rows <= 16) run(integral_constant<uint, 2>{});
  else run(integral_constant<uint, 4>{});
}
using MoeExpertGgufMmaKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *, device uchar *,
                                    device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                    device bfloat *, device bfloat *, device const uint *, constant MoeGgufExpertParams &,
                                    uint2, uint, uint);
template [[host_name("moe_expert_gguf_mma_m8_a")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<8, EpNone, 1>;
template [[host_name("moe_expert_gguf_mma_m8_g")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<8, EpUpWithGate, 1>;
template [[host_name("moe_expert_gguf_mma_m32_a")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<32, EpNone, 1>;
template [[host_name("moe_expert_gguf_mma_m32_g")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<32, EpUpWithGate, 1>;
// Flint's shared expert of two slices (CompiledMoeRouting "_s2").
template [[host_name("moe_expert_gguf_mma_m8_a_s2")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<8, EpNone, 2>;
template [[host_name("moe_expert_gguf_mma_m8_g_s2")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<8, EpUpWithGate, 2>;
template [[host_name("moe_expert_gguf_mma_m32_a_s2")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<32, EpNone, 2>;
template [[host_name("moe_expert_gguf_mma_m32_g_s2")]] kernel MoeExpertGgufMmaKernel moe_expert_gguf_mma<32, EpUpWithGate, 2>;
