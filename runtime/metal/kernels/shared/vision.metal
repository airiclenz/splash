// Vision encoder kernels for the Qwen3.5 vision tower: pixel patchify,
// position/rope preparation, dense bf16 GEMMs with fused epilogues,
// LayerNorm, 2D-rope QKV preparation, non-causal flash attention, and the
// embedding injection into the language model's prefill hidden rows, plus
// the one-time zero fill of a new scratch arena.
//
// Conventions match the text-model kernels: 256 threads (8 simdgroups) per
// threadgroup, MPP cooperative-tensor matmuls with fp32 accumulation and
// hand-written epilogues, grid {row_tile, output_tile}. On Apple7/8, which
// have no bfloat arithmetic, the GEMMs run vision_gemm_sgf_* register tiles
// instead: the same tiles, grid and epilogues on simdgroup_float8x8 matrices
// with bf16 widened to fp32; attention likewise runs vision_attention_sgf, a
// register-matrix QK and PV with the same grid and layouts. Attention runs
// on the native head dimension (72); token counts are padded to the 128-key
// tile so K/V tiles read in full, with padded keys zeroed and masked by the
// softmax.
//
// Token order everywhere is spatial-merge-block-major: each consecutive group
// of four patches is one 2x2 spatial unit, so the merger reads contiguous
// rows and the merged token index is row-major over the merged grid.

#include "metal/abi/Vision.h"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant ushort kVisionHeads = 16;
constant ushort kVisionHeadDim = 72;
// Q/K are stored padded to 80 dims (matmul K extents must be multiples of
// 16); the zeroed tail contributes nothing to the dot products.
constant ushort kVisionQkDim = 80;
constant ushort kVisionHidden = 1152;
constant ushort kVisionPatch = 16;
constant ushort kVisionMerge = 2;
constant ushort kVisionPatchDim = 1536;  // 3 channels x 2 temporal x 16 x 16
constant ushort kVisionGridSide = 48;    // learned position table is 48 x 48
constant float kVisionRopeTheta = 10000.0f;

// Block-major (row, col) of one patch token.
inline uint2 vision_patch_position(uint token, uint grid_width) {
    uint blocks_w = grid_width / kVisionMerge;
    uint block = token / (kVisionMerge * kVisionMerge);
    uint inner = token % (kVisionMerge * kVisionMerge);
    return uint2((block / blocks_w) * kVisionMerge + inner / kVisionMerge,
                 (block % blocks_w) * kVisionMerge + inner % kVisionMerge);
}

// Patchify: uint8 RGB pixels (H, W, 3) -> bf16 patch rows (tokens, 1536) in
// block-major token order with per-patch feature order (channel, temporal,
// row, col). Normalization is mean 0.5 / std 0.5, and the single frame is
// duplicated across the temporal patch dimension.

kernel void vision_patchify(
    device const uchar *pixels [[buffer(0)]],
    device bfloat *patches [[buffer(1)]],
    constant VisionGridParams &params [[buffer(2)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    uint2 position = vision_patch_position(token, params.grid_width);
    uint image_width = params.grid_width * kVisionPatch;
    device bfloat *row = patches + ulong(token) * kVisionPatchDim;
    for (uint feature = thread_index; feature < kVisionPatchDim;
         feature += 256) {
        uint channel = feature / (2 * kVisionPatch * kVisionPatch);
        uint within = feature % (kVisionPatch * kVisionPatch);
        uint patch_row = within / kVisionPatch;
        uint patch_col = within % kVisionPatch;
        ulong pixel = (ulong(position.x * kVisionPatch + patch_row) *
                           image_width +
                       position.y * kVisionPatch + patch_col) * 3 + channel;
        row[feature] = bfloat(float(pixels[pixel]) * (1.0f / 127.5f) - 1.0f);
    }
}

// Grid-derived inputs: the bilinear (align-corners) resample of the learned
// 48x48 position table into bf16 position rows, and the 2D rope tables with
// 18 frequencies each for the row and column coordinate, duplicated across
// both rotation halves.

inline void vision_bilinear_taps(uint index, uint size, thread uint *tap,
                                 thread float *weight) {
    float source = size > 1
        ? float(index) * float(kVisionGridSide - 1) / float(size - 1)
        : 0.0f;
    float lower = floor(source);
    for (uint offset = 0; offset < 2; ++offset) {
        float position = lower + float(offset);
        tap[offset] = uint(clamp(position, 0.0f, float(kVisionGridSide - 1)));
        weight[offset] = max(1.0f - fabs(source - position), 0.0f);
    }
}

kernel void vision_prepare_positions(
    device const bfloat *table [[buffer(0)]],
    device bfloat *positions [[buffer(1)]],
    device float *rope_cos [[buffer(2)]],
    device float *rope_sin [[buffer(3)]],
    constant VisionGridParams &params [[buffer(4)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    uint2 position = vision_patch_position(token, params.grid_width);
    uint row_taps[2], col_taps[2];
    float row_weights[2], col_weights[2];
    vision_bilinear_taps(position.x, params.grid_height, row_taps, row_weights);
    vision_bilinear_taps(position.y, params.grid_width, col_taps, col_weights);

    device bfloat *out = positions + ulong(token) * kVisionHidden;
    for (uint dim = thread_index; dim < kVisionHidden; dim += 256) {
        float blended = 0.0f;
        for (uint h = 0; h < 2; ++h) {
            for (uint w = 0; w < 2; ++w) {
                blended += row_weights[h] * col_weights[w] *
                    float(table[(ulong(row_taps[h]) * kVisionGridSide +
                                 col_taps[w]) * kVisionHidden + dim]);
            }
        }
        out[dim] = bfloat(blended);
    }

    if (thread_index < kVisionHeadDim) {
        constexpr uint frequencies = kVisionHeadDim / 4;  // 18
        uint frequency = thread_index % frequencies;
        bool column = (thread_index / frequencies) & 1;
        float inverse = pow(kVisionRopeTheta,
                            -float(2 * frequency) / float(kVisionHeadDim / 2));
        float angle = float(column ? position.y : position.x) * inverse;
        rope_cos[ulong(token) * kVisionHeadDim + thread_index] = cos(angle);
        rope_sin[ulong(token) * kVisionHeadDim + thread_index] = sin(angle);
    }
}

// Dense bf16 GEMM: output[M, N] = input[M, K] * weights[N, K]^T + bias[N]

enum VisionActivation { VisionNone, VisionGeluTanh, VisionGeluErf };

// Abramowitz & Stegun 7.1.26 (max error ~1.5e-7, far below bf16 rounding).
inline float vision_erf(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    x = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float polynomial = t * (0.254829592f + t * (-0.284496736f +
        t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    return sign * (1.0f - polynomial * precise::exp(-x * x));
}

// The GEMM epilogue's activation on one fp32 output value.
template <VisionActivation Activation>
inline float vision_activate(float value) {
    if constexpr (Activation == VisionGeluTanh) {
        return 0.5f * value *
            (1.0f + precise::tanh(
                0.7978845608028654f *
                (value + 0.044715f * value * value * value)));
    } else if constexpr (Activation == VisionGeluErf) {
        return 0.5f * value *
            (1.0f + vision_erf(value * 0.7071067811865476f));
    }
    return value;
}

template <ushort TileM, ushort TileN, VisionActivation Activation,
          bool AddResidual>
inline void vision_gemm_tile(
    device bfloat *input,
    device bfloat *weights,
    device bfloat *bias,
    device bfloat *output,
    device bfloat *residual,
    uint output_size,
    uint input_size,
    uint output_origin,
    uint row_origin)
{
    device bfloat *rows = input + ulong(row_origin) * input_size;
    device bfloat *tile_weights = weights + ulong(output_origin) * input_size;
    auto a = tensor(rows,
                    dextents<int, 2>{int(input_size), TileM},
                    array<int, 2>{1, int(input_size)});
    auto b = tensor(tile_weights,
                    dextents<int, 2>{int(input_size), TileN},
                    array<int, 2>{1, int(input_size)});
    auto c = tensor(output + ulong(row_origin) * output_size,
                    dextents<int, 2>{int(output_size), TileM},
                    array<int, 2>{1, int(output_size)});
    // Static BK chunks with multiply_accumulate mode: static extents avoid
    // per-load bounds checks, accumulation stays in the destination
    // cooperative tensor, and a mem_none barrier per chunk keeps simdgroups
    // cache-adjacent along K.
    constexpr ushort BK = 128;
    constexpr auto descriptor = matmul2d_descriptor(
        TileM, TileN, BK, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<8>> operation;
    auto a0 = a.template slice<BK, TileM>(0, 0);
    auto b0 = b.template slice<BK, TileN>(0, 0);
    auto accumulated = operation.template get_destination_cooperative_tensor<
        decltype(a0), decltype(b0), float>();
    #pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        accumulated[i] = 0.0f;
    }
    for (uint chunk = 0; chunk < input_size / BK; ++chunk) {
        auto a_slice = a.template slice<BK, TileM>(chunk * BK, 0);
        auto b_slice = b.template slice<BK, TileN>(chunk * BK, 0);
        operation.run(a_slice, b_slice, accumulated);
        threadgroup_barrier(mem_flags::mem_none);
    }

    auto converted = operation.template get_destination_cooperative_tensor<
        decltype(a0), decltype(b0), bfloat>();
    #pragma unroll
    for (ushort i = 0; i < accumulated.get_capacity(); ++i) {
        auto index = accumulated.get_multidimensional_index(i);
        float value = vision_activate<Activation>(
            accumulated[i] + float(bias[output_origin + index[0]]));
        if constexpr (AddResidual) {
            value += float(residual[
                ulong(row_origin + index[1]) * output_size +
                output_origin + index[0]]);
        }
        converted[i] = bfloat(value);
    }
    converted.store(c.slice<TileN, TileM>(output_origin, 0));
}

#define VISION_GEMM_KERNEL(name, tile_m, tile_n, activation, add_residual)    \
kernel void name(                                                             \
    device bfloat *input [[buffer(0)]],                                       \
    device bfloat *weights [[buffer(1)]],                                     \
    device bfloat *bias [[buffer(2)]],                                        \
    device bfloat *output [[buffer(3)]],                                      \
    device bfloat *residual [[buffer(4)]],                                    \
    constant VisionGemmParams &params [[buffer(5)]],                          \
    uint2 group [[threadgroup_position_in_grid]])                             \
{                                                                             \
    vision_gemm_tile<tile_m, tile_n, activation, add_residual>(               \
        input, weights, bias, output, residual,                               \
        params.output_size, params.input_size,                                \
        group.y * tile_n, group.x * tile_m);                                  \
}

VISION_GEMM_KERNEL(vision_gemm_m64n128, 64, 128, VisionNone, false)
VISION_GEMM_KERNEL(vision_gemm_m64n128_residual, 64, 128, VisionNone, true)
VISION_GEMM_KERNEL(vision_gemm_m64n128_gelu_tanh, 64, 128, VisionGeluTanh, false)
// The merger's 256-divisible output sizes use the wider tile.
VISION_GEMM_KERNEL(vision_gemm_m32n256, 32, 256, VisionNone, false)
VISION_GEMM_KERNEL(vision_gemm_m32n256_gelu_erf, 32, 256, VisionGeluErf,
                   false)

// Apple7/8 register tile: vision_gemm_tile's output, grid and epilogue on
// simdgroup_float8x8 matrices, for GPUs without bfloat arithmetic. bf16
// operands widen exactly to fp32 (a 16-bit shift) and accumulate in fp32.
// The eight simdgroups each own a 32 x 32 output block, 4 x 4 fragments of
// 8 x 8, laid out (TileM / 32) x (TileN / 32) over the tile. Operands load
// straight from device memory into each lane's fragment registers: no
// threadgroup memory and no barriers.
//
// K advances in 32-deep chunks of four 8-deep steps over a permuted K order:
// fragment depth d of step s is k = 4 d + s within the chunk, so each lane
// reads whole 8-byte runs: 8 consecutive input values of its row (depths fn
// and fn + 1 of every step) and 4 consecutive weight values of each of its two
// columns (depth fm of every step). Fragments are rebuilt from float2 lane
// values for every multiply: holding accumulators or operands in arrays of
// simdgroup matrices measured several times slower on Apple8.
constant ushort kVisionRegisterFragments = 4;  // 8x8 fragments per side
constant ushort kVisionRegisterSteps = 4;      // 8-deep steps per chunk
constant ushort kVisionRegisterDepth = 8 * kVisionRegisterSteps;

// Value `step` of four packed bf16, widened to fp32.
inline float vision_widen(uint2 packed, ushort step) {
    const uint word = packed[step >> 1];
    return as_type<float>((step & 1) ? (word & 0xFFFF0000u) : (word << 16));
}

template <ushort TileM, ushort TileN, VisionActivation Activation,
          bool AddResidual>
inline void vision_gemm_register_tile(
    device bfloat *input,
    device bfloat *weights,
    device bfloat *bias,
    device bfloat *output,
    device bfloat *residual,
    uint output_size,
    uint input_size,
    uint output_origin,
    uint row_origin,
    ushort simd_group,
    ushort simd_lane)
{
    constexpr ushort F = kVisionRegisterFragments;
    constexpr ushort SimdgroupsN = TileN / (8 * F);
    static_assert((TileM / (8 * F)) * SimdgroupsN == 8,
                  "eight 32 x 32 simdgroup blocks cover the tile");
    // simdgroup_matrix lane layout: thread_elements() are M[fm][fn], M[fm][fn + 1].
    const ushort quad = simd_lane >> 2;
    const ushort fm = (quad & 4) | ((simd_lane >> 1) & 3);
    const ushort fn = ((quad & 2) << 1) | ((simd_lane & 1) << 1);
    const uint first_row = row_origin + (simd_group / SimdgroupsN) * 8 * F;
    const uint first_column =
        output_origin + (simd_group % SimdgroupsN) * 8 * F;

    device const bfloat *lane_input[F];
    device const bfloat *lane_weights[F];
    #pragma unroll
    for (ushort i = 0; i < F; ++i) {
        lane_input[i] = input + ulong(first_row + i * 8 + fm) * input_size +
            kVisionRegisterSteps * fn;
        lane_weights[i] = weights +
            ulong(first_column + i * 8 + fn) * input_size +
            kVisionRegisterSteps * fm;
    }

    float2 accumulated[F][F];
    #pragma unroll
    for (ushort i = 0; i < F; ++i) {
        #pragma unroll
        for (ushort j = 0; j < F; ++j) {
            accumulated[i][j] = float2(0.0f);
        }
    }
    for (uint k = 0; k < input_size; k += kVisionRegisterDepth) {
        // rows[i][h]: input row fragment i, depth fn + h of every step.
        // columns[j][h]: weight column fn + h of fragment j, depth fm.
        uint2 rows[F][2];
        uint2 columns[F][2];
        #pragma unroll
        for (ushort i = 0; i < F; ++i) {
            #pragma unroll
            for (ushort h = 0; h < 2; ++h) {
                rows[i][h] = *reinterpret_cast<device const uint2 *>(
                    lane_input[i] + k + kVisionRegisterSteps * h);
                columns[i][h] = *reinterpret_cast<device const uint2 *>(
                    lane_weights[i] + ulong(h) * input_size + k);
            }
        }
        #pragma unroll
        for (ushort step = 0; step < kVisionRegisterSteps; ++step) {
            float2 a[F];
            float2 b[F];
            #pragma unroll
            for (ushort i = 0; i < F; ++i) {
                a[i] = float2(vision_widen(rows[i][0], step),
                              vision_widen(rows[i][1], step));
                b[i] = float2(vision_widen(columns[i][0], step),
                              vision_widen(columns[i][1], step));
            }
            #pragma unroll
            for (ushort i = 0; i < F; ++i) {
                #pragma unroll
                for (ushort j = 0; j < F; ++j) {
                    simdgroup_float8x8 left, right, sum, result;
                    reinterpret_cast<thread float2 &>(
                        left.thread_elements()) = a[i];
                    reinterpret_cast<thread float2 &>(
                        right.thread_elements()) = b[j];
                    reinterpret_cast<thread float2 &>(
                        sum.thread_elements()) = accumulated[i][j];
                    simdgroup_multiply_accumulate(result, left, right, sum);
                    accumulated[i][j] = reinterpret_cast<thread float2 &>(
                        result.thread_elements());
                }
            }
        }
    }

    #pragma unroll
    for (ushort j = 0; j < F; ++j) {
        const uint column = first_column + j * 8 + fn;
        const float2 column_bias = float2(
            float(bias[column]), float(bias[column + 1]));
        #pragma unroll
        for (ushort i = 0; i < F; ++i) {
            const ulong offset =
                ulong(first_row + i * 8 + fm) * output_size + column;
            const float2 sum = accumulated[i][j] + column_bias;
            float2 value = float2(vision_activate<Activation>(sum.x),
                                  vision_activate<Activation>(sum.y));
            if constexpr (AddResidual) {
                value += float2(*reinterpret_cast<device const bfloat2 *>(
                    residual + offset));
            }
            *reinterpret_cast<device bfloat2 *>(output + offset) =
                bfloat2(value);
        }
    }
}

#define VISION_GEMM_REGISTER_KERNEL(name, tile_m, tile_n, activation,        \
                                    add_residual)                             \
kernel void name(                                                             \
    device bfloat *input [[buffer(0)]],                                       \
    device bfloat *weights [[buffer(1)]],                                     \
    device bfloat *bias [[buffer(2)]],                                        \
    device bfloat *output [[buffer(3)]],                                      \
    device bfloat *residual [[buffer(4)]],                                    \
    constant VisionGemmParams &params [[buffer(5)]],                          \
    uint2 group [[threadgroup_position_in_grid]],                             \
    ushort simd_lane [[thread_index_in_simdgroup]],                           \
    ushort simd_group [[simdgroup_index_in_threadgroup]])                     \
{                                                                             \
    vision_gemm_register_tile<tile_m, tile_n, activation, add_residual>(      \
        input, weights, bias, output, residual,                               \
        params.output_size, params.input_size,                                \
        group.y * tile_n, group.x * tile_m, simd_group, simd_lane);           \
}

VISION_GEMM_REGISTER_KERNEL(vision_gemm_sgf_m64n128, 64, 128, VisionNone,
                            false)
VISION_GEMM_REGISTER_KERNEL(vision_gemm_sgf_m64n128_residual, 64, 128,
                            VisionNone, true)
VISION_GEMM_REGISTER_KERNEL(vision_gemm_sgf_m64n128_gelu_tanh, 64, 128,
                            VisionGeluTanh, false)
VISION_GEMM_REGISTER_KERNEL(vision_gemm_sgf_m32n256, 32, 256, VisionNone,
                            false)
VISION_GEMM_REGISTER_KERNEL(vision_gemm_sgf_m32n256_gelu_erf, 32, 256,
                            VisionGeluErf, false)

// LayerNorm with bias (fp32 statistics), one threadgroup per row.

kernel void vision_layer_norm(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device const bfloat *bias [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant VisionNormParams &params [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup float reductions[8];
    device const bfloat *row = input + ulong(group) * params.width;
    device bfloat *out = output + ulong(group) * params.width;

    float sum = 0.0f;
    for (uint column = thread_index; column < params.width; column += 256) {
        sum += float(row[column]);
    }
    sum = simd_sum(sum);
    if (simd_lane == 0) reductions[simd_group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    #pragma unroll
    for (ushort i = 0; i < 8; ++i) total += reductions[i];
    float mean = total / float(params.width);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float squares = 0.0f;
    for (uint column = thread_index; column < params.width; column += 256) {
        float centered = float(row[column]) - mean;
        squares += centered * centered;
    }
    squares = simd_sum(squares);
    if (simd_lane == 0) reductions[simd_group] = squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float variance = 0.0f;
    #pragma unroll
    for (ushort i = 0; i < 8; ++i) variance += reductions[i];
    variance /= float(params.width);
    float inverse = rsqrt(variance + 1e-6f);

    for (uint column = thread_index; column < params.width; column += 256) {
        float normalized = (float(row[column]) - mean) * inverse;
        out[column] = bfloat(normalized * float(weight[column]) +
                             float(bias[column]));
    }
}

// QKV preparation: apply 2D rope to Q and K, scatter into per-head layouts.
// Q, K: (heads, padded_tokens, 80); V: (heads, key_tile, 128, 72). Padded
// tokens are written as zeros so full 128-key tiles are always readable and
// hold finite values under the softmax mask.

kernel void vision_qkv_prepare(
    device const bfloat *qkv [[buffer(0)]],
    device const float *rope_cos [[buffer(1)]],
    device const float *rope_sin [[buffer(2)]],
    device bfloat *queries [[buffer(3)]],
    device bfloat *keys [[buffer(4)]],
    device bfloat *values [[buffer(5)]],
    constant VisionQkvParams &params [[buffer(6)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    constexpr ushort half_dim = kVisionHeadDim / 2;
    const bool valid = token < params.tokens;
    device const bfloat *row =
        qkv + ulong(token) * 3 * kVisionHeads * kVisionHeadDim;
    device const float *cos_row = rope_cos + ulong(token) * kVisionHeadDim;
    device const float *sin_row = rope_sin + ulong(token) * kVisionHeadDim;
    uint key_tile = token / 128, within = token % 128;
    uint tiles = params.padded_tokens / 128;

    for (uint index = thread_index;
         index < uint(kVisionHeads) * kVisionQkDim; index += 256) {
        uint head = index / kVisionQkDim;
        uint dim = index % kVisionQkDim;
        ulong plane = (ulong(head) * params.padded_tokens + token) *
            kVisionQkDim + dim;
        if (!valid || dim >= kVisionHeadDim) {
            queries[plane] = bfloat(0.0f);
            keys[plane] = bfloat(0.0f);
            if (dim < kVisionHeadDim) {
                values[(((ulong(head) * tiles + key_tile) * 128) + within) *
                       kVisionHeadDim + dim] = bfloat(0.0f);
            }
            continue;
        }
        uint base = head * kVisionHeadDim + dim;
        uint rotated = head * kVisionHeadDim +
            (dim < half_dim ? dim + half_dim : dim - half_dim);
        float sign = dim < half_dim ? -1.0f : 1.0f;
        float cos_value = cos_row[dim];
        float sin_value = sin_row[dim];
        queries[plane] = bfloat(float(row[base]) * cos_value +
            sign * float(row[rotated]) * sin_value);
        keys[plane] = bfloat(float(row[kVisionHidden + base]) * cos_value +
            sign * float(row[kVisionHidden + rotated]) * sin_value);
        values[(((ulong(head) * tiles + key_tile) * 128) + within) *
               kVisionHeadDim + dim] =
            bfloat(float(row[2 * kVisionHidden + base]));
    }
}

// Non-causal flash attention over the per-head layout.
// Grid: {ceil(tokens / 64), heads}. Output: (heads, padded_tokens, 72).

kernel void vision_attention(
    device bfloat *queries [[buffer(0)]],
    device bfloat *keys [[buffer(1)]],
    device bfloat *values [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant VisionAttentionParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    constexpr ushort M = 64, N = 128, D = kVisionHeadDim;
    constexpr ushort DQ = kVisionQkDim;
    constexpr ushort RowsPerSimdgroup = M / 8;
    // Preserve fp32 QK scores through softmax. After every lane has read the
    // scores, the first half holds bf16 probabilities and the second half
    // exposes row scales to PV. Running softmax state stays in registers, so
    // the complete threadgroup allocation is exactly 32 KiB.
    threadgroup float score_storage[M * N];
    threadgroup bfloat *probabilities =
        reinterpret_cast<threadgroup bfloat *>(score_storage);
    threadgroup float *row_sum = score_storage + M * N / 2;
    threadgroup float *previous_scale = row_sum + M;
    float running_max[RowsPerSimdgroup];
    float running_sum[RowsPerSimdgroup];
    for (ushort i = 0; i < RowsPerSimdgroup; ++i) {
        running_max[i] = -INFINITY;
        running_sum[i] = 0.0f;
    }

    uint head = group.y;
    uint query_start = group.x * M;
    ulong qk_plane = ulong(head) * params.padded_tokens * DQ;
    device bfloat *q = queries + qk_plane + ulong(query_start) * DQ;
    device bfloat *k = keys + qk_plane;
    device bfloat *v = values + ulong(head) * D * params.padded_tokens;
    device bfloat *out = output +
        ulong(head) * params.padded_tokens * D + ulong(query_start) * D;
    uint context = params.tokens;
    uint tiles = (context + N - 1) / N;

    auto qt = tensor(q, dextents<int, 2>{DQ, M}, array<int, 2>{1, DQ});
    constexpr auto qk_descriptor =
        matmul2d_descriptor(M, N, DQ, false, true, false);
    matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
    auto q0 = qt.slice<DQ, M>(0, 0);

    auto pt = tensor(probabilities, dextents<int, 2>{N, M},
                     array<int, 2>{1, N});
    auto p0 = pt.slice<N, M>(0, 0);
    auto st = tensor(score_storage, dextents<int, 2>{N, M},
                     array<int, 2>{1, N});
    auto s0 = st.slice<N, M>(0, 0);
    auto first_v = tensor(v, dextents<int, 2>{D, N},
                          array<int, 2>{1, D});
    auto first_v0 = first_v.slice<D, N>(0, 0);
    constexpr auto pv_descriptor =
        matmul2d_descriptor(M, D, N, false, false, false);
    matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
    auto running = pv.template get_destination_cooperative_tensor<
        decltype(p0), decltype(first_v0), float>();
    for (ushort i = 0; i < running.get_capacity(); ++i) running[i] = 0.0f;
    for (uint tile = 0; tile < tiles; ++tile) {
        device bfloat *tile_k = k + ulong(tile) * N * DQ;
        auto kt = tensor(tile_k, dextents<int, 2>{DQ, N},
                         array<int, 2>{1, DQ});
        auto k0 = kt.slice<DQ, N>(0, 0);
        auto scores = qk.template get_destination_cooperative_tensor<
            decltype(q0), decltype(k0), float>();
        qk.run(q0, k0, scores);
        scores.store(s0);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each simdgroup owns eight rows. Phase one moves scores into
        // registers and computes the running softmax state; after one
        // barrier, phase two overwrites the (aliased) storage with bf16
        // probabilities.
        float exponents[RowsPerSimdgroup][4];
        float scales[RowsPerSimdgroup];
        for (ushort i = 0; i < RowsPerSimdgroup; ++i) {
            ushort row = simd_group * RowsPerSimdgroup + i;
            float values_local[4];
            float tile_max = -INFINITY;
            for (ushort j = 0; j < 4; ++j) {
                uint column = simd_lane + j * 32;
                uint key_position = tile * N + column;
                float score = score_storage[row * N + column] * params.scale;
                values_local[j] =
                    key_position < context ? score : -INFINITY;
                tile_max = max(tile_max, values_local[j]);
            }
            tile_max = simd_max(tile_max);
            float next_max = max(running_max[i], tile_max);
            float scale = fast::exp(running_max[i] - next_max);
            float tile_sum = 0.0f;
            for (ushort j = 0; j < 4; ++j) {
                exponents[i][j] = fast::exp(values_local[j] - next_max);
                tile_sum += exponents[i][j];
            }
            tile_sum = simd_sum(tile_sum);
            scales[i] = scale;
            running_sum[i] = running_sum[i] * scale + tile_sum;
            running_max[i] = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort i = 0; i < RowsPerSimdgroup; ++i) {
            ushort row = simd_group * RowsPerSimdgroup + i;
            if (simd_lane == 0) {
                previous_scale[row] = scales[i];
                row_sum[row] = running_sum[i];
            }
            for (ushort j = 0; j < 4; ++j) {
                uint column = simd_lane + j * 32;
                probabilities[row * N + column] = bfloat(exponents[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device bfloat *tile_v = v + ulong(tile) * N * D;
        auto vt = tensor(tile_v, dextents<int, 2>{D, N},
                         array<int, 2>{1, D});
        auto v0 = vt.slice<D, N>(0, 0);
        auto partial_output = pv.template get_destination_cooperative_tensor<
            decltype(p0), decltype(v0), float>();
        pv.run(p0, v0, partial_output);
        for (ushort i = 0; i < running.get_capacity(); ++i) {
            auto index = running.get_multidimensional_index(i);
            running[i] = running[i] * previous_scale[index[1]] +
                partial_output[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto ot = tensor(out, dextents<int, 2>{D, M}, array<int, 2>{1, D});
    auto converted = pv.template get_destination_cooperative_tensor<
        decltype(p0), decltype(first_v0), bfloat>();
    for (ushort i = 0; i < running.get_capacity(); ++i) {
        auto index = running.get_multidimensional_index(i);
        converted[i] = bfloat(running[i] / row_sum[index[1]]);
    }
    converted.store(ot.slice<D, M>(0, 0));
}

// Apple7/8 register attention: vision_attention's grid, layouts and output on
// simdgroup_float8x8 matrices, for GPUs without bfloat arithmetic. bf16
// operands widen exactly to fp32 and every product accumulates in fp32. A
// lane holds two columns of one row of every 8 x 8 fragment, so the score
// fragment of keys 8j..8j+7 is, after exponentiation, the probability operand
// of PV depth step j as it stands, and a row reduction is two shuffles (the
// row's four lanes differ in lane bits 0 and 3). Probabilities stay fp32 into
// PV, where the MPP kernel rounds them to bf16.
//
// The eight simdgroups pair up on 16 query rows each: one runs an online
// softmax over the even 64-key blocks, the other over the odd ones, so every
// key and value fragment a lane loads feeds two row fragments. The odd half
// hands its maximum, sum and output to the even half through threadgroup
// memory once, after its last block. K and V load straight from device
// memory into fragment registers.
//
// QK depth advances in 16-deep chunks of two 8-deep steps over a permuted
// order: fragment depth d of step s is k = 2 d + s within the chunk, so a lane
// reads four consecutive query values (depths fn and fn + 1 of both steps)
// and two consecutive values of each of its two keys (depth fm).

// A += left * right on 8 x 8 fragments given as each lane's two elements.
inline void vision_fragment_multiply_accumulate(thread float2 &accumulated,
                                                float2 left, float2 right) {
    simdgroup_float8x8 left_matrix, right_matrix, sum, result;
    reinterpret_cast<thread float2 &>(left_matrix.thread_elements()) = left;
    reinterpret_cast<thread float2 &>(right_matrix.thread_elements()) = right;
    reinterpret_cast<thread float2 &>(sum.thread_elements()) = accumulated;
    simdgroup_multiply_accumulate(result, left_matrix, right_matrix, sum);
    accumulated = reinterpret_cast<thread float2 &>(result.thread_elements());
}

kernel void vision_attention_sgf(
    device const bfloat *queries [[buffer(0)]],
    device const bfloat *keys [[buffer(1)]],
    device const bfloat *values [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant VisionAttentionParams &params [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simd_group [[simdgroup_index_in_threadgroup]])
{
    constexpr ushort M = 64, D = kVisionHeadDim, DQ = kVisionQkDim;
    constexpr ushort RowFragments = 2;  // 8-row fragments per simdgroup
    constexpr ushort RowGroups = M / (8 * RowFragments);
    constexpr ushort Halves = 2;        // simdgroups sharing one row group
    constexpr ushort BlockKeys = 64;    // keys per online-softmax step
    constexpr ushort KeyFragments = BlockKeys / 8;
    constexpr ushort OutputFragments = D / 8;
    constexpr ushort ChunkSteps = 2;
    constexpr ushort ChunkDepth = 8 * ChunkSteps;
    constexpr ushort Chunks = DQ / ChunkDepth;
    static_assert(RowGroups * Halves == 8, "eight simdgroups cover the tile");
    static_assert(DQ % ChunkDepth == 0 && D % 8 == 0,
                  "QK depth and head dimension tile into whole fragments");
    // The odd half's state, in lane order: 18 KiB + 2 KiB.
    threadgroup float2 handed_output[RowGroups][RowFragments][OutputFragments]
                                    [32];
    threadgroup float handed_max[RowGroups][RowFragments][32];
    threadgroup float handed_sum[RowGroups][RowFragments][32];

    // simdgroup_matrix lane layout: thread_elements() are M[fm][fn], M[fm][fn + 1].
    const ushort quad = simd_lane >> 2;
    const ushort fm = (quad & 4) | ((simd_lane >> 1) & 3);
    const ushort fn = ((quad & 2) << 1) | ((simd_lane & 1) << 1);
    const ushort row_group = simd_group % RowGroups;
    const ushort key_half = simd_group / RowGroups;
    const uint head = group.y;
    const uint first_row = group.x * M + row_group * 8 * RowFragments + fm;
    const ulong qk_plane = ulong(head) * params.padded_tokens * DQ;
    const ulong d_plane = ulong(head) * params.padded_tokens * D;
    // Scores in log2 units, so the softmax runs on exp2.
    const float scale = params.scale * M_LOG2E_F;
    const uint context = params.tokens;
    const uint blocks = (context + BlockKeys - 1) / BlockKeys;

    uint2 query_words[RowFragments][Chunks];
    #pragma unroll
    for (ushort r = 0; r < RowFragments; ++r) {
        device const bfloat *lane_query = queries + qk_plane +
            ulong(first_row + r * 8) * DQ + 2 * fn;
        #pragma unroll
        for (ushort c = 0; c < Chunks; ++c) {
            query_words[r][c] = *reinterpret_cast<device const uint2 *>(
                lane_query + c * ChunkDepth);
        }
    }
    device const bfloat *lane_keys = keys + qk_plane + ulong(fn) * DQ + 2 * fm;
    device const bfloat *lane_values = values + d_plane + ulong(fm) * D + fn;

    // running_sum is this lane's share of its row's sum; the row's lanes
    // share the running maximum, so the shares add up at the end.
    float running_max[RowFragments];
    float running_sum[RowFragments];
    float2 accumulated[RowFragments][OutputFragments];
    #pragma unroll
    for (ushort r = 0; r < RowFragments; ++r) {
        running_max[r] = -INFINITY;
        running_sum[r] = 0.0f;
        #pragma unroll
        for (ushort c = 0; c < OutputFragments; ++c) {
            accumulated[r][c] = float2(0.0f);
        }
    }
    for (uint block = key_half; block < blocks; block += Halves) {
        device const bfloat *block_keys =
            lane_keys + ulong(block) * BlockKeys * DQ;
        float2 scores[RowFragments][KeyFragments];
        #pragma unroll
        for (ushort r = 0; r < RowFragments; ++r) {
            #pragma unroll
            for (ushort j = 0; j < KeyFragments; ++j) {
                scores[r][j] = float2(0.0f);
            }
        }
        #pragma unroll
        for (ushort c = 0; c < Chunks; ++c) {
            // key_words[j]: keys fn and fn + 1 of fragment j, depth fm.
            uint2 key_words[KeyFragments];
            #pragma unroll
            for (ushort j = 0; j < KeyFragments; ++j) {
                device const bfloat *key =
                    block_keys + ulong(j) * 8 * DQ + c * ChunkDepth;
                key_words[j] = uint2(
                    *reinterpret_cast<device const uint *>(key),
                    *reinterpret_cast<device const uint *>(key + DQ));
            }
            #pragma unroll
            for (ushort step = 0; step < ChunkSteps; ++step) {
                float2 query[RowFragments];
                #pragma unroll
                for (ushort r = 0; r < RowFragments; ++r) {
                    query[r] = float2(vision_widen(query_words[r][c], step),
                                      vision_widen(query_words[r][c],
                                                   2 + step));
                }
                #pragma unroll
                for (ushort j = 0; j < KeyFragments; ++j) {
                    const float2 key =
                        float2(vision_widen(key_words[j], step),
                               vision_widen(key_words[j], 2 + step));
                    #pragma unroll
                    for (ushort r = 0; r < RowFragments; ++r) {
                        vision_fragment_multiply_accumulate(scores[r][j],
                                                            query[r], key);
                    }
                }
            }
        }

        // Padded keys score -inf; every block holds at least one valid key,
        // so the row maximum stays finite.
        const uint first_key = block * BlockKeys + fn;
        #pragma unroll
        for (ushort r = 0; r < RowFragments; ++r) {
            float block_max = -INFINITY;
            #pragma unroll
            for (ushort j = 0; j < KeyFragments; ++j) {
                const uint key = first_key + j * 8;
                scores[r][j] = float2(
                    key < context ? scores[r][j].x * scale : -INFINITY,
                    key + 1 < context ? scores[r][j].y * scale : -INFINITY);
                block_max = max(block_max, max(scores[r][j].x, scores[r][j].y));
            }
            block_max = max(block_max, simd_shuffle_xor(block_max, 1));
            block_max = max(block_max, simd_shuffle_xor(block_max, 8));
            const float next_max = max(running_max[r], block_max);
            const float correction = exp2(running_max[r] - next_max);
            running_max[r] = next_max;
            float block_sum = 0.0f;
            #pragma unroll
            for (ushort j = 0; j < KeyFragments; ++j) {
                scores[r][j] = exp2(scores[r][j] - next_max);
                block_sum += scores[r][j].x + scores[r][j].y;
            }
            running_sum[r] = running_sum[r] * correction + block_sum;
            #pragma unroll
            for (ushort c = 0; c < OutputFragments; ++c) {
                accumulated[r][c] *= correction;
            }
        }

        device const bfloat *block_values =
            lane_values + ulong(block) * BlockKeys * D;
        #pragma unroll
        for (ushort j = 0; j < KeyFragments; ++j) {
            #pragma unroll
            for (ushort c = 0; c < OutputFragments; ++c) {
                const uint2 value_word = uint2(
                    *reinterpret_cast<device const uint *>(
                        block_values + j * 8 * D + c * 8),
                    0u);
                const float2 value = float2(vision_widen(value_word, 0),
                                            vision_widen(value_word, 1));
                #pragma unroll
                for (ushort r = 0; r < RowFragments; ++r) {
                    vision_fragment_multiply_accumulate(accumulated[r][c],
                                                        scores[r][j], value);
                }
            }
        }
    }

    if (key_half == 1) {
        #pragma unroll
        for (ushort r = 0; r < RowFragments; ++r) {
            handed_max[row_group][r][simd_lane] = running_max[r];
            handed_sum[row_group][r][simd_lane] = running_sum[r];
            #pragma unroll
            for (ushort c = 0; c < OutputFragments; ++c) {
                handed_output[row_group][r][c][simd_lane] = accumulated[r][c];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (key_half == 1) {
        return;
    }
    // The odd half may own no block (-inf maximum): its scale is then zero.
    #pragma unroll
    for (ushort r = 0; r < RowFragments; ++r) {
        const float other_max = handed_max[row_group][r][simd_lane];
        const float row_max = max(running_max[r], other_max);
        const float own_scale = exp2(running_max[r] - row_max);
        const float other_scale = exp2(other_max - row_max);
        float row_sum = running_sum[r] * own_scale +
            handed_sum[row_group][r][simd_lane] * other_scale;
        row_sum += simd_shuffle_xor(row_sum, 1);
        row_sum += simd_shuffle_xor(row_sum, 8);
        device bfloat *lane_output =
            output + d_plane + ulong(first_row + r * 8) * D + fn;
        #pragma unroll
        for (ushort c = 0; c < OutputFragments; ++c) {
            const float2 combined = accumulated[r][c] * own_scale +
                handed_output[row_group][r][c][simd_lane] * other_scale;
            *reinterpret_cast<device bfloat2 *>(lane_output + c * 8) =
                bfloat2(combined / row_sum);
        }
    }
}

// Gather padded attention output back to (tokens, 1152) rows.

kernel void vision_attention_pack(
    device const bfloat *padded [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant VisionQkvParams &params [[buffer(2)]],
    uint token [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]])
{
    device bfloat *row = output + ulong(token) * kVisionHidden;
    for (uint index = thread_index; index < kVisionHidden; index += 256) {
        uint head = index / kVisionHeadDim;
        uint dim = index % kVisionHeadDim;
        row[index] = padded[
            (ulong(head) * params.padded_tokens + token) * kVisionHeadDim +
            dim];
    }
}

// Overwrites language-model embedding rows with encoded image rows. Encoded
// after the token-embedding gather on the same serial compute encoder, so
// layer 0 sees the injected rows.

kernel void vision_inject_embeddings(
    device const bfloat *source [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant VisionInjectParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]])
{
    uint elements = params.rows * params.width;
    for (uint element = index; element < elements; element += grid_size) {
        uint row = element / params.width;
        uint dim = element % params.width;
        output[ulong(params.destination_row + row) * params.width + dim] =
            source[ulong(params.source_row + row) * params.width + dim];
    }
}

// Zero fill: writes zeros over word_count 16-byte words with a grid-stride
// loop, so a capped grid covers a buffer of any size. A new scratch arena runs
// it once before its first encode: padding rows and tokens are read by whole
// tiles and must be finite.

kernel void vision_zero_words(
    device uint4 *words [[buffer(0)]],
    constant uint &word_count [[buffer(1)]],
    uint index [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]])
{
    for (ulong word = index; word < word_count; word += grid_size) {
        words[word] = uint4(0);
    }
}
