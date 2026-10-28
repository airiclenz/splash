#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/activation.h"
#include "metal/kernels/common/rms_inverse.h"

// The convolution inputs before a command's first row: the rows a state
// cell carries, or after `consumed` rows (in `rows`) the last of those rows
// and the carried ones: at(r, c) is row r of the carried state the rows
// leave.
struct GdnConvHistory {
  device const bfloat *carried;
  device const bfloat *rows;
  uint conv_dim;
  uint rows_width;
  uint consumed;

  bfloat at(uint row, uint channel) const {
    constexpr uint Carried = SPLASH_GDN_CONVOLUTION_TAPS - 1;
    const uint source = consumed + row;
    return source < Carried ? carried[source * conv_dim + channel]
                            : rows[(source - Carried) * rows_width + channel];
  }
  // Row `row` itself, for a caller that reads many channels of it.
  device const bfloat *row(uint row) const {
    constexpr uint Carried = SPLASH_GDN_CONVOLUTION_TAPS - 1;
    const uint source = consumed + row;
    return source < Carried ? carried + source * conv_dim
                            : rows + (source - Carried) * rows_width;
  }
};

// Causal convolution of one channel at one token of the command, reading the
// preceding tokens from the history, rounded to bf16 and gated by SiLU.
inline bfloat gdn_conv_silu(device const bfloat *packed, GdnConvHistory history,
                            device const bfloat *conv_weights,
                            uint packed_width, uint token, uint channel) {
  constexpr uint Taps = SPLASH_GDN_CONVOLUTION_TAPS, Carried = Taps - 1;
  float value = 0.0f;
  for (uint tap = 0; tap < Taps; ++tap) {
    uint position = token + tap;
    bfloat input = position < Carried
                       ? history.at(position, channel)
                       : packed[(position - Carried) * packed_width + channel];
    value += float(input) * float(conv_weights[channel * Taps + tap]);
  }
  value = float(bfloat(value));
  return bfloat(splash_silu(value));
}

// The gates of one (token, value head): beta = sigmoid(b) and
// decay = exp(a_scale * softplus(bf16(a + dt_bias))), the softplus rounded to
// bf16 as the reference does.
struct GdnGates {
  bfloat beta;
  float decay;
};

inline GdnGates gdn_gates(device const bfloat *packed_row,
                          device const bfloat *dt_bias,
                          device const float *a_scale, uint b_offset,
                          uint a_offset, uint head) {
  float b = float(packed_row[b_offset + head]);
  GdnGates gates;
  gates.beta = bfloat(splash_sigmoid(b));
  bfloat x = bfloat(float(packed_row[a_offset + head]) + float(dt_bias[head]));
  float xf = float(x);
  bfloat softplus =
      bfloat(max(xf, 0.0f) +
             fast::log2(1.0f + fast::exp2(-1.44269504089f * abs(xf))) *
                 0.69314718056f);
  gates.decay = fast::exp(a_scale[head] * float(softplus));
  return gates;
}

// The position of value head `head` among the GDN output's head blocks: the
// head itself, or with `tiled` llama.cpp's GGUF order, which puts value head
// j of every key head next to each other (GDNGatePrefillParams).
template <uint KeyHeads, uint ValueHeads>
inline uint gdn_output_head(uint head, bool tiled) {
  constexpr uint HeadsPerKey = ValueHeads / KeyHeads;
  return tiled ? (head % HeadsPerKey) * KeyHeads + head / HeadsPerKey : head;
}

// Gated RMSNorm of the recurrent row of one task (token, value head) by one
// simdgroup, with no threadgroup memory or barriers, stored at the head's
// output position. Lane `lane` holds dimensions 32g + lane; the four
// simd_sum partials are added in g order. Reassociation is off and the
// operations are those of decode's gdn_decode_gate, so the two write the
// same rows bit for bit. Prefill runs SPLASH_GDN_GATE_TASKS_PER_GROUP tasks
// per threadgroup: against one 128-thread group per task with three
// barriers, the gate took about 22% less time on an M2 Pro with the same
// bits. The norm weights are read in their stored type W: bfloat for an MLX
// target's, float for a GGUF's F32 norms.
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint PackedWidth, class W>
inline void gdn_gate_phase(device const bfloat *recurrent,
                           device const bfloat *packed,
                           device const W *norm_weight, device bfloat *hidden,
                           uint task, bool tiled, bool sigmoid_gate,
                           uint lane) {
#pragma clang fp reassociate(off)
  constexpr uint Groups = HeadDim / 32, ZOffset = ConvDim;
  static_assert(HeadDim % 32 == 0, "whole 32-dimension groups per lane");
  const uint token = task / ValueHeads;
  const uint head = task % ValueHeads;
  const ulong base = ulong(task) * HeadDim;
  const ulong hidden_base =
      (ulong(token) * ValueHeads +
       gdn_output_head<KeyHeads, ValueHeads>(head, tiled)) *
      HeadDim;
  bfloat gate[Groups];
  W weight[Groups];
  float value[Groups];
  for (uint g = 0; g < Groups; ++g) {
    const uint dim = 32 * g + lane;
    gate[g] = packed[token * PackedWidth + ZOffset + head * HeadDim + dim];
    weight[g] = norm_weight[dim];
    value[g] = float(recurrent[base + dim]);
  }
  float total = 0.0f;
  for (uint g = 0; g < Groups; ++g)
    total += simd_sum(value[g] * value[g]);
  const float inverse = rsqrt(total / HeadDim + kRmsEpsilon);
  for (uint g = 0; g < Groups; ++g) {
    const bfloat normalized = bfloat((value[g] * inverse) * float(weight[g]));
    const float z = float(gate[g]);
    // Qwen3.5 gates by silu(z), Qwen3.8-Flash-Next by sigmoid(z).
    hidden[hidden_base + 32 * g + lane] =
        bfloat((float(normalized) * (sigmoid_gate ? 1.0f : z)) /
               (1.0f + fast::exp2(-1.44269504089f * z)));
  }
}
