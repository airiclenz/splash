#pragma once

#include "metal/abi/KernelABI.h"

// Four-tap causal convolution of one channel at one token of the command,
// reading the three preceding tokens from the carried state, rounded to bf16
// and gated by SiLU.
inline bfloat gdn_conv_silu(device const bfloat *packed,
                            device const bfloat *conv_state_in,
                            device const bfloat *conv_weights,
                            uint packed_width, uint conv_dim, uint token,
                            uint channel) {
  float value = 0.0f;
  for (uint tap = 0; tap < 4; ++tap) {
    uint position = token + tap;
    bfloat input = position < 3
                       ? conv_state_in[position * conv_dim + channel]
                       : packed[(position - 3) * packed_width + channel];
    value += float(input) * float(conv_weights[channel * 4 + tap]);
  }
  value = float(bfloat(value));
  return bfloat(value / (1.0f + fast::exp2(-1.44269504089f * value)));
}

// Row `row` of the carried state after consumed_tokens: the last three inputs
// seen, still taken from the incoming state when fewer were consumed.
inline bfloat gdn_conv_carry(device const bfloat *packed,
                             device const bfloat *conv_state_in,
                             uint packed_width, uint conv_dim,
                             uint consumed_tokens, uint row, uint channel) {
  uint source = consumed_tokens + row;
  return source < 3 ? conv_state_in[source * conv_dim + channel]
                    : packed[(source - 3) * packed_width + channel];
}

// The gates of one (token, value head): beta = sigmoid(b) and
// decay = exp(a_scale * softplus(bf16(a + dt_bias))), the softplus rounded to
// bf16 as the reference does.
inline void gdn_write_gates(device const bfloat *packed_row,
                            device const bfloat *dt_bias,
                            device const float *a_scale, uint b_offset,
                            uint a_offset, uint head, device bfloat &beta,
                            device float &decay) {
  float b = float(packed_row[b_offset + head]);
  beta = bfloat(1.0f / (1.0f + fast::exp2(-1.44269504089f * b)));
  bfloat x = bfloat(float(packed_row[a_offset + head]) + float(dt_bias[head]));
  float xf = float(x);
  bfloat softplus =
      bfloat(max(xf, 0.0f) +
             fast::log2(1.0f + fast::exp2(-1.44269504089f * abs(xf))) *
                 0.69314718056f);
  decay = fast::exp(a_scale[head] * float(softplus));
}

// Gated RMSNorm of one (token, value head) task by one simdgroup, with no
// threadgroup memory or barriers. Lane `lane` holds dims 32q + lane
// (q = 0..3); each quarter is simd_summed and the four partials are added in
// q order, the order the per-simdgroup partials were added in before.
template <uint ValueHeads, uint HeadDim, uint ConvDim>
inline void gdn_gate_phase(device const bfloat *recurrent,
                           device const bfloat *packed,
                           device const bfloat *norm_weight,
                           device bfloat *hidden, uint task,
                           uint packed_width, uint lane) {
  constexpr uint ZOffset = ConvDim;
  constexpr uint Quarters = HeadDim / 32;
  static_assert(HeadDim % 32 == 0, "whole 32-dim quarters per lane");
  const uint token = task / ValueHeads;
  const uint head = task % ValueHeads;
  const ulong base = ulong(task) * HeadDim;
  float value[Quarters];
  for (uint q = 0; q < Quarters; ++q)
    value[q] = float(recurrent[base + 32 * q + lane]);
  float total = 0.0f;
  for (uint q = 0; q < Quarters; ++q)
    total += simd_sum(value[q] * value[q]);
  const float scale = rsqrt(total / HeadDim + 1e-6f);
  device const bfloat *gates =
      packed + token * packed_width + ZOffset + head * HeadDim;
  for (uint q = 0; q < Quarters; ++q) {
    const uint dim = 32 * q + lane;
    bfloat normalized = bfloat(value[q] * scale * float(norm_weight[dim]));
    float gate = float(gates[dim]);
    float silu = gate / (1.0f + fast::exp2(-1.44269504089f * gate));
    hidden[base + dim] = bfloat(float(normalized) * silu);
  }
}
