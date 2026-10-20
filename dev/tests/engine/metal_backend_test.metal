#include <metal_stdlib>

using namespace metal;

kernel void test_copy_u32(device const uint *source [[buffer(0)]],
                         device uint *destination [[buffer(1)]],
                         constant uint &count [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        destination[gid] = source[gid];
    }
}

kernel void test_add_u32(device uint *values [[buffer(0)]],
                       constant uint &count [[buffer(1)]],
                       constant uint &increment [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        values[gid] += increment;
    }
}

kernel void sparse_fill_copy_u32(device uint *sparseValues [[buffer(0)]],
                                 device uint *readback [[buffer(1)]],
                                 constant uint &count [[buffer(2)]],
                                 constant uint &value [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid < count) {
        sparseValues[gid] = value + gid;
        readback[gid] = sparseValues[gid];
    }
}

// The host-encode benchmark's dispatch: it does no work, so a graph's time is
// all host encoding. Its name is longer than the short-string limit, as most
// production pipeline names are.
kernel void graph_encode_benchmark_noop_dispatch() {}
