// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Benchmark 6: L1-resident array iterated many times.
// Validates: l1_hit_rate (>95%), bytes_transferred_l1
//
// Design: Each workgroup works on a small array that fits in L1 cache.
//         L1 on gfx942 is 32KB per CU (16KB used for data, 16KB for instructions).
//         We use 8KB per workgroup = 2048 floats.
//         Each thread in a 256-thread workgroup reads 8 elements, iterated 200 times.
//         First pass: L1 miss, fetch from L2.
//         Subsequent passes: L1 hits.
//         Expected L1 hit rate: ~99.5% (199/200 passes hit).
//
// Usage: cache_l1_resident [iterations]

#include "common.h"

#define ELEMS_PER_WG 2048  // 8KB per workgroup

__global__ void l1_resident_read(const float* __restrict__ src, float* __restrict__ out,
                                  int iterations) {
    // Each workgroup reads from its own 8KB region
    size_t wg_offset = (size_t)blockIdx.x * ELEMS_PER_WG;
    int tid = threadIdx.x;
    int elems_per_thread = ELEMS_PER_WG / blockDim.x;  // 8 for blockDim=256

    float sum = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        for (int e = 0; e < elems_per_thread; e++) {
            size_t idx = wg_offset + tid * elems_per_thread + e;
            sum += src[idx];
        }
    }
    // Write one output per thread to avoid dead-code elimination
    out[wg_offset + tid] = sum;
}

int main(int argc, char** argv) {
    int iterations = parse_int(argc, argv, 1, 200);
    // Use enough workgroups to keep GPU busy (at least num_CU workgroups)
    int num_wgs = 304 * 4;  // 4x oversubscription on MI300X
    size_t total_elems = (size_t)num_wgs * ELEMS_PER_WG;

    gpu_warmup();

    float *d_src, *d_out;
    size_t bytes = total_elems * sizeof(float);
    HIP_CHECK(hipMalloc(&d_src, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));

    float* h_src = (float*)malloc(bytes);
    for (size_t i = 0; i < total_elems; i++) h_src[i] = 1.0f;
    HIP_CHECK(hipMemcpy(d_src, h_src, bytes, hipMemcpyHostToDevice));
    free(h_src);

    int blockSize = 256;

    hipLaunchKernelGGL(l1_resident_read, dim3(num_wgs), dim3(blockSize), 0, 0,
                       d_src, d_out, iterations);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_out));

    printf("l1_resident: %d workgroups, %d iterations, expected l1_hit_rate>95%%\n",
           num_wgs, iterations);
    return 0;
}
