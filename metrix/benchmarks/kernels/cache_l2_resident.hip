// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Benchmark 5: L2-resident array iterated many times.
// Validates: l2_hit_rate (>90%), l2_bandwidth
//
// Design: 32MB array (fits in MI300X 256MB L2 cache).
//         Read the same array 100 times.
//         First pass: compulsory misses (L2 miss, fetch from HBM).
//         Subsequent passes: L2 hits.
//         Expected L2 hit rate: ~99% (99/100 passes hit).
//
// Usage: cache_l2_resident [size_mb] [iterations]

#include "common.h"

__global__ void l2_resident_read(const float* __restrict__ src, float* __restrict__ out,
                                  size_t N, int iterations) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        for (int i = 0; i < iterations; i++) {
            sum += src[idx];
        }
        out[idx] = sum;
    }
}

int main(int argc, char** argv) {
    int size_mb = parse_int(argc, argv, 1, 32);
    int iterations = parse_int(argc, argv, 2, 100);
    size_t N = (size_t)size_mb * 1024ULL * 1024ULL / sizeof(float);

    gpu_warmup();

    float *d_src, *d_out;
    size_t bytes = N * sizeof(float);
    HIP_CHECK(hipMalloc(&d_src, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));

    // Initialize with known values
    float* h_src = (float*)malloc(bytes);
    for (size_t i = 0; i < N; i++) h_src[i] = 1.0f;
    HIP_CHECK(hipMemcpy(d_src, h_src, bytes, hipMemcpyHostToDevice));
    free(h_src);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(l2_resident_read, dim3(gridSize), dim3(blockSize), 0, 0,
                       d_src, d_out, N, iterations);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_out));

    printf("l2_resident: %d MB, %d iterations, expected l2_hit_rate>90%%\n",
           size_mb, iterations);
    return 0;
}
