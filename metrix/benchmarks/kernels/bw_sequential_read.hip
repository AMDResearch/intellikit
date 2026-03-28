// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Benchmark 1: Sequential coalesced read from HBM.
// Validates: hbm_read_bandwidth, bytes_transferred_hbm, coalescing_efficiency,
//            global_load_efficiency, l2_hit_rate (~0%)
//
// Design: Each thread reads one float from a large array (512MB default).
//         Stride-1 access => perfectly coalesced.
//         Array >> L2 cache => L2 hit rate ~0%.
//
// Usage: bw_sequential_read [size_mb]
//   size_mb: array size in MB (default 512)

#include "common.h"

__global__ void sequential_read(const float* __restrict__ src, float* __restrict__ dst, size_t N) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Read from src, write a single reduced value to avoid optimizing away
        dst[idx] = src[idx];
    }
}

int main(int argc, char** argv) {
    int size_mb = parse_int(argc, argv, 1, 512);
    size_t N = (size_t)size_mb * 1024ULL * 1024ULL / sizeof(float);

    gpu_warmup();

    float *d_src, *d_dst;
    size_t bytes = N * sizeof(float);
    HIP_CHECK(hipMalloc(&d_src, bytes));
    HIP_CHECK(hipMalloc(&d_dst, bytes));
    HIP_CHECK(hipMemset(d_src, 1, bytes));
    HIP_CHECK(hipMemset(d_dst, 0, bytes));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch kernel
    hipLaunchKernelGGL(sequential_read, dim3(gridSize), dim3(blockSize), 0, 0,
                       d_src, d_dst, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));

    printf("sequential_read: %d MB, %zu elements\n", size_mb, N);
    return 0;
}
