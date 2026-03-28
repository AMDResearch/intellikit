// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Benchmark 2: Sequential coalesced write to HBM.
// Validates: hbm_write_bandwidth, global_store_efficiency
//
// Design: Each thread writes a constant to a large array.
//         Write-only (no read), stride-1 => perfectly coalesced stores.
//
// Usage: bw_sequential_write [size_mb]

#include "common.h"

__global__ void sequential_write(float* __restrict__ dst, size_t N) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = 1.0f;
    }
}

int main(int argc, char** argv) {
    int size_mb = parse_int(argc, argv, 1, 512);
    size_t N = (size_t)size_mb * 1024ULL * 1024ULL / sizeof(float);

    gpu_warmup();

    float* d_dst;
    size_t bytes = N * sizeof(float);
    HIP_CHECK(hipMalloc(&d_dst, bytes));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(sequential_write, dim3(gridSize), dim3(blockSize), 0, 0,
                       d_dst, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_dst));

    printf("sequential_write: %d MB, %zu elements\n", size_mb, N);
    return 0;
}
