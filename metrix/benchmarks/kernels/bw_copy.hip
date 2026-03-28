// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Benchmark 3: Read-write copy (1:1 ratio).
// Validates: hbm_bandwidth_utilization, bytes_transferred_hbm (should be 2*N*4),
//            l1_hit_rate (divergence test: IntelliKit only counts read misses)
//
// Design: c[i] = a[i] for two 512MB arrays.
//         bytes_hbm = N*4 (read) + N*4 (write) = 2*N*4
//
// Usage: bw_copy [size_mb]

#include "common.h"

__global__ void copy_kernel(const float* __restrict__ src, float* __restrict__ dst, size_t N) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
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

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(copy_kernel, dim3(gridSize), dim3(blockSize), 0, 0,
                       d_src, d_dst, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));

    printf("copy: %d MB, %zu elements, expected bytes_hbm=%zu\n",
           size_mb, N, 2 * bytes);
    return 0;
}
