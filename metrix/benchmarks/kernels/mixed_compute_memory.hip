// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Benchmark 11: Tunable arithmetic intensity.
// Validates: hbm_arithmetic_intensity, l2_arithmetic_intensity, l1_arithmetic_intensity
//
// Design: Load one float from global memory, perform K FMAs, then store.
//   K=1:  AI = 2*64 / (2*4) = 16 FLOP/byte  (1 read + 1 write, 4 bytes each)
//   K=10: AI = 20*64 / 8 = 160 FLOP/byte
//   K=100: AI = 200*64 / 8 = 1600 FLOP/byte
//
// More precisely:
//   FLOPS per thread = K * 2 (each FMA = multiply + add)
//   FLOPS per wave = K * 2 * 64
//   Bytes per wave (HBM) = 2 * 64 * 4 = 512 (1 read + 1 write per thread)
//   AI_hbm = (K * 2 * 64) / 512 = K / 4
//
// Usage: mixed_compute_memory [K] [size_mb]

#include "common.h"

// Template-like approach with runtime K
__global__ void mixed_kernel(const float* __restrict__ src, float* __restrict__ dst,
                              size_t N, int K) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = src[idx];
        float b = 1.000001f;
        float c = 0.000001f;

        #pragma unroll 1
        for (int k = 0; k < K; k++) {
            val = __builtin_fmaf(val, b, c);
        }

        dst[idx] = val;
    }
}

int main(int argc, char** argv) {
    int K = parse_int(argc, argv, 1, 10);
    int size_mb = parse_int(argc, argv, 2, 256);
    size_t N = (size_t)size_mb * 1024ULL * 1024ULL / sizeof(float);

    gpu_warmup();

    float *d_src, *d_dst;
    size_t bytes = N * sizeof(float);
    HIP_CHECK(hipMalloc(&d_src, bytes));
    HIP_CHECK(hipMalloc(&d_dst, bytes));
    HIP_CHECK(hipMemset(d_src, 1, bytes));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(mixed_kernel, dim3(gridSize), dim3(blockSize), 0, 0,
                       d_src, d_dst, N, K);
    HIP_CHECK(hipDeviceSynchronize());

    // Expected arithmetic intensity:
    // FLOPS per thread = K * 2 (FMA = 2 ops)
    // Bytes per thread = 8 (4 read + 4 write)
    // AI = K * 2 / 8 = K / 4 FLOP/byte
    printf("mixed_compute_memory: K=%d, %d MB, expected AI=%.2f FLOP/byte\n",
           K, size_mb, (float)K / 4.0f);

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));
    return 0;
}
