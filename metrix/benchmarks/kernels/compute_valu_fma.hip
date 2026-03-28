// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Benchmark 8: Pure FP32 FMA loop in registers, no memory access.
// Validates: compute.total_flops, compute.hbm_gflops
//
// Design: Each thread executes N FMA instructions (a = a * b + c) in a loop.
//         No global memory access => hbm_arithmetic_intensity should be very high.
//
//         Expected total_flops = num_waves * iterations * 2 * 64
//           (each FMA = 2 flops, 64 lanes per wave)
//         num_waves = grid_size / 64
//
// Usage: compute_valu_fma [iterations]

#include "common.h"

__global__ void valu_fma_kernel(float* __restrict__ out, int iterations) {
    float a = 1.0f;
    float b = 1.000001f;
    float c = 0.000001f;

    // Pure FMA loop - compiler should emit v_fma_f32
    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        a = __builtin_fmaf(a, b, c);
    }

    // Single store to prevent dead-code elimination
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = a;
}

int main(int argc, char** argv) {
    int iterations = parse_int(argc, argv, 1, 100000);
    // Use moderate grid to keep GPU busy without too many waves
    int blockSize = 256;
    int num_wgs = 304;  // One per CU
    int gridSize = num_wgs;
    size_t total_threads = (size_t)gridSize * blockSize;

    gpu_warmup();

    float* d_out;
    HIP_CHECK(hipMalloc(&d_out, total_threads * sizeof(float)));

    hipLaunchKernelGGL(valu_fma_kernel, dim3(gridSize), dim3(blockSize), 0, 0,
                       d_out, iterations);
    HIP_CHECK(hipDeviceSynchronize());

    // Expected: total_flops = total_threads * iterations * 2
    // But metrix counts per-wave: num_waves * SQ_INSTS_VALU_FMA_F32 * 2 * 64
    size_t num_waves = total_threads / 64;
    printf("valu_fma: %d iterations, %zu threads, %zu waves\n",
           iterations, total_threads, num_waves);
    printf("  expected total_flops = %zu\n",
           (size_t)num_waves * iterations * 2 * 64);

    HIP_CHECK(hipFree(d_out));
    return 0;
}
