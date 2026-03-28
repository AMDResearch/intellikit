// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Benchmark 9: MFMA (Matrix Fused Multiply-Add) instructions.
// Validates: SQ_INSTS_VALU_MFMA_MOPS_F32 contribution to compute.total_flops
//
// Design: Uses inline assembly to emit v_mfma_f32_16x16x4_f32 instructions.
//         Each MFMA instruction produces 512 FP32 operations (16*16*2).
//         gfx942+ only.
//
// Fallback: If MFMA asm fails, uses a small GEMM via rocBLAS as alternative.
//           For now, inline asm version targeting gfx942.
//
// Usage: compute_mfma [iterations]

#include "common.h"

// v_mfma_f32_16x16x4_f32: 16x16 output, 4 elements of K dimension
// Each instruction: 16*16*2 = 512 FP32 ops (multiply + add for each output element)
// Input: 4 floats per thread in A, 4 floats per thread in B
// Output: 16 floats per thread (16x16 matrix spread across 64 threads)

__global__ void mfma_kernel(float* __restrict__ out, int iterations) {
    // Initialize accumulators and inputs in registers
    float a = 1.0f;
    float b = 1.0f;
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        // v_mfma_f32_16x16x4_f32 : D[0:3] = A * B + C[0:3]
        asm volatile(
            "v_mfma_f32_16x16x4_f32 %0, %4, %5, %0\n"
            "v_mfma_f32_16x16x4_f32 %1, %4, %5, %1\n"
            "v_mfma_f32_16x16x4_f32 %2, %4, %5, %2\n"
            "v_mfma_f32_16x16x4_f32 %3, %4, %5, %3\n"
            : "+v"(c[0]), "+v"(c[1]), "+v"(c[2]), "+v"(c[3])
            : "v"(a), "v"(b)
        );
    }

    // Prevent dead-code elimination
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = c[0] + c[1] + c[2] + c[3];
}

int main(int argc, char** argv) {
    int iterations = parse_int(argc, argv, 1, 10000);
    int blockSize = 256;
    int num_wgs = 304;
    size_t total_threads = (size_t)num_wgs * blockSize;

    gpu_warmup();

    float* d_out;
    HIP_CHECK(hipMalloc(&d_out, total_threads * sizeof(float)));

    hipLaunchKernelGGL(mfma_kernel, dim3(num_wgs), dim3(blockSize), 0, 0,
                       d_out, iterations);
    HIP_CHECK(hipDeviceSynchronize());

    // Each iteration: 4 MFMA instructions, each producing 512 ops
    // Per wave: iterations * 4 MFMA instructions counted in SQ_INSTS_VALU_MFMA_MOPS_F32
    // total_flops formula: 512 * SQ_INSTS_VALU_MFMA_MOPS_F32
    size_t num_waves = total_threads / 64;
    size_t mfma_per_wave = (size_t)iterations * 4;
    printf("mfma: %d iterations, %zu waves, %zu mfma_insts/wave\n",
           iterations, num_waves, mfma_per_wave);
    printf("  expected total_flops (mfma term) = %zu\n",
           512 * num_waves * mfma_per_wave);

    HIP_CHECK(hipFree(d_out));
    return 0;
}
