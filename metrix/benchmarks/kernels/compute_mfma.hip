// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Benchmark 9: MFMA (Matrix Fused Multiply-Add) instructions.
// Validates: SQ_INSTS_VALU_MFMA_MOPS_F32 contribution to compute.total_flops
//
// Design: Uses __builtin_amdgcn_mfma_f32_16x16x4f32 intrinsic.
//         Each MFMA instruction produces 512 FP32 operations (16*16*2).
//         gfx942+ only.
//
// Usage: compute_mfma [iterations]

#include "common.h"

// Use the compiler intrinsic instead of raw inline asm.
// __builtin_amdgcn_mfma_f32_16x16x4f32(float A, float B, float4 C, int cbsz, int abid, int blgp)
// Returns float4 (4 accumulators per lane, 64 lanes = 16x16 output matrix)
// Each call = 16*16*2 = 512 FP32 ops

typedef float float4 __attribute__((ext_vector_type(4)));

__global__ void mfma_kernel(float* __restrict__ out, int iterations) {
    float a = 1.0f;
    float b = 1.0f;
    float4 c = {0.0f, 0.0f, 0.0f, 0.0f};

    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        c = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0);
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

    // Each iteration: 1 MFMA instruction producing 512 ops
    // Per wave: iterations MFMA instructions counted in SQ_INSTS_VALU_MFMA_MOPS_F32
    // total_flops formula: 512 * SQ_INSTS_VALU_MFMA_MOPS_F32
    size_t num_waves = total_threads / 64;
    size_t mfma_per_wave = (size_t)iterations;
    printf("mfma: %d iterations, %zu waves, %zu mfma_insts/wave\n",
           iterations, num_waves, mfma_per_wave);
    printf("  expected total_flops (mfma term) = %zu\n",
           512 * num_waves * mfma_per_wave);

    HIP_CHECK(hipFree(d_out));
    return 0;
}
