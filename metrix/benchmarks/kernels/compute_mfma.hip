/*
 * compute_mfma — MFMA (Matrix Fused Multiply-Add) compute kernel.
 *
 * Uses a small GEMM via hipBLAS-style manual loop that the compiler
 * should lower to MFMA instructions on gfx942+.
 *
 * Alternative approach: inline asm for v_mfma_f32_16x16x16_f16.
 * We provide both — select via mode flag.
 *
 * Validates: SQ_INSTS_VALU_MFMA_MOPS_F32 (or F16),
 *            the 512 * MFMA_count term in compute.total_flops
 *
 * Usage: ./compute_mfma [mode] [iters]
 *   mode 0 = inline asm MFMA (default, gfx942 only)
 *   mode 1 = small FP16 GEMM loop (portable)
 */

#include "common.h"
#include <hip/hip_fp16.h>

// Mode 0: Direct MFMA inline assembly (gfx942/gfx90a)
// v_mfma_f32_4x4x4f16: 4x4x4 FP16 MFMA
// Inputs: two packed f16x2 values in VGPRs (a, b)
// Accumulator: 4 floats per thread (returned as v[0:3])
__global__ void mfma_asm_kernel(float* __restrict__ out, int iters) {
    // v_mfma_f32_4x4x4f16 needs a 4-VGPR accumulator
    float4 c = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    unsigned int a = 0x3C003C00;  // packed f16x2: (1.0h, 1.0h)
    unsigned int b = 0x3C003C00;

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        asm volatile(
            "v_mfma_f32_4x4x4f16 %0, %4, %5, %0"
            : "+v"(c.x), "+v"(c.y), "+v"(c.z), "+v"(c.w)
            : "v"(a), "v"(b)
        );
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = c.x + c.y + c.z + c.w;
    }
}

// Mode 1: Small FP32 multiply-accumulate loop — compiler may or may not use MFMA
__global__ void gemm_loop_kernel(float* __restrict__ out, int iters) {
    float acc = 0.0f;
    float a = 1.0001f;
    float b = 0.9999f;

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        acc = __builtin_fmaf(a, b, acc);
        acc = __builtin_fmaf(a, b, acc);
        acc = __builtin_fmaf(a, b, acc);
        acc = __builtin_fmaf(a, b, acc);
    }

    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

int main(int argc, char** argv) {
    int mode = parse_int(argc, argv, 1, 0);
    int iters = parse_int(argc, argv, 2, 50000);
    int num_blocks = 256;
    int block = 256;

    float* d_out;
    HIP_CHECK(hipMalloc(&d_out, num_blocks * sizeof(float)));
    HIP_CHECK(hipMemset(d_out, 0, num_blocks * sizeof(float)));

    if (mode == 0) {
        mfma_asm_kernel<<<num_blocks, block>>>(d_out, 100);
        HIP_CHECK(hipDeviceSynchronize());
        mfma_asm_kernel<<<num_blocks, block>>>(d_out, iters);
    } else {
        gemm_loop_kernel<<<num_blocks, block>>>(d_out, 100);
        HIP_CHECK(hipDeviceSynchronize());
        gemm_loop_kernel<<<num_blocks, block>>>(d_out, iters);
    }
    HIP_CHECK(hipDeviceSynchronize());

    printf("{\"benchmark\": \"compute_mfma\", \"mode\": %d, "
           "\"mode_name\": \"%s\", \"iters\": %d, \"num_blocks\": %d}\n",
           mode, mode == 0 ? "inline_asm" : "gemm_loop", iters, num_blocks);

    HIP_CHECK(hipFree(d_out));
    return 0;
}
