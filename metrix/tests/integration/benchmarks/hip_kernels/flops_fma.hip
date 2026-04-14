// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// FLOPS / Arithmetic Intensity Microbenchmark
//
// Purpose: Exercise the compute pipeline with FMA chains to validate
// total_flops, hbm_gflops, and arithmetic intensity metrics.
//
// Modes:
//   0 = pure FMA    — register-only FMA chain, minimal memory, known FLOP count
//   1 = low AI      — copy kernel (0 compute, lots of memory) → AI near 0
//   2 = high AI     — 1 load + many FMAs per element → high AI
//
// For mode 0 (pure FMA):
//   FLOP count = num_blocks * block_size * iters * 2  (each FMA = 2 FLOP)
//   With defaults: 512 * 256 * 100000 * 2 = 26,214,400,000 FLOP
//   (But FLOP counting by hardware is per-wavefront:
//    131072 threads / 64 lanes = 2048 wavefronts on CDNA
//    2048 * 100000 * 2 * 64 = 26,214,400,000)
//
// Usage: ./flops_fma [mode] [fma_per_element] [total_bytes]

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call) do {                                         \
    hipError_t err = (call);                                         \
    if (err != hipSuccess) {                                         \
        fprintf(stderr, "HIP error: %s at %s:%d\n",                 \
                hipGetErrorString(err), __FILE__, __LINE__);         \
        exit(1);                                                     \
    }                                                                \
} while(0)

// Pure FMA: register-only compute, minimal memory
__global__ void pure_fma_kernel(float* __restrict__ out, int iters) {
    float a = 1.0001f, b = 0.9999f, c = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        c = __builtin_fmaf(a, b, c);
    }
    // Single write to prevent DCE
    if (threadIdx.x == 0) out[blockIdx.x] = c;
}

// Low arithmetic intensity: pure copy, zero compute
__global__ void copy_only_kernel(const float* __restrict__ src,
                                  float* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * (size_t)blockDim.x;
    for (size_t i = idx; i < N; i += stride) {
        dst[i] = src[i];
    }
}

// High arithmetic intensity: 1 load, K FMAs, 1 store
__global__ void high_ai_kernel(const float* __restrict__ src,
                                float* __restrict__ dst,
                                size_t N, int K) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * (size_t)blockDim.x;
    for (size_t i = idx; i < N; i += stride) {
        float val = src[i];
        float acc = 0.0f;
        #pragma unroll 1
        for (int j = 0; j < K; j++) {
            acc = __builtin_fmaf(val, val, acc);
        }
        dst[i] = acc;
    }
}

int main(int argc, char** argv) {
    int mode = 0;           // 0=pure FMA, 1=low AI, 2=high AI
    int K = 100;            // FMAs per element (mode 2)
    size_t total_bytes = 64ULL * 1024 * 1024;  // 64 MB for memory modes

    if (argc > 1) mode = atoi(argv[1]);
    if (argc > 2) K = atoi(argv[2]);
    if (argc > 3) total_bytes = (size_t)atoll(argv[3]);

    fprintf(stderr, "FLOPS: mode=%d (%s), K=%d\n",
            mode,
            mode == 0 ? "pure_fma" : mode == 1 ? "low_ai" : "high_ai",
            K);

    if (mode == 0) {
        // Pure FMA: 512 blocks * 256 threads * 100000 FMAs
        int num_blocks = 512;
        int block_size = 256;
        int iters = 100000;

        float *d_out;
        HIP_CHECK(hipMalloc(&d_out, num_blocks * sizeof(float)));
        HIP_CHECK(hipMemset(d_out, 0, num_blocks * sizeof(float)));

        // Warmup
        pure_fma_kernel<<<num_blocks, block_size>>>(d_out, 100);
        HIP_CHECK(hipDeviceSynchronize());

        // Measured
        pure_fma_kernel<<<num_blocks, block_size>>>(d_out, iters);
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipFree(d_out));
    } else {
        size_t N = total_bytes / sizeof(float);

        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, 0));
        int block_size = 256;
        int num_blocks = prop.multiProcessorCount * 4;

        float *d_src, *d_dst;
        HIP_CHECK(hipMalloc(&d_src, total_bytes));
        HIP_CHECK(hipMalloc(&d_dst, total_bytes));
        HIP_CHECK(hipMemset(d_src, 0x3F, total_bytes));
        HIP_CHECK(hipMemset(d_dst, 0, total_bytes));

        // Warmup
        if (mode == 1) {
            copy_only_kernel<<<num_blocks, block_size>>>(d_src, d_dst, N);
        } else {
            high_ai_kernel<<<num_blocks, block_size>>>(d_src, d_dst, N, K);
        }
        HIP_CHECK(hipDeviceSynchronize());

        // Measured
        if (mode == 1) {
            copy_only_kernel<<<num_blocks, block_size>>>(d_src, d_dst, N);
        } else {
            high_ai_kernel<<<num_blocks, block_size>>>(d_src, d_dst, N, K);
        }
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipFree(d_src));
        HIP_CHECK(hipFree(d_dst));
    }

    return 0;
}
