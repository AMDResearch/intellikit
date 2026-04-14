// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Atomic Contention Microbenchmark
//
// Purpose: Measure atomic operation latency under different contention
// levels to validate the memory.atomic_latency metric.
//
// Modes:
//   0 = high contention  — all threads atomicAdd to a single address
//   1 = low contention   — each thread atomicAdd to its own address
//   2 = moderate          — threads share per-block counters
//
// Usage: ./atomic_contention [mode] [iterations]

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

// High contention: all threads atomic to single address
__global__ void atomic_single(int* __restrict__ counter, int iters) {
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        atomicAdd(counter, 1);
    }
}

// Low contention: each thread atomics to its own address
__global__ void atomic_private(int* __restrict__ counters, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        atomicAdd(&counters[tid], 1);
    }
}

// Moderate contention: per-block counters
__global__ void atomic_per_block(int* __restrict__ counters, int iters) {
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        atomicAdd(&counters[blockIdx.x], 1);
    }
}

int main(int argc, char** argv) {
    int mode = 0;      // 0=high, 1=low, 2=moderate
    int iters = 1000;

    if (argc > 1) mode = atoi(argv[1]);
    if (argc > 2) iters = atoi(argv[2]);

    int num_blocks = 64;
    int block_size = 256;
    int total_threads = num_blocks * block_size;

    fprintf(stderr, "Atomic: mode=%d (%s), iters=%d\n",
            mode,
            mode == 0 ? "single" : mode == 1 ? "private" : "per-block",
            iters);

    int *d_counters;
    size_t alloc_size;
    switch (mode) {
    case 0: alloc_size = sizeof(int); break;
    case 1: alloc_size = total_threads * sizeof(int); break;
    case 2: alloc_size = num_blocks * sizeof(int); break;
    default: alloc_size = sizeof(int); break;
    }

    HIP_CHECK(hipMalloc(&d_counters, alloc_size));
    HIP_CHECK(hipMemset(d_counters, 0, alloc_size));

    // Warmup
    switch (mode) {
    case 0: atomic_single<<<num_blocks, block_size>>>(d_counters, 10); break;
    case 1: atomic_private<<<num_blocks, block_size>>>(d_counters, 10); break;
    case 2: atomic_per_block<<<num_blocks, block_size>>>(d_counters, 10); break;
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Reset counters
    HIP_CHECK(hipMemset(d_counters, 0, alloc_size));

    // Measured
    switch (mode) {
    case 0: atomic_single<<<num_blocks, block_size>>>(d_counters, iters); break;
    case 1: atomic_private<<<num_blocks, block_size>>>(d_counters, iters); break;
    case 2: atomic_per_block<<<num_blocks, block_size>>>(d_counters, iters); break;
    }
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_counters));
    return 0;
}
