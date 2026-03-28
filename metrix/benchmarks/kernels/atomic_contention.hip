// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Benchmark 10: Global atomic contention.
// Validates: memory.atomic_latency (gfx942 only)
//
// Design: All threads atomicAdd to a single global address.
//         Maximum contention => high atomic latency (hundreds of cycles).
//         gfx942 uses TCC_EA0_ATOMIC_LEVEL_sum / TCC_EA0_ATOMIC_sum.
//
// Usage: atomic_contention [iterations]

#include "common.h"

__global__ void atomic_contention_kernel(int* __restrict__ counter, int iterations) {
    for (int i = 0; i < iterations; i++) {
        atomicAdd(counter, 1);
    }
}

// Low-contention version: each workgroup atomics to its own address
__global__ void atomic_low_contention(int* __restrict__ counters, int iterations) {
    for (int i = 0; i < iterations; i++) {
        atomicAdd(&counters[blockIdx.x], 1);
    }
}

int main(int argc, char** argv) {
    int mode = parse_int(argc, argv, 1, 1);  // 1=high contention, 0=low contention
    int iterations = parse_int(argc, argv, 2, 1000);
    int blockSize = 256;
    int num_wgs = 64;  // Fewer WGs to avoid timeout at high contention

    gpu_warmup();

    if (mode == 1) {
        // High contention: single counter
        int* d_counter;
        HIP_CHECK(hipMalloc(&d_counter, sizeof(int)));
        HIP_CHECK(hipMemset(d_counter, 0, sizeof(int)));

        hipLaunchKernelGGL(atomic_contention_kernel, dim3(num_wgs), dim3(blockSize), 0, 0,
                           d_counter, iterations);
        HIP_CHECK(hipDeviceSynchronize());

        int result;
        HIP_CHECK(hipMemcpy(&result, d_counter, sizeof(int), hipMemcpyDeviceToHost));
        printf("atomic_contention: high-contention, %d iterations, %d WGs, result=%d\n",
               iterations, num_wgs, result);

        HIP_CHECK(hipFree(d_counter));
    } else {
        // Low contention: per-WG counters
        int* d_counters;
        HIP_CHECK(hipMalloc(&d_counters, num_wgs * sizeof(int)));
        HIP_CHECK(hipMemset(d_counters, 0, num_wgs * sizeof(int)));

        hipLaunchKernelGGL(atomic_low_contention, dim3(num_wgs), dim3(blockSize), 0, 0,
                           d_counters, iterations);
        HIP_CHECK(hipDeviceSynchronize());

        printf("atomic_contention: low-contention, %d iterations, %d WGs\n",
               iterations, num_wgs);

        HIP_CHECK(hipFree(d_counters));
    }

    return 0;
}
