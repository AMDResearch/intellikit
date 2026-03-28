// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Benchmark 7: LDS bank conflict measurement.
// Validates: lds_bank_conflicts
//
// Design: Two modes controlled by command-line argument.
//   mode=0 (conflict-free): Each thread accesses consecutive LDS elements.
//     Thread i reads lds[i] => no bank conflicts.
//   mode=1 (high-conflict): All threads access same bank.
//     Thread i reads lds[i * 32] => all 64 threads in a wave hit bank 0.
//     Expect high conflicts.
//
// Usage: lds_bank_conflicts [mode] [iterations]
//   mode: 0=conflict-free, 1=high-conflict (default: 0)

#include "common.h"

#define LDS_SIZE 4096  // Elements in shared memory

__global__ void lds_conflict_free(float* __restrict__ out, int iterations) {
    __shared__ float lds[LDS_SIZE];

    int tid = threadIdx.x;
    // Initialize LDS
    for (int i = tid; i < LDS_SIZE; i += blockDim.x) {
        lds[i] = (float)i;
    }
    __syncthreads();

    float sum = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        // Conflict-free: sequential access, each thread reads its own bank
        sum += lds[tid];
    }
    out[blockIdx.x * blockDim.x + tid] = sum;
}

__global__ void lds_high_conflict(float* __restrict__ out, int iterations) {
    __shared__ float lds[LDS_SIZE];

    int tid = threadIdx.x;
    // Initialize LDS
    for (int i = tid; i < LDS_SIZE; i += blockDim.x) {
        lds[i] = (float)i;
    }
    __syncthreads();

    float sum = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        // High conflict: stride-32 access, all threads in a wave hit same bank
        // LDS has 32 banks, each 4 bytes wide.
        // lds[tid * 32 % LDS_SIZE] => all threads mod 32 go to bank 0
        int idx = (tid * 32) % LDS_SIZE;
        sum += lds[idx];
    }
    out[blockIdx.x * blockDim.x + tid] = sum;
}

int main(int argc, char** argv) {
    int mode = parse_int(argc, argv, 1, 0);
    int iterations = parse_int(argc, argv, 2, 10000);
    int num_wgs = 304;  // One per CU

    gpu_warmup();

    int blockSize = 256;
    size_t out_size = (size_t)num_wgs * blockSize * sizeof(float);
    float* d_out;
    HIP_CHECK(hipMalloc(&d_out, out_size));

    if (mode == 0) {
        hipLaunchKernelGGL(lds_conflict_free, dim3(num_wgs), dim3(blockSize), 0, 0,
                           d_out, iterations);
    } else {
        hipLaunchKernelGGL(lds_high_conflict, dim3(num_wgs), dim3(blockSize), 0, 0,
                           d_out, iterations);
    }
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_out));

    printf("lds_bank_conflicts: mode=%s, %d iterations\n",
           mode == 0 ? "conflict-free" : "high-conflict", iterations);
    return 0;
}
