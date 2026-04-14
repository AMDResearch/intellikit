// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// LDS Bank Conflict Microbenchmark
//
// Purpose: Demonstrate LDS access patterns with and without bank conflicts
// to validate lds_utilization and lds_bank_conflicts metrics.
//
// Both CDNA and RDNA have 32 LDS banks with 4-byte bank width.
//   - Sequential: lds[threadIdx.x] — no conflicts
//   - Stride-32:  lds[(threadIdx.x * 32) % LDS_SIZE] — all threads in
//     a wavefront hit the same bank → maximum conflicts
//
// Modes:
//   0 = no conflicts    (sequential access)
//   1 = max conflicts   (stride-32 access)
//   2 = moderate         (stride-2 access)
//
// Usage: ./lds_conflicts [mode] [iterations]

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

#define LDS_SIZE 8192  // 8192 floats = 32 KB (fits in 64 KB LDS)

// No bank conflicts: sequential access
__global__ void lds_no_conflict(float* __restrict__ out, int iters) {
    __shared__ float lds[LDS_SIZE];
    int tid = threadIdx.x;

    // Initialize LDS
    if (tid < LDS_SIZE) lds[tid] = 1.0f;
    __syncthreads();

    float acc = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        acc += lds[tid % LDS_SIZE];
    }
    if (tid == 0) out[blockIdx.x] = acc;
}

// Maximum bank conflicts: all threads in a wavefront hit same bank
__global__ void lds_max_conflict(float* __restrict__ out, int iters) {
    __shared__ float lds[LDS_SIZE];
    int tid = threadIdx.x;

    // Initialize LDS
    if (tid < LDS_SIZE) lds[tid] = 1.0f;
    __syncthreads();

    float acc = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        // stride-32: all threads in a wavefront access same bank
        acc += lds[(tid * 32) % LDS_SIZE];
    }
    if (tid == 0) out[blockIdx.x] = acc;
}

// Moderate bank conflicts: stride-2 access
__global__ void lds_moderate_conflict(float* __restrict__ out, int iters) {
    __shared__ float lds[LDS_SIZE];
    int tid = threadIdx.x;

    // Initialize LDS
    if (tid < LDS_SIZE) lds[tid] = 1.0f;
    __syncthreads();

    float acc = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        // stride-2: half the bank conflicts of stride-32
        acc += lds[(tid * 2) % LDS_SIZE];
    }
    if (tid == 0) out[blockIdx.x] = acc;
}

int main(int argc, char** argv) {
    int mode = 0;        // 0=no conflict, 1=max conflict, 2=moderate
    int iters = 10000;

    if (argc > 1) mode = atoi(argv[1]);
    if (argc > 2) iters = atoi(argv[2]);

    int num_blocks = 128;
    int block_size = 256;

    fprintf(stderr, "LDS: mode=%d (%s), iters=%d, blocks=%d\n",
            mode,
            mode == 0 ? "sequential" : mode == 1 ? "stride-32" : "stride-2",
            iters, num_blocks);

    float *d_out;
    HIP_CHECK(hipMalloc(&d_out, num_blocks * sizeof(float)));
    HIP_CHECK(hipMemset(d_out, 0, num_blocks * sizeof(float)));

    // Warmup
    switch (mode) {
    case 0: lds_no_conflict<<<num_blocks, block_size>>>(d_out, 10); break;
    case 1: lds_max_conflict<<<num_blocks, block_size>>>(d_out, 10); break;
    case 2: lds_moderate_conflict<<<num_blocks, block_size>>>(d_out, 10); break;
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Measured
    switch (mode) {
    case 0: lds_no_conflict<<<num_blocks, block_size>>>(d_out, iters); break;
    case 1: lds_max_conflict<<<num_blocks, block_size>>>(d_out, iters); break;
    case 2: lds_moderate_conflict<<<num_blocks, block_size>>>(d_out, iters); break;
    }
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_out));
    return 0;
}
