// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Coalescing Efficiency Microbenchmark
//
// Purpose: Demonstrate the difference between coalesced and non-coalesced
// global memory access patterns to validate coalescing_efficiency,
// global_load_efficiency, and global_store_efficiency metrics.
//
// Modes:
//   0 = coalesced     — thread i reads/writes data[i] (stride-1)
//   1 = strided       — thread i reads/writes data[i * STRIDE]
//   2 = scattered     — thread i reads/writes data[hash(i)]
//
// Usage: ./coalescing [mode] [stride]
//   Default: mode=0 (coalesced), stride=32

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

// Coalesced: thread i accesses element i — perfect coalescing
__global__ void coalesced_kernel(const float* __restrict__ src,
                                  float* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * (size_t)blockDim.x;
    for (size_t i = idx; i < N; i += stride) {
        dst[i] = src[i] + 1.0f;
    }
}

// Strided: thread i accesses element i*STRIDE — poor coalescing
__global__ void strided_kernel(const float* __restrict__ src,
                                float* __restrict__ dst,
                                size_t N, size_t access_stride) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_stride = gridDim.x * (size_t)blockDim.x;
    for (size_t i = tid; i < N; i += grid_stride) {
        size_t idx = (i * access_stride) % N;
        dst[idx] = src[idx] + 1.0f;
    }
}

// Scattered: thread i accesses a pseudo-random location — worst coalescing
__device__ inline size_t simple_hash(size_t x, size_t N) {
    // Multiplicative hash — spreads accesses across address space
    x = ((x >> 16) ^ x) * 0x45d9f3bULL;
    x = ((x >> 16) ^ x) * 0x45d9f3bULL;
    x = (x >> 16) ^ x;
    return x % N;
}

__global__ void scattered_kernel(const float* __restrict__ src,
                                  float* __restrict__ dst, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_stride = gridDim.x * (size_t)blockDim.x;
    for (size_t i = tid; i < N; i += grid_stride) {
        size_t idx = simple_hash(i, N);
        dst[idx] = src[idx] + 1.0f;
    }
}

int main(int argc, char** argv) {
    int mode = 0;            // 0=coalesced, 1=strided, 2=scattered
    size_t stride = 32;      // stride for mode 1
    size_t total_bytes = 128ULL * 1024 * 1024;  // 128 MB

    if (argc > 1) mode = atoi(argv[1]);
    if (argc > 2) stride = (size_t)atoll(argv[2]);

    size_t N = total_bytes / sizeof(float);

    fprintf(stderr, "Coalescing: mode=%d (%s), stride=%zu, N=%zu\n",
            mode, mode == 0 ? "coalesced" : mode == 1 ? "strided" : "scattered",
            stride, N);

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    int block_size = 256;
    int num_blocks = prop.multiProcessorCount * 4;

    // For strided mode, allocate enough for the worst-case access
    size_t alloc_bytes = total_bytes;
    if (mode == 1) {
        // Accesses wrap via modulo, so same buffer size is fine
        alloc_bytes = total_bytes;
    }

    float *d_src, *d_dst;
    HIP_CHECK(hipMalloc(&d_src, alloc_bytes));
    HIP_CHECK(hipMalloc(&d_dst, alloc_bytes));
    HIP_CHECK(hipMemset(d_src, 0x3F, alloc_bytes));
    HIP_CHECK(hipMemset(d_dst, 0, alloc_bytes));

    // Warmup
    switch (mode) {
    case 0: coalesced_kernel<<<num_blocks, block_size>>>(d_src, d_dst, N); break;
    case 1: strided_kernel<<<num_blocks, block_size>>>(d_src, d_dst, N, stride); break;
    case 2: scattered_kernel<<<num_blocks, block_size>>>(d_src, d_dst, N); break;
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Measured
    switch (mode) {
    case 0: coalesced_kernel<<<num_blocks, block_size>>>(d_src, d_dst, N); break;
    case 1: strided_kernel<<<num_blocks, block_size>>>(d_src, d_dst, N, stride); break;
    case 2: scattered_kernel<<<num_blocks, block_size>>>(d_src, d_dst, N); break;
    }
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));
    return 0;
}
