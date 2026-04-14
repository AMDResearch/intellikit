// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Cache-Thrashing Microbenchmark
//
// Purpose: Access memory with large strides over a buffer much larger than
// L2 cache, producing low cache hit rates. Validates that l2_hit_rate
// reports low values for cache-unfriendly access patterns.
//
// Design:
//   - 256 MB buffer >> any L2 cache
//   - Large stride (cache-line-sized) to defeat prefetching
//   - Each thread touches widely separated addresses → cache misses
//   - Grid-stride loop ensures all data is touched
//
// Usage: ./cache_thrash [total_mb] [stride_elements]

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

__global__ void cache_thrash_kernel(const float* __restrict__ src,
                                     float* __restrict__ out,
                                     size_t N, size_t stride) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_stride = gridDim.x * (size_t)blockDim.x;
    float acc = 0.0f;

    // Each thread accesses elements separated by `stride`, wrapping around
    for (size_t i = tid; i < N; i += grid_stride) {
        size_t idx = (i * stride) % N;
        acc += src[idx];
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

int main(int argc, char** argv) {
    size_t total_mb = 256;
    // Stride in elements: 16 floats = 64 bytes = typical cache line
    size_t stride_elements = 16;

    if (argc > 1) total_mb = (size_t)atoll(argv[1]);
    if (argc > 2) stride_elements = (size_t)atoll(argv[2]);

    size_t N = (total_mb * 1024 * 1024) / sizeof(float);
    size_t bytes = N * sizeof(float);

    fprintf(stderr, "Cache-thrash: buffer=%zu MB, stride=%zu elements\n",
            total_mb, stride_elements);

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    int block_size = 256;
    int num_blocks = prop.multiProcessorCount * 4;

    float *d_src, *d_out;
    HIP_CHECK(hipMalloc(&d_src, bytes));
    HIP_CHECK(hipMalloc(&d_out, num_blocks * sizeof(float)));
    HIP_CHECK(hipMemset(d_src, 0x3F, bytes));
    HIP_CHECK(hipMemset(d_out, 0, num_blocks * sizeof(float)));

    // Warmup
    cache_thrash_kernel<<<num_blocks, block_size>>>(d_src, d_out, N, stride_elements);
    HIP_CHECK(hipDeviceSynchronize());

    // Measured
    cache_thrash_kernel<<<num_blocks, block_size>>>(d_src, d_out, N, stride_elements);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
