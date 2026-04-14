// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Cache-Friendly Microbenchmark
//
// Purpose: Repeatedly read a small array that fits in L2 (and L1) cache
// to produce high cache hit rates. Validates l2_hit_rate and l1_hit_rate
// metrics report elevated values.
//
// Design:
//   - 256 KB array (fits in any L2: 4MB+ on CDNA, 2MB+ on RDNA)
//   - Few blocks to reduce L2 set contention across CUs
//   - Many iterations over the same data → high temporal reuse
//   - Warmup pass fills caches, measured pass should hit heavily
//
// Usage: ./cache_friendly [array_kb] [iterations] [num_blocks]

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

__global__ void cache_friendly_kernel(const float* __restrict__ src,
                                       float* __restrict__ out,
                                       size_t N, int iters) {
    float acc = 0.0f;
    for (int it = 0; it < iters; it++) {
        size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) % N;
        acc += src[idx];
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

int main(int argc, char** argv) {
    size_t array_kb = 256;       // 256 KB — fits in any L2
    int iters = 500;
    int num_blocks = 16;         // Few blocks → less cache contention
    int block_size = 256;

    if (argc > 1) array_kb = (size_t)atoll(argv[1]);
    if (argc > 2) iters = atoi(argv[2]);
    if (argc > 3) num_blocks = atoi(argv[3]);

    size_t N = (array_kb * 1024) / sizeof(float);
    size_t bytes = N * sizeof(float);

    fprintf(stderr, "Cache-friendly: array=%zu KB, iters=%d, blocks=%d\n",
            array_kb, iters, num_blocks);

    float *d_src, *d_out;
    HIP_CHECK(hipMalloc(&d_src, bytes));
    HIP_CHECK(hipMalloc(&d_out, num_blocks * sizeof(float)));

    // Initialize with non-zero data
    float* h = (float*)malloc(bytes);
    for (size_t i = 0; i < N; i++) h[i] = 1.0f;
    HIP_CHECK(hipMemcpy(d_src, h, bytes, hipMemcpyHostToDevice));
    free(h);
    HIP_CHECK(hipMemset(d_out, 0, num_blocks * sizeof(float)));

    // Warmup — fill caches
    cache_friendly_kernel<<<num_blocks, block_size>>>(d_src, d_out, N, 10);
    HIP_CHECK(hipDeviceSynchronize());

    // Measured — should hit L2 heavily
    cache_friendly_kernel<<<num_blocks, block_size>>>(d_src, d_out, N, iters);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
