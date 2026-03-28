// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Benchmark 4: Parameterized stride access.
// Validates: coalescing_efficiency at stride 1/2/4/8
//
// Design: Thread i reads src[i * stride]. With 64 threads per wave:
//   stride=1: 64 consecutive floats = 256 bytes = 2 cache lines (128B) => coalescing ~100%
//   stride=2: 64 threads span 512 bytes => 4 cache lines => coalescing ~50%
//   stride=4: 64 threads span 1024 bytes => 8 cache lines => coalescing ~25%
//   stride=8: 64 threads span 2048 bytes => 16 cache lines => coalescing ~12.5%
//
// Usage: bw_strided_access [stride] [size_mb]

#include "common.h"

__global__ void strided_read(const float* __restrict__ src, float* __restrict__ dst,
                             size_t N, int stride) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx = tid * stride;
    if (idx < N) {
        dst[tid % (N / stride)] = src[idx];
    }
}

int main(int argc, char** argv) {
    int stride = parse_int(argc, argv, 1, 1);
    int size_mb = parse_int(argc, argv, 2, 512);
    size_t N = (size_t)size_mb * 1024ULL * 1024ULL / sizeof(float);

    gpu_warmup();

    float *d_src, *d_dst;
    size_t src_bytes = N * sizeof(float);
    size_t dst_elems = N / stride;
    size_t dst_bytes = dst_elems * sizeof(float);
    HIP_CHECK(hipMalloc(&d_src, src_bytes));
    HIP_CHECK(hipMalloc(&d_dst, dst_bytes));
    HIP_CHECK(hipMemset(d_src, 1, src_bytes));

    int blockSize = 256;
    int gridSize = (dst_elems + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(strided_read, dim3(gridSize), dim3(blockSize), 0, 0,
                       d_src, d_dst, N, stride);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));

    printf("strided_access: stride=%d, %d MB, %zu elements\n", stride, size_mb, N);
    return 0;
}
