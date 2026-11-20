// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Streaming vector add: memory bound, so HBM bandwidth is the metric that matters.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 24;  // 16M floats, 64 MiB per buffer: far larger than L2
    std::vector<float> h(n, 1.0f);
    float *a, *b, *c;
    hipMalloc(&a, n * sizeof(float));
    hipMalloc(&b, n * sizeof(float));
    hipMalloc(&c, n * sizeof(float));
    hipMemcpy(a, h.data(), n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(b, h.data(), n * sizeof(float), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(vector_add, dim3((n + 255) / 256), dim3(256), 0, 0, a, b, c, n);
    hipDeviceSynchronize();
    hipMemcpy(h.data(), c, sizeof(float), hipMemcpyDeviceToHost);
    printf("c[0] = %.1f\n", h[0]);
    hipFree(a);
    hipFree(b);
    hipFree(c);
    return 0;
}
