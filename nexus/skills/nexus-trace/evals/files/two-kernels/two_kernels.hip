// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Launches two different kernels, once each.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

__global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

__global__ void square(float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = y[i] * y[i];
}

int main() {
    const int n = 1 << 16;
    std::vector<float> h(n, 1.0f);
    float *x, *y;
    hipMalloc(&x, n * sizeof(float));
    hipMalloc(&y, n * sizeof(float));
    hipMemcpy(x, h.data(), n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(y, h.data(), n * sizeof(float), hipMemcpyHostToDevice);
    dim3 grid((n + 255) / 256), block(256);
    hipLaunchKernelGGL(saxpy, grid, block, 0, 0, 2.0f, x, y, n);
    hipLaunchKernelGGL(square, grid, block, 0, 0, y, n);
    hipDeviceSynchronize();
    hipMemcpy(h.data(), y, sizeof(float), hipMemcpyDeviceToHost);
    printf("y[0] = %.1f\n", h[0]);  // (2*1 + 1)^2 = 9
    hipFree(x);
    hipFree(y);
    return 0;
}
