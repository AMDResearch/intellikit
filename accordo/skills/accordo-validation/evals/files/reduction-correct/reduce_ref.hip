// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Reference: one atomicAdd per element.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

__global__ void reduce_sum(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(output, input[i]);
}

int main() {
    const int n = 1024;
    std::vector<float> h(n, 1.0f);
    float *in, *out;
    float result = 0.0f;
    hipMalloc(&in, n * sizeof(float));
    hipMalloc(&out, sizeof(float));
    hipMemcpy(in, h.data(), n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(out, &result, sizeof(float), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(reduce_sum, dim3((n + 255) / 256), dim3(256), 0, 0, in, out, n);
    hipDeviceSynchronize();
    hipMemcpy(&result, out, sizeof(float), hipMemcpyDeviceToHost);
    printf("sum = %.1f\n", result);
    hipFree(in);
    hipFree(out);
    return 0;
}
