// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Optimized: shared-memory tree reduction, one atomicAdd per block.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

__global__ void reduce_sum(const float* input, float* output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 4; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, sdata[0]);
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
