// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// One random gather between two coalesced accesses.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

__global__ void gather_scale(const float* in, const int* idx, float* out, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j = idx[i];
    float v = in[j];
    out[i] = v * s;
}

int main() {
    const int n = 1 << 22;
    std::vector<int> h_idx(n);
    std::iota(h_idx.begin(), h_idx.end(), 0);
    std::shuffle(h_idx.begin(), h_idx.end(), std::mt19937(42));  // fixed seed: same run every time
    std::vector<float> h_in(n, 1.0f);
    float *in, *out;
    int* idx;
    hipMalloc(&in, n * sizeof(float));
    hipMalloc(&out, n * sizeof(float));
    hipMalloc(&idx, n * sizeof(int));
    hipMemcpy(in, h_in.data(), n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(idx, h_idx.data(), n * sizeof(int), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(gather_scale, dim3((n + 255) / 256), dim3(256), 0, 0, in, idx, out, n, 2.0f);
    hipDeviceSynchronize();
    float first;
    hipMemcpy(&first, out, sizeof(float), hipMemcpyDeviceToHost);
    printf("out[0] = %.1f\n", first);
    hipFree(in);
    hipFree(out);
    hipFree(idx);
    return 0;
}
