// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Mini pipeline: five GPU kernels exercising common patterns
// (elementwise, scaling, fused bias+ReLU, shift, atomic histogram).
//
// Compile:  hipcc -O2 -o mini_pipeline mini_pipeline.hip
// Run:      ./mini_pipeline

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <limits>
#include <utility>
#include <vector>

__global__ void vector_add(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_scale(float* data, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

__global__ void vector_bias_relu(float* data, float bias, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = data[idx] + bias;
        data[idx] = (v > 0.0f) ? v : 0.0f;
    }
}

__global__ void vector_shift(const float* in, float* out, float shift, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] + shift;
    }
}

__global__ void histogram_atomic(const int* keys, int n, unsigned int* bins,
                                 int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int key = keys[idx];
        atomicAdd(&bins[key % num_bins], 1u);
    }
}

#define HIP_CHECK(cmd)                                                         \
    do {                                                                       \
        hipError_t err = (cmd);                                                \
        if (err != hipSuccess) {                                               \
            std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__,        \
                         __LINE__, hipGetErrorString(err));                     \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static void fill_inputs(std::vector<float>& a, std::vector<float>& b) {
    for (size_t i = 0; i < a.size(); ++i) {
        float x = static_cast<float>(i % 1024) * 0.001f;
        a[i] = std::sinf(x);
        b[i] = std::cosf(x) - 0.5f;
    }
}

static void fill_hot_keys(std::vector<int>& keys, int hot_bins) {
    for (size_t i = 0; i < keys.size(); ++i) {
        if ((i & 15u) != 0u) {
            keys[i] = 0;
        } else {
            keys[i] = static_cast<int>((i >> 4) % static_cast<size_t>(hot_bins));
        }
    }
}

int main() {
    const int n = 1 << 18;
    const int iters = 20;

    const float scale = 1.15f;
    const float bias = -0.10f;
    const float shift = 0.03f;
    const int histogram_multiplier = 4;
    const int num_bins = 32;
    const int hot_bins = 4;
    const long long hist_n_ll = static_cast<long long>(n) * histogram_multiplier;
    if (hist_n_ll <= 0 || hist_n_ll > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "Invalid histogram size for n=%d\n", n);
        return 2;
    }
    const int hist_n = static_cast<int>(hist_n_ll);

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const size_t hist_bytes = static_cast<size_t>(hist_n) * sizeof(int);
    const size_t bins_bytes = static_cast<size_t>(num_bins) * sizeof(unsigned int);

    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_out(n);
    std::vector<float> h_ref(n);
    std::vector<float> h_tmp(n);

    std::vector<int> h_keys(hist_n);
    std::vector<unsigned int> h_bins(num_bins, 0u);
    std::vector<unsigned int> h_bins_ref(num_bins, 0u);

    fill_inputs(h_a, h_b);
    fill_hot_keys(h_keys, hot_bins);
    h_ref = h_a;

    for (int key : h_keys) {
        h_bins_ref[key % num_bins] += 1u;
    }
    for (int i = 0; i < num_bins; ++i) {
        h_bins_ref[i] *= static_cast<unsigned int>(iters);
    }

    float *d_a = nullptr, *d_b = nullptr, *d_tmp = nullptr, *d_out = nullptr;
    int* d_keys = nullptr;
    unsigned int* d_bins = nullptr;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_tmp, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));
    HIP_CHECK(hipMalloc(&d_keys, hist_bytes));
    HIP_CHECK(hipMalloc(&d_bins, bins_bytes));

    HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_keys, h_keys.data(), hist_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_bins, 0, bins_bytes));

    const dim3 block(256);
    const dim3 grid((n + block.x - 1) / block.x);
    const dim3 hist_grid((hist_n + block.x - 1) / block.x);

    for (int iter = 0; iter < iters; ++iter) {
        vector_add<<<grid, block>>>(d_a, d_b, d_tmp, n);
        vector_scale<<<grid, block>>>(d_tmp, scale, n);
        vector_bias_relu<<<grid, block>>>(d_tmp, bias, n);
        vector_shift<<<grid, block>>>(d_tmp, d_out, shift, n);
        histogram_atomic<<<hist_grid, block>>>(d_keys, hist_n, d_bins, num_bins);
        std::swap(d_a, d_out);
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_out.data(), d_a, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_bins.data(), d_bins, bins_bytes, hipMemcpyDeviceToHost));

    for (int iter = 0; iter < iters; ++iter) {
        for (int i = 0; i < n; ++i) {
            float v = h_ref[i] + h_b[i];
            v *= scale;
            v += bias;
            v = (v > 0.0f) ? v : 0.0f;
            h_tmp[i] = v + shift;
        }
        std::swap(h_ref, h_tmp);
    }

    float max_abs_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_abs_err = std::max(max_abs_err, std::fabs(h_out[i] - h_ref[i]));
    }

    int bad_bins = 0;
    unsigned long long total_counts = 0;
    for (int i = 0; i < num_bins; ++i) {
        total_counts += static_cast<unsigned long long>(h_bins[i]);
        if (h_bins[i] != h_bins_ref[i]) {
            ++bad_bins;
        }
    }
    const unsigned long long expected_counts =
        static_cast<unsigned long long>(hist_n) * static_cast<unsigned long long>(iters);

    if (max_abs_err > 5e-5f || bad_bins != 0 || total_counts != expected_counts) {
        std::fprintf(stderr,
                     "mini_pipeline: FAIL (max_abs_err=%g, bad_bins=%d, total=%llu, expected=%llu)\n",
                     max_abs_err, bad_bins, total_counts, expected_counts);
        return 1;
    }

    double checksum = 0.0;
    for (int i = 0; i < std::min(n, 16); ++i) {
        checksum += h_out[i];
    }
    std::printf(
        "mini_pipeline: PASS (n=%d, iters=%d, hist_n=%d, max_abs_err=%g, hot_bin0=%u, checksum16=%.6f)\n",
        n, iters, hist_n, max_abs_err, h_bins[0], checksum);

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_tmp));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_keys));
    HIP_CHECK(hipFree(d_bins));
    return 0;
}
