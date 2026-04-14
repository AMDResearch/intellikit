// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// HBM Bandwidth Microbenchmark
//
// Purpose: Saturate HBM bandwidth to validate metrix bandwidth metrics.
//
// Three kernel modes:
//   0 = copy   (read + write)  — measures both directions
//   1 = read   (read-only)     — isolates read bandwidth
//   2 = write  (write-only)    — isolates write bandwidth
//
// Four data types (selectable via argv):
//   0 = float4  (16 bytes/element — best bus utilization)
//   1 = double2 (16 bytes/element)
//   2 = float   (4 bytes/element  — scalar baseline)
//   3 = double  (8 bytes/element)
//
// Usage: ./hbm_bandwidth [mode] [dtype] [total_bytes]
//   Default: mode=0 (copy), dtype=0 (float4), total_bytes=256MB
//
// Design choices:
//   - 256MB buffers >> any L2 cache (4-8MB CDNA, 2-6MB RDNA)
//   - Grid-stride (persistent) loop saturates memory bus
//   - Auto-tuned grid: ~4 wavefronts per CU via hipDeviceProp_t
//   - Dummy write on read-only prevents dead-code elimination
//   - Two kernel launches: warmup + measured (profiler captures second)

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

// ---------------------------------------------------------------------------
// Kernel templates
// ---------------------------------------------------------------------------

// Copy: read src, write dst — grid-stride persistent loop
template <typename T>
__global__ void hbm_copy(const T* __restrict__ src, T* __restrict__ dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * (size_t)blockDim.x;
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

// Read-only: accumulate loads, single dummy write to prevent DCE
template <typename T>
__global__ void hbm_read(const T* __restrict__ src, T* __restrict__ dummy, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * (size_t)blockDim.x;
    T val{};
    for (size_t i = idx; i < n; i += stride) {
        val = src[i];
    }
    // Single dummy write per thread 0 — negligible write traffic
    if (idx == 0) dummy[0] = val;
}

// Write-only: fill dst with a constant value
template <typename T>
__global__ void hbm_write(T* __restrict__ dst, T val, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * (size_t)blockDim.x;
    for (size_t i = idx; i < n; i += stride) {
        dst[i] = val;
    }
}

// ---------------------------------------------------------------------------
// Grid configuration
// ---------------------------------------------------------------------------

static void get_grid_config(int blockSize, int* numBlocks) {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    int cuCount = prop.multiProcessorCount;
    int wavefrontSize = prop.warpSize;  // 64 on CDNA, 32 on RDNA
    int wavesPerBlock = blockSize / wavefrontSize;

    // Target ~4 wavefronts per CU for occupancy
    *numBlocks = cuCount * 4 / wavesPerBlock;
    if (*numBlocks < 1) *numBlocks = 1;

    fprintf(stderr, "Grid config: %d CUs, wavefront=%d, blocks=%d, blockSize=%d\n",
            cuCount, wavefrontSize, *numBlocks, blockSize);
}

// ---------------------------------------------------------------------------
// Type-specific dispatch
// ---------------------------------------------------------------------------

template <typename T>
static void run_benchmark(int mode, T fill_val, size_t total_bytes) {
    size_t n = total_bytes / sizeof(T);

    T *d_src = nullptr, *d_dst = nullptr;
    HIP_CHECK(hipMalloc(&d_src, total_bytes));
    HIP_CHECK(hipMalloc(&d_dst, total_bytes));
    HIP_CHECK(hipMemset(d_src, 0x3F, total_bytes));
    HIP_CHECK(hipMemset(d_dst, 0, total_bytes));

    int blockSize = 256;
    int numBlocks;
    get_grid_config(blockSize, &numBlocks);

    // Warmup
    if (mode == 0)      hbm_copy<<<numBlocks, blockSize>>>(d_src, d_dst, n);
    else if (mode == 1) hbm_read<<<numBlocks, blockSize>>>(d_src, d_dst, n);
    else                hbm_write<<<numBlocks, blockSize>>>(d_dst, fill_val, n);
    HIP_CHECK(hipDeviceSynchronize());

    // Measured pass (profiler captures this one)
    if (mode == 0)      hbm_copy<<<numBlocks, blockSize>>>(d_src, d_dst, n);
    else if (mode == 1) hbm_read<<<numBlocks, blockSize>>>(d_src, d_dst, n);
    else                hbm_write<<<numBlocks, blockSize>>>(d_dst, fill_val, n);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    int mode = 0;   // 0=copy, 1=read, 2=write
    int dtype = 0;  // 0=float4, 1=double2, 2=float, 3=double
    size_t total_bytes = 256ULL * 1024 * 1024;  // 256 MB

    if (argc > 1) mode = atoi(argv[1]);
    if (argc > 2) dtype = atoi(argv[2]);
    if (argc > 3) total_bytes = (size_t)atoll(argv[3]);

    fprintf(stderr, "HBM benchmark: mode=%d (%s), dtype=%d, bytes=%zu\n",
            mode, mode == 0 ? "copy" : mode == 1 ? "read" : "write",
            dtype, total_bytes);

    switch (dtype) {
    case 0: {
        float4 fill = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        run_benchmark<float4>(mode, fill, total_bytes);
        break;
    }
    case 1: {
        double2 fill = make_double2(1.0, 1.0);
        run_benchmark<double2>(mode, fill, total_bytes);
        break;
    }
    case 2: {
        run_benchmark<float>(mode, 1.0f, total_bytes);
        break;
    }
    case 3: {
        run_benchmark<double>(mode, 1.0, total_bytes);
        break;
    }
    default:
        fprintf(stderr, "Unknown dtype %d (use 0-3)\n", dtype);
        return 1;
    }

    return 0;
}
