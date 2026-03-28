// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Common macros and helpers for metrix validation microbenchmarks.

#pragma once

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            fprintf(stderr, "HIP error %d (%s) at %s:%d\n", (int)err,         \
                    hipGetErrorString(err), __FILE__, __LINE__);               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// Parse a size argument like "512M" or "32M" into element count (float)
inline size_t parse_size_mb(const char* arg, size_t elem_size = sizeof(float)) {
    size_t mb = atoi(arg);
    return (mb * 1024ULL * 1024ULL) / elem_size;
}

// Parse a generic integer argument with a default
inline int parse_int(int argc, char** argv, int idx, int default_val) {
    if (idx < argc) return atoi(argv[idx]);
    return default_val;
}

// Warm up the GPU and establish clock frequencies
inline void gpu_warmup() {
    float* d;
    HIP_CHECK(hipMalloc(&d, 1024));
    HIP_CHECK(hipFree(d));
    HIP_CHECK(hipDeviceSynchronize());
}
