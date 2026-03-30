/*
 * common.h — Shared macros for metrix counter-validation microbenchmarks.
 *
 * All benchmarks:
 *   - Accept optional command-line args for problem size / parameters
 *   - Print a one-line JSON summary to stdout for the validation harness
 *   - Use HIP_CHECK for error handling
 */

#pragma once

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(call)                                                    \
    do {                                                                   \
        hipError_t err = (call);                                           \
        if (err != hipSuccess) {                                           \
            fprintf(stderr, "HIP error: %s (%d) at %s:%d\n",              \
                    hipGetErrorString(err), err, __FILE__, __LINE__);      \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// Default block size for 1-D kernels
#define DEFAULT_BLOCK_SIZE 256

// Parse a size_t from argv, or return default
static inline size_t parse_size(int argc, char** argv, int idx, size_t def) {
    if (idx < argc) {
        char* end;
        size_t v = strtoull(argv[idx], &end, 10);
        if (*end == '\0' && v > 0) return v;
    }
    return def;
}

// Parse an int from argv, or return default
static inline int parse_int(int argc, char** argv, int idx, int def) {
    if (idx < argc) {
        char* end;
        long v = strtol(argv[idx], &end, 10);
        if (*end == '\0') return static_cast<int>(v);
    }
    return def;
}
