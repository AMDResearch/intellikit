/*
 * atomic_contention — Global atomic operations for atomic_latency.
 *
 * Mode 0 (high contention): All threads atomicAdd to a single address.
 * Mode 1 (low contention):  Each thread atomicAdd to its own address.
 *
 * Validates: memory.atomic_latency (gfx942 only)
 *   High contention → high latency (hundreds of cycles)
 *   Low contention → low latency
 *
 * Usage: ./atomic_contention [mode] [iterations]
 */

#include "common.h"

__global__ void atomic_high_contention(int* __restrict__ counter, int iters) {
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        atomicAdd(counter, 1);
    }
}

__global__ void atomic_low_contention(int* __restrict__ counters, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        atomicAdd(&counters[tid], 1);
    }
}

int main(int argc, char** argv) {
    int mode = parse_int(argc, argv, 1, 0);
    int iters = parse_int(argc, argv, 2, 1000);
    int num_blocks = 64;
    int block = 256;
    int total_threads = num_blocks * block;

    int* d_counters;
    HIP_CHECK(hipMalloc(&d_counters, total_threads * sizeof(int)));
    HIP_CHECK(hipMemset(d_counters, 0, total_threads * sizeof(int)));

    if (mode == 0) {
        atomic_high_contention<<<num_blocks, block>>>(d_counters, 10);
        HIP_CHECK(hipDeviceSynchronize());
        atomic_high_contention<<<num_blocks, block>>>(d_counters, iters);
    } else {
        atomic_low_contention<<<num_blocks, block>>>(d_counters, 10);
        HIP_CHECK(hipDeviceSynchronize());
        atomic_low_contention<<<num_blocks, block>>>(d_counters, iters);
    }
    HIP_CHECK(hipDeviceSynchronize());

    printf("{\"benchmark\": \"atomic_contention\", \"mode\": %d, "
           "\"mode_name\": \"%s\", \"iterations\": %d}\n",
           mode, mode == 0 ? "high_contention" : "low_contention", iters);

    HIP_CHECK(hipFree(d_counters));
    return 0;
}
