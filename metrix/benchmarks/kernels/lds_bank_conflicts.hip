/*
 * lds_bank_conflicts — LDS access with and without bank conflicts.
 *
 * Mode 0 (conflict-free): Each thread accesses lds[threadIdx.x]
 * Mode 1 (high-conflict): Each thread accesses lds[threadIdx.x * 32]
 *     (all threads in a wavefront hit the same bank)
 *
 * Validates: memory.lds_bank_conflicts
 *   Mode 0: SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS ~ 0
 *   Mode 1: SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS >> 0
 *
 * Usage: ./lds_bank_conflicts [mode] [iterations]
 *   mode 0 = conflict-free (default)
 *   mode 1 = high-conflict
 */

#include "common.h"

// 32 KB LDS — well within per-CU limits
#define LDS_SIZE 8192  // 8K floats = 32 KB

__global__ void lds_conflict_free_kernel(float* __restrict__ out, int iters) {
    __shared__ float lds[LDS_SIZE];
    int tid = threadIdx.x;

    // Initialize
    if (tid < LDS_SIZE) lds[tid] = 1.0f;
    __syncthreads();

    float acc = 0.0f;
    for (int i = 0; i < iters; i++) {
        // Conflict-free: consecutive threads access consecutive banks
        acc += lds[tid % LDS_SIZE];
    }

    if (tid == 0) out[blockIdx.x] = acc;
}

__global__ void lds_high_conflict_kernel(float* __restrict__ out, int iters) {
    __shared__ float lds[LDS_SIZE];
    int tid = threadIdx.x;

    if (tid < LDS_SIZE) lds[tid] = 1.0f;
    __syncthreads();

    float acc = 0.0f;
    for (int i = 0; i < iters; i++) {
        // High conflict: stride-32 means all threads in a wavefront
        // access the same bank (bank = addr % 32)
        int idx = (tid * 32) % LDS_SIZE;
        acc += lds[idx];
    }

    if (tid == 0) out[blockIdx.x] = acc;
}

int main(int argc, char** argv) {
    int mode = parse_int(argc, argv, 1, 0);
    int iters = parse_int(argc, argv, 2, 10000);
    int num_blocks = 128;

    float* d_out;
    HIP_CHECK(hipMalloc(&d_out, num_blocks * sizeof(float)));
    HIP_CHECK(hipMemset(d_out, 0, num_blocks * sizeof(float)));

    int block = 256;

    // Warmup
    if (mode == 0) {
        lds_conflict_free_kernel<<<num_blocks, block>>>(d_out, 1);
    } else {
        lds_high_conflict_kernel<<<num_blocks, block>>>(d_out, 1);
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Measured run
    if (mode == 0) {
        lds_conflict_free_kernel<<<num_blocks, block>>>(d_out, iters);
    } else {
        lds_high_conflict_kernel<<<num_blocks, block>>>(d_out, iters);
    }
    HIP_CHECK(hipDeviceSynchronize());

    printf("{\"benchmark\": \"lds_bank_conflicts\", \"mode\": %d, "
           "\"mode_name\": \"%s\", \"iterations\": %d}\n",
           mode, mode == 0 ? "conflict_free" : "high_conflict", iters);

    HIP_CHECK(hipFree(d_out));
    return 0;
}
