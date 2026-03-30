/*
 * bw_sequential_read — Pure coalesced read from HBM.
 *
 * Every thread reads one float from a contiguous array much larger than
 * L2 cache, accumulates to prevent dead-code elimination.
 *
 * Validates: memory.hbm_read_bandwidth, memory.bytes_transferred_hbm,
 *            memory.coalescing_efficiency, memory.global_load_efficiency,
 *            memory.l2_hit_rate (expect ~0%), memory.l1_hit_rate (expect ~0%)
 *
 * Usage: ./bw_sequential_read [N_elements]
 *   Default N = 128M floats = 512 MB (well above MI300X 256 MB L2)
 */

#include "common.h"

__global__ void sequential_read_kernel(const float* __restrict__ src,
                                       float* __restrict__ out,
                                       size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[0] += src[idx];  // accumulate to prevent elimination
    }
}

int main(int argc, char** argv) {
    size_t N = parse_size(argc, argv, 1, 128ULL * 1024 * 1024);  // 128M floats

    float* d_src;
    float* d_out;
    HIP_CHECK(hipMalloc(&d_src, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float)));
    HIP_CHECK(hipMemset(d_src, 0, N * sizeof(float)));
    HIP_CHECK(hipMemset(d_out, 0, sizeof(float)));

    int block = DEFAULT_BLOCK_SIZE;
    int grid = (N + block - 1) / block;

    // Warmup
    sequential_read_kernel<<<grid, block>>>(d_src, d_out, N);
    HIP_CHECK(hipDeviceSynchronize());

    // Measured run (profiled by rocprofv3)
    sequential_read_kernel<<<grid, block>>>(d_src, d_out, N);
    HIP_CHECK(hipDeviceSynchronize());

    size_t bytes = N * sizeof(float);
    printf("{\"benchmark\": \"bw_sequential_read\", \"N\": %zu, \"bytes\": %zu}\n", N, bytes);

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
