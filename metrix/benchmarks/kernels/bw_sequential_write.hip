/*
 * bw_sequential_write — Pure coalesced write to HBM.
 *
 * Every thread writes a constant to a contiguous position in a large array.
 *
 * Validates: memory.hbm_write_bandwidth, memory.global_store_efficiency,
 *            memory.bytes_transferred_hbm (write component)
 *
 * Usage: ./bw_sequential_write [N_elements]
 */

#include "common.h"

__global__ void sequential_write_kernel(float* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = 1.0f;
    }
}

int main(int argc, char** argv) {
    size_t N = parse_size(argc, argv, 1, 128ULL * 1024 * 1024);

    float* d_dst;
    HIP_CHECK(hipMalloc(&d_dst, N * sizeof(float)));

    int block = DEFAULT_BLOCK_SIZE;
    int grid = (N + block - 1) / block;

    sequential_write_kernel<<<grid, block>>>(d_dst, N);
    HIP_CHECK(hipDeviceSynchronize());

    sequential_write_kernel<<<grid, block>>>(d_dst, N);
    HIP_CHECK(hipDeviceSynchronize());

    printf("{\"benchmark\": \"bw_sequential_write\", \"N\": %zu, \"bytes\": %zu}\n",
           N, N * sizeof(float));

    HIP_CHECK(hipFree(d_dst));
    return 0;
}
