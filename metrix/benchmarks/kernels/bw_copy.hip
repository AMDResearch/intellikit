/*
 * bw_copy — Read-write copy kernel (1:1 read/write ratio).
 *
 * dst[i] = src[i]  for all i.  Both arrays >> L2 size.
 *
 * Validates: memory.hbm_bandwidth_utilization, memory.bytes_transferred_hbm
 *            (should be ~2 * N * sizeof(float)),
 *            combined read+write bandwidth.
 *
 * Also exposes L1 hit rate formula divergence: IntelliKit only subtracts
 * TCP_TCC_READ_REQ_sum, omniperf also subtracts write/atomic misses.
 *
 * Usage: ./bw_copy [N_elements]
 */

#include "common.h"

__global__ void copy_kernel(const float* __restrict__ src,
                            float* __restrict__ dst, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = src[idx];
    }
}

int main(int argc, char** argv) {
    size_t N = parse_size(argc, argv, 1, 128ULL * 1024 * 1024);

    float *d_src, *d_dst;
    HIP_CHECK(hipMalloc(&d_src, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dst, N * sizeof(float)));
    HIP_CHECK(hipMemset(d_src, 0, N * sizeof(float)));

    int block = DEFAULT_BLOCK_SIZE;
    int grid = (N + block - 1) / block;

    copy_kernel<<<grid, block>>>(d_src, d_dst, N);
    HIP_CHECK(hipDeviceSynchronize());

    copy_kernel<<<grid, block>>>(d_src, d_dst, N);
    HIP_CHECK(hipDeviceSynchronize());

    printf("{\"benchmark\": \"bw_copy\", \"N\": %zu, \"bytes\": %zu}\n",
           N, 2 * N * sizeof(float));

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));
    return 0;
}
