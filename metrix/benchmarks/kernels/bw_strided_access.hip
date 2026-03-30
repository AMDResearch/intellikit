/*
 * bw_strided_access — Parameterized stride to validate coalescing metrics.
 *
 * Thread i reads src[i * stride].  Array is sized so that all accesses
 * are in-bounds even at maximum stride.
 *
 * Expected coalescing_efficiency:
 *   stride 1 → ~100%
 *   stride 2 → ~50%
 *   stride 4 → ~25%
 *   stride 8 → ~12.5%
 *
 * Validates: memory.coalescing_efficiency, memory.global_load_efficiency
 *
 * Usage: ./bw_strided_access [N_elements] [stride]
 */

#include "common.h"

__global__ void strided_read_kernel(const float* __restrict__ src,
                                    float* __restrict__ out,
                                    size_t N, int stride) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        size_t src_idx = idx * stride;
        out[0] += src[src_idx];
    }
}

int main(int argc, char** argv) {
    size_t N = parse_size(argc, argv, 1, 32ULL * 1024 * 1024);  // 32M threads
    int stride = parse_int(argc, argv, 2, 1);

    // Allocate enough for worst-case strided access
    size_t alloc_elems = N * stride;
    float *d_src, *d_out;
    HIP_CHECK(hipMalloc(&d_src, alloc_elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float)));
    HIP_CHECK(hipMemset(d_src, 0, alloc_elems * sizeof(float)));
    HIP_CHECK(hipMemset(d_out, 0, sizeof(float)));

    int block = DEFAULT_BLOCK_SIZE;
    int grid = (N + block - 1) / block;

    strided_read_kernel<<<grid, block>>>(d_src, d_out, N, stride);
    HIP_CHECK(hipDeviceSynchronize());

    strided_read_kernel<<<grid, block>>>(d_src, d_out, N, stride);
    HIP_CHECK(hipDeviceSynchronize());

    printf("{\"benchmark\": \"bw_strided_access\", \"N\": %zu, \"stride\": %d, "
           "\"bytes_touched\": %zu}\n", N, stride, alloc_elems * sizeof(float));

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
