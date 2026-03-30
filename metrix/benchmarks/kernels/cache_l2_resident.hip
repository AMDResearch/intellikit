/*
 * cache_l2_resident — Array fits in L2, iterated many times.
 *
 * A 32 MB float array is read repeatedly inside the kernel.  The first
 * pass populates L2; subsequent passes should hit L2.
 *
 * MI300X L2 = 256 MB, MI210 L2 = 16 MB.
 * We use 8 MB by default so it fits in both.
 *
 * Validates: memory.l2_hit_rate (expect >90%),
 *            memory.l2_bandwidth, memory.bytes_transferred_l2
 *
 * Usage: ./cache_l2_resident [N_elements] [iterations]
 */

#include "common.h"

__global__ void l2_resident_kernel(const float* __restrict__ src,
                                   float* __restrict__ out,
                                   size_t N, int iters) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (int i = 0; i < iters; i++) {
        if (idx < N) {
            acc += src[idx];
        }
    }
    if (idx == 0) out[0] = acc;
}

int main(int argc, char** argv) {
    // 2M floats = 8 MB — fits in both MI300X and MI210 L2
    size_t N = parse_size(argc, argv, 1, 2ULL * 1024 * 1024);
    int iters = parse_int(argc, argv, 2, 100);

    float *d_src, *d_out;
    HIP_CHECK(hipMalloc(&d_src, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float)));

    // Initialize with non-zero data
    float* h_src = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) h_src[i] = 1.0f;
    HIP_CHECK(hipMemcpy(d_src, h_src, N * sizeof(float), hipMemcpyHostToDevice));
    free(h_src);
    HIP_CHECK(hipMemset(d_out, 0, sizeof(float)));

    int block = DEFAULT_BLOCK_SIZE;
    int grid = (N + block - 1) / block;

    // Warmup to populate L2
    l2_resident_kernel<<<grid, block>>>(d_src, d_out, N, 1);
    HIP_CHECK(hipDeviceSynchronize());

    // Measured run — should see high L2 hit rate
    l2_resident_kernel<<<grid, block>>>(d_src, d_out, N, iters);
    HIP_CHECK(hipDeviceSynchronize());

    printf("{\"benchmark\": \"cache_l2_resident\", \"N\": %zu, \"iterations\": %d, "
           "\"array_bytes\": %zu}\n", N, iters, N * sizeof(float));

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
