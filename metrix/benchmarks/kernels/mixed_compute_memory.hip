/*
 * mixed_compute_memory — Tunable arithmetic intensity.
 *
 * Each thread reads one float from HBM, performs K FMA operations on it,
 * writes the result back.
 *
 * Arithmetic intensity = (K * 2 * wavefront_size) / bytes_per_thread
 *   bytes_per_thread = 4 (read) + 4 (write) = 8
 *   For K=100:  AI = 200 * 64 / 8 = 1600 FLOP/byte (very compute-bound)
 *   For K=1:    AI = 2 * 64 / 8 = 16 FLOP/byte (moderate)
 *
 * Note: the per-thread byte count is 8 (one read + one write of float),
 * but the derived metric divides total FLOPS by total HBM bytes, which
 * accounts for cache line granularity.
 *
 * Validates: compute.hbm_arithmetic_intensity,
 *            compute.l2_arithmetic_intensity,
 *            compute.l1_arithmetic_intensity
 *
 * Usage: ./mixed_compute_memory [N_elements] [K_fma_ops]
 */

#include "common.h"

__global__ void mixed_kernel(const float* __restrict__ src,
                             float* __restrict__ dst,
                             size_t N, int K) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = src[idx];
        float acc = 0.0f;

        #pragma unroll 1
        for (int i = 0; i < K; i++) {
            acc = __builtin_fmaf(val, val, acc);
        }

        dst[idx] = acc;
    }
}

int main(int argc, char** argv) {
    size_t N = parse_size(argc, argv, 1, 64ULL * 1024 * 1024);  // 64M floats = 256 MB
    int K = parse_int(argc, argv, 2, 100);

    float *d_src, *d_dst;
    HIP_CHECK(hipMalloc(&d_src, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dst, N * sizeof(float)));
    HIP_CHECK(hipMemset(d_src, 0x3F, N * sizeof(float)));  // ~1.0f-ish

    int block = DEFAULT_BLOCK_SIZE;
    int grid = (N + block - 1) / block;

    mixed_kernel<<<grid, block>>>(d_src, d_dst, N, 1);
    HIP_CHECK(hipDeviceSynchronize());

    mixed_kernel<<<grid, block>>>(d_src, d_dst, N, K);
    HIP_CHECK(hipDeviceSynchronize());

    long long total_waves = ((long long)N + 63) / 64;
    long long expected_flops = total_waves * K * 2 * 64;

    printf("{\"benchmark\": \"mixed_compute_memory\", \"N\": %zu, \"K\": %d, "
           "\"bytes_rw\": %zu, \"expected_flops\": %lld}\n",
           N, K, 2 * N * sizeof(float), expected_flops);

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));
    return 0;
}
