/*
 * cache_l1_resident — Tiny array fits in L1 per CU, iterated many times.
 *
 * Each workgroup reads the same small buffer (fits in per-CU vL1D cache)
 * repeatedly.  gfx942 L1 = 32 KB, gfx90a L1 = 16 KB.
 * We use 4 KB (1024 floats) to fit comfortably.
 *
 * Validates: memory.l1_hit_rate (expect >95%),
 *            memory.bytes_transferred_l1
 *
 * Usage: ./cache_l1_resident [N_per_block] [iterations]
 */

#include "common.h"

__global__ void l1_resident_kernel(const float* __restrict__ src,
                                   float* __restrict__ out,
                                   size_t N_per_block, int iters) {
    float acc = 0.0f;
    size_t idx = threadIdx.x;
    for (int i = 0; i < iters; i++) {
        if (idx < N_per_block) {
            acc += src[idx];
        }
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

int main(int argc, char** argv) {
    // 1024 floats = 4 KB per block — fits in L1
    size_t N_per_block = parse_size(argc, argv, 1, 1024);
    int iters = parse_int(argc, argv, 2, 1000);
    int num_blocks = 256;  // enough blocks to utilize all CUs

    float *d_src, *d_out;
    HIP_CHECK(hipMalloc(&d_src, N_per_block * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, num_blocks * sizeof(float)));

    float* h_src = (float*)malloc(N_per_block * sizeof(float));
    for (size_t i = 0; i < N_per_block; i++) h_src[i] = 1.0f;
    HIP_CHECK(hipMemcpy(d_src, h_src, N_per_block * sizeof(float), hipMemcpyHostToDevice));
    free(h_src);
    HIP_CHECK(hipMemset(d_out, 0, num_blocks * sizeof(float)));

    int block = (N_per_block < 256) ? N_per_block : 256;

    // Warmup
    l1_resident_kernel<<<num_blocks, block>>>(d_src, d_out, N_per_block, 1);
    HIP_CHECK(hipDeviceSynchronize());

    // Measured run
    l1_resident_kernel<<<num_blocks, block>>>(d_src, d_out, N_per_block, iters);
    HIP_CHECK(hipDeviceSynchronize());

    printf("{\"benchmark\": \"cache_l1_resident\", \"N_per_block\": %zu, "
           "\"iterations\": %d, \"num_blocks\": %d}\n", N_per_block, iters, num_blocks);

    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
