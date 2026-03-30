/*
 * compute_valu_fma — Pure FP32 FMA compute, no memory traffic.
 *
 * Each thread executes N FMA instructions in registers.
 * Prevents dead-code elimination by writing the final result.
 *
 * Expected:
 *   SQ_INSTS_VALU_FMA_F32 = num_wavefronts * iters
 *   compute.total_flops = num_wavefronts * iters * 2 * 64
 *     (FMA = 2 FLOPS, wavefront_size = 64)
 *   All other FLOP counters ~ 0
 *   HBM traffic ~ 0 (compute-bound)
 *   hbm_arithmetic_intensity → very high
 *
 * Usage: ./compute_valu_fma [num_blocks] [iters_per_thread]
 */

#include "common.h"

__global__ void valu_fma_kernel(float* __restrict__ out, int iters) {
    float a = 1.0001f;
    float b = 0.9999f;
    float c = 0.0f;

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        c = __builtin_fmaf(a, b, c);
    }

    // Single write per block to prevent elimination without adding
    // significant memory traffic
    if (threadIdx.x == 0) {
        out[blockIdx.x] = c;
    }
}

int main(int argc, char** argv) {
    int num_blocks = parse_int(argc, argv, 1, 512);
    int iters = parse_int(argc, argv, 2, 100000);
    int block = 256;

    float* d_out;
    HIP_CHECK(hipMalloc(&d_out, num_blocks * sizeof(float)));
    HIP_CHECK(hipMemset(d_out, 0, num_blocks * sizeof(float)));

    // Warmup
    valu_fma_kernel<<<num_blocks, block>>>(d_out, 100);
    HIP_CHECK(hipDeviceSynchronize());

    // Measured run
    valu_fma_kernel<<<num_blocks, block>>>(d_out, iters);
    HIP_CHECK(hipDeviceSynchronize());

    long long total_threads = (long long)num_blocks * block;
    long long total_waves = total_threads / 64;  // wavefront_size = 64
    long long expected_fma_insts = total_waves * iters;
    long long expected_flops = expected_fma_insts * 2 * 64;

    printf("{\"benchmark\": \"compute_valu_fma\", \"num_blocks\": %d, "
           "\"block_size\": %d, \"iters\": %d, "
           "\"expected_fma_instructions\": %lld, "
           "\"expected_total_flops\": %lld}\n",
           num_blocks, block, iters, expected_fma_insts, expected_flops);

    HIP_CHECK(hipFree(d_out));
    return 0;
}
