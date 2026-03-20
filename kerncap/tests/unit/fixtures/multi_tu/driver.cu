#include <cstdio>
#include "kernels.cuh"

__global__ void some_other_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= 2.0f;
}

void do_mul_mat_vec_q(const float* a, const float* b, float* c, int n) {
    launch_mul_mat_vec_q(a, b, c, n);
}

int main() {
    printf("driver\n");
    return 0;
}
