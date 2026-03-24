#include "helpers.cuh"

template <typename T>
__global__ void mul_mat_vec_q(const T* a, const T* b, T* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = fma_helper(a[idx], b[idx]);
    }
}

template __global__ void mul_mat_vec_q<float>(const float*, const float*, float*, int);
