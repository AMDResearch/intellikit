#include <hip/hip_runtime.h>

// Row-major traversal by column: each lane walks a different row, so adjacent
// lanes are a whole row apart -- one cache line per lane.
__global__ void scale_columns(float* __restrict__ a, int nx, int ny, float s) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= ny) return;
    for (int col = 0; col < nx; col++) {
        a[row * nx + col] *= s;
    }
}

// The same work with lanes along a row: coalesced.
__global__ void scale_rows(float* __restrict__ a, int nx, int ny, float s) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= nx) return;
    for (int row = 0; row < ny; row++) {
        a[row * nx + col] *= s;
    }
}

void scale_both(float* a, int nx, int ny) {
    hipLaunchKernelGGL(scale_columns, dim3(ny / 64), dim3(64), 0, 0, a, nx, ny, 0.5f);
    hipLaunchKernelGGL(scale_rows, dim3(nx / 64), dim3(64), 0, 0, a, nx, ny, 2.0f);
}
