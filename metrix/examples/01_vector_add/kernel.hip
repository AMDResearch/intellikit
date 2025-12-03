#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Simple vector addition - baseline for memory profiling
__global__ void vector_add(float* c, const float* a, const float* b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

#define HIP_CHECK(cmd)                                              \
  do {                                                              \
    hipError_t error = (cmd);                                       \
    if (error != hipSuccess) {                                      \
      fprintf(stderr, "HIP error: %s\n", hipGetErrorString(error)); \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

int main() {
  int n = 1024 * 1024;  // 1M elements = 4MB per array
  size_t size = n * sizeof(float);

  printf("Vector Addition Example\n");
  printf("=======================\n");
  printf("Array size: %d elements (%.2f MB per array)\n", n,
         size / (1024.0 * 1024.0));
  printf("Total data: %.2f MB\n\n", 3 * size / (1024.0 * 1024.0));

  // Allocate host memory
  float* h_a = (float*)malloc(size);
  float* h_b = (float*)malloc(size);
  float* h_c = (float*)malloc(size);

  // Initialize
  for (int i = 0; i < n; i++) {
    h_a[i] = (float)i;
    h_b[i] = (float)(i * 2);
  }

  // Allocate device memory
  float *d_a, *d_b, *d_c;
  HIP_CHECK(hipMalloc(&d_a, size));
  HIP_CHECK(hipMalloc(&d_b, size));
  HIP_CHECK(hipMalloc(&d_c, size));

  // Copy to device
  HIP_CHECK(hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice));

  // Launch kernel
  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;

  printf("Launching kernel: grid=%d, block=%d\n", gridSize, blockSize);
  vector_add<<<gridSize, blockSize>>>(d_c, d_a, d_b, n);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // Copy result back
  HIP_CHECK(hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost));

  // Verify
  bool correct = true;
  for (int i = 0; i < n && correct; i++) {
    if (h_c[i] != h_a[i] + h_b[i]) {
      printf("Error at index %d: got %f, expected %f\n", i, h_c[i],
             h_a[i] + h_b[i]);
      correct = false;
    }
  }

  printf("Result: %s\n", correct ? "✓ PASS" : "✗ FAIL");

  // Cleanup
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
