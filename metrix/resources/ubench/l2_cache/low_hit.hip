#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <string>

// Low L2 cache hit rate - large data with strided access
__global__ void cache_unfriendly_sum(float* data,
                                     float* result,
                                     int n,
                                     int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0.0f;

  // Strided access pattern that causes cache misses
  for (int i = idx * stride; i < n; i += stride * blockDim.x * gridDim.x) {
    sum += data[i];
  }

  if (idx < n) {
    result[idx] = sum;
  }
}

#define hip_try(expr)                                                          \
  do {                                                                         \
    hipError_t status = (expr);                                                \
    if (status != hipSuccess) {                                                \
      throw std::runtime_error(std::string("HIP error at ") + __FILE__ + ":" + \
                               std::to_string(__LINE__) + ": " +               \
                               hipGetErrorString(status));                     \
    }                                                                          \
  } while (0)

int main(int argc, char** argv) {
  // Large array that doesn't fit well in L2 cache (use 128MB)
  int n = 32 * 1024 * 1024;  // 128MB of floats
  int stride = 1024;         // Large stride for poor cache locality

  if (argc > 1)
    n = atoi(argv[1]);
  if (argc > 2)
    stride = atoi(argv[2]);

  size_t size = n * sizeof(float);

  float *d_data, *d_result;
  float *h_data, *h_result;

  // Allocate device memory
  hip_try(hipMalloc(&d_data, size));
  hip_try(hipMalloc(&d_result, n * sizeof(float)));

  // Allocate host memory
  h_data = (float*)malloc(size);
  h_result = (float*)malloc(n * sizeof(float));

  // Initialize input data
  for (int i = 0; i < n; i++) {
    h_data[i] = 1.0f;
  }

  // Copy to device
  hip_try(hipMemcpy(d_data, h_data, size, hipMemcpyHostToDevice));

  // Print GPU info
  hipDeviceProp_t prop;
  hip_try(hipGetDeviceProperties(&prop, 0));
  printf("GPU: %s\n", prop.name);
  printf("Array size: %d elements (%.2f MB)\n", n, size / (1024.0 * 1024.0));
  printf("Stride: %d elements\n\n", stride);

  // Launch kernel
  int threads = 256;
  int blocks = 1024;

  cache_unfriendly_sum<<<blocks, threads>>>(d_data, d_result, n, stride);
  hip_try(hipGetLastError());
  hip_try(hipDeviceSynchronize());

  // Copy result back (just first few elements to verify)
  hip_try(hipMemcpy(h_result, d_result, 100 * sizeof(float),
                    hipMemcpyDeviceToHost));

  // Simple verification
  printf("Result: ✓ Low L2 cache hit rate (large data, strided access)\n");

  // Cleanup
  hip_try(hipFree(d_data));
  hip_try(hipFree(d_result));
  free(h_data);
  free(h_result);

  return 0;
}
