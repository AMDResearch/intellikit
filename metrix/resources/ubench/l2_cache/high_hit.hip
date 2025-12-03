#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <string>

// High L2 cache hit rate - small data accessed repeatedly
__global__ void cache_friendly_sum(float* data,
                                   float* result,
                                   int n,
                                   int iterations) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0.0f;

  // Access the same small array multiple times (high temporal locality)
  for (int iter = 0; iter < iterations; iter++) {
    if (idx < n) {
      sum += data[idx];
    }
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
  // Small array that fits in L2 cache (MI300X has 256MB L2, use ~2MB)
  int n = 512 * 1024;    // 2MB of floats
  int iterations = 100;  // Access same data 100 times

  if (argc > 1)
    n = atoi(argv[1]);
  if (argc > 2)
    iterations = atoi(argv[2]);

  size_t size = n * sizeof(float);

  float *d_data, *d_result;
  float *h_data, *h_result;

  // Allocate device memory
  hip_try(hipMalloc(&d_data, size));
  hip_try(hipMalloc(&d_result, size));

  // Allocate host memory
  h_data = (float*)malloc(size);
  h_result = (float*)malloc(size);

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
  printf("Iterations: %d\n", iterations);
  printf("Total data accessed: %.2f MB\n\n",
         (size * iterations) / (1024.0 * 1024.0));

  // Launch kernel
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  cache_friendly_sum<<<blocks, threads>>>(d_data, d_result, n, iterations);
  hip_try(hipGetLastError());
  hip_try(hipDeviceSynchronize());

  // Copy result back
  hip_try(hipMemcpy(h_result, d_result, size, hipMemcpyDeviceToHost));

  // Verify results
  bool correct = true;
  float expected = (float)iterations;
  for (int i = 0; i < n && i < 100; i++) {
    if (h_result[i] != expected) {
      printf("Error: result[%d] = %.0f, expected %.0f\n", i, h_result[i],
             expected);
      correct = false;
      break;
    }
  }

  if (correct) {
    printf("Result: ✓ High L2 cache hit rate (small data, repeated access)\n");
  } else {
    printf("Result: ✗ Incorrect\n");
  }

  // Cleanup
  hip_try(hipFree(d_data));
  hip_try(hipFree(d_result));
  free(h_data);
  free(h_result);

  return 0;
}
