#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <string>

// Coalesced access - consecutive threads access consecutive memory locations
__global__ void memcpy_coalesced(float* dst, const float* src, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n) {
    dst[idx] = src[idx];  // Coalesced: thread i accesses element i
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
  // Default parameters
  int num_blocks = 256;
  int num_threads = 256;

  // Parse command line arguments
  if (argc > 1)
    num_blocks = atoi(argv[1]);
  if (argc > 2)
    num_threads = atoi(argv[2]);

  int n = num_blocks * num_threads;
  size_t size = n * sizeof(float);

  float *d_src, *d_dst;
  float *h_src, *h_dst;

  // Allocate device memory
  hip_try(hipMalloc(&d_src, size));
  hip_try(hipMalloc(&d_dst, size));

  // Allocate host memory
  h_src = (float*)malloc(size);
  h_dst = (float*)malloc(size);

  // Initialize source data
  for (int i = 0; i < n; i++) {
    h_src[i] = (float)i;
  }

  // Copy to device
  hip_try(hipMemcpy(d_src, h_src, size, hipMemcpyHostToDevice));
  hip_try(hipMemset(d_dst, 0, size));

  // Print GPU info
  hipDeviceProp_t prop;
  hip_try(hipGetDeviceProperties(&prop, 0));
  printf("GPU: %s\n", prop.name);
  printf("Blocks: %d, Threads per block: %d, Total threads: %d\n", num_blocks,
         num_threads, n);
  printf("Data size: %.2f MB\n\n", size / (1024.0 * 1024.0));

  // Launch kernel - coalesced access
  memcpy_coalesced<<<num_blocks, num_threads>>>(d_dst, d_src, n);
  hip_try(hipGetLastError());
  hip_try(hipDeviceSynchronize());

  // Copy result back
  hip_try(hipMemcpy(h_dst, d_dst, size, hipMemcpyDeviceToHost));

  // Verify results
  bool correct = true;
  for (int i = 0; i < n; i++) {
    if (h_dst[i] != h_src[i]) {
      printf("Error: dst[%d] = %.0f, expected %.0f\n", i, h_dst[i], h_src[i]);
      correct = false;
      break;
    }
  }

  if (correct) {
    printf("Result: ✓ Coalesced memcpy correct\n");
  } else {
    printf("Result: ✗ Incorrect\n");
  }

  // Cleanup
  hip_try(hipFree(d_src));
  hip_try(hipFree(d_dst));
  free(h_src);
  free(h_dst);

  return 0;
}
