#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <string>

// High contention - all threads add to the same slot
__global__ void atomic_same_slot(unsigned int* data) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // All threads atomically add to data[0] - maximum contention
  atomicAdd(&data[0], 1);
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
  int num_blocks = 32;
  int num_threads = 256;

  // Parse command line arguments
  if (argc > 1)
    num_blocks = atoi(argv[1]);
  if (argc > 2)
    num_threads = atoi(argv[2]);

  int total_threads = num_threads * num_blocks;
  size_t size = sizeof(unsigned int);

  unsigned int* d_data;
  unsigned int* h_data;

  // Allocate device memory
  hip_try(hipMalloc(&d_data, size));

  // Allocate host memory
  h_data = (unsigned int*)malloc(size);

  // Initialize to zero
  hip_try(hipMemset(d_data, 0, size));

  // Print GPU info
  hipDeviceProp_t prop;
  hip_try(hipGetDeviceProperties(&prop, 0));
  printf("GPU: %s\n", prop.name);
  printf("Blocks: %d, Threads per block: %d, Total threads: %d\n\n", num_blocks,
         num_threads, total_threads);

  // Launch kernel - all threads add to same slot (high contention)
  atomic_same_slot<<<num_blocks, num_threads>>>(d_data);
  hip_try(hipGetLastError());
  hip_try(hipDeviceSynchronize());

  // Copy result back
  hip_try(hipMemcpy(h_data, d_data, size, hipMemcpyDeviceToHost));

  printf("Expected result: %u\n", total_threads);
  printf("Actual result: %u\n", h_data[0]);

  if (h_data[0] == total_threads) {
    printf("Result: ✓ Correct\n");
  } else {
    printf("Result: ✗ Incorrect\n");
  }

  // Cleanup
  hip_try(hipFree(d_data));
  free(h_data);

  return 0;
}
