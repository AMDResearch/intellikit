#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <string>

#define TILE_DIM 32

// Matrix transpose WITH bank conflicts (no padding)
__global__ void transpose_conflict(float* out,
                                   const float* in,
                                   int width,
                                   int height) {
  __shared__ float tile[TILE_DIM]
                       [TILE_DIM];  // No padding - causes bank conflicts

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  // Load input tile with coalesced reads
  if (x < width && y < height) {
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
  }

  __syncthreads();

  // Transpose coordinates
  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  // Store output tile with bank conflicts on shared memory read
  if (x < height && y < width) {
    out[y * height + x] =
        tile[threadIdx.x][threadIdx.y];  // Bank conflicts here!
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
  // Default parameters - square matrix
  int width = 1024;
  int height = 1024;

  if (argc > 1)
    width = height = atoi(argv[1]);

  size_t size = width * height * sizeof(float);

  float *d_in, *d_out;
  float *h_in, *h_out;

  // Allocate device memory
  hip_try(hipMalloc(&d_in, size));
  hip_try(hipMalloc(&d_out, size));

  // Allocate host memory
  h_in = (float*)malloc(size);
  h_out = (float*)malloc(size);

  // Initialize input matrix
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      h_in[i * width + j] = i * width + j;
    }
  }

  // Copy to device
  hip_try(hipMemcpy(d_in, h_in, size, hipMemcpyHostToDevice));
  hip_try(hipMemset(d_out, 0, size));

  // Print GPU info
  hipDeviceProp_t prop;
  hip_try(hipGetDeviceProperties(&prop, 0));
  printf("GPU: %s\n", prop.name);
  printf("Matrix: %dx%d\n", width, height);
  printf("Data size: %.2f MB\n\n", size / (1024.0 * 1024.0));

  // Launch kernel
  dim3 blocks((width + TILE_DIM - 1) / TILE_DIM,
              (height + TILE_DIM - 1) / TILE_DIM);
  dim3 threads(TILE_DIM, TILE_DIM);

  transpose_conflict<<<blocks, threads>>>(d_out, d_in, width, height);
  hip_try(hipGetLastError());
  hip_try(hipDeviceSynchronize());

  // Copy result back
  hip_try(hipMemcpy(h_out, d_out, size, hipMemcpyDeviceToHost));

  // Verify results (spot check a few elements)
  bool correct = true;
  for (int i = 0; i < height && i < 10; i++) {
    for (int j = 0; j < width && j < 10; j++) {
      float expected = j * height + i;  // Transposed
      if (h_out[i * height + j] != expected) {
        printf("Error: out[%d][%d] = %.0f, expected %.0f\n", i, j,
               h_out[i * height + j], expected);
        correct = false;
        break;
      }
    }
    if (!correct)
      break;
  }

  if (correct) {
    printf("Result: ✓ Transpose correct (WITH bank conflicts)\n");
  } else {
    printf("Result: ✗ Incorrect\n");
  }

  // Cleanup
  hip_try(hipFree(d_in));
  hip_try(hipFree(d_out));
  free(h_in);
  free(h_out);

  return 0;
}
