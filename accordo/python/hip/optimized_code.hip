/****************************************************************************
MIT License

Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
****************************************************************************/


#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <cstdint>
#include <iostream>

__global__ void reduction_kernel(double* input, double* result, std::size_t count) {
  extern __shared__ double shared_data[];
  const auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  const auto local_id = threadIdx.x;

  double sum = 0.0;
  if (thread_id < count) {
    sum = input[thread_id] / (thread_id + 1);
  }
  shared_data[local_id] = sum;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (local_id < stride) {
      shared_data[local_id] += shared_data[local_id + stride];
    }
    __syncthreads();
  }

  if (local_id == 0) {
    atomicAdd(result, shared_data[0]);
  }
}

int main() {
  const std::size_t count{10000000};
  thrust::device_vector<double> input(count, 1);
  thrust::device_vector<double> result(1, 0);
  thrust::sequence(input.begin(), input.end(), 1);

  const std::size_t block_size{512};
  const std::size_t num_blocks{(count + block_size - 1) / block_size};

  reduction_kernel<<<num_blocks, block_size, block_size * sizeof(double)>>>(
      input.data().get(), result.data().get(), count);

  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess || result[0] != count) {
    std::cout << "Kernel failed.";
    return -1;
  } else {
    std::cout << "Success!";
  }
  std::cout << std::endl;
  return static_cast<int>(result[0]);
}