#pragma once

template <typename T>
__device__ T fma_helper(T a, T b) {
    return a + b;
}
