#include <hip/hip_runtime.h>
#include <cstdio>

void scale_both(float* a, int nx, int ny);

int main() {
    const int nx = 1024, ny = 1024;
    float* a;
    if (hipMalloc(&a, sizeof(float) * nx * ny) != hipSuccess) return 1;
    if (hipMemset(a, 0, sizeof(float) * nx * ny) != hipSuccess) return 1;
    scale_both(a, nx, ny);
    if (hipDeviceSynchronize() != hipSuccess) return 1;
    std::printf("done\n");
    return 0;
}
