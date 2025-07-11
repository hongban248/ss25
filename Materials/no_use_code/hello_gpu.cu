#include <iostream>
#include <cuda_runtime.h>

// CUDA Kernel function to print "Hello from GPU!"
__global__ void helloFromGPU() {
    printf("Hello from GPU! ThreadIdx: %d\n", threadIdx.x);
}

int main() {
    // Call the kernel function
    helloFromGPU<<<2, 10>>>();

    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();

    return 0;
}
