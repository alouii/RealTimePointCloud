#include <cuda_runtime.h>
#include <iostream>

__global__ void dummyKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] += 1.0f;
}

void runDummyCUDA(int N) {
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemset(d_data, 0, N * sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    dummyKernel<<<blocks, threads>>>(d_data, N);
    cudaDeviceSynchronize();
    cudaFree(d_data);
}
