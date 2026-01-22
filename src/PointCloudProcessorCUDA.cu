#include <cuda_runtime.h>
#include <pcl/point_types.h>

struct float3 { float x, y, z; }; // temporary struct

__global__ void transformKernel(float3* points, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) points[idx].z += 0.01f;
}

extern "C" void runTransformKernel(float3* d_points, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    transformKernel<<<blocks, threads>>>(d_points, N);
    cudaDeviceSynchronize();
}
