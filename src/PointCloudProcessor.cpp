#include "PointCloudProcessor.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// Declare external CUDA function
extern void runTransformKernel(float3* d_points, int N);

void PointCloudProcessor::processPointCloudCUDA(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    int N = cloud->points.size();
    float3* d_points;
    cudaMalloc(&d_points, N * sizeof(float3));

    std::vector<float3> h_points(N);
    for (int i = 0; i < N; ++i) {
        h_points[i].x = cloud->points[i].x;
        h_points[i].y = cloud->points[i].y;
        h_points[i].z = cloud->points[i].z;
    }

    cudaMemcpy(d_points, h_points.data(), N * sizeof(float3), cudaMemcpyHostToDevice);

    // Call the CUDA kernel
    runTransformKernel(d_points, N);

    cudaMemcpy(h_points.data(), d_points, N * sizeof(float3), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        cloud->points[i].x = h_points[i].x;
        cloud->points[i].y = h_points[i].y;
        cloud->points[i].z = h_points[i].z;
    }

    cudaFree(d_points);
}
