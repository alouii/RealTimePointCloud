#include "../include/PointCloudProcessor.h"
#include <cuda_runtime.h>
#include <iostream>

pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudProcessor::depthToPointCloud(
        const std::vector<float>& depthData, int width, int height) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->width = width;
    cloud->height = height;
    cloud->is_dense = false;
    cloud->points.resize(width * height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float z = depthData[y * width + x];
            cloud->at(x, y).x = x * z;
            cloud->at(x, y).y = y * z;
            cloud->at(x, y).z = z;
        }
    }
    return cloud;
}

// CUDA placeholder
__global__ void transformKernel(float3* points, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        points[idx].z += 0.01f; // simple z-offset
    }
}

void PointCloudProcessor::processPointCloudCUDA(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    int N = cloud->points.size();
    float3* d_points;
    cudaMalloc(&d_points, N * sizeof(float3));

    // Copy to device
    std::vector<float3> h_points(N);
    for (int i = 0; i < N; ++i) {
        h_points[i].x = cloud->points[i].x;
        h_points[i].y = cloud->points[i].y;
        h_points[i].z = cloud->points[i].z;
    }
    cudaMemcpy(d_points, h_points.data(), N * sizeof(float3), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    transformKernel<<<blocks, threads>>>(d_points, N);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(h_points.data(), d_points, N * sizeof(float3), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        cloud->points[i].x = h_points[i].x;
        cloud->points[i].y = h_points[i].y;
        cloud->points[i].z = h_points[i].z;
    }

    cudaFree(d_points);
}
