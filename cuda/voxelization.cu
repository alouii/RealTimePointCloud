#include "point_cloud_types.h"
#include <cuda_runtime.h>

// Additional voxelization utilities
__global__ void compute_voxel_grid_size_kernel(
    const Point* points,
    int num_points,
    float voxel_size,
    int* grid_size
) {
    // Compute required grid dimensions
    __shared__ float s_min[3];
    __shared__ float s_max[3];
    
    int tid = threadIdx.x;
    
    if (tid < 3) {
        s_min[tid] = INFINITY;
        s_max[tid] = -INFINITY;
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        Point p = points[idx];
        float coords[3] = {p.x, p.y, p.z};
        
        for (int i = 0; i < 3; ++i) {
            atomicMin(reinterpret_cast<int*>(&s_min[i]), 
                     __float_as_int(coords[i]));
            atomicMax(reinterpret_cast<int*>(&s_max[i]), 
                     __float_as_int(coords[i]));
        }
    }
    __syncthreads();
    
    if (tid == 0) {
        for (int i = 0; i < 3; ++i) {
            float range = s_max[i] - s_min[i];
            grid_size[i] = static_cast<int>(ceilf(range / voxel_size)) + 1;
        }
    }
}

// Optimized voxel hashing
__device__ unsigned int hash_voxel_coords(int x, int y, int z, int grid_size) {
    // Simple but effective hashing
    const unsigned int p1 = 73856093u;
    const unsigned int p2 = 19349663u;
    const unsigned int p3 = 83492791u;
    
    return ((x * p1) ^ (y * p2) ^ (z * p3)) % grid_size;
}

__global__ void adaptive_voxelize_kernel(
    const Point* points,
    Voxel* voxels,
    int num_points,
    float base_voxel_size,
    int max_points_per_voxel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    Point p = points[idx];
    
    // Adaptive voxel size based on point density
    float local_density = 1.0f; // Simplified
    float adaptive_size = base_voxel_size * (1.0f + local_density * 0.5f);
    
    int vx = static_cast<int>(floorf(p.x / adaptive_size));
    int vy = static_cast<int>(floorf(p.y / adaptive_size));
    int vz = static_cast<int>(floorf(p.z / adaptive_size));
    
    unsigned int hash = hash_voxel_coords(vx, vy, vz, num_points);
    
    // Store voxel assignment (simplified)
}
