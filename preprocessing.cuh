#ifndef PREPROCESSING_CUH
#define PREPROCESSING_CUH

#include "point_cloud_types.h"
#include "memory_pool.h"

// Preprocessing operations
namespace preprocessing {

// Range filtering - remove points outside specified range
void filter_by_range(
    const Point* input_points,
    Point* output_points,
    int* output_indices,
    int num_points,
    float max_range,
    int* num_valid_points,
    cudaStream_t stream = 0
);

// Voxelization - downsample point cloud using voxel grid
void voxelize(
    const Point* input_points,
    Voxel* output_voxels,
    int* voxel_indices,
    int num_points,
    float voxel_size,
    const BoundingBox& bbox,
    int* num_voxels,
    cudaStream_t stream = 0
);

// Dynamic voxelization with variable voxel sizes
void dynamic_voxelize(
    const Point* input_points,
    Voxel* output_voxels,
    int* voxel_point_indices,
    int num_points,
    float* voxel_sizes,
    int num_voxel_levels,
    int* num_voxels,
    cudaStream_t stream = 0
);

// Compute point cloud statistics
void compute_statistics(
    const Point* points,
    int num_points,
    BoundingBox* bbox,
    float* mean_intensity,
    cudaStream_t stream = 0
);

// Ground plane removal using RANSAC
void remove_ground_plane(
    const Point* input_points,
    Point* output_points,
    int num_points,
    float ground_threshold,
    int* num_non_ground_points,
    cudaStream_t stream = 0
);

// Outlier removal using statistical method
void remove_outliers(
    const Point* input_points,
    Point* output_points,
    int num_points,
    int k_neighbors,
    float std_multiplier,
    int* num_inliers,
    cudaStream_t stream = 0
);

// Normalization
void normalize_points(
    Point* points,
    int num_points,
    const BoundingBox& bbox,
    cudaStream_t stream = 0
);

// Point shuffling for batch processing
void shuffle_points(
    Point* points,
    int num_points,
    unsigned int seed,
    cudaStream_t stream = 0
);

} // namespace preprocessing

#endif // PREPROCESSING_CUH
