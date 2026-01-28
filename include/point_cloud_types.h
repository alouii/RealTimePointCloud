#ifndef POINT_CLOUD_TYPES_H
#define POINT_CLOUD_TYPES_H

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>

// Point structure with CUDA compatibility
struct Point {
    float x, y, z;
    float intensity;
    
    __host__ __device__ Point() : x(0), y(0), z(0), intensity(0) {}
    __host__ __device__ Point(float x_, float y_, float z_, float i_ = 0) 
        : x(x_), y(y_), z(z_), intensity(i_) {}
};

// Voxel structure for spatial hashing
struct Voxel {
    int x, y, z;
    int point_count;
    float center_x, center_y, center_z;
    
    __host__ __device__ Voxel() : x(0), y(0), z(0), point_count(0),
                                   center_x(0), center_y(0), center_z(0) {}
};

// Bounding box
struct BoundingBox {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    
    __host__ __device__ BoundingBox() 
        : min_x(INFINITY), min_y(INFINITY), min_z(INFINITY),
          max_x(-INFINITY), max_y(-INFINITY), max_z(-INFINITY) {}
};

// Feature vector for inference
struct FeatureVector {
    static constexpr int FEATURE_DIM = 64;
    float features[FEATURE_DIM];
    
    __host__ __device__ FeatureVector() {
        for (int i = 0; i < FEATURE_DIM; ++i) features[i] = 0.0f;
    }
};

// Detection result
struct Detection {
    BoundingBox bbox;
    float confidence;
    int class_id;
    
    __host__ __device__ Detection() : confidence(0.0f), class_id(-1) {}
};

// Pipeline configuration
struct PipelineConfig {
    // Input parameters
    int max_points;
    float voxel_size;
    float max_range;
    
    // Processing parameters
    int batch_size;
    int num_threads;
    bool use_dynamic_voxelization;
    
    // Model parameters
    std::string model_path;
    int num_classes;
    float confidence_threshold;
    float nms_threshold;
    
    // Memory parameters
    size_t memory_pool_size;
    bool use_pinned_memory;
    
    PipelineConfig();
    void load_from_file(const std::string& config_file);
};

// Point cloud data container
class PointCloudData {
public:
    // Host data
    std::vector<Point> h_points;
    
    // Device data
    Point* d_points;
    int* d_indices;
    float* d_features;
    Detection* d_detections;
    
    // Metadata
    int num_points;
    int capacity;
    BoundingBox bbox;
    
    PointCloudData(int capacity = 100000);
    ~PointCloudData();
    
    void allocate_device_memory();
    void free_device_memory();
    void copy_to_device();
    void copy_from_device();
    void resize(int new_capacity);
    
private:
    bool device_allocated;
};

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#endif // POINT_CLOUD_TYPES_H
