#include "point_cloud_types.h"
#include <fstream>
#include <iostream>
#include <algorithm>

// PipelineConfig implementation
PipelineConfig::PipelineConfig() 
    : max_points(100000)
    , voxel_size(0.1f)
    , max_range(50.0f)
    , batch_size(1)
    , num_threads(4)
    , use_dynamic_voxelization(false)
    , num_classes(10)
    , confidence_threshold(0.5f)
    , nms_threshold(0.5f)
    , memory_pool_size(1024 * 1024 * 1024) // 1GB
    , use_pinned_memory(true)
{
}

void PipelineConfig::load_from_file(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << config_file << std::endl;
        return;
    }
    
    // Simple key-value parser
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        size_t pos = line.find('=');
        if (pos == std::string::npos) continue;
        
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        
        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        
        // Parse values
        if (key == "max_points") max_points = std::stoi(value);
        else if (key == "voxel_size") voxel_size = std::stof(value);
        else if (key == "max_range") max_range = std::stof(value);
        else if (key == "batch_size") batch_size = std::stoi(value);
        else if (key == "num_threads") num_threads = std::stoi(value);
        else if (key == "model_path") model_path = value;
        else if (key == "num_classes") num_classes = std::stoi(value);
        else if (key == "confidence_threshold") confidence_threshold = std::stof(value);
        else if (key == "nms_threshold") nms_threshold = std::stof(value);
    }
    
    file.close();
}

// PointCloudData implementation
PointCloudData::PointCloudData(int cap)
    : d_points(nullptr)
    , d_indices(nullptr)
    , d_features(nullptr)
    , d_detections(nullptr)
    , num_points(0)
    , capacity(cap)
    , device_allocated(false)
{
    h_points.reserve(capacity);
}

PointCloudData::~PointCloudData() {
    free_device_memory();
}

void PointCloudData::allocate_device_memory() {
    if (device_allocated) return;
    
    CUDA_CHECK(cudaMalloc(&d_points, capacity * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_indices, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_features, capacity * FeatureVector::FEATURE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_detections, 1000 * sizeof(Detection))); // Max 1000 detections
    
    device_allocated = true;
}

void PointCloudData::free_device_memory() {
    if (!device_allocated) return;
    
    if (d_points) CUDA_CHECK(cudaFree(d_points));
    if (d_indices) CUDA_CHECK(cudaFree(d_indices));
    if (d_features) CUDA_CHECK(cudaFree(d_features));
    if (d_detections) CUDA_CHECK(cudaFree(d_detections));
    
    d_points = nullptr;
    d_indices = nullptr;
    d_features = nullptr;
    d_detections = nullptr;
    
    device_allocated = false;
}

void PointCloudData::copy_to_device() {
    if (!device_allocated) allocate_device_memory();
    
    if (h_points.size() > capacity) {
        resize(h_points.size());
    }
    
    num_points = h_points.size();
    CUDA_CHECK(cudaMemcpy(d_points, h_points.data(), 
                          num_points * sizeof(Point),
                          cudaMemcpyHostToDevice));
}

void PointCloudData::copy_from_device() {
    h_points.resize(num_points);
    CUDA_CHECK(cudaMemcpy(h_points.data(), d_points,
                          num_points * sizeof(Point),
                          cudaMemcpyDeviceToHost));
}

void PointCloudData::resize(int new_capacity) {
    if (new_capacity <= capacity) return;
    
    capacity = new_capacity;
    h_points.reserve(capacity);
    
    if (device_allocated) {
        Point* new_d_points;
        int* new_d_indices;
        float* new_d_features;
        
        CUDA_CHECK(cudaMalloc(&new_d_points, capacity * sizeof(Point)));
        CUDA_CHECK(cudaMalloc(&new_d_indices, capacity * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&new_d_features, 
                              capacity * FeatureVector::FEATURE_DIM * sizeof(float)));
        
        // Copy old data
        if (num_points > 0) {
            CUDA_CHECK(cudaMemcpy(new_d_points, d_points,
                                  num_points * sizeof(Point),
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(new_d_indices, d_indices,
                                  num_points * sizeof(int),
                                  cudaMemcpyDeviceToDevice));
        }
        
        // Free old memory
        CUDA_CHECK(cudaFree(d_points));
        CUDA_CHECK(cudaFree(d_indices));
        CUDA_CHECK(cudaFree(d_features));
        
        // Update pointers
        d_points = new_d_points;
        d_indices = new_d_indices;
        d_features = new_d_features;
    }
}
