#include "pipeline.h"
#include "preprocessing.cuh"
#include "inference.cuh"
#include <iostream>
#include <cassert>
#include <cmath>

#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAILED: " << message << std::endl; \
            return false; \
        } \
    } while(0)

#define RUN_TEST(test_func) \
    do { \
        std::cout << "Running " << #test_func << "... "; \
        if (test_func()) { \
            std::cout << "PASSED" << std::endl; \
            passed++; \
        } else { \
            std::cout << "FAILED" << std::endl; \
            failed++; \
        } \
        total++; \
    } while(0)

bool test_point_cloud_data() {
    PointCloudData data(1000);
    
    // Add points
    for (int i = 0; i < 100; ++i) {
        data.h_points.push_back(Point(i * 0.1f, i * 0.2f, i * 0.3f, 0.5f));
    }
    
    TEST_ASSERT(data.h_points.size() == 100, "Point count mismatch");
    
    // Test device allocation
    data.allocate_device_memory();
    data.copy_to_device();
    data.copy_from_device();
    
    TEST_ASSERT(data.h_points.size() == 100, "Points lost in transfer");
    
    return true;
}

bool test_memory_pool() {
    auto& pool = GPUMemoryPool::get_instance();
    
    size_t initial_allocated = pool.get_total_allocated();
    
    // Allocate memory
    void* ptr1 = pool.allocate(1024);
    TEST_ASSERT(ptr1 != nullptr, "Allocation failed");
    TEST_ASSERT(pool.get_total_in_use() > initial_allocated, "In-use not updated");
    
    // Deallocate
    pool.deallocate(ptr1, 1024);
    
    // Reuse memory
    void* ptr2 = pool.allocate(512);
    TEST_ASSERT(ptr2 != nullptr, "Reallocation failed");
    
    pool.deallocate(ptr2, 512);
    
    return true;
}

bool test_range_filtering() {
    const int num_points = 1000;
    std::vector<Point> h_input(num_points);
    
    // Create points at various distances
    for (int i = 0; i < num_points; ++i) {
        float dist = i * 0.1f;
        h_input[i] = Point(dist, 0, 0, 0.5f);
    }
    
    // Allocate device memory
    Point *d_input, *d_output;
    int *d_indices, *d_num_valid;
    CUDA_CHECK(cudaMalloc(&d_input, num_points * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_output, num_points * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_indices, num_points * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_valid, sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), 
                          num_points * sizeof(Point),
                          cudaMemcpyHostToDevice));
    
    // Filter with range 50.0
    int h_num_valid;
    preprocessing::filter_by_range(d_input, d_output, d_indices,
                                  num_points, 50.0f, &h_num_valid);
    
    TEST_ASSERT(h_num_valid > 0 && h_num_valid < num_points,
                "Filtering produced unexpected result");
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_num_valid));
    
    return true;
}

bool test_bounding_box_computation() {
    const int num_points = 100;
    std::vector<Point> h_points(num_points);
    
    float min_val = -10.0f;
    float max_val = 10.0f;
    
    for (int i = 0; i < num_points; ++i) {
        h_points[i] = Point(
            min_val + (max_val - min_val) * float(rand()) / RAND_MAX,
            min_val + (max_val - min_val) * float(rand()) / RAND_MAX,
            min_val + (max_val - min_val) * float(rand()) / RAND_MAX,
            0.5f
        );
    }
    
    Point* d_points;
    CUDA_CHECK(cudaMalloc(&d_points, num_points * sizeof(Point)));
    CUDA_CHECK(cudaMemcpy(d_points, h_points.data(),
                          num_points * sizeof(Point),
                          cudaMemcpyHostToDevice));
    
    BoundingBox bbox;
    float mean_intensity;
    preprocessing::compute_statistics(d_points, num_points, &bbox, &mean_intensity);
    
    TEST_ASSERT(bbox.min_x >= min_val && bbox.max_x <= max_val,
                "BBox X range incorrect");
    TEST_ASSERT(bbox.min_y >= min_val && bbox.max_y <= max_val,
                "BBox Y range incorrect");
    TEST_ASSERT(bbox.min_z >= min_val && bbox.max_z <= max_val,
                "BBox Z range incorrect");
    
    CUDA_CHECK(cudaFree(d_points));
    
    return true;
}

bool test_voxelization() {
    const int num_points = 1000;
    std::vector<Point> h_points(num_points);
    
    for (int i = 0; i < num_points; ++i) {
        h_points[i] = Point(
            float(rand()) / RAND_MAX * 10.0f,
            float(rand()) / RAND_MAX * 10.0f,
            float(rand()) / RAND_MAX * 10.0f,
            0.5f
        );
    }
    
    Point* d_points;
    Voxel* d_voxels;
    int* d_indices;
    
    CUDA_CHECK(cudaMalloc(&d_points, num_points * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_voxels, num_points * sizeof(Voxel)));
    CUDA_CHECK(cudaMalloc(&d_indices, num_points * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_points, h_points.data(),
                          num_points * sizeof(Point),
                          cudaMemcpyHostToDevice));
    
    BoundingBox bbox;
    bbox.min_x = bbox.min_y = bbox.min_z = 0.0f;
    bbox.max_x = bbox.max_y = bbox.max_z = 10.0f;
    
    int num_voxels;
    preprocessing::voxelize(d_points, d_voxels, d_indices,
                          num_points, 0.5f, bbox, &num_voxels);
    
    TEST_ASSERT(num_voxels > 0 && num_voxels <= num_points,
                "Voxelization produced unexpected result");
    
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_voxels));
    CUDA_CHECK(cudaFree(d_indices));
    
    return true;
}

bool test_feature_extraction() {
    const int num_points = 100;
    std::vector<Point> h_points(num_points);
    
    for (int i = 0; i < num_points; ++i) {
        h_points[i] = Point(i * 0.1f, i * 0.1f, i * 0.1f, 0.5f);
    }
    
    Point* d_points;
    float* d_features;
    
    CUDA_CHECK(cudaMalloc(&d_points, num_points * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_features, num_points * 64 * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_points, h_points.data(),
                          num_points * sizeof(Point),
                          cudaMemcpyHostToDevice));
    
    inference::FeatureExtractor extractor(3, 64);
    extractor.extract_features(d_points, num_points, d_features);
    
    // Verify features are computed
    std::vector<float> h_features(num_points * 64);
    CUDA_CHECK(cudaMemcpy(h_features.data(), d_features,
                          num_points * 64 * sizeof(float),
                          cudaMemcpyDeviceToHost));
    
    bool has_nonzero = false;
    for (float f : h_features) {
        if (std::abs(f) > 1e-6) {
            has_nonzero = true;
            break;
        }
    }
    
    TEST_ASSERT(has_nonzero, "Features are all zero");
    
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_features));
    
    return true;
}

bool test_pipeline_end_to_end() {
    PipelineConfig config;
    config.max_points = 10000;
    config.voxel_size = 0.1f;
    config.max_range = 50.0f;
    
    PointCloudPipeline pipeline(config);
    
    // Generate test data
    std::vector<Point> points(1000);
    for (int i = 0; i < 1000; ++i) {
        points[i] = Point(
            float(rand()) / RAND_MAX * 20.0f - 10.0f,
            float(rand()) / RAND_MAX * 20.0f - 10.0f,
            float(rand()) / RAND_MAX * 20.0f - 10.0f,
            0.5f
        );
    }
    
    // Process
    auto detections = pipeline.process_frame(points);
    auto stats = pipeline.get_stats();
    
    TEST_ASSERT(stats.num_input_points == 1000, "Input point count mismatch");
    TEST_ASSERT(stats.total_time_ms > 0, "Processing time is zero");
    TEST_ASSERT(stats.throughput_fps > 0, "Throughput is zero");
    
    return true;
}

bool test_batch_processing() {
    PipelineConfig config;
    PointCloudPipeline pipeline(config);
    
    std::vector<std::vector<Point>> batch;
    for (int b = 0; b < 4; ++b) {
        std::vector<Point> points(500);
        for (int i = 0; i < 500; ++i) {
            points[i] = Point(
                float(rand()) / RAND_MAX * 20.0f,
                float(rand()) / RAND_MAX * 20.0f,
                float(rand()) / RAND_MAX * 20.0f,
                0.5f
            );
        }
        batch.push_back(points);
    }
    
    auto results = pipeline.process_batch(batch);
    
    TEST_ASSERT(results.size() == 4, "Batch size mismatch");
    
    return true;
}

bool test_config_loading() {
    // Create temporary config file
    std::ofstream config_file("test_config.txt");
    config_file << "max_points=50000\n";
    config_file << "voxel_size=0.2\n";
    config_file << "max_range=100.0\n";
    config_file << "num_classes=5\n";
    config_file.close();
    
    PipelineConfig config;
    config.load_from_file("test_config.txt");
    
    TEST_ASSERT(config.max_points == 50000, "max_points not loaded");
    TEST_ASSERT(std::abs(config.voxel_size - 0.2f) < 1e-6, "voxel_size not loaded");
    TEST_ASSERT(std::abs(config.max_range - 100.0f) < 1e-6, "max_range not loaded");
    TEST_ASSERT(config.num_classes == 5, "num_classes not loaded");
    
    // Cleanup
    std::remove("test_config.txt");
    
    return true;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Point Cloud Pipeline Unit Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    int total = 0;
    int passed = 0;
    int failed = 0;
    
    RUN_TEST(test_point_cloud_data);
    RUN_TEST(test_memory_pool);
    RUN_TEST(test_range_filtering);
    RUN_TEST(test_bounding_box_computation);
    RUN_TEST(test_voxelization);
    RUN_TEST(test_feature_extraction);
    RUN_TEST(test_pipeline_end_to_end);
    RUN_TEST(test_batch_processing);
    RUN_TEST(test_config_loading);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total:  " << total << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "========================================" << std::endl;
    
    return (failed == 0) ? 0 : 1;
}
