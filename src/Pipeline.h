#ifndef PIPELINE_H
#define PIPELINE_H

#include "point_cloud_types.h"
#include "memory_pool.h"
#include "preprocessing.cuh"
#include "inference.cuh"
#include <memory>
#include <chrono>

// Main pipeline class
class PointCloudPipeline {
public:
    explicit PointCloudPipeline(const PipelineConfig& config);
    ~PointCloudPipeline();
    
    // Process single frame
    std::vector<Detection> process_frame(const std::vector<Point>& points);
    
    // Batch processing
    std::vector<std::vector<Detection>> process_batch(
        const std::vector<std::vector<Point>>& point_clouds
    );
    
    // Async processing
    void process_frame_async(const std::vector<Point>& points);
    std::vector<Detection> get_results();
    bool results_ready() const;
    
    // Configuration
    void update_config(const PipelineConfig& config);
    const PipelineConfig& get_config() const { return config_; }
    
    // Performance monitoring
    struct PipelineStats {
        double preprocessing_time_ms;
        double inference_time_ms;
        double postprocessing_time_ms;
        double total_time_ms;
        int num_input_points;
        int num_detections;
        double throughput_fps;
    };
    
    PipelineStats get_stats() const { return stats_; }
    void reset_stats();
    
    // Warm-up for stable performance
    void warmup(int num_iterations = 10);
    
private:
    PipelineConfig config_;
    PipelineStats stats_;
    
    // CUDA streams for async operations
    cudaStream_t preprocessing_stream_;
    cudaStream_t inference_stream_;
    cudaStream_t postprocessing_stream_;
    
    // Memory buffers
    DeviceBuffer<Point> d_input_points_;
    DeviceBuffer<Point> d_filtered_points_;
    DeviceBuffer<Voxel> d_voxels_;
    DeviceBuffer<float> d_features_;
    DeviceBuffer<Detection> d_detections_;
    DeviceBuffer<int> d_indices_;
    
    // Processing components
    std::unique_ptr<inference::FeatureExtractor> feature_extractor_;
    std::unique_ptr<inference::DetectionNetwork> detection_network_;
    
    // Internal methods
    void preprocess(const Point* input, int num_points, 
                   Point** output, int* num_output);
    void run_inference(const Point* points, int num_points,
                      Detection* detections, int* num_detections);
    void postprocess(Detection* detections, int* num_detections);
    
    void initialize_streams();
    void cleanup_streams();
    
    bool async_processing_;
    bool results_ready_;
    std::vector<Detection> async_results_;
};

// Utility class for timing
class Timer {
public:
    Timer() { reset(); }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
    
    double elapsed_seconds() const {
        return elapsed_ms() / 1000.0;
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// CUDA timing utilities
class CudaTimer {
public:
    CudaTimer();
    ~CudaTimer();
    
    void start(cudaStream_t stream = 0);
    void stop(cudaStream_t stream = 0);
    float elapsed_ms();
    
private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
};

#endif // PIPELINE_H
