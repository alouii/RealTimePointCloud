#include "pipeline.h"
#include <iostream>
#include <algorithm>

// CudaTimer implementation
CudaTimer::CudaTimer() {
    CUDA_CHECK(cudaEventCreate(&start_event_));
    CUDA_CHECK(cudaEventCreate(&stop_event_));
}

CudaTimer::~CudaTimer() {
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
}

void CudaTimer::start(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(start_event_, stream));
}

void CudaTimer::stop(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(stop_event_, stream));
}

float CudaTimer::elapsed_ms() {
    CUDA_CHECK(cudaEventSynchronize(stop_event_));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event_, stop_event_));
    return ms;
}

// PointCloudPipeline implementation
PointCloudPipeline::PointCloudPipeline(const PipelineConfig& config)
    : config_(config)
    , async_processing_(false)
    , results_ready_(false)
{
    // Initialize CUDA streams
    initialize_streams();
    
    // Create processing components
    feature_extractor_ = std::make_unique<inference::FeatureExtractor>(
        3, FeatureVector::FEATURE_DIM);
    
    detection_network_ = std::make_unique<inference::DetectionNetwork>(config_);
    
    // Pre-allocate device memory
    d_input_points_.resize(config_.max_points);
    d_filtered_points_.resize(config_.max_points);
    d_voxels_.resize(config_.max_points / 10); // Estimate
    d_features_.resize(config_.max_points * FeatureVector::FEATURE_DIM);
    d_detections_.resize(1000); // Max detections
    d_indices_.resize(config_.max_points);
    
    // Initialize stats
    reset_stats();
    
    std::cout << "Point Cloud Pipeline initialized:" << std::endl;
    std::cout << "  Max points: " << config_.max_points << std::endl;
    std::cout << "  Voxel size: " << config_.voxel_size << " m" << std::endl;
    std::cout << "  Max range: " << config_.max_range << " m" << std::endl;
}

PointCloudPipeline::~PointCloudPipeline() {
    cleanup_streams();
}

void PointCloudPipeline::initialize_streams() {
    CUDA_CHECK(cudaStreamCreate(&preprocessing_stream_));
    CUDA_CHECK(cudaStreamCreate(&inference_stream_));
    CUDA_CHECK(cudaStreamCreate(&postprocessing_stream_));
}

void PointCloudPipeline::cleanup_streams() {
    CUDA_CHECK(cudaStreamDestroy(preprocessing_stream_));
    CUDA_CHECK(cudaStreamDestroy(inference_stream_));
    CUDA_CHECK(cudaStreamDestroy(postprocessing_stream_));
}

std::vector<Detection> PointCloudPipeline::process_frame(
    const std::vector<Point>& points
) {
    Timer total_timer;
    CudaTimer gpu_timer;
    
    stats_.num_input_points = points.size();
    
    // Copy input to device
    d_input_points_.copy_from_host(points.data(), points.size());
    
    // Preprocessing
    Timer prep_timer;
    Point* filtered_points;
    int num_filtered;
    preprocess(d_input_points_.data(), points.size(), 
              &filtered_points, &num_filtered);
    stats_.preprocessing_time_ms = prep_timer.elapsed_ms();
    
    // Inference
    Timer inf_timer;
    Detection* detections = d_detections_.data();
    int num_detections;
    run_inference(filtered_points, num_filtered, detections, &num_detections);
    stats_.inference_time_ms = inf_timer.elapsed_ms();
    
    // Post-processing
    Timer post_timer;
    postprocess(detections, &num_detections);
    stats_.postprocessing_time_ms = post_timer.elapsed_ms();
    
    // Copy results back
    std::vector<Detection> results(num_detections);
    CUDA_CHECK(cudaMemcpy(results.data(), detections,
                          num_detections * sizeof(Detection),
                          cudaMemcpyDeviceToHost));
    
    stats_.num_detections = num_detections;
    stats_.total_time_ms = total_timer.elapsed_ms();
    stats_.throughput_fps = 1000.0 / stats_.total_time_ms;
    
    return results;
}

std::vector<std::vector<Detection>> PointCloudPipeline::process_batch(
    const std::vector<std::vector<Point>>& point_clouds
) {
    std::vector<std::vector<Detection>> all_results;
    all_results.reserve(point_clouds.size());
    
    for (const auto& points : point_clouds) {
        all_results.push_back(process_frame(points));
    }
    
    return all_results;
}

void PointCloudPipeline::process_frame_async(const std::vector<Point>& points) {
    async_processing_ = true;
    results_ready_ = false;
    
    // Launch async processing (simplified - would use proper async)
    async_results_ = process_frame(points);
    results_ready_ = true;
}

std::vector<Detection> PointCloudPipeline::get_results() {
    if (!results_ready_) {
        std::cerr << "Results not ready yet!" << std::endl;
        return {};
    }
    
    results_ready_ = false;
    return async_results_;
}

bool PointCloudPipeline::results_ready() const {
    return results_ready_;
}

void PointCloudPipeline::preprocess(
    const Point* input,
    int num_points,
    Point** output,
    int* num_output
) {
    // Range filtering
    int num_valid;
    preprocessing::filter_by_range(
        input,
        d_filtered_points_.data(),
        d_indices_.data(),
        num_points,
        config_.max_range,
        &num_valid,
        preprocessing_stream_
    );
    
    // Compute bounding box
    BoundingBox bbox;
    float mean_intensity;
    preprocessing::compute_statistics(
        d_filtered_points_.data(),
        num_valid,
        &bbox,
        &mean_intensity,
        preprocessing_stream_
    );
    
    // Voxelization (optional)
    if (config_.use_dynamic_voxelization) {
        int num_voxels;
        preprocessing::voxelize(
            d_filtered_points_.data(),
            d_voxels_.data(),
            d_indices_.data(),
            num_valid,
            config_.voxel_size,
            bbox,
            &num_voxels,
            preprocessing_stream_
        );
        *num_output = num_voxels;
    } else {
        *num_output = num_valid;
    }
    
    *output = d_filtered_points_.data();
    
    CUDA_CHECK(cudaStreamSynchronize(preprocessing_stream_));
}

void PointCloudPipeline::run_inference(
    const Point* points,
    int num_points,
    Detection* detections,
    int* num_detections
) {
    // Extract features
    feature_extractor_->extract_features(
        points,
        num_points,
        d_features_.data(),
        inference_stream_
    );
    
    // Run detection network
    detection_network_->detect(
        points,
        num_points,
        detections,
        num_detections,
        inference_stream_
    );
    
    CUDA_CHECK(cudaStreamSynchronize(inference_stream_));
}

void PointCloudPipeline::postprocess(Detection* detections, int* num_detections) {
    // Filter by confidence
    int num_confident;
    inference::postprocess::filter_by_confidence(
        detections,
        detections,
        *num_detections,
        config_.confidence_threshold,
        &num_confident,
        postprocessing_stream_
    );
    
    // Non-maximum suppression
    int* keep_indices;
    CUDA_CHECK(cudaMalloc(&keep_indices, num_confident * sizeof(int)));
    
    int num_keep;
    inference::postprocess::non_max_suppression_3d(
        detections,
        num_confident,
        config_.nms_threshold,
        keep_indices,
        &num_keep,
        postprocessing_stream_
    );
    
    *num_detections = num_keep;
    
    CUDA_CHECK(cudaFree(keep_indices));
    CUDA_CHECK(cudaStreamSynchronize(postprocessing_stream_));
}

void PointCloudPipeline::update_config(const PipelineConfig& config) {
    config_ = config;
    
    // Update network parameters
    detection_network_->set_confidence_threshold(config_.confidence_threshold);
    detection_network_->set_nms_threshold(config_.nms_threshold);
}

void PointCloudPipeline::reset_stats() {
    stats_ = PipelineStats{0, 0, 0, 0, 0, 0, 0};
}

void PointCloudPipeline::warmup(int num_iterations) {
    std::cout << "Warming up pipeline..." << std::endl;
    
    // Generate dummy data
    std::vector<Point> dummy_points(1000);
    for (int i = 0; i < 1000; ++i) {
        dummy_points[i] = Point(
            static_cast<float>(rand()) / RAND_MAX * 10.0f,
            static_cast<float>(rand()) / RAND_MAX * 10.0f,
            static_cast<float>(rand()) / RAND_MAX * 10.0f,
            static_cast<float>(rand()) / RAND_MAX
        );
    }
    
    for (int i = 0; i < num_iterations; ++i) {
        process_frame(dummy_points);
    }
    
    std::cout << "Warmup complete!" << std::endl;
}
