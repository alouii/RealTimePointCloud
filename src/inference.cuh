#ifndef INFERENCE_CUH
#define INFERENCE_CUH

#include "point_cloud_types.h"
#include "memory_pool.h"
#include <vector>

namespace inference {

// Feature extraction using PointNet-style architecture
class FeatureExtractor {
public:
    FeatureExtractor(int input_dim, int output_dim);
    ~FeatureExtractor();
    
    void extract_features(
        const Point* points,
        int num_points,
        float* features,
        cudaStream_t stream = 0
    );
    
    void load_weights(const std::string& weight_file);
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// 3D object detection network
class DetectionNetwork {
public:
    DetectionNetwork(const PipelineConfig& config);
    ~DetectionNetwork();
    
    void detect(
        const Point* points,
        int num_points,
        Detection* detections,
        int* num_detections,
        cudaStream_t stream = 0
    );
    
    void load_model(const std::string& model_path);
    void set_confidence_threshold(float threshold);
    void set_nms_threshold(float threshold);
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    PipelineConfig config_;
};

// Post-processing operations
namespace postprocess {

// Non-Maximum Suppression for 3D bounding boxes
void non_max_suppression_3d(
    Detection* detections,
    int num_detections,
    float iou_threshold,
    int* keep_indices,
    int* num_keep,
    cudaStream_t stream = 0
);

// Compute IoU between 3D boxes
void compute_iou_3d(
    const BoundingBox* boxes1,
    const BoundingBox* boxes2,
    int num_boxes1,
    int num_boxes2,
    float* iou_matrix,
    cudaStream_t stream = 0
);

// Filter detections by confidence
void filter_by_confidence(
    const Detection* input_detections,
    Detection* output_detections,
    int num_detections,
    float threshold,
    int* num_filtered,
    cudaStream_t stream = 0
);

// Bounding box refinement
void refine_bounding_boxes(
    Detection* detections,
    const Point* points,
    const int* point_assignments,
    int num_detections,
    int num_points,
    cudaStream_t stream = 0
);

} // namespace postprocess

// Warp-level optimized operations
namespace warp_ops {

// Warp-level reduction for feature aggregation
template<typename T>
__device__ T warp_reduce_sum(T val);

template<typename T>
__device__ T warp_reduce_max(T val);

template<typename T>
__device__ T warp_reduce_min(T val);

// Warp-level prefix sum
template<typename T>
__device__ T warp_scan(T val);

// Efficient point sampling using warp voting
__device__ bool warp_vote_any(bool predicate);
__device__ bool warp_vote_all(bool predicate);
__device__ unsigned int warp_ballot(bool predicate);

} // namespace warp_ops

// Matrix operations optimized for point clouds
namespace matrix_ops {

void matrix_multiply(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    cudaStream_t stream = 0
);

void batch_matrix_multiply(
    const float* A,
    const float* B,
    float* C,
    int batch_size,
    int M, int N, int K,
    cudaStream_t stream = 0
);

void pointwise_convolution(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int num_points,
    int in_channels,
    int out_channels,
    cudaStream_t stream = 0
);

} // namespace matrix_ops

} // namespace inference

#endif // INFERENCE_CUH
