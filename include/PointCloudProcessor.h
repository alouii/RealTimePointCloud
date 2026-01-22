#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

class PointCloudProcessor {
public:
    PointCloudProcessor() = default;

    // Convert depth frame to point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr depthToPointCloud(
        const std::vector<float>& depthData, int width, int height);

    // CUDA placeholder: apply a transformation to cloud points
    void processPointCloudCUDA(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
};
