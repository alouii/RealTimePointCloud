#include "PointCloudProcessor.h"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudProcessor::depthToPointCloud(
    const std::vector<float>& depth, int width, int height)
{
    auto cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cloud->width = width;
    cloud->height = height;
    cloud->is_dense = false;
    cloud->points.resize(width * height);

    for (int i = 0; i < width * height; ++i) {
        cloud->points[i].x = (float)(i % width);
        cloud->points[i].y = (float)(i / width);
        cloud->points[i].z = depth[i];  // depth value
    }

    return cloud;
}
