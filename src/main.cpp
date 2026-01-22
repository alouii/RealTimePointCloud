#include <iostream>
#include <vector>
#include <thread>
#include "PointCloudProcessor.h"
#include "IMUProcessor.h"
#include "PointNetNN.h"
//#include <pcl/visualization/cloud_viewer.h>
#include </home/aloui/US/pcl/visualization/include/pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>

int main(){
    int maxPoints=100000,numClasses=5;
    PointCloudProcessor pc(maxPoints);
    IMUProcessor imu;
    PointNetNN pointnet(maxPoints,numClasses);
    pcl::visualization::CloudViewer viewer("Point Cloud Viewer");

    std::vector<Point3D> points(maxPoints);
    float* d_output; cudaMalloc(&d_output,maxPoints*numClasses*sizeof(float));

    for(int frame=0;frame<1000;frame++){
        IMUData imuData{0,0,0,1};
        imu.update(imuData);

        for(int i=0;i<maxPoints;i++) points[i]={float(i%100),float(i%50),float(i%30)};
        pc.fuseWithIMU(points,imu);
        cudaMemcpy(pc.getDevicePoints(),points.data(),maxPoints*sizeof(Point3D),cudaMemcpyHostToDevice);
        pc.cudaFilter(25.0f);
        pointnet.forwardGPU(pc.getDevicePoints(),d_output);
        pc.copyToHost(points);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for(auto&p:points) cloud->points.push_back(pcl::PointXYZ(p.x,p.y,p.z));
        viewer.showCloud(cloud);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    cudaFree(d_output);
    return 0;
}

