#pragma once
#include <vector>
#include "IMUProcessor.h"

struct Point3D { float x,y,z; };

class PointCloudProcessor {
public:
    PointCloudProcessor(int maxPoints);
    ~PointCloudProcessor();
    void fuseWithIMU(std::vector<Point3D>& points, const IMUProcessor& imu);
    void cudaFilter(float threshold);
    Point3D* getDevicePoints() { return d_points; }
    void copyToHost(std::vector<Point3D>& points);

private:
    int maxPoints;
    Point3D* d_points;
};
