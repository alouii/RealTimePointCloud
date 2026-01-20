#pragma once
#include "PointCloudProcessor.h"

class PointNetNN{
public:
    PointNetNN(int numPoints,int numClasses);
    ~PointNetNN();
    void forwardGPU(Point3D* d_points,float* d_output);

private:
    int numPoints,numClasses;
    float *d_weights1,*d_weights2,*d_weights3;
};
