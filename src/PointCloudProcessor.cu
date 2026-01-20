#include "PointCloudProcessor.h"
#include <cuda_runtime.h>

__global__ void removeOutliersKernel(Point3D* points,int N,float threshold){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N && points[idx].z>threshold) points[idx].z=0;
}

PointCloudProcessor::PointCloudProcessor(int maxPoints):maxPoints(maxPoints){
    cudaMalloc(&d_points,maxPoints*sizeof(Point3D));
}
PointCloudProcessor::~PointCloudProcessor(){ cudaFree(d_points); }

void PointCloudProcessor::fuseWithIMU(std::vector<Point3D>& points,const IMUProcessor& imu){
    float R[3][3]; imu.getRotationMatrix(R);
    for(auto& p: points){
        Point3D tmp=p;
        p.x=R[0][0]*tmp.x+R[0][1]*tmp.y+R[0][2]*tmp.z;
        p.y=R[1][0]*tmp.x+R[1][1]*tmp.y+R[1][2]*tmp.z;
        p.z=R[2][0]*tmp.x+R[2][1]*tmp.y+R[2][2]*tmp.z;
    }
}

void PointCloudProcessor::cudaFilter(float threshold){
    int blockSize=256,numBlocks=(maxPoints+blockSize-1)/blockSize;
    removeOutliersKernel<<<numBlocks,blockSize>>>(d_points,maxPoints,threshold);
    cudaDeviceSynchronize();
}

void PointCloudProcessor::copyToHost(std::vector<Point3D>& points){
    cudaMemcpy(points.data(),d_points,maxPoints*sizeof(Point3D),cudaMemcpyDeviceToHost);
}
