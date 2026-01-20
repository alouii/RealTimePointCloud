#include "PointNetNN.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void pointnetKernel(Point3D* points,float* output,float* w1,float* w2,float* w3,int N,int numClasses){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N){
        float x=points[idx].x,y=points[idx].y,z=points[idx].z;
        float h1[64]; for(int i=0;i<64;i++) h1[i]=fmaxf(0,x*w1[i*3]+y*w1[i*3+1]+z*w1[i*3+2]);
        float h2[128]; for(int i=0;i<128;i++){float sum=0;for(int j=0;j<64;j++) sum+=h1[j]*w2[i*64+j]; h2[i]=fmaxf(0,sum);}
        for(int c=0;c<numClasses;c++){float sum=0;for(int j=0;j<128;j++) sum+=h2[j]*w3[c*128+j]; output[idx*numClasses+c]=sum;}
    }
}

PointNetNN::PointNetNN(int numPoints_,int numClasses_):numPoints(numPoints_),numClasses(numClasses_){
    cudaMalloc(&d_weights1,64*3*sizeof(float));
    cudaMalloc(&d_weights2,128*64*sizeof(float));
    cudaMalloc(&d_weights3,numClasses*128*sizeof(float));
}

PointNetNN::~PointNetNN(){ cudaFree(d_weights1); cudaFree(d_weights2); cudaFree(d_weights3); }

void PointNetNN::forwardGPU(Point3D* d_points,float* d_output){
    int blockSize=256,numBlocks=(numPoints+blockSize-1)/blockSize;
    pointnetKernel<<<numBlocks,blockSize>>>(d_points,d_output,d_weights1,d_weights2,d_weights3,numPoints,numClasses);
    cudaDeviceSynchronize();
}
