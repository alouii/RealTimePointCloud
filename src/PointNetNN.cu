#include "PointNetNN.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void pointnetKernel(Point3D* points, float* output,
                               float* w1, float* w2, float* w3,
                               int N, int numClasses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;

    float x = points[idx].x;
    float y = points[idx].y;
    float z = points[idx].z;

    // Layer 1: compute one neuron at a time
    for(int i=0;i<64;i++){
        float val = fmaxf(0, x*w1[i*3]+y*w1[i*3+1]+z*w1[i*3+2]);
        // propagate to next layer immediately to avoid large arrays
        float val2 = 0;
        for(int j=0;j<128;j++){
            val2 += val * w2[j*64 + i];
            val2 = fmaxf(0, val2);
            output[idx*numClasses+j%numClasses] = val2; // simple demo
        }
    }
}

