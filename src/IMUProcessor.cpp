#include "IMUProcessor.h"
IMUProcessor::IMUProcessor() : qx(0),qy(0),qz(0),qw(1) {}
void IMUProcessor::update(const IMUData& imu) { qx=imu.qx;qy=imu.qy;qz=imu.qz;qw=imu.qw; }
void IMUProcessor::getRotationMatrix(float R[3][3]){
    R[0][0]=1-2*qy*qy-2*qz*qz; R[0][1]=2*qx*qy-2*qz*qw; R[0][2]=2*qx*qz+2*qy*qw;
    R[1][0]=2*qx*qy+2*qz*qw; R[1][1]=1-2*qx*qx-2*qz*qz; R[1][2]=2*qy*qz-2*qx*qw;
    R[2][0]=2*qx*qz-2*qy*qw; R[2][1]=2*qy*qz+2*qx*qw; R[2][2]=1-2*qx*qx-2*qy*qy;
}
