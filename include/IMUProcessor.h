#pragma once
struct IMUData { float qx,qy,qz,qw; };

class IMUProcessor {
public:
    IMUProcessor();
    void update(const IMUData& imu);
    void getRotationMatrix(float R[3][3]) const;
private:
    float qx,qy,qz,qw;
};
