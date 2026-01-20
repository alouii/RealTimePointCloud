Project: Real-Time Point Cloud Processing + PointNet Inference (Real Data Ready)
What you will get

Realistic ToF + IMU dataset (synthetic for demo or you can replace with your own sensors).

PointNet weights trained for simple classification or segmentation.

End-to-end C++/CUDA pipeline:

IMU+ToF fusion

CUDA-accelerated filtering

PointNet inference on GPU

Real-time visualization with PCL

Optimized for performance (using CUDA streams & pinned memory).

Full tutorial to run and adapt for your own data.

1. Project Structure
RealTimePointCloudNN/
│
├─ CMakeLists.txt
├─ README.md
├─ include/
│   ├─ PointCloudProcessor.h
│   ├─ IMUProcessor.h
│   └─ PointNetNN.h
├─ src/
│   ├─ main.cpp
│   ├─ PointCloudProcessor.cu
│   ├─ IMUProcessor.cpp
│   └─ PointNetNN.cu
├─ data/
│   ├─ demo_pointcloud.npy         # demo ToF points
│   ├─ demo_imu.npy                # demo IMU rotations
│   └─ pointnet_weights.bin        # pre-trained PointNet weights
└─ models/
    └─ pointnet_model.py           # Python script used to train weights

2. Key Features

IMU + ToF Fusion:

Converts ToF depth maps into 3D points.

Applies IMU rotation for real-world alignment.

CUDA Acceleration:

Filters outliers and applies optional downsampling.

Runs lightweight PointNet inference on GPU.

PointNet Inference:

Pre-trained weights for demo classes (e.g., 5 classes of objects).

Per-point predictions or global classification.

Real-Time Visualization:

PCL viewer with ~30 FPS demo.

Easy to replace with ROS2 or OpenGL for robotics pipelines.

3. How It Works

Load point cloud + IMU data.

Apply IMU rotation to each point.

Copy points to GPU.

Run CUDA kernel for filtering.

Run PointNet inference on GPU.

Copy results back and visualize in real-time.

4. README Tutorial
Requirements

Ubuntu 20+ / Windows 11

CUDA 12+

PCL 1.12+

CMake 3.18+

Optional: Python 3.10 for weight training

Build
git clone https://github.com/YourRepo/RealTimePointCloudNN.git
cd RealTimePointCloudNN
mkdir build && cd build
cmake ..
make -j

Run
./RTCP


Visualizes the point cloud in real-time.

Uses PointNet inference with demo weights.

IMU + ToF fusion applied automatically.
