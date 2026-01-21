cmake_minimum_required(VERSION 3.18)
project(RealTimePointCloudNN)

find_package(CUDA REQUIRED)
find_package(PCL 1.12 REQUIRED)

set(CMAKE_CXX_STANDARD 17)
include_directories(include ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -allow-unsupported-compiler")

cuda_add_executable(RTCP src/main.cpp src/PointCloudProcessor.cu src/IMUProcessor.cpp src/PointNetNN.cu)

target_link_libraries(RTCP ${PCL_LIBRARIES} cuda)
