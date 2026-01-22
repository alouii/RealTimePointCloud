cmake_minimum_required(VERSION 3.18)
project(RealTimePointCloudNN LANGUAGES C CXX CUDA)

# ----------------------------
# Compiler setup
# ----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional: force GCC 12
# set(CMAKE_C_COMPILER "/usr/bin/gcc-12")
# set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")

# ----------------------------
# Find CUDA
# ----------------------------
find_package(CUDA REQUIRED)

# ----------------------------
# Find PCL (core only)
# ----------------------------
#set(PCL_FIND_VISUALIZATION OFF)
find_package(PCL 1.15 REQUIRED COMPONENTS common io octree)

# ----------------------------
# Find OpenCV
# ----------------------------
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# Include directories
include_directories(
    ${PCL_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)
add_definitions(${PCL_DEFINITIONS})

# ----------------------------
# Source files
# ----------------------------
set(SOURCES
    src/main.cpp
    src/PointCloudProcessor.cpp
    src/PointNetNN.cu
    src/IMUProcessor.cpp
)

# ----------------------------
# Executable target
# ----------------------------
add_executable(RealTimePointCloud ${SOURCES})

# ----------------------------
# Link libraries
# ----------------------------
target_link_libraries(RealTimePointCloud
    ${PCL_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
)
