cmake_minimum_required(VERSION 3.18)
project(PointCloudPipeline CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

# Optional: PCL for visualization
find_package(PCL 1.10 QUIET)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

if(PCL_FOUND)
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
    add_definitions(-DUSE_PCL_VISUALIZATION)
endif()

# CUDA compilation flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
    -O3;
    -use_fast_math;
    -Xptxas -O3;
    --expt-relaxed-constexpr;
)

# Source files
set(CUDA_SOURCES
    cuda/preprocessing.cu
    cuda/inference.cu
    cuda/memory_pool.cu
    cuda/voxelization.cu
)

set(CPP_SOURCES
    src/pipeline.cpp
    src/point_cloud.cpp
    src/config.cpp
    src/timer.cpp
)

if(PCL_FOUND)
    set(CPP_SOURCES ${CPP_SOURCES} src/visualizer.cpp)
endif()

# Create CUDA library
add_library(cuda_kernels STATIC ${CUDA_SOURCES})
set_target_properties(cuda_kernels PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Main executable
add_executable(point_cloud_pipeline src/main.cpp ${CPP_SOURCES})
target_link_libraries(point_cloud_pipeline 
    cuda_kernels
    ${CUDA_LIBRARIES}
    OpenMP::OpenMP_CXX
)

if(PCL_FOUND)
    target_link_libraries(point_cloud_pipeline ${PCL_LIBRARIES})
endif()

# Benchmark executable
add_executable(benchmark tests/benchmark.cpp ${CPP_SOURCES})
target_link_libraries(benchmark 
    cuda_kernels
    ${CUDA_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Test executable
add_executable(unit_tests tests/unit_tests.cpp ${CPP_SOURCES})
target_link_libraries(unit_tests 
    cuda_kernels
    ${CUDA_LIBRARIES}
    OpenMP::OpenMP_CXX
)

if(PCL_FOUND)
    target_link_libraries(benchmark ${PCL_LIBRARIES})
    target_link_libraries(unit_tests ${PCL_LIBRARIES})
endif()

# Set output directories
set_target_properties(point_cloud_pipeline benchmark unit_tests
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
