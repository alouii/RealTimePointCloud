
cmake_minimum_required(VERSION 3.18)
project(RealTimePointCloudNN LANGUAGES C CXX CUDA)

# ----------------------------
# Compiler setup
# ----------------------------
set(PCL_FIND_VISUALIZATION OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use GCC 12 (optional if environment variables are set)
set(CMAKE_C_COMPILER "/usr/bin/gcc-12")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")

# ----------------------------
# Find CUDA
# ----------------------------
find_package(CUDA REQUIRED)

# ----------------------------
# Find PCL
# ----------------------------
# Optional: disable visualization if you don't want VTK/MPI
# set(PCL_FIND_VISUALIZATION OFF)
find_package(PCL 1.15
 REQUIRED)

# Include directories
include_directories(
    ${PCL_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)
add_definitions(${PCL_DEFINITIONS})

# ----------------------------
# Source files
# ----------------------------
set(SOURCES
    src/main.cpp
    src/PointNetNN.cu
    src/PointCloudProcessor.cu
    src/IMUProcessor.cpp
)

# ----------------------------
# Executable
# ----------------------------
add_executable(RTCP ${SOURCES})

# ----------------------------
# Link libraries
# ----------------------------
target_link_libraries(RTCP
    ${PCL_LIBRARIES}
    ${CUDA_LIBRARIES}
)

# ----------------------------
# Compiler flags (optional)
# ----------------------------
# Avoid unsupported compiler errors for newer GCC versions
# set_target_properties(RTCP PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_STANDARD 17
# )
# target_compile_options(RTCP PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -Wall -Wextra -allow-unsupported-compiler>)
