#!/bin/bash

# Point Cloud Pipeline - Build and Run Script

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "Point Cloud Pipeline - Build Script"
echo "========================================"
echo ""

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: CUDA compiler (nvcc) not found!${NC}"
    echo "Please install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo -e "${GREEN}✓${NC} Found CUDA version: $CUDA_VERSION"

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: CMake not found!${NC}"
    echo "Install with: sudo apt-get install cmake"
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
echo -e "${GREEN}✓${NC} Found CMake version: $CMAKE_VERSION"

# Check for required libraries
echo ""
echo "Checking dependencies..."

check_library() {
    if pkg-config --exists $1 2>/dev/null; then
        VERSION=$(pkg-config --modversion $1)
        echo -e "${GREEN}✓${NC} $1 (version $VERSION)"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $1 not found (optional)"
        return 1
    fi
}

check_library "eigen3"
check_library "pcl_common-1.10" || check_library "pcl_common-1.12"

# Create build directory
echo ""
echo "Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo ""
echo "Configuring project..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo ""
echo "Building project..."
make -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================"
    echo "Build completed successfully!"
    echo "========================================${NC}"
    echo ""
    echo "Executables are located in: build/bin/"
    echo ""
    echo "Available commands:"
    echo "  ./bin/point_cloud_pipeline    - Main application"
    echo "  ./bin/benchmark               - Performance benchmarks"
    echo "  ./bin/unit_tests              - Unit test suite"
    echo ""
    echo "Example usage:"
    echo "  ./bin/point_cloud_pipeline -s 10000 -v"
    echo "  ./bin/benchmark all"
    echo "  ./bin/unit_tests"
    echo ""
else
    echo -e "${RED}========================================"
    echo "Build failed!"
    echo "========================================${NC}"
    exit 1
fi

# Offer to run tests
read -p "Would you like to run unit tests? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Running unit tests..."
    ./bin/unit_tests
fi
