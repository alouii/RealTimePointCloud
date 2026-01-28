# Quick Start Guide

## Installation (5 minutes)

### 1. Prerequisites
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install cmake build-essential libeigen3-dev

# Install CUDA Toolkit (if not already installed)
# Download from: https://developer.nvidia.com/cuda-downloads
```

### 2. Build
```bash
# Clone or navigate to project directory
cd point-cloud-pipeline

# Run build script
./build.sh
```

## Basic Usage

### Process Synthetic Data
```bash
cd build
./bin/point_cloud_pipeline -s 10000
```

### Process Your Own Data
```bash
# Binary format: [x, y, z, intensity] as floats
./bin/point_cloud_pipeline -i your_data.bin -v
```

### Run Benchmarks
```bash
./bin/benchmark all
```

## Example Output

```
========================================
Point Cloud Processing Pipeline
========================================

Initializing pipeline...
  Max points: 100000
  Voxel size: 0.1 m
  Max range: 50.0 m

Warming up pipeline...
Warmup complete!

Processing point cloud...
========================================

Processing Statistics:
  Input points:       10000
  Detections:         5
  Preprocessing:      2.5 ms
  Inference:          8.3 ms
  Postprocessing:     1.2 ms
  Total time:         12.0 ms
  Throughput:         83.3 FPS

Detections:
  [0] Class 3 (confidence: 0.85)
      BBox: [-5.2, -3.1, 0.0] -> [-2.1, 1.5, 2.3]
  [1] Class 1 (confidence: 0.78)
      BBox: [2.3, -1.5, 0.0] -> [5.1, 2.3, 1.8]
```

## Configuration

Edit `config.txt` to customize:

```ini
# Key parameters
max_points = 100000          # Maximum points to process
voxel_size = 0.1            # Voxel size for downsampling (m)
max_range = 50.0            # Maximum point range (m)
confidence_threshold = 0.5   # Detection confidence threshold
```

## Performance Tips

1. **GPU Selection**: Set `CUDA_VISIBLE_DEVICES=0` to select GPU
2. **Memory**: Increase `memory_pool_size` for larger point clouds
3. **Throughput**: Enable `use_pinned_memory` for faster transfers
4. **Batch Processing**: Increase `batch_size` for multiple frames

## Troubleshooting

### "CUDA out of memory"
```bash
# Reduce max_points in config.txt
max_points = 50000
```

### "PCL not found"
```bash
# Install PCL (optional, for visualization)
sudo apt-get install libpcl-dev

# Or use terminal visualization (built-in)
./bin/point_cloud_pipeline -s 10000
```

### Build errors
```bash
# Check CUDA version
nvcc --version

# Ensure CMake >= 3.18
cmake --version

# Update if needed
sudo apt-get install cmake
```

## Next Steps

- Read the full [README.md](README.md)
- Explore the [API examples](README.md#api-usage)
- Run [benchmarks](README.md#benchmarking)
- Check out the [project structure](README.md#project-structure)

## Support

For issues, please check:
1. [README.md](README.md) - Full documentation
2. [Troubleshooting](README.md#troubleshooting) section
3. GitHub Issues (if applicable)
