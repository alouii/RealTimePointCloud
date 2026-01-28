FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    wget \
    libeigen3-dev \
    libpcl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/point-cloud-pipeline

# Build the project
WORKDIR /workspace/point-cloud-pipeline
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Set up environment
ENV PATH="/workspace/point-cloud-pipeline/build/bin:${PATH}"

# Default command
CMD ["/bin/bash"]
