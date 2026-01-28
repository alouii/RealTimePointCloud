#include "memory_pool.h"
#include <algorithm>
#include <iostream>

// GPU Memory Pool implementation
void* GPUMemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Try to find a free block of sufficient size
    for (auto& block : blocks_) {
        if (!block.in_use && block.size >= size) {
            block.in_use = true;
            total_in_use_ += block.size;
            return block.ptr;
        }
    }
    
    // No suitable block found, allocate new one
    void* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    
    Block new_block;
    new_block.ptr = ptr;
    new_block.size = size;
    new_block.in_use = true;
    
    blocks_.push_back(new_block);
    total_allocated_ += size;
    total_in_use_ += size;
    
    return ptr;
}

void GPUMemoryPool::deallocate(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& block : blocks_) {
        if (block.ptr == ptr) {
            block.in_use = false;
            total_in_use_ -= block.size;
            return;
        }
    }
    
    std::cerr << "Warning: Attempting to deallocate unknown pointer" << std::endl;
}

void GPUMemoryPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& block : blocks_) {
        if (block.ptr) {
            cudaFree(block.ptr);
        }
    }
    
    blocks_.clear();
    total_allocated_ = 0;
    total_in_use_ = 0;
}

// Pinned Memory Allocator implementation
void* PinnedMemoryAllocator::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    void* ptr;
    CUDA_CHECK(cudaMallocHost(&ptr, size));
    
    allocations_[ptr] = size;
    
    return ptr;
}

void PinnedMemoryAllocator::deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        CUDA_CHECK(cudaFreeHost(ptr));
        allocations_.erase(it);
    } else {
        std::cerr << "Warning: Attempting to deallocate unknown pinned memory" << std::endl;
    }
}

void PinnedMemoryAllocator::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& [ptr, size] : allocations_) {
        cudaFreeHost(ptr);
    }
    
    allocations_.clear();
}
