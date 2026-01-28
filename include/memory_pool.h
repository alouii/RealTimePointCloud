#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include "point_cloud_types.h"
#include <unordered_map>
#include <mutex>

// GPU Memory Pool for efficient memory reuse
class GPUMemoryPool {
public:
    static GPUMemoryPool& get_instance() {
        static GPUMemoryPool instance;
        return instance;
    }
    
    void* allocate(size_t size);
    void deallocate(void* ptr, size_t size);
    void clear();
    size_t get_total_allocated() const { return total_allocated_; }
    size_t get_total_in_use() const { return total_in_use_; }
    
private:
    GPUMemoryPool() : total_allocated_(0), total_in_use_(0) {}
    ~GPUMemoryPool() { clear(); }
    
    GPUMemoryPool(const GPUMemoryPool&) = delete;
    GPUMemoryPool& operator=(const GPUMemoryPool&) = delete;
    
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    std::vector<Block> blocks_;
    std::mutex mutex_;
    size_t total_allocated_;
    size_t total_in_use_;
};

// Pinned memory allocator for fast CPU-GPU transfers
class PinnedMemoryAllocator {
public:
    static PinnedMemoryAllocator& get_instance() {
        static PinnedMemoryAllocator instance;
        return instance;
    }
    
    void* allocate(size_t size);
    void deallocate(void* ptr);
    void clear();
    
private:
    PinnedMemoryAllocator() = default;
    ~PinnedMemoryAllocator() { clear(); }
    
    PinnedMemoryAllocator(const PinnedMemoryAllocator&) = delete;
    PinnedMemoryAllocator& operator=(const PinnedMemoryAllocator&) = delete;
    
    std::unordered_map<void*, size_t> allocations_;
    std::mutex mutex_;
};

// RAII wrapper for GPU memory
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : ptr_(nullptr), size_(0), capacity_(0) {}
    
    explicit DeviceBuffer(size_t count) : size_(0), capacity_(0) {
        resize(count);
    }
    
    ~DeviceBuffer() {
        free();
    }
    
    void resize(size_t count) {
        if (count > capacity_) {
            free();
            size_t bytes = count * sizeof(T);
            ptr_ = static_cast<T*>(GPUMemoryPool::get_instance().allocate(bytes));
            capacity_ = count;
        }
        size_ = count;
    }
    
    void free() {
        if (ptr_) {
            GPUMemoryPool::get_instance().deallocate(ptr_, capacity_ * sizeof(T));
            ptr_ = nullptr;
            capacity_ = 0;
            size_ = 0;
        }
    }
    
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    
    void copy_from_host(const T* host_data, size_t count) {
        if (count > capacity_) resize(count);
        CUDA_CHECK(cudaMemcpy(ptr_, host_data, count * sizeof(T), 
                              cudaMemcpyHostToDevice));
        size_ = count;
    }
    
    void copy_to_host(T* host_data) const {
        CUDA_CHECK(cudaMemcpy(host_data, ptr_, size_ * sizeof(T), 
                              cudaMemcpyDeviceToHost));
    }
    
    void zero() {
        CUDA_CHECK(cudaMemset(ptr_, 0, size_ * sizeof(T)));
    }
    
private:
    T* ptr_;
    size_t size_;
    size_t capacity_;
    
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

#endif // MEMORY_POOL_H
