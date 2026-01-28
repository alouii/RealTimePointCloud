#include "pipeline.h"
#include "memory_pool.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

class Benchmark {
public:
    Benchmark(const std::string& name) : name_(name) {}
    
    void add_result(double time_ms) {
        times_.push_back(time_ms);
    }
    
    void print_summary() const {
        if (times_.empty()) return;
        
        double sum = 0;
        double min_time = times_[0];
        double max_time = times_[0];
        
        for (double t : times_) {
            sum += t;
            min_time = std::min(min_time, t);
            max_time = std::max(max_time, t);
        }
        
        double avg = sum / times_.size();
        
        // Compute standard deviation
        double variance = 0;
        for (double t : times_) {
            variance += (t - avg) * (t - avg);
        }
        double std_dev = std::sqrt(variance / times_.size());
        
        std::cout << "\n" << name_ << ":" << std::endl;
        std::cout << "  Iterations:  " << times_.size() << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Average:     " << avg << " ms" << std::endl;
        std::cout << "  Min:         " << min_time << " ms" << std::endl;
        std::cout << "  Max:         " << max_time << " ms" << std::endl;
        std::cout << "  Std Dev:     " << std_dev << " ms" << std::endl;
        std::cout << "  Throughput:  " << (1000.0 / avg) << " FPS" << std::endl;
    }
    
private:
    std::string name_;
    std::vector<double> times_;
};

std::vector<Point> generate_random_points(int num_points, float range) {
    std::vector<Point> points(num_points);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-range, range);
    std::uniform_real_distribution<float> intensity(0.0f, 1.0f);
    
    for (int i = 0; i < num_points; ++i) {
        points[i] = Point(dist(gen), dist(gen), dist(gen), intensity(gen));
    }
    
    return points;
}

void benchmark_memory_operations() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Memory Operations Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    
    const int num_iterations = 100;
    const size_t sizes[] = {1024, 1024 * 1024, 10 * 1024 * 1024};
    
    for (size_t size : sizes) {
        Benchmark bench("Memory Allocation (" + std::to_string(size / 1024) + " KB)");
        
        for (int i = 0; i < num_iterations; ++i) {
            Timer timer;
            void* ptr = GPUMemoryPool::get_instance().allocate(size);
            double elapsed = timer.elapsed_ms();
            bench.add_result(elapsed);
            GPUMemoryPool::get_instance().deallocate(ptr, size);
        }
        
        bench.print_summary();
    }
    
    GPUMemoryPool::get_instance().clear();
}

void benchmark_preprocessing(PointCloudPipeline& pipeline) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Preprocessing Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    
    const int point_counts[] = {1000, 10000, 100000};
    const int num_iterations = 50;
    
    for (int count : point_counts) {
        auto points = generate_random_points(count, 50.0f);
        Benchmark bench("Preprocessing (" + std::to_string(count) + " points)");
        
        for (int i = 0; i < num_iterations; ++i) {
            Timer timer;
            pipeline.process_frame(points);
            bench.add_result(timer.elapsed_ms());
        }
        
        bench.print_summary();
    }
}

void benchmark_end_to_end(PointCloudPipeline& pipeline) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "End-to-End Pipeline Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    
    const int num_points = 50000;
    const int num_iterations = 100;
    
    auto points = generate_random_points(num_points, 50.0f);
    
    Benchmark total_bench("Total Pipeline");
    Benchmark prep_bench("Preprocessing Stage");
    Benchmark inf_bench("Inference Stage");
    Benchmark post_bench("Postprocessing Stage");
    
    for (int i = 0; i < num_iterations; ++i) {
        Timer timer;
        auto detections = pipeline.process_frame(points);
        total_bench.add_result(timer.elapsed_ms());
        
        auto stats = pipeline.get_stats();
        prep_bench.add_result(stats.preprocessing_time_ms);
        inf_bench.add_result(stats.inference_time_ms);
        post_bench.add_result(stats.postprocessing_time_ms);
    }
    
    total_bench.print_summary();
    prep_bench.print_summary();
    inf_bench.print_summary();
    post_bench.print_summary();
}

void benchmark_batch_processing(PointCloudPipeline& pipeline) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Batch Processing Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    
    const int batch_sizes[] = {1, 4, 8, 16};
    const int num_points = 10000;
    const int num_iterations = 20;
    
    for (int batch_size : batch_sizes) {
        std::vector<std::vector<Point>> batch;
        for (int i = 0; i < batch_size; ++i) {
            batch.push_back(generate_random_points(num_points, 50.0f));
        }
        
        Benchmark bench("Batch Size " + std::to_string(batch_size));
        
        for (int i = 0; i < num_iterations; ++i) {
            Timer timer;
            pipeline.process_batch(batch);
            double elapsed = timer.elapsed_ms();
            bench.add_result(elapsed / batch_size); // Per-frame time
        }
        
        bench.print_summary();
    }
}

void benchmark_scalability() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Scalability Analysis" << std::endl;
    std::cout << "========================================" << std::endl;
    
    PipelineConfig config;
    PointCloudPipeline pipeline(config);
    pipeline.warmup(5);
    
    const int point_counts[] = {1000, 5000, 10000, 50000, 100000};
    
    std::cout << "\n" << std::setw(12) << "Points" 
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "FPS"
              << std::setw(20) << "Points/sec" << std::endl;
    std::cout << std::string(62, '-') << std::endl;
    
    for (int count : point_counts) {
        auto points = generate_random_points(count, 50.0f);
        
        // Average over multiple runs
        double total_time = 0;
        const int runs = 20;
        
        for (int i = 0; i < runs; ++i) {
            Timer timer;
            pipeline.process_frame(points);
            total_time += timer.elapsed_ms();
        }
        
        double avg_time = total_time / runs;
        double fps = 1000.0 / avg_time;
        double points_per_sec = count * fps;
        
        std::cout << std::setw(12) << count
                  << std::setw(15) << std::fixed << std::setprecision(2) << avg_time
                  << std::setw(15) << std::fixed << std::setprecision(2) << fps
                  << std::setw(20) << std::scientific << std::setprecision(2) << points_per_sec
                  << std::endl;
    }
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Point Cloud Pipeline Benchmark Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Initialize pipeline
    PipelineConfig config;
    config.max_points = 200000;
    config.voxel_size = 0.1f;
    config.max_range = 100.0f;
    
    PointCloudPipeline pipeline(config);
    
    std::cout << "\nWarming up GPU..." << std::endl;
    pipeline.warmup(10);
    
    // Run benchmarks
    if (argc > 1) {
        std::string test(argv[1]);
        
        if (test == "memory") {
            benchmark_memory_operations();
        } else if (test == "preprocessing") {
            benchmark_preprocessing(pipeline);
        } else if (test == "end-to-end") {
            benchmark_end_to_end(pipeline);
        } else if (test == "batch") {
            benchmark_batch_processing(pipeline);
        } else if (test == "scalability") {
            benchmark_scalability();
        } else if (test == "all") {
            benchmark_memory_operations();
            benchmark_preprocessing(pipeline);
            benchmark_end_to_end(pipeline);
            benchmark_batch_processing(pipeline);
            benchmark_scalability();
        } else {
            std::cout << "Unknown test: " << test << std::endl;
            std::cout << "Available tests: memory, preprocessing, end-to-end, batch, scalability, all" << std::endl;
        }
    } else {
        // Run all benchmarks by default
        benchmark_memory_operations();
        benchmark_preprocessing(pipeline);
        benchmark_end_to_end(pipeline);
        benchmark_batch_processing(pipeline);
        benchmark_scalability();
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Print memory statistics
    std::cout << "\nGPU Memory Pool Statistics:" << std::endl;
    std::cout << "  Total allocated: " 
              << GPUMemoryPool::get_instance().get_total_allocated() / (1024.0 * 1024.0) 
              << " MB" << std::endl;
    std::cout << "  Currently in use: " 
              << GPUMemoryPool::get_instance().get_total_in_use() / (1024.0 * 1024.0) 
              << " MB" << std::endl;
    
    return 0;
}
