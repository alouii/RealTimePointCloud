#include "pipeline.h"
#include "visualizer.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

// Load point cloud from binary file
std::vector<Point> load_point_cloud_bin(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {};
    }
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t num_points = file_size / (4 * sizeof(float)); // x, y, z, intensity
    std::vector<Point> points(num_points);
    
    for (size_t i = 0; i < num_points; ++i) {
        float data[4];
        file.read(reinterpret_cast<char*>(data), 4 * sizeof(float));
        points[i] = Point(data[0], data[1], data[2], data[3]);
    }
    
    file.close();
    std::cout << "Loaded " << num_points << " points from " << filename << std::endl;
    return points;
}

// Generate synthetic point cloud for testing
std::vector<Point> generate_synthetic_cloud(int num_points) {
    std::vector<Point> points(num_points);
    
    for (int i = 0; i < num_points; ++i) {
        float theta = 2.0f * M_PI * float(rand()) / RAND_MAX;
        float phi = M_PI * float(rand()) / RAND_MAX;
        float radius = 10.0f + 20.0f * float(rand()) / RAND_MAX;
        
        points[i].x = radius * sin(phi) * cos(theta);
        points[i].y = radius * sin(phi) * sin(theta);
        points[i].z = radius * cos(phi);
        points[i].intensity = float(rand()) / RAND_MAX;
    }
    
    return points;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -i <input_file>    : Input point cloud file (.bin)" << std::endl;
    std::cout << "  -c <config_file>   : Configuration file" << std::endl;
    std::cout << "  -s <num_points>    : Generate synthetic cloud with N points" << std::endl;
    std::cout << "  -v                 : Enable visualization" << std::endl;
    std::cout << "  -b <num_frames>    : Benchmark mode (process N frames)" << std::endl;
    std::cout << "  -h                 : Print this help message" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Point Cloud Processing Pipeline" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Parse command line arguments
    std::string input_file;
    std::string config_file = "config.txt";
    int synthetic_points = 0;
    bool enable_viz = false;
    int benchmark_frames = 0;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_file = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            config_file = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            synthetic_points = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            enable_viz = true;
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            benchmark_frames = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // Load configuration
    PipelineConfig config;
    config.load_from_file(config_file);
    
    // Initialize pipeline
    std::cout << "\nInitializing pipeline..." << std::endl;
    PointCloudPipeline pipeline(config);
    
    // Warmup
    pipeline.warmup(5);
    
    // Load or generate point cloud
    std::vector<Point> points;
    if (!input_file.empty()) {
        points = load_point_cloud_bin(input_file);
    } else if (synthetic_points > 0) {
        std::cout << "Generating synthetic point cloud with " 
                  << synthetic_points << " points..." << std::endl;
        points = generate_synthetic_cloud(synthetic_points);
    } else {
        std::cout << "No input specified, using default synthetic cloud" << std::endl;
        points = generate_synthetic_cloud(10000);
    }
    
    if (points.empty()) {
        std::cerr << "Error: No valid point cloud data" << std::endl;
        return 1;
    }
    
    // Benchmark mode
    if (benchmark_frames > 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Running benchmark (" << benchmark_frames << " frames)..." << std::endl;
        std::cout << "========================================" << std::endl;
        
        Timer total_timer;
        double total_preprocessing = 0;
        double total_inference = 0;
        double total_postprocessing = 0;
        
        for (int i = 0; i < benchmark_frames; ++i) {
            auto detections = pipeline.process_frame(points);
            auto stats = pipeline.get_stats();
            
            total_preprocessing += stats.preprocessing_time_ms;
            total_inference += stats.inference_time_ms;
            total_postprocessing += stats.postprocessing_time_ms;
            
            if ((i + 1) % 10 == 0) {
                std::cout << "Processed " << (i + 1) << " frames..." << std::endl;
            }
        }
        
        double total_time = total_timer.elapsed_ms();
        double avg_fps = benchmark_frames / (total_time / 1000.0);
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Benchmark Results:" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Total frames:       " << benchmark_frames << std::endl;
        std::cout << "Total time:         " << total_time / 1000.0 << " seconds" << std::endl;
        std::cout << "Average FPS:        " << avg_fps << std::endl;
        std::cout << "Avg preprocessing:  " << total_preprocessing / benchmark_frames << " ms" << std::endl;
        std::cout << "Avg inference:      " << total_inference / benchmark_frames << " ms" << std::endl;
        std::cout << "Avg postprocessing: " << total_postprocessing / benchmark_frames << " ms" << std::endl;
        
        return 0;
    }
    
    // Process single frame
    std::cout << "\n========================================" << std::endl;
    std::cout << "Processing point cloud..." << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto detections = pipeline.process_frame(points);
    auto stats = pipeline.get_stats();
    
    std::cout << "\nProcessing Statistics:" << std::endl;
    std::cout << "  Input points:       " << stats.num_input_points << std::endl;
    std::cout << "  Detections:         " << stats.num_detections << std::endl;
    std::cout << "  Preprocessing:      " << stats.preprocessing_time_ms << " ms" << std::endl;
    std::cout << "  Inference:          " << stats.inference_time_ms << " ms" << std::endl;
    std::cout << "  Postprocessing:     " << stats.postprocessing_time_ms << " ms" << std::endl;
    std::cout << "  Total time:         " << stats.total_time_ms << " ms" << std::endl;
    std::cout << "  Throughput:         " << stats.throughput_fps << " FPS" << std::endl;
    
    std::cout << "\nDetections:" << std::endl;
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        std::cout << "  [" << i << "] Class " << det.class_id 
                  << " (confidence: " << det.confidence << ")" << std::endl;
        std::cout << "      BBox: [" << det.bbox.min_x << ", " << det.bbox.min_y 
                  << ", " << det.bbox.min_z << "] -> ["
                  << det.bbox.max_x << ", " << det.bbox.max_y 
                  << ", " << det.bbox.max_z << "]" << std::endl;
    }
    
    // Visualization
    if (enable_viz) {
#ifdef USE_PCL_VISUALIZATION
        std::cout << "\nStarting visualization..." << std::endl;
        Visualizer viz("Point Cloud Pipeline");
        viz.update_point_cloud(points);
        viz.add_detections(detections);
        viz.spin();
#else
        std::cout << "\nVisualization not available (PCL not installed)" << std::endl;
        std::cout << "Using terminal visualizer instead:" << std::endl;
        TerminalVisualizer term_viz(80, 30);
        term_viz.render(points, detections);
#endif
    }
    
    std::cout << "\nDone!" << std::endl;
    return 0;
}
