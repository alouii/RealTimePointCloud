//run with ./point_cloud_pipeline -i data/sample_cloud.pcd -c config.txt -v
#include "pipeline.h"
#include "visualizer.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

#ifdef USE_PCL_VISUALIZATION
#include <pcl/io/pcd_io.h>  
#include <pcl/point_types.h>
#endif

// --- Point Cloud Loaders ---

// Load binary .bin point cloud (x, y, z, intensity)
std::vector<Point> load_point_cloud_bin(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {};
    }
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (file_size % (4 * sizeof(float)) != 0) {
        std::cerr << "Warning: file size is not multiple of 4 floats. Possible corruption." << std::endl;
    }

    size_t num_points = file_size / (4 * sizeof(float));
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

#ifdef USE_PCL_VISUALIZATION
// Load PCD point cloud
std::vector<Point> load_point_cloud_pcd(const std::string& filename) {
    pcl::PointCloud<pcl::PointXYZI> cloud;
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(filename, cloud) == -1) {
        std::cerr << "Failed to load PCD file: " << filename << std::endl;
        return {};
    }

    std::vector<Point> points(cloud.points.size());
    for (size_t i = 0; i < cloud.points.size(); ++i) {
        points[i] = Point(cloud.points[i].x,
                          cloud.points[i].y,
                          cloud.points[i].z,
                          cloud.points[i].intensity);
    }

    std::cout << "Loaded " << cloud.points.size() << " points from " << filename << std::endl;
    return points;
}
#endif

// Generate synthetic point cloud
std::vector<Point> generate_synthetic_cloud(int num_points) {
    std::vector<Point> points(num_points);
    for (int i = 0; i < num_points; ++i) {
        float theta = 2.0f * M_PI * float(rand()) / RAND_MAX;
        float phi   = M_PI * float(rand()) / RAND_MAX;
        float r     = 10.0f + 20.0f * float(rand()) / RAND_MAX;

        points[i].x = r * sin(phi) * cos(theta);
        points[i].y = r * sin(phi) * sin(theta);
        points[i].z = r * cos(phi);
        points[i].intensity = float(rand()) / RAND_MAX;
    }
    return points;
}

// --- Main ---

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -i <input_file>    : Input point cloud (.bin or .pcd)\n";
    std::cout << "  -c <config_file>   : Configuration file\n";
    std::cout << "  -s <num_points>    : Generate synthetic cloud\n";
    std::cout << "  -v                 : Enable visualization\n";
    std::cout << "  -b <num_frames>    : Benchmark N frames\n";
    std::cout << "  -h                 : Print help\n";
}

int main(int argc, char** argv) {
    std::string input_file;
    std::string config_file = "config.txt";
    int synthetic_points = 0;
    bool enable_viz = false;
    int benchmark_frames = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) input_file = argv[++i];
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) config_file = argv[++i];
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) synthetic_points = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) enable_viz = true;
        else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) benchmark_frames = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0) { print_usage(argv[0]); return 0; }
    }

    // Load config
    PipelineConfig config;
    config.load_from_file(config_file);

    std::cout << "\nInitializing pipeline..." << std::endl;
    PointCloudPipeline pipeline(config);
    pipeline.warmup(5);

    // Load point cloud
    std::vector<Point> points;
    if (!input_file.empty()) {
#ifdef USE_PCL_VISUALIZATION
        /*if (input_file.ends_with(".pcd")) points = load_point_cloud_pcd(input_file);
        else points = load_point_cloud_bin(input_file);#c++20
        */
       auto has_suffix = [](const std::string& str, const std::string& suffix) {
            return str.size() >= suffix.size() &&
            str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
        };
        if (has_suffix(input_file, ".pcd")) points = load_point_cloud_pcd(input_file);
        else points = load_point_cloud_bin(input_file);
#else
        points = load_point_cloud_bin(input_file);
#endif
    } else if (synthetic_points > 0) points = generate_synthetic_cloud(synthetic_points);
    else points = generate_synthetic_cloud(10000);

    if (points.empty()) {
        std::cerr << "Error: no valid point cloud data" << std::endl;
        return 1;
    }

    // Truncate points if exceeding max_points
    if (points.size() > config.max_points) {
        std::cerr << "Warning: input has " << points.size() 
                  << " points, but max_points is " << config.max_points 
                  << ". Truncating.\n";
        points.resize(config.max_points);
    }

    // --- Process frame ---
    auto detections = pipeline.process_frame(points);
    auto stats = pipeline.get_stats();

    std::cout << "\nProcessing Statistics:\n";
    std::cout << "  Input points:       " << stats.num_input_points << "\n";
    std::cout << "  Detections:         " << stats.num_detections << "\n";
    std::cout << "  Preprocessing:      " << stats.preprocessing_time_ms << " ms\n";
    std::cout << "  Inference:          " << stats.inference_time_ms << " ms\n";
    std::cout << "  Postprocessing:     " << stats.postprocessing_time_ms << " ms\n";
    std::cout << "  Total time:         " << stats.total_time_ms << " ms\n";

    // --- Visualization ---
    if (enable_viz) {
#ifdef USE_PCL_VISUALIZATION
        Visualizer viz("Point Cloud Pipeline");
        viz.update_point_cloud(points);
        viz.add_detections(detections);
        viz.spin();
#else
        TerminalVisualizer term_viz(80, 30);
        term_viz.render(points, detections);
#endif
    }

    std::cout << "\nDone!\n";
    return 0;
}
