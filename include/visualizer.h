#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "point_cloud_types.h"
#include <vector>
#include <string>

#ifdef USE_PCL_VISUALIZATION
#include <pcl/visualization/pcl_visualizer.h>
//#include"/home/aloui/US/pcl/visualization/include/pcl/visualization/pcl_visualizer.h"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#endif

class Visualizer {
public:
    Visualizer(const std::string& window_name = "Point Cloud Visualizer");
    ~Visualizer();
    
    // Visualization methods
    void update_point_cloud(const std::vector<Point>& points);
    void add_detections(const std::vector<Detection>& detections);
    void clear_detections();
    
    // Display control
    void spin_once(int time_ms = 1);
    void spin();
    bool was_stopped() const;
    
    // Visualization settings
    void set_point_size(int size);
    void set_background_color(float r, float g, float b);
    void set_camera_position(float x, float y, float z);
    void show_coordinate_system(bool show = true);
    
    // Color mapping
    void color_by_intensity(bool enable);
    void color_by_height(bool enable);
    
    // Save/Load
    void save_screenshot(const std::string& filename);
    void save_camera_parameters(const std::string& filename);
    void load_camera_parameters(const std::string& filename);
    
private:
#ifdef USE_PCL_VISUALIZATION
    pcl::visualization::PCLVisualizer::Ptr viewer_;
    int detection_id_counter_;
#endif
    
    bool use_intensity_coloring_;
    bool use_height_coloring_;
    std::string window_name_;
    
    void initialize_viewer();
    void update_visualization();
};

// Simple ASCII art visualization for terminal
class TerminalVisualizer {
public:
    TerminalVisualizer(int width = 80, int height = 40);
    
    void render(const std::vector<Point>& points, 
               const std::vector<Detection>& detections);
    void clear();
    
private:
    int width_;
    int height_;
    std::vector<char> buffer_;
    
    void project_point(float x, float y, float z, int& screen_x, int& screen_y);
};

#endif // VISUALIZER_H
