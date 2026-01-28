#include "visualizer.h"
#include <iostream>

#ifdef USE_PCL_VISUALIZATION

Visualizer::Visualizer(const std::string& window_name)
    : window_name_(window_name)
    , detection_id_counter_(0)
    , use_intensity_coloring_(false)
    , use_height_coloring_(false)
{
    initialize_viewer();
}

Visualizer::~Visualizer() = default;

void Visualizer::initialize_viewer() {
    viewer_ = pcl::visualization::PCLVisualizer::Ptr(
        new pcl::visualization::PCLVisualizer(window_name_)
    );
    
    viewer_->setBackgroundColor(0.05, 0.05, 0.05);
    viewer_->addCoordinateSystem(1.0);
    viewer_->initCameraParameters();
    viewer_->setCameraPosition(0, 0, 50, 0, 0, 0, 0, 1, 0);
}

void Visualizer::update_point_cloud(const std::vector<Point>& points) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZI>
    );
    
    cloud->width = points.size();
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->points.resize(cloud->width * cloud->height);
    
    for (size_t i = 0; i < points.size(); ++i) {
        cloud->points[i].x = points[i].x;
        cloud->points[i].y = points[i].y;
        cloud->points[i].z = points[i].z;
        cloud->points[i].intensity = points[i].intensity;
    }
    
    if (!viewer_->updatePointCloud(cloud, "point_cloud")) {
        viewer_->addPointCloud(cloud, "point_cloud");
        viewer_->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "point_cloud"
        );
    }
}

void Visualizer::add_detections(const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        std::string id = "bbox_" + std::to_string(detection_id_counter_++);
        
        // Add bounding box
        viewer_->addCube(
            det.bbox.min_x, det.bbox.max_x,
            det.bbox.min_y, det.bbox.max_y,
            det.bbox.min_z, det.bbox.max_z,
            1.0, 0.0, 0.0, id
        );
        viewer_->setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
            id
        );
        
        // Add text label
        pcl::PointXYZ text_pos(
            (det.bbox.min_x + det.bbox.max_x) / 2,
            (det.bbox.min_y + det.bbox.max_y) / 2,
            det.bbox.max_z + 0.5f
        );
        
        std::string text = "Class " + std::to_string(det.class_id) + 
                          " (" + std::to_string(int(det.confidence * 100)) + "%)";
        viewer_->addText3D(text, text_pos, 0.5, 1.0, 1.0, 1.0, id + "_text");
    }
}

void Visualizer::clear_detections() {
    for (int i = 0; i < detection_id_counter_; ++i) {
        std::string id = "bbox_" + std::to_string(i);
        viewer_->removeShape(id);
        viewer_->removeText3D(id + "_text");
    }
    detection_id_counter_ = 0;
}

void Visualizer::spin_once(int time_ms) {
    viewer_->spinOnce(time_ms);
}

void Visualizer::spin() {
    viewer_->spin();
}

bool Visualizer::was_stopped() const {
    return viewer_->wasStopped();
}

void Visualizer::set_point_size(int size) {
    viewer_->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, size, "point_cloud"
    );
}

void Visualizer::set_background_color(float r, float g, float b) {
    viewer_->setBackgroundColor(r, g, b);
}

void Visualizer::set_camera_position(float x, float y, float z) {
    viewer_->setCameraPosition(x, y, z, 0, 0, 0, 0, 1, 0);
}

void Visualizer::show_coordinate_system(bool show) {
    if (show) {
        viewer_->addCoordinateSystem(1.0);
    } else {
        viewer_->removeCoordinateSystem();
    }
}

void Visualizer::color_by_intensity(bool enable) {
    use_intensity_coloring_ = enable;
}

void Visualizer::color_by_height(bool enable) {
    use_height_coloring_ = enable;
}

void Visualizer::save_screenshot(const std::string& filename) {
    viewer_->saveScreenshot(filename);
}

void Visualizer::save_camera_parameters(const std::string& filename) {
    viewer_->saveCameraParameters(filename);
}

void Visualizer::load_camera_parameters(const std::string& filename) {
    viewer_->loadCameraParameters(filename);
}

#else

// Stub implementation when PCL is not available
Visualizer::Visualizer(const std::string& window_name)
    : window_name_(window_name)
    , use_intensity_coloring_(false)
    , use_height_coloring_(false)
{
    std::cout << "Visualizer created (PCL visualization disabled)" << std::endl;
}

Visualizer::~Visualizer() = default;

void Visualizer::initialize_viewer() {}
void Visualizer::update_point_cloud(const std::vector<Point>&) {}
void Visualizer::add_detections(const std::vector<Detection>&) {}
void Visualizer::clear_detections() {}
void Visualizer::spin_once(int) {}
void Visualizer::spin() {}
bool Visualizer::was_stopped() const { return false; }
void Visualizer::set_point_size(int) {}
void Visualizer::set_background_color(float, float, float) {}
void Visualizer::set_camera_position(float, float, float) {}
void Visualizer::show_coordinate_system(bool) {}
void Visualizer::color_by_intensity(bool) {}
void Visualizer::color_by_height(bool) {}
void Visualizer::save_screenshot(const std::string&) {}
void Visualizer::save_camera_parameters(const std::string&) {}
void Visualizer::load_camera_parameters(const std::string&) {}

#endif

// Terminal Visualizer implementation
TerminalVisualizer::TerminalVisualizer(int width, int height)
    : width_(width), height_(height)
{
    buffer_.resize(width_ * height_, ' ');
}

void TerminalVisualizer::render(
    const std::vector<Point>& points,
    const std::vector<Detection>& detections
) {
    clear();
    
    // Simple 2D projection
    for (const auto& p : points) {
        int screen_x, screen_y;
        project_point(p.x, p.y, p.z, screen_x, screen_y);
        
        if (screen_x >= 0 && screen_x < width_ && 
            screen_y >= 0 && screen_y < height_) {
            buffer_[screen_y * width_ + screen_x] = '.';
        }
    }
    
    // Draw detections
    for (const auto& det : detections) {
        int x, y;
        project_point(
            (det.bbox.min_x + det.bbox.max_x) / 2,
            (det.bbox.min_y + det.bbox.max_y) / 2,
            (det.bbox.min_z + det.bbox.max_z) / 2,
            x, y
        );
        
        if (x >= 0 && x < width_ && y >= 0 && y < height_) {
            buffer_[y * width_ + x] = 'X';
        }
    }
    
    // Print buffer
    std::cout << "\033[2J\033[H"; // Clear screen
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            std::cout << buffer_[y * width_ + x];
        }
        std::cout << '\n';
    }
    std::cout << std::flush;
}

void TerminalVisualizer::clear() {
    std::fill(buffer_.begin(), buffer_.end(), ' ');
}

void TerminalVisualizer::project_point(
    float x, float y, float z,
    int& screen_x, int& screen_y
) {
    // Simple orthographic projection
    float scale = 2.0f;
    screen_x = static_cast<int>(x * scale + width_ / 2);
    screen_y = static_cast<int>(-y * scale + height_ / 2);
}
