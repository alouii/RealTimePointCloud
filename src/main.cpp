#include <iostream>
#include <vector>
#include "PointCloudProcessor.h"
#include "IMUProcessor.h"

// OpenCV for USB camera
#include <opencv2/opencv.hpp>

int main() {
    std::cout << "RealTimePointCloudNN started!" << std::endl;

    cv::VideoCapture cap(0); // USB camera 0
    if (!cap.isOpened()) {
        std::cerr << "Cannot open USB camera" << std::endl;
        return -1;
    }

    PointCloudProcessor pc;
    IMUProcessor imu;

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Convert frame to depth-like grayscale (dummy)
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<float> depthData(gray.rows * gray.cols);
        for (int y = 0; y < gray.rows; ++y)
            for (int x = 0; x < gray.cols; ++x)
                depthData[y * gray.cols + x] = gray.at<uchar>(y, x) / 255.0f;

        auto cloud = pc.depthToPointCloud(depthData, gray.cols, gray.rows);
        pc.processPointCloudCUDA(cloud);

        std::cout << "Processed cloud: " << cloud->points.size() << " points" << std::endl;

        // Show camera for visualization
        cv::imshow("USB Camera", frame);
        if (cv::waitKey(1) == 27) break; // ESC to quit
    }

    std::cout << "Pipeline finished." << std::endl;
    return 0;
}
