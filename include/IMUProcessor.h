#pragma once
#include <vector>

class IMUProcessor {
public:
    IMUProcessor() = default;

    // Dummy rotation matrix
    std::vector<std::vector<float>> getRotationMatrix() const {
        return std::vector<std::vector<float>>(3, std::vector<float>(3, 0.0f));
    }
};
