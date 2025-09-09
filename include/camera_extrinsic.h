#ifndef D435_CAMERA_INFO_H
#define D435_CAMERA_INFO_H
#include <string>
#include <Eigen/Core>

class camera_extrinsic {
public:
    camera_extrinsic() {
    }

    camera_extrinsic(std::string serial, int width, int height, int fps, const Eigen::Matrix4f &T)
        : serial(std::move(serial)), width(width), height(height), fps(fps), T(T) {
    }

    ~camera_extrinsic() = default;

    std::string serial;
    int width;
    int height;
    int fps;
    Eigen::Matrix4f T; //从相机坐标系映射到世界坐标系的变换矩阵
};

#endif //D435_CAMERA_INFO_H
