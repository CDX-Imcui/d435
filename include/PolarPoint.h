#ifndef D435_POLARPOINT_H
#define D435_POLARPOINT_H
#include <pcl/register_point_struct.h> // 注册自定义点类型用
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
struct PolarPoint {
    float r; // 半径
    float theta; // 极角
    float phi; // 方位角
    inline PolarPoint() : r(0.f), theta(0.f), phi(0.f) {
    }

    inline PolarPoint(float _r, float _t, float _p) : r(_r), theta(_t), phi(_p) {
    }
}; // 定义并注册 PolarPoint 为 PCL 点类型

POINT_CLOUD_REGISTER_POINT_STRUCT(PolarPoint, (float, r, r) (float, theta, theta) (float, phi, phi))

#endif //D435_POLARPOINT_H