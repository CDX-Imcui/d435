#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <cmath>
#include <random>

struct PolarPoint {
    float r;
    float theta;
    float phi;

    inline PolarPoint() : r(0.f), theta(0.f), phi(0.f) {
    }

    inline PolarPoint(float _r, float _t, float _p) : r(_r), theta(_t), phi(_p) {
    }
};

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PolarPoint,
    (float, r, r)
    (float, theta, theta)
    (float, phi, phi)
)

int main() {
    // 读取 world.ply (极坐标点云)
    pcl::PointCloud<PolarPoint>::Ptr polar_cloud(new pcl::PointCloud<PolarPoint>());
    if (pcl::io::loadPLYFile("world.ply", *polar_cloud) < 0) {
        PCL_ERROR("无法读取 world.ply\n");
        return -1;
    }

    int extra_points = 500; // 红点数量

    // 输出点云 (xyzrgb)
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    xyz_cloud->width = polar_cloud->width + extra_points;
    xyz_cloud->height = polar_cloud->height;
    xyz_cloud->is_dense = false;
    xyz_cloud->points.resize(polar_cloud->size() + extra_points);

    // 填充极坐标转换结果
    for (size_t i = 0; i < polar_cloud->points.size(); ++i) {
        const auto &p = polar_cloud->points[i];
        pcl::PointXYZRGB q;
        q.x = p.r * std::sin(p.theta) * std::cos(p.phi);
        q.y = p.r * std::sin(p.theta) * std::sin(p.phi);
        q.z = p.r * std::cos(p.theta);
        q.r = 255;
        q.g = 255;
        q.b = 255; // 普通点设为白色
        xyz_cloud->points[i] = q;
    }

    // 一堆红点标记原点
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(-0.03f, 0.03f); // 0.03 单位范围
    for (int i = 0; i < extra_points; ++i) {
        pcl::PointXYZRGB red_point;
        red_point.x = dist(gen);
        red_point.y = dist(gen);
        red_point.z = dist(gen);
        red_point.r = 255;
        red_point.g = 0;
        red_point.b = 0;
        xyz_cloud->points[polar_cloud->points.size() + i] = red_point;
    }

    pcl::io::savePLYFileBinary("xyz.ply", *xyz_cloud);
    return 0;
}
