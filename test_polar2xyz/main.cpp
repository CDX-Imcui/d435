#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <cmath>

struct PolarPoint {
    float r;
    float theta;
    float phi;
    inline PolarPoint() : r(0.f), theta(0.f), phi(0.f) {}
    inline PolarPoint(float _r, float _t, float _p) : r(_r), theta(_t), phi(_p) {}
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

    // 输出点云 (xyz)
    pcl::PointCloud<pcl::PointXYZ>::Ptr xyz_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    xyz_cloud->width = polar_cloud->width;
    xyz_cloud->height = polar_cloud->height;
    xyz_cloud->is_dense = false;
    xyz_cloud->points.resize(polar_cloud->size());

    for (size_t i = 0; i < polar_cloud->points.size(); ++i) {
        const auto &p = polar_cloud->points[i];
        pcl::PointXYZ q;
        q.x = p.r * std::sin(p.theta) * std::cos(p.phi);
        q.y = p.r * std::sin(p.theta) * std::sin(p.phi);
        q.z = p.r * std::cos(p.theta);
        xyz_cloud->points[i] = q;
    }

    pcl::io::savePLYFileBinary("xyz.ply", *xyz_cloud);
    return 0;
}
