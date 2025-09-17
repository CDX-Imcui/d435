#include <iostream>
#include <string>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <thread>
#include <chrono>
#include <X11/Xlib.h>
#include "camera_extrinsic.h"
#include "multi_RGBD.h"
#include "utils.h"

int main() try {
    XInitThreads(); // 让 X11 支持多线程
    // devices();
    Eigen::Matrix4f T1 = Eigen::Matrix4f::Identity(); //四元数(w, x, y, z)表示任意旋转。融合点云时需要将点云旋转，与相机回正方向相反
    // 旋转轴（ux，uy，uz）是单位向量（0，0，1）右手法则 45° x=ux⋅sin(θ/2),y=uy⋅sin(θ/2),z=uz⋅sin(θ/2),w=cos(θ/2)。
    T1.block<3, 3>(0, 0) = Eigen::Quaternionf(0.92388, 0.0f, 0.0f, 0.38268f).toRotationMatrix();
    T1.block<3, 1>(0, 3) = Eigen::Vector3f(-0.05, -0.05, 0); // 平移向量t
    Eigen::Matrix4f T2 = Eigen::Matrix4f::Identity();
    T2.block<3, 3>(0, 0) = Eigen::Quaternionf(0.92388, 0.0f, 0.0f, -0.38268f).toRotationMatrix(); //45°
    T2.block<3, 1>(0, 3) = Eigen::Vector3f(-0.05, 0.05, 0);
    // 外参
    camera_extrinsic extrinsic1("239722073505", 640, 480, 90, T1),
            extrinsic2("239722072145", 640, 480, 90, T2);

    multi_RGBD multi_rgbd;
    multi_rgbd.addCamera(extrinsic1);
    // multi_rgbd.addCamera(extrinsic2);
    pcl::PointCloud<PolarPoint>::Ptr world_cloud;

    std::this_thread::sleep_for(std::chrono::seconds(1)); //等待硬件初始化
    auto start = std::chrono::high_resolution_clock::now();

    int count = 100;
    for (int i = 1; i <= count; ++i) {
        // std::cout << "\r采集 " << i << " 次" << std::flush;
        world_cloud = multi_rgbd.getPointCloud(0.005f);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "耗时: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / static_cast<
                double>(count) << " ms" <<
            std::endl;

    // pcl::visualization::PCLVisualizer viewer("Point Cloud");
    // viewer.addPointCloud<pcl::PointXYZRGB>(world_cloud, "cloud");
    // viewer.spin(); // 阻塞，直到窗口关闭

    pcl::io::savePLYFileBinary("../world.ply", *world_cloud);
    std::cout << "points.size(): " << world_cloud->points.size() << std::endl;

    // pcl::PCLPointCloud2 cloud2;
    // pcl::toPCLPointCloud2(*world_cloud, cloud2);
    // std::cout << "Fields in cloud: " << std::endl;
    // for (const auto& field : cloud2.fields) {
    //     std::cout << "  " << field.name << " ("
    //               << pcl::getFieldType(field.datatype) << ")" << std::endl;
    // }
    return 0;

    return EXIT_SUCCESS;
} catch (const rs2::error &e) {
    std::cout << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n"
            << e.what() << endl;
    return EXIT_FAILURE;
} catch (const std::exception &e) {
    std::cout << e.what() << endl;
    return EXIT_FAILURE;
}
