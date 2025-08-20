#include <X11/Xlib.h>              // 必须最先包含以便调用 XInitThreads()
#ifdef Success
#undef Success                    // 防止 X11 宏 Success 与 Eigen 枚举冲突
#endif

#include <iostream>
#include<stdlib.h>
#include <string>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>

#include "rs2_D435.h" // 自定义re2_D435相机类

using namespace std;
using namespace cv;


void savePictures(const std::string &serial, rs2::depth_frame depth, rs2::video_frame color) {
    Mat img_depth1(Size(depth.get_width(), depth.get_height()),CV_16U, (void *) depth.get_data(), Mat::AUTO_STEP);
    Mat img_color1(Size(color.get_width(), color.get_height()),CV_8UC3, (void *) color.get_data(), Mat::AUTO_STEP);
    imwrite(std::string("../") + serial + "_depth.png", img_depth1);
    imwrite(std::string("../") + serial + "_color.png", img_color1);
}


int main() try {
    // 让 X11 支持多线程 在GUI PCL调用前
    XInitThreads();

    rs2::context ctx;
    auto devices = ctx.query_devices();
    if (devices.size() == 0) {
        throw std::runtime_error("没设备");
    }
    if (devices.size() == 1) {
        throw std::runtime_error("只有1个设备");
    }

    // 获取序列号
    std::string serial1 = devices[0].get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
    std::string serial2 = devices[1].get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
    cout << "Camera 1 Serial: " << serial1 << endl;
    cout << "Camera 2 Serial: " << serial2 << endl;

    rs2_D435 camera1(serial1, 640, 480, 30); //1280x720
    Eigen::Matrix4f T1 = Eigen::Matrix4f::Identity();
    // 旋转轴（ux，uy，uz）是单位向量（0，-1，0）右手法则-45° x=ux⋅sin(θ/2),y=uy⋅sin(θ/2),z=uz⋅sin(θ/2),w=cos(θ/2)。四元数(w, x, y, z)表示任意旋转
    T1.block<3,3>(0,0) = Eigen::Quaternionf(0.92388, 0.0f, 0.38268f, 0.0f).toRotationMatrix();
    T1.block<3,1>(0,3) = Eigen::Vector3f(-0.1, 0, 0.1);//// 平移向量t（px, py, pz：相机原点在世界坐标系下的位置）

    rs2_D435 camera2(serial2, 640, 480, 30);
    Eigen::Matrix4f T2 = Eigen::Matrix4f::Identity();
    T2.block<3,3>(0,0) = Eigen::Quaternionf(0.92388, 0.0f, -0.38268f, 0.0f).toRotationMatrix();//45°
    T2.block<3,1>(0,3) = Eigen::Vector3f(0.1, 0, 0.1);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr world_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    auto pair1 = camera1.get_frame();
    rs2::depth_frame depth1 = pair1.first;
    rs2::video_frame color1 = pair1.second;
    savePictures(serial1, depth1, color1);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1 = camera1.frame2PointCloud(depth1, color1);

    auto pair2 = camera2.get_frame();
    rs2::depth_frame depth2 = pair2.first;
    rs2::video_frame color2 = pair2.second;
    savePictures(serial2, depth2, color2);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2 = camera2.frame2PointCloud(depth2, color2);

    rs2_D435::fuse(world_cloud, cloud1, T1, 0.005f);
    rs2_D435::fuse(world_cloud, cloud2, T2, 0.005f);

    pcl::io::savePLYFileBinary("../total_cloud.ply", *world_cloud);

    return EXIT_SUCCESS;
} catch (const rs2::error &e) {
    std::cout << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n"
            << e.what() << endl;
    return EXIT_FAILURE;
} catch (const std::exception &e) {
    std::cout << e.what() << endl;
    return EXIT_FAILURE;
}
