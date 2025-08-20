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



using namespace std;
using namespace cv;

//获取深度像素对应长度单位（米）的换算比例,返回比例因子
float get_depth_scale(rs2::device dev) {
    //遍历每一个传感器
    for (rs2::sensor &sensor: dev.query_sensors()) {
        // 是深度传感器
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>()) {
            return dpt.get_depth_scale();
        }
    }
    throw std::runtime_error("Device does not have a depth sensor");
}

// 根据已有的 streams 列表，选择一个对齐目标流
rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile> &streams) {
    // 优先对齐彩色流
    for (auto &sp: streams) {
        if (sp.stream_type() == RS2_STREAM_COLOR)
            return RS2_STREAM_COLOR;
    }
    // 如果没有彩色流，再尝试对齐红外流
    for (auto &sp: streams) {
        if (sp.stream_type() == RS2_STREAM_INFRARED)
            return RS2_STREAM_INFRARED;
    }
    // 默认返回深度流
    return RS2_STREAM_DEPTH;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr frame2PointCloud(const rs2::depth_frame &depth_frame,
                                                        const rs2::video_frame &color_frame,
                                                        const rs2_intrinsics &intrinsics) {
    int width = depth_frame.get_width();
    int height = depth_frame.get_height();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud->is_dense = false;

    const uint8_t *cdata = reinterpret_cast<const uint8_t *>(color_frame.get_data());
    int stride = color_frame.get_stride_in_bytes();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float z = depth_frame.get_distance(x, y);
            if (z <= 0.f || !std::isfinite(z)) continue;

            float pixel[2] = {float(x), float(y)};
            float point[3];
            rs2_deproject_pixel_to_point(point, &intrinsics, pixel, z);

            pcl::PointXYZRGB p;
            p.x = point[0];
            p.y = point[1];
            p.z = point[2];

            int idx = y * stride + x * 3;
            p.b = cdata[idx + 0];
            p.g = cdata[idx + 1];
            p.r = cdata[idx + 2];

            cloud->push_back(p);
        }
    }
    return cloud;
}

// 计算两帧位姿
Mat computeFramePose(const Mat &prev_color, const Mat &prev_depth,
                     const Mat &curr_color, const Mat &curr_depth,
                     const rs2_intrinsics &intrinsics, const float &depth_scale) {
    Ptr<ORB> orb = ORB::create(1000);
    vector<KeyPoint> prev_kpts, curr_kpts;
    Mat prev_desc, curr_desc;

    orb->detectAndCompute(prev_color, noArray(), prev_kpts, prev_desc);
    orb->detectAndCompute(curr_color, noArray(), curr_kpts, curr_desc);

    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(prev_desc, curr_desc, matches);

    vector<Point3f> obj_pts;
    vector<Point2f> img_pts;

    for (auto &m: matches) {
        int u = static_cast<int>(prev_kpts[m.queryIdx].pt.x);
        int v = static_cast<int>(prev_kpts[m.queryIdx].pt.y);
        uint16_t d = prev_depth.at<uint16_t>(v, u);
        if (d == 0) continue;
        float z = d * depth_scale; // mm->m
        float x = (u - intrinsics.ppx) * z / intrinsics.fx;
        float y = (v - intrinsics.ppy) * z / intrinsics.fy;
        obj_pts.push_back(Point3f(x, y, z));
        img_pts.push_back(curr_kpts[m.trainIdx].pt);
    }

    Mat rvec, tvec, inliers;
    Mat T = Mat::eye(4, 4,CV_64F);
    if (obj_pts.size() >= 6) {
        Mat K = (Mat_<double>(3, 3) << intrinsics.fx, 0, intrinsics.ppx, 0, intrinsics.fy, intrinsics.ppy, 0, 0, 1);
        solvePnPRansac(obj_pts, img_pts,
                       K,
                       noArray(), rvec, tvec, false, 100, 2.0, 0.99, inliers);
        Mat R;
        Rodrigues(rvec, R);
        R.copyTo(T(Rect(0, 0, 3, 3)));
        tvec.copyTo(T(Rect(3, 0, 1, 3)));
    }
    return T;
}

// 点云变换、融合、下采样
pcl::PointCloud<pcl::PointXYZRGB>::Ptr fuse(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &total_cloud,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &current_cloud,
    const Mat &T, float base_voxel_size = 0.005f) {
    // 应用位姿变换
    // 将 cv::Mat (4x4, double) 转成 Eigen::Matrix4f
    Eigen::Matrix4f eigenT;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            eigenT(i, j) = static_cast<float>(T.at<double>(i, j));
        }
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*current_cloud, *transformed, eigenT);

    if (!total_cloud) total_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    size_t prev_points = total_cloud->size();

    // 融合
    *total_cloud += *transformed;

    // // 动态下采样
    // float voxel_size = base_voxel_size;
    // if (prev_points > 0) {
    //     float factor = std::sqrt(static_cast<float>(total_cloud->size()) / prev_points);
    //     voxel_size *= factor;
    // }
    // pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    // vg.setInputCloud(total_cloud);
    // vg.setLeafSize(voxel_size, voxel_size, voxel_size);
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
    // vg.filter(*tmp);
    // total_cloud = tmp;

    return total_cloud;
}


int main() try {
    // 让 X11 支持多线程 在GUI PCL调用前
    XInitThreads();

    rs2::colorizer color_map; //声明彩色图
    rs2::pipeline pipe; //声明realsense管道
    rs2::config pipe_config; //数据流配置信息
    pipe_config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    pipe_config.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    //开始传送数据流
    rs2::pipeline_profile profile = pipe.start(pipe_config);

    //获取深度像素与长度单位的关系
    float depth_scale = get_depth_scale(profile.get_device());
    rs2_stream align_to = find_stream_to_align(profile.get_streams());
    rs2::align align(align_to); // 创建 align 对象
    // 若对齐到 COLOR，则使用 color 的 intrinsics（否则使用 depth 的）
    rs2_intrinsics intrinsics;
    if (align_to == RS2_STREAM_COLOR) {
        // 获取深度流内参
        rs2::video_stream_profile color_sp = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        intrinsics = color_sp.get_intrinsics();
    } else {
        rs2::video_stream_profile depth_sp = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
        intrinsics = depth_sp.get_intrinsics();
    }


    Mat prev_color, prev_depth;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr total_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    cv::Mat T_global = cv::Mat::eye(4, 4, CV_64F); // 全局位姿 单位矩阵
    for (int i = 1; waitKey(3000) != 'q' && i <= 2; i++) {
        // 每2秒获取一帧
        rs2::frameset data = pipe.wait_for_frames(); //等待下一帧
        rs2::frameset aligned_frames = align.process(data);
        cout << "capture " << i << " frame" << endl;

        // rs2::frame depth = aligned_frames.get_depth_frame().apply_filter(color_map); //获取深度图，加颜色滤镜
        rs2::depth_frame depth_frame = aligned_frames.get_depth_frame().as<rs2::depth_frame>();
        rs2::video_frame color_frame = aligned_frames.get_color_frame().as<rs2::video_frame>();

        Mat depth(Size(depth_frame.get_width(), depth_frame.get_height()),CV_16U, (void *) depth_frame.get_data(),
                  Mat::AUTO_STEP);
        Mat color(Size(color_frame.get_width(), color_frame.get_height()),CV_8UC3,
                  (void *) color_frame.get_data(),
                  Mat::AUTO_STEP);
        imwrite(std::string("../depth_") + std::to_string(i) + ".png", depth);
        imwrite(std::string("../color_") + std::to_string(i) + ".png", color);
        // imshow("depth", depth);
        // imshow("color", color);

        // PCL点云
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = frame2PointCloud(depth_frame, color_frame, intrinsics);

        if (!prev_color.empty()) {
            //T_rel prev -> curr
            Mat T_rel = computeFramePose(prev_color, prev_depth, color, depth, intrinsics, depth_scale);

            cv::Mat T_rel_inv = T_rel.inv(); // 逆矩阵 curr -> prev

            T_global = T_global * T_rel_inv; // 累计全局位姿: curr -> world
            total_cloud = fuse(total_cloud, cloud, T_global, 0.005f);
        } else {
            total_cloud = cloud;
        }
        prev_color = color.clone();
        prev_depth = depth.clone();
    }
    pipe.stop(); // 停止管道（单帧模式）
    // 可视化（PCLVisualizer 在主线程 + XInitThreads 已调用）
    // pcl::visualization::PCLVisualizer viewer("Point Cloud");
    // viewer.addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");
    // viewer.spin(); // 阻塞，直到窗口关闭
    pcl::io::savePLYFileBinary("../total_cloud.ply", *total_cloud);

    return EXIT_SUCCESS;
} catch (const rs2::error &e) {
    std::cout << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n"
            << e.what() << endl;
    return EXIT_FAILURE;
} catch (const std::exception &e) {
    std::cout << e.what() << endl;
    return EXIT_FAILURE;
}
