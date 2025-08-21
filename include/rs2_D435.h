//
// Created by YSC on 2025/8/20.
//

#ifndef D435_RS2_D435_H
#define D435_RS2_D435_H
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

class rs2_D435 {
public:
    explicit rs2_D435(const std::string &serial, int width = 640, int height = 480, int fps = 30) {
        cfg.enable_device(serial);
        cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
        cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
        //开始传送数据流
        profile = pipe.start(cfg);
        depth_scale = get_depth_scale(profile.get_device()); //获取深度像素与长度单位的关系
        align_to = find_stream_to_align(profile.get_streams());
        align = new rs2::align(align_to);

        if (align_to == RS2_STREAM_COLOR) {
            rs2::video_stream_profile color_sp = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
            intrinsics = color_sp.get_intrinsics();
        } else {
            rs2::video_stream_profile depth_sp = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
            intrinsics = depth_sp.get_intrinsics();
        }
    }

    std::pair<rs2::depth_frame, rs2::video_frame> get_frame() {
        // rs2::colorizer color_map; //声明彩色图
        // rs2::frame depth = aligned_frames.get_depth_frame().apply_filter(color_map); //获取深度图，加颜色滤镜

        rs2::frameset frames = pipe.wait_for_frames(); // 获取一帧
        rs2::frameset aligned_frames = align->process(frames); // 对齐到目标流
        rs2::depth_frame depth = aligned_frames.get_depth_frame().as<rs2::depth_frame>();
        rs2::video_frame color = aligned_frames.get_color_frame().as<rs2::video_frame>();
        return {depth, color};
    }

    float getDepthScale() const {
        return depth_scale;
    }

    ~rs2_D435() {
        pipe.stop(); // 停止管道
        delete align; // 删除对齐对象
    }

    //获取深度像素对应长度单位（米）的换算比例,返回比例因子
    static float get_depth_scale(const rs2::device &dev) {
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
    static rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile> &streams) {
        // 优先对齐彩色流
        for (auto &sp: streams) {
            if (sp.stream_type() == RS2_STREAM_COLOR)
                return RS2_STREAM_COLOR;
        }
        // // 尝试对齐红外流
        // for (auto &sp: streams) {
        //     if (sp.stream_type() == RS2_STREAM_INFRARED)
        //         return RS2_STREAM_INFRARED;
        // }
        return RS2_STREAM_DEPTH; // 默认返回深度流
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr frame2PointCloud(const rs2::depth_frame &depth_frame,
                                                            const rs2::video_frame &color_frame) {
        int width = depth_frame.get_width();
        int height = depth_frame.get_height();
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud->is_dense = false;//表明点云中可能包含无效点（如 NaN 或 Inf）

        const uint8_t *cdata = reinterpret_cast<const uint8_t *>(color_frame.get_data());
        int stride = color_frame.get_stride_in_bytes();//图一行的字节数；line width in memory in bytes (not the logical image width)

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float z = depth_frame.get_distance(x, y);//返回单位是米
                if (z <= 0.f || !std::isfinite(z)) continue;

                float pixel[2] = {float(x), float(y)};
                float point[3];
                //可以把rs2_deproject_pixel_to_point换成手写的SIMD向量化循环，或者用 OpenMP/TBB 对双重循环并行
                rs2_deproject_pixel_to_point(point, &intrinsics, pixel, z);

                pcl::PointXYZRGB p;
                p.x = point[0];
                p.y = -point[1];
                p.z = -point[2];

                int idx = y * stride + x * 3;//每个像素占3个字节BGR
                p.b = cdata[idx + 0];
                p.g = cdata[idx + 1];
                p.r = cdata[idx + 2];

                cloud->push_back(p);
            }
        }
        return cloud;
    }

    // 点云从相机坐标系映射到世界坐标系变换、融合、下采样
    static void fuse(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &total_cloud,
                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr &current_cloud,
                     const Eigen::Matrix4f &T, //从相机坐标系映射到世界坐标系的变换矩阵
                     float base_voxel_size = 0.005f) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGB>);
        //transformPointCloud有一个copy_all_fields字段默认true，控制是否复制除x、y、z以外的其他字段（如颜色、强度等）到输出点云
        pcl::transformPointCloud(*current_cloud, *transformed, T);

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
    }

    void print_intrinsics() {
        cout << "Intrinsics:" << endl;
        cout << "Width: " << intrinsics.width << endl;
        cout << "Height: " << intrinsics.height << endl;
        cout << "PPX: " << intrinsics.ppx << endl;
        cout << "PPY: " << intrinsics.ppy << endl;
        cout << "FX: " << intrinsics.fx << endl;
        cout << "FY: " << intrinsics.fy << endl;
        cout << "Model: " << intrinsics.model << endl;
    }

private:
    rs2::pipeline pipe; //声明realsense管道
    rs2::config cfg; //数据流配置信息
    rs2::pipeline_profile profile; //管道配置文件
    float depth_scale; //深度像素与长度单位（米）的关系  0.001
    rs2_stream align_to; // 对齐到的目标流
    rs2::align *align = nullptr; // 先用默认参数初始化 align 对象
    rs2_intrinsics intrinsics; // 内参
};


#endif //D435_RS2_D435_H
