#include <X11/Xlib.h>              // 必须最先包含以便调用 XInitThreads()
#ifdef Success
#undef Success                    // 防止 X11 宏 Success 与 Eigen 枚举冲突
#endif

#include <iostream>
#include<stdlib.h>
#include<stdio.h>
#include<string>
#include <fstream>
#include <algorithm>
#include <sstream>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<librealsense2/rs.hpp>
#include<librealsense2/rsutil.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/ply_io.h>

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


    // while (waitKey(1)) {
    rs2::frameset data = pipe.wait_for_frames(); //等待下一帧
    rs2::frameset aligned_frames = align.process(data);

    // rs2::frame depth = aligned_frames.get_depth_frame().apply_filter(color_map); //获取深度图，加颜色滤镜
    rs2::depth_frame depth_frame = aligned_frames.get_depth_frame().as<rs2::depth_frame>();
    rs2::video_frame color_frame = aligned_frames.get_color_frame().as<rs2::video_frame>();

    const int depth_w = depth_frame.get_width();
    const int depth_h = depth_frame.get_height();

    Mat depth_vis(Size(depth_w, depth_h),CV_16U, (void *) depth_frame.get_data(), Mat::AUTO_STEP);
    Mat color_image(Size(color_frame.get_width(), color_frame.get_height()),CV_8UC3,
                    (void *) color_frame.get_data(),
                    Mat::AUTO_STEP);
    imwrite("../depth_color.png", depth_vis);
    imwrite("../color_image.png", color_image);
    // imshow("depth_color", depth_vis);
    // imshow("color_image", color_image);

    // PCL点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud->is_dense = false;

    for (int y = 0; y < depth_h; y++) {
        for (int x = 0; x < depth_w; x++) {
            float z = depth_frame.get_distance(x, y); // 单位米 SDK已处理depth_scale
            if (z <= 0.f || !std::isfinite(z)) continue;
            // uint16_t d = depth_frame.at<uint16_t>(y, x);
            // if (d == 0) continue; // 无效深度
            // float depth_m = d * depth_scale;

            // 反投影到相机坐标系
            float pixel[2] = {static_cast<float>(x), static_cast<float>(y)};
            float point[3];
            rs2_deproject_pixel_to_point(point, &intrinsics, pixel, z);

            pcl::PointXYZRGB p;
            p.x = point[0];
            p.y = point[1];
            p.z = point[2];
            // color_frame可能有stride，使用get_stride_in_bytes()
            const uint8_t *cdata = reinterpret_cast<const uint8_t *>(color_frame.get_data());
            int stride = color_frame.get_stride_in_bytes(); //stride表示图像一行占用的字节数
            int idx = y * stride + x * 3; // 每个像素占3个字节BGR
            p.b = cdata[idx + 0];
            p.g = cdata[idx + 1];
            p.r = cdata[idx + 2];
            cloud->push_back(p);
        }
    }

    pipe.stop(); // 停止管道（单帧模式）

    // 可视化（PCLVisualizer 在主线程 + XInitThreads 已调用）
    // pcl::visualization::PCLVisualizer viewer("Point Cloud");
    // viewer.addPointCloud<pcl::PointXYZRGB>(cloud, "cloud");
    // viewer.spin(); // 阻塞，直到窗口关闭
    pcl::io::savePLYFileBinary("../cloud.ply", *cloud);

    //     if (cv::waitKey(1) == 'q') {
    //         break;
    //     }
    // }
    return EXIT_SUCCESS;
} catch (const rs2::error &e) {
    std::cout << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n"
            << e.what() << endl;
    return EXIT_FAILURE;
} catch (const std::exception &e) {
    std::cout << e.what() << endl;
    return EXIT_FAILURE;
}
