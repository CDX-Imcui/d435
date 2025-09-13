#ifndef D435_RS2_D435_H
#define D435_RS2_D435_H
#include <iostream>
#include <string>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <omp.h>
#include  "camera_extrinsic.h"

class rs2_D435 {
public:
    camera_extrinsic extrinsic;

    explicit rs2_D435(const camera_extrinsic &camera_info, bool haveColor = false) : extrinsic(camera_info) {
        cfg.enable_device(camera_info.serial);
        cfg.enable_stream(RS2_STREAM_DEPTH, camera_info.width, camera_info.height, RS2_FORMAT_Z16, camera_info.fps);
        if (haveColor)
            cfg.enable_stream(RS2_STREAM_COLOR, camera_info.width, camera_info.height, RS2_FORMAT_BGR8,
                              camera_info.fps);
        //开始传送数据流
        profile = pipe.start(cfg);
        depth_scale = get_depth_scale(profile.get_device()); //获取深度像素与长度单位的关系
        if (haveColor) {
            align_to = find_stream_to_align(profile.get_streams());
            align = new rs2::align(align_to);
            auto color_sp = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
            intrinsics = color_sp.get_intrinsics();
            cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
        } else {
            auto depth_sp = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
            intrinsics = depth_sp.get_intrinsics();
            cloudXYZ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
            cloudXYZ->width = camera_info.width;
            cloudXYZ->height = camera_info.height; //设置了 width height PCL 会把点云视为一个有序点云(相机图像结构)
            cloudXYZ->resize(camera_info.width * camera_info.height); //points.size() 改成 n，并构造n个默认点
            cloudXYZ->is_dense = false; //表明点云中可能包含无效点（如 NaN 或 Inf）
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr getPointXYZCloud() {
        frame = pipe.wait_for_frames(); // 获取一帧
        auto depth_frame = frame.get_depth_frame().as<rs2::depth_frame>();

        int width = depth_frame.get_width();
        int height = depth_frame.get_height();
        auto *xyz = cloudXYZ->points.data();
        const float nanv = std::numeric_limits<float>::quiet_NaN();

        for (int y = 0; y < height; y++) {
            //行遍历——提高缓存命中率
            for (int x = 0; x < width; x++) {
                float z = depth_frame.get_distance(x, y); //返回单位是米
                pcl::PointXYZ p; //需要 x朝前(z)，y朝左(-x)，z朝上(-y)

                if (z <= 0.f || !std::isfinite(z)) {
                    p.x = p.y = p.z = nanv; // 明确标记无效点
                } else {
                    float pixel[2] = {float(x), float(y)};
                    float point[3]; //point[0]：x  point[1]：y point[2]：z
                    rs2_deproject_pixel_to_point(point, &intrinsics, pixel, z); //像素坐标转相机坐标（x朝右，y朝下，z朝前）
                    //纠正朝向——相机坐标系转换到机器人坐标系
                    p.x = point[2];
                    p.y = -point[0];
                    p.z = -point[1];
                }
                xyz[y * width + x] = p; //pushback伪共享，用索引赋值，必须用 resize，否则越界
            }
        }
        return cloudXYZ; //机器人坐标系下的点云
    }


    std::pair<rs2::depth_frame, rs2::video_frame> getFrame() {
        // rs2::colorizer color_map;
        // rs2::frame depth = aligned_frames.get_depth_frame().apply_filter(color_map); //对深度图加颜色滤镜

        rs2::frameset frames = pipe.wait_for_frames(); // 获取一帧
        rs2::frameset aligned_frames = align->process(frames); // 对齐到目标流
        auto depth = aligned_frames.get_depth_frame().as<rs2::depth_frame>();
        auto color = aligned_frames.get_color_frame().as<rs2::video_frame>();
        return {depth, color};
    }


    //获取深度像素对应长度单位（米）的换算比例,返回比例因子
    static float get_depth_scale(const rs2::device &dev) {
        //遍历每一个传感器
        for (rs2::sensor &sensor: dev.query_sensors()) {
            // 是深度传感器
            if (auto dpt = sensor.as<rs2::depth_sensor>()) {
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
        cloud->clear(); //点数归零，但容量 (capacity) 不变
        cloud->width = width;
        cloud->height = height; //设置了 width height PCL 会把点云视为一个有序点云(相机图像结构)
        cloud->resize(width * height); //points.size() 改成 n，并构造n个默认点
        cloud->is_dense = false; //表明点云中可能包含无效点（如 NaN 或 Inf）

        const auto *cdata = reinterpret_cast<const uint8_t *>(color_frame.get_data());
        int stride = color_frame.get_stride_in_bytes();
        //图一行的字节数；line width in memory in bytes (not the logical image width)
        // auto start = std::chrono::high_resolution_clock::now();
        // #pragma omp parallel for
        for (int y = 0; y < height; y++) {
            //行遍历——提高缓存命中率
            for (int x = 0; x < width; x++) {
                float z = depth_frame.get_distance(x, y); //返回单位是米
                if (z <= 0.f || !std::isfinite(z)) continue;

                float pixel[2] = {float(x), float(y)};
                float point[3]; //point[0]：x  point[1]：y point[2]：z
                rs2_deproject_pixel_to_point(point, &intrinsics, pixel, z); //像素坐标转相机坐标（x朝右，y朝下，z朝前）

                pcl::PointXYZRGB p; //需要 x朝前(z)，y朝左(-x)，z朝上(-y)
                //纠正朝向——相机坐标系转换到机器人坐标系
                p.x = point[2];
                p.y = -point[0];
                p.z = -point[1];

                int idx = y * stride + x * 3; //每个像素占3个字节BGR
                p.b = cdata[idx + 0];
                p.g = cdata[idx + 1];
                p.r = cdata[idx + 2];

                // cloud->push_back(p);
                cloud->points[y * width + x] = p; //索引赋值，必须用 resize，否则越界
            }
        }
        // auto end= std::chrono::high_resolution_clock::now();
        // std::cout << "openmp 耗时: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).
        //         count() <<
        //         " ms" << std::endl;
        return cloud; //机器人坐标系下的点云
    }


    // 点云从相机坐标系映射到世界坐标系变换、融合
    static void fuse(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_world_cloud,
                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr &current_cloud,
                     const Eigen::Matrix4f &T, //变换矩阵
                     float base_voxel_size = 0.005f) {
        //transformPointCloud有一个copy_all_fields字段默认true，控制是否复制除x、y、z以外的其他字段（如颜色、强度等）到输出点云
        pcl::transformPointCloud(*current_cloud, *current_cloud, T, false);
        _world_cloud->insert(_world_cloud->end(), current_cloud->begin(), current_cloud->end());
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

    float getDepthScale() const {
        return depth_scale;
    }

    ~rs2_D435() {
        pipe.stop(); // 停止管道
        delete align; // 删除对齐对象
    }

private:
    rs2::pipeline pipe; //声明realsense管道
    rs2::config cfg; //数据流配置信息
    rs2::pipeline_profile profile; //管道配置文件
    float depth_scale; //深度像素与长度单位（米）的关系  0.001
    rs2_stream align_to; // 对齐到的目标流
    rs2::align *align = nullptr; // 先用默认参数初始化 align 对象
    rs2_intrinsics intrinsics; // 内参

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudXYZ;
    rs2::frameset frame;
};


#endif //D435_RS2_D435_H
