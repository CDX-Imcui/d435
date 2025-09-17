#ifndef D435_MULTI_RGBD_H
#define D435_MULTI_RGBD_H
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <cmath>
#include <future>
#include "camera_extrinsic.h"
#include "rs2_D435.h"// rs2_D435相机类
#include "thread_pool.h"
#include "PolarPoint.h"

class multi_RGBD {
public: //TODO 暂时写死 线程池大小
    explicit multi_RGBD(const std::size_t cameras_size = 6) : pool_(cameras_size) {
        rs2::context ctx;
        auto devices = ctx.query_devices();
        if (devices.size() == 0) {
            throw std::runtime_error("没设备");
        } //devices[0].get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);devices[1] 获取设备序列号
        width = 640;
        height = 480;
        world_cloud = pcl::PointCloud<PolarPoint>::Ptr(new pcl::PointCloud<PolarPoint>);
        world_cloud->reserve(width * height * cameras_size); //分配物理内存，points.size()是0
    };

    ~multi_RGBD() = default;

    void addCamera(const camera_extrinsic &extrinsic) {
        // 不会在vector内部直接构造 rs2_D435 对象，性能提升不大
        cameras.emplace_back(std::make_shared<rs2_D435>(extrinsic, false));
        // 如果是std::vector<rs2_D435>，存对象本身，emplace_back才能避免一次拷贝或移动
        this->extrinsic = extrinsic;
        if (extrinsic.width > width || extrinsic.height > height) {
            width = extrinsic.width;
            height = extrinsic.height;
            world_cloud->reserve(width * height * cameras.size());
        }
    }

    pcl::PointCloud<PolarPoint>::Ptr getPointCloud(float base_voxel_size = 0.005f) {
        const size_t C = cameras.size();
        if (C == 0) return world_cloud;

        // 计算每台相机点数与 prefix offsets
        std::vector<size_t> offsets(C + 1, 0);
        for (size_t i = 0; i < C; ++i)
            offsets[i + 1] = offsets[i] + static_cast<size_t>(extrinsic.width) * static_cast<size_t>(extrinsic.height);

        const size_t total = offsets[C];

        // 预分配并设置有序/无序属性（拼接成 1 行）
        world_cloud->is_dense = false;
        world_cloud->width  = static_cast<uint32_t>(total);
        world_cloud->height = 1;
        world_cloud->resize(total); // 必须用 resize，使 points.size()==total
        world_cloud->is_dense = false;
        futures.clear();
        futures.reserve(C);
        PolarPoint* base = world_cloud->points.data();
        for (int i = 0; i < C; ++i) {
            PolarPoint* dst = base + offsets[i];
            futures.emplace_back(
                pool_.enqueue([this, i,dst]() {
                    cameras[i]->getPolarPointCloudInto(dst);
                })
            );
        }
        for (auto &fut: futures) {
            fut.get(); //会阻塞直到对应任务完成，并返回结果
        }

        return world_cloud;
    }

    // // 融合、下采样
    // void fuse(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &_world,
    //           const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &current,
    //           float base_voxel_size = 0.005f) {
    //
    //     size_t prev_points = _world->size();
    //     // 动态下采样
    //     float voxel_size = base_voxel_size;
    //     if (prev_points > 0) {
    //         float factor = std::sqrt(static_cast<float>(total_cloud->size()) / prev_points);
    //         voxel_size *= factor;
    //     }
    //     pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    //     vg.setInputCloud(total_cloud);
    //     vg.setLeafSize(voxel_size, voxel_size, voxel_size);
    //     pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
    //     vg.filter(*tmp);
    //     total_cloud = tmp;
    // }

private:
    std::vector<std::shared_ptr<rs2_D435> > cameras;
    pcl::PointCloud<PolarPoint>::Ptr world_cloud;
    ThreadPool pool_; // 线程池
    camera_extrinsic extrinsic;
    std::vector<std::future<void>> futures;
    int width, height;

    static void savePictures(const std::string &serial, const rs2::depth_frame &depth, const rs2::video_frame &color) {
        cv::Mat img_depth1(cv::Size(depth.get_width(), depth.get_height()),CV_16U, (void *) depth.get_data(),
                           cv::Mat::AUTO_STEP);
        cv::Mat img_color1(cv::Size(color.get_width(), color.get_height()),CV_8UC3, (void *) color.get_data(),
                           cv::Mat::AUTO_STEP);
        cv::imwrite(std::string("../") + serial + "_depth.png", img_depth1);
        cv::imwrite(std::string("../") + serial + "_color.png", img_color1);
    }
};
#endif //D435_MULTI_RGBD_H
