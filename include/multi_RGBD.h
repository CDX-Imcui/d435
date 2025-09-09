#ifndef D435_MULTI_RGBD_H
#define D435_MULTI_RGBD_H
#include <vector>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <thread>
#include <pcl/common/transforms.h>
#include "camera_extrinsic.h"
#include "rs2_D435.h"// rs2_D435相机类



class multi_RGBD {
public:

    multi_RGBD() {
        world_cloud= pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
        world_cloud->is_dense = false;
    };
    ~multi_RGBD(){};
    void addCamera(const camera_extrinsic &extrinsic) {
        // 不会在vector内部直接构造 rs2_D435 对象，性能提升不大
        cameras.emplace_back(std::make_shared<rs2_D435>(extrinsic));// 如果是std::vector<rs2_D435>，存对象本身，emplace_back才能避免一次拷贝或移动
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr getPointCloud(float base_voxel_size = 0.005f) {
        world_cloud->clear();
        for (int i = 0; i < cameras.size(); ++i) {
            auto pair = cameras[i]->getFrame();
            std::thread([serial = cameras[i]->extrinsic.serial, depth_frame = pair.first, video_frame = pair.second]() {
                savePictures(serial, depth_frame, video_frame);//保存图片 TODO后期注释掉
            }).detach();//pair.first是 rs2::frame 类型，不是Copyable的普通对象，是RealSenseSDK管理的资源句柄.按值拷贝进lambda才不会出现悬空问题
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = cameras[i]->frame2PointCloud(pair.first, pair.second);
            fuse(world_cloud, cloud, cameras[i]->extrinsic.T, base_voxel_size);
        }
        return world_cloud;
    }
    // 点云从相机坐标系映射到世界坐标系变换、融合、下采样
    void fuse(pcl::PointCloud<pcl::PointXYZRGB>::Ptr total_cloud,
                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr &current_cloud,
                     const Eigen::Matrix4f &T, //变换矩阵
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
private:
    std::vector<std::shared_ptr<rs2_D435>> cameras;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr world_cloud;

    static void savePictures(const std::string &serial,const rs2::depth_frame& depth,const rs2::video_frame& color) {
        cv::Mat img_depth1(cv::Size(depth.get_width(), depth.get_height()),CV_16U, (void *) depth.get_data(),
                           cv::Mat::AUTO_STEP);
        cv::Mat img_color1(cv::Size(color.get_width(), color.get_height()),CV_8UC3, (void *) color.get_data(),
                           cv::Mat::AUTO_STEP);
        cv::imwrite(std::string("../") + serial + "_depth.png", img_depth1);
        cv::imwrite(std::string("../") + serial + "_color.png", img_color1);
    }
};
#endif //D435_MULTI_RGBD_H