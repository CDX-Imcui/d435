#ifndef D435_MULTI_RGBD_H
#define D435_MULTI_RGBD_H
#include <vector>
#include <pcl/point_cloud.h>
#include <cmath>
#include <future>
#include "camera_extrinsic.h"
#include "rs2_D435.h"
#include "thread_pool.h"
#include "PolarPoint.h"

class multi_RGBD {
public:
    explicit multi_RGBD(const std::size_t cameras_size = 3) : pool_(cameras_size) {
        rs2::context ctx;
        auto devices = ctx.query_devices();
        if (devices.size() == 0) {
            throw std::runtime_error("没设备");
        } //devices[0].get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);devices[1] 获取设备序列号
        world_cloud = pcl::PointCloud<PolarPoint>::Ptr(new pcl::PointCloud<PolarPoint>);
        world_cloud->is_dense = false;
    };

    void addCamera(const camera_extrinsic &ext) {
        // 如果是std::vector<rs2_D435>，存对象本身，emplace_back才能避免一次拷贝或移动
        cameras.emplace_back(std::make_shared<rs2_D435>(ext, false));
        C = cameras.size();
        offsets.push_back(offsets.back() + static_cast<size_t>(ext.width) * static_cast<size_t>(ext.height));
        world_cloud->resize(offsets.back());
        execute.push_back(0);
    }

    pcl::PointCloud<PolarPoint>::Ptr getPointCloud() {
        if (C == 0) return world_cloud;
        // 开始前清零标志位
        std::fill_n(execute.begin(), C, 0);

        futures.clear();
        futures.reserve(C);
        PolarPoint *base = world_cloud->points.data();
        for (int i = 0; i < C; ++i) {
            PolarPoint *dst = base + offsets[i];
            futures.emplace_back(
                pool_.enqueue([this, i,dst]() {
                    cameras[i]->getPolarPointCloud(dst, &this->execute[i]);
                })
            );
        }
        int j = 1;
        for (auto &fut: futures) {
            auto start = std::chrono::high_resolution_clock::now();
            fut.get(); //会阻塞直到对应任务完成，并返回结果
            auto end = std::chrono::high_resolution_clock::now();
            std::cout << "d435-" << j << "阻塞: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
            j++;
        }
        // 清理未更新分段，防止旧值残留
        for (size_t i = 0; i < C; ++i) {
            if (execute[i] == 0) {
                PolarPoint *seg = base + offsets[i];
                const size_t count = offsets[i + 1] - offsets[i];
                std::fill_n(seg, count, PolarPoint{});
            }
        }

        return world_cloud;
    }

    ~multi_RGBD() = default;

private:
    std::vector<std::shared_ptr<rs2_D435> > cameras;
    pcl::PointCloud<PolarPoint>::Ptr world_cloud;
    ThreadPool pool_; // 线程池
    std::vector<std::future<void> > futures;
    size_t C{0}; // 相机数量
    // 每个相机点云在 world_cloud 中的起始偏移量
    std::vector<size_t> offsets{0};
    std::vector<int> execute; // 标记每个相机这次是否执行了 getPolarPointCloud
};
#endif //D435_MULTI_RGBD_H
