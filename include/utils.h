//
// Created by YSC on 2025/9/11.
//

#ifndef D435_UTILS_H
#define D435_UTILS_H
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <iostream>
#include <stdexcept>
#include <opencv2/opencv.hpp>


void devices() {
    rs2::context ctx;
    auto devices = ctx.query_devices();
    if (devices.size() == 0) {
        throw std::runtime_error("没设备");
    } //devices[0].get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);devices[1] 获取设备序列号

    auto dev = devices.front();
    for (auto &sensor: dev.query_sensors()) {
        std::cout << "Sensor: " << sensor.get_info(RS2_CAMERA_INFO_NAME) << "\n";
        auto profiles = sensor.get_stream_profiles();
        for (auto &p: profiles) {
            auto v = p.as<rs2::video_stream_profile>();
            std::cout << "  Stream: " << v.stream_type()
                    << " " << v.format()
                    << " " << v.width() << "x" << v.height()
                    << " @" << v.fps() << "fps\n";
        }
    }
}
static void savePictures(const std::string &serial, const rs2::depth_frame &depth, const rs2::video_frame &color) {
    cv::Mat img_depth1(cv::Size(depth.get_width(), depth.get_height()),CV_16U, (void *) depth.get_data(),
                       cv::Mat::AUTO_STEP);
    cv::Mat img_color1(cv::Size(color.get_width(), color.get_height()),CV_8UC3, (void *) color.get_data(),
                       cv::Mat::AUTO_STEP);
    cv::imwrite(std::string("../") + serial + "_depth.png", img_depth1);
    cv::imwrite(std::string("../") + serial + "_color.png", img_color1);
}
#endif //D435_UTILS_H
