#ifndef D435_DEPTH2POINT_H
#define D435_DEPTH2POINT_H

#include <CL/cl.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/register_point_struct.h> // 注册自定义点类型用
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>


class depth2point {
public:
    // 可选的 platformIndex/deviceIndex 用于在多平台/多设备环境中选择目标设备
    depth2point(unsigned platformIndex = 0, unsigned deviceIndex = 0)
        : platformIndex_(platformIndex), deviceIndex_(deviceIndex) {
        initOpenCL();
        buildKernel();
    }

    ~depth2point() {
        releaseBuffers();
        if (kernel_) clReleaseKernel(kernel_);
        if (program_) clReleaseProgram(program_);
        if (queue_) clReleaseCommandQueue(queue_);
        if (context_) clReleaseContext(context_);
    }
     // 新接口：直接吃 Z16 原始深度，kernel 内乘 depth_scale -> 米
    void DepthToPointCloudZ16(const uint16_t* depth_raw, int width, int height,
                              const rs2_intrinsics& intr, float depth_scale,
                              pcl::PointXYZ* out_xyz)
    {
        const size_t N = static_cast<size_t>(width) * height;
        if (N == 0) throw std::invalid_argument("depth size = 0");
        ensureBuffers(N); // 复用缓冲

        cl_int err = CL_SUCCESS;

        // 把输入缓冲映射为页锁内存并 memcpy（便于零拷贝/减少拷贝）
        void* mapped = clEnqueueMapBuffer(queue_, bufDepthU16_, CL_TRUE, CL_MAP_WRITE, 0,
                                          sizeof(uint16_t) * N, 0, nullptr, nullptr, &err);
        checkCLError(err, "clEnqueueMapBuffer bufDepthU16");
        std::memcpy(mapped, depth_raw, sizeof(uint16_t) * N);
        err = clEnqueueUnmapMemObject(queue_, bufDepthU16_, mapped, 0, nullptr, nullptr);
        checkCLError(err, "clEnqueueUnmapMemObject bufDepthU16");

        // 设置参数
        err  = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &bufDepthU16_);
        err |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &bufXYZOut_);
        int w = width, h = height;
        float fx = intr.fx, fy = intr.fy, ppx = intr.ppx, ppy = intr.ppy;
        float scale = depth_scale;
        float nanv = std::numeric_limits<float>::quiet_NaN();
        err |= clSetKernelArg(kernel_, 2, sizeof(int), &w);
        err |= clSetKernelArg(kernel_, 3, sizeof(int), &h);
        err |= clSetKernelArg(kernel_, 4, sizeof(float), &fx);
        err |= clSetKernelArg(kernel_, 5, sizeof(float), &fy);
        err |= clSetKernelArg(kernel_, 6, sizeof(float), &ppx);
        err |= clSetKernelArg(kernel_, 7, sizeof(float), &ppy);
        err |= clSetKernelArg(kernel_, 8, sizeof(float), &scale);
        err |= clSetKernelArg(kernel_, 9, sizeof(float), &nanv);
        checkCLError(err, "clSetKernelArg z16_to_pointcloud");

        // 异步提交（不显式 clFinish）
        size_t globalSize = N;
        err = clEnqueueNDRangeKernel(queue_, kernel_, 1, nullptr, &globalSize,
                                     (preferredLocalSize_ > 0 ? &preferredLocalSize_ : nullptr),
                                     0, nullptr, nullptr);
        checkCLError(err, "clEnqueueNDRangeKernel z16_to_pointcloud");

        // 阻塞读回（隐式等待 kernel 完成），目标缓冲为 PCL::PointXYZ（16B 步长）
        err = clEnqueueReadBuffer(queue_, bufXYZOut_, CL_TRUE, 0,
                                  sizeof(pcl::PointXYZ) * N, out_xyz, 0, nullptr, nullptr);
        checkCLError(err, "clEnqueueReadBuffer bufXYZOut");
    }

    // out: 指向已分配好的 pcl::PointXYZ 数组（size >= width*height）
    void DepthToPointCloud(const std::vector<float> &depth, int width, int height,
                              const rs2_intrinsics &intrinsics, pcl::PointXYZ *out) {
        size_t N = depth.size();
        if (N == 0) throw std::invalid_argument("depth size = 0");
        ensureBuffers(N);

        cl_int err = CL_SUCCESS;
        // 上传数据到 GPU。   使用阻塞写，当数据传输和计算可以重叠时，非阻塞 + event才有意义
        err = clEnqueueWriteBuffer(queue_, bufDepth_, CL_TRUE, 0, sizeof(float) * N, depth.data(), 0, nullptr, nullptr);
        checkCLError(err, "clEnqueueWriteBuffer bufDepth");

        // 给内核函数 depth_to_pointcloud 设置参数
        err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &bufDepth_);
        err |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &bufXYZOut_);
        int w = width, h = height;
        float fx = intrinsics.fx, fy = intrinsics.fy, ppx = intrinsics.ppx, ppy = intrinsics.ppy;
        float nanv = std::numeric_limits<float>::quiet_NaN();
        err |= clSetKernelArg(kernel_, 2, sizeof(int), &w);
        err |= clSetKernelArg(kernel_, 3, sizeof(int), &h);
        err |= clSetKernelArg(kernel_, 4, sizeof(float), &fx);
        err |= clSetKernelArg(kernel_, 5, sizeof(float), &fy);
        err |= clSetKernelArg(kernel_, 6, sizeof(float), &ppx);
        err |= clSetKernelArg(kernel_, 7, sizeof(float), &ppy);
        err |= clSetKernelArg(kernel_, 8, sizeof(float), &nanv);
        checkCLError(err, "clSetKernelArg depth_to_pointcloud");

        // launch
        size_t globalSize = N;
        err = clEnqueueNDRangeKernel(queue_, kernel_, 1, nullptr, &globalSize,
                                     (preferredLocalSize_ > 0 ? &preferredLocalSize_ : nullptr), 0, nullptr, nullptr);
        checkCLError(err, "clEnqueueNDRangeKernel depth_to_pointcloud");
        clFinish(queue_);

        // 下载结果到已有缓冲区(blocking)
        err = clEnqueueReadBuffer(queue_, bufXYZOut_, CL_TRUE, 0, sizeof(pcl::PointXYZ) * N, out, 0, nullptr, nullptr);
        checkCLError(err, "clEnqueueReadBuffer bufXYZOut");
    }

private:
    // OpenCL 状态
    unsigned platformIndex_;
    unsigned deviceIndex_;
    cl_platform_id platform_id_ = nullptr;
    cl_device_id device_id_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;
    size_t preferredLocalSize_ = 0;

    cl_mem bufDepthU16_ = nullptr;  // uint16_t * capacity_
    // buffers for depth->xyz path
    cl_mem bufDepth_ = nullptr; // float * capacity_
    cl_mem bufXYZOut_ = nullptr; // pcl::PointXYZ * capacity_
    size_t capacity_ = 0; // 已为 depth/path 分配的最大点数

    // 16B 对齐输出；输入为 Z16
    const char* kernelSource_ = R"CLC(
    __kernel void z16_to_pointcloud(
        __global const ushort* depthRaw,  // 输入：Z16
        __global float4* outXYZ,          // 输出：与 PCL::PointXYZ 16B 对齐
        const int width,
        const int height,
        const float fx,
        const float fy,
        const float ppx,
        const float ppy,
        const float scale,                // depth_scale
        const float nanv)
    {
        int idx = get_global_id(0);
        int total = width * height;
        if (idx >= total) return;

        ushort d = depthRaw[idx];
        if (d == 0) { // 无效深度
            outXYZ[idx] = (float4)(nanv, nanv, nanv, nanv);
            return;
        }

        float z = (float)d * scale; // 米
        if (!(z > 0.0f)) {
            outXYZ[idx] = (float4)(nanv, nanv, nanv, nanv);
            return;
        }

        int x = idx % width;
        int y = idx / width;

        float X = z;
        float Y = - z * ((float)x - ppx) / fx;
        float Z = - z * ((float)y - ppy) / fy;

        // 写满 16B，w 作为 padding
        outXYZ[idx] = (float4)(X, Y, Z, nanv);
    }
    )CLC";


    // 检查 cl 返回值（可扩展为更详细的错误信息）
    static void checkCLError(cl_int err, const char *msg) {
        if (err != CL_SUCCESS) {
            std::string s = std::string(msg) + " (cl_err=" + std::to_string(err) + ")";
            throw std::runtime_error(s);
        }
    }

    void initOpenCL() {
        cl_int err;
        cl_uint numPlatforms = 0;
        err = clGetPlatformIDs(0, nullptr, &numPlatforms);
        checkCLError(err, "clGetPlatformIDs: query count");
        if (numPlatforms == 0) throw std::runtime_error("No OpenCL platforms found");
        std::vector<cl_platform_id> platforms(numPlatforms);
        err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
        checkCLError(err, "clGetPlatformIDs: get ids");

        if (platformIndex_ >= numPlatforms) platformIndex_ = 0;
        platform_id_ = platforms[platformIndex_];

        cl_uint numDevices = 0;
        err = clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
        if (err != CL_SUCCESS || numDevices == 0) {
            err = clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
            checkCLError(err, "clGetDeviceIDs(all): get count");
        }
        std::vector<cl_device_id> devices(numDevices);
        err = clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
        if (err != CL_SUCCESS || numDevices == 0) {
            err = clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);
            checkCLError(err, "clGetDeviceIDs(all): get ids");
        }
        if (deviceIndex_ >= numDevices) deviceIndex_ = 0;
        device_id_ = devices[deviceIndex_];

        context_ = clCreateContext(nullptr, 1, &device_id_, nullptr, nullptr, &err);
        checkCLError(err, "clCreateContext");

        const cl_queue_properties props[] = {0};
        queue_ = clCreateCommandQueueWithProperties(context_, device_id_, props, &err);
        checkCLError(err, "clCreateCommandQueueWithProperties");
    }

    void buildKernel() {
        cl_int err;
        program_ = clCreateProgramWithSource(context_, 1, &kernelSource_, nullptr, &err);
        checkCLError(err, "clCreateProgramWithSource");
        err = clBuildProgram(program_, 1, &device_id_, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program_, device_id_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program_, device_id_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::cerr << "OpenCL build log:\n" << std::string(log.begin(), log.end()) << std::endl;
            checkCLError(err, "clBuildProgram");
        }
        kernel_ = clCreateKernel(program_, "z16_to_pointcloud", &err);
        checkCLError(err, "clCreateKernel");
    }


    void ensureBuffers(size_t N) {
        //确保 OpenCL 输入输出缓冲区（bufIn_ 和 bufOut_）的容量足够容纳 N 个点的数据
        if (N <= capacity_) return;
        releaseBuffers();
        cl_int err;
        // 输入：Z16，分配可映射的页锁内存（便于零拷贝/减少拷贝）
        bufDepthU16_ = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                      sizeof(uint16_t) * N, nullptr, &err);
        checkCLError(err, "clCreateBuffer bufDepthU16");
        // 输出：16B 对齐到 PCL::PointXYZ
        bufXYZOut_ = clCreateBuffer(context_, CL_MEM_WRITE_ONLY,
                                    sizeof(pcl::PointXYZ) * N, nullptr, &err);
        checkCLError(err, "clCreateBuffer bufXYZOut");
        capacity_ = N;
    }

    void releaseBuffers() {
        //释放原有缓冲区
        if (bufDepthU16_) clReleaseMemObject(bufDepthU16_);
        if (bufXYZOut_) clReleaseMemObject(bufXYZOut_);
        bufDepthU16_ = bufXYZOut_ = nullptr;
        capacity_ = 0;
    }
};

#endif //D435_DEPTH2POINT_H
