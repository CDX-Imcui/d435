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

    // 输入: col-major float[16] （Eigen::Matrix4f::data()）
    // 输出: row-major float[16] （OpenCL kernel 用）
    static inline void colMajorToRowMajor4x4(const float *colMajor, float *rowMajor) {
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                rowMajor[r * 4 + c] = colMajor[c * 4 + r];
            }
        }
    }

    // depth -> pointcloud -> transformPointCloud -> polarpointcloud
    void DepthToPointCloudZ16(const uint16_t *depth_raw,
                              int width, int height,
                              const rs2_intrinsics &K,
                              float depth_scale,
                              const float *T_4x4, // 16 个 float
                              pcl::PointXYZ *out_xyz) {
        const size_t N = static_cast<size_t>(width) * height;
        if (N == 0) throw std::invalid_argument("depth size = 0");
        ensureBuffers(N); // 分配/复用 Z16 与 XYZ 缓冲
        colMajorToRowMajor4x4(T_4x4, T_row_major);

        cl_int err = CL_SUCCESS;

        // 映射并复制深度 Z16
        void *mapped = clEnqueueMapBuffer(queue_, bufDepthU16_, CL_TRUE, CL_MAP_WRITE, 0,
                                          sizeof(uint16_t) * N, 0, nullptr, nullptr, &err);
        checkCLError(err, "clEnqueueMapBuffer bufDepthU16");
        std::memcpy(mapped, depth_raw, sizeof(uint16_t) * N);
        err = clEnqueueUnmapMemObject(queue_, bufDepthU16_, mapped, 0, nullptr, nullptr);
        checkCLError(err, "clEnqueueUnmapMemObject bufDepthU16");

        // 确保/更新 T 常量缓冲
        if (!bufT_) {
            bufT_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, sizeof(float) * 16, nullptr, &err);
            checkCLError(err, "clCreateBuffer bufT");
        }
        err = clEnqueueWriteBuffer(queue_, bufT_, CL_TRUE, 0, sizeof(float) * 16,
                                   T_row_major, 0, nullptr, nullptr);
        checkCLError(err, "clEnqueueWriteBuffer bufT");

        float Khost[4] = {K.fx, K.fy, K.ppx, K.ppy};
        // 严格按内核形参顺序设置内核参数
        err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &bufDepthU16_); // raw
        err |= clSetKernelArg(kernel_, 1, sizeof(int), &width); // width
        err |= clSetKernelArg(kernel_, 2, sizeof(int), &height); // height
        err |= clSetKernelArg(kernel_, 3, sizeof(Khost), &Khost); // KParams
        err |= clSetKernelArg(kernel_, 4, sizeof(float), &depth_scale); // depth_scale
        err |= clSetKernelArg(kernel_, 5, sizeof(cl_mem), &bufT_); // T (constant buffer)
        err |= clSetKernelArg(kernel_, 6, sizeof(cl_mem), &bufXYZOut_); // out_xyz
        checkCLError(err, "clSetKernelArg z16_to_points");

        // 以 2D NDRange 调度（与 kernel 内 get_global_id(0/1) 匹配）
        const size_t global[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
        err = clEnqueueNDRangeKernel(queue_, kernel_, 2, nullptr, global,
                                     (preferredLocalSize_ > 0 ? &preferredLocalSize_ : nullptr), 0, nullptr, nullptr);
        checkCLError(err, "clEnqueueNDRangeKernel z16_to_points");

        // 阻塞（隐式等待 kernel 完成） 读回 XYZ PCL::PointXYZ 为 16B 步长，对齐 float4 写出
        err = clEnqueueReadBuffer(queue_, bufXYZOut_, CL_TRUE, 0,
                                  sizeof(pcl::PointXYZ) * N, out_xyz, 0, nullptr, nullptr);
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

    cl_mem bufDepthU16_ = nullptr; // uint16_t * capacity_
    // buffers for depth->xyz path
    // cl_mem bufDepth_ = nullptr; // float * capacity_
    cl_mem bufXYZOut_ = nullptr; // pcl::PointXYZ * capacity_
    size_t capacity_ = 0; // 已为 depth/path 分配的最大点数

    cl_mem bufT_ = nullptr; // 4x4 行主序矩阵（16f）常量缓冲
    float T_row_major[16];

    // 16B 对齐输出；输入为 Z16
    const char *kernelSource_ = R"CLC(
    typedef struct { float fx, fy, ppx, ppy; } KParams;

__kernel void z16_to_points(__global const ushort* raw,
                            int width, int height,
                            KParams K, float depth_scale,
                            __constant float* T,          // 16f
                            __global float4* out_xyz)     // 与 PCL::PointXYZ 的 16B 对齐
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float z = (float)raw[idx] * depth_scale;
    if (!(z > 0.f) || !isfinite(z)) {
        out_xyz[idx] = (float4)(NAN, NAN, NAN, 0.f);
        return;
    }

    // depth->相机坐标->机器人坐标
    float xc = z;
    float yc = -z * ((float)x - K.ppx) / K.fx;
    float zc = -z * ((float)y - K.ppy) / K.fy;

    // 点 (xc, yc, zc) 乘以一个 4×4 齐次变换矩阵 T，得到在 世界坐标系下的点 (X, Y, Z)。替代 pcl::transformPointCloud
    // |X|       |xc|
    // |Y| = T * |yc|
    // |Z|       |zc|
    // |1|       |1 |
    // 行主序 T: [r0 r1 r2 r3; r4 r5 r6 r7; r8 r9 r10 r11; r12 r13 r14 r15]
    float X = T[0]*xc + T[1]*yc + T[2]*zc + T[3];
    float Y = T[4]*xc + T[5]*yc + T[6]*zc + T[7];
    float Z = T[8]*xc + T[9]*yc + T[10]*zc + T[11];

    out_xyz[idx] = (float4)(X, Y, Z, 0.f); // 16B 对齐写出
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
        kernel_ = clCreateKernel(program_, "z16_to_points", &err);
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
        if (bufT_) clReleaseMemObject(bufT_);
        bufDepthU16_ = bufXYZOut_ = nullptr;
        capacity_ = 0;
    }
};

#endif //D435_DEPTH2POINT_H
