#ifndef D435_OPENCL_H
#define D435_OPENCL_H

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

struct PolarPoint {
    float r;
    float theta;
    float phi;
    inline PolarPoint() : r(0.f), theta(0.f), phi(0.f) {}
    inline PolarPoint(float _r, float _t, float _p) : r(_r), theta(_t), phi(_p) {}
};// 定义并注册 PolarPoint 为 PCL 点类型

POINT_CLOUD_REGISTER_POINT_STRUCT(PolarPoint,
    (float, r, r)
    (float, theta, theta)
    (float, phi, phi)
)

class OpenCLConverter {
public:
    // 可选的 platformIndex/deviceIndex 用于在多平台/多设备环境中选择目标设备
    OpenCLConverter(unsigned platformIndex = 0, unsigned deviceIndex = 0)
        : platformIndex_(platformIndex), deviceIndex_(deviceIndex), context_(nullptr),
          queue_(nullptr), program_(nullptr), kernel_(nullptr),
          bufIn_(nullptr), bufOut_(nullptr), capacity_(0),
          preferredLocalSize_(0) {
        initOpenCL();
        buildKernel();
    }

    ~OpenCLConverter() {
        releaseBuffers();
        if (kernel_) clReleaseKernel(kernel_);
        if (program_) clReleaseProgram(program_);
        if (queue_) clReleaseCommandQueue(queue_);
        if (context_) clReleaseContext(context_);
    }

    void convert(const pcl::PointCloud<pcl::PointXYZ>::Ptr& world_cloud,
                 pcl::PointCloud<PolarPoint>::Ptr& polar_cloud)
    {
        if (!world_cloud) throw std::invalid_argument("world_cloud is null");
        size_t N = world_cloud->points.size();
        if (N == 0) {
            if (!polar_cloud) polar_cloud.reset(new pcl::PointCloud<PolarPoint>());
            polar_cloud->clear();
            return;
        }

        // 准备输入 buffer（pack 为 cl_float4，以保持内核简洁）
        std::vector<cl_float4> input;
        input.resize(N);
        for (size_t i = 0; i < N; ++i) {
            input[i].s[0] = world_cloud->points[i].x;
            input[i].s[1] = world_cloud->points[i].y;
            input[i].s[2] = world_cloud->points[i].z;
            input[i].s[3] = 0.0f;
        }

        // 调用通用 convert（复用已有逻辑）
        std::vector<cl_float4> output;
        convert(input, output);

        // 将输出写回 PCL 极坐标点云
        if (!polar_cloud) polar_cloud.reset(new pcl::PointCloud<PolarPoint>());
        polar_cloud->resize(N);
        for (size_t i = 0; i < N; ++i) {
            polar_cloud->points[i].r = output[i].s[0];
            polar_cloud->points[i].theta = output[i].s[1];
            polar_cloud->points[i].phi = output[i].s[2];
        }
    }

    void convert(const std::vector<cl_float4>& in, std::vector<cl_float4>& out) {
        size_t N = in.size();
        if (N == 0) { out.clear(); return; }
        out.resize(N);
        ensureBuffers(N);
        cl_int err = CL_SUCCESS;

        // 上传：使用阻塞写（可根据需要改为非阻塞 + event）
        err = clEnqueueWriteBuffer(queue_, bufIn_, CL_TRUE, 0, sizeof(cl_float4) * N, in.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("clEnqueueWriteBuffer failed");

        // 设置内核参数
        err  = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &bufIn_);
        err |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &bufOut_);
        int nn = static_cast<int>(N);
        err |= clSetKernelArg(kernel_, 2, sizeof(int), &nn);
        if (err != CL_SUCCESS) throw std::runtime_error("clSetKernelArg failed");

        // 非对齐的 global size，按 N
        size_t globalSize = N;
        size_t localSize = 0;
        if (preferredLocalSize_ > 0) localSize = preferredLocalSize_;

        // 如果没有指定 localSize 则传 nullptr（OpenCL 让 runtime 自动选择）
        err = clEnqueueNDRangeKernel(queue_, kernel_, 1, nullptr, &globalSize,
                                     (localSize ? &localSize : nullptr), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("clEnqueueNDRangeKernel failed");
        clFinish(queue_);

        // 读取结果（阻塞）
        err = clEnqueueReadBuffer(queue_, bufOut_, CL_TRUE, 0, sizeof(cl_float4) * N, out.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("clEnqueueReadBuffer failed");
    }

    // 设置建议的 local work size（0 表示交给 runtime 自动选择）
    void setPreferredLocalSize(size_t local) { preferredLocalSize_ = local; }

    // 返回当前分配容量（点数）
    size_t capacity() const { return capacity_; }

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

    // buffers
    cl_mem bufIn_ = nullptr;   // cl_float4 * capacity_
    cl_mem bufOut_ = nullptr;  // cl_float4 * capacity_
    size_t capacity_ = 0;

    // 内核源码（使用 float4 便于对齐）
    const char* kernelSource_ = R"CLC(
    __kernel void cartesian_to_spherical(
        __global const float4* in,
        __global float4* out,
        const int n)
    {
        int i = get_global_id(0);
        if (i >= n) return;
        float x = in[i].x;
        float y = in[i].y;
        float z = in[i].z;
        float r = sqrt(x*x + y*y + z*z);
        float theta = (r > 1e-6f) ? acos(z / r) : 0.0f;
        float phi = atan2(y, x);
        out[i] = (float4)(r, theta, phi, 0.0f);
    }
    )CLC";

    size_t preferredLocalSize_ = 0;

        // Helper: 检查 cl 返回值（可扩展为更详细的错误信息）
    static void checkCLError(cl_int err, const char* msg) {
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

        if (platformIndex_ >= numPlatforms) {
            std::cerr << "Warning: platformIndex out of range, using 0\n";
            platformIndex_ = 0;
        }
        platform_id_ = platforms[platformIndex_];

        // 选择 GPU 设备（若无 GPU 可改为 CL_DEVICE_TYPE_ALL）
        cl_uint numDevices = 0;//在选定的平台上查找一个GPU设备，并将其ID保存到device_
        err = clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
        if (err != CL_SUCCESS || numDevices == 0) {
            // 退回到 ALL 类型（尽量找到可用设备）
            err = clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
            checkCLError(err, "clGetDeviceIDs(all): get count");
        }
        std::vector<cl_device_id> devices(numDevices);
        err = clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
        if (err != CL_SUCCESS || numDevices == 0) {
            // 再次尝试 ALL
            err = clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);
            checkCLError(err, "clGetDeviceIDs(all): get ids");
        }

        if (deviceIndex_ >= numDevices) {
            std::cerr << "Warning: deviceIndex out of range, using 0\n";
            deviceIndex_ = 0;
        }
        device_id_ = devices[deviceIndex_];

        //基于选定的设备创建OpenCL上下文，表示一个OpenCL环境的实例，OpenCL程序运行和资源管理的“容器”，后续所有OpenCL操作都在该上下文中进行
        context_ = clCreateContext(nullptr, 1, &device_id_, nullptr, nullptr, &err);
        checkCLError(err, "clCreateContext");

        // 创建命令队列（尝试使用带属性的新 API，否则回退）
#ifdef CL_VERSION_2_0
        cl_queue_properties props[] = {0};
        queue_ = clCreateCommandQueueWithProperties(context_, device_id_, props, &err);
        if (err != CL_SUCCESS) {
            // fallback
            queue_ = clCreateCommandQueue(context_, device_id_, 0, &err);
            checkCLError(err, "clCreateCommandQueue fallback");
        }
#else
        queue_ = clCreateCommandQueue(context_, device_id_, 0, &err);
        checkCLError(err, "clCreateCommandQueue");
#endif
    }

    void buildKernel() {
        cl_int err;
        program_ = clCreateProgramWithSource(context_, 1, &kernelSource_, nullptr, &err);
        checkCLError(err, "clCreateProgramWithSource");

        err = clBuildProgram(program_, 1, &device_id_, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            // 打印 build log
            size_t log_size = 0;
            clGetProgramBuildInfo(program_, device_id_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size);
            clGetProgramBuildInfo(program_, device_id_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::string logstr(log.begin(), log.end());
            std::cerr << "OpenCL build log:\n" << logstr << std::endl;
            checkCLError(err, "clBuildProgram");
        }

        kernel_ = clCreateKernel(program_, "cartesian_to_spherical", &err);
        checkCLError(err, "clCreateKernel");
    }


    void ensureBuffers(size_t N) {//确保 OpenCL 输入输出缓冲区（bufIn_ 和 bufOut_）的容量足够容纳 N 个点的数据
        if (N > capacity_) {
            releaseBuffers();//缓冲区复用，仅在需要更大容量时重分配
            cl_int err;//重新分配能容纳 N 个点的新缓冲区
            bufIn_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, sizeof(cl_float4) * N, nullptr, &err);
            bufOut_ = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, sizeof(cl_float4) * N, nullptr, &err);
            capacity_ = N;//更新 capacity_，避免频繁分配和释放内存，提高效率
        }
    }

    void releaseBuffers() {//释放原有缓冲区
        if (bufIn_) clReleaseMemObject(bufIn_);
        if (bufOut_) clReleaseMemObject(bufOut_);
        bufIn_ = bufOut_ = nullptr;
        capacity_ = 0;
    }
};

#endif //D435_OPENCL_H