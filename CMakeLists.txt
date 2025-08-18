cmake_minimum_required(VERSION 3.16)
project(d435)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)
#find_package(Threads REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io visualization)

# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

add_executable(d435 main.cpp)

target_link_libraries(d435
        ${OpenCV_LIBS}
        ${realsense2_LIBRARY}
        ${PCL_LIBRARIES}
)