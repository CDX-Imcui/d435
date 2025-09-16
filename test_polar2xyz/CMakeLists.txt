cmake_minimum_required(VERSION 3.16)
project(test)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

find_package(realsense2 REQUIRED)
find_package(Threads REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io visualization)
find_package(OpenCL REQUIRED)

# 添加头文件目录
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${OpenCL_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
)
file(GLOB_RECURSE HEADER_FILES
        ${PROJECT_SOURCE_DIR}/include/*.h
)

add_executable(test
        ${HEADER_FILES}
        main.cpp
)

target_link_libraries(test
        ${OpenCV_LIBS}
        ${realsense2_LIBRARY}
        ${PCL_LIBRARIES}
        Threads::Threads
        ${OpenCL_LIBRARIES}
)

