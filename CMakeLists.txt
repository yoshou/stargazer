cmake_minimum_required(VERSION 3.1.0)


if (WIN32)
find_package(Ceres CONFIG REQUIRED PATHS ${CMAKE_SOURCE_DIR}/third-party/windows/vcpkg/installed/x64-windows)
find_package(glog REQUIRED PATHS ${CMAKE_SOURCE_DIR}/third-party/windows/vcpkg/installed/x64-windows)
else ()
set(CMAKE_C_FLAGS " -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(CMAKE_C_FLAGS_RELEASE "-O2")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-pg -Wall -Wextra -Wno-unused-parameter -mavx -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-pg -O0 -g -mavx -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-pg -O2 -g -mavx -march=native")
endif ()

if (WIN32)
find_package(OpenCV REQUIRED PATHS ${CMAKE_SOURCE_DIR}/third-party/windows/vcpkg/installed/x64-windows)

include_directories(
    ./src
)

include(src/CMakeLists.txt)
include(third-party/CMakeLists.txt)
else ()
set(Boost_INCLUDE_DIR /usr/local/vcpkg/installed/x64-linux/include)
find_package(OpenCV 4 REQUIRED)
find_package(realsense2 REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(yaml-cpp REQUIRED)

find_package(protobuf 3.17 REQUIRED)
find_package(gRPC REQUIRED)

include_directories(
    SYSTEM ${OpenCV_INCLUDE_DIRS}
    SYSTEM ${REALSENSE2_INCLUDE_DIRS}
    SYSTEM ${Boost_INCLUDE_DIRS}
    SYSTEM ${EIGEN3_INCLUDE_DIR}
    SYSTEM ${PCL_INCLUDE_DIRS}
    SYSTEM ${CMAKE_BINARY_DIR}/libjpeg-turbo/include
    SYSTEM ${CMAKE_BINARY_DIR}/third-party/glm
    SYSTEM ${CMAKE_BINARY_DIR}/third-party/cereal/include
    SYSTEM ${CMAKE_BINARY_DIR}/third-party/spdlog/include
    SYSTEM ${CMAKE_BINARY_DIR}/third-party/json/include
    SYSTEM ${PROTOBUF_INCLUDE_DIR}
    ./src
    ./coalsack/src
)

include(lib/CMakeLists.txt)
include(reconstruction_app/CMakeLists.txt)
include(coalsack/CMakeLists.txt)
include(third-party/CMakeLists.txt)
include(protos/CMakeLists.txt)
endif()
