if(WIN32)
    set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third-party/vcpkg/installed/x64-windows")
else()
    set(CMAKE_PREFIX_PATH "/usr/local/vcpkg/installed/x64-linux")
    set(Boost_INCLUDE_DIR /usr/local/vcpkg/installed/x64-linux/include)
endif()

find_package(OpenCV REQUIRED)
find_package(Ceres CONFIG REQUIRED)

set_target_properties(glog::glog PROPERTIES
    INTERFACE_LINK_LIBRARIES "gflags"
)
set_target_properties(multiview PROPERTIES
    INTERFACE_LINK_LIBRARIES "gflags"
)
set_target_properties(opencv_sfm PROPERTIES
    INTERFACE_LINK_LIBRARIES "gflags"
)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(nanoflann CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/third-party/vcpkg/installed/x64-include
)
