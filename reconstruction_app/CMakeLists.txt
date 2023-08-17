cmake_minimum_required(VERSION 3.8)

include(FetchContent)

FetchContent_Declare(imgui
    PREFIX imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui
    GIT_TAG v1.89
    SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/imgui"
)

FetchContent_MakeAvailable(imgui)

find_package(gRPC CONFIG REQUIRED)

project(reconstruction_app)

if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif()

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_LIST_DIR}/viewer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/views.cpp
    ${CMAKE_CURRENT_LIST_DIR}/capture.cpp
    ${CMAKE_CURRENT_LIST_DIR}/calibration.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reconstruction.cpp
    ${CMAKE_CURRENT_LIST_DIR}/viewer_app.cpp
    ${CMAKE_BINARY_DIR}/third-party/imgui/imgui.cpp
    ${CMAKE_BINARY_DIR}/third-party/imgui/imgui_draw.cpp
    ${CMAKE_BINARY_DIR}/third-party/imgui/imgui_tables.cpp
    ${CMAKE_BINARY_DIR}/third-party/imgui/imgui_widgets.cpp
    ${CMAKE_BINARY_DIR}/third-party/imgui/backends/imgui_impl_glfw.cpp
    ${CMAKE_BINARY_DIR}/third-party/imgui/backends/imgui_impl_opengl3.cpp
)

add_dependencies(${PROJECT_NAME}
    coalsack
    stargazer_sensor_grpc_proto
    stargazer
    libjpeg-turbo
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_BINARY_DIR}/third-party/imgui
    ${CMAKE_BINARY_DIR}/third-party/imgui/backends
)

target_link_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_BINARY_DIR}/libjpeg-turbo/lib
)

target_compile_definitions(${PROJECT_NAME}
    PUBLIC
)

target_link_libraries(${PROJECT_NAME}
    stargazer
    stargazer_sensor_grpc_proto
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    opencv_features2d
    opencv_calib3d
    opencv_sfm
    multiview
    $<$<NOT:$<BOOL:${WIN32}>>:GL>
    $<$<NOT:$<BOOL:${WIN32}>>:GLU>
    $<$<NOT:$<BOOL:${WIN32}>>:dl>
    glfw
    $<$<BOOL:${WIN32}>:GLEW::GLEW>
    $<$<NOT:$<BOOL:${WIN32}>>:GLEW::GLEW>
    $<$<NOT:$<BOOL:${WIN32}>>:pthread>
    lz4::lz4
    realsense2::realsense2
    Boost::program_options
    libturbojpeg.a
    glog::glog
    ceres
    gRPC::grpc++
)

target_compile_options(${PROJECT_NAME}
    PRIVATE
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wno-unused-parameter>
    $<$<CONFIG:RELEASE>:-O2>
    $<$<CONFIG:DEBUG>:-g -O0>)

target_sources(${PROJECT_NAME}
    PRIVATE
)
