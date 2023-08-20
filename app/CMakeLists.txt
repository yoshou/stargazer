cmake_minimum_required(VERSION 3.8)

if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif()

add_executable(stargazer_viewer
    ${CMAKE_CURRENT_LIST_DIR}/viewer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/views.cpp
    ${CMAKE_CURRENT_LIST_DIR}/capture.cpp
    ${CMAKE_CURRENT_LIST_DIR}/calibration.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reconstruction.cpp
    ${CMAKE_CURRENT_LIST_DIR}/viewer_app.cpp
)

add_dependencies(stargazer_viewer
    coalsack
    stargazer_sensor_grpc_proto
    stargazer
    libjpeg-turbo
)

target_include_directories(stargazer_viewer
    PUBLIC
)

target_link_directories(stargazer_viewer
    PUBLIC
    ${CMAKE_BINARY_DIR}/libjpeg-turbo/lib
)

target_compile_definitions(stargazer_viewer
    PUBLIC
)

target_link_libraries(stargazer_viewer
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
    glad::glad
    $<$<NOT:$<BOOL:${WIN32}>>:pthread>
    Boost::program_options
    libturbojpeg.a
    glog::glog
    ceres
    gRPC::grpc++
    imgui::imgui
)

target_compile_options(stargazer_viewer
    PRIVATE
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wno-unused-parameter>
    $<$<CONFIG:RELEASE>:-O2>
    $<$<CONFIG:DEBUG>:-g -O0>)

target_sources(stargazer_viewer
    PRIVATE
)
