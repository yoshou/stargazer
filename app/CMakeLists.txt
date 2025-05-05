cmake_minimum_required(VERSION 3.18)

if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif()

add_executable(stargazer_viewer
    ${CMAKE_CURRENT_LIST_DIR}/viewer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/views.cpp
    ${CMAKE_CURRENT_LIST_DIR}/capture_pipeline.cpp
    ${CMAKE_CURRENT_LIST_DIR}/calibration_pipeline.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reconstruction_pipeline.cpp
    ${CMAKE_CURRENT_LIST_DIR}/config.cpp
    ${CMAKE_CURRENT_LIST_DIR}/viewer_app.cpp
)

add_dependencies(stargazer_viewer
    stargazer
    stargazer_sensor_grpc_proto
    stargazer_voxelpose
    stargazer_preprocess
    stargazer_mvpose
)

target_include_directories(stargazer_viewer
    PRIVATE
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_INCLUDE_DIRS}>
)

target_compile_definitions(stargazer_viewer
    PUBLIC
    $<$<BOOL:${MSVC}>:_USE_MATH_DEFINES>
    $<$<BOOL:${MSVC}>:GLOG_NO_ABBREVIATED_SEVERITIES>
)

target_link_libraries(stargazer_viewer
    PRIVATE
    stargazer
    stargazer_sensor_grpc_proto
    stargazer_voxelpose
    stargazer_preprocess
    stargazer_mvpose
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    opencv_features2d
    opencv_calib3d
    opencv_aruco
    $<$<NOT:$<BOOL:${WIN32}>>:GL>
    $<$<NOT:$<BOOL:${WIN32}>>:GLU>
    $<$<NOT:$<BOOL:${WIN32}>>:dl>
    glfw
    glad::glad
    $<$<NOT:$<BOOL:${WIN32}>>:pthread>
    spdlog::spdlog_header_only
    cereal::cereal
    Boost::asio
    JPEG::JPEG
    lz4::lz4
    Ceres::ceres
    gRPC::grpc++
    imgui::imgui
    glm::glm
)

target_compile_options(stargazer_viewer
    PRIVATE
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wno-unused-parameter>
    $<$<CXX_COMPILER_ID:Clang>:$<$<CONFIG:RELEASE>:-O2>>
    $<$<CXX_COMPILER_ID:Clang>:$<$<CONFIG:DEBUG>:-g -O0>>
    $<$<BOOL:${MSVC}>:/bigobj>)

target_sources(stargazer_viewer
    PRIVATE
)

set_target_properties(stargazer_viewer PROPERTIES
    CXX_STANDARD 20)
