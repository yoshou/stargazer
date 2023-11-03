cmake_minimum_required(VERSION 3.18)

if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif()

if(NOT MSVC)
    enable_language(CUDA)
endif()

find_package(CUDAToolkit)

add_library(voxelpose
    ${CMAKE_CURRENT_LIST_DIR}/voxelpose_cuda.cu
)

target_compile_features(voxelpose PRIVATE cxx_std_17)

set_target_properties(voxelpose PROPERTIES
    LINKER_LANGUAGE CUDA C CXX
    CUDA_SEPARABLE_COMPILATION ON)

add_executable(stargazer_viewer
    ${CMAKE_CURRENT_LIST_DIR}/viewer.cpp
    ${CMAKE_CURRENT_LIST_DIR}/views.cpp
    ${CMAKE_CURRENT_LIST_DIR}/capture.cpp
    ${CMAKE_CURRENT_LIST_DIR}/calibration.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reconstruction.cpp
    ${CMAKE_CURRENT_LIST_DIR}/config_file.cpp
    ${CMAKE_CURRENT_LIST_DIR}/viewer_app.cpp
)

add_dependencies(stargazer_viewer
    coalsack
    stargazer_sensor_grpc_proto
    stargazer
    voxelpose
)

target_include_directories(stargazer_viewer
    PRIVATE
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_INCLUDE_DIRS}>
)

target_link_directories(stargazer_viewer
    PUBLIC
    ${CMAKE_BINARY_DIR}/libjpeg-turbo/lib
)

target_compile_definitions(stargazer_viewer
    PUBLIC
    $<$<BOOL:${MSVC}>:_USE_MATH_DEFINES>
    $<$<BOOL:${MSVC}>:GLOG_NO_ABBREVIATED_SEVERITIES>
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
    opencv_aruco
    multiview
    $<$<NOT:$<BOOL:${WIN32}>>:GL>
    $<$<NOT:$<BOOL:${WIN32}>>:GLU>
    $<$<NOT:$<BOOL:${WIN32}>>:dl>
    glfw
    glad::glad
    $<$<NOT:$<BOOL:${WIN32}>>:pthread>
    Boost::program_options
    JPEG::JPEG
    glog::glog
    ceres
    gRPC::grpc++
    imgui::imgui
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_LIBS}>
    voxelpose
    $<$<BOOL:${MSVC}>:${CUDAToolkit_LIBRARY_DIR}/cudnn.lib>
    $<$<NOT:$<BOOL:${MSVC}>>:cudnn>
    CUDA::cudart
    CUDA::cublas
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
