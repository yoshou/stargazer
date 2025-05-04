cmake_minimum_required(VERSION 3.18)

enable_language(CUDA)

find_package(CUDAToolkit)

add_library(stargazer_voxelpose SHARED
    ${CMAKE_CURRENT_LIST_DIR}/voxelpose_cuda.cu
    ${CMAKE_CURRENT_LIST_DIR}/voxelpose.cpp
)

target_compile_features(stargazer_voxelpose PRIVATE cxx_std_17 cuda_std_11)

target_include_directories(stargazer_voxelpose
    PRIVATE
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_INCLUDE_DIRS}>
)

set_target_properties(stargazer_voxelpose PROPERTIES
    LINKER_LANGUAGE CUDA C CXX
    CUDA_ARCHITECTURES OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(stargazer_voxelpose
    glm::glm
    opencv_imgproc
    opencv_highgui
    opencv_features2d
    opencv_calib3d
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_LIBS}>
    $<$<BOOL:${MSVC}>:${CUDAToolkit_LIBRARY_DIR}/cudnn.lib>
    $<$<NOT:$<BOOL:${MSVC}>>:cudnn>
    CUDA::cublas
    CUDA::cudart
    CUDA::cuda_driver
    spdlog::spdlog_header_only
    cereal::cereal
)

add_library(stargazer_preprocess SHARED
    ${CMAKE_CURRENT_LIST_DIR}/preprocess.cu
)

target_include_directories(stargazer_preprocess
    PRIVATE
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_INCLUDE_DIRS}>
)

set_target_properties(stargazer_preprocess PROPERTIES
    LINKER_LANGUAGE CUDA C CXX
    CUDA_ARCHITECTURES OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(stargazer_preprocess
    nvinfer
    opencv_imgproc
    opencv_highgui
    opencv_features2d
    opencv_calib3d
    glm::glm
    unofficial::sqlite3::sqlite3
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_LIBS}>
    CUDA::cudart
    CUDA::cuda_driver
    spdlog::spdlog_header_only
    cereal::cereal
)

add_library(stargazer_mvpose
    ${CMAKE_CURRENT_LIST_DIR}/mvpose.cpp
)

target_include_directories(stargazer_mvpose
    PRIVATE
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_INCLUDE_DIRS}>
)

target_link_libraries(stargazer_mvpose
    glm::glm
    opencv_imgproc
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_LIBS}>
    CUDA::cudart
    CUDA::cuda_driver
)

add_library(stargazer
    ${CMAKE_CURRENT_LIST_DIR}/parameters.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reconstruction/triangulation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reconstruction/correspondance.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reconstruction/reconstruction.cpp
    ${CMAKE_CURRENT_LIST_DIR}/calibration/bundle_adjust_data.cpp
)

target_compile_definitions(stargazer
    PUBLIC
    $<$<BOOL:${MSVC}>:GLOG_NO_ABBREVIATED_SEVERITIES>
    GLOG_USE_GLOG_EXPORT
)

target_include_directories(stargazer
    PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/reconstruction
    ${CMAKE_CURRENT_LIST_DIR}/calibration
)

target_link_libraries(stargazer
    glm::glm
    opencv_imgproc
    $<$<NOT:$<BOOL:${MSVC}>>:pthread>
    $<$<NOT:$<BOOL:${MSVC}>>:stdc++fs>
)
