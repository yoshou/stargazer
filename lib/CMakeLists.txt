cmake_minimum_required(VERSION 3.18)

set(ENABLE_CUDA ON TRUE CACHE BOOL "Enable CUDA support")

if (ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit)
else()
    set(CMAKE_PREFIX_PATH /opt/rocm)

    enable_language(HIP)
    find_package(hip REQUIRED)
    find_package(hipblas REQUIRED)
    find_package(miopen REQUIRED)
    find_package(migraphx REQUIRED)
endif()

add_library(stargazer_voxelpose SHARED
    $<$<BOOL:${ENABLE_CUDA}>:${CMAKE_CURRENT_LIST_DIR}/voxelpose.cu>
    $<$<NOT:$<BOOL:${ENABLE_CUDA}>>:${CMAKE_CURRENT_LIST_DIR}/voxelpose.hip>
    ${CMAKE_CURRENT_LIST_DIR}/voxelpose.cpp
)

target_compile_features(stargazer_voxelpose PRIVATE cxx_std_17 cuda_std_11)

target_include_directories(stargazer_voxelpose
    PRIVATE
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_INCLUDE_DIRS}>
)

target_compile_definitions(stargazer_voxelpose
    PUBLIC
    $<$<BOOL:${ENABLE_CUDA}>:USE_CUDA>
)

if (ENABLE_CUDA)
    set_target_properties(stargazer_voxelpose PROPERTIES
        CXX_STANDARD 20
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
        opencv_dnn
        $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_LIBS}>
        $<$<BOOL:${MSVC}>:${CUDAToolkit_LIBRARY_DIR}/cudnn.lib>
        $<$<NOT:$<BOOL:${MSVC}>>:cudnn>
        CUDA::cublas
        CUDA::cudart
        CUDA::cuda_driver
        spdlog::spdlog_header_only
        cereal::cereal
)
else()
    set_target_properties(stargazer_voxelpose PROPERTIES
        CXX_STANDARD 20
        LINKER_LANGUAGE HIP C CXX)

    target_link_libraries(stargazer_voxelpose
        glm::glm
        opencv_imgproc
        opencv_highgui
        opencv_features2d
        opencv_calib3d
        opencv_dnn
        hip
        roc::hipblas
        MIOpen
        migraphx::c
        spdlog::spdlog_header_only
        cereal::cereal
    )
endif()

add_library(stargazer_preprocess SHARED
    $<$<BOOL:${ENABLE_CUDA}>:${CMAKE_CURRENT_LIST_DIR}/preprocess.cu>
    $<$<NOT:$<BOOL:${ENABLE_CUDA}>>:${CMAKE_CURRENT_LIST_DIR}/preprocess.hip>
    
)

target_include_directories(stargazer_preprocess
    PRIVATE
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_INCLUDE_DIRS}>
)

target_compile_definitions(stargazer_preprocess
    PUBLIC
    $<$<BOOL:${ENABLE_CUDA}>:USE_CUDA>
)

if (ENABLE_CUDA)
    set_target_properties(stargazer_preprocess PROPERTIES
        CXX_STANDARD 20
        LINKER_LANGUAGE CUDA C CXX
        CUDA_ARCHITECTURES OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON)

    target_link_libraries(stargazer_preprocess
        opencv_imgproc
        opencv_highgui
        opencv_features2d
        opencv_calib3d
        glm::glm
        unofficial::sqlite3::sqlite3
        $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_LIBS}>
        CUDA::cudart
        CUDA::cuda_driver
        Eigen3::Eigen
        spdlog::spdlog_header_only
        cereal::cereal
    )
else()
    set_target_properties(stargazer_voxelpose PROPERTIES
        CXX_STANDARD 20
        LINKER_LANGUAGE HIP C CXX)

    target_link_libraries(stargazer_preprocess
        opencv_imgproc
        opencv_highgui
        opencv_features2d
        opencv_calib3d
        hip
        roc::hipblas
        MIOpen
        glm::glm
        unofficial::sqlite3::sqlite3
        $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_LIBS}>
        Eigen3::Eigen
        spdlog::spdlog_header_only
        cereal::cereal
    )
endif()

add_library(stargazer_mvpose
    ${CMAKE_CURRENT_LIST_DIR}/mvpose.cpp
)

target_include_directories(stargazer_mvpose
    PRIVATE
    $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_INCLUDE_DIRS}>
)

target_compile_definitions(stargazer_mvpose
    PUBLIC
    $<$<BOOL:${ENABLE_CUDA}>:USE_CUDA>
)

if (ENABLE_CUDA)
    target_link_libraries(stargazer_mvpose
        glm::glm
        Eigen3::Eigen
        opencv_imgproc
        $<$<BOOL:${ENABLE_ONNXRUNTIME_EXT}>:${ONNXRUNTIME_LIBS}>
        CUDA::cudart
        CUDA::cuda_driver
        nvinfer
    )
else()
    target_link_libraries(stargazer_mvpose
        glm::glm
        Eigen3::Eigen
        opencv_imgproc
    )
endif()


set_target_properties(stargazer_mvpose PROPERTIES
    CXX_STANDARD 20)

add_library(stargazer
    ${CMAKE_CURRENT_LIST_DIR}/parameters.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reconstruction/triangulation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reconstruction/correspondance.cpp
    ${CMAKE_CURRENT_LIST_DIR}/reconstruction/reconstruction.cpp
    ${CMAKE_CURRENT_LIST_DIR}/calibration/bundle_adjust_data.cpp
    ${CMAKE_CURRENT_LIST_DIR}/calibration/calibration.cpp
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
    Eigen3::Eigen
    opencv_imgproc
    $<$<NOT:$<BOOL:${MSVC}>>:pthread>
    $<$<NOT:$<BOOL:${MSVC}>>:stdc++fs>
)

set_target_properties(stargazer PROPERTIES
    CXX_STANDARD 20)
