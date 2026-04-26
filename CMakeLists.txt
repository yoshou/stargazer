cmake_minimum_required(VERSION 3.18)

project(stargazer C CXX)

enable_testing()

if (WIN32)
else ()
set(CMAKE_C_FLAGS "-Wall -Wextra -pedantic -mavx -march=native")
set(CMAKE_C_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-g3 -Og")
set(CMAKE_C_FLAGS_MINSIZEREL "-Os")

set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -mavx -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g3 -Og")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os")
endif ()

if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ENABLE_ONNXRUNTIME_EXT 1)
else()
    set(ENABLE_ONNXRUNTIME_EXT 0)
endif()

if(${ENABLE_ONNXRUNTIME_EXT})
    set(ONNXRUNTIME_INCLUDE_DIRS $ENV{ONNXRUNTIME_ROOT}/include)
    if(MSVC)
        set(ONNXRUNTIME_LIBS $ENV{ONNXRUNTIME_ROOT}/lib/onnxruntime.lib $ENV{ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.lib $ENV{ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.lib)
    else()
        set(ONNXRUNTIME_LIBS $ENV{ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
    endif()
endif()

# Check required model files (download with: python3 models/download_models.py)
set(DUST3R_ONNX ${CMAKE_SOURCE_DIR}/models/dust3r/dust3r_512x288.onnx)
set(DUST3R_ONNX_DATA ${CMAKE_SOURCE_DIR}/models/dust3r/dust3r_512x288.onnx.data)
if(NOT EXISTS ${DUST3R_ONNX} OR NOT EXISTS ${DUST3R_ONNX_DATA})
    message(FATAL_ERROR "DUSt3R model files not found. Run: python3 models/download_models.py")
endif()

# MASt3R model is optional until generated with scripts/export_mast3r_onnx.py.
set(MAST3R_ONNX ${CMAKE_SOURCE_DIR}/models/mast3r/mast3r_512x288.onnx)
set(MAST3R_ONNX_DATA ${CMAKE_SOURCE_DIR}/models/mast3r/mast3r_512x288.onnx.data)

set(VOXELPOSE_BACKBONE ${CMAKE_SOURCE_DIR}/models/voxelpose/backbone.onnx)
set(VOXELPOSE_PROPOSAL_TRT ${CMAKE_SOURCE_DIR}/models/voxelpose/proposal_v2v_net-fp16.trt)
set(VOXELPOSE_POSE_TRT ${CMAKE_SOURCE_DIR}/models/voxelpose/pose_v2v_net-fp16.trt)
if(NOT EXISTS ${VOXELPOSE_BACKBONE} OR NOT EXISTS ${VOXELPOSE_PROPOSAL_TRT} OR NOT EXISTS ${VOXELPOSE_POSE_TRT})
    message(FATAL_ERROR "VoxelPose model files not found. Run: python3 models/download_models.py")
endif()

set(MVPOSE_DET_TRT ${CMAKE_SOURCE_DIR}/models/mvpose/rtmdet_m_640-8xb32_coco-person-fp16.trt)
set(MVPOSE_DET_ONNX ${CMAKE_SOURCE_DIR}/models/mvpose/rtmdet_m_640-8xb32_coco-person.onnx)
set(MVPOSE_POSE_ONNX ${CMAKE_SOURCE_DIR}/models/mvpose/rtmpose-l_8xb32-270e_coco-wholebody-384x288.onnx)
set(MVPOSE_ASSOC_ONNX ${CMAKE_SOURCE_DIR}/models/mvpose/associator.onnx)
if(NOT EXISTS ${MVPOSE_DET_TRT} OR NOT EXISTS ${MVPOSE_DET_ONNX} OR NOT EXISTS ${MVPOSE_POSE_ONNX} OR NOT EXISTS ${MVPOSE_ASSOC_ONNX})
    message(FATAL_ERROR "MVPose model files not found. Run: python3 models/download_models.py")
endif()

include(third-party/CMakeLists.txt)
include(protos/CMakeLists.txt)
include(app/fonts/CMakeLists.txt)
include(lib/CMakeLists.txt)
include(app/CMakeLists.txt)
include(lib/test/CMakeLists.txt)

include_directories(
    ${CMAKE_SOURCE_DIR}/coalsack/include
)
