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

# Restore DUSt3R ONNX data file from chunks if needed
set(DUST3R_ONNX_DATA ${CMAKE_SOURCE_DIR}/models/dust3r_512x288.onnx.data)
if(NOT EXISTS ${DUST3R_ONNX_DATA})
    message(STATUS "Restoring DUSt3R ONNX data from chunks...")
    execute_process(
        COMMAND python3 ${CMAKE_SOURCE_DIR}/models/restore_onnx.py --all ${CMAKE_SOURCE_DIR}/models
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/models
        RESULT_VARIABLE RESTORE_RESULT
    )
    if(NOT RESTORE_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to restore DUSt3R ONNX data")
    endif()
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
