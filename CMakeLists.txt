cmake_minimum_required(VERSION 3.18)

project(stargazer C CXX)

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

include(third-party/CMakeLists.txt)
include(protos/CMakeLists.txt)
include(fonts/CMakeLists.txt)
include(lib/CMakeLists.txt)
include(app/CMakeLists.txt)

include_directories(
    ${CMAKE_SOURCE_DIR}/coalsack/include
)
