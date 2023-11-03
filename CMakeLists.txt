cmake_minimum_required(VERSION 3.18)

project(stargazer C CXX)

set(CMAKE_CXX_STANDARD 17)

if (WIN32)
else ()
set(CMAKE_C_FLAGS " -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(CMAKE_C_FLAGS_RELEASE "-O2")

set(CMAKE_CXX_FLAGS "-pg -Wall -Wextra -Wno-unused-parameter -mavx -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-pg -O0 -g -mavx -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-pg -O2 -g -mavx -march=native")
endif ()

if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ENABLE_ONNXRUNTIME_EXT 1)
else()
    set(ENABLE_ONNXRUNTIME_EXT 0)
endif()

if(${ENABLE_ONNXRUNTIME_EXT})
    set(ONNXRUNTIME_INCLUDE_DIRS $ENV{ONNXRUNTIME_ROOT}/include)
    set(ONNXRUNTIME_LIBS $ENV{ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
endif()

include(third-party/CMakeLists.txt)
include(protos/CMakeLists.txt)
include(fonts/CMakeLists.txt)
include(lib/CMakeLists.txt)
include(app/CMakeLists.txt)

include_directories(
    ${CMAKE_SOURCE_DIR}/coalsack/include
)
