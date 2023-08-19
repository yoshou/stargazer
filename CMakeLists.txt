cmake_minimum_required(VERSION 3.1.0)

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

include(third-party/CMakeLists.txt)
include(coalsack/CMakeLists.txt)
include(protos/CMakeLists.txt)
include(fonts/CMakeLists.txt)
include(lib/CMakeLists.txt)
include(reconstruction_app/CMakeLists.txt)
