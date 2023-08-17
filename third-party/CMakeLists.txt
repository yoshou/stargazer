if (WIN32)
    set(VCPKG_ROOT "C:/Users/yoshi/Desktop/marionette/third-party/windows/vcpkg")
    set(CMAKE_PREFIX_PATH "${VCPKG_ROOT}/installed/x64-windows")
    set(CMAKE_LIBRARY_PATH "${CMAKE_PREFIX_PATH}/lib")

    find_package(OpenCV REQUIRED)
    find_package(Ceres CONFIG REQUIRED)

    set_target_properties(glog::glog PROPERTIES
        INTERFACE_LINK_LIBRARIES "gflags"
    )

    add_custom_target(tinygltf)
    add_custom_target(nuklear)
    add_custom_target(nanoflann)
    add_custom_target(glm)
    add_custom_target(eigen)
    # add_custom_target(json)

    include_directories(
        ${CMAKE_SOURCE_DIR}/third-party/windows/tinygltf
        ${CMAKE_SOURCE_DIR}/third-party/windows/nuklear
        ${CMAKE_SOURCE_DIR}/third-party/windows/nanoflann/include
        ${CMAKE_SOURCE_DIR}/third-party/windows/glm
        ${CMAKE_SOURCE_DIR}/third-party/windows/eigen
        # ${CMAKE_SOURCE_DIR}/third-party/windows/json/include
        ${VCPKG_ROOT}/installed/x64-windows/include
    )

    find_package(Boost REQUIRED COMPONENTS program_options)
    find_package(yaml-cpp REQUIRED)
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(flann CONFIG REQUIRED)
    find_package(PCL CONFIG REQUIRED)
    find_package(nanoflann CONFIG REQUIRED)
    # find_package(unofficial-nuklear CONFIG REQUIRED)
    find_package(glm CONFIG REQUIRED)
    find_package(gRPC CONFIG REQUIRED)
else()
    set(Boost_INCLUDE_DIR /usr/local/vcpkg/installed/x64-linux/include)
    find_package(OpenCV 4 REQUIRED)
    find_package(realsense2 REQUIRED)
    find_package(Boost REQUIRED COMPONENTS program_options)
    find_package(Eigen3 REQUIRED)
    find_package(PCL REQUIRED)
    find_package(yaml-cpp REQUIRED)

    find_package(protobuf 3.17 REQUIRED)
    find_package(gRPC REQUIRED)

    include_directories(
        SYSTEM ${OpenCV_INCLUDE_DIRS}
        SYSTEM ${REALSENSE2_INCLUDE_DIRS}
        SYSTEM ${Boost_INCLUDE_DIRS}
        SYSTEM ${EIGEN3_INCLUDE_DIR}
        SYSTEM ${PCL_INCLUDE_DIRS}
        SYSTEM ${CMAKE_BINARY_DIR}/libjpeg-turbo/include
        SYSTEM ${CMAKE_BINARY_DIR}/third-party/glm
        SYSTEM ${CMAKE_BINARY_DIR}/third-party/cereal/include
        SYSTEM ${CMAKE_BINARY_DIR}/third-party/spdlog/include
        SYSTEM ${CMAKE_BINARY_DIR}/third-party/json/include
        SYSTEM ${PROTOBUF_INCLUDE_DIR}
        ./src
        ./coalsack/src
    )

    set(GLEW_INCLUDE_DIR /usr/local/vcpkg/installed/x64-linux/include)
    set(GLEW_LIBRARY /usr/local/vcpkg/installed/x64-linux/lib/libGLEW.a)
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)

    include(ExternalProject)

    if(NOT TARGET libjpeg-turbo)
        ExternalProject_Add(libjpeg-turbo
            PREFIX libjpeg-turbo
            GIT_REPOSITORY "https://github.com/libjpeg-turbo/libjpeg-turbo.git"
            GIT_TAG "2.1.0"
            SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/libjpeg-turbo"
            CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/libjpeg-turbo"
            "-DCMAKE_GENERATOR=${CMAKE_GENERATOR}"
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
            "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
        )
    endif()

    if(NOT TARGET glm)
        ExternalProject_Add(glm
            PREFIX glm
            GIT_REPOSITORY https://github.com/g-truc/glm.git
            SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/glm"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
        )
    endif()

    if(NOT TARGET spdlog)
        ExternalProject_Add(spdlog
            PREFIX spdlog
            GIT_REPOSITORY https://github.com/gabime/spdlog.git
            GIT_TAG "v1.8.2"
            SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/spdlog"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
        )
    endif()

    if(NOT TARGET googletest)
        ExternalProject_Add(googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG release-1.12.1
            SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/googletest"
            BINARY_DIR "${CMAKE_BINARY_DIR}/third-party/googletest-build"
            INSTALL_COMMAND ""
        )
    endif()

    if(NOT TARGET json)
        ExternalProject_Add(json
            PREFIX json
            GIT_REPOSITORY https://github.com/nlohmann/json
            GIT_TAG v3.10.2
            SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/json"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
        )
        target_include_directories(json
            PUBLIC ${CMAKE_BINARY_DIR}/third-party/json/include
        )
    endif()

    if(NOT TARGET yaml-cpp)
        ExternalProject_Add(yaml-cpp
            PREFIX yaml-cpp
            GIT_REPOSITORY https://github.com/jbeder/yaml-cpp
            GIT_TAG yaml-cpp-0.7.0
            SOURCE_DIR "${CMAKE_SOURCE_DIR}/third-party/yaml-cpp/"
            INSTALL_DIR "${CMAKE_BINARY_DIR}/yaml-cpp"
            CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/yaml-cpp"
            "-DCMAKE_GENERATOR=${CMAKE_GENERATOR}"
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
            "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
        )
    endif()

    if(NOT TARGET tinygltf)
        ExternalProject_Add(tinygltf
            PREFIX tinygltf
            GIT_REPOSITORY https://github.com/syoyo/tinygltf
            GIT_TAG v2.5.0
            SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/tinygltf"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
        )
    endif()

    if(NOT TARGET nuklear)
        ExternalProject_Add(nuklear
            PREFIX nuklear
            GIT_REPOSITORY https://github.com/Immediate-Mode-UI/Nuklear
            GIT_TAG master
            SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/nuklear"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
        )
    endif()

    if(NOT TARGET nanoflann)
        ExternalProject_Add(nanoflann
            PREFIX nanoflann
            GIT_REPOSITORY https://github.com/jlblancoc/nanoflann
            GIT_TAG v1.3.2
            SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/nanoflann"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
        )
    endif()

    include_directories(
        ${CMAKE_BINARY_DIR}/third-party/tinygltf
        ${CMAKE_BINARY_DIR}/third-party/nuklear
        ${CMAKE_BINARY_DIR}/third-party/nanoflann/include
    )

    link_directories(
        ${CMAKE_BINARY_DIR}/libjpeg-turbo/lib
    )

    set_target_properties(glog::glog PROPERTIES
        INTERFACE_LINK_LIBRARIES "gflags"
    )

    set_target_properties(multiview PROPERTIES
        INTERFACE_LINK_LIBRARIES "gflags"
    )
endif()
