FROM mcr.microsoft.com/devcontainers/cpp:ubuntu-22.04

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.22.2"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends gfortran bison python3 python3-setuptools libarchive-dev libblas-dev liblapack-dev \
    libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev pkg-config

RUN su vscode -c "${VCPKG_ROOT}/vcpkg install boost-program-options boost-asio boost-graph ceres opencv4[core,highgui,calib3d,contrib] glm grpc glfw3 glad nanoflann nlohmann-json imgui[opengl3-binding,glfw-binding] libjpeg-turbo"

ENV PATH $PATH:${VCPKG_ROOT}/installed/x64-linux
ENV PATH $PATH:${VCPKG_ROOT}/installed/x64-linux/tools/protobuf
ENV PATH $PATH:${VCPKG_ROOT}/installed/x64-linux/tools/grpc

WORKDIR /onnxruntime/

RUN curl -L \
    https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-gpu-1.15.1.tgz \
    -o /tmp/onnxruntime-linux-x64-gpu-1.15.1.tgz

RUN tar -zxvf /tmp/onnxruntime-linux-x64-gpu-1.15.1.tgz --strip-components 1 -C /onnxruntime

ENV ONNXRUNTIME_ROOT /onnxruntime
