FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG INSTALL_ZSH="true"
ARG UPGRADE_PACKAGES="true"
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

COPY ./common-debian.sh /tmp/
RUN yes | unminimize 2>&1 \ 
    && bash /tmp/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" "true" "true" \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install build-essential cmake cppcheck valgrind clang lldb llvm gdb \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

ENV VCPKG_ROOT=/usr/local/vcpkg \
    VCPKG_DOWNLOADS=/usr/local/vcpkg-downloads
ENV PATH="${PATH}:${VCPKG_ROOT}"

COPY ./install-vcpkg.sh /tmp/
RUN /tmp/install-vcpkg.sh ${USERNAME} \
    && rm -f /tmp/install-vcpkg.sh

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.22.2"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

# Install TensorRT
ENV NVARCH=x86_64
    
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates\
    && curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${NVARCH}/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm -rf /var/lib/apt/lists/*

ENV TENSORRT_VERSION=8.6

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && TENSORRT_MAJOR_VERSION=`echo ${TENSORRT_VERSION} | cut -d '.' -f 1` \
    && apt-get install -y libnvinfer${TENSORRT_MAJOR_VERSION}=${TENSORRT_VERSION}.* \
                       libnvinfer-plugin${TENSORRT_MAJOR_VERSION}=${TENSORRT_VERSION}.* \
                       libnvinfer-dev=${TENSORRT_VERSION}.* \
                       libnvinfer-headers-dev=${TENSORRT_VERSION}.* \
                       libnvinfer-headers-plugin-dev=${TENSORRT_VERSION}.* \
                       libnvinfer-plugin-dev=${TENSORRT_VERSION}.* \
                       libnvonnxparsers${TENSORRT_MAJOR_VERSION}=${TENSORRT_VERSION}.* \
                       libnvonnxparsers-dev=${TENSORRT_VERSION}.*

# Install ONNX Runtime
ENV ONNXRUNTIME_VERSION=1.17.3

WORKDIR /onnxruntime/

RUN curl -L \
    https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-cuda12-${ONNXRUNTIME_VERSION}.tgz \
    -o /tmp/onnxruntime.tgz \
    && tar -zxvf /tmp/onnxruntime.tgz --strip-components 1 -C /onnxruntime

ENV ONNXRUNTIME_ROOT=/onnxruntime

# Install dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends gfortran bison python3 python3-setuptools libarchive-dev libblas-dev liblapack-dev \
    libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev pkg-config

RUN update-alternatives --set c++ /usr/bin/clang++ \
    && update-alternatives --set cc /usr/bin/clang

RUN su vscode -c "${VCPKG_ROOT}/vcpkg install boost-asio boost-graph ceres opencv4[core,highgui,calib3d,contrib,jpeg,dnn] glm grpc nanoflann nlohmann-json libjpeg-turbo sqlite3 cereal spdlog glfw3 glad imgui[opengl3-binding,glfw-binding]"

ENV PATH=$PATH:${VCPKG_ROOT}/installed/x64-linux
ENV PATH=$PATH:${VCPKG_ROOT}/installed/x64-linux/tools/protobuf
ENV PATH=$PATH:${VCPKG_ROOT}/installed/x64-linux/tools/grpc

USER ${USERNAME}
