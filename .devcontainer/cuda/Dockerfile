FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.22.2"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
    chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

# Install CUDA
ENV NVARCH x86_64

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates\
    && curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${NVARCH}/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm -rf /var/lib/apt/lists/*

ENV NV_CUDA_LIB_VERSION "12.6.3-1"

ENV NV_CUDA_CUDART_DEV_VERSION 12.6.77-1
ENV NV_NVML_DEV_VERSION 12.6.77-1
ENV NV_LIBCUSPARSE_DEV_VERSION 12.5.4.2-1
ENV NV_LIBNPP_DEV_VERSION 12.3.1.54-1
ENV NV_LIBNPP_DEV_PACKAGE libnpp-dev-12-6=${NV_LIBNPP_DEV_VERSION}

ENV NV_LIBCUBLAS_DEV_VERSION 12.6.4.1-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME libcublas-dev-12-6
ENV NV_LIBCUBLAS_DEV_PACKAGE ${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}

ENV NV_CUDA_NSIGHT_COMPUTE_VERSION 12.6.3-1
ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE cuda-nsight-compute-12-6=${NV_CUDA_NSIGHT_COMPUTE_VERSION}

ENV NV_NVPROF_VERSION 12.6.80-1
ENV NV_NVPROF_DEV_PACKAGE cuda-nvprof-12-6=${NV_NVPROF_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-dev-12-6=${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-command-line-tools-12-6=${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-12-6=${NV_CUDA_LIB_VERSION} \
    cuda-libraries-dev-12-6=${NV_CUDA_LIB_VERSION} \
    cuda-nvml-dev-12-6=${NV_NVML_DEV_VERSION} \
    ${NV_NVPROF_DEV_PACKAGE} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    libcusparse-dev-12-6=${NV_LIBCUSPARSE_DEV_VERSION} \
    ${NV_LIBCUBLAS_DEV_PACKAGE} \
    ${NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME}
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# Install cuDNN
ENV NV_CUDNN_VERSION 9.5.1.17-1
ENV NV_CUDNN_PACKAGE_NAME libcudnn9-cuda-12
ENV NV_CUDNN_PACKAGE libcudnn9-cuda-12=${NV_CUDNN_VERSION}
ENV NV_CUDNN_PACKAGE_DEV libcudnn9-dev-cuda-12=${NV_CUDNN_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*

# Install TensorRT
ENV TENSORRT_VERSION 10.4

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

# Install dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends xxd gfortran bison python3 python3-setuptools libarchive-dev libblas-dev liblapack-dev \
    libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev pkg-config

RUN wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc \
    && wget -qO /etc/apt/sources.list.d/lunarg-vulkan-noble.list http://packages.lunarg.com/vulkan/lunarg-vulkan-noble.list \
    && sudo apt update \
    && sudo apt install -y vulkan-sdk mesa-vulkan-drivers

RUN update-alternatives --set c++ /usr/bin/clang++ \
    && update-alternatives --set cc /usr/bin/clang

RUN su vscode -c "${VCPKG_ROOT}/vcpkg install boost-program-options boost-asio boost-graph ceres opencv4[core,highgui,calib3d,contrib,jpeg,dnn] glm grpc glfw3 glad nanoflann nlohmann-json imgui[vulkan-binding,opengl3-binding,glfw-binding] libjpeg-turbo lz4 sqlite3 cereal spdlog"

ENV PATH $PATH:${VCPKG_ROOT}/installed/x64-linux
ENV PATH $PATH:${VCPKG_ROOT}/installed/x64-linux/tools/protobuf
ENV PATH $PATH:${VCPKG_ROOT}/installed/x64-linux/tools/grpc

# Install ONNX Runtime
ENV ONNXRUNTIME_VERSION 1.23.2

WORKDIR /usr/local/onnxruntime/

RUN curl -L \
    https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz \
    -o /tmp/onnxruntime.tgz \
    && tar -zxvf /tmp/onnxruntime.tgz --strip-components 1 -C /usr/local/onnxruntime

ENV ONNXRUNTIME_ROOT /usr/local/onnxruntime
