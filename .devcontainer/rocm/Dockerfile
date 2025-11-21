FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="none"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
    chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

# Install ROCm
RUN wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null

RUN cat <<EOF >> /etc/apt/sources.list.d/rocm.list
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.0 noble main
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/7.0/ubuntu noble main
EOF

RUN cat <<EOF >> /etc/apt/preferences.d/rocm-pin-600
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
EOF

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends rocm

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

# Install ONNX Runtime
ENV ONNXRUNTIME_VERSION=1.23.2

WORKDIR /usr/local/onnxruntime/

RUN curl -L \
    https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz \
    -o /tmp/onnxruntime.tgz \
    && tar -zxvf /tmp/onnxruntime.tgz --strip-components 1 -C /usr/local/onnxruntime

ENV ONNXRUNTIME_ROOT=/usr/local/onnxruntime
