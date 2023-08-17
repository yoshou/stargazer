FROM mcr.microsoft.com/devcontainers/cpp:0-ubuntu-22.04

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="3.22.2"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends gfortran bison python3 python3-setuptools libarchive-dev libblas-dev liblapack-dev

RUN su vscode -c "${VCPKG_ROOT}/vcpkg install boost-program-options boost-asio opencv4[core,sfm] glm eigen3 suitesparse ceres[suitesparse] grpc yaml-cpp"

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev pkg-config
    
RUN su vscode -c "${VCPKG_ROOT}/vcpkg install glfw3 glew glad"

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends autoconf automake libtool libudev-dev
    
RUN su vscode -c "${VCPKG_ROOT}/vcpkg install realsense2"
    
RUN su vscode -c "${VCPKG_ROOT}/vcpkg install pcl"

WORKDIR /fbxsdk/

RUN curl -L \
    https://damassets.autodesk.net/content/dam/autodesk/www/adn/fbx/2020-0-1/fbx202001_fbxsdk_linux.tar.gz \
    -o /tmp/fbx202001_fbxsdk_linux.tar.gz

RUN mkdir -p /fbxsdk/install

RUN tar -zxvf /tmp/fbx202001_fbxsdk_linux.tar.gz -C /tmp

RUN echo "yes\nn" | /tmp/fbx202001_fbxsdk_linux /fbxsdk

RUN rm /tmp/fbx202001_fbxsdk_linux.tar.gz
RUN rm -rf /tmp/fbx202001_fbxsdk_linux

RUN chmod -R 755 /fbxsdk

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends python3-pip

RUN pip3 install -U pip \
    && pip3 install --no-cache-dir numpy

ENV PATH $PATH:${VCPKG_ROOT}/installed/x64-linux
ENV PATH $PATH:${VCPKG_ROOT}/installed/x64-linux/tools/protobuf
ENV PATH $PATH:${VCPKG_ROOT}/installed/x64-linux/tools/grpc
