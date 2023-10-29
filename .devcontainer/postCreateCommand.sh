#!/bin/sh

sudo dpkg -i nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-11.8_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-11.8/nv-tensorrt-local-0628887B-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo dpkg -i /var/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-11.8/libnvinfer8_8.6.1.6-1+cuda11.8_amd64.deb
sudo dpkg -i /var/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-11.8/libnvinfer-plugin8_8.6.1.6-1+cuda11.8_amd64.deb
sudo dpkg -i /var/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-11.8/libnvonnxparsers8_8.6.1.6-1+cuda11.8_amd64.deb