# stargazer

Multi-view 3D reconstruction and pose estimation system

## Install

```console
$ git clone --recurse-submodules https://github.com/yoshou/stargazer.git
$ cd stargazer && mkdir build && cd build
$ cmake .. && make -j
```

## Dependencies

* [OpenCV](https://github.com/opencv/opencv)
* [Eigen3](https://gitlab.com/libeigen/eigen)
* [Ceres Solver](http://ceres-solver.org)
* [ONNX Runtime](https://github.com/microsoft/onnxruntime)
* [gRPC](https://github.com/grpc/grpc)
* [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/)
* [GLFW](https://www.glfw.org)
* [Dear ImGui](https://github.com/ocornut/imgui)
* [GLM](https://github.com/g-truc/glm)
* [Boost](https://www.boost.org)
* [spdlog](https://github.com/gabime/spdlog)
* [cereal](https://github.com/USCiLab/cereal)
* [coalsack](https://github.com/yoshou/coalsack)

## Usage

The processing pipeline is defined as a node graph in a JSON configuration file.

```json
{
    "subgraphs": [
        {
            "name": "my_subgraph",
            "nodes": [
                { "name": "load_blob", "type": "load_blob" },
                { "name": "decode_jpeg", "type": "decode_jpeg",
                  "inputs": { "default": "load_blob" } },
                { "name": "approximate_time_sync", "type": "approximate_time_sync",
                  "inputs": { "camera0": "decode_jpeg" } },
                { "name": "dust3r_pose", "type": "dust3r_calibration",
                  "inputs": { "default": "approximate_time_sync" },
                  "model_path": "models/dust3r_512x288.onnx" }
            ]
        }
    ],
    "pipelines": {
        "my_pipeline": {
            "subgraphs": [
                { "name": "main", "extends": ["my_subgraph"] }
            ]
        }
    },
    "pipeline": "my_pipeline"
}
```

#### Run the GUI viewer

```console
$ ./stargazer_viewer --config ../config/calibration_dust3r.json
```

#### Run the headless gRPC server

```console
$ ./stargazer_grpc_server --config ../config/calibration_dust3r.json
```

#### Control the pipeline via CLI

```console
$ ../scripts/pipeline_ctl.py start
$ ../scripts/pipeline_ctl.py status
$ ../scripts/pipeline_ctl.py stop
```
