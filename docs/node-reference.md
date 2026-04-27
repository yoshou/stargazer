# Node Parameter Reference

This document lists all node type keys recognised by `graph_builder.cpp` together with
their configurable parameters.  Parameters map to the corresponding `set_xxx()` setter
on the node class.

Parameters are set either directly as top-level fields on the node JSON object or, where
a default is provided at the subgraph level, inherited from the enclosing subgraph.

---

## Calibration Nodes

### `dust3r_calibration`

Jointly estimates camera intrinsics and extrinsics using the DUSt3R model.
See [`dust3r_calibration_node`](../lib/nodes/calibration/dust3r_calibration_node.hpp).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | `""` | Path to the DUSt3R ONNX model file |

### `dust3r_pose_estimation`

Estimates relative camera poses given known intrinsics using the DUSt3R model.
See [`dust3r_pose_node`](../lib/nodes/calibration/dust3r_pose_node.hpp).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | `""` | Path to the DUSt3R ONNX model file |

### `mast3r_calibration`

Jointly estimates camera intrinsics and extrinsics using the MASt3R model.
See [`mast3r_calibration_node`](../lib/nodes/calibration/mast3r_calibration_node.hpp).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | `""` | Path to the MASt3R ONNX model file |

### `extrinsic_calibration`

Bundle-adjustment extrinsic (and optionally intrinsic) calibration from 2D observations.
See [`extrinsic_calibration_node`](../lib/nodes/calibration/extrinsic_calibration_node.hpp).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `only_extrinsic` | bool | `false` | Hold intrinsics fixed during optimisation |
| `robust` | bool | `false` | Enable outlier-robust fitting |

### `intrinsic_calibration`

Calibrates camera intrinsics from checkerboard observations.
See [`intrinsic_calibration_node`](../lib/nodes/calibration/intrinsic_calibration_node.hpp).

_(no configurable parameters)_

### `pattern_board_calibration_target_detector`

Detects ordered calibration board corners from blob positions.
See [`pattern_board_calibration_target_detector_node`](../lib/nodes/calibration/pattern_board_calibration_target_detector_node.hpp).

_(no configurable parameters)_

### `scene_calibration`

Computes scene coordinate axis from ArUco marker triangulation.
See [`scene_calibration_node`](../lib/nodes/calibration/scene_calibration_node.hpp).

_(no configurable parameters)_

### `three_point_bar_calibration_target_detector`

Detects three-point bar calibration target from blob positions.
See [`three_point_bar_calibration_target_detector_node`](../lib/nodes/calibration/three_point_bar_calibration_target_detector_node.hpp).

_(no configurable parameters)_

---

## Core Nodes

### `contrail_render`

Renders accumulated 2D point trajectories onto an image.
See [`contrail_render_node`](../lib/nodes/core/contrail_render_node.hpp).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `camera_name` | string | `""` | Identifier for the source camera |
| `width` | int | `0` | Output image width in pixels |
| `height` | int | `0` | Output image height in pixels |

### `feature_render`

Renders reconstruction feature heatmaps as colour images.
See [`feature_render_node`](../lib/nodes/core/feature_render_node.hpp).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `camera_name` | string | `""` | Identifier for the source camera |
| `width` | int | `0` | Output image width in pixels |
| `height` | int | `0` | Output image height in pixels |

### `gate`

Conditionally passes or suppresses messages.
See [`gate_node`](../lib/nodes/core/gate_node.hpp).

_(no configurable parameters — the enabled flag is set programmatically)_

### `grpc_server`

Exposes sensor data over a gRPC interface.
See [`grpc_server_node`](../lib/nodes/core/grpc_server_node.hpp).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `address` | string | `""` | Server listen address, e.g. `"0.0.0.0:50051"` |

### `image_property`

Caches and exposes images as UI properties.
See [`image_property_node`](../lib/nodes/core/image_property_node.hpp).

_(no configurable parameters)_

### `keypoint_to_float2_map`

Converts keypoint fields to float2 list fields inside an object message.
See [`keypoint_to_float2_map_node`](../lib/nodes/core/keypoint_to_float2_map_node.hpp).

_(no configurable parameters)_

### `marker_property`

Caches and exposes 3D marker positions as UI properties.
See [`marker_property_node`](../lib/nodes/core/marker_property_node.hpp).

_(no configurable parameters)_

### `object_map`

Routes object fields to individual output ports.
See [`object_map_node`](../lib/nodes/core/object_map_node.hpp).

_(no configurable parameters — outputs are registered programmatically)_

### `object_mux`

Emits each field of an object message followed by a sentinel.
See [`object_mux_node`](../lib/nodes/core/object_mux_node.hpp).

_(no configurable parameters)_

### `object_to_frame`

Wraps an object message in a frame message with frame number and timestamp.
See [`object_to_frame_node`](../lib/nodes/core/object_to_frame_node.hpp).

_(no configurable parameters)_

### `unframe_image_fields`

Converts framed image fields inside an object to bare image messages.
See [`unframe_image_fields_node`](../lib/nodes/core/unframe_image_fields_node.hpp).

_(no configurable parameters)_

---

## I/O Nodes

> **Note:** `dump_blob` and `dump_keypoint` are managed by the coalsack library and are not
> part of the stargazer `node_type` enum or `graph_builder.cpp`.  They are not configurable
> through the stargazer pipeline JSON.  Only the nodes listed below are supported.

### `dump_reconstruction`

Records reconstruction results to a SQLite database.
See [`dump_reconstruction_node`](../lib/nodes/io/dump_reconstruction_node.hpp).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | string | `""` | Path to the SQLite database file |
| `topic_name` | string | `""` | Topic name written to the `topics` table |

### `dump_se3`

Records SE3 pose data to a SQLite database.
See [`dump_se3_node`](../lib/nodes/io/dump_se3_node.hpp).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | string | `""` | Path to the SQLite database file |
| `topic_name` | string | `""` | Topic name written to the `topics` table |

### `load_blob`

Plays back blob frames from a SQLite database.
See [`load_blob_node`](../lib/nodes/io/load_blob_node.hpp).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | string | `""` | Path to the SQLite database file |
| `topic_name` | string | `""` | Topic name to read from the database |

### `load_marker`

Plays back 3D marker positions from a SQLite database.
See [`load_marker_node`](../lib/nodes/io/load_marker_node.hpp).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | string | `""` | Path to the SQLite database file |
| `topic_name` | string | `""` | Topic name to read from the database |

### `load_panoptic`

Plays back Panoptic dataset JPEG images from a SQLite database.
See [`load_panoptic_node`](../lib/nodes/io/load_panoptic_node.hpp).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | string | `""` | Path to the SQLite database file |
| `topic_name` | string | `""` | Topic name to read from the database |
| `fps` | int | `30` | Playback frame rate in frames per second |

### `load_parameter`

Reads camera or scene parameters from a parameter resource.
See [`load_parameter_node`](../lib/nodes/io/load_parameter_node.hpp).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `id` | string | `""` | Identifier of the parameter resource to watch |

### `store_parameter`

Writes received parameters to a parameter resource.
See [`store_parameter_node`](../lib/nodes/io/store_parameter_node.hpp).

_(no configurable parameters)_

---

## Reconstruction Nodes

### `epipolar_reconstruction`

Triangulates 3D points from multi-camera 2D correspondences using epipolar geometry.
See [`epipolar_reconstruct_node`](../lib/nodes/reconstruct/epipolar_reconstruct_node.hpp).

_(no configurable parameters)_

### `mvpose_reconstruction`

Multi-view human pose estimation using the MVPose neural network model.
See [`mvpose_reconstruct_node`](../lib/nodes/reconstruct/mvpose_reconstruct_node.hpp).

_(no configurable parameters)_

### `mvp_reconstruction`

Multi-view pose estimation using the MVP model (Panoptic dataset variant).
See [`mvp_reconstruct_node`](../lib/nodes/reconstruct/mvp_reconstruct_node.hpp).

_(no configurable parameters)_

### `voxelpose_reconstruction`

Voxel-space 3D human pose reconstruction using the VoxelPose neural network model.
See [`voxelpose_reconstruct_node`](../lib/nodes/reconstruct/voxelpose_reconstruct_node.hpp).

_(no configurable parameters)_

### `reconstruction_result_markers`

Extracts 3D point positions from a reconstruction result message.
See [`reconstruction_result_markers_node`](../lib/nodes/reconstruct/reconstruction_result_markers_node.hpp).

_(no configurable parameters)_
