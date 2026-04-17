# Pipeline Configuration Reference

This document describes the JSON format used to configure the stargazer processing pipeline.
The pipeline configuration file is passed to `stargazer_viewer` or `stargazer_grpc_server` at
startup.

---

## Top-Level Structure

A configuration file contains up to three top-level keys:

```
{
  "subgraphs": [ ... ],
  "pipelines": { ... },
  "pipeline":  "pipeline_name"
}
```

| Key | Required | Description |
|-----|----------|-------------|
| `subgraphs` | No | Reusable named subgraph templates |
| `pipelines` | Yes | Named pipeline definitions |
| `pipeline` | Yes | Name of the pipeline to activate on startup |

---

## Subgraph Definition

A subgraph is a reusable fragment of a processing graph.

```
{
  "name": "playback_color_subgraph",
  "nodes": [ ... ],
  "outputs": [ "decode_jpeg" ],
  "db_path": "../data/capture.db",
  "width": 960,
  "height": 540,
  "fps": 30
}
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique identifier for the subgraph template |
| `nodes` | array | Node definitions |
| `outputs` | array of strings | Node names whose outputs are exposed outside the subgraph |
| `subgraphs` | array | Nested subgraph instantiations with optional `extends` |
| `db_path` | string | Default database path — consumed by `load_blob`, `load_marker`, `load_panoptic`, `dump_se3`, and `dump_reconstruction` nodes |
| `width`, `height` | int | Stored in the subgraph params object; available for any node that reads them, but **not** automatically passed to `load_blob` or `load_marker` by `graph_builder.cpp` |
| `fps` | int | Propagated to `load_panoptic` only; not consumed by `load_blob` or `load_marker` |
| `id` | string | Device/entity ID propagated to `load_parameter` nodes |

Extra keys at the subgraph level are propagated as default `properties` to nodes of matching
types within the subgraph.

### Subgraph Inheritance

A subgraph can extend one or more templates using the `extends` key:

```json
{
  "name": "camera101",
  "extends": ["playback_color_subgraph"],
  "nodes": [
    { "name": "load_blob", "topic_name": "image_101" }
  ],
  "id": "000000000026"
}
```

When `extends` is specified:
- All nodes from the referenced template are included.
- Nodes listed in `nodes` **override** the corresponding node in the template (matched by
  `name`).  Unspecified fields are inherited.
- Extra fields (`id`, `db_path`, etc.) at the instance level override template defaults.

---

## Pipeline Definition

A pipeline definition assembles subgraph instances into a complete processing graph.

```json
"pipelines": {
  "calibration_dust3r": {
    "subgraphs": [
      {
        "name": "calibration",
        "extends": ["calibration_dust3r_subgraph"]
      }
    ]
  }
}
```

---

## Node Definition

Each element of a `nodes` array describes one graph node.

```
{
  "name": "decode_jpeg",
  "type": "decode_jpeg",
  "inputs": {
    "default": "load_blob"
  },
  "properties": [ ... ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Node instance name; used as connection target by other nodes |
| `type` | string | Node type key (see [node-reference.md](node-reference.md)) |
| `inputs` | object | Maps input port names to source node names |
| `properties` | array | UI-visible property descriptors (see below) |

Any additional fields not listed above are treated as **node parameters** and are passed to
the corresponding node setter as determined by `graph_builder.cpp`.  For example:

```json
{ "name": "load_panoptic", "type": "load_panoptic", "topic_name": "hd_camera_01", "fps": 30 }
```

### Inputs Syntax

The `inputs` object maps an input port name to a source node path:

```
"<input_port>": "<source_node_name>"
```

For nodes inside a subgraph, cross-subgraph references use a slash-separated path:

```
"camera.camera101": "camera101/decode_jpeg"
```

### Property Descriptor

Properties expose node state to the viewer UI.

```json
{
  "id": "image",
  "label": "Image",
  "source_key": "image",
  "target": "image",
  "resource_kind": "raw"
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Internal identifier |
| `label` | Yes | Human-readable label shown in the UI |
| `source_key` | Yes | Key on the node that provides the value |
| `target` | No | UI rendering target (e.g. `"image"`) |
| `resource_kind` | No | Resource format hint (`"raw"` for binary image data) |
| `format` | No | Value format hint (`"integer"`, `"float"`, etc.) |
| `default_value` | No | Initial value before the node produces output |

---

## Practical Examples

### Example 1 — Basic Blob Playback (`capture.json`)

The `playback_subgraph` template in `config/capture.json` demonstrates the simplest
source-to-display chain: read a blob from SQLite, decode it as JPEG, and expose the
image as a UI property.

```json
{
  "name": "playback_subgraph",
  "nodes": [
    { "name": "load_blob",    "type": "load_blob" },
    {
      "name": "decode_jpeg",
      "type": "decode_jpeg",
      "inputs": { "default": "load_blob" }
    },
    {
      "name": "callback_image",
      "type": "image_property",
      "inputs": { "default": "decode_jpeg" },
      "properties": [
        {
          "id": "image",
          "label": "Image",
          "source_key": "image",
          "target": "image",
          "resource_kind": "raw"
        }
      ]
    }
  ],
  "outputs": ["callback_image"],
  "db_path": "../data/data_20221004_1/capture.db",
  "width": 820,
  "height": 616,
  "fps": 90
}
```

Key points:
- `db_path` at the subgraph level is inherited by the `load_blob` node and controls which
  database file is opened.  `width`, `height`, and `fps` are stored in the subgraph params
  but are **not** wired to `load_blob` or `load_marker` by `graph_builder.cpp`; they are
  present for other nodes (such as camera capture nodes) that do consume them.
- `outputs` exposes `callback_image` so that other subgraphs can wire into its output.

---

### Example 2 — DUSt3R Calibration with Subgraph Inheritance (`calibration_dust3r.json`)

`config/calibration_dust3r.json` shows how the `extends` mechanism creates per-camera
subgraph instances from a single template.

```json
{
  "name": "playback_color_subgraph",
  "nodes": [
    { "name": "load_blob",    "type": "load_blob" },
    {
      "name": "decode_jpeg",
      "type": "decode_jpeg",
      "inputs": { "default": "load_blob" }
    }
  ],
  "outputs": ["decode_jpeg"],
  "db_path": "../data/data_20250713_1/calibrate.db",
  "height": 540,
  "width": 960,
  "fps": 30
}
```

Each camera instance overrides only `topic_name` in `load_blob` and sets `id`:

```json
{
  "name": "camera101",
  "extends": ["playback_color_subgraph"],
  "nodes": [
    { "name": "load_blob", "topic_name": "image_101" }
  ],
  "id": "000000000026"
}
```

The parent calibration subgraph then wires five camera instances into a single
`dust3r_calibration` node:

```json
{
  "name": "approximate_time_sync",
  "type": "approximate_time_sync",
  "inputs": {
    "camera101": "camera101/decode_jpeg",
    "camera102": "camera102/decode_jpeg",
    "camera103": "camera103/decode_jpeg",
    "camera104": "camera104/decode_jpeg",
    "camera105": "camera105/decode_jpeg"
  },
  "interval": 33.333
},
{
  "name": "dust3r_pose",
  "type": "dust3r_calibration",
  "inputs": {
    "default":          "approximate_time_sync",
    "camera.camera101": "load_camera101",
    "camera.camera102": "load_camera102",
    "camera.camera103": "load_camera103",
    "camera.camera104": "load_camera104",
    "camera.camera105": "load_camera105",
    "estimate":         "estimate_action"
  },
  "model_path": "../models/dust3r_512x288.onnx"
}
```

Key points:
- Cross-subgraph references use the `<subgraph_name>/<node_name>` path syntax.
- `model_path` is a node parameter passed directly to `dust3r_calibration_node::set_model_path()`.
- `estimate_action` is an `action` node whose `action_id` lets the viewer button trigger
  the calibration.

---

### Example 3 — Multi-Camera 3D Reconstruction (`reconstruction_image.json`)

`config/reconstruction_image.json` shows the full reconstruction pipeline.  Per-camera
subgraph instances are each wired to an `epipolar_reconstruct_node` (or one of the
neural-network variants) after time-synchronisation.

The `playback_color_subgraph` template provides the source and display nodes for each camera.
A camera-specific override only changes `topic_name` on `load_blob` and the `db_path`
default.  The reconstruction subgraph then assembles all cameras:

```json
{
  "name": "load_marker",
  "type": "load_marker"
},
{
  "name": "display_image",
  "type": "image_property",
  "inputs": { "default": "decode_jpeg" },
  "properties": [
    {
      "id": "image",
      "label": "Image",
      "source_key": "image",
      "target": "image",
      "resource_kind": "raw"
    }
  ]
}
```

Key points:
- `load_marker` is a source node with no inputs; its `db_path` and `topic_name` come from
  subgraph-level defaults.
- `image_property` nodes expose their cached frame to the viewer via the `image` property.
- The outer pipeline definition selects which subgraph to activate based on the
  `"pipeline"` key at the top level of the JSON.
