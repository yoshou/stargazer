from __future__ import annotations

import sys

from pipeline import Config, camera_id, config_dict, image_properties, node, pipeline_def, prop, require_output_path, subgraph


COLOR_CAMERAS = [
    {"suffix": 101, "id": 26, "address": "192.168.0.26"},
    {"suffix": 102, "id": 27, "address": "192.168.0.27"},
    {"suffix": 103, "id": 28, "address": "192.168.0.28"},
    {"suffix": 104, "id": 29, "address": "192.168.0.29"},
    {"suffix": 105, "id": 30, "address": "192.168.0.30"},
]


def playback_color_subgraph() -> dict:
    return subgraph(
        "playback_color_subgraph",
        db_path="../data/data_20250713_1/calibrate.db",
        height=540,
        width=960,
        fps=30,
        nodes=[
            node("load_blob", "load_blob"),
            node("decode_jpeg", "decode_jpeg", inputs={"default": "load_blob"}),
            node(
                "display_image",
                "image_property",
                inputs={"default": "decode_jpeg"},
                properties=image_properties(include_received=True),
            ),
            node("load_marker", "load_marker"),
        ],
        outputs=["load_marker"],
    )


def intrinsic_calibration_properties() -> list[dict]:
    return [
        prop("collected", label="Collected", source_key="collected", format="integer", default_value=0),
        prop("rms", label="rms", source_key="rms", format="fixed3", default_value=0.0),
        prop("fx", label="fx", source_key="intrinsics.fx", format="fixed3", default_value=0.0),
        prop("fy", label="fy", source_key="intrinsics.fy", format="fixed3", default_value=0.0),
        prop("cx", label="cx", source_key="intrinsics.cx", format="fixed3", default_value=0.0),
        prop("cy", label="cy", source_key="intrinsics.cy", format="fixed3", default_value=0.0),
        prop("k0", label="k0", source_key="intrinsics.k0", format="fixed3", default_value=0.0),
        prop("k1", label="k1", source_key="intrinsics.k1", format="fixed3", default_value=0.0),
        prop("k2", label="k2", source_key="intrinsics.k2", format="fixed3", default_value=0.0),
        prop("p0", label="p0", source_key="intrinsics.p0", format="fixed3", default_value=0.0),
        prop("p1", label="p1", source_key="intrinsics.p1", format="fixed3", default_value=0.0),
    ]


def intrinsic_calibration_subgraph() -> dict:
    return subgraph(
        "intrinsic_calibration_subgraph",
        nodes=[
            node("load_camera101", "load_parameter", id=camera_id(26)),
            node("frame_number_numbering", "frame_number_numbering"),
            node("object_mux", "object_mux", inputs={"default": "frame_number_numbering"}),
            node("calibrate_action", "action", action_id="calibrate", label="Calibrate", icon="refresh"),
            node(
                "intrinsic_calibration",
                "intrinsic_calibration",
                properties=intrinsic_calibration_properties(),
                inputs={
                    "calibrate": "calibrate_action",
                    "default": "object_mux",
                    "camera": "load_camera101",
                },
            ),
            node("store_camera", "store_parameter", inputs={"default": "intrinsic_calibration"}),
        ],
    )


def playback_color_camera_subgraph(camera: dict) -> dict:
    suffix = camera["suffix"]
    return subgraph(
        f"playback_color_camera{suffix}_subgraph",
        db_path="../data/data_20250713_1/calibrate.db",
        height=540,
        width=960,
        fps=30,
        id=camera_id(camera["id"]),
        nodes=[
            node("load_blob", "load_blob", topic_name=f"image_{suffix}"),
            node("decode_jpeg", "decode_jpeg", inputs={"default": "load_blob"}),
            node(
                "display_image",
                "image_property",
                inputs={"default": "decode_jpeg"},
                properties=image_properties(include_received=True),
                camera_name=f"camera{suffix}",
            ),
            node("load_marker", "load_marker", topic_name=f"marker_{suffix}"),
        ],
        outputs=["load_marker"],
    )


def capture_color_camera_subgraph(camera: dict) -> dict:
    suffix = camera["suffix"]
    return subgraph(
        f"capture_color_camera{suffix}_subgraph",
        address=camera["address"],
        id=camera_id(camera["id"]),
        height=1296,
        width=2304,
        fps=30,
        stream="color",
        format="R8G8B8_UINT",
        exposure=7000,
        gain=10,
        emitter_enabled=False,
        resize_width=960,
        resize_height=540,
        deploy_port=31400,
        endpoint_address="192.168.0.254",
        image_port=50101,
        marker_port=50201,
        nodes=[
            node("libcamera_capture", "libcamera_capture"),
            node("fifo", "fifo", inputs={"default": "libcamera_capture"}),
            node("resize", "resize", inputs={"default": "fifo"}),
            node("fifo_image", "fifo", inputs={"default": "resize"}),
            node("encode_jpeg", "encode_jpeg", inputs={"default": "fifo_image"}),
            node("p2p_tcp_talker_image", "p2p_tcp_talker", inputs={"default": "encode_jpeg"}),
            node("fifo_marker", "fifo", inputs={"default": "resize"}),
            node(
                "detect_circle_grid",
                "detect_circle_grid",
                inputs={"default": "fifo_marker"},
                min_threshold=150,
                max_threshold=200,
                threshold_step=20,
                min_dist_between_blobs=3.0,
                min_area=5.0,
                max_area=100.0,
                filter_by_area=True,
                min_circularity=0.5,
                max_circularity=1.0,
                filter_by_circularity=True,
                filter_by_inertia=False,
                filter_by_convexity=False,
                blob_color=0,
                filter_by_color=True,
            ),
            node("p2p_tcp_talker_marker", "p2p_tcp_talker", inputs={"default": "detect_circle_grid"}),
            node("p2p_tcp_listener_image", "p2p_tcp_listener", inputs={"default": "p2p_tcp_talker_image"}),
            node("decode_jpeg", "decode_jpeg", inputs={"default": "p2p_tcp_listener_image"}),
            node(
                "display_image",
                "image_property",
                inputs={"default": "decode_jpeg"},
                properties=image_properties(include_received=True),
                camera_name=f"camera{suffix}",
            ),
            node("p2p_tcp_listener_marker", "p2p_tcp_listener", inputs={"default": "p2p_tcp_talker_marker"}),
        ],
        outputs=["p2p_tcp_listener_marker"],
    )


def intrinsic_pipeline(name: str, target_subgraph: str) -> tuple[str, dict]:
    return pipeline_def(
        name,
        subgraphs=[
            subgraph("target_camera", extends=[target_subgraph]),
            subgraph(
                "intrinsic_calibration",
                extends=["intrinsic_calibration_subgraph"],
                nodes=[node("frame_number_numbering", inputs={"default": "target_camera/decode_jpeg"})],
            ),
        ],
    )


def build_config() -> Config:
    data = config_dict(
        pipeline="intrinsic_calibration",
        subgraphs=[
            playback_color_subgraph(),
            intrinsic_calibration_subgraph(),
            *[playback_color_camera_subgraph(camera) for camera in COLOR_CAMERAS],
            *[capture_color_camera_subgraph(camera) for camera in COLOR_CAMERAS],
        ],
        pipelines=dict(
            [
                intrinsic_pipeline("intrinsic_calibration", "playback_color_camera101_subgraph"),
                intrinsic_pipeline("intrinsic_calibration_live", "capture_color_camera101_subgraph"),
            ]
        ),
    )
    return Config.from_dict(data)


def main(argv: list[str]) -> int:
    output_path = require_output_path(argv)
    build_config().to_json(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
