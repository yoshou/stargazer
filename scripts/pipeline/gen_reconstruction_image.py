from __future__ import annotations

import sys

from pipeline import Config, camera_id, config_dict, image_properties, node, pipeline_def, require_output_path, subgraph


COLOR_CAMERAS = [
    {"suffix": 101, "id": 26, "address": "192.168.0.26", "master": True},
    {"suffix": 102, "id": 27, "address": "192.168.0.27", "master": False},
    {"suffix": 103, "id": 28, "address": "192.168.0.28", "master": False},
    {"suffix": 104, "id": 29, "address": "192.168.0.29", "master": False},
    {"suffix": 105, "id": 30, "address": "192.168.0.30", "master": False},
]
PANOPTIC_CAMERAS = ["00_03", "00_06", "00_12", "00_13", "00_23"]


def feature_properties() -> list[dict]:
    return image_properties(include_received=True, label="Feature", target="point", resource_kind="feature")


def playback_color_subgraph() -> dict:
    return subgraph(
        "playback_color_subgraph",
        db_path="../data/data_20250713_1/capture.db",
        width=960,
        height=540,
        fps=30,
        nodes=[
            node("load_blob", "load_blob"),
            node("decode_jpeg", "decode_jpeg", inputs={"default": "load_blob"}),
            node(
                "display_image",
                "image_property",
                inputs={"default": "decode_jpeg"},
                properties=image_properties(resource_kind="raw"),
            ),
            node("load_marker", "load_marker"),
        ],
        outputs=["load_marker"],
    )


def playback_panoptic_subgraph() -> dict:
    return subgraph(
        "playback_panoptic_subgraph",
        db_path="../data/panoptic",
        width=1920,
        height=1080,
        fps=30,
        nodes=[
            node("load_panoptic", "load_panoptic"),
            node("decode_jpeg", "decode_jpeg", inputs={"default": "load_panoptic"}),
            node(
                "display_image",
                "image_property",
                inputs={"default": "decode_jpeg"},
                properties=image_properties(resource_kind="raw"),
            ),
        ],
    )


def record_subgraph() -> dict:
    return subgraph(
        "record_subgraph",
        db_path="../data/data_20250506_1/capture.db",
        nodes=[node("record", "record")],
        outputs=["record"],
    )


def color_cluster_subgraph(name: str, *, master: bool) -> dict:
    common_nodes = [
        node("fifo", "fifo", inputs={"default": "libcamera_capture"}),
        node("timestamp", "timestamp", inputs={"default": "fifo"}),
    ]
    if master:
        prefix_nodes = [node("libcamera_capture", "libcamera_capture")]
        sync_nodes = [
            node(
                "broadcast_talker",
                "broadcast_talker",
                inputs={"default": "timestamp"},
                address="192.168.0.255",
                port=40000,
            )
        ]
    else:
        prefix_nodes = [
            node(
                "video_time_sync_control",
                "video_time_sync_control",
                gain=0.05,
                interval=33.333333,
                max_interval=50.0,
                min_interval=25.0,
            ),
            node(
                "libcamera_capture",
                "libcamera_capture",
                inputs={"interval": "video_time_sync_control"},
            ),
        ]
        sync_nodes = [
            node(
                "broadcast_listener",
                "broadcast_listener",
                address="192.168.0.1",
                port=40000,
            ),
            node(
                "video_time_sync_control_update",
                "video_time_sync_control",
                inputs={"default": "timestamp", "ref": "broadcast_listener"},
                gain=0.05,
                interval=33.333333,
                max_interval=50.0,
                min_interval=25.0,
            ),
        ]
    tail_nodes = [
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
    ]
    return subgraph(
        name,
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
        nodes=prefix_nodes + common_nodes + sync_nodes + tail_nodes,
        outputs=["p2p_tcp_talker_image", "p2p_tcp_talker_marker"],
    )


def color_receiver_subgraph() -> dict:
    return subgraph(
        "raspi_color_receiver",
        endpoint_address="192.168.0.254",
        fps=30,
        width=960,
        height=540,
        image_port=50101,
        marker_port=50201,
        nodes=[
            node("p2p_tcp_listener_image", "p2p_tcp_listener"),
            node("decode_jpeg", "decode_jpeg", inputs={"default": "p2p_tcp_listener_image"}),
            node(
                "display_image",
                "image_property",
                inputs={"default": "decode_jpeg"},
                properties=image_properties(resource_kind="raw"),
            ),
            node("p2p_tcp_listener_marker", "p2p_tcp_listener"),
        ],
        outputs=["p2p_tcp_listener_marker"],
    )


def make_receiver(camera_name: str) -> dict:
    return subgraph(
        f"{camera_name}_receiver",
        extends=["raspi_color_receiver"],
        nodes=[
            node("p2p_tcp_listener_image", inputs={"default": f"{camera_name}.p2p_tcp_talker_image"}),
            node("display_image", camera_name=camera_name),
            node("p2p_tcp_listener_marker", inputs={"default": f"{camera_name}.p2p_tcp_talker_marker"}),
        ],
    )


def make_record(camera_suffix: int) -> dict:
    camera_name = f"camera{camera_suffix}"
    return subgraph(
        f"record{camera_suffix}",
        extends=["record_subgraph"],
        nodes=[node("record", inputs={"default": f"{camera_name}_receiver/callback_image"})],
    )


def make_playback_camera(camera: dict) -> dict:
    suffix = camera["suffix"]
    return subgraph(
        f"camera{suffix}",
        extends=["playback_color_subgraph"],
        id=camera_id(camera["id"]),
        nodes=[
            node("load_blob", topic_name=f"image_{suffix}"),
            node("load_marker", topic_name=f"marker_{suffix}"),
            node("display_image", camera_name=f"camera{suffix}"),
        ],
    )


def make_panoptic_camera(name: str) -> dict:
    return subgraph(
        name,
        extends=["playback_panoptic_subgraph"],
        id=name,
        fps=30,
        nodes=[
            node("load_panoptic", topic_name=name),
            node("display_image", camera_name=name),
        ],
    )


def feature_render(camera_name: str, *, width: int, height: int) -> dict:
    return node(
        f"feature_{camera_name}",
        "feature_render",
        camera_name=camera_name,
        width=width,
        height=height,
        inputs={"default": "reconstruct"},
        properties=feature_properties(),
    )


def voxelpose_reconstruction_subgraph() -> dict:
    nodes = [
        node(f"load_camera{camera['suffix']}", "load_parameter", id=camera_id(camera["id"]))
        for camera in COLOR_CAMERAS
    ]
    nodes.extend(
        [
            node("load_axis", "load_parameter", id="scene"),
            node("numbering", "frame_number_numbering"),
            node("object_to_frame", "object_to_frame", inputs={"default": "numbering"}),
            node("unframe_image_fields", "unframe_image_fields", inputs={"default": "object_to_frame"}),
            node("queue", "parallel_queue", num_threads=1, inputs={"default": "unframe_image_fields"}),
            node(
                "reconstruct",
                "voxelpose_reconstruction",
                inputs={
                    "default": "queue",
                    **{f"camera.camera{camera['suffix']}": f"load_camera{camera['suffix']}" for camera in COLOR_CAMERAS},
                    "axis": "load_axis",
                },
            ),
        ]
    )
    nodes.extend(feature_render(f"camera{camera['suffix']}", width=960, height=540) for camera in COLOR_CAMERAS)
    nodes.extend(
        [
            node("markers", "reconstruction_result_markers", inputs={"default": "reconstruct"}),
            node("ordering", "frame_number_ordering", inputs={"default": "markers"}),
            node("marker_property", "marker_property", inputs={"default": "ordering"}),
            node("grpc_server", "grpc_server", address="0.0.0.0:50052", inputs={"sphere": "ordering"}),
        ]
    )
    return subgraph("voxelpose_reconstruction_subgraph", nodes=nodes)


def mvpose_reconstruction_subgraph() -> dict:
    nodes = [
        node(f"load_camera{camera['suffix']}", "load_parameter", id=camera_id(camera["id"]))
        for camera in COLOR_CAMERAS
    ]
    nodes.extend(
        [
            node("load_axis", "load_parameter", id="scene"),
            node("numbering", "frame_number_numbering"),
            node("queue", "parallel_queue", num_threads=1, inputs={"default": "numbering"}),
            node(
                "reconstruct",
                "mvpose_reconstruction",
                inputs={
                    "default": "queue",
                    **{f"camera.camera{camera['suffix']}": f"load_camera{camera['suffix']}" for camera in COLOR_CAMERAS},
                    "axis": "load_axis",
                },
            ),
            node(
                "dump_reconstruction",
                "dump_reconstruction",
                db_path="../data/mvpose_dump.db",
                topic_name="mvpose_reconstruction",
                inputs={"default": "reconstruct"},
            ),
            node("marker_property", "marker_property", inputs={"default": "reconstruct"}),
        ]
    )
    nodes.extend(feature_render(f"camera{camera['suffix']}", width=960, height=540) for camera in COLOR_CAMERAS)
    nodes.extend(feature_render(name, width=1920, height=1080) for name in PANOPTIC_CAMERAS)
    return subgraph("mvpose_reconstruction_subgraph", nodes=nodes)


def mvp_reconstruction_subgraph() -> dict:
    nodes = [
        node(f"load_camera{camera['suffix']}", "load_parameter", id=camera_id(camera["id"]))
        for camera in COLOR_CAMERAS
    ]
    nodes.extend(
        [
            node("load_axis", "load_parameter", id="scene"),
            node("numbering", "frame_number_numbering"),
            node("queue", "parallel_queue", num_threads=1, inputs={"default": "numbering"}),
            node(
                "reconstruct",
                "mvp_reconstruction",
                inputs={
                    "default": "queue",
                    **{f"camera.camera{camera['suffix']}": f"load_camera{camera['suffix']}" for camera in COLOR_CAMERAS},
                    "axis": "load_axis",
                },
            ),
            node("marker_property", "marker_property", inputs={"default": "reconstruct"}),
        ]
    )
    return subgraph("mvp_reconstruction_subgraph", nodes=nodes)


def sync_subgraph(name: str, *, interval: float, image_inputs: dict, marker_inputs: dict | None = None) -> dict:
    nodes = [node("approximate_time_sync_images", "approximate_time_sync", interval=interval, inputs=image_inputs)]
    if marker_inputs is not None:
        nodes.append(node("approximate_time_sync_markers", "approximate_time_sync", interval=interval, inputs=marker_inputs))
    return subgraph(name, nodes=nodes)


def record_color_pipeline() -> tuple[str, dict]:
    subgraphs = []
    for camera in COLOR_CAMERAS:
        camera_name = f"camera{camera['suffix']}"
        subgraphs.extend(
            [
                subgraph(
                    camera_name,
                    extends=["raspi_color_cluster_master" if camera["master"] else "raspi_color_cluster_slave"],
                    address=camera["address"],
                    id=camera_id(camera["id"]),
                ),
                make_receiver(camera_name),
                make_record(camera["suffix"]),
            ]
        )
    subgraphs.append(
        sync_subgraph(
            "sync",
            interval=11.6,
            image_inputs={f"camera{camera['suffix']}": f"camera{camera['suffix']}_receiver/decode_jpeg" for camera in COLOR_CAMERAS},
            marker_inputs={
                f"camera{camera['suffix']}": f"camera{camera['suffix']}_receiver/p2p_tcp_listener_marker"
                for camera in COLOR_CAMERAS
            },
        )
    )
    return pipeline_def("record_color", subgraphs=subgraphs)


def capture_color_pipeline() -> tuple[str, dict]:
    subgraphs = []
    for camera in COLOR_CAMERAS:
        camera_name = f"camera{camera['suffix']}"
        subgraphs.extend(
            [
                subgraph(
                    camera_name,
                    extends=["raspi_color_cluster_master" if camera["master"] else "raspi_color_cluster_slave"],
                    address=camera["address"],
                    id=camera_id(camera["id"]),
                ),
                make_receiver(camera_name),
            ]
        )
    subgraphs.append(
        sync_subgraph(
            "sync",
            interval=11.6,
            image_inputs={f"camera{camera['suffix']}": f"camera{camera['suffix']}_receiver/decode_jpeg" for camera in COLOR_CAMERAS},
            marker_inputs={
                f"camera{camera['suffix']}": f"camera{camera['suffix']}_receiver/p2p_tcp_listener_marker"
                for camera in COLOR_CAMERAS
            },
        )
    )
    return pipeline_def("capture_color", subgraphs=subgraphs)


def playback_color_pipeline() -> tuple[str, dict]:
    subgraphs = [make_playback_camera(camera) for camera in COLOR_CAMERAS]
    subgraphs.append(
        sync_subgraph(
            "sync",
            interval=33.833,
            image_inputs={f"camera{camera['suffix']}": f"camera{camera['suffix']}/decode_jpeg" for camera in COLOR_CAMERAS},
            marker_inputs={f"camera{camera['suffix']}": f"camera{camera['suffix']}/load_marker" for camera in COLOR_CAMERAS},
        )
    )
    return pipeline_def("playback_color", subgraphs=subgraphs)


def playback_panoptic_pipeline() -> tuple[str, dict]:
    subgraphs = [make_panoptic_camera(name) for name in PANOPTIC_CAMERAS]
    subgraphs.append(
        sync_subgraph(
            "sync",
            interval=33.833,
            image_inputs={name: f"{name}.decode_jpeg" for name in PANOPTIC_CAMERAS},
        )
    )
    return pipeline_def("playback_panoptic", subgraphs=subgraphs)


def voxelpose_pipeline() -> tuple[str, dict]:
    subgraphs = [make_playback_camera(camera) for camera in COLOR_CAMERAS]
    subgraphs.append(
        sync_subgraph(
            "sync",
            interval=33.833,
            image_inputs={f"camera{camera['suffix']}": f"camera{camera['suffix']}/decode_jpeg" for camera in COLOR_CAMERAS},
        )
    )
    subgraphs.append(
        subgraph(
            "voxelpose_reconstruction",
            extends=["voxelpose_reconstruction_subgraph"],
            nodes=[node("numbering", inputs={"default": "approximate_time_sync_images"})],
        )
    )
    return pipeline_def("voxelpose_reconstruction", subgraphs=subgraphs)


def build_config() -> Config:
    data = config_dict(
        pipeline="voxelpose_reconstruction",
        subgraphs=[
            record_subgraph(),
            playback_color_subgraph(),
            playback_panoptic_subgraph(),
            voxelpose_reconstruction_subgraph(),
            mvpose_reconstruction_subgraph(),
            mvp_reconstruction_subgraph(),
            color_cluster_subgraph("raspi_color_cluster_master", master=True),
            color_cluster_subgraph("raspi_color_cluster_slave", master=False),
            color_receiver_subgraph(),
        ],
        pipelines=dict(
            [
                record_color_pipeline(),
                capture_color_pipeline(),
                playback_color_pipeline(),
                playback_panoptic_pipeline(),
                voxelpose_pipeline(),
                pipeline_def("mvpose_reconstruction", subgraphs=[subgraph("mvpose_reconstruction", extends=["mvpose_reconstruction_subgraph"])]),
                pipeline_def("mvp_reconstruction", subgraphs=[subgraph("mvp_reconstruction", extends=["mvp_reconstruction_subgraph"])]),
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
