from __future__ import annotations

import sys

from pipeline import Config, camera_id, config_dict, image_properties, node, node_ref, pipeline_def, require_output_path, subgraph


IR_CAPTURE_CAMERAS = list(range(1, 25))
IR_PLAYBACK_CAMERAS = list(range(1, 21))
IMU_TOPICS = [
    "imu1",
    "imu2",
    "imu3",
    "imu4",
    "imu5",
    "imu6",
]


def playback_subgraph() -> dict:
    return subgraph(
        "playback_subgraph",
        db_path="../data/data_20221004_1/capture.db",
        width=820,
        height=616,
        fps=90,
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


def record_subgraph() -> dict:
    return subgraph(
        "record_subgraph",
        db_path="../data/data_20250506_1/capture.db",
        nodes=[node("record", "record")],
        outputs=["record"],
    )


def ir_cluster_subgraph(name: str, *, master: bool) -> dict:
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
                gain=0.1,
                interval=11.111111,
                max_interval=12.5,
                min_interval=10.0,
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
                gain=0.1,
                interval=11.111111,
                max_interval=12.5,
                min_interval=10.0,
            ),
        ]
    tail_nodes = [
        node("scale", "scale", inputs={"default": "fifo"}, alpha=1.1, beta=0),
        node(
            "gaussian_blur",
            "gaussian_blur",
            inputs={"default": "scale"},
            kernel_width=3,
            kernel_height=3,
            sigma_x=1.5,
            sigma_y=1.5,
        ),
        node("action_mask_generate", "action", action_id="generate", label="Mask", icon="edit"),
        node("action_mask_clear", "action", action_id="clear", label="Unmask", icon="eraser"),
        node(
            "mask_generator",
            "mask_generator",
            inputs={
                "default": "gaussian_blur",
                "generate": "action_mask_generate",
                "clear": "action_mask_clear",
            },
            path="/tmp/mask.png",
        ),
        node("mask", "mask", inputs={"default": "gaussian_blur", "mask": "mask_generator"}),
        node("fifo_image", "fifo", inputs={"default": "mask"}),
        node("encode_jpeg", "encode_jpeg", inputs={"default": "fifo_image"}),
        node("p2p_tcp_talker_image", "p2p_tcp_talker", inputs={"default": "encode_jpeg"}),
        node("fifo_marker", "fifo", inputs={"default": "mask"}),
        node(
            "fast_blob_detector",
            "fast_blob_detector",
            inputs={"default": "fifo_marker"},
            min_dist_between_blobs=1,
            step_threshold=5,
            min_threshold=50,
            max_threshold=100,
            min_area=6,
            max_area=1000,
            min_circularity=0.8,
        ),
        node("p2p_tcp_talker_marker", "p2p_tcp_talker", inputs={"default": "fast_blob_detector"}),
    ]
    return subgraph(
        name,
        height=616,
        width=820,
        fps=90,
        stream="infrared",
        format="Y8_UINT",
        exposure=5000,
        gain=10,
        emitter_enabled=True,
        deploy_port=31400,
        nodes=prefix_nodes + common_nodes + sync_nodes + tail_nodes,
        outputs=["p2p_tcp_talker_image", "p2p_tcp_talker_marker"],
    )


def ir_receiver_subgraph() -> dict:
    return subgraph(
        "raspi_ir_receiver",
        endpoint_address="192.168.0.254",
        fps=90,
        width=820,
        height=616,
        nodes=[
            node("p2p_tcp_listener_image", "p2p_tcp_listener"),
            node("decode_jpeg", "decode_jpeg", inputs={"default": "p2p_tcp_listener_image"}),
            node("p2p_tcp_listener_marker", "p2p_tcp_listener"),
        ],
        outputs=["p2p_tcp_listener_marker"],
    )


def epipolar_reconstruction_subgraph() -> dict:
    camera_inputs = {f"camera.camera{index}": f"load_camera{index}" for index in IR_PLAYBACK_CAMERAS}
    nodes = [
        node(f"load_camera{index}", "load_parameter", id=camera_id(index))
        for index in IR_PLAYBACK_CAMERAS
    ]
    nodes.extend(
        [
            node("load_axis", "load_parameter", id="scene"),
            node("object_to_frame", "object_to_frame"),
            node("frame_number_numbering", "frame_number_numbering", inputs={"default": "object_to_frame"}),
            node("parallel_queue", "parallel_queue", inputs={"default": "frame_number_numbering"}),
            node(
                "epipolar_reconstruction",
                "epipolar_reconstruction",
                inputs={"default": "parallel_queue", **camera_inputs, "axis": "load_axis"},
            ),
            node("frame_number_ordering", "frame_number_ordering", inputs={"default": "epipolar_reconstruction"}),
            node("marker_property", "marker_property", inputs={"default": "frame_number_ordering"}),
            node("grpc_server", "grpc_server", inputs={"sphere": "frame_number_ordering"}),
        ]
    )
    return subgraph("epipolar_reconstruction_subgraph", nodes=nodes)


def imu_subgraph() -> dict:
    nodes = [
        node("imu_server", "grpc_server", address="0.0.0.0:50053"),
        node("demux", "frame_demux", outputs=IMU_TOPICS, inputs={"default": "imu_server"}),
    ]
    for index, topic in enumerate(IMU_TOPICS, start=1):
        nodes.append(
            node(
                f"dump_imu_{index}",
                "dump_se3",
                db_path="../data/imu.db",
                topic_name=topic,
                inputs={topic: f"demux:{topic}"},
            )
        )
    return subgraph("imu_subgraph", nodes=nodes)


def make_capture_receiver(camera_name: str) -> dict:
    return subgraph(
        f"{camera_name}_receiver",
        extends=["raspi_ir_receiver"],
        nodes=[node("display_image", camera_name=camera_name)],
    )


def make_record(camera_name: str, index: int) -> dict:
    return subgraph(
        f"record{index}",
        extends=["record_subgraph"],
        nodes=[node("record", inputs={"default": node_ref(f"{camera_name}_receiver", "callback_image")})],
    )


def make_playback_camera(index: int) -> dict:
    camera_name = f"camera{index}"
    return subgraph(
        camera_name,
        extends=["playback_subgraph"],
        id=camera_id(index),
        nodes=[
            node("load_blob", topic_name=f"image_{index}"),
            node("load_marker", topic_name=f"marker_{index}"),
            node("display_image", camera_name=camera_name),
        ],
    )


def record_ir_pipeline() -> tuple[str, dict]:
    subgraphs = []
    for index in IR_CAPTURE_CAMERAS:
        camera_name = f"camera{index}"
        subgraphs.extend(
            [
                subgraph(
                    camera_name,
                    extends=["raspi_ir_cluster_master" if index == 1 else "raspi_ir_cluster_slave"],
                    address=f"192.168.0.{index}",
                    id=camera_id(index),
                ),
                make_capture_receiver(camera_name),
                make_record(camera_name, index),
            ]
        )
    subgraphs.append(
        subgraph(
            "sync",
            nodes=[
                node(
                    "approximate_time_sync_images",
                    "approximate_time_sync",
                    interval=11.611,
                    inputs={f"camera{index}": node_ref(f"camera{index}_receiver", "decode_jpeg") for index in IR_CAPTURE_CAMERAS},
                ),
                node(
                    "approximate_time_sync_markers",
                    "approximate_time_sync",
                    interval=11.611,
                    inputs={
                        f"camera{index}": node_ref(f"camera{index}_receiver", "p2p_tcp_listener_marker")
                        for index in IR_CAPTURE_CAMERAS
                    },
                ),
            ],
        )
    )
    return pipeline_def("record_ir", subgraphs=subgraphs)


def capture_ir_pipeline() -> tuple[str, dict]:
    subgraphs = []
    for index in IR_CAPTURE_CAMERAS:
        camera_name = f"camera{index}"
        subgraphs.extend(
            [
                subgraph(
                    camera_name,
                    extends=["raspi_ir_cluster_master" if index == 1 else "raspi_ir_cluster_slave"],
                    address=f"192.168.0.{index}",
                    id=camera_id(index),
                ),
                make_capture_receiver(camera_name),
            ]
        )
    subgraphs.append(
        subgraph(
            "sync",
            nodes=[
                node(
                    "approximate_time_sync_images",
                    "approximate_time_sync",
                    interval=11.611,
                    inputs={f"camera{index}": node_ref(f"camera{index}_receiver", "decode_jpeg") for index in IR_CAPTURE_CAMERAS},
                ),
                node(
                    "approximate_time_sync_markers",
                    "approximate_time_sync",
                    interval=11.611,
                    inputs={
                        f"camera{index}": node_ref(f"camera{index}_receiver", "p2p_tcp_listener_marker")
                        for index in IR_CAPTURE_CAMERAS
                    },
                ),
            ],
        )
    )
    return pipeline_def("capture_ir", subgraphs=subgraphs)


def playback_ir_pipeline() -> tuple[str, dict]:
    subgraphs = [make_playback_camera(index) for index in IR_PLAYBACK_CAMERAS]
    subgraphs.append(
        subgraph(
            "sync",
            nodes=[
                node(
                    "approximate_time_sync_images",
                    "approximate_time_sync",
                    interval=11.611,
                    inputs={f"camera{index}": node_ref(f"camera{index}", "decode_jpeg") for index in IR_PLAYBACK_CAMERAS},
                ),
                node(
                    "approximate_time_sync_markers",
                    "approximate_time_sync",
                    interval=11.611,
                    inputs={f"camera{index}": node_ref(f"camera{index}", "load_marker") for index in IR_PLAYBACK_CAMERAS},
                ),
            ],
        )
    )
    return pipeline_def("playback_ir", subgraphs=subgraphs)


def epipolar_pipeline() -> tuple[str, dict]:
    subgraphs = [make_playback_camera(index) for index in IR_PLAYBACK_CAMERAS]
    subgraphs.append(
        subgraph(
            "sync",
            nodes=[
                node(
                    "approximate_time_sync_markers",
                    "approximate_time_sync",
                    interval=11.611,
                    inputs={f"camera{index}": node_ref(f"camera{index}", "load_marker") for index in IR_PLAYBACK_CAMERAS},
                ),
                node("keypoint_to_float2", "keypoint_to_float2_map", inputs={"default": "approximate_time_sync_markers"}),
            ],
        )
    )
    subgraphs.append(
        subgraph(
            "epipolar_reconstruction",
            extends=["epipolar_reconstruction_subgraph"],
            nodes=[node("object_to_frame", inputs={"default": "keypoint_to_float2"})],
        )
    )
    return pipeline_def("epipolar_reconstruction", subgraphs=subgraphs)


def build_config() -> Config:
    data = config_dict(
        pipeline="epipolar_reconstruction",
        subgraphs=[
            playback_subgraph(),
            record_subgraph(),
            epipolar_reconstruction_subgraph(),
            imu_subgraph(),
            ir_cluster_subgraph("raspi_ir_cluster_master", master=True),
            ir_cluster_subgraph("raspi_ir_cluster_slave", master=False),
            ir_receiver_subgraph(),
        ],
        pipelines=dict(
            [
                record_ir_pipeline(),
                capture_ir_pipeline(),
                playback_ir_pipeline(),
                epipolar_pipeline(),
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
