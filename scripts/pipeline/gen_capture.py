from __future__ import annotations

import sys

from pipeline import (
    Config,
    camera_id,
    config_dict,
    image_properties,
    node,
    pipeline_def,
    require_output_path,
    subgraph,
)


IR_CAMERAS = list(range(1, 25))
COLOR_CAMERAS = [
    {"suffix": 101, "id": 26, "address": "192.168.0.26", "master": True},
    {"suffix": 102, "id": 27, "address": "192.168.0.27", "master": False},
    {"suffix": 103, "id": 28, "address": "192.168.0.28", "master": False},
    {"suffix": 104, "id": 29, "address": "192.168.0.29", "master": False},
    {"suffix": 105, "id": 30, "address": "192.168.0.30", "master": False},
]
PANOPTIC_CAMERAS = ["00_03", "00_06", "00_12", "00_13", "00_23"]


def playback_blob_subgraph(
    name: str,
    *,
    db_path: str,
    width: int,
    height: int,
    fps: int,
    image_node_name: str,
    include_received: bool,
    resource_kind: str | None,
) -> dict:
    return subgraph(
        name,
        db_path=db_path,
        width=width,
        height=height,
        fps=fps,
        nodes=[
            node("load_blob", "load_blob"),
            node("decode_jpeg", "decode_jpeg", inputs={"default": "load_blob"}),
            node(
                image_node_name,
                "image_property",
                inputs={"default": "decode_jpeg"},
                properties=image_properties(
                    include_received=include_received,
                    resource_kind=resource_kind,
                ),
            ),
            node("load_marker", "load_marker"),
        ],
        outputs=[image_node_name, "load_marker"],
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
                "callback_image",
                "image_property",
                inputs={"default": "decode_jpeg"},
                properties=image_properties(include_received=True, resource_kind="raw"),
            ),
        ],
        outputs=["callback_image"],
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


def receiver_subgraph(name: str, *, width: int, height: int, fps: int, image_node_name: str) -> dict:
    ports = {}
    if width == 960:
        ports = {"image_port": 50101, "marker_port": 50201}
    return subgraph(
        name,
        endpoint_address="192.168.0.254",
        fps=fps,
        width=width,
        height=height,
        nodes=[
            node("p2p_tcp_listener_image", "p2p_tcp_listener"),
            node("decode_jpeg", "decode_jpeg", inputs={"default": "p2p_tcp_listener_image"}),
            node(
                image_node_name,
                "image_property",
                inputs={"default": "decode_jpeg"},
                properties=image_properties(include_received=True, resource_kind="raw"),
            ),
            node("p2p_tcp_listener_marker", "p2p_tcp_listener"),
        ],
        outputs=[image_node_name, "p2p_tcp_listener_marker"],
        **ports,
    )


def receiver_instance(name: str, camera_name: str, *, image_node_name: str) -> dict:
    return subgraph(
        name,
        extends=["raspi_color_receiver" if camera_name.startswith("camera10") else "raspi_ir_receiver"],
        nodes=[
            node("p2p_tcp_listener_image", inputs={"default": f"{camera_name}.p2p_tcp_talker_image"}),
            node(image_node_name, camera_name=camera_name),
            node("p2p_tcp_listener_marker", inputs={"default": f"{camera_name}.p2p_tcp_talker_marker"}),
        ],
    )


def make_receiver_instance(name: str, template: str, camera_name: str, *, image_node_name: str, connect_streams: bool) -> dict:
    override_nodes = [node(image_node_name, camera_name=camera_name)]
    if connect_streams:
        override_nodes = [
            node("p2p_tcp_listener_image", inputs={"default": f"{camera_name}.p2p_tcp_talker_image"}),
            node(image_node_name, camera_name=camera_name),
            node("p2p_tcp_listener_marker", inputs={"default": f"{camera_name}.p2p_tcp_talker_marker"}),
        ]
    return subgraph(name, extends=[template], nodes=override_nodes)


def make_record_instance(name: str, input_source: str) -> dict:
    return subgraph(
        name,
        extends=["record_subgraph"],
        nodes=[node("record", inputs={"default": input_source})],
    )


def make_playback_camera(template: str, *, name: str, camera_name: str, id_value: int, image_topic: str, marker_topic: str) -> dict:
    return subgraph(
        name,
        extends=[template],
        id=camera_id(id_value),
        nodes=[
            node("load_blob", topic_name=image_topic),
            node("load_marker", topic_name=marker_topic),
            node("callback_image", camera_name=camera_name),
        ],
    )


def make_panoptic_camera(name: str, *, fps: int | None = None) -> dict:
    params = {"id": name}
    if fps is not None:
        params["fps"] = fps
    return subgraph(
        name,
        extends=["playback_panoptic_subgraph"],
        nodes=[
            node("load_panoptic", topic_name=name),
            node("callback_image", camera_name=name),
        ],
        **params,
    )


def make_sync(name: str, *, interval: float, image_inputs: dict, marker_inputs: dict | None = None) -> dict:
    nodes = [node("approximate_time_sync_images", "approximate_time_sync", interval=interval, inputs=image_inputs)]
    if marker_inputs is not None:
        nodes.append(node("approximate_time_sync_markers", "approximate_time_sync", interval=interval, inputs=marker_inputs))
    return subgraph(name, nodes=nodes)


def build_record_ir() -> tuple[str, dict]:
    subgraphs = []
    for index in IR_CAMERAS:
        camera_name = f"camera{index}"
        subgraphs.extend(
            [
                subgraph(
                    camera_name,
                    extends=["raspi_ir_cluster_master" if index == 1 else "raspi_ir_cluster_slave"],
                    address=f"192.168.0.{index}",
                    id=camera_id(index),
                ),
                make_receiver_instance(
                    f"{camera_name}_receiver",
                    "raspi_ir_receiver",
                    camera_name,
                    image_node_name="callback_image",
                    connect_streams=False,
                ),
                make_record_instance(
                    f"record{index}",
                    f"{camera_name}_receiver.callback_image",
                ),
            ]
        )
    subgraphs.append(
        make_sync(
            "sync",
            interval=11.611,
            image_inputs={f"camera{index}": f"camera{index}_receiver.decode_jpeg" for index in IR_CAMERAS},
            marker_inputs={
                f"camera{index}": f"camera{index}_receiver.p2p_tcp_listener_marker"
                for index in IR_CAMERAS
            },
        )
    )
    return pipeline_def("record_ir", subgraphs=subgraphs)


def build_capture_ir() -> tuple[str, dict]:
    subgraphs = []
    for index in IR_CAMERAS:
        camera_name = f"camera{index}"
        subgraphs.extend(
            [
                subgraph(
                    camera_name,
                    extends=["raspi_ir_cluster_master" if index == 1 else "raspi_ir_cluster_slave"],
                    address=f"192.168.0.{index}",
                    id=camera_id(index),
                ),
                make_receiver_instance(
                    f"{camera_name}_receiver",
                    "raspi_ir_receiver",
                    camera_name,
                    image_node_name="callback_image",
                    connect_streams=True,
                ),
            ]
        )
    subgraphs.append(
        make_sync(
            "sync",
            interval=11.611,
            image_inputs={f"camera{index}": f"camera{index}_receiver.decode_jpeg" for index in IR_CAMERAS},
            marker_inputs={
                f"camera{index}": f"camera{index}_receiver.p2p_tcp_listener_marker"
                for index in IR_CAMERAS
            },
        )
    )
    return pipeline_def("capture_ir", subgraphs=subgraphs)


def build_playback_ir() -> tuple[str, dict]:
    subgraphs = [
        make_playback_camera(
            "playback_subgraph",
            name=f"camera{index}",
            camera_name=f"camera{index}",
            id_value=index,
            image_topic=f"image_{index}",
            marker_topic=f"marker_{index}",
        )
        for index in range(1, 21)
    ]
    subgraphs.append(
        make_sync(
            "sync",
            interval=11.611,
            image_inputs={f"camera{index}": f"camera{index}.decode_jpeg" for index in range(1, 21)},
            marker_inputs={f"camera{index}": f"camera{index}.load_marker" for index in range(1, 21)},
        )
    )
    return pipeline_def("playback_ir", subgraphs=subgraphs)


def build_record_color() -> tuple[str, dict]:
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
                make_receiver_instance(
                    f"{camera_name}_receiver",
                    "raspi_color_receiver",
                    camera_name,
                    image_node_name="callback_image",
                    connect_streams=True,
                ),
                make_record_instance(
                    f"record{camera['suffix']}",
                    f"{camera_name}_receiver.callback_image",
                ),
            ]
        )
    subgraphs.append(
        make_sync(
            "sync",
            interval=11.6,
            image_inputs={
                f"camera{camera['suffix']}": f"camera{camera['suffix']}_receiver.decode_jpeg"
                for camera in COLOR_CAMERAS
            },
            marker_inputs={
                f"camera{camera['suffix']}": f"camera{camera['suffix']}_receiver.p2p_tcp_listener_marker"
                for camera in COLOR_CAMERAS
            },
        )
    )
    return pipeline_def("record_color", subgraphs=subgraphs)


def build_capture_color() -> tuple[str, dict]:
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
                make_receiver_instance(
                    f"{camera_name}_receiver",
                    "raspi_color_receiver",
                    camera_name,
                    image_node_name="callback_image",
                    connect_streams=True,
                ),
            ]
        )
    subgraphs.append(
        make_sync(
            "sync",
            interval=11.6,
            image_inputs={
                f"camera{camera['suffix']}": f"camera{camera['suffix']}_receiver.decode_jpeg"
                for camera in COLOR_CAMERAS
            },
            marker_inputs={
                f"camera{camera['suffix']}": f"camera{camera['suffix']}_receiver.p2p_tcp_listener_marker"
                for camera in COLOR_CAMERAS
            },
        )
    )
    return pipeline_def("capture_color", subgraphs=subgraphs)


def build_playback_color() -> tuple[str, dict]:
    subgraphs = [
        make_playback_camera(
            "playback_color_subgraph",
            name=f"camera{camera['suffix']}",
            camera_name=f"camera{camera['suffix']}",
            id_value=camera["id"],
            image_topic=f"image_{camera['suffix']}",
            marker_topic=f"marker_{camera['suffix']}",
        )
        for camera in COLOR_CAMERAS
    ]
    subgraphs.append(
        make_sync(
            "sync",
            interval=33.833,
            image_inputs={
                f"camera{camera['suffix']}": f"camera{camera['suffix']}.decode_jpeg"
                for camera in COLOR_CAMERAS
            },
            marker_inputs={
                f"camera{camera['suffix']}": f"camera{camera['suffix']}.load_marker"
                for camera in COLOR_CAMERAS
            },
        )
    )
    return pipeline_def("playback_color", subgraphs=subgraphs)


def build_playback_panoptic() -> tuple[str, dict]:
    subgraphs = [make_panoptic_camera(name) for name in PANOPTIC_CAMERAS]
    subgraphs.append(
        make_sync(
            "sync",
            interval=33.833,
            image_inputs={name: f"{name}.decode_jpeg" for name in PANOPTIC_CAMERAS},
        )
    )
    return pipeline_def("playback_panoptic", subgraphs=subgraphs)


def build_config() -> Config:
    pipelines = dict(
        [
            build_record_color(),
            build_record_ir(),
            build_capture_ir(),
            build_capture_color(),
            build_playback_ir(),
            build_playback_color(),
            build_playback_panoptic(),
        ]
    )
    data = config_dict(
        pipeline="capture_ir",
        subgraphs=[
            playback_blob_subgraph(
                "playback_subgraph",
                db_path="../data/data_20221004_1/capture.db",
                width=820,
                height=616,
                fps=90,
                image_node_name="callback_image",
                include_received=True,
                resource_kind="raw",
            ),
            playback_panoptic_subgraph(),
            playback_blob_subgraph(
                "playback_color_subgraph",
                db_path="../data/data_20250713_1/capture.db",
                width=960,
                height=540,
                fps=30,
                image_node_name="callback_image",
                include_received=True,
                resource_kind="raw",
            ),
            record_subgraph(),
            ir_cluster_subgraph("raspi_ir_cluster_master", master=True),
            ir_cluster_subgraph("raspi_ir_cluster_slave", master=False),
            receiver_subgraph("raspi_ir_receiver", width=820, height=616, fps=90, image_node_name="callback_image"),
            color_cluster_subgraph("raspi_color_cluster_master", master=True),
            color_cluster_subgraph("raspi_color_cluster_slave", master=False),
            receiver_subgraph("raspi_color_receiver", width=960, height=540, fps=30, image_node_name="callback_image"),
        ],
        pipelines=pipelines,
    )
    return Config.from_dict(data)


def main(argv: list[str]) -> int:
    output_path = require_output_path(argv)
    build_config().to_json(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
