#!/usr/bin/env python3
"""
pipeline_ctl.py - Stargazer PipelineControl gRPC CLI client

Usage:
  pipeline_ctl.py [--address HOST:PORT] <command> [args...]

Commands:
  start                   Start the pipeline
  stop                    Stop the pipeline
  status                  Get pipeline status
  collect-on              Enable marker collecting
  collect-off             Disable marker collecting
  action <action_id>      Dispatch an action
  list-actions            List available actions
  list-nodes              List pipeline nodes
  list-properties [node]  List properties (optionally filtered by node name)
  get-property <node> <key>  Get a node property value
                             Image properties are saved to /tmp/ and path is printed

Options:
  --address HOST:PORT     gRPC server address (default: localhost:50052)
"""
import argparse
import sys
import os
import time
import importlib.util
import subprocess
import tempfile

# Determine the script directory so we can find generated proto stubs
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROTO_STUBS_DIR = os.path.join(_SCRIPT_DIR, "_grpc_stubs")


def _ensure_stubs():
    """Generate Python gRPC stubs from .proto if not already generated."""
    pb2_path = os.path.join(_PROTO_STUBS_DIR, "pipeline_control_pb2.py")
    pb2_grpc_path = os.path.join(_PROTO_STUBS_DIR, "pipeline_control_pb2_grpc.py")
    proto_src = os.path.join(_SCRIPT_DIR, "..", "protos", "pipeline_control.proto")

    if os.path.exists(pb2_path) and os.path.exists(pb2_grpc_path):
        # Check if proto is newer than generated stubs
        if os.path.getmtime(proto_src) <= os.path.getmtime(pb2_path):
            return

    os.makedirs(_PROTO_STUBS_DIR, exist_ok=True)
    init_path = os.path.join(_PROTO_STUBS_DIR, "__init__.py")
    if not os.path.exists(init_path):
        open(init_path, "w").close()

    proto_src = os.path.abspath(proto_src)
    proto_dir = os.path.dirname(proto_src)

    result = subprocess.run(
        [
            sys.executable, "-m", "grpc_tools.protoc",
            f"-I{proto_dir}",
            f"--python_out={_PROTO_STUBS_DIR}",
            f"--grpc_python_out={_PROTO_STUBS_DIR}",
            proto_src,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Error generating gRPC stubs:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        print("Install grpcio-tools: pip install grpcio grpcio-tools", file=sys.stderr)
        sys.exit(1)


def _load_stubs():
    _ensure_stubs()
    sys.path.insert(0, _SCRIPT_DIR)

    spec2 = importlib.util.spec_from_file_location(
        "pipeline_control_pb2",
        os.path.join(_PROTO_STUBS_DIR, "pipeline_control_pb2.py"),
    )
    pb2 = importlib.util.module_from_spec(spec2)
    sys.modules["pipeline_control_pb2"] = pb2
    spec2.loader.exec_module(pb2)

    spec_grpc = importlib.util.spec_from_file_location(
        "pipeline_control_pb2_grpc",
        os.path.join(_PROTO_STUBS_DIR, "pipeline_control_pb2_grpc.py"),
    )
    pb2_grpc = importlib.util.module_from_spec(spec_grpc)
    sys.modules["pipeline_control_pb2_grpc"] = pb2_grpc
    spec_grpc.loader.exec_module(pb2_grpc)

    return pb2, pb2_grpc


def _make_channel(address):
    try:
        import grpc
    except ImportError:
        print("grpcio not installed. Run: pip install grpcio grpcio-tools", file=sys.stderr)
        sys.exit(1)
    return grpc.insecure_channel(address)


def _empty(pb2):
    from google.protobuf import empty_pb2
    return empty_pb2.Empty()


def cmd_start(stub, pb2, args):
    stub.Start(_empty(pb2))
    print("Started")


def cmd_stop(stub, pb2, args):
    stub.Stop(_empty(pb2))
    print("Stopped")


def cmd_status(stub, pb2, args):
    status = stub.GetStatus(_empty(pb2))
    print(f"running:    {'yes' if status.running else 'no'}")
    print(f"collecting: {'yes' if status.collecting else 'no'}")


def cmd_collect_on(stub, pb2, args):
    stub.EnableCollecting(_empty(pb2))
    print("Collecting enabled")


def cmd_collect_off(stub, pb2, args):
    stub.DisableCollecting(_empty(pb2))
    print("Collecting disabled")


def cmd_action(stub, pb2, args):
    if not args.action_id:
        print("Error: action_id required", file=sys.stderr)
        sys.exit(1)
    req = pb2.ActionRequest(action_id=args.action_id)
    stub.DispatchAction(req)
    print(f"Dispatched action: {args.action_id}")


def cmd_list_actions(stub, pb2, args):
    response = stub.ListActions(_empty(pb2))
    if not response.actions:
        print("(no actions)")
        return
    for action in response.actions:
        parts = [f"id={action.id}", f"label={action.label!r}"]
        if action.icon:
            parts.append(f"icon={action.icon}")
        print("  " + "  ".join(parts))


def cmd_list_nodes(stub, pb2, args):
    response = stub.ListNodes(_empty(pb2))
    if not response.nodes:
        print("(no nodes)")
        return
    for node in response.nodes:
        print(f"  {node.name}  [{node.type}]")


def cmd_list_properties(stub, pb2, args):
    node_name = getattr(args, "node_name", "") or ""
    req = pb2.NodeRequest(node_name=node_name)
    response = stub.ListProperties(req)
    if not response.properties:
        print("(no properties)")
        return
    for prop in response.properties:
        type_hint = f"  ({prop.type_hint})" if prop.type_hint else ""
        print(f"  {prop.node_name}  {prop.key!r}  label={prop.label!r}{type_hint}")


def _format_camera(cam):
    lines = [
        f"  fx={cam.fx:.6f}  fy={cam.fy:.6f}",
        f"  cx={cam.cx:.6f}  cy={cam.cy:.6f}",
        f"  size={cam.width}x{cam.height}",
        f"  distortion: k1={cam.k1:.6f} k2={cam.k2:.6f} p1={cam.p1:.6f} p2={cam.p2:.6f} k3={cam.k3:.6f}",
    ]
    if cam.pose:
        pose = list(cam.pose)
        rows = [pose[i*4:(i+1)*4] for i in range(4)]
        lines.append("  pose:")
        for row in rows:
            lines.append("    " + "  ".join(f"{v:10.6f}" for v in row))
    return "\n".join(lines)


def _format_mat4(mat):
    vals = list(mat.values)
    if len(vals) != 16:
        return f"  mat4: {vals}"
    rows = [vals[i*4:(i+1)*4] for i in range(4)]
    lines = ["  mat4:"]
    for row in rows:
        lines.append("    " + "  ".join(f"{v:10.6f}" for v in row))
    return "\n".join(lines)


def _format_vec3_list(v3l):
    pts = v3l.points
    if not pts:
        return "  (empty points list)"
    lines = [f"  {len(pts)} points:"]
    for i, p in enumerate(pts[:10]):
        lines.append(f"    [{i}] ({p.x:.4f}, {p.y:.4f}, {p.z:.4f})")
    if len(pts) > 10:
        lines.append(f"    ... ({len(pts) - 10} more)")
    return "\n".join(lines)


def cmd_get_property(stub, pb2, args):
    req = pb2.PropertyRequest(node_name=args.node_name, key=args.key)
    response = stub.GetNodeProperty(req)

    if not response.found:
        print(f"Property not found: {args.node_name}.{args.key}")
        sys.exit(1)

    which = response.WhichOneof("value")
    if which == "string_value":
        print(response.string_value)
    elif which == "int_value":
        print(response.int_value)
    elif which == "double_value":
        print(response.double_value)
    elif which == "bool_value":
        print("true" if response.bool_value else "false")
    elif which == "image_value":
        data = response.image_value
        if not data:
            print("(null image)")
            return
        ts = int(time.time())
        node_safe = args.node_name.replace("/", "_")
        key_safe = args.key.replace("/", "_").replace(".", "_")
        path = f"/tmp/stargazer_{node_safe}_{key_safe}_{ts}.jpg"
        with open(path, "wb") as f:
            f.write(data)
        print(path)
    elif which == "camera_value":
        print(_format_camera(response.camera_value))
    elif which == "mat4_value":
        print(_format_mat4(response.mat4_value))
    elif which == "vec3_list_value":
        print(_format_vec3_list(response.vec3_list_value))
    else:
        print("(no value)")


def main():
    parser = argparse.ArgumentParser(
        description="Stargazer pipeline control CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--address", default="localhost:50052", help="gRPC server address")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("start")
    subparsers.add_parser("stop")
    subparsers.add_parser("status")
    subparsers.add_parser("collect-on")
    subparsers.add_parser("collect-off")

    p_action = subparsers.add_parser("action")
    p_action.add_argument("action_id")

    subparsers.add_parser("list-actions")
    subparsers.add_parser("list-nodes")

    p_list_props = subparsers.add_parser("list-properties")
    p_list_props.add_argument("node_name", nargs="?", default="")

    p_get = subparsers.add_parser("get-property")
    p_get.add_argument("node_name")
    p_get.add_argument("key")

    args = parser.parse_args()

    pb2, pb2_grpc = _load_stubs()
    channel = _make_channel(args.address)
    stub = pb2_grpc.PipelineControlStub(channel)

    dispatch = {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "collect-on": cmd_collect_on,
        "collect-off": cmd_collect_off,
        "action": cmd_action,
        "list-actions": cmd_list_actions,
        "list-nodes": cmd_list_nodes,
        "list-properties": cmd_list_properties,
        "get-property": cmd_get_property,
    }

    try:
        dispatch[args.command](stub, pb2, args)
    except Exception as e:
        import grpc
        if isinstance(e, grpc.RpcError):
            print(f"gRPC error [{e.code().name}]: {e.details()}", file=sys.stderr)
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
