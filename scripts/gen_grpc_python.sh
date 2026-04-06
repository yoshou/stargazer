#!/usr/bin/env bash
# gen_grpc_python.sh
# Generate Python gRPC stubs from pipeline_control.proto
# Usage: bash scripts/gen_grpc_python.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_FILE="$SCRIPT_DIR/../protos/pipeline_control.proto"
OUT_DIR="$SCRIPT_DIR/_grpc_stubs"

mkdir -p "$OUT_DIR"
touch "$OUT_DIR/__init__.py"

python3 -m grpc_tools.protoc \
  -I"$(dirname "$PROTO_FILE")" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_FILE"

echo "Generated stubs in $OUT_DIR"
