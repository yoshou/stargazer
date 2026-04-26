#!/usr/bin/env python3
"""
Download model files from Hugging Face Hub.
Usage: python3 models/download_models.py
"""
import os
import sys

REPO_ID = "yoshou/stargazer-models"
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub is not installed.")
        print("  pip install huggingface_hub")
        sys.exit(1)

    files = [
        "dust3r/dust3r_512x288.onnx",
        "dust3r/dust3r_512x288.onnx.data",
        "mast3r/mast3r_512x288.onnx",
        "mast3r/mast3r_512x288.onnx.data",
        "mvpose/associator.onnx",
        "mvpose/libmmdeploy_tensorrt_ops.so",
        "mvpose/rtmdet_m_640-8xb32_coco-person-fp16.trt",
        "mvpose/rtmdet_m_640-8xb32_coco-person.onnx",
        "mvpose/rtmpose-l_8xb32-270e_coco-wholebody-384x288-fp16.trt",
        "mvpose/rtmpose-l_8xb32-270e_coco-wholebody-384x288.onnx",
        "voxelpose/backbone-fp16.trt",
        "voxelpose/backbone.onnx",
        "voxelpose/pose_v2v_net-fp16.trt",
        "voxelpose/pose_v2v_net.onnx",
        "voxelpose/proposal_v2v_net-fp16.trt",
        "voxelpose/proposal_v2v_net.onnx",
    ]

    for filename in files:
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest):
            print(f"  [skip] {filename}")
            continue

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"Downloading {filename} ...", flush=True)
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=MODELS_DIR,
        )
        print(f"  -> Done", flush=True)

    print("\nAll model files are ready.")


if __name__ == "__main__":
    main()
