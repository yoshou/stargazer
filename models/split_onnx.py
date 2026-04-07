#!/usr/bin/env python3
"""
Split large ONNX files into smaller chunks for Git version control.
"""
import os
import sys
import hashlib

# Chunk size: 10MB (Git-friendly size)
CHUNK_SIZE = 10 * 1024 * 1024


def calculate_md5(filepath):
    """Calculate MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def split_file(input_file):
    """Split a file into chunks."""
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return False
    
    base_name = os.path.basename(input_file)
    output_dir = os.path.dirname(input_file)
    
    # Calculate MD5 of original file
    original_md5 = calculate_md5(input_file)
    
    # Create chunks directory
    chunks_dir = os.path.join(output_dir, f"{base_name}.chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Split file
    chunk_index = 0
    with open(input_file, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            
            chunk_file = os.path.join(chunks_dir, f"chunk_{chunk_index:04d}")
            with open(chunk_file, "wb") as cf:
                cf.write(chunk)
            
            chunk_index += 1
    
    # Write metadata
    metadata_file = os.path.join(chunks_dir, "metadata.txt")
    with open(metadata_file, "w") as mf:
        mf.write(f"original_file={base_name}\n")
        mf.write(f"num_chunks={chunk_index}\n")
        mf.write(f"md5={original_md5}\n")
        mf.write(f"chunk_size={CHUNK_SIZE}\n")
    
    print(f"Split {input_file} into {chunk_index} chunks")
    print(f"Original MD5: {original_md5}")
    print(f"Chunks saved to: {chunks_dir}")
    
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: split_onnx.py <onnx_file1> [onnx_file2] ...")
        sys.exit(1)
    
    for input_file in sys.argv[1:]:
        if not split_file(input_file):
            sys.exit(1)
    
    print("\nAll files split successfully!")


if __name__ == "__main__":
    main()
