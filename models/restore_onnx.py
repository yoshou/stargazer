#!/usr/bin/env python3
"""
Restore ONNX files from chunks.
"""
import os
import sys
import hashlib


def calculate_md5(filepath):
    """Calculate MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def restore_file(chunks_dir, verify=True):
    """Restore a file from chunks."""
    if not os.path.exists(chunks_dir):
        print(f"Error: {chunks_dir} not found")
        return False
    
    # Read metadata
    metadata_file = os.path.join(chunks_dir, "metadata.txt")
    if not os.path.exists(metadata_file):
        print(f"Error: metadata.txt not found in {chunks_dir}")
        return False
    
    metadata = {}
    with open(metadata_file, "r") as mf:
        for line in mf:
            key, value = line.strip().split("=", 1)
            metadata[key] = value
    
    original_file = metadata["original_file"]
    num_chunks = int(metadata["num_chunks"])
    original_md5 = metadata["md5"]
    
    output_dir = os.path.dirname(chunks_dir)
    output_file = os.path.join(output_dir, original_file)
    
    # Restore file
    with open(output_file, "wb") as of:
        for i in range(num_chunks):
            chunk_file = os.path.join(chunks_dir, f"chunk_{i:04d}")
            if not os.path.exists(chunk_file):
                print(f"Error: {chunk_file} not found")
                return False
            
            with open(chunk_file, "rb") as cf:
                of.write(cf.read())
    
    print(f"Restored {output_file}")
    
    # Verify MD5
    if verify:
        reconstructed_md5 = calculate_md5(output_file)
        if reconstructed_md5 == original_md5:
            print(f"✓ MD5 verification passed: {reconstructed_md5}")
        else:
            print(f"✗ MD5 verification failed!")
            print(f"  Expected: {original_md5}")
            print(f"  Got:      {reconstructed_md5}")
            return False
    
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: restore_onnx.py <chunks_dir1> [chunks_dir2] ...")
        print("   or: restore_onnx.py --all <directory>")
        sys.exit(1)
    
    if sys.argv[1] == "--all":
        # Restore all .chunks directories in the specified directory
        target_dir = sys.argv[2] if len(sys.argv) > 2 else "."
        chunks_dirs = []
        for entry in os.listdir(target_dir):
            if entry.endswith(".onnx.chunks"):
                chunks_dirs.append(os.path.join(target_dir, entry))
        
        if not chunks_dirs:
            print(f"No .onnx.chunks directories found in {target_dir}")
            sys.exit(1)
        
        for chunks_dir in sorted(chunks_dirs):
            if not restore_file(chunks_dir):
                sys.exit(1)
    else:
        # Restore specified chunks directories
        for chunks_dir in sys.argv[1:]:
            if not restore_file(chunks_dir):
                sys.exit(1)
    
    print("\nAll files restored successfully!")


if __name__ == "__main__":
    main()
