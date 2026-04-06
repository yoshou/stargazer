#!/usr/bin/env python3
"""Convert a TTF binary file to a C header with an embedded unsigned char array.

Usage: convert_font.py <input.ttf> <output.hpp>
"""
import sys
import os

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <input.ttf> <output.hpp>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, "rb") as f:
    data = f.read()

size = len(data)
source_name = os.path.basename(input_file)

COLS = 16  # bytes per line

with open(output_file, "w") as f:
    f.write(f"// Auto-generated from {source_name} ({size} bytes)\n")
    f.write("// Do not edit manually — regenerated at build time\n")
    f.write("#pragma once\n")
    f.write(f"static const unsigned int font_awesome_data_size = {size};\n")
    f.write("static const unsigned char font_awesome_data[] = {\n")
    for i in range(0, size, COLS):
        chunk = data[i:i + COLS]
        line = ", ".join(f"0x{b:02x}" for b in chunk)
        if i + COLS < size:
            f.write(f"    {line},\n")
        else:
            f.write(f"    {line}\n")
    f.write("};\n")
