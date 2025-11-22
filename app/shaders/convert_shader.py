#!/usr/bin/env python3
import sys
import subprocess
import re

if len(sys.argv) != 3:
    print("Usage: convert_shader.py <input.spv> <output.inc>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# Run xxd and get output
result = subprocess.run(['xxd', '-i', input_file], capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error running xxd: {result.stderr}")
    sys.exit(1)

# Process the output
lines = result.stdout.split('\n')
data_lines = []

for line in lines:
    # Skip the declaration and length lines
    if 'unsigned char' in line or 'unsigned int' in line:
        continue
    # Skip the closing brace and semicolon
    if line.strip() == '};':
        continue
    # Keep data lines
    if line.strip():
        data_lines.append(line)

# Ensure the last data line ends with a comma
if data_lines and not data_lines[-1].rstrip().endswith(','):
    data_lines[-1] = data_lines[-1].rstrip() + ','

# Write to output file (just the data, no declaration)
with open(output_file, 'w') as f:
    f.write('\n'.join(data_lines))
    if data_lines:
        f.write('\n')
