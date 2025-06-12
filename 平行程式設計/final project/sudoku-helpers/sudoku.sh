#!/bin/bash

OUTPUT_DIR="/home/110611008/Parallelization-of-Sudoku-Solvers/final-project/mazes"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Generate 100 Sudoku files
for i in $(seq 100 100); do
    FILENAME="${OUTPUT_DIR}/16x16_Medium_${i}.txt"
    ./generate_sudoku "$FILENAME"
    sleep 1
done
