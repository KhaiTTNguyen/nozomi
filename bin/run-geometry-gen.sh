#!/bin/bash

GPU_DEVICE=0  # default value

# Parse all arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu=*)
            GPU_DEVICE="${1#*=}"
            shift
            ;;
        --gpu)
            GPU_DEVICE="$2"
            shift 2
            ;;
        --gpus=*)
            GPU_DEVICE="${1#*=}"
            shift
            ;;
        --gpus)
            GPU_DEVICE="$2"
            shift 2
            ;;
        *)
            break  # Stop parsing, pass remaining args to Python
            ;;
    esac
done

echo "Using GPU(s): $GPU_DEVICE"
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python3 -m simulation_toolkit.cli.main substrate "$@"