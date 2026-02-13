#!/bin/bash

GPU_DEVICE=0  # default value

# Function to validate and show available GPUs
validate_gpus() {
    local gpu_list="$1"
    
    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Warning: nvidia-smi not found. Cannot validate GPU numbers."
        return 0
    fi
    
    # Get available GPUs
    local available_gpus=$(nvidia-smi -L 2>/dev/null | wc -l)
    
    if [ $available_gpus -eq 0 ]; then
        echo "Error: No NVIDIA GPUs detected on this system"
        exit 1
    fi
    
    # Convert comma-separated list to array
    IFS=',' read -ra GPU_ARRAY <<< "$gpu_list"
    
    for gpu in "${GPU_ARRAY[@]}"; do
        # Remove any whitespace
        gpu=$(echo "$gpu" | xargs)
        
        # Check if it's a number
        if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
            echo "Error: '$gpu' is not a valid GPU number (must be integer)"
            exit 1
        fi
        
        # Check if GPU number is within available range
        if [ "$gpu" -ge "$available_gpus" ]; then
            echo "Error: GPU $gpu does not exist"
            echo "Valid GPU IDs: 0-$((available_gpus-1))"
            exit 1
        fi
    done
}

# Parse arguments (same as before)
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
            break
            ;;
    esac
done

# Validate the GPU numbers
validate_gpus "$GPU_DEVICE"

echo "Using GPU(s): $GPU_DEVICE"
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python3 -m simulation_toolkit.cli.main substrate "$@"