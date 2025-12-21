#!/bin/bash

# GlobalFlowNet Video Stabilization Script
# Usage: bash 02_Implementation/globalflownet/run_globalflownet.sh --repo <path> --input <video> --ckpt <checkpoint> --out <output>

set -e

# Default values
REPO_PATH=""
INPUT_VIDEO=""
CHECKPOINT=""
OUTPUT_VIDEO=""
GPU_ID=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo)
            REPO_PATH="$2"
            shift 2
            ;;
        --input)
            INPUT_VIDEO="$2"
            shift 2
            ;;
        --ckpt)
            CHECKPOINT="$2"
            shift 2
            ;;
        --out)
            OUTPUT_VIDEO="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --repo <path> --input <video> --ckpt <checkpoint> --out <output> [--gpu <id>]"
            echo "  --repo: Path to GlobalFlowNet repository"
            echo "  --input: Input shaky video file"
            echo "  --ckpt: Path to pretrained checkpoint"
            echo "  --out: Output stabilized video file"
            echo "  --gpu: GPU ID to use (default: 0)"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$REPO_PATH" || -z "$INPUT_VIDEO" || -z "$CHECKPOINT" || -z "$OUTPUT_VIDEO" ]]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 --repo <path> --input <video> --ckpt <checkpoint> --out <output>"
    exit 1
fi

# Check if files exist
if [[ ! -d "$REPO_PATH" ]]; then
    echo "Error: Repository path does not exist: $REPO_PATH"
    exit 1
fi

if [[ ! -f "$INPUT_VIDEO" ]]; then
    echo "Error: Input video does not exist: $INPUT_VIDEO"
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Error: Checkpoint file does not exist: $CHECKPOINT"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_VIDEO")
mkdir -p "$OUTPUT_DIR"

echo "Starting GlobalFlowNet video stabilization..."
echo "Repository: $REPO_PATH"
echo "Input: $INPUT_VIDEO"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_VIDEO"
echo "GPU: $GPU_ID"

# Change to repository directory
cd "$REPO_PATH"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Convert paths to absolute paths to avoid relative path issues
INPUT_VIDEO_ABS=$(cd "$(dirname "$INPUT_VIDEO")" && pwd)/$(basename "$INPUT_VIDEO")
OUTPUT_VIDEO_ABS=$(cd "$(dirname "$OUTPUT_VIDEO")" && pwd)/$(basename "$OUTPUT_VIDEO")

# Try to run the GlobalFlowNet stabilizeVideo.py script
if [[ -f "Code/stabilizeVideo.py" ]]; then
    echo "Running GlobalFlowNet stabilizeVideo.py..."
    cd Code
    python3 stabilizeVideo.py \
        --inpVideoPath "$INPUT_VIDEO_ABS" \
        --outVideoPath "$OUTPUT_VIDEO_ABS" \
        --maxAffineCrop 0.8
    cd ..
elif [[ -f "stabilizeVideo.py" ]]; then
    echo "Running stabilizeVideo.py..."
    python3 stabilizeVideo.py \
        --inpVideoPath "$INPUT_VIDEO_ABS" \
        --outVideoPath "$OUTPUT_VIDEO_ABS" \
        --maxAffineCrop 0.8
else
    echo "Error: GlobalFlowNet stabilizeVideo.py not found"
    echo "Expected location: Code/stabilizeVideo.py or stabilizeVideo.py"
    echo "Available files:"
    find . -name "*.py" | head -10
    exit 1
fi

echo "GlobalFlowNet stabilization complete! Output saved to: $OUTPUT_VIDEO"
