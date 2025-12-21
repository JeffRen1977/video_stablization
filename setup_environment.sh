#!/bin/bash

# Video Stabilization Environment Setup Script
# This script sets up the environment for the video stabilization case study
# 
# IMPORTANT: This script REQUIRES an existing virtual environment in the parent directory (../virtual_env/)
# It will NOT create a new virtual environment. If the virtual_env doesn't exist, the script will exit with an error.
# 
# To create the virtual environment manually (if needed):
#   cd ..  # Go to parent directory
#   python3 -m venv virtual_env
#   cd video_stablization
#   source ../virtual_env/bin/activate

set -e

# Get the script directory and ensure we're in the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Setting up Video Stabilization Environment..."
echo "Project directory: $SCRIPT_DIR"

# Verify we're in the right directory (should contain README.md and requirements.txt)
if [ ! -f "README.md" ] || [ ! -f "requirements.txt" ]; then
    echo "Error: This script must be run from the video_stablization project root directory"
    echo "Current directory: $SCRIPT_DIR"
    exit 1
fi

# Check if virtual environment exists in parent directory
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PARENT_DIR/virtual_env"

if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found in parent directory: $VENV_PATH"
    echo ""
    echo "This script requires an existing virtual environment in the parent directory."
    echo "Please create it manually:"
    echo "  cd $PARENT_DIR"
    echo "  python3 -m venv virtual_env"
    echo "  cd video_stablization"
    echo "  source ../virtual_env/bin/activate"
    echo ""
    exit 1
fi

echo "Using existing virtual environment at $VENV_PATH"
source "$VENV_PATH/bin/activate"

# Verify venv activation
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

if [ "$VIRTUAL_ENV" != "$VENV_PATH" ]; then
    echo "Error: Virtual environment path mismatch"
    echo "Expected: $VENV_PATH"
    echo "Got: $VIRTUAL_ENV"
    exit 1
fi

echo "Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements (excluding scikit-video which has Python 2 syntax issues)
echo "Installing Python dependencies..."
echo "Note: scikit-video will be installed separately with --no-compile flag"
pip install -r requirements.txt

# Install scikit-video separately with --no-compile to avoid Python 2 syntax errors
echo "Installing scikit-video (required for GlobalFlowNet)..."
pip install scikit-video==1.1.11 --no-compile || {
    echo "Warning: scikit-video installation failed. You can install it manually:"
    echo "  pip install scikit-video==1.1.11 --no-compile"
    echo "This package has Python 2 syntax and requires the --no-compile flag."
}

# Create necessary directories
echo "Creating project directories..."
mkdir -p thirdparty
mkdir -p 03_Evaluation/results
mkdir -p 04_Experiments/results
mkdir -p samples/stabilization

# Clone NNDVS repository
echo "Cloning NNDVS repository..."
if [ ! -d "thirdparty/NNDVS" ]; then
    git clone https://github.com/liuzhen03/NNDVS.git thirdparty/NNDVS
    echo "NNDVS repository cloned successfully"
else
    echo "NNDVS repository already exists"
fi

# Clone GlobalFlowNet repository
echo "Cloning GlobalFlowNet repository..."
if [ ! -d "thirdparty/GlobalFlowNet" ]; then
    git clone https://github.com/GlobalFlowNet/GlobalFlowNet.git thirdparty/GlobalFlowNet
    echo "GlobalFlowNet repository cloned successfully"
else
    echo "GlobalFlowNet repository already exists"
fi


# Create sample data preparation script
echo "Creating sample data preparation script..."
cat > samples/stabilization/prepare_samples.py << 'EOF'
#!/usr/bin/env python3
"""
Sample Data Preparation Script
Creates synthetic shaky videos for testing video stabilization methods.
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path

def create_synthetic_shaky_video(output_path: str, duration: int = 10, fps: int = 30):
    """Create a synthetic shaky video for testing."""
    
    # Video parameters
    width, height = 640, 480
    total_frames = duration * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create a simple pattern that will show motion clearly
    for frame_idx in range(total_frames):
        # Create frame with moving pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some geometric patterns
        t = frame_idx / fps
        
        # Moving circle
        center_x = int(width // 2 + 50 * np.sin(2 * np.pi * t))
        center_y = int(height // 2 + 30 * np.cos(2 * np.pi * t))
        cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), -1)
        
        # Moving rectangle
        rect_x = int(100 + 100 * np.sin(1.5 * np.pi * t))
        rect_y = int(100 + 50 * np.cos(1.2 * np.pi * t))
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 60, rect_y + 40), (255, 0, 0), -1)
        
        # Add text
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add synthetic camera shake
        shake_x = int(5 * np.sin(10 * np.pi * t) + 3 * np.sin(20 * np.pi * t))
        shake_y = int(3 * np.cos(8 * np.pi * t) + 2 * np.cos(15 * np.pi * t))
        
        # Apply shake by shifting the frame
        M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
        frame = cv2.warpAffine(frame, M, (width, height))
        
        out.write(frame)
    
    out.release()
    print(f"Synthetic shaky video created: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create synthetic shaky videos for testing")
    parser.add_argument("--output", default="samples/stabilization/shaky.mp4", 
                       help="Output video path")
    parser.add_argument("--duration", type=int, default=10, 
                       help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=30, 
                       help="Video FPS")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    create_synthetic_shaky_video(args.output, args.duration, args.fps)
    print("Sample data preparation complete!")

if __name__ == "__main__":
    main()
EOF

chmod +x samples/stabilization/prepare_samples.py

echo ""
echo "Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source ../virtual_env/bin/activate"
echo "2. Verify checkpoints exist:"
echo "   - thirdparty/NNDVS/pretrained/pretrained_model.pth.tar"
echo "   - thirdparty/GlobalFlowNet/Code/GlobalFlowNets/trainedModels/GFlowNet.pth"
echo "3. Create sample videos: python samples/stabilization/prepare_samples.py --output samples/stabilization/shaky.mp4 --duration 5 --fps 30"
echo "4. Run complete pipeline: bash run_complete_pipeline.sh"
echo "   Or run methods individually:"
echo "   - NNDVS: bash 02_Implementation/nndvs/run_nndvs.sh --repo thirdparty/NNDVS --input samples/stabilization/shaky.mp4 --ckpt thirdparty/NNDVS/pretrained/pretrained_model.pth.tar --out 04_Experiments/results/nndvs_out.mp4"
echo "   - GlobalFlowNet: bash 02_Implementation/globalflownet/run_globalflownet.sh --repo thirdparty/GlobalFlowNet --input samples/stabilization/shaky.mp4 --ckpt thirdparty/GlobalFlowNet/Code/GlobalFlowNets/trainedModels/GFlowNet.pth --out 04_Experiments/results/globalflownet_out.mp4"
echo "5. Evaluate results: python 03_Evaluation/eval_video.py --input 04_Experiments/results/nndvs_out.mp4 --original samples/stabilization/shaky.mp4 --output 03_Evaluation/results/evaluation_results.json"
echo ""
echo "For detailed instructions, see README.md or doc/QUICK_START.md"

