#!/bin/bash

################################################################################
# Complete Video Stabilization Pipeline Script
# 
# This script runs the complete video stabilization pipeline:
# 1. Environment setup and verification
# 2. Sample video creation
# 3. NNDVS stabilization
# 4. GlobalFlowNet stabilization
# 5. Evaluation of results
# 6. Comparison of methods
#
# IMPORTANT: This script REQUIRES an existing virtual environment in the parent 
# directory (../virtual_env/). It will NOT create a new virtual environment.
#
# Usage:
#   bash run_complete_pipeline.sh [--input <video>] [--skip-setup] [--skip-eval]
#
# Options:
#   --input <video>    : Use existing video instead of creating sample
#   --skip-setup       : Skip environment setup verification
#   --skip-eval        : Skip evaluation and comparison steps
#   --help             : Show this help message
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
INPUT_VIDEO=""
SKIP_SETUP=false
SKIP_EVAL=false
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$PROJECT_DIR")"
VENV_PATH="$PARENT_DIR/virtual_env"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_VIDEO="$2"
            shift 2
            ;;
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--input <video>] [--skip-setup] [--skip-eval] [--help]"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

################################################################################
# Helper Functions
################################################################################

print_section() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "$1 is not installed"
        return 1
    fi
    return 0
}

################################################################################
# Step 1: Environment Setup and Verification
################################################################################

if [ "$SKIP_SETUP" = false ]; then
    print_section "Step 1: Environment Setup and Verification"
    
    print_step "Checking project directory..."
    if [ ! -d "$PROJECT_DIR" ]; then
        print_error "Project directory not found: $PROJECT_DIR"
        exit 1
    fi
    cd "$PROJECT_DIR" || exit 1
    print_success "Project directory: $PROJECT_DIR"
    
    print_step "Checking virtual environment..."
    if [ ! -d "$VENV_PATH" ]; then
        print_error "Virtual environment not found at $VENV_PATH"
        echo ""
        echo "This script requires an existing virtual environment in the parent directory."
        echo "Please create it manually:"
        echo "  cd $PARENT_DIR"
        echo "  python3 -m venv virtual_env"
        echo "  cd video_stablization"
        echo "  source ../virtual_env/bin/activate"
        echo ""
        exit 1
    else
        print_success "Virtual environment found at $VENV_PATH"
    fi
    
    print_step "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
    if [ "$VIRTUAL_ENV" != "$VENV_PATH" ]; then
        print_error "Failed to activate virtual environment"
        exit 1
    fi
    print_success "Virtual environment activated: $VIRTUAL_ENV"
    
    print_step "Verifying Python version..."
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
        print_error "Python 3.9+ required. Found: $PYTHON_VERSION"
        exit 1
    fi
    print_success "Python version: $PYTHON_VERSION"
    
    print_step "Checking required packages..."
    MISSING_PACKAGES=()
    for package in torch cv2 numpy scipy; do
        if ! python -c "import $package" 2>/dev/null; then
            MISSING_PACKAGES+=("$package")
        fi
    done
    
    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        print_warning "Missing packages: ${MISSING_PACKAGES[*]}"
        print_step "Installing requirements..."
        pip install -q -r requirements.txt
        print_success "Requirements installed"
    else
        print_success "All required packages are installed"
    fi
    
    print_step "Checking FFmpeg..."
    if ! check_command ffmpeg; then
        print_warning "FFmpeg not found. Video processing may fail."
        print_warning "Install with: brew install ffmpeg (macOS) or sudo apt-get install ffmpeg (Linux)"
    else
        print_success "FFmpeg is available"
    fi
    
    print_step "Checking checkpoint files..."
    NNDVS_CHECKPOINT="$PROJECT_DIR/thirdparty/NNDVS/pretrained/pretrained_model.pth.tar"
    GLOBALFLOWNET_CHECKPOINT="$PROJECT_DIR/thirdparty/GlobalFlowNet/Code/GlobalFlowNets/trainedModels/GFlowNet.pth"
    
    if [ ! -f "$NNDVS_CHECKPOINT" ]; then
        print_error "NNDVS checkpoint not found: $NNDVS_CHECKPOINT"
        exit 1
    fi
    print_success "NNDVS checkpoint found"
    
    if [ ! -f "$GLOBALFLOWNET_CHECKPOINT" ]; then
        print_error "GlobalFlowNet checkpoint not found: $GLOBALFLOWNET_CHECKPOINT"
        exit 1
    fi
    print_success "GlobalFlowNet checkpoint found"
fi

################################################################################
# Step 2: Prepare Input Video
################################################################################

print_section "Step 2: Prepare Input Video"

if [ -n "$INPUT_VIDEO" ]; then
    if [ ! -f "$INPUT_VIDEO" ]; then
        print_error "Input video not found: $INPUT_VIDEO"
        exit 1
    fi
    INPUT_VIDEO_PATH="$INPUT_VIDEO"
    print_success "Using provided input video: $INPUT_VIDEO_PATH"
else
    print_step "Creating sample shaky video..."
    SAMPLE_VIDEO="$PROJECT_DIR/samples/stabilization/shaky.mp4"
    
    if [ ! -f "$SAMPLE_VIDEO" ]; then
        python samples/stabilization/prepare_samples.py \
            --output "$SAMPLE_VIDEO" \
            --duration 5 \
            --fps 30
        
        if [ ! -f "$SAMPLE_VIDEO" ]; then
            print_error "Failed to create sample video"
            exit 1
        fi
        print_success "Sample video created: $SAMPLE_VIDEO"
    else
        print_success "Sample video already exists: $SAMPLE_VIDEO"
    fi
    INPUT_VIDEO_PATH="$SAMPLE_VIDEO"
fi

# Create output directories
mkdir -p "$PROJECT_DIR/04_Experiments/results"
mkdir -p "$PROJECT_DIR/03_Evaluation/results/comparison"

################################################################################
# Step 3: Run NNDVS Stabilization
################################################################################

print_section "Step 3: Run NNDVS Stabilization"

NNDVS_OUTPUT="$PROJECT_DIR/04_Experiments/results/nndvs_out.mp4"
NNDVS_SCRIPT="$PROJECT_DIR/02_Implementation/nndvs/run_nndvs.sh"

print_step "Running NNDVS stabilization..."
bash "$NNDVS_SCRIPT" \
    --repo "$PROJECT_DIR/thirdparty/NNDVS" \
    --input "$INPUT_VIDEO_PATH" \
    --ckpt "$NNDVS_CHECKPOINT" \
    --out "$NNDVS_OUTPUT"

if [ ! -f "$NNDVS_OUTPUT" ]; then
    print_error "NNDVS stabilization failed - output file not created"
    exit 1
fi
print_success "NNDVS stabilization completed: $NNDVS_OUTPUT"

################################################################################
# Step 4: Run GlobalFlowNet Stabilization
################################################################################

print_section "Step 4: Run GlobalFlowNet Stabilization"

GLOBALFLOWNET_OUTPUT="$PROJECT_DIR/04_Experiments/results/globalflownet_out.mp4"
GLOBALFLOWNET_SCRIPT="$PROJECT_DIR/02_Implementation/globalflownet/run_globalflownet.sh"

print_step "Running GlobalFlowNet stabilization..."
bash "$GLOBALFLOWNET_SCRIPT" \
    --repo "$PROJECT_DIR/thirdparty/GlobalFlowNet" \
    --input "$INPUT_VIDEO_PATH" \
    --ckpt "$GLOBALFLOWNET_CHECKPOINT" \
    --out "$GLOBALFLOWNET_OUTPUT"

if [ ! -f "$GLOBALFLOWNET_OUTPUT" ]; then
    print_error "GlobalFlowNet stabilization failed - output file not created"
    exit 1
fi
print_success "GlobalFlowNet stabilization completed: $GLOBALFLOWNET_OUTPUT"

################################################################################
# Step 5: Evaluate Results
################################################################################

if [ "$SKIP_EVAL" = false ]; then
    print_section "Step 5: Evaluate Results"
    
    print_step "Evaluating NNDVS output..."
    NNDVS_EVAL_OUTPUT="$PROJECT_DIR/03_Evaluation/results/nndvs_evaluation.json"
    python "$PROJECT_DIR/03_Evaluation/eval_video.py" \
        --input "$NNDVS_OUTPUT" \
        --original "$INPUT_VIDEO_PATH" \
        --output "$NNDVS_EVAL_OUTPUT"
    
    if [ -f "$NNDVS_EVAL_OUTPUT" ]; then
        print_success "NNDVS evaluation completed: $NNDVS_EVAL_OUTPUT"
    else
        print_warning "NNDVS evaluation may have failed"
    fi
    
    print_step "Evaluating GlobalFlowNet output..."
    GLOBALFLOWNET_EVAL_OUTPUT="$PROJECT_DIR/03_Evaluation/results/globalflownet_evaluation.json"
    python "$PROJECT_DIR/03_Evaluation/eval_video.py" \
        --input "$GLOBALFLOWNET_OUTPUT" \
        --original "$INPUT_VIDEO_PATH" \
        --output "$GLOBALFLOWNET_EVAL_OUTPUT"
    
    if [ -f "$GLOBALFLOWNET_EVAL_OUTPUT" ]; then
        print_success "GlobalFlowNet evaluation completed: $GLOBALFLOWNET_EVAL_OUTPUT"
    else
        print_warning "GlobalFlowNet evaluation may have failed"
    fi
    
    ################################################################################
    # Step 6: Compare Methods
    ################################################################################
    
    print_section "Step 6: Compare Methods"
    
    print_step "Running method comparison..."
    COMPARISON_OUTPUT="$PROJECT_DIR/03_Evaluation/results/comparison"
    python "$PROJECT_DIR/03_Evaluation/compare_methods.py" \
        --input "$INPUT_VIDEO_PATH" \
        --methods nndvs globalflownet \
        --output "$COMPARISON_OUTPUT"
    
    if [ -d "$COMPARISON_OUTPUT" ]; then
        print_success "Comparison completed: $COMPARISON_OUTPUT"
    else
        print_warning "Comparison may have failed"
    fi
fi

################################################################################
# Step 7: Summary
################################################################################

print_section "Pipeline Complete!"

echo -e "${GREEN}Summary:${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Input Video:     $INPUT_VIDEO_PATH"
echo "NNDVS Output:    $NNDVS_OUTPUT"
echo "GlobalFlowNet:  $GLOBALFLOWNET_OUTPUT"

if [ "$SKIP_EVAL" = false ]; then
    echo ""
    echo "Evaluation Results:"
    echo "  - NNDVS:        $NNDVS_EVAL_OUTPUT"
    echo "  - GlobalFlowNet: $GLOBALFLOWNET_EVAL_OUTPUT"
    echo "  - Comparison:   $COMPARISON_OUTPUT"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Show file sizes
if [ -f "$NNDVS_OUTPUT" ] && [ -f "$GLOBALFLOWNET_OUTPUT" ]; then
    echo ""
    echo "Output File Sizes:"
    ls -lh "$NNDVS_OUTPUT" "$GLOBALFLOWNET_OUTPUT" | awk '{print "  " $9 ": " $5}'
fi

# Show evaluation summary if available
if [ "$SKIP_EVAL" = false ] && [ -f "$NNDVS_EVAL_OUTPUT" ] && [ -f "$GLOBALFLOWNET_EVAL_OUTPUT" ]; then
    echo ""
    echo "Quick Evaluation Summary:"
    python -c "
import json
import sys

try:
    with open('$NNDVS_EVAL_OUTPUT', 'r') as f:
        nndvs = json.load(f)
    with open('$GLOBALFLOWNET_EVAL_OUTPUT', 'r') as f:
        gfn = json.load(f)
    
    print(f\"  NNDVS FPS:        {nndvs.get('fps', 'N/A'):.2f}\")
    print(f\"  GlobalFlowNet FPS: {gfn.get('fps', 'N/A'):.2f}\")
    print(f\"  NNDVS Processing:  {nndvs.get('processing_time', {}).get('total', 'N/A'):.2f}s\")
    print(f\"  GlobalFlowNet Processing: {gfn.get('processing_time', {}).get('total', 'N/A'):.2f}s\")
except Exception as e:
    print(f\"  Could not parse evaluation results: {e}\")
" 2>/dev/null || echo "  (Evaluation results parsing failed)"
fi

echo ""
print_success "Pipeline execution completed successfully!"
echo ""
echo "To view results:"
echo "  - Videos: ls -lh $PROJECT_DIR/04_Experiments/results/*.mp4"
if [ "$SKIP_EVAL" = false ]; then
    echo "  - Evaluation: cat $PROJECT_DIR/03_Evaluation/results/comparison/comparison_summary.json"
fi
echo ""

