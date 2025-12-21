# Quick Start Guide - Video Stabilization

> **Important**: This project uses the **existing virtual environment** in the parent directory (`../virtual_env/`).

## 🚀 Easiest Way: Complete Pipeline Script

**Run everything with one command:**

```bash
# Navigate to project directory
cd /path/to/Mobile_Computational_photograph/video_stablization

# Run complete pipeline (handles everything automatically)
bash run_complete_pipeline.sh
```

This single command will:
- ✅ Verify environment and dependencies
- ✅ Create sample video (if needed)
- ✅ Run NNDVS stabilization
- ✅ Run GlobalFlowNet stabilization
- ✅ Evaluate both methods
- ✅ Compare results
- ✅ Display summary

**Options:**
```bash
# Use your own video
bash run_complete_pipeline.sh --input my_video.mp4

# Skip setup verification (faster)
bash run_complete_pipeline.sh --skip-setup

# Skip evaluation (just run stabilization)
bash run_complete_pipeline.sh --skip-eval
```

---

## Manual Step-by-Step (Alternative)

If you prefer to run steps manually:

### 1. Activate Existing Virtual Environment
```bash
# Navigate to project directory
cd /path/to/Mobile_Computational_photograph/video_stablization

# Activate the existing virtual environment from parent directory
source ../virtual_env/bin/activate

# Verify activation
which python  # Should show: .../virtual_env/bin/python
```

### 2. Install Dependencies (First Time Only)
```bash
# Install main dependencies
pip install -r requirements.txt

# Install scikit-video separately (has Python 2 syntax, needs --no-compile flag)
pip install scikit-video==1.1.11 --no-compile
```

### 3. Create Test Video
```bash
python samples/stabilization/prepare_samples.py \
    --output samples/stabilization/shaky.mp4 \
    --duration 5 --fps 30
```

### 4. Run Stabilization

**NNDVS:**
```bash
bash 02_Implementation/nndvs/run_nndvs.sh \
    --repo thirdparty/NNDVS \
    --input samples/stabilization/shaky.mp4 \
    --ckpt thirdparty/NNDVS/pretrained/pretrained_model.pth.tar \
    --out 04_Experiments/results/nndvs_out.mp4
```

**GlobalFlowNet:**
```bash
bash 02_Implementation/globalflownet/run_globalflownet.sh \
    --repo thirdparty/GlobalFlowNet \
    --input samples/stabilization/shaky.mp4 \
    --ckpt thirdparty/GlobalFlowNet/Code/GlobalFlowNets/trainedModels/GFlowNet.pth \
    --out 04_Experiments/results/globalflownet_out.mp4
```

### 5. Compare Results
```bash
python 03_Evaluation/compare_methods.py \
    --input samples/stabilization/shaky.mp4 \
    --methods nndvs globalflownet \
    --output 03_Evaluation/results/comparison/
```

## ✅ Verification Commands

```bash
# Check environment
python --version  # Should be 3.9+
which python      # Should point to ../virtual_env/bin/python

# Check packages
python -c "import torch, cv2, numpy, cvxopt, skvideo; print('✅ All OK')"

# Check files
ls thirdparty/NNDVS/pretrained/pretrained_model.pth.tar
ls thirdparty/GlobalFlowNet/Code/GlobalFlowNets/trainedModels/GFlowNet.pth
```

## 📁 Directory Structure

```
video_stablization/          ← Run commands from here
├── samples/stabilization/   ← Input videos
├── 02_Implementation/      ← Run scripts
├── 04_Experiments/results/  ← Output videos
└── 03_Evaluation/results/    ← Evaluation metrics
```

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| `CUDA not available` | Normal - code uses CPU automatically |
| `Checkpoint not found` | Verify files exist in `thirdparty/` |
| `FFmpeg error` | Install: `brew install ffmpeg` (macOS) |

For detailed instructions, see `README.md`.

