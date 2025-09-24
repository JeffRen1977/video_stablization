# Video Stabilization Case Study

A practical implementation of modern video stabilization techniques, focusing on deep learning-based methods with comprehensive evaluation metrics.

## Overview

This case study demonstrates three influential video stabilization approaches:

1. **NNDVS (Minimum Latency Deep Online Video Stabilization)** - ICCV 2023
   - Online, low-latency stabilization
   - Real-time processing capabilities
   - Repository: [liuzhen03/NNDVS](https://github.com/liuzhen03/NNDVS)

2. **GlobalFlowNet** - WACV 2023
   - Global motion estimation with distillation
   - Strong baseline for comparison
   - Repository: [GlobalFlowNet/GlobalFlowNet](https://github.com/GlobalFlowNet/GlobalFlowNet)

3. **Fast-Stab (Fast Full-frame Video Stabilization)** - ICCV 2023
   - Iterative optimization with full-frame outpainting
   - High-quality results with maintained field-of-view
   - Repository: [liuzhen03/Fast-Stab](https://github.com/liuzhen03/Fast-Stab)

## Project Structure

```
video_stabilization/
├── experiments/
│   └── stabilization/
│       ├── run_nndvs.sh          # NNDVS execution script
│       ├── eval_video.py         # Comprehensive evaluation script
│       └── results/              # Output videos and results
├── samples/
│   └── stabilization/
│       ├── prepare_samples.py    # Synthetic video generation
│       └── shaky.mp4            # Sample input video
├── thirdparty/
│   ├── NNDVS/                   # NNDVS repository
│   ├── GlobalFlowNet/           # GlobalFlowNet repository
│   └── Fast-Stab/               # Fast-Stab repository
├── data/
│   └── checkpoints/             # Pretrained model weights
├── requirements.txt             # Python dependencies
├── setup_environment.sh         # Environment setup script
└── README.md                    # This file
```

## Quick Start

### 1. Environment Setup

```bash
# Run the setup script
bash setup_environment.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Download Pretrained Models

Download the required checkpoints to `data/checkpoints/`:

- **NNDVS**: Download from the [official repository](https://github.com/liuzhen03/NNDVS)
- **GlobalFlowNet**: Download from the [official repository](https://github.com/GlobalFlowNet/GlobalFlowNet)
- **Fast-Stab**: Download from the [official repository](https://github.com/liuzhen03/Fast-Stab)

### 3. Create Sample Data

```bash
# Generate synthetic shaky video
python samples/stabilization/prepare_samples.py \
    --output samples/stabilization/shaky.mp4 \
    --duration 10 \
    --fps 30
```

### 4. Run Video Stabilization

```bash
# Run NNDVS stabilization
bash experiments/stabilization/run_nndvs.sh \
    --repo thirdparty/NNDVS \
    --input samples/stabilization/shaky.mp4 \
    --ckpt data/checkpoints/nndvs_pretrained.pth \
    --out experiments/stabilization/results/nndvs_out.mp4
```

### 5. Evaluate Results

```bash
# Comprehensive evaluation
python experiments/stabilization/eval_video.py \
    --input experiments/stabilization/results/nndvs_out.mp4 \
    --original samples/stabilization/shaky.mp4 \
    --output evaluation_results.json \
    --plots experiments/stabilization/results/plots/
```

## Evaluation Metrics

The evaluation script provides comprehensive metrics:

### Temporal Smoothness
- **Mean Translation**: Average frame-to-frame translation magnitude
- **Standard Deviation**: Consistency of motion
- **Rotation Metrics**: Angular stability measures

### Quality Metrics (if ground truth available)
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **Frame-wise Analysis**: Per-frame quality assessment

### Performance Metrics
- **Processing Speed**: Frames per second
- **Memory Usage**: Computational requirements
- **Boundary Crop Ratio**: Fraction of frame lost due to cropping

### Visual Analysis
- **Motion Trajectory Plots**: Camera path visualization
- **Quality Trend Analysis**: Frame-wise quality progression
- **Comparative Visualizations**: Side-by-side comparisons

## Advanced Usage

### Custom Video Input

```bash
# Use your own shaky video
bash experiments/stabilization/run_nndvs.sh \
    --repo thirdparty/NNDVS \
    --input /path/to/your/video.mp4 \
    --ckpt data/checkpoints/nndvs_pretrained.pth \
    --out /path/to/output.mp4
```

### Batch Processing

```bash
# Process multiple videos
for video in samples/*.mp4; do
    basename=$(basename "$video" .mp4)
    bash experiments/stabilization/run_nndvs.sh \
        --repo thirdparty/NNDVS \
        --input "$video" \
        --ckpt data/checkpoints/nndvs_pretrained.pth \
        --out "experiments/stabilization/results/${basename}_stabilized.mp4"
done
```

### Comparative Analysis

```bash
# Run multiple methods and compare
python experiments/stabilization/compare_methods.py \
    --input samples/stabilization/shaky.mp4 \
    --methods nndvs globalflownet faststab \
    --output experiments/stabilization/results/comparison/
```

## Method Comparison

| Method | Type | Latency | Quality | Field of View | Use Case |
|--------|------|---------|---------|---------------|----------|
| NNDVS | Online | Low | Good | Cropped | Real-time capture |
| GlobalFlowNet | Offline | High | Very Good | Cropped | Post-processing |
| Fast-Stab | Offline | High | Excellent | Full-frame | High-quality output |

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use CPU instead
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt
   ```

3. **Video Codec Issues**
   ```bash
   # Install additional codecs
   pip install imageio-ffmpeg
   ```

### Performance Optimization

- **GPU Memory**: Reduce batch size or use gradient checkpointing
- **CPU Processing**: Use multiple workers for data loading
- **Storage**: Use SSD for faster I/O operations

## Research Applications

This case study supports research in:

- **Mobile Computational Photography**: Real-time stabilization for mobile devices
- **Computer Vision**: Robust feature tracking and SLAM
- **Video Processing**: Quality enhancement and post-production
- **Deep Learning**: Online vs offline processing tradeoffs

## Citation

If you use this case study in your research, please cite the original papers:

```bibtex
@inproceedings{nndvs2023,
  title={Minimum Latency Deep Online Video Stabilization},
  author={Zhen, Liu and others},
  booktitle={ICCV},
  year={2023}
}
```

## License

This case study is provided for educational and research purposes. Please refer to the individual repository licenses for the specific methods used.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve this case study.

## Acknowledgments

- NNDVS authors for providing the online stabilization implementation
- GlobalFlowNet team for the global motion estimation approach
- Fast-Stab authors for the full-frame stabilization method
- OpenCV community for computer vision tools
