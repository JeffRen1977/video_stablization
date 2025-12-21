# Chapter 17.3 - Evaluation Framework

This directory contains the evaluation framework for comparing video stabilization methods.

## Files

- `eval_video.py` - Comprehensive evaluation script
- `compare_methods.py` - Multi-method comparison tool
- `results/` - Evaluation results and metrics

## Usage

> **Prerequisites**: Activate the existing virtual environment from parent directory:
> ```bash
> # From video_stablization directory
> source ../virtual_env/bin/activate  # Uses existing virtual_env in parent directory
> ```

### Evaluate Single Method

```bash
# From the video_stablization directory
python 03_Evaluation/eval_video.py \
    --input 04_Experiments/results/nndvs_out.mp4 \
    --original samples/stabilization/shaky.mp4 \
    --output 03_Evaluation/results/evaluation_results.json
```

### Compare Multiple Methods

```bash
# From the video_stablization directory
python 03_Evaluation/compare_methods.py \
    --input samples/stabilization/shaky.mp4 \
    --methods nndvs globalflownet \
    --output 03_Evaluation/results/comparison/
```

## Evaluation Metrics

- **Temporal Smoothness**: Frame-to-frame motion consistency
- **Quality Metrics**: PSNR/SSIM (when ground truth available)
- **Boundary Analysis**: Crop ratio and field of view preservation
- **Performance**: Processing speed and memory usage

## Related Chapter Section

- **Chapter 17.3**: Evaluation Framework

