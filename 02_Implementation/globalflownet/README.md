# GlobalFlowNet Implementation - Chapter 17 Section 2.2

This directory contains the implementation code for GlobalFlowNet (Video Stabilization using Deep Distilled Global Motion Estimates).

## Files

- `run_globalflownet.sh` - Execution script for GlobalFlowNet stabilization

## Usage

> **Prerequisites**: Activate the existing virtual environment from parent directory:
> ```bash
> # From video_stablization directory
> source ../virtual_env/bin/activate  # Uses existing virtual_env in parent directory
> ```

```bash
# From the video_stablization directory
bash 02_Implementation/globalflownet/run_globalflownet.sh \
    --repo thirdparty/GlobalFlowNet \
    --input samples/stabilization/shaky.mp4 \
    --ckpt thirdparty/GlobalFlowNet/Code/GlobalFlowNets/trainedModels/GFlowNet.pth \
    --out 04_Experiments/results/globalflownet_out.mp4
```

## Related Chapter Section

- **Chapter 17.2.2**: GlobalFlowNet Method Implementation
- **Paper**: `../../Papers/GlobalFlowNet- Video Stabilization using Deep Distilled Global Motion Estimates.pdf`

## Original Repository

- [GlobalFlowNet/GlobalFlowNet](https://github.com/GlobalFlowNet/GlobalFlowNet)

