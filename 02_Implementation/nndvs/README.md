# NNDVS Implementation - Chapter 17 Section 2.1

This directory contains the implementation code for NNDVS (Minimum Latency Deep Online Video Stabilization).

## Files

- `run_nndvs.sh` - Execution script for NNDVS stabilization

## Usage

> **Prerequisites**: Activate the existing virtual environment from parent directory:
> ```bash
> # From video_stablization directory
> source ../virtual_env/bin/activate  # Uses existing virtual_env in parent directory
> ```

```bash
# From the video_stablization directory
bash 02_Implementation/nndvs/run_nndvs.sh \
    --repo thirdparty/NNDVS \
    --input samples/stabilization/shaky.mp4 \
    --ckpt thirdparty/NNDVS/pretrained/pretrained_model.pth.tar \
    --out 04_Experiments/results/nndvs_out.mp4
```

## Related Chapter Section

- **Chapter 17.2.1**: NNDVS Method Implementation
- **Paper**: `../../Papers/Minimum Latency Deep Online Video Stabilization.pdf`

## Original Repository

- [liuzhen03/NNDVS](https://github.com/liuzhen03/NNDVS)

