# 17 Video Stabilization Implementation

## Abstract

This chapter presents a practical case study on modern video stabilization techniques, focusing on two influential methods: NNDVS (Minimum Latency Deep Online Video Stabilization) and GlobalFlowNet. We provide a detailed implementation and evaluation of these approaches, highlighting their architectures, algorithmic pipelines, and experimental results. NNDVS, an online low-latency deep learning solution, is compared against GlobalFlowNet, an offline method leveraging deep distilled global motion estimates. Through comprehensive testing and analysis, we compare their performance across key metrics such as temporal smoothness, field of view preservation, and computational efficiency. The chapter concludes with a discussion of the observed strengths and limitations of each method and proposes future improvements to address current challenges in video stabilization for mobile computational photography.

**Keywords**: Video Stabilization, NNDVS, GlobalFlowNet, Deep Learning, Optical Flow, Mobile Photography, Real-time Processing, Online Stabilization, Offline Processing, Motion Estimation, Computational Photography.

## 17.1 Overview

This chapter provides a comprehensive, hands-on exploration of modern video stabilization techniques through a practical case study. We implement and evaluate two influential methods that represent different approaches to solving the stabilization problem in mobile computational photography. Please refer to the GitHub repository: https://github.com/JeffRen1977/video_stablization

### 17.1.1 The Video Stabilization Problem

Video stabilization addresses the challenge of removing unwanted camera shake from video sequences. When recording with handheld devices, camera motion introduces jitter that degrades visual quality. The goal is to estimate and compensate for this unwanted motion while preserving intentional camera movements.

**Core Challenge**: Distinguish between intentional camera motion (e.g., panning, tracking) and unwanted shake, then apply appropriate corrections.

### 17.1.2 Two Approaches to Stabilization

This case study demonstrates two influential video stabilization approaches:

**1. NNDVS (Minimum Latency Deep Online Video Stabilization)** - ICCV 2023
- **Type**: Online, low-latency stabilization
- **Key Feature**: Real-time processing capabilities
- **Repository**: [liuzhen03/NNDVS](https://github.com/liuzhen03/NNDVS)

**2. GlobalFlowNet** - WACV 2023
- **Type**: Offline global motion estimation
- **Key Feature**: Deep distilled global motion estimates
- **Repository**: [GlobalFlowNet/GlobalFlowNet](https://github.com/GlobalFlowNet/GlobalFlowNet)

### 17.1.3 Basic Stabilization Pipeline

Most video stabilization methods follow a three-step pipeline:

1. **Motion Estimation**: Extract camera motion between consecutive frames
2. **Path Smoothing**: Filter the motion trajectory to remove jitter
3. **Frame Warping**: Apply smoothed transformations to generate stable output

The fundamental difference between methods lies in how they implement each step and whether they process frames online (real-time) or offline (batch processing).

## 17.2 Environment Setup

### 17.2.1 Clone and Setup

**Table 17.2.1-1.** Clone the repository

```bash
# Clone the repository
git clone https://github.com/JeffRen1977/video_stablization.git
cd video_stablization

# Make setup script executable and run it
chmod +x setup_environment.sh
./setup_environment.sh
```

### 17.2.2 Activate Virtual Environment

**Table 17.2.2-1.** Activate the virtual environment

The project uses a virtual environment located in the parent directory:

```bash
# Activate the existing virtual environment from parent directory
source ../virtual_env/bin/activate

# Verify activation - you should see (virtual_env) in your terminal prompt
which python
# Should show: .../Mobile_Computational_photograph/virtual_env/bin/python
```

**Note**: If the virtual environment doesn't exist, create it in the parent directory:
```bash
cd ..
python3 -m venv virtual_env
cd video_stablization
source ../virtual_env/bin/activate
```

### 17.2.3 Verify Installation

**Table 17.2.3-1.** Verification of installation

```bash
# Check Python version
python --version

# Verify key packages
python -c "import torch, cv2, numpy; print('All packages installed successfully!')"
```

## 17.3 Quick Start

### 17.3.1 Create Sample Data

**Table 17.3.1-1.** Prepare samples

```bash
# Generate synthetic shaky video for testing
python samples/stabilization/prepare_samples.py \
    --output samples/stabilization/shaky.mp4 \
    --duration 5 --fps 30
```

### 17.3.2 Run Video Stabilization

#### 17.3.2.1 NNDVS (Deep Learning)

**Table 17.3.2.1-1.** Run NNDVS

```bash
bash 02_Implementation/nndvs/run_nndvs.sh \
    --input samples/stabilization/shaky.mp4 \
    --output 04_Experiments/results/nndvs_out.mp4
```

#### 17.3.2.2 GlobalFlowNet (Global Motion)

**Table 17.3.2.2-1.** Run GlobalFlowNet

```bash
bash 02_Implementation/globalflownet/run_globalflownet.sh \
    --input samples/stabilization/shaky.mp4 \
    --output 04_Experiments/results/globalflownet_out.mp4
```

#### 17.3.2.3 Complete Pipeline (Recommended)

**Table 17.3.2.3-1.** Run complete pipeline

For convenience, use the complete pipeline script that runs everything:

```bash
# Run the complete pipeline (setup, stabilization, evaluation)
bash run_complete_pipeline.sh --input samples/stabilization/shaky.mp4
```

### 17.3.3 Evaluate Results

**Table 17.3.3-1.** Evaluate results

```bash
# Evaluate stabilization quality
python 03_Evaluation/eval_video.py \
    --input 04_Experiments/results/nndvs_out.mp4 \
    --original samples/stabilization/shaky.mp4 \
    --output 03_Evaluation/results/evaluation_results.json
```

### 17.3.4 Compare Methods

**Table 17.3.4-1.** Compare methods

```bash
# Compare multiple stabilization approaches
python 03_Evaluation/compare_methods.py \
    --input samples/stabilization/shaky.mp4 \
    --methods nndvs globalflownet \
    --output 03_Evaluation/results/comparison/
```

## 17.4 Video Stabilization Algorithms: Architecture and Analysis

### 17.4.1 NNDVS (Minimum Latency Deep Online Video Stabilization)

#### 17.4.1.1 Architecture Overview

NNDVS employs a **U-Net-based path smoothing network** designed for real-time video stabilization. The method focuses on the latter two steps of the stabilization pipeline: path optimization and novel view rendering, while adopting off-the-shelf high-quality deep motion models for motion estimation.

**Table 17.4.1.1-1.** U-Net based path smoothing network

```python
class PathSmoothUNet(nn.Module):
    def __init__(self, in_chn, wf=32, depth=4, relu_slope=0.2):
        # U-Net architecture with configurable depth and width factor
        self.down_path = nn.ModuleList()  # Encoder path
        self.up_path = nn.ModuleList()    # Decoder path
        self.last = conv3x3(prev_channels, 2, bias=True)  # Output 2D flow
```

**Key Components:**
- **Input**: Motion trajectory windows (4 × net_radius frames, where net_radius=15)
- **Encoder**: Progressive downsampling with skip connections
- **Decoder**: Upsampling with feature concatenation
- **Output**: 2D stabilization flow field
- **Model Size**: 16.8 MB pretrained weights

**Why U-Net?** The U-Net architecture is ideal for this task because:
- Skip connections preserve fine-grained motion details
- Encoder-decoder structure enables multi-scale motion understanding
- Efficient inference suitable for real-time processing

#### 17.4.1.2 Motion Estimation Pipeline

NNDVS uses a hybrid approach for motion estimation combining traditional computer vision with deep learning:

**Table 17.4.1.2-1.** KLT tracker

```python
class KltTracker():
    def __init__(self, back_threshold=1.0, nms=True, check_optical_flow=False):
        self.detector = cv2.FastFeatureDetector_create()
        self.back_threshold = back_threshold
        self.check_optical_flow = check_optical_flow
    
    def track_features(self, src, dst):
        # FAST feature detection + KLT tracking
        kp_src = self.detector.detect(src_gray, None)
        kp_dst, st = self.check_trace(src_gray, dst_gray, kp_src)
        return pts_src, pts_dst
```

**Motion Estimation Steps:**

1. **Feature Detection**: FAST feature detector identifies robust corner features
2. **Feature Tracking**: KLT (Kanade-Lucas-Tomasi) optical flow tracks features across frames
3. **Global Motion Estimation**: Homography estimation using RANSAC to handle outliers
4. **Motion Validation**: Bidirectional consistency checking ensures reliable motion

**Why FAST+KLT?** This combination provides:
- Fast computation suitable for real-time processing
- Robust feature detection in various lighting conditions
- Efficient tracking with sub-pixel accuracy

#### 17.4.1.3 Algorithm Pipeline

The complete NNDVS pipeline processes video in a sliding window fashion:

1. **Motion Estimation**: Extract camera motion using FAST+KLT with global homography estimation
2. **Sliding Window Processing**: Process motion in overlapping windows (net_radius=15 frames)
3. **Path Smoothing**: Apply U-Net to predict stabilization flow for the last frame in window
4. **Frame Warping**: Apply predicted transformations using flow warping

**Table 17.4.1.3-1.** Flow warper

```python
class FlowWarper():
    def warp_image(self, img, trans):
        trans = -trans + self.base_grid
        new_img = cv2.remap(img, trans[:, :, 0], trans[:, :, 1], cv2.INTER_LINEAR)
        return new_img
```

**Key Design Choice**: Sliding window processing enables online operation while maintaining temporal context. The window size (net_radius=15) balances latency and quality.

#### 17.4.1.4 Training Strategy

- **Dataset**: MotionStab dataset with stable/unstable motion pairs
- **Loss Function**: Hybrid loss combining spatial and temporal consistency
- **Training Data**: Synthesized videos with known camera trajectories
- **Data Augmentation**: Various motion patterns (Regular, QuickRotation, Crowd scenarios)

#### 17.4.1.5 Experimental Results

**Strengths:**
- **Low Latency**: Real-time processing capability
- **Online Processing**: No need for future frames
- **Memory Efficient**: Small model size (16.8 MB)
- **No Cropping**: Full field of view preservation

**Shortcomings:**
- **Quality Limitation**: Basic U-Net architecture may struggle with complex motion patterns
- **Window Dependency**: Performance depends on window size selection
- **Motion Estimation Dependency**: Quality limited by input motion estimation accuracy

### 17.4.2 GlobalFlowNet (Video Stabilization using Deep Distilled Global Motion Estimates)

#### 17.4.2.1 Algorithm Pipeline

GlobalFlowNet extends PWCNet (Pyramid, Warping, and Cost volume) with deep distilled global motion estimates. The method introduces a novel approach to video stabilization by combining local optical flow estimation with global motion understanding through DCT-based filtering.

**Pipeline Steps:**

1. **Feature Extraction**: Multi-scale convolutional feature extraction using PWCNet
2. **Cost Volume**: Correlation-based motion estimation between frame pairs
3. **Global Motion**: DCT coefficient filtering for global motion estimation
4. **Flow Prediction**: Multi-level flow prediction and refinement
5. **Stabilization**: Apply composed stabilization with affine and photometric components
6. **Cropping**: Apply boundary cropping to avoid stabilization artifacts

#### 17.4.2.2 Architecture Overview

**Table 17.4.2.2-1.** Global PWC base

```python
class GlobalPWCBase(PWCNet):
    def forward(self, im1, im2, multiScaleFlows=False):
        # Multi-scale feature extraction
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        
        # Cost volume correlation
        corr6 = self.corr(c16, c26)
        
        # DCT-based global motion filtering
        flow6 = self.filterFlow(self.predict_flow6(x), level=6)
```

**Key Components:**
- **Base Network**: PWCNet for optical flow estimation
- **Global Motion**: DCT-based filtering for global motion estimation
- **Multi-scale Processing**: 6-level pyramid architecture
- **Model Size**: 37.5 MB pretrained weights

**Why PWCNet?** PWCNet provides:
- Efficient multi-scale feature extraction
- Robust optical flow estimation
- Proven performance in motion estimation tasks

#### 17.4.2.3 DCT-Based Global Motion Estimation

The core innovation of GlobalFlowNet lies in its DCT-based global motion filtering:

**Table 17.4.2.3-1.** Using DCT-based global motion filtering

```python
class GlobalPWCDCT(GlobalPWCBase):
    def filterFlow(self, flow, level):
        cf = self.cfs[str(level)]
        UX, UY = self.getUniformGrid(flow.shape)
        filFlow = 0.0
        for u in range(cf + 1):
            for v in range(cf + 1):
                base = self.getDCTBase(UX, UY, u, v)
                coeffs = torch.tensordot(base, flow, dims=([-1, -2], [-1, -2]))
                filFlow = filFlow + coeffs[:, :, None, None] * base[None, None]
        return filFlow
```

**DCT Filtering Process:**

1. **DCT Basis Generation**: Create 2D DCT basis functions for each frequency component
2. **Coefficient Extraction**: Compute DCT coefficients for the optical flow field
3. **Frequency Filtering**: Apply configurable frequency cutoffs (cfs) at different pyramid levels
4. **Flow Reconstruction**: Reconstruct filtered flow field from DCT coefficients

**Why DCT?** Discrete Cosine Transform provides:
- Efficient frequency-domain representation
- Natural separation of global vs. local motion
- Smooth filtering without artifacts

#### 17.4.2.4 Multi-Scale Processing Architecture

GlobalFlowNet employs a sophisticated multi-scale processing pipeline:

**Table 17.4.2.4-1.** Using multiple scale processing

```python
# 6-level pyramid processing
flow6 = self.filterFlow(self.predict_flow6(x), level=6)
up_flow6 = self.filterFlow(self.deconv6(flow6), level=6)
up_feat6 = self.filterFlow(self.upfeat6(x), level=6)

# Warping and refinement at each level
warp5 = self.warp(c25, up_flow6 * 0.625)
corr5 = self.corr(c15, warp5)
```

**Pyramid Levels:**
- **Level 6**: Coarsest scale (1/64 resolution) - captures global motion
- **Level 5**: 1/32 resolution
- **Level 4**: 1/16 resolution
- **Level 3**: 1/8 resolution
- **Level 2**: 1/4 resolution
- **Level 1**: Full resolution - preserves fine details

**Why Multi-scale?** Processing at multiple scales enables:
- Robust handling of various motion magnitudes
- Efficient computation at coarse levels
- Fine detail preservation at full resolution

#### 17.4.2.5 Stabilization Pipeline

GlobalFlowNet uses a composed stabilization approach:

**Table 17.4.2.5-1.** GlobalFlowNet stabilization approach

```python
class ComposedStabilizer():
    def __init__(self, Video, OptNet, stabilizers=[], span=11, cutoffFreq=5, crop=.8):
        self.stabilizers = []
        for stabilizer in stabilizers:
            if stabilizer == 'GNetAffine':
                AffineStabilizer = JoinedAdaptiveGNetStabilizer(self.frames, OptNet, dSpan=32, crop=crop)
                self.stabilizers.append(AffineStabilizer)
            elif stabilizer == 'MSPhotometric':
                Stabilizer = MSPhotometric(self.frames, OptNet, span=13, cutoffFreq=5)
```

**Stabilization Components:**

1. **GNetAffine**: Neural network-based affine transformation estimation
2. **MSPhotometric**: Multi-scale photometric stabilization
3. **Adaptive Span**: Dynamic window size adjustment (dSpan=32)
4. **Frequency Cutoff**: Configurable frequency filtering (cutoffFreq=5)

#### 17.4.2.6 Experimental Results

**Strengths:**
- **High Quality**: Sophisticated PWCNet architecture produces excellent results
- **Global Motion Understanding**: DCT-based filtering accurately captures overall scene movement
- **Robust Processing**: Multi-scale approach handles various motion types
- **Well-Established**: Based on proven PWCNet framework

**Shortcomings:**
- **High Latency**: Requires full video for offline processing
- **Memory Intensive**: Large model (37.5 MB) with high GPU memory requirements
- **Field of View Loss**: Cropping (crop=0.8) reduces usable frame area
- **Batch Dependency**: Cannot process frames independently

## 17.5 Experimental Results and Comparison

### 17.5.1 Test Setup

- **Input Video**: Synthetic shaky video (5 seconds, 30 FPS, 640x480)
- **Evaluation Metrics**: Temporal smoothness, boundary crop ratio, performance metrics
- **Test Environment**: CPU-only processing (no GPU acceleration)

### 17.5.2 Methods Compared

#### 17.5.2.1 NNDVS (Minimum Latency Deep Online Video Stabilization)

- **Type**: Deep learning-based online stabilization
- **Architecture**: U-Net path smoothing network
- **Model Size**: 16.8 MB
- **Processing**: Real-time, sliding window approach

#### 17.5.2.2 OpenCV-based Stabilization (Baseline)

- **Type**: Traditional computer vision approach
- **Architecture**: FAST feature detection + KLT tracking + affine transformation
- **Processing**: Frame-by-frame optical flow estimation

### 17.5.3 Experimental Results

#### 17.5.3.1 Temporal Smoothness Analysis

**Table 17.5.3.1-1.** Temporal smoothness analysis

| Method | Mean Translation | Std Translation | Mean Rotation | Std Rotation | Max Translation | Min Translation |
|--------|------------------|-----------------|---------------|--------------|-----------------|-----------------|
| **NNDVS** | 5.37 | 2.31 | 4.23 | 0.72 | 10.34 | 0.66 |
| **OpenCV** | 7.90 | 3.25 | 6.45 | 2.85 | 15.32 | 1.01 |

**Key Observations:**
- NNDVS shows **32% better mean translation** (5.37 vs 7.90 pixels)
- **29% better translation consistency** (2.31 vs 3.25 std)
- **34% better rotation smoothness** (4.23 vs 6.45 degrees)
- **75% better rotation consistency** (0.72 vs 2.85 std)

#### 17.5.3.2 Performance Metrics

**Table 17.5.3.2-1.** Performance metrics

| Method | Frame Count | FPS | Duration (s) | Evaluation Time (s) | Output Size (KB) |
|--------|-------------|-----|--------------|-------------------|------------------|
| **NNDVS** | 150 | 30.0 | 5.00 | 0.00045 | 482 |
| **OpenCV** | 149 | 30.0 | 4.97 | 0.00038 | 565 |

**Key Observations:**
- Both methods achieve real-time processing (30 FPS)
- Similar evaluation times (~0.0004 seconds per evaluation)
- NNDVS produces slightly smaller output (482 vs 565 KB)

#### 17.5.3.3 Quality Metrics

**Table 17.5.3.3-1.** Quality metrics

| Method | Boundary Crop Ratio | Field of View Preservation | Processing Type |
|--------|-------------------|---------------------------|-----------------|
| **NNDVS** | 0.0% | Full preservation | Online |
| **OpenCV** | 0.0% | Full preservation | Online |

**Key Observations:**
- Both methods preserve full field of view (0% crop ratio)
- No boundary artifacts or content loss
- Maintains original video dimensions

### 17.5.4 Technical Analysis

#### 17.5.4.1 NNDVS Advantages

1. **Superior Motion Smoothing**: Deep learning approach captures complex motion patterns
2. **Consistent Performance**: Lower standard deviation indicates more stable results
3. **Real-time Capability**: Maintains 30 FPS processing with neural network inference
4. **Robust Architecture**: U-Net design handles various motion types effectively

#### 17.5.4.2 OpenCV Advantages

1. **Simplicity**: Traditional computer vision approach, easier to understand
2. **No Model Dependencies**: No need for pretrained neural network weights
3. **Fast Evaluation**: Slightly faster evaluation time (0.00038 vs 0.00045 seconds)
4. **Wide Compatibility**: Works on any system with OpenCV

#### 17.5.4.3 Limitations Observed

1. **CPU Processing**: Both methods tested on CPU-only, limiting performance
2. **Synthetic Data**: Results based on synthetic shaky video, may not reflect real-world scenarios
3. **Single Video Test**: Limited to one test video, needs broader evaluation
4. **No Ground Truth**: Quality metrics (PSNR/SSIM) not available without reference

### 17.5.5 Method Comparison Summary

**Table 17.5.5-1.** Method comparison

| Method | Type | Latency | Quality | Field of View | Use Case |
|--------|------|---------|---------|---------------|----------|
| **NNDVS** | Online | Low | Good | Full | Real-time capture |
| **GlobalFlowNet** | Offline | High | Very Good | Cropped | Post-processing |
| **OpenCV** | Online | Low | Fair | Full | Simple applications |

## 17.6 Shortcomings and Future Improvements

### 17.6.1 Current Limitations

#### 17.6.1.1 NNDVS Limitations

1. **Motion Estimation Dependency**: Quality limited by input motion accuracy from FAST+KLT
2. **Window Size Sensitivity**: Performance varies with sliding window size (net_radius=15)
3. **Simple Architecture**: Basic U-Net may not capture complex motion patterns
4. **No Global Context**: Lacks global motion understanding beyond local path smoothing
5. **Feature Tracking Failures**: KLT tracking can fail in low-texture regions
6. **Homography Assumption**: Assumes planar scene motion, limiting effectiveness in 3D scenes

#### 17.6.1.2 GlobalFlowNet Limitations

1. **High Latency**: Cannot process frames in real-time due to batch processing requirement
2. **Memory Intensive**: Requires significant GPU memory (37.5 MB model + intermediate features)
3. **Field of View Loss**: Cropping reduces usable frame area (crop=0.8 by default)
4. **Batch Processing**: Cannot handle streaming video, requires full video for processing
5. **DCT Filtering Overhead**: DCT computation adds computational complexity
6. **Multi-scale Dependency**: Performance depends on pyramid level configuration

### 17.6.2 Proposed Improvements

#### 17.6.2.1 For NNDVS

**Enhanced Motion Estimation:**
- Replace FAST+KLT with deep learning-based optical flow (RAFT, PWCNet)
- Implement robust feature matching with learned descriptors
- Add motion validation using temporal consistency

**Attention Mechanisms:**
- Add spatial attention to focus on important motion regions
- Implement temporal attention for long-range dependencies
- Use self-attention in U-Net skip connections

**Multi-scale Processing:**
- Implement pyramid-based processing similar to GlobalFlowNet
- Add multi-resolution path smoothing
- Use scale-aware loss functions

**Global Motion Integration:**
- Combine local path smoothing with global motion estimation
- Add DCT-based global motion filtering
- Implement adaptive global-local motion fusion

#### 17.6.2.2 For GlobalFlowNet

**Real-time Adaptation:**
- Develop online processing variants with sliding window approach
- Implement incremental DCT computation
- Add frame-by-frame processing capability

**Memory Optimization:**
- Implement model compression and quantization
- Use gradient checkpointing for memory efficiency
- Develop lightweight DCT filtering variants

**Field of View Preservation:**
- Develop inpainting methods to avoid cropping
- Implement content-aware resizing
- Add seam carving for boundary preservation

**Streaming Support:**
- Enable frame-by-frame processing
- Implement buffered processing with minimal latency
- Add adaptive quality adjustment based on processing time

#### 17.6.2.3 General Improvements

**Hybrid Approaches:**
- Combine online and offline processing
- Use NNDVS for real-time preview, GlobalFlowNet for final output
- Implement quality-adaptive processing

**Perceptual Metrics:**
- Integrate human perception-based evaluation
- Add perceptual loss functions
- Implement user study validation

**Mobile Optimization:**
- Develop mobile-specific architectures
- Implement model pruning and quantization
- Add hardware-specific optimizations (NPU, GPU)

**Joint Processing:**
- Integrate with denoising, exposure compensation
- Add super-resolution capabilities
- Implement end-to-end video enhancement pipeline

### 17.6.3 Future Research Directions

1. **Neural Architecture Search**: Automatically discover optimal architectures for video stabilization
2. **Self-supervised Learning**: Reduce dependency on labeled training data
3. **Multi-modal Fusion**: Combine visual, inertial, and depth information
4. **Edge Computing**: Optimize for mobile and embedded devices
5. **Real-time Global Motion**: Develop efficient global motion estimation for online processing
6. **Advanced Motion Models**: Implement 3D scene flow estimation and object-aware motion segmentation

## 17.7 Conclusion

This chapter presented a practical implementation and evaluation of two modern video stabilization methods: NNDVS and GlobalFlowNet. Through hands-on experimentation, we explored:

- **Fundamental Concepts**: The three-step stabilization pipeline (motion estimation, path smoothing, frame warping)
- **Architecture Design**: U-Net-based path smoothing (NNDVS) vs. DCT-based global motion filtering (GlobalFlowNet)
- **Trade-offs**: Online vs. offline processing, quality vs. latency, field of view preservation
- **Practical Implementation**: Real-world considerations including CPU/GPU compatibility, path handling, and evaluation metrics

**Key Takeaways:**

1. **Online vs. Offline**: NNDVS enables real-time stabilization but with quality trade-offs, while GlobalFlowNet provides higher quality but requires batch processing.

2. **Motion Estimation Matters**: The quality of stabilization is fundamentally limited by motion estimation accuracy, whether using traditional (FAST+KLT) or deep learning (PWCNet) approaches.

3. **Architecture Choices**: U-Net provides efficient real-time processing, while multi-scale pyramid architectures enable robust global motion understanding.

4. **Evaluation Metrics**: Temporal smoothness, field of view preservation, and computational efficiency are all critical considerations for practical deployment.

**For Readers**: This case study provides a foundation for understanding video stabilization in mobile computational photography. The code repository (https://github.com/JeffRen1977/video_stablization) contains all implementations, allowing readers to experiment with different methods and understand the practical challenges in real-world deployment.

**Future Work**: The field continues to evolve toward hybrid approaches that combine the benefits of online and offline processing, with increasing focus on mobile optimization and perceptual quality metrics.

