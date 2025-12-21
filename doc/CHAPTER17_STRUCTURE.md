# Chapter 17 - Video Stabilization: Code Organization Guide

This document explains how the code in this repository is organized to follow Chapter 17 of "Mobile Computational Photography v2".

## Repository Structure for Chapter 17

```
video_stabilization/
├── README.md                          # Main chapter guide
├── Papers/                            # 📄 Research papers referenced in Chapter 17
│   ├── Minimum Latency Deep Online Video Stabilization.pdf
│   ├── GlobalFlowNet- Video Stabilization using Deep Distilled Global Motion Estimates.pdf
│   └── README.md                     # Paper citations and details
│
├── 01_Introduction/                   # 📖 Section 1: Introduction to Video Stabilization
│   ├── notebooks/
│   │   └── video_stabilization_analysis.ipynb
│   └── README.md
│
├── 02_Implementation/                 # 🔧 Section 2: Implementation of Methods
│   ├── nndvs/                        # NNDVS implementation
│   │   ├── run_nndvs.sh
│   │   └── README.md
│   └── globalflownet/                # GlobalFlowNet implementation
│       ├── run_globalflownet.sh
│       └── README.md
│
├── 03_Evaluation/                     # 📊 Section 3: Evaluation Framework
│   ├── eval_video.py                 # Main evaluation script
│   ├── compare_methods.py            # Comparison tool
│   ├── results/                      # Evaluation results
│   └── README.md
│
├── 04_Experiments/                    # 🧪 Section 4: Experimental Results
│   ├── results/                      # Experimental outputs
│   └── README.md
│
├── samples/                           # 📹 Sample data for testing
│   └── stabilization/
│       ├── prepare_samples.py
│       └── shaky.mp4
│
├── thirdparty/                        # 🔗 Original open-source implementations
│   ├── NNDVS/                        # Original NNDVS repository
│   └── GlobalFlowNet/                # Original GlobalFlowNet repository
│
├── requirements.txt                   # Python dependencies
├── setup_environment.sh              # Environment setup script
└── CHAPTER17_STRUCTURE.md            # This file - Structure guide
```

**Note**: The virtual environment (`venv/`) is located in the parent directory.

## How to Use This Repository with Chapter 17

### Step 1: Read the Papers
Start by reading the papers in the `Papers/` folder to understand the theoretical foundations.

### Step 2: Follow the Implementation Sections
- **Section 1**: Understand why video stabilization matters
- **Section 2**: Implement and run the two methods
- **Section 3**: Evaluate the results
- **Section 4**: Analyze experimental findings

### Step 3: Run the Code
Follow the README.md instructions to set up and run the implementations.

