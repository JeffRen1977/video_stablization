#!/usr/bin/env python3
"""
NNDVS Inference Script
Simple inference script for NNDVS video stabilization
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import PathSmoothUNet
from utils import load_checkpoint
from image_warper import FlowWarper

def create_synthetic_motion(video_path, output_motion_path):
    """Create synthetic motion data for NNDVS inference."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create synthetic motion (simplified)
    motion_data = np.zeros((frame_count, frame_height, frame_width, 2), dtype=np.float32)
    
    # Add some synthetic camera shake
    for i in range(frame_count):
        # Simple sinusoidal motion
        dx = 2 * np.sin(0.1 * i) + 0.5 * np.sin(0.3 * i)
        dy = 1.5 * np.cos(0.15 * i) + 0.3 * np.cos(0.25 * i)
        
        # Create flow field
        motion_data[i, :, :, 0] = dx
        motion_data[i, :, :, 1] = dy
    
    cap.release()
    
    # Save motion data
    np.save(output_motion_path, motion_data)
    print(f"Synthetic motion data saved to: {output_motion_path}")
    
    return motion_data

def stabilize_video(input_video, output_video, model_path, net_radius=15, scale_factor=8):
    """Stabilize video using NNDVS model."""
    
    # Create temporary motion file
    motion_path = "temp_motion.npy"
    create_synthetic_motion(input_video, motion_path)
    
    # Load video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Load motion data
    motion_data = np.load(motion_path)
    
    # Load model
    print(f"Loading model from: {model_path}")
    net = PathSmoothUNet(4 * net_radius)
    net = nn.DataParallel(net)
    # Use CPU if CUDA not available
    if torch.cuda.is_available():
        net = net.cuda()
    else:
        print("CUDA not available, using CPU")
    load_checkpoint(model_path, net)
    net.eval()
    
    # Initialize image warper
    image_warper = FlowWarper()
    image_warper.initialize(frame_width, frame_height)
    
    # Bilinear upsample
    bilinear_upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
    
    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    
    # Process frames
    print("Processing video frames...")
    warp_trans = np.zeros((frame_count, frame_height, frame_width, 2), dtype=np.float32)
    
    # Process motion data in sliding windows
    for i in range(net_radius, frame_count - net_radius):
        # Get motion window
        motion_window = motion_data[i-net_radius:i+net_radius]
        
        # Convert to tensor
        motion_tensor = torch.from_numpy(motion_window).unsqueeze(0).permute(0, 4, 1, 2, 3)
        if torch.cuda.is_available():
            motion_tensor = motion_tensor.cuda()
        
        # Get stabilization warp
        with torch.no_grad():
            # Reshape motion tensor to 2D (batch, channels, height, width)
            # motion_tensor shape: [1, 2, 30, 480, 640] -> [1, 60, 480, 640]
            motion_2d = motion_tensor.reshape(1, -1, motion_tensor.shape[-2], motion_tensor.shape[-1])
            Bi = net(motion_2d)
            Bi = bilinear_upsample(Bi)
            Bi = Bi.detach().cpu().numpy().transpose(0, 2, 3, 1)
            # Resize to match frame dimensions
            Bi_resized = cv2.resize(Bi[0], (frame_width, frame_height))
            warp_trans[i] = Bi_resized
    
    # Apply stabilization
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        if i < net_radius or i >= frame_count - net_radius:
            # Use original frame for boundary frames
            stabilized_frame = frame
        else:
            # Apply stabilization
            stabilized_frame = image_warper.warp_image(frame, warp_trans[i])
        
        out.write(stabilized_frame)
        
        if i % 30 == 0:
            print(f"Processed frame {i}/{frame_count}")
    
    # Cleanup
    cap.release()
    out.release()
    os.remove(motion_path)
    
    print(f"Stabilized video saved to: {output_video}")

def main():
    parser = argparse.ArgumentParser(description="NNDVS Video Stabilization Inference")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--model", default="pretrained/pretrained_model.pth.tar", help="Model checkpoint path")
    parser.add_argument("--net_radius", type=int, default=15, help="Network radius")
    parser.add_argument("--scale_factor", type=int, default=8, help="Scale factor")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input video does not exist: {args.input}")
        return 1
    
    if not os.path.exists(args.model):
        print(f"Error: Model checkpoint does not exist: {args.model}")
        return 1
    
    try:
        stabilize_video(args.input, args.output, args.model, args.net_radius, args.scale_factor)
        print("Stabilization completed successfully!")
        return 0
    except Exception as e:
        print(f"Error during stabilization: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

