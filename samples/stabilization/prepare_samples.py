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
