#!/usr/bin/env python3
"""
Video Stabilization Evaluation Script

This script evaluates video stabilization quality using multiple metrics:
- PSNR/SSIM (if ground truth is available)
- Temporal smoothness metrics
- Boundary crop analysis
- Runtime performance

Usage:
    python eval_video.py --input <stabilized_video> [--gt <ground_truth>] [--original <original_video>]
"""

import argparse
import cv2
import numpy as np
import time
import os
from pathlib import Path
import json
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Structural Similarity Index between two images."""
    # Simplified SSIM calculation
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1 = np.var(img1)
    sigma2 = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
           ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
    
    return ssim

def calculate_optical_flow(prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
    """Calculate optical flow between consecutive frames."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect features in previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    
    if prev_pts is None or len(prev_pts) == 0:
        return np.array([])
    
    # Calculate optical flow
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, 
        prev_pts, None,
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Select good points
    good_pts = status.ravel() == 1
    if np.sum(good_pts) > 0:
        flow = curr_pts[good_pts] - prev_pts[good_pts]
        return flow
    else:
        return np.array([])

def calculate_temporal_smoothness(video_path: str) -> Dict[str, float]:
    """Calculate temporal smoothness metrics for a video."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_translations = []
    frame_rotations = []
    prev_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if prev_frame is not None:
            # Calculate optical flow
            flow = calculate_optical_flow(prev_frame, frame)
            if flow is not None and len(flow) > 0:
                # Calculate translation magnitude
                translation = np.mean(flow, axis=0)
                translation_mag = np.linalg.norm(translation)
                frame_translations.append(translation_mag)
                
                # Simple rotation estimate (simplified)
                rotation = np.std(flow, axis=0)
                rotation_mag = np.linalg.norm(rotation)
                frame_rotations.append(rotation_mag)
        
        prev_frame = frame
    
    cap.release()
    
    if not frame_translations:
        return {"mean_translation": 0.0, "std_translation": 0.0, "mean_rotation": 0.0, "std_rotation": 0.0}
    
    return {
        "mean_translation": np.mean(frame_translations),
        "std_translation": np.std(frame_translations),
        "mean_rotation": np.mean(frame_rotations),
        "std_rotation": np.std(frame_rotations),
        "max_translation": np.max(frame_translations),
        "min_translation": np.min(frame_translations)
    }

def calculate_boundary_crop_ratio(original_path: str, stabilized_path: str) -> float:
    """Calculate the fraction of frame lost due to cropping."""
    cap_orig = cv2.VideoCapture(original_path)
    cap_stab = cv2.VideoCapture(stabilized_path)
    
    if not cap_orig.isOpened() or not cap_stab.isOpened():
        raise ValueError("Cannot open one or both videos")
    
    # Get first frame dimensions
    ret_orig, frame_orig = cap_orig.read()
    ret_stab, frame_stab = cap_stab.read()
    
    if not ret_orig or not ret_stab:
        raise ValueError("Cannot read frames from videos")
    
    orig_area = frame_orig.shape[0] * frame_orig.shape[1]
    stab_area = frame_stab.shape[0] * frame_stab.shape[1]
    
    cap_orig.release()
    cap_stab.release()
    
    crop_ratio = 1.0 - (stab_area / orig_area)
    return max(0.0, crop_ratio)

def evaluate_video_quality(stabilized_path: str, 
                          ground_truth_path: Optional[str] = None,
                          original_path: Optional[str] = None) -> Dict:
    """Comprehensive video quality evaluation."""
    
    results = {
        "stabilized_video": stabilized_path,
        "ground_truth_video": ground_truth_path,
        "original_video": original_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print(f"Evaluating video: {stabilized_path}")
    
    # Temporal smoothness metrics
    print("Calculating temporal smoothness...")
    smoothness_metrics = calculate_temporal_smoothness(stabilized_path)
    results["temporal_smoothness"] = smoothness_metrics
    
    # Boundary crop analysis
    if original_path and os.path.exists(original_path):
        print("Calculating boundary crop ratio...")
        crop_ratio = calculate_boundary_crop_ratio(original_path, stabilized_path)
        results["boundary_crop_ratio"] = crop_ratio
    else:
        results["boundary_crop_ratio"] = None
    
    # Quality metrics (if ground truth available)
    if ground_truth_path and os.path.exists(ground_truth_path):
        print("Calculating quality metrics against ground truth...")
        psnr_values, ssim_values = calculate_quality_metrics(stabilized_path, ground_truth_path)
        results["quality_metrics"] = {
            "mean_psnr": np.mean(psnr_values),
            "std_psnr": np.std(psnr_values),
            "mean_ssim": np.mean(ssim_values),
            "std_ssim": np.std(ssim_values),
            "frame_psnr": psnr_values,
            "frame_ssim": ssim_values
        }
    else:
        results["quality_metrics"] = None
    
    # Performance metrics
    print("Measuring performance...")
    start_time = time.time()
    cap = cv2.VideoCapture(stabilized_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    results["performance"] = {
        "frame_count": frame_count,
        "fps": fps,
        "duration_seconds": duration,
        "evaluation_time_seconds": time.time() - start_time
    }
    
    return results

def calculate_quality_metrics(video1_path: str, video2_path: str) -> Tuple[List[float], List[float]]:
    """Calculate PSNR and SSIM for each frame pair."""
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    psnr_values = []
    ssim_values = []
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # Resize frames to match if necessary
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        
        psnr = calculate_psnr(frame1, frame2)
        ssim = calculate_ssim(frame1, frame2)
        
        psnr_values.append(psnr)
        ssim_values.append(ssim)
    
    cap1.release()
    cap2.release()
    
    return psnr_values, ssim_values

def plot_metrics(results: Dict, output_dir: str):
    """Generate plots for the evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot temporal smoothness
    if "temporal_smoothness" in results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        smoothness = results["temporal_smoothness"]
        metrics = ["mean_translation", "std_translation", "mean_rotation", "std_rotation"]
        values = [smoothness[m] for m in metrics]
        
        ax1.bar(metrics, values)
        ax1.set_title("Temporal Smoothness Metrics")
        ax1.set_ylabel("Magnitude")
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot frame-wise PSNR if available
        if results["quality_metrics"] and "frame_psnr" in results["quality_metrics"]:
            frame_psnr = results["quality_metrics"]["frame_psnr"]
            ax2.plot(frame_psnr)
            ax2.set_title("Frame-wise PSNR")
            ax2.set_xlabel("Frame Number")
            ax2.set_ylabel("PSNR (dB)")
        else:
            ax2.text(0.5, 0.5, "No ground truth available\nfor quality metrics", 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("Quality Metrics")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "evaluation_metrics.png"), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate video stabilization quality")
    parser.add_argument("--input", required=True, help="Path to stabilized video")
    parser.add_argument("--gt", help="Path to ground truth video (optional)")
    parser.add_argument("--original", help="Path to original shaky video (optional)")
    parser.add_argument("--output", default="evaluation_results.json", help="Output JSON file")
    parser.add_argument("--plots", help="Directory to save plots (optional)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input video does not exist: {args.input}")
        return 1
    
    try:
        results = evaluate_video_quality(
            args.input, 
            args.gt, 
            args.original
        )
        
        # Save results (convert numpy types to Python types for JSON serialization)
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy_types(results)
        
        with open(args.output, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\nEvaluation complete! Results saved to: {args.output}")
        
        # Print summary
        print("\n=== EVALUATION SUMMARY ===")
        print(f"Stabilized video: {args.input}")
        
        if results["temporal_smoothness"]:
            smoothness = results["temporal_smoothness"]
            print(f"Temporal smoothness:")
            print(f"  Mean translation: {smoothness['mean_translation']:.4f}")
            print(f"  Std translation: {smoothness['std_translation']:.4f}")
            print(f"  Mean rotation: {smoothness['mean_rotation']:.4f}")
            print(f"  Std rotation: {smoothness['std_rotation']:.4f}")
        
        if results["boundary_crop_ratio"] is not None:
            print(f"Boundary crop ratio: {results['boundary_crop_ratio']:.4f}")
        
        if results["quality_metrics"]:
            quality = results["quality_metrics"]
            print(f"Quality metrics (vs ground truth):")
            print(f"  Mean PSNR: {quality['mean_psnr']:.2f} dB")
            print(f"  Mean SSIM: {quality['mean_ssim']:.4f}")
        
        if results["performance"]:
            perf = results["performance"]
            print(f"Performance:")
            print(f"  Frame count: {perf['frame_count']}")
            print(f"  FPS: {perf['fps']:.2f}")
            print(f"  Duration: {perf['duration_seconds']:.2f}s")
        
        # Generate plots if requested
        if args.plots:
            plot_metrics(results, args.plots)
            print(f"Plots saved to: {args.plots}")
        
        return 0
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
