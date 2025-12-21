#!/usr/bin/env python3
"""
Video Stabilization Methods Comparison Script

This script runs multiple video stabilization methods and provides
comprehensive comparison analysis including quality metrics, performance,
and visual comparisons.

Usage:
    python compare_methods.py --input <video> --methods nndvs globalflownet --output <dir>
"""

import argparse
import os
import subprocess
import time
import json
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

from eval_video import evaluate_video_quality

class VideoStabilizationComparator:
    """Compare different video stabilization methods."""
    
    def __init__(self, input_video: str, output_dir: str):
        self.input_video = input_video
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Method configurations
        self.methods = {
            'nndvs': {
                'repo': 'thirdparty/NNDVS',
                'checkpoint': 'thirdparty/NNDVS/pretrained/pretrained_model.pth.tar',
                'script': '02_Implementation/nndvs/run_nndvs.sh',
                'description': 'NNDVS - Online Low-Latency'
            },
            'globalflownet': {
                'repo': 'thirdparty/GlobalFlowNet',
                'checkpoint': 'thirdparty/GlobalFlowNet/Code/GlobalFlowNets/trainedModels/GFlowNet.pth',
                'script': '02_Implementation/globalflownet/run_globalflownet.sh',
                'description': 'GlobalFlowNet - Global Motion'
            },
        }
    
    def run_method(self, method_name: str) -> Dict[str, Any]:
        """Run a specific stabilization method."""
        if method_name not in self.methods:
            raise ValueError(f"Unknown method: {method_name}")
        
        method_config = self.methods[method_name]
        output_video = self.output_dir / f"{method_name}_output.mp4"
        
        print(f"Running {method_name} ({method_config['description']})...")
        
        # Check if required files exist
        if not os.path.exists(method_config['repo']):
            print(f"Warning: Repository not found: {method_config['repo']}")
            return {"error": "Repository not found", "method": method_name}
        
        if not os.path.exists(method_config['checkpoint']):
            print(f"Warning: Checkpoint not found: {method_config['checkpoint']}")
            return {"error": "Checkpoint not found", "method": method_name}
        
        # Run the stabilization
        start_time = time.time()
        
        try:
            if method_name == 'nndvs':
                cmd = [
                    'bash', method_config['script'],
                    '--repo', method_config['repo'],
                    '--input', self.input_video,
                    '--ckpt', method_config['checkpoint'],
                    '--out', str(output_video)
                ]
            else:
                # For other methods, we'd need to implement their specific scripts
                print(f"Method {method_name} not yet implemented in wrapper")
                return {"error": "Method not implemented", "method": method_name}
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"Error running {method_name}: {result.stderr}")
                return {"error": result.stderr, "method": method_name}
            
            processing_time = time.time() - start_time
            
            return {
                "method": method_name,
                "output_video": str(output_video),
                "processing_time": processing_time,
                "success": True
            }
            
        except subprocess.TimeoutExpired:
            return {"error": "Processing timeout", "method": method_name}
        except Exception as e:
            return {"error": str(e), "method": method_name}
    
    def evaluate_methods(self, method_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate all successful methods."""
        evaluation_results = {}
        
        for result in method_results:
            if not result.get("success", False):
                continue
            
            method_name = result["method"]
            output_video = result["output_video"]
            
            print(f"Evaluating {method_name}...")
            
            try:
                eval_result = evaluate_video_quality(
                    output_video,
                    original_path=self.input_video
                )
                evaluation_results[method_name] = eval_result
            except Exception as e:
                print(f"Error evaluating {method_name}: {e}")
                evaluation_results[method_name] = {"error": str(e)}
        
        return evaluation_results
    
    def generate_comparison_report(self, method_results: List[Dict[str, Any]], 
                                 evaluation_results: Dict[str, Any]) -> None:
        """Generate comprehensive comparison report."""
        
        # Create comparison summary
        comparison_data = []
        
        for result in method_results:
            if not result.get("success", False):
                continue
            
            method_name = result["method"]
            method_config = self.methods[method_name]
            
            comparison_entry = {
                "method": method_name,
                "description": method_config["description"],
                "processing_time": result.get("processing_time", 0),
                "output_video": result.get("output_video", ""),
                "success": True
            }
            
            # Add evaluation metrics if available
            if method_name in evaluation_results and "error" not in evaluation_results[method_name]:
                eval_data = evaluation_results[method_name]
                
                if "temporal_smoothness" in eval_data:
                    smoothness = eval_data["temporal_smoothness"]
                    comparison_entry.update({
                        "mean_translation": smoothness.get("mean_translation", 0),
                        "std_translation": smoothness.get("std_translation", 0),
                        "mean_rotation": smoothness.get("mean_rotation", 0),
                        "std_rotation": smoothness.get("std_rotation", 0)
                    })
                
                if "boundary_crop_ratio" in eval_data:
                    comparison_entry["crop_ratio"] = eval_data["boundary_crop_ratio"]
                
                if "quality_metrics" in eval_data and eval_data["quality_metrics"]:
                    quality = eval_data["quality_metrics"]
                    comparison_entry.update({
                        "mean_psnr": quality.get("mean_psnr", 0),
                        "mean_ssim": quality.get("mean_ssim", 0)
                    })
                
                if "performance" in eval_data:
                    perf = eval_data["performance"]
                    comparison_entry.update({
                        "frame_count": perf.get("frame_count", 0),
                        "fps": perf.get("fps", 0),
                        "duration": perf.get("duration_seconds", 0)
                    })
            
            comparison_data.append(comparison_entry)
        
        # Save comparison data
        comparison_file = self.output_dir / "comparison_results.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        # Generate visualizations
        self.create_comparison_plots(comparison_data)
        
        # Print summary
        self.print_comparison_summary(comparison_data)
    
    def create_comparison_plots(self, comparison_data: List[Dict[str, Any]]) -> None:
        """Create comparison visualization plots."""
        
        if not comparison_data:
            print("No data available for plotting")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Video Stabilization Methods Comparison', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        methods = [d["method"] for d in comparison_data]
        descriptions = [d["description"] for d in comparison_data]
        
        # 1. Processing Time
        processing_times = [d.get("processing_time", 0) for d in comparison_data]
        axes[0, 0].bar(methods, processing_times, alpha=0.7)
        axes[0, 0].set_title('Processing Time (seconds)')
        axes[0, 0].set_ylabel('Time (s)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Temporal Smoothness (Mean Translation)
        mean_translations = [d.get("mean_translation", 0) for d in comparison_data]
        axes[0, 1].bar(methods, mean_translations, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Temporal Smoothness (Mean Translation)')
        axes[0, 1].set_ylabel('Translation Magnitude')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Boundary Crop Ratio
        crop_ratios = [d.get("crop_ratio", 0) for d in comparison_data]
        axes[0, 2].bar(methods, crop_ratios, alpha=0.7, color='lightcoral')
        axes[0, 2].set_title('Boundary Crop Ratio')
        axes[0, 2].set_ylabel('Crop Ratio')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Quality Metrics (PSNR)
        mean_psnr = [d.get("mean_psnr", 0) for d in comparison_data]
        axes[1, 0].bar(methods, mean_psnr, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Quality Metrics (Mean PSNR)')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Quality Metrics (SSIM)
        mean_ssim = [d.get("mean_ssim", 0) for d in comparison_data]
        axes[1, 1].bar(methods, mean_ssim, alpha=0.7, color='gold')
        axes[1, 1].set_title('Quality Metrics (Mean SSIM)')
        axes[1, 1].set_ylabel('SSIM')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Performance (FPS)
        fps_values = [d.get("fps", 0) for d in comparison_data]
        axes[1, 2].bar(methods, fps_values, alpha=0.7, color='plum')
        axes[1, 2].set_title('Performance (FPS)')
        axes[1, 2].set_ylabel('Frames per Second')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # Adjust layout and save
        plt.tight_layout()
        plot_file = self.output_dir / "comparison_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plots saved to: {plot_file}")
    
    def print_comparison_summary(self, comparison_data: List[Dict[str, Any]]) -> None:
        """Print a summary of the comparison results."""
        
        print("\n" + "="*80)
        print("VIDEO STABILIZATION METHODS COMPARISON SUMMARY")
        print("="*80)
        
        if not comparison_data:
            print("No successful methods to compare")
            return
        
        # Create a formatted table
        print(f"{'Method':<15} {'Description':<25} {'Time(s)':<8} {'Smoothness':<12} {'Crop%':<8} {'PSNR':<8} {'SSIM':<8}")
        print("-" * 80)
        
        for data in comparison_data:
            method = data["method"]
            desc = data["description"][:24]
            time_s = f"{data.get('processing_time', 0):.2f}"
            smoothness = f"{data.get('mean_translation', 0):.3f}"
            crop = f"{data.get('crop_ratio', 0)*100:.1f}%"
            psnr = f"{data.get('mean_psnr', 0):.2f}"
            ssim = f"{data.get('mean_ssim', 0):.3f}"
            
            print(f"{method:<15} {desc:<25} {time_s:<8} {smoothness:<12} {crop:<8} {psnr:<8} {ssim:<8}")
        
        print("\nKey Insights:")
        print("- Lower smoothness values indicate better temporal stability")
        print("- Lower crop ratios preserve more of the original field of view")
        print("- Higher PSNR/SSIM values indicate better quality (if ground truth available)")
        print("- Processing time varies based on method complexity and hardware")
        
        print(f"\nDetailed results saved to: {self.output_dir}/comparison_results.json")

def main():
    parser = argparse.ArgumentParser(description="Compare video stabilization methods")
    parser.add_argument("--input", required=True, help="Input shaky video")
    parser.add_argument("--methods", nargs='+', 
                       choices=['nndvs', 'globalflownet'],
                       default=['nndvs'],
                       help="Methods to compare")
    parser.add_argument("--output", required=True, help="Output directory for results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input video does not exist: {args.input}")
        return 1
    
    # Create comparator
    comparator = VideoStabilizationComparator(args.input, args.output)
    
    print(f"Comparing methods: {', '.join(args.methods)}")
    print(f"Input video: {args.input}")
    print(f"Output directory: {args.output}")
    
    # Run methods
    method_results = []
    for method in args.methods:
        result = comparator.run_method(method)
        method_results.append(result)
    
    # Evaluate successful methods
    successful_results = [r for r in method_results if r.get("success", False)]
    if not successful_results:
        print("No methods completed successfully")
        return 1
    
    evaluation_results = comparator.evaluate_methods(successful_results)
    
    # Generate comparison report
    comparator.generate_comparison_report(method_results, evaluation_results)
    
    print("\nComparison complete!")
    return 0

if __name__ == "__main__":
    exit(main())

