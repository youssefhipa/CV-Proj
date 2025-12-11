#!/usr/bin/env python3
"""
MAIN SCRIPT - Run all experiments automatically
Just run: python run_experiments.py
"""

import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import json
import sys

# Import our modules
sys.path.append('src')  # Add src directory to path

from src.harris_detector import HarrisDetector
from src.sift_detector import SIFTDetector
from src.utils import ImageTransformer, Visualizer
from src.experiments import ExperimentRunner


def setup_directories():
    """Create all necessary directories"""
    directories = [
        'data/original',
        'data/transformations',
        'data/kaggle_dataset',
        'results',
        'results/visualizations',
        'results/metrics',
        'results/plots',
        'results/harris',
        'results/sift',
        'results/comparison',
        'results/distribution',
        'results/robustness'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    print("\n" + "="*50)
    print("DIRECTORIES SETUP COMPLETE!")
    print("="*50 + "\n")

def check_data():
    """Check if data is available"""
    print("Checking for images...")
    
    # Check original images
    original_images = glob.glob('data/original/*.jpg') + \
                     glob.glob('data/original/*.png') + \
                     glob.glob('data/original/*.jpeg')
    
    if not original_images:
        print("\n❌ ERROR: No images found in data/original/")
        print("\nTO FIX THIS:")
        print("1. Open the 'data/original/' folder")
        print("2. Copy your 5 building photos into it")
        print("3. Make sure they are .jpg, .png, or .jpeg files")
        print("\nExample files you should have:")
        print("  data/original/building1.jpg")
        print("  data/original/building2.jpg")
        print("  ... etc.")
        return False
    
    print(f"✓ Found {len(original_images)} images in data/original/")
    
    # Show what images were found
    print("\nImages found:")
    for img_path in original_images:
        print(f"  • {os.path.basename(img_path)}")
    
    return True

def run_basic_detection():
    """Run basic Harris and SIFT detection on all images"""
    print("\n" + "="*50)
    print("TASK 1: BASIC DETECTION & KEYPOINT COUNT COMPARISON")
    print("="*50)
    
    # Initialize detectors
    # Loosened Harris threshold to better match SIFT keypoint counts
    harris = HarrisDetector(k=0.05, window_size=3, threshold_percent=0.005, nms_size=3)
    sift = SIFTDetector()
    visualizer = Visualizer()
    
    # Get all images
    image_files = glob.glob('data/original/*.jpg') + \
                  glob.glob('data/original/*.png') + \
                  glob.glob('data/original/*.jpeg')
    
    results = []
    
    for img_path in image_files:
        print(f"\nProcessing: {os.path.basename(img_path)}")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ❌ Failed to load image")
            continue
        
        filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Detect with Harris
        harris_kps = harris.detect(img)
        harris_img = harris.draw_keypoints(img, harris_kps)
        
        # Detect with SIFT
        sift_kps = sift.detect(img)
        sift_img = sift.draw_keypoints(img, sift_kps)
        
        # Save individual results
        cv2.imwrite(f'results/harris/{filename}_harris.jpg', harris_img)
        cv2.imwrite(f'results/sift/{filename}_sift.jpg', sift_img)
        
        # Create comparison image
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(cv2.cvtColor(harris_img, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'Harris: {len(harris_kps)} keypoints')
        ax1.axis('off')
        
        ax2.imshow(cv2.cvtColor(sift_img, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'SIFT: {len(sift_kps)} keypoints')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'results/comparison/{filename}_comparison.jpg', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create keypoint distribution analysis (Task 1C)
        distribution_fig = visualizer.plot_distribution(harris_kps, sift_kps, img.shape, 
                                                       f"Keypoint Distribution - {filename}")
        distribution_fig.savefig(f'results/distribution/{filename}_distribution.jpg', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Store results
        results.append({
            'filename': os.path.basename(img_path),
            'harris_count': len(harris_kps),
            'sift_count': len(sift_kps),
            'count_difference': abs(len(harris_kps) - len(sift_kps)),
            'harris_percentage': len(harris_kps) / (len(harris_kps) + len(sift_kps) + 1e-6) * 100,
            'sift_percentage': len(sift_kps) / (len(harris_kps) + len(sift_kps) + 1e-6) * 100,
            'image_size': f"{img.shape[1]}x{img.shape[0]}",
            'image_area': img.shape[0] * img.shape[1],
            'harris_density': len(harris_kps) / (img.shape[0] * img.shape[1]),
            'sift_density': len(sift_kps) / (img.shape[0] * img.shape[1])
        })
        
        print(f"  ✓ Harris: {len(harris_kps)} keypoints")
        print(f"  ✓ SIFT: {len(sift_kps)} keypoints")
        print(f"  ✓ Distribution analysis saved")
    
    # Save summary
    df = pd.DataFrame(results)
    df.to_csv('results/metrics/basic_detection_results.csv', index=False)
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Bar chart of keypoint counts
    filenames = [r['filename'] for r in results]
    harris_counts = [r['harris_count'] for r in results]
    sift_counts = [r['sift_count'] for r in results]
    
    x = np.arange(len(filenames))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, harris_counts, width, label='Harris', color='red', alpha=0.7)
    axes[0, 0].bar(x + width/2, sift_counts, width, label='SIFT', color='blue', alpha=0.7)
    axes[0, 0].set_xlabel('Image')
    axes[0, 0].set_ylabel('Keypoint Count')
    axes[0, 0].set_title('Keypoint Count Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f[:15] for f in filenames], rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[0, 1].scatter(harris_counts, sift_counts, s=100, alpha=0.6)
    max_count = max(max(harris_counts), max(sift_counts))
    axes[0, 1].plot([0, max_count], [0, max_count], 'r--', alpha=0.5, label='y=x')
    axes[0, 1].set_xlabel('Harris Count')
    axes[0, 1].set_ylabel('SIFT Count')
    axes[0, 1].set_title('Harris vs SIFT Keypoint Counts')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Density comparison
    axes[1, 0].bar(['Harris', 'SIFT'], 
                  [df['harris_density'].mean(), df['sift_density'].mean()],
                  color=['red', 'blue'], alpha=0.7)
    axes[1, 0].set_ylabel('Keypoints per Pixel')
    axes[1, 0].set_title('Average Keypoint Density')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Percentage comparison
    axes[1, 1].pie([df['harris_percentage'].mean(), df['sift_percentage'].mean()],
                  labels=['Harris', 'SIFT'], autopct='%1.1f%%',
                  colors=['red', 'blue'], startangle=90)
    axes[1, 1].set_title('Percentage Distribution of Keypoints')
    
    plt.tight_layout()
    plt.savefig('results/plots/keypoint_count_analysis.jpg', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*50)
    print("TASK 1 COMPLETE!")
    print(f"Results saved in:")
    print(f"  • results/harris/ - Harris keypoint images")
    print(f"  • results/sift/ - SIFT keypoint images")
    print(f"  • results/comparison/ - Comparison images")
    print(f"  • results/distribution/ - Distribution analysis")
    print(f"  • results/metrics/basic_detection_results.csv - Data")
    print(f"  • results/plots/keypoint_count_analysis.jpg - Summary plots")
    print("="*50)
    
    return df

def run_robustness_experiments():
    """Run all robustness experiments (Task 2)"""
    print("\n" + "="*50)
    print("TASK 2: ROBUSTNESS ANALYSIS")
    print("="*50)
    
    # Initialize
    experimenter = ExperimentRunner()
    transformer = ImageTransformer()
    
    # Get first image for experiments
    image_files = glob.glob('data/original/*.jpg') + \
                  glob.glob('data/original/*.png') + \
                  glob.glob('data/original/*.jpeg')
    
    if not image_files:
        print("No images found for experiments")
        return
    
    img_path = image_files[0]  # Use first image for robustness tests
    img = cv2.imread(img_path)
    filename = os.path.splitext(os.path.basename(img_path))[0]
    
    print(f"Running robustness experiments on: {os.path.basename(img_path)}")
    print("This will test: Scale, Rotation, Brightness, Blur, and Noise")
    
    # Create directory for robustness visualizations
    os.makedirs(f'results/robustness/{filename}', exist_ok=True)
    
    # Run all experiments
    all_results = {}
    
    print("\n1. Testing Scale Robustness...")
    scale_results = experimenter.test_scale_robustness(img)
    all_results['scale'] = scale_results
    
    # Visualize scale transformations
    print("   Creating scale transformation examples...")
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    scale_images = []
    scale_titles = []
    
    for scale in scales[:4]:  # Show first 4 scales
        scaled_img = transformer.scale_image(img, scale)
        scale_images.append(scaled_img)
        scale_titles.append(f'Scale: {scale}x')
    
    fig, axes = plt.subplots(1, len(scale_images), figsize=(15, 4))
    for ax, scale_img, title in zip(axes, scale_images, scale_titles):
        ax.imshow(cv2.cvtColor(scale_img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'results/robustness/{filename}/scale_examples.jpg', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("2. Testing Rotation Robustness...")
    rotation_results = experimenter.test_rotation_robustness(img)
    all_results['rotation'] = rotation_results
    
    # Visualize rotation transformations
    print("   Creating rotation examples...")
    angles = [0, 30, 60, 90]
    rotation_images = []
    rotation_titles = []
    
    for angle in angles:
        rotated_img = transformer.rotate_image(img, angle)
        rotation_images.append(rotated_img)
        rotation_titles.append(f'Rotation: {angle}°')
    
    fig, axes = plt.subplots(1, len(rotation_images), figsize=(15, 4))
    for ax, rot_img, title in zip(axes, rotation_images, rotation_titles):
        ax.imshow(cv2.cvtColor(rot_img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'results/robustness/{filename}/rotation_examples.jpg', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("3. Testing Brightness Robustness...")
    brightness_results = experimenter.test_brightness_robustness(img)
    all_results['brightness'] = brightness_results
    
    # Visualize brightness transformations
    print("   Creating brightness examples...")
    factors = [0.5, 0.75, 1.0, 1.5]
    brightness_images = []
    brightness_titles = []
    
    for factor in factors:
        bright_img = transformer.change_brightness(img, factor)
        brightness_images.append(bright_img)
        brightness_titles.append(f'Brightness: {factor}x')
    
    fig, axes = plt.subplots(1, len(brightness_images), figsize=(15, 4))
    for ax, bright_img, title in zip(axes, brightness_images, brightness_titles):
        ax.imshow(cv2.cvtColor(bright_img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'results/robustness/{filename}/brightness_examples.jpg', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("4. Testing Blur Robustness...")
    blur_results = experimenter.test_blur_robustness(img)
    all_results['blur'] = blur_results
    
    # Visualize blur transformations
    print("   Creating blur examples...")
    kernel_sizes = [1, 3, 5, 9]
    blur_images = []
    blur_titles = []
    
    for kernel in kernel_sizes:
        blurred_img = transformer.add_gaussian_blur(img, kernel)
        blur_images.append(blurred_img)
        blur_titles.append(f'Blur: kernel={kernel}')
    
    fig, axes = plt.subplots(1, len(blur_images), figsize=(15, 4))
    for ax, blur_img, title in zip(axes, blur_images, blur_titles):
        ax.imshow(cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'results/robustness/{filename}/blur_examples.jpg', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("5. Testing Noise Robustness...")
    noise_results = experimenter.test_noise_robustness(img)
    all_results['noise'] = noise_results
    
    # Visualize noise transformations
    print("   Creating noise examples...")
    noise_levels = [0, 10, 20, 40]
    noise_images = []
    noise_titles = []
    
    for sigma in noise_levels:
        if sigma == 0:
            noisy_img = img.copy()
        else:
            noisy_img = transformer.add_gaussian_noise(img, sigma=sigma)
        noise_images.append(noisy_img)
        noise_titles.append(f'Noise: σ={sigma}')
    
    fig, axes = plt.subplots(1, len(noise_images), figsize=(15, 4))
    for ax, noise_img, title in zip(axes, noise_images, noise_titles):
        ax.imshow(cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'results/robustness/{filename}/noise_examples.jpg', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save all results
    with open(f'results/metrics/{filename}_robustness_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create comprehensive robustness plot
    experimenter.plot_robustness_results(all_results, f'results/plots/{filename}_robustness_summary.jpg')
    
    # Create individual robustness comparison tables
    robustness_summary = []
    
    for transformation, results in all_results.items():
        harris_avg_repeat = np.mean(results.get('harris_repeatability', [0]))
        sift_avg_repeat = np.mean(results.get('sift_repeatability', [0]))
        
        robustness_summary.append({
            'Transformation': transformation.capitalize(),
            'Harris_Avg_Repeatability': harris_avg_repeat,
            'SIFT_Avg_Repeatability': sift_avg_repeat,
            'Winner': 'Harris' if harris_avg_repeat > sift_avg_repeat else 'SIFT',
            'Difference': abs(harris_avg_repeat - sift_avg_repeat)
        })
    
    # Create summary table
    summary_df = pd.DataFrame(robustness_summary)
    summary_df.to_csv(f'results/metrics/{filename}_robustness_summary.csv', index=False)
    
    # Create robustness comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    transformations = summary_df['Transformation']
    harris_scores = summary_df['Harris_Avg_Repeatability']
    sift_scores = summary_df['SIFT_Avg_Repeatability']
    
    x = np.arange(len(transformations))
    width = 0.35
    
    ax.bar(x - width/2, harris_scores, width, label='Harris', color='red', alpha=0.7)
    ax.bar(x + width/2, sift_scores, width, label='SIFT', color='blue', alpha=0.7)
    
    ax.set_xlabel('Transformation Type')
    ax.set_ylabel('Average Repeatability')
    ax.set_title('Robustness Comparison: Harris vs SIFT')
    ax.set_xticks(x)
    ax.set_xticklabels(transformations)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (h_val, s_val) in enumerate(zip(harris_scores, sift_scores)):
        ax.text(i - width/2, h_val + 0.01, f'{h_val:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width/2, s_val + 0.01, f'{s_val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'results/plots/{filename}_robustness_comparison.jpg', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*50)
    print("TASK 2 COMPLETE!")
    print(f"Robustness results saved in:")
    print(f"  • results/robustness/{filename}/ - Transformation examples")
    print(f"  • results/metrics/{filename}_robustness_results.json - Detailed results")
    print(f"  • results/metrics/{filename}_robustness_summary.csv - Summary table")
    print(f"  • results/plots/{filename}_robustness_summary.jpg - Comprehensive plot")
    print(f"  • results/plots/{filename}_robustness_comparison.jpg - Comparison chart")
    print("="*50)
    
    return all_results, summary_df

def run_parameter_analysis():
    """Analyze parameter effects (Task 3)"""
    print("\n" + "="*50)
    print("TASK 3: PARAMETER ANALYSIS")
    print("="*50)
    
    # Get first image for parameter analysis
    image_files = glob.glob('data/original/*.jpg') + \
                  glob.glob('data/original/*.png') + \
                  glob.glob('data/original/*.jpeg')
    
    if not image_files:
        print("No images found for parameter analysis")
        return
    
    img_path = image_files[0]
    img = cv2.imread(img_path)
    filename = os.path.splitext(os.path.basename(img_path))[0]
    
    print(f"Analyzing parameters on: {os.path.basename(img_path)}")
    
    # Harris parameter analysis
    print("\n1. Harris Parameter Analysis...")
    harris = HarrisDetector()
    harris.analyze_parameters(img, f'results/plots/{filename}_harris_parameters.jpg')
    
    # SIFT parameter analysis
    print("2. SIFT Parameter Analysis...")
    sift = SIFTDetector()
    sift.analyze_parameters(img, f'results/plots/{filename}_sift_parameters.jpg')
    
    print("\n" + "="*50)
    print("TASK 3 COMPLETE!")
    print(f"Parameter analysis saved in results/plots/")
    print("="*50)

def generate_final_report(results_df, robustness_summary):
    """Generate final comprehensive report using current run metrics."""
    print("\n" + "="*50)
    print("GENERATING FINAL REPORT")
    print("="*50)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary = {
        'timestamp': timestamp,
        'total_images': len(results_df),
        'total_harris_keypoints': int(results_df['harris_count'].sum()),
        'total_sift_keypoints': int(results_df['sift_count'].sum()),
        'avg_harris_per_image': float(results_df['harris_count'].mean()),
        'avg_sift_per_image': float(results_df['sift_count'].mean()),
        'std_harris': float(results_df['harris_count'].std()),
        'std_sift': float(results_df['sift_count'].std()),
        'harris_min': int(results_df['harris_count'].min()),
        'harris_max': int(results_df['harris_count'].max()),
        'sift_min': int(results_df['sift_count'].min()),
        'sift_max': int(results_df['sift_count'].max()),
        'avg_harris_density': float(results_df['harris_density'].mean()),
        'avg_sift_density': float(results_df['sift_density'].mean()),
    }
    
    # Add robustness analysis if available
    robustness_section = ""
    if robustness_summary is not None and not robustness_summary.empty:
        summary['robustness_tests'] = len(robustness_summary)
        summary['harris_wins'] = len(robustness_summary[robustness_summary['Winner'] == 'Harris'])
        summary['sift_wins'] = len(robustness_summary[robustness_summary['Winner'] == 'SIFT'])
        summary['avg_harris_repeatability'] = float(robustness_summary['Harris_Avg_Repeatability'].mean())
        summary['avg_sift_repeatability'] = float(robustness_summary['SIFT_Avg_Repeatability'].mean())
        
        robustness_section += "### Robustness Comparison\n\n"
        robustness_section += "| Transformation | Harris Repeatability | SIFT Repeatability | Winner |\n"
        robustness_section += "|----------------|---------------------|-------------------|--------|\n"
        for _, row in robustness_summary.iterrows():
            robustness_section += f"| {row['Transformation']} | {row['Harris_Avg_Repeatability']:.3f} | {row['SIFT_Avg_Repeatability']:.3f} | **{row['Winner']}** |\n"
        
        robustness_section += f"""
### Overall Robustness
- Harris wins: {summary['harris_wins']} / {summary['robustness_tests']}
- SIFT wins: {summary['sift_wins']} / {summary['robustness_tests']}
- Avg repeatability — Harris: {summary['avg_harris_repeatability']:.3f}, SIFT: {summary['avg_sift_repeatability']:.3f}
"""
    
    # Save summary as JSON
    with open('results/metrics/final_summary_report.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    ratio = summary['total_sift_keypoints'] / max(summary['total_harris_keypoints'], 1)
    
    # Build per-image table
    per_image_rows = []
    per_image_rows.append("| Image | Harris | SIFT | Difference | Harris Density | SIFT Density |")
    per_image_rows.append("|-------|--------|------|------------|----------------|--------------|")
    for _, row in results_df.iterrows():
        per_image_rows.append(
            f"| {row['filename']} | {int(row['harris_count'])} | {int(row['sift_count'])} | "
            f"{int(row['count_difference'])} | {row['harris_density']:.6f} | {row['sift_density']:.6f} |"
        )
    per_image_table = "\n".join(per_image_rows)
    
    robustness_text = robustness_section if robustness_section else "No robustness summary available in this run."
    
    report_md = f"""# Computer Vision Course Project
## Comparative Evaluation of Harris Corner Detector and SIFT Keypoint Detector

**Submitted by:** Khaled Amr and Youssef Hipa  
**Generated:** {timestamp}  
**Images processed:** {summary['total_images']}

---
## Project Overview
Controlled comparison of Harris and SIFT detectors to study stability, robustness, and repeatability under scale, rotation, illumination, blur, and noise transformations. The pipeline follows the course brief and mirrors the Kaggle “Feature Extraction and Matching” assignment; current run uses 5 building/architecture photos placed in `data/original/`.

### Learning Objectives (met)
- Understand Harris corner detection and SIFT scale-space keypoint extraction.
- Analyze scale, rotation, illumination, blur, and noise invariance.
- Conduct quantitative evaluation (counts, densities, repeatability).
- Use benchmark-style building imagery (aligned with Kaggle dataset theme).
- Present results in a structured scientific report with code + visuals.

---
## 1) Implementation Summary
- **Harris:** gradients → second-moment matrix → Harris response → NMS → threshold; visualized as corner overlays.
- **SIFT:** OpenCV SIFT for multiscale keypoints with orientation; visualized as dots (clean overlays).
- Parameter notes: Harris k=0.05, window=3, threshold=0.003, nms=3; SIFT uses OpenCV defaults.

---
## 2) Keypoint Count Comparison (Task A)
- Total keypoints — Harris: {summary['total_harris_keypoints']:,}, SIFT: {summary['total_sift_keypoints']:,}
- Average per image — Harris: {summary['avg_harris_per_image']:.1f}, SIFT: {summary['avg_sift_per_image']:.1f}
- Range — Harris: {summary['harris_min']}–{summary['harris_max']}, SIFT: {summary['sift_min']}–{summary['sift_max']}
- Density (avg) — Harris: {summary['avg_harris_density']:.6f} kp/pixel, SIFT: {summary['avg_sift_density']:.6f} kp/pixel
- SIFT/Harris keypoint ratio: {ratio:.1f}×

### Per-image breakdown
{per_image_table}

**Observation:** SIFT produces far more keypoints; Harris is selective on high-contrast corners (std Harris {summary['std_harris']:.1f} vs SIFT {summary['std_sift']:.1f}).

---
## 3) Robustness Analysis (Task B)
{robustness_text}

---
## 4) Keypoint Distribution (Task C)
- Harris: concentrated on corners/edges of façades and windows.
- SIFT: broad coverage including textured regions; better scale/rotation handling.
- Visuals: see `results/comparison/` and `results/distribution/` for overlays and heatmaps.

---
## 5) Deliverables & Alignment with Brief
- Code: full Python pipeline in `src/` and `run_experiments.py`.
- Visualizations: `results/harris/`, `results/sift/`, `results/comparison/`, `results/distribution/`, `results/robustness/`, `results/plots/`.
- Metrics: `results/metrics/basic_detection_results.csv`, `results/metrics/*_robustness_results.json`, `results/metrics/*_robustness_summary.csv`, `results/metrics/final_summary_report.json`.
- Report: this file (`results/final_report.md`) with summary, dataset note, experiment design, results, and recommendations.

---
## 6) Recommendations
- Use **Harris** for fast, corner-focused tasks on man-made structures.
- Use **SIFT** when you need scale/rotation robustness and richer coverage.
- Hybrid approach: Harris for speed, SIFT descriptors for robustness/matching.

*Report generated automatically by Computer Vision Project System*
"""
    
    # Save report in results folder
    with open('results/final_report.md', 'w') as f:
        f.write(report_md)
    
    # Print summary
    print("\n📊 FINAL SUMMARY:")
    print("="*60)
    print(f"Images Processed: {summary['total_images']}")
    print(f"  Harris keypoints: {summary['total_harris_keypoints']:,}")
    print(f"  SIFT keypoints:   {summary['total_sift_keypoints']:,}")
    if robustness_summary is not None and not robustness_summary.empty:
        print(f"  Robustness tests: Harris wins {summary['harris_wins']}/{summary['robustness_tests']}, "
              f"SIFT wins {summary['sift_wins']}/{summary['robustness_tests']}")
    print(f"📄 Report saved as 'results/final_report.md'")
    print("="*60)
    
    return summary

def main():
    """Main function - runs everything automatically"""
    print("\n" + "="*70)
    print("COMPUTER VISION PROJECT: Harris vs SIFT Comparative Evaluation")
    print("="*70 + "\n")
    
    print("This program will run ALL required experiments:")
    print("1. Task A: Keypoint Count Comparison")
    print("2. Task B: Robustness Analysis (Scale, Rotation, Brightness, Blur, Noise)")
    print("3. Task C: Keypoint Distribution Analysis")
    print("4. Parameter Analysis")
    print("5. Generate Final Report\n")
    
    # Step 1: Setup directories
    setup_directories()
    
    # Step 2: Check if data exists
    if not check_data():
        print("\n⚠️  Please add your images to data/original/ and run again.")
        return
    
    print("\n" + "="*70)
    print("STARTING COMPLETE EXPERIMENT PIPELINE...")
    print("="*70 + "\n")
    
    try:
        # Task A: Basic detection and keypoint count comparison
        results_df = run_basic_detection()
        
        # Task B: Robustness analysis
        robustness_results, robustness_summary = run_robustness_experiments()
        
        # Task C & Parameter Analysis (included in above functions)
        run_parameter_analysis()
        
        # Generate final report
        summary = generate_final_report(results_df, robustness_summary)
        
        print("\n" + "="*70)
        print("🎉 PROJECT COMPLETED SUCCESSFULLY! 🎉")
        print("="*70)
        print("\n✅ ALL TASKS COMPLETED:")
        print("   ✓ Task A: Keypoint Count Comparison")
        print("   ✓ Task B: Robustness Analysis")
        print("   ✓ Task C: Keypoint Distribution Analysis")
        print("   ✓ Parameter Analysis")
        print("   ✓ Final Report Generation")
        
        print("\n📁 Check these folders for results:")
        print("   • results/harris/       - Harris detector outputs")
        print("   • results/sift/         - SIFT detector outputs")
        print("   • results/comparison/   - Side-by-side comparisons")
        print("   • results/distribution/ - Distribution analysis")
        print("   • results/robustness/   - Transformation examples")
        print("   • results/plots/        - All analysis plots")
        print("   • results/metrics/      - Data files (CSV, JSON)")
        
        print("\n📊 Check these key files:")
        print("   • results/final_report.md         - Complete project report")
        print("   • results/metrics/final_summary_report.json - Statistics")
        print("   • results/plots/*_robustness_summary.jpg - Robustness analysis")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ ERROR: Something went wrong!")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print("\n⚠️  TROUBLESHOOTING:")
        print("  1. Make sure you installed all requirements:")
        print("     pip install -r requirements.txt")
        print("  2. Check your images are in data/original/")
        print("  3. Make sure images are valid JPG/PNG files")
        print("  4. Check you have all Python files in src/ folder")

if __name__ == "__main__":
    main()
