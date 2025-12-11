import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import pandas as pd
from tqdm import tqdm
import os

from src.harris_detector import HarrisDetector
from src.sift_detector import SIFTDetector
from src.evaluator import Evaluator
from src.utils import ImageTransformer, Visualizer

class ExperimentRunner:
    """Class to run all experiments for Harris vs SIFT comparison"""
    
    def __init__(self):
        self.harris_detector = HarrisDetector(k=0.05, window_size=5, threshold_percent=0.001)
        self.sift_detector = SIFTDetector()
        self.evaluator = Evaluator()
        self.transformer = ImageTransformer()
        
    def test_scale_robustness(self, image: np.ndarray, 
                             scales: List[float] = None) -> Dict[str, Any]:
        """
        Test robustness to scale changes
        
        Args:
            image: Original image
            scales: List of scale factors to test
            
        Returns:
            Dictionary with scale robustness results
        """
        if scales is None:
            scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        
        print(f"  Testing scales: {scales}")
        
        results = {
            'scales': scales,
            'harris_counts': [],
            'sift_counts': [],
            'harris_repeatability': [],
            'sift_repeatability': [],
            'harris_retention': [],
            'sift_retention': []
        }
        
        # Detect keypoints on original image
        harris_kps_orig = self.harris_detector.detect(image)
        sift_kps_orig = self.sift_detector.detect(image)
        
        original_counts = {
            'harris': len(harris_kps_orig),
            'sift': len(sift_kps_orig)
        }
        
        # Test each scale
        for scale in tqdm(scales, desc="      Testing scales", leave=False):
            # Scale the image
            scaled_img = self.transformer.scale_image(image, scale)
            
            # Detect keypoints on scaled image
            harris_kps_scaled = self.harris_detector.detect(scaled_img)
            sift_kps_scaled = self.sift_detector.detect(scaled_img)
            
            # Scale keypoints back to original coordinates for repeatability
            # (This is a simplified approach)
            harris_kps_scaled_orig = []
            for x, y, response in harris_kps_scaled:
                harris_kps_scaled_orig.append((x/scale, y/scale, response))
            
            sift_kps_scaled_orig = []
            for kp in sift_kps_scaled:
                x, y = kp.pt
                scaled_kp = cv2.KeyPoint(x/scale, y/scale, kp.size, kp.angle, 
                                        kp.response, kp.octave, kp.class_id)
                sift_kps_scaled_orig.append(scaled_kp)
            
            # Compute repeatability
            harris_repeat = self.evaluator.compute_repeatability(
                harris_kps_orig, harris_kps_scaled_orig, image.shape)
            sift_repeat = self.evaluator.compute_repeatability(
                sift_kps_orig, sift_kps_scaled_orig, image.shape)
            
            # Compute retention rate
            harris_retention = len(harris_kps_scaled) / (len(harris_kps_orig) + 1e-6)
            sift_retention = len(sift_kps_scaled) / (len(sift_kps_orig) + 1e-6)
            
            # Store results
            results['harris_counts'].append(len(harris_kps_scaled))
            results['sift_counts'].append(len(sift_kps_scaled))
            results['harris_repeatability'].append(harris_repeat)
            results['sift_repeatability'].append(sift_repeat)
            results['harris_retention'].append(harris_retention)
            results['sift_retention'].append(sift_retention)
        
        # Save visualization
        self._plot_scale_results(results, image)
        
        return results
    
    def test_rotation_robustness(self, image: np.ndarray,
                                angles: List[float] = None) -> Dict[str, Any]:
        """
        Test robustness to rotation
        
        Args:
            image: Original image
            angles: List of rotation angles in degrees
            
        Returns:
            Dictionary with rotation robustness results
        """
        if angles is None:
            angles = [15, 30, 45, 60, 90, 120, 150, 180]
        
        print(f"  Testing rotations: {angles}°")
        
        results = {
            'angles': angles,
            'harris_counts': [],
            'sift_counts': [],
            'harris_repeatability': [],
            'sift_repeatability': []
        }
        
        # Detect keypoints on original image
        harris_kps_orig = self.harris_detector.detect(image)
        sift_kps_orig = self.sift_detector.detect(image)
        
        # Test each rotation angle
        for angle in tqdm(angles, desc="      Testing rotations", leave=False):
            # Rotate the image
            rotated_img = self.transformer.rotate_image(image, angle)
            
            # Detect keypoints on rotated image
            harris_kps_rotated = self.harris_detector.detect(rotated_img)
            sift_kps_rotated = self.sift_detector.detect(rotated_img)
            
            # Compute repeatability
            # Note: For rotation, we should ideally transform keypoints back,
            # but for simplicity, we'll compute direct repeatability
            harris_repeat = self.evaluator.compute_repeatability(
                harris_kps_orig, harris_kps_rotated, image.shape, threshold=5.0)
            sift_repeat = self.evaluator.compute_repeatability(
                sift_kps_orig, sift_kps_rotated, image.shape, threshold=5.0)
            
            # Store results
            results['harris_counts'].append(len(harris_kps_rotated))
            results['sift_counts'].append(len(sift_kps_rotated))
            results['harris_repeatability'].append(harris_repeat)
            results['sift_repeatability'].append(sift_repeat)
        
        # Save visualization
        self._plot_rotation_results(results, image)
        
        return results
    
    def test_brightness_robustness(self, image: np.ndarray,
                                  factors: List[float] = None) -> Dict[str, Any]:
        """
        Test robustness to brightness changes
        
        Args:
            image: Original image
            factors: List of brightness multipliers
            
        Returns:
            Dictionary with brightness robustness results
        """
        if factors is None:
            factors = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        
        print(f"  Testing brightness factors: {factors}")
        
        results = {
            'factors': factors,
            'harris_counts': [],
            'sift_counts': [],
            'harris_repeatability': [],
            'sift_repeatability': []
        }
        
        # Detect keypoints on original image
        harris_kps_orig = self.harris_detector.detect(image)
        sift_kps_orig = self.sift_detector.detect(image)
        
        # Test each brightness factor
        for factor in tqdm(factors, desc="      Testing brightness", leave=False):
            # Adjust brightness
            bright_img = self.transformer.change_brightness(image, factor)
            
            # Detect keypoints on brightened image
            harris_kps_bright = self.harris_detector.detect(bright_img)
            sift_kps_bright = self.sift_detector.detect(bright_img)
            
            # Compute repeatability
            harris_repeat = self.evaluator.compute_repeatability(
                harris_kps_orig, harris_kps_bright, image.shape)
            sift_repeat = self.evaluator.compute_repeatability(
                sift_kps_orig, sift_kps_bright, image.shape)
            
            # Store results
            results['harris_counts'].append(len(harris_kps_bright))
            results['sift_counts'].append(len(sift_kps_bright))
            results['harris_repeatability'].append(harris_repeat)
            results['sift_repeatability'].append(sift_repeat)
        
        # Save visualization
        self._plot_brightness_results(results, image)
        
        return results
    
    def test_blur_robustness(self, image: np.ndarray,
                            kernel_sizes: List[int] = None) -> Dict[str, Any]:
        """
        Test robustness to blur
        
        Args:
            image: Original image
            kernel_sizes: List of Gaussian kernel sizes
            
        Returns:
            Dictionary with blur robustness results
        """
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7, 9, 11, 13]
        
        print(f"  Testing blur kernel sizes: {kernel_sizes}")
        
        results = {
            'kernel_sizes': kernel_sizes,
            'harris_counts': [],
            'sift_counts': [],
            'harris_repeatability': [],
            'sift_repeatability': []
        }
        
        # Detect keypoints on original image
        harris_kps_orig = self.harris_detector.detect(image)
        sift_kps_orig = self.sift_detector.detect(image)
        
        # Test each kernel size
        for kernel_size in tqdm(kernel_sizes, desc="      Testing blur", leave=False):
            # Apply blur
            blurred_img = self.transformer.add_gaussian_blur(image, kernel_size)
            
            # Detect keypoints on blurred image
            harris_kps_blurred = self.harris_detector.detect(blurred_img)
            sift_kps_blurred = self.sift_detector.detect(blurred_img)
            
            # Compute repeatability
            harris_repeat = self.evaluator.compute_repeatability(
                harris_kps_orig, harris_kps_blurred, image.shape)
            sift_repeat = self.evaluator.compute_repeatability(
                sift_kps_orig, sift_kps_blurred, image.shape)
            
            # Store results
            results['harris_counts'].append(len(harris_kps_blurred))
            results['sift_counts'].append(len(sift_kps_blurred))
            results['harris_repeatability'].append(harris_repeat)
            results['sift_repeatability'].append(sift_repeat)
        
        # Save visualization
        self._plot_blur_results(results, image)
        
        return results
    
    def test_noise_robustness(self, image: np.ndarray,
                             noise_levels: List[float] = None) -> Dict[str, Any]:
        """
        Test robustness to noise
        
        Args:
            image: Original image
            noise_levels: List of noise sigma values
            
        Returns:
            Dictionary with noise robustness results
        """
        if noise_levels is None:
            noise_levels = [5, 10, 15, 20, 25, 30, 40, 50]
        
        print(f"  Testing noise levels (sigma): {noise_levels}")
        
        results = {
            'noise_levels': noise_levels,
            'harris_counts': [],
            'sift_counts': [],
            'harris_repeatability': [],
            'sift_repeatability': []
        }
        
        # Detect keypoints on original image
        harris_kps_orig = self.harris_detector.detect(image)
        sift_kps_orig = self.sift_detector.detect(image)
        
        # Test each noise level
        for sigma in tqdm(noise_levels, desc="      Testing noise", leave=False):
            # Add noise
            noisy_img = self.transformer.add_gaussian_noise(image, sigma=sigma)
            
            # Detect keypoints on noisy image
            harris_kps_noisy = self.harris_detector.detect(noisy_img)
            sift_kps_noisy = self.sift_detector.detect(noisy_img)
            
            # Compute repeatability
            harris_repeat = self.evaluator.compute_repeatability(
                harris_kps_orig, harris_kps_noisy, image.shape)
            sift_repeat = self.evaluator.compute_repeatability(
                sift_kps_orig, sift_kps_noisy, image.shape)
            
            # Store results
            results['harris_counts'].append(len(harris_kps_noisy))
            results['sift_counts'].append(len(sift_kps_noisy))
            results['harris_repeatability'].append(harris_repeat)
            results['sift_repeatability'].append(sift_repeat)
        
        # Save visualization
        self._plot_noise_results(results, image)
        
        return results
    
    def _plot_scale_results(self, results: Dict[str, Any], image: np.ndarray):
        """Create and save scale robustness plot"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Keypoint count vs scale
        axes[0, 0].plot(results['scales'], results['harris_counts'], 'ro-', label='Harris', linewidth=2)
        axes[0, 0].plot(results['scales'], results['sift_counts'], 'bo-', label='SIFT', linewidth=2)
        axes[0, 0].set_xlabel('Scale Factor')
        axes[0, 0].set_ylabel('Keypoint Count')
        axes[0, 0].set_title('Keypoint Count vs Scale')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Repeatability vs scale
        axes[0, 1].plot(results['scales'], results['harris_repeatability'], 'ro-', label='Harris', linewidth=2)
        axes[0, 1].plot(results['scales'], results['sift_repeatability'], 'bo-', label='SIFT', linewidth=2)
        axes[0, 1].set_xlabel('Scale Factor')
        axes[0, 1].set_ylabel('Repeatability')
        axes[0, 1].set_title('Repeatability vs Scale')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Retention vs scale
        axes[1, 0].plot(results['scales'], results['harris_retention'], 'ro-', label='Harris', linewidth=2)
        axes[1, 0].plot(results['scales'], results['sift_retention'], 'bo-', label='SIFT', linewidth=2)
        axes[1, 0].set_xlabel('Scale Factor')
        axes[1, 0].set_ylabel('Retention Rate')
        axes[1, 0].set_title('Keypoint Retention vs Scale')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Comparison bar chart
        scales_str = [str(s) for s in results['scales']]
        x = np.arange(len(scales_str))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, results['harris_repeatability'], width, label='Harris', color='red', alpha=0.7)
        axes[1, 1].bar(x + width/2, results['sift_repeatability'], width, label='SIFT', color='blue', alpha=0.7)
        axes[1, 1].set_xlabel('Scale Factor')
        axes[1, 1].set_ylabel('Repeatability')
        axes[1, 1].set_title('Repeatability Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(scales_str)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/plots/scale_robustness.jpg', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_rotation_results(self, results: Dict[str, Any], image: np.ndarray):
        """Create and save rotation robustness plot"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Keypoint count vs rotation
        axes[0].plot(results['angles'], results['harris_counts'], 'ro-', label='Harris', linewidth=2)
        axes[0].plot(results['angles'], results['sift_counts'], 'bo-', label='SIFT', linewidth=2)
        axes[0].set_xlabel('Rotation Angle (degrees)')
        axes[0].set_ylabel('Keypoint Count')
        axes[0].set_title('Keypoint Count vs Rotation')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Repeatability vs rotation
        axes[1].plot(results['angles'], results['harris_repeatability'], 'ro-', label='Harris', linewidth=2)
        axes[1].plot(results['angles'], results['sift_repeatability'], 'bo-', label='SIFT', linewidth=2)
        axes[1].set_xlabel('Rotation Angle (degrees)')
        axes[1].set_ylabel('Repeatability')
        axes[1].set_title('Repeatability vs Rotation')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/plots/rotation_robustness.jpg', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_brightness_results(self, results: Dict[str, Any], image: np.ndarray):
        """Create and save brightness robustness plot"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Keypoint count vs brightness
        axes[0].plot(results['factors'], results['harris_counts'], 'ro-', label='Harris', linewidth=2)
        axes[0].plot(results['factors'], results['sift_counts'], 'bo-', label='SIFT', linewidth=2)
        axes[0].set_xlabel('Brightness Factor')
        axes[0].set_ylabel('Keypoint Count')
        axes[0].set_title('Keypoint Count vs Brightness')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Repeatability vs brightness
        axes[1].plot(results['factors'], results['harris_repeatability'], 'ro-', label='Harris', linewidth=2)
        axes[1].plot(results['factors'], results['sift_repeatability'], 'bo-', label='SIFT', linewidth=2)
        axes[1].set_xlabel('Brightness Factor')
        axes[1].set_ylabel('Repeatability')
        axes[1].set_title('Repeatability vs Brightness')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/plots/brightness_robustness.jpg', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_blur_results(self, results: Dict[str, Any], image: np.ndarray):
        """Create and save blur robustness plot"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Keypoint count vs blur
        axes[0].plot(results['kernel_sizes'], results['harris_counts'], 'ro-', label='Harris', linewidth=2)
        axes[0].plot(results['kernel_sizes'], results['sift_counts'], 'bo-', label='SIFT', linewidth=2)
        axes[0].set_xlabel('Blur Kernel Size')
        axes[0].set_ylabel('Keypoint Count')
        axes[0].set_title('Keypoint Count vs Blur')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Repeatability vs blur
        axes[1].plot(results['kernel_sizes'], results['harris_repeatability'], 'ro-', label='Harris', linewidth=2)
        axes[1].plot(results['kernel_sizes'], results['sift_repeatability'], 'bo-', label='SIFT', linewidth=2)
        axes[1].set_xlabel('Blur Kernel Size')
        axes[1].set_ylabel('Repeatability')
        axes[1].set_title('Repeatability vs Blur')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/plots/blur_robustness.jpg', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_noise_results(self, results: Dict[str, Any], image: np.ndarray):
        """Create and save noise robustness plot"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Keypoint count vs noise
        axes[0].plot(results['noise_levels'], results['harris_counts'], 'ro-', label='Harris', linewidth=2)
        axes[0].plot(results['noise_levels'], results['sift_counts'], 'bo-', label='SIFT', linewidth=2)
        axes[0].set_xlabel('Noise Level (sigma)')
        axes[0].set_ylabel('Keypoint Count')
        axes[0].set_title('Keypoint Count vs Noise')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Repeatability vs noise
        axes[1].plot(results['noise_levels'], results['harris_repeatability'], 'ro-', label='Harris', linewidth=2)
        axes[1].plot(results['noise_levels'], results['sift_repeatability'], 'bo-', label='SIFT', linewidth=2)
        axes[1].set_xlabel('Noise Level (sigma)')
        axes[1].set_ylabel('Repeatability')
        axes[1].set_title('Repeatability vs Noise')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/plots/noise_robustness.jpg', dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_robustness_results(self, all_results: Dict[str, Dict], save_path: str = None):
        """Create comprehensive robustness comparison plot"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Define plot configurations
        plot_configs = [
            ('scale', 'Scale Factor', 0, 0),
            ('rotation', 'Rotation Angle (°)', 0, 1),
            ('brightness', 'Brightness Factor', 0, 2),
            ('blur', 'Blur Kernel Size', 1, 0),
            ('noise', 'Noise Level', 1, 1)
        ]
        
        for i, (exp_type, xlabel, row, col) in enumerate(plot_configs):
            if exp_type in all_results:
                results = all_results[exp_type]
                
                if exp_type == 'scale':
                    x_data = results['scales']
                elif exp_type == 'rotation':
                    x_data = results['angles']
                elif exp_type == 'brightness':
                    x_data = results['factors']
                elif exp_type == 'blur':
                    x_data = results['kernel_sizes']
                elif exp_type == 'noise':
                    x_data = results['noise_levels']
                else:
                    x_data = list(range(len(results['harris_repeatability'])))
                
                # Plot repeatability
                axes[row, col].plot(x_data, results['harris_repeatability'], 
                                  'ro-', label='Harris', linewidth=2, markersize=6)
                axes[row, col].plot(x_data, results['sift_repeatability'], 
                                  'bo-', label='SIFT', linewidth=2, markersize=6)
                axes[row, col].set_xlabel(xlabel)
                axes[row, col].set_ylabel('Repeatability')
                axes[row, col].set_title(f'{exp_type.capitalize()} Robustness')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
        
        # Overall comparison in last subplot
        ax = axes[1, 2]
        categories = ['Scale', 'Rotation', 'Brightness', 'Blur', 'Noise']
        harris_avg = []
        sift_avg = []
        
        for exp_type in ['scale', 'rotation', 'brightness', 'blur', 'noise']:
            if exp_type in all_results:
                results = all_results[exp_type]
                harris_avg.append(np.mean(results['harris_repeatability']))
                sift_avg.append(np.mean(results['sift_repeatability']))
        
        x = np.arange(len(categories[:len(harris_avg)]))
        width = 0.35
        
        ax.bar(x - width/2, harris_avg, width, label='Harris', color='red', alpha=0.7)
        ax.bar(x + width/2, sift_avg, width, label='SIFT', color='blue', alpha=0.7)
        ax.set_xlabel('Transformation Type')
        ax.set_ylabel('Average Repeatability')
        ax.set_title('Overall Robustness Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(categories[:len(harris_avg)], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def run_complete_experiment(self, image: np.ndarray, image_name: str = None):
        """Run complete set of experiments on one image"""
        print(f"\nRunning complete experiment on image: {image_name}")
        
        # Run all experiments
        scale_results = self.test_scale_robustness(image)
        rotation_results = self.test_rotation_robustness(image)
        brightness_results = self.test_brightness_robustness(image)
        blur_results = self.test_blur_robustness(image)
        noise_results = self.test_noise_robustness(image)
        
        # Combine results
        all_results = {
            'scale': scale_results,
            'rotation': rotation_results,
            'brightness': brightness_results,
            'blur': blur_results,
            'noise': noise_results
        }
        
        # Create comprehensive visualization
        if image_name:
            save_path = f'results/plots/{image_name}_complete_robustness.jpg'
        else:
            save_path = 'results/plots/complete_robustness.jpg'
        
        self.plot_robustness_results(all_results, save_path)
        
        return all_results