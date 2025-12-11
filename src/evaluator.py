import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import pandas as pd
from tqdm import tqdm

class Evaluator:
    """Class to evaluate and compare Harris and SIFT detectors"""
    
    def __init__(self):
        self.metrics_history = []
    
    def count_keypoints(self, harris_kps: List[Tuple], sift_kps: List[cv2.KeyPoint]) -> Dict[str, int]:
        """Count keypoints from both detectors"""
        return {
            'harris_count': len(harris_kps),
            'sift_count': len(sift_kps),
            'difference': abs(len(harris_kps) - len(sift_kps)),
            'harris_percentage': len(harris_kps) / (len(harris_kps) + len(sift_kps) + 1e-6) * 100,
            'sift_percentage': len(sift_kps) / (len(harris_kps) + len(sift_kps) + 1e-6) * 100
        }
    
    def compute_spatial_distribution(self, keypoints: List, image_shape: Tuple) -> np.ndarray:
        """Compute spatial distribution heatmap of keypoints"""
        heatmap = np.zeros(image_shape[:2], dtype=np.float32)
        
        for kp in keypoints:
            if isinstance(kp, tuple):  # Harris keypoint
                x, y, _ = kp
            else:  # SIFT keypoint
                x, y = kp.pt
            
            x, y = int(x), int(y)
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                # Add Gaussian blob at keypoint location
                radius = 5
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < image_shape[0] and 0 <= nx < image_shape[1]:
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist <= radius:
                                heatmap[ny, nx] += np.exp(-dist**2 / (2 * (radius/2)**2))
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def compute_repeatability(self, kps1: List, kps2: List, 
                             image_shape: Tuple, threshold: float = 3.0) -> float:
        """
        Compute repeatability rate between two sets of keypoints
        
        Args:
            kps1, kps2: Keypoints from two images (should be corresponding)
            image_shape: Shape of the image
            threshold: Distance threshold for considering keypoints as repeated
        """
        if not kps1 or not kps2:
            return 0.0
        
        # Convert keypoints to (x, y) coordinates
        points1 = []
        for kp in kps1:
            if isinstance(kp, tuple):  # Harris
                x, y, _ = kp
            else:  # SIFT
                x, y = kp.pt
            points1.append((x, y))
        
        points2 = []
        for kp in kps2:
            if isinstance(kp, tuple):  # Harris
                x, y, _ = kp
            else:  # SIFT
                x, y = kp.pt
            points2.append((x, y))
        
        # Use KDTree for efficient nearest neighbor search
        tree1 = KDTree(points1)
        tree2 = KDTree(points2)
        
        # Find correspondences
        matches1 = 0
        for point in points1:
            dist, idx = tree2.query(point)
            if dist <= threshold:
                matches1 += 1
        
        matches2 = 0
        for point in points2:
            dist, idx = tree1.query(point)
            if dist <= threshold:
                matches2 += 1
        
        # Compute repeatability as average match rate
        repeatability1 = matches1 / len(points1) if points1 else 0
        repeatability2 = matches2 / len(points2) if points2 else 0
        
        return (repeatability1 + repeatability2) / 2
    
    def evaluate_robustness(self, original_img: np.ndarray, 
                           transformed_img: np.ndarray,
                           harris_detector, sift_detector) -> Dict[str, Any]:
        """Evaluate robustness of detectors to transformations"""
        
        # Detect keypoints on original image
        harris_kps_orig = harris_detector.detect(original_img)
        sift_kps_orig = sift_detector.detect(original_img)
        
        # Detect keypoints on transformed image
        harris_kps_trans = harris_detector.detect(transformed_img)
        sift_kps_trans = sift_detector.detect(transformed_img)
        
        # Compute repeatability
        harris_repeatability = self.compute_repeatability(
            harris_kps_orig, harris_kps_trans, original_img.shape)
        sift_repeatability = self.compute_repeatability(
            sift_kps_orig, sift_kps_trans, original_img.shape)
        
        # Count keypoints
        orig_counts = self.count_keypoints(harris_kps_orig, sift_kps_orig)
        trans_counts = self.count_keypoints(harris_kps_trans, sift_kps_trans)
        
        # Compute keypoint retention rate
        harris_retention = len(harris_kps_trans) / (len(harris_kps_orig) + 1e-6)
        sift_retention = len(sift_kps_trans) / (len(sift_kps_orig) + 1e-6)
        
        results = {
            'harris_repeatability': harris_repeatability,
            'sift_repeatability': sift_repeatability,
            'harris_retention': harris_retention,
            'sift_retention': sift_retention,
            'orig_harris_count': orig_counts['harris_count'],
            'orig_sift_count': orig_counts['sift_count'],
            'trans_harris_count': trans_counts['harris_count'],
            'trans_sift_count': trans_counts['sift_count'],
            'repeatability_difference': abs(harris_repeatability - sift_repeatability),
            'retention_difference': abs(harris_retention - sift_retention)
        }
        
        return results
    
    def run_comprehensive_evaluation(self, images: List[Tuple[str, np.ndarray]], 
                                    harris_detector, sift_detector) -> pd.DataFrame:
        """Run comprehensive evaluation on all images"""
        all_results = []
        
        for img_name, img in tqdm(images, desc="Evaluating images"):
            # Detect keypoints
            harris_kps = harris_detector.detect(img)
            sift_kps = sift_detector.detect(img)
            
            # Count keypoints
            counts = self.count_keypoints(harris_kps, sift_kps)
            
            # Compute spatial distributions
            harris_dist = self.compute_spatial_distribution(harris_kps, img.shape)
            sift_dist = self.compute_spatial_distribution(sift_kps, img.shape)
            
            # Compute distribution similarity
            distribution_similarity = np.corrcoef(
                harris_dist.flatten(), sift_dist.flatten())[0, 1]
            
            # Store results
            result = {
                'image_name': img_name,
                'harris_count': counts['harris_count'],
                'sift_count': counts['sift_count'],
                'count_difference': counts['difference'],
                'harris_percentage': counts['harris_percentage'],
                'sift_percentage': counts['sift_percentage'],
                'distribution_similarity': distribution_similarity,
                'image_width': img.shape[1],
                'image_height': img.shape[0],
                'image_area': img.shape[0] * img.shape[1],
                'harris_density': counts['harris_count'] / (img.shape[0] * img.shape[1]),
                'sift_density': counts['sift_count'] / (img.shape[0] * img.shape[1])
            }
            
            all_results.append(result)
        
        return pd.DataFrame(all_results)
    
    def plot_comparison(self, df: pd.DataFrame, save_path: str = None):
        """Create comprehensive comparison plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Keypoint count comparison
        axes[0, 0].bar(['Harris', 'SIFT'], 
                      [df['harris_count'].mean(), df['sift_count'].mean()])
        axes[0, 0].set_title('Average Keypoint Count')
        axes[0, 0].set_ylabel('Count')
        
        # 2. Keypoint density comparison
        axes[0, 1].bar(['Harris', 'SIFT'],
                      [df['harris_density'].mean(), df['sift_density'].mean()])
        axes[0, 1].set_title('Keypoint Density (per pixel)')
        axes[0, 1].set_ylabel('Density')
        
        # 3. Count scatter plot
        axes[0, 2].scatter(df['harris_count'], df['sift_count'], alpha=0.6)
        axes[0, 2].plot([0, df['harris_count'].max()], 
                       [0, df['harris_count'].max()], 'r--', alpha=0.5)
        axes[0, 2].set_xlabel('Harris Count')
        axes[0, 2]