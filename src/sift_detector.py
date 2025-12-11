import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt

class SIFTDetector:
    """SIFT Keypoint Detector using OpenCV implementation"""
    
    def __init__(self, n_features: int = 0, n_octave_layers: int = 3, 
                 contrast_threshold: float = 0.04, edge_threshold: float = 10, 
                 sigma: float = 1.6):
        """
        Initialize SIFT Detector
        
        Args:
            n_features: Number of best features to retain (0 for all)
            n_octave_layers: Number of layers in each octave
            contrast_threshold: Contrast threshold to filter weak features
            edge_threshold: Edge threshold to filter edge-like features
            sigma: Sigma of Gaussian at octave 0
        """
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )
        self.params = {
            'n_features': n_features,
            'n_octave_layers': n_octave_layers,
            'contrast_threshold': contrast_threshold,
            'edge_threshold': edge_threshold,
            'sigma': sigma
        }
    
    def detect(self, image: np.ndarray) -> List[cv2.KeyPoint]:
        """
        Detect SIFT keypoints in an image
        
        Returns:
            List of OpenCV KeyPoint objects
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints = self.sift.detect(gray, None)
        return keypoints
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detect SIFT keypoints and compute descriptors
        
        Returns:
            Tuple of (keypoints, descriptors)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def draw_keypoints(self, image: np.ndarray, keypoints: List[cv2.KeyPoint], 
                      color: Tuple = (0, 255, 0), max_points: int = 2000, radius: int = 2) -> np.ndarray:
        """
        Draw SIFT keypoints as simple dots (no circles/scale lines)
        
        Args:
            image: Input image
            keypoints: List of SIFT keypoints
            color: Color for keypoints
            max_points: Cap number of points to draw for clarity
            radius: Dot radius in pixels
        """
        img_copy = image.copy()
        
        # Keep strongest points if too many
        if len(keypoints) > max_points:
            keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:max_points]
        
        for kp in keypoints:
            x, y = map(int, kp.pt)
            cv2.circle(img_copy, (x, y), radius, color, thickness=-1)
        
        return img_copy
    
    def get_keypoint_statistics(self, keypoints: List[cv2.KeyPoint]) -> Dict[str, Any]:
        """
        Compute statistics about SIFT keypoints
        
        Returns:
            Dictionary containing various statistics
        """
        if not keypoints:
            return {}
        
        # Extract properties
        responses = [kp.response for kp in keypoints]
        sizes = [kp.size for kp in keypoints]
        angles = [kp.angle for kp in keypoints]
        octaves = [kp.octave for kp in keypoints]
        
        # Get positions
        positions = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
        
        stats = {
            'count': len(keypoints),
            'avg_response': np.mean(responses),
            'std_response': np.std(responses),
            'min_response': np.min(responses),
            'max_response': np.max(responses),
            'avg_size': np.mean(sizes),
            'std_size': np.std(sizes),
            'avg_angle': np.mean(angles),
            'std_angle': np.std(angles),
            'octave_distribution': np.bincount(np.array(octaves) + 128),  # OpenCV stores octave as (octave * 256 + layer)
            'positions': positions
        }
        
        return stats
    
    def visualize_scale_space(self, image: np.ndarray, save_path: str = None):
        """
        Visualize keypoints at different scales
        
        Note: This is a simplified visualization since OpenCV doesn't expose
        the full scale-space pyramid directly
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create images at different scales
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        fig, axes = plt.subplots(1, len(scales), figsize=(15, 4))
        
        for i, scale in enumerate(scales):
            # Scale image
            h, w = gray.shape
            scaled_size = (int(w * scale), int(h * scale))
            scaled_img = cv2.resize(gray, scaled_size)
            
            # Detect keypoints at this scale
            kps = self.sift.detect(scaled_img, None)
            
            # Draw keypoints
            img_kps = cv2.cvtColor(scaled_img, cv2.COLOR_GRAY2BGR)
            for kp in kps:
                x, y = map(int, kp.pt)
                cv2.circle(img_kps, (x, y), 2, (0, 255, 0), thickness=-1)
            
            axes[i].imshow(cv2.cvtColor(img_kps, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f'Scale: {scale}\nKeypoints: {len(kps)}')
            axes[i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def analyze_parameters(self, image: np.ndarray, save_path: str = None):
        """Analyze the effect of different SIFT parameters"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Test different contrast thresholds
        contrast_thresholds = [0.01, 0.04, 0.08]
        for i, ct in enumerate(contrast_thresholds):
            detector = cv2.SIFT_create(contrastThreshold=ct)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kps = detector.detect(gray, None)
            img_kps = image.copy()
            for kp in kps:
                x, y = map(int, kp.pt)
                cv2.circle(img_kps, (x, y), 2, (0, 255, 0), thickness=-1)
            axes[i].imshow(cv2.cvtColor(img_kps, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f'ContrastThresh={ct}\nKeypoints={len(kps)}')
            axes[i].axis('off')
        
        # Test different edge thresholds
        edge_thresholds = [5, 10, 20]
        for i, et in enumerate(edge_thresholds):
            detector = cv2.SIFT_create(edgeThreshold=et)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kps = detector.detect(gray, None)
            img_kps = image.copy()
            for kp in kps:
                x, y = map(int, kp.pt)
                cv2.circle(img_kps, (x, y), 2, (0, 255, 0), thickness=-1)
            axes[i+3].imshow(cv2.cvtColor(img_kps, cv2.COLOR_BGR2RGB))
            axes[i+3].set_title(f'EdgeThresh={et}\nKeypoints={len(kps)}')
            axes[i+3].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
