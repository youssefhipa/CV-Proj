import cv2
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class HarrisDetector:
    """Enhanced Harris Corner Detector with various improvements"""
    
    def __init__(self, k: float = 0.05, window_size: int = 5, 
                 threshold_percent: float = 0.001, nms_size: int = 3):
        """
        Initialize Harris Detector
        
        Args:
            k: Harris detector constant (0.04-0.06)
            window_size: Size of Gaussian window
            threshold_percent: Percentage of strongest corners to keep (0-1)
            nms_size: Size of non-maximum suppression window
        """
        self.k = k
        self.window_size = window_size
        self.threshold_percent = threshold_percent
        self.nms_size = nms_size
        
    def compute_gradients(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute image gradients using Sobel operator"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Compute gradients
        Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        return Ix, Iy
    
    def compute_harris_response(self, Ix: np.ndarray, Iy: np.ndarray) -> np.ndarray:
        """Compute Harris response matrix"""
        # Compute products of gradients
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy
        
        # Apply Gaussian smoothing
        kernel = (self.window_size, self.window_size)
        Ixx = cv2.GaussianBlur(Ixx, kernel, 1.5)
        Iyy = cv2.GaussianBlur(Iyy, kernel, 1.5)
        Ixy = cv2.GaussianBlur(Ixy, kernel, 1.5)
        
        # Compute Harris response
        detM = Ixx * Iyy - Ixy * Ixy
        traceM = Ixx + Iyy
        R = detM - self.k * (traceM ** 2)
        
        return R
    
    def detect(self, image: np.ndarray, return_response: bool = False) -> List[Tuple[int, int, float]]:
        """
        Detect Harris corners in an image
        
        Returns:
            List of keypoints as (x, y, response) tuples
        """
        # 1. Compute gradients
        Ix, Iy = self.compute_gradients(image)
        
        # 2. Compute Harris response
        R = self.compute_harris_response(Ix, Iy)
        
        # 3. Non-maximum suppression
        R_nms = cv2.dilate(R, None)
        local_maxima = (R == R_nms)
        
        # 4. Adaptive thresholding
        R_local_max = R[local_maxima]
        if len(R_local_max) > 0:
            threshold_value = np.percentile(R_local_max, 100 * (1 - self.threshold_percent))
        else:
            threshold_value = 0
        
        # Create mask for strong corners
        strong_corners = (R >= threshold_value) & local_maxima
        
        # 5. Extract keypoints
        keypoints = []
        rows, cols = R.shape
        
        for y in range(rows):
            for x in range(cols):
                if strong_corners[y, x]:
                    response = float(R[y, x])
                    # Normalize response for consistency
                    norm_response = (response - threshold_value) / (R.max() - threshold_value + 1e-6)
                    norm_response = np.clip(norm_response, 0.0, 1.0)
                    keypoints.append((x, y, norm_response))
        
        if return_response:
            return keypoints, R
        return keypoints
    
    def detect_with_scale(self, image: np.ndarray, scales: List[float] = None) -> List[Tuple[int, int, float, float]]:
        """
        Detect Harris corners at multiple scales
        
        Returns:
            List of keypoints as (x, y, response, scale) tuples
        """
        if scales is None:
            scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        
        all_keypoints = []
        
        for scale in scales:
            # Scale image
            h, w = image.shape[:2]
            scaled_size = (int(w * scale), int(h * scale))
            scaled_img = cv2.resize(image, scaled_size)
            
            # Detect keypoints at this scale
            kps = self.detect(scaled_img)
            
            # Scale keypoints back to original image coordinates
            for x, y, response in kps:
                orig_x = x / scale
                orig_y = y / scale
                all_keypoints.append((orig_x, orig_y, response, scale))
        
        return all_keypoints
    
    def draw_keypoints(self, image: np.ndarray, keypoints: List[Tuple], 
                      color: Tuple = (0, 0, 255), max_points: int = 500) -> np.ndarray:
        """
        Draw keypoints on image with response-based coloring
        
        Args:
            image: Input image
            keypoints: List of (x, y, response) tuples
            color: Base color for keypoints
            max_points: Maximum number of keypoints to display
        """
        img_copy = image.copy()
        
        # Sort keypoints by response and take top ones
        if len(keypoints) > max_points:
            keypoints = sorted(keypoints, key=lambda x: x[2], reverse=True)[:max_points]
        
        for x, y, response in keypoints:
            # Color coding based on response strength
            if response > 0.8:
                kp_color = (0, 255, 0)  # Strong - Green
                radius = 4
            elif response > 0.5:
                kp_color = (0, 200, 255)  # Medium - Orange
                radius = 3
            else:
                kp_color = color  # Weak - Red
                radius = 2
            
            cv2.circle(img_copy, (int(x), int(y)), radius, kp_color, 1)
            
        return img_copy
    
    def analyze_parameters(self, image: np.ndarray, save_path: str = None):
        """Analyze the effect of different parameters on detection"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Test different k values
        k_values = [0.04, 0.05, 0.06]
        for i, k in enumerate(k_values):
            self.k = k
            kps = self.detect(image)
            img_kps = self.draw_keypoints(image, kps)
            axes[i].imshow(cv2.cvtColor(img_kps, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f'k={k}, Keypoints={len(kps)}')
            axes[i].axis('off')
        
        # Test different window sizes
        window_sizes = [3, 5, 7]
        for i, ws in enumerate(window_sizes):
            self.window_size = ws
            kps = self.detect(image)
            img_kps = self.draw_keypoints(image, kps)
            axes[i+3].imshow(cv2.cvtColor(img_kps, cv2.COLOR_BGR2RGB))
            axes[i+3].set_title(f'Window={ws}, Keypoints={len(kps)}')
            axes[i+3].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        # Reset to default values
        self.k = 0.05
        self.window_size = 5