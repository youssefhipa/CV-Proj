import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
import os
import glob
from tqdm import tqdm

class ImageTransformer:
    """Class to apply various transformations to images for robustness testing"""
    
    @staticmethod
    def scale_image(image: np.ndarray, scale_factor: float) -> np.ndarray:
        """Scale image by a factor"""
        h, w = image.shape[:2]
        new_size = (int(w * scale_factor), int(h * scale_factor))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    
    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by specified angle in degrees"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    
    @staticmethod
    def change_brightness(image: np.ndarray, brightness_factor: float) -> np.ndarray:
        """Change image brightness"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def add_gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
        """Add Gaussian blur to image"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 25) -> np.ndarray:
        """Add Gaussian noise to image"""
        row, col, ch = image.shape
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = np.clip(image + gauss, 0, 255).astype(np.uint8)
        return noisy
    
    @staticmethod
    def add_salt_pepper_noise(image: np.ndarray, prob: float = 0.05) -> np.ndarray:
        """Add salt and pepper noise to image"""
        noisy = np.copy(image)
        
        # Salt noise
        num_salt = np.ceil(prob * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 255
        
        # Pepper noise
        num_pepper = np.ceil(prob * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy[coords[0], coords[1], :] = 0
        
        return noisy

class Visualizer:
    """Class for visualizing keypoints and results"""
    
    @staticmethod
    def draw_keypoints(image: np.ndarray, keypoints: List[Tuple], 
                       color: Tuple = (0, 255, 0), radius: int = 3) -> np.ndarray:
        """Draw keypoints on image"""
        img_copy = image.copy()
        for kp in keypoints:
            if len(kp) == 2:  # (x, y)
                x, y = kp
                cv2.circle(img_copy, (int(x), int(y)), radius, color, 1)
            elif len(kp) == 3:  # (x, y, response)
                x, y, _ = kp
                cv2.circle(img_copy, (int(x), int(y)), radius, color, 1)
        return img_copy
    
    @staticmethod
    def draw_comparison(images: List[np.ndarray], titles: List[str], 
                       figsize: Tuple = (15, 10)) -> plt.Figure:
        """Create a comparison plot of multiple images"""
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=figsize)
        
        if n == 1:
            axes = [axes]
        
        for ax, img, title in zip(axes, images, titles):
            if len(img.shape) == 3:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_distribution(harris_kps: List[Tuple], sift_kps: List[Tuple], 
                         image_shape: Tuple, title: str = "Keypoint Distribution"):
        """Plot spatial distribution of keypoints"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Create empty images for distribution
        harris_dist = np.zeros(image_shape[:2])
        sift_dist = np.zeros(image_shape[:2])
        
        # Mark keypoint locations
        for kp in harris_kps:
            x, y = int(kp[0]), int(kp[1])
            if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
                harris_dist[y, x] = 1
        
        for kp in sift_kps:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
                sift_dist[y, x] = 1
        
        # Plot Harris distribution
        axes[0].imshow(harris_dist, cmap='hot')
        axes[0].set_title("Harris Keypoints Distribution")
        axes[0].axis('off')
        
        # Plot SIFT distribution
        axes[1].imshow(sift_dist, cmap='hot')
        axes[1].set_title("SIFT Keypoints Distribution")
        axes[1].axis('off')
        
        # Plot combined
        combined_dist = harris_dist + sift_dist * 2
        im = axes[2].imshow(combined_dist, cmap='jet')
        axes[2].set_title("Combined (Blue=Harris, Red=SIFT)")
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig

class DataLoader:
    """Class to load and manage dataset"""
    
    @staticmethod
    def load_dataset(data_path: str) -> List[Tuple[str, np.ndarray]]:
        """Load all images from a directory"""
        images = []
        for img_path in tqdm(glob.glob(os.path.join(data_path, "*.jpg")) + 
                            glob.glob(os.path.join(data_path, "*.png")) +
                            glob.glob(os.path.join(data_path, "*.jpeg"))):
            img = cv2.imread(img_path)
            if img is not None:
                images.append((os.path.basename(img_path), img))
        return images
    
    @staticmethod
    def create_transformations(image: np.ndarray, base_name: str) -> Dict[str, Tuple[str, np.ndarray]]:
        """Create various transformations of an image"""
        transformations = {}
        
        # Scale transformations
        scales = [0.5, 0.75, 1.25, 1.5, 2.0]
        for scale in scales:
            scaled = ImageTransformer.scale_image(image, scale)
            transformations[f"{base_name}_scale_{scale}"] = scaled
        
        # Rotation transformations
        rotations = [15, 30, 45, 60, 90]
        for angle in rotations:
            rotated = ImageTransformer.rotate_image(image, angle)
            transformations[f"{base_name}_rotate_{angle}"] = rotated
        
        # Brightness transformations
        brightness_factors = [0.5, 0.75, 1.25, 1.5]
        for factor in brightness_factors:
            bright = ImageTransformer.change_brightness(image, factor)
            transformations[f"{base_name}_bright_{factor}"] = bright
        
        # Blur transformations
        blur_sizes = [3, 5, 7, 9]
        for size in blur_sizes:
            blurred = ImageTransformer.add_gaussian_blur(image, size)
            transformations[f"{base_name}_blur_{size}"] = blurred
        
        # Noise transformations
        noise_sigmas = [10, 20, 30, 40]
        for sigma in noise_sigmas:
            noisy = ImageTransformer.add_gaussian_noise(image, sigma=sigma)
            transformations[f"{base_name}_noise_{sigma}"] = noisy
        
        return transformations