import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path
import yaml

def preprocess_image(image_path, show_steps=True):
    """Apply preprocessing steps as described in the proposal"""
    # Load image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Gaussian filter for noise reduction
    # (5,5) kernel: removes camera noise while preserving player/ball details
    # sigma=0: auto-calculated for optimal smoothing
    gaussian_filtered = cv2.GaussianBlur(img_rgb, (5, 5), 0)

    # 2. Convert to grayscale for edge detection
    gray = cv2.cvtColor(gaussian_filtered, cv2.COLOR_RGB2GRAY)

    # 3. Edge detection
    # 50: low threshold catches subtle field lines and player boundaries
    # 150: high threshold (3:1 ratio) ensures strong edges without noise
    edges = cv2.Canny(gray, 50, 150)

    # 4. Enhance contrast
    # clipLimit=2.0: prevents noise amplification while boosting contrast
    # tileGridSize=(8,8): adapts to local lighting variations across field
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    if show_steps:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(gaussian_filtered)
        axes[0, 1].set_title('Gaussian Filtered')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(gray, cmap='gray')
        axes[0, 2].set_title('Grayscale')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(edges, cmap='gray')
        axes[1, 0].set_title('Edge Detection')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(enhanced, cmap='gray')
        axes[1, 1].set_title('Enhanced Contrast')
        axes[1, 1].axis('off')

        # Show difference
        diff = cv2.absdiff(gray, enhanced)
        axes[1, 2].imshow(diff, cmap='hot')
        axes[1, 2].set_title('Enhancement Difference')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

    return {
        'original': img_rgb,
        'gaussian': gaussian_filtered,
        'gray': gray,
        'edges': edges,
        'enhanced': enhanced
    }