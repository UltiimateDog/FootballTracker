import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path
import yaml

def load_annotations(label_path):
    """Load YOLO format annotations"""
    annotations = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append([class_id, x_center, y_center, width, height])
    return annotations


def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    """Convert YOLO format to bounding box coordinates"""
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return x1, y1, x2, y2


def visualize_annotations(image_path, annotations, title="", class_names=None):
    """Visualize image with annotations"""
    if class_names is None:
        class_names = ['ball', 'goalkeeper', 'player', 'referee']

    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Color map for different classes
    colors = {
        0: (255, 0, 0),  # ball - red
        1: (0, 255, 0),  # goalkeeper - green
        2: (0, 0, 255),  # player - blue
        3: (255, 255, 0)  # referee - yellow
    }

    for ann in annotations:
        class_id, x_center, y_center, width, height = ann
        x1, y1, x2, y2 = yolo_to_bbox(x_center, y_center, width, height, w, h)

        color = colors.get(class_id, (128, 128, 128))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Add class label
        label = class_names[class_id]
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

    return img