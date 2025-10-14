import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path
import yaml
from tqdm import tqdm


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

def xywh_to_xyxy(box, img_w, img_h):
    """Convert normalized YOLO [x_center, y_center, w, h] to pixel [x1, y1, x2, y2]."""
    x_c, y_c, w, h = box
    x1 = (x_c - w / 2) * img_w
    y1 = (y_c - h / 2) * img_h
    x2 = (x_c + w / 2) * img_w
    y2 = (y_c + h / 2) * img_h
    return [x1, y1, x2, y2]

def iou(boxA, boxB):
    """Compute IoU between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

def evaluate_dataset(model, dataset_dir, class_names, iou_threshold=0.5, visualize_balls=True):
    results = {cls: {"detected": 0, "total": 0} for cls in class_names}
    detected_ball_images = []

    image_paths = list(Path(dataset_dir).glob("*.jpg"))

    for img_path in tqdm(image_paths, desc=f"Evaluating {dataset_dir.name}"):
        label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_boxes_raw = load_annotations(label_path)

        gt_boxes = []
        for cls_id, x, y, bw, bh in gt_boxes_raw:
            box_xyxy = xywh_to_xyxy([x, y, bw, bh], w, h)
            gt_boxes.append((int(cls_id), box_xyxy))
            results[class_names[int(cls_id)]]["total"] += 1

        # Model prediction
        pred = model(img_path, verbose=False)[0]
        pred_boxes = [(int(b.cls), b.xyxy[0].cpu().numpy()) for b in pred.boxes]

        # Count matches
        for cls_id, box_gt in gt_boxes:
            for cls_p, box_p in pred_boxes:
                if cls_id == cls_p and iou(box_gt, box_p) >= iou_threshold:
                    results[class_names[cls_id]]["detected"] += 1
                    break

        # Store ball detections
        if any(class_names[int(b.cls)] == "ball" for b in pred.boxes):
            detected_ball_images.append((img_path, pred))

    # Calculate percentages
    percentages = {cls: (v["detected"] / v["total"] * 100 if v["total"] > 0 else 0)
                   for cls, v in results.items()}

    print(f"\n📊 Results for {dataset_dir.name}:")
    for cls, pct in percentages.items():
        print(f"  {cls}: {pct:.2f}% detected ({results[cls]['detected']}/{results[cls]['total']})")

    # Visualize ball detections
    if visualize_balls and detected_ball_images:
        print(f"\n🎯 Found {len(detected_ball_images)} images with detected balls. Showing up to 3 examples:")
        for i, (img_path, pred) in enumerate(detected_ball_images[:3]):
            img = cv2.imread(str(img_path))
            visualize_annotations(img_path, [], title=f"Detected ball: {img_path.name}", class_names=class_names)
            pred.show()

    return results, percentages

