"""
Player Projection Module

This module handles the projection of YOLO-detected players, balls, and other objects
from 2D image coordinates to 3D world coordinates and then to top-down field view.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# Import field tracker components
from Complete.field_tracker.calibrationRoutines import calibrate_from_image
from Complete.field_tracker.projectionHelpers import project_to_screen
from Complete.field_tracker.Constants import (
    DEFAULT_GUESS_FX, DEFAULT_GUESS_ROT, DEFAULT_GUESS_TRANS,
    SOCCER_FIELD_WIDTH, SOCCER_FIELD_HEIGHT,
    corner_front_left_world, corner_back_left_world
)


class PlayerProjector:
    """
    Projects YOLO detections to top-down field view using camera calibration.
    """
    
    def __init__(self):
        self.class_names = ['ball', 'goalkeeper', 'player', 'referee']
        self.class_colors = {
            0: (255, 0, 0),    # ball - red
            1: (0, 255, 0),    # goalkeeper - green  
            2: (0, 0, 255),    # player - blue
            3: (255, 255, 0)   # referee - yellow
        }
        self.point_sizes = {
            0: 8,  # ball - larger
            1: 6,  # goalkeeper
            2: 4,  # player
            3: 5   # referee
        }
    
    def unproject_to_field(self, pixel_point: Tuple[int, int], K: np.ndarray, 
                          to_device_from_world: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Unproject a 2D pixel point to 3D world coordinates on the field (y=0 plane).
        
        Args:
            pixel_point: (x, y) pixel coordinates
            K: Camera intrinsic matrix
            to_device_from_world: Camera pose transformation matrix
            
        Returns:
            (x, z) world coordinates on the field, or None if unprojection fails
        """
        try:
            # Convert pixel to normalized camera coordinates
            x_pixel, y_pixel = pixel_point
            x_norm = (x_pixel - K[0, 2]) / K[0, 0]
            y_norm = (y_pixel - K[1, 2]) / K[1, 1]
            
            # Camera ray direction in camera coordinates
            ray_camera = np.array([x_norm, y_norm, 1.0])
            
            # Transform ray to world coordinates
            to_world_from_device = np.linalg.inv(to_device_from_world)
            ray_world = to_world_from_device[:3, :3] @ ray_camera
            camera_pos_world = to_world_from_device[:3, 3]
            
            # Intersect ray with field plane (y = 0)
            if abs(ray_world[1]) < 1e-6:  # Ray parallel to field
                return None
                
            t = -camera_pos_world[1] / ray_world[1]
            if t < 0:  # Intersection behind camera
                return None
                
            # Calculate intersection point
            intersection = camera_pos_world + t * ray_world
            return (intersection[0], intersection[2])
            
        except Exception:
            return None
    
    def yolo_to_center_point(self, detection: List[float], img_width: int, img_height: int) -> Tuple[int, int]:
        """
        Convert YOLO detection to center point in pixel coordinates.
        
        Args:
            detection: [class_id, x_center, y_center, width, height] in normalized coordinates
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            (x, y) center point in pixel coordinates
        """
        _, x_center, y_center, _, _ = detection
        x_pixel = int(x_center * img_width)
        y_pixel = int(y_center * img_height)
        return (x_pixel, y_pixel)
    
    def create_top_view_field(self, width: int = 800, height: int = 600) -> np.ndarray:
        """
        Create a blank top-down field view with field lines.
        
        Args:
            width: Output image width
            height: Output image height
            
        Returns:
            Field image with lines drawn
        """
        field_img = np.zeros((height, width, 3), dtype=np.uint8)
        field_img.fill(34)  # Dark green background
        
        # Calculate field boundaries in image coordinates
        field_width_m = SOCCER_FIELD_WIDTH
        field_height_m = SOCCER_FIELD_HEIGHT
        
        # Add margins
        margin_x = width * 0.1
        margin_y = height * 0.1
        field_width_px = width - 2 * margin_x
        field_height_px = height - 2 * margin_y
        
        # Field outline
        field_rect = (
            int(margin_x), int(margin_y),
            int(field_width_px), int(field_height_px)
        )
        cv2.rectangle(field_img, 
                     (field_rect[0], field_rect[1]),
                     (field_rect[0] + field_rect[2], field_rect[1] + field_rect[3]),
                     (255, 255, 255), 2)
        
        # Center line
        center_x = int(margin_x + field_width_px / 2)
        cv2.line(field_img, 
                (center_x, int(margin_y)),
                (center_x, int(margin_y + field_height_px)),
                (255, 255, 255), 2)
        
        # Center circle
        center_y = int(margin_y + field_height_px / 2)
        circle_radius = int(9.15 * field_width_px / field_width_m)  # 9.15m radius
        cv2.circle(field_img, (center_x, center_y), circle_radius, (255, 255, 255), 2)
        
        return field_img
    
    def world_to_field_image(self, world_point: Tuple[float, float], 
                           img_width: int, img_height: int) -> Tuple[int, int]:
        """
        Convert world coordinates to field image pixel coordinates.
        
        Args:
            world_point: (x, z) coordinates in world space
            img_width: Field image width
            img_height: Field image height
            
        Returns:
            (x, y) pixel coordinates in field image
        """
        world_x, world_z = world_point
        
        # Field dimensions
        field_width_m = SOCCER_FIELD_WIDTH
        field_height_m = SOCCER_FIELD_HEIGHT
        
        # Image margins
        margin_x = img_width * 0.1
        margin_y = img_height * 0.1
        field_width_px = img_width - 2 * margin_x
        field_height_px = img_height - 2 * margin_y
        
        # Convert world coordinates to image coordinates
        # World: x=[-52.5, 52.5], z=[-34, 34]
        # Image: x=[margin_x, margin_x + field_width_px], y=[margin_y, margin_y + field_height_px]
        # Fix x-axis inversion by flipping the x coordinate
        
        x_normalized = (world_x + field_width_m/2) / field_width_m
        z_normalized = (-world_z + field_height_m/2) / field_height_m
        
        img_x = int(margin_x + x_normalized * field_width_px)
        img_y = int(margin_y + z_normalized * field_height_px)
        
        return img_x, img_y
    
    def process_detections_to_topview(self, image: np.ndarray, detections: List[List[float]], 
                                    field_width: int = 800, field_height: int = 600) -> np.ndarray:
        """
        Process YOLO detections and create top-down field view with projected players.
        
        Args:
            image: Input image
            detections: List of YOLO detections [class_id, x_center, y_center, width, height]
            field_width: Output field image width
            field_height: Output field image height
            
        Returns:
            Top-down field view with players projected as points
        """
        img_height, img_width = image.shape[:2]
        
        # Calibrate camera
        guess_fx = DEFAULT_GUESS_FX
        guess_rot = np.array(DEFAULT_GUESS_ROT)
        guess_trans = DEFAULT_GUESS_TRANS
        
        K, to_device_from_world, _, _, _ = calibrate_from_image(
            image, guess_fx, guess_rot, guess_trans
        )
        
        # Create field image
        field_img = self.create_top_view_field(field_width, field_height)
        
        if to_device_from_world is None:
            # Add text indicating calibration failed
            cv2.putText(field_img, "Camera calibration failed", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return field_img
        
        # Process each detection
        projected_count = 0
        for detection in detections:
            class_id = int(detection[0])
            
            # Get center point of detection
            center_pixel = self.yolo_to_center_point(detection, img_width, img_height)
            
            # Unproject to field coordinates
            world_point = self.unproject_to_field(center_pixel, K, to_device_from_world)
            
            if world_point is not None:
                # Convert to field image coordinates
                field_pixel = self.world_to_field_image(world_point, field_width, field_height)
                
                # Check if point is within field bounds
                if (0 <= field_pixel[0] < field_width and 0 <= field_pixel[1] < field_height):
                    color = self.class_colors.get(class_id, (128, 128, 128))
                    size = self.point_sizes.get(class_id, 4)
                    
                    # Draw point
                    cv2.circle(field_img, field_pixel, size, color, -1)
                    
                    projected_count += 1
        
        # Add info text
        info_text = f"Projected: {projected_count}/{len(detections)} detections"
        cv2.putText(field_img, info_text, (10, field_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return field_img


def process_yolo_predictions_to_topview(model_predictions: Any, image: np.ndarray, 
                                      field_width: int = 800, field_height: int = 600) -> np.ndarray:
    """
    Convenience function to process YOLO model predictions directly.
    
    Args:
        model_predictions: YOLO model prediction results
        image: Input image
        field_width: Output field image width  
        field_height: Output field image height
        
    Returns:
        Top-down field view with projected detections
    """
    projector = PlayerProjector()
    
    # Convert YOLO predictions to detection format
    detections = []
    if hasattr(model_predictions, 'boxes') and model_predictions.boxes is not None:
        for box in model_predictions.boxes:
            # Extract box data
            cls_id = int(box.cls.cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy()
            
            # Convert to YOLO format (normalized center coordinates)
            img_h, img_w = image.shape[:2]
            x1, y1, x2, y2 = xyxy
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h
            
            detections.append([cls_id, x_center, y_center, width, height])
    
    return projector.process_detections_to_topview(image, detections, field_width, field_height)