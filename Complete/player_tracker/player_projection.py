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
from Complete.player_tracker.Constants import (
    CLASS_NAMES, CLASS_COLORS, POINT_SIZES,
    DEFAULT_FIELD_WIDTH, DEFAULT_FIELD_HEIGHT, FIELD_BACKGROUND_COLOR,
    FIELD_LINE_COLOR, FIELD_LINE_THICKNESS, FIELD_MARGIN_RATIO,
    CENTER_CIRCLE_RADIUS, PENALTY_AREA_WIDTH, PENALTY_AREA_HEIGHT,
    GOAL_AREA_WIDTH, GOAL_AREA_HEIGHT, PENALTY_SPOT_DISTANCE, SPOT_RADIUS,
    RAY_PARALLEL_THRESHOLD, FIELD_PLANE_Y,
    INFO_TEXT_FONT, INFO_TEXT_SCALE, INFO_TEXT_COLOR, INFO_TEXT_THICKNESS, INFO_TEXT_MARGIN,
    ERROR_TEXT_FONT, ERROR_TEXT_SCALE, ERROR_TEXT_COLOR, ERROR_TEXT_THICKNESS, ERROR_TEXT_POSITION
)


class PlayerProjector:
    """
    Projects YOLO detections to top-down field view using camera calibration.
    """
    
    def __init__(self):
        self.class_names = CLASS_NAMES
        self.class_colors = CLASS_COLORS
        self.point_sizes = POINT_SIZES
    
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
            if abs(ray_world[1]) < RAY_PARALLEL_THRESHOLD:  # Ray parallel to field
                return None
                
            t = -camera_pos_world[1] / ray_world[1]
            if t < 0:  # Intersection behind camera
                return None
                
            # Calculate intersection point
            intersection = camera_pos_world + t * ray_world
            return (intersection[0], intersection[2])
            
        except Exception:
            return None
    
    def yolo_to_ground_point(self, detection: List[float], img_width: int, img_height: int) -> Tuple[int, int]:
        """
        Convert YOLO detection to ground contact point (bottom center for players, center for ball).
        
        Args:
            detection: [class_id, x_center, y_center, width, height] in normalized coordinates
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            (x, y) ground contact point in pixel coordinates
        """
        class_id, x_center, y_center, width, height = detection
        x_pixel = int(x_center * img_width)
        
        # For players/goalkeepers/referees: use bottom of bounding box (feet)
        # For ball: use center (ball is on the ground)
        if int(class_id) == 0:  # ball
            y_pixel = int(y_center * img_height)
        else:  # players, goalkeepers, referees
            y_pixel = int((y_center + height/2) * img_height)
            
        return (x_pixel, y_pixel)
    
    def create_top_view_field(self, width: int = DEFAULT_FIELD_WIDTH, height: int = DEFAULT_FIELD_HEIGHT) -> np.ndarray:
        """
        Create a blank top-down field view with field lines.
        
        Args:
            width: Output image width
            height: Output image height
            
        Returns:
            Field image with lines drawn
        """
        field_img = np.zeros((height, width, 3), dtype=np.uint8)
        field_img.fill(FIELD_BACKGROUND_COLOR)
        
        # Calculate field boundaries in image coordinates
        field_width_m = SOCCER_FIELD_WIDTH
        field_height_m = SOCCER_FIELD_HEIGHT
        
        # Add margins
        margin_x = width * FIELD_MARGIN_RATIO
        margin_y = height * FIELD_MARGIN_RATIO
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
                     FIELD_LINE_COLOR, FIELD_LINE_THICKNESS)
        
        # Center line
        center_x = int(margin_x + field_width_px / 2)
        cv2.line(field_img, 
                (center_x, int(margin_y)),
                (center_x, int(margin_y + field_height_px)),
                FIELD_LINE_COLOR, FIELD_LINE_THICKNESS)
        
        # Center circle
        center_y = int(margin_y + field_height_px / 2)
        circle_radius = int(CENTER_CIRCLE_RADIUS * field_width_px / field_width_m)
        cv2.circle(field_img, (center_x, center_y), circle_radius, FIELD_LINE_COLOR, FIELD_LINE_THICKNESS)
        
        # Penalty areas
        penalty_width = int(PENALTY_AREA_WIDTH * field_width_px / field_width_m)
        penalty_height = int(PENALTY_AREA_HEIGHT * field_height_px / field_height_m)
        penalty_y = int(margin_y + (field_height_px - penalty_width) / 2)
        
        # Left penalty area
        cv2.rectangle(field_img,
                     (int(margin_x), penalty_y),
                     (int(margin_x + penalty_height), penalty_y + penalty_width),
                     FIELD_LINE_COLOR, FIELD_LINE_THICKNESS)
        
        # Right penalty area
        cv2.rectangle(field_img,
                     (int(margin_x + field_width_px - penalty_height), penalty_y),
                     (int(margin_x + field_width_px), penalty_y + penalty_width),
                     FIELD_LINE_COLOR, FIELD_LINE_THICKNESS)
        
        # Goal areas
        goal_width = int(GOAL_AREA_WIDTH * field_width_px / field_width_m)
        goal_height = int(GOAL_AREA_HEIGHT * field_height_px / field_height_m)
        goal_y = int(margin_y + (field_height_px - goal_width) / 2)
        
        # Left goal area
        cv2.rectangle(field_img,
                     (int(margin_x), goal_y),
                     (int(margin_x + goal_height), goal_y + goal_width),
                     FIELD_LINE_COLOR, FIELD_LINE_THICKNESS)
        
        # Right goal area
        cv2.rectangle(field_img,
                     (int(margin_x + field_width_px - goal_height), goal_y),
                     (int(margin_x + field_width_px), goal_y + goal_width),
                     FIELD_LINE_COLOR, FIELD_LINE_THICKNESS)
        
        # Penalty spots
        penalty_spot_x = int(PENALTY_SPOT_DISTANCE * field_width_px / field_width_m)
        cv2.circle(field_img, (int(margin_x + penalty_spot_x), center_y), SPOT_RADIUS, FIELD_LINE_COLOR, -1)
        cv2.circle(field_img, (int(margin_x + field_width_px - penalty_spot_x), center_y), SPOT_RADIUS, FIELD_LINE_COLOR, -1)
        
        # Center spot
        cv2.circle(field_img, (center_x, center_y), SPOT_RADIUS, FIELD_LINE_COLOR, -1)
        
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
        margin_x = img_width * FIELD_MARGIN_RATIO
        margin_y = img_height * FIELD_MARGIN_RATIO
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
                                    field_width: int = DEFAULT_FIELD_WIDTH, field_height: int = DEFAULT_FIELD_HEIGHT) -> np.ndarray:
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
                       ERROR_TEXT_POSITION, ERROR_TEXT_FONT, ERROR_TEXT_SCALE, ERROR_TEXT_COLOR, ERROR_TEXT_THICKNESS)
            return field_img
        
        # Process each detection
        projected_count = 0
        for detection in detections:
            class_id = int(detection[0])
            
            # Get ground contact point of detection
            ground_pixel = self.yolo_to_ground_point(detection, img_width, img_height)
            
            # Unproject to field coordinates
            world_point = self.unproject_to_field(ground_pixel, K, to_device_from_world)
            
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
        cv2.putText(field_img, info_text, (INFO_TEXT_MARGIN, field_height - 20), 
                   INFO_TEXT_FONT, INFO_TEXT_SCALE, INFO_TEXT_COLOR, INFO_TEXT_THICKNESS)
        
        return field_img


def process_yolo_predictions_to_topview(model_predictions: Any, image: np.ndarray, 
                                      field_width: int = DEFAULT_FIELD_WIDTH, field_height: int = DEFAULT_FIELD_HEIGHT) -> np.ndarray:
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