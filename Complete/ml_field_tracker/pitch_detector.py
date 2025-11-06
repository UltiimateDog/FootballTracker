import cv2
import numpy as np
from typing import Optional, Tuple
from ultralytics import YOLO
from .config import SoccerPitchConfiguration
from .view_transformer import ViewTransformer
from .pitch_annotator import draw_pitch, draw_points_on_pitch, Color


class KeyPoints:
    """Simple keypoints container similar to supervision's KeyPoints"""
    def __init__(self, xy: np.ndarray):
        self.xy = [xy]  # Wrap in list to match supervision format


class MLPitchDetector:
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize ML-based pitch detector using YOLO keypoint detection.
        
        Args:
            model_path: Path to the trained YOLO keypoint detection model
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.model = YOLO(model_path).to(device=device)
        self.config = SoccerPitchConfiguration()
        self.device = device
        
    def detect_keypoints(self, frame: np.ndarray) -> Optional[KeyPoints]:
        """
        Detect field keypoints in the frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            KeyPoints object or None if detection fails
        """
        try:
            result = self.model(frame, verbose=False)[0]
            
            # Extract keypoints from YOLO result
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints_data = result.keypoints.data[0]  # First detection
                # Filter out invalid keypoints (confidence > 0.5)
                valid_mask = keypoints_data[:, 2] > 0.5
                valid_keypoints = keypoints_data[valid_mask][:, :2]  # x, y only
                
                if len(valid_keypoints) >= 4:  # Need minimum 4 points for homography
                    return KeyPoints(valid_keypoints.cpu().numpy())
            
            return None
        except Exception as e:
            print(f"Keypoint detection failed: {e}")
            return None
    
    def create_view_transformer(self, keypoints: KeyPoints) -> Optional[ViewTransformer]:
        """
        Create view transformer from detected keypoints to field coordinates.
        
        Args:
            keypoints: Detected field keypoints
            
        Returns:
            ViewTransformer object or None if creation fails
        """
        try:
            detected_points = keypoints.xy[0]
            
            # Use first N detected points and corresponding field vertices
            n_points = min(len(detected_points), len(self.config.vertices))
            if n_points < 4:
                return None
                
            source_points = detected_points[:n_points].astype(np.float32)
            target_points = np.array(self.config.vertices[:n_points], dtype=np.float32)
            
            return ViewTransformer(source_points, target_points)
        except Exception as e:
            print(f"View transformer creation failed: {e}")
            return None
    
    def annotate_keypoints(self, frame: np.ndarray, keypoints: KeyPoints) -> np.ndarray:
        """
        Annotate detected keypoints on the frame.
        
        Args:
            frame: Input frame
            keypoints: Detected keypoints
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for i, point in enumerate(keypoints.xy[0]):
            x, y = int(point[0]), int(point[1])
            # Draw keypoint
            cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)
            # Draw label
            cv2.putText(annotated_frame, str(i+1), (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def generate_top_view(self, frame: np.ndarray, transformer: ViewTransformer, 
                         detections: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate top-down view of the field.
        
        Args:
            frame: Input frame
            transformer: View transformer
            detections: Optional player/ball detections to transform
            
        Returns:
            Top-down view image
        """
        # Create base pitch
        pitch = draw_pitch(self.config)
        
        # If detections provided, transform and draw them
        if detections is not None and len(detections) > 0:
            try:
                transformed_points = transformer.transform_points(detections.astype(np.float32))
                pitch = draw_points_on_pitch(
                    self.config, 
                    transformed_points,
                    face_color=Color.RED,
                    radius=15,
                    pitch=pitch
                )
            except Exception as e:
                print(f"Failed to transform detections: {e}")
        
        return pitch
    
    def process_frame(self, frame: np.ndarray, show_keypoints: bool = True, 
                     generate_topview: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process a single frame for pitch detection.
        
        Args:
            frame: Input frame
            show_keypoints: Whether to show detected keypoints
            generate_topview: Whether to generate top-down view
            
        Returns:
            Tuple of (annotated_frame, top_view_frame)
        """
        # Detect keypoints
        keypoints = self.detect_keypoints(frame)
        
        annotated_frame = frame.copy()
        top_view = None
        
        if keypoints is not None:
            if show_keypoints:
                annotated_frame = self.annotate_keypoints(annotated_frame, keypoints)
            
            if generate_topview:
                transformer = self.create_view_transformer(keypoints)
                if transformer is not None:
                    top_view = self.generate_top_view(frame, transformer)
        
        return annotated_frame, top_view