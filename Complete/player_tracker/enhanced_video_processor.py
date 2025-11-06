"""
Enhanced Video Processing with Combined Views and Frame Persistence

Provides video processing with options for top-view only or combined view,
and maintains last known positions when detections fail.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union, Any
from ultralytics import YOLO

from Complete.field_tracker.Constants import (
    VIDEO_EXTENSIONS, VIDEO_FOURCC, FRAME_PROGRESS_INTERVAL, VIDEO_OUTPUT_EXT
)
from Complete.player_tracker.player_projection import process_yolo_predictions_to_topview
from Complete.ml_field_tracker.pitch_detector import MLPitchDetector
from Complete.player_tracker.Constants import (
    DEFAULT_FIELD_WIDTH_PROCESSING, DEFAULT_FIELD_HEIGHT_PROCESSING,
    ENHANCED_TOPVIEW_SUFFIX, ENHANCED_COMBINED_SUFFIX
)


def process_video_enhanced(input_path: Union[str, Path], 
                          model_path: Union[str, Path],
                          output_path: Optional[Union[str, Path]] = None,
                          combined_view: bool = False,
                          field_width: int = DEFAULT_FIELD_WIDTH_PROCESSING,
                          field_height: int = DEFAULT_FIELD_HEIGHT_PROCESSING,
                          field_tracker_type: str = "traditional",
                          pitch_model_path: Optional[Union[str, Path]] = None) -> None:
    """
    Enhanced video processing with combined view option and frame persistence.
    
    Args:
        input_path: Path to input video
        model_path: Path to YOLO model file
        output_path: Path for output (auto-generated if None)
        combined_view: If True, creates side-by-side original+topview
        field_width: Width of field view
        field_height: Height of field view
        field_tracker_type: Type of field tracker to use ("traditional" or "ml")
        pitch_model_path: Path to pitch detection model (required if field_tracker_type="ml")
    """
    input_path = Path(input_path)
    model_path = Path(model_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Load YOLO model
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(str(model_path))
    
    # Initialize field tracker if ML type is selected
    pitch_detector = None
    if field_tracker_type == "ml":
        if pitch_model_path is None:
            raise ValueError("pitch_model_path is required when field_tracker_type='ml'")
        if not Path(pitch_model_path).exists():
            raise FileNotFoundError(f"Pitch model path does not exist: {pitch_model_path}")
        print(f"Loading ML pitch detector from {pitch_model_path}...")
        pitch_detector = MLPitchDetector(str(pitch_model_path))
    
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {input_path.name}")
    print(f"Mode: {'Combined view' if combined_view else 'Top-view only'}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    
    # Set output path
    if output_path is None:
        suffix = ENHANCED_COMBINED_SUFFIX if combined_view else ENHANCED_TOPVIEW_SUFFIX
        output_path = input_path.parent / f"{input_path.stem}{suffix}{VIDEO_OUTPUT_EXT}"
    
    # Determine output dimensions
    if combined_view:
        # Get first frame to calculate dimensions
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        frame_height, frame_width = first_frame.shape[:2]
        target_height = field_height  # Use actual field height (680px)
        new_width = int(frame_width * target_height / frame_height)
        output_width = new_width + field_width
        output_height = target_height
    else:
        output_width = field_width
        output_height = field_height
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
    
    # Frame persistence variables
    last_valid_field_view = None
    last_valid_annotated = None
    
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection
            results = model(frame, verbose=False)
            
            # Check if we have valid detections
            has_detections = len(results[0].boxes) > 0 if results[0].boxes is not None else False
            
            if has_detections:
                # Generate new views
                if field_tracker_type == "ml" and pitch_detector is not None:
                    field_view = _process_with_ml_tracker(results[0], frame, field_width, field_height, pitch_detector)
                else:
                    field_view = process_yolo_predictions_to_topview(
                        results[0], frame, field_width, field_height
                    )
                last_valid_field_view = field_view.copy()
                
                if combined_view:
                    original_resized = cv2.resize(frame, (new_width, target_height))
                    last_valid_annotated = original_resized.copy()
            else:
                # Use last valid views if available
                if last_valid_field_view is not None:
                    field_view = last_valid_field_view
                else:
                    # Create empty field view
                    field_view = np.zeros((field_height, field_width, 3), dtype=np.uint8)
                
                if combined_view:
                    if last_valid_annotated is not None:
                        original_resized = last_valid_annotated
                    else:
                        # Use original frame resized
                        original_resized = cv2.resize(frame, (new_width, target_height))
            
            # Create output frame
            if combined_view:
                output_frame = np.hstack([original_resized, field_view])
            else:
                output_frame = field_view
            
            # Write frame
            out.write(output_frame)
            
            frame_idx += 1
            if frame_idx % FRAME_PROGRESS_INTERVAL == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Processed {frame_idx}/{total_frames} frames ({progress:.1f}%)")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    print(f"✅ Output video saved to: {output_path}")


def _process_with_ml_tracker(yolo_results: Any, image: np.ndarray, 
                           field_width: int, field_height: int,
                           pitch_detector: MLPitchDetector) -> np.ndarray:
    """
    Process YOLO results using ML field tracker for projection.
    
    Args:
        yolo_results: YOLO detection results
        image: Input image
        field_width: Output field width
        field_height: Output field height
        pitch_detector: ML pitch detector instance
        
    Returns:
        Top-down field view with projected players
    """
    # Get field transformation from ML tracker
    _, top_view = pitch_detector.process_frame(image, show_keypoints=False, generate_topview=True)
    
    if top_view is None:
        # Fallback to traditional method if ML tracker fails
        return process_yolo_predictions_to_topview(yolo_results, image, field_width, field_height)
    
    # Create field view and project YOLO detections
    from Complete.player_tracker.player_projection import PlayerProjector
    projector = PlayerProjector()
    field_img = projector.create_top_view_field(field_width, field_height)
    
    # Convert YOLO predictions to detection format and draw on field
    if hasattr(yolo_results, 'boxes') and yolo_results.boxes is not None:
        for box in yolo_results.boxes:
            cls_id = int(box.cls.cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy()
            
            # Use center bottom for players, center for ball
            x1, y1, x2, y2 = xyxy
            if cls_id == 0:  # ball
                point_x, point_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            else:  # players
                point_x, point_y = int((x1 + x2) / 2), int(y2)
            
            # Simple projection using ML field detection (placeholder)
            # This would need proper homography from ML tracker
            color = projector.class_colors.get(cls_id, (128, 128, 128))
            size = projector.point_sizes.get(cls_id, 4)
            
            # For now, use a simple mapping - this should be replaced with proper homography
            field_x = int((point_x / image.shape[1]) * field_width)
            field_y = int((point_y / image.shape[0]) * field_height)
            
            if 0 <= field_x < field_width and 0 <= field_y < field_height:
                cv2.circle(field_img, (field_x, field_y), size, color, -1)
    
    return field_img


if __name__ == "__main__":
    # Example usage
    input_path = "/Users/alanpehz/Documents/Personal/True Computer Vision/FootballTracker/Complete/test_content/demo1.mp4"
    model_path = "/Users/alanpehz/Documents/Personal/True Computer Vision/FootballTracker/Complete/models/ball_and_player_model.pt"
    pitch_model_path = "/Users/alanpehz/Documents/Personal/True Computer Vision/FootballTracker/Complete/models/pitch_tracker.pt"
        
    # Combined view with traditional tracker
    # process_video_enhanced(input_path, model_path, combined_view=True)
    
    # Combined view with ML tracker
    process_video_enhanced(input_path, model_path, combined_view=True, field_tracker_type="ml", pitch_model_path=pitch_model_path)