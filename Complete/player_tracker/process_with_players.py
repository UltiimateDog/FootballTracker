"""
Process Media with Player Tracking

Main processing pipeline that combines field detection with YOLO player tracking
to generate top-down field views with player positions.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union, Any
from ultralytics import YOLO

from Complete.field_tracker.Constants import (
    VIDEO_EXTENSIONS, VIDEO_FOURCC, FRAME_PROGRESS_INTERVAL,
    TOPVIEW_SUFFIX, VIDEO_OUTPUT_EXT
)
from Complete.player_tracker.player_projection import process_yolo_predictions_to_topview
from Complete.ml_field_tracker.pitch_detector import MLPitchDetector
from Complete.player_tracker.Constants import (
    DEFAULT_FIELD_WIDTH_PROCESSING, DEFAULT_FIELD_HEIGHT_PROCESSING,
    TOPVIEW_SUFFIX_PLAYERS, COMBINED_SUFFIX, TARGET_HEIGHT_COMBINED
)


def process_media_with_players(input_path: Union[str, Path], 
                             model_path: Union[str, Path],
                             output_path: Optional[Union[str, Path]] = None,
                             field_width: int = DEFAULT_FIELD_WIDTH_PROCESSING,
                             field_height: int = DEFAULT_FIELD_HEIGHT_PROCESSING,
                             field_tracker_type: str = "traditional",
                             pitch_model_path: Optional[Union[str, Path]] = None) -> None:
    """
    Process an image or video with YOLO player detection and field projection.
    
    Args:
        input_path: Path to input image or video
        model_path: Path to YOLO model file
        output_path: Path for output (auto-generated if None)
        field_width: Width of output field view (default maintains FIFA 105m x 68m proportions)
        field_height: Height of output field view (default maintains FIFA 105m x 68m proportions)
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
    
    # Detect if input is video
    is_video = input_path.suffix.lower() in VIDEO_EXTENSIONS
    
    if is_video:
        _process_video_with_players(input_path, model, output_path, field_width, field_height, field_tracker_type, pitch_detector)
    else:
        _process_image_with_players(input_path, model, output_path, field_width, field_height, field_tracker_type, pitch_detector)


def _process_image_with_players(input_path: Path, model: YOLO, 
                               output_path: Optional[Path],
                               field_width: int, field_height: int,
                               field_tracker_type: str, pitch_detector: Optional[MLPitchDetector]) -> None:
    """Process single image with player tracking."""
    # Load image
    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"Cannot read image: {input_path}")
    
    print(f"Processing image: {input_path.name}")
    
    # Run YOLO detection
    results = model(img, verbose=False)
    
    # Generate top-down view
    if field_tracker_type == "ml" and pitch_detector is not None:
        field_view = _process_with_ml_tracker(results[0], img, field_width, field_height, pitch_detector)
    else:
        field_view = process_yolo_predictions_to_topview(
            results[0], img, field_width, field_height
        )
    
    # Set output path
    if output_path is None:
        suffix = f"{TOPVIEW_SUFFIX_PLAYERS}_players{input_path.suffix}"
        output_path = input_path.parent / f"{input_path.stem}{suffix}"
    
    # Save result
    cv2.imwrite(str(output_path), field_view)
    print(f"✅ Output saved to: {output_path}")


def _process_video_with_players(input_path: Path, model: YOLO,
                               output_path: Optional[Path],
                               field_width: int, field_height: int,
                               field_tracker_type: str, pitch_detector: Optional[MLPitchDetector]) -> None:
    """Process video with player tracking."""
    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {input_path.name}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    
    # Set output path
    if output_path is None:
        suffix = f"{TOPVIEW_SUFFIX_PLAYERS}_players{VIDEO_OUTPUT_EXT}"
        output_path = input_path.parent / f"{input_path.stem}{suffix}"
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (field_width, field_height))
    
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection
            results = model(frame, verbose=False)
            
            # Generate top-down view
            if field_tracker_type == "ml" and pitch_detector is not None:
                field_view = _process_with_ml_tracker(results[0], frame, field_width, field_height, pitch_detector)
            else:
                field_view = process_yolo_predictions_to_topview(
                    results[0], frame, field_width, field_height
                )
            
            # Write frame
            out.write(field_view)
            
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


def create_side_by_side_view(input_path: Union[str, Path],
                           model_path: Union[str, Path],
                           output_path: Optional[Union[str, Path]] = None,
                           field_tracker_type: str = "traditional",
                           pitch_model_path: Optional[Union[str, Path]] = None) -> None:
    """
    Create side-by-side view: original image with detections + top-down field view.
    
    Args:
        input_path: Path to input image
        model_path: Path to YOLO model
        output_path: Path for output (auto-generated if None)
        field_tracker_type: Type of field tracker to use ("traditional" or "ml")
        pitch_model_path: Path to pitch detection model (required if field_tracker_type="ml")
    """
    input_path = Path(input_path)
    model_path = Path(model_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    # Load image and model
    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"Cannot read image: {input_path}")
    
    model = YOLO(str(model_path))
    
    # Initialize pitch detector if needed
    pitch_detector = None
    if field_tracker_type == "ml":
        if pitch_model_path is None:
            raise ValueError("pitch_model_path is required when field_tracker_type='ml'")
        pitch_detector = MLPitchDetector(str(pitch_model_path))
    
    # Run detection
    results = model(img, verbose=False)
    
    # Get annotated image from YOLO
    annotated_img = results[0].plot()
    
    # Generate top-down view with proper FIFA proportions
    if field_tracker_type == "ml" and pitch_detector is not None:
        field_view = _process_with_ml_tracker(results[0], img, DEFAULT_FIELD_WIDTH_PROCESSING, DEFAULT_FIELD_HEIGHT_PROCESSING, pitch_detector)
    else:
        field_view = process_yolo_predictions_to_topview(results[0], img, DEFAULT_FIELD_WIDTH_PROCESSING, DEFAULT_FIELD_HEIGHT_PROCESSING)
    
    # Resize images to same height (use field view height as target)
    field_height = field_view.shape[0]
    img_height, img_width = annotated_img.shape[:2]
    new_width = int(img_width * field_height / img_height)
    annotated_resized = cv2.resize(annotated_img, (new_width, field_height))
    
    # Create side-by-side image
    combined = np.hstack([annotated_resized, field_view])
    
    # Set output path
    if output_path is None:
        suffix = f"{COMBINED_SUFFIX}{input_path.suffix}"
        output_path = input_path.parent / f"{input_path.stem}{suffix}"
    
    # Save result
    cv2.imwrite(str(output_path), combined)
    print(f"✅ Combined view saved to: {output_path}")


if __name__ == "__main__":
    # Example usage with proper FIFA field proportions (105m x 68m)
    input_path = "/Users/alanpehz/Documents/Personal/True Computer Vision/FootballTracker/Complete/test_content/2e57b9_1_9_png.rf.4ddf27c8067f98fd10da07374f376097.jpg"
    model_path = "/Users/alanpehz/Documents/Personal/True Computer Vision/FootballTracker/Complete/models/ball_and_player_model.pt"
    pitch_model_path = "/Users/alanpehz/Documents/Personal/True Computer Vision/FootballTracker/Complete/models/pitch_tracker.pt"
    
    # Process with traditional field tracker
    process_media_with_players(input_path, model_path, field_tracker_type="traditional")
    
    # Process with ML field tracker
    process_media_with_players(input_path, model_path, field_tracker_type="ml", pitch_model_path=pitch_model_path)
    
    # Create side-by-side comparison with ML tracker
    create_side_by_side_view(input_path, model_path, field_tracker_type="ml", pitch_model_path=pitch_model_path)