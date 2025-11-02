"""
Enhanced Video Processing with Combined Views and Frame Persistence

Provides video processing with options for top-view only or combined view,
and maintains last known positions when detections fail.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union
from ultralytics import YOLO

from Complete.field_tracker.Constants import (
    VIDEO_EXTENSIONS, VIDEO_FOURCC, FRAME_PROGRESS_INTERVAL, VIDEO_OUTPUT_EXT
)
from Complete.player_tracker.player_projection import process_yolo_predictions_to_topview


def process_video_enhanced(input_path: Union[str, Path], 
                          model_path: Union[str, Path],
                          output_path: Optional[Union[str, Path]] = None,
                          combined_view: bool = False,
                          field_width: int = 800,
                          field_height: int = 600) -> None:
    """
    Enhanced video processing with combined view option and frame persistence.
    
    Args:
        input_path: Path to input video
        model_path: Path to YOLO model file
        output_path: Path for output (auto-generated if None)
        combined_view: If True, creates side-by-side original+topview
        field_width: Width of field view
        field_height: Height of field view
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
        suffix = "_combined" if combined_view else "_topview"
        output_path = input_path.parent / f"{input_path.stem}{suffix}{VIDEO_OUTPUT_EXT}"
    
    # Determine output dimensions
    if combined_view:
        # Get first frame to calculate dimensions
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        frame_height, frame_width = first_frame.shape[:2]
        target_height = field_height
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


if __name__ == "__main__":
    # Example usage
    input_path = "/Users/alanpehz/Documents/Personal/True Computer Vision/FootballTracker/Complete/test_content/demo2.mp4"
    model_path = "/Users/alanpehz/Documents/Personal/True Computer Vision/FootballTracker/Complete/models/ball_and_player_model.pt"
        
    # Combined view
    process_video_enhanced(input_path, model_path, combined_view=True)