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
import supervision as sv

from Complete.field_tracker.Constants import (
    VIDEO_EXTENSIONS, VIDEO_FOURCC, FRAME_PROGRESS_INTERVAL, VIDEO_OUTPUT_EXT
)
from Complete.player_tracker.player_projection import process_yolo_predictions_to_topview
from Complete.ml_field_tracker.pitch_detector import MLPitchDetector
from Complete.player_tracker.team_tracker import TeamClassifier
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
                          pitch_model_path: Optional[Union[str, Path]] = None,
                          use_team_colors: bool = True) -> None:
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
        use_team_colors: If True, uses team classifier to color players by team
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
    
    # Initialize team classifier if enabled
    team_classifier = None
    team_colors = [(0, 255, 255), (255, 0, 255)]  # Yellow and Magenta for teams
    if use_team_colors:
        print("Initializing team classifier...")
        team_classifier = TeamClassifier(device='cpu', batch_size=16)
    
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
    team_classifier_trained = False
    training_crops = []
    
    # Initialize player tracker
    player_tracker = sv.ByteTrack()
    print("Initialized individual player tracking...")
    
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
                # Collect training data for team classifier
                if use_team_colors and team_classifier is not None and not team_classifier_trained:
                    player_crops = _extract_player_crops(results[0], frame)
                    training_crops.extend(player_crops)
                    
                    # Train after collecting enough samples (first 100 frames)
                    if frame_idx == 100 and len(training_crops) > 10:
                        print(f"Training team classifier with {len(training_crops)} player crops...")
                        team_classifier.fit(training_crops)
                        team_classifier_trained = True
                        print("Team classifier training completed.")
                
                # Generate new views with player tracking
                if field_tracker_type == "ml" and pitch_detector is not None:
                    field_view = _process_with_ml_tracker(results[0], frame, field_width, field_height, pitch_detector, team_classifier if team_classifier_trained else None, team_colors, player_tracker)
                else:
                    field_view = _process_yolo_predictions_to_topview_with_teams(
                        results[0], frame, field_width, field_height, team_classifier if team_classifier_trained else None, team_colors, player_tracker
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
                           pitch_detector: MLPitchDetector,
                           team_classifier: Optional[TeamClassifier] = None,
                           team_colors: list = [(0, 255, 255), (255, 0, 255)],
                           player_tracker: Optional[sv.ByteTrack] = None) -> np.ndarray:
    """
    Process YOLO results using ML field tracker for projection.
    
    Args:
        yolo_results: YOLO detection results
        image: Input image
        field_width: Output field width
        field_height: Output field height
        pitch_detector: ML pitch detector instance
        team_classifier: Optional team classifier for player colors
        team_colors: Colors for different teams
        
    Returns:
        Top-down field view with projected players
    """
    # Get field transformation from ML tracker
    _, top_view = pitch_detector.process_frame(image, show_keypoints=False, generate_topview=True)
    
    if top_view is None:
        # Fallback to traditional method if ML tracker fails
        return _process_yolo_predictions_to_topview_with_teams(yolo_results, image, field_width, field_height, team_classifier, team_colors)
    
    # Create field view and project YOLO detections
    from Complete.player_tracker.player_projection import PlayerProjector
    projector = PlayerProjector()
    field_img = projector.create_top_view_field(field_width, field_height)
    
    # Get player crops for team classification
    player_crops = []
    if team_classifier is not None and hasattr(yolo_results, 'boxes') and yolo_results.boxes is not None:
        for box in yolo_results.boxes:
            cls_id = int(box.cls.cpu().numpy())
            if cls_id == 2:  # Only for players
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    player_crops.append(crop)
    
    # Predict team assignments
    team_assignments = []
    if team_classifier is not None and len(player_crops) > 0:
        team_assignments = team_classifier.predict(player_crops)
    
    # Convert YOLO predictions to detection format and draw on field
    if hasattr(yolo_results, 'boxes') and yolo_results.boxes is not None:
        player_idx = 0
        for box in yolo_results.boxes:
            cls_id = int(box.cls.cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy()
            
            # Use center bottom for players, center for ball
            x1, y1, x2, y2 = xyxy
            if cls_id == 0:  # ball
                point_x, point_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            else:  # players
                point_x, point_y = int((x1 + x2) / 2), int(y2)
            
            # Determine color based on class and team
            if cls_id == 2 and team_classifier is not None and player_idx < len(team_assignments):  # player
                color = team_colors[team_assignments[player_idx]]
                player_idx += 1
            else:
                color = projector.class_colors.get(cls_id, (128, 128, 128))
            
            size = projector.point_sizes.get(cls_id, 4)
            
            # For now, use a simple mapping - this should be replaced with proper homography
            field_x = int((point_x / image.shape[1]) * field_width)
            field_y = int((point_y / image.shape[0]) * field_height)
            
            if 0 <= field_x < field_width and 0 <= field_y < field_height:
                cv2.circle(field_img, (field_x, field_y), size, color, -1)
    
    return field_img


def _extract_player_crops(yolo_results: Any, image: np.ndarray) -> list:
    """Extract player crops from YOLO detections for team classification."""
    crops = []
    if hasattr(yolo_results, 'boxes') and yolo_results.boxes is not None:
        for box in yolo_results.boxes:
            cls_id = int(box.cls.cpu().numpy())
            if cls_id == 2:  # Only players
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)
    return crops


def _process_yolo_predictions_to_topview_with_teams(model_predictions: Any, image: np.ndarray, 
                                                   field_width: int, field_height: int,
                                                   team_classifier: Optional[TeamClassifier] = None,
                                                   team_colors: list = [(0, 255, 255), (255, 0, 255)],
                                                   player_tracker: Optional[sv.ByteTrack] = None) -> np.ndarray:
    """Process YOLO predictions with team classification and individual player tracking."""
    from Complete.player_tracker.player_projection import PlayerProjector
    
    projector = PlayerProjector()
    
    # Convert YOLO to supervision format for tracking
    detections_sv = sv.Detections.from_ultralytics(model_predictions)
    
    # Apply player tracking if available
    tracked_detections = detections_sv
    if player_tracker is not None:
        tracked_detections = player_tracker.update_with_detections(detections_sv)
    
    # Get player crops for team classification
    player_crops = []
    player_boxes = []
    if team_classifier is not None:
        for i, (box, cls_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.class_id)):
            if cls_id == 2:  # Only for players
                x1, y1, x2, y2 = box.astype(int)
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    player_crops.append(crop)
                    player_boxes.append(i)
    
    # Predict team assignments
    team_assignments = []
    if team_classifier is not None and len(player_crops) > 0:
        team_assignments = team_classifier.predict(player_crops)
    
    # Convert to detection format with tracking IDs and team colors
    detections = []
    player_idx = 0
    for i, (box, cls_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.class_id)):
        x1, y1, x2, y2 = box
        
        # Convert to YOLO format (normalized center coordinates)
        img_h, img_w = image.shape[:2]
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        
        # Get tracker ID if available
        tracker_id = tracked_detections.tracker_id[i] if tracked_detections.tracker_id is not None else None
        
        # Determine color based on class and team
        if cls_id == 2 and team_classifier is not None and player_idx < len(team_assignments):  # player
            color = team_colors[team_assignments[player_idx]]
            player_idx += 1
        else:
            color = projector.class_colors.get(cls_id, (128, 128, 128))
        
        detections.append([cls_id, x_center, y_center, width, height, color, tracker_id])
    
    return _process_detections_to_topview_with_colors(projector, image, detections, field_width, field_height)


def _process_detections_to_topview_with_colors(projector: 'PlayerProjector', image: np.ndarray, 
                                             detections: list, field_width: int, field_height: int) -> np.ndarray:
    """Process detections with custom colors."""
    from Complete.field_tracker.calibrationRoutines import calibrate_from_image
    from Complete.field_tracker.Constants import DEFAULT_GUESS_FX, DEFAULT_GUESS_ROT, DEFAULT_GUESS_TRANS
    from Complete.player_tracker.Constants import INFO_TEXT_MARGIN, INFO_TEXT_FONT, INFO_TEXT_SCALE, INFO_TEXT_COLOR, INFO_TEXT_THICKNESS, ERROR_TEXT_POSITION, ERROR_TEXT_FONT, ERROR_TEXT_SCALE, ERROR_TEXT_COLOR, ERROR_TEXT_THICKNESS
    
    img_height, img_width = image.shape[:2]
    
    # Calibrate camera
    guess_fx = DEFAULT_GUESS_FX
    guess_rot = np.array(DEFAULT_GUESS_ROT)
    guess_trans = DEFAULT_GUESS_TRANS
    
    K, to_device_from_world, _, _, _ = calibrate_from_image(
        image, guess_fx, guess_rot, guess_trans
    )
    
    # Create field image
    field_img = projector.create_top_view_field(field_width, field_height)
    
    if to_device_from_world is None:
        # Add text indicating calibration failed
        cv2.putText(field_img, "Camera calibration failed", 
                   ERROR_TEXT_POSITION, ERROR_TEXT_FONT, ERROR_TEXT_SCALE, ERROR_TEXT_COLOR, ERROR_TEXT_THICKNESS)
        return field_img
    
    # Process each detection
    projected_count = 0
    for detection in detections:
        if len(detection) == 7:  # With custom color and tracker ID
            class_id, x_center, y_center, width, height, color, tracker_id = detection
        elif len(detection) == 6:  # With custom color
            class_id, x_center, y_center, width, height, color = detection
            tracker_id = None
        else:  # Without custom color
            class_id, x_center, y_center, width, height = detection
            color = projector.class_colors.get(int(class_id), (128, 128, 128))
            tracker_id = None
        
        class_id = int(class_id)
        
        # Get ground contact point of detection
        ground_pixel = projector.yolo_to_ground_point([class_id, x_center, y_center, width, height], img_width, img_height)
        
        # Unproject to field coordinates
        world_point = projector.unproject_to_field(ground_pixel, K, to_device_from_world)
        
        if world_point is not None:
            # Convert to field image coordinates
            field_pixel = projector.world_to_field_image(world_point, field_width, field_height)
            
            # Check if point is within field bounds
            if (0 <= field_pixel[0] < field_width and 0 <= field_pixel[1] < field_height):
                size = projector.point_sizes.get(class_id, 4)
                
                # Draw point with custom color
                cv2.circle(field_img, field_pixel, size, color, -1)
                
                # Draw tracker ID for players if available
                if tracker_id is not None and class_id == 2:  # Only for players
                    cv2.putText(field_img, str(tracker_id), 
                               (field_pixel[0] + 8, field_pixel[1] - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                projected_count += 1
    
    # Add info text
    info_text = f"Projected: {projected_count}/{len(detections)} detections"
    cv2.putText(field_img, info_text, (INFO_TEXT_MARGIN, field_height - 20), 
               INFO_TEXT_FONT, INFO_TEXT_SCALE, INFO_TEXT_COLOR, INFO_TEXT_THICKNESS)
    
    return field_img


if __name__ == "__main__":
    # Example usage
    input_path = "/Users/alanpehz/Documents/Personal/True Computer Vision/FootballTracker/Complete/test_content/demo2.mp4"
    model_path = "/Users/alanpehz/Documents/Personal/True Computer Vision/FootballTracker/Complete/models/ball_and_player_model.pt"
    pitch_model_path = "/Users/alanpehz/Documents/Personal/True Computer Vision/FootballTracker/Complete/models/pitch_tracker.pt"
        
    # Combined view with traditional tracker
    process_video_enhanced(input_path, model_path, combined_view=True)
    
    # Combined view with ML tracker and team colors
    # process_video_enhanced(input_path, model_path, combined_view=True, field_tracker_type="ml", pitch_model_path=pitch_model_path, use_team_colors=True)