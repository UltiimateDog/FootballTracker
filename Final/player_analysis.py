"""
Player Analysis Module

Analyzes video to extract player crops, speeds, and create summary image.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union, Any, Dict, List, Tuple
from ultralytics import YOLO
import supervision as sv
import math

from Final.players.team_tracker import TeamClassifier
from Final.field.calibrationRoutines import calibrate_from_image
from Final.Constants import DEFAULT_GUESS_FX, DEFAULT_GUESS_ROT, DEFAULT_GUESS_TRANS


def analyze_players_from_video(input_path: Union[str, Path],
                              model_path: Union[str, Path],
                              output_path: Optional[Union[str, Path]] = None) -> None:
    """
    Analyze video to extract player and ball information and create summary image.

    Args:
        input_path: Path to input video
        model_path: Path to YOLO model file
        output_path: Path for output image (auto-generated if None)
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

    # Initialize team classifier
    print("Initializing team classifier...")
    team_classifier = TeamClassifier(device='cpu', batch_size=16)
    # Disable progress bars for cleaner output
    import os
    os.environ['TQDM_DISABLE'] = '1'

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Analyzing video: {input_path.name}")
    print(f"Total frames: {total_frames}, FPS: {fps}")

    # Initialize tracking
    player_tracker = sv.ByteTrack()
    player_data = {}  # {player_id: {'crops': [], 'positions': [], 'teams': []}}
    ball_data = {'positions': []}  # Ball tracking data
    team_classifier_trained = False
    training_crops = []

    # Calibration persistence
    last_calibration = {'K': None, 'pose': None, 'frame_count': 0}

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            has_detections = len(results[0].boxes) > 0 if results[0].boxes is not None else False

            if has_detections:
                # Train team classifier
                if not team_classifier_trained:
                    player_crops = _extract_player_crops(results[0], frame)
                    training_crops.extend(player_crops)

                    if frame_idx == 100 and len(training_crops) > 10:
                        print("Training team classifier...")
                        team_classifier.fit(training_crops)
                        team_classifier_trained = True

                # Process detections
                _process_frame_for_analysis(results[0], frame, player_tracker, player_data,
                                          team_classifier if team_classifier_trained else None, fps, ball_data, last_calibration)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")

    finally:
        cap.release()

    # Calculate speeds and create summary
    _calculate_player_speeds(player_data, fps)
    _calculate_ball_speed(ball_data, fps)

    # Set output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_player_ball_analysis.jpg"

    # Create summary image
    _create_player_summary_image(player_data, ball_data, output_path)

    print(f"✅ Player analysis saved to: {output_path}")


def _extract_player_crops(yolo_results: Any, image: np.ndarray) -> List[np.ndarray]:
    """Extract player crops from YOLO detections."""
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


def _process_frame_for_analysis(yolo_results: Any, frame: np.ndarray,
                               player_tracker: sv.ByteTrack, player_data: Dict,
                               team_classifier: Optional[TeamClassifier], fps: int, ball_data: Dict, last_calibration: Dict) -> None:
    """Process single frame to extract player and ball data."""
    # Try to calibrate camera for this frame
    K, to_device_from_world = _calibrate_frame(frame)

    if to_device_from_world is not None:
        # Successful calibration - update cache
        last_calibration['K'] = K
        last_calibration['pose'] = to_device_from_world
        last_calibration['frame_count'] = 0
    else:
        # Calibration failed - try to use cached calibration
        if last_calibration['K'] is not None and last_calibration['frame_count'] < 30:
            K = last_calibration['K']
            to_device_from_world = last_calibration['pose']
            last_calibration['frame_count'] += 1
        else:
            return  # No valid calibration available, skip frame

    # Convert to supervision format
    detections_sv = sv.Detections.from_ultralytics(yolo_results)

    # Process ball detections (class 0)
    ball_mask = detections_sv.class_id == 0
    if np.any(ball_mask):
        ball_boxes = detections_sv.xyxy[ball_mask]
        if len(ball_boxes) > 0:
            # Use first ball detection
            x1, y1, x2, y2 = ball_boxes[0]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            ball_pos = _unproject_to_field((center_x, center_y), K, to_device_from_world)
            if ball_pos is not None:
                ball_data['positions'].append(ball_pos)

    # Filter only players
    player_mask = detections_sv.class_id == 2
    if not np.any(player_mask):
        return

    player_detections = sv.Detections(
        xyxy=detections_sv.xyxy[player_mask],
        class_id=detections_sv.class_id[player_mask],
        confidence=detections_sv.confidence[player_mask] if detections_sv.confidence is not None else None
    )

    # Apply tracking
    tracked_detections = player_tracker.update_with_detections(player_detections)

    if tracked_detections.tracker_id is None:
        return

    # Extract crops and positions
    for i, (box, tracker_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
        x1, y1, x2, y2 = box.astype(int)

        # Extract crop
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Calculate real field position using calibration
        center_x = (x1 + x2) / 2
        bottom_y = y2  # Use bottom of bbox for players (feet position)
        field_pos = _unproject_to_field((center_x, bottom_y), K, to_device_from_world)
        if field_pos is None:
            continue  # Skip if unprojection fails

        # Initialize player data if new
        if tracker_id not in player_data:
            player_data[tracker_id] = {
                'crops': [],
                'positions': [],
                'teams': [],
                'best_crop': None,
                'best_crop_size': 0
            }

        # Store crop (keep best quality one)
        crop_size = crop.shape[0] * crop.shape[1]
        if crop_size > player_data[tracker_id]['best_crop_size']:
            player_data[tracker_id]['best_crop'] = crop.copy()
            player_data[tracker_id]['best_crop_size'] = crop_size

        # Store position
        player_data[tracker_id]['positions'].append(field_pos)

        # Get team assignment
        if team_classifier is not None:
            team = team_classifier.predict([crop])[0]
            player_data[tracker_id]['teams'].append(team)



def _calibrate_frame(frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Calibrate camera for the given frame."""
    try:
        guess_fx = DEFAULT_GUESS_FX
        guess_rot = np.array(DEFAULT_GUESS_ROT)
        guess_trans = DEFAULT_GUESS_TRANS

        K, to_device_from_world, _, _, _ = calibrate_from_image(
            frame, guess_fx, guess_rot, guess_trans
        )
        return K, to_device_from_world
    except Exception:
        return None, None


def _unproject_to_field(pixel_point: Tuple[float, float], K: np.ndarray,
                       to_device_from_world: np.ndarray) -> Optional[Tuple[float, float]]:
    """Unproject pixel point to real field coordinates."""
    try:
        from Complete.player_tracker.Constants import RAY_PARALLEL_THRESHOLD, FIELD_PLANE_Y

        x_pixel, y_pixel = pixel_point

        # Convert pixel to normalized camera coordinates
        x_norm = (x_pixel - K[0, 2]) / K[0, 0]
        y_norm = (y_pixel - K[1, 2]) / K[1, 1]

        # Camera ray direction in camera coordinates
        ray_camera = np.array([x_norm, y_norm, 1.0])

        # Transform ray to world coordinates
        to_world_from_device = np.linalg.inv(to_device_from_world)
        ray_world = to_world_from_device[:3, :3] @ ray_camera
        camera_pos_world = to_world_from_device[:3, 3]

        # Intersect ray with field plane (y = 0)
        if abs(ray_world[1]) < RAY_PARALLEL_THRESHOLD:
            return None  # Ray parallel to field

        t = -camera_pos_world[1] / ray_world[1]
        if t < 0:
            return None  # Intersection behind camera

        # Calculate intersection point
        intersection = camera_pos_world + t * ray_world
        return (intersection[0], intersection[2])  # Return (x, z) coordinates

    except Exception:
        return None


def _calculate_player_speeds(player_data: Dict, fps: int) -> None:
    """Calculate average speed for each player."""
    for player_id, data in player_data.items():
        positions = data['positions']
        if len(positions) < 2:
            data['avg_speed'] = 0.0
            continue

        total_distance = 0.0
        for i in range(1, len(positions)):
            x1, y1 = positions[i-1]
            x2, y2 = positions[i]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance += distance

        # Calculate speed in km/h
        time_seconds = len(positions) / fps
        speed_ms = total_distance / time_seconds if time_seconds > 0 else 0
        speed_kmh = speed_ms * 3.6
        data['avg_speed'] = speed_kmh

        # Get most frequent team
        if data['teams']:
            data['team'] = max(set(data['teams']), key=data['teams'].count)
        else:
            data['team'] = 0


def _calculate_ball_speed(ball_data: Dict, fps: int) -> None:
    """Calculate ball average speed."""
    positions = ball_data['positions']
    if len(positions) < 2:
        ball_data['avg_speed'] = 0.0
        return

    total_distance = 0.0
    for i in range(1, len(positions)):
        x1, y1 = positions[i-1]
        x2, y2 = positions[i]
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_distance += distance

    time_seconds = len(positions) / fps
    speed_ms = total_distance / time_seconds if time_seconds > 0 else 0
    ball_data['avg_speed'] = speed_ms * 3.6  # km/h


def _create_player_summary_image(player_data: Dict, ball_data: Dict, output_path: Path) -> None:
    """Create summary image with player crops and ball statistics."""
    if not player_data and not ball_data['positions']:
        print("No players or ball detected!")
        return

    # Sort players by ID and add ball if detected
    sorted_players = sorted(player_data.items())
    if ball_data['positions']:
        sorted_players.append(('BALL', ball_data))

    # Calculate grid layout
    num_items = len(sorted_players)
    cols = min(4, num_items)
    rows = (num_items + cols - 1) // cols

    # Image dimensions
    crop_size = 150
    text_height = 80
    cell_width = crop_size + 20
    cell_height = crop_size + text_height + 20

    img_width = cols * cell_width + 40
    img_height = rows * cell_height + 60

    # Create summary image
    summary_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 240

    # Title
    cv2.putText(summary_img, "Player & Ball Analysis", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # Team colors
    team_colors = [(0, 255, 255), (255, 0, 255)]  # Yellow, Magenta

    for idx, (item_id, data) in enumerate(sorted_players):
        row = idx // cols
        col = idx % cols

        x_offset = col * cell_width + 20
        y_offset = row * cell_height + 60

        # Draw player crop or ball placeholder
        if item_id == 'BALL':
            # Create ball placeholder
            ball_img = np.ones((crop_size, crop_size, 3), dtype=np.uint8) * 50
            cv2.circle(ball_img, (crop_size//2, crop_size//2), crop_size//3, (0, 255, 0), -1)
            cv2.putText(ball_img, "BALL", (crop_size//2-25, crop_size//2+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            summary_img[y_offset:y_offset + crop_size,
                       x_offset:x_offset + crop_size] = ball_img
            border_color = (0, 255, 0)  # Green for ball
        elif data.get('best_crop') is not None:
            crop = data['best_crop']
            crop_resized = cv2.resize(crop, (crop_size, crop_size))
            team = data.get('team', 0)
            border_color = team_colors[team] if team < len(team_colors) else (128, 128, 128)
            summary_img[y_offset:y_offset + crop_size,
                       x_offset:x_offset + crop_size] = crop_resized
        else:
            border_color = (128, 128, 128)

        # Add border
        cv2.rectangle(summary_img,
                     (x_offset - 2, y_offset - 2),
                     (x_offset + crop_size + 2, y_offset + crop_size + 2),
                     border_color, 4)

        # Add text information
        text_y = y_offset + crop_size + 20

        # ID/Name
        if item_id == 'BALL':
            cv2.putText(summary_img, "Ball",
                       (x_offset, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            cv2.putText(summary_img, f"Player {item_id}",
                       (x_offset, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Team (only for players)
        if item_id != 'BALL':
            team_name = "Team A" if data.get('team', 0) == 0 else "Team B"
            cv2.putText(summary_img, team_name,
                       (x_offset, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Speed
        speed = data.get('avg_speed', 0.0)
        cv2.putText(summary_img, f"{speed:.1f} km/h",
                   (x_offset, text_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Detections count
        detections = len(data['positions'])
        cv2.putText(summary_img, f"{detections} detections",
                   (x_offset, text_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    # Save image
    cv2.imwrite(str(output_path), summary_img)


if __name__ == "__main__":
    # Example usage
    input_path = "/Users/alanpehz/Documents/Personal/True Computer Vision/FootballTracker/Final/test_content/demo2.mp4"
    model_path = "/Users/alanpehz/Documents/Personal/True Computer Vision/FootballTracker/Final/models/ball_and_player_model.pt"
    
    analyze_players_from_video(input_path, model_path)