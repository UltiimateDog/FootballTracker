# Player Tracker Module

This module extends the field tracker functionality by integrating YOLO object detection to project players, balls, goalkeepers, and referees onto a top-down field view.

## Overview

The player tracker combines:
- **Field Detection**: Uses the field_tracker module for camera calibration
- **YOLO Detection**: Detects players, balls, goalkeepers, and referees
- **3D Projection**: Projects 2D detections to 3D world coordinates
- **Top-Down Visualization**: Creates tactical field view with player positions

## Features

- **Multi-Class Detection**: Supports ball, goalkeeper, player, and referee detection
- **Accurate Projection**: Uses camera calibration for precise world coordinate mapping
- **Real-Time Processing**: Supports both images and videos
- **Tactical View**: Generates clean top-down field visualization
- **Side-by-Side Comparison**: Shows original detections alongside tactical view

## Architecture

### Core Components

#### `player_projection.py` - Projection Engine
- **PlayerProjector Class**: Main projection functionality
- **Key Methods**:
  - `unproject_to_field()`: Projects 2D pixels to 3D field coordinates
  - `process_detections_to_topview()`: Creates top-down view with players
  - `create_top_view_field()`: Generates clean field visualization
- **Features**:
  - Handles camera calibration integration
  - Validates projections within field bounds
  - Color-coded visualization by object class

#### `process_with_players.py` - Processing Pipeline
- **Main Functions**:
  - `process_media_with_players()`: Process images/videos with player tracking
  - `create_side_by_side_view()`: Generate comparison views
- **Features**:
  - YOLO model integration
  - Batch video processing
  - Flexible output options

## Usage

### Basic Usage

```python
from Complete.player_tracker.process_with_players import process_media_with_players

# Process image with player tracking
process_media_with_players(
    input_path="match_image.jpg",
    model_path="ball_and_player_model.pt",
    output_path="tactical_view.jpg"
)

# Process video
process_media_with_players(
    input_path="match_video.mp4", 
    model_path="ball_and_player_model.pt"
)
```

### Advanced Usage

```python
from Complete.player_tracker.player_projection import PlayerProjector
from ultralytics import YOLO
import cv2

# Load components
projector = PlayerProjector()
model = YOLO("ball_and_player_model.pt")
img = cv2.imread("match.jpg")

# Run detection
results = model(img)

# Create tactical view
field_view = projector.process_detections_to_topview(
    image=img,
    detections=results[0],
    field_width=1000,
    field_height=800
)

cv2.imwrite("tactical_view.jpg", field_view)
```

### Side-by-Side Comparison

```python
from Complete.player_tracker.process_with_players import create_side_by_side_view

create_side_by_side_view(
    input_path="match.jpg",
    model_path="model.pt",
    output_path="comparison.jpg"
)
```

## Technical Details

### Projection Pipeline

1. **YOLO Detection**: Detect objects in image coordinates
2. **Camera Calibration**: Estimate camera pose using field features
3. **Ray Casting**: Cast rays from camera through detected points
4. **Field Intersection**: Find intersection with field plane (y=0)
5. **Coordinate Mapping**: Map world coordinates to tactical view
6. **Visualization**: Draw objects on clean field representation

### Coordinate Systems

- **Image Coordinates**: Pixel coordinates (0,0) at top-left
- **World Coordinates**: Field-centered, X=right, Y=up, Z=forward
- **Field Coordinates**: Top-down view, origin at field center

### Object Classes

| Class ID | Name       | Color  | Size | Description |
|----------|------------|--------|------|-------------|
| 0        | Ball       | Red    | 8px  | Football    |
| 1        | Goalkeeper | Green  | 6px  | Goalkeepers |
| 2        | Player     | Blue   | 4px  | Field players |
| 3        | Referee    | Yellow | 5px  | Match officials |

### Field Visualization

- **Field Dimensions**: FIFA standard (105m × 68m)
- **Center Circle**: 9.15m radius
- **Clean Design**: White lines on dark green background
- **Margins**: 10% border for better visualization

## Configuration

### Customizable Parameters

```python
# Field view dimensions
field_width = 800    # Output image width
field_height = 600   # Output image height

# Object visualization
class_colors = {
    0: (255, 0, 0),    # Ball - red
    1: (0, 255, 0),    # Goalkeeper - green
    2: (0, 0, 255),    # Player - blue
    3: (255, 255, 0)   # Referee - yellow
}

point_sizes = {
    0: 8,  # Ball
    1: 6,  # Goalkeeper  
    2: 4,  # Player
    3: 5   # Referee
}
```

### YOLO Model Requirements

The YOLO model should be trained with:
- **Classes**: ball, goalkeeper, player, referee (in that order)
- **Format**: YOLOv8 (.pt file)
- **Input**: Standard image formats (jpg, png, etc.)

## Performance

- **Image Processing**: ~2-3 seconds per image
- **Video Processing**: ~1-2x real-time (depends on resolution)
- **Accuracy**: Depends on field visibility and YOLO model quality
- **Memory Usage**: Minimal (frame-by-frame processing)

## Limitations

- Requires visible field lines for camera calibration
- Projection accuracy depends on field detection quality
- Objects must be on the field plane (y=0) for accurate projection
- Performance varies with lighting and image quality

## Error Handling

- **Calibration Failure**: Shows error message on field view
- **Invalid Projections**: Filters out points outside field bounds
- **Model Loading**: Clear error messages for missing models
- **File Handling**: Robust path validation and error reporting

## Integration with Field Tracker

This module seamlessly integrates with the field_tracker package:

```python
# Uses field_tracker components
from Complete.field_tracker.calibrationRoutines import calibrate_from_image
from Complete.field_tracker.Constants import SOCCER_FIELD_WIDTH, SOCCER_FIELD_HEIGHT
```

## Future Enhancements

- **Player Tracking**: Track individual players across frames
- **Team Classification**: Distinguish between team colors
- **Formation Analysis**: Detect tactical formations
- **Heat Maps**: Generate player movement heat maps
- **Statistics**: Calculate distances, speeds, and positions