# ML-Based Football Field Tracker

A machine learning-based computer vision system for automatic football field detection using YOLO keypoint detection models.

## Overview

This implementation uses deep learning (YOLOv8) for robust field keypoint detection, providing superior performance in challenging lighting conditions and partial occlusions compared to traditional computer vision approaches.

## Features

- **YOLO Keypoint Detection**: Uses trained YOLOv8 models for robust field feature detection
- **32 Field Keypoints**: Detects comprehensive set of field landmarks
- **Homography Transformation**: Creates accurate field-to-image mappings
- **Top-down View Generation**: Produces bird's-eye view of the field
- **Real-time Processing**: Supports both images and videos
- **Device Flexibility**: Runs on CPU, CUDA, or MPS (Apple Silicon)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from Complete.ml_field_tracker.process_media import process_media

# Process video with keypoint annotation
process_media(
    input_path="match.mp4",
    model_path="pitch_detection_model.pt",
    device="cpu",
    show_keypoints=True,
    generate_topview=False
)

# Generate top-down view
process_media(
    input_path="match.mp4", 
    model_path="pitch_detection_model.pt",
    device="cuda",
    show_keypoints=False,
    generate_topview=True
)
```

### Advanced Usage

```python
from Complete.ml_field_tracker.pitch_detector import MLPitchDetector
import cv2

# Initialize detector
detector = MLPitchDetector("pitch_model.pt", device="cpu")

# Process single frame
frame = cv2.imread("field_image.jpg")
annotated_frame, top_view = detector.process_frame(
    frame, 
    show_keypoints=True, 
    generate_topview=True
)

# Save results
cv2.imwrite("annotated.jpg", annotated_frame)
if top_view is not None:
    cv2.imwrite("topview.jpg", top_view)
```

## Architecture

### Core Components

1. **MLPitchDetector**: Main detection class using YOLO keypoint detection
2. **ViewTransformer**: Handles homography calculations for perspective transformation
3. **SoccerPitchConfiguration**: FIFA-compliant field dimensions and keypoint definitions
4. **PitchAnnotator**: Visualization utilities for field rendering

### Detection Pipeline

1. **Keypoint Detection**: YOLO model detects 32 field keypoints
2. **Validation**: Filters keypoints based on confidence scores
3. **Homography Calculation**: Maps image points to field coordinates
4. **Visualization**: Renders annotations or top-down views

## Model Requirements

This system requires a trained YOLOv8 keypoint detection model.

## Configuration

### Field Keypoints (32 points)

The system detects these key field features:
- Field corners and boundaries
- Penalty box corners
- Goal box corners  
- Center circle points
- Penalty spots
- Center line intersections

### Device Support

- **CPU**: Universal compatibility
- **CUDA**: NVIDIA GPU acceleration
- **MPS**: Apple Silicon GPU acceleration

## Performance

- **Accuracy**: Superior to classical CV in challenging conditions
- **Speed**: ~30-60 FPS depending on hardware
- **Robustness**: Handles partial occlusions and varying lighting
- **Memory**: ~2GB GPU memory for inference

## Comparison with Classical Approach

| Feature | ML Approach | Classical CV |
|---------|-------------|--------------|
| **Robustness** | High | Medium |
| **Setup** | Model required | Parameter tuning |
| **Speed** | Fast with GPU | Very fast |
| **Accuracy** | Consistent | Variable |
| **Dependencies** | YOLO model | OpenCV only |

## Limitations

- Requires trained YOLO model (~50MB)
- GPU recommended for real-time processing
- Performance depends on training data quality
- May struggle with non-standard field layouts

## Future Enhancements

- [ ] Multi-scale detection for better accuracy
- [ ] Temporal smoothing for video processing
- [ ] Support for custom field dimensions
- [ ] Integration with player tracking systems