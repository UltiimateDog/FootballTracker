# Football Field Tracker

A computer vision package for automatic football field detection, camera pose estimation, and field line projection from broadcast footage.

## Overview

This package implements a complete pipeline for analyzing football match footage by:
- Detecting key field features (lines, circles, corners)
- Estimating camera pose using Perspective-n-Point (PnP) algorithm
- Projecting 3D field coordinates onto 2D image space
- Generating annotated output with field overlays
- Creating top-down (bird's-eye) view transformations

## Features

- **Automatic Field Detection**: Detects field boundaries, center circle, penalty areas, and goal lines
- **Camera Calibration**: Estimates camera position, orientation, and focal length
- **Real-time Processing**: Supports both images and videos
- **Field Overlay**: Projects accurate field lines onto the original footage
- **Top-down View**: Generates bird's-eye perspective of the field
- **Robust Detection**: Handles various camera angles and lighting conditions

## Architecture

### Core Modules

#### `processMedia.py` - Main Processing Pipeline
- **Purpose**: Entry point for processing images and videos
- **Key Functions**:
  - `process_media()`: Main processing function supporting both images and videos
- **Features**:
  - Automatic file type detection
  - Progress tracking for video processing
  - Flexible output naming
  - Support for both annotated and top-view outputs

#### `shapeDetection.py` - Field Feature Detection
- **Purpose**: Computer vision algorithms for detecting field elements
- **Key Functions**:
  - `find_back_front_lines()`: Detects horizontal field boundaries
  - `find_main_line()`: Detects center line (vertical)
  - `find_central_circle()`: Locates center circle using flood fill
  - `find_goal_line()`: Detects goal area boundaries
  - `find_key_points()`: Orchestrates all detection algorithms
- **Algorithms**:
  - Canny edge detection
  - Hough line transform
  - Flood fill for circle detection
  - Morphological operations for noise reduction

#### `calibrationRoutines.py` - Camera Pose Estimation
- **Purpose**: Estimates camera parameters using detected field features
- **Key Functions**:
  - `calibrate_from_image()`: Main calibration pipeline
  - `find_extrinsic_intrinsic_matrices()`: PnP solver implementation
  - `extend_key_points_set()`: Enhances point set for better accuracy
  - `display_yaw_and_focal_length()`: Overlays camera info on image
  - `display_top_view()`: Generates bird's-eye view transformation
- **Algorithms**:
  - Perspective-n-Point (PnP) algorithm
  - Camera matrix estimation
  - Iterative refinement with extended point sets

#### `projectionHelpers.py` - 3D to 2D Projection
- **Purpose**: Projects 3D world coordinates to 2D image coordinates
- **Key Functions**:
  - `project_to_screen()`: Core projection function
  - `draw_pitch_lines()`: Renders complete field overlay
  - `draw_central_circle()`: Renders center circle
  - `draw_penalty_areas()`: Renders penalty boxes
- **Features**:
  - Accurate FIFA-compliant field dimensions
  - Customizable line colors and thickness
  - Efficient batch projection

### Data Structures

#### `KeyPoints.py` - Field Point Management
- **Purpose**: Manages detected field feature points
- **Key Features**:
  - Stores 2D pixel coordinates of field features
  - Maps 2D points to 3D world coordinates
  - Computes focal length from center circle
  - Visualization methods for debugging
- **Detected Points**:
  - Center circle points (4 cardinal directions)
  - Field corners (4 corners)
  - Middle line intersections
  - Goal line intersections

#### `KeyLines.py` - Field Line Management
- **Purpose**: Manages detected field lines
- **Key Features**:
  - Stores lines in polar coordinates (rho, theta)
  - Visualization methods for debugging
- **Detected Lines**:
  - Front and back field boundaries
  - Center line (main line)
  - Left and right goal lines

#### `helpers.py` - Utility Functions
- **Purpose**: Common utility functions for visualization and geometry
- **Key Functions**:
  - `draw_point()`: Visualizes key points
  - `draw_line()`: Visualizes lines in polar coordinates
  - `intersect()`: Computes intersection of two lines
- **Features**:
  - Color-coded visualization
  - Robust geometric calculations

#### `Constants.py` - Configuration Management
- **Purpose**: Centralized configuration for all parameters
- **Categories**:
  - **World Coordinates**: FIFA-compliant field dimensions
  - **Camera Calibration**: PnP algorithm parameters
  - **Video Processing**: Codec and format settings
  - **Edge Detection**: Canny thresholds and parameters
  - **Hough Detection**: Line detection parameters
  - **Circle Detection**: Flood fill and validation parameters
  - **Display Settings**: Colors, fonts, and overlay parameters

## Usage

### Basic Usage

```python
from Complete.field_tracker.processMedia import process_media

# Process an image
process_media("input.jpg", "output_annotated.jpg", top_view=False)

# Generate top-down view
process_media("input.jpg", "output_topview.jpg", top_view=True)

# Process a video
process_media("match.mp4", "match_annotated.mp4", top_view=False)
```

### Advanced Usage

```python
from Complete.field_tracker.shapeDetection import find_key_points
from Complete.field_tracker.calibrationRoutines import calibrate_from_image
from Complete.field_tracker.projectionHelpers import draw_pitch_lines
import cv2
import numpy as np

# Load image
img = cv2.imread("football_field.jpg")

# Detect field features
key_points, key_lines = find_key_points(img)

# Estimate camera pose
guess_fx = 2000
guess_rot = np.array([[0.25, 0, 0]])
guess_trans = (0, 0, 80)

K, pose, rot, trans, img = calibrate_from_image(
    img, guess_fx, guess_rot, guess_trans
)

# Draw field overlay
if pose is not None:
    img = draw_pitch_lines(K, pose, img)

cv2.imwrite("result.jpg", img)
```

## Algorithm Details

### Field Detection Pipeline

1. **Edge Detection**: Apply Canny edge detection to identify field boundaries
2. **Line Detection**: Use Hough transform to detect straight lines
3. **Line Classification**: Classify lines as horizontal (field boundaries) or vertical (center line)
4. **Circle Detection**: Use flood fill to identify center circle
5. **Goal Line Detection**: Detect penalty area boundaries using constrained search

### Camera Calibration Process

1. **Feature Matching**: Associate 2D image points with 3D world coordinates
2. **Initial Estimation**: Use detected center circle to estimate focal length
3. **PnP Solving**: Apply OpenCV's solvePnP for camera pose estimation
4. **Validation**: Check pose validity using distance and angle constraints
5. **Refinement**: Extend point set with projected corners for improved accuracy
6. **Iteration**: Re-run PnP with extended point set for final pose

### Coordinate Systems

- **World Coordinates**: Origin at field center, X-axis right, Y-axis forward, Z-axis up
- **Camera Coordinates**: Standard computer vision convention
- **Image Coordinates**: Pixel coordinates with origin at top-left

## Configuration

### Key Parameters

All parameters are centralized in `Constants.py`:

- **Field Dimensions**: Based on FIFA regulations (105m × 68m)
- **Camera Parameters**: Typical broadcast camera settings
- **Detection Thresholds**: Tuned for football field imagery
- **Visualization Settings**: Colors and line thickness

### Customization

To adapt for different scenarios:

1. **Field Dimensions**: Modify world coordinate constants for non-standard fields
2. **Camera Settings**: Adjust initial guess parameters for different camera types
3. **Detection Sensitivity**: Tune Canny and Hough parameters for different image quality
4. **Validation Thresholds**: Modify distance and angle limits for different camera positions

## Dependencies

- **OpenCV**: Computer vision algorithms
- **NumPy**: Numerical computations
- **pathlib**: File path handling

## Performance

- **Image Processing**: ~1-2 seconds per image
- **Video Processing**: Real-time capable (depends on resolution)
- **Memory Usage**: Minimal (processes frame by frame)
- **Accuracy**: Sub-pixel precision for well-lit, high-contrast fields

## Limitations

- Requires visible field lines and center circle
- Performance degrades with poor lighting or low contrast
- Assumes standard FIFA field dimensions
- Works best with broadcast-quality footage

## Future Enhancements

- Support for non-standard field dimensions
- Improved robustness in challenging lighting conditions
- Real-time processing optimizations
- Support for additional field features (corner arcs, goal posts)
- Machine learning-based feature detection