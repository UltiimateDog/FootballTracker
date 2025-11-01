"""
Football Field Tracker - Constants Configuration

This file contains all constants used across the field tracker modules.
Constants are organized by functionality and include usage information.
"""

import cv2

# =============================================================================
# WORLD COORDINATES - Used in: projectionHelpers.py, calibrationRoutines.py
# =============================================================================
# Points in world coordinate system with origin at center
# x: right, y: foreground, z: top
# Based on official FIFA soccer field dimensions (105m x 68m)

# Center circle points (radius = 9.15m per FIFA regulations)
right_circle_world = [9.15, 0, 0]
left_circle_world = [-9.15, 0, 0]
behind_circle_world = [0, 0, 9.15]
front_circle_world = [0, 0, -9.15]

# Middle line coordinates (field length = 68m, so ±34m from center)
front_middle_line_world = [0, 0, -34]
back_middle_line_world = [0, 0, 34]

# Field corners (field width = 105m, so ±52.5m from center)
corner_back_left_world = [-52.5, 0, 34]
corner_front_left_world = [-52.5, 0, -34]
corner_back_right_world = [52.5, 0, 34]
corner_front_right_world = [52.5, 0, -34]

# Penalty area coordinates (16.5m from goal line, 40.32m width)
PENALTY_LEFT_FRONT_GOAL_WORLD = [-52.5, 0, -20.16]
PENALTY_LEFT_FRONT_FIELD_WORLD = [-36, 0, -20.16]  # 16.5m from goal
PENALTY_LEFT_BACK_FIELD_WORLD = [-36, 0, 20.16]
PENALTY_LEFT_BACK_GOAL_WORLD = [-52.5, 0, 20.16]

PENALTY_RIGHT_FRONT_GOAL_WORLD = [52.5, 0, -20.16]
PENALTY_RIGHT_FRONT_FIELD_WORLD = [36, 0, -20.16]
PENALTY_RIGHT_BACK_FIELD_WORLD = [36, 0, 20.16]
PENALTY_RIGHT_BACK_GOAL_WORLD = [52.5, 0, 20.16]

DIST_TO_CENTER = 77.0  # Typical camera distance to field center

# =============================================================================
# CAMERA CALIBRATION - Used in: processMedia.py, calibrationRoutines.py
# =============================================================================

# Initial camera parameter estimates for PnP algorithm
DEFAULT_GUESS_FX = 2000  # Focal length guess (typical for broadcast cameras)
DEFAULT_GUESS_ROT = [[0.25, 0, 0]]  # Small rotation around x-axis
DEFAULT_GUESS_TRANS = (0, 0, 80)  # Camera height ~80m above field

# PnP validation thresholds
MIN_DISTANCE_TO_CENTER = 40.0  # Minimum realistic camera distance (meters)
MAX_DISTANCE_TO_CENTER = 100.0  # Maximum realistic camera distance (meters)
MIN_PNP_POINTS = 4  # Minimum points required for PnP algorithm

# =============================================================================
# VIDEO PROCESSING - Used in: processMedia.py
# =============================================================================

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]  # Supported formats
VIDEO_FOURCC = "mp4v"  # Video codec for output
FRAME_PROGRESS_INTERVAL = 30  # Print progress every N frames

# Output file naming
TOPVIEW_SUFFIX = "_topview"
ANNOTATED_SUFFIX = "_annotated"
VIDEO_OUTPUT_EXT = ".mp4"

# =============================================================================
# DISPLAY & VISUALIZATION - Used in: calibrationRoutines.py, projectionHelpers.py
# =============================================================================

# Text overlay settings
TEXT_POSITION = (1280, 120)  # Position for camera info text
TEXT_FONT = cv2.FONT_HERSHEY_COMPLEX
TEXT_SCALE = 1
TEXT_COLOR = (0, 255, 0)  # Green color (BGR format)
TEXT_THICKNESS = 2

# Line drawing settings
LINE_COLOR = (0, 165, 255)  # Orange color for field lines (BGR format)
LINE_THICKNESS = 3

# Top-down view settings
SOCCER_FIELD_WIDTH = 105  # FIFA standard field width (meters)
SOCCER_FIELD_HEIGHT = 68  # FIFA standard field height (meters)
TOP_VIEW_PIXELS_WIDTH = 1668  # Pixel width for top view rendering

# Center circle rendering
CIRCLE_RADIUS = 9.15  # FIFA regulation center circle radius (meters)
CIRCLE_RESOLUTION = 25  # Number of points to approximate circle

# =============================================================================
# EDGE DETECTION - Used in: shapeDetection.py
# =============================================================================

# Canny edge detection parameters
CANNY_LOW_THRESHOLD = 50  # Lower threshold for edge detection
CANNY_HIGH_THRESHOLD = 200  # Upper threshold for edge detection
CANNY_APERTURE_SIZE = 3  # Sobel kernel size

# Special settings for circle detection (more sensitive)
CANNY_CIRCLE_LOW_THRESHOLD = 20
CANNY_CIRCLE_HIGH_THRESHOLD = 100

# =============================================================================
# HOUGH LINE DETECTION - Used in: shapeDetection.py
# =============================================================================

# Basic Hough parameters
HOUGH_RHO = 1  # Distance resolution in pixels

# Back/front line detection (horizontal lines ~90°)
HOUGH_THETA_DIVISOR = 4  # Higher precision for horizontal lines
HOUGH_THRESHOLD = 500  # High threshold for strong lines
HOUGH_MIN_THETA_BACK_FRONT = 80  # Nearly horizontal lines
HOUGH_MAX_THETA_BACK_FRONT = 100

# Main line detection (vertical lines)
HOUGH_THETA_DIVISOR_MAIN = 2
HOUGH_THRESHOLD_MAIN = 200
HOUGH_THRESHOLD_MAIN_OTHER = 250  # Higher threshold for alternative search
HOUGH_MAX_THETA_MAIN = 40  # Vertical-ish lines
HOUGH_MIN_THETA_MAIN_OTHER = 130

# Goal line detection
HOUGH_THRESHOLD_GOAL = 60  # Lower threshold for goal lines
HOUGH_THRESHOLD_GOAL_SECOND = 200  # Second pass with higher threshold
HOUGH_MAX_THETA_LEFT_GOAL = 80
HOUGH_MIN_THETA_RIGHT_GOAL = 100

# =============================================================================
# MORPHOLOGICAL OPERATIONS - Used in: shapeDetection.py
# =============================================================================

# Kernel sizes for image processing
DILATE_KERNEL_SIZE = (7, 7)  # Initial dilation for bold contours
FINAL_DILATE_KERNEL_SIZE = (15, 15)  # Final dilation for circle mask
FINAL_ERODE_KERNEL_SIZE = (10, 10)  # Erosion to clean up mask

# =============================================================================
# CIRCLE DETECTION - Used in: shapeDetection.py
# =============================================================================

# Flood fill parameters for center circle detection
CIRCLE_SEEDS = [-150, -100, -50, 50, 100, 150]  # Horizontal offsets for seed points
CIRCLE_FILL_VALUE = 128  # Gray value for flood fill
CIRCLE_RANGE_MIN = 127  # Range for extracting filled area
CIRCLE_RANGE_MAX = 129

# Circle center estimation weights
CIRCLE_CENTER_WEIGHT_FRONT = 0.3  # Weight for front middle point
CIRCLE_CENTER_WEIGHT_BACK = 0.7  # Weight for back middle point (more reliable)

# Circle validation parameters
CIRCLE_BOUNDARY_OFFSET = 10  # Pixel tolerance for circle boundaries
CIRCLE_SIZE_RATIO = 2  # Minimum width/height ratio for valid circle

# =============================================================================
# GOAL LINE DETECTION - Used in: shapeDetection.py
# =============================================================================

GOAL_LINE_OFFSET = 30  # Offset for parallel line to back line
BLACK_PIXEL = [0, 0, 0]  # RGB values for masking pixels

# Field division for goal line detection
WIDTH_FRACTION_LEFT = 2 / 5  # Left 40% of field
WIDTH_FRACTION_RIGHT = 3 / 5  # Right 60% of field

# Line matching tolerances
THETA_TOLERANCE = 0.03  # Angular tolerance for line matching (radians)
MIN_DISTANCE_THRESHOLD = 1000  # Initial distance threshold for line matching