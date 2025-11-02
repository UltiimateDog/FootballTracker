"""
Player Tracker Constants

This module contains all constants used across the player tracking modules.
Constants are organized by functionality and include usage information.
"""

import cv2
from typing import Dict, List, Tuple

# =============================================================================
# YOLO CLASS CONFIGURATION
# Used in: player_projection.py
# =============================================================================

# YOLO detection class names in order (class_id -> name mapping)
CLASS_NAMES: List[str] = ['ball', 'goalkeeper', 'player', 'referee']

# BGR color mapping for each detection class (for visualization)
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (255, 0, 0),    # ball - red (most important, stands out)
    1: (0, 255, 0),    # goalkeeper - green
    2: (0, 0, 255),    # player - blue
    3: (255, 255, 0)   # referee - yellow
}

# Point sizes for drawing detections on field view
POINT_SIZES: Dict[int, int] = {
    0: 8,  # ball - larger (easier to spot)
    1: 6,  # goalkeeper
    2: 4,  # player - smaller (many on field)
    3: 5   # referee
}

# =============================================================================
# FIELD VISUALIZATION CONSTANTS
# Used in: player_projection.py
# =============================================================================

# Default dimensions for top-down field view (pixels)
# Maintains FIFA field proportions: 105m x 68m (1.544:1 ratio)
DEFAULT_FIELD_WIDTH: int = 1050  # 10 pixels per meter
DEFAULT_FIELD_HEIGHT: int = 680   # 10 pixels per meter

# Field appearance settings
FIELD_BACKGROUND_COLOR: int = 34           # Dark green background (RGB value)
FIELD_LINE_COLOR: Tuple[int, int, int] = (255, 255, 255)  # White lines
FIELD_LINE_THICKNESS: int = 2              # Line thickness in pixels
FIELD_MARGIN_RATIO: float = 0.1            # 10% margin around field edges

# =============================================================================
# SOCCER FIELD DIMENSIONS (FIFA Standard)
# Used in: player_projection.py for accurate field rendering
# =============================================================================

# Official FIFA field measurements in meters
CENTER_CIRCLE_RADIUS: float = 9.15         # Center circle radius
PENALTY_AREA_WIDTH: float = 40.3           # Penalty area width (goal line)
PENALTY_AREA_HEIGHT: float = 16.5          # Penalty area depth from goal
GOAL_AREA_WIDTH: float = 18.3              # Goal area width (6-yard box)
GOAL_AREA_HEIGHT: float = 5.5              # Goal area depth from goal
PENALTY_SPOT_DISTANCE: float = 11.0        # Distance from goal line to penalty spot
SPOT_RADIUS: int = 3                       # Radius for drawing spots (pixels)

# =============================================================================
# PROJECTION CONSTANTS
# Used in: player_projection.py for 3D to 2D projection calculations
# =============================================================================

# Threshold for detecting rays parallel to field plane
RAY_PARALLEL_THRESHOLD: float = 1e-6       # Prevents division by zero
FIELD_PLANE_Y: float = 0.0                 # Field plane at y=0 in world coordinates

# =============================================================================
# TEXT DISPLAY CONSTANTS
# Used in: player_projection.py for overlay text
# =============================================================================

# Info text settings (frame count, detection stats)
INFO_TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
INFO_TEXT_SCALE: float = 0.6
INFO_TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)  # White
INFO_TEXT_THICKNESS: int = 1
INFO_TEXT_MARGIN: int = 10                 # Pixels from edge

# Error text settings (calibration failures)
ERROR_TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
ERROR_TEXT_SCALE: float = 1.0              # Larger for visibility
ERROR_TEXT_COLOR: Tuple[int, int, int] = (0, 0, 255)    # Red
ERROR_TEXT_THICKNESS: int = 2
ERROR_TEXT_POSITION: Tuple[int, int] = (50, 50)         # Fixed position

# =============================================================================
# MEDIA PROCESSING CONSTANTS
# Used in: process_with_players.py
# =============================================================================

# Default field dimensions for media processing
# Maintains FIFA field proportions: 105m x 68m (1.544:1 ratio)
DEFAULT_FIELD_WIDTH_PROCESSING: int = 1050  # 10 pixels per meter
DEFAULT_FIELD_HEIGHT_PROCESSING: int = 680   # 10 pixels per meter

# File naming suffixes
TOPVIEW_SUFFIX_PLAYERS: str = "_topview"   # For top-down view outputs
COMBINED_SUFFIX: str = "_combined"         # For side-by-side view outputs
TARGET_HEIGHT_COMBINED: int = 600          # Standard height for combined views

# =============================================================================
# ENHANCED VIDEO PROCESSING CONSTANTS
# Used in: enhanced_video_processor.py
# =============================================================================

# Output file suffixes for enhanced processing
ENHANCED_TOPVIEW_SUFFIX: str = "_topview"  # Top-view only output
ENHANCED_COMBINED_SUFFIX: str = "_combined" # Combined view output