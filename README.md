# Football Tracker

## Getting Started

**Start Here:** For anyone reviewing this project, please begin by reading the main analysis notebook:
`ball_and_player_trackers/football_tracker_analysis.ipynb`

This notebook contains the core implementation and analysis of the computer vision system for automatic ball detection and speed estimation in football matches.

**Next:** After reviewing the main analysis, continue with:
`ball_and_player_trackers/yolo_training.ipynb`

This notebook demonstrates the YOLOv8 model training process for football ball and player detection, including dataset preparation, training configuration, and model evaluation.

**Then:** For advanced model fine-tuning, proceed to:
`ball_and_player_trackers/yolo_retraining.ipynb`

This notebook shows YOLOv8 fine-tuning for improved football ball and player detection, featuring class balance analysis, data augmentation strategies, and comprehensive model evaluation with visualization of results.

**Finally:** Complete your review with:
`ball_and_player_trackers/Final_testing.ipynb`

This notebook demonstrates final model evaluation and testing on validation and test datasets, providing comprehensive performance metrics and detection results visualization for the trained football detection model.

## Project Overview

This project implements a computer vision system for football match analysis with the following key features:

- **Automatic Ball Detection**: Uses YOLO format object detection to identify footballs in match footage
- **Player Tracking**: Detects and tracks players, goalkeepers, and referees
- **Speed Estimation**: Calculates ball speed and movement patterns
- **Data Visualization**: Provides comprehensive analysis and visualization of detection results

### Dataset Information
- **Classes**: ball (0), goalkeeper (1), player (2), referee (3)
- **Format**: YOLO format annotations (class x_center y_center width height)
- **Primary Focus**: Ball detection and player tracking (goalkeepers and referees are detected but not the main focus)

## Additional Files

*Space reserved for documentation of additional analysis files and notebooks*

---

*This project demonstrates the application of computer vision techniques to sports analytics, specifically focusing on football match analysis and automated detection systems.*