import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from Complete.ml_field_tracker.pitch_detector import MLPitchDetector


def process_media(input_path: str, output_path: Optional[str] = None, 
                 model_path: str = None, device: str = 'cpu',
                 show_keypoints: bool = True, generate_topview: bool = False):
    """
    Process an image or video using ML-based pitch detection.
    
    Args:
        input_path: Path to input image or video
        output_path: Path for output (auto-generated if None)
        model_path: Path to YOLO keypoint detection model
        device: Device for inference ('cpu', 'cuda', 'mps')
        show_keypoints: Whether to show detected keypoints
        generate_topview: Whether to generate top-down view
    """
    if model_path is None:
        raise ValueError("model_path is required for ML pitch detection")
    
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    # Initialize detector
    detector = MLPitchDetector(model_path, device)
    
    # Check if input is video
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
    is_video = input_path.suffix.lower() in video_extensions
    
    if is_video:
        _process_video(input_path, output_path, detector, show_keypoints, generate_topview)
    else:
        _process_image(input_path, output_path, detector, show_keypoints, generate_topview)


def _process_video(input_path: Path, output_path: Optional[str], 
                  detector: MLPitchDetector, show_keypoints: bool, generate_topview: bool):
    """Process video file"""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output path
    if output_path is None:
        suffix = "_topview.mp4" if generate_topview else "_annotated.mp4"
        output_path = input_path.parent / f"{input_path.stem}{suffix}"
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    print(f"Processing {total_frames} frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        annotated_frame, top_view = detector.process_frame(
            frame, show_keypoints, generate_topview
        )
        
        # Use top view if requested and available, otherwise use annotated frame
        output_frame = top_view if (generate_topview and top_view is not None) else annotated_frame
        
        # Resize to match original dimensions if needed
        if output_frame.shape[:2] != (height, width):
            output_frame = cv2.resize(output_frame, (width, height))
        
        out.write(output_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")
    
    cap.release()
    out.release()
    print(f"✅ Output video saved to: {output_path}")


def _process_image(input_path: Path, output_path: Optional[str], 
                  detector: MLPitchDetector, show_keypoints: bool, generate_topview: bool):
    """Process single image"""
    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"Cannot read image: {input_path}")
    
    # Process image
    annotated_img, top_view = detector.process_frame(
        img, show_keypoints, generate_topview
    )
    
    # Use top view if requested and available, otherwise use annotated image
    output_img = top_view if (generate_topview and top_view is not None) else annotated_img
    
    # Setup output path
    if output_path is None:
        suffix = "_topview" if generate_topview else "_annotated"
        output_path = input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}"
    
    cv2.imwrite(str(output_path), output_img)
    print(f"✅ Output image saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    input_path = "/Users/alanpehz/Documents/Personal/True Computer Vision/FootballTracker/Complete/test_content/demo1.mp4"
    model_path = "/Users/alanpehz/Documents/Personal/True Computer Vision/FootballTracker/Complete/models/pitch_tracker.pt"
    
    # Process with keypoint annotation
    process_media(
        input_path=input_path,
        model_path=model_path,
        device='cpu',
        show_keypoints=True,
        generate_topview=False
    )

    '''
    # Process with top-down view
    process_media(
        input_path=input_path,
        model_path=model_path,
        device='cpu',
        show_keypoints=False,
        generate_topview=True
    )
    '''