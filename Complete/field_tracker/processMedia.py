import cv2
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

from Complete.field_tracker.calibrationRoutines import calibrate_from_image, display_yaw_and_focal_length, \
    display_top_view
from Complete.field_tracker.projectionHelpers import draw_pitch_lines
from Complete.field_tracker.shapeDetection import find_key_points
from Complete.field_tracker.Constants import (
    DEFAULT_GUESS_FX, DEFAULT_GUESS_ROT, DEFAULT_GUESS_TRANS,
    VIDEO_EXTENSIONS, VIDEO_FOURCC, FRAME_PROGRESS_INTERVAL,
    TOPVIEW_SUFFIX, ANNOTATED_SUFFIX, VIDEO_OUTPUT_EXT
)


def process_media(input_path, output_path=None, top_view=False):
    """
    Process an image or video to perform camera pose estimation and annotation.
    Optionally generates a top-down (bird’s-eye) view of the field.

    Args:
        input_path (str or Path): Path to an image or video file.
        output_path (str or Path): Path to save the annotated output.
        top_view (bool): Whether to generate top-view (bird’s-eye) transformation.
    """

    # Default camera parameter guesses
    guess_fx = DEFAULT_GUESS_FX
    guess_rot = np.array(DEFAULT_GUESS_ROT)
    guess_trans = DEFAULT_GUESS_TRANS

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Detect if input is a video file
    is_video = input_path.suffix.lower() in VIDEO_EXTENSIONS

    if is_video:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")

        fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_path is None:
            suffix = f"{TOPVIEW_SUFFIX}{VIDEO_OUTPUT_EXT}" if top_view else f"{ANNOTATED_SUFFIX}{VIDEO_OUTPUT_EXT}"
            output_path = input_path.parent / f"{input_path.stem}{suffix}"

        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            K, to_device_from_world, rot, trans, frame = calibrate_from_image(
                frame, guess_fx, guess_rot, guess_trans
            )

            if not top_view:
                # Regular annotated frame
                key_points, key_lines = find_key_points(frame)
                frame = key_points.draw(frame)

                if to_device_from_world is not None:
                    frame = draw_pitch_lines(K, to_device_from_world, frame)
                    frame = display_yaw_and_focal_length(
                        frame, guess_rot[0, 1] * 180 / np.pi, K[0, 0]
                    )
            else:
                # Bird's-eye / top view
                if to_device_from_world is not None:
                    frame = display_top_view(K, to_device_from_world, frame)

            # Update guesses for next frame
            if to_device_from_world is not None:
                guess_rot = rot
                guess_trans = trans
                guess_fx = K[0, 0]

            out.write(frame)

            frame_idx += 1
            if frame_idx % FRAME_PROGRESS_INTERVAL == 0:
                print(f"Processed {frame_idx} frames...")

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"✅ Output video saved to: {output_path}")

    else:
        # Process single image
        img = cv2.imread(str(input_path))
        if img is None:
            raise ValueError(f"Cannot read image: {input_path}")

        K, to_device_from_world, rot, trans, img = calibrate_from_image(
            img, guess_fx, guess_rot, guess_trans
        )

        if not top_view:
            key_points, key_lines = find_key_points(img)
            img = key_points.draw(img)
            if to_device_from_world is not None:
                img = draw_pitch_lines(K, to_device_from_world, img)
                img = display_yaw_and_focal_length(
                    img, guess_rot[0, 1] * 180 / np.pi, K[0, 0]
                )
        else:
            if to_device_from_world is not None:
                img = display_top_view(K, to_device_from_world, img)
            else:
                print("⚠️ Top-view unavailable: calibration failed.")

        if output_path is None:
            suffix = TOPVIEW_SUFFIX + input_path.suffix if top_view else ANNOTATED_SUFFIX + input_path.suffix
            output_path = input_path.parent / f"{input_path.stem}{suffix}"

        cv2.imwrite(str(output_path), img)
        print(f"✅ Output image saved to: {output_path}")


if __name__ == "__main__":
    # Modify these parameters for testing
    input_path = "/Users/alanpehz/Documents/Personal/True Computer Vision/FootballTracker/Complete/test_content/demo1.mp4"
    output_path = "/Users/alanpehz/Documents/Personal/True Computer Vision/FootballTracker/Complete/test_results/demo1.mp4"  # Will auto-generate if None
    top_view = False
    
    process_media(input_path, output_path, top_view)
