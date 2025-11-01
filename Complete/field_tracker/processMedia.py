import cv2
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

from Complete.field_tracker.calibrationRoutines import calibrate_from_image, display_yaw_and_focal_length, \
    display_top_view
from Complete.field_tracker.projectionHelpers import draw_pitch_lines
from Complete.field_tracker.shapeDetection import find_key_points


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
    guess_fx = 2000
    guess_rot = np.array([[0.25, 0, 0]])
    guess_trans = (0, 0, 80)

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Detect if input is a video file
    video_exts = [".mp4", ".avi", ".mov", ".mkv"]
    is_video = input_path.suffix.lower() in video_exts

    if is_video:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_path is None:
            suffix = "_topview.mp4" if top_view else "_annotated.mp4"
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
            if frame_idx % 30 == 0:
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
            suffix = "_topview" + input_path.suffix if top_view else "_annotated" + input_path.suffix
            output_path = input_path.parent / f"{input_path.stem}{suffix}"

        cv2.imwrite(str(output_path), img)
        print(f"✅ Output image saved to: {output_path}")


if __name__ == "__main__":
    # Modify these parameters for testing
    input_path = "/Users/alanpehz/Documents/Personal/True Computer Vision/FootballTracker/Complete/test_content/2e57b9_1_9_png.rf.4ddf27c8067f98fd10da07374f376097.jpg"
    output_path = "/Users/alanpehz/Documents/Personal/True Computer Vision/FootballTracker/Complete/test_results/2e57b9_1_9_png-topview.jpg"  # Will auto-generate if None
    top_view = True
    
    process_media(input_path, output_path, top_view)
