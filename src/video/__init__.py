"""
Video Processing Module
Handles watermark removal in videos frame by frame
With enhanced progress tracking and batch processing support
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generator

import cv2
import numpy as np
from PIL import Image


class VideoStatus(Enum):
    """Video processing status"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class VideoInfo:
    """Video metadata information"""

    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str


# Defer heavy imports to be used only when needed
try:
    from moviepy.editor import VideoFileClip

    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False
    VideoFileClip = None

from ..detector import detect_watermark
from ..inpainter import remove_watermark


def get_video_info(video_path: str) -> VideoInfo:
    """
    Get video metadata information

    Args:
        video_path: Path to video file

    Returns:
        VideoInfo object with video metadata

    Raises:
        ValueError: If video file cannot be opened
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Get codec information
    codec_code = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((codec_code >> 8 * i) & 0xFF) for i in range(4)])

    cap.release()

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        duration=duration,
        codec=codec,
    )


def estimate_processing_time(video_path: str, frame_interval: int = 1) -> float:
    """
    Estimate the processing time for video watermark removal

    Args:
        video_path: Path to video file
        frame_interval: Frame processing interval

    Returns:
        Estimated processing time in seconds
    """
    video_info = get_video_info(video_path)

    # Estimate based on frame processing time (0.5 seconds per frame as approximation)
    estimated_time_per_frame = 0.5  # Adjust based on actual processing speed
    processed_frames_count = video_info.total_frames // frame_interval

    return processed_frames_count * estimated_time_per_frame


def extract_frames(
    video_path: str, frame_interval: int = 1
) -> Generator[tuple, None, None]:
    """
    Extract frames from video at specified intervals

    Args:
        video_path: Path to input video
        frame_interval: Interval between extracted frames (1 = every frame, 2 = every 2nd frame, etc.)

    Yields:
        Tuple of (frame_index, PIL Image)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            yield frame_count, pil_image

        frame_count += 1

    cap.release()


def process_video_frame(
    frame: Image.Image,
    detection_method: str = "auto",
    model_path: str = None,
    device: str = "cpu",
) -> Image.Image:
    """
    Process a single video frame to remove watermarks

    Args:
        frame: Input PIL Image frame
        detection_method: Method for watermark detection
        model_path: Path to inpainting model
        device: Device for processing

    Returns:
        Processed PIL Image with watermarks removed
    """
    # Detect watermark in the frame
    mask = detect_watermark(frame, method=detection_method)

    # Remove watermark using inpainting
    processed_frame = remove_watermark(
        image=frame, mask=mask, model_path=model_path, device=device
    )

    return processed_frame


def remove_watermark_from_video(
    video_input_path: str,
    video_output_path: str,
    detection_method: str = "auto",
    model_path: str = None,
    device: str = "cpu",
    frame_interval: int = 1,
    progress_callback: Callable[[float], Any] = None,
) -> bool:
    """
    Remove watermarks from video file

    Args:
        video_input_path: Path to input video file
        video_output_path: Path for output video file
        detection_method: Watermark detection method
        model_path: Path to inpainting model
        device: Device for processing
        frame_interval: Frame processing interval
        progress_callback: Callback function to report progress (0.0 to 1.0)

    Returns:
        True if successful, False otherwise
    """
    if not HAS_MOVIEPY:
        print("MoviePy not available. Video processing requires 'moviepy' package.")
        return False

    # Create a temporary directory for processed frames
    temp_dir = os.path.join(os.path.dirname(video_output_path), "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Count total frames to calculate progress
        cap = cv2.VideoCapture(video_input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        processed_frame_paths = []
        frame_count = 0

        for frame_idx, frame in extract_frames(video_input_path, frame_interval):
            # Process the frame
            processed_frame = process_video_frame(
                frame=frame,
                detection_method=detection_method,
                model_path=model_path,
                device=device,
            )

            # Save processed frame
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
            processed_frame.save(frame_path)
            processed_frame_paths.append(frame_path)

            frame_count += 1

            # Report progress
            if progress_callback:
                progress = min(1.0, (frame_idx + 1) / total_frames)
                progress_callback(progress)

        # Reconstruct video from processed frames
        # Load the original video and replace frames
        original_clip = VideoFileClip(video_input_path)

        # For this implementation, we'll use moviepy to process the video
        # In a real implementation, we might need a more sophisticated approach

        # Create a function to process each frame
        def process_clip_frame(get_frame, t):
            frame_time = t
            current_frame_idx = int(frame_time * fps)

            # Since we pre-processed specific frames, we need to map time to processed frames
            # For now, we'll use a simplified approach
            if current_frame_idx % frame_interval == 0:
                # Find closest processed frame
                closest_idx = min(
                    range(len(processed_frame_paths)),
                    key=lambda i: abs(
                        int(processed_frame_paths[i].split("_")[-1].split(".")[0])
                        - current_frame_idx
                    ),
                )

                # Load the processed frame
                processed_img = Image.open(processed_frame_paths[closest_idx])
                return np.array(processed_img)
            else:
                # Return original frame for skipped frames
                original_frame = get_frame(t)
                return original_frame

        # Apply processing to the clip
        processed_clip = original_clip.fl(process_clip_frame)

        # Write the output video
        processed_clip.write_videofile(
            video_output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=os.path.join(temp_dir, "temp-audio.m4a"),
            remove_temp=True,
        )

        # Close clips to free memory
        original_clip.close()
        processed_clip.close()

        # Clean up temporary frames
        for frame_path in processed_frame_paths:
            os.remove(frame_path)
        os.rmdir(temp_dir)

        return True

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        # Clean up temp directory in case of error
        import shutil

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return False
