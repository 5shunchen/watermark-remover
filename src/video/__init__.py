"""
Video Processing Module
Handles watermark removal in videos frame by frame
With enhanced progress tracking and batch processing support

Technical requirements:
- Uses FFmpeg for frame extraction and re-encoding
- Preserves original audio track
- Supports MP4/AVI/MOV formats
- Progress bar displayed with tqdm
"""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generator

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


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
    Remove watermarks from video file using FFmpeg for frame extraction and re-encoding

    Args:
        video_input_path: Path to input video file (MP4/AVI/MOV supported)
        video_output_path: Path for output video file
        detection_method: Watermark detection method
        model_path: Path to inpainting model
        device: Device for processing
        frame_interval: Frame processing interval
        progress_callback: Callback function to report progress (0.0 to 1.0)

    Returns:
        True if successful, False otherwise

    Performance target: > 10 fps for 1080p video (CPU mode)
    """
    # Verify input file exists
    if not os.path.exists(video_input_path):
        print(f"Error: Input video file not found: {video_input_path}")
        return False

    # Create temporary directories
    temp_dir = tempfile.mkdtemp(prefix="video_processing_")
    frames_dir = os.path.join(temp_dir, "frames")
    processed_dir = os.path.join(temp_dir, "processed")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    try:
        # Step 1: Extract frames using FFmpeg
        print(f"Extracting frames from {video_input_path}...")
        frame_pattern = os.path.join(frames_dir, "frame_%06d.png")
        ffmpeg_extract_cmd = [
            "ffmpeg",
            "-i",
            video_input_path,
            "-vf",
            "fps=original",
            frame_pattern,
            "-loglevel",
            "error",
        ]
        result = subprocess.run(
            ffmpeg_extract_cmd, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            print(f"Error extracting frames: {result.stderr}")
            return False

        # Get list of extracted frames
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
        total_frames = len(frame_files)

        if total_frames == 0:
            print("Error: No frames extracted from video")
            return False

        print(f"Extracted {total_frames} frames")

        # Step 2: Get video info for audio extraction and output settings
        cap = cv2.VideoCapture(video_input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Step 3: Process each frame with tqdm progress bar
        print(f"Processing {total_frames} frames...")
        processed_frame_paths = []

        with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
            for i, frame_file in enumerate(frame_files):
                frame_idx = int(frame_file.split("_")[1].split(".")[0])

                # Check if this frame should be processed based on interval
                if frame_idx % frame_interval != 0:
                    # Skip this frame, copy original
                    frame_path = os.path.join(frames_dir, frame_file)
                    processed_path = os.path.join(processed_dir, frame_file)
                    shutil.copy(frame_path, processed_path)
                    processed_frame_paths.append(processed_path)
                    pbar.update(1)
                    continue

                # Load frame
                frame_path = os.path.join(frames_dir, frame_file)
                frame = Image.open(frame_path)

                # Process the frame
                try:
                    processed_frame = process_video_frame(
                        frame=frame,
                        detection_method=detection_method,
                        model_path=model_path,
                        device=device,
                    )

                    # Save processed frame
                    processed_path = os.path.join(processed_dir, frame_file)
                    processed_frame.save(processed_path)
                    processed_frame_paths.append(processed_path)

                    # Update progress
                    pbar.update(1)
                    if progress_callback:
                        progress = (i + 1) / total_frames
                        progress_callback(progress)

                except Exception as e:
                    print(f"Error processing frame {frame_file}: {e}")
                    # Copy original frame on error
                    processed_path = os.path.join(processed_dir, frame_file)
                    shutil.copy(frame_path, processed_path)
                    processed_frame_paths.append(processed_path)
                    pbar.update(1)

        # Step 4: Reconstruct video using FFmpeg with original audio
        print("Reconstructing video with FFmpeg...")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(video_output_path) or ".", exist_ok=True)

        # Pattern for processed frames
        processed_pattern = os.path.join(processed_dir, "frame_%06d.png")

        # First, try to extract audio from original video
        temp_audio_path = os.path.join(temp_dir, "audio.aac")
        ffmpeg_audio_cmd = [
            "ffmpeg",
            "-i",
            video_input_path,
            "-vn",
            "-acodec",
            "copy",
            temp_audio_path,
            "-loglevel",
            "error",
        ]
        audio_result = subprocess.run(
            ffmpeg_audio_cmd, capture_output=True, text=True, check=False
        )

        # Build FFmpeg command for video reconstruction
        if audio_result.returncode == 0 and os.path.exists(temp_audio_path):
            # Video with audio
            ffmpeg_merge_cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                processed_pattern,
                "-i",
                temp_audio_path,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
                video_output_path,
                "-loglevel",
                "error",
            ]
        else:
            # Video without audio (audio extraction failed or no audio in source)
            print(
                "Note: No audio track found or extraction failed, creating video without audio"
            )
            ffmpeg_merge_cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                processed_pattern,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                video_output_path,
                "-loglevel",
                "error",
            ]

        result = subprocess.run(
            ffmpeg_merge_cmd, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            print(f"Error reconstructing video: {result.stderr}")
            return False

        print(f"Video processing completed: {video_output_path}")
        return True

    except Exception as e:
        print(f"Error processing video: {e}")
        return False

    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
