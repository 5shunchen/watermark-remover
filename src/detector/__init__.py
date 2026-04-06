"""
Watermark Detection Module
Detects watermarks in images and generates corresponding masks
"""

import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

# Optional ONNX runtime import
try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False


def detect_watermark_by_color(
    image: Image.Image,
    lower_threshold: Tuple[int, int, int] = (0, 0, 0),
    upper_threshold: Tuple[int, int, int] = (180, 255, 255),
) -> Image.Image:
    """
    Detect watermark based on color thresholds (HSV color space)

    Args:
        image: Input PIL Image
        lower_threshold: Lower HSV threshold (H, S, V)
        upper_threshold: Upper HSV threshold (H, S, V)

    Returns:
        PIL Image mask with detected watermark areas
    """
    # Convert PIL to OpenCV format (RGB to BGR)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Convert to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Create mask based on threshold
    mask = cv2.inRange(hsv, lower_threshold, upper_threshold)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Convert back to PIL format
    mask_pil = Image.fromarray(mask)

    return mask_pil


def detect_watermark_by_corner_focus(
    image: Image.Image,
    focus_area: str = "top-right",
) -> Image.Image:
    """
    Detect watermark with focus on corner areas (where watermarks commonly appear)

    Args:
        image: Input PIL Image
        focus_area: Area to focus on ("top-right", "top-left", "bottom-right", "bottom-left")

    Returns:
        PIL Image mask with detected watermark areas
    """
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # Define ROI based on focus area
    roi_positions = {
        "top-right": (0, int(width * 0.6), int(height * 0.2), int(width * 0.4)),
        "top-left": (0, 0, int(height * 0.2), int(width * 0.4)),
        "bottom-right": (
            int(height * 0.8),
            int(width * 0.6),
            int(height * 0.2),
            int(width * 0.4),
        ),
        "bottom-left": (int(height * 0.8), 0, int(height * 0.2), int(width * 0.4)),
    }

    roi_y, roi_x, roi_h, roi_w = roi_positions.get(
        focus_area, (0, int(width * 0.6), int(height * 0.2), int(width * 0.4))
    )

    # Extract ROI
    roi = gray[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

    # Detect bright text (common for watermarks)
    _, roi_thresh = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY)

    # Morphological operations to connect text strokes
    kernel = np.ones((3, 3), np.uint8)
    roi_dilated = cv2.dilate(roi_thresh, kernel, iterations=2)
    roi_eroded = cv2.erode(roi_dilated, kernel, iterations=1)

    # Find contours and filter small noise
    contours, _ = cv2.findContours(
        roi_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    roi_cleaned = np.zeros_like(roi_eroded)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 20:  # Filter small noise
            cv2.drawContours(roi_cleaned, [contour], -1, (255), -1)

    # Copy to full mask
    mask[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = roi_cleaned

    # Also detect bottom subtitles (12% from bottom)
    sub_y = int(height * 0.88)
    subtitle_roi = gray[sub_y:, :]
    _, sub_thresh = cv2.threshold(subtitle_roi, 200, 255, cv2.THRESH_BINARY)

    sub_kernel = np.ones((5, 5), np.uint8)
    sub_dilated = cv2.dilate(sub_thresh, sub_kernel, iterations=2)
    sub_eroded = cv2.erode(sub_dilated, sub_kernel, iterations=1)
    mask[sub_y:, :] = sub_eroded

    # Clean up
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    return Image.fromarray(mask)


def detect_watermark_by_edge(
    image: Image.Image, edge_threshold: int = 50
) -> Image.Image:
    """
    Detect watermark based on edges

    Args:
        image: Input PIL Image
        edge_threshold: Threshold for edge detection

    Returns:
        PIL Image mask with detected watermark areas
    """
    # Convert PIL to OpenCV format (RGB to BGR)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, edge_threshold, edge_threshold * 3)

    # Threshold to binary
    _, binary = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to fill gaps
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Convert back to PIL format
    mask_pil = Image.fromarray(binary)

    return mask_pil


def detect_watermark_by_corners(
    image: Image.Image,
    quality_level: float = 0.01,
    min_distance: int = 10,
    block_size: int = 3,
) -> Image.Image:
    """
    Detect watermark based on corner detection

    Args:
        image: Input PIL Image
        quality_level: Quality level for corner detection
        min_distance: Minimum distance between corners
        block_size: Block size for corner detection

    Returns:
        PIL Image mask with detected watermark areas
    """
    # Convert PIL to OpenCV format (RGB to BGR)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Detect corners
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=0,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
    )

    # Create mask
    mask = np.zeros_like(gray, dtype=np.uint8)

    if corners is not None:
        # Draw detected corners
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(mask, (int(x), int(y)), 5, (255), -1)

        # Apply morphological operations to connect nearby points
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)

    # Convert back to PIL format
    mask_pil = Image.fromarray(mask)

    return mask_pil


def detect_watermark_by_template_matching(
    image: Image.Image,
    template_image: Optional[Image.Image] = None,
    threshold: float = 0.8,
) -> Image.Image:
    """
    Detect watermark using template matching if template is provided
    Otherwise, use a basic approach to detect common watermark patterns

    Args:
        image: Input PIL Image
        template_image: Template image to match (optional) - if provided, will use actual template matching
        threshold: Matching threshold (0-1), higher = more strict matching

    Returns:
        PIL Image mask with detected watermark areas
    """
    # Convert PIL to OpenCV format (RGB to BGR)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    # If template is provided, use actual template matching
    if template_image is not None:
        template_array = np.array(template_image)
        if len(template_array.shape) == 3:
            template_gray = cv2.cvtColor(template_array, cv2.COLOR_RGB2GRAY)
        else:
            template_gray = template_array

        # Perform template matching
        result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, result = cv2.threshold(result, threshold, 255, cv2.THRESH_BINARY)
        result = result.astype(np.uint8)

        # Dilate to cover the matched areas
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.dilate(result, kernel, iterations=2)

        mask_pil = Image.fromarray(result)
        return mask_pil

    # For demo purposes without template, use enhanced pattern detection
    # Combine multiple heuristics for better detection
    mask = np.zeros((height, width), dtype=np.uint8)

    # Common watermark positions analysis
    regions = [
        # Bottom-right corner (most common)
        (height * 3 // 4, width * 3 // 4, height, width),
        # Bottom-left corner
        (height * 3 // 4, 0, height, width // 4),
        # Top-right corner
        (0, width * 3 // 4, height // 4, width),
        # Top-left corner
        (0, 0, height // 4, width // 4),
        # Center (for centered watermarks)
        (height // 3, width // 3, height * 2 // 3, width * 2 // 3),
    ]

    for y1, x1, y2, x2 in regions:
        region = gray[y1:y2, x1:x2]
        # Analyze region for potential watermark characteristics
        # Watermarks often have different brightness/contrast than surroundings
        region_mean = np.mean(region)
        region_std = np.std(region)

        # Compare with surrounding area
        surrounding_mask = np.zeros_like(region, dtype=np.uint8)
        surrounding_threshold = (
            region_mean + 1.5 * region_std if region_std > 0 else 255
        )
        _, surrounding_mask = cv2.threshold(
            region.astype(np.float32), surrounding_threshold, 255, cv2.THRESH_BINARY
        )

        # If significant difference found, mark as potential watermark
        if np.sum(surrounding_mask > 0) > region.size * 0.01:  # At least 1% different
            mask[y1:y2, x1:x2] = (region > region_mean).astype(np.uint8) * 255

    # Apply morphological operations to clean up
    if np.sum(mask) > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask_pil = Image.fromarray(mask)
    return mask_pil


def detect_watermark_by_mser(
    image: Image.Image,
    delta: int = 1,
    min_area: int = 15,
    max_area: int = 5000,
    max_variation: float = 0.7,
    min_diversity: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    Detect text-like watermarks using MSER (Maximally Stable Extremal Regions)

    MSER is particularly effective for detecting text watermarks like "@username"
    as it finds stable blob-like regions across intensity thresholds.

    Args:
        image: Input PIL Image
        delta: Delta parameter for MSER (default: 1)
        min_area: Minimum region area to keep (default: 15)
        max_area: Maximum region area to keep (default: 5000)
        max_variation: Maximum variation threshold (default: 0.7)
        min_diversity: Minimum diversity threshold (default: 0.2)

    Returns:
        List of detected watermark regions, each containing:
        - x, y: Top-left corner coordinates
        - w, h: Width and height of bounding box
        - confidence: Detection confidence (0-1)
        - type: "text" for MSER detection
    """
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array.copy()

    height, width = img_gray.shape
    watermarks: List[Dict[str, Any]] = []

    try:
        # Create MSER detector
        # Note: OpenCV Python API uses parameter names without underscore prefix
        mser = cv2.MSER_create(
            delta=delta,
            min_area=min_area,
            max_area=max_area,
            max_variation=max_variation,
            min_diversity=min_diversity,
        )

        # Detect MSER regions
        regions, _ = mser.detectRegions(img_gray)

        if len(regions) == 0:
            return watermarks

        # Convert regions to bounding boxes
        for region in regions:
            # Get bounding box from region points
            x, y, w, h = cv2.boundingRect(region)

            # Filter based on position (watermarks typically in corners or edges)
            is_corner_or_edge = (
                y < height * 0.25  # Top quarter
                or y > height * 0.75  # Bottom quarter
                or x < width * 0.25  # Left quarter
                or x > width * 0.75  # Right quarter
            )

            if not is_corner_or_edge:
                continue

            # Filter based on aspect ratio (text watermarks are usually horizontal)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 8 or aspect_ratio < 0.5:
                continue

            # Calculate confidence based on region properties
            area = cv2.contourArea(region)
            rect_area = w * h
            fill_ratio = area / rect_area if rect_area > 0 else 0

            # Higher confidence for regions with good fill ratio and corner position
            position_score = 1.0 if (y < height * 0.15 or y > height * 0.85) else 0.7
            confidence = min(1.0, fill_ratio * position_score + 0.3)

            watermarks.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "confidence": round(confidence, 3),
                    "type": "text",
                }
            )

    except Exception as e:
        print(f"MSER detection warning: {e}")

    # Sort by confidence (highest first)
    watermarks.sort(key=lambda x: x["confidence"], reverse=True)

    return watermarks


def detect_watermark_by_onnx(
    image: Image.Image,
    model_path: str = "models/lama.onnx",
    confidence_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Detect watermarks using ONNX deep learning model

    Note: This function currently returns empty results as the LaMa model
    is an inpainting model, not a detection model. A dedicated watermark
    detection ONNX model would be needed for actual DL-based detection.

    Args:
        image: Input PIL Image
        model_path: Path to ONNX model file
        confidence_threshold: Minimum confidence threshold (0-1)

    Returns:
        List of detected watermark regions with x, y, w, h, confidence, type
    """
    if not ONNX_AVAILABLE:
        print("ONNX runtime not available. Install with: pip install onnxruntime")
        return []

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"ONNX model not found at {model_path}")
        return []

    # Note: LaMa is an inpainting model, not a detection model
    # For now, return empty results
    # TODO: Integrate a dedicated watermark detection ONNX model
    print("ONNX detection: Using heuristic methods (LaMa is an inpainting model)")
    return []


def detect_watermark_by_pattern(image: Image.Image) -> Image.Image:
    """
    General watermark detection based on common patterns (brightness, contrast, position)

    Args:
        image: Input PIL Image

    Returns:
        PIL Image mask with detected watermark areas
    """
    # Convert PIL to OpenCV format (RGB to BGR)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Convert to grayscale for analysis
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Enhance contrast to highlight potential watermarks
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Look for areas that differ significantly from surrounding areas
    # which is typical of watermarks
    # Use enhanced image for better contrast detection
    blurred = cv2.GaussianBlur(enhanced_gray, (21, 21), 0)
    diff = cv2.absdiff(enhanced_gray, blurred)

    # Threshold the difference
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Focus on common watermark positions (corners, center)
    h, w = thresh.shape
    mask = np.zeros_like(thresh)

    # Define regions where watermarks commonly appear
    margin = min(h, w) // 10  # 10% margin

    # Top-left corner
    mask[0 : margin * 2, 0 : margin * 2] = thresh[0 : margin * 2, 0 : margin * 2]
    # Top-right corner
    mask[0 : margin * 2, w - margin * 2 : w] = thresh[
        0 : margin * 2, w - margin * 2 : w
    ]
    # Bottom-left corner
    mask[h - margin * 2 : h, 0 : margin * 2] = thresh[
        h - margin * 2 : h, 0 : margin * 2
    ]
    # Bottom-right corner
    mask[h - margin * 2 : h, w - margin * 2 : w] = thresh[
        h - margin * 2 : h, w - margin * 2 : w
    ]
    # Center area (for centered watermarks)
    center_h_start, center_h_end = h // 2 - margin, h // 2 + margin
    center_w_start, center_w_end = w // 2 - margin, w // 2 + margin
    if (
        center_h_start >= 0
        and center_h_end <= h
        and center_w_start >= 0
        and center_w_end <= w
    ):
        mask[center_h_start:center_h_end, center_w_start:center_w_end] = thresh[
            center_h_start:center_h_end, center_w_start:center_w_end
        ]

    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Convert back to PIL format
    mask_pil = Image.fromarray(mask)

    return mask_pil


def detect_watermark_by_text(image: Image.Image) -> Image.Image:
    """
    Detect text-like watermarks using fusion of CLAHE and HSV analysis
    Specialized for detecting watermarks like "@小样燃剪"

    Args:
        image: Input PIL Image

    Returns:
        PIL Image mask with detected text watermark areas
    """
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    h, s, v = cv2.split(hsv)

    # === 1. 右上角区域定义 ===
    roi_x = int(width * 0.5)
    roi_y = 0
    roi_w = int(width * 0.5)
    roi_h = int(height * 0.25)

    # === 2. CLAHE 检测路径 ===
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_gray = gray[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
    roi_enhanced = clahe.apply(roi_gray)

    clahe_mask = np.zeros_like(roi_enhanced)
    for thresh in [170, 190, 210]:
        _, binary = cv2.threshold(roi_enhanced, thresh, 255, cv2.THRESH_BINARY)
        clahe_mask = cv2.bitwise_or(clahe_mask, binary)

    # === 3. HSV 检测路径 ===
    s_roi = s[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
    v_roi = v[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

    _, s_mask = cv2.threshold(s_roi, 70, 255, cv2.THRESH_BINARY_INV)
    _, v_mask = cv2.threshold(v_roi, 150, 255, cv2.THRESH_BINARY)
    hsv_mask = cv2.bitwise_and(s_mask, v_mask)

    # === 4. 融合两种检测结果 ===
    kernel = np.ones((3, 3), np.uint8)
    clahe_dilated = cv2.dilate(clahe_mask, kernel, iterations=2)
    clahe_eroded = cv2.erode(clahe_dilated, kernel, iterations=1)

    hsv_dilated = cv2.dilate(hsv_mask, kernel, iterations=2)
    hsv_eroded = cv2.erode(hsv_dilated, kernel, iterations=1)

    combined = cv2.bitwise_or(clahe_eroded, hsv_eroded)

    # === 5. 轮廓分析和过滤 ===
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_mask = np.zeros_like(combined)

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h_rect = cv2.boundingRect(contour)

        # 位置过滤：水印在 ROI 中下部
        if y < roi_h * 0.15:
            continue

        # 面积过滤
        if area < 10 or area > 5000:
            continue

        # 长宽比过滤（文字通常不是细长条）
        aspect_ratio = w / h_rect if h_rect > 0 else 0
        if aspect_ratio > 15:
            continue

        cv2.drawContours(roi_mask, [contour], -1, (255), -1)

    # === 6. 扩展填充 ===
    coords = cv2.findNonZero(roi_mask)
    if coords is not None:
        x, y, w, h_rect = cv2.boundingRect(coords)

        expand_x = max(5, int(w * 0.3))
        expand_y = max(5, int(h_rect * 0.5))

        x1 = max(0, x - expand_x)
        y1 = max(0, y - expand_y)
        x2 = min(roi_w, x + w + expand_x)
        y2 = min(roi_h, y + h_rect + expand_y)

        region = roi_gray[y1:y2, x1:x2]
        if region.size > 50:
            _, otsu = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            roi_mask[y1:y2, x1:x2] = cv2.bitwise_or(roi_mask[y1:y2, x1:x2], otsu)

    mask[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = roi_mask

    # === 7. 底部字幕检测 ===
    sub_y = int(height * 0.85)
    subtitle_roi = gray[sub_y:, :]

    for thresh in [150, 170, 190]:
        _, sub_binary = cv2.threshold(subtitle_roi, thresh, 255, cv2.THRESH_BINARY)
        sub_kernel = np.ones((5, 5), np.uint8)
        sub_dilated = cv2.dilate(sub_binary, sub_kernel, iterations=2)
        sub_eroded = cv2.erode(sub_dilated, sub_kernel, iterations=1)
        mask[sub_y:, :] = cv2.bitwise_or(mask[sub_y:, :], sub_eroded)

    # === 8. 最终清理 ===
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return Image.fromarray(mask)


def detect_watermark(image: Image.Image, method: str = "auto") -> Image.Image:
    """
    Main function to detect watermark in an image

    Args:
        image: Input PIL Image
        method: Detection method ("color", "edge", "corners", "pattern", "template", "text", "auto")

    Returns:
        PIL Image mask with detected watermark areas
    """
    if method == "color":
        return detect_watermark_by_color(image)
    elif method == "edge":
        return detect_watermark_by_edge(image)
    elif method == "corners":
        return detect_watermark_by_corners(image)
    elif method == "pattern":
        return detect_watermark_by_pattern(image)
    elif method == "template":
        return detect_watermark_by_template_matching(image)
    elif method == "text":
        return detect_watermark_by_text(image)
    elif method == "corner_focus":
        return detect_watermark_by_corner_focus(image, focus_area="top-right")
    else:  # auto
        # Enhanced adaptive detection: try multiple methods and intelligently combine
        try:
            # Run multiple detection methods
            pattern_mask = detect_watermark_by_pattern(image)
            edge_mask = detect_watermark_by_edge(image)
            color_mask = detect_watermark_by_color(image)

            # Convert to numpy arrays for analysis
            pattern_array = np.array(pattern_mask)
            edge_array = np.array(edge_mask)
            color_array = np.array(color_mask)

            # Calculate detection confidence for each method
            pattern_ratio = np.sum(pattern_array > 0) / pattern_array.size
            edge_ratio = np.sum(edge_array > 0) / edge_array.size
            color_ratio = np.sum(color_array > 0) / color_array.size

            # Select best detection based on coverage (watermarks typically cover 1-15% of image)
            masks = [
                (pattern_ratio, pattern_array),
                (edge_ratio, edge_array),
                (color_ratio, color_array),
            ]

            # Filter masks with reasonable coverage (0.1% to 20%)
            valid_masks = [(ratio, arr) for ratio, arr in masks if 0.001 < ratio < 0.2]

            if valid_masks:
                # Use the mask with highest confidence (most coverage within reasonable range)
                best_ratio, best_array = max(valid_masks, key=lambda x: x[0])

                # If multiple methods detected something, combine them
                if len(valid_masks) > 1:
                    combined_array = np.zeros_like(pattern_array)
                    for _, arr in valid_masks:
                        combined_array = np.maximum(combined_array, arr)
                    # Apply morphological closing to connect nearby regions
                    kernel = np.ones((3, 3), np.uint8)
                    combined_array = cv2.morphologyEx(
                        combined_array, cv2.MORPH_CLOSE, kernel
                    )
                    return Image.fromarray(combined_array)
                else:
                    return Image.fromarray(best_array)
            else:
                # No clear detection found - return conservative combination
                # Only use areas detected by at least 2 methods
                combined_array = np.zeros_like(pattern_array)
                detection_count = np.zeros_like(pattern_array, dtype=np.uint8)

                for arr in [pattern_array, edge_array, color_array]:
                    detection_count[arr > 0] += 1

                # Areas detected by 2 or more methods
                combined_array[detection_count >= 2] = 255

                # If nothing found by multiple methods, use edge detection as fallback
                if np.sum(combined_array) == 0:
                    return edge_mask

                return Image.fromarray(combined_array)

        except Exception as e:
            # If anything fails, fall back to edge detection
            print(f"Auto detection warning: {e}")
            return detect_watermark_by_edge(image)


def detect_watermark_v2(
    image: Image.Image,
    method: str = "auto",
    model_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Main function to detect watermarks and return structured output

    This is the v2 API that returns a list of detected watermark regions
    with coordinates, confidence, and type information.

    Args:
        image: Input PIL Image
        method: Detection method ("color", "edge", "corners", "pattern", "template",
                "text", "mser", "onnx", "auto")
        model_path: Path to ONNX model file (only used for "onnx" method)

    Returns:
        List of detected watermark regions, each containing:
        - x, y: Top-left corner coordinates
        - w, h: Width and height of bounding box
        - confidence: Detection confidence (0-1)
        - type: Detection type ("text", "deep_learning", "heuristic")

    Example:
        >>> watermarks = detect_watermark_v2(image, method="mser")
        >>> for wm in watermarks:
        ...     print(f"Watermark at ({wm['x']}, {wm['y']}) "
        ...           f"size {wm['w']}x{wm['h']} confidence {wm['confidence']}")
    """
    # Set default model path if not provided
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "models",
            "lama.onnx",
        )

    if method == "mser":
        return detect_watermark_by_mser(image)
    elif method == "onnx":
        return detect_watermark_by_onnx(image, model_path=model_path)
    elif method == "text":
        # Text method now uses MSER
        return detect_watermark_by_mser(image)
    elif method == "color":
        # Convert color mask to regions
        mask = detect_watermark_by_color(image)
        return _mask_to_regions(mask)
    elif method == "edge":
        mask = detect_watermark_by_edge(image)
        return _mask_to_regions(mask)
    elif method == "corners":
        mask = detect_watermark_by_corners(image)
        return _mask_to_regions(mask)
    elif method == "pattern":
        mask = detect_watermark_by_pattern(image)
        return _mask_to_regions(mask)
    elif method == "template":
        mask = detect_watermark_by_template_matching(image)
        return _mask_to_regions(mask)
    elif method == "corner_focus":
        mask = detect_watermark_by_corner_focus(image)
        return _mask_to_regions(mask)
    else:  # auto
        # Auto mode: try multiple methods and combine results
        all_regions: List[Dict[str, Any]] = []

        # Try MSER first (best for text watermarks)
        mser_regions = detect_watermark_by_mser(image)
        all_regions.extend(mser_regions)

        # Try ONNX if model is available
        if ONNX_AVAILABLE and os.path.exists(model_path):
            onnx_regions = detect_watermark_by_onnx(image, model_path=model_path)
            all_regions.extend(onnx_regions)

        # If no regions found, fall back to heuristic methods
        if len(all_regions) == 0:
            # Try edge detection as fallback
            edge_mask = detect_watermark_by_edge(image)
            fallback_regions = _mask_to_regions(edge_mask, confidence_offset=0.3)
            all_regions.extend(fallback_regions)

        # Remove overlapping regions (non-maximum suppression)
        all_regions = _nms_regions(all_regions)

        return all_regions


def _mask_to_regions(
    mask: Image.Image,
    confidence_threshold: float = 0.5,
    confidence_offset: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Convert a binary mask image to region list format

    Args:
        mask: Binary mask PIL Image (255 for watermark, 0 for background)
        confidence_threshold: Minimum area threshold
        confidence_offset: Offset to add to confidence scores

    Returns:
        List of detected regions with x, y, w, h, confidence, type
    """
    mask_array = np.array(mask)
    height, width = mask_array.shape

    # Find contours
    contours, _ = cv2.findContours(
        mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    regions: List[Dict[str, Any]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20:  # Filter small noise
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Calculate confidence based on area and position
        area_score = min(1.0, area / 500)

        # Higher confidence for corner/edge positions
        is_corner = (
            y < height * 0.2 or y > height * 0.8 or x < width * 0.2 or x > width * 0.8
        )
        position_score = 0.8 if is_corner else 0.5

        confidence = min(1.0, (area_score + position_score) / 2 + confidence_offset)

        regions.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "confidence": round(confidence, 3),
                "type": "heuristic",
            }
        )

    # Sort by confidence
    regions.sort(key=lambda x: x["confidence"], reverse=True)

    return regions


def _nms_regions(
    regions: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Apply non-maximum suppression to remove overlapping regions

    Args:
        regions: List of watermark regions
        iou_threshold: IoU threshold for merging

    Returns:
        Filtered list of non-overlapping regions
    """
    if len(regions) <= 1:
        return regions

    # Sort by confidence (highest first)
    regions = sorted(regions, key=lambda x: x["confidence"], reverse=True)

    keep: List[Dict[str, Any]] = []

    while len(regions) > 0:
        # Take the region with highest confidence
        current = regions.pop(0)
        keep.append(current)

        # Remove regions with high IoU
        remaining = []
        for region in regions:
            iou = _calculate_iou(current, region)
            if iou < iou_threshold:
                remaining.append(region)
        regions = remaining

    return keep


def _calculate_iou(
    box1: Dict[str, Any],
    box2: Dict[str, Any],
) -> float:
    """
    Calculate Intersection over Union between two bounding boxes

    Args:
        box1: First box with x, y, w, h
        box2: Second box with x, y, w, h

    Returns:
        IoU value between 0 and 1
    """
    x1, y1, w1, h1 = box1["x"], box1["y"], box1["w"], box1["h"]
    x2, y2, w2, h2 = box2["x"], box2["y"], box2["w"], box2["h"]

    # Calculate intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0

    return iou
