"""
Watermark Detection Module
Detects watermarks in images and generates corresponding masks
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


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
        "bottom-right": (int(height * 0.8), int(width * 0.6), int(height * 0.2), int(width * 0.4)),
        "bottom-left": (int(height * 0.8), 0, int(height * 0.2), int(width * 0.4)),
    }

    roi_y, roi_x, roi_h, roi_w = roi_positions.get(focus_area, (0, int(width * 0.6), int(height * 0.2), int(width * 0.4)))

    # Extract ROI
    roi = gray[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

    # Detect bright text (common for watermarks)
    _, roi_thresh = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY)

    # Morphological operations to connect text strokes
    kernel = np.ones((3, 3), np.uint8)
    roi_dilated = cv2.dilate(roi_thresh, kernel, iterations=2)
    roi_eroded = cv2.erode(roi_dilated, kernel, iterations=1)

    # Find contours and filter small noise
    contours, _ = cv2.findContours(roi_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    Detect text-like watermarks using MSER and morphological analysis
    Specialized for detecting watermarks like "@小样燃剪"

    Args:
        image: Input PIL Image

    Returns:
        PIL Image mask with detected text watermark areas
    """
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # === 1. 右上角水印精确检测 ===
    # ROI: 右上角 30% 宽度 x 15% 高度
    roi_x = int(width * 0.6)
    roi_y = 0
    roi_w = int(width * 0.4)
    roi_h = int(height * 0.15)

    roi = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # 计算 ROI 的局部统计特性
    roi_mean = np.mean(roi)
    roi_std = np.std(roi)

    # 使用 Otsu 自适应阈值
    otsu_result, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thresh = int(otsu_result)

    # 多阈值检测 - 捕捉不同亮度的文字
    detected = False
    for thresh_val in [int(roi_mean + roi_std), int(roi_mean + 1.5 * roi_std), otsu_thresh, 180, 200]:
        if detected:
            break
        _, roi_binary = cv2.threshold(roi, int(thresh_val), 255, cv2.THRESH_BINARY)

        # 形态学膨胀连接文字笔画
        kernel = np.ones((3, 3), np.uint8)
        roi_dilated = cv2.dilate(roi_binary, kernel, iterations=2)

        # 找到轮廓
        contours, _ = cv2.findContours(roi_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            x_rect, y_rect, w, h = cv2.boundingRect(contour)

            # 文字特征：合理的面积和长宽比
            if 20 < area < 8000 and 0.2 < w / h < 12:
                # 绘制到 ROI mask
                roi_mask = np.zeros_like(roi_dilated)
                cv2.drawContours(roi_mask, [contour], -1, (255), -1)
                # 合并到总 mask
                mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = cv2.bitwise_or(
                    mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w], roi_mask
                )
                detected = True

    # === 2. 颜色空间检测 - HSV 中检测低饱和度高亮度区域 ===
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 水印通常是低饱和度 + 高亮度
    _, v_mask = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY)
    _, s_mask = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY_INV)

    # 结合两个条件
    color_mask = cv2.bitwise_and(v_mask, s_mask)

    # 只保留右上角区域
    color_roi = color_mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    kernel = np.ones((3, 3), np.uint8)
    color_dilated = cv2.dilate(color_roi, kernel, iterations=2)
    contours, _ = cv2.findContours(color_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 20 < area < 8000:
            cv2.drawContours(mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w],
                           [contour], -1, (255), -1)

    # === 3. 底部字幕检测 ===
    sub_y = int(height * 0.85)
    subtitle_roi = gray[sub_y:, :]

    for thresh_val in [160, 180, 200]:
        _, sub_binary = cv2.threshold(subtitle_roi, thresh_val, 255, cv2.THRESH_BINARY)
        sub_kernel = np.ones((5, 5), np.uint8)
        sub_dilated = cv2.dilate(sub_binary, sub_kernel, iterations=2)
        sub_eroded = cv2.erode(sub_dilated, sub_kernel, iterations=1)
        mask[sub_y:, :] = cv2.bitwise_or(mask[sub_y:, :], sub_eroded)

    # === 4. 最终形态学清理 ===
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
