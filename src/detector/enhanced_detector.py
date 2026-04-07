"""
Enhanced Watermark Detection Module
使用阈值分割 + 形态学 + 位置过滤来精准检测文字水印
"""

import cv2
import numpy as np
from PIL import Image


def detect_watermark_enhanced(image: Image.Image) -> Image.Image:
    """
    增强版水印检测 - 精准识别右上角水印和底部字幕

    使用阈值分割检测亮色文字 + 位置过滤 + 形态学优化

    Args:
        image: Input PIL Image

    Returns:
        PIL Image mask with detected watermark areas (white=watermark, black=background)
    """
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array.copy()

    height, width = img_gray.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    # === 1. 检测右上角水印（通常是白色/亮色文字）===
    # 定义扫描区域：右上角 20% 高度，40% 宽度
    scan_x1 = int(width * 0.55)
    scan_y1 = int(height * 0.05)
    scan_x2 = int(width * 0.95)
    scan_y2 = int(height * 0.25)

    # 确保区域有效
    if scan_x1 >= scan_x2 or scan_y1 >= scan_y2:
        scan_x1, scan_y1 = int(width * 0.6), 0
        scan_x2, scan_y2 = int(width * 0.95), int(height * 0.25)

    scan_region = img_gray[scan_y1:scan_y2, scan_x1:scan_x2]

    # 使用 Otsu 阈值 + 亮度过滤检测亮色文字
    _, scan_thresh = cv2.threshold(scan_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_threshold = max(_, 120)  # 至少 120，确保只检测亮区
    _, scan_binary = cv2.threshold(scan_region, bright_threshold, 255, cv2.THRESH_BINARY)

    # 形态学连接文字笔画
    dilated = cv2.dilate(scan_binary, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # 查找轮廓并绘制到 mask
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 15:  # 降低面积阈值，检测小文字
            cv2.drawContours(mask[scan_y1:scan_y2, scan_x1:scan_x2], [contour], -1, (255), -1)

    # === 2. 检测底部字幕（通常是白色文字，位于底部 15% 区域）===
    sub_y = int(height * 0.85)
    subtitle_roi = img_gray[sub_y:, :]

    # 使用 Otsu 阈值检测亮色文字
    _, sub_thresh = cv2.threshold(subtitle_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, sub_binary = cv2.threshold(subtitle_roi, max(_, 120), 255, cv2.THRESH_BINARY)

    # 形态学操作连接文字
    sub_dilated = cv2.dilate(sub_binary, kernel, iterations=3)
    sub_eroded = cv2.erode(sub_dilated, kernel, iterations=2)

    # 找到字幕的边界框，绘制一个连续的条带
    coords = cv2.findNonZero(sub_eroded)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # 扩展边界确保完全覆盖
        expand_x = max(10, int(w * 0.1))
        expand_y = max(5, int(h * 0.3))
        x1 = max(0, x - expand_x)
        y1 = max(0, sub_y + y - expand_y)
        x2 = min(width, x + w + expand_x)
        y2 = min(height, sub_y + y + h + expand_y)
        mask[y1:y2, x1:x2] = 255

    # === 3. 最终形态学优化 ===
    if np.sum(mask) > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return Image.fromarray(mask)
