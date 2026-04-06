#!/usr/bin/env python3
"""
迭代 14 测试 - 综合 CLAHE + HSV + 形态学
结合多种检测方法的优势
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
import cv2
import numpy as np
from inpainter.enhanced import remove_watermark as remove_watermark_enhanced


def detect_fusion_watermark(image: Image.Image) -> Image.Image:
    """
    融合 CLAHE、HSV 和形态学的检测方法
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
    roi_gray = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    roi_enhanced = clahe.apply(roi_gray)

    clahe_mask = np.zeros_like(roi_enhanced)
    for thresh in [170, 190, 210]:
        _, binary = cv2.threshold(roi_enhanced, thresh, 255, cv2.THRESH_BINARY)
        clahe_mask = cv2.bitwise_or(clahe_mask, binary)

    # === 3. HSV 检测路径 ===
    s_roi = s[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    v_roi = v[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    _, s_mask = cv2.threshold(s_roi, 70, 255, cv2.THRESH_BINARY_INV)
    _, v_mask = cv2.threshold(v_roi, 150, 255, cv2.THRESH_BINARY)
    hsv_mask = cv2.bitwise_and(s_mask, v_mask)

    # === 4. 融合两种检测结果 ===
    # 先对 CLAHE mask 进行形态学操作
    kernel = np.ones((3, 3), np.uint8)
    clahe_dilated = cv2.dilate(clahe_mask, kernel, iterations=2)
    clahe_eroded = cv2.erode(clahe_dilated, kernel, iterations=1)

    # 对 HSV mask 进行形态学操作
    hsv_dilated = cv2.dilate(hsv_mask, kernel, iterations=2)
    hsv_eroded = cv2.erode(hsv_dilated, kernel, iterations=1)

    # 融合：取并集
    combined = cv2.bitwise_or(clahe_eroded, hsv_eroded)

    # === 5. 轮廓分析和过滤 ===
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_mask = np.zeros_like(combined)

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # 位置过滤：水印在 ROI 中下部
        if y < roi_h * 0.15:
            continue

        # 面积过滤
        if area < 10 or area > 5000:
            continue

        # 长宽比过滤（文字通常不是细长条）
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 15:  # 太细长，可能是头发
            continue

        cv2.drawContours(roi_mask, [contour], -1, (255), -1)

    # === 6. 扩展填充 ===
    coords = cv2.findNonZero(roi_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)

        expand_x = max(5, int(w * 0.3))
        expand_y = max(5, int(h * 0.5))

        x1 = max(0, x - expand_x)
        y1 = max(0, y - expand_y)
        x2 = min(roi_w, x + w + expand_x)
        y2 = min(roi_h, y + h + expand_y)

        # Otsu 填充
        region = roi_gray[y1:y2, x1:x2]
        if region.size > 50:
            _, otsu = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            roi_mask[y1:y2, x1:x2] = cv2.bitwise_or(roi_mask[y1:y2, x1:x2], otsu)

    # 复制到完整 mask
    mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = roi_mask

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


def test_iteration_14():
    print("🧪 迭代 14 - 融合检测测试")
    print("=" * 60)

    test_image_path = Path("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
    image = Image.open(test_image_path)
    print(f"\n📷 图片：{image.size}")

    print("\n🔍 融合检测水印...")
    mask = detect_fusion_watermark(image)
    mask_arr = np.array(mask)
    white_pixels = np.sum(mask_arr > 0)
    percentage = (white_pixels / mask_arr.size) * 100
    print(f"   检测到水印区域：{white_pixels} 像素 ({percentage:.2f}%)")

    mask.save("output/mask_iter14.png")

    mask_rgb = Image.new('RGB', mask.size)
    mask_rgb_data = np.zeros((mask.size[1], mask.size[0], 3), dtype=np.uint8)
    mask_rgb_data[mask_arr > 0] = [255, 0, 0]
    mask_rgb = Image.fromarray(mask_rgb_data)
    mask_rgb.save("output/mask_iter14_visual.png")

    print("\n🎨 测试 inpainting 方法:")
    methods = ["aggressive", "multi", "ns"]
    for method in methods:
        print(f"   方法：{method}")
        result = remove_watermark_enhanced(image=image, mask=mask, method=method)
        result.save(f"output/cleaned_iter14_{method}.png")
        print(f"      ✅ 结果已保存")

    image.save("output/original_iter14.png")

    print("\n" + "=" * 60)
    print("✅ 迭代 14 完成！")


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    test_iteration_14()
