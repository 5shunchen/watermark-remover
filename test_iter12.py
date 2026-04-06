#!/usr/bin/env python3
"""
迭代 12 测试 - 使用边缘密度和纹理分析过滤
头发区域：高边缘密度，复杂纹理
水印文字：低边缘密度，简单几何形状
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
import cv2
import numpy as np
from inpainter.enhanced import remove_watermark as remove_watermark_enhanced


def detect_smart_filter_watermark(image: Image.Image) -> Image.Image:
    """
    使用边缘密度和纹理分析过滤的水印检测
    """
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # === 1. 右上角区域检测 ===
    roi_x = int(width * 0.55)
    roi_y = int(height * 0.08)
    roi_w = int(width * 0.40)
    roi_h = int(height * 0.15)

    roi = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    roi_h_size, roi_w_size = roi.shape

    # CLAHE 增强
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    roi_enhanced = clahe.apply(roi)

    # 初步检测
    combined = np.zeros_like(roi)
    for thresh in [170, 190, 210]:
        _, binary = cv2.threshold(roi_enhanced, thresh, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_or(combined, binary)

    # 形态学清理
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # 找到所有轮廓
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_mask = np.zeros_like(combined)

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # 基本过滤
        if area < 10 or area > 5000:
            continue

        if y < roi_h_size * 0.2:  # 太靠上的可能是头发
            continue

        # 创建当前轮廓的 mask
        contour_mask = np.zeros_like(combined)
        cv2.drawContours(contour_mask, [contour], -1, (255), -1)

        # 计算边缘密度（使用 Canny）
        roi_crop = roi[y:y+h, x:x+w]
        if roi_crop.size < 100:
            continue

        edges = cv2.Canny(roi_crop, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # 水印文字通常边缘密度较低（相比头发）
        # 头发：高边缘密度（>0.3）
        # 文字：低边缘密度（<0.2）
        if edge_density < 0.25:
            cv2.drawContours(roi_mask, [contour], -1, (255), -1)

    # 如果有有效区域，扩展并填充
    coords = cv2.findNonZero(roi_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)

        # 扩展
        expand_x = max(3, int(w * 0.3))
        expand_y = max(3, int(h * 0.5))

        x1 = max(0, x - expand_x)
        y1 = max(0, y - expand_y)
        x2 = min(roi_w_size, x + w + expand_x)
        y2 = min(roi_h_size, y + h + expand_y)

        # Otsu 填充
        region = roi[y1:y2, x1:x2]
        if region.size > 50:
            _, otsu = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            roi_mask[y1:y2, x1:x2] = cv2.bitwise_or(roi_mask[y1:y2, x1:x2], otsu)

    # 复制到完整 mask
    mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = roi_mask

    # === 2. 底部字幕检测 ===
    sub_y = int(height * 0.85)
    subtitle_roi = gray[sub_y:, :]

    for thresh in [150, 170, 190]:
        _, sub_binary = cv2.threshold(subtitle_roi, thresh, 255, cv2.THRESH_BINARY)
        sub_kernel = np.ones((5, 5), np.uint8)
        sub_dilated = cv2.dilate(sub_binary, sub_kernel, iterations=2)
        sub_eroded = cv2.erode(sub_dilated, sub_kernel, iterations=1)
        mask[sub_y:, :] = cv2.bitwise_or(mask[sub_y:, :], sub_eroded)

    # === 3. 最终清理 ===
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return Image.fromarray(mask)


def test_iteration_12():
    print("🧪 迭代 12 - 边缘密度过滤测试")
    print("=" * 60)

    test_image_path = Path("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
    image = Image.open(test_image_path)
    print(f"\n📷 图片：{image.size}")

    print("\n🔍 检测水印...")
    mask = detect_smart_filter_watermark(image)
    mask_arr = np.array(mask)
    white_pixels = np.sum(mask_arr > 0)
    percentage = (white_pixels / mask_arr.size) * 100
    print(f"   检测到水印区域：{white_pixels} 像素 ({percentage:.2f}%)")

    mask.save("output/mask_iter12.png")

    mask_rgb = Image.new('RGB', mask.size)
    mask_rgb_data = np.zeros((mask.size[1], mask.size[0], 3), dtype=np.uint8)
    mask_rgb_data[mask_arr > 0] = [255, 0, 0]
    mask_rgb = Image.fromarray(mask_rgb_data)
    mask_rgb.save("output/mask_iter12_visual.png")

    print("\n🎨 测试 inpainting 方法:")
    methods = ["aggressive", "multi", "ns"]
    for method in methods:
        print(f"   方法：{method}")
        result = remove_watermark_enhanced(image=image, mask=mask, method=method)
        result.save(f"output/cleaned_iter12_{method}.png")
        print(f"      ✅ 结果已保存")

    image.save("output/original_iter12.png")

    print("\n" + "=" * 60)
    print("✅ 迭代 12 完成！")


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    test_iteration_12()
