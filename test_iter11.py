#!/usr/bin/env python3
"""
迭代 11 测试 - 基于诊断的手动调优
从 diagnose.py 结果：CLAHE 增强后阈值 180 有 222 像素 (1.76%)，文字清晰
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
import cv2
import numpy as np
from inpainter.enhanced import remove_watermark as remove_watermark_enhanced


def detect_tuned_watermark(image: Image.Image) -> Image.Image:
    """
    基于诊断结果手动调优的水印检测
    """
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # === 1. 右上角区域 - 基于诊断的精确参数 ===
    # ROI 定义（覆盖水印区域）
    roi_x = int(width * 0.55)
    roi_y = int(height * 0.08)  # 避开顶部头发
    roi_w = int(width * 0.40)
    roi_h = int(height * 0.15)

    roi = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    roi_h_size, roi_w_size = roi.shape

    # CLAHE 增强
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    roi_enhanced = clahe.apply(roi)

    # 使用诊断确定的最佳阈值范围
    combined = np.zeros_like(roi)
    for thresh in [160, 175, 190, 205]:
        _, binary = cv2.threshold(roi_enhanced, thresh, 255, cv2.THRESH_BINARY)

        # 形态学膨胀连接
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=2)
        combined = cv2.bitwise_or(combined, dilated)

    # 形态学清理
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.erode(combined, kernel, iterations=1)

    # 找到所有轮廓，过滤噪点
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_mask = np.zeros_like(combined)
    valid_regions = []

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # 基于诊断：水印文字应该有合理的面积
        # 太小的是噪点，太大的是头发区域
        if 15 < area < 2000:
            # 位置过滤：水印在 ROI 的中下部
            if y > roi_h_size * 0.2:
                valid_regions.append((x, y, w, h, contour))
                cv2.drawContours(roi_mask, [contour], -1, (255), -1)

    # 如果有有效区域，合并并扩展
    if valid_regions:
        # 找到所有有效区域的合并 bounding box
        all_x = []
        all_y = []
        for x, y, w, h, _ in valid_regions:
            all_x.extend([x, x + w])
            all_y.extend([y, y + h])

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        # 扩展 bounding box
        expand_x = max(5, int((max_x - min_x) * 0.4))
        expand_y = max(5, int((max_y - min_y) * 0.6))

        x1 = max(0, min_x - expand_x)
        y1 = max(0, min_y - expand_y)
        x2 = min(roi_w_size, max_x + expand_x)
        y2 = min(roi_h_size, max_y + expand_y)

        # 在扩展区域内使用 Otsu 精确分割
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


def test_iteration_11():
    print("🧪 迭代 11 - 基于诊断的手动调优测试")
    print("=" * 60)

    test_image_path = Path("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
    image = Image.open(test_image_path)
    print(f"\n📷 图片：{image.size}")

    print("\n🔍 调优检测水印...")
    mask = detect_tuned_watermark(image)
    mask_arr = np.array(mask)
    white_pixels = np.sum(mask_arr > 0)
    percentage = (white_pixels / mask_arr.size) * 100
    print(f"   检测到水印区域：{white_pixels} 像素 ({percentage:.2f}%)")

    mask.save("output/mask_iter11.png")

    mask_rgb = Image.new('RGB', mask.size)
    mask_rgb_data = np.zeros((mask.size[1], mask.size[0], 3), dtype=np.uint8)
    mask_rgb_data[mask_arr > 0] = [255, 0, 0]
    mask_rgb = Image.fromarray(mask_rgb_data)
    mask_rgb.save("output/mask_iter11_visual.png")

    print("\n🎨 测试 inpainting 方法:")
    methods = ["aggressive", "multi", "ns"]
    for method in methods:
        print(f"   方法：{method}")
        result = remove_watermark_enhanced(image=image, mask=mask, method=method)
        result.save(f"output/cleaned_iter11_{method}.png")
        print(f"      ✅ 结果已保存")

    image.save("output/original_iter11.png")

    print("\n" + "=" * 60)
    print("✅ 迭代 11 完成！")


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    test_iteration_11()
