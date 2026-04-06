#!/usr/bin/env python3
"""
迭代 10 测试 - 完整检测策略
使用极低阈值 + 严格位置过滤 + 形态学闭合
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
import cv2
import numpy as np
from inpainter.enhanced import remove_watermark as remove_watermark_enhanced


def detect_complete_watermark(image: Image.Image) -> Image.Image:
    """
    完整水印检测策略
    """
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # === 1. 右上角区域检测 ===
    roi_x = int(width * 0.5)
    roi_y = 0
    roi_w = int(width * 0.5)
    roi_h = int(height * 0.25)

    roi = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    roi_h_size, roi_w_size = roi.shape

    # 多尺度阈值检测
    combined = np.zeros_like(roi)

    # Scale 1: CLAHE 增强
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    roi_enhanced = clahe.apply(roi)

    for thresh in [140, 160, 180, 200, 220]:
        _, binary = cv2.threshold(roi_enhanced, thresh, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_or(combined, binary)

    # Scale 2: 原始图低阈值
    for thresh in [100, 120, 140, 160]:
        _, binary = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_or(combined, binary)

    # 形态学闭合连接区域
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.dilate(combined, kernel, iterations=3)
    combined = cv2.erode(combined, kernel, iterations=2)

    # 只保留 ROI 底部的连通区域（水印位置）
    # 找到所有连通分量
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, connectivity=8)

    roi_mask = np.zeros_like(combined)
    for i in range(1, num_labels):  # 跳过背景
        x, y, w, h, area = stats[i]

        # 只保留底部区域的分量（y 坐标在 ROI 的 30% 以下）
        if y > roi_h_size * 0.2 and area > 20:
            roi_mask[labels == i] = 255

    # 找到最终 mask 的 bounding box 并扩展
    coords = cv2.findNonZero(roi_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)

        # 扩展覆盖完整水印
        expand_x = max(10, int(w * 0.3))
        expand_y = max(10, int(h * 0.8))

        x1 = max(0, x - expand_x)
        y1 = max(0, y - expand_y)
        x2 = min(roi_w_size, x + w + expand_x)
        y2 = min(roi_h_size, y + h + expand_y)

        # 在扩展区域内使用 Otsu 阈值
        region = roi[y1:y2, x1:x2]
        if region.size > 50:
            _, otsu = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            roi_mask[y1:y2, x1:x2] = cv2.bitwise_or(roi_mask[y1:y2, x1:x2], otsu)

    # 复制到完整 mask
    mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = roi_mask

    # === 2. 底部字幕检测 ===
    sub_y = int(height * 0.85)
    subtitle_roi = gray[sub_y:, :]

    for thresh in [140, 160, 180]:
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


def test_iteration_10():
    print("🧪 迭代 10 - 完整检测策略测试")
    print("=" * 60)

    test_image_path = Path("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
    image = Image.open(test_image_path)
    print(f"\n📷 图片：{image.size}")

    print("\n🔍 完整检测水印...")
    mask = detect_complete_watermark(image)
    mask_arr = np.array(mask)
    white_pixels = np.sum(mask_arr > 0)
    percentage = (white_pixels / mask_arr.size) * 100
    print(f"   检测到水印区域：{white_pixels} 像素 ({percentage:.2f}%)")

    mask.save("output/mask_iter10.png")

    mask_rgb = Image.new('RGB', mask.size)
    mask_rgb_data = np.zeros((mask.size[1], mask.size[0], 3), dtype=np.uint8)
    mask_rgb_data[mask_arr > 0] = [255, 0, 0]
    mask_rgb = Image.fromarray(mask_rgb_data)
    mask_rgb.save("output/mask_iter10_visual.png")

    print("\n🎨 测试 inpainting 方法:")
    methods = ["aggressive", "multi", "ns"]
    for method in methods:
        print(f"   方法：{method}")
        result = remove_watermark_enhanced(image=image, mask=mask, method=method)
        result.save(f"output/cleaned_iter10_{method}.png")
        print(f"      ✅ 结果已保存")

    image.save("output/original_iter10.png")

    print("\n" + "=" * 60)
    print("✅ 迭代 10 完成！")


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    test_iteration_10()
