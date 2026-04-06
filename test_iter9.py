#!/usr/bin/env python3
"""
迭代 9 测试 - 检测 + 扩展策略
核心思想：检测到文字后，扩展 mask 覆盖完整水印区域
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
import cv2
import numpy as np
from inpainter.enhanced import remove_watermark as remove_watermark_enhanced


def detect_expand_watermark(image: Image.Image) -> Image.Image:
    """
    检测 + 扩展策略：先检测文字种子，然后扩展覆盖完整水印
    """
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # === 1. 右上角区域检测 ===
    roi_x = int(width * 0.55)
    roi_y = int(height * 0.05)
    roi_w = int(width * 0.45)
    roi_h = int(height * 0.20)

    roi = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    roi_h_size, roi_w_size = roi.shape

    # CLAHE 增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_enhanced = clahe.apply(roi)

    # 检测文字种子
    seed_mask = np.zeros_like(roi_enhanced)
    for thresh_val in [160, 180, 200]:
        _, roi_binary = cv2.threshold(roi_enhanced, thresh_val, 255, cv2.THRESH_BINARY)
        seed_mask = cv2.bitwise_or(seed_mask, roi_binary)

    # 形态学清理
    kernel = np.ones((3, 3), np.uint8)
    seed_dilated = cv2.dilate(seed_mask, kernel, iterations=2)
    seed_eroded = cv2.erode(seed_dilated, kernel, iterations=1)

    # 找到文字种子的 bounding box
    contours, _ = cv2.findContours(seed_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 找到所有轮廓的合并 bounding box
        all_points = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:  # 小种子也保留
                x, y, w, h = cv2.boundingRect(contour)
                all_points.append((x, y, x + w, y + h))

        if all_points:
            # 计算合并的 bounding box
            min_x = min(p[0] for p in all_points)
            min_y = min(p[1] for p in all_points)
            max_x = max(p[2] for p in all_points)
            max_y = max(p[3] for p in all_points)

            # 扩展 bounding box 覆盖完整水印
            expand_x = int((max_x - min_x) * 0.3)  # 水平扩展 30%
            expand_y = int((max_y - min_y) * 0.5)  # 垂直扩展 50%

            min_x = max(0, min_x - expand_x)
            min_y = max(0, min_y - expand_y)
            max_x = min(roi_w_size, max_x + expand_x)
            max_y = min(roi_h_size, max_y + expand_y)

            # 在扩展的 bounding box 内填充 mask
            seed_mask[min_y:max_y, min_x:max_x] = 255

    # 复制种子 mask 到完整 mask
    mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = seed_mask

    # === 2. 底部字幕检测 ===
    sub_y = int(height * 0.85)
    subtitle_roi = gray[sub_y:, :]

    for thresh_val in [140, 160, 180]:
        _, sub_binary = cv2.threshold(subtitle_roi, thresh_val, 255, cv2.THRESH_BINARY)
        sub_kernel = np.ones((5, 5), np.uint8)
        sub_dilated = cv2.dilate(sub_binary, sub_kernel, iterations=2)
        sub_eroded = cv2.erode(sub_dilated, sub_kernel, iterations=1)
        mask[sub_y:, :] = cv2.bitwise_or(mask[sub_y:, :], sub_eroded)

    # === 3. 最终形态学清理 ===
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return Image.fromarray(mask)


def test_iteration_9():
    print("🧪 迭代 9 - 检测 + 扩展策略测试")
    print("=" * 60)

    # 加载测试图片
    test_image_path = Path("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
    image = Image.open(test_image_path)
    print(f"\n📷 图片：{image.size}")

    # 使用检测 + 扩展方法
    print("\n🔍 检测 + 扩展水印...")
    mask = detect_expand_watermark(image)
    mask_arr = np.array(mask)
    white_pixels = np.sum(mask_arr > 0)
    percentage = (white_pixels / mask_arr.size) * 100
    print(f"   检测到水印区域：{white_pixels} 像素 ({percentage:.2f}%)")

    # 保存 mask
    mask.save("output/mask_iter9.png")

    # 创建可视化 mask
    mask_rgb = Image.new('RGB', mask.size)
    mask_rgb_data = np.zeros((mask.size[1], mask.size[0], 3), dtype=np.uint8)
    mask_rgb_data[mask_arr > 0] = [255, 0, 0]
    mask_rgb = Image.fromarray(mask_rgb_data)
    mask_rgb.save("output/mask_iter9_visual.png")

    # 测试 inpainting 方法
    print("\n🎨 测试 inpainting 方法:")

    methods = ["aggressive", "multi", "ns"]
    for method in methods:
        print(f"\n   方法：{method}")
        result = remove_watermark_enhanced(
            image=image,
            mask=mask,
            method=method
        )
        result.save(f"output/cleaned_iter9_{method}.png")
        print(f"      ✅ 结果已保存：output/cleaned_iter9_{method}.png")

    # 保存原图
    image.save("output/original_iter9.png")

    print("\n" + "=" * 60)
    print("✅ 迭代 9 完成！请检查结果图片")


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    test_iteration_9()
