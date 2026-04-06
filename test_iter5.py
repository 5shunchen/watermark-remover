#!/usr/bin/env python3
"""
迭代 5 测试 - 使用增强版 inpainter 和多策略检测
目标：完美去除"@小样燃剪"水印
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
import cv2
import numpy as np
from detector import detect_watermark
from inpainter.enhanced import remove_watermark as remove_watermark_enhanced


def detect_optimized_watermark(image: Image.Image) -> Image.Image:
    """
    优化的水印检测 - 结合多种策略
    """
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # === 1. 右上角水印检测 - 扩大区域 ===
    roi_x = int(width * 0.55)
    roi_y = 0
    roi_w = int(width * 0.45)
    roi_h = int(height * 0.18)

    roi = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # 多阈值检测
    for thresh_val in [140, 160, 180, 200]:
        _, roi_binary = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        roi_dilated = cv2.dilate(roi_binary, kernel, iterations=2)

        contours, _ = cv2.findContours(roi_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 15 < area < 8000:
                roi_mask = np.zeros_like(roi_dilated)
                cv2.drawContours(roi_mask, [contour], -1, (255), -1)
                mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = cv2.bitwise_or(
                    mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w], roi_mask
                )

    # === 2. HSV 颜色空间检测 ===
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 低饱和度 + 高亮度 = 可能的白色水印
    _, v_mask = cv2.threshold(v, 180, 255, cv2.THRESH_BINARY)
    _, s_mask = cv2.threshold(s, 60, 255, cv2.THRESH_BINARY_INV)
    color_mask = cv2.bitwise_and(v_mask, s_mask)

    # 只保留右上角区域
    color_roi = color_mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    kernel = np.ones((3, 3), np.uint8)
    color_dilated = cv2.dilate(color_roi, kernel, iterations=2)

    contours, _ = cv2.findContours(color_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if 15 < area < 8000:
            cv2.drawContours(mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w],
                           [contour], -1, (255), -1)

    # === 3. 底部字幕检测 ===
    sub_y = int(height * 0.85)
    subtitle_roi = gray[sub_y:, :]

    for thresh_val in [150, 170, 190]:
        _, sub_binary = cv2.threshold(subtitle_roi, thresh_val, 255, cv2.THRESH_BINARY)
        sub_kernel = np.ones((5, 5), np.uint8)
        sub_dilated = cv2.dilate(sub_binary, sub_kernel, iterations=2)
        sub_eroded = cv2.erode(sub_dilated, sub_kernel, iterations=1)
        mask[sub_y:, :] = cv2.bitwise_or(mask[sub_y:, :], sub_eroded)

    # === 4. 形态学清理 ===
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return Image.fromarray(mask)


def test_iteration_5():
    print("🧪 迭代 5 - 增强版 inpainter 测试")
    print("=" * 60)

    # 加载测试图片
    test_image_path = Path("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
    image = Image.open(test_image_path)
    print(f"\n📷 图片：{image.size}")

    # 使用优化的检测方法
    print("\n🔍 检测水印...")
    mask = detect_optimized_watermark(image)
    mask_arr = np.array(mask)
    white_pixels = np.sum(mask_arr > 0)
    percentage = (white_pixels / mask_arr.size) * 100
    print(f"   检测到水印区域：{white_pixels} 像素 ({percentage:.2f}%)")

    # 保存 mask
    mask.save("output/mask_iter5.png")

    # 创建可视化 mask
    mask_rgb = Image.new('RGB', mask.size)
    mask_rgb_data = np.zeros((mask.size[1], mask.size[0], 3), dtype=np.uint8)
    mask_rgb_data[mask_arr > 0] = [255, 0, 0]
    mask_rgb = Image.fromarray(mask_rgb_data)
    mask_rgb.save("output/mask_iter5_visual.png")

    # 测试不同的 inpainting 方法
    print("\n🎨 测试不同 inpainting 方法:")

    methods = ["telea", "ns", "multi", "aggressive"]
    for method in methods:
        print(f"\n   方法：{method}")
        result = remove_watermark_enhanced(
            image=image,
            mask=mask,
            method=method
        )
        result.save(f"output/cleaned_iter5_{method}.png")
        print(f"      ✅ 结果已保存：output/cleaned_iter5_{method}.png")

    # 保存原图
    image.save("output/original_iter5.png")

    print("\n" + "=" * 60)
    print("✅ 迭代 5 完成！请检查结果图片")


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    test_iteration_5()
