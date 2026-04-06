#!/usr/bin/env python3
"""
迭代 13 测试 - HSV 颜色空间检测
水印特征：白色/低饱和度 + 高亮度
头发特征：有颜色/高饱和度
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
import cv2
import numpy as np
from inpainter.enhanced import remove_watermark as remove_watermark_enhanced


def detect_hsv_watermark(image: Image.Image) -> Image.Image:
    """
    使用 HSV 颜色空间检测水印
    """
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    h, s, v = cv2.split(hsv)

    # === 1. 右上角区域 - HSV 检测 ===
    roi_x = int(width * 0.55)
    roi_y = int(height * 0.05)
    roi_w = int(width * 0.45)
    roi_h = int(height * 0.25)

    # 提取 ROI 的 HSV 通道
    s_roi = s[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    v_roi = v[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    gray_roi = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # 水印特征：低饱和度 + 高亮度
    # 白色/灰色文字：S < 50, V > 150
    _, s_mask = cv2.threshold(s_roi, 60, 255, cv2.THRESH_BINARY_INV)
    _, v_mask = cv2.threshold(v_roi, 160, 255, cv2.THRESH_BINARY)

    # 结合两个条件
    hsv_mask = cv2.bitwise_and(s_mask, v_mask)

    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    hsv_dilated = cv2.dilate(hsv_mask, kernel, iterations=2)
    hsv_eroded = cv2.erode(hsv_dilated, kernel, iterations=1)

    # 找到轮廓并过滤
    contours, _ = cv2.findContours(hsv_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_mask = np.zeros_like(hsv_eroded)

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # 水印文字应该在 ROI 的下半部分
        if area > 10 and y > roi_h * 0.2:
            cv2.drawContours(roi_mask, [contour], -1, (255), -1)

    # === 2. 补充：CLAHE 增强检测 ===
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_enhanced = clahe.apply(gray_roi)

    for thresh in [180, 200, 220]:
        _, binary = cv2.threshold(roi_enhanced, thresh, 255, cv2.THRESH_BINARY)
        roi_mask = cv2.bitwise_or(roi_mask, binary)

    # 形态学清理
    kernel = np.ones((3, 3), np.uint8)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)

    # 复制到完整 mask
    mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = roi_mask

    # === 3. 底部字幕检测 ===
    sub_y = int(height * 0.85)
    subtitle_roi = gray[sub_y:, :]

    for thresh in [150, 170, 190]:
        _, sub_binary = cv2.threshold(subtitle_roi, thresh, 255, cv2.THRESH_BINARY)
        sub_kernel = np.ones((5, 5), np.uint8)
        sub_dilated = cv2.dilate(sub_binary, sub_kernel, iterations=2)
        sub_eroded = cv2.erode(sub_dilated, sub_kernel, iterations=1)
        mask[sub_y:, :] = cv2.bitwise_or(mask[sub_y:, :], sub_eroded)

    # === 4. 最终清理 ===
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return Image.fromarray(mask)


def test_iteration_13():
    print("🧪 迭代 13 - HSV 颜色空间检测测试")
    print("=" * 60)

    test_image_path = Path("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
    image = Image.open(test_image_path)
    print(f"\n📷 图片：{image.size}")

    print("\n🔍 HSV 检测水印...")
    mask = detect_hsv_watermark(image)
    mask_arr = np.array(mask)
    white_pixels = np.sum(mask_arr > 0)
    percentage = (white_pixels / mask_arr.size) * 100
    print(f"   检测到水印区域：{white_pixels} 像素 ({percentage:.2f}%)")

    mask.save("output/mask_iter13.png")

    mask_rgb = Image.new('RGB', mask.size)
    mask_rgb_data = np.zeros((mask.size[1], mask.size[0], 3), dtype=np.uint8)
    mask_rgb_data[mask_arr > 0] = [255, 0, 0]
    mask_rgb = Image.fromarray(mask_rgb_data)
    mask_rgb.save("output/mask_iter13_visual.png")

    print("\n🎨 测试 inpainting 方法:")
    methods = ["aggressive", "multi", "ns"]
    for method in methods:
        print(f"   方法：{method}")
        result = remove_watermark_enhanced(image=image, mask=mask, method=method)
        result.save(f"output/cleaned_iter13_{method}.png")
        print(f"      ✅ 结果已保存")

    image.save("output/original_iter13.png")

    print("\n" + "=" * 60)
    print("✅ 迭代 13 完成！")


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    test_iteration_13()
