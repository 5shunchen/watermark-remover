#!/usr/bin/env python3
"""
水印检测精确度测试 - 迭代 4
目标：精确检测"@小样燃剪"水印，避免过度检测
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
import cv2
import numpy as np


def detect_precise_watermark(image: Image.Image) -> Image.Image:
    """
    精确检测"@小样燃剪"类型水印
    策略：聚焦右上角，使用自适应阈值和文字特征分析
    """
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # === 1. 精确聚焦右上角 ===
    # ROI: 右上角 25% 宽度 x 20% 高度
    roi_x = int(width * 0.65)
    roi_y = 0
    roi_w = int(width * 0.35)
    roi_h = int(height * 0.2)

    roi = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # 自适应阈值 - 更适合检测文字
    roi_adaptive = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    roi_dilated = cv2.dilate(roi_adaptive, kernel, iterations=2)
    roi_eroded = cv2.erode(roi_dilated, kernel, iterations=1)

    # 轮廓分析
    contours, _ = cv2.findContours(roi_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_mask = np.zeros_like(roi_eroded)

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # 精确的文字特征过滤
        if 30 < area < 3000 and 0.3 < w / h < 8:
            cv2.drawContours(roi_mask, [contour], -1, (255), -1)

    # 复制到完整 mask
    mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = roi_mask

    # === 2. 固定阈值补充检测（检测半透明文字）===
    for thresh_val in [140, 160, 180]:
        _, roi_binary = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            if 20 < area < 5000 and 0.2 < w / h < 10:
                # 确保在 ROI 范围内
                roi_contour = contour.copy()
                cv2.drawContours(mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w],
                               [roi_contour], -1, (255), -1)

    # === 3. 底部字幕检测 ===
    sub_y = int(height * 0.88)
    subtitle_roi = gray[sub_y:, :]

    _, sub_binary = cv2.threshold(subtitle_roi, 180, 255, cv2.THRESH_BINARY)
    sub_kernel = np.ones((5, 5), np.uint8)
    sub_dilated = cv2.dilate(sub_binary, sub_kernel, iterations=2)
    sub_eroded = cv2.erode(sub_dilated, sub_kernel, iterations=1)
    mask[sub_y:, :] = sub_eroded

    # === 4. 最终清理 ===
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    return Image.fromarray(mask)


def test_precise():
    print("🧪 迭代 4 - 精确检测测试")
    print("=" * 50)

    # 加载测试图片
    test_image_path = Path("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
    image = Image.open(test_image_path)
    print(f"\n📷 图片：{image.size}")

    # 精确检测
    mask = detect_precise_watermark(image)
    mask_arr = np.array(mask)
    white_pixels = np.sum(mask_arr > 0)
    percentage = (white_pixels / mask_arr.size) * 100

    print(f"\n🔍 检测到水印区域：{white_pixels} 像素 ({percentage:.2f}%)")

    # 保存 mask 和可视化
    mask.save("output/mask_precise_iter4.png")

    # 创建可视化 mask
    mask_rgb = Image.new('RGB', mask.size)
    mask_rgb_data = np.zeros((mask.size[1], mask.size[0], 3), dtype=np.uint8)
    mask_rgb_data[mask_arr > 0] = [255, 0, 0]
    mask_rgb = Image.fromarray(mask_rgb_data)
    mask_rgb.save("output/mask_precise_iter4_visual.png")

    # 保存原图参考
    image.save("output/original_ref.png")

    print("\n✅ 迭代 4 完成！检查输出图片评估效果")


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    test_precise()
