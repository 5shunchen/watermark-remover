#!/usr/bin/env python3
"""
真实素材去水印测试 - v3 最终优化
针对 `@小样燃剪` 类型水印精确检测
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
import cv2
import numpy as np
from detector import detect_watermark
from inpainter import remove_watermark


def detect_corner_watermark(image: Image.Image) -> Image.Image:
    """
    精确检测右上角文字水印
    """
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # === 1. 右上角水印检测 ===
    # 定义右上角 ROI (更精确的区域)
    roi_x = int(width * 0.6)
    roi_y = 0
    roi_w = int(width * 0.4)
    roi_h = int(height * 0.2)

    roi = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # 自适应阈值检测亮色文字
    _, roi_thresh = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY)

    # 形态学操作连接文字笔画
    kernel = np.ones((3, 3), np.uint8)
    roi_dilated = cv2.dilate(roi_thresh, kernel, iterations=2)
    roi_eroded = cv2.erode(roi_dilated, kernel, iterations=1)

    # 找到轮廓，过滤小噪点
    contours, _ = cv2.findContours(roi_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_cleaned = np.zeros_like(roi_eroded)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 20:  # 过滤太小的噪点
            cv2.drawContours(roi_cleaned, [contour], -1, (255), -1)

    # 复制到完整 mask
    mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = roi_cleaned

    # === 2. 底部字幕检测 ===
    # 底部 12% 区域
    sub_y = int(height * 0.88)
    subtitle_roi = gray[sub_y:, :]

    # 检测底部亮色文字
    _, sub_thresh = cv2.threshold(subtitle_roi, 200, 255, cv2.THRESH_BINARY)

    # 形态学操作
    sub_kernel = np.ones((5, 5), np.uint8)
    sub_dilated = cv2.dilate(sub_thresh, sub_kernel, iterations=2)
    sub_eroded = cv2.erode(sub_dilated, sub_kernel, iterations=1)

    # 复制到 mask
    mask[sub_y:, :] = sub_eroded

    # === 3. 清理 ===
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    return Image.fromarray(mask)


def test_final():
    print("🧪 最终优化版去水印测试")
    print("=" * 50)

    # 加载测试图片
    test_image_path = Path("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
    image = Image.open(test_image_path)
    print(f"\n📷 图片：{image.size}, {image.mode}")

    # 使用优化的检测方法
    print("\n🔍 精确检测水印...")
    mask = detect_corner_watermark(image)
    mask.save("output/mask_final.png")

    mask_arr = np.array(mask)
    print(f"   检测到水印区域：{np.sum(mask_arr > 0)} 像素 ({np.sum(mask_arr > 0) / mask_arr.size * 100:.2f}%)")

    # 保存 mask 可视化版本
    mask_rgb = Image.new('RGB', mask.size)
    mask_rgb_data = np.zeros((mask.size[1], mask.size[0], 3), dtype=np.uint8)
    mask_arr = np.array(mask)
    mask_rgb_data[mask_arr > 0] = [255, 0, 0]  # 红色标记
    mask_rgb = Image.fromarray(mask_rgb_data)
    mask_rgb.save("output/mask_final_visual.png")
    print("   Mask 已保存：output/mask_final_visual.png")

    # 执行去水印
    print("\n🎨 执行去水印...")

    # 测试不同算法
    for method in ["telea", "ns"]:
        result = remove_watermark(
            image=image,
            mask=mask,
            device="cpu",
            method=method
        )
        result.save(f"output/cleaned_final_{method}.png")
        print(f"   ✅ [{method}] 结果已保存：output/cleaned_final_{method}.png")

    # 保存原图对比
    image.save("output/original.png")

    print("\n" + "=" * 50)
    print("✅ 最终测试完成！")
    print("\n输出文件:")
    print("  - output/original.png (原图)")
    print("  - output/mask_final*.png (检测的掩码)")
    print("  - output/cleaned_final_*.png (去水印结果)")

if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    test_final()
