#!/usr/bin/env python3
"""
真实素材去水印测试 - 优化版
针对 `@小样燃剪` 类型水印优化检测
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
import cv2
import numpy as np
from detector import detect_watermark
from inpainter import remove_watermark

def detect_watermark_manual(image: Image.Image) -> Image.Image:
    """
    手动优化水印检测 - 针对右上角文字水印
    """
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    # 创建空白 mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # 重点检测右上角区域 (水印常见位置)
    # 定义右上角 ROI
    roi_h = height // 5
    roi_w = width // 3
    roi = gray[0:roi_h, width - roi_w:width]

    # 对 ROI 区域进行阈值检测 (检测亮色文字)
    # 水印通常是白色或亮色
    _, roi_thresh = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)

    # 形态学操作连接文字
    kernel = np.ones((3, 3), np.uint8)
    roi_dilated = cv2.dilate(roi_thresh, kernel, iterations=2)
    roi_eroded = cv2.erode(roi_dilated, kernel, iterations=1)

    # 将检测到的区域复制到完整 mask
    mask[0:roi_h, width - roi_w:width] = roi_eroded

    # 检测底部字幕区域 (避免破坏)
    # 底部 15% 区域
    subtitle_roi = gray[int(height * 0.85):, :]
    _, subtitle_thresh = cv2.threshold(subtitle_roi, 220, 255, cv2.THRESH_BINARY)

    # 底部字幕也检测出来
    subtitle_mask = np.zeros_like(subtitle_thresh)
    subtitle_kernel = np.ones((5, 5), np.uint8)
    subtitle_dilated = cv2.dilate(subtitle_thresh, subtitle_kernel, iterations=3)
    subtitle_eroded = cv2.erode(subtitle_dilated, subtitle_kernel, iterations=2)
    subtitle_mask[:] = subtitle_eroded

    # 合并两个区域的检测
    mask[int(height * 0.85):, :] = np.maximum(
        mask[int(height * 0.85):, :],
        subtitle_mask
    )

    # 应用形态学清理
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return Image.fromarray(mask)


def test_optimized():
    print("🧪 优化版去水印测试")
    print("=" * 50)

    # 加载测试图片
    test_image_path = Path("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
    image = Image.open(test_image_path)

    print(f"\n📷 图片：{image.size}, {image.mode}")

    # 方法 1: 使用 auto 检测
    print("\n🔍 方法 1: auto 检测")
    mask_auto = detect_watermark(image, method="auto")
    mask_auto.save("output/mask_optimized_auto.png")

    # 方法 2: 使用手动优化检测
    print("\n🔍 方法 2: 手动优化检测")
    mask_manual = detect_watermark_manual(image)
    mask_manual.save("output/mask_optimized_manual.png")

    # 比较两个 mask
    mask_auto_arr = np.array(mask_auto)
    mask_manual_arr = np.array(mask_manual)

    print(f"\n   Auto 检测：{np.sum(mask_auto_arr > 0)} 像素")
    print(f"   手动检测：{np.sum(mask_manual_arr > 0)} 像素")

    # 使用手动优化的 mask 进行去水印
    print("\n🎨 执行去水印 (使用手动优化的 mask):")

    # 使用 ns 算法 (Navier-Stokes, 质量更高)
    result = remove_watermark(
        image=image,
        mask=mask_manual,
        device="cpu",
        method="ns"
    )
    result.save("output/cleaned_optimized.png")
    print("   ✅ 结果已保存：output/cleaned_optimized.png")

    # 保存 mask 用于对比
    mask_auto.convert('RGB').save("output/mask_optimized_auto_rgb.png")
    mask_manual.convert('RGB').save("output/mask_optimized_manual_rgb.png")

    print("\n" + "=" * 50)
    print("✅ 优化测试完成！")

if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    test_optimized()
