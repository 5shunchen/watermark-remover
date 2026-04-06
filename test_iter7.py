#!/usr/bin/env python3
"""
迭代 7 测试 - 使用更激进的检测策略
目标：完整检测"@小样燃剪"水印
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
import cv2
import numpy as np
from inpainter.enhanced import remove_watermark as remove_watermark_enhanced


def detect_aggressive_watermark(image: Image.Image) -> Image.Image:
    """
    激进的水印检测 - 确保覆盖完整水印区域
    """
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # === 1. 全图范围的低阈值检测 ===
    # 使用非常低的阈值检测暗淡的文字
    for thresh_val in [80, 100, 120, 140, 160, 180, 200]:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

        # 强形态学膨胀连接文字笔画
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=3)
        eroded = cv2.erode(dilated, kernel, iterations=2)

        # 找到轮廓
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            # 检查是否在右上角区域（扩大范围）
            is_top_right = (y < height * 0.35 and x > width * 0.45)
            # 检查是否在底部区域（字幕）
            is_bottom = y > height * 0.80

            # 放宽文字特征限制
            if 5 < area < 15000 and 0.1 < w / h < 20:
                if is_top_right or is_bottom:
                    cv2.drawContours(mask, [contour], -1, (255), -1)

    # === 2. Canny 边缘检测补充 ===
    edges = cv2.Canny(gray, 30, 100)
    edge_kernel = np.ones((7, 7), np.uint8)
    edge_dilated = cv2.dilate(edges, edge_kernel, iterations=3)
    _, edge_mask = cv2.threshold(edge_dilated, 50, 255, cv2.THRESH_BINARY)

    # 只保留右上角和底部
    edge_final = np.zeros_like(edge_mask)
    edge_final[0:int(height*0.35), int(width*0.45):] = edge_mask[0:int(height*0.35), int(width*0.45):]
    edge_final[int(height*0.80):, :] = edge_mask[int(height*0.80):, :]

    mask = cv2.bitwise_or(mask, edge_final)

    # === 3. 最终形态学清理 - 连接邻近区域 ===
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)

    return Image.fromarray(mask)


def test_iteration_7():
    print("🧪 迭代 7 - 激进检测测试")
    print("=" * 60)

    # 加载测试图片
    test_image_path = Path("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
    image = Image.open(test_image_path)
    print(f"\n📷 图片：{image.size}")

    # 使用激进的检测方法
    print("\n🔍 激进检测水印...")
    mask = detect_aggressive_watermark(image)
    mask_arr = np.array(mask)
    white_pixels = np.sum(mask_arr > 0)
    percentage = (white_pixels / mask_arr.size) * 100
    print(f"   检测到水印区域：{white_pixels} 像素 ({percentage:.2f}%)")

    # 保存 mask
    mask.save("output/mask_iter7.png")

    # 创建可视化 mask
    mask_rgb = Image.new('RGB', mask.size)
    mask_rgb_data = np.zeros((mask.size[1], mask.size[0], 3), dtype=np.uint8)
    mask_rgb_data[mask_arr > 0] = [255, 0, 0]
    mask_rgb = Image.fromarray(mask_rgb_data)
    mask_rgb.save("output/mask_iter7_visual.png")

    # 测试不同的 inpainting 方法
    print("\n🎨 测试不同 inpainting 方法:")

    methods = ["ns", "multi", "aggressive"]
    for method in methods:
        print(f"\n   方法：{method}")
        result = remove_watermark_enhanced(
            image=image,
            mask=mask,
            method=method
        )
        result.save(f"output/cleaned_iter7_{method}.png")
        print(f"      ✅ 结果已保存：output/cleaned_iter7_{method}.png")

    # 保存原图
    image.save("output/original_iter7.png")

    print("\n" + "=" * 60)
    print("✅ 迭代 7 完成！请检查结果图片")


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    test_iteration_7()
