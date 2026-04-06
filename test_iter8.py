#!/usr/bin/env python3
"""
迭代 8 测试 - 智能精确检测
策略：结合局部对比度分析和文字特征检测
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
import cv2
import numpy as np
from inpainter.enhanced import remove_watermark as remove_watermark_enhanced


def detect_smart_watermark(image: Image.Image) -> Image.Image:
    """
    智能水印检测 - 平衡检测率和精确度
    """
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # === 1. 定义精确的 ROI 区域 ===
    # 右上角区域（水印位置）
    tr_x = int(width * 0.55)
    tr_y = int(height * 0.05)
    tr_w = int(width * 0.40)
    tr_h = int(height * 0.15)

    # 底部区域（字幕位置）
    b_y = int(height * 0.85)

    # === 2. 右上角水印检测 ===
    roi = gray[tr_y:tr_y + tr_h, tr_x:tr_x + tr_w]

    # 使用局部自适应阈值（更适合检测与背景有对比的文字）
    roi_adaptive = cv2.adaptiveThreshold(
        roi, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=5
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

        # 文字特征过滤
        if 20 < area < 5000 and 0.3 < w / h < 10:
            cv2.drawContours(roi_mask, [contour], -1, (255), -1)

    # 复制到完整 mask
    mask[tr_y:tr_y + tr_h, tr_x:tr_x + tr_w] = roi_mask

    # === 3. 固定阈值补充检测（捕捉暗淡文字）===
    for thresh_val in [120, 150, 180]:
        _, roi_binary = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        roi_dilated = cv2.dilate(roi_binary, kernel, iterations=2)

        contours, _ = cv2.findContours(roi_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 15 < area < 5000:
                temp_mask = np.zeros_like(roi_dilated)
                cv2.drawContours(temp_mask, [contour], -1, (255), -1)
                mask[tr_y:tr_y + tr_h, tr_x:tr_x + tr_w] = cv2.bitwise_or(
                    mask[tr_y:tr_y + tr_h, tr_x:tr_x + tr_w], temp_mask
                )

    # === 4. 底部字幕检测 ===
    subtitle_roi = gray[b_y:, :]

    for thresh_val in [140, 160, 180]:
        _, sub_binary = cv2.threshold(subtitle_roi, thresh_val, 255, cv2.THRESH_BINARY)
        sub_kernel = np.ones((5, 5), np.uint8)
        sub_dilated = cv2.dilate(sub_binary, sub_kernel, iterations=2)
        sub_eroded = cv2.erode(sub_dilated, sub_kernel, iterations=1)
        mask[b_y:, :] = cv2.bitwise_or(mask[b_y:, :], sub_eroded)

    # === 5. 最终形态学清理 ===
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return Image.fromarray(mask)


def test_iteration_8():
    print("🧪 迭代 8 - 智能精确检测测试")
    print("=" * 60)

    # 加载测试图片
    test_image_path = Path("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
    image = Image.open(test_image_path)
    print(f"\n📷 图片：{image.size}")

    # 使用智能检测方法
    print("\n🔍 智能检测水印...")
    mask = detect_smart_watermark(image)
    mask_arr = np.array(mask)
    white_pixels = np.sum(mask_arr > 0)
    percentage = (white_pixels / mask_arr.size) * 100
    print(f"   检测到水印区域：{white_pixels} 像素 ({percentage:.2f}%)")

    # 保存 mask
    mask.save("output/mask_iter8.png")

    # 创建可视化 mask
    mask_rgb = Image.new('RGB', mask.size)
    mask_rgb_data = np.zeros((mask.size[1], mask.size[0], 3), dtype=np.uint8)
    mask_rgb_data[mask_arr > 0] = [255, 0, 0]
    mask_rgb = Image.fromarray(mask_rgb_data)
    mask_rgb.save("output/mask_iter8_visual.png")

    # 测试 aggressive inpainting 方法
    print("\n🎨 测试 inpainting 方法:")

    methods = ["aggressive", "multi"]
    for method in methods:
        print(f"\n   方法：{method}")
        result = remove_watermark_enhanced(
            image=image,
            mask=mask,
            method=method
        )
        result.save(f"output/cleaned_iter8_{method}.png")
        print(f"      ✅ 结果已保存：output/cleaned_iter8_{method}.png")

    # 保存原图
    image.save("output/original_iter8.png")

    print("\n" + "=" * 60)
    print("✅ 迭代 8 完成！请检查结果图片")


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    test_iteration_8()
