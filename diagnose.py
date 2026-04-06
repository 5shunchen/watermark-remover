#!/usr/bin/env python3
"""
诊断工具 - 可视化不同阈值的检测效果
"""
from PIL import Image
import cv2
import numpy as np
from pathlib import Path

# 加载图片
image = Image.open("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
img_array = np.array(image)
img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

height, width = gray.shape

# 定义右上角 ROI
roi_x = int(width * 0.5)
roi_y = 0
roi_w = int(width * 0.5)
roi_h = int(height * 0.2)

roi = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

# 使用 CLAHE 增强
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
roi_enhanced = clahe.apply(roi)

# 保存增强前后的 ROI
Image.fromarray(roi).save("output/diag_roi_orig.png")
Image.fromarray(roi_enhanced).save("output/diag_roi_enhanced.png")

print("=== 原始 ROI 阈值检测 ===")
for thresh in [100, 120, 140, 160, 180, 200, 220]:
    _, binary = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)
    count = np.sum(binary > 0)
    pct = count / binary.size * 100
    print(f"阈值 {thresh}: {count} 像素 ({pct:.2f}%)")
    cv2.imwrite(f"output/diag_orig_t{thresh}.png", binary)

print("\n=== CLAHE 增强后 ROI 阈值检测 ===")
for thresh in [120, 140, 160, 180, 200, 220, 240]:
    _, binary = cv2.threshold(roi_enhanced, thresh, 255, cv2.THRESH_BINARY)
    count = np.sum(binary > 0)
    pct = count / binary.size * 100
    print(f"阈值 {thresh}: {count} 像素 ({pct:.2f}%)")
    cv2.imwrite(f"output/diag_enhanced_t{thresh}.png", binary)

print("\n✅ 诊断图片已保存到 output/")
