#!/usr/bin/env python3
"""
分析原图右上角水印区域的像素特征
"""
from PIL import Image
import numpy as np
import cv2

# 加载图片
image = Image.open("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
img_array = np.array(image)
img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

height, width = gray.shape
print(f"图片尺寸：{width}x{height}")

# 定义右上角 ROI
roi_x = int(width * 0.6)
roi_y = 0
roi_w = int(width * 0.4)
roi_h = int(height * 0.2)

roi = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
roi_color = img_bgr[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

print(f"\nROI 区域：x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
print(f"ROI 尺寸：{roi_w}x{roi_h}")

# 分析 ROI 的像素统计特性
print(f"\nROI 像素统计:")
print(f"  最小值：{np.min(roi)}")
print(f"  最大值：{np.max(roi)}")
print(f"  平均值：{np.mean(roi):.2f}")
print(f"  标准差：{np.std(roi):.2f}")
print(f"  中位数：{np.median(roi):.2f}")

# 分析直方图
hist = np.histogram(roi, bins=256, range=(0, 256))
print(f"\n直方图分析:")
for i in range(255, 200, -10):
    count = hist[0][i]
    if count > 0:
        print(f"  亮度 {i}: {count} 像素")

# 保存 ROI 用于视觉参考
roi_pil = Image.fromarray(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB))
roi_pil.save("output/roi_analysis.png")
print("\n✅ ROI 已保存到 output/roi_analysis.png")

# 测试不同阈值的效果
print("\n=== 测试不同阈值 ===")
for thresh in [120, 140, 160, 180, 200, 220]:
    _, binary = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)
    white_count = np.sum(binary > 0)
    percentage = (white_count / binary.size) * 100
    print(f"阈值 {thresh}: {white_count} 像素 ({percentage:.2f}%)")
