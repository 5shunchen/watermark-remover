#!/usr/bin/env python3
"""
迭代 16 测试 - PyTorch 频域修复
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
import cv2
import numpy as np
from detector import detect_watermark
from inpainter.lama_inpainter import remove_watermark_lama
from inpainter.enhanced import remove_watermark as remove_watermark_cv


def test_iteration_16():
    print("🧪 迭代 16 - PyTorch 频域修复测试")
    print("=" * 60)

    test_image_path = Path("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
    image = Image.open(test_image_path)
    print(f"\n📷 图片：{image.size}")

    # 检测水印
    print("\n🔍 检测水印...")
    mask = detect_watermark(image, method="text")
    mask_arr = np.array(mask)
    white_pixels = np.sum(mask_arr > 0)
    percentage = (white_pixels / mask_arr.size) * 100
    print(f"   检测到水印区域：{white_pixels} 像素 ({percentage:.2f}%)")

    mask.save("output/mask_iter16.png")

    # 测试不同方法
    print("\n🎨 测试修复方法:")

    # OpenCV 方法
    methods_cv = ["telea", "ns", "multi", "aggressive"]
    for method in methods_cv:
        result = remove_watermark_cv(image=image, mask=mask, method=method)
        result.save(f"output/cleaned_iter16_cv_{method}.png")
        print(f"   ✅ [CV-{method}] 已保存")

    # PyTorch 方法
    print("\n   PyTorch 方法:")
    result_pt = remove_watermark_lama(image=image, mask=mask, use_pytorch=True)
    result_pt.save("output/cleaned_iter16_pytorch.png")
    print(f"   ✅ [PyTorch-Frequency] 已保存")

    image.save("output/original_iter16.png")

    print("\n" + "=" * 60)
    print("✅ 迭代 16 完成！")


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    test_iteration_16()
