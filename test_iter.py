#!/usr/bin/env python3
"""
水印去除迭代优化测试
针对"@小样燃剪"水印专项优化
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
import numpy as np
from detector import detect_watermark
from inpainter import remove_watermark

def test_iteration(iteration_num: int):
    print(f"\n{'='*60}")
    print(f"🔄 迭代 {iteration_num} - 水印检测与去除测试")
    print(f"{'='*60}")

    # 加载测试图片
    test_image_path = Path("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
    image = Image.open(test_image_path)
    print(f"\n📷 图片：{image.size}")

    # 测试不同检测方法
    methods = ["auto", "text", "corner_focus", "pattern", "edge"]

    for method in methods:
        print(f"\n🔍 检测方法：{method}")
        try:
            mask = detect_watermark(image, method=method)
            mask_arr = np.array(mask)
            white_pixels = np.sum(mask_arr > 0)
            percentage = (white_pixels / mask_arr.size) * 100

            print(f"   检测区域：{white_pixels} 像素 ({percentage:.2f}%)")

            # 保存 mask
            mask.save(f"output/mask_{method}_iter{iteration_num}.png")

            # 执行去水印
            result = remove_watermark(image, mask, method="ns")
            result.save(f"output/cleaned_{method}_iter{iteration_num}.png")

            print(f"   ✅ 结果已保存")

        except Exception as e:
            print(f"   ❌ 错误：{e}")

    # 保存原图
    image.save("output/original.png")

    print(f"\n✅ 迭代 {iteration_num} 完成!")

if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    import sys
    iteration = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    test_iteration(iteration)
