#!/usr/bin/env python3
"""
最终测试 - 使用当前最佳配置
检测：融合 CLAHE + HSV
修复：multi-pass 策略
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
import cv2
import numpy as np
from detector import detect_watermark
from inpainter.enhanced import remove_watermark as remove_watermark_enhanced


def test_final():
    print("🧪 最终测试 - 当前最佳配置")
    print("=" * 60)

    test_image_path = Path("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
    image = Image.open(test_image_path)
    print(f"\n📷 图片：{image.size}")

    # 使用 text 方法检测（融合 CLAHE + HSV）
    print("\n🔍 检测水印...")
    mask = detect_watermark(image, method="text")
    mask_arr = np.array(mask)
    white_pixels = np.sum(mask_arr > 0)
    percentage = (white_pixels / mask_arr.size) * 100
    print(f"   检测到水印区域：{white_pixels} 像素 ({percentage:.2f}%)")

    mask.save("output/final_mask.png")

    # 执行去水印
    print("\n🎨 执行去水印...")

    methods = ["telea", "ns", "multi", "aggressive"]
    for method in methods:
        result = remove_watermark_enhanced(image=image, mask=mask, method=method)
        result.save(f"output/final_cleaned_{method}.png")
        print(f"   ✅ [{method}] 结果已保存")

    image.save("output/final_original.png")

    print("\n" + "=" * 60)
    print("✅ 最终测试完成！")
    print("\n📁 输出文件:")
    print("   - output/final_mask.png (检测 mask)")
    print("   - output/final_cleaned_*.png (去水印结果)")
    print("   - output/final_original.png (原图)")
    print("\n📝 技术报告：docs/WATERMARK_REMOVAL_REPORT.md")


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    test_final()
