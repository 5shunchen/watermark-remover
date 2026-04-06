#!/usr/bin/env python3
"""
真实素材去水印测试
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
from detector import detect_watermark
from inpainter import remove_watermark, Inpainter
import time

def test_real_image():
    print("🧪 真实素材去水印测试")
    print("=" * 50)

    # 加载测试图片
    test_image_path = Path("test-photo/Snipaste_2026-04-04_23-05-17.PNG")
    if not test_image_path.exists():
        print(f"❌ 测试图片不存在：{test_image_path}")
        return

    print(f"\n📷 加载图片：{test_image_path}")
    image = Image.open(test_image_path)
    print(f"   图片尺寸：{image.size}")
    print(f"   图片模式：{image.mode}")

    # 测试不同检测方法
    print("\n🔍 测试水印检测方法:")

    methods = ["auto", "pattern", "edge", "color"]
    masks = {}

    for method in methods:
        try:
            print(f"\n   方法：{method}")
            mask = detect_watermark(image, method=method)
            masks[method] = mask

            # 计算检测到的水印区域
            mask_array = mask.resize(image.size).convert('L')
            import numpy as np
            mask_np = np.array(mask_array)
            white_pixels = np.sum(mask_np > 127)
            percentage = (white_pixels / mask_np.size) * 100
            print(f"      检测到水印区域：{white_pixels} 像素 ({percentage:.2f}%)")

            # 保存 mask
            mask.save(f"output/mask_{method}.png")
            print(f"      Mask 已保存：output/mask_{method}.png")

        except Exception as e:
            print(f"      ❌ 错误：{e}")

    # 使用最佳 mask 进行去水印
    print("\n🎨 执行去水印:")

    # 使用 auto 方法的结果
    if "auto" in masks:
        best_mask = masks["auto"]

        # 测试不同的 inpainting 方法
        inpaint_methods = ["telea", "ns"]

        for inpaint_method in inpaint_methods:
            print(f"\n   使用 {inpaint_method} 算法:")
            start_time = time.time()

            try:
                result = remove_watermark(
                    image=image,
                    mask=best_mask,
                    device="cpu",
                    method=inpaint_method
                )

                elapsed = time.time() - start_time
                print(f"      处理时间：{elapsed:.2f}秒")

                # 保存结果
                output_path = f"output/cleaned_{inpaint_method}.png"
                result.save(output_path)
                print(f"      结果已保存：{output_path}")

            except Exception as e:
                print(f"      ❌ 错误：{e}")

    print("\n" + "=" * 50)
    print("✅ 测试完成！")
    print("\n输出文件:")
    print("  - output/mask_*.png (检测的掩码)")
    print("  - output/cleaned_*.png (去水印结果)")

if __name__ == "__main__":
    # 创建输出目录
    Path("output").mkdir(exist_ok=True)
    test_real_image()
