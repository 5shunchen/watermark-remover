#!/usr/bin/env python3
"""
迭代 17 测试 - 使用 lama-cleaner (LaMa 模型)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
import cv2
import numpy as np
from detector import detect_watermark


def remove_watermark_lama_cleaner(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Use lama-cleaner library for AI inpainting
    """
    from lama_cleaner.model_manager import ModelManager
    from lama_cleaner.schema import Config

    # Initialize model manager
    model = ModelManager(
        name="lama",
        device="cpu",
        hf_model_id="docker://justinzyf/lama",
    )

    # Convert PIL to numpy
    image_np = np.array(image)
    mask_np = np.array(mask)

    # Ensure mask is binary
    mask_np = (mask_np > 127).astype(np.uint8) * 255

    # Create config
    config = Config(
        ldm_steps=20,
        ldm_strength=1.0,
        hd_strategy="Original",
        hd_strategy_crop_margin=128,
        hd_strategy_crop_trigger_size=128,
        hd_strategy_resize_limit=128,
    )

    # Run inpainting
    result = model(image_np, mask_np, config)

    return Image.fromarray(result)


def test_iteration_17():
    print("🧪 迭代 17 - LaMa 模型测试")
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

    mask.save("output/mask_iter17.png")

    # 使用 LaMa 修复
    print("\n🎨 LaMa 模型修复中...")
    try:
        result = remove_watermark_lama_cleaner(image, mask)
        result.save("output/cleaned_iter17_lama.png")
        print("   ✅ [LaMa] 结果已保存")
    except Exception as e:
        print(f"   ❌ LaMa 失败：{e}")
        print("   使用 OpenCV 备用方案...")
        from inpainter.enhanced import remove_watermark
        result = remove_watermark(image, mask, method="ns")
        result.save("output/cleaned_iter17_cv.png")
        print("   ✅ [CV-ns] 备用结果已保存")

    image.save("output/original_iter17.png")

    print("\n" + "=" * 60)
    print("✅ 迭代 17 完成！")


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    test_iteration_17()
