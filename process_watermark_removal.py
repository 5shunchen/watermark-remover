"""
完整的去水印处理流程
"""
import numpy as np
from PIL import Image
import sys
import os
from pathlib import Path

# 添加项目源码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from detector import detect_watermark
from inpainter import remove_watermark
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2

def run_complete_watermark_removal(input_path):
    """执行完整的去水印流程"""

    print("🔍 步骤1: 加载输入图像")
    original_img = Image.open(input_path)
    print(f"🖼️  原始图像尺寸: {original_img.size}, 模式: {original_img.mode}")

    print("\n🔎 步骤2: 检测水印")
    mask = detect_watermark(original_img, method="auto")
    print(f"🎭 生成的mask尺寸: {mask.size}, 模式: {mask.mode}")

    # 确保mask与原图尺寸一致
    if mask.size != original_img.size:
        mask = mask.resize(original_img.size, resample=Image.LANCZOS)

    print("\n✨ 步骤3: 使用LaMa模型修复图像")
    cleaned_img = remove_watermark(original_img, mask, device="cpu")
    print(f"✅ 修复后图像尺寸: {cleaned_img.size}, 模式: {cleaned_img.mode}")

    print("\n📁 步骤4: 保存结果")
    # 创建输出目录
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 保存结果
    mask_path = output_dir / "generated_mask.png"
    cleaned_path = output_dir / "cleaned_result.png"

    mask.save(mask_path)
    cleaned_img.save(cleaned_path)

    print(f"💾 Mask已保存至: {mask_path}")
    print(f"💾 结果已保存至: {cleaned_path}")

    print("\n📊 步骤5: 计算质量评分")
    # 将图像转换为numpy数组用于PSNR计算
    original_np = np.array(original_img)
    cleaned_np = np.array(cleaned_img)

    # 如果是彩色图像，确保维度匹配
    if len(original_np.shape) == 3 and len(cleaned_np.shape) == 3:
        # 彩色图像的PSNR计算
        psnr_score = psnr(original_np, cleaned_np, data_range=255)
    elif len(original_np.shape) == 2 and len(cleaned_np.shape) == 2:
        # 灰度图像的PSNR计算
        psnr_score = psnr(original_np, cleaned_np, data_range=255)
    else:
        # 如果维度不匹配，转为灰度再计算
        original_gray = cv2.cvtColor(original_np, cv2.COLOR_RGB2GRAY) if len(original_np.shape) == 3 else original_np
        cleaned_gray = cv2.cvtColor(cleaned_np, cv2.COLOR_RGB2GRAY) if len(cleaned_np.shape) == 3 else cleaned_np
        psnr_score = psnr(original_gray, cleaned_gray, data_range=255)

    print(f"📈 PSNR质量评分: {psnr_score:.2f} dB")

    print("\n🎯 完整的去水印流程完成！")

    # 返回结果路径和评分
    return {
        'mask_path': str(mask_path),
        'cleaned_path': str(cleaned_path),
        'psnr_score': psnr_score,
        'original_size': original_img.size
    }

if __name__ == "__main__":
    # 处理提供的输入文件
    input_file = "simple_test_original.png"

    if os.path.exists(input_file):
        print(f"🚀 开始处理: {input_file}")
        result = run_complete_watermark_removal(input_file)

        print(f"\n📋 处理摘要:")
        print(f"   输入文件: {input_file}")
        print(f"   生成mask: {result['mask_path']}")
        print(f"   输出文件: {result['cleaned_path']}")
        print(f"   图像尺寸: {result['original_size']}")
        print(f"   PSNR评分: {result['psnr_score']:.2f} dB")

        if result['psnr_score'] > 20:
            print("✅ 图像质量良好")
        else:
            print("⚠️  图像质量一般，可能需要进一步处理")
    else:
        print(f"❌ 错误: 找不到输入文件 {input_file}")