"""
水印检测功能
"""
import numpy as np
from PIL import Image
import sys
import os
from pathlib import Path

# 添加项目源码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from detector import detect_watermark

def detect_watermark_in_image(image_path):
    """检测图像中的水印区域并生成mask"""

    print(f"🔍 读取图像文件: {image_path}")

    if not os.path.exists(image_path):
        print(f"❌ 错误: 文件 {image_path} 不存在")
        print("💡 提示: 请检查文件路径是否正确")
        return None

    # 读取图像
    img = Image.open(image_path)
    print(f"🖼️  图像信息: 尺寸={img.size}, 模式={img.mode}")

    # 检测水印
    print("\n🔎 正在检测水印区域...")
    mask = detect_watermark(img, method="auto")
    print(f"🎭 检测完成，生成mask: 尺寸={mask.size}, 模式={mask.mode}")

    # 确保输出目录存在
    mask_dir = Path("output/masks")
    mask_dir.mkdir(parents=True, exist_ok=True)

    # 保存mask
    filename = Path(image_path).stem
    mask_path = mask_dir / f"{filename}_mask.png"
    mask.save(mask_path)
    print(f"💾 mask已保存至: {mask_path}")

    # 分析检测结果
    mask_array = np.array(mask)
    detected_pixels = np.sum(mask_array > 127)  # 阈值以上像素数
    total_pixels = mask_array.size
    confidence = detected_pixels / total_pixels if total_pixels > 0 else 0

    print(f"\n📊 检测结果分析:")
    print(f"   检测到的水印像素数: {detected_pixels}")
    print(f"   总像素数: {total_pixels}")
    print(f"   水印覆盖比例: {confidence:.4f}")
    print(f"   检测置信度: {confidence:.4f}")

    # 如果检测到水印，估算可能的位置
    if detected_pixels > 0:
        # 获取非零像素的坐标
        coords = np.where(mask_array > 127)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()

            print(f"   估算水印边界框: ({x_min}, {y_min}, {x_max}, {y_max})")
            print(f"   估算水印位置: 左上角({x_min}, {y_min}), 右下角({x_max}, {y_max})")

    return {
        'mask_path': str(mask_path),
        'confidence': confidence,
        'detected_pixels': detected_pixels,
        'image_size': img.size
    }

if __name__ == "__main__":
    image_path = "simple_test_original.png"

    result = detect_watermark_in_image(image_path)

    if result:
        print(f"\n✅ 水印检测完成!")
        print(f"   文件: {image_path}")
        print(f"   Mask路径: {result['mask_path']}")
        print(f"   置信度: {result['confidence']:.4f}")
        print(f"   检测到像素: {result['detected_pixels']}")
    else:
        print(f"\n❌ 水印检测失败")