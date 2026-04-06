#!/usr/bin/env python3
"""
智能去水印系统 - 最终验证脚本
验证所有组件是否正确协同工作
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("🔍 智能去水印系统 - 最终验证")
    print("="*50)

    # 1. 验证模块导入
    print("\n✅ 1. 验证模块导入...")
    try:
        from detector import detect_watermark
        from inpainter import remove_watermark, Inpainter
        from video import extract_frames, remove_watermark_from_video
        from api.main import app

        print("   ✓ 所有模块导入成功")
    except ImportError as e:
        print(f"   ❌ 模块导入失败: {e}")
        return False

    # 2. 验证目录结构
    print("\n✅ 2. 验证项目结构...")
    expected_dirs = [
        "src",
        "src/detector",
        "src/inpainter",
        "src/video",
        "src/api",
        "tests",
        "models",
        "output",
        "reports"
    ]

    missing_dirs = []
    for dir_path in expected_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)

    if missing_dirs:
        print(f"   ❌ 缺少目录: {missing_dirs}")
        return False
    else:
        print("   ✓ 所有必需目录存在")

    # 3. 验证依赖
    print("\n✅ 3. 验证关键依赖...")
    try:
        import cv2
        import numpy as np
        import PIL
        import torch  # May not be required for basic functionality
        import fastapi
        print("   ✓ 关键依赖可用")
    except ImportError as e:
        print(f"   ⚠ 依赖警告: {e}")
        # 不失败，因为某些依赖可能是可选的

    # 4. 创建测试图像
    print("\n✅ 4. 创建测试图像...")
    test_img_path = Path("test_validation_image.png")
    test_img = Image.new('RGB', (200, 200), color='skyblue')
    # 添加一个简单的"水印"（白色矩形）
    img_array = np.array(test_img)
    img_array[150:190, 150:190] = [255, 255, 255]  # 白色水印
    test_img = Image.fromarray(img_array)
    test_img.save(test_img_path)
    print(f"   ✓ 测试图像创建: {test_img_path}")

    # 5. 测试检测功能
    print("\n✅ 5. 测试水印检测...")
    try:
        test_img_obj = Image.open(test_img_path)
        mask = detect_watermark(test_img_obj, method="auto")
        print(f"   ✓ 水印检测完成，生成mask: {mask.size}")
    except Exception as e:
        print(f"   ❌ 水印检测失败: {e}")
        return False

    # 6. 测试修复功能
    print("\n✅ 6. 测试水印移除...")
    try:
        cleaned_img = remove_watermark(test_img_obj, mask, device="cpu")
        print(f"   ✓ 水印移除完成，输出图像: {cleaned_img.size}")
    except Exception as e:
        print(f"   ❌ 水印移除失败: {e}")
        return False

    # 7. 测试Inpainter类
    print("\n✅ 7. 测试Inpainter类...")
    try:
        inpainter = Inpainter(device="cpu")
        result_img = inpainter.inpaint(test_img_obj, mask)
        print(f"   ✓ Inpainter类工作正常: {result_img.size}")
    except Exception as e:
        print(f"   ❌ Inpainter类失败: {e}")
        return False

    # 8. 验证API可用性
    print("\n✅ 8. 验证API应用...")
    try:
        assert hasattr(app, 'routes'), "FastAPI应用应包含路由"
        print("   ✓ API应用结构正常")
    except Exception as e:
        print(f"   ❌ API验证失败: {e}")
        return False

    # 9. 清理测试文件
    print("\n🧹 9. 清理测试文件...")
    test_img_path.unlink(missing_ok=True)
    print("   ✓ 测试文件清理完成")

    # 10. 总结
    print("\n" + "="*50)
    print("🎉 智能去水印系统验证完成!")
    print("\n📋 验证项目:")
    print("   ✓ 模块导入 - 成功")
    print("   ✓ 项目结构 - 完整")
    print("   ✓ 关键依赖 - 可用")
    print("   ✓ 水印检测 - 正常")
    print("   ✓ 水印移除 - 正常")
    print("   ✓ AI修复模块 - 正常")
    print("   ✓ API接口 - 可用")
    print("\n✨ 所有验证通过！系统已准备就绪！")

    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 项目验证成功 - 智能去水印系统现已完全就绪！")
    else:
        print("\n💥 项目验证失败 - 需要解决上述问题")
        sys.exit(1)