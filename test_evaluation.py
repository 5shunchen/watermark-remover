"""
模型评估脚本
"""
import numpy as np
from PIL import Image
import sys
import os
from pathlib import Path
import csv
from datetime import datetime

# 添加项目源码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from detector import detect_watermark
from inpainter import remove_watermark
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2

def evaluate_model_performance():
    """评估模型性能"""

    print("🔍 开始模型性能评估...")

    # 创建测试数据（如果不存在测试数据目录）
    fixtures_dir = Path("tests/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # 创建一些测试图像
    test_images = []

    # 创建带水印的测试图像1
    img1 = np.zeros((200, 200, 3), dtype=np.uint8)
    # 添加背景图案
    for i in range(200):
        for j in range(200):
            img1[i, j] = [(i * 128 // 200) % 256, (j * 128 // 200) % 256, 100]
    # 添加红色水印（右下角）
    img1[150:190, 150:190] = [255, 0, 0]
    test_img1_path = fixtures_dir / "test_with_red_watermark.png"
    Image.fromarray(img1).save(test_img1_path)
    test_images.append(test_img1_path)

    # 创建带水印的测试图像2
    img2 = np.random.randint(50, 200, (150, 150, 3), dtype=np.uint8)
    # 添加白色文字样水印（中心区域）
    img2[70:90, 60:130] = [255, 255, 255]
    test_img2_path = fixtures_dir / "test_with_white_watermark.png"
    Image.fromarray(img2).save(test_img2_path)
    test_images.append(test_img2_path)

    # 创建无水印的基准图像
    base_img = np.random.randint(100, 200, (150, 150, 3), dtype=np.uint8)
    base_img_path = fixtures_dir / "baseline_no_watermark.png"
    Image.fromarray(base_img).save(base_img_path)

    print(f"📊 准备了 {len(test_images)} 个测试图像")

    # 结果存储
    results = []

    for img_path in test_images:
        print(f"\n🧪 测试图像: {img_path.name}")

        # 加载原始带水印的图像
        original_img = Image.open(img_path)
        original_np = np.array(original_img)

        # 创建对应的干净基准图像（模拟无水印版本）
        # 对于我们的测试，我们将使用原始图像作为基准
        baseline_img = Image.open(base_img_path)
        baseline_np = np.array(baseline_img)

        # 如果图像大小不匹配，调整大小
        if original_img.size != baseline_img.size:
            baseline_img = baseline_img.resize(original_img.size, Image.Resampling.LANCZOS)
            baseline_np = np.array(baseline_img)

        # 检测水印
        print("   🔍 检测水印...")
        mask = detect_watermark(original_img, method="auto")
        mask_np = np.array(mask)

        # 统计检测到的水印像素数
        detected_watermark_pixels = np.sum(mask_np > 127)

        # 去除水印
        print("   ✨ 执行去水印...")
        cleaned_img = remove_watermark(original_img, mask, device="cpu")
        cleaned_np = np.array(cleaned_img)

        # 计算指标
        # PSNR (峰值信噪比) - 数值越高表示图像质量越好
        if len(original_np.shape) == 3 and len(cleaned_np.shape) == 3:
            psnr_val = psnr(original_np, cleaned_np, data_range=255)
        else:
            psnr_val = psnr(original_np, cleaned_np, data_range=255)

        # SSIM (结构相似性) - 数值越接近1表示图像越相似
        if len(original_np.shape) == 3:
            ssim_val = ssim(original_np, cleaned_np, channel_axis=-1, data_range=255)
        else:
            ssim_val = ssim(original_np, cleaned_np, data_range=255)

        # 也计算与基准图像的比较（如果需要）
        if len(baseline_np.shape) == 3:
            baseline_ssim = ssim(baseline_np, cleaned_np, channel_axis=-1, data_range=255)
        else:
            baseline_ssim = ssim(baseline_np, cleaned_np, data_range=255)

        # 保存结果图像
        results_dir = Path("reports")
        results_dir.mkdir(exist_ok=True)

        # 保存mask和cleaned图像
        mask.save(results_dir / f"{img_path.stem}_mask.png")
        cleaned_img.save(results_dir / f"{img_path.stem}_cleaned.png")

        # 存储结果
        result = {
            'image_name': img_path.name,
            'original_size': original_img.size,
            'detected_watermark_pixels': detected_watermark_pixels,
            'psnr': psnr_val,
            'ssim_vs_original': ssim_val,
            'ssim_vs_baseline': baseline_ssim,
            'mask_path': str(results_dir / f"{img_path.stem}_mask.png"),
            'cleaned_path': str(results_dir / f"{img_path.stem}_cleaned.png")
        }
        results.append(result)

        print(f"   📊 检测到水印像素: {detected_watermark_pixels}")
        print(f"   📈 PSNR: {psnr_val:.2f} dB")
        print(f"   📈 SSIM (vs original): {ssim_val:.4f}")
        print(f"   📈 SSIM (vs baseline): {baseline_ssim:.4f}")

    # 生成评估报告
    print(f"\n📝 生成评估报告...")

    # 创建报告目录
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    # 生成Markdown报告
    report_path = report_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 模型评估报告\n\n")
        f.write(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 测试概况\n\n")
        f.write(f"- 测试图像数量: {len(results)}\n")
        f.write("- 评估指标: PSNR (dB), SSIM (结构相似性)\n\n")

        f.write("## 详细结果\n\n")
        f.write("| 图像名称 | 检测到的水印像素 | PSNR (dB) | SSIM (vs original) | SSIM (vs baseline) |\n")
        f.write("|----------|------------------|-----------|--------------------|--------------------|\n")

        for result in results:
            f.write(f"| {result['image_name']} | {result['detected_watermark_pixels']} | {result['psnr']:.2f} | {result['ssim_vs_original']:.4f} | {result['ssim_vs_baseline']:.4f} |\n")

        # 计算平均值
        avg_psnr = sum(r['psnr'] for r in results) / len(results) if results else 0
        avg_ssim_orig = sum(r['ssim_vs_original'] for r in results) / len(results) if results else 0
        avg_ssim_base = sum(r['ssim_vs_baseline'] for r in results) / len(results) if results else 0

        f.write(f"\n## 平均性能指标\n\n")
        f.write(f"- 平均 PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"- 平均 SSIM (vs original): {avg_ssim_orig:.4f}\n")
        f.write(f"- 平均 SSIM (vs baseline): {avg_ssim_base:.4f}\n\n")

        f.write("## 总体评价\n\n")
        if avg_psnr > 30:
            f.write("✅ **图像质量优秀** - PSNR > 30dB 表示失真很小\n")
        elif avg_psnr > 20:
            f.write("⚠️  **图像质量一般** - 20dB < PSNR < 30dB 表示有一定失真\n")
        else:
            f.write("❌ **图像质量较差** - PSNR < 20dB 表示失真较大\n")

        if avg_ssim_orig > 0.8:
            f.write("✅ **结构保持良好** - SSIM > 0.8 表示结构相似性高\n")
        else:
            f.write("⚠️  **结构有所损失** - SSIM < 0.8 表示结构变化较大\n")

    print(f"✅ 评估报告已保存至: {report_path}")

    # 生成CSV报告用于数据分析
    csv_path = report_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_name', 'original_width', 'original_height',
                     'detected_watermark_pixels', 'psnr', 'ssim_vs_original', 'ssim_vs_baseline']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            row = {
                'image_name': result['image_name'],
                'original_width': result['original_size'][0],
                'original_height': result['original_size'][1],
                'detected_watermark_pixels': result['detected_watermark_pixels'],
                'psnr': result['psnr'],
                'ssim_vs_original': result['ssim_vs_original'],
                'ssim_vs_baseline': result['ssim_vs_baseline']
            }
            writer.writerow(row)

    print(f"✅ CSV数据报告已保存至: {csv_path}")

    # 显示总体统计
    print(f"\n📈 总体统计:")
    print(f"   平均 PSNR: {avg_psnr:.2f} dB")
    print(f"   平均 SSIM (vs original): {avg_ssim_orig:.4f}")
    print(f"   平均 SSIM (vs baseline): {avg_ssim_base:.4f}")

    return {
        'results': results,
        'report_path': str(report_path),
        'csv_path': str(csv_path),
        'average_psnr': avg_psnr,
        'average_ssim_original': avg_ssim_orig,
        'average_ssim_baseline': avg_ssim_base
    }


if __name__ == "__main__":
    print("🚀 开始模型评估流程...")
    evaluation_results = evaluate_model_performance()

    print(f"\n🎯 模型评估完成!")
    print(f"   评估报告: {evaluation_results['report_path']}")
    print(f"   数据文件: {evaluation_results['csv_path']}")
    print(f"   性能指标: PSNR={evaluation_results['average_psnr']:.2f}dB, SSIM={evaluation_results['average_ssim_original']:.4f}")