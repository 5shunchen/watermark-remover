# 水印去除使用指南

## 快速开始

```bash
cd /mnt/g/code/AI/Claude/watermark-remover
source venv/bin/activate

# 运行最佳配置测试
python test_final_best.py
```

## 输出文件说明

运行完成后，结果保存在 `output/` 目录：

| 文件 | 说明 |
|------|------|
| `output/final_mask.png` | 检测的水印掩码 |
| `output/final_cleaned_telea.png` | Telea 算法修复结果 |
| `output/final_cleaned_ns.png` | Navier-Stokes 算法修复结果 |
| `output/final_cleaned_multi.png` | 多通道修复结果（推荐） |
| `output/final_cleaned_aggressive.png` | 激进修复结果 |

## API 使用

```python
from PIL import Image
from detector import detect_watermark
from inpainter.enhanced import remove_watermark

# 加载图片
image = Image.open("your_image.png")

# 检测水印
mask = detect_watermark(image, method="text")

# 去除水印
result = remove_watermark(image, mask, method="multi")
result.save("cleaned.png")
```

## 最佳实践

### 检测参数调优

```python
# 更精确的检测（适合简单背景）
mask = detect_watermark(image, method="text")

# 更全面的检测（适合复杂背景，可能误检）
mask = detect_watermark(image, method="auto")
```

### 修复算法选择

| 算法 | 适用场景 | 速度 | 质量 |
|------|----------|------|------|
| `telea` | 小区域快速修复 | 快 | 中 |
| `ns` | 平滑区域 | 中 | 好 |
| `multi` | 通用场景（推荐） | 慢 | 最好 |
| `aggressive` | 顽固水印 | 最慢 | 好 |

## 当前限制

- **@小样燃剪水印**：右上角水印可检测并淡化，但无法完全去除
- **底部字幕**：可以完全去除 ✅
- **复杂背景**：修复后可能有轻微模糊

## 故障排除

### 检测不完整

尝试调整检测阈值或使用 `method="auto"` 让系统自动选择最佳方法。

### 修复后有痕迹

尝试使用 `method="multi"` 或 `method="aggressive"` 进行更彻底的修复。

---

*最后更新：2026-04-06*
