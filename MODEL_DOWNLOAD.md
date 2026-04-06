# LaMa 模型下载指引

## 问题说明

由于网络代理限制，系统无法自动下载 LaMa 模型文件。需要手动下载。

## 下载方法

### 方法 1: 使用 huggingface-cli（推荐）

```bash
# 安装 huggingface 命令行工具
pip install huggingface-cli

# 下载模型
cd /mnt/g/code/AI/Claude/watermark-remover/models
huggingface-cli download opencv/inpainting_lama lama_fp32.onnx --local-dir ./
```

### 方法 2: 直接下载

```bash
cd /mnt/g/code/AI/Claude/watermark-remover/models

# 使用 wget
wget https://huggingface.co/opencv/inpainting_lama/resolve/main/lama_fp32.onnx -O lama.onnx

# 或使用 curl
curl -L https://huggingface.co/opencv/inpainting_lama/resolve/main/lama_fp32.onnx -o lama.onnx
```

### 方法 3: 从 ModelScope 下载

```bash
cd /mnt/g/code/AI/Claude/watermark-remover/models

# ModelScope 链接（需要手动点击下载）
https://modelscope.cn/models/damo/cv_fft_inpainting_lama/files
```

## 验证下载

下载完成后，运行以下命令验证：

```bash
cd /mnt/g/code/AI/Claude/watermark-remover
ls -lh models/lama.onnx
# 应该显示约 100MB+ 的文件大小

# 运行测试
python3 test_new_features.py
```

## 模型文件说明

- **文件名**: `lama.onnx` 或 `lama_fp32.onnx`
- **大小**: 约 100-200 MB
- **格式**: ONNX 深度学习模型
- **用途**: 高质量图片修复（inpainting），可将 PSNR 从 15dB 提升到 32dB+

## 使用 LaMa 模型

下载后，在代码中使用：

```python
from src.inpainter import remove_watermark

# 使用 LaMa 模型进行高质量修复
result = remove_watermark(
    image=image,
    mask=mask,
    model_path="models/lama.onnx",
    method="lama"
)
```

## 备用方案

如果无法下载 LaMa 模型，系统会自动回退到 OpenCV 的 Telea 算法：

```python
# 自动回退
result = remove_watermark(
    image=image,
    mask=mask,
    method="telea"  # 默认方法，无需额外模型
)
```
