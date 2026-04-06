---
name: quality_metrics
description: 计算图像质量指标 PSNR 和 SSIM
---

每次评估去水印效果时，用以下代码计算指标：
```python
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def evaluate_quality(original: np.ndarray, result: np.ndarray) -> dict:
    psnr = peak_signal_noise_ratio(original, result)
    ssim = structural_similarity(original, result, channel_axis=2)
    return {
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 4),
        "quality": "good" if psnr > 32 else "poor"
    }
```

合格标准：PSNR > 32dB，SSIM > 0.85
低于标准时，在日志中打印警告并记录到 reports/quality_log.csv
