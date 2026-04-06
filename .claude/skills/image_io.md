---
name: image_io
description: 标准化的图片读取和保存方法，所有模块统一使用
---

## 读取图片
始终使用以下方式，自动处理路径和格式：
```python
import cv2
import numpy as np
from pathlib import Path

def load_image(path: str) -> np.ndarray:
    """加载图片，自动转换为 RGB numpy array"""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"图片不存在: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(img: np.ndarray, path: str) -> None:
    """保存图片，自动创建父目录"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
```

## 注意事项
- 项目统一用 RGB，不用 BGR
- 路径统一用 pathlib.Path，不用字符串拼接
