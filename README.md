# 智能去水印项目 (Watermark Remover)

## 项目概述
这是一个基于AI技术的智能去水印工具，支持图片和视频的水印检测与移除。项目采用模块化设计，主要包括水印检测、AI修复和视频处理等功能模块。

## 功能特性

### 1. 水印检测 (Detector)
- **颜色检测**: 基于HSV色彩空间的颜色阈值检测
- **边缘检测**: 基于边缘特征的水印定位
- **角点检测**: 基于Harris角点检测算法
- **模式检测**: 基于对比度和纹理特征的通用检测
- **自适应检测**: 组合多种算法提高检测精度

### 2. AI修复 (Inpainter)
- **多种修复算法**: 支持OpenCV内置算法和扩展AI模型接口
- **掩码预处理**: 对检测到的掩码进行形态学处理
- **多设备支持**: 支持CPU/GPU加速推理
- **多格式兼容**: 支持RGB、RGBA、灰度等多种图像模式

### 3. 视频处理 (Video)
- **帧提取**: 支持按间隔提取视频帧
- **批量处理**: 逐帧处理视频内容
- **视频重构**: 将处理后的帧重新组合成视频
- **进度追踪**: 实时显示处理进度

### 4. Web接口 (API)
- **FastAPI框架**: 高性能异步Web API
- **RESTful设计**: 符合REST标准的API设计
- **文件上传**: 支持图片/视频文件上传
- **实时处理**: 上传后即时处理并返回结果

## 项目结构
```
watermark-remover/
├── src/                    # 源代码
│   ├── detector/          # 水印检测模块
│   │   └── __init__.py    # 检测算法实现
│   ├── inpainter/         # AI修复模块  
│   │   └── __init__.py    # 修复算法实现
│   ├── video/             # 视频处理模块
│   │   └── __init__.py    # 视频处理功能
│   └── api/               # Web API接口
│       └── main.py        # FastAPI应用入口
├── tests/                 # 测试用例
│   ├── test_detector.py   # 检测模块测试
│   ├── test_inpainter.py  # 修复模块测试
│   └── test_video.py      # 视频模块测试
├── models/                # 预训练模型
├── reports/               # 评估报告输出
├── output/                # 处理结果输出
├── requirements.txt       # 项目依赖
├── Makefile              # 构建脚本
└── CLAUDE.md             # 项目文档
```

## 快速开始

### 1. 环境搭建
```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行服务
```bash
# 启动API服务
make run-api
# 或者
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. 运行测试
```bash
# 运行所有测试
make test
# 或者
pytest tests/ -v
```

### 4. 使用示例
```python
from src.detector import detect_watermark
from src.inpainter import remove_watermark
from PIL import Image

# 加载图像
image = Image.open("input.jpg")

# 检测水印
mask = detect_watermark(image, method="auto")

# 移除水印
cleaned_image = remove_watermark(image, mask, device="cpu")

# 保存结果
cleaned_image.save("output.jpg")
```

## 性能指标
根据测试评估：
- **PSNR**: >30dB 表示优秀的图像质量
- **SSIM**: >0.98 表示高度的结构相似性
- **检测准确率**: 在测试集中表现良好
- **处理速度**: CPU模式下单张图片约需几秒

## API接口

### 检测水印
POST /detect/
- 参数: file (图像文件)
- 返回: 水印掩码图像

### 移除水印
POST /remove/
- 参数: file (图像文件), detection_method (检测方法), device (处理设备)
- 返回: 移除水印后的图像

### 完整处理
POST /process/
- 参数: file (图像文件), detection_method (检测方法), device (处理设备)
- 返回: 完整处理后的图像

## 技术栈
- **编程语言**: Python 3.11+
- **图像处理**: OpenCV, Pillow, scikit-image
- **深度学习**: PyTorch (预留), ONNX Runtime
- **Web框架**: FastAPI, Uvicorn
- **视频处理**: MoviePy
- **测试框架**: pytest
- **代码格式化**: Black, isort

## 开发规范
- 所有函数必须有Python类型注解
- 每个模块需编写对应的pytest单元测试
- 使用black格式化代码，isort整理imports
- 提交前运行make test

## 部署说明
1. 生产环境部署建议使用GPU加速
2. 可配置并发处理多个请求
3. 建议设置适当的资源限制和超时时间

## 当前状态
✅ 项目已完整实现所有计划功能
✅ 所有模块均已通过测试
✅ API接口可正常使用
✅ 模型评估结果良好
✅ 文档完整
