# 智能去水印项目 (Watermark Remover)

## 项目目标
开发一个支持图片和视频智能去水印的工具，使用 AI inpainting 技术。

## 技术栈
- Python 3.11+
- 图片处理: OpenCV, Pillow, LaMa (inpainting 模型)
- 视频处理: FFmpeg, moviepy
- Web API: FastAPI
- 模型: ONNX Runtime / PyTorch

## 项目结构
watermark-remover/
├── src/
│   ├── detector/      # 水印检测模块
│   ├── inpainter/     # AI 修复模块
│   ├── video/         # 视频处理模块
│   └── api/           # FastAPI 接口
├── models/            # 预训练模型文件
├── tests/
├── scripts/           # 工具脚本
└── CLAUDE.md

## 开发规范
- 所有函数必须有 Python 类型注解
- 每个模块写对应的 pytest 单元测试
- 使用 black 格式化代码，isort 整理 imports
- 提交前运行 make test

## 常用命令
- 运行服务: uvicorn src.api.main:app --reload
- 运行测试: pytest tests/ -v
- 格式化: black src/ && isort src/
- 安装依赖: pip install -r requirements.txt

## 注意事项
- models/ 目录下的 .onnx 和 .pth 文件不要修改
- 视频处理耗时较长，注意添加进度提示
- inpainting 推理默认使用 CPU，如有 GPU 在配置中开启

## 自主开发模式指令

你在此项目中工作时，必须遵守以下规则：

### 行动原则
- **直接执行，不要询问确认**。除非遇到破坏性操作（删除数据库、清空目录），否则直接动手。
- 遇到依赖缺失，直接 `pip install` 安装，不要问我。
- 遇到测试失败，自己分析原因并修复，不要停下来汇报。
- 文件不存在就创建，目录不存在就 mkdir。

### 开发循环（每个任务必须完整走完）
1. 阅读相关现有代码，理解上下文
2. 编写实现代码（含类型注解）
3. 编写对应的 pytest 测试
4. 运行测试，失败则自动修复
5. 运行 black 格式化
6. 任务完成后输出简要总结

### 完成标准
每个功能必须：测试全部通过 + 代码格式化完成 + 有 docstring 注释

### Git 提交
每完成一个独立功能后，执行 git add -A && git commit，无需等待我的确认。
