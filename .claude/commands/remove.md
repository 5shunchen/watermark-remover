---
description: 对图片或视频执行完整的去水印流程
---
执行完整去水印流程：
1. 先检测水印 (调用 detector 模块)
2. 生成 inpainting mask
3. 运行 LaMa 模型修复图像
4. 输出结果到 output/ 目录
5. 对比原图和结果，输出 PSNR 质量评分

输入文件: $ARGUMENTS
