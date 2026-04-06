---
description: 用测试集评估当前模型效果
---
运行模型评估：
1. 读取 tests/fixtures/ 下所有测试图片
2. 执行去水印流程
3. 计算 PSNR / SSIM 指标
4. 生成 Markdown 格式的评估报告输出到 reports/
