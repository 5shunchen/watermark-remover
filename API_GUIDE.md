# Watermark Remover API 使用指南

## 快速开始

### 1. 启动服务

```bash
uvicorn src.api.main:app --reload --port 8000
```

访问 http://localhost:8000 查看主页

## API 端点

### 健康检查

```bash
# 检查 API 状态
curl http://localhost:8000/api/v1/health

# 获取使用统计
curl http://localhost:8000/api/v1/stats
```

### 图片去水印

#### 方法一：直接上传并处理（推荐）

```bash
curl -X POST http://localhost:8000/api/v1/process \
  -F "file=@image.jpg" \
  -F "detection_method=auto" \
  -F "device=cpu" \
  -o result.png
```

#### 方法二：先上传后处理

```bash
# 1. 上传文件
UPLOAD_RESPONSE=$(curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@image.jpg")

# 提取 file_id
FILE_ID=$(echo $UPLOAD_RESPONSE | jq -r '.file_id')

# 2. 处理文件
curl -X POST http://localhost:8000/api/v1/process/$FILE_ID \
  -F "detection_method=auto" \
  -o result.png
```

### 水印检测

#### 检测并返回 mask

```bash
# 上传并检测
curl -X POST http://localhost:8000/api/v1/detect-upload \
  -F "file=@image.jpg" \
  -F "method=auto" \
  -o mask.png
```

### 文件管理

```bash
# 列出所有文件
curl http://localhost:8000/api/v1/files

# 获取文件信息
curl http://localhost:8000/api/v1/files/{file_id}

# 删除文件
curl -X DELETE http://localhost:8000/api/v1/files/{file_id}
```

## 检测方法

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| `auto` | 智能自动检测（推荐） | 大多数情况 |
| `color` | 颜色阈值检测 | 彩色水印 |
| `edge` | 边缘检测 | 文字水印 |
| `corners` | 角点检测 | 角落水印 |
| `pattern` | 图案分析 | 规则水印 |
| `text` | 文字检测（推荐） | @用户名等文字水印 |

## 修复算法

| 算法 | 速度 | 质量 | 说明 |
|------|------|------|------|
| `telea` | 快 | 高 | 默认推荐 |
| `ns` | 慢 | 很高 | Navier-Stokes |
| `ns_original` | 中 | 高 | 原始 NS 算法 |

## Python SDK 示例

```python
import requests

# 上传图片并处理
def remove_watermark(image_path: str, output_path: str):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {
            'detection_method': 'auto',
            'device': 'cpu'
        }
        
        response = requests.post(
            'http://localhost:8000/api/v1/process',
            files=files,
            data=data
        )
        
        result = response.json()
        
        if result['success']:
            # 下载处理后的图片
            download_url = result['output_url']
            img_response = requests.get(f'http://localhost:8000{download_url}')
            
            with open(output_path, 'wb') as out_f:
                out_f.write(img_response.content)
            
            print(f"处理完成！耗时：{result['processing_time']}s")
        else:
            print(f"处理失败：{result['message']}")

# 使用示例
remove_watermark('input.jpg', 'output.png')
```

## cURL 完整示例

```bash
#!/bin/bash

# Watermark Remover API 测试脚本

API_URL="http://localhost:8000/api/v1"

echo "=== 1. 健康检查 ==="
curl -s "$API_URL/health" | jq .

echo -e "\n=== 2. 查看统计 ==="
curl -s "$API_URL/stats" | jq .

echo -e "\n=== 3. 上传图片 ==="
UPLOAD_RESPONSE=$(curl -s -X POST "$API_URL/upload" \
  -F "file=@test.jpg")
echo $UPLOAD_RESPONSE | jq .

FILE_ID=$(echo $UPLOAD_RESPONSE | jq -r '.file_id')

echo -e "\n=== 4. 处理图片 ==="
PROCESS_RESPONSE=$(curl -s -X POST "$API_URL/process/$FILE_ID" \
  -F "detection_method=auto" \
  -F "device=cpu")
echo $PROCESS_RESPONSE | jq .

echo -e "\n=== 5. 下载结果 ==="
OUTPUT_URL=$(echo $PROCESS_RESPONSE | jq -r '.output_url')
curl -s "$API_URL$OUTPUT_URL" -o result.png
echo "结果已保存到 result.png"

echo -e "\n=== 6. 清理文件 ==="
curl -s -X DELETE "$API_URL/files/$FILE_ID" | jq .

echo -e "\n完成！"
```

## 文档地址

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **自定义文档**: http://localhost:8000/api-docs

## 错误处理

API 返回标准 HTTP 状态码：

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求错误（文件格式/大小问题） |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

错误响应格式：

```json
{
  "detail": "错误描述信息"
}
```

## 限制

- 文件大小：最大 10MB
- 支持格式：JPG, PNG, WebP, GIF
- 处理超时：默认 60 秒

## 生产环境建议

1. **使用数据库**：替换内存存储为 Redis/PostgreSQL
2. **添加认证**：API Key 或 JWT 验证
3. **限流**：使用速率限制防止滥用
4. **队列系统**：使用 Celery + Redis 处理长时间任务
5. **对象存储**：使用 S3/OSS 存储文件
6. **CDN**：加速静态文件分发
