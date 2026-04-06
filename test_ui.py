#!/usr/bin/env python3
"""
UI 自动化测试脚本 - 测试去水印功能
"""

import requests
import base64
from io import BytesIO
from PIL import Image

BASE_URL = "http://localhost:8000"

def create_test_image():
    """创建一个简单的测试图片"""
    img = Image.new('RGB', (100, 100), color='white')
    # 添加一些内容模拟水印
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 30, 30], fill='red')
    draw.text((50, 50), "Watermark", fill='blue')
    return img

def test_health():
    """测试健康检查"""
    print("\n=== 测试健康检查 ===")
    response = requests.get(f"{BASE_URL}/api/v1/health")
    print(f"状态码：{response.status_code}")
    print(f"响应：{response.json()}")
    assert response.status_code == 200
    print("✓ 健康检查通过")

def test_video_page():
    """测试视频页面"""
    print("\n=== 测试视频页面 ===")
    response = requests.get(f"{BASE_URL}/video")
    print(f"状态码：{response.status_code}")
    assert response.status_code == 200
    assert "视频去水印" in response.text
    print("✓ 视频页面加载成功")

def test_main_page():
    """测试主页"""
    print("\n=== 测试主页 ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"状态码：{response.status_code}")
    assert response.status_code == 200
    print("✓ 主页加载成功")

def test_image_upload_and_process():
    """测试图片上传和处理"""
    print("\n=== 测试图片上传和处理 ===")

    # 创建测试图片
    test_img = create_test_image()
    img_buffer = BytesIO()
    test_img.save(img_buffer, format='JPEG')
    img_buffer.seek(0)

    files = {'file': ('test.jpg', img_buffer, 'image/jpeg')}
    data = {
        'detection_method': 'auto',
        'device': 'cpu',
        'return_mask': False
    }

    print("发送 POST 请求到 /api/v1/process...")
    response = requests.post(f"{BASE_URL}/api/v1/process", files=files, data=data)
    print(f"状态码：{response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"响应：{result}")

        if result.get('success'):
            print("✓ 图片处理成功")

            # 下载处理后的图片
            output_url = result.get('output_url')
            if output_url:
                download_response = requests.get(f"{BASE_URL}{output_url}")
                print(f"下载状态码：{download_response.status_code}")
                if download_response.status_code == 200:
                    print("✓ 处理后的图片下载成功")
                else:
                    print("✗ 下载失败")
        else:
            print(f"✗ 处理失败：{result.get('message', 'Unknown error')}")
    else:
        print(f"✗ 请求失败：{response.status_code}")
        print(f"响应内容：{response.text}")

def test_detect_methods():
    """测试检测方法 API"""
    print("\n=== 测试检测方法 API ===")
    response = requests.get(f"{BASE_URL}/api/v1/detection-methods")
    print(f"状态码：{response.status_code}")
    if response.status_code == 200:
        methods = response.json().get('methods', [])
        print(f"可用检测方法：{len(methods)} 种")
        for m in methods:
            print(f"  - {m['id']}: {m['name']}")
        print("✓ 检测方法 API 正常")

def test_inpaint_methods():
    """测试修复算法 API"""
    print("\n=== 测试修复算法 API ===")
    response = requests.get(f"{BASE_URL}/api/v1/inpaint-methods")
    print(f"状态码：{response.status_code}")
    if response.status_code == 200:
        methods = response.json().get('methods', [])
        print(f"可用修复算法：{len(methods)} 种")
        for m in methods:
            print(f"  - {m['id']}: {m['name']}")
        print("✓ 修复算法 API 正常")

def test_video_methods():
    """测试视频处理方法 API"""
    print("\n=== 测试视频处理方法 API ===")
    response = requests.get(f"{BASE_URL}/api/v1/video-methods")
    print(f"状态码：{response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"检测方法：{len(result.get('detection_methods', []))} 种")
        print(f"质量选项：{len(result.get('quality_options', []))} 种")
        print("✓ 视频处理方法 API 正常")

def test_static_files():
    """测试静态文件"""
    print("\n=== 测试静态文件 ===")

    # 测试 CSS
    css_files = ['style.css', 'video.css', 'app.js', 'video.js']
    for css in css_files:
        response = requests.get(f"{BASE_URL}/static/{css}")
        if response.status_code == 200:
            print(f"✓ /static/{css} 访问成功")
        else:
            print(f"✗ /static/{css} 失败：{response.status_code}")

if __name__ == "__main__":
    print("=" * 60)
    print("UI 自动化测试 - Watermark Remover")
    print("=" * 60)

    try:
        test_health()
        test_main_page()
        test_video_page()
        test_detect_methods()
        test_inpaint_methods()
        test_video_methods()
        test_static_files()
        test_image_upload_and_process()

        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)

    except requests.exceptions.ConnectionError as e:
        print(f"\n✗ 连接错误：无法连接到服务器")
        print(f"请确保服务器正在运行：python3 -m uvicorn src.api.main:app --reload --port 8000")
    except Exception as e:
        print(f"\n✗ 测试失败：{e}")
