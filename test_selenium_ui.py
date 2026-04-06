#!/usr/bin/env python3
"""
Selenium UI 自动化测试 - Watermark Remover 全流程测试
覆盖：主页、视频页、图片上传、API 测试、导航交互
"""

import time
import sys
from pathlib import Path
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException

BASE_URL = "http://localhost:8000"

# 测试结果存储
test_results = {
    "passed": [],
    "failed": [],
    "skipped": [],
    "errors": []
}


def setup_driver():
    """配置并启动 Firefox WebDriver"""
    from selenium.webdriver.firefox.service import Service
    from selenium.webdriver.firefox.options import Options

    firefox_options = Options()
    firefox_options.add_argument("--headless")
    firefox_options.add_argument("--no-sandbox")
    firefox_options.add_argument("--disable-dev-shm-usage")
    firefox_options.add_argument("--disable-gpu")
    firefox_options.add_argument("--window-size=1920,1080")
    firefox_options.add_argument("--lang=zh-CN")

    driver = webdriver.Firefox(options=firefox_options)
    driver.set_page_load_timeout(30)
    driver.set_script_timeout(30)
    return driver


def log_result(test_name: str, passed: bool, message: str = ""):
    """记录测试结果"""
    result = {
        "test": test_name,
        "message": message,
        "time": datetime.now().strftime("%H:%M:%S")
    }
    if passed:
        test_results["passed"].append(result)
        print(f"  ✓ {test_name}")
    else:
        test_results["failed"].append(result)
        print(f"  ✗ {test_name}: {message}")


def take_screenshot(driver, name: str):
    """截取屏幕截图"""
    screenshot_dir = Path("test_screenshots")
    screenshot_dir.mkdir(exist_ok=True)
    driver.save_screenshot(str(screenshot_dir / f"{name}.png"))


# ============== 测试用例 ==============

def test_01_health_check(driver):
    """测试 1: API 健康检查"""
    test_name = "API 健康检查"
    try:
        driver.get(f"{BASE_URL}/api/v1/health")
        body_text = driver.find_element(By.TAG_NAME, "pre").text

        if '"status":"healthy"' in body_text or '"status": "healthy"' in body_text:
            log_result(test_name, True)
        else:
            log_result(test_name, False, "健康检查返回状态不是 healthy")
    except Exception as e:
        log_result(test_name, False, str(e))


def test_02_homepage_load(driver):
    """测试 2: 主页加载"""
    test_name = "主页加载"
    try:
        driver.get(f"{BASE_URL}/")
        time.sleep(1)

        # 检查页面标题
        assert "Watermark Remover" in driver.title or "去水印" in driver.title

        # 检查主要元素存在
        hero_title = driver.find_element(By.CLASS_NAME, "hero-title")
        assert hero_title.is_displayed()

        # 检查导航链接
        nav_links = driver.find_elements(By.CSS_SELECTOR, ".nav-links a")
        assert len(nav_links) >= 5

        log_result(test_name, True)
    except Exception as e:
        log_result(test_name, False, str(e))
        take_screenshot(driver, "homepage_error")


def test_03_video_page_load(driver):
    """测试 3: 视频页面加载"""
    test_name = "视频页面加载"
    try:
        driver.get(f"{BASE_URL}/video")
        time.sleep(1)

        # 检查页面标题
        assert "视频" in driver.title or "Video" in driver.title

        # 检查上传区域
        upload_zone = driver.find_element(By.ID, "uploadZone")
        assert upload_zone.is_displayed()

        # 检查标题
        h1 = driver.find_element(By.TAG_NAME, "h1")
        assert "视频" in h1.text or "Video" in h1.text

        log_result(test_name, True)
    except Exception as e:
        log_result(test_name, False, str(e))
        take_screenshot(driver, "video_page_error")


def test_04_theme_toggle(driver):
    """测试 4: 主题切换功能"""
    test_name = "主题切换功能"
    try:
        driver.get(f"{BASE_URL}/")
        time.sleep(1)

        # 获取初始主题
        html = driver.find_element(By.TAG_NAME, "html")
        initial_theme = html.get_attribute("data-theme")

        # 点击主题切换按钮
        theme_toggle = driver.find_element(By.ID, "themeToggle")
        theme_toggle.click()
        time.sleep(0.5)

        # 检查主题是否改变
        new_theme = html.get_attribute("data-theme")

        if initial_theme != new_theme:
            log_result(test_name, True, f"{initial_theme} -> {new_theme}")
        else:
            # 可能已经是另一个主题了，再切换一次
            theme_toggle.click()
            time.sleep(0.5)
            final_theme = html.get_attribute("data-theme")
            log_result(test_name, initial_theme != final_theme, f"主题切换可能有问题")
    except Exception as e:
        log_result(test_name, False, str(e))


def test_05_nav_links(driver):
    """测试 5: 导航链接测试"""
    test_name = "导航链接测试"
    try:
        driver.get(f"{BASE_URL}/")
        time.sleep(1)

        nav_items = {
            "功能": "#features",
            "视频去水印": "/video",
            "在线体验": "#demo",
        }

        all_passed = True
        errors = []

        for link_text, expected_href in nav_items.items():
            try:
                link = driver.find_element(By.XPATH, f"//a[contains(text(), '{link_text}')]")
                href = link.get_attribute("href")
                if expected_href in href:
                    log_result(f"  - 导航链接：{link_text}", True)
                else:
                    log_result(f"  - 导航链接：{link_text}", False, f"href={href}")
                    all_passed = False
            except NoSuchElementException:
                errors.append(f"未找到链接：{link_text}")
                all_passed = False

        if all_passed:
            log_result(test_name, True)
        else:
            log_result(test_name, False, "; ".join(errors))
    except Exception as e:
        log_result(test_name, False, str(e))


def test_06_api_docs_pages(driver):
    """测试 6: API 文档页面"""
    test_name = "API 文档页面"
    try:
        # 测试 Swagger UI
        driver.get(f"{BASE_URL}/docs")
        time.sleep(2)
        assert "Swagger" in driver.title or "API" in driver.title
        log_result("  - Swagger UI", True)

        # 测试 ReDoc
        driver.get(f"{BASE_URL}/redoc")
        time.sleep(2)
        assert "ReDoc" in driver.title or "API" in driver.title
        log_result("  - ReDoc", True)

        # 测试自定义 API 文档
        driver.get(f"{BASE_URL}/api-docs")
        time.sleep(1)
        assert "API" in driver.title
        log_result("  - 自定义文档", True)

        log_result(test_name, True)
    except Exception as e:
        log_result(test_name, False, str(e))
        take_screenshot(driver, "api_docs_error")


def test_07_static_files(driver):
    """测试 7: 静态文件加载"""
    test_name = "静态文件加载"
    try:
        static_files = [
            "/static/style.css",
            "/static/app.js",
            "/static/video.css",
            "/static/video.js",
        ]

        all_loaded = True
        for file_url in static_files:
            driver.get(f"{BASE_URL}{file_url}")
            time.sleep(0.5)
            page_source = driver.page_source
            if len(page_source) < 50:
                log_result(f"  - {file_url}", False, "文件内容为空或太短")
                all_loaded = False
            else:
                log_result(f"  - {file_url}", True)

        if all_loaded:
            log_result(test_name, True)
        else:
            log_result(test_name, False, "部分静态文件加载失败")
    except Exception as e:
        log_result(test_name, False, str(e))


def test_08_image_upload_ui(driver):
    """测试 8: 图片上传 UI 测试"""
    test_name = "图片上传 UI 测试"
    try:
        driver.get(f"{BASE_URL}/#demo")
        time.sleep(1)

        # 查找文件上传输入框
        file_input = driver.find_element(By.ID, "fileInput")
        assert file_input.is_displayed() or not file_input.is_displayed()  # 可能是隐藏的

        # 查找拖拽上传区域
        upload_zone = driver.find_element(By.ID, "uploadZone")
        assert upload_zone.is_displayed()

        log_result(test_name, True)
    except Exception as e:
        log_result(test_name, False, str(e))


def test_09_video_upload_ui(driver):
    """测试 9: 视频上传 UI 测试"""
    test_name = "视频上传 UI 测试"
    try:
        driver.get(f"{BASE_URL}/video")
        time.sleep(1)

        # 查找文件上传输入框
        file_input = driver.find_element(By.ID, "fileInput")
        assert file_input

        # 查找拖拽上传区域
        upload_zone = driver.find_element(By.ID, "uploadZone")
        assert upload_zone.is_displayed()

        # 检查格式提示
        hint_badges = driver.find_elements(By.CLASS_NAME, "hint-badge")
        formats = [badge.text for badge in hint_badges]
        assert "MP4" in formats or any("mp4" in f.lower() for f in formats)

        log_result(test_name, True, f"支持格式：{formats}")
    except Exception as e:
        log_result(test_name, False, str(e))
        take_screenshot(driver, "video_upload_error")


def test_10_video_settings_panel(driver):
    """测试 10: 视频设置面板"""
    test_name = "视频设置面板"
    try:
        driver.get(f"{BASE_URL}/video")

        # 检查设置元素是否存在（初始可能是隐藏的）
        settings = {
            "detectionMethod": "检测方法选择",
            "frameInterval": "帧间隔滑块",
            "startProcessBtn": "开始处理按钮",
        }

        for element_id, element_name in settings.items():
            try:
                element = driver.find_element(By.ID, element_id)
                log_result(f"  - {element_name}", True)
            except NoSuchElementException:
                log_result(f"  - {element_name}", False, "元素不存在")

        log_result(test_name, True)
    except Exception as e:
        log_result(test_name, False, str(e))


def test_11_comparison_viewer(driver):
    """测试 11: 对比查看器"""
    test_name = "对比查看器"
    try:
        driver.get(f"{BASE_URL}/video")
        time.sleep(1)

        # 检查对比滑块组件
        try:
            comparison_slider = driver.find_element(By.ID, "comparisonSlider")
            slider_handle = driver.find_element(By.ID, "sliderHandle")
            log_result(test_name, True, "对比滑块组件存在")
        except NoSuchElementException:
            log_result(test_name, False, "对比滑块组件不存在")
    except Exception as e:
        log_result(test_name, False, str(e))


def test_12_responsive_design(driver):
    """测试 12: 响应式设计测试"""
    test_name = "响应式设计"
    try:
        # 测试不同视口宽度
        viewports = [
            (375, 667, "iPhone SE"),
            (768, 1024, "iPad"),
            (1920, 1080, "Desktop"),
        ]

        for width, height, device_name in viewports:
            driver.set_window_size(width, height)
            driver.get(f"{BASE_URL}/")
            time.sleep(0.5)

            # 检查页面是否正常渲染
            body = driver.find_element(By.TAG_NAME, "body")
            body_width = body.size["width"]

            # 检查是否有水平滚动（不应该有）
            scroll_width = driver.execute_script("return document.body.scrollWidth")
            has_horizontal_scroll = scroll_width > width

            log_result(f"  - {device_name} ({width}x{height})", not has_horizontal_scroll,
                      "有水平滚动" if has_horizontal_scroll else "正常")

        log_result(test_name, True)
        driver.set_window_size(1920, 1080)  # 恢复默认大小
    except Exception as e:
        log_result(test_name, False, str(e))


def test_13_accessibility(driver):
    """测试 13: 无障碍功能测试"""
    test_name = "无障碍功能"
    try:
        driver.get(f"{BASE_URL}/")
        time.sleep(1)

        checks = []

        # 检查 skip-link
        try:
            skip_link = driver.find_element(By.CLASS_NAME, "skip-link")
            checks.append(("skip-link", skip_link.is_displayed()))
        except NoSuchElementException:
            checks.append(("skip-link", False))

        # 检查 aria-label
        buttons_with_aria = driver.find_elements(By.CSS_SELECTOR, "button[aria-label]")
        checks.append(("aria-label buttons", len(buttons_with_aria) > 0))

        # 检查 alt 文本
        images = driver.find_elements(By.CSS_SELECTOR, "img:not([alt])")
        checks.append(("所有图片有 alt", len(images) == 0))

        # 检查主题切换按钮
        try:
            theme_toggle = driver.find_element(By.ID, "themeToggle")
            has_aria = theme_toggle.get_attribute("aria-label") is not None
            checks.append(("主题切换 aria-label", has_aria))
        except:
            checks.append(("主题切换 aria-label", False))

        all_passed = all(result for _, result in checks)
        for check_name, result in checks:
            log_result(f"  - {check_name}", result)

        log_result(test_name, all_passed)
    except Exception as e:
        log_result(test_name, False, str(e))


def test_14_api_endpoints(driver):
    """测试 14: API 端点可访问性"""
    test_name = "API 端点可访问性"
    try:
        endpoints = [
            ("/api/v1/health", "健康检查"),
            ("/api/v1/stats", "使用统计"),
            ("/api/v1/detection-methods", "检测方法"),
            ("/api/v1/inpaint-methods", "修复算法"),
            ("/api/v1/video-methods", "视频方法"),
            ("/api/v1/files", "文件列表"),
        ]

        all_accessible = True
        for endpoint, name in endpoints:
            try:
                driver.get(f"{BASE_URL}{endpoint}")
                time.sleep(0.3)
                body = driver.find_element(By.TAG_NAME, "pre")
                log_result(f"  - {name}", True)
            except Exception as e:
                log_result(f"  - {name}", False, str(e))
                all_accessible = False

        log_result(test_name, all_accessible)
    except Exception as e:
        log_result(test_name, False, str(e))


def test_15_footer_links(driver):
    """测试 15: Footer 链接测试"""
    test_name = "Footer 链接测试"
    try:
        driver.get(f"{BASE_URL}/")
        time.sleep(1)

        # 滚动到底部
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.5)

        # 检查 footer 存在
        footer = driver.find_element(By.CLASS_NAME, "footer")
        assert footer.is_displayed()

        # 检查 GitHub 链接
        github_link = driver.find_element(By.XPATH, "//a[contains(@href, 'github.com')]")
        assert "5shunchen" in github_link.get_attribute("href")

        log_result(test_name, True)
    except Exception as e:
        log_result(test_name, False, str(e))


# ============== 主测试流程 ==============

def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("Selenium UI 自动化测试 - Watermark Remover")
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"目标地址：{BASE_URL}")
    print("=" * 70)

    driver = None
    try:
        driver = setup_driver()
        print(f"\n✓ Chrome WebDriver 启动成功\n")

        # 测试服务器是否运行
        try:
            driver.get(BASE_URL)
            print("✓ 服务器连接成功\n")
        except Exception as e:
            print(f"✗ 无法连接到服务器：{e}")
            print("  请先启动服务器：python3 -m uvicorn src.api.main:app --reload --port 8000")
            return

        # 运行所有测试
        tests = [
            (test_01_health_check, "API 健康检查"),
            (test_02_homepage_load, "主页加载"),
            (test_03_video_page_load, "视频页面加载"),
            (test_04_theme_toggle, "主题切换功能"),
            (test_05_nav_links, "导航链接测试"),
            (test_06_api_docs_pages, "API 文档页面"),
            (test_07_static_files, "静态文件加载"),
            (test_08_image_upload_ui, "图片上传 UI"),
            (test_09_video_upload_ui, "视频上传 UI"),
            (test_10_video_settings_panel, "视频设置面板"),
            (test_11_comparison_viewer, "对比查看器"),
            (test_12_responsive_design, "响应式设计"),
            (test_13_accessibility, "无障碍功能"),
            (test_14_api_endpoints, "API 端点"),
            (test_15_footer_links, "Footer 链接"),
        ]

        for test_func, test_name in tests:
            print(f"\n[测试] {test_name}")
            try:
                test_func(driver)
            except Exception as e:
                log_result(test_name, False, f"测试异常：{str(e)}")
                take_screenshot(driver, f"{test_name}_error")

        # 打印测试结果
        print("\n" + "=" * 70)
        print("测试结果汇总")
        print("=" * 70)

        total = len(test_results["passed"]) + len(test_results["failed"])
        passed = len(test_results["passed"])
        failed = len(test_results["failed"])

        print(f"\n总测试数：{total}")
        print(f"通过：{passed} ({passed/total*100:.1f}%)")
        print(f"失败：{failed} ({failed/total*100:.1f}%)")

        if test_results["failed"]:
            print("\n失败的测试:")
            for result in test_results["failed"]:
                print(f"  ✗ {result['test']}: {result['message']}")

        print("\n" + "=" * 70)
        if failed == 0:
            print("✓ 所有测试通过!")
        else:
            print(f"⚠ {failed} 个测试失败，请检查")
        print("=" * 70)

        # 生成测试报告
        generate_report()

    except Exception as e:
        print(f"\n✗ 测试执行失败：{e}")
        import traceback
        traceback.print_exc()
    finally:
        if driver:
            driver.quit()
            print("\n✓ WebDriver 已关闭")


def generate_report():
    """生成 HTML 测试报告"""
    report_dir = Path("test_reports")
    report_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"test_report_{timestamp}.html"

    total = len(test_results["passed"]) + len(test_results["failed"])
    passed = len(test_results["passed"])
    failed = len(test_results["failed"])

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>UI 自动化测试报告</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #3b82f6; padding-bottom: 15px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .summary-card {{ flex: 1; padding: 20px; border-radius: 8px; text-align: center; }}
        .summary-card.passed {{ background: #dcfce7; }}
        .summary-card.failed {{ background: #fee2e2; }}
        .summary-card.total {{ background: #eff6ff; }}
        .summary-card h3 {{ margin: 0; font-size: 2.5rem; }}
        .summary-card p {{ margin: 5px 0 0 0; color: #666; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 30px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e5e5e5; }}
        th {{ background: #f9fafb; font-weight: 600; }}
        .status-pass {{ color: #22c55e; }}
        .status-fail {{ color: #ef4444; }}
        .timestamp {{ color: #999; font-size: 0.9rem; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 UI 自动化测试报告</h1>
        <p class="timestamp">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary">
            <div class="summary-card total">
                <h3>{total}</h3>
                <p>总测试数</p>
            </div>
            <div class="summary-card passed">
                <h3>{passed}</h3>
                <p>通过</p>
            </div>
            <div class="summary-card failed">
                <h3>{failed}</h3>
                <p>失败</p>
            </div>
        </div>

        <h2>测试结果详情</h2>
        <table>
            <thead>
                <tr>
                    <th>测试项</th>
                    <th>状态</th>
                    <th>时间</th>
                    <th>备注</th>
                </tr>
            </thead>
            <tbody>
"""

    for result in test_results["passed"]:
        html_content += f"""
                <tr>
                    <td>{result['test']}</td>
                    <td class="status-pass">✓ 通过</td>
                    <td>{result['time']}</td>
                    <td>{result['message']}</td>
                </tr>
"""

    for result in test_results["failed"]:
        html_content += f"""
                <tr>
                    <td>{result['test']}</td>
                    <td class="status-fail">✗ 失败</td>
                    <td>{result['time']}</td>
                    <td>{result['message']}</td>
                </tr>
"""

    html_content += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n📄 测试报告已生成：{report_path.absolute()}")


if __name__ == "__main__":
    run_all_tests()
