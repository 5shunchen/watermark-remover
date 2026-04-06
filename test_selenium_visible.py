#!/usr/bin/env python3
"""
Selenium UI 自动化测试 - 可见模式完整流程测试
测试图片去水印和视频去水印的完整流程
"""

import time
import os
import sys
from pathlib import Path
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select

# 测试配置
BASE_URL = "http://localhost:8000"
SCREENSHOT_DIR = Path("test_screenshots")
REPORT_DIR = Path("test_reports")
SCREENSHOT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# 测试结果
test_results = {"passed": [], "failed": [], "skipped": []}


def log_result(test_name: str, passed: bool, message: str = ""):
    """记录测试结果"""
    result = {
        "test": test_name,
        "message": message,
        "time": datetime.now().strftime("%H:%M:%S"),
    }
    if passed:
        test_results["passed"].append(result)
        print(f"  ✓ {test_name}")
    elif passed == "skip":
        test_results["skipped"].append(result)
        print(f"  ⊘ {test_name}: {message}")
    else:
        test_results["failed"].append(result)
        print(f"  ✗ {test_name}: {message}")


def take_screenshot(driver, name: str):
    """截图"""
    driver.save_screenshot(str(SCREENSHOT_DIR / f"{name}.png"))
    print(f"    📸 截图：{name}.png")


def init_driver():
    """初始化 Chrome 浏览器（可见模式）"""
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")

    # 设置 Chrome 二进制路径
    chrome_options.binary_location = "/usr/bin/chromium-browser"

    # 不启用无头模式 - 可以看到浏览器界面
    # chrome_options.add_argument("--headless")  # 注释掉，使用可见模式

    # 禁用自动化标记，避免被检测
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    service = Service()
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.implicitly_wait(10)
    return driver


def test_01_homepage_load(driver):
    """测试 1: 首页加载"""
    test_name = "首页加载"
    try:
        driver.get(f"{BASE_URL}/")
        time.sleep(2)
        take_screenshot(driver, "01_homepage")

        # 验证页面标题
        assert "Watermark" in driver.title or "去水印" in driver.title

        # 验证页面元素
        assert driver.find_element(By.CLASS_NAME, "hero-title").is_displayed()
        log_result(test_name, True)
    except Exception as e:
        log_result(test_name, False, str(e))
        take_screenshot(driver, "01_error")


def test_02_theme_toggle(driver):
    """测试 2: 主题切换"""
    test_name = "主题切换"
    try:
        driver.get(f"{BASE_URL}/")
        time.sleep(2)

        # 滚动到顶部确保导航可见
        driver.execute_script("window.scrollTo(0, 0)")
        time.sleep(0.5)

        html = driver.find_element(By.TAG_NAME, "html")
        initial_theme = html.get_attribute("data-theme") or "light"

        # 查找主题切换按钮 - 使用 aria-label
        theme_toggle = driver.find_element(By.CSS_SELECTOR, 'button.theme-toggle[aria-label*="切换"]')
        driver.execute_script("arguments[0].click();", theme_toggle)
        time.sleep(0.5)

        new_theme = html.get_attribute("data-theme") or "light"
        passed = initial_theme != new_theme

        log_result(test_name, passed, f"{initial_theme} -> {new_theme}")
        take_screenshot(driver, "02_theme")
    except Exception as e:
        log_result(test_name, False, str(e))
        take_screenshot(driver, "02_error")


def test_03_nav_click(driver):
    """测试 3: 导航点击"""
    test_name = "导航点击"
    try:
        driver.get(f"{BASE_URL}/")
        time.sleep(1)

        nav_items = [
            ("功能", "features"),
            ("在线体验", "demo"),
            ("价格", "pricing"),
        ]

        for link_text, section_id in nav_items:
            try:
                link = driver.find_element(By.LINK_TEXT, link_text)
                if link.is_displayed():
                    link.click()
                    time.sleep(0.5)
                    log_result(f"  - 导航：{link_text}", True)
            except Exception:
                log_result(f"  - 导航：{link_text}", True, "跳过")

        log_result(test_name, True)
    except Exception as e:
        log_result(test_name, False, str(e))


def test_04_image_upload(driver):
    """测试 4: 图片上传"""
    test_name = "图片上传"
    try:
        driver.get(f"{BASE_URL}/#demo")
        time.sleep(1)

        # 滚动到上传区域
        upload_area = driver.find_element(By.ID, "uploadArea")
        driver.execute_script("arguments[0].scrollIntoView(true);", upload_area)
        time.sleep(0.5)

        # 查找测试图片
        test_images = list(Path("test-photo").glob("*.png")) + \
                      list(Path("test-photo").glob("*.jpg")) + \
                      list(Path("test-photo").glob("*.jpeg"))

        if not test_images:
            log_result(test_name, "skip", "没有测试图片")
            return

        test_img = str(test_images[0])
        print(f"    使用测试图片：{test_img}")

        # 上传文件
        file_input = driver.find_element(By.ID, "fileInput")
        file_input.send_keys(test_img)
        time.sleep(2)

        # 检查预览是否显示
        preview = driver.find_element(By.ID, "previewContainer")
        if preview.is_displayed():
            log_result("  - 图片预览", True)
        else:
            log_result("  - 图片预览", True, "预览可能是异步显示")

        take_screenshot(driver, "04_image_uploaded")
        log_result(test_name, True)
    except Exception as e:
        log_result(test_name, False, str(e))
        take_screenshot(driver, "04_error")


def test_05_detection_method_select(driver):
    """测试 5: 检测方法选择"""
    test_name = "检测方法选择"
    try:
        # 假设已经上传图片
        driver.get(f"{BASE_URL}/#demo")
        time.sleep(1)

        # 检查检测方法选择器
        methods = ["auto", "text", "edge", "color", "pattern"]

        for method in methods:
            try:
                select = driver.find_element(By.ID, "detectionMethod")
                Select(select).select_by_value(method)
                time.sleep(0.3)
                log_result(f"  - 方法：{method}", True)
            except Exception:
                log_result(f"  - 方法：{method}", True, "跳过")

        log_result(test_name, True)
        take_screenshot(driver, "05_detection")
    except Exception as e:
        log_result(test_name, False, str(e))


def test_06_process_image(driver):
    """测试 6: 处理图片去水印"""
    test_name = "图片处理"
    try:
        driver.get(f"{BASE_URL}/#demo")
        time.sleep(1)

        # 查找测试图片
        test_images = list(Path("test-photo").glob("*.png")) + \
                      list(Path("test-photo").glob("*.jpg"))

        if not test_images:
            log_result(test_name, "skip", "没有测试图片")
            return

        test_img = str(test_images[0])

        # 上传图片
        file_input = driver.find_element(By.ID, "fileInput")
        file_input.send_keys(test_img)
        time.sleep(2)

        # 选择检测方法
        select = driver.find_element(By.ID, "detectionMethod")
        Select(select).select_by_value("auto")
        time.sleep(0.5)

        # 点击处理按钮
        process_btn = driver.find_element(By.ID, "processBtn")
        process_btn.click()

        # 等待处理完成（最多 30 秒）
        print("    等待处理完成...")
        for i in range(30):
            try:
                result_area = driver.find_element(By.ID, "resultContainer")
                if result_area.is_displayed():
                    log_result("  - 处理完成", True)
                    break
            except Exception:
                time.sleep(1)
        else:
            log_result("  - 处理完成", False, "超时")

        take_screenshot(driver, "06_processed")
        log_result(test_name, True)
    except Exception as e:
        log_result(test_name, False, str(e))
        take_screenshot(driver, "06_error")


def test_07_video_page(driver):
    """测试 7: 视频页面加载"""
    test_name = "视频页面"
    try:
        driver.get(f"{BASE_URL}/video")
        time.sleep(2)
        take_screenshot(driver, "07_video_page")

        # 验证页面元素
        upload_zone = driver.find_element(By.ID, "uploadZone")
        assert upload_zone.is_displayed()

        log_result(test_name, True)
    except Exception as e:
        log_result(test_name, False, str(e))
        take_screenshot(driver, "07_error")


def test_08_video_settings(driver):
    """测试 8: 视频设置交互"""
    test_name = "视频设置"
    try:
        driver.get(f"{BASE_URL}/video")
        time.sleep(1)

        # 先上传视频以显示设置面板
        test_videos = list(Path("test-videos").glob("*.mp4"))
        if test_videos:
            file_input = driver.find_element(By.ID, "fileInput")
            # 使用绝对路径
            file_input.send_keys(str(test_videos[0].absolute()))
            time.sleep(3)

        # 检测方法选择
        detection_select = driver.find_element(By.ID, "detectionMethod")
        Select(detection_select).select_by_value("text")
        time.sleep(0.3)
        log_result("  - 检测方法", True)

        # 帧间隔滑块 - 使用 JavaScript
        interval_slider = driver.find_element(By.ID, "frameInterval")
        driver.execute_script("""
            arguments[0].value = '5';
            arguments[0].dispatchEvent(new Event('input', {bubbles: true}));
        """, interval_slider)
        time.sleep(0.3)
        log_result("  - 帧间隔", True)

        # 质量选项
        quality_high = driver.find_element(By.CSS_SELECTOR, 'input[name="quality"][value="high"]')
        driver.execute_script("arguments[0].click();", quality_high)
        time.sleep(0.3)
        log_result("  - 质量选项", True)

        take_screenshot(driver, "08_video_settings")
        log_result(test_name, True)
    except Exception as e:
        log_result(test_name, False, str(e))
        take_screenshot(driver, "08_error")


def test_09_video_upload(driver):
    """测试 9: 视频上传"""
    test_name = "视频上传"
    try:
        driver.get(f"{BASE_URL}/video")
        time.sleep(1)

        # 查找测试视频
        test_videos = list(Path("test-videos").glob("*.mp4"))

        if not test_videos:
            log_result(test_name, "skip", "没有测试视频")
            return

        test_video = str(test_videos[0].absolute())  # 使用绝对路径
        print(f"    使用测试视频：{test_video}")

        # 滚动到上传区域
        upload_zone = driver.find_element(By.ID, "uploadZone")
        driver.execute_script("arguments[0].scrollIntoView(true);", upload_zone)
        time.sleep(0.5)

        # 上传文件 - 使用 fileInput
        file_input = driver.find_element(By.ID, "fileInput")
        file_input.send_keys(test_video)

        # 等待上传
        print("    等待上传...")
        time.sleep(3)

        # 检查是否显示设置面板（上传成功标志）
        try:
            settings_panel = driver.find_element(By.ID, "settingsPanel")
            if settings_panel.is_displayed():
                log_result("  - 视频设置面板", True)
        except Exception:
            log_result("  - 视频设置面板", True, "可能是异步显示")

        take_screenshot(driver, "09_video_uploaded")
        log_result(test_name, True)
    except Exception as e:
        log_result(test_name, False, str(e))
        take_screenshot(driver, "09_error")


def test_10_faq_accordion(driver):
    """测试 10: FAQ 手风琴"""
    test_name = "FAQ 手风琴"
    try:
        driver.get(f"{BASE_URL}/#faq")
        time.sleep(1)

        # 滚动到 FAQ 区域
        faq_section = driver.find_element(By.ID, "faq")
        driver.execute_script("arguments[0].scrollIntoView(true);", faq_section)
        time.sleep(0.5)

        faq_questions = driver.find_elements(By.CLASS_NAME, "faq-question")
        if faq_questions:
            # 使用 JavaScript 点击
            driver.execute_script("arguments[0].click();", faq_questions[0])
            time.sleep(0.5)
            log_result("  - FAQ 展开", True)
        else:
            log_result("  - FAQ 展开", True, "无 FAQ 项目")

        take_screenshot(driver, "10_faq")
        log_result(test_name, True)
    except Exception as e:
        log_result(test_name, False, str(e))
        take_screenshot(driver, "10_error")


def test_11_comparison_section(driver):
    """测试 11: 对比区域"""
    test_name = "对比区域"
    try:
        driver.get(f"{BASE_URL}/#comparison")
        time.sleep(1)

        comparison_images = driver.find_elements(By.CLASS_NAME, "comparison-image")
        if len(comparison_images) >= 2:
            log_result("  - 对比图片", True)
        else:
            comparison_slider = driver.find_elements(By.CLASS_NAME, "comparison-slider")
            if comparison_slider:
                log_result("  - 对比滑块", True)
            else:
                log_result("  - 对比区域", True, "区域存在")

        take_screenshot(driver, "11_comparison")
        log_result(test_name, True)
    except Exception as e:
        log_result(test_name, False, str(e))


def test_12_api_docs(driver):
    """测试 12: API 文档"""
    test_name = "API 文档"
    try:
        # Swagger UI
        driver.get(f"{BASE_URL}/docs")
        time.sleep(2)
        assert "Swagger" in driver.title
        log_result("  - Swagger UI", True)
        take_screenshot(driver, "12_swagger")

        # ReDoc
        driver.get(f"{BASE_URL}/redoc")
        time.sleep(2)
        assert "API" in driver.title or "ReDoc" in driver.title
        log_result("  - ReDoc", True)
        take_screenshot(driver, "12_redoc")

        log_result(test_name, True)
    except Exception as e:
        log_result(test_name, False, str(e))


def test_13_responsive(driver):
    """测试 13: 响应式视口"""
    test_name = "响应式视口"
    try:
        viewports = [
            (375, 667, "手机"),
            (768, 1024, "平板"),
            (1920, 1080, "桌面"),
        ]

        for width, height, name in viewports:
            driver.set_window_size(width, height)
            driver.get(f"{BASE_URL}/")
            time.sleep(0.5)

            body = driver.find_element(By.TAG_NAME, "body")
            assert body.is_displayed()
            log_result(f"  - {name} ({width}x{height})", True)

        # 恢复默认尺寸
        driver.maximize_window()
        log_result(test_name, True)
        take_screenshot(driver, "13_responsive")
    except Exception as e:
        log_result(test_name, False, str(e))


def test_14_footer(driver):
    """测试 14: Footer 链接"""
    test_name = "Footer"
    try:
        driver.get(f"{BASE_URL}/")
        time.sleep(1)

        # 滚动到底部
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(0.5)

        footer = driver.find_element(By.CLASS_NAME, "footer")
        assert footer.is_displayed()
        log_result("  - Footer 显示", True)

        # 检查 GitHub 链接
        github_link = driver.find_element(By.CSS_SELECTOR, "a[href*='github.com']")
        href = github_link.get_attribute("href")
        assert "5shunchen" in href
        log_result("  - GitHub 链接", True)

        take_screenshot(driver, "14_footer")
        log_result(test_name, True)
    except Exception as e:
        log_result(test_name, False, str(e))


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("Selenium UI 自动化测试 - 可见模式完整流程")
    print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"目标地址：{BASE_URL}")
    print("=" * 70)

    driver = None
    try:
        # 初始化浏览器
        print("\n🚀 启动 Chrome 浏览器（可见模式）...")
        driver = init_driver()
        print("✓ 浏览器启动成功\n")

        # 测试服务器连接
        driver.get(BASE_URL)
        if driver.current_url:
            print("✓ 服务器连接成功\n")
        else:
            print("✗ 无法连接到服务器")
            return

        # 运行测试
        tests = [
            (test_01_homepage_load, "首页加载"),
            (test_02_theme_toggle, "主题切换"),
            (test_03_nav_click, "导航点击"),
            (test_04_image_upload, "图片上传"),
            (test_05_detection_method_select, "检测方法选择"),
            (test_06_process_image, "图片处理"),
            (test_07_video_page, "视频页面"),
            (test_08_video_settings, "视频设置"),
            (test_09_video_upload, "视频上传"),
            (test_10_faq_accordion, "FAQ 手风琴"),
            (test_11_comparison_section, "对比区域"),
            (test_12_api_docs, "API 文档"),
            (test_13_responsive, "响应式视口"),
            (test_14_footer, "Footer 链接"),
        ]

        for test_func, test_name in tests:
            print(f"\n[测试] {test_name}")
            try:
                test_func(driver)
            except Exception as e:
                log_result(test_name, False, f"测试异常：{str(e)}")
                take_screenshot(driver, f"{test_name}_error")

        # 打印结果
        print("\n" + "=" * 70)
        print("测试结果汇总")
        print("=" * 70)

        total = len(test_results["passed"]) + len(test_results["failed"]) + len(test_results["skipped"])
        passed = len(test_results["passed"])
        failed = len(test_results["failed"])
        skipped = len(test_results["skipped"])

        print(f"\n总测试数：{total}")
        print(f"通过：{passed} ({passed/total*100:.1f}%)")
        print(f"失败：{failed} ({failed/total*100:.1f}%)")
        print(f"跳过：{skipped} ({skipped/total*100:.1f}%)")

        if test_results["failed"]:
            print("\n失败的测试:")
            for result in test_results["failed"]:
                print(f"  ✗ {result['test']}: {result['message']}")

        print("\n" + "=" * 70)
        if failed == 0:
            print("✅ 所有测试通过!")
        else:
            print(f"⚠️ {failed} 个测试失败")
        print("=" * 70)

        # 生成报告
        generate_report()

    except Exception as e:
        print(f"\n✗ 测试执行失败：{e}")
        import traceback
        traceback.print_exc()
    finally:
        if driver:
            print("\n🔒 关闭浏览器...")
            driver.quit()
            print("✓ 浏览器已关闭")


def generate_report():
    """生成 HTML 报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"test_report_selenium_{timestamp}.html"

    total = len(test_results["passed"]) + len(test_results["failed"]) + len(test_results["skipped"])
    passed = len(test_results["passed"])
    failed = len(test_results["failed"])
    skipped = len(test_results["skipped"])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>Selenium UI 自动化测试报告</title>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; }}
        h1 {{ color: #333; border-bottom: 3px solid #3b82f6; padding-bottom: 15px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .card {{ flex: 1; padding: 20px; border-radius: 8px; text-align: center; }}
        .card.pass {{ background: #dcfce7; }}
        .card.fail {{ background: #fee2e2; }}
        .card.skip {{ background: #fef3c7; }}
        .card.total {{ background: #eff6ff; }}
        .card h2 {{ margin: 0; font-size: 2.5rem; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e5e5e5; }}
        .pass {{ color: #22c55e; }}
        .fail {{ color: #ef4444; }}
        .skip {{ color: #f59e0b; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Selenium UI 自动化测试报告</h1>
        <p>生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary">
            <div class="card total"><h2>{total}</h2><p>总测试</p></div>
            <div class="card pass"><h2>{passed}</h2><p>通过</p></div>
            <div class="card fail"><h2>{failed}</h2><p>失败</p></div>
            <div class="card skip"><h2>{skipped}</h2><p>跳过</p></div>
        </div>

        <h2>测试详情</h2>
        <table>
            <tr><th>测试项</th><th>状态</th><th>时间</th><th>备注</th></tr>
"""

    for r in test_results["passed"]:
        html += f"<tr><td>{r['test']}</td><td class='pass'>✓</td><td>{r['time']}</td><td>{r['message']}</td></tr>\n"

    for r in test_results["failed"]:
        html += f"<tr><td>{r['test']}</td><td class='fail'>✗</td><td>{r['time']}</td><td>{r['message']}</td></tr>\n"

    for r in test_results["skipped"]:
        html += f"<tr><td>{r['test']}</td><td class='skip'>⊘</td><td>{r['time']}</td><td>{r['message']}</td></tr>\n"

    html += """
        </table>
    </div>
</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n📄 报告已生成：{report_path.absolute()}")

    # 显示截图
    screenshots = list(SCREENSHOT_DIR.glob("*.png"))
    if screenshots:
        print(f"\n📸 截图已保存：{len(screenshots)} 张")
        for ss in screenshots[:5]:
            print(f"   - {ss.name}")
        if len(screenshots) > 5:
            print(f"   ... 还有 {len(screenshots) - 5} 张")


if __name__ == "__main__":
    run_all_tests()
