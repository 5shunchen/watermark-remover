"""
FastAPI Interface for Watermark Removal
Provides REST API endpoints for watermark detection and removal
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Import API router
from src.api.routes import router as api_router

app = FastAPI(
    title="Watermark Remover API",
    version="2.0.0",
    description="""
## Watermark Remover API

完整的智能去水印 API，支持图片和视频处理。

### 主要功能

- **图片去水印**: 上传图片，一键去除水印
- **水印检测**: 智能识别水印位置并生成 mask
- **批量处理**: 支持多张图片同时处理
- **文件管理**: 上传、存储、下载管理

### 技术特点

- 多种检测算法（颜色、边缘、角点、图案）
- 多种修复算法（Telea、Navier-Stokes）
- 支持 CPU/GPU 处理
    """,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Mount static files
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")

# Include API router
app.include_router(api_router)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the landing page"""
    html_path = PROJECT_ROOT / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Watermark Remover</h1><p>Landing page not found</p>")


@app.get("/video", response_class=HTMLResponse)
async def video_page():
    """Serve the video watermark remover page"""
    html_path = PROJECT_ROOT / "templates" / "video.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Video Watermark Remover</h1><p>Page not found</p>")


@app.get("/api-docs", response_class=HTMLResponse, tags=["Root"])
async def api_docs():
    """Serve the API documentation page"""
    html_path = PROJECT_ROOT / "templates" / "api-docs.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>API Docs</h1><p>Documentation not found</p>")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
