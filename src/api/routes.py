"""
Complete REST API for Watermark Remover
Provides comprehensive endpoints for image/video processing, batch operations, and file management
"""

import io
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from src.detector import detect_watermark
from src.inpainter import remove_watermark
from src.video import remove_watermark_from_video, get_video_info, VideoStatus

# Create router
router = APIRouter(prefix="/api/v1", tags=["Watermark Remover API"])

# Storage directories
UPLOAD_DIR = Path("/tmp/watermark-remover/uploads")
OUTPUT_DIR = Path("/tmp/watermark-remover/outputs")
MASK_DIR = Path("/tmp/watermark-remover/masks")

# Ensure directories exist
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, MASK_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# In-memory storage (use Redis/Database in production)
upload_storage: Dict[str, dict] = {}
job_storage: Dict[str, dict] = {}
batch_storage: Dict[str, dict] = {}
video_job_storage: Dict[str, dict] = {}


# ============== Utility Functions ==============


def save_upload_file(file: UploadFile, file_id: str) -> Path:
    """Save uploaded file to storage"""
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path


def get_image_info(file_path: Path) -> dict:
    """Get image metadata"""
    with Image.open(file_path) as img:
        return {
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "size_bytes": file_path.stat().st_size,
            "mode": img.mode,
        }


def cleanup_temp_files(*paths: str) -> None:
    """Clean up temporary files"""
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


# ============== Health & Status Endpoints ==============


@router.get("/health", tags=["Health"])
async def health_check():
    """
    检查 API 健康状态

    返回 API 版本和各服务组件状态。
    """
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "detector": "ok",
            "inpainter": "ok",
            "storage": "ok",
            "api": "ok",
        },
    }


@router.get("/stats", tags=["Health"])
async def get_usage_stats():
    """
    获取使用统计信息

    返回总处理数、今日处理数、用户数等统计数据。
    """
    now = datetime.now()
    today = now.date()

    return {
        "total_processed": len(job_storage),
        "today_processed": len(
            [
                j
                for j in job_storage.values()
                if datetime.fromisoformat(j["created_at"]).date() == today
            ]
        ),
        "total_files": len(upload_storage),
        "active_jobs": len([j for j in job_storage.values() if j["status"] == "processing"]),
    }


# ============== File Upload Endpoints ==============


@router.post("/upload", tags=["Files"])
async def upload_file(file: UploadFile = File(...)):
    """
    上传文件用于处理

    - 支持格式：JPG, PNG, WebP, GIF
    - 返回：文件 ID 用于后续处理
    - 文件大小限制：最大 10MB
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型。允许的类型：{allowed_types}",
        )

    # Validate file size (10MB limit)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning

    if file_size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件大小超过 10MB 限制")

    # Generate unique file ID
    file_id = str(uuid.uuid4())

    # Save file
    file_path = save_upload_file(file, file_id)

    # Store metadata
    upload_storage[file_id] = {
        "file_id": file_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file_path.stat().st_size,
        "upload_time": datetime.now().isoformat(),
        "path": str(file_path),
        "status": "uploaded",
    }

    return {
        "success": True,
        "file_id": file_id,
        "filename": file.filename,
        "size": file_path.stat().st_size,
        "content_type": file.content_type,
        "upload_time": upload_storage[file_id]["upload_time"],
    }


@router.get("/files/{file_id}", tags=["Files"])
async def get_file_info(file_id: str):
    """
    获取已上传文件的信息
    """
    if file_id not in upload_storage:
        raise HTTPException(status_code=404, detail="文件不存在")

    file_data = upload_storage[file_id]
    return {
        "file_id": file_data["file_id"],
        "filename": file_data["filename"],
        "content_type": file_data["content_type"],
        "size": file_data["size"],
        "upload_time": file_data["upload_time"],
        "status": file_data["status"],
        "metadata": get_image_info(Path(file_data["path"])),
    }


@router.delete("/files/{file_id}", tags=["Files"])
async def delete_file(file_id: str):
    """
    删除已上传的文件
    """
    if file_id not in upload_storage:
        raise HTTPException(status_code=404, detail="文件不存在")

    file_data = upload_storage[file_id]
    file_path = Path(file_data["path"])

    # Delete file from storage
    if file_path.exists():
        os.remove(file_path)

    # Remove from storage dict
    del upload_storage[file_id]

    return {"success": True, "message": "文件已成功删除"}


@router.get("/files", tags=["Files"])
async def list_files(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    列出所有已上传的文件
    """
    files = list(upload_storage.values())
    total = len(files)
    files = files[offset : offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "files": [
            {
                "file_id": f["file_id"],
                "filename": f["filename"],
                "size": f["size"],
                "upload_time": f["upload_time"],
                "status": f["status"],
            }
            for f in files
        ],
    }


# ============== Image Processing Endpoints ==============


@router.post("/process", tags=["Image Processing"])
async def process_image(
    file: UploadFile = File(...),
    detection_method: str = Form("auto"),
    device: str = Form("cpu"),
    return_mask: bool = Form(False),
    inpaint_method: str = Form("lama"),
):
    """
    完整图片处理：检测并移除水印

    上传图片并一键获取处理结果。

    ### 参数
    - **detection_method**: 水印检测方法 (auto, color, edge, corners, pattern, text, enhanced)
    - **device**: 处理设备 (cpu 或 cuda)
    - **return_mask**: 是否返回检测掩码
    - **inpaint_method**: 修复方法 (lama, telea, ns) - 默认 lama (AI 模型，最佳质量)
    """
    start_time = time.time()
    file_id = str(uuid.uuid4())

    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Detect watermark
        mask = detect_watermark(image, method=detection_method)

        # Remove watermark
        result_image = remove_watermark(
            image=image,
            mask=mask,
            device=device,
            method=inpaint_method,
        )

        # Save result
        result_filename = f"processed_{file_id}.png"
        result_path = OUTPUT_DIR / result_filename
        result_image.save(result_path)

        result = {
            "success": True,
            "file_id": file_id,
            "output_url": f"/api/v1/download/{result_filename}",
            "download_url": f"/api/v1/download/{result_filename}",
            "processing_time": round(time.time() - start_time, 2),
            "message": "图片处理成功",
            "method_used": detection_method,
        }

        if return_mask:
            mask_filename = f"mask_{file_id}.png"
            mask_path = MASK_DIR / mask_filename
            mask.save(mask_path)
            result["mask_url"] = f"/api/v1/download/masks/{mask_filename}"

        # Store job
        job_storage[file_id] = {
            "job_id": file_id,
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "processing_time": result["processing_time"],
        }

        return result

    except Exception as e:
        return {
            "success": False,
            "file_id": file_id,
            "processing_time": round(time.time() - start_time, 2),
            "message": f"处理失败：{str(e)}",
            "error": str(e),
        }


@router.post("/process/{file_id}", tags=["Image Processing"])
async def process_uploaded_image(
    file_id: str,
    detection_method: str = Form("auto"),
    device: str = Form("cpu"),
    return_mask: bool = Form(False),
):
    """
    处理已上传的图片

    使用之前上传的文件 ID 进行处理，避免重复上传。
    """
    if file_id not in upload_storage:
        raise HTTPException(status_code=404, detail="文件不存在")

    start_time = time.time()
    file_data = upload_storage[file_id]
    file_path = Path(file_data["path"])

    try:
        # Load image
        image = Image.open(file_path).convert("RGB")

        # Detect watermark
        mask = detect_watermark(image, method=detection_method)

        # Remove watermark
        result_image = remove_watermark(
            image=image,
            mask=mask,
            device=device,
        )

        # Save result
        result_filename = f"processed_{file_id}.png"
        result_path = OUTPUT_DIR / result_filename
        result_image.save(result_path)

        # Update file status
        upload_storage[file_id]["status"] = "processed"

        result = {
            "success": True,
            "file_id": file_id,
            "output_url": f"/api/v1/download/{result_filename}",
            "processing_time": round(time.time() - start_time, 2),
            "message": "图片处理成功",
        }

        if return_mask:
            mask_filename = f"mask_{file_id}.png"
            mask_path = MASK_DIR / mask_filename
            mask.save(mask_path)
            result["mask_url"] = f"/api/v1/download/masks/{mask_filename}"

        return result

    except Exception as e:
        return {
            "success": False,
            "file_id": file_id,
            "processing_time": round(time.time() - start_time, 2),
            "message": f"处理失败：{str(e)}",
        }


@router.post("/detect/{file_id}", tags=["Detection"])
async def detect_watermark_api(
    file_id: str,
    method: str = Form("auto"),
):
    """
    检测水印并返回掩码

    返回检测掩码 PNG 图片，可用于自定义修复区域。
    """
    if file_id not in upload_storage:
        raise HTTPException(status_code=404, detail="文件不存在")

    file_data = upload_storage[file_id]
    file_path = Path(file_data["path"])

    try:
        # Load image
        image = Image.open(file_path).convert("RGB")

        # Detect watermark
        mask = detect_watermark(image, method=method)

        # Save mask
        mask_filename = f"mask_{file_id}.png"
        mask_path = MASK_DIR / mask_filename
        mask.save(mask_path)

        return FileResponse(
            mask_path,
            media_type="image/png",
            filename=f"mask_{file_data['filename']}",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败：{str(e)}")


@router.post("/detect-upload", tags=["Detection"])
async def detect_and_return_mask(
    file: UploadFile = File(...),
    method: str = Form("auto"),
):
    """
    上传图片并直接返回水印检测掩码

    用于预览水印检测效果。
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        mask = detect_watermark(image, method=method)

        # Convert mask to bytes
        buffer = io.BytesIO()
        mask.save(buffer, format="PNG")
        buffer.seek(0)

        return FileResponse(
            buffer,
            media_type="image/png",
            filename=f"mask_{file.filename}",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败：{str(e)}")


# ============== Download Endpoints ==============


@router.get("/download/{filename}", tags=["Files"])
async def download_file(filename: str):
    """
    下载处理后的文件
    """
    # Determine file location
    if filename.startswith("mask_"):
        file_path = MASK_DIR / filename
    else:
        file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(
        file_path,
        media_type="image/png",
        filename=filename,
    )


@router.get("/download/masks/{filename}", tags=["Files"])
async def download_mask(filename: str):
    """
    下载水印掩码文件
    """
    file_path = MASK_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(
        file_path,
        media_type="image/png",
        filename=filename,
    )


# ============== Detection Methods Info ==============


@router.get("/detection-methods", tags=["Info"])
async def get_detection_methods():
    """
    获取所有可用的检测方法说明
    """
    return {
        "methods": [
            {
                "id": "auto",
                "name": "智能自动",
                "description": "自动分析图片并选择最佳检测方法",
                "recommended": True,
            },
            {
                "id": "color",
                "name": "颜色检测",
                "description": "基于 HSV 颜色阈值检测水印",
                "use_case": "适用于彩色水印",
            },
            {
                "id": "edge",
                "name": "边缘检测",
                "description": "使用 Canny 算法检测边缘",
                "use_case": "适用于文字水印",
            },
            {
                "id": "corners",
                "name": "角点检测",
                "description": "检测图片角点区域",
                "use_case": "适用于角落水印",
            },
            {
                "id": "pattern",
                "name": "图案检测",
                "description": "分析常见水印位置和图案",
                "use_case": "适用于规则水印",
            },
            {
                "id": "text",
                "name": "文字检测",
                "description": "使用 CLAHE 和 HSV 融合检测文字",
                "use_case": "适用于文字类水印如@用户名",
                "recommended": True,
            },
        ]
    }


@router.get("/inpaint-methods", tags=["Info"])
async def get_inpaint_methods():
    """
    获取所有可用的修复算法说明
    """
    return {
        "methods": [
            {
                "id": "telea",
                "name": "Telea 算法",
                "description": "快速高质量的修复算法",
                "speed": "快",
                "quality": "高",
                "recommended": True,
            },
            {
                "id": "ns",
                "name": "Navier-Stokes",
                "description": "基于流体力学的高质量修复",
                "speed": "慢",
                "quality": "很高",
            },
            {
                "id": "ns_original",
                "name": "Navier-Stokes 原始版",
                "description": "原始 NS 算法实现",
                "speed": "中等",
                "quality": "高",
            },
        ]
    }


# ============== Video Processing Endpoints ==============


@router.post("/upload-video", tags=["Video Processing"])
async def upload_video(file: UploadFile = File(...)):
    """
    上传视频文件用于处理

    - 支持格式：MP4, AVI, MOV, MKV, WebM
    - 返回：文件 ID 用于后续处理
    - 文件大小限制：最大 100MB
    """
    # Validate file type
    allowed_types = [
        "video/mp4",
        "video/x-msvideo",
        "video/quicktime",
        "video/x-matroska",
        "video/webm",
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型。允许的类型：{allowed_types}",
        )

    # Validate file size (100MB limit for videos)
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)

    if file_size > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件大小超过 100MB 限制")

    # Generate unique file ID
    file_id = str(uuid.uuid4())

    # Save file
    file_path = save_upload_file(file, file_id)

    # Get video info
    try:
        video_info = get_video_info(str(file_path))
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"无法读取视频文件：{str(e)}")

    # Store metadata
    upload_storage[file_id] = {
        "file_id": file_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file_path.stat().st_size,
        "upload_time": datetime.now().isoformat(),
        "path": str(file_path),
        "status": "uploaded",
        "type": "video",
        "video_info": {
            "width": video_info.width,
            "height": video_info.height,
            "fps": video_info.fps,
            "total_frames": video_info.total_frames,
            "duration": video_info.duration,
            "codec": video_info.codec,
        },
    }

    return {
        "success": True,
        "file_id": file_id,
        "filename": file.filename,
        "size": file_path.stat().st_size,
        "content_type": file.content_type,
        "upload_time": upload_storage[file_id]["upload_time"],
        "video_info": upload_storage[file_id]["video_info"],
    }


@router.post("/process-video/{file_id}", tags=["Video Processing"])
async def process_video(
    file_id: str,
    detection_method: str = Form("auto"),
    frame_interval: int = Form(1),
    quality: str = Form("high"),
):
    """
    处理视频去水印

    对已上传的视频进行水印检测和移除处理。
    处理时间较长，建议异步处理。

    ### 参数
    - **detection_method**: 水印检测方法 (auto/color/edge/corners/pattern/text)
    - **frame_interval**: 帧处理间隔 (1=每帧处理，2=每隔一帧处理)
    - **quality**: 输出质量 (low/medium/high)
    """
    if file_id not in upload_storage:
        raise HTTPException(status_code=404, detail="文件不存在")

    file_data = upload_storage[file_id]
    if file_data.get("type") != "video":
        raise HTTPException(status_code=400, detail="不是视频文件")

    job_id = str(uuid.uuid4())
    file_path = Path(file_data["path"])

    # Create job entry
    video_job_storage[job_id] = {
        "job_id": job_id,
        "file_id": file_id,
        "status": "processing",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "detection_method": detection_method,
        "frame_interval": frame_interval,
        "quality": quality,
    }

    # Process video in background
    async def process_video_background():
        output_filename = f"processed_{file_id}.mp4"
        output_path = OUTPUT_DIR / output_filename

        def progress_callback(progress: float):
            video_job_storage[job_id]["progress"] = min(1.0, progress * 100)

        try:
            success = remove_watermark_from_video(
                video_input_path=str(file_path),
                video_output_path=str(output_path),
                detection_method=detection_method,
                frame_interval=frame_interval,
                progress_callback=progress_callback,
            )

            if success:
                video_job_storage[job_id]["status"] = "completed"
                video_job_storage[job_id]["progress"] = 100
                video_job_storage[job_id]["output_path"] = str(output_path)
                video_job_storage[job_id]["output_filename"] = output_filename
            else:
                video_job_storage[job_id]["status"] = "failed"
                video_job_storage[job_id]["error"] = "处理失败"

        except Exception as e:
            video_job_storage[job_id]["status"] = "failed"
            video_job_storage[job_id]["error"] = str(e)

    # Start background processing
    import asyncio

    asyncio.create_task(process_video_background())

    return {
        "success": True,
        "job_id": job_id,
        "status": "processing",
        "message": "视频处理已开始",
        "estimated_time": "根据视频长度而定",
    }


@router.get("/video-status/{job_id}", tags=["Video Processing"])
async def get_video_status(job_id: str):
    """
    查询视频处理进度状态
    """
    if job_id not in video_job_storage:
        raise HTTPException(status_code=404, detail="任务不存在")

    job_data = video_job_storage[job_id]
    return {
        "job_id": job_data["job_id"],
        "status": job_data["status"],
        "progress": job_data.get("progress", 0),
        "created_at": job_data["created_at"],
        "output_url": (
            f"/api/v1/download-video/{job_data['output_filename']}"
            if job_data.get("output_filename")
            else None
        ),
        "error": job_data.get("error"),
    }


@router.get("/download-video/{filename}", tags=["Video Processing"])
async def download_video(filename: str):
    """
    下载处理后的视频
    """
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=filename,
    )


@router.get("/video-methods", tags=["Video Processing"])
async def get_video_methods():
    """
    获取视频处理相关说明
    """
    return {
        "detection_methods": [
            {
                "id": "auto",
                "name": "智能自动",
                "description": "自动分析视频帧并选择最佳检测方法",
                "recommended": True,
            },
            {
                "id": "text",
                "name": "文字检测",
                "description": "专门检测视频中的文字水印",
                "use_case": "适用于@用户名等文字水印",
                "recommended": True,
            },
            {
                "id": "color",
                "name": "颜色检测",
                "description": "基于颜色阈值检测",
                "use_case": "适用于彩色水印",
            },
            {
                "id": "pattern",
                "name": "图案检测",
                "description": "分析固定位置的水印图案",
                "use_case": "适用于台标/固定位置水印",
            },
        ],
        "quality_options": [
            {
                "id": "high",
                "name": "高质量",
                "description": "最佳输出质量，文件较大",
                "bitrate": "8000k",
            },
            {
                "id": "medium",
                "name": "中等质量",
                "description": "平衡质量和文件大小",
                "bitrate": "4000k",
            },
            {
                "id": "low",
                "name": "低质量",
                "description": "较小文件大小，快速处理",
                "bitrate": "2000k",
            },
        ],
    }
