"""
FastAPI Interface for Watermark Removal
Provides REST API endpoints for watermark detection and removal
"""

import asyncio
import io
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Import using absolute imports
from src.detector import detect_watermark
from src.inpainter import remove_watermark
from src.video import remove_watermark_from_video

app = FastAPI(title="Watermark Remover API", version="1.0.0")

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Mount static files
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the landing page"""
    html_path = PROJECT_ROOT / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Watermark Remover</h1><p>Landing page not found</p>")


@app.get("/")
async def root():
    return {"message": "Watermark Remover API", "version": "1.0.0"}


@app.post("/detect/")
async def detect_watermark_endpoint(file: UploadFile = File(...)):
    """
    Detect watermark in an uploaded image and return the mask
    """
    # Read and convert the uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Detect watermark
    mask = detect_watermark(image, method="auto")

    # Save mask temporarily
    temp_filename = f"temp_mask_{uuid.uuid4().hex}.png"
    mask_path = os.path.join("/tmp", temp_filename)
    mask.save(mask_path)

    return FileResponse(
        mask_path, media_type="image/png", filename=f"mask_{file.filename}"
    )


@app.post("/remove/")
async def remove_watermark_endpoint(
    file: UploadFile = File(...),
    detection_method: str = Form("auto"),
    device: str = Form("cpu"),
):
    """
    Remove watermark from an uploaded image
    """
    # Read and convert the uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Detect watermark
    mask = detect_watermark(image, method=detection_method)

    # Remove watermark using inpainting
    result_image = remove_watermark(image=image, mask=mask, device=device)

    # Save result temporarily
    temp_filename = f"temp_result_{uuid.uuid4().hex}.png"
    result_path = os.path.join("/tmp", temp_filename)
    result_image.save(result_path)

    return FileResponse(
        result_path, media_type="image/png", filename=f"clean_{file.filename}"
    )


@app.post("/remove-video/")
async def remove_watermark_from_video_endpoint(
    file: UploadFile = File(...),
    detection_method: str = Form("auto"),
    device: str = Form("cpu"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Remove watermark from an uploaded video
    """
    # Save uploaded video temporarily
    temp_input_path = f"/tmp/temp_video_input_{uuid.uuid4().hex}.mp4"
    temp_output_path = f"/tmp/temp_video_output_{uuid.uuid4().hex}.mp4"

    # Write uploaded video to temp file
    with open(temp_input_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    # Process video to remove watermark
    success = remove_watermark_from_video(
        video_input_path=temp_input_path,
        video_output_path=temp_output_path,
        detection_method=detection_method,
        device=device,
    )

    if not success:
        return {"error": "Failed to process video"}

    # Remove temporary input file
    background_tasks.add_task(os.remove, temp_input_path)

    return FileResponse(
        temp_output_path,
        media_type="video/mp4",
        filename=f"clean_{file.filename}",
        background=background_tasks,
    )


@app.post("/process/")
async def process_image_endpoint(
    file: UploadFile = File(...),
    detection_method: str = Form("auto"),
    device: str = Form("cpu"),
):
    """
    Complete processing: detect and remove watermark in one step
    """
    # Read and convert the uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Detect watermark
    mask = detect_watermark(image, method=detection_method)

    # Remove watermark using inpainting
    result_image = remove_watermark(image=image, mask=mask, device=device)

    # Save result temporarily
    temp_filename = f"temp_processed_{uuid.uuid4().hex}.png"
    result_path = os.path.join("/tmp", temp_filename)
    result_image.save(result_path)

    return FileResponse(
        result_path, media_type="image/png", filename=f"processed_{file.filename}"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
